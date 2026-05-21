#########################################################################
##  Bound operators for ONNX ATen fallback ops                         ##
#########################################################################
"""``Bound`` subclasses for selected ``aten::ATen`` ONNX-fallback nodes."""

import torch

from .base import Bound, Interval

# PyTorch ScalarType integer codes used in ONNX ATen fallback for ``torch.zeros`` etc.
_SCALAR_TYPE_TO_DTYPE = {
    0: torch.uint8,
    1: torch.int8,
    2: torch.int16,
    3: torch.int32,
    4: torch.int64,
    5: torch.half,
    6: torch.float32,
    7: torch.float64,
    8: torch.complex32,
    9: torch.complex64,
    10: torch.complex128,
    11: torch.bool,
    15: torch.bfloat16,
}


def _tensor_to_int_list(t: torch.Tensor):
    if t is None or (isinstance(t, torch.Tensor) and t.numel() == 0):
        return None
    if not isinstance(t, torch.Tensor):
        return t
    flat = t.reshape(-1).tolist()
    return [int(x) for x in flat]


def _forward_subtree(node) -> object:
    """Recursively materialize ``forward_value`` for a node not on the main output path."""
    cached = getattr(node, "forward_value", None)
    if cached is not None:
        return cached
    args = [_forward_subtree(inp) for inp in node.inputs]
    fv = node.forward(*args)
    if isinstance(fv, torch.Tensor):
        node.output_shape = fv.shape
    node.forward_value = fv
    return fv


def _apply_pending_inplace_writes(buffer_node, buffer: torch.Tensor) -> torch.Tensor:
    """Apply registered ``aten::copy_`` side effects onto a freshly materialized buffer."""
    pending = getattr(buffer_node, "_pending_inplace_writes", None)
    if not pending:
        return buffer
    out = buffer
    for copy_node in pending:
        if type(copy_node).__name__ != "BoundAtenJitCopy":
            continue
        src_node = copy_node.inputs[1]
        src_val = _forward_subtree(src_node)
        if not isinstance(src_val, torch.Tensor):
            continue
        chain = copy_node._slice_chain()
        out = BoundAtenJitCopy._write_slice_chain(out, src_val, chain)
    return out


class BoundATenOnnxZeros(Bound):
    """``torch.zeros`` as ``aten::ATen[operator="zeros"]`` with tensor arguments (ONNX export)."""

    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.use_default_ibp = False
        self.no_jacobian = True

    def forward(
        self, shape, scalar_type=None, layout=None, device=None, requires_grad=False
    ):
        del layout
        if isinstance(shape, torch.Tensor):
            shape_list = [int(x) for x in shape.reshape(-1).tolist()]
        elif isinstance(shape, (list, tuple)):
            shape_list = []
            for x in shape:
                if isinstance(x, torch.Tensor):
                    shape_list.extend(int(t) for t in x.reshape(-1).tolist())
                else:
                    shape_list.append(int(x))
        else:
            shape_list = [int(shape)]
        if scalar_type is None or (
            isinstance(scalar_type, torch.Tensor) and scalar_type.numel() == 0
        ):
            dtype = torch.float32
        else:
            code = (
                int(scalar_type.item())
                if isinstance(scalar_type, torch.Tensor)
                else int(scalar_type)
            )
            dtype = _SCALAR_TYPE_TO_DTYPE.get(code, torch.float32)
        if device is None or (isinstance(device, torch.Tensor) and device.numel() == 0):
            dev = torch.device("cpu")
        elif isinstance(device, torch.device):
            dev = device
        elif isinstance(device, str):
            dev = torch.device(device)
        else:
            dev = torch.device("cpu")
        if requires_grad is None or (
            isinstance(requires_grad, torch.Tensor) and requires_grad.numel() == 0
        ):
            rg = False
        else:
            rg = (
                bool(requires_grad.item())
                if isinstance(requires_grad, torch.Tensor)
                else bool(requires_grad)
            )
        out = torch.zeros(*shape_list, dtype=dtype, device=dev, requires_grad=rg)
        return _apply_pending_inplace_writes(self, out)

    def interval_propagate(self, shape, scalar_type=None, layout=None, device=None, requires_grad=None):
        def _bound(iv):
            if iv is None:
                return None
            return iv[0]

        args_l = [_bound(shape), _bound(scalar_type), _bound(layout), _bound(device), _bound(requires_grad)]
        args_u = [
            shape[1] if shape is not None else None,
            scalar_type[1] if scalar_type is not None else None,
            layout[1] if layout is not None else None,
            device[1] if device is not None else None,
            requires_grad[1] if requires_grad is not None else None,
        ]
        o_l = self.forward(*args_l)
        o_u = self.forward(*args_u)
        pending = getattr(self, "_pending_inplace_writes", None)
        if pending:
            for copy_node in pending:
                val_node = copy_node.inputs[1]
                val_iv = getattr(val_node, "interval", None)
                if val_iv is None:
                    continue
                o_l, o_u = copy_node.interval_propagate_base((o_l, o_u), val_iv)
        return Interval.make_interval(o_l, o_u, shape)


def _norm_arg(norm):
    if norm is None:
        return None
    if isinstance(norm, str):
        return norm
    if isinstance(norm, torch.Tensor):
        if norm.numel() == 1 and norm.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            return None
    return norm


def _complex_center_radius(lower, upper):
    center = (lower + upper) / 2
    if lower.is_complex():
        real_radius = (upper.real - lower.real).abs() / 2
        imag_radius = (upper.imag - lower.imag).abs() / 2
    else:
        real_radius = (upper - lower).abs() / 2
        imag_radius = torch.zeros_like(real_radius)
    return center, real_radius, imag_radius


def _complex_interval(center, real_radius, imag_radius):
    if center.is_complex():
        lower = (center.real - real_radius).to(center.real.dtype) + 1j * (
            center.imag - imag_radius
        ).to(center.real.dtype)
        upper = (center.real + real_radius).to(center.real.dtype) + 1j * (
            center.imag + imag_radius
        ).to(center.real.dtype)
        return lower, upper
    return center - real_radius, center + real_radius


def _fft_numel(shape, dims):
    n = 1
    for dim in dims:
        n *= int(shape[dim])
    return n


def _fft_scale(norm, n, inverse=False):
    if norm == "forward":
        return 1.0 if inverse else 1.0 / n
    if norm == "ortho":
        return n ** -0.5
    return 1.0 / n if inverse else 1.0


def _sum_radius_for_fft(radius, dims, scale, out):
    summed = radius
    for dim in sorted(dims, reverse=True):
        summed = summed.sum(dim=dim, keepdim=True)
    return (summed * scale).expand_as(out.real if out.is_complex() else out)


def _linear_interval_with_weight(equation, x_l, x_u, weight):
    center, real_radius, imag_radius = _complex_center_radius(x_l, x_u)
    out_center = torch.einsum(equation, center, weight)
    weight_abs_real = weight.real.abs() if weight.is_complex() else weight.abs()
    weight_abs_imag = weight.imag.abs() if weight.is_complex() else torch.zeros_like(weight)
    out_real_radius = (
        torch.einsum(equation, real_radius, weight_abs_real)
        + torch.einsum(equation, imag_radius, weight_abs_imag)
    )
    out_imag_radius = (
        torch.einsum(equation, real_radius, weight_abs_imag)
        + torch.einsum(equation, imag_radius, weight_abs_real)
    )
    return _complex_interval(out_center, out_real_radius, out_imag_radius)


class BoundATenFftIrfftn(Bound):
    """``torch.fft.irfftn`` exported as ``aten::ATen[operator="fft_irfftn"]``."""

    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.use_default_ibp = True

    def forward(self, *inputs):
        inp, s, dim, norm = inputs[0], inputs[1], inputs[2], inputs[3]
        s_list = _tensor_to_int_list(s)
        dim_list = _tensor_to_int_list(dim)
        norm_s = _norm_arg(norm)
        return torch.fft.irfftn(inp, s=s_list, dim=dim_list, norm=norm_s)


class BoundATenFftRfftn(Bound):
    """``torch.fft.rfftn`` exported as ``aten::ATen[operator="fft_rfftn"]``."""

    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.use_default_ibp = True

    def forward(self, *inputs):
        # Typical: (x, norm, dim, norm_dup) with None for norm — see ONNX graph.
        x = inputs[0]
        dim = inputs[2] if len(inputs) > 2 else None
        dim_list = _tensor_to_int_list(dim) if dim is not None else None
        norm = None
        for cand in (inputs[1], inputs[3] if len(inputs) > 3 else None):
            n = _norm_arg(cand)
            if isinstance(n, str):
                norm = n
                break
        return torch.fft.rfftn(x, dim=dim_list, norm=norm)


def _discover_slices_from_diff(base: torch.Tensor, out: torch.Tensor, eps: float = 1e-9):
    """Find axis-aligned slice ranges where ``out`` differs from ``base``."""
    diff = (out - base).abs()
    if diff.is_complex():
        diff = diff.real + diff.imag
    slices = []
    for dim in range(base.ndim):
        if base.shape[dim] == 1:
            continue
        m = diff
        for d in range(base.ndim):
            if d != dim:
                m = m.amax(dim=d, keepdim=True)
        m = m.squeeze()
        if m.ndim == 0:
            if float(m) <= eps:
                continue
            slices.append((dim, 0, 1))
            continue
        idx = (m > eps).nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        slices.append((dim, int(idx[0].item()), int(idx[-1].item()) + 1))
    return slices


def _slice_index(slices: list) -> tuple:
    ndim = max(s[0] for s in slices) + 1 if slices else 0
    idx: list = [slice(None)] * (ndim if ndim else 1)
    for dim, start, end in slices:
        idx[dim] = slice(start, end)
    return tuple(idx)


class BoundAssign(Bound):
    """Write ``values`` into fixed slices of a (typically constant) base buffer.

    Used for FNO-style ``out_fft[slices_x] = v`` after JIT rewrite of ``aten::copy_``.
    Phase A: base is unperturbed; only ``values`` may carry perturbation.
    """

    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.slice_chain = list(attr.get("slice_chain", []))
        self.slices = list(attr.get("slices", []))
        self._slices_inferred = bool(self.slices)
        self.use_default_ibp = False

    def _maybe_infer_slices(self, base: torch.Tensor, values: torch.Tensor):
        if self._slices_inferred:
            return
        probe = base.clone()
        if probe.shape == values.shape:
            self.slices = []
        else:
            # Provisional write to discover index ranges (FNO slice assignment).
            idx = [slice(None)] * probe.ndim
            for dim in range(probe.ndim):
                if probe.shape[dim] != values.shape[dim]:
                    size = values.shape[dim]
                    start = max(0, (probe.shape[dim] - size) // 2)
                    idx[dim] = slice(start, start + size)
            probe[tuple(idx)] = values
            self.slices = _discover_slices_from_diff(base, probe)
        self._slices_inferred = True

    def _write_via_slice_chain(self, out, values):
        view = out
        for dim, start, end, step in self.slice_chain:
            end_i = view.shape[dim] if end > 2**62 else end
            if start < 0:
                start = view.shape[dim] + start
            length = end_i - start
            view = torch.narrow(view, dim, start, length)
            if step == -1:
                view = torch.flip(view, dims=(dim,))
        view.copy_(values)
        return out

    def forward(self, base, values):
        if values.is_complex() and not base.is_complex():
            base = base.to(values.dtype)
        out = base.clone()
        if self.slice_chain:
            return self._write_via_slice_chain(out, values)
        if not self._slices_inferred:
            self._maybe_infer_slices(base, values)
        if self.slices:
            out[_slice_index(self.slices)] = values
        elif base.shape == values.shape:
            out = values.clone()
        return out

    def interval_propagate(self, base, values):
        if self.is_input_perturbed(0):
            raise NotImplementedError(
                "BoundAssign Phase A requires an unperturbed base buffer."
            )
        z_l, z_u = base[0], base[1]
        v_l, v_u = values[0], values[1]
        if self.slice_chain:
            o_l = self._write_via_slice_chain(z_l.clone(), v_l)
            o_u = self._write_via_slice_chain(z_u.clone(), v_u)
            return Interval.make_interval(o_l, o_u, values)
        if not self.slices:
            if not self._slices_inferred:
                self._maybe_infer_slices(z_l, v_l)
        if not self.slices:
            return Interval.make_interval(v_l, v_u, values)
        idx = _slice_index(self.slices)
        o_l = z_l.clone()
        o_u = z_u.clone()
        o_l[idx] = v_l
        o_u[idx] = v_u
        return Interval.make_interval(o_l, o_u, values)

    def bound_backward(self, last_lA, last_uA, base, values, **kwargs):
        del base, kwargs

        def _to_values(A):
            if A is None:
                return None
            if not self.slices:
                return A
            idx = _slice_index(self.slices)
            if isinstance(A, torch.Tensor):
                return A[idx]
            raise NotImplementedError(f"BoundAssign: unsupported A type {type(A)}")

        return [(None, None), (_to_values(last_lA), _to_values(last_uA))], 0, 0


class BoundAtenJitSlice(Bound):
    """``aten::slice`` in the raw JIT graph (dim/start/end/step as inputs)."""

    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.use_default_ibp = False
        self.ibp_intermediate = True

    def forward(self, x, dim, start, end, step):
        dim_i = int(dim) if not isinstance(dim, int) else dim
        start_i = int(start.item() if isinstance(start, torch.Tensor) else start)
        end_i = int(end.item() if isinstance(end, torch.Tensor) else end)
        step_i = int(step.item() if isinstance(step, torch.Tensor) else step)
        dim_size = x.shape[dim_i]
        # JIT uses INT_MAX as "to end of dimension".
        if end_i > dim_size or end_i > 2**62:
            end_i = dim_size
        if start_i < 0:
            start_i = dim_size + start_i
        length = end_i - start_i
        out = torch.narrow(x, dim_i, start_i, length)
        if step_i == -1:
            out = torch.flip(out, dims=(dim_i,))
        return out

    def interval_propagate(self, *v):
        return Interval.make_interval(
            self.forward(v[0][0], *[x[0] for x in v[1:]]),
            self.forward(v[0][1], *[x[0] for x in v[1:]]),
            v[0],
        )


class BoundAtenClone(Bound):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.use_default_ibp = True

    def forward(self, x, *_optional):
        del _optional
        return x.clone()


class BoundAtenJitCopy(Bound):
    """``aten::copy_`` on a (possibly sliced) destination view."""

    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.use_default_ibp = False

    def forward(self, dest, src, non_blocking=False):
        del non_blocking
        dest.copy_(src)
        return dest

    @staticmethod
    def _scalar_value(node):
        value = getattr(node, "forward_value", None)
        if value is None:
            value = getattr(node, "value", None)
        if value is None:
            inputs = []
            for inp in node.inputs:
                inp_value = getattr(inp, "forward_value", None)
                if inp_value is None:
                    inp_value = getattr(inp, "value", None)
                if inp_value is None:
                    inp_value = BoundAtenJitCopy._scalar_value(inp)
                inputs.append(inp_value)
            value = node.forward(*inputs)
        if isinstance(value, torch.Tensor):
            return int(value.reshape(-1)[0].item())
        return int(value)

    def _slice_chain(self):
        chain = []
        cur = self.inputs[0]
        while type(cur).__name__ == "BoundAtenJitSlice":
            chain.append(
                tuple(self._scalar_value(inp) for inp in cur.inputs[1:5])
            )
            cur = cur.inputs[0]
        return list(reversed(chain))

    @staticmethod
    def _write_slice_chain(out, values, chain):
        view = out
        for dim, start, end, step in chain:
            dim_size = view.shape[dim]
            if end > dim_size or end > 2**62:
                end = dim_size
            if start < 0:
                start = dim_size + start
            if step != 1:
                raise NotImplementedError("BoundAtenJitCopy only supports step=1 slices")
            view = torch.narrow(view, dim, start, end - start)
        view.copy_(values)
        return out

    def interval_propagate_base(self, base, values):
        chain = self._slice_chain()
        base_l, base_u = base[0], base[1]
        val_l, val_u = values[0], values[1]
        if val_l.is_complex() and not base_l.is_complex():
            base_l = base_l.to(val_l.dtype)
            base_u = base_u.to(val_u.dtype)
        out_l = self._write_slice_chain(base_l.clone(), val_l, chain)
        out_u = self._write_slice_chain(base_u.clone(), val_u, chain)
        return Interval.make_interval(out_l, out_u, values)

    def interval_propagate(self, dest, values, non_blocking=None):
        del non_blocking
        if self.is_input_perturbed(0):
            raise NotImplementedError(
                "BoundAtenJitCopy requires an unperturbed destination view."
            )
        out_l = dest[0].clone()
        out_u = dest[1].clone()
        out_l.copy_(values[0])
        out_u.copy_(values[1])
        return Interval.make_interval(out_l, out_u, values)


class BoundAtenJitAdd(Bound):
    """``aten::add`` with optional scalar ``alpha`` (third JIT input)."""

    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.use_default_ibp = True

    def forward(self, x, y, alpha=1):
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.item() if alpha.numel() == 1 else alpha
        return x + y * alpha


class BoundAtenJitSize(Bound):
    """``aten::size`` — shape metadata for JIT graphs (unperturbed)."""

    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.use_default_ibp = True
        self.no_jacobian = True
        self.never_perturbed = True

    def forward(self, x, dim):
        dim_i = int(dim.item() if isinstance(dim, torch.Tensor) else dim)
        return torch.tensor(x.shape[dim_i], device=x.device, dtype=torch.int64)


class BoundPrimNumToTensor(Bound):
    """``prim::NumToTensor`` — scalar / size int → 0-D tensor."""

    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.use_default_ibp = True
        self.no_jacobian = True
        self.never_perturbed = True  # shape/metadata path

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return x
        return torch.tensor(x, device=self.device, dtype=torch.int64)


class BoundAtenJitInt(Bound):
    """``aten::Int`` — tensor/scalar → Python ``int`` for slice indices."""

    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.use_default_ibp = True
        self.no_jacobian = True
        self.never_perturbed = True

    def forward(self, x):
        if isinstance(x, int):
            return x
        if isinstance(x, torch.Tensor):
            return int(x.reshape(-1)[0].item())
        return int(x)


class BoundAtenJitFloorDivide(Bound):
    """``aten::floor_divide`` on shape scalars (JIT metadata path)."""

    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.use_default_ibp = True
        self.no_jacobian = True
        self.never_perturbed = True

    def forward(self, a, b):
        a_i = int(a.item() if isinstance(a, torch.Tensor) else a)
        b_i = int(b.item() if isinstance(b, torch.Tensor) else b)
        return torch.tensor(a_i // b_i, device=self.device, dtype=torch.int64)


class BoundAtenExpand(Bound):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.use_default_ibp = True

    def forward(self, x, sizes, *implicit):
        del implicit
        if isinstance(sizes, (list, tuple)):
            shape_list = [
                int(s.item() if isinstance(s, torch.Tensor) else s) for s in sizes
            ]
        elif isinstance(sizes, torch.Tensor):
            shape_list = [int(sizes.item())] if sizes.numel() == 1 else [
                int(t) for t in sizes.reshape(-1).tolist()
            ]
        else:
            shape_list = [int(sizes)]
        return x.expand(*shape_list)


class BoundJitConstant(Bound):
    """``prim::Constant`` in a raw JIT graph (scalars, devices, empty/None markers)."""

    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.use_default_ibp = True
        self.no_jacobian = True
        self.never_perturbed = True

    def forward(self):
        if "value" not in self.attr:
            return None
        v = self.attr["value"]
        if isinstance(v, str):
            return v
        if isinstance(v, torch.Tensor):
            return v.to(self.device) if v.numel() else v
        return torch.tensor(v, device=self.device)


class BoundPrimListConstruct(Bound):
    """``prim::ListConstruct`` for JIT shape lists (feeds ``aten::zeros`` etc.)."""

    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.use_default_ibp = True
        self.no_jacobian = True
        self.never_perturbed = True

    def forward(self, *inputs):
        out = []
        for item in inputs:
            if isinstance(item, torch.Tensor) and item.numel() == 1:
                out.append(int(item.item()))
            else:
                out.append(item)
        return out

    def interval_propagate(self, *v):
        return Interval.make_interval(
            [item[0] for item in v],
            [item[1] for item in v],
            v[0] if v else None,
        )


class BoundAtenJitEinsum(Bound):
    """``aten::einsum`` with equation string as the first JIT input."""

    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.use_default_ibp = False

    def forward(self, equation, operands, *rest):
        del rest
        if isinstance(equation, torch.Tensor):
            equation = equation.item() if equation.numel() == 1 else str(equation)
        if isinstance(operands, (list, tuple)):
            return torch.einsum(equation, *operands)
        return torch.einsum(equation, operands)

    def interval_propagate(self, equation, operands, *rest):
        del rest
        eq = equation[0]
        if isinstance(eq, torch.Tensor):
            eq = eq.item() if eq.numel() == 1 else str(eq)
        lower_ops, upper_ops = operands[0], operands[1]
        if not isinstance(lower_ops, (list, tuple)) or len(lower_ops) != 2:
            raise NotImplementedError("BoundAtenJitEinsum supports two operands in IBP")
        first_perturbed = not torch.equal(lower_ops[0], upper_ops[0])
        second_perturbed = not torch.equal(lower_ops[1], upper_ops[1])
        if first_perturbed and second_perturbed:
            raise NotImplementedError(
                "BoundAtenJitEinsum IBP does not support two perturbed operands"
            )
        if first_perturbed:
            lower, upper = _linear_interval_with_weight(
                eq, lower_ops[0], upper_ops[0], lower_ops[1]
            )
            return Interval.make_interval(lower, upper, operands)
        if second_perturbed:
            # Swap the operands in the equation so the perturbed operand is first.
            lhs, rhs = eq.split("->")
            in_a, in_b = lhs.split(",")
            swapped_eq = f"{in_b},{in_a}->{rhs}"
            lower, upper = _linear_interval_with_weight(
                swapped_eq, lower_ops[1], upper_ops[1], lower_ops[0]
            )
            return Interval.make_interval(lower, upper, operands)
        out = self.forward(eq, lower_ops)
        return Interval.make_interval(out, out)


class BoundAtenJitFftIrfftn(Bound):
    """``aten::fft_irfftn`` in the raw JIT graph."""

    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.use_default_ibp = False

    def forward(self, x, s, dim, norm):
        return torch.fft.irfftn(
            x,
            s=_tensor_to_int_list(s),
            dim=_tensor_to_int_list(dim),
            norm=_norm_arg(norm),
        )

    def interval_propagate(self, x, s, dim, norm):
        dim_list = _tensor_to_int_list(dim[0])
        s_list = _tensor_to_int_list(s[0])
        norm_arg = _norm_arg(norm[0])
        center, real_radius, imag_radius = _complex_center_radius(x[0], x[1])
        out_center = self.forward(center, s[0], dim[0], norm[0])
        dims = dim_list if dim_list is not None else list(range(out_center.ndim))
        if s_list is not None:
            n = 1
            for size in s_list:
                n *= int(size)
        else:
            n = _fft_numel(out_center.shape, dims)
        radius = _sum_radius_for_fft(
            real_radius + imag_radius, dims, _fft_scale(norm_arg, n, inverse=True), out_center
        )
        return Interval.make_interval(out_center - radius, out_center + radius, x)


class BoundATenIndexPut(Bound):
    """``index_put_`` / ``index_put`` as ``aten::ATen[Placeholder, name=index_put_]``.

    ONNX optimization may fuse slice assignment into a single-input Placeholder
    that only receives the pre-write tensor; in that case LiRPA cannot recover
    index/value operands from the graph and this layer returns the base tensor
    unchanged (see module docstring in ``aten_bound_dispatch``).
    """

    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.use_default_ibp = False

    def forward(self, *inputs):
        if len(inputs) == 1:
            return inputs[0]
        accumulate = inputs[-1]
        values = inputs[-2]
        self_tensor = inputs[0]
        index_tensors = list(inputs[1:-2])
        if isinstance(accumulate, torch.Tensor) and accumulate.numel() == 1:
            accumulate = bool(accumulate.item())
        else:
            accumulate = bool(accumulate)
        return torch.index_put(self_tensor, index_tensors, values, accumulate=accumulate)

    def interval_propagate(self, *v):
        if len(v) == 1:
            return v[0][0], v[0][1]
        if self.is_input_perturbed(0):
            raise NotImplementedError(
                "BoundATenIndexPut Phase A requires an unperturbed base tensor."
            )
        base_l, base_u = v[0][0], v[0][1]
        val_l, val_u = v[-2][0], v[-2][1]
        out_l = self.forward(base_l, *[x[0] for x in v[1:-2]], val_l, v[-1][0])
        out_u = self.forward(base_u, *[x[0] for x in v[1:-2]], val_u, v[-1][0])
        return Interval.make_interval(out_l, out_u, v[-2])

    def bound_backward(self, last_lA, last_uA, *inputs, **kwargs):
        del kwargs
        if len(inputs) == 1:
            return [(last_lA, last_uA)], 0, 0
        # Phase A: only values input (index -2) may be perturbed.
        n = len(inputs)

        def _to_values(A):
            if A is None:
                return None
            raise NotImplementedError(
                "BoundATenIndexPut bound_backward for general indices is not implemented; "
                "use bound_opts['onnx_optimize_graph']=False and custom::Assign."
            )

        ret = [(None, None)] * n
        ret[-2] = (_to_values(last_lA), _to_values(last_uA))
        return ret, 0, 0
