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


class BoundZeros(Bound):
    """``torch.zeros`` (``onnx::ATen[operator=zeros]`` and raw-JIT ``aten::zeros``)."""

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
        return torch.zeros(*shape_list, dtype=dtype, device=dev, requires_grad=rg)

    def interval_propagate(self, shape, scalar_type=None, layout=None, device=None, requires_grad=None):
        del layout, requires_grad
        shape_l = shape[0] if shape is not None else None
        scalar_l = scalar_type[0] if scalar_type is not None else None
        device_l = device[0] if device is not None else None
        out = self.forward(shape_l, scalar_l, None, device_l, False)
        return Interval.make_interval(out, out, shape)


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


class BoundAtenClone(Bound):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.use_default_ibp = True

    def forward(self, x, *_optional):
        del _optional
        return x.clone()


class BoundIndexPut(Bound):
    """Slice-chain assign bound for raw-JIT ``custom::Assign`` nodes.

    Raw-JIT graphs rewrite ``aten::copy_`` into ``custom::Assign``, which
    dispatches here.  Two input shapes are supported:

    1. **Static slice chain** — ``inputs = [base, src]`` with
       ``attr["slice_chain"]`` providing the ``(dim, start, end, step)`` write
       region resolved at graph-parse time.
    2. **Lazy slice chain** — ``inputs = [base, src, dest_view]`` when slice
       indices depend on dynamic shape metadata; the chain is resolved on the
       first forward call by walking ``dest_view`` through ``BoundSlice`` nodes.

    CROWN backward supports the slice-chain form with an unperturbed base buffer
    only; the perturbed ``src`` input receives the gathered cotangent.
    """

    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.slice_chain = list(attr.get("slice_chain", []))
        self.slices = list(attr.get("slices", []))
        self._slices_inferred = bool(self.slices)
        self._slice_chain_resolved = bool(self.slice_chain)
        self.use_default_ibp = False
        self._n_inputs = len(inputs) if inputs is not None else 0
        if self._n_inputs not in (2, 3):
            raise NotImplementedError(
                "BoundIndexPut expects the slice-chain form with 2 or 3 inputs; "
                f"got {self._n_inputs}.  Use bound_opts={{'onnx_optimize_graph': False}}."
            )

    @staticmethod
    def _scalar_value(node):
        """Recursively materialize a scalar (handles ``aten::size`` / chains)."""
        value = getattr(node, "forward_value", None)
        if value is None:
            value = getattr(node, "value", None)
        if value is None:
            inp_values = []
            for inp in node.inputs:
                inp_value = getattr(inp, "forward_value", None)
                if inp_value is None:
                    inp_value = getattr(inp, "value", None)
                if inp_value is None:
                    inp_value = BoundIndexPut._scalar_value(inp)
                inp_values.append(inp_value)
            value = node.forward(*inp_values)
        if isinstance(value, torch.Tensor):
            return int(value.reshape(-1)[0].item())
        return int(value)

    def _ensure_slice_chain(self):
        """Resolve the slice chain lazily from ``dest_view`` input (if any)."""
        if self._slice_chain_resolved:
            return
        if self._n_inputs < 3:
            self._slice_chain_resolved = True
            return
        dest_view = self.inputs[2]
        chain = []
        cur = dest_view
        while type(cur).__name__ == "BoundSlice":
            # JIT slice layout: inputs[1:5] = (dim, start, end, step).
            chain.append(
                tuple(self._scalar_value(inp) for inp in cur.inputs[1:5])
            )
            cur = cur.inputs[0]
        self.slice_chain = list(reversed(chain))
        self._slice_chain_resolved = True

    # -- slice-chain helpers --------------------------------------------------

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

    def _forward_slice_chain(self, base, values):
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

    # -- forward / IBP / CROWN ------------------------------------------------

    def forward(self, *inputs):
        self._ensure_slice_chain()
        return self._forward_slice_chain(inputs[0], inputs[1])

    def interval_propagate(self, *v):
        self._ensure_slice_chain()
        if self.is_input_perturbed(0):
            raise NotImplementedError(
                "BoundIndexPut requires an unperturbed base buffer."
            )
        base_l, base_u = v[0][0], v[0][1]
        val_l, val_u = v[1][0], v[1][1]
        if self.slice_chain:
            o_l = self._write_via_slice_chain(base_l.clone(), val_l)
            o_u = self._write_via_slice_chain(base_u.clone(), val_u)
            return Interval.make_interval(o_l, o_u, v[1])
        if not self._slices_inferred:
            self._maybe_infer_slices(base_l, val_l)
        if not self.slices:
            return Interval.make_interval(val_l, val_u, v[1])
        idx = _slice_index(self.slices)
        o_l = base_l.clone()
        o_u = base_u.clone()
        o_l[idx] = val_l
        o_u[idx] = val_u
        return Interval.make_interval(o_l, o_u, v[1])

    def bound_backward(self, last_lA, last_uA, *inputs, **kwargs):
        del kwargs
        self._ensure_slice_chain()

        def _to_values(A):
            """Gather the cotangent along the written slice chain."""
            if A is None:
                return None
            if isinstance(A, torch.Tensor):
                view = A
                for dim, start, end, step in self.slice_chain:
                    a_dim = dim + 1
                    shape_dim = view.shape[a_dim]
                    end_i = shape_dim if end > 2**62 else end
                    if start < 0:
                        start = shape_dim + start
                    length = end_i - start
                    view = torch.narrow(view, a_dim, start, length)
                    if step == -1:
                        view = torch.flip(view, dims=(a_dim,))
                if self.slice_chain:
                    return view.contiguous()
                if self.slices:
                    idx = _slice_index(self.slices)
                    return A[idx]
                return A
            raise NotImplementedError(
                f"BoundIndexPut: unsupported A type {type(A)}"
            )

        ret = [(None, None)] * len(inputs)
        ret[1] = (_to_values(last_lA), _to_values(last_uA))
        return ret, 0, 0


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


class BoundAtenJitSub(Bound):
    """``aten::sub`` on shape scalars (JIT metadata path)."""

    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.use_default_ibp = True
        self.no_jacobian = True
        self.never_perturbed = True

    def forward(self, a, b, alpha=None):
        del alpha
        a_i = int(a.item() if isinstance(a, torch.Tensor) else a)
        b_i = int(b.item() if isinstance(b, torch.Tensor) else b)
        return torch.tensor(a_i - b_i, device=self.device, dtype=torch.int64)


class BoundAtenJitRemainder(Bound):
    """``aten::remainder`` on shape scalars (JIT metadata path)."""

    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.use_default_ibp = True
        self.no_jacobian = True
        self.never_perturbed = True

    def forward(self, a, b):
        a_i = int(a.item() if isinstance(a, torch.Tensor) else a)
        b_i = int(b.item() if isinstance(b, torch.Tensor) else b)
        return torch.tensor(a_i % b_i, device=self.device, dtype=torch.int64)


class BoundFftShift(Bound):
    """``aten::fft_fftshift`` — permutation along FFT frequency axes."""

    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.use_default_ibp = False

    @staticmethod
    def _resolve_dims(dim):
        if dim is None:
            return None
        if isinstance(dim, torch.Tensor):
            if dim.numel() == 0:
                return None
            return [int(x) for x in dim.reshape(-1).tolist()]
        if isinstance(dim, (list, tuple)):
            return [int(x) for x in dim]
        return int(dim)

    def forward(self, x, dim=None):
        dims = self._resolve_dims(dim)
        if dims is None:
            return torch.fft.fftshift(x)
        return torch.fft.fftshift(x, dim=dims)

    def interval_propagate(self, x_iv, dim=None):
        dim_arg = dim[0] if isinstance(dim, (list, tuple)) and len(dim) == 2 else dim
        dims = self._resolve_dims(dim_arg)
        lower = torch.fft.fftshift(x_iv[0], dim=dims)
        upper = torch.fft.fftshift(x_iv[1], dim=dims)
        return Interval.make_interval(lower, upper, x_iv)

    def bound_backward(self, last_lA, last_uA, *inputs, **kwargs):
        del kwargs
        dim = inputs[1] if len(inputs) > 1 else None
        dims = self._resolve_dims(getattr(dim, "forward_value", dim))

        def _adjoint(A):
            if A is None:
                return None
            if dims is None:
                return torch.fft.ifftshift(A)
            return torch.fft.ifftshift(A, dim=dims)

        lA = _adjoint(last_lA)
        uA = _adjoint(last_uA)
        return [(lA, uA), (None, None)], 0, 0


class BoundPrimListConstruct(Bound):
    """``prim::ListConstruct`` for JIT shape lists (feeds ``aten::zeros`` etc.)."""

    propagates_perturbed_inputs = True

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

    def bound_backward(self, last_lA, last_uA, *inputs, **kwargs):
        """Unpack a packed-tuple ``A`` from the consumer into per-element ``A``\\s.

        ``prim::ListConstruct`` outputs a Python ``list``; CROWN cannot carry a
        single tensor ``A`` across that boundary.  The convention introduced
        here: the consumer (e.g. :class:`BoundComplexEinsum`) passes ``last_lA``
        / ``last_uA`` as a Python ``tuple`` / ``list`` of length
        ``len(self.inputs)`` whose entries are the per-list-element adjoints
        (or ``None`` for unperturbed elements).  This method simply unpacks
        that tuple back into one ``(lA_i, uA_i)`` pair per producer input.
        """
        del kwargs

        def _idx(A, i):
            if A is None:
                return None
            if isinstance(A, (list, tuple)):
                return A[i]
            raise NotImplementedError(
                "BoundPrimListConstruct.bound_backward expects packed "
                "per-element A from the consumer; got "
                f"{type(A).__name__}."
            )

        n = len(inputs)
        return [(_idx(last_lA, i), _idx(last_uA, i)) for i in range(n)], 0, 0


