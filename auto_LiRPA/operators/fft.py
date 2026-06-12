#########################################################################
##  Exact trailing-axis FFT bounds (rfftn / irfftn)                      ##
#########################################################################
"""Exact CROWN/IBP rules for ``rfftn`` / ``irfftn`` on contiguous trailing axes.

Neuralop FNO graphs emit ``rfftn`` / ``irfftn`` along trailing dimensions
``(-2, -1)`` or ``(-3, -2, -1)``.  Those layouts use the staged exact linear
rule implemented here.  Other axis layouts fall back to a sum-radius IBP rule
without CROWN support (see :class:`BoundRFFT` / :class:`BoundIRFFT`).
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import torch

from .aten_fallback import (
    _complex_center_radius,
    _complex_interval,
    _fft_numel,
    _fft_scale,
    _norm_arg,
    _sum_radius_for_fft,
    _tensor_to_int_list,
)
from .base import Interval
from .complex_pair import BoundLinearComplexPair


def _resolve_fft_dims(dim_arg, ndim):
    """JIT ``dim`` operand -> ``list[int]`` of positive axes (default: all axes)."""
    dim_list = _tensor_to_int_list(dim_arg)
    if dim_list is None:
        return list(range(ndim))
    return [d % ndim for d in dim_list]


def _can_exact_nd(dims, ndim):
    """Exact path: contiguous trailing FFT axes (neuralop ``rfftn`` / ``irfftn`` layout)."""
    if not dims:
        return False
    dims = sorted(d % ndim for d in dims)
    return dims == list(range(ndim - len(dims), ndim))


def _normalize_exact_fft_dims(dims, ndim):
    """Return sorted negative axis indices for contiguous trailing FFT dims."""
    dims_pos = sorted(d % ndim for d in dims)
    return [d - ndim for d in dims_pos]


def _cfft_norm_scale(norm: Optional[str], N: int, inverse: bool = False) -> float:
    if norm == "ortho":
        return 1.0 / math.sqrt(N)
    if norm == "forward":
        return 1.0 / N if not inverse else 1.0
    return 1.0 if not inverse else 1.0 / N


def _parse_norm_str(norm_arg):
    norm_str = _norm_arg(norm_arg)
    if not (isinstance(norm_str, str) or norm_str is None):
        norm_str = None
    return norm_str


def _forward_value(node):
    if node is None:
        return None
    return getattr(node, "forward_value", node)


def _rfft_norm_scale(norm: Optional[str], N: int) -> float:
    """Forward-direction normalization scalar matching ``torch.fft.rfft``."""
    if norm == "ortho":
        return 1.0 / math.sqrt(N)
    if norm == "forward":
        return 1.0 / N
    return 1.0


def _irfft_norm_scale(norm: Optional[str], N: int) -> float:
    """Inverse-direction normalization scalar matching ``torch.fft.irfft``."""
    if norm == "ortho":
        return 1.0 / math.sqrt(N)
    if norm == "forward":
        return 1.0
    return 1.0 / N


class BoundFFTNExact(BoundLinearComplexPair):
    """Shared exact trailing-axis / loose-ND configuration for ``rfftn`` / ``irfftn``."""

    _loose_backward_msg: str

    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self._matrices = {}
        self._exact_N: Optional[int] = None
        self._exact_norm: Optional[str] = None
        self._exact_dim: Optional[int] = None
        self._exact_active: Optional[bool] = None
        self._loose_dims: Optional[list] = None
        self._exact_fft_dims: Optional[list] = None
        self._exact_sizes: Optional[dict] = None

    @property
    def _is_scalar_axis_exact(self) -> bool:
        return (
            self._exact_fft_dims is not None
            and len(self._exact_fft_dims) == 1
            and self._exact_N is not None
        )

    @staticmethod
    def _matrix_key(N, norm, dtype, device):
        return (int(N), norm, dtype, str(device))

    def _cfft_matrix_key(self, N, norm, inverse, dtype, device):
        return ("cfft", int(N), norm, bool(inverse), dtype, str(device))

    def _get_cfft_matrices(self, N, norm, dtype, device, inverse=False):
        """Full complex FFT matrices ``(N, N)`` with norm baked in."""
        key = self._cfft_matrix_key(N, norm, inverse, dtype, device)
        cached = self._matrices.get(key)
        if cached is not None:
            return cached
        n_idx = torch.arange(N, dtype=dtype, device=device)
        k_idx = torch.arange(N, dtype=dtype, device=device)
        phase = (2.0 * math.pi / N) * torch.outer(k_idx, n_idx)
        scale = _cfft_norm_scale(norm, N, inverse=inverse)
        C = torch.cos(phase) * scale
        S = torch.sin(phase) * scale
        cached = (C, S, C.abs(), S.abs())
        self._matrices[key] = cached
        return cached

    @staticmethod
    def _move_axis_to_last(t: torch.Tensor, dim: int) -> torch.Tensor:
        return t.movedim(dim, -1)

    @staticmethod
    def _move_axis_from_last(t: torch.Tensor, dim: int) -> torch.Tensor:
        return t.movedim(-1, dim)

    def _apply_cfft_absA_pair(self, r_re, r_im, dim, N, norm, inverse=False):
        _, _, absC, absS = self._get_cfft_matrices(
            N, norm, r_re.dtype, r_re.device, inverse=inverse
        )
        r_re_m = self._move_axis_to_last(r_re, dim)
        r_im_m = self._move_axis_to_last(r_im, dim)
        out_re = torch.einsum("...n,kn->...k", r_re_m, absC) + torch.einsum(
            "...n,kn->...k", r_im_m, absS
        )
        out_im = torch.einsum("...n,kn->...k", r_re_m, absS) + torch.einsum(
            "...n,kn->...k", r_im_m, absC
        )
        return (
            self._move_axis_from_last(out_re, dim),
            self._move_axis_from_last(out_im, dim),
        )

    def _spatial_sizes_list(self):
        if self._exact_sizes is None or self._exact_fft_dims is None:
            raise RuntimeError("FFT sizes not configured.")
        return [self._exact_sizes[d] for d in self._exact_fft_dims]

    def _nd_ifft_axis_scale(self, N: int) -> float:
        """Post-ifft scale per axis matching ``torch.fft.rfftn`` autograd."""
        norm = self._exact_norm
        if norm == "ortho":
            return math.sqrt(N)
        if norm == "forward":
            return 1.0
        if norm == "backward":
            return float(N)
        return float(N)

    def _nd_fft_adjoint_axis_scale(self, N: int) -> float:
        """Per-axis scale for the adjoint of ``torch.fft.ifft`` along one axis."""
        norm = self._exact_norm
        if norm == "ortho":
            return 1.0
        if norm == "forward":
            return float(N)
        return 1.0 / float(N)

    def _nd_rfft_adjoint(self, c_complex: torch.Tensor) -> torch.Tensor:
        """Adjoint of ``rfftn`` on contiguous trailing dims (PyTorch decomposed form)."""
        dim_last = self._exact_fft_dims[-1]
        N_last = self._exact_sizes[dim_last]
        K = N_last // 2 + 1
        t = self._move_axis_to_last(c_complex, dim_last)
        full = torch.zeros(
            *t.shape[:-1],
            N_last,
            dtype=c_complex.dtype,
            device=c_complex.device,
        )
        full[..., :K] = t
        t = self._move_axis_from_last(full, dim_last)
        for dim in reversed(self._exact_fft_dims):
            N = self._exact_sizes[dim]
            t_m = self._move_axis_to_last(t, dim)
            t_m = torch.fft.ifft(t_m, dim=-1) * self._nd_ifft_axis_scale(N)
            t = self._move_axis_from_last(t_m, dim)
        return t.real.contiguous()

    def _nd_irfft_adjoint(self, c_real: torch.Tensor) -> torch.Tensor:
        """Adjoint of ``irfftn`` on contiguous trailing dims; returns a real-pair spectrum.

        PyTorch decomposes ``irfftn`` as ``ifft`` along leading FFT axes followed by
        ``irfft`` on the last axis.  This applies the adjoints in reverse order.
        """
        dim_last = self._exact_fft_dims[-1]
        N_last = self._exact_sizes[dim_last]
        C_inv, S_inv, _, _ = self._get_matrices(
            N_last, self._exact_norm, c_real.dtype, c_real.device
        )
        c_m = self._move_axis_to_last(c_real, dim_last)
        re = torch.einsum("...n,nk->...k", c_m, C_inv)
        im = -torch.einsum("...n,nk->...k", c_m, S_inv)
        re = self._move_axis_from_last(re, dim_last)
        im = self._move_axis_from_last(im, dim_last)
        t = torch.complex(re, im)
        for dim in reversed(self._exact_fft_dims[:-1]):
            N = self._exact_sizes[dim]
            t_m = self._move_axis_to_last(t, dim)
            t_m = torch.fft.fft(t_m, dim=-1, norm=self._exact_norm)
            t_m = t_m * self._nd_fft_adjoint_axis_scale(N)
            t = self._move_axis_from_last(t_m, dim)
        return torch.view_as_real(t.contiguous())

    def configure_1d(self, N, norm=None, dtype=None, device=None):
        """Pre-configure the exact 1D path (for direct test access)."""
        self._exact_active = True
        self._exact_dim = -1
        self._exact_N = int(N)
        self._exact_norm = norm
        self._loose_dims = None
        self._exact_fft_dims = [-1]
        self._exact_sizes = {-1: int(N)}
        if dtype is None:
            dtype = torch.get_default_dtype()
        if device is None:
            device = torch.device("cpu")
        self._warm_matrix_cache(N, norm, dtype, device)

    def _warm_matrix_cache(self, N, norm, dtype, device):  # pragma: no cover - abstract
        raise NotImplementedError

    def _get_matrices(self, N, norm, dtype, device):  # pragma: no cover - abstract
        raise NotImplementedError

    def _configure_from_inputs(self, x_shape, *args):  # pragma: no cover - abstract
        raise NotImplementedError

    def _ensure_configured(self, x_shape, *args):
        if self._exact_active is None:
            self._configure_from_inputs(x_shape, *args)

    def _ensure_configured_from_bound(self, bound_inputs):
        if self._exact_active is not None:
            return
        x = bound_inputs[0]
        if x is None:
            return
        x_shape = getattr(x, "output_shape", None) or _forward_value(x).shape
        self._ensure_configured_from_values(x_shape, bound_inputs)

    def _ensure_configured_from_values(self, x_shape, values):  # pragma: no cover - abstract
        raise NotImplementedError

    def _interval_propagate_fft(self, inputs, normalize_fn, exact_fn, loose_fn):
        x_iv = inputs[0]
        x_lower, x_upper = x_iv[0], x_iv[1]
        lowers = [iv[0] if iv is not None else None for iv in inputs]
        config_args = normalize_fn(*lowers)
        self._ensure_configured(x_lower.shape, *config_args[1:])
        if self._exact_active:
            return exact_fn(x_lower, x_upper, x_iv)
        return loose_fn(x_lower, x_upper, *config_args[1:], x_iv=x_iv)

    def _fft_bound_backward(self, last_lA, last_uA, *inputs, adjoint: Callable):
        if inputs:
            self._ensure_configured_from_bound(list(inputs))
        elif self._exact_active is None:
            raise RuntimeError(
                f"{type(self).__name__}.bound_backward requires graph inputs "
                "or a prior configure_1d call."
            )
        if not self._exact_active:
            raise NotImplementedError(self._loose_backward_msg)
        lA = adjoint(last_lA)
        uA = adjoint(last_uA)
        if not inputs:
            return [(lA, uA)], 0, 0
        n_inputs = len(self.inputs) if self.inputs else len(inputs)
        return [(lA, uA)] + [(None, None)] * (n_inputs - 1), 0, 0


class BoundRFFT(BoundFFTNExact):
    """``aten::fft_rfftn`` / ``aten::ATen[operator="fft_rfftn"]`` via real-pair lowering.

    Accepts both raw JIT ``(x, s, dim, norm)`` and ONNX ``(x, norm, dim, norm_dup)``.

    Contiguous trailing-axis layouts (including 1D last-axis and neuralop 2D/3D
    FNO ``rfftn``) use the exact linear rule with materialized cosine/sine
    matrices.  Non-trailing or non-contiguous axis sets fall back to a
    sum-radius IBP rule without CROWN support.
    """

    _loose_backward_msg = (
        "BoundRFFT.bound_backward is only implemented for the exact ND path; "
        "the legacy loose path is currently IBP-only."
    )

    def _warm_matrix_cache(self, N, norm, dtype, device):
        self._get_matrices(N, norm, dtype, device)

    def _get_matrices(self, N, norm, dtype, device):
        """Materialize ``(C, S, |C|, |S|)`` of shape ``(K, N)`` with norm baked in."""
        key = self._matrix_key(N, norm, dtype, device)
        cached = self._matrices.get(key)
        if cached is not None:
            return cached
        K = N // 2 + 1
        n_idx = torch.arange(N, dtype=dtype, device=device)
        k_idx = torch.arange(K, dtype=dtype, device=device)
        phase = (2.0 * math.pi / N) * torch.outer(k_idx, n_idx)
        scale = _rfft_norm_scale(norm, N)
        C = torch.cos(phase) * scale
        S = torch.sin(phase) * scale
        cached = (C, S, C.abs(), S.abs())
        self._matrices[key] = cached
        return cached

    @staticmethod
    def _normalize_rfftn_inputs(*inputs):
        """Return ``(x, dim, norm)`` for JIT or ONNX ``fft_rfftn`` argument layouts."""
        x = inputs[0]
        if len(inputs) < 2:
            return x, None, None
        n1 = _norm_arg(inputs[1])
        if isinstance(n1, str):
            dim = inputs[2] if len(inputs) > 2 else None
            norm = n1
            if len(inputs) > 3:
                n3 = _norm_arg(inputs[3])
                if isinstance(n3, str):
                    norm = norm or n3
            return x, dim, norm
        dim = inputs[2] if len(inputs) > 2 else None
        norm = inputs[3] if len(inputs) > 3 else None
        return x, dim, norm

    def _configure_from_inputs(self, x_shape, dim_arg, norm_arg):
        ndim = len(x_shape)
        dims = _resolve_fft_dims(dim_arg, ndim)
        norm_str = _parse_norm_str(norm_arg)
        if _can_exact_nd(dims, ndim):
            self._exact_active = True
            self._exact_fft_dims = _normalize_exact_fft_dims(dims, ndim)
            self._exact_sizes = {
                d: int(x_shape[d]) for d in self._exact_fft_dims
            }
            self._exact_norm = norm_str
            self._loose_dims = None
            if len(self._exact_fft_dims) == 1:
                self._exact_dim = self._exact_fft_dims[0]
                self._exact_N = self._exact_sizes[self._exact_dim]
            else:
                self._exact_dim = None
                self._exact_N = None
        else:
            self._exact_active = False
            self._exact_dim = None
            self._exact_N = None
            self._exact_fft_dims = None
            self._exact_sizes = None
            self._exact_norm = norm_str
            self._loose_dims = dims

    def _ensure_configured_from_values(self, x_shape, values):
        _, dim, norm = self._normalize_rfftn_inputs(
            *[_forward_value(inp) for inp in values]
        )
        self._ensure_configured(x_shape, dim, norm)

    def apply_A(self, x: torch.Tensor) -> torch.Tensor:
        if self._exact_fft_dims is None:
            raise RuntimeError("BoundRFFT.apply_A called before configuration.")
        if not x.is_floating_point():
            x = x.to(torch.get_default_dtype())
        if self._is_scalar_axis_exact:
            assert x.shape[self._exact_dim] == self._exact_N
            z = torch.fft.rfft(
                x, n=self._exact_N, dim=self._exact_dim, norm=self._exact_norm
            )
        else:
            z = torch.fft.rfftn(
                x, dim=self._exact_fft_dims, norm=self._exact_norm
            )
        return torch.view_as_real(z).contiguous()

    def apply_AT(self, c: torch.Tensor) -> torch.Tensor:
        if self._exact_fft_dims is None:
            raise RuntimeError("BoundRFFT.apply_AT called before configuration.")
        if c.is_complex():
            c_complex = c
        elif c.shape[-1] == 2:
            c_complex = self.pair_to_complex(c)
        else:
            raise AssertionError(
                "BoundRFFT.apply_AT expects a complex A or a real-pair A "
                "with trailing size-2 axis."
            )
        if self._is_scalar_axis_exact:
            C, S, _, _ = self._get_matrices(
                self._exact_N, self._exact_norm, c.dtype, c.device
            )
            c_re = c_complex.real
            c_im = c_complex.imag
            return torch.einsum("...k,kn->...n", c_re, C) - torch.einsum(
                "...k,kn->...n", c_im, S
            )
        return self._nd_rfft_adjoint(c_complex)

    def apply_absA_radius(self, r: torch.Tensor) -> torch.Tensor:
        if self._exact_fft_dims is None:
            raise RuntimeError(
                "BoundRFFT.apply_absA_radius called before configuration."
            )
        if self._is_scalar_axis_exact:
            assert r.shape[self._exact_dim] == self._exact_N
            _, _, absC, absS = self._get_matrices(
                self._exact_N, self._exact_norm, r.dtype, r.device
            )
            r_m = self._move_axis_to_last(r, self._exact_dim)
            out_re = torch.einsum("...n,kn->...k", r_m, absC)
            out_im = torch.einsum("...n,kn->...k", r_m, absS)
            out_re = self._move_axis_from_last(out_re, self._exact_dim)
            out_im = self._move_axis_from_last(out_im, self._exact_dim)
            return torch.stack([out_re, out_im], dim=-1)

        r_re = r
        r_im = torch.zeros_like(r)
        for dim in self._exact_fft_dims[:-1]:
            N = self._exact_sizes[dim]
            r_re, r_im = self._apply_cfft_absA_pair(
                r_re, r_im, dim, N, self._exact_norm, inverse=False
            )
        dim_last = self._exact_fft_dims[-1]
        N = self._exact_sizes[dim_last]
        _, _, absC, absS = self._get_matrices(
            N, self._exact_norm, r.dtype, r.device
        )
        r_re_m = self._move_axis_to_last(r_re, dim_last)
        r_im_m = self._move_axis_to_last(r_im, dim_last)
        out_re = torch.einsum("...n,kn->...k", r_re_m, absC) + torch.einsum(
            "...n,kn->...k", r_im_m, absS
        )
        out_im = torch.einsum("...n,kn->...k", r_re_m, absS) + torch.einsum(
            "...n,kn->...k", r_im_m, absC
        )
        out_re = self._move_axis_from_last(out_re, dim_last)
        out_im = self._move_axis_from_last(out_im, dim_last)
        return torch.stack([out_re, out_im], dim=-1)

    def forward(self, *inputs):
        x, dim, norm = self._normalize_rfftn_inputs(*inputs)
        self._configure_from_inputs(x.shape, dim, norm)
        if self._exact_active:
            if self._is_scalar_axis_exact:
                return torch.fft.rfft(
                    x, n=self._exact_N, dim=self._exact_dim, norm=self._exact_norm
                )
            return torch.fft.rfftn(
                x, dim=self._exact_fft_dims, norm=self._exact_norm
            )
        dim_list = _tensor_to_int_list(dim) if dim is not None else None
        return torch.fft.rfftn(x, dim=dim_list, norm=_norm_arg(norm))

    def interval_propagate(self, *inputs):
        return self._interval_propagate_fft(
            inputs,
            self._normalize_rfftn_inputs,
            self._interval_propagate_exact,
            self._interval_propagate_loose,
        )

    def _interval_propagate_exact(self, x_lower, x_upper, x_iv):
        if x_lower.is_complex() or x_upper.is_complex():
            assert torch.all(x_lower.imag == 0) and torch.all(x_upper.imag == 0), (
                "BoundRFFT exact path requires a real-valued input interval."
            )
            x_lower = x_lower.real.contiguous()
            x_upper = x_upper.real.contiguous()
        y_pair_l, y_pair_u = self.pair_interval_propagate(x_lower, x_upper)
        lower = self.pair_to_complex(y_pair_l)
        upper = self.pair_to_complex(y_pair_u)
        return Interval.make_interval(lower, upper, x_iv)

    def _interval_propagate_loose(self, x_lower, x_upper, dim_arg, norm_arg, *, x_iv):
        """Sum-radius IBP rule for non-trailing or non-contiguous RFFT layouts."""
        dim_list = _tensor_to_int_list(dim_arg) if dim_arg is not None else None
        norm_str = _norm_arg(norm_arg)
        center, real_radius, _imag_radius = _complex_center_radius(x_lower, x_upper)
        out_center = torch.fft.rfftn(center, dim=dim_list, norm=norm_str)
        dims = dim_list if dim_list is not None else list(range(center.ndim))
        n = _fft_numel(center.shape, dims)
        radius = _sum_radius_for_fft(
            real_radius, dims, _fft_scale(norm_str, n), out_center
        )
        lower, upper = _complex_interval(out_center, radius, radius)
        return Interval.make_interval(lower, upper, x_iv)

    def bound_backward(self, last_lA, last_uA, *inputs, **kwargs):
        if inputs:
            bound_inputs = inputs
        elif kwargs:
            bound_inputs = (
                kwargs.get("x"),
                kwargs.get("_s"),
                kwargs.get("dim"),
                kwargs.get("norm"),
            )
        else:
            bound_inputs = ()

        def _adjoint(A):
            if A is None:
                return None
            if A.is_complex():
                pair = self.complex_to_pair(A)
            else:
                assert A.shape[-1] == 2, (
                    "BoundRFFT.bound_backward expects a complex A or a real-pair A "
                    "with trailing size-2 axis."
                )
                pair = A
            return self.apply_AT(pair)

        return self._fft_bound_backward(
            last_lA, last_uA, *bound_inputs, adjoint=_adjoint
        )


class BoundIRFFT(BoundFFTNExact):
    """``aten::fft_irfftn`` / ``aten::ATen[operator="fft_irfftn"]`` via real-pair lowering.

    Forward: complex spectrum ``(..., K)`` -> real signal ``(..., N)`` with
    ``K = N // 2 + 1``.  Under the pair lowering the input is ``(..., K, 2)``
    where the trailing axis holds ``[Re, Im]`` per frequency bin.

    Contiguous trailing-axis layouts use the exact staged rule (including 1D
    last-axis and neuralop 2D/3D FNO ``irfftn``).  Other layouts fall back to a
    sum-radius IBP rule without CROWN support.
    """

    _loose_backward_msg = (
        "BoundIRFFT.bound_backward is only implemented for the exact ND path; "
        "the legacy loose path is currently IBP-only."
    )

    def _warm_matrix_cache(self, N, norm, dtype, device):
        self._get_matrices(N, norm, dtype, device)

    def _get_matrices(self, N, norm, dtype, device):
        """Materialize ``(C_inv, S_inv, |C_inv|, |S_inv|)`` of shape ``(N, K)``."""
        key = self._matrix_key(N, norm, dtype, device)
        cached = self._matrices.get(key)
        if cached is not None:
            return cached
        K = N // 2 + 1
        n_idx = torch.arange(N, dtype=dtype, device=device)
        k_idx = torch.arange(K, dtype=dtype, device=device)
        phase = (2.0 * math.pi / N) * torch.outer(n_idx, k_idx)
        scale = _irfft_norm_scale(norm, N)
        w = torch.full((K,), 2.0, dtype=dtype, device=device)
        w[0] = 1.0
        if N % 2 == 0:
            w[-1] = 1.0
        gain = scale * w
        C_inv = torch.cos(phase) * gain
        S_inv = torch.sin(phase) * gain
        cached = (C_inv, S_inv, C_inv.abs(), S_inv.abs())
        self._matrices[key] = cached
        return cached

    @staticmethod
    def _normalize_irfftn_inputs(*inputs):
        """``(x, s, dim, norm)`` for both JIT and ONNX IRFFT argument layouts."""
        x = inputs[0]
        s = inputs[1] if len(inputs) > 1 else None
        dim = inputs[2] if len(inputs) > 2 else None
        norm = inputs[3] if len(inputs) > 3 else None
        return x, s, dim, norm

    def _configure_from_inputs(self, x_shape, s_arg, dim_arg, norm_arg):
        ndim = len(x_shape)
        dims = _resolve_fft_dims(dim_arg, ndim)
        norm_str = _parse_norm_str(norm_arg)
        s_list = _tensor_to_int_list(s_arg) if s_arg is not None else None
        if _can_exact_nd(dims, ndim):
            self._exact_active = True
            self._exact_fft_dims = _normalize_exact_fft_dims(dims, ndim)
            if s_list and len(s_list) == len(self._exact_fft_dims):
                self._exact_sizes = {
                    d: int(s_list[i])
                    for i, d in enumerate(self._exact_fft_dims)
                }
            elif s_list:
                self._exact_sizes = {
                    d: int(s_list[-1]) if d == self._exact_fft_dims[-1] else int(x_shape[d])
                    for d in self._exact_fft_dims
                }
            else:
                self._exact_sizes = {
                    d: int(x_shape[d]) for d in self._exact_fft_dims
                }
            self._exact_norm = norm_str
            self._loose_dims = None
            if len(self._exact_fft_dims) == 1:
                self._exact_dim = self._exact_fft_dims[0]
                if s_list:
                    self._exact_N = int(s_list[-1])
                else:
                    K = int(x_shape[self._exact_dim])
                    self._exact_N = 2 * (K - 1)
                self._exact_sizes[self._exact_dim] = self._exact_N
            else:
                self._exact_dim = None
                self._exact_N = None
        else:
            self._exact_active = False
            self._exact_dim = None
            self._exact_N = None
            self._exact_fft_dims = None
            self._exact_sizes = None
            self._exact_norm = norm_str
            self._loose_dims = dims

    def _ensure_configured_from_values(self, x_shape, values):
        _, s, dim, norm = self._normalize_irfftn_inputs(
            *[_forward_value(inp) for inp in values]
        )
        self._ensure_configured(x_shape, s, dim, norm)

    def apply_A(self, z_pair: torch.Tensor) -> torch.Tensor:
        if self._exact_fft_dims is None:
            raise RuntimeError("BoundIRFFT.apply_A called before configuration.")
        assert z_pair.shape[-1] == 2, "apply_A expects real-pair last-axis of size 2."
        if not z_pair.is_floating_point():
            z_pair = z_pair.to(torch.get_default_dtype())
        z_complex = self.pair_to_complex(z_pair)
        if self._is_scalar_axis_exact:
            return torch.fft.irfft(
                z_complex, n=self._exact_N, dim=self._exact_dim, norm=self._exact_norm
            )
        return torch.fft.irfftn(
            z_complex,
            s=self._spatial_sizes_list(),
            dim=self._exact_fft_dims,
            norm=self._exact_norm,
        )

    def apply_AT(self, c: torch.Tensor) -> torch.Tensor:
        if self._exact_fft_dims is None:
            raise RuntimeError("BoundIRFFT.apply_AT called before configuration.")
        assert not c.is_complex(), (
            "BoundIRFFT.apply_AT expects a real cotangent (IRFFT output is real)."
        )
        if self._is_scalar_axis_exact:
            assert c.shape[self._exact_dim] == self._exact_N
            C_inv, S_inv, _, _ = self._get_matrices(
                self._exact_N, self._exact_norm, c.dtype, c.device
            )
            c_m = self._move_axis_to_last(c, self._exact_dim)
            re = torch.einsum("...n,nk->...k", c_m, C_inv)
            im = -torch.einsum("...n,nk->...k", c_m, S_inv)
            re = self._move_axis_from_last(re, self._exact_dim)
            im = self._move_axis_from_last(im, self._exact_dim)
            return torch.stack([re, im], dim=-1)
        return self._nd_irfft_adjoint(c)

    def apply_absA_radius(self, r_pair: torch.Tensor) -> torch.Tensor:
        if self._exact_fft_dims is None:
            raise RuntimeError(
                "BoundIRFFT.apply_absA_radius called before configuration."
            )
        assert r_pair.shape[-1] == 2, "apply_absA_radius expects real-pair input."
        r_re = r_pair[..., 0]
        r_im = r_pair[..., 1]
        if self._is_scalar_axis_exact:
            _, _, absC, absS = self._get_matrices(
                self._exact_N, self._exact_norm, r_pair.dtype, r_pair.device
            )
            r_re_m = self._move_axis_to_last(r_re, self._exact_dim)
            r_im_m = self._move_axis_to_last(r_im, self._exact_dim)
            out = torch.einsum("...k,nk->...n", r_re_m, absC) + torch.einsum(
                "...k,nk->...n", r_im_m, absS
            )
            return self._move_axis_from_last(out, self._exact_dim)

        dim_last = self._exact_fft_dims[-1]
        N = self._exact_sizes[dim_last]
        _, _, absC, absS = self._get_matrices(
            N, self._exact_norm, r_pair.dtype, r_pair.device
        )
        r_re_m = self._move_axis_to_last(r_re, dim_last)
        r_im_m = self._move_axis_to_last(r_im, dim_last)
        out = torch.einsum("...k,nk->...n", r_re_m, absC) + torch.einsum(
            "...k,nk->...n", r_im_m, absS
        )
        out = self._move_axis_from_last(out, dim_last)
        r_re = out
        r_im = torch.zeros_like(out)
        for dim in reversed(self._exact_fft_dims[:-1]):
            N = self._exact_sizes[dim]
            r_re, r_im = self._apply_cfft_absA_pair(
                r_re, r_im, dim, N, self._exact_norm, inverse=True
            )
        return r_re

    def forward(self, *inputs):
        x, s, dim, norm = self._normalize_irfftn_inputs(*inputs)
        self._configure_from_inputs(x.shape, s, dim, norm)
        if self._exact_active:
            if self._is_scalar_axis_exact:
                return torch.fft.irfft(
                    x, n=self._exact_N, dim=self._exact_dim, norm=self._exact_norm
                )
            return torch.fft.irfftn(
                x,
                s=self._spatial_sizes_list(),
                dim=self._exact_fft_dims,
                norm=self._exact_norm,
            )
        s_list = _tensor_to_int_list(s) if s is not None else None
        dim_list = _tensor_to_int_list(dim) if dim is not None else None
        return torch.fft.irfftn(x, s=s_list, dim=dim_list, norm=_norm_arg(norm))

    def interval_propagate(self, *inputs):
        return self._interval_propagate_fft(
            inputs,
            self._normalize_irfftn_inputs,
            self._interval_propagate_exact,
            self._interval_propagate_loose,
        )

    def _interval_propagate_exact(self, x_lower, x_upper, x_iv):
        if x_lower.is_complex():
            pair_lower = self.complex_to_pair(x_lower)
            pair_upper = self.complex_to_pair(x_upper)
        else:
            assert x_lower.shape[-1] == 2, (
                "BoundIRFFT exact path expects a complex or real-pair input interval."
            )
            pair_lower = x_lower
            pair_upper = x_upper
        y_lower, y_upper = self.pair_interval_propagate(pair_lower, pair_upper)
        return Interval.make_interval(y_lower, y_upper, x_iv)

    def _interval_propagate_loose(
        self, x_lower, x_upper, s_arg, dim_arg, norm_arg, *, x_iv
    ):
        """Sum-radius IBP rule for non-trailing or non-contiguous IRFFT layouts."""
        dim_list = _tensor_to_int_list(dim_arg) if dim_arg is not None else None
        s_list = _tensor_to_int_list(s_arg) if s_arg is not None else None
        norm_str = _norm_arg(norm_arg)
        center, real_radius, imag_radius = _complex_center_radius(x_lower, x_upper)
        out_center = torch.fft.irfftn(
            center, s=s_list, dim=dim_list, norm=norm_str
        )
        dims = dim_list if dim_list is not None else list(range(out_center.ndim))
        if s_list is not None:
            n = 1
            for size in s_list:
                n *= int(size)
        else:
            n = _fft_numel(out_center.shape, dims)
        radius = _sum_radius_for_fft(
            real_radius + imag_radius,
            dims,
            _fft_scale(norm_str, n, inverse=True),
            out_center,
        )
        return Interval.make_interval(out_center - radius, out_center + radius, x_iv)

    def bound_backward(self, last_lA, last_uA, *inputs, **kwargs):
        if inputs:
            bound_inputs = inputs
        elif kwargs:
            bound_inputs = (
                kwargs.get("x"),
                kwargs.get("s"),
                kwargs.get("dim"),
                kwargs.get("norm"),
            )
        else:
            bound_inputs = ()

        def _adjoint(A):
            if A is None:
                return None
            assert not A.is_complex(), (
                "BoundIRFFT.bound_backward expects a real cotangent "
                "(IRFFT output is real)."
            )
            pair = self.apply_AT(A)
            return self.pair_to_complex(pair)

        return self._fft_bound_backward(
            last_lA, last_uA, *bound_inputs, adjoint=_adjoint
        )
