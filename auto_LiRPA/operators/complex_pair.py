#########################################################################
##  Real-pair bounds for complex linear ops (RFFT etc.)                ##
#########################################################################
"""Exact CROWN/IBP rules for complex-valued ops lowered to a real ``[..., 2]`` pair.

Any operator that is linear over the complex numbers (RFFT, IRFFT, fixed-weight
spectral multiplication, slice, scatter) becomes a *real* linear map ``y = A x``
when the complex tensor ``z = a + i b`` is represented as the real-pair tensor
``stack([a, b], dim=-1)``.  CROWN and IBP rules are then exact:

* center: ``y0 = A x0``;
* radius: ``r_y = |A| r_x``;
* backward: ``c_x = A^T c_y``.

This module provides :class:`BoundLinearComplexPair` (an abstract base
encapsulating the three primitives ``apply_A`` / ``apply_AT`` /
``apply_absA_radius``) and the concrete subclasses :class:`BoundRFFT`,
:class:`BoundIRFFT`, and :class:`BoundComplexEinsum`.  Their 1D last-axis
FFT paths use the exact rule; ND FFTs keep a sum-radius fallback until a
follow-up PR generalizes the exact rule to multi-axis FFTs.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

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
from .base import Bound, Interval


def _resolve_fft_dims(dim_arg, ndim):
    """JIT ``dim`` operand -> ``list[int]`` of positive axes (default: all axes)."""
    dim_list = _tensor_to_int_list(dim_arg)
    if dim_list is None:
        return list(range(ndim))
    return [d % ndim for d in dim_list]


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


class BoundLinearComplexPair(Bound):
    """Abstract base for ops that are linear under the real-pair lowering.

    Subclasses implement three primitives operating on the canonical last-axis
    layout:

    * ``apply_A(x)``: real ``(..., N)`` -> real-pair ``(..., K, 2)``.
    * ``apply_AT(c)``: real-pair ``(..., K, 2)`` -> real ``(..., N)``.
    * ``apply_absA_radius(r)``: real radius ``(..., N)`` -> real-pair radius
      ``(..., K, 2)`` equal to ``stack([|C| r, |S| r], dim=-1)``.

    The base provides helper methods that compose these primitives into the
    standard IBP/CROWN rules and convert between native complex and real-pair
    representations at the LiRPA graph boundary.
    """

    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.is_complex_pair = True
        self.pair_layout = "last_dim"
        self.use_default_ibp = False

    def apply_A(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def apply_AT(self, c: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def apply_absA_radius(self, r: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def pair_interval_propagate(
        self, x_lower: torch.Tensor, x_upper: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(y_pair_lower, y_pair_upper)`` real-pair tensors."""
        center = (x_lower + x_upper) * 0.5
        radius = (x_upper - x_lower) * 0.5
        y_center = self.apply_A(center)
        y_radius = self.apply_absA_radius(radius)
        return y_center - y_radius, y_center + y_radius

    @staticmethod
    def pair_to_complex(pair: torch.Tensor) -> torch.Tensor:
        """``(..., 2)`` real -> ``(...)`` complex."""
        return torch.view_as_complex(pair.contiguous())

    @staticmethod
    def complex_to_pair(z: torch.Tensor) -> torch.Tensor:
        """``(...)`` complex -> ``(..., 2)`` real."""
        return torch.view_as_real(z).contiguous()


class BoundRFFT(BoundLinearComplexPair):
    """``aten::fft_rfftn`` / ``aten::ATen[operator="fft_rfftn"]`` via real-pair lowering.

    Accepts both raw JIT ``(x, s, dim, norm)`` and ONNX ``(x, norm, dim, norm_dup)``.

    The 1D path (FFT taken along the last axis only) uses the exact linear
    rule ``y_pair = [C; -S] x`` with materialized ``C[k, n] = s cos(2π k n / N)``
    and ``S[k, n] = s sin(2π k n / N)`` and ``torch.fft.rfft`` for the forward.
    Multi-axis or off-last-axis FFTs fall back to the legacy sum-radius rule
    absorbed from the previous ``BoundAtenJitFftRfftn``.
    """

    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self._matrices = {}
        self._exact_N: Optional[int] = None
        self._exact_norm: Optional[str] = None
        self._exact_dim: Optional[int] = None
        self._exact_active: Optional[bool] = None
        # ND fallback bookkeeping.
        self._loose_dims: Optional[list] = None

    @staticmethod
    def _matrix_key(N, norm, dtype, device):
        return (int(N), norm, dtype, str(device))

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

    def _can_exact_1d(self, dims, ndim):
        """Exact path: a single FFT axis equal to the last input axis."""
        return len(dims) == 1 and dims[0] == ndim - 1

    @staticmethod
    def _normalize_rfftn_inputs(*inputs):
        """Return ``(x, dim, norm)`` for JIT or ONNX ``fft_rfftn`` argument layouts."""
        x = inputs[0]
        if len(inputs) < 2:
            return x, None, None
        n1 = _norm_arg(inputs[1])
        if isinstance(n1, str):
            # ONNX: (x, norm, dim, norm_dup?)
            dim = inputs[2] if len(inputs) > 2 else None
            norm = n1
            if len(inputs) > 3:
                n3 = _norm_arg(inputs[3])
                if isinstance(n3, str):
                    norm = norm or n3
            return x, dim, norm
        # JIT: (x, s, dim, norm)
        dim = inputs[2] if len(inputs) > 2 else None
        norm = inputs[3] if len(inputs) > 3 else None
        return x, dim, norm

    def _configure_from_inputs(self, x_shape, dim_arg, norm_arg):
        ndim = len(x_shape)
        dims = _resolve_fft_dims(dim_arg, ndim)
        norm_str = _norm_arg(norm_arg)
        if not (isinstance(norm_str, str) or norm_str is None):
            norm_str = None
        if self._can_exact_1d(dims, ndim):
            self._exact_active = True
            self._exact_dim = dims[0]
            self._exact_N = int(x_shape[self._exact_dim])
            self._exact_norm = norm_str
            self._loose_dims = None
        else:
            self._exact_active = False
            self._exact_dim = None
            self._exact_N = None
            self._exact_norm = norm_str
            self._loose_dims = dims

    def configure_1d(self, N, norm=None, dtype=None, device=None):
        """Pre-configure the exact 1D path (for direct test access)."""
        self._exact_active = True
        self._exact_dim = -1
        self._exact_N = int(N)
        self._exact_norm = norm
        if dtype is None:
            dtype = torch.get_default_dtype()
        if device is None:
            device = torch.device("cpu")
        self._get_matrices(N, norm, dtype, device)

    def dense_real_pair_matrix(
        self, N=None, norm=None, dtype=None, device=None
    ) -> torch.Tensor:
        """Return ``A_pair[k, comp, n]`` of shape ``(K, 2, N)`` with comp 0 = C, comp 1 = -S."""
        if N is None:
            N = self._exact_N
        if norm is None:
            norm = self._exact_norm
        if N is None:
            raise RuntimeError(
                "dense_real_pair_matrix requires an N; call configure_1d first."
            )
        if dtype is None:
            dtype = torch.get_default_dtype()
        if device is None:
            device = torch.device("cpu")
        C, S, _, _ = self._get_matrices(N, norm, dtype, device)
        return torch.stack([C, -S], dim=-2)

    def apply_A(self, x: torch.Tensor) -> torch.Tensor:
        if self._exact_N is None:
            raise RuntimeError("BoundRFFT.apply_A called before exact-1D configuration.")
        assert x.shape[-1] == self._exact_N, (
            f"apply_A: last-axis length {x.shape[-1]} != configured N {self._exact_N}"
        )
        if not x.is_floating_point():
            x = x.to(torch.get_default_dtype())
        z = torch.fft.rfft(x, n=self._exact_N, dim=-1, norm=self._exact_norm)
        return torch.view_as_real(z).contiguous()

    def apply_AT(self, c: torch.Tensor) -> torch.Tensor:
        if self._exact_N is None:
            raise RuntimeError("BoundRFFT.apply_AT called before exact-1D configuration.")
        assert c.shape[-1] == 2, "apply_AT expects real-pair last-axis of size 2."
        C, S, _, _ = self._get_matrices(self._exact_N, self._exact_norm, c.dtype, c.device)
        c_re = c[..., 0]
        c_im = c[..., 1]
        return torch.einsum("...k,kn->...n", c_re, C) - torch.einsum("...k,kn->...n", c_im, S)

    def apply_absA_radius(self, r: torch.Tensor) -> torch.Tensor:
        if self._exact_N is None:
            raise RuntimeError(
                "BoundRFFT.apply_absA_radius called before exact-1D configuration."
            )
        assert r.shape[-1] == self._exact_N, (
            f"apply_absA_radius: last-axis length {r.shape[-1]} != configured N {self._exact_N}"
        )
        _, _, absC, absS = self._get_matrices(
            self._exact_N, self._exact_norm, r.dtype, r.device
        )
        out_re = torch.einsum("...n,kn->...k", r, absC)
        out_im = torch.einsum("...n,kn->...k", r, absS)
        return torch.stack([out_re, out_im], dim=-1)

    def forward(self, *inputs):
        x, dim, norm = self._normalize_rfftn_inputs(*inputs)
        self._configure_from_inputs(x.shape, dim, norm)
        if self._exact_active:
            return torch.fft.rfft(
                x, n=self._exact_N, dim=self._exact_dim, norm=self._exact_norm
            )
        dim_list = _tensor_to_int_list(dim) if dim is not None else None
        return torch.fft.rfftn(x, dim=dim_list, norm=_norm_arg(norm))

    def interval_propagate(self, *inputs):
        x_iv = inputs[0]
        x_lower, x_upper = x_iv[0], x_iv[1]
        lowers = [iv[0] if iv is not None else None for iv in inputs]
        _, dim_arg, norm_arg = self._normalize_rfftn_inputs(*lowers)
        if self._exact_active is None:
            self._configure_from_inputs(x_lower.shape, dim_arg, norm_arg)
        if self._exact_active:
            return self._interval_propagate_exact(x_lower, x_upper, x_iv)
        return self._interval_propagate_loose(x_lower, x_upper, dim_arg, norm_arg, x_iv)

    def _interval_propagate_exact(self, x_lower, x_upper, x_iv):
        if x_lower.is_complex() or x_upper.is_complex():
            # RFFT consumes a real input; defensively coerce a complex interval
            # with zero imaginary part to its real part rather than failing.
            assert torch.all(x_lower.imag == 0) and torch.all(x_upper.imag == 0), (
                "BoundRFFT exact path requires a real-valued input interval."
            )
            x_lower = x_lower.real.contiguous()
            x_upper = x_upper.real.contiguous()
        y_pair_l, y_pair_u = self.pair_interval_propagate(x_lower, x_upper)
        lower = self.pair_to_complex(y_pair_l)
        upper = self.pair_to_complex(y_pair_u)
        return Interval.make_interval(lower, upper, x_iv)

    def _interval_propagate_loose(self, x_lower, x_upper, dim_arg, norm_arg, x_iv):
        """Legacy sum-radius IBP rule for multi-axis or off-last-axis RFFT."""
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
            bound_inputs = list(inputs)
        else:
            bound_inputs = [
                kwargs.get("x"),
                kwargs.get("_s"),
                kwargs.get("dim"),
                kwargs.get("norm"),
            ]

        def _fv(node):
            if node is None:
                return None
            return getattr(node, "forward_value", node)

        x = bound_inputs[0]
        _, dim, norm = self._normalize_rfftn_inputs(
            *[_fv(inp) for inp in bound_inputs]
        )
        if self._exact_active is None:
            x_shape = getattr(x, "output_shape", None) or _fv(x).shape
            self._configure_from_inputs(x_shape, dim, norm)
        if not self._exact_active:
            raise NotImplementedError(
                "BoundRFFT.bound_backward is only implemented for the exact 1D path; "
                "the multi-axis loose path is currently IBP-only."
            )

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
            # ``apply_AT`` returns a real tensor (RFFT input is real); leave it.
            return self.apply_AT(pair)

        lA = _adjoint(last_lA)
        uA = _adjoint(last_uA)
        # Only the data input ``x`` carries an A coefficient.
        n_inputs = len(self.inputs) if self.inputs else len(bound_inputs)
        return [(lA, uA)] + [(None, None)] * (n_inputs - 1), 0, 0


class BoundIRFFT(BoundLinearComplexPair):
    """``aten::fft_irfftn`` / ``aten::ATen[operator="fft_irfftn"]`` via real-pair lowering.

    Forward: complex spectrum ``(..., K)`` -> real signal ``(..., N)`` with
    ``K = N // 2 + 1``.  Under the pair lowering the input is ``(..., K, 2)``
    where the trailing axis holds ``[Re, Im]`` per frequency bin.

    The 1D last-axis linear map is
        y_n = sum_k w_k s [Re(z_k) cos(2 pi k n / N) - Im(z_k) sin(2 pi k n / N)]
    with the Hermitian-unfolding weights ``w_0 = 1``, ``w_k = 2`` for
    ``0 < k < N/2``, ``w_{N/2} = 1`` when ``N`` is even, and inverse-direction
    scale ``s`` controlled by ``norm``.  Materialized matrices of shape
    ``(N, K)``:
        C_inv[n, k] = w_k * s * cos(2 pi k n / N),
        S_inv[n, k] = w_k * s * sin(2 pi k n / N).

    Multi-axis or off-last-axis IRFFTs fall back to the legacy sum-radius rule
    (kept as a stopgap; the exact rule extends to ND in a follow-up PR).
    """

    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self._matrices = {}
        self._exact_N: Optional[int] = None
        self._exact_norm: Optional[str] = None
        self._exact_dim: Optional[int] = None
        self._exact_active: Optional[bool] = None
        self._loose_dims: Optional[list] = None

    @staticmethod
    def _matrix_key(N, norm, dtype, device):
        return (int(N), norm, dtype, str(device))

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

    def _can_exact_1d(self, dims, ndim):
        return len(dims) == 1 and dims[0] == ndim - 1

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
        norm_str = _norm_arg(norm_arg)
        if not (isinstance(norm_str, str) or norm_str is None):
            norm_str = None
        if self._can_exact_1d(dims, ndim):
            self._exact_active = True
            self._exact_dim = dims[0]
            s_list = _tensor_to_int_list(s_arg) if s_arg is not None else None
            if s_list:
                self._exact_N = int(s_list[-1])
            else:
                K = int(x_shape[self._exact_dim])
                self._exact_N = 2 * (K - 1)
            self._exact_norm = norm_str
            self._loose_dims = None
        else:
            self._exact_active = False
            self._exact_dim = None
            self._exact_N = None
            self._exact_norm = norm_str
            self._loose_dims = dims

    def configure_1d(self, N, norm=None, dtype=None, device=None):
        """Pre-configure the exact 1D path (for direct test access)."""
        self._exact_active = True
        self._exact_dim = -1
        self._exact_N = int(N)
        self._exact_norm = norm
        if dtype is None:
            dtype = torch.get_default_dtype()
        if device is None:
            device = torch.device("cpu")
        self._get_matrices(N, norm, dtype, device)

    def dense_real_pair_matrix(
        self, N=None, norm=None, dtype=None, device=None
    ) -> torch.Tensor:
        """Return ``A_pair[n, k, comp]`` of shape ``(N, K, 2)`` with comp 0 = C_inv, comp 1 = -S_inv."""
        if N is None:
            N = self._exact_N
        if norm is None:
            norm = self._exact_norm
        if N is None:
            raise RuntimeError(
                "dense_real_pair_matrix requires an N; call configure_1d first."
            )
        if dtype is None:
            dtype = torch.get_default_dtype()
        if device is None:
            device = torch.device("cpu")
        C_inv, S_inv, _, _ = self._get_matrices(N, norm, dtype, device)
        return torch.stack([C_inv, -S_inv], dim=-1)

    def apply_A(self, z_pair: torch.Tensor) -> torch.Tensor:
        if self._exact_N is None:
            raise RuntimeError("BoundIRFFT.apply_A called before exact-1D configuration.")
        assert z_pair.shape[-1] == 2, "apply_A expects real-pair last-axis of size 2."
        if not z_pair.is_floating_point():
            z_pair = z_pair.to(torch.get_default_dtype())
        z_complex = self.pair_to_complex(z_pair)
        return torch.fft.irfft(
            z_complex, n=self._exact_N, dim=-1, norm=self._exact_norm
        )

    def apply_AT(self, c: torch.Tensor) -> torch.Tensor:
        if self._exact_N is None:
            raise RuntimeError("BoundIRFFT.apply_AT called before exact-1D configuration.")
        assert c.shape[-1] == self._exact_N, (
            f"apply_AT: last-axis length {c.shape[-1]} != configured N {self._exact_N}"
        )
        C_inv, S_inv, _, _ = self._get_matrices(
            self._exact_N, self._exact_norm, c.dtype, c.device
        )
        re = torch.einsum("...n,nk->...k", c, C_inv)
        im = -torch.einsum("...n,nk->...k", c, S_inv)
        return torch.stack([re, im], dim=-1)

    def apply_absA_radius(self, r_pair: torch.Tensor) -> torch.Tensor:
        if self._exact_N is None:
            raise RuntimeError(
                "BoundIRFFT.apply_absA_radius called before exact-1D configuration."
            )
        assert r_pair.shape[-1] == 2, "apply_absA_radius expects real-pair input."
        _, _, absC, absS = self._get_matrices(
            self._exact_N, self._exact_norm, r_pair.dtype, r_pair.device
        )
        r_re = r_pair[..., 0]
        r_im = r_pair[..., 1]
        out = torch.einsum("...k,nk->...n", r_re, absC) + torch.einsum(
            "...k,nk->...n", r_im, absS
        )
        return out

    def forward(self, *inputs):
        x, s, dim, norm = self._normalize_irfftn_inputs(*inputs)
        self._configure_from_inputs(x.shape, s, dim, norm)
        if self._exact_active:
            return torch.fft.irfft(
                x, n=self._exact_N, dim=self._exact_dim, norm=self._exact_norm
            )
        s_list = _tensor_to_int_list(s) if s is not None else None
        dim_list = _tensor_to_int_list(dim) if dim is not None else None
        return torch.fft.irfftn(x, s=s_list, dim=dim_list, norm=_norm_arg(norm))

    def interval_propagate(self, *inputs):
        x_iv = inputs[0]
        x_lower, x_upper = x_iv[0], x_iv[1]
        lowers = [iv[0] if iv is not None else None for iv in inputs]
        _, s_arg, dim_arg, norm_arg = self._normalize_irfftn_inputs(*lowers)
        if self._exact_active is None:
            self._configure_from_inputs(x_lower.shape, s_arg, dim_arg, norm_arg)
        if self._exact_active:
            return self._interval_propagate_exact(x_lower, x_upper, x_iv)
        return self._interval_propagate_loose(
            x_lower, x_upper, s_arg, dim_arg, norm_arg, x_iv
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
        self, x_lower, x_upper, s_arg, dim_arg, norm_arg, x_iv
    ):
        """Legacy sum-radius IBP rule for multi-axis or off-last-axis IRFFT."""
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
            bound_inputs = list(inputs)
        else:
            bound_inputs = [
                kwargs.get("x"),
                kwargs.get("s"),
                kwargs.get("dim"),
                kwargs.get("norm"),
            ]

        def _fv(node):
            if node is None:
                return None
            return getattr(node, "forward_value", node)

        x = bound_inputs[0]
        _, s, dim, norm = self._normalize_irfftn_inputs(
            *[_fv(inp) for inp in bound_inputs]
        )
        if self._exact_active is None:
            x_shape = getattr(x, "output_shape", None) or _fv(x).shape
            self._configure_from_inputs(x_shape, s, dim, norm)
        if not self._exact_active:
            raise NotImplementedError(
                "BoundIRFFT.bound_backward is only implemented for the exact 1D path; "
                "the multi-axis loose path is currently IBP-only."
            )

        def _adjoint(A):
            if A is None:
                return None
            assert not A.is_complex(), (
                "BoundIRFFT.bound_backward expects a real cotangent "
                "(IRFFT output is real)."
            )
            # ``apply_AT`` returns a real-pair tensor; the upstream chain
            # (slice / index-put / RFFT) reasons in native complex tensors, so
            # collapse the pair axis back into a complex cotangent here.
            pair = self.apply_AT(A)
            return self.pair_to_complex(pair)

        lA = _adjoint(last_lA)
        uA = _adjoint(last_uA)
        n_inputs = len(self.inputs) if self.inputs else len(bound_inputs)
        return [(lA, uA)] + [(None, None)] * (n_inputs - 1), 0, 0


class BoundComplexEinsum(BoundLinearComplexPair):
    """``aten::einsum`` with a fixed (unperturbed) complex weight operand.

    Forward: ``Y = einsum(eq, X, W)`` where one operand is the perturbed input
    ``X`` and the other is the fixed weight ``W``.  The map ``X -> Y`` is
    complex-linear with elementwise block matrix ``[[Re(W), -Im(W)], [Im(W),
    Re(W)]]`` composed with the einsum contraction.  Under the real-pair
    lowering this gives, for the forward equation ``eq`` oriented so the
    perturbed operand is first,

        Y_re = einsum(eq, X_re, W_re) - einsum(eq, X_im, W_im),
        Y_im = einsum(eq, X_re, W_im) + einsum(eq, X_im, W_re).

    The adjoint equation ``eq_adj`` flips ``X``'s indices with ``Y``'s output
    indices so that, for a cotangent ``c`` on ``Y``,

        d_re =  einsum(eq_adj, c_re, W_re) + einsum(eq_adj, c_im, W_im),
        d_im = -einsum(eq_adj, c_re, W_im) + einsum(eq_adj, c_im, W_re),

    which equals the pair form of ``d = c * conj(W)``.  The radius rule
    substitutes ``|Re(W)|`` and ``|Im(W)|`` for ``Re(W)`` and ``Im(W)``.

    Note: integration of CROWN through the JIT graph requires
    ``BoundPrimListConstruct.bound_backward`` to route per-element adjoints to
    the operand list; the implementation here exposes the pair-form
    primitives and a direct ``bound_backward`` for tests.
    """

    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self._eq: Optional[str] = None
        self._weight: Optional[torch.Tensor] = None
        self._perturbed_index: Optional[int] = None

    def configure(
        self,
        equation: str,
        weight: torch.Tensor,
        perturbed_index: int = 0,
    ) -> None:
        """Pre-configure for direct (non-JIT) use; ``perturbed_index in {0, 1}``."""
        if perturbed_index not in (0, 1):
            raise ValueError("perturbed_index must be 0 or 1")
        self._eq = equation
        self._weight = weight
        self._perturbed_index = perturbed_index

    @staticmethod
    def _parse_equation(eq: str) -> Tuple[str, str, str]:
        try:
            lhs, rhs = eq.split("->")
        except ValueError as e:
            raise NotImplementedError(
                f"BoundComplexEinsum requires explicit '->' equation; got {eq!r}"
            ) from e
        parts = lhs.split(",")
        if len(parts) != 2:
            raise NotImplementedError(
                f"BoundComplexEinsum supports 2-operand einsum; got {eq!r}"
            )
        return parts[0].strip(), parts[1].strip(), rhs.strip()

    def _forward_equation(self) -> str:
        """Equation re-oriented so the perturbed operand is the first argument."""
        in_a, in_b, out = self._parse_equation(self._eq)
        if self._perturbed_index == 0:
            return f"{in_a},{in_b}->{out}"
        return f"{in_b},{in_a}->{out}"

    def _adjoint_equation(self, extra_leading: int = 0) -> str:
        """Equation mapping a Y-cotangent and the weight back to an X-cotangent.

        ``extra_leading`` is the number of leading non-contracting axes on the
        cotangent that are not present in the forward equation (CROWN's
        specification / specification-batch axes).  An ``...`` ellipsis is
        prepended to absorb them and they appear on the output side as well.
        """
        in_a, in_b, out = self._parse_equation(self._eq)
        x_idx, w_idx = (in_a, in_b) if self._perturbed_index == 0 else (in_b, in_a)
        if extra_leading > 0:
            return f"...{out},{w_idx}->...{x_idx}"
        return f"{out},{w_idx}->{x_idx}"

    def _weight_parts(self, dtype, device):
        w = self._weight
        if w.device != device or w.dtype != (
            dtype if not w.is_complex() else _complex_dtype_for(dtype)
        ):
            target_dtype = _complex_dtype_for(dtype) if w.is_complex() else dtype
            w = w.to(device=device, dtype=target_dtype)
        if w.is_complex():
            return w.real, w.imag, w.real.abs(), w.imag.abs()
        zero = torch.zeros_like(w)
        return w, zero, w.abs(), zero

    def apply_A(self, x_pair: torch.Tensor) -> torch.Tensor:
        if self._eq is None or self._weight is None:
            raise RuntimeError(
                "BoundComplexEinsum.apply_A called before configuration."
            )
        assert x_pair.shape[-1] == 2, "apply_A expects real-pair last-axis of size 2."
        if not x_pair.is_floating_point():
            x_pair = x_pair.to(torch.get_default_dtype())
        eq = self._forward_equation()
        w_re, w_im, _, _ = self._weight_parts(x_pair.dtype, x_pair.device)
        x_re = x_pair[..., 0]
        x_im = x_pair[..., 1]
        y_re = torch.einsum(eq, x_re, w_re) - torch.einsum(eq, x_im, w_im)
        y_im = torch.einsum(eq, x_re, w_im) + torch.einsum(eq, x_im, w_re)
        return torch.stack([y_re, y_im], dim=-1)

    def apply_AT(self, c_pair: torch.Tensor) -> torch.Tensor:
        if self._eq is None or self._weight is None:
            raise RuntimeError(
                "BoundComplexEinsum.apply_AT called before configuration."
            )
        assert c_pair.shape[-1] == 2, "apply_AT expects real-pair last-axis of size 2."
        # The cotangent has the output shape of the forward einsum, possibly
        # prepended with CROWN specification axes.  Count those leading axes so
        # the adjoint equation can absorb them via an ellipsis.
        _, _, out = self._parse_equation(self._eq)
        n_out = len(out)
        n_cotangent = c_pair.ndim - 1  # strip trailing real-pair axis
        extra_leading = max(0, n_cotangent - n_out)
        eq_adj = self._adjoint_equation(extra_leading=extra_leading)
        w_re, w_im, _, _ = self._weight_parts(c_pair.dtype, c_pair.device)
        c_re = c_pair[..., 0]
        c_im = c_pair[..., 1]
        d_re = torch.einsum(eq_adj, c_re, w_re) + torch.einsum(eq_adj, c_im, w_im)
        d_im = -torch.einsum(eq_adj, c_re, w_im) + torch.einsum(eq_adj, c_im, w_re)
        return torch.stack([d_re, d_im], dim=-1)

    def apply_absA_radius(self, r_pair: torch.Tensor) -> torch.Tensor:
        if self._eq is None or self._weight is None:
            raise RuntimeError(
                "BoundComplexEinsum.apply_absA_radius called before configuration."
            )
        assert r_pair.shape[-1] == 2, "apply_absA_radius expects real-pair input."
        eq = self._forward_equation()
        _, _, abs_wr, abs_wi = self._weight_parts(r_pair.dtype, r_pair.device)
        r_re = r_pair[..., 0]
        r_im = r_pair[..., 1]
        r_y_re = torch.einsum(eq, r_re, abs_wr) + torch.einsum(eq, r_im, abs_wi)
        r_y_im = torch.einsum(eq, r_re, abs_wi) + torch.einsum(eq, r_im, abs_wr)
        return torch.stack([r_y_re, r_y_im], dim=-1)

    @staticmethod
    def _resolve_equation(equation) -> str:
        if isinstance(equation, str):
            return equation
        if isinstance(equation, torch.Tensor):
            return equation.item() if equation.numel() == 1 else str(equation)
        return str(equation)

    def forward(self, equation, operands, *rest):
        del rest
        eq = self._resolve_equation(equation)
        if isinstance(operands, (list, tuple)):
            ops = tuple(operands)
        else:
            ops = (operands,)
        if len(ops) != 2:
            raise NotImplementedError(
                "BoundComplexEinsum supports 2-operand einsum; "
                f"got {len(ops)} operand(s)."
            )
        return torch.einsum(eq, *ops)

    def interval_propagate(self, equation, operands, *rest):
        del rest
        eq = self._resolve_equation(equation[0])
        lower_ops, upper_ops = operands[0], operands[1]
        if not isinstance(lower_ops, (list, tuple)) or len(lower_ops) != 2:
            raise NotImplementedError(
                "BoundComplexEinsum.interval_propagate requires 2 operands."
            )
        first_perturbed = not torch.equal(lower_ops[0], upper_ops[0])
        second_perturbed = not torch.equal(lower_ops[1], upper_ops[1])
        if first_perturbed and second_perturbed:
            raise NotImplementedError(
                "BoundComplexEinsum IBP does not support two perturbed operands."
            )
        if not (first_perturbed or second_perturbed):
            out = self.forward(eq, list(lower_ops))
            return Interval.make_interval(out, out, operands)
        if first_perturbed:
            x_l, x_u, w = lower_ops[0], upper_ops[0], lower_ops[1]
            self.configure(eq, w, perturbed_index=0)
        else:
            x_l, x_u, w = lower_ops[1], upper_ops[1], lower_ops[0]
            self.configure(eq, w, perturbed_index=1)
        if x_l.is_complex():
            x_pair_l = self.complex_to_pair(x_l)
            x_pair_u = self.complex_to_pair(x_u)
            output_complex = True
        else:
            assert x_l.shape[-1] == 2, (
                "BoundComplexEinsum: real perturbed operand must already be in "
                "real-pair form (last axis of size 2)."
            )
            x_pair_l, x_pair_u = x_l, x_u
            output_complex = False
        y_pair_l, y_pair_u = self.pair_interval_propagate(x_pair_l, x_pair_u)
        if output_complex:
            y_l = self.pair_to_complex(y_pair_l)
            y_u = self.pair_to_complex(y_pair_u)
        else:
            y_l, y_u = y_pair_l, y_pair_u
        return Interval.make_interval(y_l, y_u, operands)

    def _autoconfigure_from_inputs(self, inputs) -> None:
        """Populate ``_eq`` / ``_weight`` / ``_perturbed_index`` from input nodes.

        Used when ``bound_backward`` runs before any ``forward`` / ``interval_propagate``
        call has cached the einsum equation and the unperturbed operand
        (the typical CROWN-only path).  Supported shapes:

        * **ONNX** form: ``inputs = (x, w)``.
        * **JIT** form: ``inputs = (equation_const, list_construct, path_const)``
          where the list-construct holds ``[x, w]`` or ``[w, x]``.
        """
        if not inputs:
            return
        # JIT shape: equation is a BoundConstant string; operands sit in a
        # prim::ListConstruct node at inputs[1].
        if (len(inputs) >= 2
                and type(inputs[1]).__name__ == "BoundPrimListConstruct"):
            eq_node = inputs[0]
            equation = self._resolve_equation(
                getattr(eq_node, 'value',
                        getattr(eq_node, 'forward_value', None))
            )
            list_inputs = inputs[1].inputs
            if len(list_inputs) != 2:
                raise NotImplementedError(
                    "BoundComplexEinsum: JIT operand list must have 2 elements; "
                    f"got {len(list_inputs)}"
                )
            perturbed_index = 0 if list_inputs[0].perturbed else 1
            weight_node = list_inputs[1 - perturbed_index]
            weight = getattr(
                weight_node, 'param',
                getattr(weight_node, 'forward_value',
                        getattr(weight_node, 'value', None))
            )
            if weight is None:
                raise NotImplementedError(
                    "BoundComplexEinsum: could not resolve unperturbed weight "
                    "from JIT graph inputs."
                )
            self.configure(equation, weight, perturbed_index=perturbed_index)
            return
        # ONNX shape: (x, w) with one perturbed input.
        if len(inputs) == 2:
            perturbed_index = 0 if inputs[0].perturbed else 1
            weight_node = inputs[1 - perturbed_index]
            weight = getattr(
                weight_node, 'param',
                getattr(weight_node, 'forward_value',
                        getattr(weight_node, 'value', None))
            )
            equation = getattr(self, '_eq', None)
            if equation is None or weight is None:
                raise NotImplementedError(
                    "BoundComplexEinsum: ONNX-form bound_backward requires a "
                    "prior forward / interval_propagate to cache the equation."
                )
            self.configure(equation, weight, perturbed_index=perturbed_index)

    def bound_backward(self, last_lA, last_uA, *inputs, **kwargs):
        """Adjoint application of the complex-einsum linear map.

        Two graph shapes are handled:

        * **ONNX / unit-test** form: ``inputs = (x, w)`` or ``inputs = ()``.
          Returns a per-input list of ``(lA, uA)`` tensors with ``(None, None)``
          for the unperturbed weight slot.
        * **JIT** form: ``inputs = (equation_const, list_construct, path_const)``.
          Returns ``[(None, None), (packed_l, packed_u), (None, None)]`` where
          ``packed_l`` / ``packed_u`` are length-2 tuples of per-list-element
          adjoints (``None`` for the unperturbed slot).
          :class:`BoundPrimListConstruct.bound_backward` unpacks these tuples.
        """
        del kwargs
        if self._eq is None or self._weight is None:
            self._autoconfigure_from_inputs(inputs)
            if self._eq is None or self._weight is None:
                raise RuntimeError(
                    "BoundComplexEinsum.bound_backward called before "
                    "configuration; call ``configure`` or run ``interval_"
                    "propagate`` first."
                )

        def _adjoint(A):
            if A is None:
                return None
            if A.is_complex():
                pair = self.complex_to_pair(A)
                pair_out = self.apply_AT(pair)
                # Keep complex semantics through complex-domain ops upstream.
                return self.pair_to_complex(pair_out)
            assert A.shape[-1] == 2, (
                "BoundComplexEinsum.bound_backward expects a complex A or a "
                "real-pair A with trailing size-2 axis."
            )
            return self.apply_AT(A)

        lA = _adjoint(last_lA)
        uA = _adjoint(last_uA)

        # No-input test calling convention.
        if not inputs:
            return [(lA, uA)], 0, 0

        # JIT layout: pack per-list-element As for the list-construct slot;
        # propagate ``(None, None)`` to all other (metadata) slots.
        if (len(inputs) >= 2
                and type(inputs[1]).__name__ == "BoundPrimListConstruct"):
            pi = self._perturbed_index
            packed_l = (lA, None) if pi == 0 else (None, lA)
            packed_u = (uA, None) if pi == 0 else (None, uA)
            ret = [(None, None)] * len(inputs)
            ret[1] = (packed_l, packed_u)
            return ret, 0, 0

        # ONNX layout: 2 positional input slots.
        pi = self._perturbed_index
        ret = [(None, None)] * len(inputs)
        ret[pi] = (lA, uA)
        return ret, 0, 0


def _complex_dtype_for(real_dtype: torch.dtype) -> torch.dtype:
    """Pick the complex dtype matching ``real_dtype`` for materialized weights."""
    if real_dtype == torch.float64:
        return torch.complex128
    if real_dtype == torch.float32:
        return torch.complex64
    if real_dtype == torch.float16 or real_dtype == torch.bfloat16:
        return torch.complex32
    return torch.complex64
