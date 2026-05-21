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
``apply_absA_radius``) and :class:`BoundRFFT`, the first concrete subclass.
``BoundRFFT`` replaces the legacy ``BoundAtenJitFftRfftn``; its 1D path uses
the exact rule, its ND path keeps the legacy sum-radius rule until a follow-up
PR generalizes the exact rule to multi-axis FFTs.
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
    """``aten::fft_rfftn`` bounds via real-pair lowering.

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

    def forward(self, x, _s, dim, norm):
        del _s
        self._configure_from_inputs(x.shape, dim, norm)
        if self._exact_active:
            return torch.fft.rfft(
                x, n=self._exact_N, dim=self._exact_dim, norm=self._exact_norm
            )
        dim_list = _tensor_to_int_list(dim) if dim is not None else None
        return torch.fft.rfftn(x, dim=dim_list, norm=_norm_arg(norm))

    def interval_propagate(self, x, _s, dim, norm):
        del _s
        x_lower, x_upper = x[0], x[1]
        if self._exact_active is None:
            self._configure_from_inputs(x_lower.shape, dim[0], norm[0])
        if self._exact_active:
            return self._interval_propagate_exact(x_lower, x_upper, x)
        return self._interval_propagate_loose(x_lower, x_upper, dim[0], norm[0], x)

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

    def bound_backward(self, last_lA, last_uA, x, _s, dim, norm, **kwargs):
        del kwargs
        if self._exact_active is None:
            self._configure_from_inputs(x.output_shape, dim.forward_value, norm.forward_value)
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
            return self.apply_AT(pair)

        lA = _adjoint(last_lA)
        uA = _adjoint(last_uA)
        # Inputs are (x, _s, dim, norm); only x carries an A coefficient.
        return [(lA, uA), (None, None), (None, None), (None, None)], 0, 0
