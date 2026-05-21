#########################################################################
##  ATen ONNX-fallback dispatch for auto_LiRPA                         ##
#########################################################################
"""Resolve ``Bound`` classes for ``aten::ATen`` nodes from ONNX export.

PyTorch's ONNX ATen fallback uses ``operator="Placeholder"`` with the real
ATen name in ``attr["name"]`` (e.g. ``index_put_``). It also uses multi-token
operator names such as ``fft_irfftn`` where ``str.capitalize()`` does not
produce a valid ``BoundATen...`` class name.

``operator="zeros"`` from ONNX passes shape/dtype as tensor arguments; that
is routed to ``BoundATenOnnxZeros`` instead of ``BoundATenZeros`` (which expects
``attr["shape"]`` from a different export path).
"""
from __future__ import annotations

from typing import Any, Dict


def resolve_aten_bound_class(attr: Dict[str, Any], globals_ns: Dict[str, Any]):
    """Return the ``Bound`` subclass for an ``aten::ATen`` node's attributes.

    Parameters
    ----------
    attr
        Node ``attr`` dict (must contain ``operator``).
    globals_ns
        Namespace that already contains ``BoundATen*`` symbols (typically
        ``bound_ops`` after ``from .operators import *``).
    """
    op = attr.get("operator", "")
    if op == "Placeholder":
        name = attr.get("name", "")
        if name == "index_put_":
            return globals_ns["BoundATenIndexPut"]
        raise KeyError(f"unsupported ATen Placeholder name={name!r}")

    # Explicit names where ``op.capitalize()`` is not the class suffix (e.g. fft_irfftn).
    if op == "fft_irfftn":
        return globals_ns["BoundATenFftIrfftn"]
    if op == "fft_rfftn":
        return globals_ns["BoundATenFftRfftn"]
    # ONNX ATen fallback uses tensor arguments instead of attr["shape"] (see BoundATenZeros).
    if op == "zeros":
        return globals_ns["BoundATenOnnxZeros"]

    cls_name = f"BoundATen{op.capitalize()}"
    if cls_name not in globals_ns:
        raise KeyError(cls_name)
    return globals_ns[cls_name]


# Raw JIT ``aten::`` ops (when ``onnx_optimize_graph`` is False).
_JIT_ATEN_MAP = {
    "aten::zeros": "BoundATenOnnxZeros",
    "aten::fft_rfftn": "BoundRFFT",
    "aten::fft_irfftn": "BoundAtenJitFftIrfftn",
    "aten::slice": "BoundAtenJitSlice",
    "aten::einsum": "BoundAtenJitEinsum",
    "aten::clone": "BoundAtenClone",
    "aten::expand": "BoundAtenExpand",
    "aten::add": "BoundAtenJitAdd",
    "aten::size": "BoundAtenJitSize",
    "aten::Int": "BoundAtenJitInt",
    "aten::floor_divide": "BoundAtenJitFloorDivide",
    "aten::copy_": "BoundAtenJitCopy",
}

_JIT_PRIM_MAP = {
    "prim::Constant": "BoundJitConstant",
    "prim::ListConstruct": "BoundPrimListConstruct",
    "prim::NumToTensor": "BoundPrimNumToTensor",
}


def resolve_jit_aten_bound_class(op: str, globals_ns: Dict[str, Any]):
    """Resolve ``Bound`` for a native JIT ``aten::`` op name."""
    if op in _JIT_ATEN_MAP:
        return globals_ns[_JIT_ATEN_MAP[op]]
    raise KeyError(op)


def resolve_jit_prim_bound_class(op: str, globals_ns: Dict[str, Any]):
    """Resolve ``Bound`` for a native JIT ``prim::`` op name."""
    if op in _JIT_PRIM_MAP:
        return globals_ns[_JIT_PRIM_MAP[op]]
    raise KeyError(op)
