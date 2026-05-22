#########################################################################
##  JIT graph helpers (in-place copy → explicit assign for LiRPA)        ##
#########################################################################
"""Rewrite traced JIT graphs so in-place ``aten::copy_`` becomes ``custom::Assign``."""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from .parse_graph import Node


def _node_by_name(nodesOP: List[Node]) -> Dict[str, Node]:
    return {n.name: n for n in nodesOP}


def _slice_chain_nodes(dest_view: str, by: Dict[str, Node]) -> Tuple[str, Set[str]]:
    """Return base buffer name and slice nodes used as the copy destination view."""
    chain: Set[str] = set()
    cur = dest_view
    while cur in by and by[cur].op == "aten::slice":
        chain.add(cur)
        cur = by[cur].inputs[0]
    return cur, chain


def _resolve_graph_int(by: Dict[str, Node], name: str) -> int:
    """Resolve a traced scalar to ``int`` (follows ``aten::Int`` / ``prim::NumToTensor``)."""
    n = by[name]
    while n.op in ("aten::Int", "prim::NumToTensor"):
        n = by[n.inputs[0]]
    if n.op != "prim::Constant" or "value" not in n.attr:
        raise ValueError(f"cannot resolve scalar constant for {name!r} ({n.op})")
    v = n.attr["value"]
    if hasattr(v, "item"):
        return int(v.item())
    return int(v)


def _ordered_slice_specs(dest_view: str, by: Dict[str, Node]) -> List[Tuple[int, int, int, int]]:
    """Return ``(dim, start, end, step)`` outer-to-inner for a slice view chain."""
    specs: List[Tuple[int, int, int, int]] = []
    cur = dest_view
    while cur in by and by[cur].op == "aten::slice":
        sn = by[cur]
        specs.append(
            tuple(_resolve_graph_int(by, inp) for inp in sn.inputs[1:5])  # type: ignore[misc]
        )
        cur = sn.inputs[0]
    return list(reversed(specs))


def rewrite_inplace_copy_to_assign(
    nodesOP: List[Node], nodesOut: List[str] | None = None
) -> Tuple[List[Node], List[str] | None]:
    """Replace ``aten::copy_`` with ``custom::Assign`` and rewire buffer users.

    Slice views that are write targets keep reading the pre-assign buffer; other
    uses of the buffer (including the module output) are rewired to the assign node.
    """
    by = _node_by_name(nodesOP)
    rewire: Dict[str, str] = {}
    slice_targets: Set[str] = set()
    # Map original ``aten::copy_`` index in ``nodesOP`` → its replacement
    # ``custom::Assign`` ``Node``.  Inserting in-place preserves topological
    # order so downstream consumers (e.g. ``aten::fft_irfftn`` reading the
    # rewired buffer) appear after the assign node in ``nodesOP``.
    assigns_by_idx: Dict[int, Node] = {}

    for idx, n in enumerate(nodesOP):
        if n.op != "aten::copy_":
            continue
        dest_view, src, _non_blocking = n.inputs[0], n.inputs[1], n.inputs[2]
        base_name, chain = _slice_chain_nodes(dest_view, by)
        slice_targets |= chain
        assign_name = f"{base_name}/assign"
        try:
            slice_chain = _ordered_slice_specs(dest_view, by)
        except ValueError:
            slice_chain = []
        # ``dest_view`` is the deepest slice in the write chain.  We keep it as
        # a metadata input so ``BoundIndexPut`` can resolve the slice indices
        # lazily at forward time when ``_ordered_slice_specs`` could not.
        assign_inputs = [base_name, src, dest_view] if dest_view != base_name else [base_name, src]
        assigns_by_idx[idx] = Node(
            name=assign_name,
            ori_name=None,
            inputs=assign_inputs,
            attr={"slice_chain": slice_chain},
            op="custom::Assign",
            param={},
            input_index=None,
            bound_node=None,
            output_index=0,
            perturbation=None,
        )
        rewire[base_name] = assign_name

    if not assigns_by_idx:
        return nodesOP, nodesOut

    def map_inputs(inputs: List[str], node_name: str) -> List[str]:
        mapped = []
        for i in inputs:
            if node_name in slice_targets:
                mapped.append(i)
            else:
                mapped.append(rewire.get(i, i))
        return mapped

    out: List[Node] = []
    for idx, n in enumerate(nodesOP):
        if n.op == "aten::copy_":
            out.append(assigns_by_idx[idx])
            continue
        out.append(n._replace(inputs=map_inputs(list(n.inputs), n.name)))

    if nodesOut is not None:
        nodesOut = [rewire.get(o, o) for o in nodesOut]

    return out, nodesOut


