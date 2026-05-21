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
    assign_nodes: List[Node] = []
    slice_targets: Set[str] = set()
    kept: List[Node] = []

    for n in nodesOP:
        if n.op != "aten::copy_":
            kept.append(n)
            continue
        dest_view, src, _non_blocking = n.inputs[0], n.inputs[1], n.inputs[2]
        base_name, chain = _slice_chain_nodes(dest_view, by)
        slice_targets |= chain
        assign_name = f"{base_name}/assign"
        try:
            slice_chain = _ordered_slice_specs(dest_view, by)
        except ValueError:
            slice_chain = []
        assign_nodes.append(
            Node(
                name=assign_name,
                ori_name=None,
                inputs=[base_name, src],
                attr={"slice_chain": slice_chain},
                op="custom::Assign",
                param={},
                input_index=None,
                bound_node=None,
                output_index=0,
                perturbation=None,
            )
        )
        rewire[base_name] = assign_name

    if not assign_nodes:
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
    for n in kept + assign_nodes:
        if n.op == "custom::Assign":
            out.append(n)
        else:
            out.append(n._replace(inputs=map_inputs(list(n.inputs), n.name)))

    if nodesOut is not None:
        nodesOut = [rewire.get(o, o) for o in nodesOut]

    return out, nodesOut


def _bound_buffer_base(dest_view):
    """Root buffer node for a JIT slice view chain."""
    cur = dest_view
    while type(cur).__name__ == "BoundAtenJitSlice":
        cur = cur.inputs[0]
    return cur


def register_inplace_buffer_writes(model) -> None:
    """Register ``aten::copy_`` side effects on the buffer they mutate.

    JIT graphs use ``copy_`` into a slice of a ``zeros`` buffer while ``irfftn`` reads
    the buffer node directly, so ``copy_`` has no SSA consumers and is dropped by graph
    optimization. LiRPA runs pending writes when the base buffer's forward value is
    first materialized.
    """
    from .operators.aten_fallback import BoundAtenJitCopy

    for node in list(model.nodes()):
        if not isinstance(node, BoundAtenJitCopy):
            continue
        dest_view = node.inputs[0]
        base = _bound_buffer_base(dest_view)
        node._inplace_base = base
        pending = getattr(base, "_pending_inplace_writes", None)
        if pending is None:
            pending = []
            base._pending_inplace_writes = pending
        if node not in pending:
            pending.append(node)
