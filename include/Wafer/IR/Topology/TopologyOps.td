//===- TopologyOps.td - Wafer target topology ops ---------*- tablegen -*-===//

#ifndef WAFER_IR_TOPOLOGY_TOPOLOGYOPS_TD
#define WAFER_IR_TOPOLOGY_TOPOLOGYOPS_TD

def Wafer_TargetTopologyOp : Wafer_Op<"target.topology", [Symbol]> {
  let summary = "regular Wafer target topology facts";
  let description = [{
    `wafer.target.topology` records the regular target topology accepted by the
    compiler pipeline. It is the fact source for card grid shape, card-level
    C2C interconnect kind, per-card tile grid shape and unavailable tile
    coordinate exceptions. Downstream mesh selection, SPMD seed policy,
    communication lowering and ABI/package resource views derive regular
    adjacency from these grids instead of carrying a fully expanded tile/link
    graph.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    DenseI64ArrayAttr:$card_grid,
    StrAttr:$card_interconnect,
    DenseI64ArrayAttr:$tile_grid,
    DenseI64ArrayAttr:$unavailable_tiles
  );

  let hasVerifier = 1;
  let assemblyFormat = "$sym_name attr-dict";
}

def Wafer_ExecutionMeshOp : Wafer_Op<"execution.mesh", [Symbol]> {
  let summary = "logical card-partition mesh";
  let description = [{
    `wafer.execution.mesh` records only the logical card-partition domain
    selected for a program. Its axis-shape product is `num_partitions`.
    Tile participation and placement are selected downstream and
    are not part of this IR object.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    StrArrayAttr:$axes,
    DenseI64ArrayAttr:$shape
  );

  let hasVerifier = 1;
  let assemblyFormat = "$sym_name attr-dict";
}

#endif // WAFER_IR_TOPOLOGY_TOPOLOGYOPS_TD
