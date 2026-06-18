//===- TopologyOps.td - Wafer target topology ops ---------*- tablegen -*-===//

#ifndef WAFER_IR_TARGET_TOPOLOGYOPS_TD
#define WAFER_IR_TARGET_TOPOLOGYOPS_TD

def Wafer_TargetTopologyOp : Wafer_Op<"target.topology", [Symbol]> {
  let summary = "explicit Wafer physical tile topology graph";
  let description = [{
    `wafer.target.topology` records the physical tile graph accepted by the
    compiler pipeline. It is the fact source for tile coordinates, encoded tile
    ids, availability, connectivity and tile-id codec provenance. Downstream
    mesh selection, SPMD seed policy, communication lowering and ABI/package
    resource views query this explicit graph instead of reconstructing tile ids
    from row-major or card-major formulas.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    StrArrayAttr:$axes,
    StrAttr:$id_encoding,
    DenseI64ArrayAttr:$tile_coords,
    DenseI64ArrayAttr:$tile_ids,
    DenseI64ArrayAttr:$available_tile_ids,
    DenseI64ArrayAttr:$bad_tile_ids,
    DenseI64ArrayAttr:$pg_disabled_tile_ids,
    DenseI64ArrayAttr:$links
  );

  let hasVerifier = 1;
  let assemblyFormat = "$sym_name attr-dict";
}

#endif // WAFER_IR_TARGET_TOPOLOGYOPS_TD
