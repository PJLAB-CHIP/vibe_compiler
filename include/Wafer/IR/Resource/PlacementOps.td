//===- PlacementOps.td - Wafer Placement operation definitions ----*- tablegen -*-===//

#ifndef WAFER_IR_OPS_PLACEMENTOPS_TD
#define WAFER_IR_OPS_PLACEMENTOPS_TD

def Wafer_PlacementMapOp : Wafer_Op<"placement.map"> {
  let summary = "accepted logical-rank to physical-tile mapping";
  let description = [{
    `wafer.placement.map` records an accepted placement mapping from dense
    logical ranks to Wafer physical tile coordinates. Coordinates are flattened
    as `[card_y, card_x, tile_y, tile_x]` tuples in logical rank order. The op
    carries only the accepted mapping and target topology facts needed for
    verification; candidate search trace, cost scores, SPM/DDR allocation, DTE
    packets, and runtime handles are not placement-map semantics.
  }];

  let arguments = (ins
    I64Attr:$logical_rank_count,
    DenseI64ArrayAttr:$physical_tile_coords,
    I64Attr:$card_y_count,
    I64Attr:$card_x_count,
    I64Attr:$tile_y_count,
    I64Attr:$tile_x_count,
    DenseI64ArrayAttr:$bad_tile_ids
  );

  let hasVerifier = 1;
  let assemblyFormat = "attr-dict";
}

#endif // WAFER_IR_OPS_PLACEMENTOPS_TD
