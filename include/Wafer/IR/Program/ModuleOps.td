//===- ModuleOps.td - Wafer target module operations ----------*- tablegen -*-===//

#ifndef WAFER_IR_PROGRAM_MODULEOPS_TD
#define WAFER_IR_PROGRAM_MODULEOPS_TD

def Wafer_CardModuleOp : Wafer_Op<"card.module", [
    IsolatedFromAbove,
    NoRegionArguments,
    RecursiveMemoryEffects,
    SymbolTable
  ] # GraphRegionNoTerminator.traits> {
  let summary = "module containing the selected execution for one card";
  let description = [{
    `wafer.card.module` is the selected execution and verification scope for
    one target card. `card_id` is a row-major card identifier in the enclosing
    `wafer.target.topology`. The body owns all-and-only
    available `wafer.tile.module` operations for that card.

    This operation is a module-like container. Card-owned symbols represent
    shared DDR resources; downstream module splitting forms standalone per-Tile
    module boundaries. The operation does not carry search state, a shadow
    schedule, routes, scores, or target launch slots.
  }];

  let arguments = (ins I64Attr:$card_id);
  let regions = (region SizedRegion<1>:$body);

  let hasVerifier = 1;
  let assemblyFormat = [{
    `card_id` `=` $card_id attr-dict-with-keyword $body
  }];
}

def Wafer_TileModuleOp : Wafer_Op<"tile.module", [
    HasParent<"CardModuleOp">,
    IsolatedFromAbove,
    NoRegionArguments,
    RecursiveMemoryEffects,
    SymbolTable
  ] # GraphRegionNoTerminator.traits> {
  let summary = "module containing the selected execution for one Tile";
  let description = [{
    `wafer.tile.module` owns the selected execution for one target Tile.
    `tile_id` is row-major within the parent card's Tile grid. Different Tile
    modules may contain different executable symbols, loops, operation sets,
    temporal tile shapes, and execution lengths. Each Tile module is its own
    symbol table so the same entry symbol can have a distinct implementation
    on every Tile.

    The isolated graph region makes Tile-local SSA ownership explicit: SSA
    values, including SPM roots and aliases, cannot cross a TileModule
    boundary. Cross-Tile dataflow is represented by typed communication and
    card-owned DDR resources, not by implicit SSA capture.
  }];

  let arguments = (ins I64Attr:$tile_id);
  let regions = (region SizedRegion<1>:$body);

  let hasVerifier = 1;
  let assemblyFormat = [{
    `tile_id` `=` $tile_id attr-dict-with-keyword $body
  }];
}

#endif // WAFER_IR_PROGRAM_MODULEOPS_TD
