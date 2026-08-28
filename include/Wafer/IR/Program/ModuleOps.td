//===- ModuleOps.td - Wafer target module operations ----------*- tablegen -*-===//

#ifndef WAFER_IR_PROGRAM_MODULEOPS_TD
#define WAFER_IR_PROGRAM_MODULEOPS_TD

def Wafer_TileModuleOp : Wafer_Op<"tile.module", [
    IsolatedFromAbove,
    NoRegionArguments,
    RecursiveMemoryEffects,
    SymbolTable
  ] # GraphRegionNoTerminator.traits> {
  let summary = "module containing the selected execution for one Tile";
  let description = [{
    `wafer.tile.module` is directly nested under `builtin.module` and owns the
    selected execution for one target Tile. `card_id` identifies the physical
    card and `tile_id` is row-major within that card's Tile grid. Different Tile
    modules may contain different executable symbols, loops, operation sets,
    temporal tile shapes, and execution lengths. Each Tile module is its own
    symbol table so the same entry symbol can have a distinct implementation
    on every Tile.

    The isolated graph region makes Tile-local SSA ownership explicit: SSA
    values, including SPM roots and aliases, cannot cross a TileModule
    boundary. Cross-Tile dataflow is represented by typed communication and
    shared DDR resources, not by implicit SSA capture.
  }];

  let arguments = (ins I64Attr:$card_id, I64Attr:$tile_id);
  let regions = (region SizedRegion<1>:$body);

  let hasVerifier = 1;
  let assemblyFormat = [{
    `card_id` `=` $card_id `tile_id` `=` $tile_id attr-dict-with-keyword $body
  }];
}

#endif // WAFER_IR_PROGRAM_MODULEOPS_TD
