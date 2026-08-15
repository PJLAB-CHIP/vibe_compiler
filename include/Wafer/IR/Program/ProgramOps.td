//===- ProgramOps.td - Wafer whole-card program operations ----*- tablegen -*-===//

#ifndef WAFER_IR_PROGRAM_PROGRAMOPS_TD
#define WAFER_IR_PROGRAM_PROGRAMOPS_TD

def Wafer_CardProgramOp : Wafer_Op<"card.program", [
    IsolatedFromAbove,
    NoRegionArguments,
    RecursiveMemoryEffects,
    SymbolTable
  ] # GraphRegionNoTerminator.traits> {
  let summary = "whole-card Wafer MPMD program";
  let description = [{
    `wafer.card.program` is the selected, card-local verification scope for a
    whole-card MPMD program. `card_id` is a row-major physical card identifier
    in the enclosing `wafer.target.topology`. The body owns all-and-only
    available `wafer.tile.program` operations for that card.

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

def Wafer_TileProgramOp : Wafer_Op<"tile.program", [
    HasParent<"CardProgramOp">,
    IsolatedFromAbove,
    NoRegionArguments,
    RecursiveMemoryEffects,
    SymbolTable
  ] # GraphRegionNoTerminator.traits> {
  let summary = "one physical Tile program in a whole-card MPMD program";
  let description = [{
    `wafer.tile.program` owns the selected execution for one physical Tile.
    `tile_id` is row-major within the parent card's Tile grid. Different Tile
    programs may contain different executable symbols, loops, operation sets,
    temporal tile shapes, and execution lengths. Each Tile program is its own
    symbol table so the same entry symbol can have a distinct implementation
    on every physical Tile.

    The isolated graph region makes Tile-local SSA ownership explicit: SSA
    values, including SPM roots and aliases, cannot cross a Tile-program
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

#endif // WAFER_IR_PROGRAM_PROGRAMOPS_TD
