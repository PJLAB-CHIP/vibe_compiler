//===- MoveOps.td - Wafer movement operation definitions ----*- tablegen -*-===//

#ifndef WAFER_IR_OPS_MOVEOPS_TD
#define WAFER_IR_OPS_MOVEOPS_TD

class Wafer_MoveOp<string mnemonic> : Wafer_Op<mnemonic, [
    DeclareOpInterfaceMethods<WaferLayoutOpInterface>,
    DeclareOpInterfaceMethods<WaferResourceEffectInterface>,
    MemoryEffects<[MemRead<Wafer_SPMResource>,
                   MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_MovementResource>]>
  ]>;

def Wafer_MoveExtractSliceOp : Wafer_MoveOp<"move.extract_slice"> {
  let summary = "tile-local explicit slice extraction movement";
  let description = [{
    `wafer.move.extract_slice` copies a statically described slice from an SPM
    tile buffer into another SPM tile buffer. It is explicit movement, not an
    aliasing view, so SPM allocation and resource planning can see the read
    and write.
  }];

  let arguments = (ins
    Wafer_AnyTileBuffer:$source,
    DenseI64ArrayAttr:$offsets,
    DenseI64ArrayAttr:$sizes,
    DenseI64ArrayAttr:$strides
  );
  let results = (outs Wafer_AnyTileBuffer:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "$source attr-dict `:` type($source) `->` type($result)";
}

def Wafer_MoveInsertSliceOp : Wafer_MoveOp<"move.insert_slice"> {
  let summary = "tile-local explicit slice insertion movement";
  let description = [{
    `wafer.move.insert_slice` copies a source tile buffer into a statically
    described slice of a destination tile buffer and returns the updated tile
    buffer value.
  }];

  let arguments = (ins
    Wafer_AnyTileBuffer:$source,
    Wafer_AnyTileBuffer:$dest,
    DenseI64ArrayAttr:$offsets,
    DenseI64ArrayAttr:$sizes,
    DenseI64ArrayAttr:$strides
  );
  let results = (outs Wafer_AnyTileBuffer:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "$source `into` $dest attr-dict `:` type($source) `into` type($dest) `->` type($result)";
}

def Wafer_MoveCopyOp : Wafer_MoveOp<"move.copy"> {
  let summary = "tile-local explicit same-shape copy movement";
  let arguments = (ins Wafer_AnyTileBuffer:$source);
  let results = (outs Wafer_AnyTileBuffer:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def Wafer_MoveTransposeOp : Wafer_MoveOp<"move.transpose"> {
  let summary = "tile-local explicit transpose movement";
  let arguments = (ins
    Wafer_AnyTileBuffer:$source,
    DenseI64ArrayAttr:$permutation
  );
  let results = (outs Wafer_AnyTileBuffer:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "$source attr-dict `:` type($source) `->` type($result)";
}

def Wafer_MoveBroadcastOp : Wafer_MoveOp<"move.broadcast"> {
  let summary = "tile-local explicit broadcast movement";
  let arguments = (ins
    Wafer_AnyTileBuffer:$source,
    DenseI64ArrayAttr:$dimensions
  );
  let results = (outs Wafer_AnyTileBuffer:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "$source attr-dict `:` type($source) `->` type($result)";
}

#endif // WAFER_IR_OPS_MOVEOPS_TD
