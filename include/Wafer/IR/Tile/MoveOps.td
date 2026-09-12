//===- MoveOps.td - Wafer movement operation definitions ----*- tablegen -*-===//

#ifndef WAFER_IR_OPS_MOVEOPS_TD
#define WAFER_IR_OPS_MOVEOPS_TD

class Wafer_MoveOp<string mnemonic, list<Trait> traits = []>
    : Wafer_Op<mnemonic, !listconcat([
    WaferTileDataflowOpInterface,
    MemoryEffects<[MemRead<Wafer_SPMResource>,
                   MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_MovementResource>]>
  ], traits)>;

def Wafer_MoveExtractSliceOp : Wafer_MoveOp<"tile.extract_slice"> {
  let summary = "tile-local explicit slice extraction movement";
  let description = [{
    `wafer.tile.extract_slice` copies a statically described slice from an SPM
    storage into another SPM storage. It is explicit movement, not an
    aliasing view, so SPM allocation and resource planning can see the read
    and write.
  }];

  let arguments = (ins
    Arg<Wafer_AnySPMMemRef, "", [MemRead]>:$source,
    DenseI64ArrayAttr:$offsets,
    DenseI64ArrayAttr:$sizes,
    DenseI64ArrayAttr:$strides
  );
  let results = (outs Res<Wafer_AnySPMMemRef, "", [MemAlloc, MemWrite]>:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "$source attr-dict `:` type($source) `->` type($result)";
}

def Wafer_MoveInsertSliceOp
    : Wafer_MoveOp<"tile.insert_slice", [DestinationStyleOpInterface]> {
  let summary = "tile-local explicit slice insertion movement";
  let description = [{
    `wafer.tile.insert_slice` copies a source storage into a statically
    described slice of a destination storage. It has pure buffer destination
    semantics and therefore does not return a duplicate SSA alias of `dest`.
    Functional out-of-place updates must first create an explicit
    `wafer.tile.copy` of the old destination.
  }];

  let arguments = (ins
    Arg<Wafer_AnySPMMemRef, "", [MemRead]>:$source,
    Arg<Wafer_AnySPMMemRef, "", [MemWrite]>:$dest,
    DenseI64ArrayAttr:$offsets,
    DenseI64ArrayAttr:$sizes,
    DenseI64ArrayAttr:$strides,
    OptionalAttr<Wafer_DDRResourceAttr>:$ddr_resource
  );
  let extraClassDeclaration = [{
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return getDestMutable();
    }
  }];
  let hasVerifier = 1;
  let assemblyFormat =
      "$source `into` $dest attr-dict `:` type($source) `into` type($dest)";
}

def Wafer_MoveCopyOp : Wafer_MoveOp<"tile.copy"> {
  let summary = "tile-local explicit same-shape copy movement";
  let arguments = (ins
    Arg<Wafer_AnySPMMemRef, "", [MemRead]>:$source,
    OptionalAttr<Wafer_DDRResourceAttr>:$ddr_resource
  );
  let results = (outs Res<Wafer_AnySPMMemRef, "", [MemAlloc, MemWrite]>:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def Wafer_MoveCopyIntoOp : Wafer_MoveOp<"tile.copy_into"> {
  let summary = "tile-local explicit copy into an existing storage view";
  let description = [{
    `wafer.tile.copy_into` copies one SPM storage into an existing same-shaped
    SPM destination view. Source and destination may use different physical
    layouts; the operation then represents the actual layout-changing movement
    into that already selected destination. Dynamic placement is represented
    by the destination memref view itself, so the movement operation does not
    carry a shadow offset plan. The operation is used only when current
    SSA/effect analysis proves that updating the destination storage preserves
    functional value semantics.
  }];

  let arguments = (ins
    Arg<Wafer_AnySPMMemRef, "", [MemRead]>:$source,
    Arg<Wafer_AnySPMMemRef, "", [MemWrite]>:$dest
  );
  let hasVerifier = 1;
  let assemblyFormat =
      "$source `into` $dest attr-dict `:` type($source) `into` type($dest)";
}

def Wafer_MoveReshapeOp : Wafer_MoveOp<"tile.reshape_copy"> {
  let summary = "tile-local explicit materializing reshape movement";
  let description = [{
    `wafer.tile.reshape_copy` preserves canonical logical linear element order
    while copying into a differently shaped allocation. It is the explicit
    movement counterpart of the alias-only `wafer.tile.reshape` view and is
    required when selected physical layouts do not preserve identical element
    mappings across the reshape.
  }];

  let arguments = (ins Arg<Wafer_AnySPMMemRef, "", [MemRead]>:$source);
  let results = (outs Res<Wafer_AnySPMMemRef, "", [MemAlloc, MemWrite]>:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def Wafer_MoveTransposeOp : Wafer_MoveOp<"tile.transpose"> {
  let summary = "tile-local explicit transpose movement";
  let arguments = (ins
    Arg<Wafer_AnySPMMemRef, "", [MemRead]>:$source,
    DenseI64ArrayAttr:$permutation
  );
  let results = (outs Res<Wafer_AnySPMMemRef, "", [MemAlloc, MemWrite]>:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "$source attr-dict `:` type($source) `->` type($result)";
}

def Wafer_MoveBroadcastOp : Wafer_MoveOp<"tile.broadcast"> {
  let summary = "tile-local explicit broadcast movement";
  let arguments = (ins
    Arg<Wafer_AnySPMMemRef, "", [MemRead]>:$source,
    DenseI64ArrayAttr:$dimensions
  );
  let results = (outs Res<Wafer_AnySPMMemRef, "", [MemAlloc, MemWrite]>:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "$source attr-dict `:` type($source) `->` type($result)";
}

#endif // WAFER_IR_OPS_MOVEOPS_TD
