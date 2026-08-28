//===- StorageOps.td - Wafer SPM operation definitions ----*- tablegen -*-===//

#ifndef WAFER_IR_TILE_STORAGEOPS_TD
#define WAFER_IR_TILE_STORAGEOPS_TD

include "mlir/Interfaces/DestinationStyleOpInterface.td"

def Wafer_StorageLoadOp : Wafer_Op<"tile.load", [
    WaferTileDataflowOpInterface,
    DestinationStyleOpInterface,
    MemoryEffects<[MemRead<Wafer_DDRResource>,
                   MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_MovementResource>]>
]> {
  let summary = "load a DDR boundary buffer into an SPM buffer";
  let arguments = (ins Arg<Wafer_AnyDDRMemRef, "", [MemRead]>:$source, Arg<Wafer_AnySPMMemRef, "", [MemWrite]>:$dest);
  let hasVerifier = 1;
  let assemblyFormat =
      "$source `into` $dest attr-dict `:` type($source) `into` type($dest)";
  let extraClassDeclaration = [{
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return getDestMutable();
    }
  }];
}

def Wafer_StorageStoreOp : Wafer_Op<"tile.store", [
    WaferTileDataflowOpInterface,
    MemoryEffects<[MemRead<Wafer_SPMResource>,
                   MemWrite<Wafer_DDRResource>,
                   MemWrite<Wafer_MovementResource>]>
]> {
  let summary = "store an SPM buffer to a DDR boundary buffer";
  let arguments = (ins Arg<Wafer_AnySPMMemRef, "", [MemRead]>:$source, Arg<Wafer_AnyDDRMemRef, "", [MemWrite]>:$dest);
  let hasVerifier = 1;
  let assemblyFormat =
      "$source `,` $dest attr-dict `:` type($source) `->` type($dest)";
}

#endif // WAFER_IR_TILE_STORAGEOPS_TD
