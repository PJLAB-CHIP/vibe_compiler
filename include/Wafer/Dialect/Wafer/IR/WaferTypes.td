//===- WaferTypes.td - Wafer type definitions -------------*- tablegen -*-===//

#ifndef WAFER_DIALECT_WAFER_IR_WAFERTYPES_TD
#define WAFER_DIALECT_WAFER_IR_WAFERTYPES_TD

include "Wafer/Dialect/Wafer/IR/WaferAttrs.td"
include "Wafer/Dialect/Wafer/IR/WaferBase.td"
include "mlir/IR/AttrTypeBase.td"

def Wafer_TileBufferType : TypeDef<Wafer_Dialect, "TileBuffer"> {
  let mnemonic = "tile_buffer";
  let summary = "tile-local layout-aware buffer";
  let description = [{
    `!wafer.tile_buffer` is the tile-region buffer abstraction used after
    tensor-level grouping and before storage realization. It carries the
    logical ranked tensor type, accepted Wafer physical layout family, and
    Wafer memory space. It does not carry physical address, offset, packet, or
    runtime buffer-object facts.
  }];

  let parameters = (ins
    "::mlir::Type":$tensorType,
    "::mlir::Attribute":$layout,
    "::mlir::Attribute":$memorySpace
  );

  let genVerifyDecl = 1;
  let assemblyFormat = "`<` $tensorType `,` $layout `,` $memorySpace `>`";
}

#endif // WAFER_DIALECT_WAFER_IR_WAFERTYPES_TD
