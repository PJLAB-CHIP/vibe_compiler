//===- WaferTypes.td - Wafer type definitions -------------*- tablegen -*-===//

#ifndef WAFER_IR_WAFERTYPES_TD
#define WAFER_IR_WAFERTYPES_TD

include "Wafer/IR/WaferAttrs.td"
include "Wafer/IR/WaferBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonTypeConstraints.td"

def Wafer_StorageType : TypeDef<Wafer_Dialect, "Storage"> {
  let mnemonic = "storage";
  let summary = "layout-aware storage value";
  let description = [{
    `!wafer.storage` is the tile-region storage abstraction used after
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

def Wafer_AnyStorage : Type<
  CPred<"::llvm::isa<::wafer::StorageType>($_self)">,
  "Wafer storage">;

def Wafer_AnyRankedTensorOrStorage : AnyTypeOf<[
  AnyRankedTensor,
  Wafer_AnyStorage
]>;

#endif // WAFER_IR_WAFERTYPES_TD
