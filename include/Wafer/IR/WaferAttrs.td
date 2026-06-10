//===- WaferAttrs.td - Wafer shared attributes ------------*- tablegen -*-===//

#ifndef WAFER_IR_WAFERATTRS_TD
#define WAFER_IR_WAFERATTRS_TD

include "Wafer/IR/WaferBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

def Wafer_TargetWafer : I32EnumAttrCase<"Wafer", 0, "wafer">;
def Wafer_Target : I32EnumAttr<"Target", "Wafer target profile", [
  Wafer_TargetWafer
]> {
  let cppNamespace = "::wafer";
  let genSpecializedAttr = 0;
}

def Wafer_TargetAttr : EnumAttr<Wafer_Dialect, Wafer_Target, "target"> {
  let assemblyFormat = "`<` $value `>`";
}

def Wafer_MemorySpaceSPM : I32EnumAttrCase<"SPM", 0, "spm">;
def Wafer_MemorySpaceDDR : I32EnumAttrCase<"DDR", 1, "ddr">;
def Wafer_MemorySpace : I32EnumAttr<"MemorySpace", "Wafer addressable storage space", [
  Wafer_MemorySpaceSPM,
  Wafer_MemorySpaceDDR
]> {
  let cppNamespace = "::wafer";
  let genSpecializedAttr = 0;
}

def Wafer_MemLayoutTensor : I32EnumAttrCase<"Tensor", 0, "tensor">;
def Wafer_MemLayoutNTensor : I32EnumAttrCase<"NTensor", 1, "ntensor">;
def Wafer_MemLayoutCx : I32EnumAttrCase<"Cx", 2, "cx">;
def Wafer_MemLayoutNCx : I32EnumAttrCase<"NCx", 3, "ncx">;
def Wafer_MemLayout : I32EnumAttr<"MemLayout", "Wafer physical memory layout family", [
  Wafer_MemLayoutTensor,
  Wafer_MemLayoutNTensor,
  Wafer_MemLayoutCx,
  Wafer_MemLayoutNCx
]> {
  let cppNamespace = "::wafer";
  let genSpecializedAttr = 0;
}

def Wafer_MemoryAttr : AttrDef<Wafer_Dialect, "Memory"> {
  let mnemonic = "memory";
  let summary = "Wafer memory-space and physical layout marker";
  let description = [{
    `#wafer.memory<space, layout>` is the memory-space attribute carried by
    Wafer-owned memref values. The memref shape and element type remain the
    logical tensor view; this attribute records which Wafer addressable memory
    space contains the buffer and which physical layout family the buffer uses.
    It does not encode a physical SPM offset, runtime handle, packet field, or
    ABI descriptor.
  }];

  let parameters = (ins
    "wafer::MemorySpace":$space,
    "wafer::MemLayout":$layout
  );
  let assemblyFormat = "`<` $space `,` $layout `>`";
}

def Wafer_SPMPlacementAttr : AttrDef<Wafer_Dialect, "SPMPlacement"> {
  let mnemonic = "spm_placement";
  let summary = "Accepted Wafer SPM byte placement fact";
  let description = [{
    `#wafer.spm_placement<offset, size, alignment, bank_begin, bank_limit>`
    records an accepted tile-local SPM byte range for a Wafer SPM memref
    definition. The `size` is the physical storage size computed from the
    memref type and Wafer layout, not merely the logical compact tensor byte
    count. `bank_begin` and `bank_limit` describe the half-open span of 256B
    bank lines touched by the placed range.

    This attribute is an accepted placement fact for instruction-level IR. It
    does not encode allocator search trace, runtime handles, packet fields, or
    DDR resource ownership.
  }];

  let parameters = (ins
    "int64_t":$offset,
    "int64_t":$size,
    "int64_t":$alignment,
    "int64_t":$bankBegin,
    "int64_t":$bankLimit
  );
  let assemblyFormat = "`<` $offset `,` $size `,` $alignment `,` "
                       "$bankBegin `,` $bankLimit `>`";
}

def Wafer_PlacementSingleTile : I32EnumAttrCase<"SingleTile", 0, "single_tile">;
def Wafer_PlacementTileMesh : I32EnumAttrCase<"TileMesh", 1, "tile_mesh">;
def Wafer_PlacementCardMesh : I32EnumAttrCase<"CardMesh", 2, "card_mesh">;
def Wafer_Placement : I32EnumAttr<"Placement", "Wafer physical placement scope", [
  Wafer_PlacementSingleTile,
  Wafer_PlacementTileMesh,
  Wafer_PlacementCardMesh
]> {
  let cppNamespace = "::wafer";
  let genSpecializedAttr = 0;
}

def Wafer_PlacementAttr
    : EnumAttr<Wafer_Dialect, Wafer_Placement, "placement"> {
  let assemblyFormat = "`<` $value `>`";
}

def Wafer_ComputeElementwiseKindAdd : I32EnumAttrCase<"Add", 0, "add">;
def Wafer_ComputeElementwiseKindSub : I32EnumAttrCase<"Sub", 1, "sub">;
def Wafer_ComputeElementwiseKindMul : I32EnumAttrCase<"Mul", 2, "mul">;
def Wafer_ComputeElementwiseKindDiv : I32EnumAttrCase<"Div", 3, "div">;
def Wafer_ComputeElementwiseKindMax : I32EnumAttrCase<"Max", 4, "max">;
def Wafer_ComputeElementwiseKindMin : I32EnumAttrCase<"Min", 5, "min">;
def Wafer_ComputeElementwiseKindNeg : I32EnumAttrCase<"Neg", 6, "neg">;
def Wafer_ComputeElementwiseKindRecip : I32EnumAttrCase<"Recip", 7, "recip">;
def Wafer_ComputeElementwiseKindSqrt : I32EnumAttrCase<"Sqrt", 8, "sqrt">;
def Wafer_ComputeElementwiseKindRsqrt : I32EnumAttrCase<"Rsqrt", 9, "rsqrt">;
def Wafer_ComputeElementwiseKindExp : I32EnumAttrCase<"Exp", 10, "exp">;
def Wafer_ComputeElementwiseKindTanh : I32EnumAttrCase<"Tanh", 11, "tanh">;
def Wafer_ComputeElementwiseKindEq : I32EnumAttrCase<"Eq", 12, "eq">;
def Wafer_ComputeElementwiseKindNe : I32EnumAttrCase<"Ne", 13, "ne">;
def Wafer_ComputeElementwiseKindLt : I32EnumAttrCase<"Lt", 14, "lt">;
def Wafer_ComputeElementwiseKindLe : I32EnumAttrCase<"Le", 15, "le">;
def Wafer_ComputeElementwiseKindGt : I32EnumAttrCase<"Gt", 16, "gt">;
def Wafer_ComputeElementwiseKindGe : I32EnumAttrCase<"Ge", 17, "ge">;
def Wafer_ComputeElementwiseKind
    : I32EnumAttr<"ComputeElementwiseKind",
                  "Wafer target-abstract elementwise compute kind", [
  Wafer_ComputeElementwiseKindAdd,
  Wafer_ComputeElementwiseKindSub,
  Wafer_ComputeElementwiseKindMul,
  Wafer_ComputeElementwiseKindDiv,
  Wafer_ComputeElementwiseKindMax,
  Wafer_ComputeElementwiseKindMin,
  Wafer_ComputeElementwiseKindNeg,
  Wafer_ComputeElementwiseKindRecip,
  Wafer_ComputeElementwiseKindSqrt,
  Wafer_ComputeElementwiseKindRsqrt,
  Wafer_ComputeElementwiseKindExp,
  Wafer_ComputeElementwiseKindTanh,
  Wafer_ComputeElementwiseKindEq,
  Wafer_ComputeElementwiseKindNe,
  Wafer_ComputeElementwiseKindLt,
  Wafer_ComputeElementwiseKindLe,
  Wafer_ComputeElementwiseKindGt,
  Wafer_ComputeElementwiseKindGe
]> {
  let cppNamespace = "::wafer";
  let genSpecializedAttr = 0;
}

def Wafer_ComputeElementwiseKindAttr
    : EnumAttr<Wafer_Dialect, Wafer_ComputeElementwiseKind, "elementwise_kind"> {
  let assemblyFormat = "`<` $value `>`";
}

def Wafer_ComputeReduceKindSum : I32EnumAttrCase<"Sum", 0, "sum">;
def Wafer_ComputeReduceKindMax : I32EnumAttrCase<"Max", 1, "max">;
def Wafer_ComputeReduceKindMin : I32EnumAttrCase<"Min", 2, "min">;
def Wafer_ComputeReduceKindAvg : I32EnumAttrCase<"Avg", 3, "avg">;
def Wafer_ComputeReduceKind
    : I32EnumAttr<"ComputeReduceKind",
                  "Wafer target-abstract local reduce kind", [
  Wafer_ComputeReduceKindSum,
  Wafer_ComputeReduceKindMax,
  Wafer_ComputeReduceKindMin,
  Wafer_ComputeReduceKindAvg
]> {
  let cppNamespace = "::wafer";
  let genSpecializedAttr = 0;
}

def Wafer_ComputeReduceKindAttr
    : EnumAttr<Wafer_Dialect, Wafer_ComputeReduceKind, "reduce_kind"> {
  let assemblyFormat = "`<` $value `>`";
}

def Wafer_InstrQueueCT : I32EnumAttrCase<"CT", 0, "ct">;
def Wafer_InstrQueueNE : I32EnumAttrCase<"NE", 1, "ne">;
def Wafer_InstrQueueRDMA : I32EnumAttrCase<"RDMA", 2, "rdma">;
def Wafer_InstrQueueWDMA : I32EnumAttrCase<"WDMA", 3, "wdma">;
def Wafer_InstrQueueTDMA : I32EnumAttrCase<"TDMA", 4, "tdma">;
def Wafer_InstrQueue : I32EnumAttr<"InstrQueue",
                                   "Wafer instruction issue queue family", [
  Wafer_InstrQueueCT,
  Wafer_InstrQueueNE,
  Wafer_InstrQueueRDMA,
  Wafer_InstrQueueWDMA,
  Wafer_InstrQueueTDMA
]> {
  let cppNamespace = "::wafer";
  let genSpecializedAttr = 0;
}

#endif // WAFER_IR_WAFERATTRS_TD
