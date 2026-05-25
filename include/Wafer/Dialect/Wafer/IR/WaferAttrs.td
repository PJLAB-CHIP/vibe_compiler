//===- WaferAttrs.td - Wafer shared attributes ------------*- tablegen -*-===//

#ifndef WAFER_DIALECT_WAFER_IR_WAFERATTRS_TD
#define WAFER_DIALECT_WAFER_IR_WAFERATTRS_TD

include "Wafer/Dialect/Wafer/IR/WaferBase.td"
include "mlir/IR/EnumAttr.td"

def Wafer_TargetV0 : I32EnumAttrCase<"WaferV0", 0, "wafer_v0">;
def Wafer_Target : I32EnumAttr<"Target", "Wafer target profile", [
  Wafer_TargetV0
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

def Wafer_MemorySpaceAttr
    : EnumAttr<Wafer_Dialect, Wafer_MemorySpace, "memory_space"> {
  let assemblyFormat = "`<` $value `>`";
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

def Wafer_MemLayoutAttr
    : EnumAttr<Wafer_Dialect, Wafer_MemLayout, "mem_layout"> {
  let assemblyFormat = "`<` $value `>`";
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

#endif // WAFER_DIALECT_WAFER_IR_WAFERATTRS_TD
