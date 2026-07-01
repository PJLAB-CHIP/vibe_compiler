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

def Wafer_SPMOffsetAttr : AttrDef<Wafer_Dialect, "SPMOffset"> {
  let mnemonic = "spm_offset";
  let summary = "Accepted Wafer SPM byte offset fact";
  let description = [{
    `#wafer.spm_offset<offset>` records an accepted tile-local SPM byte offset
    for a Wafer SPM memref definition. The range size, alignment and bank-line
    span are recomputed from the memref type, Wafer layout and target policy.

    This attribute is an accepted SPM memory-planning fact for
    instruction-level IR. It does not encode planner search trace, runtime
    handles, packet fields, or DDR memory ownership.
  }];

  let parameters = (ins
    "int64_t":$offset
  );
  let assemblyFormat = "`<` $offset `>`";
}

def Wafer_DDROffsetAttr : AttrDef<Wafer_Dialect, "DDROffset"> {
  let mnemonic = "ddr_offset";
  let summary = "Accepted Wafer DDR byte offset fact";
  let description = [{
    `#wafer.ddr_offset<offset>` records an accepted symbolic DDR byte offset
    for a compiler-managed Wafer DDR memref allocation. The range size,
    alignment, lifetime and read/write demand are recomputed from the memref
    type, Wafer layout, SSA uses and movement descriptors.

    This attribute is an accepted DDR memory-planning fact for
    instruction-level IR. It does not encode external binding requirements,
    runtime handles, packet fields, physical DDR addresses or planner search
    trace.
  }];

  let parameters = (ins
    "int64_t":$offset
  );
  let assemblyFormat = "`<` $offset `>`";
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
def Wafer_ComputeElementwiseKindSelect : I32EnumAttrCase<"Select", 18, "select">;
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
  Wafer_ComputeElementwiseKindGe,
  Wafer_ComputeElementwiseKindSelect
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

def Wafer_InstrElementwiseKindAdd : I32EnumAttrCase<"Add", 0, "add">;
def Wafer_InstrElementwiseKindSub : I32EnumAttrCase<"Sub", 1, "sub">;
def Wafer_InstrElementwiseKindMul : I32EnumAttrCase<"Mul", 2, "mul">;
def Wafer_InstrElementwiseKindDiv : I32EnumAttrCase<"Div", 3, "div">;
def Wafer_InstrElementwiseKindMax : I32EnumAttrCase<"Max", 4, "max">;
def Wafer_InstrElementwiseKindMin : I32EnumAttrCase<"Min", 5, "min">;
def Wafer_InstrElementwiseKindNeg : I32EnumAttrCase<"Neg", 6, "neg">;
def Wafer_InstrElementwiseKindRecip : I32EnumAttrCase<"Recip", 7, "recip">;
def Wafer_InstrElementwiseKindSqrt : I32EnumAttrCase<"Sqrt", 8, "sqrt">;
def Wafer_InstrElementwiseKindRsqrt : I32EnumAttrCase<"Rsqrt", 9, "rsqrt">;
def Wafer_InstrElementwiseKindExp : I32EnumAttrCase<"Exp", 10, "exp">;
def Wafer_InstrElementwiseKindTanh : I32EnumAttrCase<"Tanh", 11, "tanh">;
def Wafer_InstrElementwiseKindEq : I32EnumAttrCase<"Eq", 12, "eq">;
def Wafer_InstrElementwiseKindNe : I32EnumAttrCase<"Ne", 13, "ne">;
def Wafer_InstrElementwiseKindLt : I32EnumAttrCase<"Lt", 14, "lt">;
def Wafer_InstrElementwiseKindLe : I32EnumAttrCase<"Le", 15, "le">;
def Wafer_InstrElementwiseKindGt : I32EnumAttrCase<"Gt", 16, "gt">;
def Wafer_InstrElementwiseKindGe : I32EnumAttrCase<"Ge", 17, "ge">;
def Wafer_InstrElementwiseKind
    : I32EnumAttr<"InstrElementwiseKind",
                  "Wafer instruction CT elementwise target kind", [
  Wafer_InstrElementwiseKindAdd,
  Wafer_InstrElementwiseKindSub,
  Wafer_InstrElementwiseKindMul,
  Wafer_InstrElementwiseKindDiv,
  Wafer_InstrElementwiseKindMax,
  Wafer_InstrElementwiseKindMin,
  Wafer_InstrElementwiseKindNeg,
  Wafer_InstrElementwiseKindRecip,
  Wafer_InstrElementwiseKindSqrt,
  Wafer_InstrElementwiseKindRsqrt,
  Wafer_InstrElementwiseKindExp,
  Wafer_InstrElementwiseKindTanh,
  Wafer_InstrElementwiseKindEq,
  Wafer_InstrElementwiseKindNe,
  Wafer_InstrElementwiseKindLt,
  Wafer_InstrElementwiseKindLe,
  Wafer_InstrElementwiseKindGt,
  Wafer_InstrElementwiseKindGe
]> {
  let cppNamespace = "::wafer";
  let genSpecializedAttr = 0;
}

def Wafer_InstrElementwiseKindAttr
    : EnumAttr<Wafer_Dialect, Wafer_InstrElementwiseKind,
               "instr_elementwise_kind"> {
  let assemblyFormat = "`<` $value `>`";
}

def Wafer_InstrReduceKindSum : I32EnumAttrCase<"Sum", 0, "sum">;
def Wafer_InstrReduceKindMax : I32EnumAttrCase<"Max", 1, "max">;
def Wafer_InstrReduceKindMin : I32EnumAttrCase<"Min", 2, "min">;
def Wafer_InstrReduceKindAvg : I32EnumAttrCase<"Avg", 3, "avg">;
def Wafer_InstrReduceKind
    : I32EnumAttr<"InstrReduceKind",
                  "Wafer instruction CT reduce target kind", [
  Wafer_InstrReduceKindSum,
  Wafer_InstrReduceKindMax,
  Wafer_InstrReduceKindMin,
  Wafer_InstrReduceKindAvg
]> {
  let cppNamespace = "::wafer";
  let genSpecializedAttr = 0;
}

def Wafer_InstrReduceKindAttr
    : EnumAttr<Wafer_Dialect, Wafer_InstrReduceKind, "instr_reduce_kind"> {
  let assemblyFormat = "`<` $value `>`";
}

def Wafer_InstrConvertKindInt8Fp16 : I32EnumAttrCase<"Int8Fp16", 139, "int8_fp16">;
def Wafer_InstrConvertKindInt8Bf16 : I32EnumAttrCase<"Int8Bf16", 140, "int8_bf16">;
def Wafer_InstrConvertKindInt8Fp32 : I32EnumAttrCase<"Int8Fp32", 141, "int8_fp32">;
def Wafer_InstrConvertKindInt8Tf32 : I32EnumAttrCase<"Int8Tf32", 142, "int8_tf32">;
def Wafer_InstrConvertKindInt16Fp16 : I32EnumAttrCase<"Int16Fp16", 143, "int16_fp16">;
def Wafer_InstrConvertKindInt16Bf16 : I32EnumAttrCase<"Int16Bf16", 144, "int16_bf16">;
def Wafer_InstrConvertKindInt16Fp32 : I32EnumAttrCase<"Int16Fp32", 145, "int16_fp32">;
def Wafer_InstrConvertKindInt16Tf32 : I32EnumAttrCase<"Int16Tf32", 146, "int16_tf32">;
def Wafer_InstrConvertKindInt32Fp16 : I32EnumAttrCase<"Int32Fp16", 147, "int32_fp16">;
def Wafer_InstrConvertKindInt32Bf16 : I32EnumAttrCase<"Int32Bf16", 148, "int32_bf16">;
def Wafer_InstrConvertKindInt32Fp32 : I32EnumAttrCase<"Int32Fp32", 149, "int32_fp32">;
def Wafer_InstrConvertKindInt32Tf32 : I32EnumAttrCase<"Int32Tf32", 150, "int32_tf32">;
def Wafer_InstrConvertKindBf16Int8 : I32EnumAttrCase<"Bf16Int8", 151, "bf16_int8">;
def Wafer_InstrConvertKindBf16Int16 : I32EnumAttrCase<"Bf16Int16", 152, "bf16_int16">;
def Wafer_InstrConvertKindBf16Int32 : I32EnumAttrCase<"Bf16Int32", 153, "bf16_int32">;
def Wafer_InstrConvertKindBf16Fp16 : I32EnumAttrCase<"Bf16Fp16", 154, "bf16_fp16">;
def Wafer_InstrConvertKindBf16Fp32 : I32EnumAttrCase<"Bf16Fp32", 155, "bf16_fp32">;
def Wafer_InstrConvertKindBf16Tf32 : I32EnumAttrCase<"Bf16Tf32", 156, "bf16_tf32">;
def Wafer_InstrConvertKindFp16Int8 : I32EnumAttrCase<"Fp16Int8", 157, "fp16_int8">;
def Wafer_InstrConvertKindFp16Int16 : I32EnumAttrCase<"Fp16Int16", 158, "fp16_int16">;
def Wafer_InstrConvertKindFp16Int32 : I32EnumAttrCase<"Fp16Int32", 159, "fp16_int32">;
def Wafer_InstrConvertKindFp16Bf16 : I32EnumAttrCase<"Fp16Bf16", 160, "fp16_bf16">;
def Wafer_InstrConvertKindFp16Fp32 : I32EnumAttrCase<"Fp16Fp32", 161, "fp16_fp32">;
def Wafer_InstrConvertKindFp16Tf32 : I32EnumAttrCase<"Fp16Tf32", 162, "fp16_tf32">;
def Wafer_InstrConvertKindFp32Int8 : I32EnumAttrCase<"Fp32Int8", 163, "fp32_int8">;
def Wafer_InstrConvertKindFp32Int16 : I32EnumAttrCase<"Fp32Int16", 164, "fp32_int16">;
def Wafer_InstrConvertKindFp32Int32 : I32EnumAttrCase<"Fp32Int32", 165, "fp32_int32">;
def Wafer_InstrConvertKindFp32Fp16 : I32EnumAttrCase<"Fp32Fp16", 166, "fp32_fp16">;
def Wafer_InstrConvertKindFp32Bf16 : I32EnumAttrCase<"Fp32Bf16", 167, "fp32_bf16">;
def Wafer_InstrConvertKindFp32Tf32 : I32EnumAttrCase<"Fp32Tf32", 168, "fp32_tf32">;
def Wafer_InstrConvertKindTf32Int8 : I32EnumAttrCase<"Tf32Int8", 169, "tf32_int8">;
def Wafer_InstrConvertKindTf32Int16 : I32EnumAttrCase<"Tf32Int16", 170, "tf32_int16">;
def Wafer_InstrConvertKindTf32Int32 : I32EnumAttrCase<"Tf32Int32", 171, "tf32_int32">;
def Wafer_InstrConvertKindTf32Fp16 : I32EnumAttrCase<"Tf32Fp16", 172, "tf32_fp16">;
def Wafer_InstrConvertKindTf32Bf16 : I32EnumAttrCase<"Tf32Bf16", 173, "tf32_bf16">;
def Wafer_InstrConvertKindTf32Fp32 : I32EnumAttrCase<"Tf32Fp32", 174, "tf32_fp32">;
def Wafer_InstrConvertKind
    : I32EnumAttr<"InstrConvertKind",
                  "Wafer instruction CT convert target kind", [
  Wafer_InstrConvertKindInt8Fp16,
  Wafer_InstrConvertKindInt8Bf16,
  Wafer_InstrConvertKindInt8Fp32,
  Wafer_InstrConvertKindInt8Tf32,
  Wafer_InstrConvertKindInt16Fp16,
  Wafer_InstrConvertKindInt16Bf16,
  Wafer_InstrConvertKindInt16Fp32,
  Wafer_InstrConvertKindInt16Tf32,
  Wafer_InstrConvertKindInt32Fp16,
  Wafer_InstrConvertKindInt32Bf16,
  Wafer_InstrConvertKindInt32Fp32,
  Wafer_InstrConvertKindInt32Tf32,
  Wafer_InstrConvertKindBf16Int8,
  Wafer_InstrConvertKindBf16Int16,
  Wafer_InstrConvertKindBf16Int32,
  Wafer_InstrConvertKindBf16Fp16,
  Wafer_InstrConvertKindBf16Fp32,
  Wafer_InstrConvertKindBf16Tf32,
  Wafer_InstrConvertKindFp16Int8,
  Wafer_InstrConvertKindFp16Int16,
  Wafer_InstrConvertKindFp16Int32,
  Wafer_InstrConvertKindFp16Bf16,
  Wafer_InstrConvertKindFp16Fp32,
  Wafer_InstrConvertKindFp16Tf32,
  Wafer_InstrConvertKindFp32Int8,
  Wafer_InstrConvertKindFp32Int16,
  Wafer_InstrConvertKindFp32Int32,
  Wafer_InstrConvertKindFp32Fp16,
  Wafer_InstrConvertKindFp32Bf16,
  Wafer_InstrConvertKindFp32Tf32,
  Wafer_InstrConvertKindTf32Int8,
  Wafer_InstrConvertKindTf32Int16,
  Wafer_InstrConvertKindTf32Int32,
  Wafer_InstrConvertKindTf32Fp16,
  Wafer_InstrConvertKindTf32Bf16,
  Wafer_InstrConvertKindTf32Fp32
]> {
  let cppNamespace = "::wafer";
  let genSpecializedAttr = 0;
}

def Wafer_InstrConvertKindAttr
    : EnumAttr<Wafer_Dialect, Wafer_InstrConvertKind, "instr_convert_kind"> {
  let assemblyFormat = "`<` $value `>`";
}

def Wafer_InstrFamilyCT : I32EnumAttrCase<"CT", 0, "ct">;
def Wafer_InstrFamilyNE : I32EnumAttrCase<"NE", 1, "ne">;
def Wafer_InstrFamilyRDMA : I32EnumAttrCase<"RDMA", 2, "rdma">;
def Wafer_InstrFamilyWDMA : I32EnumAttrCase<"WDMA", 3, "wdma">;
def Wafer_InstrFamilyTDMA : I32EnumAttrCase<"TDMA", 4, "tdma">;
def Wafer_InstrFamilyDTE : I32EnumAttrCase<"DTE", 5, "dte">;
def Wafer_InstrFamily : I32EnumAttr<"InstrFamily",
                                    "Wafer hardware instruction family", [
  Wafer_InstrFamilyCT,
  Wafer_InstrFamilyNE,
  Wafer_InstrFamilyRDMA,
  Wafer_InstrFamilyWDMA,
  Wafer_InstrFamilyTDMA,
  Wafer_InstrFamilyDTE
]> {
  let cppNamespace = "::wafer";
  let genSpecializedAttr = 0;
}

#endif // WAFER_IR_WAFERATTRS_TD
