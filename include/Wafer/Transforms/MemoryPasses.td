//===- MemoryPasses.td - Bufferization and memory planning -*- tablegen -*-===//

#ifndef WAFER_TRANSFORMS_MEMORYPASSES_TD
#define WAFER_TRANSFORMS_MEMORYPASSES_TD

def BufferizeInstrFunctionBoundariesPass
    : Pass<"wafer-bufferize-instr-function-boundaries", "mlir::ModuleOp"> {
  let summary =
      "Bufferize Instr functions with the fixed Wafer function-boundary contract";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::bufferization::BufferizationDialect",
                           "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::tensor::TensorDialect",
                           "wafer::WaferDialect"
  ];
}

def PlanSPMMemoryPass
    : Pass<"wafer-plan-spm-memory", "mlir::ModuleOp"> {
  let summary =
      "Plan Wafer SPM memref allocations with tile-local byte offsets";
  let dependentDialects = ["mlir::memref::MemRefDialect",
                           "wafer::WaferDialect"
  ];
  let options = [
    Option<"spmBase", "spm-base", "int64_t", /*default=*/"65536",
           "Inclusive byte offset for ordinary tile-local SPM planning">,
    Option<"spmLimit", "spm-limit", "int64_t", /*default=*/"3080192",
           "Exclusive byte limit for ordinary tile-local SPM planning">,
    Option<"spmAlignment", "spm-alignment", "int64_t", /*default=*/"256",
           "Preferred byte alignment for ordinary tile-local SPM planning">
  ];
  let statistics = [
    Statistic<"numTimelineScopes", "timeline-scopes",
              "Number of operation scopes served by managed structured timelines">,
    Statistic<"numAssignedAllocations", "assigned-allocations",
              "Number of SPM allocations assigned a physical byte offset">
  ];
}

def PlanDDRMemoryPass
    : Pass<"wafer-plan-ddr-memory", "mlir::ModuleOp"> {
  let summary =
      "Plan and verify DDR memory demands in instruction-level Wafer IR";
  let dependentDialects = ["mlir::async::AsyncDialect",
                           "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect", "wafer::WaferDialect"
  ];
  let options = [
    Option<"ddrCapacityBytes", "ddr-capacity-bytes", "int64_t",
           /*default=*/"9223372036854775807",
           "Total DDR bytes available to the planned DDR memory footprint">,
    Option<"ddrLargestContiguousBytes", "ddr-largest-contiguous-bytes",
           "int64_t", /*default=*/"9223372036854775807",
           "Largest contiguous DDR range available to a single root demand">,
    Option<"ddrBandwidthLimitBytes", "ddr-bandwidth-limit-bytes", "int64_t",
           /*default=*/"9223372036854775807",
           "Total DDR movement bytes allowed by this planning gate">,
    Option<"ddrAlignmentBytes", "ddr-alignment-bytes", "int64_t",
           /*default=*/"256",
           "Preferred byte alignment for compiler-managed DDR planning">
  ];
  let statistics = [
    Statistic<"numTimelineScopes", "timeline-scopes",
              "Number of operation scopes served by managed structured timelines">,
    Statistic<"numAssignedAllocations", "assigned-allocations",
              "Number of DDR allocations assigned a physical byte offset">
  ];
}

#endif // WAFER_TRANSFORMS_MEMORYPASSES_TD
