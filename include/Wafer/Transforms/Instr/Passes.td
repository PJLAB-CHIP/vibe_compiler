//===- Passes.td - Instr-level transforms ------------------*- tablegen -*-===//

#ifndef WAFER_TRANSFORMS_INSTR_PASSES_TD
#define WAFER_TRANSFORMS_INSTR_PASSES_TD

def PlanSPMMemoryPass
    : Pass<"wafer-plan-spm-memory", "mlir::ModuleOp"> {
  let summary =
      "Plan Wafer SPM memref allocations with tile-local byte offsets";
  let dependentDialects = ["mlir::memref::MemRefDialect",
                           "wafer::WaferDialect"];
  let options = [
    Option<"spmBase", "spm-base", "int64_t", /*default=*/"65536",
           "Inclusive byte offset for ordinary tile-local SPM planning">,
    Option<"spmLimit", "spm-limit", "int64_t", /*default=*/"3080192",
           "Exclusive byte limit for ordinary tile-local SPM planning">,
    Option<"spmAlignment", "spm-alignment", "int64_t", /*default=*/"256",
           "Preferred byte alignment for ordinary tile-local SPM planning">,
    Option<"emitCapacityDiagnostics", "emit-capacity-diagnostics", "bool",
           /*default=*/"true",
           "Emit an error diagnostic for a proven capacity rejection">
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
                           "mlir::scf::SCFDialect",
                           "wafer::WaferDialect"];
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

def PlaceRequiredNCCJoinsPass
    : Pass<"wafer-place-required-ncc-joins", "mlir::func::FuncOp"> {
  let summary =
      "Place joins required by outstanding NCC accesses in one function";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::async::AsyncDialect",
                           "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect",
                           "wafer::WaferDialect"];
  let statistics = [
    Statistic<"numRequiredJoinsInserted", "required-joins-inserted",
              "Number of required NCC joins inserted in the function">
  ];
}

def RebuildRequiredNCCJoinsPass
    : Pass<"wafer-rebuild-required-ncc-joins", "mlir::func::FuncOp"> {
  let summary =
      "Erase derived NCC joins and recompute required joins in one function";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::async::AsyncDialect",
                           "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect",
                           "wafer::WaferDialect"];
  let statistics = [
    Statistic<"numDerivedJoinsRemoved", "derived-joins-removed",
              "Number of existing derived NCC joins removed before rebuilding">,
    Statistic<"numRequiredJoinsInserted", "required-joins-inserted",
              "Number of required joins present after rebuilding">
  ];
}

#endif // WAFER_TRANSFORMS_INSTR_PASSES_TD
