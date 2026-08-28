//===- Passes.td - Instr to LLVM conversion --------------*- tablegen -*-===//

#ifndef WAFER_CONVERSION_INSTRTOLLVM_PASSES_TD
#define WAFER_CONVERSION_INSTRTOLLVM_PASSES_TD

def LowerInstrToTargetLLVMPass
    : Pass<"wafer-lower-instr-to-target-llvm", "mlir::ModuleOp"> {
  let summary =
      "Lower memory-planned wafer.instr ops to target LLVM CRT calls";
  let dependentDialects = ["mlir::cf::ControlFlowDialect",
                           "mlir::LLVM::LLVMDialect",
                           "wafer::WaferDialect"];
  let options = [
    Option<"defaultDDRArenaArgumentIndex",
           "default-ddr-arena-argument-index", "int64_t", /*default=*/"-1",
           "Entry argument carrying the default DDR arena base; -1 rejects "
           "compiler-managed DDR allocations">,
    Option<"cardId", "card-id", "int64_t", /*default=*/"-1",
           "Card containing the accepted Tile module; required "
           "when Direct DTE instructions are present">,
    Option<"tileId", "tile-id", "int64_t", /*default=*/"-1",
           "Tile whose accepted Direct DTE endpoints are lowered; "
           "required when Direct DTE instructions are present">,
    Option<"transportStatusArgumentIndex",
           "transport-status-argument-index", "int64_t", /*default=*/"-1",
           "Entry i64 argument carrying the launch-observable Direct DTE "
           "status address; its presence is the Direct DTE lifecycle contract">,
    Option<"transportPreparedBeforeEntry",
           "transport-prepared-before-entry", "bool", /*default=*/"false",
           "Debug direct-lowering projection stating that transport "
           "initialization completed in an earlier launch phase">,
    Option<"profileRecordArgumentIndex",
           "profile-record-argument-index", "int64_t", /*default=*/"-1",
           "Entry i64 argument carrying the per-launch profile record base; "
           "-1 disables profile instrumentation">
  ];
}

#endif // WAFER_CONVERSION_INSTRTOLLVM_PASSES_TD
