//===- CardExecutableTestSupport.h - Compiler executable fixtures -*- C++ -*-===//

#ifndef WAFER_UNITTESTS_COMPILER_CARDEXECUTABLETESTSUPPORT_H
#define WAFER_UNITTESTS_COMPILER_CARDEXECUTABLETESTSUPPORT_H

#include "Wafer/Planning/Baseline/CardBaselineCompilation.h"
#include "Wafer/Planning/Baseline/CardBaselinePlacement.h"
#include "Wafer/CodeGen/Executable/CardExecutableInternal.h"
#include "Wafer/Transforms/Bufferization/SelectedBufferMaterialization.h"
#include "Wafer/Driver/CompilationInternal.h"
#include "Wafer/Planning/PhysicalDataflow/TemporalTileShape.h"
#include "Wafer/Program/ProgramData.h"

#include "Wafer/Driver/Compilation.h"
#include "Wafer/Support/OptimizationConfig.h"
#include "Wafer/Support/TargetPolicy.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace wafer::compiler::testing {

frontend::ProgramPartitionSlice
singlePartitionSlice(llvm::ArrayRef<int64_t> shape);
frontend::ProgramBoundaryBinding
boundary(int64_t index, llvm::ArrayRef<int64_t> shape);

frontend::FrontendProgramVerificationResult programMetadata();
frontend::FrontendProgramVerificationResult branchMetadata();
frontend::FrontendProgramVerificationResult dependentProgramMetadata();
frontend::FrontendProgramVerificationResult largeTemporalProgramMetadata();
frontend::FrontendProgramVerificationResult
largeProducerStageProgramMetadata();
frontend::FrontendProgramVerificationResult layoutPipelineProgramMetadata();
frontend::FrontendProgramVerificationResult twoReductionAxisProgramMetadata();
frontend::FrontendProgramVerificationResult broadcastProgramMetadata();
frontend::FrontendProgramVerificationResult reductionDemandProgramMetadata();
frontend::FrontendProgramVerificationResult windowDemandProgramMetadata();
frontend::FrontendProgramVerificationResult stridedDemandProgramMetadata();
frontend::FrontendProgramVerificationResult multiPieceDemandProgramMetadata();
frontend::FrontendProgramVerificationResult multiProducerJoinProgramMetadata();

struct ParsedProgram {
  std::shared_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};

ParsedProgram parseProgram();
ParsedProgram parseBranchProgram();
ParsedProgram parseDependentProgram();
ParsedProgram parseThreeStageDependentProgram();
ParsedProgram parseLargeTemporalProgram();
ParsedProgram parseLargeProducerStageProgram();
ParsedProgram parseLayoutPipelineProgram();
ParsedProgram parseTwoReductionAxisProgram();
ParsedProgram parseBroadcastProgram();
ParsedProgram parseReductionDemandProgram();
ParsedProgram parseWindowDemandProgram();
ParsedProgram parseStridedDemandProgram();
ParsedProgram parseMultiPieceDemandProgram();
ParsedProgram parseMultiProducerJoinProgram();

size_t countOccurrences(llvm::StringRef text, llvm::StringRef needle);
ExecutionConfig executionConfig();

void expectCompleteTileDomain(
    const detail::CardExecutableLoweringResult &executable,
    llvm::ArrayRef<std::string> tileDataflowIRTrace);

void expectDemandProgramCompletesExecutableGate(
    ParsedProgram &parsed,
    const frontend::FrontendProgramVerificationResult &metadata);

} // namespace wafer::compiler::testing

#endif // WAFER_UNITTESTS_COMPILER_CARDEXECUTABLETESTSUPPORT_H
