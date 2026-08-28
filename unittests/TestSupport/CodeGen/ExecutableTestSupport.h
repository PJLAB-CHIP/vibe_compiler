//===- ExecutableTestSupport.h - Compiler executable fixtures -*- C++
//-*-===//

#ifndef WAFER_UNITTESTS_COMPILER_EXECUTABLETESTSUPPORT_H
#define WAFER_UNITTESTS_COMPILER_EXECUTABLETESTSUPPORT_H

#include "Wafer/CodeGen/DeviceExecutableInternal.h"
#include "Wafer/Driver/ExecutableCompilation.h"
#include "Wafer/Driver/CompilationInternal.h"
#include "Wafer/Program/ProgramData.h"

#include "Wafer/Driver/Compilation.h"
#include "Wafer/Support/OptimizationConfig.h"
#include "Wafer/Target/Core/TargetMemory.h"

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
frontend::ProgramBoundaryBinding boundary(int64_t index,
                                          llvm::ArrayRef<int64_t> shape);

frontend::FrontendProgramVerificationResult programMetadata();
frontend::FrontendProgramVerificationResult branchMetadata();
frontend::FrontendProgramVerificationResult dependentProgramMetadata();
frontend::FrontendProgramVerificationResult largeTemporalProgramMetadata();
frontend::FrontendProgramVerificationResult largeProducerStageProgramMetadata();
frontend::FrontendProgramVerificationResult
largeTransposedWeightProgramMetadata();
frontend::FrontendProgramVerificationResult layoutPipelineProgramMetadata();
frontend::FrontendProgramVerificationResult twoReductionAxisProgramMetadata();
frontend::FrontendProgramVerificationResult broadcastProgramMetadata();
frontend::FrontendProgramVerificationResult reductionDemandProgramMetadata();
frontend::FrontendProgramVerificationResult windowDemandProgramMetadata();
frontend::FrontendProgramVerificationResult stridedDemandProgramMetadata();
frontend::FrontendProgramVerificationResult multiPieceDemandProgramMetadata();
frontend::FrontendProgramVerificationResult
multiProducerJoinProgramMetadata(int64_t extent);

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
ParsedProgram parseLargeTransposedWeightProgram();
ParsedProgram parseLayoutPipelineProgram();
ParsedProgram parseTwoReductionAxisProgram();
ParsedProgram parseBroadcastProgram();
ParsedProgram parseReductionDemandProgram();
ParsedProgram parseWindowDemandProgram();
ParsedProgram parseStridedDemandProgram();
ParsedProgram parseMultiPieceDemandProgram();
ParsedProgram parseMultiProducerJoinProgram(int64_t extent);

size_t countOccurrences(llvm::StringRef text, llvm::StringRef needle);
ExecutionConfig executionConfig();

void expectCompleteTileDomain(
    const detail::ExecutableLoweringResult &executable,
    llvm::ArrayRef<std::string> tileDataflowIRTrace);

} // namespace wafer::compiler::testing

#endif // WAFER_UNITTESTS_COMPILER_EXECUTABLETESTSUPPORT_H
