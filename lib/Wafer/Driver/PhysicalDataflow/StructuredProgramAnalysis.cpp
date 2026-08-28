//===- StructuredProgramAnalysis.cpp
//---------------------------------------===//

#include "Wafer/Driver/PhysicalDataflow/StructuredProgramAnalysis.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace wafer::compiler::detail {
namespace {

mlir::FailureOr<mlir::func::FuncOp>
getStructuredProgram(mlir::ModuleOp module, std::string &failureReason) {
  mlir::func::FuncOp program;
  for (mlir::func::FuncOp function : module.getOps<mlir::func::FuncOp>()) {
    if (function.isExternal())
      continue;
    if (program) {
      failureReason = "structured program analysis requires exactly one "
                      "defined tensor program";
      return mlir::failure();
    }
    program = function;
  }
  if (!program) {
    failureReason = "structured program analysis requires exactly one "
                    "defined tensor program";
    return mlir::failure();
  }
  return program;
}

mlir::FailureOr<StaticOutputDomains>
getStaticOutputDomains(mlir::func::FuncOp program, std::string &failureReason) {
  if (program.getNumResults() == 0) {
    failureReason =
        "structured program analysis requires tensor output domains";
    return mlir::failure();
  }
  StaticOutputDomains domains;
  domains.reserve(program.getNumResults());
  for (mlir::Type resultType : program.getResultTypes()) {
    auto ranked = mlir::dyn_cast<mlir::RankedTensorType>(resultType);
    if (!ranked || !ranked.hasStaticShape() ||
        llvm::any_of(ranked.getShape(),
                     [](int64_t extent) { return extent <= 0; })) {
      failureReason =
          "structured program analysis requires static tensor output domains";
      return mlir::failure();
    }
    domains.emplace_back(ranked.getShape());
  }
  return domains;
}

} // namespace

mlir::FailureOr<std::unique_ptr<StructuredProgramAnalysis>>
analyzeStructuredProgram(
    mlir::ModuleOp tensorProgram,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics) {
  if (!tensorProgram || executionConfig.getNumPartitions() != 1 ||
      program.numPartitions != executionConfig.getNumPartitions()) {
    diagnostics << "wafer-compile: structured program analysis requires one "
                   "logical card partition\n";
    return mlir::failure();
  }

  std::string failureReason;
  mlir::FailureOr<TargetTopology> topology =
      TargetTopology::create(tensorProgram, &failureReason);
  if (mlir::failed(topology)) {
    diagnostics << "wafer-compile: cannot construct physical topology: "
                << failureReason << '\n';
    return mlir::failure();
  }
  constexpr CardId cardId(0);
  std::optional<llvm::ArrayRef<TileId>> availableTileIds =
      topology->getAvailableTileIds(cardId);
  if (!availableTileIds ||
      availableTileIds->size() !=
          static_cast<size_t>(executionConfig.getTileCount())) {
    diagnostics << "wafer-compile: target topology does not provide the "
                   "exact configured Tile domain\n";
    return mlir::failure();
  }

  mlir::FailureOr<mlir::func::FuncOp> structuredProgram =
      getStructuredProgram(tensorProgram, failureReason);
  if (mlir::failed(structuredProgram)) {
    diagnostics << "wafer-compile: " << failureReason << '\n';
    return mlir::failure();
  }
  mlir::FailureOr<StructuredDAGAnalysis> dag =
      StructuredDAGAnalysis::create(*structuredProgram, &failureReason);
  if (mlir::failed(dag)) {
    diagnostics << "wafer-compile: cannot derive structured DAG: "
                << failureReason << '\n';
    return mlir::failure();
  }
  mlir::FailureOr<StaticOutputDomains> outputDomains =
      getStaticOutputDomains(*structuredProgram, failureReason);
  if (mlir::failed(outputDomains)) {
    diagnostics << "wafer-compile: " << failureReason << '\n';
    return mlir::failure();
  }

  llvm::SmallVector<StructuredOperationNodeMapping, 16> operationNodes;
  operationNodes.reserve(dag->getNodes().size());
  for (const StructuredDAGNode &node : dag->getNodes())
    operationNodes.push_back({node.operation, node.id});
  llvm::SmallVector<TileId, 16> copiedTileIds(availableTileIds->begin(),
                                              availableTileIds->end());
  return std::make_unique<StructuredProgramAnalysis>(
      std::move(*topology), std::move(copiedTileIds), std::move(*dag),
      std::move(*outputDomains), std::move(operationNodes));
}

} // namespace wafer::compiler::detail
