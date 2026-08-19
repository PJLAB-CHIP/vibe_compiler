//===- WaferTensorProgramToTileRegion.cpp - Structured shard lowering ----===//

#include "Internal.h"

#include "Wafer/Support/CompileTiming.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"

#include <memory>

using namespace wafer;
using namespace wafer::tensor_program_to_tile_region;

namespace {

static void normalizeMapOps(
    mlir::func::FuncOp function,
    llvm::MutableArrayRef<CandidatePeerEndpoint> peerEndpoints,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> &operationNodes) {
  llvm::SmallVector<mlir::linalg::MapOp, 8> maps;
  function.walk([&](mlir::linalg::MapOp map) { maps.push_back(map); });

  for (mlir::linalg::MapOp map : maps) {
    mlir::IRRewriter rewriter(function.getContext());
    rewriter.setInsertionPoint(map);
    auto linalgOp = mlir::cast<mlir::linalg::LinalgOp>(map.getOperation());
    auto generic = rewriter.create<mlir::linalg::GenericOp>(
        map.getLoc(), map->getResultTypes(), map.getInputs(),
        mlir::ValueRange{map.getInit()}, linalgOp.getIndexingMapsArray(),
        linalgOp.getIteratorTypesArray(),
        [&](mlir::OpBuilder &builder, mlir::Location loc,
            mlir::ValueRange arguments) {
          mlir::IRMapping mapping;
          mlir::Block &source = map.getMapper().front();
          for (auto [original, converted] :
               llvm::zip(source.getArguments(),
                         arguments.take_front(source.getNumArguments())))
            mapping.map(original, converted);
          for (mlir::Operation &operation : source.without_terminator())
            builder.clone(operation, mapping);
          auto yield =
              mlir::cast<mlir::linalg::YieldOp>(source.getTerminator());
          llvm::SmallVector<mlir::Value, 2> yielded;
          for (mlir::Value value : yield.getValues())
            yielded.push_back(mapping.lookupOrDefault(value));
          builder.create<mlir::linalg::YieldOp>(loc, yielded);
        });
    for (CandidatePeerEndpoint &endpoint : peerEndpoints)
      for (auto [oldResult, newResult] :
           llvm::zip_equal(map->getResults(), generic->getResults()))
        if (endpoint.value == oldResult)
          endpoint.value = newResult;
    for (StructuredOperationNodeMapping &mapping : operationNodes)
      if (mapping.operation == map.getOperation())
        mapping.operation = generic.getOperation();
    rewriter.replaceOp(map, generic->getResults());
  }
}

static mlir::LogicalResult rewriteTensorProgramInPlace(
    mlir::ModuleOp module, unsigned functionalArgumentCount,
    int64_t currentLogicalPartition, std::string *failureReason,
    llvm::SmallVector<CandidatePeerEndpoint, 8> peerEndpoints,
    llvm::ArrayRef<CandidateSelectedDDRStage> selectedDDRStages,
    TileRegionEmissionRelations *emissionRelations,
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes,
    bool requireOneStructuredRootPerRegion) {
  wafer::support::ScopedCompileTimingSpan totalTiming(
      "conversion-phase", "rewriteTensorProgramInPlace", "total");
  auto phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "analysis-phase", "rewriteTensorProgramInPlace", "verify-scope");
  mlir::func::FuncOp function = findSingleStandaloneTensorProgram(module);
  if (!function) {
    setFailureReason(
        failureReason,
        "standalone module must contain exactly one tensor program function");
    return mlir::failure();
  }
  TensorProgramScope scope(function, functionalArgumentCount);
  if (mlir::failed(verifyTensorProgramScope(function, functionalArgumentCount,
                                            failureReason,
                                            scope.getBoundaryArgumentCount())))
    return mlir::failure();

  phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "transformation-phase", "rewriteTensorProgramInPlace", "normalizeMapOps");
  llvm::SmallVector<StructuredOperationNodeMapping, 16>
      normalizedOperationNodes(operationNodes.begin(), operationNodes.end());
  normalizeMapOps(function, peerEndpoints, normalizedOperationNodes);
  llvm::SmallVector<mlir::Operation *, 16> sourceOperations;
  for (mlir::Operation &operation : scope.getBody().without_terminator())
    sourceOperations.push_back(&operation);

  mlir::func::ReturnOp oldReturn = scope.getReturn();
  mlir::IRRewriter rewriter(module.getContext());
  rewriter.setInsertionPoint(oldReturn);
  TileRegionBodyEmitter emitter(failureReason, currentLogicalPartition,
                                peerEndpoints, selectedDDRStages,
                                emissionRelations, normalizedOperationNodes);
  phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "conversion-phase", "rewriteTensorProgramInPlace",
      "TileRegionBodyEmitter::emit");
  mlir::FailureOr<TileRegionOp> tileRegion =
      requireOneStructuredRootPerRegion || !selectedDDRStages.empty()
          ? emitter.emitStructuredStages(scope, rewriter)
          : emitter.emit(scope, rewriter);
  if (mlir::failed(tileRegion))
    return mlir::failure();

  phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "transformation-phase", "rewriteTensorProgramInPlace", "replace-return");
  rewriter.setInsertionPoint(oldReturn);
  llvm::SmallVector<mlir::Value, 4> returnedTensors;
  for (mlir::Value result : (*tileRegion).getResults()) {
    auto tensor = rewriter.create<mlir::bufferization::ToTensorOp>(
        function.getLoc(), result, /*restrict=*/true, /*writeable=*/true);
    returnedTensors.push_back(tensor.getResult());
  }
  rewriter.create<mlir::func::ReturnOp>(function.getLoc(), returnedTensors);
  rewriter.eraseOp(oldReturn);

  // Tiling/fusion can place a nested user under a source operation whose block
  // position is not reverse topological order for the flattened use graph.
  // Delete the old program as an explicit dead closure. Any source value still
  // retained by emitted IR is a conversion failure in this private clone.
  phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "transformation-phase", "rewriteTensorProgramInPlace",
      "erase-source-closure");
  llvm::SmallVector<mlir::Operation *, 16> pending(sourceOperations);
  bool changed = true;
  while (changed && !pending.empty()) {
    changed = false;
    for (size_t index = 0; index < pending.size();) {
      mlir::Operation *operation = pending[index];
      if (!operation->use_empty()) {
        ++index;
        continue;
      }
      pending.erase(pending.begin() + index);
      rewriter.eraseOp(operation);
      changed = true;
    }
  }
  if (!pending.empty()) {
    std::string detail;
    llvm::raw_string_ostream os(detail);
    os << "tile-region lowering retained source operation "
       << pending.front()->getName();
    if (!pending.front()->use_empty())
      os << " through " << (*pending.front()->getUsers().begin())->getName();
    setFailureReason(failureReason, os.str());
    return mlir::failure();
  }
  return mlir::success();
}

} // namespace

mlir::LogicalResult wafer::tensor_program_to_tile_region::
    convertTensorProgramToTileRegionModuleInPlace(
        mlir::ModuleOp module, mlir::MLIRContext *context,
        unsigned functionalArgumentCount, int64_t currentLogicalPartition,
        std::string *failureReason, bool suppressDiagnostics, bool verifyResult,
        bool populateFallbackFailureReason,
        llvm::ArrayRef<CandidatePeerEndpoint> peerEndpoints,
        llvm::ArrayRef<CandidateSelectedDDRStage> selectedDDRStages,
        TileRegionEmissionRelations *emissionRelations,
        llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes,
        bool requireOneStructuredRootPerRegion) {
  wafer::support::ScopedCompileTimingSpan timing(
      "conversion", "convertTensorProgramToTileRegionModuleInPlace", "total");
  // The caller owns the already-private candidate and is the sole rollback
  // boundary.  A failed in-place conversion leaves that disposable candidate
  // mutated; adding another whole-module clone here would duplicate the same
  // rollback scope without improving failure isolation.
  if (emissionRelations) {
    emissionRelations->selectedDDRStages.clear();
    emissionRelations->materializedBuffers.clear();
  }
  llvm::SmallVector<CandidatePeerEndpoint, 8> mappedEndpoints(
      peerEndpoints.begin(), peerEndpoints.end());
  mlir::LogicalResult conversionResult = mlir::success();
  {
    wafer::support::ScopedCompileTimingSpan rewriteTiming(
        "conversion-phase", "convertTensorProgramToTileRegionModuleInPlace",
        "rewriteTensorProgramInPlace");
    if (suppressDiagnostics) {
      mlir::ScopedDiagnosticHandler handler(
          context, [](mlir::Diagnostic &) { return mlir::success(); });
      conversionResult = rewriteTensorProgramInPlace(
          module, functionalArgumentCount, currentLogicalPartition,
          failureReason, mappedEndpoints, selectedDDRStages, emissionRelations,
          operationNodes, requireOneStructuredRootPerRegion);
    } else {
      conversionResult = rewriteTensorProgramInPlace(
          module, functionalArgumentCount, currentLogicalPartition,
          failureReason, mappedEndpoints, selectedDDRStages, emissionRelations,
          operationNodes, requireOneStructuredRootPerRegion);
    }
  }

  if (mlir::failed(conversionResult)) {
    if (populateFallbackFailureReason &&
        (!failureReason || failureReason->empty()))
      setFailureReason(failureReason,
                       "tensor-program-to-tile-region lowering failed");
    return mlir::failure();
  }

  if (verifyResult) {
    wafer::support::ScopedCompileTimingSpan verifyTiming(
        "analysis-phase", "convertTensorProgramToTileRegionModuleInPlace",
        "verify");
    if (mlir::failed(mlir::verify(module))) {
      mlir::ScopedDiagnosticHandler dumpHandler(
          module.getContext(),
          [&](mlir::Diagnostic &diagnostic) {
            diagnostic.print(llvm::errs());
            llvm::errs() << "\n";
            return mlir::success();
          });
      (void)mlir::verify(module);
      setFailureReason(failureReason,
                       "lowered tile-region module failed verifier");
      return mlir::failure();
    }
  }

  return mlir::success();
}
