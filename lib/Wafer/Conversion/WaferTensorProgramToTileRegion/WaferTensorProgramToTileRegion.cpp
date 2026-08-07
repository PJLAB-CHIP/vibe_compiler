//===- WaferTensorProgramToTileRegion.cpp - Tensor program lowering ------===//

#include "Internal.h"

#include "Wafer/Support/CompileTiming.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"

#include <limits>

using namespace wafer;
using namespace wafer::tensor_program_to_tile_region;

wafer::detail::CheckedStaticTileProductStatus
wafer::detail::checkedStaticTileProduct(llvm::ArrayRef<int64_t> ranges,
                                        llvm::ArrayRef<int64_t> tileSizes,
                                        uint64_t &product) {
  product = 1;
  if (ranges.size() != tileSizes.size())
    return CheckedStaticTileProductStatus::InvalidInput;

  for (auto [range, tileSize] : llvm::zip(ranges, tileSizes)) {
    if (range <= 0 || tileSize <= 0 || tileSize > range)
      return CheckedStaticTileProductStatus::InvalidInput;

    uint64_t unsignedRange = static_cast<uint64_t>(range);
    uint64_t unsignedTileSize = static_cast<uint64_t>(tileSize);
    uint64_t tileCount = unsignedRange / unsignedTileSize;
    tileCount += unsignedRange % unsignedTileSize != 0;
    if (product > std::numeric_limits<uint64_t>::max() / tileCount)
      return CheckedStaticTileProductStatus::Overflow;
    product *= tileCount;
  }
  return CheckedStaticTileProductStatus::Success;
}

namespace {
static void normalizeMapOps(mlir::func::FuncOp function) {
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
    rewriter.replaceOp(map, generic->getResults());
  }
}

struct FunctionalOutputDestination {
  mlir::Value empty;
  /// View operations in returned-to-base order.  Reversing each relation in
  /// this order maps the public output destination back to the exact DPS init
  /// type without recovering anything from a symbol or operation name.
  llvm::SmallVector<mlir::Operation *, 4> returnedViews;
};

static mlir::FailureOr<FunctionalOutputDestination>
findFunctionalOutputDestination(mlir::Value returned) {
  llvm::DenseSet<mlir::Value> visited;
  mlir::Value current = returned;
  FunctionalOutputDestination destination;
  while (visited.insert(current).second) {
    auto result = mlir::dyn_cast<mlir::OpResult>(current);
    if (!result)
      return mlir::failure();
    if (mlir::isa<mlir::tensor::ExpandShapeOp, mlir::tensor::CollapseShapeOp>(
            result.getOwner())) {
      if (!current.hasOneUse())
        return mlir::failure();
      destination.returnedViews.push_back(result.getOwner());
      current = result.getOwner()->getOperand(0);
      continue;
    }
    auto dps =
        mlir::dyn_cast<mlir::DestinationStyleOpInterface>(result.getOwner());
    if (!dps ||
        result.getResultNumber() >= static_cast<unsigned>(dps.getNumDpsInits()))
      return mlir::failure();
    mlir::Value init = dps.getDpsInitOperand(result.getResultNumber())->get();
    if (init.getDefiningOp<mlir::tensor::EmptyOp>()) {
      destination.empty = init;
      return destination;
    }
    current = init;
  }
  return mlir::failure();
}

static mlir::FailureOr<mlir::Value> materializeFunctionalOutputDestination(
    mlir::OpBuilder &builder, mlir::Value publicDestination,
    const FunctionalOutputDestination &destination) {
  mlir::Value current = publicDestination;
  for (mlir::Operation *view : destination.returnedViews) {
    if (auto expand = mlir::dyn_cast<mlir::tensor::ExpandShapeOp>(view)) {
      current = builder
                    .create<mlir::tensor::CollapseShapeOp>(
                        view->getLoc(), expand.getSrcType(), current,
                        expand.getReassociationIndices())
                    .getResult();
      continue;
    }
    if (auto collapse = mlir::dyn_cast<mlir::tensor::CollapseShapeOp>(view)) {
      current = builder
                    .create<mlir::tensor::ExpandShapeOp>(
                        view->getLoc(), collapse.getSrcType(), current,
                        collapse.getReassociationIndices())
                    .getResult();
      continue;
    }
    return mlir::failure();
  }
  if (!destination.empty || current.getType() != destination.empty.getType())
    return mlir::failure();
  return current;
}

static mlir::LogicalResult rewriteTensorProgramInPlace(
    mlir::ModuleOp module, int64_t currentLogicalRank,
    std::string *failureReason,
    std::optional<TargetImplementationKind> selectedAlternative,
    bool useDirectMappedBoundaryTransfer) {
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
  if (mlir::failed(verifyTensorProgramScope(function, failureReason)))
    return mlir::failure();

  phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "transformation-phase", "rewriteTensorProgramInPlace", "normalizeMapOps");
  normalizeMapOps(function);
  TensorProgramScope scope(function);
  llvm::SmallVector<mlir::Operation *, 16> sourceOperations;
  for (mlir::Operation &operation : scope.getBody().without_terminator())
    sourceOperations.push_back(&operation);

  mlir::func::ReturnOp oldReturn = scope.getReturn();
  mlir::IRRewriter rewriter(module.getContext());
  rewriter.setInsertionPoint(oldReturn);
  TileRegionBodyEmitter emitter(failureReason, currentLogicalRank,
                                selectedAlternative,
                                useDirectMappedBoundaryTransfer);
  phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "conversion-phase", "rewriteTensorProgramInPlace",
      "TileRegionBodyEmitter::emit");
  mlir::FailureOr<TileRegionOp> tileRegion = emitter.emit(scope, rewriter);
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

  // Tiling/fusion may place a nested user under a source operation whose
  // block position is not a reverse topological order for the flattened use
  // graph.  Delete the old program as an explicit dead closure instead of
  // relying on lexical order (and on eraseOp's assertion).  A value retained
  // by newly emitted IR is a conversion failure in this private clone.
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
        int64_t currentLogicalRank, std::string *failureReason,
        bool suppressDiagnostics, bool verifyResult,
        bool populateFallbackFailureReason,
        std::optional<TargetImplementationKind> selectedAlternative,
        bool useDirectMappedBoundaryTransfer) {
  wafer::support::ScopedCompileTimingSpan timing(
      "conversion", "convertTensorProgramToTileRegionModuleInPlace", "total");
  // Rewrite a private clone and commit only after the complete scheduling
  // scope has lowered and verified.
  mlir::OwningOpRef<mlir::ModuleOp> candidate;
  {
    wafer::support::ScopedCompileTimingSpan cloneTiming(
        "conversion-phase", "convertTensorProgramToTileRegionModuleInPlace",
        "clone");
    candidate = module.clone();
  }
  mlir::LogicalResult conversionResult = mlir::success();
  {
    wafer::support::ScopedCompileTimingSpan rewriteTiming(
        "conversion-phase", "convertTensorProgramToTileRegionModuleInPlace",
        "rewriteTensorProgramInPlace");
    if (suppressDiagnostics) {
      mlir::ScopedDiagnosticHandler handler(
          context, [](mlir::Diagnostic &) { return mlir::success(); });
      conversionResult = rewriteTensorProgramInPlace(
          *candidate, currentLogicalRank, failureReason, selectedAlternative,
          useDirectMappedBoundaryTransfer);
    } else {
      conversionResult = rewriteTensorProgramInPlace(
          *candidate, currentLogicalRank, failureReason, selectedAlternative,
          useDirectMappedBoundaryTransfer);
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
    if (mlir::failed(mlir::verify(*candidate))) {
      setFailureReason(failureReason,
                       "lowered tile-region module failed verifier");
      return mlir::failure();
    }
  }

  {
    wafer::support::ScopedCompileTimingSpan commitTiming(
        "conversion-phase", "convertTensorProgramToTileRegionModuleInPlace",
        "commit");
    module->setAttrs((*candidate)->getAttrs());
    module.getBodyRegion().takeBody(candidate->getBodyRegion());
  }

  return mlir::success();
}

mlir::LogicalResult wafer::lowerTensorProgramToTileRegionModule(
    mlir::func::FuncOp function, mlir::OwningOpRef<mlir::ModuleOp> &module,
    std::string *failureReason, int64_t currentLogicalRank,
    std::optional<TargetImplementationKind> selectedAlternative,
    bool useDirectMappedBoundaryTransfer) {
  if (failureReason)
    failureReason->clear();
  if (mlir::failed(verifyTensorProgramScope(function, failureReason)))
    return mlir::failure();
  module = detail::cloneTensorProgramToStandaloneModule(function);
  return convertTensorProgramToTileRegionModuleInPlace(
      *module, function.getContext(), currentLogicalRank, failureReason,
      /*suppressDiagnostics=*/true, /*verifyResult=*/true,
      /*populateFallbackFailureReason=*/true, selectedAlternative,
      useDirectMappedBoundaryTransfer);
}

mlir::LogicalResult
wafer::tensor_program_to_tile_region::prepareCompleteRankTensorProgram(
    mlir::ModuleOp sourceModule, PreparedCompleteRankTensorProgram &candidate,
    std::string *failureReason) {
  if (failureReason)
    failureReason->clear();
  if (!sourceModule) {
    if (failureReason)
      *failureReason = "complete-rank preparation requires a module";
    return mlir::failure();
  }
  candidate = {};
  candidate.module = mlir::cast<mlir::ModuleOp>(sourceModule->clone());
  mlir::func::FuncOp function =
      tensor_program_to_tile_region::findSingleStandaloneTensorProgram(
          *candidate.module);
  if (!function || function.isExternal() ||
      !llvm::hasSingleElement(function.getBody())) {
    if (failureReason)
      *failureReason =
          "complete-rank tensor program must contain one defined single-block "
          "function";
    return mlir::failure();
  }

  candidate.functionalType = function.getFunctionType();
  candidate.inputCount = function.getNumArguments();
  candidate.outputCount = function.getNumResults();
  auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(
      function.getBody().front().getTerminator());
  if (!returnOp || returnOp.getNumOperands() != candidate.outputCount ||
      candidate.outputCount == 0) {
    if (failureReason)
      *failureReason = "complete-rank functional result boundary is invalid";
    return mlir::failure();
  }

  llvm::SmallVector<FunctionalOutputDestination, 4> outputDestinations;
  llvm::DenseSet<mlir::Value> uniqueDestinations;
  outputDestinations.reserve(candidate.outputCount);
  for (auto [index, returned] : llvm::enumerate(returnOp.getOperands())) {
    auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(
        candidate.functionalType.getResult(index));
    mlir::FailureOr<FunctionalOutputDestination> destination =
        findFunctionalOutputDestination(returned);
    if (!resultType || mlir::failed(destination) || !destination->empty ||
        !uniqueDestinations.insert(destination->empty).second) {
      if (failureReason)
        *failureReason =
            "complete-rank functional result has no unique tensor.empty "
            "destination chain";
      return mlir::failure();
    }
    if (!resultType.hasStaticShape()) {
      if (failureReason)
        *failureReason =
            "complete-rank functional output restoration requires static "
            "result shape";
      return mlir::failure();
    }
    outputDestinations.push_back(*destination);
  }

  for (mlir::Type resultType : candidate.functionalType.getResults())
    function.insertArgument(function.getNumArguments(), resultType,
                            mlir::DictionaryAttr{}, function.getLoc());
  mlir::OpBuilder outputViewBuilder(&function.getBody().front(),
                                    function.getBody().front().begin());
  for (auto [index, destination] : llvm::enumerate(outputDestinations)) {
    mlir::FailureOr<mlir::Value> restoredDestination =
        materializeFunctionalOutputDestination(
            outputViewBuilder,
            function.getArgument(candidate.inputCount + index), destination);
    if (mlir::failed(restoredDestination)) {
      if (failureReason)
        *failureReason = "complete-rank functional output view relation cannot "
                         "be inverted for its DPS destination";
      return mlir::failure();
    }
    destination.empty.replaceAllUsesWith(*restoredDestination);
    if (mlir::Operation *producer = destination.empty.getDefiningOp();
        producer && producer->use_empty())
      producer->erase();
  }
  return mlir::success();
}

static mlir::LogicalResult lowerPreparedCompleteRankTensorProgramImpl(
    wafer::tensor_program_to_tile_region::PreparedCompleteRankTensorProgram
        &&candidate,
    mlir::OwningOpRef<mlir::ModuleOp> &module, std::string *failureReason,
    int64_t currentLogicalRank, llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    std::optional<wafer::CandidateTileTraversalKind> traversalKind,
    std::optional<wafer::CompleteRankTraversalComposition> composition,
    llvm::ArrayRef<wafer::CandidateTraversalConnectionChoice> connectionChoices,
    std::optional<wafer::TargetImplementationKind> selectedAlternative,
    bool useDirectMappedBoundaryTransfer,
    bool materializeOnlyRemainingConservativeRoots,
    llvm::ArrayRef<mlir::Operation *> coveredTopLevelOperations) {
  if (failureReason)
    failureReason->clear();
  if (!candidate.module || !candidate.functionalType ||
      candidate.outputCount == 0 || currentLogicalRank < 0) {
    if (failureReason)
      *failureReason =
          "prepared complete-rank lowering requires a valid artifact and "
          "non-negative logical rank";
    return mlir::failure();
  }
  mlir::FunctionType functionalType = candidate.functionalType;
  const unsigned inputCount = candidate.inputCount;
  const unsigned outputCount = candidate.outputCount;
  module = std::move(candidate.module);
  mlir::func::FuncOp function =
      tensor_program_to_tile_region::findSingleStandaloneTensorProgram(*module);
  if (!function || function.isExternal() ||
      !llvm::hasSingleElement(function.getBody()) ||
      function.getNumArguments() != inputCount + outputCount ||
      function.getNumResults() != outputCount) {
    if (failureReason)
      *failureReason = "prepared complete-rank structured boundary is invalid";
    return mlir::failure();
  }

  mlir::LogicalResult traversalResult = mlir::failure();
  auto scope = tensor_program_to_tile_region::TensorProgramScope(function);
  for (mlir::Operation *operation : coveredTopLevelOperations) {
    if (!operation || operation->getBlock() != &scope.getBody()) {
      if (failureReason)
        *failureReason =
            "implementation coverage left the prepared top-level SSA scope";
      return mlir::failure();
    }
  }
  if (!connectionChoices.empty()) {
    mlir::FailureOr<llvm::SmallVector<
        tensor_program_to_tile_region::CandidateTraversalConnection, 16>>
        connections =
            tensor_program_to_tile_region::collectCandidateTraversalConnections(
                function, failureReason);
    if (mlir::failed(connections) ||
        connections->size() != connectionChoices.size()) {
      if (mlir::succeeded(connections) && failureReason)
        *failureReason =
            "connection action vector does not cover current SSA edges";
      return mlir::failure();
    }
    traversalResult =
        tensor_program_to_tile_region::materializeJointCompleteRankTraversals(
            scope, candidateTileSizes, *connections, connectionChoices,
            failureReason);
  } else if (!composition) {
    traversalResult =
        materializeOnlyRemainingConservativeRoots
            ? tensor_program_to_tile_region::
                  materializeRemainingConservativeCompleteRankTraversals(
                      scope, coveredTopLevelOperations, failureReason)
            : tensor_program_to_tile_region::
                  materializeConservativeCompleteRankTraversals(scope,
                                                                failureReason);
  } else if (*composition == wafer::CompleteRankTraversalComposition::Coupled) {
    if (!traversalKind) {
      if (failureReason)
        *failureReason =
            "coupled complete-rank candidate requires a traversal kind";
      return mlir::failure();
    }
    traversalResult =
        tensor_program_to_tile_region::materializeCompleteCandidateTraversal(
            scope, candidateTileSizes, candidateReductionTileSizes,
            *traversalKind, failureReason);
  } else {
    if (!candidateReductionTileSizes.empty()) {
      if (failureReason)
        *failureReason = "separated complete-rank traversal does not invent "
                         "a reduction partition";
      return mlir::failure();
    }
    traversalResult = tensor_program_to_tile_region::
        materializeSeparatedCompleteRankTraversals(scope, candidateTileSizes,
                                                   failureReason);
  }
  if (mlir::failed(traversalResult))
    return mlir::failure();
  function =
      tensor_program_to_tile_region::findSingleStandaloneTensorProgram(*module);
  if (!function)
    return mlir::failure();
  if (mlir::failed(
          tensor_program_to_tile_region::
              convertTensorProgramToTileRegionModuleInPlace(
                  *module, module->getContext(), currentLogicalRank,
                  failureReason,
                  /*suppressDiagnostics=*/true,
                  /*verifyResult=*/true,
                  /*populateFallbackFailureReason=*/true, selectedAlternative,
                  useDirectMappedBoundaryTransfer)))
    return mlir::failure();

  function =
      tensor_program_to_tile_region::findSingleStandaloneTensorProgram(*module);
  if (!function || !function.getBody().hasOneBlock()) {
    if (failureReason)
      *failureReason = "lowered complete-rank function boundary is invalid";
    return mlir::failure();
  }
  mlir::OwningOpRef<mlir::func::FuncOp> loweredSnapshot =
      mlir::cast<mlir::func::FuncOp>(function->clone());
  mlir::Block &loweredEntry = loweredSnapshot->getBody().front();
  auto loweredReturn =
      mlir::dyn_cast<mlir::func::ReturnOp>(loweredEntry.getTerminator());
  if (!loweredReturn ||
      loweredEntry.getNumArguments() != inputCount + outputCount) {
    if (failureReason)
      *failureReason = "lowered complete-rank DPS boundary is invalid";
    return mlir::failure();
  }

  function.getBody().dropAllReferences();
  function.getBody().getBlocks().clear();
  function.setType(functionalType);
  mlir::Block *restoredEntry = function.addEntryBlock();
  mlir::OpBuilder builder(restoredEntry, restoredEntry->end());
  mlir::IRMapping mapping;
  for (unsigned index = 0; index < inputCount; ++index)
    mapping.map(loweredEntry.getArgument(index),
                restoredEntry->getArgument(index));
  for (unsigned index = 0; index < outputCount; ++index) {
    auto resultType =
        mlir::cast<mlir::RankedTensorType>(functionalType.getResult(index));
    mlir::Value destination = builder
                                  .create<mlir::tensor::EmptyOp>(
                                      function.getLoc(), resultType.getShape(),
                                      resultType.getElementType())
                                  .getResult();
    mapping.map(loweredEntry.getArgument(inputCount + index), destination);
  }
  for (mlir::Operation &operation : loweredEntry.without_terminator())
    builder.clone(operation, mapping);
  llvm::SmallVector<mlir::Value, 4> restoredResults;
  for (mlir::Value returned : loweredReturn.getOperands())
    restoredResults.push_back(mapping.lookupOrDefault(returned));
  builder.create<mlir::func::ReturnOp>(function.getLoc(), restoredResults);

  if (mlir::failed(mlir::verify(*module))) {
    if (failureReason)
      *failureReason = "restored complete-rank Tile module failed verifier";
    return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult wafer::tensor_program_to_tile_region::
    lowerPreparedCompleteRankTensorProgramToTileRegion(
        PreparedCompleteRankTensorProgram &&candidate,
        mlir::OwningOpRef<mlir::ModuleOp> &module, std::string *failureReason,
        int64_t currentLogicalRank,
        llvm::ArrayRef<mlir::Operation *> coveredTopLevelOperations) {
  return lowerPreparedCompleteRankTensorProgramImpl(
      std::move(candidate), module, failureReason, currentLogicalRank,
      /*candidateTileSizes=*/{}, /*candidateReductionTileSizes=*/{},
      /*traversalKind=*/std::nullopt, /*composition=*/std::nullopt,
      /*connectionChoices=*/{}, /*selectedAlternative=*/std::nullopt,
      /*useDirectMappedBoundaryTransfer=*/false,
      /*materializeOnlyRemainingConservativeRoots=*/true,
      coveredTopLevelOperations);
}

static mlir::LogicalResult lowerCompleteRankTensorProgramToTileRegionImpl(
    mlir::ModuleOp sourceModule, mlir::OwningOpRef<mlir::ModuleOp> &module,
    std::string *failureReason, int64_t currentLogicalRank,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    std::optional<wafer::CandidateTileTraversalKind> traversalKind,
    std::optional<wafer::CompleteRankTraversalComposition> composition,
    llvm::ArrayRef<wafer::CandidateTraversalConnectionChoice> connectionChoices,
    std::optional<wafer::TargetImplementationKind> selectedAlternative,
    bool useDirectMappedBoundaryTransfer) {
  wafer::tensor_program_to_tile_region::PreparedCompleteRankTensorProgram
      candidate;
  if (mlir::failed(wafer::tensor_program_to_tile_region::
                       prepareCompleteRankTensorProgram(sourceModule, candidate,
                                                        failureReason)))
    return mlir::failure();
  return lowerPreparedCompleteRankTensorProgramImpl(
      std::move(candidate), module, failureReason, currentLogicalRank,
      candidateTileSizes, candidateReductionTileSizes, traversalKind,
      composition, connectionChoices, selectedAlternative,
      useDirectMappedBoundaryTransfer,
      /*materializeOnlyRemainingConservativeRoots=*/false,
      /*coveredTopLevelOperations=*/{});
}

mlir::LogicalResult wafer::lowerCompleteRankTensorProgramToTileRegionModule(
    mlir::ModuleOp sourceModule, mlir::OwningOpRef<mlir::ModuleOp> &module,
    std::string *failureReason, int64_t currentLogicalRank) {
  return lowerCompleteRankTensorProgramToTileRegionImpl(
      sourceModule, module, failureReason, currentLogicalRank,
      /*candidateTileSizes=*/{}, /*candidateReductionTileSizes=*/{},
      /*traversalKind=*/std::nullopt, /*composition=*/std::nullopt,
      /*connectionChoices=*/{},
      /*selectedAlternative=*/std::nullopt,
      /*useDirectMappedBoundaryTransfer=*/false);
}

mlir::LogicalResult
wafer::lowerCompleteRankCandidateTensorProgramToTileRegionModule(
    mlir::ModuleOp sourceModule, llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    CandidateTileTraversalKind traversalKind,
    CompleteRankTraversalComposition composition,
    mlir::OwningOpRef<mlir::ModuleOp> &module, std::string *failureReason,
    int64_t currentLogicalRank,
    std::optional<TargetImplementationKind> selectedAlternative,
    bool useDirectMappedBoundaryTransfer) {
  if (candidateTileSizes.empty()) {
    if (failureReason)
      *failureReason = "complete-rank candidate tile sizes must be non-empty";
    return mlir::failure();
  }
  return lowerCompleteRankTensorProgramToTileRegionImpl(
      sourceModule, module, failureReason, currentLogicalRank,
      candidateTileSizes, candidateReductionTileSizes, traversalKind,
      composition, /*connectionChoices=*/{}, selectedAlternative,
      useDirectMappedBoundaryTransfer);
}

mlir::LogicalResult
wafer::lowerCompleteRankConnectionTensorProgramToTileRegionModule(
    mlir::ModuleOp sourceModule, llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<CandidateTraversalConnectionAction> connectionActions,
    mlir::OwningOpRef<mlir::ModuleOp> &module, std::string *failureReason,
    int64_t currentLogicalRank,
    std::optional<TargetImplementationKind> selectedAlternative,
    bool useDirectMappedBoundaryTransfer) {
  if (connectionActions.empty()) {
    if (failureReason)
      *failureReason =
          "connection traversal requires finite current-SSA actions";
    return mlir::failure();
  }
  llvm::SmallVector<CandidateTraversalConnectionChoice, 16> choices;
  choices.reserve(connectionActions.size());
  for (CandidateTraversalConnectionAction action : connectionActions) {
    CandidateTraversalConnectionChoice choice;
    choice.action = action;
    if (action != CandidateTraversalConnectionAction::CoupledResident)
      choice.producerTileSizes.assign(candidateTileSizes.begin(),
                                      candidateTileSizes.end());
    choice.consumerTileSizes.assign(candidateTileSizes.begin(),
                                    candidateTileSizes.end());
    choices.push_back(std::move(choice));
  }
  return lowerCompleteRankTensorProgramToTileRegionImpl(
      sourceModule, module, failureReason, currentLogicalRank,
      candidateTileSizes, /*candidateReductionTileSizes=*/{},
      CandidateTileTraversalKind::ResultDriven,
      /*composition=*/std::nullopt, choices, selectedAlternative,
      useDirectMappedBoundaryTransfer);
}

mlir::LogicalResult
wafer::lowerCompleteRankConnectionChoicesTensorProgramToTileRegionModule(
    mlir::ModuleOp sourceModule,
    llvm::ArrayRef<CandidateTraversalConnectionChoice> connectionChoices,
    mlir::OwningOpRef<mlir::ModuleOp> &module, std::string *failureReason,
    int64_t currentLogicalRank,
    std::optional<TargetImplementationKind> selectedAlternative,
    bool useDirectMappedBoundaryTransfer) {
  if (connectionChoices.empty()) {
    if (failureReason)
      *failureReason =
          "connection traversal requires finite current-SSA choices";
    return mlir::failure();
  }
  return lowerCompleteRankTensorProgramToTileRegionImpl(
      sourceModule, module, failureReason, currentLogicalRank,
      /*candidateTileSizes=*/{}, /*candidateReductionTileSizes=*/{},
      CandidateTileTraversalKind::ResultDriven,
      /*composition=*/std::nullopt, connectionChoices, selectedAlternative,
      useDirectMappedBoundaryTransfer);
}

mlir::FailureOr<unsigned>
wafer::getCompleteRankCandidateConnectionCount(mlir::ModuleOp sourceModule,
                                               std::string *failureReason) {
  auto topology =
      getCompleteRankCandidateConnectionTopology(sourceModule, failureReason);
  if (mlir::failed(topology))
    return mlir::failure();
  return topology->connectionCount;
}

mlir::FailureOr<wafer::CandidateTraversalConnectionTopology>
wafer::getCompleteRankCandidateConnectionTopology(mlir::ModuleOp sourceModule,
                                                  std::string *failureReason) {
  if (failureReason)
    failureReason->clear();
  mlir::func::FuncOp function =
      sourceModule
          ? tensor_program_to_tile_region::findSingleStandaloneTensorProgram(
                sourceModule)
          : mlir::func::FuncOp{};
  if (!function || function.isExternal() || !function.getBody().hasOneBlock()) {
    if (failureReason)
      *failureReason =
          "connection query requires one standalone tensor program";
    return mlir::failure();
  }
  auto connections =
      tensor_program_to_tile_region::collectCandidateTraversalConnections(
          function, failureReason);
  if (mlir::failed(connections))
    return mlir::failure();
  if (connections->size() > std::numeric_limits<unsigned>::max()) {
    if (failureReason)
      *failureReason = "connection count is not representable";
    return mlir::failure();
  }
  CandidateTraversalConnectionTopology topology;
  topology.connectionCount = static_cast<unsigned>(connections->size());
  auto getElementBytes =
      [](mlir::RankedTensorType type) -> std::optional<uint64_t> {
    mlir::Type elementType = type.getElementType();
    if (!elementType.isIntOrFloat())
      return std::nullopt;
    const uint64_t bitWidth = elementType.getIntOrFloatBitWidth();
    if (bitWidth == 0)
      return std::nullopt;
    // This query feeds a byte-addressed structural estimate. Conservatively
    // round sub-byte and other non-whole-byte scalar widths up to their storage
    // byte count; exact post-materialization accounting still comes from IR.
    return bitWidth / 8 + (bitWidth % 8 != 0);
  };
  llvm::DenseMap<mlir::Operation *, unsigned> operationOrdinals;
  for (auto [ordinal, operation] :
       llvm::enumerate(function.getBody().front().without_terminator()))
    operationOrdinals[&operation] = static_cast<unsigned>(ordinal);
  for (auto [index, connection] : llvm::enumerate(*connections)) {
    mlir::Operation *producer = connection.getProducer();
    mlir::Operation *consumer = connection.getConsumer();
    auto producerType = connection.producerResult
                            ? mlir::dyn_cast<mlir::RankedTensorType>(
                                  connection.producerResult.getType())
                            : mlir::RankedTensorType{};
    auto consumerOperandType =
        connection.consumerOperand
            ? mlir::dyn_cast<mlir::RankedTensorType>(
                  connection.consumerOperand->get().getType())
            : mlir::RankedTensorType{};
    auto consumerType = consumer && consumer->getNumResults() == 1
                            ? mlir::dyn_cast<mlir::RankedTensorType>(
                                  consumer->getResult(0).getType())
                            : mlir::RankedTensorType{};
    auto producerOrdinal = operationOrdinals.find(producer);
    auto consumerOrdinal = operationOrdinals.find(consumer);
    std::optional<uint64_t> producerElementBytes =
        producerType ? getElementBytes(producerType) : std::nullopt;
    std::optional<uint64_t> consumerOperandElementBytes =
        consumerOperandType ? getElementBytes(consumerOperandType)
                            : std::nullopt;
    std::optional<uint64_t> consumerResultElementBytes =
        consumerType ? getElementBytes(consumerType) : std::nullopt;
    if (!connection.isValid() || !producerType || !consumerOperandType ||
        !consumerType || !producerType.hasStaticShape() ||
        !consumerOperandType.hasStaticShape() ||
        !consumerType.hasStaticShape() || !producerElementBytes ||
        !consumerOperandElementBytes || !consumerResultElementBytes ||
        producerOrdinal == operationOrdinals.end() ||
        consumerOrdinal == operationOrdinals.end()) {
      if (failureReason)
        *failureReason =
            "connection query requires exact static byte-addressable "
            "connection domains";
      return mlir::failure();
    }
    CandidateTraversalConnectionDomain domain;
    domain.producerOperationOrdinal = producerOrdinal->second;
    domain.producerResultNumber = connection.producerResult.getResultNumber();
    domain.consumerOperationOrdinal = consumerOrdinal->second;
    domain.consumerOperandNumber =
        connection.consumerOperand->getOperandNumber();
    domain.producerResultElementBytes = *producerElementBytes;
    domain.consumerOperandElementBytes = *consumerOperandElementBytes;
    domain.consumerResultElementBytes = *consumerResultElementBytes;
    domain.producerResultShape.assign(producerType.getShape().begin(),
                                      producerType.getShape().end());
    domain.consumerOperandShape.assign(consumerOperandType.getShape().begin(),
                                       consumerOperandType.getShape().end());
    domain.consumerResultShape.assign(consumerType.getShape().begin(),
                                      consumerType.getShape().end());
    topology.domains.push_back(std::move(domain));
    topology.requiresGeneralDAGBeam |= llvm::any_of(
        llvm::drop_begin(*connections, index + 1),
        [&](const tensor_program_to_tile_region::CandidateTraversalConnection
                &other) {
          return connection.producerResult == other.producerResult &&
                 connection.consumerOperand != other.consumerOperand;
        });
  }
  return topology;
}
