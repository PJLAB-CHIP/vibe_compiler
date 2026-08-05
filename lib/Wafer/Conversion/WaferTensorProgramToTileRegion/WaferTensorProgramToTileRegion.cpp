//===- WaferTensorProgramToTileRegion.cpp - Tensor program lowering ------===//

#include "Internal.h"

#include "Wafer/Support/CompileTiming.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
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

static mlir::FailureOr<mlir::Value>
findFunctionalOutputDestination(mlir::Value returned) {
  llvm::DenseSet<mlir::Value> visited;
  mlir::Value current = returned;
  while (visited.insert(current).second) {
    auto result = mlir::dyn_cast<mlir::OpResult>(current);
    if (!result)
      return mlir::failure();
    auto dps =
        mlir::dyn_cast<mlir::DestinationStyleOpInterface>(result.getOwner());
    if (!dps ||
        result.getResultNumber() >= static_cast<unsigned>(dps.getNumDpsInits()))
      return mlir::failure();
    mlir::Value init = dps.getDpsInitOperand(result.getResultNumber())->get();
    if (init.getDefiningOp<mlir::tensor::EmptyOp>())
      return init;
    current = init;
  }
  return mlir::failure();
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

mlir::LogicalResult wafer::lowerCompleteRankTensorProgramToTileRegionModule(
    mlir::ModuleOp sourceModule, mlir::OwningOpRef<mlir::ModuleOp> &module,
    std::string *failureReason, int64_t currentLogicalRank) {
  if (failureReason)
    failureReason->clear();
  if (!sourceModule || currentLogicalRank < 0) {
    if (failureReason)
      *failureReason =
          "complete-rank Tile lowering requires a module and non-negative "
          "logical rank";
    return mlir::failure();
  }
  module = mlir::cast<mlir::ModuleOp>(sourceModule->clone());
  mlir::func::FuncOp function =
      tensor_program_to_tile_region::findSingleStandaloneTensorProgram(*module);
  if (!function || function.isExternal() ||
      !llvm::hasSingleElement(function.getBody())) {
    if (failureReason)
      *failureReason =
          "complete-rank tensor program must contain one defined single-block "
          "function";
    return mlir::failure();
  }

  mlir::FunctionType functionalType = function.getFunctionType();
  const unsigned inputCount = function.getNumArguments();
  const unsigned outputCount = function.getNumResults();
  auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(
      function.getBody().front().getTerminator());
  if (!returnOp || returnOp.getNumOperands() != outputCount ||
      outputCount == 0) {
    if (failureReason)
      *failureReason = "complete-rank functional result boundary is invalid";
    return mlir::failure();
  }

  llvm::SmallVector<mlir::Value, 4> outputDestinations;
  llvm::DenseSet<mlir::Value> uniqueDestinations;
  outputDestinations.reserve(outputCount);
  for (auto [index, returned] : llvm::enumerate(returnOp.getOperands())) {
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(functionalType.getResult(index));
    mlir::FailureOr<mlir::Value> destination =
        findFunctionalOutputDestination(returned);
    if (!resultType || mlir::failed(destination) ||
        (*destination).getType() != resultType ||
        !uniqueDestinations.insert(*destination).second) {
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

  for (mlir::Type resultType : functionalType.getResults())
    function.insertArgument(function.getNumArguments(), resultType,
                            mlir::DictionaryAttr{}, function.getLoc());
  for (auto [index, destination] : llvm::enumerate(outputDestinations)) {
    destination.replaceAllUsesWith(function.getArgument(inputCount + index));
    if (mlir::Operation *producer = destination.getDefiningOp();
        producer && producer->use_empty())
      producer->erase();
  }

  if (mlir::failed(
          tensor_program_to_tile_region::
              materializeConservativeCompleteRankTraversals(
                  tensor_program_to_tile_region::TensorProgramScope(function),
                  failureReason)))
    return mlir::failure();

  if (mlir::failed(tensor_program_to_tile_region::
                       convertTensorProgramToTileRegionModuleInPlace(
                           *module, sourceModule.getContext(),
                           currentLogicalRank, failureReason,
                           /*suppressDiagnostics=*/true,
                           /*verifyResult=*/true,
                           /*populateFallbackFailureReason=*/true,
                           /*selectedAlternative=*/std::nullopt,
                           /*useDirectMappedBoundaryTransfer=*/false)))
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
