//===- WaferTensorProgramToTileRegion.cpp - Tensor program lowering ------===//

#include "Internal.h"

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

static mlir::LogicalResult rewriteTensorProgramInPlace(
    mlir::ModuleOp module, int64_t currentLogicalRank,
    std::string *failureReason,
    std::optional<TargetImplementationKind> selectedAlternative) {
  mlir::func::FuncOp function = findSingleStandaloneTensorProgram(module);
  if (!function) {
    setFailureReason(
        failureReason,
        "standalone module must contain exactly one tensor program function");
    return mlir::failure();
  }
  if (mlir::failed(verifyTensorProgramScope(function, failureReason)))
    return mlir::failure();

  normalizeMapOps(function);
  TensorProgramScope scope(function);
  llvm::SmallVector<mlir::Operation *, 16> sourceOperations;
  for (mlir::Operation &operation : scope.getBody().without_terminator())
    sourceOperations.push_back(&operation);

  mlir::func::ReturnOp oldReturn = scope.getReturn();
  mlir::IRRewriter rewriter(module.getContext());
  rewriter.setInsertionPoint(oldReturn);
  TileRegionBodyEmitter emitter(failureReason, currentLogicalRank,
                                selectedAlternative);
  mlir::FailureOr<TileRegionOp> tileRegion = emitter.emit(scope, rewriter);
  if (mlir::failed(tileRegion))
    return mlir::failure();

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
        std::optional<TargetImplementationKind> selectedAlternative) {
  // Rewrite a private clone and commit only after the complete scheduling
  // scope has lowered and verified.
  mlir::OwningOpRef<mlir::ModuleOp> candidate = module.clone();
  mlir::LogicalResult conversionResult = mlir::success();
  if (suppressDiagnostics) {
    mlir::ScopedDiagnosticHandler handler(
        context, [](mlir::Diagnostic &) { return mlir::success(); });
    conversionResult = rewriteTensorProgramInPlace(
        *candidate, currentLogicalRank, failureReason, selectedAlternative);
  } else {
    conversionResult = rewriteTensorProgramInPlace(
        *candidate, currentLogicalRank, failureReason, selectedAlternative);
  }

  if (mlir::failed(conversionResult)) {
    if (populateFallbackFailureReason &&
        (!failureReason || failureReason->empty()))
      setFailureReason(failureReason,
                       "tensor-program-to-tile-region lowering failed");
    return mlir::failure();
  }

  if (verifyResult && mlir::failed(mlir::verify(*candidate))) {
    setFailureReason(failureReason,
                     "lowered tile-region module failed verifier");
    return mlir::failure();
  }

  module->setAttrs((*candidate)->getAttrs());
  module.getBodyRegion().takeBody(candidate->getBodyRegion());

  return mlir::success();
}

mlir::LogicalResult wafer::lowerTensorProgramToTileRegionModule(
    mlir::func::FuncOp function, mlir::OwningOpRef<mlir::ModuleOp> &module,
    std::string *failureReason, int64_t currentLogicalRank,
    std::optional<TargetImplementationKind> selectedAlternative) {
  if (failureReason)
    failureReason->clear();
  if (mlir::failed(verifyTensorProgramScope(function, failureReason)))
    return mlir::failure();
  module = detail::cloneTensorProgramToStandaloneModule(function);
  return convertTensorProgramToTileRegionModuleInPlace(
      *module, function.getContext(), currentLogicalRank, failureReason,
      /*suppressDiagnostics=*/true, /*verifyResult=*/true,
      /*populateFallbackFailureReason=*/true, selectedAlternative);
}
