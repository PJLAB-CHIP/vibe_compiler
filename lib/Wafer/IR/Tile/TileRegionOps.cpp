//===- TileRegionOps.cpp - Wafer TileRegion verifier implementation
//----------===//

#include "Wafer/IR/WaferDialect.h"

#include "WaferIRVerification.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace wafer;
using namespace wafer::detail;

namespace {

struct StorageTrace {
  bool valid = true;
  bool hasSPMRoot = false;
};

static mlir::scf::YieldOp getSingleBlockYield(mlir::Region &region) {
  if (region.empty() || !region.hasOneBlock())
    return {};
  return mlir::dyn_cast<mlir::scf::YieldOp>(region.front().getTerminator());
}

static bool isAllocationOwnedBy(mlir::memref::AllocOp allocation,
                                TileRegionOp owner) {
  for (mlir::Operation *parent = allocation->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (parent == owner.getOperation())
      return true;
    if (mlir::isa<TileRegionOp>(parent))
      return false;
  }
  return false;
}

static void mergeTrace(StorageTrace &destination, StorageTrace source) {
  destination.valid &= source.valid;
  destination.hasSPMRoot |= source.hasSPMRoot;
}

static bool isShapedDataType(mlir::Type type) {
  return mlir::isa<mlir::ShapedType>(type);
}

static bool isDDRDataType(mlir::Type type) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
  return memrefType && hasWaferMemorySpace(memrefType, wafer::MemorySpace::DDR);
}

static bool isSPMDataType(mlir::Type type) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
  return memrefType && hasWaferMemorySpace(memrefType, wafer::MemorySpace::SPM);
}

static bool isResident(TileRegionOp region) {
  return static_cast<bool>(region.getResidentAttr());
}

static bool isOwnedByTileModule(mlir::Value value, TileRegionOp region) {
  if (auto toMemref = value.getDefiningOp<mlir::bufferization::ToMemrefOp>()) {
    auto allocation = toMemref.getTensor().getDefiningOp<
        mlir::bufferization::AllocTensorOp>();
    TileModuleOp owner = region->getParentOfType<TileModuleOp>();
    if (allocation && owner &&
        allocation->getParentOfType<TileModuleOp>() == owner) {
      auto memory = allocation.getMemorySpace();
      return memory && isSPMDataType(toMemref.getMemref().getType()) &&
             mlir::isa<MemoryAttr>(*memory) &&
             mlir::cast<MemoryAttr>(*memory).getSpace() == MemorySpace::SPM;
    }
  }
  while (mlir::Operation *producer = value.getDefiningOp()) {
    if (auto view = mlir::dyn_cast<mlir::ViewLikeOpInterface>(producer)) {
      value = view.getViewSource();
      continue;
    }
    break;
  }
  auto allocation = value.getDefiningOp<mlir::memref::AllocOp>();
  TileModuleOp owner = region->getParentOfType<TileModuleOp>();
  return allocation && owner &&
         allocation->getParentOfType<TileModuleOp>() == owner;
}

static bool isStructuralOrPhysicalBoundaryType(mlir::Type type) {
  return mlir::isa<mlir::RankedTensorType>(type) || isDDRDataType(type);
}

/// Traces the storage roots of a value crossing a tile-region boundary.
/// Unknown shaped producers are traversed only to detect an erased SPM
/// dependency; they are never accepted as an alias producer for an SPM result.
static StorageTrace
traceSPMStorage(mlir::Value value, TileRegionOp owner,
                llvm::DenseSet<mlir::Value> &active,
                llvm::DenseMap<mlir::Value, StorageTrace> &memo) {
  StorageTrace trace;
  if (!value)
    return trace;
  if (auto found = memo.find(value); found != memo.end())
    return found->second;
  if (!active.insert(value).second)
    return trace;

  auto finish = [&](StorageTrace result) {
    active.erase(value);
    memo[value] = result;
    return result;
  };
  auto traceValue = [&](mlir::Value source) {
    return traceSPMStorage(source, owner, active, memo);
  };

  if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    if (blockArg.getOwner() == &owner.getBody().front()) {
      unsigned index = blockArg.getArgNumber();
      trace.valid = index < owner.getInputs().size() &&
                    owner.getInputs()[index].getType() == blockArg.getType();
      trace.hasSPMRoot = trace.valid && isSPMBuffer(blockArg.getType());
      return finish(trace);
    }

    auto forOp = mlir::dyn_cast_or_null<mlir::scf::ForOp>(
        blockArg.getOwner() ? blockArg.getOwner()->getParentOp() : nullptr);
    if (forOp && blockArg.getOwner() == forOp.getBody() &&
        blockArg.getArgNumber() > 0) {
      unsigned index = blockArg.getArgNumber() - 1;
      if (index >= forOp.getInitArgs().size())
        return finish(StorageTrace{/*valid=*/false, /*hasSPMRoot=*/false});
      mergeTrace(trace, traceValue(forOp.getInitArgs()[index]));
      mlir::scf::YieldOp yield =
          mlir::dyn_cast<mlir::scf::YieldOp>(forOp.getBody()->getTerminator());
      if (!yield || index >= yield.getResults().size())
        trace.valid = false;
      else
        mergeTrace(trace, traceValue(yield.getResults()[index]));
      return finish(trace);
    }

    trace.valid = !isSPMBuffer(blockArg.getType());
    return finish(trace);
  }

  auto result = mlir::dyn_cast<mlir::OpResult>(value);
  if (!result)
    return finish(StorageTrace{/*valid=*/!isSPMBuffer(value.getType()),
                               /*hasSPMRoot=*/false});
  mlir::Operation *producer = result.getOwner();

  // A preceding TileRegion is already the verified ownership boundary for
  // its result. Do not reopen that region and recursively retrace all of its
  // inputs when verifying a later sibling. The producer's own verifier checks
  // its yield locally, while this consumer checks the typed DDR boundary.
  // Keeping these contracts compositional makes a sequential region chain
  // linear instead of repeatedly expanding the complete prefix DAG.
  if (mlir::isa<TileRegionOp>(producer)) {
    trace.valid = isDDRDataType(value.getType());
    return finish(trace);
  }

  if (auto allocation = mlir::dyn_cast<mlir::memref::AllocOp>(producer)) {
    trace.hasSPMRoot = isSPMBuffer(allocation.getType());
    trace.valid = !trace.hasSPMRoot || isAllocationOwnedBy(allocation, owner);
    return finish(trace);
  }
  if (auto toTensor = mlir::dyn_cast<mlir::bufferization::ToTensorOp>(producer))
    return finish(traceValue(toTensor.getMemref()));
  if (auto toMemref = mlir::dyn_cast<mlir::bufferization::ToMemrefOp>(producer))
    return finish(traceValue(toMemref.getTensor()));
  if (auto viewLike = mlir::dyn_cast<mlir::ViewLikeOpInterface>(producer))
    return finish(traceValue(viewLike.getViewSource()));
  if (auto select = mlir::dyn_cast<mlir::SelectLikeOpInterface>(producer)) {
    mergeTrace(trace, traceValue(select.getTrueValue()));
    mergeTrace(trace, traceValue(select.getFalseValue()));
    return finish(trace);
  }
  if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(producer)) {
    unsigned index = result.getResultNumber();
    for (mlir::scf::YieldOp yield :
         {getSingleBlockYield(ifOp.getThenRegion()),
          getSingleBlockYield(ifOp.getElseRegion())}) {
      if (!yield || index >= yield.getResults().size()) {
        trace.valid = false;
        continue;
      }
      mergeTrace(trace, traceValue(yield.getResults()[index]));
    }
    return finish(trace);
  }
  if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(producer)) {
    unsigned index = result.getResultNumber();
    if (index >= forOp.getInitArgs().size())
      trace.valid = false;
    else
      mergeTrace(trace, traceValue(forOp.getInitArgs()[index]));
    mlir::scf::YieldOp yield =
        mlir::dyn_cast<mlir::scf::YieldOp>(forOp.getBody()->getTerminator());
    if (!yield || index >= yield.getResults().size())
      trace.valid = false;
    else
      mergeTrace(trace, traceValue(yield.getResults()[index]));
    return finish(trace);
  }

  // A shaped result of an unknown producer cannot establish SPM aliasing.
  // Still inspect its shaped operands so an erased SPM dependency cannot be
  // smuggled through a tensor/generic-memref result.
  if (mlir::isa<mlir::ShapedType>(value.getType())) {
    for (mlir::Value operand : producer->getOperands()) {
      if (!mlir::isa<mlir::ShapedType>(operand.getType()))
        continue;
      StorageTrace operandTrace = traceValue(operand);
      if (operandTrace.hasSPMRoot)
        trace.valid = false;
      trace.hasSPMRoot |= operandTrace.hasSPMRoot;
    }
  }
  if (isSPMBuffer(value.getType()))
    trace.valid = false;
  return finish(trace);
}

} // namespace

mlir::OperandRange
TileRegionOp::getEntrySuccessorOperands(mlir::RegionBranchPoint point) {
  assert(point == getBody() && "wafer.tile.region only enters its body region");
  return getInputs();
}

void TileRegionOp::getSuccessorRegions(
    mlir::RegionBranchPoint point,
    llvm::SmallVectorImpl<mlir::RegionSuccessor> &regions) {
  if (point.isParent()) {
    regions.emplace_back(&getBody(), getBody().getArguments());
    return;
  }
  assert(point == getBody() &&
         "wafer.tile.region has no region successor other than its body");
  regions.emplace_back(getResults());
}

void TileRegionOp::getRegionInvocationBounds(
    llvm::ArrayRef<mlir::Attribute> operands,
    llvm::SmallVectorImpl<mlir::InvocationBounds> &invocationBounds) {
  (void)operands;
  invocationBounds.emplace_back(/*lb=*/1, /*ub=*/1);
}

mlir::LogicalResult TileRegionOp::verify() {
  if (getOperation()->getParentOfType<TileRegionOp>())
    return emitOpError("must be an outer, non-nested Tile execution region");

  for (auto [index, input] : llvm::enumerate(getInputs())) {
    if (!isShapedDataType(input.getType()))
      continue;
    if (!isStructuralOrPhysicalBoundaryType(input.getType()) &&
        !(isResident(*this) && isSPMDataType(input.getType())))
      return emitOpError("shaped data input at index ")
             << index << " must be a ranked tensor or Wafer DDR memref, got "
             << input.getType();
  }

  for (auto [index, result] : llvm::enumerate(getResults())) {
    if (isShapedDataType(result.getType()) &&
        !isStructuralOrPhysicalBoundaryType(result.getType()) &&
        !(isResident(*this) && isSPMDataType(result.getType())))
      return emitOpError("shaped data result at index ")
             << index << " must be a ranked tensor or Wafer DDR memref, got "
             << result.getType();
  }
  return mlir::success();
}

mlir::LogicalResult TileRegionOp::verifyRegions() {
  if (getBody().empty())
    return emitOpError("expected non-empty body region");

  mlir::Block &block = getBody().front();
  if (block.getNumArguments() != getInputs().size())
    return emitOpError("expected ")
           << getInputs().size()
           << " body block arguments matching wafer.tile.region inputs, got "
           << block.getNumArguments();

  for (auto [index, inputAndArg] :
       llvm::enumerate(llvm::zip(getInputs(), block.getArguments()))) {
    mlir::Type inputType = std::get<0>(inputAndArg).getType();
    mlir::Type blockArgType = std::get<1>(inputAndArg).getType();
    if (blockArgType != inputType)
      return emitOpError("body block argument type ")
             << blockArgType << " does not match input type " << inputType
             << " at index " << index;
  }

  auto yield = mlir::dyn_cast<TileYieldOp>(block.getTerminator());
  if (!yield)
    return emitOpError("expected wafer.tile.yield terminator");

  if (yield.getValues().size() != getNumResults())
    return emitOpError(
               "expected tile.yield value count to match result count, got ")
           << yield.getValues().size() << " values and " << getNumResults()
           << " results";

  for (auto [index, yieldedAndResult] :
       llvm::enumerate(llvm::zip(yield.getValues(), getResults()))) {
    mlir::Value yielded = std::get<0>(yieldedAndResult);
    mlir::Type yieldedType = yielded.getType();
    mlir::Type resultType = std::get<1>(yieldedAndResult).getType();
    if (yieldedType != resultType)
      return emitOpError("tile.yield type ")
             << yieldedType << " does not match wafer.tile.region result type "
             << resultType << " at index " << index;
  }

  for (mlir::NamedAttribute attr : getOperation()->getAttrs())
    if (!wafer::detail::isExternalDiscardableAttribute(getOperation(), attr))
      return emitOpError("does not accept semantic attribute '")
             << attr.getName().getValue() << "'";

  return mlir::success();
}

mlir::LogicalResult wafer::verifyStructuralTileRegions(mlir::ModuleOp module) {
  if (!module)
    return mlir::failure();
  mlir::WalkResult result = module.walk([&](TileRegionOp region) {
    if (!region->getParentOfType<TileModuleOp>()) {
      region.emitOpError(
          "structural TileRegion must be nested in one TileModule");
      return mlir::WalkResult::interrupt();
    }
    for (auto [index, value] : llvm::enumerate(region.getInputs())) {
      if (mlir::isa<mlir::ShapedType>(value.getType()) &&
          !mlir::isa<mlir::RankedTensorType>(value.getType())) {
        region.emitOpError("structural shaped input at index ")
            << index << " must be a ranked tensor";
        return mlir::WalkResult::interrupt();
      }
    }
    for (auto [index, value] : llvm::enumerate(region.getResults())) {
      if (mlir::isa<mlir::ShapedType>(value.getType()) &&
          !mlir::isa<mlir::RankedTensorType>(value.getType())) {
        region.emitOpError("structural shaped result at index ")
            << index << " must be a ranked tensor";
        return mlir::WalkResult::interrupt();
      }
    }
    mlir::WalkResult body = region.walk([&](mlir::Operation *operation) {
      if (operation == region.getOperation() ||
          mlir::isa<TileYieldOp>(operation))
        return mlir::WalkResult::advance();
      for (mlir::Type type : operation->getOperandTypes())
        if (mlir::isa<mlir::MemRefType>(type)) {
          operation->emitOpError(
              "is not legal in structural TileRegion: memref operand");
          return mlir::WalkResult::interrupt();
        }
      for (mlir::Type type : operation->getResultTypes())
        if (mlir::isa<mlir::MemRefType>(type)) {
          operation->emitOpError(
              "is not legal in structural TileRegion: memref result");
          return mlir::WalkResult::interrupt();
        }
      llvm::StringRef dialect = operation->getName().getDialectNamespace();
      const bool allowed =
          dialect == "builtin" || dialect == "affine" || dialect == "arith" ||
          dialect == "math" || dialect == "tensor" || dialect == "linalg" ||
          dialect == "scf" || dialect == "cf" ||
          mlir::isa<mlir::func::CallOp>(operation) ||
          mlir::isa<LinalgExtAttentionOp, LinalgExtOnlineAttentionOp,
                    LinalgExtCollectiveYieldOp>(operation) ||
          mlir::isa<WaferLinalgExtCollectiveOpInterface>(operation);
      if (!allowed) {
        operation->emitOpError("is not legal in structural TileRegion form");
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
    return body.wasInterrupted() ? mlir::WalkResult::interrupt()
                                 : mlir::WalkResult::advance();
  });
  return result.wasInterrupted() ? mlir::failure() : mlir::success();
}

mlir::LogicalResult
wafer::verifyTileRegionStorageBoundaries(mlir::ModuleOp module) {
  mlir::WalkResult result = module.walk([&](TileRegionOp region) {
    llvm::DenseMap<mlir::Value, StorageTrace> inputTraceMemo;
    for (auto [index, input] : llvm::enumerate(region.getInputs())) {
      if (!isShapedDataType(input.getType()))
        continue;
      if (!isDDRDataType(input.getType()) &&
          !(isResident(region) && isSPMDataType(input.getType()) &&
            isOwnedByTileModule(input, region))) {
        region.emitOpError("physical shaped input at index ")
            << index << " must be a Wafer DDR memref";
        return mlir::WalkResult::interrupt();
      }
      llvm::DenseSet<mlir::Value> active;
      StorageTrace trace =
          traceSPMStorage(input, region, active, inputTraceMemo);
      if (!isDDRDataType(input.getType()) && isResident(region))
        continue;
      if (!trace.valid || trace.hasSPMRoot) {
        region.emitOpError("shaped data input at index ")
            << index << " depends on SPM storage across the region boundary";
        return mlir::WalkResult::interrupt();
      }
    }

    auto yield = mlir::dyn_cast_or_null<TileYieldOp>(
        region.getBody().empty() ? nullptr
                                 : region.getBody().front().getTerminator());
    if (!yield)
      return mlir::WalkResult::advance();
    llvm::DenseMap<mlir::Value, StorageTrace> resultTraceMemo;
    for (auto [index, yielded] : llvm::enumerate(yield.getValues())) {
      if (!isShapedDataType(yielded.getType()))
        continue;
      if (!isDDRDataType(yielded.getType()) &&
          !(isResident(region) && isSPMDataType(yielded.getType()) &&
            isOwnedByTileModule(yielded, region))) {
        region.emitOpError("physical shaped result at index ")
            << index << " must be a Wafer DDR memref";
        return mlir::WalkResult::interrupt();
      }
      llvm::DenseSet<mlir::Value> active;
      StorageTrace trace =
          traceSPMStorage(yielded, region, active, resultTraceMemo);
      if (!isDDRDataType(yielded.getType()) && isResident(region))
        continue;
      if (!trace.valid || trace.hasSPMRoot) {
        region.emitOpError("result at index ")
            << index
            << " depends on unsupported SPM storage across the region "
               "boundary";
        return mlir::WalkResult::interrupt();
      }
    }
    return mlir::WalkResult::advance();
  });
  return result.wasInterrupted() ? mlir::failure() : mlir::success();
}
