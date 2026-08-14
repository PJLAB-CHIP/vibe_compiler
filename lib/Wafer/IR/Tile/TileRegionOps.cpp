//===- TileRegionOps.cpp - Wafer TileRegion verifier implementation
//----------===//

#include "Wafer/IR/WaferDialect.h"

#include "OpVerifierUtils.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
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

/// Proves storage provenance for a value crossing a tile-region boundary.
/// Unknown shaped producers are traversed only to detect an erased SPM
/// dependency; they are never accepted as an alias producer for an SPM result.
static StorageTrace traceSPMStorage(mlir::Value value, TileRegionOp owner,
                                    llvm::DenseSet<mlir::Value> &active,
                                    llvm::DenseMap<mlir::Value, StorageTrace>
                                        &memo) {
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

mlir::OperandRange TileRegionOp::getEntrySuccessorOperands(
    mlir::RegionBranchPoint point) {
  assert(point == getBody() &&
         "wafer.tile.region only enters its body region");
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
    return emitOpError("must be an outer, non-nested SPM residency region");

  llvm::DenseMap<mlir::Value, StorageTrace> inputTraceMemo;
  for (auto [index, input] : llvm::enumerate(getInputs())) {
    if (!isShapedDataType(input.getType()))
      continue;
    if (!isDDRDataType(input.getType()))
      return emitOpError("shaped data input at index ")
             << index << " must be a Wafer DDR memref, got " << input.getType();
    llvm::DenseSet<mlir::Value> active;
    StorageTrace trace =
        traceSPMStorage(input, *this, active, inputTraceMemo);
    if (!trace.valid || trace.hasSPMRoot)
      return emitOpError("shaped data input at index ")
             << index
             << " carries SPM storage provenance across the region "
                "boundary";
  }

  for (auto [index, result] : llvm::enumerate(getResults())) {
    if (isShapedDataType(result.getType()) && !isDDRDataType(result.getType()))
      return emitOpError("shaped data result at index ")
             << index << " must be a Wafer DDR memref, got "
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

  llvm::DenseMap<mlir::Value, StorageTrace> resultTraceMemo;
  for (auto [index, yieldedAndResult] :
       llvm::enumerate(llvm::zip(yield.getValues(), getResults()))) {
    mlir::Value yielded = std::get<0>(yieldedAndResult);
    mlir::Type yieldedType = yielded.getType();
    mlir::Type resultType = std::get<1>(yieldedAndResult).getType();
    if (yieldedType != resultType)
      return emitOpError("tile.yield type ")
             << yieldedType << " does not match wafer.tile.region result type "
             << resultType << " at index " << index;

    llvm::DenseSet<mlir::Value> active;
    StorageTrace trace =
        traceSPMStorage(yielded, *this, active, resultTraceMemo);
    if (!trace.valid)
      return emitOpError("result at index ")
             << index
             << " has unsupported SPM storage provenance; SPM results must "
                "alias a matching region input or a region-owned memref.alloc";
    if (trace.hasSPMRoot)
      return emitOpError("result at index ")
             << index
             << "cannot carry SPM storage provenance across the "
                "wafer.tile.region boundary";
  }

  for (mlir::NamedAttribute attr : getOperation()->getAttrs())
    return emitOpError("does not accept semantic attributes");

  return mlir::success();
}
