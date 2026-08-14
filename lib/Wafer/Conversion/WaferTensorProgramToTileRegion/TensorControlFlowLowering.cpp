//===- TensorControlFlowLowering.cpp - Tensor/control-flow lowering -===//

#include "Internal.h"

#include "Wafer/Analysis/PhysicalDataflow/TransferRealizability.h"
#include "Wafer/Conversion/WaferTensorProgramToTileRegion/DependentDataflow.h"
#include "Wafer/Support/TargetPolicy.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include <functional>
#include <limits>

using namespace wafer;

namespace wafer::tensor_program_to_tile_region {

// One external-to-external insert copy owns only a bounded share of the SPM
// window. The remaining shares cover concurrent source/consumer working sets
// and a second independent stream. This is a conservative lowering model, not
// an execution-time or bandwidth claim; exact lifetime packing remains the
// final legality owner.
static constexpr uint64_t kExternalTensorCopyWorkingSetSlots = 4;

/// Ends one compiler-owned SPM value after its final tensor-level observation.
/// The deallocation is an explicit residency release: Tile-to-Instr completion
/// reconstruction must complete any pending NCC access before it, and exact
/// SPM planning may then reuse the released range.  Caller-owned DDR values
/// and tensor values with another semantic use are never released here.
static void releasePrivateSPMValueAfterLastUse(mlir::Value tensorValue,
                                               mlir::Value buffer,
                                               mlir::Location loc,
                                               mlir::OpBuilder &builder) {
  if (!tensorValue.hasOneUse() || !isWaferSPMMemRefType(buffer.getType()))
    return;
  builder.create<mlir::memref::DeallocOp>(loc, buffer);
}

mlir::FailureOr<mlir::Type>
TileRegionBodyEmitter::convertControlFlowType(mlir::Type type) {
  if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type))
    return makeSPMMemRefType(tensorType, MemLayout::Tensor);
  if (isScalarType(type))
    return type;
  return failType("control-flow value is not a ranked tensor or scalar");
}

mlir::LogicalResult
TileRegionBodyEmitter::recordControlFlowValue(mlir::Value original,
                                              mlir::Value converted) {
  if (mlir::isa<mlir::RankedTensorType>(original.getType())) {
    record(original, MemLayout::Tensor, converted);
    return mlir::success();
  }
  if (isScalarType(original.getType())) {
    scalarValues[original] = converted;
    return mlir::success();
  }
  return fail("control-flow value is not a ranked tensor or scalar");
}

mlir::FailureOr<mlir::Value>
TileRegionBodyEmitter::materializeControlFlowValue(mlir::Value original,
                                                   mlir::OpBuilder &builder) {
  if (mlir::isa<mlir::RankedTensorType>(original.getType()))
    return getOrMaterialize(original, MemLayout::Tensor, builder);
  if (isScalarType(original.getType()))
    return getScalarValue(original);
  return failValue("control-flow yield is not a ranked tensor or scalar");
}

mlir::LogicalResult
TileRegionBodyEmitter::convertSupportOp(mlir::Operation *op,
                                        mlir::OpBuilder &builder) {
  if (auto allocation = mlir::dyn_cast<mlir::memref::AllocOp>(op)) {
    if ((!isWaferDDRMemRefType(allocation.getType()) &&
         !isWaferSPMMemRefType(allocation.getType())) ||
        !allocation.getDynamicSizes().empty() ||
        !allocation.getSymbolOperands().empty())
      return fail("compiler-owned tensor buffer requires one static Wafer "
                  "SPM or DDR memref.alloc");
    if (compilerOwnedBuffers.contains(allocation.getResult()))
      return mlir::success();
    mlir::Operation *cloned = builder.clone(*allocation.getOperation());
    compilerOwnedBuffers[allocation.getResult()] = cloned->getResult(0);
    auto selected = llvm::find_if(
        selectedDDRStages, [&](const CandidateSelectedDDRStage &stage) {
          return stage.buffer == allocation.getResult();
        });
    // A selected compiler-owned DDR allocation is an explicit op-stage
    // destination. RegionCut creates it for a local edge; the
    // IndependentDDRStages materializer creates it for a cross-Tile fragment
    // assembly. JointDataflow PeerFragments never creates this allocation, so
    // recognizing the artifact here cannot infer or select a baseline mode.
    if (selected != selectedDDRStages.end()) {
      selectedDDRStageExternalBuffers.insert(cloned->getResult(0));
      if (emissionRelations)
        emissionRelations->selectedDDRStages.push_back(
            MaterializedSelectedDDRStage{
                mlir::cast<mlir::memref::AllocOp>(cloned), selected->strategy});
    }
    return mlir::success();
  }

  if (auto toTensor = mlir::dyn_cast<mlir::bufferization::ToTensorOp>(op)) {
    auto converted = compilerOwnedBuffers.find(toTensor.getMemref());
    if (converted == compilerOwnedBuffers.end())
      return mlir::success();
    if (isWaferSPMMemRefType(converted->second.getType())) {
      record(toTensor.getResult(), MemLayout::Tensor, converted->second);
      return mlir::success();
    }
    externalBuffers[toTensor.getResult()] = converted->second;
    if (toTensor.getWritable())
      writableExternalBuffers.insert(toTensor.getResult());
    return mlir::success();
  }

  if (auto materialize =
          mlir::dyn_cast<mlir::bufferization::MaterializeInDestinationOp>(op)) {
    auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(
        materialize.getResult().getType());
    if (!resultType || !resultType.hasStaticShape() ||
        materialize.getDest().getType() != resultType ||
        materialize.getSource().getType() != resultType)
      return fail("tensor materialize-in-destination requires equal static "
                  "ranked tensor types");
    mlir::FailureOr<mlir::Value> source =
        getOrMaterialize(materialize.getSource(), MemLayout::Tensor, builder);
    if (mlir::failed(source)) {
      if (failureReason && llvm::StringRef(*failureReason).starts_with(
                               "missing buffer for value")) {
        llvm::raw_string_ostream diagnostic(*failureReason);
        diagnostic << "; materialize-in-destination source has no lowered "
                      "buffer; source=";
        if (mlir::Operation *definition =
                materialize.getSource().getDefiningOp())
          diagnostic << definition->getName();
        else
          diagnostic << "block-argument";
      }
      return mlir::failure();
    }
    mlir::FailureOr<mlir::Value> dest =
        getOrMaterialize(materialize.getDest(), MemLayout::Tensor, builder);
    if (mlir::failed(dest)) {
      if (failureReason && llvm::StringRef(*failureReason).starts_with(
                               "missing buffer for value"))
        failureReason->append(
            "; materialize-in-destination destination has no lowered buffer");
      return mlir::failure();
    }

    llvm::SmallVector<int64_t, 4> offsets(resultType.getRank(), 0);
    llvm::SmallVector<int64_t, 4> strides(resultType.getRank(), 1);
    builder.create<MoveInsertSliceOp>(
        materialize.getLoc(), *source, *dest,
        mlir::DenseI64ArrayAttr::get(materialize.getContext(), offsets),
        mlir::DenseI64ArrayAttr::get(materialize.getContext(),
                                     resultType.getShape()),
        mlir::DenseI64ArrayAttr::get(materialize.getContext(), strides));
    record(materialize.getResult(), MemLayout::Tensor, *dest);
    return mlir::success();
  }

  if (auto constant = mlir::dyn_cast<mlir::arith::ConstantOp>(op)) {
    if (constant->getNumResults() == 0)
      return mlir::success();

    mlir::Value originalResult = constant.getResult();
    if (onlyFeedsUnreadDpsInit(originalResult))
      return mlir::success();
    auto tensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(originalResult.getType());
    if (!tensorType) {
      mlir::Operation *cloned = builder.clone(*constant.getOperation());
      scalarValues[originalResult] = cloned->getResult(0);
      scalarAttrs[originalResult] = constant.getValue();
      return mlir::success();
    }

    if (getScalarSplatAttr(tensorType, constant.getValue())) {
      // A top-level splat is a scalar value with a shaped demand, not an
      // eager full-tensor residency requirement.  Record the exact attr and
      // let getOrMaterialize create the SPM fill for the concrete full or
      // sliced SSA value that is actually consumed.  Candidate tiling
      // propagates this attr through tensor.extract_slice below; eagerly
      // filling the source shape here would retain an unused full-shape SPM
      // allocation beside every tile-local fill.
      tensorAttrs[originalResult] = constant.getValue();
      return mlir::success();
    }

    mlir::Operation *cloned = builder.clone(*constant.getOperation());
    auto ddr = builder.create<mlir::bufferization::ToMemrefOp>(
        constant.getLoc(), makeDDRMemRefType(tensorType), cloned->getResult(0),
        /*read_only=*/true);
    auto destination = builder.create<mlir::memref::AllocOp>(
        constant.getLoc(), makeSPMMemRefType(tensorType, MemLayout::Tensor));
    builder.create<StorageLoadOp>(constant.getLoc(), ddr.getMemref(),
                                  destination.getResult());
    record(originalResult, MemLayout::Tensor, destination.getResult());
    externalBuffers[originalResult] = ddr.getMemref();
    tensorAttrs[originalResult] = constant.getValue();
    return mlir::success();
  }

  if (auto apply = mlir::dyn_cast<mlir::affine::AffineApplyOp>(op)) {
    llvm::SmallVector<mlir::Value, 4> operands;
    operands.reserve(apply.getMapOperands().size());
    for (mlir::Value operand : apply.getMapOperands()) {
      mlir::FailureOr<mlir::Value> converted = getScalarValue(operand);
      if (mlir::failed(converted))
        return mlir::failure();
      operands.push_back(*converted);
    }
    auto converted = builder.create<mlir::affine::AffineApplyOp>(
        apply.getLoc(), apply.getAffineMap(), operands);
    scalarValues[apply.getResult()] = converted.getResult();
    return mlir::success();
  }

  if (auto empty = mlir::dyn_cast<mlir::tensor::EmptyOp>(op)) {
    if (onlyFeedsUnreadDpsInit(empty.getResult()))
      return mlir::success();
    bool onlyFeedsSlices = !empty.getResult().use_empty();
    for (mlir::OpOperand &use : empty.getResult().getUses()) {
      auto extractSlice =
          mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(use.getOwner());
      if (!extractSlice || extractSlice.getSource() != empty.getResult()) {
        onlyFeedsSlices = false;
        break;
      }
    }
    // tensor.empty carries no observable contents.  When traversal tiling has
    // replaced every use with a tile slice, materialize each live tile below
    // instead of allocating the original full tensor in SPM.
    if (onlyFeedsSlices)
      return mlir::success();
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(empty.getType());
    if (!tensorType)
      return fail("tensor.empty result is not a ranked tensor");
    auto alloc = builder.create<mlir::memref::AllocOp>(
        empty.getLoc(), makeSPMMemRefType(tensorType, MemLayout::Tensor));
    record(empty.getResult(), MemLayout::Tensor, alloc.getResult());
    return mlir::success();
  }

  if (auto extract = mlir::dyn_cast<mlir::tensor::ExtractOp>(op))
    return convertTensorExtract(extract, builder);
  if (auto pad = mlir::dyn_cast<mlir::tensor::PadOp>(op))
    return convertTensorPad(pad, builder);
  if (auto extractSlice = mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(op))
    return convertTensorExtractSlice(extractSlice, builder);
  if (auto insertSlice = mlir::dyn_cast<mlir::tensor::InsertSliceOp>(op))
    return convertTensorInsertSlice(insertSlice, builder);
  if (auto expandShape = mlir::dyn_cast<mlir::tensor::ExpandShapeOp>(op))
    return convertTensorReshape(expandShape.getOperation(),
                                expandShape.getSrc(), expandShape.getResult(),
                                builder);
  if (auto collapseShape = mlir::dyn_cast<mlir::tensor::CollapseShapeOp>(op))
    return convertTensorReshape(collapseShape.getOperation(),
                                collapseShape.getSrc(),
                                collapseShape.getResult(), builder);
  if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(op))
    return convertScfIf(ifOp, builder);
  if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(op))
    return convertScfFor(forOp, builder);

  // Preserve ordinary pure scalar semantics inside structured control flow.
  // Traversal tiling itself creates index arithmetic here, and source
  // programs may carry any registered scalar arith/math operation.  Clone the
  // live operation with explicitly converted scalar operands instead of
  // maintaining an opcode allowlist or recognizing a generated loop shape.
  if (op->getNumRegions() == 0 && op->getNumSuccessors() == 0 &&
      mlir::isMemoryEffectFree(op) &&
      llvm::all_of(op->getOperandTypes(),
                   [&](mlir::Type type) { return isScalarType(type); }) &&
      llvm::all_of(op->getResultTypes(),
                   [&](mlir::Type type) { return isScalarType(type); })) {
    mlir::IRMapping mapping;
    for (mlir::Value operand : op->getOperands()) {
      mlir::FailureOr<mlir::Value> converted = getScalarValue(operand);
      if (mlir::failed(converted))
        return mlir::failure();
      mapping.map(operand, *converted);
    }
    mlir::Operation *cloned = builder.clone(*op, mapping);
    for (auto [original, converted] :
         llvm::zip_equal(op->getResults(), cloned->getResults()))
      scalarValues[original] = converted;
    return mlir::success();
  }

  return mlir::success();
}

mlir::LogicalResult
TileRegionBodyEmitter::convertNestedOp(mlir::Operation *op,
                                       mlir::OpBuilder &builder) {
  if (mlir::isa<WaferLinalgExtCollectiveOpInterface>(op))
    return convertLinalgExtCollective(op, builder);
  if (mlir::isa<mlir::linalg::LinalgOp>(op))
    return convertStructuredOp(op, builder);
  if (mlir::isa<
          mlir::affine::AffineApplyOp, mlir::arith::ConstantOp,
          mlir::bufferization::MaterializeInDestinationOp,
          mlir::bufferization::ToMemrefOp, mlir::bufferization::ToTensorOp,
          mlir::tensor::EmptyOp, mlir::memref::AllocOp, mlir::tensor::ExtractOp,
          mlir::tensor::PadOp, mlir::tensor::ExtractSliceOp,
          mlir::tensor::InsertSliceOp, mlir::tensor::ExpandShapeOp,
          mlir::tensor::CollapseShapeOp, mlir::scf::IfOp, mlir::scf::ForOp>(op))
    return convertSupportOp(op, builder);
  if (op->getNumRegions() == 0 && op->getNumSuccessors() == 0 &&
      mlir::isMemoryEffectFree(op) &&
      llvm::all_of(op->getOperandTypes(),
                   [&](mlir::Type type) { return isScalarType(type); }) &&
      llvm::all_of(op->getResultTypes(),
                   [&](mlir::Type type) { return isScalarType(type); }))
    return convertSupportOp(op, builder);
  return fail("unsupported op inside structured control-flow " +
              op->getName().getStringRef().str());
}

mlir::LogicalResult
TileRegionBodyEmitter::convertScfYield(mlir::scf::YieldOp yield,
                                       mlir::OpBuilder &builder) {
  llvm::SmallVector<mlir::Value, 4> yielded;
  auto convertedFor = mlir::dyn_cast_or_null<mlir::scf::ForOp>(
      builder.getInsertionBlock() ? builder.getInsertionBlock()->getParentOp()
                                  : nullptr);
  for (auto [index, value] : llvm::enumerate(yield.getResults())) {
    if (convertedFor && index < convertedFor.getRegionIterArgs().size() &&
        isWaferDDRMemRefType(
            convertedFor.getRegionIterArgs()[index].getType())) {
      auto external = externalBuffers.find(value);
      if (external == externalBuffers.end())
        return fail("external loop yield has no DDR buffer version");
      yielded.push_back(external->second);
      continue;
    }
    mlir::FailureOr<mlir::Value> converted =
        materializeControlFlowValue(value, builder);
    if (mlir::failed(converted))
      return mlir::failure();
    yielded.push_back(*converted);
  }
  builder.create<mlir::scf::YieldOp>(yield.getLoc(), yielded);
  return mlir::success();
}

mlir::LogicalResult
TileRegionBodyEmitter::convertScfBlock(mlir::Block &source,
                                       mlir::OpBuilder &builder) {
  for (mlir::Operation &op : source.without_terminator()) {
    if (mlir::failed(convertNestedOp(&op, builder)))
      return mlir::failure();
  }
  auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(source.getTerminator());
  if (!yield)
    return fail("structured control-flow body must terminate with scf.yield");
  return convertScfYield(yield, builder);
}

void TileRegionBodyEmitter::eraseImplicitYield(mlir::Block *block) {
  if (!block || block->empty())
    return;
  if (mlir::isa<mlir::scf::YieldOp>(block->back()))
    block->back().erase();
}

mlir::LogicalResult
TileRegionBodyEmitter::convertScfIf(mlir::scf::IfOp ifOp,
                                    mlir::OpBuilder &builder) {
  mlir::FailureOr<mlir::Value> condition = getScalarValue(ifOp.getCondition());
  if (mlir::failed(condition))
    return mlir::failure();

  llvm::SmallVector<mlir::Type, 4> resultTypes;
  for (mlir::Type type : ifOp->getResultTypes()) {
    mlir::FailureOr<mlir::Type> converted = convertControlFlowType(type);
    if (mlir::failed(converted))
      return mlir::failure();
    resultTypes.push_back(*converted);
  }

  bool hasElse = !ifOp.getElseRegion().empty();
  auto convertedIf = builder.create<mlir::scf::IfOp>(ifOp.getLoc(), resultTypes,
                                                     *condition, hasElse);

  {
    StateSnapshot outer = snapshotState();
    mlir::Block *thenBlock = convertedIf.thenBlock();
    eraseImplicitYield(thenBlock);
    mlir::OpBuilder thenBuilder(thenBlock, thenBlock->end());
    if (mlir::failed(convertScfBlock(*ifOp.thenBlock(), thenBuilder)))
      return mlir::failure();
    restoreState(outer);
  }

  if (hasElse) {
    StateSnapshot outer = snapshotState();
    mlir::Block *elseBlock = convertedIf.elseBlock();
    eraseImplicitYield(elseBlock);
    mlir::OpBuilder elseBuilder(elseBlock, elseBlock->end());
    if (mlir::failed(convertScfBlock(*ifOp.elseBlock(), elseBuilder)))
      return mlir::failure();
    restoreState(outer);
  }

  for (auto [original, converted] :
       llvm::zip(ifOp->getResults(), convertedIf->getResults())) {
    if (mlir::failed(recordControlFlowValue(original, converted)))
      return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult
TileRegionBodyEmitter::convertScfFor(mlir::scf::ForOp forOp,
                                     mlir::OpBuilder &builder) {
  mlir::FailureOr<mlir::Value> lowerBound =
      getScalarValue(forOp.getLowerBound());
  mlir::FailureOr<mlir::Value> upperBound =
      getScalarValue(forOp.getUpperBound());
  mlir::FailureOr<mlir::Value> step = getScalarValue(forOp.getStep());
  if (mlir::failed(lowerBound) || mlir::failed(upperBound) ||
      mlir::failed(step))
    return mlir::failure();

  struct ExternalCarry {
    bool external = false;
    bool writable = false;
    std::optional<unsigned> outputIndex;
    mlir::Value baseBuffer;
  };
  llvm::SmallVector<mlir::Value, 4> initArgs;
  llvm::SmallVector<ExternalCarry, 4> externalCarries;
  auto sourceYield =
      mlir::cast<mlir::scf::YieldOp>(forOp.getBody()->getTerminator());
  auto tracesExternalDestination = [&](mlir::Value value,
                                       mlir::Value expectedDestination) {
    llvm::DenseSet<mlir::Value> visited;
    std::function<bool(mlir::Value, mlir::Value)> trace =
        [&](mlir::Value current, mlir::Value expected) -> bool {
      if (current == expected)
        return true;
      if (!visited.insert(current).second)
        return false;
      if (auto insert = current.getDefiningOp<mlir::tensor::InsertSliceOp>())
        return trace(insert.getDest(), expected);
      auto nestedFor = current.getDefiningOp<mlir::scf::ForOp>();
      auto result = mlir::dyn_cast<mlir::OpResult>(current);
      if (!nestedFor || !result ||
          result.getResultNumber() >= nestedFor.getNumRegionIterArgs())
        return false;
      unsigned index = result.getResultNumber();
      auto nestedYield =
          mlir::cast<mlir::scf::YieldOp>(nestedFor.getBody()->getTerminator());
      return trace(nestedFor.getInitArgs()[index], expected) &&
             trace(nestedYield.getResults()[index],
                   nestedFor.getRegionIterArgs()[index]);
    };
    return trace(value, expectedDestination);
  };

  for (auto [index, init] : llvm::enumerate(forOp.getInitArgs())) {
    ExternalCarry carry;
    auto external = externalBuffers.find(init);
    const bool tracesDestination = tracesExternalDestination(
        sourceYield.getResults()[index], forOp.getRegionIterArgs()[index]);
    if (external != externalBuffers.end() &&
        writableExternalBuffers.contains(init) && tracesDestination) {
      initArgs.push_back(external->second);
      carry.external = true;
      carry.writable = writableExternalBuffers.contains(init);
      carry.baseBuffer = external->second;
      if (auto base = directYieldBuffers.find(init);
          base != directYieldBuffers.end())
        carry.baseBuffer = base->second;
      if (auto index = externalOutputIndices.find(init);
          index != externalOutputIndices.end())
        carry.outputIndex = index->second;
      externalCarries.push_back(carry);
      continue;
    }
    mlir::FailureOr<mlir::Value> converted =
        materializeControlFlowValue(init, builder);
    if (mlir::failed(converted))
      return mlir::failure();
    initArgs.push_back(*converted);
    externalCarries.push_back(carry);
  }

  auto convertedFor = builder.create<mlir::scf::ForOp>(
      forOp.getLoc(), *lowerBound, *upperBound, *step, initArgs);
  eraseImplicitYield(convertedFor.getBody());

  StateSnapshot outer = snapshotState();
  scalarValues[forOp.getInductionVar()] = convertedFor.getInductionVar();
  for (auto [original, converted, carry] :
       llvm::zip(forOp.getRegionIterArgs(), convertedFor.getRegionIterArgs(),
                 externalCarries)) {
    if (carry.external) {
      externalBuffers[original] = converted;
      if (carry.writable)
        writableExternalBuffers.insert(original);
      if (carry.outputIndex)
        externalOutputIndices[original] = *carry.outputIndex;
      directYieldBuffers[original] = carry.baseBuffer;
    } else if (mlir::failed(recordControlFlowValue(original, converted))) {
      return mlir::failure();
    }
  }

  mlir::OpBuilder bodyBuilder(convertedFor.getBody(),
                              convertedFor.getBody()->end());
  if (mlir::failed(convertScfBlock(*forOp.getBody(), bodyBuilder)))
    return mlir::failure();
  restoreState(outer);

  for (auto [original, converted, carry] : llvm::zip(
           forOp->getResults(), convertedFor->getResults(), externalCarries)) {
    if (carry.external) {
      externalBuffers[original] = converted;
      if (carry.writable)
        writableExternalBuffers.insert(original);
      if (carry.outputIndex)
        externalOutputIndices[original] = *carry.outputIndex;
      directYieldBuffers[original] = carry.baseBuffer;
    } else if (mlir::failed(recordControlFlowValue(original, converted))) {
      return mlir::failure();
    }
  }
  return mlir::success();
}

mlir::LogicalResult
TileRegionBodyEmitter::convertTensorExtract(mlir::tensor::ExtractOp extract,
                                            mlir::OpBuilder &builder) {
  auto bufferIt = externalBuffers.find(extract.getTensor());
  if (bufferIt == externalBuffers.end())
    return fail("tensor.extract from tile-local tensor is not representable");

  llvm::SmallVector<mlir::Value, 4> indices;
  for (mlir::Value index : extract.getIndices()) {
    auto scalarIt = scalarValues.find(index);
    if (scalarIt == scalarValues.end())
      return fail("missing index value for tensor.extract");
    indices.push_back(scalarIt->second);
  }

  auto load = builder.create<mlir::memref::LoadOp>(extract.getLoc(),
                                                   bufferIt->second, indices);
  scalarValues[extract.getResult()] = load.getResult();
  return mlir::success();
}

mlir::LogicalResult
TileRegionBodyEmitter::convertTensorPad(mlir::tensor::PadOp pad,
                                        mlir::OpBuilder &builder) {
  mlir::RankedTensorType sourceType = pad.getSourceType();
  mlir::RankedTensorType resultType = pad.getResultType();
  llvm::ArrayRef<int64_t> low = pad.getStaticLow();
  llvm::ArrayRef<int64_t> high = pad.getStaticHigh();
  if (!sourceType.hasStaticShape() || !resultType.hasStaticShape() ||
      sourceType.getRank() != resultType.getRank() ||
      sourceType.getElementType() != resultType.getElementType() ||
      !allStatic(low) || !allStatic(high) ||
      low.size() != static_cast<size_t>(sourceType.getRank()) ||
      high.size() != static_cast<size_t>(sourceType.getRank()))
    return fail("tensor.pad requires matching static ranked tensors and "
                "static low/high padding");
  for (int64_t dim = 0; dim < sourceType.getRank(); ++dim) {
    if (low[dim] < 0 || high[dim] < 0 ||
        sourceType.getDimSize(dim) >
            std::numeric_limits<int64_t>::max() - low[dim] ||
        sourceType.getDimSize(dim) + low[dim] >
            std::numeric_limits<int64_t>::max() - high[dim] ||
        resultType.getDimSize(dim) !=
            sourceType.getDimSize(dim) + low[dim] + high[dim])
      return fail("tensor.pad result shape does not match explicit low/high "
                  "padding");
  }

  // tensor.pad may compute a position-dependent value in its region. That
  // semantics cannot be collapsed into one target fill. getConstantPaddingValue
  // accepts only an actual constant or a value captured from outside the pad
  // region, which is precisely the scalar-fill contract represented here.
  mlir::Value paddingValue = pad.getConstantPaddingValue();
  if (!paddingValue)
    return fail("tensor.pad requires one position-independent padding value");
  mlir::FailureOr<mlir::Value> scalar = mlir::failure();
  mlir::Attribute constantAttr;
  if (mlir::matchPattern(paddingValue, mlir::m_Constant(&constantAttr))) {
    auto typedAttr = mlir::dyn_cast<mlir::TypedAttr>(constantAttr);
    if (!typedAttr || typedAttr.getType() != sourceType.getElementType())
      return fail("tensor.pad constant value type must match the source "
                  "element type");
    scalar = builder.create<mlir::arith::ConstantOp>(pad.getLoc(), typedAttr)
                 .getResult();
  } else {
    // A position-independent value captured from outside the pad region has
    // already been converted with the surrounding scalar SSA graph.
    scalar = getScalarValue(paddingValue);
  }
  if (mlir::failed(scalar))
    return mlir::failure();
  if ((*scalar).getType() != sourceType.getElementType())
    return fail("tensor.pad value type must match the source element type");

  mlir::FailureOr<mlir::Value> source =
      getOrMaterialize(pad.getSource(), MemLayout::Tensor, builder);
  if (mlir::failed(source))
    return mlir::failure();
  if (llvm::all_of(low, [](int64_t value) { return value == 0; }) &&
      llvm::all_of(high, [](int64_t value) { return value == 0; })) {
    record(pad.getResult(), MemLayout::Tensor, *source);
    return mlir::success();
  }

  mlir::MemRefType resultBufferType =
      makeSPMMemRefType(resultType, MemLayout::Tensor);
  mlir::Value destination =
      builder.create<mlir::memref::AllocOp>(pad.getLoc(), resultBufferType);
  builder.create<ComputeFillOp>(pad.getLoc(), destination, *scalar,
                                /*fill_domain=*/FillDomainAttr{});
  llvm::SmallVector<int64_t, 4> strides(sourceType.getRank(), 1);
  builder.create<MoveInsertSliceOp>(
      pad.getLoc(), *source, destination,
      builder.getDenseI64ArrayAttr(low),
      builder.getDenseI64ArrayAttr(sourceType.getShape()),
      builder.getDenseI64ArrayAttr(strides));
  record(pad.getResult(), MemLayout::Tensor, destination);
  return mlir::success();
}

bool TileRegionBodyEmitter::allStatic(llvm::ArrayRef<int64_t> values) const {
  return llvm::all_of(values, [](int64_t value) {
    return value != mlir::ShapedType::kDynamic;
  });
}

mlir::FailureOr<mlir::Value> TileRegionBodyEmitter::materializeMemRefSubview(
    mlir::Location loc, mlir::Value sourceMemRef,
    mlir::RankedTensorType tileTensorType,
    llvm::ArrayRef<mlir::OpFoldResult> offsets, llvm::ArrayRef<int64_t> sizes,
    llvm::ArrayRef<int64_t> strides, mlir::OpBuilder &builder) {
  auto sourceType = mlir::dyn_cast<mlir::MemRefType>(sourceMemRef.getType());
  if (!sourceType)
    return failValue("tile view source is not a memref");
  if (sourceType.getElementType() != tileTensorType.getElementType())
    return failValue("tile view element type mismatch");
  if (sourceType.getRank() != static_cast<int64_t>(offsets.size()) ||
      sourceType.getRank() != static_cast<int64_t>(sizes.size()) ||
      sourceType.getRank() != static_cast<int64_t>(strides.size()))
    return failValue("tile view rank mismatch");

  llvm::SmallVector<mlir::OpFoldResult, 4> convertedOffsets;
  llvm::SmallVector<mlir::OpFoldResult, 4> mixedSizes;
  llvm::SmallVector<mlir::OpFoldResult, 4> mixedStrides;
  convertedOffsets.reserve(offsets.size());
  for (mlir::OpFoldResult offset : offsets) {
    if (auto value = mlir::dyn_cast<mlir::Value>(offset)) {
      mlir::FailureOr<mlir::Value> converted = getScalarValue(value);
      if (mlir::failed(converted))
        return mlir::failure();
      convertedOffsets.push_back(*converted);
    } else {
      convertedOffsets.push_back(offset);
    }
  }
  for (int64_t size : sizes)
    mixedSizes.push_back(builder.getIndexAttr(size));
  for (int64_t stride : strides)
    mixedStrides.push_back(builder.getIndexAttr(stride));

  auto subviewType = mlir::cast<mlir::MemRefType>(
      mlir::memref::SubViewOp::inferRankReducedResultType(
          tileTensorType.getShape(), sourceType, convertedOffsets, mixedSizes,
          mixedStrides));
  auto subview = builder.create<mlir::memref::SubViewOp>(
      loc, subviewType, sourceMemRef, convertedOffsets, mixedSizes,
      mixedStrides);
  return subview.getResult();
}

bool TileRegionBodyEmitter::hasNoObservableDestUseExceptInsert(
    mlir::tensor::InsertSliceOp insertSlice) const {
  mlir::Value dest = insertSlice.getDest();
  mlir::OpOperand *destOperand = &insertSlice->getOpOperand(1);
  auto areMutuallyExclusive = [](mlir::Operation *lhs, mlir::Operation *rhs) {
    // Find each scf.if on lhs's ancestor chain and compare the immediate
    // nested operation containing rhs. Operations in opposite regions of the
    // same if cannot observe one functional tensor version in one execution.
    // This covers both conditional forwarding and the finite static overlap
    // classes produced for a boundary wave without relying on generated op
    // names or a particular nesting depth.
    mlir::Operation *lhsChild = lhs;
    for (mlir::Operation *parent = lhs->getParentOp(); parent;
         lhsChild = parent, parent = parent->getParentOp()) {
      auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(parent);
      if (!ifOp)
        continue;
      mlir::Operation *rhsChild = rhs;
      while (rhsChild && rhsChild->getParentOp() != parent)
        rhsChild = rhsChild->getParentOp();
      if (!rhsChild)
        continue;
      mlir::Region *lhsRegion = lhsChild->getParentRegion();
      mlir::Region *rhsRegion = rhsChild->getParentRegion();
      const bool lhsThen = lhsRegion == &ifOp.getThenRegion();
      const bool lhsElse = lhsRegion == &ifOp.getElseRegion();
      const bool rhsThen = rhsRegion == &ifOp.getThenRegion();
      const bool rhsElse = rhsRegion == &ifOp.getElseRegion();
      if ((lhsThen && rhsElse) || (lhsElse && rhsThen))
        return true;
    }
    return false;
  };
  for (mlir::OpOperand &use : dest.getUses()) {
    if (&use == destOperand)
      continue;
    if (areMutuallyExclusive(insertSlice.getOperation(), use.getOwner()))
      continue;
    // emit() creates DDR boundary conversions after snapshotting the source
    // operation worklist.  Such a conversion is the physical binding for the
    // matching TileRegion block argument, not a second semantic observer of
    // the functional tensor destination.
    if (auto toMemref =
            mlir::dyn_cast<mlir::bufferization::ToMemrefOp>(use.getOwner())) {
      auto external = externalBuffers.find(dest);
      auto tileArgument =
          external == externalBuffers.end()
              ? mlir::BlockArgument{}
              : mlir::dyn_cast<mlir::BlockArgument>(external->second);
      auto tileRegion = tileArgument
                            ? mlir::dyn_cast_or_null<TileRegionOp>(
                                  tileArgument.getOwner()->getParentOp())
                            : TileRegionOp{};
      if (tileRegion &&
          tileArgument.getArgNumber() < tileRegion.getInputs().size() &&
          tileRegion.getInputs()[tileArgument.getArgNumber()] ==
              toMemref.getMemref())
        continue;
    }
    if (isUnreadDpsInitUse(use))
      continue;
    auto extractSlice =
        mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(use.getOwner());
    if (extractSlice && extractSlice.getSource() == dest &&
        onlyFeedsUnreadDpsInit(extractSlice.getResult()))
      continue;
    // A slice that is used only as the destination of another functional
    // insert does not read the old tensor version.  Once the backing tensor
    // is a compiler-owned writable DDR buffer, both inserts are explicit
    // ordered writes to subviews of that buffer; forcing the old full tensor
    // through SPM would manufacture a read and defeat temporal tiling.
    if (extractSlice && extractSlice.getSource() == dest) {
      llvm::DenseSet<mlir::Value> visited;
      if (onlyFeedsTensorInsertDestinations(extractSlice.getResult(), visited))
        continue;
    }
    return false;
  }
  return true;
}

bool TileRegionBodyEmitter::onlyFeedsTensorInsertDestinations(
    mlir::Value value, llvm::DenseSet<mlir::Value> &visited) const {
  if (value.use_empty() || !visited.insert(value).second)
    return false;
  for (mlir::OpOperand &use : value.getUses()) {
    if (auto insert =
            mlir::dyn_cast<mlir::tensor::InsertSliceOp>(use.getOwner())) {
      if (&use == &insert->getOpOperand(/*dest=*/1))
        continue;
      return false;
    }
    auto extract = mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(use.getOwner());
    if (!extract || extract.getSource() != value ||
        !onlyFeedsTensorInsertDestinations(extract.getResult(), visited))
      return false;
  }
  return true;
}

mlir::LogicalResult TileRegionBodyEmitter::convertTensorExtractSlice(
    mlir::tensor::ExtractSliceOp extractSlice, mlir::OpBuilder &builder) {
  if (!allStatic(extractSlice.getStaticSizes()) ||
      !allStatic(extractSlice.getStaticStrides()))
    return fail(
        "dynamic tensor.extract_slice size/stride is not representable");

  auto resultTensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(extractSlice.getType());
  if (!resultTensorType)
    return fail("tensor.extract_slice result is not a ranked tensor");

  if (onlyFeedsUnreadDpsInit(extractSlice.getResult()))
    return mlir::success();

  // A slice of tensor.empty is itself an undefined tile.  Allocate only that
  // tile in SPM; compact traversal offsets may be dynamic even though the tile
  // shape is static.
  if (extractSlice.getSource().getDefiningOp<mlir::tensor::EmptyOp>()) {
    auto alloc = builder.create<mlir::memref::AllocOp>(
        extractSlice.getLoc(),
        makeSPMMemRefType(resultTensorType, MemLayout::Tensor));
    record(extractSlice.getResult(), MemLayout::Tensor, alloc.getResult());
    return mlir::success();
  }

  bool hasFillInitMarker = fillInitAttrs.contains(extractSlice.getSource()) ||
                           fillInitScalars.contains(extractSlice.getSource());
  if (hasFillInitMarker &&
      onlyFeedsScalarInitializedComputeInit(extractSlice.getResult())) {
    mlir::Attribute fillInitAttr =
        fillInitAttrs.lookup(extractSlice.getSource());
    if (fillInitAttr)
      fillInitAttrs[extractSlice.getResult()] = fillInitAttr;
    mlir::Value fillInitScalar =
        fillInitScalars.lookup(extractSlice.getSource());
    if (fillInitScalar)
      fillInitScalars[extractSlice.getResult()] = fillInitScalar;
    return mlir::success();
  }

  mlir::Attribute tensorAttr = tensorAttrs.lookup(extractSlice.getSource());
  if (tensorAttr && getScalarSplatAttr(resultTensorType, tensorAttr)) {
    // Copy the mapped value before inserting. DenseMap insertion may rehash;
    // retaining an iterator/reference into tensorAttrs across operator[] is a
    // use-after-free when a deeply tiled traversal first grows this map.
    tensorAttrs[extractSlice.getResult()] = tensorAttr;
    return mlir::success();
  }

  if (auto externalIt = externalBuffers.find(extractSlice.getSource());
      externalIt != externalBuffers.end()) {
    mlir::FailureOr<mlir::Value> tileView = materializeMemRefSubview(
        extractSlice.getLoc(), externalIt->second, resultTensorType,
        extractSlice.getMixedOffsets(), extractSlice.getStaticSizes(),
        extractSlice.getStaticStrides(), builder);
    if (mlir::failed(tileView))
      return mlir::failure();

    // tensor.extract_slice is a functional read-only view. Preserve an
    // external source as a DDR subview and let the first concrete consumer
    // tile request materialize it. This avoids promoting an entire boundary
    // slice to SPM merely because a later tiled op captures the tensor value.
    externalBuffers[extractSlice.getResult()] = *tileView;
    return mlir::success();
  }

  mlir::FailureOr<mlir::Value> source =
      getOrMaterialize(extractSlice.getSource(), MemLayout::Tensor, builder);
  if (mlir::failed(source))
    return mlir::failure();

  if (!allStatic(extractSlice.getStaticOffsets())) {
    mlir::FailureOr<mlir::Value> tileView = materializeMemRefSubview(
        extractSlice.getLoc(), *source, resultTensorType,
        extractSlice.getMixedOffsets(), extractSlice.getStaticSizes(),
        extractSlice.getStaticStrides(), builder);
    if (mlir::failed(tileView))
      return mlir::failure();
    auto move = builder.create<MoveCopyOp>(
        extractSlice.getLoc(),
        makeSPMMemRefType(resultTensorType, MemLayout::Tensor), *tileView);
    record(extractSlice.getResult(), MemLayout::Tensor, move.getResult());
    return mlir::success();
  }

  mlir::MLIRContext *context = extractSlice.getContext();
  auto offsets =
      mlir::DenseI64ArrayAttr::get(context, extractSlice.getStaticOffsets());
  auto sizes =
      mlir::DenseI64ArrayAttr::get(context, extractSlice.getStaticSizes());
  auto strides =
      mlir::DenseI64ArrayAttr::get(context, extractSlice.getStaticStrides());
  auto move = builder.create<MoveExtractSliceOp>(
      extractSlice.getLoc(),
      makeSPMMemRefType(resultTensorType, MemLayout::Tensor), *source, offsets,
      sizes, strides);
  record(extractSlice.getResult(), MemLayout::Tensor, move.getResult());
  mlir::Attribute fillInitAttr = fillInitAttrs.lookup(extractSlice.getSource());
  if (fillInitAttr)
    fillInitAttrs[extractSlice.getResult()] = fillInitAttr;
  mlir::Value fillInitScalar = fillInitScalars.lookup(extractSlice.getSource());
  if (fillInitScalar)
    fillInitScalars[extractSlice.getResult()] = fillInitScalar;
  return mlir::success();
}

mlir::LogicalResult TileRegionBodyEmitter::convertTensorInsertSlice(
    mlir::tensor::InsertSliceOp insertSlice, mlir::OpBuilder &builder) {
  if (!allStatic(insertSlice.getStaticSizes()) ||
      !allStatic(insertSlice.getStaticStrides()))
    return fail("dynamic tensor.insert_slice size/stride is not representable");

  auto resultTensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(insertSlice.getType());
  if (!resultTensorType)
    return fail("tensor.insert_slice result is not a ranked tensor");
  auto sourceTensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(insertSlice.getSourceType());
  if (!sourceTensorType)
    return fail("tensor.insert_slice source is not a ranked tensor");

  auto externalIt = externalBuffers.find(insertSlice.getDest());
  std::optional<unsigned> outputIndex;
  if (auto outputIndexIt = externalOutputIndices.find(insertSlice.getDest());
      outputIndexIt != externalOutputIndices.end())
    outputIndex = outputIndexIt->second;
  bool isLoopYield = insertSlice.getResult().hasOneUse() &&
                     mlir::isa<mlir::scf::YieldOp>(
                         insertSlice.getResult().use_begin()->getOwner());
  bool selectedDDRStageDestination = false;
  if (externalIt != externalBuffers.end()) {
    mlir::Value root = externalIt->second;
    llvm::DenseSet<mlir::Value> visited;
    while (root && visited.insert(root).second) {
      if (auto subview = root.getDefiningOp<mlir::memref::SubViewOp>())
        root = subview.getSource();
      else if (auto cast = root.getDefiningOp<mlir::memref::CastOp>())
        root = cast.getSource();
      else if (auto expand = root.getDefiningOp<mlir::memref::ExpandShapeOp>())
        root = expand.getSrc();
      else if (auto collapse =
                   root.getDefiningOp<mlir::memref::CollapseShapeOp>())
        root = collapse.getSrc();
      else
        break;
    }
    selectedDDRStageDestination =
        selectedDDRStageExternalBuffers.contains(root);
  }
  const bool isCompilerOwnedRegionCutDestination = selectedDDRStageDestination;
  // A writable external destination is an explicit DDR tensor version. An
  // insert may update that version in place exactly when the old functional
  // destination has no other semantic observer. The new version may then be
  // consumed by arbitrary structured operations, returned, or both; requiring
  // a return-only use chain would incorrectly force shared cache values back
  // through one full-shape SPM buffer.
  if (externalIt != externalBuffers.end() &&
      (isCompilerOwnedRegionCutDestination ||
       (writableExternalBuffers.contains(insertSlice.getDest()) &&
        hasNoObservableDestUseExceptInsert(insertSlice)))) {
    mlir::Value externalBuffer = externalIt->second;
    auto recordExternalResult = [&] {
      mlir::Value result = insertSlice.getResult();
      externalBuffers[result] = externalBuffer;
      writableExternalBuffers.insert(result);
      if (outputIndex) {
        externalOutputIndices[result] = *outputIndex;
        mlir::Value baseBuffer =
            directYieldBuffers.lookup(insertSlice.getDest());
        directYieldBuffers[result] = baseBuffer ? baseBuffer : externalBuffer;
      }
    };

    auto sourceExternalIt = externalBuffers.find(insertSlice.getSource());
    auto stripMemRefViews = [](mlir::Value value) {
      llvm::DenseSet<mlir::Value> visited;
      while (value && visited.insert(value).second) {
        if (auto subview = value.getDefiningOp<mlir::memref::SubViewOp>()) {
          value = subview.getSource();
          continue;
        }
        if (auto cast = value.getDefiningOp<mlir::memref::CastOp>()) {
          value = cast.getSource();
          continue;
        }
        if (auto expand = value.getDefiningOp<mlir::memref::ExpandShapeOp>()) {
          value = expand.getSrc();
          continue;
        }
        if (auto collapse =
                value.getDefiningOp<mlir::memref::CollapseShapeOp>()) {
          value = collapse.getSrc();
          continue;
        }
        break;
      }
      return value;
    };
    bool canStreamExternalCopy =
        sourceExternalIt != externalBuffers.end() &&
        sourceTensorType.hasStaticShape() &&
        sourceTensorType.getRank() == resultTensorType.getRank() &&
        insertSlice.getMixedOffsets().size() ==
            static_cast<size_t>(sourceTensorType.getRank()) &&
        stripMemRefViews(sourceExternalIt->second) !=
            stripMemRefViews(externalBuffer);
    if (canStreamExternalCopy) {
      mlir::Type elementType = sourceTensorType.getElementType();
      if (!mlir::isa<mlir::IntegerType, mlir::FloatType>(elementType))
        return fail("external tensor copy requires integer or float elements");
      unsigned elementBitWidth = elementType.getIntOrFloatBitWidth();
      if (elementBitWidth == 0 || elementBitWidth % 8 != 0)
        return fail(
            "external tensor copy element width is not byte-addressable");
      const uint64_t elementBytes = elementBitWidth / 8;
      const WaferTargetPolicy targetPolicy = getDefaultWaferTargetPolicy();
      const uint64_t spmWindowBytes = static_cast<uint64_t>(
          targetPolicy.memory.spmLimit - targetPolicy.memory.spmBase);
      const uint64_t spmTileBudgetBytes =
          spmWindowBytes / kExternalTensorCopyWorkingSetSlots;
      if (spmTileBudgetBytes == 0)
        return fail("target SPM is too small for external tensor copy");
      int64_t preferredExtent =
          targetPolicy.tileSearch.preferredTileSizes.empty()
              ? 1
              : targetPolicy.tileSearch.preferredTileSizes.front();
      llvm::SmallVector<int64_t, 4> tileShape;
      tileShape.reserve(sourceTensorType.getRank());
      for (int64_t extent : sourceTensorType.getShape())
        tileShape.push_back(std::min(extent, preferredExtent));

      auto getTileBytes = [&]() -> std::optional<uint64_t> {
        uint64_t elements = 1;
        for (int64_t extent : tileShape) {
          if (extent <= 0 ||
              static_cast<uint64_t>(extent) >
                  std::numeric_limits<uint64_t>::max() / elements)
            return std::nullopt;
          elements *= static_cast<uint64_t>(extent);
        }
        if (elementBytes > std::numeric_limits<uint64_t>::max() / elements)
          return std::nullopt;
        return elements * elementBytes;
      };
      while (true) {
        std::optional<uint64_t> bytes = getTileBytes();
        if (bytes && *bytes <= spmTileBudgetBytes)
          break;
        auto largest = llvm::max_element(tileShape);
        if (largest == tileShape.end() || *largest <= 1)
          return fail("external tensor copy tile does not fit target SPM");
        *largest = (*largest + 1) / 2;
      }

      llvm::SmallVector<mlir::OpFoldResult, 4> baseOffsets;
      baseOffsets.reserve(insertSlice.getMixedOffsets().size());
      for (mlir::OpFoldResult offset : insertSlice.getMixedOffsets()) {
        if (auto value = mlir::dyn_cast<mlir::Value>(offset)) {
          mlir::FailureOr<mlir::Value> converted = getScalarValue(value);
          if (mlir::failed(converted))
            return mlir::failure();
          baseOffsets.push_back(*converted);
        } else {
          baseOffsets.push_back(offset);
        }
      }

      auto createIndexValue = [&](mlir::OpBuilder &nestedBuilder,
                                  mlir::OpFoldResult value) {
        if (auto dynamic = mlir::dyn_cast<mlir::Value>(value))
          return dynamic;
        return nestedBuilder
            .create<mlir::arith::ConstantIndexOp>(
                insertSlice.getLoc(), mlir::cast<mlir::IntegerAttr>(
                                          mlir::cast<mlir::Attribute>(value))
                                          .getInt())
            .getResult();
      };
      auto createSubview =
          [&](mlir::OpBuilder &nestedBuilder, mlir::Value sourceMemRef,
              llvm::ArrayRef<mlir::OpFoldResult> offsets,
              llvm::ArrayRef<int64_t> sizes,
              llvm::ArrayRef<int64_t> strides) -> mlir::FailureOr<mlir::Value> {
        auto sourceType =
            mlir::dyn_cast<mlir::MemRefType>(sourceMemRef.getType());
        if (!sourceType ||
            sourceType.getRank() != static_cast<int64_t>(offsets.size()))
          return mlir::failure();
        llvm::SmallVector<mlir::OpFoldResult, 4> mixedSizes;
        llvm::SmallVector<mlir::OpFoldResult, 4> mixedStrides;
        for (int64_t size : sizes)
          mixedSizes.push_back(nestedBuilder.getIndexAttr(size));
        for (int64_t stride : strides)
          mixedStrides.push_back(nestedBuilder.getIndexAttr(stride));
        auto subviewType = mlir::cast<mlir::MemRefType>(
            mlir::memref::SubViewOp::inferRankReducedResultType(
                sizes, sourceType, offsets, mixedSizes, mixedStrides));
        return nestedBuilder
            .create<mlir::memref::SubViewOp>(insertSlice.getLoc(), subviewType,
                                             sourceMemRef, offsets, mixedSizes,
                                             mixedStrides)
            .getResult();
      };

      llvm::SmallVector<mlir::OpFoldResult, 4> localOffsets;
      llvm::SmallVector<int64_t, 4> localSizes;
      std::function<mlir::LogicalResult(unsigned, mlir::OpBuilder &)> emitCopy =
          [&](unsigned dim,
              mlir::OpBuilder &nestedBuilder) -> mlir::LogicalResult {
        if (dim == static_cast<unsigned>(sourceTensorType.getRank())) {
          llvm::SmallVector<mlir::OpFoldResult, 4> destinationOffsets;
          destinationOffsets.reserve(localOffsets.size());
          for (auto [base, local, stride] : llvm::zip_equal(
                   baseOffsets, localOffsets, insertSlice.getStaticStrides())) {
            std::optional<int64_t> baseConstant =
                mlir::getConstantIntValue(base);
            std::optional<int64_t> localConstant =
                mlir::getConstantIntValue(local);
            if (baseConstant && localConstant) {
              destinationOffsets.push_back(nestedBuilder.getIndexAttr(
                  *baseConstant + *localConstant * stride));
              continue;
            }
            mlir::Value localValue = createIndexValue(nestedBuilder, local);
            if (stride != 1) {
              auto strideValue =
                  nestedBuilder.create<mlir::arith::ConstantIndexOp>(
                      insertSlice.getLoc(), stride);
              localValue = nestedBuilder.create<mlir::arith::MulIOp>(
                  insertSlice.getLoc(), localValue, strideValue);
            }
            mlir::Value baseValue = createIndexValue(nestedBuilder, base);
            destinationOffsets.push_back(
                nestedBuilder
                    .create<mlir::arith::AddIOp>(insertSlice.getLoc(),
                                                 baseValue, localValue)
                    .getResult());
          }
          llvm::SmallVector<int64_t, 4> unitStrides(localSizes.size(), 1);
          mlir::FailureOr<mlir::Value> sourceView =
              createSubview(nestedBuilder, sourceExternalIt->second,
                            localOffsets, localSizes, unitStrides);
          mlir::FailureOr<mlir::Value> destinationView =
              createSubview(nestedBuilder, externalBuffer, destinationOffsets,
                            localSizes, insertSlice.getStaticStrides());
          if (mlir::failed(sourceView) || mlir::failed(destinationView))
            return mlir::failure();
          auto tileType = mlir::RankedTensorType::get(
              localSizes, sourceTensorType.getElementType());
          auto tile = nestedBuilder.create<mlir::memref::AllocOp>(
              insertSlice.getLoc(),
              makeSPMMemRefType(tileType, MemLayout::Tensor));
          nestedBuilder.create<StorageLoadOp>(insertSlice.getLoc(), *sourceView,
                                              tile.getResult());
          nestedBuilder.create<StorageStoreOp>(
              insertSlice.getLoc(), tile.getResult(), *destinationView);
          nestedBuilder.create<mlir::memref::DeallocOp>(insertSlice.getLoc(),
                                                        tile.getResult());
          return mlir::success();
        }

        int64_t extent = sourceTensorType.getDimSize(dim);
        int64_t tileExtent = tileShape[dim];
        int64_t tailExtent = extent % tileExtent;
        int64_t mainEnd = extent - tailExtent;
        if (mainEnd == tileExtent) {
          localOffsets.push_back(nestedBuilder.getIndexAttr(0));
          localSizes.push_back(tileExtent);
          if (mlir::failed(emitCopy(dim + 1, nestedBuilder)))
            return mlir::failure();
          localSizes.pop_back();
          localOffsets.pop_back();
        } else if (mainEnd > 0) {
          auto lower = nestedBuilder.create<mlir::arith::ConstantIndexOp>(
              insertSlice.getLoc(), 0);
          auto upper = nestedBuilder.create<mlir::arith::ConstantIndexOp>(
              insertSlice.getLoc(), mainEnd);
          auto step = nestedBuilder.create<mlir::arith::ConstantIndexOp>(
              insertSlice.getLoc(), tileExtent);
          auto loop = nestedBuilder.create<mlir::scf::ForOp>(
              insertSlice.getLoc(), lower, upper, step);
          mlir::OpBuilder bodyBuilder =
              mlir::OpBuilder::atBlockBegin(loop.getBody());
          localOffsets.push_back(loop.getInductionVar());
          localSizes.push_back(tileExtent);
          if (mlir::failed(emitCopy(dim + 1, bodyBuilder)))
            return mlir::failure();
          localSizes.pop_back();
          localOffsets.pop_back();
          nestedBuilder.setInsertionPointAfter(loop);
        }
        if (tailExtent > 0) {
          localOffsets.push_back(nestedBuilder.getIndexAttr(mainEnd));
          localSizes.push_back(tailExtent);
          if (mlir::failed(emitCopy(dim + 1, nestedBuilder)))
            return mlir::failure();
          localSizes.pop_back();
          localOffsets.pop_back();
        }
        return mlir::success();
      };
      if (mlir::failed(emitCopy(/*dim=*/0, builder)))
        return fail("external tensor insert_slice streaming failed");
      recordExternalResult();
      return mlir::success();
    }

    mlir::FailureOr<mlir::Value> source =
        getOrMaterialize(insertSlice.getSource(), MemLayout::Tensor, builder);
    if (mlir::failed(source))
      return mlir::failure();

    mlir::FailureOr<mlir::Value> tileView = materializeMemRefSubview(
        insertSlice.getLoc(), externalBuffer, sourceTensorType,
        insertSlice.getMixedOffsets(), insertSlice.getStaticSizes(),
        insertSlice.getStaticStrides(), builder);
    if (mlir::failed(tileView))
      return mlir::failure();

    builder.create<StorageStoreOp>(insertSlice.getLoc(), *source, *tileView);
    releasePrivateSPMValueAfterLastUse(insertSlice.getSource(), *source,
                                       insertSlice.getLoc(), builder);
    recordExternalResult();
    return mlir::success();
  }

  if (!allStatic(insertSlice.getStaticOffsets())) {
    if (!hasNoObservableDestUseExceptInsert(insertSlice))
      return fail("dynamic tile-local tensor.insert_slice requires an "
                  "unobserved destination version");
    mlir::FailureOr<mlir::Value> source =
        getOrMaterialize(insertSlice.getSource(), MemLayout::Tensor, builder);
    mlir::FailureOr<mlir::Value> dest =
        getOrMaterialize(insertSlice.getDest(), MemLayout::Tensor, builder);
    if (mlir::failed(source) || mlir::failed(dest))
      return mlir::failure();
    mlir::FailureOr<mlir::Value> destView = materializeMemRefSubview(
        insertSlice.getLoc(), *dest, sourceTensorType,
        insertSlice.getMixedOffsets(), insertSlice.getStaticSizes(),
        insertSlice.getStaticStrides(), builder);
    if (mlir::failed(destView))
      return mlir::failure();
    builder.create<MoveCopyIntoOp>(insertSlice.getLoc(), *source, *destView);
    releasePrivateSPMValueAfterLastUse(insertSlice.getSource(), *source,
                                       insertSlice.getLoc(), builder);
    record(insertSlice.getResult(), MemLayout::Tensor, *dest);
    return mlir::success();
  }

  mlir::FailureOr<mlir::Value> source =
      getOrMaterialize(insertSlice.getSource(), MemLayout::Tensor, builder);
  mlir::FailureOr<mlir::Value> dest =
      getOrMaterialize(insertSlice.getDest(), MemLayout::Tensor, builder);
  if (mlir::failed(source) || mlir::failed(dest))
    return mlir::failure();
  if (!hasNoObservableDestUseExceptInsert(insertSlice))
    *dest = builder
                .create<MoveCopyOp>(
                    insertSlice.getLoc(),
                    makeSPMMemRefType(resultTensorType, MemLayout::Tensor),
                    *dest)
                .getResult();

  mlir::MLIRContext *context = insertSlice.getContext();
  auto offsets =
      mlir::DenseI64ArrayAttr::get(context, insertSlice.getStaticOffsets());
  auto sizes =
      mlir::DenseI64ArrayAttr::get(context, insertSlice.getStaticSizes());
  auto strides =
      mlir::DenseI64ArrayAttr::get(context, insertSlice.getStaticStrides());
  builder.create<MoveInsertSliceOp>(
      insertSlice.getLoc(), *source, *dest,
      offsets, sizes, strides);
  releasePrivateSPMValueAfterLastUse(insertSlice.getSource(), *source,
                                     insertSlice.getLoc(), builder);
  record(insertSlice.getResult(), MemLayout::Tensor, *dest);
  return mlir::success();
}

mlir::LogicalResult TileRegionBodyEmitter::convertTensorReshape(
    mlir::Operation *op, mlir::Value sourceValue, mlir::Value resultValue,
    mlir::OpBuilder &builder) {
  auto resultTensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(resultValue.getType());
  if (!resultTensorType)
    return fail("tensor reshape result is not a ranked tensor");

  const bool hasFillInit = fillInitAttrs.contains(sourceValue) ||
                           fillInitScalars.contains(sourceValue);
  if (hasFillInit && onlyFeedsScalarInitializedComputeInit(resultValue)) {
    mlir::Attribute fillInitAttr = fillInitAttrs.lookup(sourceValue);
    if (fillInitAttr)
      fillInitAttrs[resultValue] = fillInitAttr;
    mlir::Value fillInitScalar = fillInitScalars.lookup(sourceValue);
    if (fillInitScalar)
      fillInitScalars[resultValue] = fillInitScalar;
    return mlir::success();
  }

  if (auto external = externalBuffers.find(sourceValue);
      external != externalBuffers.end()) {
    mlir::Value view;
    auto sourceMemRefType =
        mlir::dyn_cast<mlir::MemRefType>(external->second.getType());
    if (!sourceMemRefType)
      return fail("external tensor reshape source is not a memref");
    if (auto collapse = mlir::dyn_cast<mlir::tensor::CollapseShapeOp>(op)) {
      mlir::MemRefType resultType =
          mlir::memref::CollapseShapeOp::computeCollapsedType(
              sourceMemRefType, collapse.getReassociationIndices());
      if (resultType.getShape() != resultTensorType.getShape())
        return fail("external tensor collapse inferred an incompatible shape");
      view = builder
                 .create<mlir::memref::CollapseShapeOp>(
                     op->getLoc(), resultType, external->second,
                     collapse.getReassociationIndices())
                 .getResult();
    } else if (auto expand = mlir::dyn_cast<mlir::tensor::ExpandShapeOp>(op)) {
      mlir::FailureOr<mlir::MemRefType> resultType =
          mlir::memref::ExpandShapeOp::computeExpandedType(
              sourceMemRefType, resultTensorType.getShape(),
              expand.getReassociationIndices());
      if (mlir::failed(resultType))
        return fail("external tensor expand has no exact memref layout");
      view = builder
                 .create<mlir::memref::ExpandShapeOp>(
                     op->getLoc(), *resultType, external->second,
                     expand.getReassociationIndices())
                 .getResult();
    } else {
      return fail("unsupported external tensor reshape operation");
    }
    externalBuffers[resultValue] = view;
    if (writableExternalBuffers.contains(sourceValue))
      writableExternalBuffers.insert(resultValue);
    std::optional<unsigned> outputIndex;
    if (auto index = externalOutputIndices.find(sourceValue);
        index != externalOutputIndices.end())
      outputIndex = index->second;
    if (outputIndex)
      externalOutputIndices[resultValue] = *outputIndex;
    mlir::Value baseBuffer = directYieldBuffers.lookup(sourceValue);
    if (baseBuffer)
      directYieldBuffers[resultValue] = baseBuffer;
    return mlir::success();
  }

  MemLayout sourceLayout = MemLayout::Tensor;
  mlir::Value source = lookupAny(sourceValue, sourceLayout);
  if (!source) {
    mlir::FailureOr<mlir::Value> loaded =
        getOrMaterialize(sourceValue, MemLayout::Tensor, builder);
    if (mlir::failed(loaded))
      return mlir::failure();
    source = *loaded;
    sourceLayout = MemLayout::Tensor;
  }

  mlir::MemRefType resultType =
      makeSPMMemRefType(resultTensorType, sourceLayout);
  auto sourceType = mlir::dyn_cast<mlir::MemRefType>(source.getType());
  if (!sourceType)
    return fail("tensor reshape source is not a memref");
  mlir::Value result;
  if (mlir::succeeded(
          analysis::TransferRealizability::proveStaticReshapeMetadataView(
              sourceType, resultType, /*destinationMayWrite=*/true)))
    result = builder
                 .create<ViewReshapeOp>(op->getLoc(), resultType, source)
                 .getResult();
  else
    result = builder
                 .create<MoveReshapeOp>(op->getLoc(), resultType, source)
                 .getResult();
  record(resultValue, sourceLayout, result);
  return mlir::success();
}

mlir::FailureOr<mlir::Value>
TileRegionBodyEmitter::getScalarValue(mlir::Value original) {
  auto it = scalarValues.find(original);
  if (it == scalarValues.end()) {
    return failValue("missing scalar value for tile compute");
  }
  return it->second;
}

bool TileRegionBodyEmitter::isUnreadDpsInitUse(mlir::OpOperand &use) const {
  if (auto allReduce =
          mlir::dyn_cast<LinalgExtCollectiveAllReduceOp>(use.getOwner())) {
    auto dps =
        mlir::cast<mlir::DestinationStyleOpInterface>(allReduce.getOperation());
    return dps.isDpsInit(&use);
  }

  auto linalgOp = mlir::dyn_cast<mlir::linalg::LinalgOp>(use.getOwner());
  if (!linalgOp)
    return false;
  llvm::ArrayRef<mlir::BlockArgument> outputArgs =
      linalgOp.getRegionOutputArgs();
  for (int64_t index = 0; index < linalgOp.getNumDpsInits(); ++index) {
    if (linalgOp.getDpsInitOperand(index) != &use)
      continue;
    return static_cast<size_t>(index) < outputArgs.size() &&
           outputArgs[index].use_empty();
  }
  return false;
}

bool TileRegionBodyEmitter::onlyFeedsUnreadDpsInit(
    mlir::Value value, llvm::DenseSet<mlir::Value> &visited) const {
  if (value.use_empty() || !visited.insert(value).second)
    return false;
  for (mlir::OpOperand &use : value.getUses()) {
    if (isUnreadDpsInitUse(use))
      continue;
    auto extractSlice =
        mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(use.getOwner());
    if (!extractSlice || extractSlice.getSource() != value ||
        !onlyFeedsUnreadDpsInit(extractSlice.getResult(), visited))
      return false;
  }
  return true;
}

bool TileRegionBodyEmitter::onlyFeedsUnreadDpsInit(mlir::Value value) const {
  llvm::DenseSet<mlir::Value> visited;
  return onlyFeedsUnreadDpsInit(value, visited);
}

bool TileRegionBodyEmitter::onlyFeedsScalarInitializedComputeInit(
    mlir::Value value, llvm::DenseSet<mlir::Value> &visited) const {
  if (value.use_empty() || !visited.insert(value).second)
    return false;

  for (mlir::OpOperand &use : value.getUses()) {
    mlir::Operation *owner = use.getOwner();
    if (auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(owner);
        linalg && linalg.isDpsInit(&use) &&
        mlir::succeeded(inferOrdinaryConv2DGeometry(linalg)))
      continue;
    if (mlir::isa<mlir::linalg::MatmulOp, mlir::linalg::MatmulTransposeAOp,
                  mlir::linalg::MatmulTransposeBOp, mlir::linalg::BatchMatmulOp,
                  mlir::linalg::BatchMatmulTransposeAOp,
                  mlir::linalg::BatchMatmulTransposeBOp>(owner)) {
      auto dpsOp = mlir::cast<mlir::linalg::LinalgOp>(owner);
      if (dpsOp.isDpsInit(&use))
        continue;
      return false;
    }

    if (auto generic = mlir::dyn_cast<mlir::linalg::GenericOp>(owner)) {
      auto linalg = mlir::cast<mlir::linalg::LinalgOp>(owner);
      if (linalg.isDpsInit(&use) && hasReductionIterator(generic))
        continue;
      return false;
    }

    if (auto expand = mlir::dyn_cast<mlir::tensor::ExpandShapeOp>(owner)) {
      if (expand.getSrc() == value &&
          onlyFeedsScalarInitializedComputeInit(expand.getResult(), visited))
        continue;
      return false;
    }
    if (auto collapse = mlir::dyn_cast<mlir::tensor::CollapseShapeOp>(owner)) {
      if (collapse.getSrc() == value &&
          onlyFeedsScalarInitializedComputeInit(collapse.getResult(), visited))
        continue;
      return false;
    }

    auto extractSlice = mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(owner);
    if (!extractSlice || extractSlice.getSource() != value ||
        !onlyFeedsScalarInitializedComputeInit(extractSlice.getResult(),
                                               visited))
      return false;
  }
  return true;
}

bool TileRegionBodyEmitter::onlyFeedsScalarInitializedComputeInit(
    mlir::Value value) const {
  llvm::DenseSet<mlir::Value> visited;
  return onlyFeedsScalarInitializedComputeInit(value, visited);
}

} // namespace wafer::tensor_program_to_tile_region
