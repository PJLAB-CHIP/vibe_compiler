//===- TileRegionCandidate.cpp - Provisional tile-region candidate -------===//

#include "Wafer/Transforms/Group/TileRegionCandidate.h"

#include "Wafer/Transforms/Group/LayoutPlanningAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

using namespace wafer;

namespace {

struct BufferVersions {
  mlir::Value tensor;
  mlir::Value nTensor;
  mlir::Value cx;
  mlir::Value nCx;
};

class TileRegionCandidateBuilder {
public:
  mlir::LogicalResult run(GroupOp group, TileRegionCandidate &candidate) {
    candidate = {};
    candidate.group = group;

    GroupLayoutPlan layoutPlan;
    if (mlir::failed(collectGroupLayoutPlan(group, layoutPlan)))
      return mlir::failure();
    if (!layoutPlan.succeeded)
      return fail(candidate, layoutPlan.failureReason);

    mlir::Location loc = group.getLoc();
    scratchModule = mlir::ModuleOp::create(loc);
    mlir::OpBuilder moduleBuilder(scratchModule->getBodyRegion());

    llvm::SmallVector<mlir::Type, 4> inputTypes;
    for (mlir::Value input : group.getInputs())
      inputTypes.push_back(input.getType());
    for (mlir::Value output : group.getOuts())
      inputTypes.push_back(output.getType());

    auto funcType =
        moduleBuilder.getFunctionType(inputTypes, group.getResultTypes());
    auto func = moduleBuilder.create<mlir::func::FuncOp>(
        loc, "tile_region_candidate", funcType);
    mlir::Block *entry = func.addEntryBlock();

    mlir::OpBuilder builder(entry, entry->end());
    auto tileRegion = builder.create<TileRegionOp>(loc, group.getResultTypes(),
                                                   entry->getArguments());
    mlir::Block *tileBlock = new mlir::Block();
    tileRegion.getBody().push_back(tileBlock);
    for (mlir::Value input : tileRegion.getInputs())
      tileBlock->addArgument(input.getType(), loc);

    builder.setInsertionPointToStart(tileBlock);
    if (mlir::failed(initializeBoundary(group, tileRegion, builder, candidate)))
      return mlir::success();

    for (mlir::Operation &op : group.getBody().front().without_terminator()) {
      if (mlir::isa<WaferTensorCollectiveOpInterface>(&op))
        return fail(candidate,
                    "collective lowering requires placement/local-rank facts");
    }

    for (OpLayoutPlan &opPlan : layoutPlan.ops) {
      if (mlir::failed(convertOp(opPlan, builder, candidate)))
        return mlir::success();
    }

    if (mlir::failed(finishRegion(group, tileRegion, builder, candidate)))
      return mlir::success();

    builder.setInsertionPointAfter(tileRegion);
    builder.create<mlir::func::ReturnOp>(loc, tileRegion.getResults());

    if (mlir::failed(mlir::verify(*scratchModule)))
      return fail(candidate, "candidate verifier failed");

    candidate.module = std::move(scratchModule);
    return mlir::success();
  }

private:
  mlir::OwningOpRef<mlir::ModuleOp> scratchModule;
  llvm::DenseMap<mlir::Value, BufferVersions> buffers;

  mlir::LogicalResult fail(TileRegionCandidate &candidate,
                           llvm::StringRef reason) {
    candidate.succeeded = false;
    candidate.failureReason = reason.str();
    return mlir::success();
  }

  mlir::Attribute layoutAttr(mlir::MLIRContext *context, MemLayout layout) {
    return MemLayoutAttr::get(context, layout);
  }

  mlir::Attribute spmAttr(mlir::MLIRContext *context) {
    return MemorySpaceAttr::get(context, MemorySpace::SPM);
  }

  mlir::Type tileBufferType(mlir::Type tensorType, MemLayout layout) {
    auto *context = tensorType.getContext();
    return TileBufferType::get(context, tensorType, layoutAttr(context, layout),
                               spmAttr(context));
  }

  void record(mlir::Value original, MemLayout layout, mlir::Value buffer) {
    BufferVersions &versions = buffers[original];
    switch (layout) {
    case MemLayout::Tensor:
      versions.tensor = buffer;
      break;
    case MemLayout::NTensor:
      versions.nTensor = buffer;
      break;
    case MemLayout::Cx:
      versions.cx = buffer;
      break;
    case MemLayout::NCx:
      versions.nCx = buffer;
      break;
    }
  }

  mlir::Value lookup(mlir::Value original, MemLayout layout) const {
    auto it = buffers.find(original);
    if (it == buffers.end())
      return {};
    const BufferVersions &versions = it->second;
    switch (layout) {
    case MemLayout::Tensor:
      return versions.tensor;
    case MemLayout::NTensor:
      return versions.nTensor;
    case MemLayout::Cx:
      return versions.cx;
    case MemLayout::NCx:
      return versions.nCx;
    }
    llvm_unreachable("unknown memory layout");
  }

  mlir::Value lookupAny(mlir::Value original, MemLayout &layout) const {
    auto it = buffers.find(original);
    if (it == buffers.end())
      return {};
    const BufferVersions &versions = it->second;
    if (versions.tensor) {
      layout = MemLayout::Tensor;
      return versions.tensor;
    }
    if (versions.cx) {
      layout = MemLayout::Cx;
      return versions.cx;
    }
    if (versions.nTensor) {
      layout = MemLayout::NTensor;
      return versions.nTensor;
    }
    if (versions.nCx) {
      layout = MemLayout::NCx;
      return versions.nCx;
    }
    return {};
  }

  mlir::FailureOr<mlir::Value>
  getOrMaterialize(mlir::Value original, MemLayout targetLayout,
                   mlir::OpBuilder &builder, TileRegionCandidate &candidate) {
    if (mlir::Value existing = lookup(original, targetLayout))
      return existing;

    MemLayout sourceLayout = MemLayout::Tensor;
    mlir::Value source = lookupAny(original, sourceLayout);
    if (!source)
      return failAndReturn(candidate, "missing tile buffer for value");
    if (sourceLayout == targetLayout)
      return source;

    auto tensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(original.getType());
    if (!tensorType)
      return failAndReturn(candidate,
                           "cannot materialize non-ranked-tensor value");

    mlir::Type resultType = tileBufferType(tensorType, targetLayout);
    auto materialize = builder.create<LayoutMaterializeOp>(original.getLoc(),
                                                           resultType, source);
    record(original, targetLayout, materialize.getResult());
    return materialize.getResult();
  }

  mlir::FailureOr<mlir::Value> failAndReturn(TileRegionCandidate &candidate,
                                             llvm::StringRef reason) {
    (void)fail(candidate, reason);
    return mlir::failure();
  }

  mlir::LogicalResult initializeBoundary(GroupOp group, TileRegionOp tileRegion,
                                         mlir::OpBuilder &builder,
                                         TileRegionCandidate &candidate) {
    mlir::Block &groupBlock = group.getBody().front();
    mlir::Block &tileBlock = tileRegion.getBody().front();
    if (groupBlock.getNumArguments() != tileBlock.getNumArguments())
      return fail(candidate, "group boundary argument count mismatch");

    for (auto [groupArg, tileArg] :
         llvm::zip(groupBlock.getArguments(), tileBlock.getArguments())) {
      auto tensorType =
          mlir::dyn_cast<mlir::RankedTensorType>(groupArg.getType());
      if (!tensorType)
        return fail(candidate, "group boundary is not a ranked tensor");
      auto load = builder.create<LoadTileOp>(
          groupArg.getLoc(), tileBufferType(tensorType, MemLayout::Tensor),
          tileArg);
      record(groupArg, MemLayout::Tensor, load.getResult());
    }
    return mlir::success();
  }

  mlir::LogicalResult convertOp(const OpLayoutPlan &opPlan,
                                mlir::OpBuilder &builder,
                                TileRegionCandidate &candidate) {
    if (opPlan.kind == OpTilingDemandKind::Failure)
      return fail(candidate, opPlan.failureReason);

    mlir::Operation *op = opPlan.op;
    if (opPlan.kind == OpTilingDemandKind::Support)
      return convertSupportOp(op, builder, candidate);
    if (opPlan.kind == OpTilingDemandKind::TensorCollective)
      return fail(candidate,
                  "collective lowering requires placement/local-rank facts");

    if (auto fill = mlir::dyn_cast<mlir::linalg::FillOp>(op))
      return convertFill(fill, builder, candidate);
    if (mlir::isa<mlir::linalg::MatmulOp>(op))
      return convertMatmul(mlir::cast<mlir::linalg::LinalgOp>(op), builder,
                           candidate);
    if (auto generic = mlir::dyn_cast<mlir::linalg::GenericOp>(op))
      return convertGeneric(generic, builder, candidate);

    return fail(candidate, ("unsupported linalg op " +
                            op->getName().getStringRef().str()));
  }

  mlir::LogicalResult convertSupportOp(mlir::Operation *op,
                                       mlir::OpBuilder &builder,
                                       TileRegionCandidate &candidate) {
    if (auto constant = mlir::dyn_cast<mlir::arith::ConstantOp>(op)) {
      if (constant->getNumResults() == 0)
        return mlir::success();
      auto tensorType =
          mlir::dyn_cast<mlir::RankedTensorType>(constant.getType());
      if (!tensorType)
        return mlir::success();
      mlir::Operation *cloned = builder.clone(*constant.getOperation());
      auto load = builder.create<LoadTileOp>(
          constant.getLoc(), tileBufferType(tensorType, MemLayout::Tensor),
          cloned->getResult(0));
      record(constant.getResult(), MemLayout::Tensor, load.getResult());
      return mlir::success();
    }

    if (mlir::isa<mlir::tensor::EmptyOp>(op))
      return fail(candidate,
                  "tensor.empty requires explicit tile-buffer allocation");

    return mlir::success();
  }

  mlir::LogicalResult convertFill(mlir::linalg::FillOp fill,
                                  mlir::OpBuilder &builder,
                                  TileRegionCandidate &candidate) {
    mlir::linalg::LinalgOp op = fill;
    if (op.getNumDpsInits() != 1 || fill->getNumResults() != 1)
      return fail(candidate, "unsupported linalg.fill arity");

    mlir::FailureOr<mlir::Value> output = getOrMaterialize(
        op.getDpsInits()[0], MemLayout::Tensor, builder, candidate);
    if (mlir::failed(output))
      return mlir::failure();
    record(fill.getResult(0), MemLayout::Tensor, *output);
    return mlir::success();
  }

  mlir::LogicalResult convertMatmul(mlir::linalg::LinalgOp op,
                                    mlir::OpBuilder &builder,
                                    TileRegionCandidate &candidate) {
    if (op.getNumDpsInputs() != 2 || op->getNumResults() != 1)
      return fail(candidate, "unsupported matmul arity");

    mlir::FailureOr<mlir::Value> lhs = getOrMaterialize(
        op.getDpsInputs()[0], MemLayout::Cx, builder, candidate);
    mlir::FailureOr<mlir::Value> rhs = getOrMaterialize(
        op.getDpsInputs()[1], MemLayout::Cx, builder, candidate);
    if (mlir::failed(lhs) || mlir::failed(rhs))
      return mlir::failure();

    auto resultTensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultTensorType)
      return fail(candidate, "matmul result is not a ranked tensor");

    auto gemm = builder.create<ComputeGemmOp>(
        op->getLoc(), tileBufferType(resultTensorType, MemLayout::Cx), *lhs,
        *rhs);
    record(op->getResult(0), MemLayout::Cx, gemm.getResult());
    return mlir::success();
  }

  std::optional<ComputeElementwiseKind>
  inferElementwiseKind(mlir::linalg::GenericOp generic,
                       TileRegionCandidate &candidate) {
    auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(
        generic.getBody()->getTerminator());
    if (!yield || yield.getValues().size() != 1) {
      (void)fail(candidate, "unsupported linalg.generic yield");
      return std::nullopt;
    }

    mlir::Operation *def = yield.getValues()[0].getDefiningOp();
    if (!def) {
      (void)fail(candidate, "unsupported linalg.generic passthrough body");
      return std::nullopt;
    }

    llvm::StringRef name = def->getName().getStringRef();
    if (name == "arith.addf" || name == "arith.addi")
      return ComputeElementwiseKind::Add;
    if (name == "arith.subf" || name == "arith.subi")
      return ComputeElementwiseKind::Sub;
    if (name == "arith.mulf" || name == "arith.muli")
      return ComputeElementwiseKind::Mul;
    if (name == "arith.divf" || name == "arith.divsi" || name == "arith.divui")
      return ComputeElementwiseKind::Div;
    if (name == "arith.maximumf" || name == "arith.maxsi" ||
        name == "arith.maxui")
      return ComputeElementwiseKind::Max;
    if (name == "arith.minimumf" || name == "arith.minsi" ||
        name == "arith.minui")
      return ComputeElementwiseKind::Min;

    (void)fail(candidate, ("unsupported linalg.generic body op " + name.str()));
    return std::nullopt;
  }

  mlir::LogicalResult convertGeneric(mlir::linalg::GenericOp generic,
                                     mlir::OpBuilder &builder,
                                     TileRegionCandidate &candidate) {
    if (generic.getNumDpsInits() != 1 || generic->getNumResults() != 1)
      return fail(candidate, "unsupported linalg.generic arity");

    std::optional<ComputeElementwiseKind> kind =
        inferElementwiseKind(generic, candidate);
    if (!kind)
      return mlir::failure();

    llvm::SmallVector<mlir::Value, 4> inputs;
    for (mlir::Value input : generic.getDpsInputs()) {
      mlir::FailureOr<mlir::Value> buffer =
          getOrMaterialize(input, MemLayout::Tensor, builder, candidate);
      if (mlir::failed(buffer))
        return mlir::failure();
      inputs.push_back(*buffer);
    }

    auto resultTensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(generic->getResult(0).getType());
    if (!resultTensorType)
      return fail(candidate, "generic result is not a ranked tensor");

    auto kindAttr =
        ComputeElementwiseKindAttr::get(generic.getContext(), *kind);
    auto elementwise = builder.create<ComputeElementwiseOp>(
        generic.getLoc(), tileBufferType(resultTensorType, MemLayout::Tensor),
        kindAttr, inputs);
    if (mlir::Attribute indexingMaps = generic->getAttr("indexing_maps"))
      elementwise->setAttr("indexing_maps", indexingMaps);

    record(generic->getResult(0), MemLayout::Tensor, elementwise.getResult());
    return mlir::success();
  }

  mlir::LogicalResult finishRegion(GroupOp group, TileRegionOp tileRegion,
                                   mlir::OpBuilder &builder,
                                   TileRegionCandidate &candidate) {
    auto yield =
        mlir::dyn_cast<GroupYieldOp>(group.getBody().front().getTerminator());
    if (!yield)
      return fail(candidate, "group terminator is not wafer.group_yield");

    unsigned inputCount = static_cast<unsigned>(group.getInputs().size());
    llvm::SmallVector<mlir::Value, 2> yieldedTensors;
    mlir::Block &tileBlock = tileRegion.getBody().front();
    for (auto [index, value] : llvm::enumerate(yield.getValues())) {
      mlir::FailureOr<mlir::Value> tensorBuffer =
          getOrMaterialize(value, MemLayout::Tensor, builder, candidate);
      if (mlir::failed(tensorBuffer))
        return mlir::failure();

      if (inputCount + index >= tileBlock.getNumArguments())
        return fail(candidate, "group result has no output boundary");
      mlir::Value output = tileBlock.getArgument(inputCount + index);
      builder.create<StoreTileOp>(value.getLoc(), *tensorBuffer, output);
      yieldedTensors.push_back(output);
    }

    builder.create<TileYieldOp>(group.getLoc(), yieldedTensors);
    return mlir::success();
  }
};

} // namespace

mlir::LogicalResult
wafer::buildTileRegionCandidate(GroupOp group, TileRegionCandidate &candidate) {
  TileRegionCandidateBuilder builder;
  return builder.run(group, candidate);
}

void wafer::dumpTileRegionCandidate(const TileRegionCandidate &candidate,
                                    llvm::StringRef groupLabel,
                                    llvm::raw_ostream &os) {
  os << "wafer.tile_region.candidate group " << groupLabel << "\n";
  if (!candidate.succeeded) {
    os << "  failure " << candidate.failureReason << "\n";
    return;
  }
  candidate.module.get().print(os);
  os << "\n";
}
