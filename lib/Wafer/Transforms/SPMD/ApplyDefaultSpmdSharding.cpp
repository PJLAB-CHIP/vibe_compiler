//===- ApplyDefaultSpmdSharding.cpp - Default SDY input seeds ------------===//

#include "Wafer/Transforms/Passes.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#ifdef WAFER_ENABLE_SHARDY
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#endif

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>

namespace wafer {

#define GEN_PASS_DEF_APPLYDEFAULTSPMDSHARDINGPASS
#include "Wafer/Transforms/WaferPasses.h.inc"

namespace {

#ifdef WAFER_ENABLE_SHARDY

constexpr llvm::StringLiteral kDefaultMeshName = "wafer_default_card_mesh";
constexpr llvm::StringLiteral kMhloShardingAttr = "mhlo.sharding";
constexpr llvm::StringLiteral kStablehloShardingAttr = "stablehlo.sharding";
constexpr llvm::StringLiteral kCustomCallTargetAttr = "call_target_name";
constexpr llvm::StringLiteral kShardingCustomCallTarget = "Sharding";

struct DefaultMeshSpec {
  llvm::SmallVector<std::string, 4> axes;
  llvm::SmallVector<int64_t, 4> shape;
  int64_t partitionCount = 1;
};

static bool checkedMul(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0)
    return false;
  if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

static mlir::FailureOr<DefaultMeshSpec>
getExecutionMeshSpec(mlir::ModuleOp moduleOp, llvm::StringRef meshName) {
  ExecutionMeshOp meshOp = moduleOp.lookupSymbol<ExecutionMeshOp>(meshName);
  if (!meshOp) {
    bool multipleMeshes = false;
    for (ExecutionMeshOp candidate : moduleOp.getOps<ExecutionMeshOp>()) {
      if (!meshOp) {
        meshOp = candidate;
        continue;
      }
      multipleMeshes = true;
      break;
    }
    if (multipleMeshes) {
      moduleOp.emitOpError(
          "requires an explicit execution-mesh option when multiple direct "
          "module execution meshes exist");
      return mlir::failure();
    }
  }
  if (!meshOp)
    return mlir::failure();

  DefaultMeshSpec spec;
  for (mlir::Attribute axisAttr : meshOp.getAxesAttr())
    spec.axes.push_back(
        mlir::cast<mlir::StringAttr>(axisAttr).getValue().str());
  for (int64_t dim : meshOp.getShapeAttr().asArrayRef()) {
    int64_t partitionCount = 0;
    if (dim <= 0 || !checkedMul(spec.partitionCount, dim, partitionCount)) {
      meshOp.emitOpError("has invalid shape for default SPMD sharding");
      return mlir::failure();
    }
    spec.shape.push_back(dim);
    spec.partitionCount = partitionCount;
  }
  if (spec.axes.empty() || spec.axes.size() != spec.shape.size()) {
    meshOp.emitOpError("has inconsistent axes and shape");
    return mlir::failure();
  }
  return spec;
}

static bool isFrontendShardingAttr(mlir::NamedAttribute attr) {
  llvm::StringRef name = attr.getName().getValue();
  if (name == kMhloShardingAttr || name == kStablehloShardingAttr) {
    if (auto stringAttr = mlir::dyn_cast<mlir::StringAttr>(attr.getValue()))
      return !stringAttr.getValue().empty();
    return true;
  }
  return name == "mhlo.spmd_parameters_sharding";
}

static mlir::StringAttr getFrontendShardingStringAttr(mlir::Operation *op) {
  if (auto attr = op->getAttrOfType<mlir::StringAttr>(kMhloShardingAttr))
    return attr;
  return op->getAttrOfType<mlir::StringAttr>(kStablehloShardingAttr);
}

static bool isFrontendShardingCustomCall(mlir::Operation *op) {
  llvm::StringRef opName = op->getName().getStringRef();
  if (opName != "stablehlo.custom_call" && opName != "mhlo.custom_call")
    return false;
  auto target = op->getAttrOfType<mlir::StringAttr>(kCustomCallTargetAttr);
  return target && target.getValue() == kShardingCustomCallTarget;
}

static bool hasShardingSeed(mlir::func::FuncOp funcOp) {
  auto funcIface = mlir::cast<mlir::FunctionOpInterface>(funcOp.getOperation());

  for (unsigned i = 0, e = funcOp.getNumArguments(); i != e; ++i) {
    if (funcIface.getArgAttr(i, mlir::sdy::kShardingAttr))
      return true;
    for (mlir::NamedAttribute attr : funcIface.getArgAttrs(i))
      if (isFrontendShardingAttr(attr))
        return true;
  }

  for (unsigned i = 0, e = funcOp.getNumResults(); i != e; ++i) {
    if (funcIface.getResultAttr(i, mlir::sdy::kShardingAttr))
      return true;
    for (mlir::NamedAttribute attr : funcIface.getResultAttrs(i))
      if (isFrontendShardingAttr(attr))
        return true;
  }

  bool found = false;
  funcOp.walk([&](mlir::Operation *op) {
    if (op == funcOp.getOperation())
      return mlir::WalkResult::advance();

    if (op->getAttr(mlir::sdy::kShardingAttr)) {
      found = true;
      return mlir::WalkResult::interrupt();
    }

    for (mlir::NamedAttribute attr : op->getAttrs()) {
      if (isFrontendShardingAttr(attr)) {
        found = true;
        return mlir::WalkResult::interrupt();
      }
    }

    if (mlir::isa<mlir::sdy::ShardingConstraintOp, mlir::sdy::ReshardOp,
                  mlir::sdy::ManualComputationOp>(op)) {
      found = true;
      return mlir::WalkResult::interrupt();
    }

    return mlir::WalkResult::advance();
  });
  return found;
}

static std::optional<int64_t> findDefaultSplitDim(mlir::RankedTensorType type,
                                                  int64_t partitionCount) {
  if (partitionCount == 1)
    return std::nullopt;

  for (auto [index, dim] : llvm::enumerate(type.getShape())) {
    if (dim > 0 && !mlir::ShapedType::isDynamic(dim) &&
        dim % partitionCount == 0)
      return static_cast<int64_t>(index);
  }
  return std::nullopt;
}

static llvm::SmallVector<mlir::sdy::AxisRefAttr>
getAxisRefs(mlir::MLIRContext *context, const DefaultMeshSpec &meshSpec) {
  llvm::SmallVector<mlir::sdy::AxisRefAttr> axes;
  axes.reserve(meshSpec.axes.size());
  for (const std::string &axis : meshSpec.axes)
    axes.push_back(mlir::sdy::AxisRefAttr::get(context, axis));
  return axes;
}

static mlir::sdy::TensorShardingAttr
buildDefaultInputSharding(mlir::MLIRContext *context,
                          mlir::RankedTensorType type,
                          const DefaultMeshSpec &meshSpec) {
  llvm::SmallVector<mlir::sdy::AxisRefAttr> meshAxes =
      getAxisRefs(context, meshSpec);
  mlir::sdy::DimensionShardingAttr replicatedDim =
      mlir::sdy::DimensionShardingAttr::get(context, {}, /*is_closed=*/true);

  llvm::SmallVector<mlir::sdy::DimensionShardingAttr> dimShardings(
      type.getRank(), replicatedDim);
  llvm::SmallVector<mlir::sdy::AxisRefAttr> replicatedAxes;

  if (std::optional<int64_t> splitDim =
          findDefaultSplitDim(type, meshSpec.partitionCount)) {
    dimShardings[*splitDim] = mlir::sdy::DimensionShardingAttr::get(
        context, meshAxes, /*is_closed=*/true);
  } else {
    replicatedAxes.append(meshAxes);
  }

  return mlir::sdy::TensorShardingAttr::get(context, kDefaultMeshName,
                                            dimShardings, replicatedAxes);
}

static mlir::sdy::TensorShardingAttr
buildReplicatedSharding(mlir::MLIRContext *context, mlir::RankedTensorType type,
                        const DefaultMeshSpec &meshSpec) {
  mlir::sdy::DimensionShardingAttr replicatedDim =
      mlir::sdy::DimensionShardingAttr::get(context, {}, /*is_closed=*/true);
  llvm::SmallVector<mlir::sdy::DimensionShardingAttr> dimShardings(
      type.getRank(), replicatedDim);
  return mlir::sdy::TensorShardingAttr::get(
      context, kDefaultMeshName, dimShardings, getAxisRefs(context, meshSpec));
}

static mlir::FailureOr<llvm::SmallVector<int64_t>>
parseOldOpShardingDevicesShape(mlir::Operation *op,
                               llvm::StringRef shardingText) {
  llvm::StringRef devicesPrefix = "devices=[";
  size_t devicesBegin = shardingText.find(devicesPrefix);
  if (devicesBegin == llvm::StringRef::npos)
    return op->emitError("unsupported frontend sharding custom call attr: ")
           << shardingText;
  devicesBegin += devicesPrefix.size();

  size_t devicesEnd = shardingText.find(']', devicesBegin);
  if (devicesEnd == llvm::StringRef::npos)
    return op->emitError("malformed frontend sharding devices list: ")
           << shardingText;

  llvm::SmallVector<int64_t> devicesShape;
  llvm::SmallVector<llvm::StringRef> pieces;
  shardingText.slice(devicesBegin, devicesEnd).split(pieces, ',');
  for (llvm::StringRef piece : pieces) {
    int64_t value = 0;
    if (piece.trim().getAsInteger(10, value) || value <= 0)
      return op->emitError("malformed frontend sharding devices dimension: ")
             << shardingText;
    devicesShape.push_back(value);
  }
  return devicesShape;
}

static mlir::FailureOr<mlir::sdy::TensorShardingAttr>
buildFrontendSharding(mlir::Operation *op, mlir::StringAttr shardingAttr,
                      mlir::RankedTensorType type,
                      const DefaultMeshSpec &meshSpec) {
  mlir::MLIRContext *context = op->getContext();
  llvm::StringRef shardingText = shardingAttr.getValue().trim();
  if (shardingText == "{replicated}")
    return buildReplicatedSharding(context, type, meshSpec);

  mlir::FailureOr<llvm::SmallVector<int64_t>> devicesShape =
      parseOldOpShardingDevicesShape(op, shardingText);
  if (mlir::failed(devicesShape))
    return mlir::failure();

  bool hasLastTileDimReplicate =
      shardingText.contains("last_tile_dim_replicate");
  llvm::SmallVector<int64_t> tensorDimFactors = *devicesShape;
  if (hasLastTileDimReplicate &&
      tensorDimFactors.size() == static_cast<size_t>(type.getRank() + 1)) {
    int64_t replicatedFactor = tensorDimFactors.pop_back_val();
    if (replicatedFactor != 1)
      return op->emitError(
                 "unsupported partial-replication frontend sharding: ")
             << shardingText;
  }
  if (tensorDimFactors.size() != static_cast<size_t>(type.getRank()))
    return op->emitError("frontend sharding rank does not match tensor rank: ")
           << shardingText;

  llvm::SmallVector<mlir::sdy::AxisRefAttr> meshAxes =
      getAxisRefs(context, meshSpec);
  mlir::sdy::DimensionShardingAttr replicatedDim =
      mlir::sdy::DimensionShardingAttr::get(context, {}, /*is_closed=*/true);
  llvm::SmallVector<mlir::sdy::DimensionShardingAttr> dimShardings(
      type.getRank(), replicatedDim);

  std::optional<int64_t> shardedDim;
  for (auto [index, factor] : llvm::enumerate(tensorDimFactors)) {
    if (factor == 1)
      continue;
    if (factor != meshSpec.partitionCount)
      return op->emitError("frontend sharding factor does not match Wafer "
                           "execution mesh partition count: ")
             << shardingText;
    if (shardedDim)
      return op->emitError("unsupported multi-dimension frontend sharding: ")
             << shardingText;
    shardedDim = static_cast<int64_t>(index);
  }

  if (!shardedDim)
    return buildReplicatedSharding(context, type, meshSpec);

  dimShardings[*shardedDim] =
      mlir::sdy::DimensionShardingAttr::get(context, meshAxes,
                                            /*is_closed=*/true);
  return mlir::sdy::TensorShardingAttr::get(
      context, kDefaultMeshName, dimShardings, /*replicated_axes=*/{});
}

static bool isCompatibleDefaultMesh(mlir::sdy::MeshOp meshOp,
                                    const DefaultMeshSpec &meshSpec) {
  llvm::ArrayRef<mlir::sdy::MeshAxisAttr> axes = meshOp.getMesh().getAxes();
  if (axes.size() != meshSpec.axes.size())
    return false;
  for (auto [axis, expectedName, expectedSize] :
       llvm::zip_equal(axes, meshSpec.axes, meshSpec.shape)) {
    if (axis.getName() != expectedName || axis.getSize() != expectedSize)
      return false;
  }
  return true;
}

static mlir::sdy::MeshOp
getOrCreateDefaultMesh(mlir::ModuleOp moduleOp,
                       const DefaultMeshSpec &meshSpec) {
  mlir::SymbolTable symbolTable(moduleOp);
  if (auto existing = symbolTable.lookup<mlir::sdy::MeshOp>(kDefaultMeshName))
    return existing;

  llvm::SmallVector<mlir::sdy::MeshAxisAttr> axes;
  axes.reserve(meshSpec.axes.size());
  for (auto [axis, size] : llvm::zip_equal(meshSpec.axes, meshSpec.shape))
    axes.push_back(
        mlir::sdy::MeshAxisAttr::get(moduleOp.getContext(), axis, size));

  mlir::OpBuilder builder(moduleOp.getContext());
  builder.setInsertionPointToStart(moduleOp.getBody());
  return builder.create<mlir::sdy::MeshOp>(
      moduleOp.getLoc(), kDefaultMeshName,
      mlir::sdy::MeshAttr::get(moduleOp.getContext(), axes));
}

static bool needsDefaultSeed(mlir::func::FuncOp funcOp) {
  if (hasShardingSeed(funcOp))
    return false;

  return llvm::any_of(
      funcOp.getFunctionType().getInputs(),
      [](mlir::Type type) { return mlir::isa<mlir::RankedTensorType>(type); });
}

static void applyDefaultInputSeeds(mlir::func::FuncOp funcOp,
                                   const DefaultMeshSpec &meshSpec) {
  mlir::MLIRContext *context = funcOp.getContext();
  auto funcIface = mlir::cast<mlir::FunctionOpInterface>(funcOp.getOperation());
  for (auto [index, type] :
       llvm::enumerate(funcOp.getFunctionType().getInputs())) {
    auto rankedType = mlir::dyn_cast<mlir::RankedTensorType>(type);
    if (!rankedType)
      continue;
    funcIface.setArgAttr(
        static_cast<unsigned>(index), mlir::sdy::kShardingAttr,
        buildDefaultInputSharding(context, rankedType, meshSpec));
  }
}

static void collectFrontendShardingCustomCalls(
    mlir::ModuleOp moduleOp, llvm::SmallVectorImpl<mlir::Operation *> &ops) {
  moduleOp.walk([&](mlir::Operation *op) {
    if (isFrontendShardingCustomCall(op))
      ops.push_back(op);
  });
}

struct FrontendShardingRewrite {
  mlir::Operation *customCall = nullptr;
  mlir::sdy::TensorShardingAttr sharding;
};

static mlir::FailureOr<llvm::SmallVector<FrontendShardingRewrite>>
buildFrontendShardingRewritePlan(mlir::ModuleOp moduleOp,
                                 const DefaultMeshSpec &meshSpec) {
  llvm::SmallVector<mlir::Operation *> customCalls;
  collectFrontendShardingCustomCalls(moduleOp, customCalls);

  llvm::SmallVector<FrontendShardingRewrite> plan;
  plan.reserve(customCalls.size());
  for (mlir::Operation *op : customCalls) {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return op->emitError("expected frontend sharding custom call with one "
                           "operand and one result");

    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType)
      return op->emitError("expected ranked tensor result on frontend "
                           "sharding custom call");

    mlir::StringAttr shardingAttr = getFrontendShardingStringAttr(op);
    if (!shardingAttr)
      return op->emitError("expected frontend sharding custom call to carry ")
             << kMhloShardingAttr << " or " << kStablehloShardingAttr;

    mlir::FailureOr<mlir::sdy::TensorShardingAttr> sharding =
        buildFrontendSharding(op, shardingAttr, resultType, meshSpec);
    if (mlir::failed(sharding))
      return mlir::failure();
    plan.push_back({op, *sharding});
  }
  return plan;
}

static void
applyFrontendShardingRewritePlan(llvm::ArrayRef<FrontendShardingRewrite> plan) {
  for (const FrontendShardingRewrite &rewrite : plan) {
    mlir::OpBuilder builder(rewrite.customCall);
    auto constraint = builder.create<mlir::sdy::ShardingConstraintOp>(
        rewrite.customCall->getLoc(), rewrite.customCall->getOperand(0),
        rewrite.sharding);
    rewrite.customCall->getResult(0).replaceAllUsesWith(constraint.getResult());
    rewrite.customCall->erase();
  }
}
#endif

struct ApplyDefaultSpmdShardingPass
    : public impl::ApplyDefaultSpmdShardingPassBase<
          ApplyDefaultSpmdShardingPass> {
  using impl::ApplyDefaultSpmdShardingPassBase<
      ApplyDefaultSpmdShardingPass>::ApplyDefaultSpmdShardingPassBase;

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    impl::ApplyDefaultSpmdShardingPassBase<
        ApplyDefaultSpmdShardingPass>::getDependentDialects(registry);
#ifdef WAFER_ENABLE_SHARDY
    registry.insert<mlir::sdy::SdyDialect>();
#endif
  }

  void runOnOperation() final {
#ifndef WAFER_ENABLE_SHARDY
    getOperation().emitOpError("requires WAFER_ENABLE_SPMD_PARTITIONER_DEPS");
    signalPassFailure();
#else
    mlir::ModuleOp moduleOp = getOperation();
    mlir::FailureOr<DefaultMeshSpec> meshSpec =
        getExecutionMeshSpec(moduleOp, executionMeshName);
    if (mlir::failed(meshSpec)) {
      moduleOp.emitOpError(
          "requires wafer.execution.mesh for default SPMD input sharding "
          "seeds");
      signalPassFailure();
      return;
    }

    llvm::SmallVector<mlir::func::FuncOp> funcs;
    moduleOp.walk([&](mlir::func::FuncOp funcOp) {
      if (needsDefaultSeed(funcOp))
        funcs.push_back(funcOp);
    });

    llvm::SmallVector<mlir::Operation *> frontendShardingCustomCalls;
    collectFrontendShardingCustomCalls(moduleOp, frontendShardingCustomCalls);

    if (funcs.empty() && frontendShardingCustomCalls.empty())
      return;

    mlir::SymbolTable symbolTable(moduleOp);
    mlir::sdy::MeshOp meshOp =
        symbolTable.lookup<mlir::sdy::MeshOp>(kDefaultMeshName);
    if (meshOp && !isCompatibleDefaultMesh(meshOp, *meshSpec)) {
      meshOp.emitOpError()
          << "conflicts with default Wafer SPMD mesh derived from "
             "wafer.execution.mesh";
      signalPassFailure();
      return;
    }

    mlir::FailureOr<llvm::SmallVector<FrontendShardingRewrite>> rewritePlan =
        buildFrontendShardingRewritePlan(moduleOp, *meshSpec);
    if (mlir::failed(rewritePlan)) {
      signalPassFailure();
      return;
    }

    if (!meshOp)
      meshOp = getOrCreateDefaultMesh(moduleOp, *meshSpec);
    applyFrontendShardingRewritePlan(*rewritePlan);
    for (mlir::func::FuncOp funcOp : funcs)
      applyDefaultInputSeeds(funcOp, *meshSpec);
#endif
  }
};

} // namespace

} // namespace wafer
