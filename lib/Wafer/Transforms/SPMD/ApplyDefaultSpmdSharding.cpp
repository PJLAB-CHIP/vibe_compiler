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
#include "llvm/Support/CommandLine.h"

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

#ifdef WAFER_ENABLE_SHARDY
namespace {

constexpr llvm::StringLiteral kDefaultMeshName = "wafer_default_tile_mesh";
constexpr llvm::StringLiteral kTileAxisName = "tile";

struct DefaultMeshSpec {
  llvm::SmallVector<std::string, 4> axes;
  llvm::SmallVector<int64_t, 4> shape;
  int64_t rankCount = 1;
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
    moduleOp.walk([&](ExecutionMeshOp candidate) {
      if (!meshOp)
        meshOp = candidate;
    });
  }
  if (!meshOp)
    return mlir::failure();

  DefaultMeshSpec spec;
  for (mlir::Attribute axisAttr : meshOp.getAxesAttr())
    spec.axes.push_back(
        mlir::cast<mlir::StringAttr>(axisAttr).getValue().str());
  for (int64_t dim : meshOp.getShapeAttr().asArrayRef()) {
    int64_t rankCount = 0;
    if (dim <= 0 || !checkedMul(spec.rankCount, dim, rankCount)) {
      meshOp.emitOpError("has invalid shape for default SPMD sharding");
      return mlir::failure();
    }
    spec.shape.push_back(dim);
    spec.rankCount = rankCount;
  }
  if (spec.axes.empty() || spec.axes.size() != spec.shape.size()) {
    meshOp.emitOpError("has inconsistent axes and shape");
    return mlir::failure();
  }
  return spec;
}

static DefaultMeshSpec getLegacyTileMeshSpec(int64_t tileCount) {
  DefaultMeshSpec spec;
  spec.axes.push_back(kTileAxisName.str());
  spec.shape.push_back(tileCount);
  spec.rankCount = tileCount;
  return spec;
}

static mlir::FailureOr<DefaultMeshSpec>
getDefaultMeshSpec(mlir::ModuleOp moduleOp, llvm::StringRef meshName,
                   int64_t fallbackTileCount) {
  mlir::FailureOr<DefaultMeshSpec> meshSpec =
      getExecutionMeshSpec(moduleOp, meshName);
  if (mlir::succeeded(meshSpec))
    return meshSpec;

  if (fallbackTileCount < 1 || fallbackTileCount > 16) {
    moduleOp.emitOpError("tile-count must be in [1, 16]");
    return mlir::failure();
  }
  return getLegacyTileMeshSpec(fallbackTileCount);
}

static bool isFrontendShardingAttr(mlir::NamedAttribute attr) {
  llvm::StringRef name = attr.getName().getValue();
  if (name == "mhlo.sharding") {
    if (auto stringAttr = mlir::dyn_cast<mlir::StringAttr>(attr.getValue()))
      return !stringAttr.getValue().empty();
    return true;
  }
  return name == "mhlo.spmd_parameters_sharding";
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
                                                  int64_t rankCount) {
  if (rankCount == 1)
    return std::nullopt;

  for (auto [index, dim] : llvm::enumerate(type.getShape())) {
    if (dim > 0 && !mlir::ShapedType::isDynamic(dim) && dim % rankCount == 0)
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
          findDefaultSplitDim(type, meshSpec.rankCount)) {
    dimShardings[*splitDim] = mlir::sdy::DimensionShardingAttr::get(
        context, meshAxes, /*is_closed=*/true);
  } else {
    replicatedAxes.append(meshAxes);
  }

  return mlir::sdy::TensorShardingAttr::get(context, kDefaultMeshName,
                                            dimShardings, replicatedAxes);
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

static mlir::sdy::MeshOp getOrCreateDefaultMesh(mlir::ModuleOp moduleOp,
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

struct ApplyDefaultSpmdShardingPass
    : public mlir::PassWrapper<ApplyDefaultSpmdShardingPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  using Base = mlir::PassWrapper<ApplyDefaultSpmdShardingPass,
                                 mlir::OperationPass<mlir::ModuleOp>>;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ApplyDefaultSpmdShardingPass)

  ApplyDefaultSpmdShardingPass() = default;
  explicit ApplyDefaultSpmdShardingPass(int64_t tileCount) {
    this->tileCount = tileCount;
  }
  ApplyDefaultSpmdShardingPass(const ApplyDefaultSpmdShardingPass &pass)
      : Base(pass) {
    tileCount = pass.tileCount;
    executionMeshName = pass.executionMeshName;
  }

  mlir::Pass::Option<int64_t> tileCount{
      *this, "tile-count",
      llvm::cl::desc("logical Wafer tile mesh size for default SPMD input "
                     "sharding seeds when no wafer.execution.mesh exists"),
      llvm::cl::init(16)};
  mlir::Pass::Option<std::string> executionMeshName{
      *this, "execution-mesh",
      llvm::cl::desc("wafer.execution.mesh symbol used for default SPMD input "
                     "sharding seeds"),
      llvm::cl::init("default_mesh")};

  llvm::StringRef getArgument() const final {
    return "wafer-apply-default-spmd-sharding";
  }

  llvm::StringRef getDescription() const final {
    return "apply Wafer default SDY function-input sharding seeds when a "
           "function has no user sharding seed";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::func::FuncDialect, mlir::sdy::SdyDialect,
                    WaferDialect>();
  }

  void runOnOperation() final {
    mlir::ModuleOp moduleOp = getOperation();
    mlir::FailureOr<DefaultMeshSpec> meshSpec =
        getDefaultMeshSpec(moduleOp, executionMeshName, tileCount);
    if (mlir::failed(meshSpec)) {
      signalPassFailure();
      return;
    }

    llvm::SmallVector<mlir::func::FuncOp> funcs;
    moduleOp.walk([&](mlir::func::FuncOp funcOp) {
      if (needsDefaultSeed(funcOp))
        funcs.push_back(funcOp);
    });

    if (funcs.empty())
      return;

    mlir::sdy::MeshOp meshOp = getOrCreateDefaultMesh(moduleOp, *meshSpec);
    if (!isCompatibleDefaultMesh(meshOp, *meshSpec)) {
      meshOp.emitOpError()
          << "conflicts with default Wafer SPMD mesh derived from "
             "wafer.execution.mesh";
      signalPassFailure();
      return;
    }

    for (mlir::func::FuncOp funcOp : funcs)
      applyDefaultInputSeeds(funcOp, *meshSpec);
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createApplyDefaultSpmdShardingPass() {
  return std::make_unique<ApplyDefaultSpmdShardingPass>();
}

std::unique_ptr<mlir::Pass>
createApplyDefaultSpmdShardingPass(int64_t tileCount) {
  return std::make_unique<ApplyDefaultSpmdShardingPass>(tileCount);
}

#endif // WAFER_ENABLE_SHARDY

} // namespace wafer
