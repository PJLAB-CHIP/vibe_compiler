//===- ApplyDefaultSpmdSharding.cpp - Default SDY input seeds ------------===//

#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"

#ifdef WAFER_ENABLE_SHARDY
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#endif

#include <memory>
#include <optional>

namespace wafer {

#ifdef WAFER_ENABLE_SHARDY
namespace {

constexpr llvm::StringLiteral kDefaultMeshName = "wafer_default_tile_mesh";
constexpr llvm::StringLiteral kTileAxisName = "tile";

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
  auto funcIface =
      mlir::cast<mlir::FunctionOpInterface>(funcOp.getOperation());

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
                                                  int64_t tileCount) {
  if (tileCount == 1)
    return std::nullopt;

  for (auto [index, dim] : llvm::enumerate(type.getShape())) {
    if (dim > 0 && !mlir::ShapedType::isDynamic(dim) &&
        dim % tileCount == 0)
      return static_cast<int64_t>(index);
  }
  return std::nullopt;
}

static mlir::sdy::TensorShardingAttr
buildDefaultInputSharding(mlir::MLIRContext *context,
                          mlir::RankedTensorType type, int64_t tileCount) {
  mlir::sdy::AxisRefAttr tileAxis =
      mlir::sdy::AxisRefAttr::get(context, kTileAxisName);
  mlir::sdy::DimensionShardingAttr replicatedDim =
      mlir::sdy::DimensionShardingAttr::get(context, {}, /*is_closed=*/true);

  llvm::SmallVector<mlir::sdy::DimensionShardingAttr> dimShardings(
      type.getRank(), replicatedDim);
  llvm::SmallVector<mlir::sdy::AxisRefAttr> replicatedAxes;

  if (std::optional<int64_t> splitDim = findDefaultSplitDim(type, tileCount)) {
    dimShardings[*splitDim] = mlir::sdy::DimensionShardingAttr::get(
        context, {tileAxis}, /*is_closed=*/true);
  } else {
    replicatedAxes.push_back(tileAxis);
  }

  return mlir::sdy::TensorShardingAttr::get(
      context, kDefaultMeshName, dimShardings, replicatedAxes);
}

static bool isCompatibleDefaultMesh(mlir::sdy::MeshOp meshOp,
                                    int64_t tileCount) {
  llvm::ArrayRef<mlir::sdy::MeshAxisAttr> axes = meshOp.getMesh().getAxes();
  return axes.size() == 1 && axes.front().getName() == kTileAxisName &&
         axes.front().getSize() == tileCount;
}

static mlir::sdy::MeshOp getOrCreateDefaultMesh(mlir::ModuleOp moduleOp,
                                                int64_t tileCount) {
  mlir::SymbolTable symbolTable(moduleOp);
  if (auto existing = symbolTable.lookup<mlir::sdy::MeshOp>(kDefaultMeshName))
    return existing;

  mlir::OpBuilder builder(moduleOp.getContext());
  builder.setInsertionPointToStart(moduleOp.getBody());
  return builder.create<mlir::sdy::MeshOp>(
      moduleOp.getLoc(), kDefaultMeshName,
      mlir::sdy::MeshAttr::get(moduleOp.getContext(),
                               {mlir::sdy::MeshAxisAttr::get(
                                   moduleOp.getContext(), kTileAxisName,
                                   tileCount)}));
}

static bool needsDefaultSeed(mlir::func::FuncOp funcOp) {
  if (hasShardingSeed(funcOp))
    return false;

  return llvm::any_of(funcOp.getFunctionType().getInputs(),
                      [](mlir::Type type) {
                        return mlir::isa<mlir::RankedTensorType>(type);
                      });
}

static void applyDefaultInputSeeds(mlir::func::FuncOp funcOp,
                                   int64_t tileCount) {
  mlir::MLIRContext *context = funcOp.getContext();
  auto funcIface =
      mlir::cast<mlir::FunctionOpInterface>(funcOp.getOperation());
  for (auto [index, type] :
       llvm::enumerate(funcOp.getFunctionType().getInputs())) {
    auto rankedType = mlir::dyn_cast<mlir::RankedTensorType>(type);
    if (!rankedType)
      continue;
    funcIface.setArgAttr(
        static_cast<unsigned>(index), mlir::sdy::kShardingAttr,
        buildDefaultInputSharding(context, rankedType, tileCount));
  }
}

struct ApplyDefaultSpmdShardingPass
    : public mlir::PassWrapper<ApplyDefaultSpmdShardingPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  using Base = mlir::PassWrapper<ApplyDefaultSpmdShardingPass,
                                 mlir::OperationPass<mlir::ModuleOp>>;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ApplyDefaultSpmdShardingPass)

  ApplyDefaultSpmdShardingPass() = default;
  ApplyDefaultSpmdShardingPass(const ApplyDefaultSpmdShardingPass &pass)
      : Base(pass) {
    tileCount = pass.tileCount;
  }

  mlir::Pass::Option<int64_t> tileCount{
      *this, "tile-count",
      llvm::cl::desc("logical Wafer tile mesh size for default SPMD input "
                     "sharding seeds"),
      llvm::cl::init(16)};

  llvm::StringRef getArgument() const final {
    return "wafer-apply-default-spmd-sharding";
  }

  llvm::StringRef getDescription() const final {
    return "apply Wafer default SDY function-input sharding seeds when a "
           "function has no user sharding seed";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::func::FuncDialect, mlir::sdy::SdyDialect>();
  }

  void runOnOperation() final {
    mlir::ModuleOp moduleOp = getOperation();
    if (tileCount < 1 || tileCount > 16) {
      moduleOp.emitOpError("tile-count must be in [1, 16]");
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

    mlir::sdy::MeshOp meshOp = getOrCreateDefaultMesh(moduleOp, tileCount);
    if (!isCompatibleDefaultMesh(meshOp, tileCount)) {
      meshOp.emitOpError()
          << "conflicts with default Wafer SPMD mesh; expected single axis "
          << '"' << kTileAxisName << "\" of size " << tileCount;
      signalPassFailure();
      return;
    }

    for (mlir::func::FuncOp funcOp : funcs)
      applyDefaultInputSeeds(funcOp, tileCount);
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createApplyDefaultSpmdShardingPass() {
  return std::make_unique<ApplyDefaultSpmdShardingPass>();
}

#endif // WAFER_ENABLE_SHARDY

} // namespace wafer
