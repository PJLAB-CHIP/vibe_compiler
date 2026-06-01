//===- Pipelines.cpp - Wafer named pipeline registration -----------------===//

#include "Wafer/Pipelines/Pipelines.h"

#include "Wafer/Conversion/Passes.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"

#include <memory>
#include <string>
#include <utility>

namespace wafer {
namespace {

constexpr llvm::StringRef kWaferTarget = "wafer";
constexpr llvm::StringRef kSingleTileMapping = "single";
constexpr llvm::StringRef kMultiTileNoCommMapping = "multi-tile-no-comm";

struct LocalPipelineOptions
    : public mlir::PassPipelineOptions<LocalPipelineOptions> {
  Option<std::string> target{
      *this, "target",
      llvm::cl::desc("User-level compile target. The supported target is "
                     "'wafer'."),
      llvm::cl::init(kWaferTarget.str())};
  Option<std::string> tileMapping{
      *this, "tile-mapping",
      llvm::cl::desc("Local tile materialization mode: 'single' or "
                     "'multi-tile-no-comm'."),
      llvm::cl::init(kSingleTileMapping.str())};
};

struct TargetPipelineOptions
    : public mlir::PassPipelineOptions<TargetPipelineOptions> {
  Option<std::string> target{
      *this, "target",
      llvm::cl::desc("User-level compile target. The supported target is "
                     "'wafer'."),
      llvm::cl::init(kWaferTarget.str())};
};

struct MaterializeCompileTargetPass
    : public mlir::PassWrapper<MaterializeCompileTargetPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializeCompileTargetPass)

  explicit MaterializeCompileTargetPass(llvm::StringRef target)
      : target(target.str()) {}

  llvm::StringRef getArgument() const final {
    return "wafer-materialize-compile-target";
  }

  llvm::StringRef getDescription() const final {
    return "materialize and verify the Wafer compile target contract";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<wafer::WaferDialect>();
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    if (target != kWaferTarget) {
      module.emitError() << "unsupported Wafer compile target '" << target
                         << "'; expected 'wafer'";
      signalPassFailure();
      return;
    }

    mlir::MLIRContext *context = module.getContext();
    wafer::TargetAttr expected =
        wafer::TargetAttr::get(context, wafer::Target::Wafer);
    mlir::Attribute current = module->getAttr("wafer.target");
    if (!current) {
      module->setAttr("wafer.target", expected);
      return;
    }

    auto currentTarget = mlir::dyn_cast<wafer::TargetAttr>(current);
    if (!currentTarget || currentTarget.getValue() != wafer::Target::Wafer) {
      module.emitError()
          << "module wafer.target does not match compile target 'wafer'";
      signalPassFailure();
    }
  }

  std::string target;
};

struct RejectPipelineConfigurationPass
    : public mlir::PassWrapper<RejectPipelineConfigurationPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RejectPipelineConfigurationPass)

  explicit RejectPipelineConfigurationPass(std::string message)
      : message(std::move(message)) {}

  llvm::StringRef getArgument() const final {
    return "wafer-reject-pipeline-configuration";
  }

  llvm::StringRef getDescription() const final {
    return "reject an invalid Wafer pipeline configuration";
  }

  void runOnOperation() final {
    getOperation().emitError() << message;
    signalPassFailure();
  }

  std::string message;
};

static void addCompileTargetContract(mlir::OpPassManager &pm,
                                     llvm::StringRef target) {
  pm.addPass(std::make_unique<MaterializeCompileTargetPass>(target));
}

static void addLocalTensorToCAbiBody(mlir::OpPassManager &pm,
                                     llvm::StringRef tileMapping) {
  pm.addPass(createFormGroupsPass());
  pm.addPass(createCheckRootTileCandidatesPass());
  if (tileMapping == kSingleTileMapping) {
    pm.addPass(createMaterializeSingleTilePass());
  } else if (tileMapping == kMultiTileNoCommMapping) {
    pm.addPass(createMaterializeMultiTileNoCommPass());
  } else {
    pm.addPass(std::make_unique<RejectPipelineConfigurationPass>(
        (llvm::Twine("unsupported local tile mapping '") + tileMapping +
         "'; expected 'single' or 'multi-tile-no-comm'")
            .str()));
  }
  pm.addPass(createCheckSPMAllocationPass());
  pm.addPass(createMaterializeDDRExternalBindingsPass());
  pm.addPass(createLowerTileRegionToCAbiPass());
}

static void addStablehloLocalTensorNormalization(mlir::OpPassManager &pm) {
  pm.addPass(createLowerStablehloReducePass());
  pm.addPass(createNormalizeConstantsPass());
  pm.addPass(createLowerStablehloDotPass());
  pm.addPass(createLowerStablehloElementwisePass());
  pm.addPass(createLowerStablehloShapePass());
}

} // namespace

void buildLocalLinalgToCAbiPipeline(mlir::OpPassManager &pm,
                                    llvm::StringRef target,
                                    llvm::StringRef tileMapping) {
  addCompileTargetContract(pm, target);
  addLocalTensorToCAbiBody(pm, tileMapping);
}

void buildLocalStablehloToCAbiPipeline(mlir::OpPassManager &pm,
                                       llvm::StringRef target,
                                       llvm::StringRef tileMapping) {
  addCompileTargetContract(pm, target);
  addStablehloLocalTensorNormalization(pm);
  addLocalTensorToCAbiBody(pm, tileMapping);
}

void buildTileCommunicationToCAbiPipeline(mlir::OpPassManager &pm,
                                          llvm::StringRef target) {
  addCompileTargetContract(pm, target);
  pm.addPass(createLowerRingAllGatherPass());
  pm.addPass(createLowerRingReduceCollectivesPass());
  pm.addPass(createLowerTileRegionToCAbiPass());
}

void registerWaferPipelines() {
  static bool registered = [] {
    mlir::PassPipelineRegistration<LocalPipelineOptions>(
        "wafer-lower-local-linalg-to-cabi",
        "Lower local structured tensor compute to the Wafer C ABI",
        [](mlir::OpPassManager &pm, const LocalPipelineOptions &options) {
          buildLocalLinalgToCAbiPipeline(pm, options.target,
                                         options.tileMapping);
        });
    mlir::PassPipelineRegistration<LocalPipelineOptions>(
        "wafer-lower-local-stablehlo-to-cabi",
        "Lower local StableHLO tensor compute to the Wafer C ABI",
        [](mlir::OpPassManager &pm, const LocalPipelineOptions &options) {
          buildLocalStablehloToCAbiPipeline(pm, options.target,
                                            options.tileMapping);
        });
    mlir::PassPipelineRegistration<TargetPipelineOptions>(
        "wafer-lower-tile-communication-to-cabi",
        "Lower tile-level Wafer communication IR to the Wafer C ABI",
        [](mlir::OpPassManager &pm, const TargetPipelineOptions &options) {
          buildTileCommunicationToCAbiPipeline(pm, options.target);
        });
    return true;
  }();
  (void)registered;
}

} // namespace wafer
