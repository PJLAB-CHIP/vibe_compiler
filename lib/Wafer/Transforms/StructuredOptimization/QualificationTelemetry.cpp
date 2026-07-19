//===- QualificationTelemetry.cpp - Pass invocation gateway -------------===//

#include "Wafer/Transforms/StructuredOptimization.h"

#include "Wafer/Support/CanonicalIRSnapshot.h"

#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/FormatVariadic.h"

#include <limits>
#include <utility>

namespace wafer {
namespace {

static uint64_t countOperations(mlir::Operation *root) {
  uint64_t count = 0;
  root->walk([&](mlir::Operation *) {
    if (count != std::numeric_limits<uint64_t>::max())
      ++count;
  });
  return count;
}

class OptimizationInvocationPass final
    : public mlir::PassWrapper<OptimizationInvocationPass,
                               mlir::OperationPass<>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptimizationInvocationPass)

  OptimizationInvocationPass(MechanismKey key,
                             OptimizationCutPoint cutPoint,
                             OptimizationPassFactory factory,
                             uint64_t invocationOrdinal)
      : key_(key), cutPoint_(cutPoint), factory_(std::move(factory)),
        invocationOrdinal_(invocationOrdinal) {}

  OptimizationInvocationPass(const OptimizationInvocationPass &other)
      : mlir::PassWrapper<OptimizationInvocationPass,
                          mlir::OperationPass<>>(other),
        key_(other.key_), cutPoint_(other.cutPoint_), factory_(other.factory_),
        invocationOrdinal_(other.invocationOrdinal_) {}

  llvm::StringRef getName() const override {
    return "OptimizationInvocationGateway";
  }

  OptimizationInvocationDescription describe() const {
    OptimizationInvocationDescription description;
    description.key = key_;
    description.cutPoint = cutPoint_;
    if (std::unique_ptr<mlir::Pass> owner = factory_()) {
      llvm::raw_string_ostream stream(description.ownerTextualPipeline);
      owner->printAsTextualPipeline(stream);
    }
    return description;
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    std::unique_ptr<mlir::Pass> pass = factory_();
    if (pass)
      pass->getDependentDialects(registry);
  }

  void runOnOperation() override {
    mlir::Operation *root = getOperation();
    std::unique_ptr<mlir::Pass> ownerPass = factory_();
    if (!ownerPass) {
      root->emitError("optimization invocation factory returned no owner pass");
      return signalPassFailure();
    }

    mlir::OperationFingerPrint before(root);
    CanonicalIRSnapshotV1 inputSnapshot;
    std::string snapshotDiagnostic;
    if (!createCanonicalIRSnapshotV1(
            root, CanonicalIRSnapshotMode::SemanticStructure, inputSnapshot,
            &snapshotDiagnostic)) {
      root->emitError() << "cannot establish optimization input snapshot: "
                        << snapshotDiagnostic;
      return signalPassFailure();
    }
    OptimizationInvocationTokenV1 invocationToken;
    std::string beginDiagnostic;
    if (!beginOptimizationInvocationV1(
            key_, cutPoint_, invocationOrdinal_, inputSnapshot.sha256Digest,
            invocationToken, &beginDiagnostic)) {
      root->emitError() << "cannot begin optimization invocation for "
                        << key_.semanticId << ": " << beginDiagnostic;
      return signalPassFailure();
    }
    uint64_t beforeWork = countOperations(root);

    mlir::OpPassManager nested(root->getName());
    nested.addPass(std::move(ownerPass));
    mlir::LogicalResult result = runPipeline(nested, root);

    mlir::OperationFingerPrint after(root);
    uint64_t afterWork = countOperations(root);
    uint64_t work = beforeWork;
    if (std::numeric_limits<uint64_t>::max() - work < afterWork)
      work = std::numeric_limits<uint64_t>::max();
    else
      work += afterWork;

    OptimizationInvocationTelemetry telemetry;
    telemetry.key = key_;
    telemetry.cutPoint = cutPoint_;
    telemetry.invocationOrdinal = invocationOrdinal_;
    telemetry.inputSnapshotDigest = inputSnapshot.sha256Digest;
    telemetry.workUnits = work;
    if (mlir::failed(result)) {
      telemetry.outcome = InvocationOutcome::Invalid;
    } else if (before == after) {
      telemetry.outcome = InvocationOutcome::NoChange;
    } else {
      telemetry.outcome = InvocationOutcome::Applied;
      // This is a transaction-level rewrite count.  Owner mechanisms that
      // expose finer rewrite events may add those to their own work summary;
      // the gateway never guesses pattern counts from op-count deltas.
      telemetry.rewriteCount = 1;
    }

    std::string diagnostic;
    if (!commitOptimizationInvocationV1(invocationToken, telemetry,
                                        &diagnostic)) {
      root->emitError() << "invalid optimization invocation telemetry for "
                        << key_.semanticId << ": " << diagnostic;
      return signalPassFailure();
    }
    if (mlir::failed(result))
      signalPassFailure();
  }

private:
  MechanismKey key_;
  OptimizationCutPoint cutPoint_;
  OptimizationPassFactory factory_;
  uint64_t invocationOrdinal_ = 0;
};

} // namespace

std::optional<OptimizationInvocationDescription>
describeOptimizationInvocationPass(const mlir::Pass &pass) {
  auto *gateway = mlir::dyn_cast<OptimizationInvocationPass>(&pass);
  if (!gateway)
    return std::nullopt;
  return gateway->describe();
}

std::unique_ptr<mlir::Pass> createOptimizationInvocationPass(
    MechanismKey key, OptimizationCutPoint cutPoint,
    OptimizationPassFactory factory, uint64_t invocationOrdinal) {
  return std::make_unique<OptimizationInvocationPass>(key, cutPoint,
                                                       std::move(factory),
                                                       invocationOrdinal);
}

} // namespace wafer
