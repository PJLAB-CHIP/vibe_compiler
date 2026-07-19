//===- DebugQualificationInstrumentation.cpp - wafer-opt gateway --------===//

#include "Wafer/Transforms/StructuredOptimization.h"

#include "Wafer/Support/CanonicalIRSnapshot.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Threading.h"

#include <limits>
#include <map>
#include <mutex>

namespace wafer {
namespace {

struct DebugInvocation {
  MechanismKey key;
  OptimizationCutPoint cutPoint;
  mlir::OperationFingerPrint before;
  AdoptionDigest inputSnapshotDigest;
  OptimizationInvocationTokenV1 token;

  DebugInvocation(MechanismKey key, OptimizationCutPoint cutPoint,
                  mlir::Operation *root, AdoptionDigest inputSnapshotDigest,
                  OptimizationInvocationTokenV1 token)
      : key(key), cutPoint(cutPoint), before(root),
        inputSnapshotDigest(inputSnapshotDigest), token(std::move(token)) {}
};

static std::optional<std::pair<MechanismKey, OptimizationCutPoint>>
classifyDebugPass(mlir::Pass *pass, mlir::Operation *root) {
  llvm::StringRef argument = pass->getArgument();
  if (argument.empty() || argument.starts_with("wafer-"))
    return std::nullopt;
  if (argument == "cse")
    return std::make_pair(
        mechanism::ScalarCommonSubexpressionElimination,
        OptimizationCutPoint::StructuredTensorModule);
  if (argument == "sccp")
    return std::make_pair(mechanism::SparseConditionalConstantPropagation,
                          OptimizationCutPoint::StructuredTensorModule);

  if (argument == "canonicalize") {
    bool hasInstruction = false;
    bool hasSelectedPayload = false;
    root->walk([&](mlir::Operation *operation) {
      llvm::StringRef dialect = operation->getName().getDialectNamespace();
      llvm::StringRef name = operation->getName().getStringRef();
      hasInstruction |= name.starts_with("wafer.instr.") || dialect == "llvm";
      hasSelectedPayload |= name.starts_with("wafer.tile.") ||
                            dialect == "memref" || dialect == "bufferization";
    });
    if (hasInstruction)
      return std::make_pair(mechanism::PostMemoryPlanningCleanup,
                            OptimizationCutPoint::FinalInstructionModule);
    if (hasSelectedPayload)
      return std::make_pair(mechanism::PreBufferizationCleanup,
                            OptimizationCutPoint::SelectedPhysicalPayloadModule);
    return std::make_pair(mechanism::StructuredTensorCleanup,
                          OptimizationCutPoint::StructuredTensorModule);
  }

  if (argument.contains("buffer") || argument.contains("one-shot"))
    return std::make_pair(mechanism::BufferizationTransformFamily,
                          OptimizationCutPoint::SelectedPhysicalPayloadModule);
  if (argument.starts_with("linalg-") || argument.contains("-linalg-"))
    return std::make_pair(mechanism::LinalgTransformFamily,
                          OptimizationCutPoint::StructuredTensorModule);
  if (argument.starts_with("tensor-") || argument.contains("-tensor-"))
    return std::make_pair(mechanism::TensorTransformFamily,
                          OptimizationCutPoint::StructuredTensorModule);
  if (argument.starts_with("scf-") || argument.contains("-scf-"))
    return std::make_pair(mechanism::ScfTransformFamily,
                          OptimizationCutPoint::StructuredTensorModule);
  if (argument.starts_with("arith-") || argument.contains("-arith-"))
    return std::make_pair(mechanism::ArithTransformFamily,
                          OptimizationCutPoint::StructuredTensorModule);
  return std::nullopt;
}

class DebugQualificationInstrumentation final
    : public mlir::PassInstrumentation {
public:
  void runBeforePipeline(
      std::optional<mlir::OperationName>,
      const PipelineParentInfo &parentInfo) override {
    if (!parentInfo.parentPass ||
        !describeOptimizationInvocationPass(*parentInfo.parentPass))
      return;
    std::lock_guard<std::mutex> lock(mutex);
    ++suppressedThreads[llvm::get_threadid()];
  }

  void runAfterPipeline(std::optional<mlir::OperationName>,
                        const PipelineParentInfo &parentInfo) override {
    if (!parentInfo.parentPass ||
        !describeOptimizationInvocationPass(*parentInfo.parentPass))
      return;
    std::lock_guard<std::mutex> lock(mutex);
    auto found = suppressedThreads.find(llvm::get_threadid());
    if (found == suppressedThreads.end())
      return;
    if (--found->second == 0)
      suppressedThreads.erase(found);
  }

  void runBeforePass(mlir::Pass *pass, mlir::Operation *operation) override {
    if (describeOptimizationInvocationPass(*pass))
      return;
    std::lock_guard<std::mutex> lock(mutex);
    if (suppressedThreads.count(llvm::get_threadid()))
      return;
    auto classification = classifyDebugPass(pass, operation);
    if (!classification)
      return;
    CanonicalIRSnapshotV1 inputSnapshot;
    std::string diagnostic;
    if (!createCanonicalIRSnapshotV1(
            operation, CanonicalIRSnapshotMode::SemanticStructure,
            inputSnapshot, &diagnostic)) {
      operation->emitError() << "cannot establish debug optimization input "
                                "snapshot: "
                             << diagnostic;
      return;
    }
    OptimizationInvocationTokenV1 token;
    if (!beginOptimizationInvocationV1(
            classification->first, classification->second,
            /*invocationOrdinal=*/0, inputSnapshot.sha256Digest, token,
            &diagnostic)) {
      operation->emitError() << "cannot begin debug optimization invocation: "
                             << diagnostic;
      return;
    }
    invocations.emplace(pass, DebugInvocation(classification->first,
                                               classification->second,
                                               operation,
                                               inputSnapshot.sha256Digest,
                                               std::move(token)));
  }

  void runAfterPass(mlir::Pass *pass, mlir::Operation *operation) override {
    finish(pass, operation, /*failed=*/false);
  }

  void runAfterPassFailed(mlir::Pass *pass,
                          mlir::Operation *operation) override {
    finish(pass, operation, /*failed=*/true);
  }

private:
  void finish(mlir::Pass *pass, mlir::Operation *operation, bool failed) {
    std::optional<DebugInvocation> invocation;
    {
      std::lock_guard<std::mutex> lock(mutex);
      auto found = invocations.find(pass);
      if (found == invocations.end())
        return;
      invocation.emplace(std::move(found->second));
      invocations.erase(found);
    }
    mlir::OperationFingerPrint after(operation);
    OptimizationInvocationTelemetry telemetry;
    telemetry.key = invocation->key;
    telemetry.cutPoint = invocation->cutPoint;
    telemetry.inputSnapshotDigest = invocation->inputSnapshotDigest;
    telemetry.outcome = failed ? InvocationOutcome::Invalid
                               : invocation->before == after
                                     ? InvocationOutcome::NoChange
                                     : InvocationOutcome::Applied;
    telemetry.rewriteCount = telemetry.outcome == InvocationOutcome::Applied;
    uint64_t operations = 0;
    operation->walk([&](mlir::Operation *) {
      if (operations != std::numeric_limits<uint64_t>::max())
        ++operations;
    });
    telemetry.workUnits = operations;
    std::string diagnostic;
    if (!commitOptimizationInvocationV1(invocation->token, telemetry,
                                        &diagnostic))
      operation->emitError() << "invalid debug optimization invocation: "
                             << diagnostic;
  }

  std::mutex mutex;
  std::map<uint64_t, unsigned> suppressedThreads;
  std::map<mlir::Pass *, DebugInvocation> invocations;
};

} // namespace

std::unique_ptr<mlir::PassInstrumentation>
createDebugOptimizationInvocationInstrumentation() {
  return std::make_unique<DebugQualificationInstrumentation>();
}

} // namespace wafer
