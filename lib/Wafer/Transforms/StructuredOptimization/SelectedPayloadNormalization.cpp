//===- SelectedPayloadNormalization.cpp - Physical payload normal form --===//

#include "Wafer/Transforms/StructuredOptimization.h"

#include "MemoryPlanning/LifetimeAnalysis.h"
#include "StructuredOptimizationInternal.h"
#include "Wafer/Analysis/IdentityViewProof.h"
#include "Wafer/Analysis/SchedulableCallClosure.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/IR/WaferInterfaces.h"
#include "Wafer/Support/CanonicalIRSnapshot.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>

namespace wafer {
namespace {

namespace mp = memory_planning::detail;

using Outcome = SelectedPayloadNormalizationOutcome;
using Status = SelectedPayloadNormalizationStatus;
using Family = SelectedPayloadNormalizationFamily;
using Work = SelectedPayloadNormalizationWorkSummary;

static bool add(uint64_t &target, uint64_t value) {
  if (value > std::numeric_limits<uint64_t>::max() - target)
    return false;
  target += value;
  return true;
}

static bool multiply(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (lhs && rhs > std::numeric_limits<uint64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

static bool consume(Work &work, uint64_t amount) {
  if (work.fuelConsumed > work.fuelLimit ||
      amount > work.fuelLimit - work.fuelConsumed)
    return false;
  work.fuelConsumed += amount;
  return true;
}

static Outcome fail(Status status, Family family, llvm::StringRef reason,
                    mlir::Operation *operation, mlir::ModuleOp module,
                    Work work) {
  Outcome outcome;
  outcome.status = status;
  outcome.diagnostic = SelectedPayloadNormalizationDiagnostic{
      family, reason.str(),
      structured_optimization::detail::getCanonicalOperationPath(operation,
                                                                 module)};
  outcome.work = work;
  return outcome;
}

static bool initialize(llvm::ArrayRef<mlir::func::FuncOp> functions,
                       Work &work) {
  bool valid = true;
  for (mlir::func::FuncOp function : functions)
    function.walk([&](mlir::Operation *operation) {
      valid &= add(work.initialOperationCount, 1);
      valid &= add(work.initialEdgeCount, operation->getNumOperands());
      valid &= add(work.initialRegionCount, operation->getNumRegions());
      if (mlir::isa<mlir::memref::SubViewOp, mlir::memref::ViewOp,
                    mlir::memref::ReinterpretCastOp>(operation))
        valid &= add(work.initialViewRootNodes, 1);

      if (auto standard =
              mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation)) {
        llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 8> effects;
        standard.getEffects(effects);
        valid &= add(work.initialEffectEntries, effects.size());
      }
      if (auto detailed =
              mlir::dyn_cast<WaferResourceEffectInterface>(operation)) {
        llvm::SmallVector<WaferResourceEffect, 8> effects;
        detailed.collectWaferResourceEffects(effects);
        valid &= add(work.initialEffectEntries, effects.size());
      }
    });
  if (!valid)
    return false;
  uint64_t operations = 0, edges = 0, regions = 0, views = 0, effects = 0;
  if (!multiply(work.initialOperationCount, 32, operations) ||
      !multiply(work.initialEdgeCount, 8, edges) ||
      !multiply(work.initialRegionCount, 16, regions) ||
      !multiply(work.initialViewRootNodes, 8, views) ||
      !multiply(work.initialEffectEntries, 4, effects))
    return false;
  work.fuelLimit = 1024;
  return add(work.fuelLimit, operations) && add(work.fuelLimit, edges) &&
         add(work.fuelLimit, regions) && add(work.fuelLimit, views) &&
         add(work.fuelLimit, effects);
}

static bool merge(Work &destination, const Work &source) {
  Work merged = destination;
  if (!add(merged.fuelConsumed, source.fuelConsumed) ||
      !add(merged.operationVisits, source.operationVisits) ||
      !add(merged.regionVisits, source.regionVisits) ||
      !add(merged.rewriteAttempts, source.rewriteAttempts) ||
      !add(merged.committedRewrites, source.committedRewrites) ||
      !add(merged.newOperations, source.newOperations) ||
      !add(merged.proofNodes, source.proofNodes))
    return false;
  destination = merged;
  return true;
}

static bool isWaferStandardResource(mlir::SideEffects::Resource *resource) {
  return llvm::isa<WaferSPMResource, WaferDDRResource, WaferMovementResource,
                   WaferComputeResource, WaferCommunicationResource,
                   WaferSyncResource>(resource);
}

static bool standardEffectCovers(
    llvm::ArrayRef<mlir::MemoryEffects::EffectInstance> standard,
    const WaferResourceEffect &detailed) {
  return llvm::any_of(standard, [&](const auto &effect) {
    bool resourceMatches = false;
    switch (detailed.resource) {
    case WaferResourceKind::SPM:
      resourceMatches = llvm::isa<WaferSPMResource>(effect.getResource());
      break;
    case WaferResourceKind::DDR:
      resourceMatches = llvm::isa<WaferDDRResource>(effect.getResource());
      break;
    case WaferResourceKind::Movement:
      resourceMatches = llvm::isa<WaferMovementResource>(effect.getResource());
      break;
    case WaferResourceKind::Compute:
      resourceMatches = llvm::isa<WaferComputeResource>(effect.getResource());
      break;
    case WaferResourceKind::Communication:
      resourceMatches =
          llvm::isa<WaferCommunicationResource>(effect.getResource());
      break;
    case WaferResourceKind::Sync:
      resourceMatches = llvm::isa<WaferSyncResource>(effect.getResource());
      break;
    }
    if (!resourceMatches)
      return false;
    if (detailed.access == WaferResourceAccess::Read)
      return llvm::isa<mlir::MemoryEffects::Read>(effect.getEffect());
    return llvm::isa<mlir::MemoryEffects::Write>(effect.getEffect());
  });
}

static std::optional<Outcome> verifyLifetimeClosure(mlir::func::FuncOp function,
                                                    mlir::ModuleOp module,
                                                    Work &work) {
  uint64_t operationCount = 0;
  bool countValid = true;
  function.walk(
      [&](mlir::Operation *) { countValid &= add(operationCount, 1); });
  uint64_t proofFuel = 0;
  if (!countValid || !multiply(operationCount, 5, proofFuel) ||
      !add(work.proofNodes, proofFuel) || !consume(work, proofFuel))
    return fail(Status::ResourceExhausted, Family::ProgramEnvelope,
                "selected payload lifetime proof exhausted its work policy",
                function, module, work);

  mp::TimelineFailure timelineFailure;
  mlir::FailureOr<mp::StructuredTimeline> timeline =
      mp::StructuredTimeline::build(function, &timelineFailure);
  if (mlir::failed(timeline))
    return fail(Status::UnsupportedSemantic, Family::Traversal,
                "selected payload requires supported structured control flow",
                timelineFailure.origin ? timelineFailure.origin
                                       : function.getOperation(),
                module, work);

  for (WaferResourceKind resource :
       {WaferResourceKind::SPM, WaferResourceKind::DDR}) {
    llvm::SmallVector<mp::LifetimeDemand, 0> noPlacementDemands;
    mp::LocalCompletionTracker completion(resource);
    mp::ValueResolver resolver;
    mp::ExplicitRootPredicate explicitRoot;
    if (resource == WaferResourceKind::DDR) {
      resolver = mp::resolveTileRegionBoundaryValue;
      explicitRoot = mp::isExplicitDDRRoot;
    }
    mp::LifetimeDataflow dataflow(
        *timeline, noPlacementDemands,
        [resource](mlir::Type type) {
          return resource == WaferResourceKind::SPM
                     ? isWaferSPMMemRefType(type)
                     : isWaferDDRMemRefType(type);
        },
        std::move(resolver), std::move(explicitRoot));
    mp::LifetimeFailure lifetimeFailure;
    if (mlir::succeeded(dataflow.run(function, &completion, &lifetimeFailure)))
      continue;
    mlir::Operation *origin = lifetimeFailure.origin ? lifetimeFailure.origin
                                                     : function.getOperation();
    switch (lifetimeFailure.kind) {
    case mp::LifetimeFailureKind::MissingAllocationEvent:
      return fail(Status::InternalInvariant, Family::AllocationRootLayout,
                  "tracked allocation has no structured timeline event", origin,
                  module, work);
    case mp::LifetimeFailureKind::UnsupportedTrackedValueProducer: {
      std::string operationText;
      llvm::raw_string_ostream operationStream(operationText);
      origin->print(operationStream);
      if (auto toMemref =
              mlir::dyn_cast<mlir::bufferization::ToMemrefOp>(origin))
        if (mlir::Operation *source = toMemref.getTensor().getDefiningOp()) {
          operationStream << " from ";
          source->print(operationStream);
        }
      std::string reason =
          "tracked memref has no supported allocation or boundary root at " +
          operationStream.str();
      return fail(Status::UnsupportedSemantic, Family::AllocationRootLayout,
                  reason, origin, module, work);
    }
    case mp::LifetimeFailureKind::UnsupportedTrackedValueEscape: {
      std::string reason =
          "tracked memref escapes through unsupported alias semantics at " +
          origin->getName().getStringRef().str();
      return fail(Status::UnsupportedSemantic, Family::MemRefRelation, reason,
                  origin, module, work);
    }
    case mp::LifetimeFailureKind::LoopCarriedAllocationInstance:
      return fail(Status::UnsupportedSemantic, Family::AllocationRootLayout,
                  "loop-carried allocation requires unsupported multi-instance "
                  "placement",
                  origin, module, work);
    case mp::LifetimeFailureKind::MissingAsyncCompletion:
      return fail(Status::InvalidIR, Family::Completion,
                  "asynchronous tracked access has no path-covering completion",
                  origin, module, work);
    case mp::LifetimeFailureKind::UnsupportedAsyncCompletionFlow:
      return fail(Status::UnsupportedSemantic, Family::Completion,
                  "asynchronous completion identity cannot be proven", origin,
                  module, work);
    case mp::LifetimeFailureKind::MissingLocalCompletion:
      return fail(Status::InvalidIR, Family::Completion,
                  "local tracked access has no path-covering fence", origin,
                  module, work);
    case mp::LifetimeFailureKind::LoopBackedgeCompletion:
      return fail(
          Status::InvalidIR, Family::Completion,
          "local tracked access reaches a loop backedge without a fence",
          origin, module, work);
    case mp::LifetimeFailureKind::InconsistentCompletionState:
      return fail(Status::InternalInvariant, Family::Completion,
                  "completion proof retained inconsistent pending state",
                  origin, module, work);
    }
  }
  return std::nullopt;
}

static Outcome rewrite(mlir::ModuleOp module) {
  Work work;
  SchedulableCallClosure closure = analyzeAllSchedulableCallClosures(module);
  if (closure.status != SchedulableCallClosureStatus::Success) {
    Status status = closure.status == SchedulableCallClosureStatus::InvalidIR
                        ? Status::InvalidIR
                        : Status::UnsupportedSemantic;
    return fail(status, Family::ProgramEnvelope,
                describeSchedulableCallClosureReason(*closure.reason),
                closure.diagnosticOperation ? closure.diagnosticOperation
                                            : module.getOperation(),
                module, work);
  }
  if (!initialize(closure.functions, work))
    return fail(Status::ResourceExhausted, Family::ProgramEnvelope,
                "selected payload work-policy overflow", module, module, work);

  mlir::IRRewriter rewriter(module.getContext());
  llvm::SmallVector<mlir::Operation *, 32> operations;
  for (mlir::func::FuncOp function : closure.functions)
    function.walk<mlir::WalkOrder::PostOrder>(
        [&](mlir::Operation *operation) { operations.push_back(operation); });
  for (mlir::Operation *operation : operations) {
    uint64_t visitFuel = 1;
    if (!add(work.operationVisits, 1) ||
        !add(work.regionVisits, operation->getNumRegions()) ||
        !add(visitFuel, operation->getNumRegions()) ||
        !consume(work, visitFuel))
      return fail(Status::ResourceExhausted, Family::ProgramEnvelope,
                  "selected payload exhausted its work policy", operation,
                  module, work);

    if (auto subview =
            mlir::dyn_cast_or_null<mlir::memref::SubViewOp>(operation)) {
      if (!add(work.rewriteAttempts, 1) || !consume(work, 2))
        return fail(Status::ResourceExhausted, Family::MemRefRelation,
                    "identity subview rewrite exhausted its work policy",
                    operation, module, work);
      IdentityViewProof proof = proveIdentityView(subview);
      uint64_t proofFuel = 0;
      if (!add(work.proofNodes, proof.inspectedDimensions) ||
          !multiply(2, proof.inspectedDimensions, proofFuel) ||
          !consume(work, proofFuel))
        return fail(Status::ResourceExhausted, Family::MemRefRelation,
                    "identity subview proof exhausted its work policy",
                    operation, module, work);
      if (proof.status != IdentityProofStatus::ProvenIdentity)
        continue;
      mlir::Value replacement = subview.getSource();
      if (replacement.getType() != subview.getType()) {
        if (!mlir::memref::CastOp::areCastCompatible(replacement.getType(),
                                                     subview.getType()))
          return fail(Status::UnsupportedSemantic, Family::MemRefRelation,
                      "identity subview requires a lossy result type change",
                      operation, module, work);
        rewriter.setInsertionPoint(subview);
        replacement = rewriter
                          .create<mlir::memref::CastOp>(
                              subview.getLoc(), subview.getType(), replacement)
                          .getResult();
        if (!add(work.newOperations, 1))
          return fail(Status::ResourceExhausted, Family::MemRefRelation,
                      "identity subview cast accounting overflow", operation,
                      module, work);
      }
      rewriter.replaceOp(subview, replacement);
      if (!add(work.committedRewrites, 1) || !consume(work, 8))
        return fail(Status::ResourceExhausted, Family::MemRefRelation,
                    "identity subview commit exhausted its work policy", module,
                    module, work);
      continue;
    }

    if (auto loop = mlir::dyn_cast_or_null<mlir::scf::ForOp>(operation)) {
      if (!add(work.rewriteAttempts, 1) || !consume(work, 2))
        return fail(Status::ResourceExhausted, Family::Traversal,
                    "traversal rewrite exhausted its work policy", operation,
                    module, work);
      StaticTripCountProof proof = proveStaticTripCount(loop);
      if (!proof.tripCount || (*proof.tripCount != 0 && *proof.tripCount != 1))
        continue;
      rewriter.setInsertionPoint(loop);
      if (*proof.tripCount == 0)
        rewriter.replaceOp(loop, loop.getInitArgs());
      else if (mlir::failed(loop.promoteIfSingleIteration(rewriter)))
        return fail(Status::InternalInvariant, Family::Traversal,
                    "proven one-trip traversal could not be promoted",
                    operation, module, work);
      if (!add(work.committedRewrites, 1) || !consume(work, 8))
        return fail(Status::ResourceExhausted, Family::Traversal,
                    "traversal commit exhausted its work policy", module,
                    module, work);
    }
  }

  // A zero/one-trip promotion can expose an exact dynamic offset expression
  // that was nested below the loop during the post-order sweep. Revisit only
  // memref views after all traversal rewrites; this is one bounded phase, not
  // an open-ended canonicalization fixed point.
  llvm::SmallVector<mlir::memref::SubViewOp, 8> exposedSubviews;
  for (mlir::func::FuncOp function : closure.functions)
    function.walk<mlir::WalkOrder::PostOrder>(
        [&](mlir::memref::SubViewOp subview) {
          exposedSubviews.push_back(subview);
        });
  for (mlir::memref::SubViewOp subview : exposedSubviews) {
    if (!add(work.rewriteAttempts, 1) || !consume(work, 2))
      return fail(Status::ResourceExhausted, Family::MemRefRelation,
                  "exposed identity subview rewrite exhausted its work policy",
                  subview, module, work);
    IdentityViewProof proof = proveIdentityView(subview);
    uint64_t proofFuel = 0;
    if (!add(work.proofNodes, proof.inspectedDimensions) ||
        !multiply(2, proof.inspectedDimensions, proofFuel) ||
        !consume(work, proofFuel))
      return fail(Status::ResourceExhausted, Family::MemRefRelation,
                  "exposed identity subview proof exhausted its work policy",
                  subview, module, work);
    if (proof.status != IdentityProofStatus::ProvenIdentity)
      continue;
    mlir::Value replacement = subview.getSource();
    if (replacement.getType() != subview.getType()) {
      if (!mlir::memref::CastOp::areCastCompatible(replacement.getType(),
                                                   subview.getType()))
        return fail(Status::UnsupportedSemantic, Family::MemRefRelation,
                    "identity subview requires a lossy result type change",
                    subview, module, work);
      rewriter.setInsertionPoint(subview);
      replacement = rewriter
                        .create<mlir::memref::CastOp>(
                            subview.getLoc(), subview.getType(), replacement)
                        .getResult();
      if (!add(work.newOperations, 1))
        return fail(Status::ResourceExhausted, Family::MemRefRelation,
                    "identity subview cast accounting overflow", subview,
                    module, work);
    }
    rewriter.replaceOp(subview, replacement);
    if (!add(work.committedRewrites, 1) || !consume(work, 8))
      return fail(Status::ResourceExhausted, Family::MemRefRelation,
                  "identity subview commit exhausted its work policy", module,
                  module, work);
  }

  // Candidate splicing can leave pure tensor/bufferization adapter closures
  // whose only consumer was folded above. Remove them explicitly so physical
  // planners do not mistake dead adapters for storage roots. Each iteration
  // erases at least one operation and is bounded by the initial op count.
  for (uint64_t iteration = 0; iteration < work.initialOperationCount;
       ++iteration) {
    mlir::Operation *dead = nullptr;
    for (mlir::func::FuncOp function : closure.functions) {
      function.walk<mlir::WalkOrder::PostOrder>([&](mlir::Operation *op) {
        if (!dead && mlir::isOpTriviallyDead(op))
          dead = op;
      });
      if (dead)
        break;
    }
    if (!dead)
      break;
    if (!add(work.operationVisits, 1) || !add(work.rewriteAttempts, 1) ||
        !add(work.committedRewrites, 1) || !consume(work, 10))
      return fail(Status::ResourceExhausted, Family::ProgramEnvelope,
                  "dead adapter cleanup exhausted its work policy", dead,
                  module, work);
    dead->erase();
  }
  Outcome outcome;
  outcome.status = Status::Success;
  outcome.changed = work.committedRewrites != 0;
  outcome.work = work;
  return outcome;
}

class SelectedPayloadNormalizationPass final
    : public mlir::PassWrapper<SelectedPayloadNormalizationPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SelectedPayloadNormalizationPass)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
  }

  void runOnOperation() override {
    Outcome outcome = normalizeSelectedPhysicalPayload(getOperation());
    if (outcome.status == Status::Success)
      return;
    auto diagnostic =
        getOperation().emitError("selected payload normalization failed");
    if (outcome.diagnostic)
      diagnostic << " at " << outcome.diagnostic->canonicalOperationPath << ": "
                 << outcome.diagnostic->reason;
    signalPassFailure();
  }
};

} // namespace

static SelectedPayloadNormalizationOutcome
verifySelectedPhysicalPayloadNormalFormImpl(mlir::ModuleOp module,
                                            bool requireLifetimeClosure) {
  Work work;
  if (mlir::failed(mlir::verify(module)))
    return fail(Status::InvalidIR, Family::ProgramEnvelope,
                "input module failed registered IR verification", module,
                module, work);
  SchedulableCallClosure closure = analyzeAllSchedulableCallClosures(module);
  if (closure.status != SchedulableCallClosureStatus::Success) {
    Status status = closure.status == SchedulableCallClosureStatus::InvalidIR
                        ? Status::InvalidIR
                        : Status::UnsupportedSemantic;
    return fail(status, Family::ProgramEnvelope,
                describeSchedulableCallClosureReason(*closure.reason),
                closure.diagnosticOperation ? closure.diagnosticOperation
                                            : module.getOperation(),
                module, work);
  }
  if (!initialize(closure.functions, work))
    return fail(Status::ResourceExhausted, Family::ProgramEnvelope,
                "selected payload work-policy overflow", module, module, work);

  Outcome failure;
  bool failed = false;
  auto verifyOperation = [&](mlir::Operation *operation) {
    if (failed)
      return mlir::WalkResult::interrupt();
    uint64_t visitFuel = 1;
    if (!add(work.operationVisits, 1) ||
        !add(work.regionVisits, operation->getNumRegions()) ||
        !add(visitFuel, operation->getNumRegions()) ||
        !consume(work, visitFuel)) {
      failure = fail(Status::ResourceExhausted, Family::ProgramEnvelope,
                     "selected payload verifier exhausted its work policy",
                     operation, module, work);
      failed = true;
      return mlir::WalkResult::interrupt();
    }
    if (auto subview = mlir::dyn_cast<mlir::memref::SubViewOp>(operation)) {
      IdentityViewProof proof = proveIdentityView(subview);
      uint64_t proofFuel = 0;
      if (!add(work.proofNodes, proof.inspectedDimensions) ||
          !multiply(2, proof.inspectedDimensions, proofFuel) ||
          !consume(work, proofFuel)) {
        failure = fail(Status::ResourceExhausted, Family::MemRefRelation,
                       "identity subview proof exhausted its work policy",
                       operation, module, work);
        failed = true;
        return mlir::WalkResult::interrupt();
      }
      if (proof.status == IdentityProofStatus::ProvenIdentity) {
        failure = fail(Status::InvalidIR, Family::MemRefRelation,
                       "identity full memref subview remains", operation,
                       module, work);
        failed = true;
        return mlir::WalkResult::interrupt();
      }
    }
    if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(operation)) {
      StaticTripCountProof proof = proveStaticTripCount(loop);
      if (proof.tripCount && (*proof.tripCount == 0 || *proof.tripCount == 1)) {
        failure = fail(Status::InvalidIR, Family::Traversal,
                       "static zero/one-trip traversal remains", operation,
                       module, work);
        failed = true;
        return mlir::WalkResult::interrupt();
      }
    }
    auto checkTrackedType = [&](mlir::Type type) {
      if (!isWaferSPMMemRefType(type) && !isWaferDDRMemRefType(type))
        return true;
      auto memref = mlir::dyn_cast<mlir::MemRefType>(type);
      if (!memref || !memref.hasStaticShape() ||
          !computeWaferPhysicalTensorInfo(memref)) {
        failure =
            fail(Status::UnsupportedSemantic, Family::AllocationRootLayout,
                 "tracked memref has unsupported dynamic shape, memory space, "
                 "or physical layout",
                 operation, module, work);
        failed = true;
        return false;
      }
      return true;
    };
    if (!llvm::all_of(operation->getOperandTypes(), checkTrackedType) ||
        !llvm::all_of(operation->getResultTypes(), checkTrackedType))
      return mlir::WalkResult::interrupt();
    if (auto allocation = mlir::dyn_cast<mlir::memref::AllocOp>(operation);
        allocation &&
        (isWaferSPMMemRefType(allocation.getType()) ||
         isWaferDDRMemRefType(allocation.getType())) &&
        (!allocation.getDynamicSizes().empty() ||
         !allocation.getSymbolOperands().empty())) {
      failure = fail(Status::UnsupportedSemantic, Family::AllocationRootLayout,
                     "tracked allocation must have static sizes and symbols",
                     operation, module, work);
      failed = true;
      return mlir::WalkResult::interrupt();
    }
    auto standard = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
    auto detailed = mlir::dyn_cast<WaferResourceEffectInterface>(operation);
    llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 8> standardEffects;
    if (standard)
      standard.getEffects(standardEffects);
    bool hasWaferStandardEffect =
        llvm::any_of(standardEffects, [](const auto &effect) {
          return isWaferStandardResource(effect.getResource());
        });
    if (hasWaferStandardEffect && !detailed) {
      failure =
          fail(Status::InvalidIR, Family::EffectCoverage,
               "standard Wafer memory effect lacks detailed resource coverage",
               operation, module, work);
      failed = true;
      return mlir::WalkResult::interrupt();
    }
    if (detailed) {
      if (!mlir::isa<mlir::MemoryEffectOpInterface>(operation)) {
        failure = fail(Status::InvalidIR, Family::EffectCoverage,
                       "detailed Wafer resource effect lacks standard memory "
                       "effect coverage",
                       operation, module, work);
        failed = true;
        return mlir::WalkResult::interrupt();
      }
      if (mlir::failed(detailed.verifyWaferResourceEffectContract())) {
        failure = fail(Status::InvalidIR, Family::EffectCoverage,
                       "detailed Wafer resource effect contract failed",
                       operation, module, work);
        failed = true;
        return mlir::WalkResult::interrupt();
      }
      llvm::SmallVector<WaferResourceEffect, 8> detailedEffects;
      detailed.collectWaferResourceEffects(detailedEffects);
      if (llvm::any_of(detailedEffects, [&](const WaferResourceEffect &effect) {
            return !standardEffectCovers(standardEffects, effect);
          })) {
        failure = fail(Status::InvalidIR, Family::EffectCoverage,
                       "detailed Wafer resource effect is not covered by the "
                       "standard memory effect interface",
                       operation, module, work);
        failed = true;
        return mlir::WalkResult::interrupt();
      }
    }
    return mlir::WalkResult::advance();
  };
  for (mlir::func::FuncOp function : closure.functions) {
    if (failed)
      break;
    function.walk<mlir::WalkOrder::PreOrder>(verifyOperation);
    if (!failed && requireLifetimeClosure)
      if (std::optional<Outcome> lifetime =
              verifyLifetimeClosure(function, module, work)) {
        failure = std::move(*lifetime);
        failed = true;
      }
  }
  if (failed)
    return failure;
  Outcome outcome;
  outcome.status = Status::Success;
  outcome.work = work;
  return outcome;
}

static SelectedPayloadNormalizationOutcome
normalizeSelectedPhysicalPayloadImpl(mlir::ModuleOp module,
                                     bool requireLifetimeClosure) {
  Work snapshotWork;
  CanonicalIRSnapshotV1 sourceGuard;
  std::string snapshotDiagnostic;
  if (!createCanonicalIRSnapshotV1(module,
                                   CanonicalIRSnapshotMode::MutationGuard,
                                   sourceGuard, &snapshotDiagnostic))
    return fail(Status::UnsupportedSemantic, Family::ProgramEnvelope,
                "input has no canonical mutation guard: " + snapshotDiagnostic,
                module, module, snapshotWork);
  auto returnUnchanged = [&](Outcome outcome) {
    CanonicalIRSnapshotV1 current;
    std::string diagnostic;
    if (!createCanonicalIRSnapshotV1(module,
                                     CanonicalIRSnapshotMode::MutationGuard,
                                     current, &diagnostic) ||
        current.structuralBytes != sourceGuard.structuralBytes)
      return fail(Status::InternalInvariant, Family::ProgramEnvelope,
                  "selected payload failure changed its source transaction",
                  module, module, outcome.work);
    return outcome;
  };
  if (mlir::failed(mlir::verify(module)))
    return returnUnchanged(
        fail(Status::InvalidIR, Family::ProgramEnvelope,
             "input module failed registered IR verification", module, module,
             snapshotWork));
  SchedulableCallClosure closure = analyzeAllSchedulableCallClosures(module);
  if (closure.status != SchedulableCallClosureStatus::Success) {
    Status status = closure.status == SchedulableCallClosureStatus::InvalidIR
                        ? Status::InvalidIR
                        : Status::UnsupportedSemantic;
    return returnUnchanged(
        fail(status, Family::ProgramEnvelope,
             describeSchedulableCallClosureReason(*closure.reason),
             closure.diagnosticOperation ? closure.diagnosticOperation
                                         : module.getOperation(),
             module, snapshotWork));
  }
  if (!initialize(closure.functions, snapshotWork))
    return returnUnchanged(fail(
        Status::ResourceExhausted, Family::ProgramEnvelope,
        "selected payload work-policy overflow", module, module, snapshotWork));
  CanonicalIRSnapshotV1 before;
  if (!createCanonicalIRSnapshotV1(module,
                                   CanonicalIRSnapshotMode::SemanticStructure,
                                   before, &snapshotDiagnostic))
    return returnUnchanged(fail(
        Status::UnsupportedSemantic, Family::ProgramEnvelope,
        "input has no canonical structural snapshot: " + snapshotDiagnostic,
        module, module, snapshotWork));
  mlir::OwningOpRef<mlir::ModuleOp> clone = module.clone();
  Outcome rewritten = rewrite(*clone);
  if (rewritten.status != Status::Success)
    return returnUnchanged(std::move(rewritten));
  Outcome verified = verifySelectedPhysicalPayloadNormalFormImpl(
      *clone, requireLifetimeClosure);
  if (verified.status != Status::Success) {
    Work verifierWork = verified.work;
    verified.work = rewritten.work;
    if (!merge(verified.work, verifierWork))
      return returnUnchanged(fail(Status::ResourceExhausted,
                                  Family::ProgramEnvelope,
                                  "selected payload work-summary overflow",
                                  module, module, rewritten.work));
    return returnUnchanged(std::move(verified));
  }
  if (!merge(rewritten.work, verified.work))
    return returnUnchanged(fail(Status::ResourceExhausted,
                                Family::ProgramEnvelope,
                                "selected payload work-summary overflow",
                                module, module, rewritten.work));
  CanonicalIRSnapshotV1 after;
  if (!createCanonicalIRSnapshotV1(*clone,
                                   CanonicalIRSnapshotMode::SemanticStructure,
                                   after, &snapshotDiagnostic))
    return returnUnchanged(
        fail(Status::InternalInvariant, Family::ProgramEnvelope,
             "normalized clone has no canonical structural snapshot: " +
                 snapshotDiagnostic,
             module, module, rewritten.work));
  rewritten.changed = before.structuralBytes != after.structuralBytes;
  if (rewritten.changed)
    module.getBodyRegion().takeBody(clone->getBodyRegion());
  else
    return returnUnchanged(std::move(rewritten));
  return rewritten;
}

SelectedPayloadNormalizationOutcome
verifySelectedPhysicalPayloadNormalForm(mlir::ModuleOp module) {
  return verifySelectedPhysicalPayloadNormalFormImpl(
      module, /*requireLifetimeClosure=*/true);
}

SelectedPayloadNormalizationOutcome
normalizeSelectedPhysicalPayload(mlir::ModuleOp module) {
  return normalizeSelectedPhysicalPayloadImpl(
      module, /*requireLifetimeClosure=*/true);
}

SelectedPayloadNormalizationOutcome structured_optimization::detail::
    normalizeSelectedPayloadStructureForPlanning(mlir::ModuleOp module) {
  return normalizeSelectedPhysicalPayloadImpl(
      module, /*requireLifetimeClosure=*/false);
}

std::unique_ptr<mlir::Pass> createSelectedPayloadNormalizationPass() {
  return std::make_unique<SelectedPayloadNormalizationPass>();
}

} // namespace wafer
