//===- RequiredTensorNormalization.cpp - Bounded required rewrites -------===//

#include "StructuredOptimizationInternal.h"

#include "Wafer/Analysis/IdentityViewProof.h"
#include "Wafer/Analysis/SchedulableCallClosure.h"
#include "Wafer/Support/CanonicalIRSnapshot.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <limits>

namespace wafer {
namespace {

static bool consumeRewriteAttempt(TensorNormalizationWorkSummary &work) {
  if (!structured_optimization::detail::addWorkCounter(work.rewriteAttempts, 1))
    return false;
  return structured_optimization::detail::consumeFuel(work, 2);
}

static bool consumeCommittedRewrite(TensorNormalizationWorkSummary &work,
                                    uint64_t newOperations = 0) {
  using namespace structured_optimization::detail;
  uint64_t operationFuel = 0;
  uint64_t totalFuel = 8;
  if (!addWorkCounter(work.committedRewrites, 1) ||
      !addWorkCounter(work.newOperations, newOperations) ||
      !multiplyWorkCounter(8, newOperations, operationFuel) ||
      !addWorkCounter(totalFuel, operationFuel))
    return false;
  return consumeFuel(work, totalFuel);
}

static TensorNormalizationOutcome rewriteRequiredForm(mlir::ModuleOp module) {
  using namespace structured_optimization::detail;
  TensorNormalizationWorkSummary work;
  SchedulableCallClosure closure = analyzeAllSchedulableCallClosures(module);
  if (closure.status != SchedulableCallClosureStatus::Success) {
    TensorNormalizationStatus status =
        closure.status == SchedulableCallClosureStatus::InvalidIR
            ? TensorNormalizationStatus::InvalidIR
            : TensorNormalizationStatus::UnsupportedSemantic;
    return makeFailure(status, TensorNormalizationFamily::ProgramEnvelope,
                       describeSchedulableCallClosureReason(*closure.reason),
                       closure.diagnosticOperation ? closure.diagnosticOperation
                                                   : module.getOperation(),
                       module, work);
  }
  if (!initializeWorkSummary(closure.functions, work))
    return makeFailure(TensorNormalizationStatus::ResourceExhausted,
                       TensorNormalizationFamily::ProgramEnvelope,
                       "required normalization work-policy overflow",
                       module.getOperation(), module, work);

  mlir::IRRewriter rewriter(module.getContext());
  llvm::DenseMap<mlir::Block *, llvm::SmallVector<mlir::Value, 4>>
      linalgIndicesByBlock;
  llvm::SmallVector<mlir::Operation *, 16> operations;
  for (mlir::func::FuncOp function : closure.functions)
    function.walk<mlir::WalkOrder::PostOrder>(
        [&](mlir::Operation *operation) { operations.push_back(operation); });

  for (mlir::Operation *operation : operations) {
    if (!addWorkCounter(work.operationVisits, 1))
      return makeFailure(TensorNormalizationStatus::ResourceExhausted,
                         TensorNormalizationFamily::ProgramEnvelope,
                         "required normalization operation counter overflow",
                         operation, module, work);
    if (!consumeFuel(work, 1))
      return makeFailure(TensorNormalizationStatus::ResourceExhausted,
                         TensorNormalizationFamily::ProgramEnvelope,
                         "required normalization exhausted its work policy",
                         operation, module, work);

    if (auto index = mlir::dyn_cast_or_null<mlir::linalg::IndexOp>(operation)) {
      llvm::SmallVector<mlir::Value, 4> &known =
          linalgIndicesByBlock[index->getBlock()];
      uint64_t dimension = index.getDim();
      if (dimension >= known.size())
        known.resize(dimension + 1);
      if (!known[dimension]) {
        known[dimension] = index.getResult();
      } else {
        if (!consumeRewriteAttempt(work))
          return makeFailure(
              TensorNormalizationStatus::ResourceExhausted,
              TensorNormalizationFamily::ScalarRegion,
              "duplicate linalg index rewrite exhausted its work policy",
              operation, module, work);
        rewriter.replaceOp(index, known[dimension]);
        if (!consumeCommittedRewrite(work))
          return makeFailure(
              TensorNormalizationStatus::ResourceExhausted,
              TensorNormalizationFamily::ScalarRegion,
              "duplicate linalg index commit exhausted its work policy",
              module.getOperation(), module, work);
        continue;
      }
    }

    if (auto dim = mlir::dyn_cast_or_null<mlir::tensor::DimOp>(operation)) {
      auto type =
          mlir::dyn_cast<mlir::RankedTensorType>(dim.getSource().getType());
      std::optional<int64_t> dimension =
          mlir::getConstantIntValue(dim.getIndex());
      if (type && type.hasStaticShape() && dimension && *dimension >= 0 &&
          *dimension < type.getRank()) {
        if (!consumeRewriteAttempt(work))
          return makeFailure(
              TensorNormalizationStatus::ResourceExhausted,
              TensorNormalizationFamily::ScalarRegion,
              "static tensor dimension rewrite exhausted its work policy",
              operation, module, work);
        mlir::Operation *indexDefinition = dim.getIndex().getDefiningOp();
        mlir::Operation *constantAnchor = dim->getParentOp();
        while (constantAnchor &&
               !mlir::isa<mlir::linalg::LinalgOp>(constantAnchor))
          constantAnchor = constantAnchor->getParentOp();
        rewriter.setInsertionPoint(constantAnchor ? constantAnchor : operation);
        mlir::Value constant = rewriter.create<mlir::arith::ConstantIndexOp>(
            dim.getLoc(), type.getDimSize(*dimension));
        rewriter.replaceOp(dim, constant);
        if (indexDefinition && indexDefinition->use_empty() &&
            mlir::isa<mlir::arith::ConstantOp>(indexDefinition))
          rewriter.eraseOp(indexDefinition);
        if (!consumeCommittedRewrite(work, /*newOperations=*/1))
          return makeFailure(
              TensorNormalizationStatus::ResourceExhausted,
              TensorNormalizationFamily::ScalarRegion,
              "static tensor dimension commit exhausted its work policy",
              module.getOperation(), module, work);
        continue;
      }
    }

    if (auto add = mlir::dyn_cast_or_null<mlir::arith::AddIOp>(operation);
        add && mlir::isa<mlir::IndexType>(add.getType())) {
      std::optional<int64_t> lhs = mlir::getConstantIntValue(add.getLhs());
      std::optional<int64_t> rhs = mlir::getConstantIntValue(add.getRhs());
      mlir::Value replacement;
      if (lhs && *lhs == 0)
        replacement = add.getRhs();
      else if (rhs && *rhs == 0)
        replacement = add.getLhs();
      if (replacement) {
        if (!consumeRewriteAttempt(work))
          return makeFailure(
              TensorNormalizationStatus::ResourceExhausted,
              TensorNormalizationFamily::ScalarRegion,
              "zero index addition rewrite exhausted its work policy",
              operation, module, work);
        rewriter.replaceOp(add, replacement);
        if (!consumeCommittedRewrite(work))
          return makeFailure(
              TensorNormalizationStatus::ResourceExhausted,
              TensorNormalizationFamily::ScalarRegion,
              "zero index addition commit exhausted its work policy",
              module.getOperation(), module, work);
        continue;
      }
    }

    if (auto sub = mlir::dyn_cast_or_null<mlir::arith::SubIOp>(operation);
        sub && mlir::isa<mlir::IndexType>(sub.getType())) {
      std::optional<int64_t> rhs = mlir::getConstantIntValue(sub.getRhs());
      if (rhs && *rhs == 0) {
        if (!consumeRewriteAttempt(work))
          return makeFailure(
              TensorNormalizationStatus::ResourceExhausted,
              TensorNormalizationFamily::ScalarRegion,
              "zero index subtraction rewrite exhausted its work policy",
              operation, module, work);
        rewriter.replaceOp(sub, sub.getLhs());
        if (!consumeCommittedRewrite(work))
          return makeFailure(
              TensorNormalizationStatus::ResourceExhausted,
              TensorNormalizationFamily::ScalarRegion,
              "zero index subtraction commit exhausted its work policy",
              module.getOperation(), module, work);
        continue;
      }
    }

    if (auto extract =
            mlir::dyn_cast_or_null<mlir::tensor::ExtractOp>(operation)) {
      auto fromElements =
          extract.getTensor().getDefiningOp<mlir::tensor::FromElementsOp>();
      if (fromElements && extract.getIndices().empty() &&
          fromElements.getElements().size() == 1) {
        if (!consumeRewriteAttempt(work))
          return makeFailure(
              TensorNormalizationStatus::ResourceExhausted,
              TensorNormalizationFamily::ScalarRegion,
              "scalar tensor roundtrip rewrite exhausted its work policy",
              operation, module, work);
        rewriter.replaceOp(extract, fromElements.getElements().front());
        if (fromElements->use_empty())
          rewriter.eraseOp(fromElements);
        if (!consumeCommittedRewrite(work))
          return makeFailure(
              TensorNormalizationStatus::ResourceExhausted,
              TensorNormalizationFamily::ScalarRegion,
              "scalar tensor roundtrip commit exhausted its work policy",
              module.getOperation(), module, work);
        continue;
      }
    }

    if (auto slice =
            mlir::dyn_cast_or_null<mlir::tensor::ExtractSliceOp>(operation)) {
      if (!consumeRewriteAttempt(work))
        return makeFailure(
            TensorNormalizationStatus::ResourceExhausted,
            TensorNormalizationFamily::TensorRelation,
            "tensor identity-view rewrite exhausted its work policy", operation,
            module, work);
      IdentityViewProof proof = proveIdentityView(slice);
      if (proof.inspectedDimensions >
          std::numeric_limits<uint64_t>::max() - work.relationNodes)
        return makeFailure(TensorNormalizationStatus::ResourceExhausted,
                           TensorNormalizationFamily::TensorRelation,
                           "tensor identity-view relation counter overflow",
                           operation, module, work);
      work.relationNodes += proof.inspectedDimensions;
      uint64_t proofFuel = 0;
      if (!multiplyWorkCounter(2, proof.inspectedDimensions, proofFuel) ||
          !consumeFuel(work, proofFuel))
        return makeFailure(
            TensorNormalizationStatus::ResourceExhausted,
            TensorNormalizationFamily::TensorRelation,
            "tensor identity-view proof exhausted its work policy", operation,
            module, work);
      if (proof.status != IdentityProofStatus::ProvenIdentity)
        continue;

      mlir::Value replacement = slice.getSource();
      uint64_t newOperations = 0;
      if (replacement.getType() != slice.getType()) {
        if (!mlir::tensor::CastOp::areCastCompatible(replacement.getType(),
                                                     slice.getType()))
          return makeFailure(
              TensorNormalizationStatus::UnsupportedSemantic,
              TensorNormalizationFamily::TensorRelation,
              "identity tensor slice requires a lossy result type change",
              operation, module, work);
        rewriter.setInsertionPoint(slice);
        replacement = rewriter
                          .create<mlir::tensor::CastOp>(
                              slice.getLoc(), slice.getType(), replacement)
                          .getResult();
        newOperations = 1;
      }
      rewriter.replaceOp(slice, replacement);
      if (!consumeCommittedRewrite(work, newOperations))
        return makeFailure(
            TensorNormalizationStatus::ResourceExhausted,
            TensorNormalizationFamily::TensorRelation,
            "tensor identity-view commit exhausted its work policy",
            module.getOperation(), module, work);
      continue;
    }

    if (auto loop = mlir::dyn_cast_or_null<mlir::scf::ForOp>(operation)) {
      if (!consumeRewriteAttempt(work))
        return makeFailure(
            TensorNormalizationStatus::ResourceExhausted,
            TensorNormalizationFamily::StructuredControl,
            "structured control rewrite exhausted its work policy", operation,
            module, work);
      StaticTripCountProof proof = proveStaticTripCount(loop);
      if (!proof.tripCount || (*proof.tripCount != 0 && *proof.tripCount != 1))
        continue;
      rewriter.setInsertionPoint(loop);
      if (*proof.tripCount == 0) {
        rewriter.replaceOp(loop, loop.getInitArgs());
      } else if (mlir::failed(loop.promoteIfSingleIteration(rewriter))) {
        return makeFailure(TensorNormalizationStatus::InternalInvariant,
                           TensorNormalizationFamily::StructuredControl,
                           "proven one-trip loop could not be promoted",
                           operation, module, work);
      }
      if (!consumeCommittedRewrite(work))
        return makeFailure(
            TensorNormalizationStatus::ResourceExhausted,
            TensorNormalizationFamily::StructuredControl,
            "structured control commit exhausted its work policy",
            module.getOperation(), module, work);
    }
  }

  TensorNormalizationOutcome outcome;
  outcome.status = TensorNormalizationStatus::Success;
  outcome.changed = work.committedRewrites != 0;
  outcome.work = work;
  return outcome;
}

class RequiredTensorNormalizationPass final
    : public mlir::PassWrapper<RequiredTensorNormalizationPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RequiredTensorNormalizationPass)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
  }

  void runOnOperation() override {
    TensorNormalizationOutcome outcome =
        normalizeRequiredTensorModule(getOperation());
    if (outcome.status == TensorNormalizationStatus::Success)
      return;
    auto diagnostic =
        getOperation().emitError("required tensor normalization failed");
    if (outcome.diagnostic)
      diagnostic << " at " << outcome.diagnostic->canonicalOperationPath << ": "
                 << outcome.diagnostic->reason;
    signalPassFailure();
  }
};

} // namespace

TensorNormalizationOutcome
normalizeRequiredTensorModule(mlir::ModuleOp module) {
  TensorNormalizationWorkSummary snapshotWork;
  CanonicalIRSnapshotV1 sourceGuard;
  std::string snapshotDiagnostic;
  if (!createCanonicalIRSnapshotV1(module,
                                   CanonicalIRSnapshotMode::MutationGuard,
                                   sourceGuard, &snapshotDiagnostic))
    return structured_optimization::detail::makeFailure(
        TensorNormalizationStatus::UnsupportedSemantic,
        TensorNormalizationFamily::ProgramEnvelope,
        "input has no canonical mutation guard: " + snapshotDiagnostic,
        module.getOperation(), module, snapshotWork);
  auto returnUnchanged = [&](TensorNormalizationOutcome outcome) {
    CanonicalIRSnapshotV1 current;
    std::string diagnostic;
    if (!createCanonicalIRSnapshotV1(module,
                                     CanonicalIRSnapshotMode::MutationGuard,
                                     current, &diagnostic) ||
        current.structuralBytes != sourceGuard.structuralBytes) {
      return structured_optimization::detail::makeFailure(
          TensorNormalizationStatus::InternalInvariant,
          TensorNormalizationFamily::ProgramEnvelope,
          "required normalization failure changed its source transaction",
          module.getOperation(), module, outcome.work);
    }
    return outcome;
  };
  if (mlir::failed(mlir::verify(module)))
    return returnUnchanged(structured_optimization::detail::makeFailure(
        TensorNormalizationStatus::InvalidIR,
        TensorNormalizationFamily::ProgramEnvelope,
        "input module failed registered IR verification", module.getOperation(),
        module, snapshotWork));
  SchedulableCallClosure closure = analyzeAllSchedulableCallClosures(module);
  if (closure.status != SchedulableCallClosureStatus::Success) {
    TensorNormalizationStatus status =
        closure.status == SchedulableCallClosureStatus::InvalidIR
            ? TensorNormalizationStatus::InvalidIR
            : TensorNormalizationStatus::UnsupportedSemantic;
    return returnUnchanged(structured_optimization::detail::makeFailure(
        status, TensorNormalizationFamily::ProgramEnvelope,
        describeSchedulableCallClosureReason(*closure.reason),
        closure.diagnosticOperation ? closure.diagnosticOperation
                                    : module.getOperation(),
        module, snapshotWork));
  }
  if (!structured_optimization::detail::initializeWorkSummary(closure.functions,
                                                              snapshotWork))
    return returnUnchanged(structured_optimization::detail::makeFailure(
        TensorNormalizationStatus::ResourceExhausted,
        TensorNormalizationFamily::ProgramEnvelope,
        "required normalization work-policy overflow", module.getOperation(),
        module, snapshotWork));
  CanonicalIRSnapshotV1 before;
  if (!createCanonicalIRSnapshotV1(module,
                                   CanonicalIRSnapshotMode::SemanticStructure,
                                   before, &snapshotDiagnostic))
    return returnUnchanged(structured_optimization::detail::makeFailure(
        TensorNormalizationStatus::UnsupportedSemantic,
        TensorNormalizationFamily::ProgramEnvelope,
        "input has no canonical structural snapshot: " + snapshotDiagnostic,
        module.getOperation(), module, snapshotWork));
  mlir::OwningOpRef<mlir::ModuleOp> clone = module.clone();
  TensorNormalizationOutcome rewrite = rewriteRequiredForm(*clone);
  if (rewrite.status != TensorNormalizationStatus::Success)
    return returnUnchanged(std::move(rewrite));

  TensorNormalizationOutcome verification =
      verifyRequiredTensorNormalForm(*clone);
  if (verification.status != TensorNormalizationStatus::Success) {
    TensorNormalizationWorkSummary verifierWork = verification.work;
    verification.work = rewrite.work;
    if (!structured_optimization::detail::mergeWork(verification.work,
                                                    verifierWork)) {
      return returnUnchanged(structured_optimization::detail::makeFailure(
          TensorNormalizationStatus::ResourceExhausted,
          TensorNormalizationFamily::ProgramEnvelope,
          "required normalization work-summary overflow", module.getOperation(),
          module, rewrite.work));
    }
    return returnUnchanged(std::move(verification));
  }
  if (!structured_optimization::detail::mergeWork(rewrite.work,
                                                  verification.work)) {
    return returnUnchanged(structured_optimization::detail::makeFailure(
        TensorNormalizationStatus::ResourceExhausted,
        TensorNormalizationFamily::ProgramEnvelope,
        "required normalization work-summary overflow", module.getOperation(),
        module, rewrite.work));
  }

  CanonicalIRSnapshotV1 after;
  if (!createCanonicalIRSnapshotV1(*clone,
                                   CanonicalIRSnapshotMode::SemanticStructure,
                                   after, &snapshotDiagnostic))
    return returnUnchanged(structured_optimization::detail::makeFailure(
        TensorNormalizationStatus::InternalInvariant,
        TensorNormalizationFamily::ProgramEnvelope,
        "normalized clone has no canonical structural snapshot: " +
            snapshotDiagnostic,
        module.getOperation(), module, rewrite.work));
  rewrite.changed = before.structuralBytes != after.structuralBytes;

  // A verified no-op must not replace the body with clone-owned operations:
  // OperationFingerPrint intentionally includes object identity and would
  // make such a replacement observable to the gateway even though the
  // semantic structure is unchanged.
  if (rewrite.changed)
    module.getBodyRegion().takeBody(clone->getBodyRegion());
  else
    return returnUnchanged(std::move(rewrite));
  return rewrite;
}

std::unique_ptr<mlir::Pass> createRequiredTensorNormalizationPass() {
  return std::make_unique<RequiredTensorNormalizationPass>();
}

} // namespace wafer
