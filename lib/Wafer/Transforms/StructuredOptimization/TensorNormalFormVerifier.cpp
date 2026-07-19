//===- TensorNormalFormVerifier.cpp - Structured tensor postcondition ----===//

#include "StructuredOptimizationInternal.h"

#include "Wafer/Analysis/DPSInitAnalysis.h"
#include "Wafer/Analysis/IdentityViewProof.h"
#include "Wafer/Analysis/SchedulableCallClosure.h"
#include "Wafer/IR/WaferInterfaces.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <limits>

namespace wafer::structured_optimization::detail {
namespace {

static bool addChecked(uint64_t &target, uint64_t value) {
  if (value > std::numeric_limits<uint64_t>::max() - target)
    return false;
  target += value;
  return true;
}

static bool multiplyChecked(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

static uint64_t getOrdinal(mlir::Operation *operation) {
  uint64_t ordinal = 0;
  for (mlir::Operation &candidate : *operation->getBlock()) {
    if (&candidate == operation)
      return ordinal;
    ++ordinal;
  }
  return ordinal;
}

static uint64_t getBlockOrdinal(mlir::Block *block) {
  uint64_t ordinal = 0;
  for (mlir::Block &candidate : *block->getParent()) {
    if (&candidate == block)
      return ordinal;
    ++ordinal;
  }
  return ordinal;
}

static uint64_t getRegionOrdinal(mlir::Region *region) {
  mlir::Operation *parent = region->getParentOp();
  for (uint64_t ordinal = 0; ordinal < parent->getNumRegions(); ++ordinal)
    if (&parent->getRegion(ordinal) == region)
      return ordinal;
  return parent->getNumRegions();
}

static bool isSupportedTensorType(mlir::Type type) {
  if (!mlir::isa<mlir::TensorType>(type))
    return true;
  auto ranked = mlir::dyn_cast<mlir::RankedTensorType>(type);
  return ranked && ranked.hasStaticShape();
}

} // namespace

bool addWorkCounter(uint64_t &target, uint64_t value) {
  return addChecked(target, value);
}

bool multiplyWorkCounter(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  return multiplyChecked(lhs, rhs, result);
}

bool initializeWorkSummary(
    llvm::ArrayRef<mlir::func::FuncOp> schedulableFunctions,
    TensorNormalizationWorkSummary &work) {
  bool valid = true;
  for (mlir::func::FuncOp function : schedulableFunctions)
    function.walk([&](mlir::Operation *operation) {
      valid &= addWorkCounter(work.initialOperationCount, 1);
      valid &=
          addWorkCounter(work.initialEdgeCount,
                         static_cast<uint64_t>(operation->getNumOperands()));
      for (mlir::Block *successor : operation->getSuccessors())
        valid &=
            addWorkCounter(work.initialEdgeCount,
                           static_cast<uint64_t>(successor->getNumArguments()));
      for (mlir::Type type : operation->getResultTypes())
        if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(type);
            shaped && shaped.hasRank())
          valid &= addWorkCounter(work.initialDimensionCount,
                                  static_cast<uint64_t>(shaped.getRank()));
    });
  if (!valid)
    return false;

  uint64_t opFuel = 0;
  uint64_t edgeFuel = 0;
  uint64_t dimensionFuel = 0;
  if (!multiplyWorkCounter(work.initialOperationCount, 32, opFuel) ||
      !multiplyWorkCounter(work.initialEdgeCount, 8, edgeFuel) ||
      !multiplyWorkCounter(work.initialDimensionCount, 8, dimensionFuel))
    return false;
  work.fuelLimit = 1024;
  return addWorkCounter(work.fuelLimit, opFuel) &&
         addWorkCounter(work.fuelLimit, edgeFuel) &&
         addWorkCounter(work.fuelLimit, dimensionFuel);
}

bool initializeWorkSummary(mlir::ModuleOp module,
                           TensorNormalizationWorkSummary &work) {
  SchedulableCallClosure closure = analyzeAllSchedulableCallClosures(module);
  return closure.status == SchedulableCallClosureStatus::Success &&
         initializeWorkSummary(closure.functions, work);
}

bool consumeFuel(TensorNormalizationWorkSummary &work, uint64_t amount) {
  if (work.fuelConsumed > work.fuelLimit ||
      amount > work.fuelLimit - work.fuelConsumed)
    return false;
  work.fuelConsumed += amount;
  return true;
}

bool mergeWork(TensorNormalizationWorkSummary &destination,
               const TensorNormalizationWorkSummary &source) {
  TensorNormalizationWorkSummary merged = destination;
  if (!addWorkCounter(merged.fuelConsumed, source.fuelConsumed) ||
      !addWorkCounter(merged.operationVisits, source.operationVisits) ||
      !addWorkCounter(merged.rewriteAttempts, source.rewriteAttempts) ||
      !addWorkCounter(merged.committedRewrites, source.committedRewrites) ||
      !addWorkCounter(merged.newOperations, source.newOperations) ||
      !addWorkCounter(merged.relationNodes, source.relationNodes))
    return false;
  destination = merged;
  return true;
}

std::string getCanonicalOperationPath(mlir::Operation *operation,
                                      mlir::ModuleOp root) {
  llvm::SmallVector<std::string, 8> components;
  for (mlir::Operation *current = operation;
       current && current != root.getOperation();
       current = current->getParentOp()) {
    mlir::Block *block = current->getBlock();
    mlir::Region *region = block ? block->getParent() : nullptr;
    if (!block || !region)
      break;
    components.push_back("r" + std::to_string(getRegionOrdinal(region)) + ".b" +
                         std::to_string(getBlockOrdinal(block)) + ".o" +
                         std::to_string(getOrdinal(current)));
  }
  std::string path = "module";
  for (const std::string &component : llvm::reverse(components))
    path += "/" + component;
  return path;
}

TensorNormalizationOutcome
makeFailure(TensorNormalizationStatus status, TensorNormalizationFamily family,
            llvm::StringRef reason, mlir::Operation *operation,
            mlir::ModuleOp root, TensorNormalizationWorkSummary work) {
  TensorNormalizationOutcome outcome;
  outcome.status = status;
  outcome.changed = false;
  outcome.diagnostic = TensorNormalizationDiagnostic{
      family, reason.str(), getCanonicalOperationPath(operation, root)};
  outcome.work = work;
  return outcome;
}

} // namespace wafer::structured_optimization::detail

namespace wafer {

TensorNormalizationOutcome
verifyRequiredTensorNormalForm(mlir::ModuleOp module) {
  using namespace structured_optimization::detail;
  TensorNormalizationWorkSummary work;
  if (mlir::failed(mlir::verify(module)))
    return makeFailure(TensorNormalizationStatus::InvalidIR,
                       TensorNormalizationFamily::ProgramEnvelope,
                       "input module failed registered IR verification",
                       module.getOperation(), module, work);

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

  TensorNormalizationOutcome failure;
  bool failed = false;
  llvm::DenseMap<mlir::Block *, llvm::SmallVector<bool, 4>>
      linalgIndexDimensionsByBlock;
  auto verifyOperation = [&](mlir::Operation *operation) {
    if (failed)
      return mlir::WalkResult::interrupt();
    if (!addWorkCounter(work.operationVisits, 1)) {
      failure = makeFailure(TensorNormalizationStatus::ResourceExhausted,
                            TensorNormalizationFamily::ProgramEnvelope,
                            "required normalization operation counter overflow",
                            operation, module, work);
      failed = true;
      return mlir::WalkResult::interrupt();
    }
    if (!consumeFuel(work, 1)) {
      failure = makeFailure(
          TensorNormalizationStatus::ResourceExhausted,
          TensorNormalizationFamily::ProgramEnvelope,
          "required normalization verifier exhausted its work policy",
          operation, module, work);
      failed = true;
      return mlir::WalkResult::interrupt();
    }

    llvm::StringRef dialect = operation->getName().getDialectNamespace();
    if (dialect == "stablehlo" || dialect == "sdy" ||
        operation->getName().getStringRef() ==
            "builtin.unrealized_conversion_cast") {
      failure = makeFailure(
          TensorNormalizationStatus::UnsupportedSemantic,
          TensorNormalizationFamily::ProgramEnvelope,
          "schedulable path retains frontend or unrealized semantics",
          operation, module, work);
      failed = true;
      return mlir::WalkResult::interrupt();
    }

    for (mlir::Type type : operation->getOperandTypes()) {
      if (!isSupportedTensorType(type)) {
        failure = makeFailure(
            TensorNormalizationStatus::UnsupportedSemantic,
            TensorNormalizationFamily::ProgramEnvelope,
            "schedulable path contains a dynamic or unranked tensor", operation,
            module, work);
        failed = true;
        return mlir::WalkResult::interrupt();
      }
    }
    for (mlir::Type type : operation->getResultTypes()) {
      if (!isSupportedTensorType(type)) {
        failure = makeFailure(
            TensorNormalizationStatus::UnsupportedSemantic,
            TensorNormalizationFamily::ProgramEnvelope,
            "schedulable path contains a dynamic or unranked tensor", operation,
            module, work);
        failed = true;
        return mlir::WalkResult::interrupt();
      }
    }

    if (auto extract = mlir::dyn_cast<mlir::tensor::ExtractOp>(operation)) {
      auto fromElements =
          extract.getTensor().getDefiningOp<mlir::tensor::FromElementsOp>();
      if (fromElements && extract.getIndices().empty() &&
          fromElements.getElements().size() == 1) {
        failure = makeFailure(
            TensorNormalizationStatus::InvalidIR,
            TensorNormalizationFamily::ScalarRegion,
            "scalar tensor from_elements/extract roundtrip remains after "
            "required normalization",
            operation, module, work);
        failed = true;
        return mlir::WalkResult::interrupt();
      }
    }

    if (auto index = mlir::dyn_cast<mlir::linalg::IndexOp>(operation)) {
      llvm::SmallVector<bool, 4> &seen =
          linalgIndexDimensionsByBlock[index->getBlock()];
      uint64_t dimension = index.getDim();
      if (dimension >= seen.size())
        seen.resize(dimension + 1);
      if (seen[dimension]) {
        failure = makeFailure(
            TensorNormalizationStatus::InvalidIR,
            TensorNormalizationFamily::ScalarRegion,
            "duplicate linalg index remains after required normalization",
            operation, module, work);
        failed = true;
        return mlir::WalkResult::interrupt();
      }
      seen[dimension] = true;
    }

    if (auto dim = mlir::dyn_cast<mlir::tensor::DimOp>(operation)) {
      auto type =
          mlir::dyn_cast<mlir::RankedTensorType>(dim.getSource().getType());
      std::optional<int64_t> dimension =
          mlir::getConstantIntValue(dim.getIndex());
      if (type && type.hasStaticShape() && dimension && *dimension >= 0 &&
          *dimension < type.getRank()) {
        failure = makeFailure(
            TensorNormalizationStatus::InvalidIR,
            TensorNormalizationFamily::ScalarRegion,
            "static tensor dimension remains after required normalization",
            operation, module, work);
        failed = true;
        return mlir::WalkResult::interrupt();
      }
    }

    if (auto add = mlir::dyn_cast<mlir::arith::AddIOp>(operation);
        add && mlir::isa<mlir::IndexType>(add.getType())) {
      std::optional<int64_t> lhs = mlir::getConstantIntValue(add.getLhs());
      std::optional<int64_t> rhs = mlir::getConstantIntValue(add.getRhs());
      if ((lhs && *lhs == 0) || (rhs && *rhs == 0)) {
        failure = makeFailure(
            TensorNormalizationStatus::InvalidIR,
            TensorNormalizationFamily::ScalarRegion,
            "zero index addition remains after required normalization",
            operation, module, work);
        failed = true;
        return mlir::WalkResult::interrupt();
      }
    }
    if (auto sub = mlir::dyn_cast<mlir::arith::SubIOp>(operation);
        sub && mlir::isa<mlir::IndexType>(sub.getType())) {
      std::optional<int64_t> rhs = mlir::getConstantIntValue(sub.getRhs());
      if (rhs && *rhs == 0) {
        failure = makeFailure(
            TensorNormalizationStatus::InvalidIR,
            TensorNormalizationFamily::ScalarRegion,
            "zero index subtraction remains after required normalization",
            operation, module, work);
        failed = true;
        return mlir::WalkResult::interrupt();
      }
    }

    if (auto dps = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(operation);
        dps && llvm::any_of(operation->getResultTypes(), [](mlir::Type type) {
          return mlir::isa<mlir::TensorType>(type);
        })) {
      if (dps.getNumDpsInits() !=
          static_cast<int64_t>(operation->getNumResults())) {
        failure = makeFailure(
            TensorNormalizationStatus::InvalidIR,
            TensorNormalizationFamily::StructuredComputeDPS,
            "structured tensor operation has no exact DPS result/init tie",
            operation, module, work);
        failed = true;
        return mlir::WalkResult::interrupt();
      }
      for (auto [result, init] :
           llvm::zip(operation->getResults(), dps.getDpsInits())) {
        if (result.getType() != init.getType()) {
          failure = makeFailure(TensorNormalizationStatus::InvalidIR,
                                TensorNormalizationFamily::StructuredComputeDPS,
                                "DPS result and init types differ", operation,
                                module, work);
          failed = true;
          return mlir::WalkResult::interrupt();
        }
      }
    }

    if (auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(operation);
        linalg && linalg.hasPureTensorSemantics()) {
      auto dps = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(operation);
      if (!dps || dps.getNumDpsInits() !=
                      static_cast<int64_t>(operation->getNumResults())) {
        failure = makeFailure(
            TensorNormalizationStatus::InvalidIR,
            TensorNormalizationFamily::StructuredComputeDPS,
            "pure tensor structured compute has no exact DPS result/init tie",
            operation, module, work);
        failed = true;
        return mlir::WalkResult::interrupt();
      }
      for (auto [result, init] :
           llvm::zip(operation->getResults(), dps.getDpsInits())) {
        if (result.getType() != init.getType()) {
          failure = makeFailure(TensorNormalizationStatus::InvalidIR,
                                TensorNormalizationFamily::StructuredComputeDPS,
                                "DPS result and init types differ", operation,
                                module, work);
          failed = true;
          return mlir::WalkResult::interrupt();
        }
      }
      for (unsigned initIndex = 0; initIndex < linalg.getNumDpsInits();
           ++initIndex) {
        DPSInitFacts facts = analyzeDPSInit(linalg, initIndex);
        uint64_t proofFuel = 0;
        if (facts.inspectedNodes > std::numeric_limits<uint64_t>::max() / 2 ||
            facts.inspectedNodes >
                std::numeric_limits<uint64_t>::max() - work.relationNodes ||
            !multiplyWorkCounter(2, facts.inspectedNodes, proofFuel) ||
            !consumeFuel(work, proofFuel)) {
          failure = makeFailure(TensorNormalizationStatus::ResourceExhausted,
                                TensorNormalizationFamily::DPSInitReduction,
                                "DPS init proof exhausted its work policy",
                                operation, module, work);
          failed = true;
          return mlir::WalkResult::interrupt();
        }
        work.relationNodes += facts.inspectedNodes;
        if (facts.readState == InitReadState::Read &&
            facts.origin == InitOrigin::Undefined) {
          failure =
              makeFailure(TensorNormalizationStatus::InvalidIR,
                          TensorNormalizationFamily::DPSInitReduction,
                          "read DPS init has undefined tensor.empty origin",
                          operation, module, work);
          failed = true;
          return mlir::WalkResult::interrupt();
        }
      }
    }

    if (auto collective =
            mlir::dyn_cast<WaferLinalgExtCollectiveOpInterface>(operation)) {
      WaferLinalgExtCollectiveInfo info;
      collective.collectWaferLinalgExtCollectiveInfo(info);
      if (mlir::failed(collective.verifyWaferLinalgExtCollectiveContract()) ||
          !info.hasCommunicationEffect) {
        failure = makeFailure(
            TensorNormalizationStatus::InvalidIR,
            TensorNormalizationFamily::LogicalCollective,
            "logical collective lacks a verified communication barrier",
            operation, module, work);
        failed = true;
        return mlir::WalkResult::interrupt();
      }
    }

    if (auto slice = mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(operation)) {
      IdentityViewProof proof = proveIdentityView(slice);
      if (proof.inspectedDimensions >
          std::numeric_limits<uint64_t>::max() - work.relationNodes) {
        failure = makeFailure(TensorNormalizationStatus::ResourceExhausted,
                              TensorNormalizationFamily::TensorRelation,
                              "tensor identity-view relation counter overflow",
                              operation, module, work);
        failed = true;
        return mlir::WalkResult::interrupt();
      }
      work.relationNodes += proof.inspectedDimensions;
      uint64_t proofFuel = 0;
      if (!multiplyWorkCounter(2, proof.inspectedDimensions, proofFuel) ||
          !consumeFuel(work, proofFuel)) {
        failure =
            makeFailure(TensorNormalizationStatus::ResourceExhausted,
                        TensorNormalizationFamily::TensorRelation,
                        "tensor identity-view proof exhausted its work policy",
                        operation, module, work);
        failed = true;
        return mlir::WalkResult::interrupt();
      }
      if (proof.status == IdentityProofStatus::ProvenIdentity) {
        failure = makeFailure(
            TensorNormalizationStatus::InvalidIR,
            TensorNormalizationFamily::TensorRelation,
            "identity full tensor slice remains after required normalization",
            operation, module, work);
        failed = true;
        return mlir::WalkResult::interrupt();
      }
    }

    if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(operation)) {
      StaticTripCountProof proof = proveStaticTripCount(loop);
      if (proof.tripCount && (*proof.tripCount == 0 || *proof.tripCount == 1)) {
        failure = makeFailure(
            TensorNormalizationStatus::InvalidIR,
            TensorNormalizationFamily::StructuredControl,
            "static zero/one-trip loop remains after required normalization",
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
  }

  if (failed)
    return failure;
  TensorNormalizationOutcome outcome;
  outcome.status = TensorNormalizationStatus::Success;
  outcome.changed = false;
  outcome.work = work;
  return outcome;
}

} // namespace wafer
