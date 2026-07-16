//===- ScheduleTensorProgram.cpp - Closed-loop task scheduling ------------===//

#include "Scheduling/ScheduleTensorProgramInternal.h"
#include "Wafer/Transforms/Passes.h"
#include "Wafer/Transforms/TensorProgramScheduling.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace wafer {
#define GEN_PASS_DEF_SCHEDULETENSORPROGRAMPASS
#include "Wafer/Transforms/WaferPasses.h.inc"

using namespace tensor_program_scheduling;

namespace {

static bool isValueOwnedBy(mlir::Value value, mlir::Operation *root) {
  mlir::Operation *owner = value.getDefiningOp();
  if (!owner) {
    auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
    owner = argument ? argument.getOwner()->getParentOp() : nullptr;
  }
  return owner && (owner == root || root->isAncestor(owner));
}

/// Check the structural invariants that are cheap to diagnose without asking
/// the generic verifier to print a malformed operation.  In particular, a
/// failed atomic splice must never leave an operand referring into the private
/// candidate module.
static std::optional<std::string>
getCommittedTaskStructuralFailure(mlir::ModuleOp module) {
  std::optional<std::string> failure;
  module.walk([&](mlir::Operation *operation) {
    if (failure)
      return;
    for (mlir::Value operand : operation->getOperands()) {
      if (!isValueOwnedBy(operand, module.getOperation())) {
        failure = "committed operation retains a value from outside the "
                  "staged rank module";
        return;
      }
    }
    if (auto toMemref =
            mlir::dyn_cast<mlir::bufferization::ToMemrefOp>(operation)) {
      auto tensorType =
          mlir::dyn_cast<mlir::TensorType>(toMemref.getTensor().getType());
      auto memrefType =
          mlir::dyn_cast<mlir::BaseMemRefType>(toMemref.getMemref().getType());
      if (!tensorType || !memrefType ||
          tensorType.getShape() != memrefType.getShape() ||
          tensorType.getElementType() != memrefType.getElementType())
        failure = "committed bufferization.to_memref boundary has "
                  "incompatible tensor and memref types";
    }
  });
  return failure;
}

struct BoundaryAliasProof {
  llvm::DenseMap<mlir::Operation *, mlir::Value> &emptyAllocations;
  llvm::DenseSet<mlir::Value> activeTensors;
  llvm::DenseSet<mlir::Value> activeMemrefs;
  llvm::DenseSet<mlir::Value> assumedTensors;
};

static bool isTensorBackedBy(mlir::Value tensor, mlir::Value expectedMemref,
                             BoundaryAliasProof &proof);

static bool isMemrefBackedBy(mlir::Value memref, mlir::Value expectedMemref,
                             BoundaryAliasProof &proof);

static bool isValueBackedBy(mlir::Value value, mlir::Value expectedMemref,
                            BoundaryAliasProof &proof) {
  if (mlir::isa<mlir::TensorType>(value.getType()))
    return isTensorBackedBy(value, expectedMemref, proof);
  if (mlir::isa<mlir::BaseMemRefType>(value.getType()))
    return isMemrefBackedBy(value, expectedMemref, proof);
  return false;
}

static bool isForTensorRecurrenceBackedBy(mlir::scf::ForOp forOp,
                                          unsigned index,
                                          mlir::Value expectedMemref,
                                          BoundaryAliasProof &proof) {
  auto yield =
      mlir::dyn_cast<mlir::scf::YieldOp>(forOp.getBody()->getTerminator());
  if (!yield || index >= forOp.getInitArgs().size() ||
      index >= forOp.getRegionIterArgs().size() ||
      index >= yield.getResults().size())
    return false;
  mlir::Value iterArg = forOp.getRegionIterArgs()[index];
  if (!proof.assumedTensors.insert(iterArg).second)
    return false;
  bool preservesStorage =
      isTensorBackedBy(forOp.getInitArgs()[index], expectedMemref, proof) &&
      isTensorBackedBy(yield.getResults()[index], expectedMemref, proof);
  proof.assumedTensors.erase(iterArg);
  return preservesStorage;
}

static bool isTensorBackedBy(mlir::Value tensor, mlir::Value expectedMemref,
                             BoundaryAliasProof &proof) {
  if (proof.assumedTensors.contains(tensor))
    return true;
  if (!proof.activeTensors.insert(tensor).second)
    return false;
  auto finish = [&](bool result) {
    proof.activeTensors.erase(tensor);
    return result;
  };

  if (auto toTensor = tensor.getDefiningOp<mlir::bufferization::ToTensorOp>())
    return finish(
        isMemrefBackedBy(toTensor.getMemref(), expectedMemref, proof));
  if (auto empty = tensor.getDefiningOp<mlir::tensor::EmptyOp>())
    return finish(proof.emptyAllocations.lookup(empty.getOperation()) ==
                  expectedMemref);
  if (auto collapse = tensor.getDefiningOp<mlir::tensor::CollapseShapeOp>())
    return finish(isTensorBackedBy(collapse.getSrc(), expectedMemref, proof));
  if (auto expand = tensor.getDefiningOp<mlir::tensor::ExpandShapeOp>())
    return finish(isTensorBackedBy(expand.getSrc(), expectedMemref, proof));

  if (auto blockArgument = mlir::dyn_cast<mlir::BlockArgument>(tensor)) {
    mlir::Block *owner = blockArgument.getOwner();
    if (auto tileRegion = mlir::dyn_cast_or_null<TileRegionOp>(
            owner ? owner->getParentOp() : nullptr)) {
      unsigned index = blockArgument.getArgNumber();
      return finish(owner == &tileRegion.getBody().front() &&
                    index < tileRegion.getInputs().size() &&
                    isValueBackedBy(tileRegion.getInputs()[index],
                                    expectedMemref, proof));
    }
    auto forOp = mlir::dyn_cast_or_null<mlir::scf::ForOp>(
        owner ? owner->getParentOp() : nullptr);
    if (!forOp || owner != forOp.getBody() || blockArgument.getArgNumber() == 0)
      return finish(false);
    return finish(isForTensorRecurrenceBackedBy(
        forOp, blockArgument.getArgNumber() - 1, expectedMemref, proof));
  }

  auto result = mlir::dyn_cast<mlir::OpResult>(tensor);
  if (!result)
    return finish(false);
  if (auto tileRegion = mlir::dyn_cast<TileRegionOp>(result.getOwner())) {
    if (tileRegion.getBody().empty())
      return finish(false);
    auto yield = mlir::dyn_cast<TileYieldOp>(
        tileRegion.getBody().front().getTerminator());
    unsigned index = result.getResultNumber();
    return finish(
        yield && index < yield.getValues().size() &&
        isValueBackedBy(yield.getValues()[index], expectedMemref, proof));
  }
  if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(result.getOwner()))
    return finish(isForTensorRecurrenceBackedBy(forOp, result.getResultNumber(),
                                                expectedMemref, proof));
  if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(result.getOwner())) {
    unsigned index = result.getResultNumber();
    auto thenYield = mlir::dyn_cast<mlir::scf::YieldOp>(
        ifOp.getThenRegion().front().getTerminator());
    auto elseYield = mlir::dyn_cast<mlir::scf::YieldOp>(
        ifOp.getElseRegion().front().getTerminator());
    return finish(
        thenYield && elseYield && index < thenYield.getResults().size() &&
        index < elseYield.getResults().size() &&
        isTensorBackedBy(thenYield.getResults()[index], expectedMemref,
                         proof) &&
        isTensorBackedBy(elseYield.getResults()[index], expectedMemref, proof));
  }
  return finish(false);
}

static bool isMemrefBackedBy(mlir::Value memref, mlir::Value expectedMemref,
                             BoundaryAliasProof &proof) {
  if (memref == expectedMemref)
    return true;
  if (!proof.activeMemrefs.insert(memref).second)
    return false;
  auto finish = [&](bool result) {
    proof.activeMemrefs.erase(memref);
    return result;
  };

  if (auto toMemref = memref.getDefiningOp<mlir::bufferization::ToMemrefOp>())
    return finish(
        isTensorBackedBy(toMemref.getTensor(), expectedMemref, proof));
  if (auto viewLike = mlir::dyn_cast_or_null<mlir::ViewLikeOpInterface>(
          memref.getDefiningOp()))
    return finish(
        isMemrefBackedBy(viewLike.getViewSource(), expectedMemref, proof));

  if (auto result = mlir::dyn_cast<mlir::OpResult>(memref)) {
    if (auto tileRegion = mlir::dyn_cast<TileRegionOp>(result.getOwner())) {
      if (tileRegion.getBody().empty())
        return finish(false);
      auto yield = mlir::dyn_cast<TileYieldOp>(
          tileRegion.getBody().front().getTerminator());
      unsigned index = result.getResultNumber();
      return finish(
          yield && index < yield.getValues().size() &&
          isValueBackedBy(yield.getValues()[index], expectedMemref, proof));
    }
  }

  if (auto blockArgument = mlir::dyn_cast<mlir::BlockArgument>(memref)) {
    mlir::Block *owner = blockArgument.getOwner();
    if (auto tileRegion = mlir::dyn_cast_or_null<TileRegionOp>(
            owner ? owner->getParentOp() : nullptr)) {
      unsigned index = blockArgument.getArgNumber();
      return finish(owner == &tileRegion.getBody().front() &&
                    index < tileRegion.getInputs().size() &&
                    isValueBackedBy(tileRegion.getInputs()[index],
                                    expectedMemref, proof));
    }
  }
  return finish(false);
}

static mlir::Value resolveCommittedTensorBoundary(
    mlir::Value tensor, mlir::MemRefType memrefType,
    llvm::DenseMap<mlir::Operation *, mlir::Value> &emptyAllocations) {
  if (auto toTensor = tensor.getDefiningOp<mlir::bufferization::ToTensorOp>())
    return toTensor.getMemref().getType() == memrefType ? toTensor.getMemref()
                                                        : mlir::Value{};

  if (auto empty = tensor.getDefiningOp<mlir::tensor::EmptyOp>()) {
    if (!memrefType.hasStaticShape())
      return {};
    mlir::Value replacement = emptyAllocations.lookup(empty.getOperation());
    if (replacement)
      return replacement.getType() == memrefType ? replacement : mlir::Value{};
    mlir::OpBuilder builder(empty);
    builder.setInsertionPointAfter(empty);
    replacement =
        builder.create<mlir::memref::AllocOp>(empty.getLoc(), memrefType);
    emptyAllocations.try_emplace(empty.getOperation(), replacement);
    return replacement;
  }

  if (auto collapse = tensor.getDefiningOp<mlir::tensor::CollapseShapeOp>()) {
    auto sourceTensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(collapse.getSrcType());
    if (!sourceTensorType || !sourceTensorType.hasStaticShape())
      return {};
    auto sourceMemrefType = mlir::MemRefType::get(
        sourceTensorType.getShape(), sourceTensorType.getElementType(),
        mlir::MemRefLayoutAttrInterface{}, memrefType.getMemorySpace());
    mlir::Value source = resolveCommittedTensorBoundary(
        collapse.getSrc(), sourceMemrefType, emptyAllocations);
    if (!source)
      return {};
    mlir::OpBuilder builder(collapse);
    builder.setInsertionPointAfter(collapse);
    return builder
        .create<mlir::memref::CollapseShapeOp>(
            collapse.getLoc(), memrefType, source,
            collapse.getReassociationIndices())
        .getResult();
  }
  if (auto expand = tensor.getDefiningOp<mlir::tensor::ExpandShapeOp>()) {
    auto sourceTensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(expand.getSrcType());
    if (!sourceTensorType || !sourceTensorType.hasStaticShape())
      return {};
    auto sourceMemrefType = mlir::MemRefType::get(
        sourceTensorType.getShape(), sourceTensorType.getElementType(),
        mlir::MemRefLayoutAttrInterface{}, memrefType.getMemorySpace());
    mlir::Value source = resolveCommittedTensorBoundary(
        expand.getSrc(), sourceMemrefType, emptyAllocations);
    if (!source)
      return {};
    mlir::OpBuilder builder(expand);
    builder.setInsertionPointAfter(expand);
    return builder
        .create<mlir::memref::ExpandShapeOp>(expand.getLoc(), memrefType,
                                             source,
                                             expand.getReassociationIndices())
        .getResult();
  }

  auto blockArgument = mlir::dyn_cast<mlir::BlockArgument>(tensor);
  if (!blockArgument)
    return {};
  mlir::Block *owner = blockArgument.getOwner();
  if (auto tileRegion = mlir::dyn_cast_or_null<TileRegionOp>(
          owner ? owner->getParentOp() : nullptr)) {
    unsigned index = blockArgument.getArgNumber();
    if (owner != &tileRegion.getBody().front() ||
        index >= tileRegion.getInputs().size())
      return {};
    mlir::Value input = tileRegion.getInputs()[index];
    if (!mlir::isa<mlir::TensorType>(input.getType()))
      return {};
    return resolveCommittedTensorBoundary(input, memrefType, emptyAllocations);
  }
  auto forOp = mlir::dyn_cast_or_null<mlir::scf::ForOp>(
      owner ? owner->getParentOp() : nullptr);
  if (!forOp || owner != forOp.getBody() || blockArgument.getArgNumber() == 0)
    return {};
  unsigned index = blockArgument.getArgNumber() - 1;
  if (index >= forOp.getInitArgs().size())
    return {};
  mlir::Value root = resolveCommittedTensorBoundary(
      forOp.getInitArgs()[index], memrefType, emptyAllocations);
  if (!root)
    return {};
  BoundaryAliasProof proof{emptyAllocations};
  return isForTensorRecurrenceBackedBy(forOp, index, root, proof)
             ? root
             : mlir::Value{};
}

static mlir::memref::GlobalOp findOrCreateDenseConstantGlobal(
    mlir::ModuleOp module, mlir::arith::ConstantOp constant,
    mlir::MemRefType memrefType, mlir::ElementsAttr elements) {
  mlir::memref::GlobalOp global;
  for (mlir::memref::GlobalOp candidate :
       module.getOps<mlir::memref::GlobalOp>()) {
    if (candidate.getType() == memrefType && candidate.getConstant() &&
        candidate.getConstantInitValue() == elements) {
      global = candidate;
      break;
    }
  }

  if (!global) {
    mlir::OpBuilder globalBuilder(module.getContext());
    mlir::SymbolTable symbolTable(module);
    global = globalBuilder.create<mlir::memref::GlobalOp>(
        constant.getLoc(), "__wafer_constant",
        globalBuilder.getStringAttr("private"), memrefType, elements,
        /*constant=*/true, /*alignment=*/mlir::IntegerAttr{});
    symbolTable.insert(global);
    global->moveBefore(&module.front());
  }

  return global;
}

/// Give a dense tensor constant a real read-only DDR backing.  A raw
/// tensor-to-memref adapter does not establish storage provenance, whereas a
/// private constant global is a typed symbol whose initializer, extent, and
/// memory space are all visible to DDR planning and target lowering.
static mlir::Value
outlineDenseConstantBoundary(mlir::bufferization::ToMemrefOp toMemref,
                             mlir::ModuleOp module) {
  auto constant = toMemref.getTensor().getDefiningOp<mlir::arith::ConstantOp>();
  auto memrefType =
      mlir::dyn_cast<mlir::MemRefType>(toMemref.getMemref().getType());
  auto elements = constant
                      ? mlir::dyn_cast<mlir::ElementsAttr>(constant.getValue())
                      : mlir::ElementsAttr{};
  if (!constant || !memrefType || !memrefType.hasStaticShape() || !elements ||
      elements.getShapedType().getShape() != memrefType.getShape() ||
      elements.getShapedType().getElementType() != memrefType.getElementType())
    return {};

  mlir::memref::GlobalOp global =
      findOrCreateDenseConstantGlobal(module, constant, memrefType, elements);

  mlir::OpBuilder builder(toMemref);
  return builder
      .create<mlir::memref::GetGlobalOp>(toMemref.getLoc(), memrefType,
                                         global.getSymName())
      .getResult();
}

/// Non-splat tensor constants cannot be synthesized by a scalar fill. Make
/// them typed read-only rank inputs before task extraction so every standalone
/// candidate sees an ordinary external tensor boundary. The exact
/// to_tensor/to_memref pair folds back to get_global after candidate commit.
static void outlineRankDenseTensorConstants(mlir::ModuleOp module) {
  llvm::SmallVector<mlir::arith::ConstantOp, 8> constants;
  module.walk([&](mlir::arith::ConstantOp constant) {
    auto tensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(constant.getType());
    auto elements = mlir::dyn_cast<mlir::ElementsAttr>(constant.getValue());
    if (tensorType && tensorType.hasStaticShape() && elements &&
        !elements.isSplat())
      constants.push_back(constant);
  });

  for (mlir::arith::ConstantOp constant : constants) {
    auto tensorType = mlir::cast<mlir::RankedTensorType>(constant.getType());
    auto elements = mlir::cast<mlir::ElementsAttr>(constant.getValue());
    mlir::MemRefType memrefType = mlir::MemRefType::get(
        tensorType.getShape(), tensorType.getElementType(),
        mlir::MemRefLayoutAttrInterface{},
        MemoryAttr::get(module.getContext(), MemorySpace::DDR,
                        MemLayout::Tensor));
    mlir::memref::GlobalOp global =
        findOrCreateDenseConstantGlobal(module, constant, memrefType, elements);
    mlir::OpBuilder builder(constant);
    builder.setInsertionPointAfter(constant);
    auto getGlobal = builder.create<mlir::memref::GetGlobalOp>(
        constant.getLoc(), memrefType, global.getSymName());
    auto tensor = builder.create<mlir::bufferization::ToTensorOp>(
        constant.getLoc(), getGlobal.getResult(), /*restrict=*/false,
        /*writable=*/false);
    constant.getResult().replaceAllUsesWith(tensor.getResult());
    constant.erase();
  }
}

/// Commit materializes tensor task boundaries as to_tensor/to_memref pairs.
/// Once producer and consumer tasks share the committed rank those pairs no
/// longer represent an ABI boundary.  Fold exact-type round trips and
/// materialize tensor.empty destinations before rank-wide lifetime planning
/// so the planner sees explicit DDR roots and their existing alias chains.
static void foldCommittedBufferizationRoundTrips(mlir::ModuleOp module) {
  llvm::SmallVector<mlir::bufferization::ToMemrefOp, 8> toErase;
  llvm::DenseMap<mlir::Operation *, mlir::Value> emptyAllocations;
  module.walk([&](mlir::bufferization::ToMemrefOp toMemref) {
    auto memrefType =
        mlir::dyn_cast<mlir::MemRefType>(toMemref.getMemref().getType());
    if (!memrefType)
      return;
    mlir::Value replacement = resolveCommittedTensorBoundary(
        toMemref.getTensor(), memrefType, emptyAllocations);
    if (!replacement)
      replacement = outlineDenseConstantBoundary(toMemref, module);
    if (!replacement)
      return;
    toMemref.getMemref().replaceAllUsesWith(replacement);
    toErase.push_back(toMemref);
  });
  for (mlir::bufferization::ToMemrefOp toMemref : toErase)
    toMemref.erase();

  // A tensor reshape whose only storage consumer was the folded to_memref is
  // now dead.  Remove the tensor view before pruning to_tensor so lifetime
  // analysis observes only the equivalent memref view chain.
  bool erasedTensorReshape = true;
  while (erasedTensorReshape) {
    erasedTensorReshape = false;
    llvm::SmallVector<mlir::Operation *, 8> deadTensorReshapes;
    module.walk([&](mlir::Operation *operation) {
      if (operation->use_empty() &&
          mlir::isa<mlir::tensor::CollapseShapeOp, mlir::tensor::ExpandShapeOp>(
              operation))
        deadTensorReshapes.push_back(operation);
    });
    for (mlir::Operation *operation : deadTensorReshapes) {
      operation->erase();
      erasedTensorReshape = true;
    }
  }

  llvm::SmallVector<mlir::bufferization::ToTensorOp, 8> deadToTensors;
  module.walk([&](mlir::bufferization::ToTensorOp toTensor) {
    if (toTensor->use_empty())
      deadToTensors.push_back(toTensor);
  });
  for (mlir::bufferization::ToTensorOp toTensor : deadToTensors)
    toTensor.erase();

  llvm::SmallVector<mlir::tensor::EmptyOp, 8> deadEmpties;
  module.walk([&](mlir::tensor::EmptyOp empty) {
    if (empty->use_empty())
      deadEmpties.push_back(empty);
  });
  for (mlir::tensor::EmptyOp empty : deadEmpties)
    empty.erase();

  llvm::SmallVector<mlir::arith::ConstantOp, 8> deadTensorConstants;
  module.walk([&](mlir::arith::ConstantOp constant) {
    if (constant->use_empty() &&
        mlir::isa<mlir::TensorType>(constant.getType()))
      deadTensorConstants.push_back(constant);
  });
  for (mlir::arith::ConstantOp constant : deadTensorConstants)
    constant.erase();
}

struct RankVariantEvaluation {
  std::string label;
  std::string partitionSignature;
  std::string failureReason;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  llvm::SmallVector<SelectedCandidate, 8> selectedCandidates;
  CandidateStats stats;
  int64_t estimatedTimePs = std::numeric_limits<int64_t>::max();
  bool accepted = false;
  bool duplicate = false;
  bool noScopes = false;
  bool failedOnCuttableTerminalFullTraversalOnlyScope = false;
};

struct FinalizedRankMetrics {
  CandidateStats stats;
  int64_t estimatedTimePs = std::numeric_limits<int64_t>::max();
};

static std::string
getPolicyLabel(const structured_scheduler::ScopeDiscoveryPolicy &policy) {
  std::string label;
  llvm::raw_string_ostream os(label);
  if (policy.cutTerminalFullTraversalOnlyRoots)
    os << "dataflow-terminal-full-only-cut";
  else
    os << (policy.allowCrossShapeDataflow ? "dataflow" : "conservative");
  os << "/shared-peers=";
  if (policy.maxSharedInputPeers < 0)
    os << "all";
  else
    os << policy.maxSharedInputPeers;
  return label;
}

static mlir::LogicalResult failRankVariant(RankVariantEvaluation &evaluation,
                                           llvm::StringRef gate,
                                           llvm::StringRef detail = {}) {
  llvm::raw_string_ostream os(evaluation.failureReason);
  os << gate;
  if (!detail.empty())
    os << ": " << detail;
  return mlir::failure();
}

/// Finalize one complete rank artifact independently.  This is deliberately
/// shared by the spill baseline and the deterministic full-buffer-resident
/// alternative: neither representation can participate in rank selection
/// until the exact same whole-rank SPM, DDR, verifier, and cost gates accept
/// it.
static std::optional<std::string>
finalizeRankArtifact(mlir::ModuleOp module, const SelectionConfig &config,
                     FinalizedRankMetrics &metrics) {
  if (mlir::failed(mlir::verify(module)))
    return "whole-rank-pre-plan-verifier";
  if (mlir::failed(planSPMMemoryModule(module, config.spmBase, config.spmLimit,
                                       config.spmAlignment)))
    return "whole-rank-spm-offsets";
  if (mlir::failed(planDDRMemoryModule(
          module, config.ddrAlignmentBytes, config.ddrCapacityBytes,
          config.ddrLargestContiguousBytes, config.ddrBandwidthLimitBytes)))
    return "whole-rank-ddr-offsets";
  if (mlir::failed(mlir::verify(module)))
    return "whole-rank-verifier";

  metrics.stats = estimateStats(module);
  if (config.mode == TileSearchMode::MinEstimatedTime) {
    if (std::optional<std::string> failure =
            getRankingCostFailure(metrics.stats))
      return (llvm::Twine("whole-rank-cost: ") + *failure).str();
  }
  metrics.estimatedTimePs = estimateCandidateTimePs(metrics.stats);
  return std::nullopt;
}

static mlir::FailureOr<std::string> getPartitionSignature(
    mlir::ModuleOp module,
    llvm::ArrayRef<structured_scheduler::StructuredSchedulingScope> scopes) {
  llvm::DenseMap<mlir::Operation *, uint64_t> operationOrdinals;
  uint64_t nextOrdinal = 0;
  module.walk([&](mlir::Operation *operation) {
    operationOrdinals.try_emplace(operation, nextOrdinal++);
  });

  std::string signature;
  llvm::raw_string_ostream os(signature);
  for (const auto &scope : scopes) {
    os << "{";
    for (mlir::Operation *operation : scope.orderedOps) {
      auto found = operationOrdinals.find(operation);
      if (found == operationOrdinals.end())
        return mlir::failure();
      os << found->second << ",";
    }
    os << "}";
  }
  return signature;
}

static mlir::LogicalResult
checkRankTerminalBudget(mlir::ModuleOp stagedModule,
                        llvm::ArrayRef<SelectedCandidate> selectedCandidates,
                        RankVariantEvaluation &evaluation) {
  uint64_t terminalOperationCount = 0;
  if (mlir::failed(accumulateStaticTerminalOperations(
          stagedModule.getOperation(), terminalOperationCount)))
    return failRankVariant(evaluation, "static-terminal-budget",
                           "source rank operation count overflows");

  for (const SelectedCandidate &selected : selectedCandidates) {
    if (!selected.module)
      return failRankVariant(evaluation, "static-terminal-budget",
                             "selected task operation count is unavailable");
    uint64_t selectedOperationCount = 0;
    detail::StaticTerminalOperationBudgetStatus status =
        detail::checkStaticTerminalOperationBudget(
            (*selected.module).getOperation(), selectedOperationCount);
    if (status == detail::StaticTerminalOperationBudgetStatus::CountOverflow ||
        selectedOperationCount >
            std::numeric_limits<uint64_t>::max() - terminalOperationCount)
      return failRankVariant(evaluation, "static-terminal-budget",
                             "selected rank operation count overflows");
    terminalOperationCount += selectedOperationCount;
  }
  if (terminalOperationCount > wafer::detail::kStaticTerminalOperationBudget) {
    std::string detail;
    llvm::raw_string_ostream os(detail);
    os << "selected rank requires " << terminalOperationCount
       << " terminal instruction issue/completion operations; limit is "
       << wafer::detail::kStaticTerminalOperationBudget;
    return failRankVariant(evaluation, "static-terminal-budget", os.str());
  }
  return mlir::success();
}

static mlir::LogicalResult evaluateRankVariantImpl(
    mlir::ModuleOp sourceModule, const SelectionConfig &config,
    const structured_scheduler::ScopeDiscoveryPolicy &policy,
    llvm::StringSet<> &seenPartitions, RankVariantEvaluation &evaluation) {
  evaluation.label = getPolicyLabel(policy);
  evaluation.module = mlir::cast<mlir::ModuleOp>(sourceModule->clone());
  outlineRankDenseTensorConstants(*evaluation.module);

  llvm::SmallVector<structured_scheduler::StructuredSchedulingScope, 8> scopes;
  if (mlir::failed(structured_scheduler::discoverStructuredSchedulingScopes(
          *evaluation.module, scopes, policy)))
    return failRankVariant(evaluation, "scope-discovery");
  if (scopes.empty()) {
    evaluation.noScopes = true;
    return mlir::success();
  }
  if (config.logicalRank < 0)
    return failRankVariant(evaluation, "missing-logical-rank",
                           "tensor-program scheduling requires an explicit "
                           "non-negative logical-rank");

  mlir::FailureOr<std::string> signature =
      getPartitionSignature(*evaluation.module, scopes);
  if (mlir::failed(signature))
    return failRankVariant(evaluation, "scope-signature",
                           "scope contains an operation outside the rank");
  evaluation.partitionSignature = std::move(*signature);
  if (!seenPartitions.insert(evaluation.partitionSignature).second) {
    evaluation.duplicate = true;
    return mlir::success();
  }

  // SelectedCandidate retains sourceTask handles until commit.  Keep every
  // standalone task module alive across selection, the terminal budget gate,
  // and the complete consumer-first commit.
  llvm::SmallVector<mlir::OwningOpRef<mlir::ModuleOp>, 8> taskModules;
  taskModules.reserve(scopes.size());
  evaluation.selectedCandidates.reserve(scopes.size());
  llvm::DenseMap<mlir::Operation *, unsigned> taskOrdinals;
  for (const auto &scope : scopes) {
    mlir::OwningOpRef<mlir::ModuleOp> taskModule =
        structured_scheduler::cloneScopeToStandaloneModule(scope);
    if (!taskModule)
      return failRankVariant(evaluation, "task-clone",
                             "cannot clone structured scheduling boundary");
    mlir::func::FuncOp task =
        structured_scheduler::findSingleTaskFunction(*taskModule);
    if (!task || mlir::failed(mlir::verify(*taskModule)))
      return failRankVariant(evaluation, "task-verifier",
                             "standalone structured scheduling task is "
                             "invalid");
    std::string symbolName = getNearestSymbolName(scope.insertionPoint);
    unsigned ordinal = taskOrdinals[scope.insertionPoint->getParentOp()]++;
    std::string taskLabel;
    llvm::raw_string_ostream labelOs(taskLabel);
    labelOs << symbolName << "#" << ordinal;
    if (config.printCandidateSummary) {
      llvm::errs() << "wafer.schedule_tensor_program discovered task "
                   << labelOs.str() << " ops=[";
      for (auto [opIndex, operation] : llvm::enumerate(scope.orderedOps)) {
        if (opIndex)
          llvm::errs() << ",";
        llvm::errs() << operation->getName();
      }
      llvm::errs() << "] inputs=[";
      for (auto [inputIndex, input] : llvm::enumerate(scope.inputs)) {
        if (inputIndex)
          llvm::errs() << ",";
        llvm::errs() << input.getType();
      }
      llvm::errs() << "] outputs=[";
      for (auto [outputIndex, output] : llvm::enumerate(scope.yieldedValues)) {
        if (outputIndex)
          llvm::errs() << ",";
        llvm::errs() << output.getType();
      }
      llvm::errs() << "]\n";
    }
    taskModules.push_back(std::move(taskModule));
    mlir::FailureOr<SelectedCandidate> selected =
        selectCandidateForScope(scope, task, labelOs.str(), config);
    if (mlir::failed(selected)) {
      // A full-traversal-only yielded root fixes this entire scope at its full
      // shape. Shared-input peer prefixes can only add independent roots;
      // they cannot make the direct dataflow closure tileable. Record the
      // precise structural case handled by the one bounded terminal recovery
      // policy so the caller need not evaluate the remaining normal prefixes
      // before trying that policy.
      if (getTaskTraversalRootCapability(task) ==
              CandidateTraversalRootCapability::FullTraversalOnly &&
          llvm::any_of(scope.orderedOps, [](mlir::Operation *operation) {
            return classifyCandidateTraversalRoot(operation) ==
                   CandidateTraversalRootCapability::Tiled;
          }))
        evaluation.failedOnCuttableTerminalFullTraversalOnlyScope = true;
      return failRankVariant(evaluation, "candidate-selection");
    }
    evaluation.selectedCandidates.push_back(std::move(*selected));
  }

  if (mlir::failed(checkRankTerminalBudget(
          *evaluation.module, evaluation.selectedCandidates, evaluation)))
    return mlir::failure();

  // Consumer-first commit keeps source SSA boundaries valid when two task
  // scopes are separated by an unsupported fusion boundary.
  for (SelectedCandidate &selected :
       llvm::reverse(evaluation.selectedCandidates)) {
    if (mlir::failed(commitSelectedTaskCandidate(selected, config)))
      return failRankVariant(evaluation, "candidate-commit", selected.label);
    if (std::optional<std::string> failure =
            getCommittedTaskStructuralFailure(*evaluation.module))
      return failRankVariant(evaluation, "candidate-commit", *failure);
  }
  foldCommittedBufferizationRoundTrips(*evaluation.module);

  // Normalize the committed rank once before deriving either final artifact.
  // In particular, static one-trip traversal recurrences and full subviews
  // must be folded here: otherwise equivalent full-buffer edges would be
  // accepted or rejected based on incidental candidate-emission wrappers.
  mlir::PassManager canonicalization(evaluation.module->getContext());
  canonicalization.addPass(mlir::createCanonicalizerPass());
  if (mlir::failed(canonicalization.run(*evaluation.module)))
    return failRankVariant(evaluation, "committed-canonicalization");

  uint64_t committedTerminalOperationCount = 0;
  detail::StaticTerminalOperationBudgetStatus committedBudgetStatus =
      detail::checkStaticTerminalOperationBudget(
          (*evaluation.module).getOperation(), committedTerminalOperationCount);
  if (committedBudgetStatus !=
      detail::StaticTerminalOperationBudgetStatus::WithinBudget) {
    std::string detail;
    llvm::raw_string_ostream os(detail);
    os << "committed rank requires " << committedTerminalOperationCount
       << " terminal instruction issue/completion operations; limit is "
       << wafer::detail::kStaticTerminalOperationBudget;
    return failRankVariant(evaluation, "static-terminal-budget", os.str());
  }

  // Preserve the spill baseline before independently finalizing residency.
  // Full-buffer handoff is one bounded, deterministic alternative rather than
  // another per-edge search dimension.  A failed or non-improving resident
  // clone therefore cannot invalidate the rank.
  mlir::OwningOpRef<mlir::ModuleOp> residentModule =
      mlir::cast<mlir::ModuleOp>((*evaluation.module)->clone());
  unsigned promotedHandoffs = promoteFullBufferHandoffs(*residentModule);

  FinalizedRankMetrics spillMetrics;
  std::optional<std::string> spillFailure =
      finalizeRankArtifact(*evaluation.module, config, spillMetrics);
  FinalizedRankMetrics residentMetrics;
  std::optional<std::string> residentFailure;
  if (promotedHandoffs != 0) {
    residentFailure =
        finalizeRankArtifact(*residentModule, config, residentMetrics);
  }

  if (spillFailure && (promotedHandoffs == 0 || residentFailure)) {
    std::string detail;
    llvm::raw_string_ostream os(detail);
    os << "spill=" << *spillFailure;
    if (promotedHandoffs != 0)
      os << ", resident=" << *residentFailure;
    return failRankVariant(evaluation, "whole-rank-artifact", os.str());
  }

  bool selectResident =
      promotedHandoffs != 0 && !residentFailure &&
      (spillFailure ||
       residentMetrics.estimatedTimePs < spillMetrics.estimatedTimePs ||
       hasStrictExecutionCostDominance(residentMetrics.stats,
                                       spillMetrics.stats));
  if (selectResident) {
    evaluation.module = std::move(residentModule);
    evaluation.stats = residentMetrics.stats;
    evaluation.estimatedTimePs = residentMetrics.estimatedTimePs;
    if (config.printCandidateSummary)
      llvm::errs() << "wafer.schedule_tensor_program selected "
                   << promotedHandoffs
                   << " full-buffer SPM handoff(s) for rank variant "
                   << evaluation.label << "\n";
  } else {
    evaluation.stats = spillMetrics.stats;
    evaluation.estimatedTimePs = spillMetrics.estimatedTimePs;
    if (promotedHandoffs != 0 && config.printCandidateSummary) {
      llvm::errs() << "wafer.schedule_tensor_program retained DDR spill for "
                   << promotedHandoffs
                   << " full-buffer handoff candidate(s) in rank variant "
                   << evaluation.label;
      if (residentFailure)
        llvm::errs() << ": " << *residentFailure;
      else
        llvm::errs() << ": resident cost is not lower";
      llvm::errs() << "\n";
    }
  }
  evaluation.accepted = true;
  return mlir::success();
}

static RankVariantEvaluation
evaluateRankVariant(mlir::ModuleOp sourceModule, const SelectionConfig &config,
                    const structured_scheduler::ScopeDiscoveryPolicy &policy,
                    llvm::StringSet<> &seenPartitions) {
  RankVariantEvaluation evaluation;
  std::string diagnostics;
  mlir::LogicalResult result = mlir::success();
  {
    mlir::ScopedDiagnosticHandler handler(
        sourceModule.getContext(), [&](mlir::Diagnostic &diagnostic) {
          llvm::raw_string_ostream os(diagnostics);
          diagnostic.print(os);
          os << "\n";
          return mlir::success();
        });
    result = evaluateRankVariantImpl(sourceModule, config, policy,
                                     seenPartitions, evaluation);
  }
  if (mlir::failed(result)) {
    llvm::StringRef captured = llvm::StringRef(diagnostics).trim();
    if (evaluation.failureReason.empty())
      evaluation.failureReason = captured.str();
    else if (!captured.empty()) {
      llvm::raw_string_ostream os(evaluation.failureReason);
      os << "; " << captured;
    }
  }
  return evaluation;
}

static bool isBetterRankVariant(const RankVariantEvaluation &candidate,
                                const RankVariantEvaluation *best) {
  return !best || candidate.estimatedTimePs < best->estimatedTimePs;
}

struct ScheduleTensorProgramPass
    : public impl::ScheduleTensorProgramPassBase<ScheduleTensorProgramPass> {
  using impl::ScheduleTensorProgramPassBase<
      ScheduleTensorProgramPass>::ScheduleTensorProgramPassBase;

  void runOnOperation() final {
    mlir::FailureOr<TileSearchMode> parsedMode =
        parseTileSearchMode(tileSearch, getOperation());
    mlir::FailureOr<TileSearchEffort> parsedEffort =
        parseTileSearchEffort(tileSearchEffort, getOperation());
    if (mlir::failed(parsedMode) || mlir::failed(parsedEffort)) {
      signalPassFailure();
      return;
    }
    std::optional<llvm::SmallVector<int64_t, 8>> parsedPreferred;
    if (!llvm::StringRef(preferredTileSizes).trim().empty()) {
      mlir::FailureOr<llvm::SmallVector<int64_t, 8>> parsed = parseI64List(
          preferredTileSizes, "preferred-tile-sizes", getOperation());
      if (mlir::failed(parsed)) {
        signalPassFailure();
        return;
      }
      parsedPreferred = std::move(*parsed);
    }

    if (maxCandidatesPerDim < -1 || maxSearchCandidates < -1 ||
        searchBeamWidth < -1) {
      getOperation()->emitError()
          << "invalid_tile_search_config: search-space overrides must be -1 "
             "or non-negative";
      signalPassFailure();
      return;
    }
    if (*parsedMode == TileSearchMode::MinEstimatedTime &&
        candidateParallelism <= 0) {
      getOperation()->emitError()
          << "invalid_tile_search_config: min-estimated-time requires "
             "positive candidate-parallelism";
      signalPassFailure();
      return;
    }

    WaferTargetPolicy targetPolicy = getDefaultWaferTargetPolicy(*parsedEffort);
    SelectionConfig config(targetPolicy);
    config.mode = *parsedMode;
    config.logicalRank = logicalRank;
    if (parsedPreferred && parsedPreferred->empty()) {
      getOperation()->emitError()
          << "invalid_tile_search_config: preferred-tile-sizes cannot be empty "
             "when explicitly provided";
      signalPassFailure();
      return;
    }
    if (parsedPreferred)
      config.preferredTileSizes = *parsedPreferred;
    if (maxCandidatesPerDim >= 0)
      config.maxCandidatesPerDim = maxCandidatesPerDim;
    if (maxSearchCandidates >= 0)
      config.maxSearchCandidates = maxSearchCandidates;
    if (searchBeamWidth >= 0)
      config.searchBeamWidth = searchBeamWidth;
    if (config.mode == TileSearchMode::MinEstimatedTime)
      config.candidateParallelism = candidateParallelism;
    config.printCandidateSummary = printCandidateSummary;

    // Partition alternatives are bounded prefixes of the deterministic
    // shared-input reuse order.  Each alternative is independently cloned,
    // selected, committed, rank-planned, verified, and costed.  A partition
    // signature prevents a no-op prefix from repeating expensive candidate
    // search.
    const structured_scheduler::ScopeDiscoveryPolicy aggressivePolicies[] = {
        {/*maxSharedInputPeers=*/0, /*allowCrossShapeDataflow=*/true,
         /*cutTerminalFullTraversalOnlyRoots=*/false},
        {/*maxSharedInputPeers=*/1, /*allowCrossShapeDataflow=*/true,
         /*cutTerminalFullTraversalOnlyRoots=*/false},
        {/*maxSharedInputPeers=*/2, /*allowCrossShapeDataflow=*/true,
         /*cutTerminalFullTraversalOnlyRoots=*/false},
        {/*maxSharedInputPeers=*/-1, /*allowCrossShapeDataflow=*/true,
         /*cutTerminalFullTraversalOnlyRoots=*/false},
    };
    const structured_scheduler::ScopeDiscoveryPolicy recoveryPolicy = {
        /*maxSharedInputPeers=*/0,
        /*allowCrossShapeDataflow=*/true,
        /*cutTerminalFullTraversalOnlyRoots=*/true};
    llvm::StringSet<> seenPartitions;
    llvm::SmallVector<std::string, 6> failures;
    std::optional<RankVariantEvaluation> best;
    bool recoveryAttempted = false;
    auto recordFailure = [&](const RankVariantEvaluation &evaluation) {
      std::string failure;
      llvm::raw_string_ostream os(failure);
      os << evaluation.label << ": "
         << (evaluation.failureReason.empty() ? "failed"
                                              : evaluation.failureReason);
      failures.push_back(std::move(failure));
    };
    for (const auto &policy : aggressivePolicies) {
      RankVariantEvaluation evaluation =
          evaluateRankVariant(getOperation(), config, policy, seenPartitions);
      if (evaluation.noScopes) {
        markAllAnalysesPreserved();
        return;
      }
      if (evaluation.duplicate)
        continue;
      if (!evaluation.accepted) {
        recordFailure(evaluation);
        if (evaluation.failedOnCuttableTerminalFullTraversalOnlyScope &&
            !recoveryAttempted) {
          // Every normal peer-prefix retains the same direct dataflow closure
          // and therefore the same non-tileable yielded root. Try the sole
          // terminal-cut policy now instead of replaying that deterministic
          // failure through the remaining prefixes. This only reorders the
          // existing frontier; it neither adds a policy nor crosses recovery
          // with peer choices.
          recoveryAttempted = true;
          RankVariantEvaluation recovery = evaluateRankVariant(
              getOperation(), config, recoveryPolicy, seenPartitions);
          if (recovery.noScopes) {
            markAllAnalysesPreserved();
            return;
          }
          if (recovery.accepted) {
            best = std::move(recovery);
            break;
          }
          if (!recovery.duplicate)
            recordFailure(recovery);
        }
        continue;
      }
      if (config.mode == TileSearchMode::FirstLegal ||
          isBetterRankVariant(evaluation, best ? &*best : nullptr))
        best = std::move(evaluation);
      if (config.mode == TileSearchMode::FirstLegal)
        break;
    }

    // If no failed scope exposed the exact capability condition above, retain
    // the recovery policy at its original bounded fallback position.
    if (!best && !recoveryAttempted) {
      recoveryAttempted = true;
      RankVariantEvaluation evaluation = evaluateRankVariant(
          getOperation(), config, recoveryPolicy, seenPartitions);
      if (evaluation.noScopes) {
        markAllAnalysesPreserved();
        return;
      }
      if (evaluation.accepted)
        best = std::move(evaluation);
      else if (!evaluation.duplicate)
        recordFailure(evaluation);
    }

    // Conservative shape/family closure is the broad legality recovery path,
    // not a cost competitor to a successful dataflow or root-capability
    // partition.
    if (!best) {
      const structured_scheduler::ScopeDiscoveryPolicy fallbackPolicy = {
          /*maxSharedInputPeers=*/0,
          /*allowCrossShapeDataflow=*/false,
          /*cutTerminalFullTraversalOnlyRoots=*/false};
      RankVariantEvaluation evaluation = evaluateRankVariant(
          getOperation(), config, fallbackPolicy, seenPartitions);
      if (evaluation.noScopes) {
        markAllAnalysesPreserved();
        return;
      }
      if (evaluation.accepted)
        best = std::move(evaluation);
      else if (!evaluation.duplicate)
        recordFailure(evaluation);
    }

    if (!best) {
      auto diagnostic = getOperation()->emitError()
                        << "no_complete_rank_variant: bounded scheduling "
                           "search found no fully legal rank artifact";
      for (const std::string &failure : failures)
        diagnostic << "\n  - " << failure;
      signalPassFailure();
      return;
    }

    if (printCandidateSummary)
      for (const SelectedCandidate &selected : best->selectedCandidates)
        printSelectedSummary(selected, config.mode);
    getOperation()->setAttrs((*best->module)->getAttrs());
    getOperation().getBodyRegion().takeBody(best->module->getBodyRegion());
    return;
  }
};

} // namespace

mlir::FailureOr<std::vector<ScheduledRankCandidate>>
buildScheduledRankCandidateFrontier(
    mlir::ModuleOp sourceModule,
    const TensorProgramSchedulingConfig &frontierConfig) {
  if (!sourceModule) {
    return mlir::failure();
  }
  if (frontierConfig.logicalRank < 0) {
    sourceModule.emitError()
        << "invalid_tensor_program_scheduling_config: logical-rank must be "
           "non-negative";
    return mlir::failure();
  }
  if (frontierConfig.candidateParallelism <= 0) {
    sourceModule.emitError() << "invalid_tensor_program_scheduling_config: "
                                "candidate-parallelism must be positive";
    return mlir::failure();
  }

  WaferTargetPolicy targetPolicy =
      getDefaultWaferTargetPolicy(TileSearchEffort::Default);
  SelectionConfig config(targetPolicy);
  config.mode = TileSearchMode::MinEstimatedTime;
  config.logicalRank = frontierConfig.logicalRank;
  config.candidateParallelism = frontierConfig.candidateParallelism;

  // Discovery order is a stable compiler-private correspondence key across
  // ranks.  Every distinct partition is independently materialized and
  // rank-planned; the whole-variant coordinator may therefore reject a cheap
  // but cross-rank-incompatible combination without rerunning local search.
  const structured_scheduler::ScopeDiscoveryPolicy policies[] = {
      {/*maxSharedInputPeers=*/0, /*allowCrossShapeDataflow=*/true,
       /*cutTerminalFullTraversalOnlyRoots=*/false},
      {/*maxSharedInputPeers=*/1, /*allowCrossShapeDataflow=*/true,
       /*cutTerminalFullTraversalOnlyRoots=*/false},
      {/*maxSharedInputPeers=*/2, /*allowCrossShapeDataflow=*/true,
       /*cutTerminalFullTraversalOnlyRoots=*/false},
      {/*maxSharedInputPeers=*/-1, /*allowCrossShapeDataflow=*/true,
       /*cutTerminalFullTraversalOnlyRoots=*/false},
      {/*maxSharedInputPeers=*/0, /*allowCrossShapeDataflow=*/true,
       /*cutTerminalFullTraversalOnlyRoots=*/true},
      {/*maxSharedInputPeers=*/0, /*allowCrossShapeDataflow=*/false,
       /*cutTerminalFullTraversalOnlyRoots=*/false},
  };

  constexpr int64_t conservativeDiscoveryOrder = 5;

  llvm::StringSet<> seenPartitions;
  // Preserve the conservative policy as a common cross-rank recovery key
  // even when its partition happens to be structurally identical to one of
  // the aggressive prefixes on this rank.
  llvm::StringSet<> conservativeSeenPartitions;
  llvm::SmallVector<std::string, 6> failures;
  std::vector<ScheduledRankCandidate> frontier;
  frontier.reserve(std::size(policies));
  for (auto [discoveryOrder, policy] : llvm::enumerate(policies)) {
    llvm::StringSet<> &policySeenPartitions =
        discoveryOrder == conservativeDiscoveryOrder
            ? conservativeSeenPartitions
            : seenPartitions;
    RankVariantEvaluation evaluation =
        evaluateRankVariant(sourceModule, config, policy, policySeenPartitions);
    if (evaluation.noScopes) {
      // A rank without schedulable structured work is still a complete local
      // alternative.  Finalization and whole-card acceptance remain shared
      // with non-empty ranks.
      frontier.emplace_back(std::move(evaluation.module), /*cost=*/0,
                            /*discoveryOrder=*/conservativeDiscoveryOrder);
      return frontier;
    }
    if (evaluation.duplicate)
      continue;
    if (evaluation.accepted) {
      frontier.emplace_back(std::move(evaluation.module),
                            evaluation.estimatedTimePs,
                            static_cast<int64_t>(discoveryOrder));
      continue;
    }

    std::string failure;
    llvm::raw_string_ostream os(failure);
    os << evaluation.label << ": "
       << (evaluation.failureReason.empty() ? "failed"
                                            : evaluation.failureReason);
    failures.push_back(std::move(failure));
  }

  if (!frontier.empty())
    return frontier;

  auto diagnostic = sourceModule.emitError()
                    << "no_complete_rank_variant: bounded scheduling search "
                       "found no fully legal rank artifact";
  for (const std::string &failure : failures)
    diagnostic << "\n  - " << failure;
  return mlir::failure();
}

mlir::FailureOr<int64_t>
estimateScheduledRankProgramTimePs(mlir::ModuleOp module) {
  if (!module)
    return mlir::failure();
  CandidateStats stats = estimateStats(module);
  if (std::optional<std::string> failure = getRankingCostFailure(stats)) {
    module.emitError() << "scheduled_rank_cost_failure: " << *failure;
    return mlir::failure();
  }
  return estimateCandidateTimePs(stats);
}

} // namespace wafer
