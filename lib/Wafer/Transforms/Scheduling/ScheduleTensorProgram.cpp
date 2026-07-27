//===- ScheduleTensorProgram.cpp - Closed-loop task scheduling ------------===//

#include "Scheduling/ScheduleTensorProgramInternal.h"
#include "Wafer/Transforms/Passes.h"
#include "Wafer/Transforms/Scheduling/RankCandidateFrontier.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace wafer {
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

struct RankArtifactAlternative {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  unsigned promotedHandoffs = 0;
  bool readyReordered = false;
};

static RankArtifactKind
getRankArtifactKind(const RankArtifactAlternative &alternative) {
  if (alternative.promotedHandoffs != 0)
    return alternative.readyReordered ? RankArtifactKind::ResidentReady
                                      : RankArtifactKind::Resident;
  return alternative.readyReordered ? RankArtifactKind::SpillReady
                                    : RankArtifactKind::Spill;
}

struct RankVariantEvaluation {
  std::string label;
  std::string partitionSignature;
  std::string failureReason;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  llvm::SmallVector<SelectedCandidate, 8> selectedCandidates;
  std::vector<RankArtifactAlternative> alternatives;
  bool accepted = false;
  bool duplicate = false;
  bool noScopes = false;
};

static std::string
getPolicyLabel(const structured_scheduler::ScopeDiscoveryPolicy &policy) {
  std::string label;
  llvm::raw_string_ostream os(label);
  if (policy.cutTerminalFullTraversalOnlyRoots)
    os << "dataflow-terminal-full-only-cut";
  else
    os << (policy.allowCrossShapeDataflow ? "dataflow" : "conservative");
  os << (policy.includeSharedInputPeers ? "/shared-closure" : "/root-closure");
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
/// until the exact same whole-rank SPM, verifier, and cost gates accept it.
/// DDR placement is a whole-variant gate owned by the all-rank coordinator.
static std::optional<std::string>
finalizeRankArtifact(mlir::ModuleOp module, const SelectionConfig &config) {
  if (mlir::failed(mlir::verify(module)))
    return "whole-rank-pre-plan-verifier";
  if (mlir::failed(planSPMMemoryModule(module, config.spmBase, config.spmLimit,
                                       config.spmAlignment)))
    return "whole-rank-spm-offsets";
  if (mlir::failed(mlir::verify(module)))
    return "whole-rank-verifier";

  CandidateStats stats = estimateStats(module);
  if (std::optional<std::string> failure = getRankingCostFailure(stats))
    return (llvm::Twine("whole-rank-cost: ") + *failure).str();
  return std::nullopt;
}

/// Task candidate evaluation is allowed to place its standalone proof module,
/// but those offsets are not semantic choices. Remove them after task commit
/// so every physical alternative starts from an unplaced generation parent.
static void clearCandidateEvaluationFacts(mlir::ModuleOp module) {
  module.walk([](mlir::memref::AllocOp allocation) {
    allocation->removeAttr(kWaferSPMOffsetAttrName);
    allocation->removeAttr(kWaferDDROffsetAttrName);
  });
  module.walk([](mlir::Operation *operation) {
    if (mlir::isa<InstrDTESendOp, InstrDTERecvOp>(operation))
      operation->removeAttr("binding");
  });
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
    if (mlir::failed(selected))
      return failRankVariant(evaluation, "candidate-selection");
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

  clearCandidateEvaluationFacts(*evaluation.module);

  // The committed, unplaced module is the generation parent. Spill and
  // resident alternatives are evaluated on separate clones, so placement and
  // cost observations can never flow back into either generation path.
  mlir::OwningOpRef<mlir::ModuleOp> spillModule =
      mlir::cast<mlir::ModuleOp>((*evaluation.module)->clone());
  mlir::OwningOpRef<mlir::ModuleOp> residentGeneration =
      mlir::cast<mlir::ModuleOp>((*evaluation.module)->clone());
  unsigned promotedHandoffs = promoteFullBufferHandoffs(*residentGeneration);
  mlir::OwningOpRef<mlir::ModuleOp> residentModule;
  if (promotedHandoffs != 0)
    residentModule = mlir::cast<mlir::ModuleOp>((*residentGeneration)->clone());

  // Ready-order variants are actual independently owned instruction modules.
  // The rewrite changes operation order in SSA while preserving value hazards
  // and fences; placement and cost are therefore recomputed from scratch.
  mlir::OwningOpRef<mlir::ModuleOp> spillReadyModule;
  {
    mlir::OwningOpRef<mlir::ModuleOp> candidate =
        mlir::cast<mlir::ModuleOp>((*evaluation.module)->clone());
    if (scheduleIndependentInstructionsByReadyOrder(
            candidate->getOperation()) != 0)
      spillReadyModule = std::move(candidate);
  }
  mlir::OwningOpRef<mlir::ModuleOp> residentReadyModule;
  if (promotedHandoffs != 0) {
    mlir::OwningOpRef<mlir::ModuleOp> candidate =
        mlir::cast<mlir::ModuleOp>((*residentGeneration)->clone());
    if (scheduleIndependentInstructionsByReadyOrder(
            candidate->getOperation()) != 0)
      residentReadyModule = std::move(candidate);
  }

  std::optional<std::string> spillFailure =
      finalizeRankArtifact(*spillModule, config);
  std::optional<std::string> residentFailure;
  if (promotedHandoffs != 0) {
    residentFailure = finalizeRankArtifact(*residentModule, config);
  }
  std::optional<std::string> spillReadyFailure;
  if (spillReadyModule)
    spillReadyFailure = finalizeRankArtifact(*spillReadyModule, config);
  std::optional<std::string> residentReadyFailure;
  if (residentReadyModule)
    residentReadyFailure = finalizeRankArtifact(*residentReadyModule, config);

  if (spillFailure && (promotedHandoffs == 0 || residentFailure)) {
    std::string detail;
    llvm::raw_string_ostream os(detail);
    os << "spill=" << *spillFailure;
    if (promotedHandoffs != 0)
      os << ", resident=" << *residentFailure;
    return failRankVariant(evaluation, "whole-rank-artifact", os.str());
  }

  if (!spillFailure)
    evaluation.alternatives.push_back({std::move(spillModule),
                                       /*promotedHandoffs=*/0,
                                       /*readyReordered=*/false});
  if (spillReadyModule && !spillReadyFailure)
    evaluation.alternatives.push_back({std::move(spillReadyModule),
                                       /*promotedHandoffs=*/0,
                                       /*readyReordered=*/true});
  if (promotedHandoffs != 0 && !residentFailure)
    evaluation.alternatives.push_back({std::move(residentModule),
                                       promotedHandoffs,
                                       /*readyReordered=*/false});
  if (residentReadyModule && !residentReadyFailure)
    evaluation.alternatives.push_back({std::move(residentReadyModule),
                                       promotedHandoffs,
                                       /*readyReordered=*/true});

  evaluation.module = nullptr;
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

} // namespace

namespace {

using SourceProducer = unsigned (*)(mlir::func::FuncOp);

static unsigned applySourceProducer(mlir::ModuleOp module,
                                    SourceProducer producer) {
  unsigned changed = 0;
  module.walk(
      [&](mlir::func::FuncOp function) { changed += producer(function); });
  return changed;
}

static void buildBoundedSourceVariants(
    mlir::ModuleOp source,
    llvm::SmallVectorImpl<mlir::OwningOpRef<mlir::ModuleOp>> &owners,
    llvm::SmallVectorImpl<mlir::ModuleOp> &variants) {
  constexpr unsigned sourceVariantLimit = 16;
  const SourceProducer producers[] = {
      materializeConsumerLocalTensorRecomputation,
      hoistStaticLoopInvariantOperations,
      reassociateElementwiseExpressions,
      balanceElementwiseReductionTrees,
      contractDistributiveExpressions,
      factorElementwiseExpressions,
  };

  variants.push_back(source);
  auto tryAdd = [&](llvm::ArrayRef<unsigned> producerIndices,
                    unsigned minimumAppliedProducers) {
    if (variants.size() >= sourceVariantLimit)
      return;
    mlir::OwningOpRef<mlir::ModuleOp> candidate =
        mlir::cast<mlir::ModuleOp>(source->clone());
    unsigned appliedProducers = 0;
    for (unsigned index : producerIndices)
      appliedProducers +=
          applySourceProducer(*candidate, producers[index]) != 0;
    if (appliedProducers < minimumAppliedProducers ||
        mlir::failed(mlir::verify(*candidate)))
      return;
    owners.push_back(std::move(candidate));
    variants.push_back(*owners.back());
  };

  // Reserve one slot for the all-applicable joint state. Singletons guarantee
  // that every producer reaches the rank frontier; canonical pairs provide a
  // bounded interaction sample without permutation duplicates.
  for (unsigned index = 0; index < std::size(producers); ++index)
    tryAdd({index}, /*minimumAppliedProducers=*/1);
  for (unsigned lhs = 0;
       lhs < std::size(producers) && variants.size() + 1 < sourceVariantLimit;
       ++lhs)
    for (unsigned rhs = lhs + 1;
         rhs < std::size(producers) && variants.size() + 1 < sourceVariantLimit;
         ++rhs)
      tryAdd({lhs, rhs}, /*minimumAppliedProducers=*/2);

  llvm::SmallVector<unsigned, 8> allProducerIndices;
  for (unsigned index = 0; index < std::size(producers); ++index)
    allProducerIndices.push_back(index);
  tryAdd(allProducerIndices, /*minimumAppliedProducers=*/2);
}

static llvm::SmallVector<SelectionConfig, 12>
buildRankSearchRecipes(mlir::ModuleOp source, const SelectionConfig &base) {
  constexpr unsigned recipeLimit = 12;
  bool hasAllGather = false;
  bool hasReduceScatter = false;
  bool hasAllReduce = false;
  llvm::SmallVector<TargetImplementationKind, 2> implementations;
  source.walk([&](mlir::Operation *operation) {
    hasAllGather |=
        mlir::isa<LinalgExtCollectiveAllGatherOp, CommAllGatherOp>(operation);
    hasReduceScatter |=
        mlir::isa<LinalgExtCollectiveReduceScatterOp, CommReduceScatterOp>(
            operation);
    hasAllReduce |=
        mlir::isa<LinalgExtCollectiveAllReduceOp, CommAllReduceOp>(operation);
    if (auto interface =
            mlir::dyn_cast<WaferTargetImplementationOpInterface>(operation)) {
      llvm::SmallVector<TargetImplementationCandidate, 2> candidates;
      interface.collectTargetImplementationCandidates(WaferTargetCapabilities{},
                                                      candidates);
      for (const TargetImplementationCandidate &candidate :
           llvm::drop_begin(candidates))
        if (!llvm::is_contained(implementations, candidate.kind))
          implementations.push_back(candidate.kind);
    }
  });

  llvm::SmallVector<SelectionConfig, 12> recipes;
  auto addRecipe = [&](CommunicationAlternative communication,
                       std::optional<TargetImplementationKind> implementation,
                       unsigned taskOrdinal,
                       bool useDirectMappedBoundaryTransfer = false) {
    if (recipes.size() >= recipeLimit)
      return;
    SelectionConfig recipe = base;
    recipe.communicationAlternative = communication;
    recipe.allowAutomaticImplementationAlternatives = false;
    recipe.forcedImplementationAlternative = implementation;
    recipe.taskAlternativeOrdinal = taskOrdinal;
    recipe.useDirectMappedBoundaryTransfer = useDirectMappedBoundaryTransfer;
    recipes.push_back(std::move(recipe));
  };

  addRecipe(CommunicationAlternative::Ring, std::nullopt, 0);
  if (hasAllGather)
    addRecipe(CommunicationAlternative::DirectAllGather, std::nullopt, 0);
  if (hasReduceScatter)
    addRecipe(CommunicationAlternative::RingReduceScatter, std::nullopt, 0);
  if (hasAllReduce)
    addRecipe(CommunicationAlternative::TreeAllReduce, std::nullopt, 0);
  if (hasAllGather && hasAllReduce)
    addRecipe(CommunicationAlternative::DirectAllGatherTreeAllReduce,
              std::nullopt, 0);
  for (TargetImplementationKind implementation : implementations)
    addRecipe(CommunicationAlternative::Ring, implementation, 0);
  addRecipe(CommunicationAlternative::Ring, std::nullopt, 1);
  addRecipe(CommunicationAlternative::Ring, std::nullopt, 0,
            /*useDirectMappedBoundaryTransfer=*/true);
  for (TargetImplementationKind implementation : implementations)
    addRecipe(CommunicationAlternative::Ring, implementation, 0,
              /*useDirectMappedBoundaryTransfer=*/true);

  // Cross the currently supported non-baseline implementation and task beam
  // with communication parameters while the fixed recipe cap has room.
  for (TargetImplementationKind implementation : implementations) {
    if (hasAllGather)
      addRecipe(CommunicationAlternative::DirectAllGather, implementation, 0);
    if (hasReduceScatter)
      addRecipe(CommunicationAlternative::RingReduceScatter, implementation, 0);
    if (hasAllReduce)
      addRecipe(CommunicationAlternative::TreeAllReduce, implementation, 0);
    if (hasAllGather)
      addRecipe(CommunicationAlternative::DirectAllGather, implementation, 0,
                /*useDirectMappedBoundaryTransfer=*/true);
    if (hasAllReduce)
      addRecipe(CommunicationAlternative::TreeAllReduce, implementation, 0,
                /*useDirectMappedBoundaryTransfer=*/true);
  }
  if (hasAllGather)
    addRecipe(CommunicationAlternative::DirectAllGather, std::nullopt, 1);
  if (hasReduceScatter)
    addRecipe(CommunicationAlternative::RingReduceScatter, std::nullopt, 1);
  if (hasAllReduce)
    addRecipe(CommunicationAlternative::TreeAllReduce, std::nullopt, 1);
  if (hasAllGather)
    addRecipe(CommunicationAlternative::DirectAllGather, std::nullopt, 0,
              /*useDirectMappedBoundaryTransfer=*/true);
  if (hasAllReduce)
    addRecipe(CommunicationAlternative::TreeAllReduce, std::nullopt, 0,
              /*useDirectMappedBoundaryTransfer=*/true);
  for (TargetImplementationKind implementation : implementations)
    addRecipe(CommunicationAlternative::Ring, implementation, 1);
  return recipes;
}

struct RankEvaluationRequest {
  unsigned sourceIndex = 0;
  unsigned recipeIndex = 0;
  unsigned policyIndex = 0;

  friend bool operator==(const RankEvaluationRequest &lhs,
                         const RankEvaluationRequest &rhs) {
    return lhs.sourceIndex == rhs.sourceIndex &&
           lhs.recipeIndex == rhs.recipeIndex &&
           lhs.policyIndex == rhs.policyIndex;
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
  config.logicalRank = frontierConfig.logicalRank;
  config.candidateParallelism = frontierConfig.candidateParallelism;
  std::unique_ptr<CandidateEvaluationExecutor> evaluationExecutor;
  if (config.candidateParallelism > 1) {
    evaluationExecutor = std::make_unique<CandidateEvaluationExecutor>(
        static_cast<unsigned>(config.candidateParallelism));
    config.evaluationExecutor = evaluationExecutor.get();
  }

  // Stable ordinals identify invocation-local semantic generations across
  // ranks. RankArtifactKind separately identifies the spill/resident and
  // ready-order derivation; no partial shared-input prefix is a candidate
  // protocol.
  const structured_scheduler::ScopeDiscoveryPolicy policies[] = {
      {/*includeSharedInputPeers=*/false, /*allowCrossShapeDataflow=*/true,
       /*cutTerminalFullTraversalOnlyRoots=*/false},
      {/*includeSharedInputPeers=*/true, /*allowCrossShapeDataflow=*/true,
       /*cutTerminalFullTraversalOnlyRoots=*/false},
      {/*includeSharedInputPeers=*/false, /*allowCrossShapeDataflow=*/true,
       /*cutTerminalFullTraversalOnlyRoots=*/true},
      {/*includeSharedInputPeers=*/false, /*allowCrossShapeDataflow=*/false,
       /*cutTerminalFullTraversalOnlyRoots=*/false},
  };

  constexpr unsigned conservativePolicyIndex = 3;
  constexpr unsigned optimizedRankEvaluationLimit = 64;
  constexpr unsigned optimizedRankFrontierLimit = 256;
  constexpr unsigned stableRecipeStride = 12;

  llvm::SmallVector<mlir::OwningOpRef<mlir::ModuleOp>, 16> sourceOwners;
  llvm::SmallVector<mlir::ModuleOp, 16> generationSources;
  buildBoundedSourceVariants(sourceModule, sourceOwners, generationSources);
  llvm::SmallVector<SelectionConfig, 12> recipes =
      buildRankSearchRecipes(sourceModule, config);

  llvm::SmallVector<std::string, 16> failures;
  std::vector<ScheduledRankCandidate> frontier;
  frontier.reserve(optimizedRankFrontierLimit + 1);
  unsigned reservedBaselineCount = 0;
  unsigned optimizedFrontierCount = 0;
  auto appendAccepted = [&](RankVariantEvaluation &evaluation,
                            int64_t stableOrdinal,
                            bool reservedPolicy) -> mlir::LogicalResult {
    bool hasBaselineSpill = false;
    for (RankArtifactAlternative &alternative : evaluation.alternatives) {
      bool isBaselineSpill =
          alternative.promotedHandoffs == 0 && !alternative.readyReordered;
      bool reserved = reservedPolicy && isBaselineSpill;
      hasBaselineSpill |= isBaselineSpill;
      reservedBaselineCount += reserved;
      if (optimizedFrontierCount >= optimizedRankFrontierLimit && !reserved)
        continue;
      frontier.emplace_back(std::move(alternative.module), stableOrdinal,
                            getRankArtifactKind(alternative), reserved);
      optimizedFrontierCount += !reserved;
    }
    return !reservedPolicy || hasBaselineSpill ? mlir::success()
                                               : mlir::failure();
  };

  // The conservative spill tuple has an allowance outside every optimization
  // cap. It must exist before producer/recipe work can add frontier members.
  llvm::StringSet<> reservedSeenPartitions;
  RankVariantEvaluation reserved = evaluateRankVariant(
      generationSources.front(), recipes.front(),
      policies[conservativePolicyIndex], reservedSeenPartitions);
  if (reserved.noScopes) {
    clearCandidateEvaluationFacts(*reserved.module);
    frontier.emplace_back(std::move(reserved.module),
                          /*stableOrdinal=*/conservativePolicyIndex,
                          RankArtifactKind::Spill,
                          /*reservedBaseline=*/true);
    return frontier;
  }
  if (!reserved.accepted ||
      mlir::failed(appendAccepted(reserved,
                                  /*stableOrdinal=*/conservativePolicyIndex,
                                  /*reservedPolicy=*/true))) {
    sourceModule.emitError()
        << "reserved_baseline_unavailable: conservative spill artifact did "
           "not pass rank-local exact gates"
        << (reserved.failureReason.empty() ? "" : ": ")
        << reserved.failureReason;
    return mlir::failure();
  }

  llvm::SmallVector<RankEvaluationRequest, 32> requests;
  auto addRequest = [&](unsigned sourceIndex, unsigned recipeIndex,
                        unsigned policyIndex) {
    RankEvaluationRequest request{sourceIndex, recipeIndex, policyIndex};
    if (requests.size() >= optimizedRankEvaluationLimit ||
        llvm::is_contained(requests, request) ||
        (sourceIndex == 0 && recipeIndex == 0 &&
         policyIndex == conservativePolicyIndex))
      return;
    requests.push_back(request);
  };

  // Stable seed order first admits every source producer, every task/
  // communication recipe, and every scope policy. Remaining allowance then
  // samples their Cartesian product in canonical source/recipe/policy order.
  for (unsigned sourceIndex = 0; sourceIndex < generationSources.size();
       ++sourceIndex)
    addRequest(sourceIndex, /*recipeIndex=*/0, /*policyIndex=*/0);
  for (unsigned recipeIndex = 0; recipeIndex < recipes.size(); ++recipeIndex)
    addRequest(/*sourceIndex=*/0, recipeIndex, /*policyIndex=*/0);
  for (unsigned policyIndex = 0; policyIndex < std::size(policies);
       ++policyIndex)
    addRequest(/*sourceIndex=*/0, /*recipeIndex=*/0, policyIndex);

  // Consume the remaining fixed budget on genuine joint states before the
  // canonical Cartesian tail. The all-applicable source crosses the far ends
  // of both other axes, and every independently materialized source crosses
  // one non-baseline recipe and one non-baseline scope policy when present.
  if (generationSources.size() > 1 && recipes.size() > 1 &&
      std::size(policies) > 1)
    addRequest(generationSources.size() - 1, recipes.size() - 1,
               std::size(policies) - 1);
  if (recipes.size() > 1)
    for (unsigned sourceIndex = 1; sourceIndex < generationSources.size();
         ++sourceIndex)
      addRequest(sourceIndex, /*recipeIndex=*/1, /*policyIndex=*/0);
  if (std::size(policies) > 1)
    for (unsigned sourceIndex = 1; sourceIndex < generationSources.size();
         ++sourceIndex)
      addRequest(sourceIndex, /*recipeIndex=*/0, /*policyIndex=*/1);
  for (unsigned sourceIndex = 1; sourceIndex < generationSources.size();
       ++sourceIndex)
    for (unsigned recipeIndex = 1; recipeIndex < recipes.size(); ++recipeIndex)
      addRequest(sourceIndex, recipeIndex, /*policyIndex=*/0);
  for (unsigned sourceIndex = 0; sourceIndex < generationSources.size();
       ++sourceIndex)
    for (unsigned recipeIndex = 0; recipeIndex < recipes.size(); ++recipeIndex)
      for (unsigned policyIndex = 0; policyIndex < std::size(policies);
           ++policyIndex)
        addRequest(sourceIndex, recipeIndex, policyIndex);

  std::vector<llvm::StringSet<>> seenPartitions(generationSources.size() *
                                                recipes.size());
  for (const RankEvaluationRequest &request : requests) {
    if (optimizedFrontierCount >= optimizedRankFrontierLimit)
      break;
    llvm::StringSet<> &seen =
        seenPartitions[request.sourceIndex * recipes.size() +
                       request.recipeIndex];
    RankVariantEvaluation evaluation = evaluateRankVariant(
        generationSources[request.sourceIndex], recipes[request.recipeIndex],
        policies[request.policyIndex], seen);
    if (evaluation.noScopes || evaluation.duplicate)
      continue;
    int64_t stableOrdinal = static_cast<int64_t>(
        (request.sourceIndex * stableRecipeStride + request.recipeIndex) *
            std::size(policies) +
        request.policyIndex);
    if (evaluation.accepted) {
      (void)appendAccepted(evaluation, stableOrdinal,
                           /*reservedPolicy=*/false);
      continue;
    }
    if (failures.size() < 16) {
      std::string failure;
      llvm::raw_string_ostream os(failure);
      os << evaluation.label << ": "
         << (evaluation.failureReason.empty() ? "failed"
                                              : evaluation.failureReason);
      failures.push_back(std::move(failure));
    }
  }

  if (reservedBaselineCount == 1)
    return frontier;

  auto diagnostic = sourceModule.emitError()
                    << "no_complete_rank_variant: bounded scheduling search "
                       "found no fully legal rank artifact";
  for (const std::string &failure : failures)
    diagnostic << "\n  - " << failure;
  return mlir::failure();
}

} // namespace wafer
