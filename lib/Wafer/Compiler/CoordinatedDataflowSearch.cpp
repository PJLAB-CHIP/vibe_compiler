//===- CoordinatedDataflowSearch.cpp - All-rank Tile frontier ------------===//

#include "CoordinatedDataflowSearch.h"

#include "BoundedRankExecutor.h"
#include "CoordinatedStructuredCandidateDerivation.h"

#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/TargetPolicy.h"
#include "Wafer/Transforms/CompleteRankMaterialization.h"
#include "Wafer/Transforms/PhysicalDataflow.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>
#include <tuple>

namespace wafer::compiler::detail {
namespace {

static std::optional<uint64_t> checkedAdd(uint64_t left, uint64_t right) {
  if (right > std::numeric_limits<uint64_t>::max() - left)
    return std::nullopt;
  return left + right;
}

static std::optional<uint64_t> checkedMultiply(uint64_t left, uint64_t right) {
  if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left)
    return std::nullopt;
  return left * right;
}

static bool isWithinEstimate(const CoordinatedWorkEstimate &actual,
                             const CoordinatedWorkEstimate &upperBound) {
  for (size_t index = 0; index < kCoordinatedWorkKindCount; ++index)
    if (actual.counts[index] > upperBound.counts[index])
      return false;
  return true;
}

static bool containsInstructionSemantics(mlir::ModuleOp module) {
  bool found = false;
  module.walk([&](mlir::Operation *operation) {
    if (!mlir::isa<WaferInstructionOpInterface, SyncNCCJoinOp>(operation))
      return mlir::WalkResult::advance();
    found = true;
    return mlir::WalkResult::interrupt();
  });
  return found;
}

static std::optional<llvm::SmallVector<int64_t, 4>>
getCommonStaticResultShape(mlir::ModuleOp module) {
  mlir::func::FuncOp function;
  for (mlir::func::FuncOp candidate : module.getOps<mlir::func::FuncOp>()) {
    if (candidate.isExternal())
      continue;
    if (function)
      return std::nullopt;
    function = candidate;
  }
  if (!function || function.getNumResults() == 0)
    return std::nullopt;

  std::optional<llvm::SmallVector<int64_t, 4>> commonShape;
  for (mlir::Type type : function.getResultTypes()) {
    auto ranked = mlir::dyn_cast<mlir::RankedTensorType>(type);
    if (!ranked || !ranked.hasStaticShape() || ranked.getRank() == 0 ||
        llvm::any_of(ranked.getShape(),
                     [](int64_t extent) { return extent <= 0; }))
      return std::nullopt;
    llvm::SmallVector<int64_t, 4> shape(ranked.getShape().begin(),
                                        ranked.getShape().end());
    if (!commonShape) {
      commonShape = std::move(shape);
      continue;
    }
    if (commonShape->size() != shape.size())
      return std::nullopt;
    for (size_t index = 0; index < shape.size(); ++index)
      (*commonShape)[index] = std::min((*commonShape)[index], shape[index]);
  }
  return commonShape;
}

/// Returns the static reduction-loop domain of a direct function result, or
/// of the exact single-use local reduction feeding a typed all-reduce result.
/// This query is intentionally rooted in SSA/results: unrelated reductions in
/// a larger function cannot seed a partial-reduction candidate.
static std::optional<llvm::SmallVector<int64_t, 2>>
getReturnedStaticReductionShape(mlir::ModuleOp module) {
  mlir::func::FuncOp function;
  for (mlir::func::FuncOp candidate : module.getOps<mlir::func::FuncOp>()) {
    if (candidate.isExternal())
      continue;
    if (function)
      return std::nullopt;
    function = candidate;
  }
  if (!function || !function.getBody().hasOneBlock())
    return std::nullopt;
  auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(
      function.getBody().front().getTerminator());
  if (!returnOp)
    return std::nullopt;

  std::optional<llvm::SmallVector<int64_t, 2>> commonReductionShape;
  for (mlir::Value result : returnOp.getOperands()) {
    mlir::Operation *root = result.getDefiningOp();
    if (auto allReduce =
            mlir::dyn_cast_or_null<LinalgExtCollectiveAllReduceOp>(root)) {
      if (allReduce.getInputs().size() != 1 ||
          !allReduce.getInputs().front().hasOneUse())
        continue;
      root = allReduce.getInputs().front().getDefiningOp();
    }
    auto linalg = mlir::dyn_cast_or_null<mlir::linalg::LinalgOp>(root);
    auto tiling = mlir::dyn_cast_or_null<mlir::TilingInterface>(root);
    auto partial =
        mlir::dyn_cast_or_null<mlir::PartialReductionOpInterface>(root);
    if (!linalg || !tiling || !partial)
      continue;
    llvm::SmallVector<int64_t, 4> loopRanges = linalg.getStaticLoopRanges();
    llvm::SmallVector<mlir::utils::IteratorType, 4> iteratorTypes =
        tiling.getLoopIteratorTypes();
    if (loopRanges.size() != iteratorTypes.size())
      return std::nullopt;
    llvm::SmallVector<int64_t, 2> reductionShape;
    for (auto [range, iteratorType] :
         llvm::zip_equal(loopRanges, iteratorTypes)) {
      if (iteratorType != mlir::utils::IteratorType::reduction)
        continue;
      if (mlir::ShapedType::isDynamic(range) || range <= 0)
        return std::nullopt;
      reductionShape.push_back(range);
    }
    if (reductionShape.empty())
      continue;
    if (commonReductionShape && *commonReductionShape != reductionShape)
      return std::nullopt;
    commonReductionShape = std::move(reductionShape);
  }
  return commonReductionShape;
}

static std::vector<StructuredTraversalProposal>
buildStructuredTraversalProposals(mlir::ModuleOp sourceModule) {
  std::optional<llvm::SmallVector<int64_t, 4>> shape =
      getCommonStaticResultShape(sourceModule);
  std::vector<StructuredTraversalProposal> proposals;
  int64_t nextOrdinal = 1;

  auto buildTileVectors = [&](llvm::ArrayRef<int64_t> domainShape) {
    llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 3> vectors;
    auto appendUnique = [&](llvm::SmallVector<int64_t, 4> tiles) {
      if (!llvm::is_contained(vectors, tiles))
        vectors.push_back(std::move(tiles));
    };
    appendUnique(llvm::SmallVector<int64_t, 4>(domainShape.size(), 1));
    llvm::SmallVector<int64_t, 4> full(domainShape.begin(), domainShape.end());
    const WaferTargetPolicy policy =
        getDefaultWaferTargetPolicy(TileSearchEffort::Default);
    for (int64_t preferred : policy.tileSearch.preferredTileSizes) {
      llvm::SmallVector<int64_t, 4> tiles;
      tiles.reserve(domainShape.size());
      for (int64_t extent : domainShape)
        tiles.push_back(std::min(extent, preferred));
      if (tiles != full &&
          tiles != llvm::SmallVector<int64_t, 4>(domainShape.size(), 1)) {
        appendUnique(std::move(tiles));
        break;
      }
    }
    appendUnique(std::move(full));
    return vectors;
  };

  llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 3> tileVectors;
  llvm::SmallVector<int64_t, 4> full;
  if (shape) {
    tileVectors = buildTileVectors(*shape);
    full.assign(shape->begin(), shape->end());
  }

  if (shape) {
    for (const llvm::SmallVector<int64_t, 4> &tiles : tileVectors)
      proposals.push_back({nextOrdinal++,
                           CompleteRankTraversalComposition::Coupled,
                           CandidateTileTraversalKind::ResultDriven,
                           CandidateTileResidencyAction::KeepSingleRegion,
                           CandidateBoundaryMovementAction::Staged,
                           CandidateLoopMovementAction::AsConstructed, tiles});
  }
  if (shape) {
    for (const llvm::SmallVector<int64_t, 4> &tiles : tileVectors)
      proposals.push_back({nextOrdinal++,
                           CompleteRankTraversalComposition::Coupled,
                           CandidateTileTraversalKind::ResultDriven,
                           CandidateTileResidencyAction::KeepSingleRegion,
                           CandidateBoundaryMovementAction::ExactDirectMapped,
                           CandidateLoopMovementAction::AsConstructed, tiles});
  }
  if (shape) {
    const llvm::SmallVector<int64_t, 4> &tiles =
        tileVectors.size() > 1 ? tileVectors[1] : tileVectors.front();
    if (tiles != full) {
      proposals.push_back(
          {nextOrdinal++, CompleteRankTraversalComposition::Coupled,
           CandidateTileTraversalKind::ResultDriven,
           CandidateTileResidencyAction::KeepSingleRegion,
           CandidateBoundaryMovementAction::Staged,
           CandidateLoopMovementAction::HoistInvariantReadOnlyBoundary, tiles});
      proposals.push_back(
          {nextOrdinal++, CompleteRankTraversalComposition::Coupled,
           CandidateTileTraversalKind::ResultDriven,
           CandidateTileResidencyAction::KeepSingleRegion,
           CandidateBoundaryMovementAction::ExactDirectMapped,
           CandidateLoopMovementAction::HoistInvariantReadOnlyBoundary, tiles});
    }
  }
  if (shape) {
    for (const llvm::SmallVector<int64_t, 4> &tiles :
         llvm::drop_begin(tileVectors))
      proposals.push_back({nextOrdinal++,
                           CompleteRankTraversalComposition::Separated,
                           CandidateTileTraversalKind::ResultDriven,
                           CandidateTileResidencyAction::KeepSingleRegion,
                           CandidateBoundaryMovementAction::Staged,
                           CandidateLoopMovementAction::AsConstructed, tiles});
  }
  // These are bounded complete-state actions, not another independent axis:
  // one large separated traversal gets a real DDR-clean region cut, and one
  // large coupled traversal gets one selective root spill. Failed actions are
  // simply absent from the actual frontier.
  if (shape)
    proposals.push_back(
        {nextOrdinal++, CompleteRankTraversalComposition::Separated,
         CandidateTileTraversalKind::ResultDriven,
         CandidateTileResidencyAction::SplitAtExplicitDDRBoundary,
         CandidateBoundaryMovementAction::Staged,
         CandidateLoopMovementAction::AsConstructed, full});
  if (shape)
    proposals.push_back({nextOrdinal++,
                         CompleteRankTraversalComposition::Coupled,
                         CandidateTileTraversalKind::ResultDriven,
                         CandidateTileResidencyAction::SelectiveSpill,
                         CandidateBoundaryMovementAction::Staged,
                         CandidateLoopMovementAction::AsConstructed, full});

  // Native/full-domain reduction is already represented by the ordinary
  // result-driven candidate. Add one structural partial/two-pass sibling only
  // when the returned SSA root proves a static PartialReduction interface.
  // The chunk model is versionless policy local to this invocation: prefer the
  // target's first tile quantum and otherwise bisect a small finite domain.
  // Exact numeric legality remains owned by PartialReductionOpInterface, and
  // failed materialization never enters the frontier.
  if (shape) {
    std::optional<llvm::SmallVector<int64_t, 2>> reductionShape =
        getReturnedStaticReductionShape(sourceModule);
    if (reductionShape) {
      const WaferTargetPolicy policy =
          getDefaultWaferTargetPolicy(TileSearchEffort::Default);
      const int64_t preferred =
          policy.tileSearch.preferredTileSizes.empty()
              ? 1
              : policy.tileSearch.preferredTileSizes.front();
      llvm::SmallVector<int64_t, 2> reductionTiles;
      bool splitsDomain = false;
      for (int64_t extent : *reductionShape) {
        int64_t tile = std::min(extent, std::max<int64_t>(1, preferred));
        if (tile == extent && extent > 1)
          tile = (extent + 1) / 2;
        splitsDomain |= tile < extent;
        reductionTiles.push_back(tile);
      }
      if (splitsDomain) {
        StructuredTraversalProposal proposal;
        proposal.stableSemanticOrdinal = nextOrdinal++;
        proposal.composition = CompleteRankTraversalComposition::Coupled;
        // A non-empty reduction domain is the typed request; ordinary
        // ResultDriven materialization proves and consumes the current
        // PartialReductionOpInterface without a semantic enum branch.
        proposal.traversalKind = CandidateTileTraversalKind::ResultDriven;
        const llvm::SmallVector<int64_t, 4> &outputTiles =
            tileVectors.size() > 1 ? tileVectors[1] : tileVectors.front();
        proposal.tileSizes.assign(outputTiles.begin(), outputTiles.end());
        proposal.reductionTileSizes = std::move(reductionTiles);
        proposals.push_back(std::move(proposal));
      }
    }
  }

  // Implementation and layout alternatives are materialized directly into a
  // complete-rank Tile clone. The interface query and bounded PBQP projection
  // are invocation-local proposal mechanisms; neither chooses a rank-local
  // winner or survives in the accepted IR.
  if (shape) {
    llvm::SmallVector<TargetImplementationKind, 2> implementations;
    sourceModule.walk([&](mlir::Operation *operation) {
      auto interface =
          mlir::dyn_cast<WaferTargetImplementationOpInterface>(operation);
      if (!interface)
        return;
      llvm::SmallVector<TargetImplementationCandidate, 2> candidates;
      interface.collectTargetImplementationCandidates(WaferTargetCapabilities{},
                                                      candidates);
      for (const TargetImplementationCandidate &candidate :
           llvm::drop_begin(candidates))
        if (!llvm::is_contained(implementations, candidate.kind))
          implementations.push_back(candidate.kind);
    });

    auto appendTypedProposal =
        [&](std::optional<TargetImplementationKind> implementation,
            std::optional<unsigned> layoutOrdinal,
            llvm::ArrayRef<int64_t> candidateTiles, bool cloneBaseline,
            CompleteRankTraversalComposition composition) {
          StructuredTraversalProposal proposal;
          proposal.stableSemanticOrdinal = nextOrdinal++;
          proposal.composition = composition;
          proposal.tileSizes.assign(candidateTiles.begin(),
                                    candidateTiles.end());
          proposal.selectedImplementation = implementation;
          proposal.physicalLayoutProposalOrdinal = layoutOrdinal;
          proposal.cloneReservedBaseline = cloneBaseline;
          proposals.push_back(std::move(proposal));
        };
    CompleteRankTraversalComposition composition =
        CompleteRankTraversalComposition::Coupled;
    appendTypedProposal(std::nullopt, std::nullopt, full,
                        /*cloneBaseline=*/false, composition);
    for (TargetImplementationKind implementation : implementations)
      appendTypedProposal(implementation, std::nullopt, full,
                          /*cloneBaseline=*/false, composition);
    for (unsigned layoutOrdinal = 0; layoutOrdinal < 4; ++layoutOrdinal)
      appendTypedProposal(std::nullopt, layoutOrdinal, full,
                          /*cloneBaseline=*/true, composition);
    for (unsigned layoutOrdinal = 0; layoutOrdinal < 4; ++layoutOrdinal)
      appendTypedProposal(std::nullopt, layoutOrdinal, full,
                          /*cloneBaseline=*/false, composition);
    for (TargetImplementationKind implementation : implementations)
      for (unsigned layoutOrdinal = 0; layoutOrdinal < 4; ++layoutOrdinal)
        appendTypedProposal(implementation, layoutOrdinal, full,
                            /*cloneBaseline=*/false, composition);
  }
  return proposals;
}

static std::string joinIntegers(llvm::ArrayRef<int64_t> values) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  llvm::interleaveComma(values, stream);
  stream.flush();
  return text;
}

static CoordinatedStructuredFrontierFacts
makeUnknownPrecloneFacts(llvm::StringRef physicalVersions,
                         llvm::StringRef loopReuseAndEffects) {
  CoordinatedStructuredFrontierFacts facts;
  for (CoordinatedStructuredMetric &metric : facts.selection) {
    metric.knowledge = CoordinatedStructuredFactKnowledge::Unknown;
    metric.disposition = "pre-materialization-current-ir";
  }
  facts.futureLiveInterface = "complete-rank-current-source";
  facts.physicalVersions = physicalVersions.str();
  facts.loopReuseAndEffects = loopReuseAndEffects.str();
  facts.collectiveAndPeerInterface = "current-source-typed-communication";
  return facts;
}

static std::string
getTraversalProposalKey(const StructuredTraversalProposal &proposal) {
  std::string key;
  llvm::raw_string_ostream stream(key);
  stream << "traversal:composition="
         << static_cast<unsigned>(proposal.composition)
         << ":kind=" << static_cast<unsigned>(proposal.traversalKind)
         << ":residency=" << static_cast<unsigned>(proposal.residencyAction)
         << ":boundary="
         << static_cast<unsigned>(proposal.boundaryMovementAction)
         << ":loop=" << static_cast<unsigned>(proposal.loopMovementAction)
         << ":tiles=" << joinIntegers(proposal.tileSizes)
         << ":reduction=" << joinIntegers(proposal.reductionTileSizes)
         << ":clone-baseline=" << proposal.cloneReservedBaseline;
  if (proposal.selectedImplementation)
    stream << ":implementation="
           << static_cast<unsigned>(*proposal.selectedImplementation);
  if (proposal.physicalLayoutProposalOrdinal)
    stream << ":layout=" << *proposal.physicalLayoutProposalOrdinal;
  stream.flush();
  return key;
}

static std::string
getTraversalCoverageClass(const StructuredTraversalProposal &proposal) {
  std::string coverage;
  llvm::raw_string_ostream stream(coverage);
  stream << "traversal:" << static_cast<unsigned>(proposal.traversalKind) << ':'
         << static_cast<unsigned>(proposal.composition) << ':'
         << static_cast<unsigned>(proposal.residencyAction) << ':'
         << static_cast<unsigned>(proposal.boundaryMovementAction) << ':'
         << static_cast<unsigned>(proposal.loopMovementAction) << ':'
         << !proposal.reductionTileSizes.empty() << ':'
         << static_cast<bool>(proposal.selectedImplementation) << ':'
         << static_cast<bool>(proposal.physicalLayoutProposalOrdinal);
  stream.flush();
  return coverage;
}

static CoordinatedStructuralProposal
makeTraversalStructuralProposal(StructuredTraversalProposal proposal) {
  CoordinatedStructuralProposal structural;
  structural.stableSemanticOrdinal = proposal.stableSemanticOrdinal;
  structural.recipe.kind = CoordinatedStructuralRecipeKind::Traversal;
  structural.canonicalKey = getTraversalProposalKey(proposal);
  structural.coverageClass = getTraversalCoverageClass(proposal);
  std::string physical =
      (llvm::Twine("traversal-physical:") +
       llvm::Twine(static_cast<unsigned>(proposal.residencyAction)) + ":" +
       llvm::Twine(static_cast<unsigned>(proposal.boundaryMovementAction)) +
       ":" +
       llvm::Twine(
           proposal.selectedImplementation
               ? static_cast<uint64_t>(*proposal.selectedImplementation) + 1
               : 0) +
       ":" +
       llvm::Twine(proposal.physicalLayoutProposalOrdinal
                       ? static_cast<uint64_t>(
                             *proposal.physicalLayoutProposalOrdinal) +
                             1
                       : 0))
          .str();
  std::string loops =
      (llvm::Twine("traversal-loop:") +
       llvm::Twine(static_cast<unsigned>(proposal.traversalKind)) + ":" +
       llvm::Twine(static_cast<unsigned>(proposal.composition)) + ":" +
       llvm::Twine(static_cast<unsigned>(proposal.loopMovementAction)) + ":" +
       joinIntegers(proposal.tileSizes) + ":" +
       joinIntegers(proposal.reductionTileSizes))
          .str();
  structural.facts = makeUnknownPrecloneFacts(physical, loops);
  structural.recipe.traversal = std::move(proposal);
  return structural;
}

static std::string getConnectionActionsKey(
    llvm::ArrayRef<CandidateTraversalConnectionChoice> choices) {
  std::string key;
  llvm::raw_string_ostream stream(key);
  stream << "connections";
  for (const CandidateTraversalConnectionChoice &choice : choices)
    stream << ':' << static_cast<unsigned>(choice.action) << '['
           << joinIntegers(choice.producerTileSizes) << "]>["
           << joinIntegers(choice.consumerTileSizes) << ']';
  stream.flush();
  return key;
}

static std::string getConnectionCoverageClass(
    llvm::ArrayRef<CandidateTraversalConnectionChoice> choices) {
  std::array<bool, 5> present = {};
  bool hasExplicitProducer = false;
  bool hasIndependentTiles = false;
  bool hasUnitProducer = false;
  bool hasUnitConsumer = false;
  bool hasNonUnitProducer = false;
  bool hasNonUnitConsumer = false;
  for (const CandidateTraversalConnectionChoice &choice : choices) {
    unsigned action = static_cast<unsigned>(choice.action);
    if (action < present.size())
      present[action] = true;
    hasExplicitProducer |= !choice.producerTileSizes.empty();
    hasIndependentTiles |= !choice.producerTileSizes.empty() &&
                           choice.producerTileSizes != choice.consumerTileSizes;
    const bool producerUnit =
        !choice.producerTileSizes.empty() &&
        llvm::all_of(choice.producerTileSizes,
                     [](int64_t tile) { return tile == 1; });
    const bool consumerUnit =
        !choice.consumerTileSizes.empty() &&
        llvm::all_of(choice.consumerTileSizes,
                     [](int64_t tile) { return tile == 1; });
    hasUnitProducer |= producerUnit;
    hasUnitConsumer |= consumerUnit;
    hasNonUnitProducer |= !choice.producerTileSizes.empty() && !producerUnit;
    hasNonUnitConsumer |= !choice.consumerTileSizes.empty() && !consumerUnit;
  }
  std::string coverage = "connections:";
  for (bool value : present)
    coverage.push_back(value ? '1' : '0');
  coverage +=
      (llvm::Twine(":producer=") +
       llvm::Twine(static_cast<unsigned>(hasExplicitProducer)) +
       ":independent=" +
       llvm::Twine(static_cast<unsigned>(hasIndependentTiles)) +
       ":unit-producer=" + llvm::Twine(static_cast<unsigned>(hasUnitProducer)) +
       ":unit-consumer=" + llvm::Twine(static_cast<unsigned>(hasUnitConsumer)) +
       ":nonunit-producer=" +
       llvm::Twine(static_cast<unsigned>(hasNonUnitProducer)) +
       ":nonunit-consumer=" +
       llvm::Twine(static_cast<unsigned>(hasNonUnitConsumer)))
          .str();
  return coverage;
}

static CoordinatedStructuralProposal makeConnectionStructuralProposal(
    int64_t stableOrdinal,
    llvm::ArrayRef<CandidateTraversalConnectionChoice> choices,
    uint64_t estimatedPeakResidentBytes, uint64_t estimatedDDRReadBytes,
    uint64_t estimatedDDRWriteBytes, uint64_t estimatedDDRIssueWork,
    uint64_t estimatedTraversalTrips,
    uint64_t estimatedTileUnderutilizationElements,
    uint64_t estimatedComputeElements, uint64_t estimatedCoordinationPenalty,
    llvm::StringRef futureVersionSignature, bool estimateOverflow) {
  CoordinatedStructuralProposal structural;
  structural.stableSemanticOrdinal = stableOrdinal;
  structural.recipe.kind = CoordinatedStructuralRecipeKind::Connections;
  structural.recipe.connections.assign(choices.begin(), choices.end());
  structural.canonicalKey = getConnectionActionsKey(choices);
  structural.coverageClass = getConnectionCoverageClass(choices);
  structural.facts = makeUnknownPrecloneFacts(
      futureVersionSignature, "connection-traversal-current-source");
  auto setEstimate = [&](CoordinatedStructuredDimension dimension,
                         uint64_t value) {
    CoordinatedStructuredMetric &metric =
        structural.facts.selection[static_cast<size_t>(dimension)];
    metric.value = value;
    metric.knowledge = estimateOverflow
                           ? CoordinatedStructuredFactKnowledge::Overflow
                           : CoordinatedStructuredFactKnowledge::Estimated;
    metric.disposition = estimateOverflow
                             ? "connection-estimate-overflow"
                             : kCoordinatedConnectionProposalEstimateModel;
  };
  setEstimate(CoordinatedStructuredDimension::ResourcePressure,
              estimatedPeakResidentBytes);
  setEstimate(CoordinatedStructuredDimension::DDRAggregateReadBytes,
              estimatedDDRReadBytes);
  setEstimate(CoordinatedStructuredDimension::DDRAggregateWriteBytes,
              estimatedDDRWriteBytes);
  setEstimate(CoordinatedStructuredDimension::DDRMaximumRankIssueWork,
              estimatedDDRIssueWork);
  setEstimate(CoordinatedStructuredDimension::InstrAggregateWork,
              estimatedTraversalTrips);
  setEstimate(CoordinatedStructuredDimension::TileUnderutilization,
              estimatedTileUnderutilizationElements);
  setEstimate(CoordinatedStructuredDimension::ComputeAggregateWork,
              estimatedComputeElements);
  setEstimate(CoordinatedStructuredDimension::ComputeMaximumRankWork,
              estimatedComputeElements);
  setEstimate(CoordinatedStructuredDimension::AllRankCouplingWork,
              estimatedCoordinationPenalty);
  return structural;
}

static CoordinatedStructuralProposal makeImplementationStructuralProposal(
    int64_t stableOrdinal, llvm::StringRef providerKey,
    std::shared_ptr<const StructuredImplementationAlternativePoint> point,
    CandidateTileResidencyAction postResidencyAction,
    int64_t logicalRankCount) {
  CoordinatedStructuralProposal structural;
  structural.stableSemanticOrdinal = stableOrdinal;
  structural.recipe.kind =
      CoordinatedStructuralRecipeKind::ImplementationAlternative;
  structural.canonicalKey =
      (llvm::Twine("implementation-provider:") + providerKey + ":" +
       point->getIdentity().stableKey +
       ":residency=" + llvm::Twine(static_cast<unsigned>(postResidencyAction)))
          .str();
  const StructuredAlternativeParameters &parameters = point->getParameters();
  llvm::StringRef parallelismCoverage =
      parameters.parallelPartitionCount <= 1 ? "serial" : "partitioned";
  structural.coverageClass =
      (llvm::Twine("implementation-provider:") + providerKey +
       ":parallelism=" + parallelismCoverage +
       ":residency=" + llvm::Twine(static_cast<unsigned>(postResidencyAction)))
          .str();
  structural.facts = makeUnknownPrecloneFacts(
      structural.canonicalKey, (llvm::Twine("implementation-provider-tiles:") +
                                joinIntegers(parameters.outputTileSizes) + ":" +
                                joinIntegers(parameters.reductionTileSizes))
                                   .str());
  const StructuredAlternativeStructuralEstimates &estimates =
      point->getStructuralEstimates();
  auto setEstimate = [&](CoordinatedStructuredDimension dimension,
                         std::optional<uint64_t> value, llvm::StringRef model) {
    if (!value)
      return;
    CoordinatedStructuredMetric &metric =
        structural.facts.selection[static_cast<size_t>(dimension)];
    metric.value = *value;
    metric.knowledge = CoordinatedStructuredFactKnowledge::Estimated;
    metric.disposition = model.str();
  };
  setEstimate(CoordinatedStructuredDimension::ResourcePressure,
              estimates.estimatedPeakLiveBytes,
              "provider-typed-peak-live-bytes-v1");
  std::optional<uint64_t> aggregateCompute =
      logicalRankCount > 0 && estimates.estimatedComputeScalarOps
          ? checkedMultiply(*estimates.estimatedComputeScalarOps,
                            static_cast<uint64_t>(logicalRankCount))
          : std::nullopt;
  setEstimate(CoordinatedStructuredDimension::ComputeAggregateWork,
              aggregateCompute, "provider-typed-compute-scalar-ops-v1");
  setEstimate(CoordinatedStructuredDimension::ComputeMaximumRankWork,
              estimates.estimatedComputeScalarOps,
              "provider-typed-compute-scalar-ops-v1");
  structural.recipe.implementationAlternative = std::move(point);
  structural.recipe.implementationPostResidencyAction = postResidencyAction;
  return structural;
}

static std::string computeStructuralProposalDigest(
    llvm::ArrayRef<CoordinatedStructuralProposal> proposals) {
  llvm::SHA256 hasher;
  for (const CoordinatedStructuralProposal &proposal : proposals) {
    hasher.update((llvm::Twine(proposal.stableSemanticOrdinal) + ":" +
                   proposal.canonicalKey + "\n")
                      .str());
  }
  return llvm::toHex(hasher.final(), /*LowerCase=*/true);
}

static std::string
computeActualAdmissionDigest(llvm::ArrayRef<std::string> admissions) {
  llvm::SHA256 hasher;
  for (const std::string &admission : admissions) {
    hasher.update(admission);
    hasher.update("\n");
  }
  return llvm::toHex(hasher.final(), /*LowerCase=*/true);
}

static std::string computeCoordinatedTileVariantContentDigestImpl(
    const CoordinatedTileVariant &variant) {
  llvm::SHA256 hasher;
  for (const CoordinatedRankTileProgram &rank : variant.ranks) {
    std::string moduleText;
    llvm::raw_string_ostream stream(moduleText);
    stream << "rank:" << rank.logicalRank << '\n';
    rank.module.get().print(stream);
    stream.flush();
    hasher.update(moduleText);
  }
  return llvm::toHex(hasher.final(), /*LowerCase=*/true);
}

} // namespace

std::string computeCoordinatedTileVariantContentDigest(
    const CoordinatedTileVariant &variant) {
  return computeCoordinatedTileVariantContentDigestImpl(variant);
}

std::optional<uint64_t> CoordinatedWorkEstimate::getTotal() const {
  uint64_t total = 0;
  for (uint64_t count : counts) {
    std::optional<uint64_t> next = checkedAdd(total, count);
    if (!next)
      return std::nullopt;
    total = *next;
  }
  return total;
}

mlir::FailureOr<CoordinatedWorkEstimate>
CoordinatedWorkLedger::getMandatoryGenerationUpperBound(int64_t rankCount) {
  if (rankCount <= 0)
    return mlir::failure();
  CoordinatedWorkEstimate estimate;
  uint64_t ranks = static_cast<uint64_t>(rankCount);
  estimate.set(CoordinatedWorkKind::StructuredExpansion, ranks);
  estimate.set(CoordinatedWorkKind::ActualTileClone, ranks);
  return estimate;
}

mlir::FailureOr<CoordinatedWorkEstimate>
CoordinatedWorkLedger::getExecutableFinalizationUpperBound(int64_t rankCount) {
  if (rankCount <= 0)
    return mlir::failure();
  uint64_t ranks = static_cast<uint64_t>(rankCount);
  CoordinatedWorkEstimate estimate;
  estimate.set(CoordinatedWorkKind::TileToInstrLowering, ranks);
  mlir::FailureOr<CoordinatedWorkEstimate> action =
      getExecutableScheduleAttemptUpperBound(rankCount);
  if (mlir::failed(action))
    return mlir::failure();
  for (size_t index = 0; index < kCoordinatedWorkKindCount; ++index) {
    std::optional<uint64_t> combined =
        checkedAdd(estimate.counts[index], action->counts[index]);
    if (!combined)
      return mlir::failure();
    estimate.counts[index] = *combined;
  }
  return estimate;
}

mlir::FailureOr<CoordinatedWorkEstimate>
CoordinatedWorkLedger::getExecutableScheduleAttemptUpperBound(
    int64_t rankCount) {
  if (rankCount <= 0)
    return mlir::failure();
  const uint64_t ranks = static_cast<uint64_t>(rankCount);
  CoordinatedWorkEstimate estimate;
  estimate.set(CoordinatedWorkKind::ExecutableScheduleAction, ranks);
  estimate.set(CoordinatedWorkKind::SPMAllocationProblem, ranks);
  estimate.set(CoordinatedWorkKind::DDRAllocationDomain, ranks);
  estimate.set(CoordinatedWorkKind::TransportValidation, 1);
  estimate.set(CoordinatedWorkKind::ABIValidation, ranks);
  return estimate;
}

CoordinatedWorkLedger::CoordinatedWorkLedger(
    int64_t rankCount, uint64_t capacity, uint64_t repairReserve,
    CoordinatedWorkEstimate generationUpperBound,
    CoordinatedWorkEstimate finalizationUpperBound)
    : rankCount(rankCount), capacity(capacity),
      mandatoryGenerationReserved(*generationUpperBound.getTotal()),
      repairReserved(repairReserve),
      mandatoryGenerationUpperBound(std::move(generationUpperBound)) {
  mandatoryBaselineReservation.id = nextReservationId++;
  finalizationReservations.push_back(
      {mandatoryBaselineReservation.id, /*mandatoryBaseline=*/true,
       /*scheduleAttemptSlots=*/1, std::move(finalizationUpperBound)});
  scheduleAttemptsReserved = 1;
}

mlir::FailureOr<CoordinatedWorkLedger>
CoordinatedWorkLedger::create(int64_t rankCount, uint64_t capacity,
                              uint64_t repairReserve) {
  mlir::FailureOr<CoordinatedWorkEstimate> generation =
      getMandatoryGenerationUpperBound(rankCount);
  mlir::FailureOr<CoordinatedWorkEstimate> finalization =
      getExecutableFinalizationUpperBound(rankCount);
  if (mlir::failed(generation) || mlir::failed(finalization))
    return mlir::failure();
  std::optional<uint64_t> generationTotal = generation->getTotal();
  std::optional<uint64_t> finalizationTotal = finalization->getTotal();
  if (!generationTotal || !finalizationTotal)
    return mlir::failure();
  std::optional<uint64_t> mandatory =
      checkedAdd(*generationTotal, *finalizationTotal);
  if (!mandatory)
    return mlir::failure();
  mandatory = checkedAdd(*mandatory, repairReserve);
  if (!mandatory || *mandatory > capacity)
    return mlir::failure();
  return CoordinatedWorkLedger(rankCount, capacity, repairReserve,
                               std::move(*generation),
                               std::move(*finalization));
}

CoordinatedWorkLedger::FinalizationReservationRecord *
CoordinatedWorkLedger::findReservation(uint64_t id) {
  auto found = llvm::find_if(finalizationReservations,
                             [&](const FinalizationReservationRecord &record) {
                               return record.id == id;
                             });
  return found == finalizationReservations.end() ? nullptr : &*found;
}

const CoordinatedWorkLedger::FinalizationReservationRecord *
CoordinatedWorkLedger::findReservation(uint64_t id) const {
  auto found = llvm::find_if(finalizationReservations,
                             [&](const FinalizationReservationRecord &record) {
                               return record.id == id;
                             });
  return found == finalizationReservations.end() ? nullptr : &*found;
}

bool CoordinatedWorkLedger::canReserve(uint64_t credits) const {
  CoordinatedWorkLedgerSnapshot snapshot = getSnapshot();
  return credits <= snapshot.unreserved;
}

void CoordinatedWorkLedger::addConsumed(const CoordinatedWorkEstimate &actual) {
  consumed += *actual.getTotal();
  for (size_t index = 0; index < kCoordinatedWorkKindCount; ++index)
    consumedByKind[index] += actual.counts[index];
}

mlir::LogicalResult CoordinatedWorkLedger::completeMandatoryBaselineGeneration(
    const CoordinatedWorkEstimate &actual) {
  if (mandatoryGenerationReserved == 0 ||
      !isWithinEstimate(actual, mandatoryGenerationUpperBound) ||
      !actual.getTotal())
    return mlir::failure();
  addConsumed(actual);
  mandatoryGenerationReserved = 0;
  mandatoryGenerationUpperBound = {};
  return mlir::success();
}

bool CoordinatedWorkLedger::tryConsumeGeneration(CoordinatedWorkKind kind,
                                                 uint64_t credits) {
  if (kind != CoordinatedWorkKind::StructuredExpansion &&
      kind != CoordinatedWorkKind::ActualTileClone)
    return false;
  CoordinatedWorkEstimate actual;
  actual.set(kind, credits);
  return tryConsumeGeneration(actual);
}

bool CoordinatedWorkLedger::tryConsumeGeneration(
    const CoordinatedWorkEstimate &actual) {
  std::optional<uint64_t> credits = actual.getTotal();
  if (!credits || *credits == 0 ||
      actual.get(CoordinatedWorkKind::RepairExpansion) != 0 ||
      actual.get(CoordinatedWorkKind::TileToInstrLowering) != 0 ||
      actual.get(CoordinatedWorkKind::ExecutableScheduleAction) != 0 ||
      actual.get(CoordinatedWorkKind::SPMAllocationProblem) != 0 ||
      actual.get(CoordinatedWorkKind::DDRAllocationDomain) != 0 ||
      actual.get(CoordinatedWorkKind::TransportValidation) != 0 ||
      actual.get(CoordinatedWorkKind::ABIValidation) != 0 ||
      !canReserve(*credits))
    return false;
  addConsumed(actual);
  return true;
}

std::optional<ExecutableFinalizationReservation>
CoordinatedWorkLedger::tryReserveExecutableFinalization(
    const CoordinatedWorkEstimate &upperBound) {
  std::optional<uint64_t> credits = upperBound.getTotal();
  const uint32_t scheduleAttemptSlots =
      upperBound.get(CoordinatedWorkKind::ExecutableScheduleAction) == 0 ? 0
                                                                         : 1;
  if (!credits || *credits == 0 || !canReserve(*credits) ||
      scheduleAttemptsConsumed + scheduleAttemptsReserved +
              scheduleAttemptSlots >
          kMaximumCoordinatedScheduleRecipeMaterializationAttempts)
    return std::nullopt;
  ExecutableFinalizationReservation reservation{nextReservationId++};
  finalizationReservations.push_back({reservation.id,
                                      /*mandatoryBaseline=*/false,
                                      scheduleAttemptSlots, upperBound});
  scheduleAttemptsReserved += scheduleAttemptSlots;
  return reservation;
}

std::optional<ExecutableFinalizationReservation>
CoordinatedWorkLedger::tryReserveExecutableScheduleAttempt() {
  mlir::FailureOr<CoordinatedWorkEstimate> upperBound =
      getExecutableScheduleAttemptUpperBound(rankCount);
  if (mlir::failed(upperBound))
    return std::nullopt;
  return tryReserveExecutableFinalization(*upperBound);
}

mlir::LogicalResult CoordinatedWorkLedger::completeExecutableFinalization(
    ExecutableFinalizationReservation reservation,
    const CoordinatedWorkEstimate &actual) {
  FinalizationReservationRecord *record = findReservation(reservation.id);
  if (!record || !isWithinEstimate(actual, record->upperBound) ||
      !actual.getTotal())
    return mlir::failure();
  const uint32_t attemptSlots = record->scheduleAttemptSlots;
  if (attemptSlots > scheduleAttemptsReserved)
    return mlir::failure();
  const bool attempted =
      actual.get(CoordinatedWorkKind::ExecutableScheduleAction) != 0;
  scheduleAttemptsReserved -= attemptSlots;
  if (attempted)
    scheduleAttemptsConsumed += attemptSlots;
  addConsumed(actual);
  llvm::erase_if(finalizationReservations,
                 [&](const FinalizationReservationRecord &candidate) {
                   return candidate.id == reservation.id;
                 });
  return mlir::success();
}

mlir::LogicalResult CoordinatedWorkLedger::releaseExecutableFinalization(
    ExecutableFinalizationReservation reservation) {
  FinalizationReservationRecord *record = findReservation(reservation.id);
  if (!record || record->mandatoryBaseline)
    return mlir::failure();
  if (record->scheduleAttemptSlots > scheduleAttemptsReserved)
    return mlir::failure();
  scheduleAttemptsReserved -= record->scheduleAttemptSlots;
  llvm::erase_if(finalizationReservations,
                 [&](const FinalizationReservationRecord &candidate) {
                   return candidate.id == reservation.id;
                 });
  return mlir::success();
}

bool CoordinatedWorkLedger::tryConsumeRepair(uint64_t credits) {
  CoordinatedWorkEstimate actual;
  actual.set(CoordinatedWorkKind::RepairExpansion, credits);
  return tryConsumeRepair(actual);
}

bool CoordinatedWorkLedger::tryConsumeRepair(
    const CoordinatedWorkEstimate &actual) {
  std::optional<uint64_t> credits = actual.getTotal();
  if (!credits || *credits == 0 ||
      actual.get(CoordinatedWorkKind::RepairExpansion) == 0 ||
      actual.get(CoordinatedWorkKind::StructuredExpansion) != 0 ||
      actual.get(CoordinatedWorkKind::TileToInstrLowering) != 0 ||
      actual.get(CoordinatedWorkKind::ExecutableScheduleAction) != 0 ||
      actual.get(CoordinatedWorkKind::SPMAllocationProblem) != 0 ||
      actual.get(CoordinatedWorkKind::DDRAllocationDomain) != 0 ||
      actual.get(CoordinatedWorkKind::TransportValidation) != 0 ||
      actual.get(CoordinatedWorkKind::ABIValidation) != 0 ||
      *credits > repairReserved)
    return false;
  repairReserved -= *credits;
  addConsumed(actual);
  return true;
}

CoordinatedWorkLedgerSnapshot CoordinatedWorkLedger::getSnapshot() const {
  CoordinatedWorkLedgerSnapshot snapshot;
  snapshot.capacity = capacity;
  snapshot.consumed = consumed;
  snapshot.mandatoryGenerationReserved = mandatoryGenerationReserved;
  snapshot.repairReserved = repairReserved;
  snapshot.scheduleAttemptsReserved = scheduleAttemptsReserved;
  snapshot.scheduleAttemptsConsumed = scheduleAttemptsConsumed;
  snapshot.consumedByKind = consumedByKind;
  for (const FinalizationReservationRecord &record : finalizationReservations)
    snapshot.finalizationReserved += *record.upperBound.getTotal();
  uint64_t committed = consumed + mandatoryGenerationReserved +
                       snapshot.finalizationReserved + repairReserved;
  snapshot.unreserved = committed <= capacity ? capacity - committed : 0;
  return snapshot;
}

mlir::LogicalResult
verifyCoordinatedTileVariant(const CoordinatedTileVariant &variant,
                             int64_t expectedRankCount) {
  if (expectedRankCount <= 0 ||
      variant.ranks.size() != static_cast<size_t>(expectedRankCount) ||
      variant.finalizationReservation.id == 0 ||
      variant.repairDepth > kMaximumCoordinatedRepairDepth ||
      (variant.reservedBaseline && variant.repairDepth != 0))
    return mlir::failure();
  for (auto [expectedRank, rank] : llvm::enumerate(variant.ranks)) {
    if (rank.logicalRank != static_cast<int64_t>(expectedRank) || !rank.module)
      return mlir::failure();
    if (containsInstructionSemantics(*rank.module)) {
      rank.module.get().emitError()
          << "coordinated Tile variant contains instruction semantics at "
             "logical rank "
          << expectedRank;
      return mlir::failure();
    }
    bool hasPhysicalFacts = false;
    rank.module.get().walk([&](mlir::Operation *operation) {
      if (auto allocation = mlir::dyn_cast<mlir::memref::AllocOp>(operation))
        hasPhysicalFacts |= allocation->hasAttr(kWaferSPMOffsetAttrName) ||
                            allocation->hasAttr(kWaferDDROffsetAttrName);
      if (auto send = mlir::dyn_cast<InstrDTESendOp>(operation))
        hasPhysicalFacts |= send.getBinding().has_value();
      if (auto recv = mlir::dyn_cast<InstrDTERecvOp>(operation))
        hasPhysicalFacts |= recv.getBinding().has_value();
    });
    if (hasPhysicalFacts) {
      rank.module.get().emitError() << "coordinated Tile variant contains "
                                       "finalization-owned physical facts at "
                                       "logical rank "
                                    << expectedRank;
      return mlir::failure();
    }
    if (mlir::failed(mlir::verify(*rank.module)))
      return mlir::failure();
  }
  return verifyCoordinatedStructuredCommunication(variant, expectedRankCount);
}

mlir::FailureOr<CoordinatedTileVariant> materializeCoordinatedTileRepair(
    const CoordinatedTileVariant &parent, CoordinatedTileRepairAction action,
    int64_t stableSemanticOrdinal, CoordinatedWorkLedger &ledger,
    std::string *failureReason) {
  auto fail =
      [&](llvm::StringRef reason) -> mlir::FailureOr<CoordinatedTileVariant> {
    if (failureReason)
      *failureReason = reason.str();
    return mlir::failure();
  };
  if (parent.reservedBaseline ||
      parent.repairDepth >= kMaximumCoordinatedRepairDepth ||
      stableSemanticOrdinal <= parent.stableSemanticOrdinal ||
      ledger.getRankCount() <= 0 ||
      mlir::failed(verifyCoordinatedTileVariant(parent, ledger.getRankCount())))
    return fail("invalid coordinated Tile repair parent or ordinal");

  mlir::FailureOr<CoordinatedWorkEstimate> finalizationUpperBound =
      CoordinatedWorkLedger::getExecutableFinalizationUpperBound(
          ledger.getRankCount());
  if (mlir::failed(finalizationUpperBound))
    return fail("cannot derive repair executable-finalization upper bound");
  std::optional<ExecutableFinalizationReservation> reservation =
      ledger.tryReserveExecutableFinalization(*finalizationUpperBound);
  if (!reservation)
    return fail(
        "coordinated repair has no executable-finalization reservation");

  CoordinatedWorkEstimate repairWork;
  repairWork.set(CoordinatedWorkKind::RepairExpansion, 1);
  repairWork.set(CoordinatedWorkKind::ActualTileClone, parent.ranks.size());
  if (!ledger.tryConsumeRepair(repairWork)) {
    (void)ledger.releaseExecutableFinalization(*reservation);
    return fail("coordinated repair reserve is exhausted");
  }

  CandidateTileResidencyAction residencyAction =
      action == CoordinatedTileRepairAction::SelectiveSpill
          ? CandidateTileResidencyAction::SelectiveSpill
          : CandidateTileResidencyAction::SplitAtExplicitDDRBoundary;
  CoordinatedTileVariant sibling;
  sibling.stableSemanticOrdinal = stableSemanticOrdinal;
  sibling.implementationAlternativeOrigin =
      parent.implementationAlternativeOrigin;
  sibling.repairDepth = parent.repairDepth + 1;
  sibling.finalizationReservation = *reservation;
  sibling.ranks.reserve(parent.ranks.size());
  for (const CoordinatedRankTileProgram &rank : parent.ranks) {
    std::string rankFailure;
    mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> rankSibling =
        materializeCompleteRankTileResidencySibling(
            *rank.module, residencyAction, &rankFailure);
    if (mlir::failed(rankSibling)) {
      (void)ledger.releaseExecutableFinalization(*reservation);
      if (failureReason)
        *failureReason =
            (llvm::Twine("logical rank ") + llvm::Twine(rank.logicalRank) +
             " repair failed" +
             (rankFailure.empty() ? llvm::Twine()
                                  : llvm::Twine(": ") + rankFailure))
                .str();
      return mlir::failure();
    }
    sibling.ranks.emplace_back(rank.logicalRank, std::move(*rankSibling),
                               /*selectedTileIR=*/nullptr);
  }
  if (mlir::failed(
          verifyCoordinatedTileVariant(sibling, ledger.getRankCount()))) {
    (void)ledger.releaseExecutableFinalization(*reservation);
    return fail("coordinated repair produced invalid actual Tile IR");
  }
  if (failureReason)
    failureReason->clear();
  return sibling;
}

std::string
computeCoordinatedTileFrontierDigest(const CoordinatedTileFrontier &frontier) {
  llvm::SHA256 hasher;
  for (const CoordinatedTileVariant &variant : frontier) {
    std::string header =
        (llvm::Twine("variant:") + llvm::Twine(variant.stableSemanticOrdinal) +
         ":" + llvm::Twine(variant.reservedBaseline ? 1 : 0) + "\n")
            .str();
    hasher.update(header);
    for (const CoordinatedRankTileProgram &rank : variant.ranks) {
      std::string moduleText;
      llvm::raw_string_ostream stream(moduleText);
      stream << "rank:" << rank.logicalRank << '\n';
      rank.module.get().print(stream);
      stream.flush();
      hasher.update(moduleText);
    }
  }
  return llvm::toHex(hasher.final(), /*LowerCase=*/true);
}

namespace {

struct ConnectionProposalState {
  llvm::SmallVector<CandidateTraversalConnectionChoice, 16> choices;
  uint64_t estimatedPeakResidentBytes = 0;
  uint64_t estimatedDDRReadBytes = 0;
  uint64_t estimatedDDRWriteBytes = 0;
  uint64_t estimatedDDRIssueWork = 0;
  uint64_t estimatedTraversalTrips = 0;
  uint64_t estimatedTileUnderutilizationElements = 0;
  uint64_t estimatedComputeElements = 0;
  uint64_t estimatedCoordinationPenalty = 0;
  std::string futureVersionSignature;
  int64_t stableSemanticOrdinal = 0;
  bool estimateOverflow = false;
};

static bool
sameConnectionChoice(const CandidateTraversalConnectionChoice &left,
                     const CandidateTraversalConnectionChoice &right) {
  return left.action == right.action &&
         left.producerTileSizes == right.producerTileSizes &&
         left.consumerTileSizes == right.consumerTileSizes;
}

static bool sameConnectionChoices(
    llvm::ArrayRef<CandidateTraversalConnectionChoice> left,
    llvm::ArrayRef<CandidateTraversalConnectionChoice> right) {
  return left.size() == right.size() &&
         llvm::all_of(llvm::zip_equal(left, right), [](const auto &pair) {
           return sameConnectionChoice(std::get<0>(pair), std::get<1>(pair));
         });
}

static bool
isSeparatedConnectionChoice(const CandidateTraversalConnectionChoice &choice) {
  return choice.action != CandidateTraversalConnectionAction::CoupledResident;
}

static llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 8>
buildConnectionTileDomainPoints(llvm::ArrayRef<int64_t> shape,
                                bool includeTargetPreferred) {
  llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 8> points;
  auto appendUnique = [&](llvm::SmallVector<int64_t, 4> point) {
    if (!llvm::is_contained(points, point))
      points.push_back(std::move(point));
  };
  appendUnique(llvm::SmallVector<int64_t, 4>(shape.begin(), shape.end()));
  if (!includeTargetPreferred)
    return points;
  appendUnique(llvm::SmallVector<int64_t, 4>(shape.size(), 1));

  const WaferTargetPolicy policy =
      getDefaultWaferTargetPolicy(TileSearchEffort::Default);
  for (int64_t preferred : policy.tileSearch.preferredTileSizes) {
    if (preferred <= 0)
      continue;
    llvm::SmallVector<int64_t, 4> point;
    point.reserve(shape.size());
    for (int64_t extent : shape)
      point.push_back(std::min(extent, preferred));
    appendUnique(std::move(point));
  }
  return points;
}

static llvm::SmallVector<std::string, 4> getLocalChoiceCoverageFeatures(
    const CandidateTraversalConnectionChoice &choice) {
  llvm::SmallVector<std::string, 4> features;
  features.push_back((llvm::Twine("action:") +
                      llvm::Twine(static_cast<unsigned>(choice.action)))
                         .str());
  if (isSeparatedConnectionChoice(choice))
    features.push_back(
        (llvm::Twine("producer:") + joinIntegers(choice.producerTileSizes))
            .str());
  features.push_back(
      (llvm::Twine("consumer:") + joinIntegers(choice.consumerTileSizes))
          .str());
  if (isSeparatedConnectionChoice(choice) &&
      choice.producerTileSizes != choice.consumerTileSizes)
    features.push_back("independent-producer-consumer-tiles");
  return features;
}

static llvm::SmallVector<CandidateTraversalConnectionChoice, 8>
buildLocalConnectionChoicePool(const CandidateTraversalConnectionDomain &domain,
                               uint64_t &rawChoiceCount) {
  using Action = CandidateTraversalConnectionAction;
  const auto producerPoints = buildConnectionTileDomainPoints(
      domain.producerResultShape, /*includeTargetPreferred=*/true);
  const auto consumerPoints = buildConnectionTileDomainPoints(
      domain.consumerResultShape, /*includeTargetPreferred=*/true);

  llvm::SmallVector<CandidateTraversalConnectionChoice, 64> all;
  for (const auto &consumer : consumerPoints) {
    CandidateTraversalConnectionChoice choice;
    choice.action = Action::CoupledResident;
    // Coupled producer demand is always recovered from the consumer tile by
    // the current TilingInterface and exact intervening SSA view relation.
    choice.consumerTileSizes = consumer;
    all.push_back(std::move(choice));
  }
  constexpr Action separatedActions[] = {
      Action::SeparatedResident, Action::SeparatedDDR, Action::CrossRegion,
      Action::SelectiveSpill};
  for (Action action : separatedActions)
    for (const auto &producer : producerPoints)
      for (const auto &consumer : consumerPoints) {
        CandidateTraversalConnectionChoice choice;
        choice.action = action;
        choice.producerTileSizes = producer;
        choice.consumerTileSizes = consumer;
        all.push_back(std::move(choice));
      }
  rawChoiceCount = all.size();

  llvm::SmallVector<CandidateTraversalConnectionChoice, 8> retained;
  std::vector<bool> selected(all.size(), false);
  llvm::StringSet<> covered;
  auto uncoveredScore = [&](size_t index) {
    unsigned score = 0;
    for (const std::string &feature :
         getLocalChoiceCoverageFeatures(all[index]))
      score += covered.find(feature) == covered.end();
    return score;
  };
  auto retain = [&](size_t index) {
    if (selected[index] ||
        retained.size() >= kMaximumCoordinatedConnectionExpansionsPerStep)
      return;
    selected[index] = true;
    for (std::string &feature : getLocalChoiceCoverageFeatures(all[index]))
      covered.insert(feature);
    retained.push_back(all[index]);
  };

  // Every action family receives one deterministic representative first. The
  // representative greedily adds still-uncovered producer/consumer points so
  // the remaining slots can cover independent tile vectors rather than repeat
  // the same full/full tuple for every storage action.
  constexpr Action allActions[] = {
      Action::CoupledResident, Action::SeparatedResident, Action::SeparatedDDR,
      Action::CrossRegion, Action::SelectiveSpill};
  for (Action action : allActions) {
    std::optional<size_t> best;
    unsigned bestScore = 0;
    for (size_t index = 0; index < all.size(); ++index) {
      if (selected[index] || all[index].action != action)
        continue;
      unsigned score = uncoveredScore(index);
      if (!best || score > bestScore) {
        best = index;
        bestScore = score;
      }
    }
    if (best)
      retain(*best);
  }
  while (retained.size() < kMaximumCoordinatedConnectionExpansionsPerStep) {
    std::optional<size_t> best;
    unsigned bestScore = 0;
    for (size_t index = 0; index < all.size(); ++index) {
      if (selected[index])
        continue;
      unsigned score = uncoveredScore(index);
      if (!best || score > bestScore) {
        best = index;
        bestScore = score;
      }
    }
    if (!best || bestScore == 0)
      break;
    retain(*best);
  }
  return retained;
}

static unsigned
getConnectionStorageClass(CandidateTraversalConnectionAction action) {
  switch (action) {
  case CandidateTraversalConnectionAction::CoupledResident:
  case CandidateTraversalConnectionAction::SeparatedResident:
    return 0;
  case CandidateTraversalConnectionAction::SeparatedDDR:
  case CandidateTraversalConnectionAction::CrossRegion:
    return 1;
  case CandidateTraversalConnectionAction::SelectiveSpill:
    return 2;
  }
  llvm_unreachable("unknown connection action");
}

static std::string
getSeparatedVersionSignature(const CandidateTraversalConnectionDomain &domain,
                             const CandidateTraversalConnectionChoice &choice) {
  return (llvm::Twine(domain.producerOperationOrdinal) + "." +
          llvm::Twine(domain.producerResultNumber) +
          ":storage=" + llvm::Twine(getConnectionStorageClass(choice.action)) +
          ":tile=" + joinIntegers(choice.producerTileSizes) + ":region-cut=" +
          llvm::Twine(choice.action ==
                              CandidateTraversalConnectionAction::CrossRegion
                          ? 1
                          : 0))
      .str();
}

static bool hasCompatibleSharedSeparatedVersion(
    llvm::ArrayRef<CandidateTraversalConnectionDomain> domains,
    llvm::ArrayRef<CandidateTraversalConnectionChoice> choices,
    const CandidateTraversalConnectionDomain &nextDomain,
    const CandidateTraversalConnectionChoice &nextChoice) {
  if (!isSeparatedConnectionChoice(nextChoice))
    return true;
  const std::string nextSignature =
      getSeparatedVersionSignature(nextDomain, nextChoice);
  for (auto [domain, choice] : llvm::zip_equal(domains, choices)) {
    if (!isSeparatedConnectionChoice(choice) ||
        domain.producerOperationOrdinal !=
            nextDomain.producerOperationOrdinal ||
        domain.producerResultNumber != nextDomain.producerResultNumber)
      continue;
    if (getSeparatedVersionSignature(domain, choice) != nextSignature)
      return false;
  }
  return true;
}

static std::string getFutureVersionSignature(
    llvm::ArrayRef<CandidateTraversalConnectionDomain> domains,
    llvm::ArrayRef<CandidateTraversalConnectionChoice> choices) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  stream << "connection-versions";
  llvm::StringSet<> seen;
  for (auto [domain, choice] : llvm::zip_equal(domains, choices)) {
    if (!isSeparatedConnectionChoice(choice))
      continue;
    std::string signature = getSeparatedVersionSignature(domain, choice);
    if (seen.insert(signature).second)
      stream << '|' << signature;
  }
  stream.flush();
  return result;
}

struct ConnectionTileWork {
  uint64_t logicalElements = 0;
  uint64_t paddedElements = 0;
  uint64_t traversalTrips = 0;
  uint64_t underutilizedElements = 0;
};

static std::optional<ConnectionTileWork>
estimateConnectionTileWork(llvm::ArrayRef<int64_t> shape,
                           llvm::ArrayRef<int64_t> tileSizes) {
  if (shape.size() != tileSizes.size())
    return std::nullopt;
  ConnectionTileWork result{1, 1, 1, 0};
  for (auto [extentValue, tileValue] : llvm::zip_equal(shape, tileSizes)) {
    if (extentValue <= 0 || tileValue <= 0 || tileValue > extentValue)
      return std::nullopt;
    const uint64_t extent = static_cast<uint64_t>(extentValue);
    const uint64_t tile = static_cast<uint64_t>(tileValue);
    const uint64_t trips = extent / tile + (extent % tile != 0);
    auto logical = checkedMultiply(result.logicalElements, extent);
    auto padded = checkedMultiply(result.paddedElements, tile);
    auto nextTrips = checkedMultiply(result.traversalTrips, trips);
    if (!logical || !padded || !nextTrips)
      return std::nullopt;
    result.logicalElements = *logical;
    result.paddedElements = *padded;
    result.traversalTrips = *nextTrips;
  }
  auto paddedWork =
      checkedMultiply(result.paddedElements, result.traversalTrips);
  if (!paddedWork || *paddedWork < result.logicalElements)
    return std::nullopt;
  result.paddedElements = *paddedWork;
  result.underutilizedElements = *paddedWork - result.logicalElements;
  return result;
}

static bool isCanonicalCoupledFullPrefix(
    llvm::ArrayRef<CandidateTraversalConnectionDomain> domains,
    llvm::ArrayRef<CandidateTraversalConnectionChoice> choices) {
  return llvm::all_of(llvm::zip_equal(domains, choices), [](const auto &pair) {
    const auto &domain = std::get<0>(pair);
    const auto &choice = std::get<1>(pair);
    return choice.action ==
               CandidateTraversalConnectionAction::CoupledResident &&
           choice.producerTileSizes.empty() &&
           choice.consumerTileSizes == domain.consumerResultShape;
  });
}

static mlir::FailureOr<std::vector<CoordinatedStructuralProposal>>
deriveStructuralFrontier(mlir::ModuleOp sourceModule,
                         const CoordinatedDataflowSearchConfig &config,
                         CoordinatedWorkLedger &ledger,
                         CoordinatedStructuredFrontierStatistics &statistics) {
  std::vector<CoordinatedStructuralProposal> proposals;
  CoordinatedStructuralProposal baseline;
  baseline.stableSemanticOrdinal = 0;
  baseline.reservedBaseline = true;
  baseline.recipe.kind = CoordinatedStructuralRecipeKind::MandatoryBaseline;
  baseline.canonicalKey = "mandatory-conservative-baseline";
  baseline.coverageClass = "mandatory-baseline";
  baseline.facts =
      makeUnknownPrecloneFacts("baseline-physical", "baseline-traversal");
  proposals.push_back(std::move(baseline));

  if (config.reservedBaselineOnly ||
      config.optimizations == OptimizationConfig::none()) {
    statistics.structuralProposalsDerived = proposals.size();
    statistics.structuralPendingPeak = proposals.size();
    statistics.structuralFrontierDigest =
        computeStructuralProposalDigest(proposals);
    return proposals;
  }

  std::vector<StructuredTraversalProposal> traversals =
      buildStructuredTraversalProposals(sourceModule);
  statistics.derivedCandidates = traversals.size();
  int64_t nextStableOrdinal = 1;
  for (StructuredTraversalProposal &proposal : traversals) {
    nextStableOrdinal =
        std::max(nextStableOrdinal, proposal.stableSemanticOrdinal + 1);
    if (!ledger.tryConsumeGeneration(
            CoordinatedWorkKind::StructuredExpansion)) {
      statistics.structuralBudgetExhausted = true;
      break;
    }
    proposals.push_back(makeTraversalStructuralProposal(std::move(proposal)));
  }

  if (!statistics.structuralBudgetExhausted) {
    const WaferTargetPolicy policy =
        getDefaultWaferTargetPolicy(TileSearchEffort::Default);
    llvm::SmallVector<int64_t, 8> tileSizeSeeds;
    tileSizeSeeds.push_back(1);
    for (int64_t preferred : policy.tileSearch.preferredTileSizes)
      if (preferred > 0 && !llvm::is_contained(tileSizeSeeds, preferred))
        tileSizeSeeds.push_back(preferred);
    llvm::SmallVector<int64_t, 4> parallelPartitionCounts{2, 4, 8, 16};

    llvm::StringSet<> providerKeys;
    for (const StructuredImplementationAlternativeProvider *provider :
         config.implementationAlternativeProviders) {
      if (!provider || provider->getStableKey().empty() ||
          !providerKeys.insert(provider->getStableKey()).second) {
        sourceModule.emitError(
            "invalid or duplicate structured implementation provider");
        return mlir::failure();
      }
      StructuredImplementationAlternativePoints points;
      std::string providerFailure;
      StructuredImplementationAlternativeQuery query;
      query.parallelPartitionCounts = parallelPartitionCounts;
      query.outputTileSizeSeeds = tileSizeSeeds;
      query.reductionTileSizeSeeds = tileSizeSeeds;
      ++statistics.implementationAlternativeQueries;
      if (mlir::failed(
              provider->query(sourceModule, query, points, &providerFailure))) {
        sourceModule.emitError()
            << "structured implementation provider query failed"
            << (providerFailure.empty() ? "" : ": ") << providerFailure;
        return mlir::failure();
      }
      const uint64_t targetSPMWindowBytes =
          static_cast<uint64_t>(policy.memory.spmLimit - policy.memory.spmBase);
      llvm::sort(points, [targetSPMWindowBytes](const auto &left,
                                                const auto &right) {
        const auto &leftEstimate = left->getStructuralEstimates();
        const auto &rightEstimate = right->getStructuralEstimates();
        // A provider may expose a typed, target-independent live-byte
        // prior. It covers the provider graph, while the complete actual
        // candidate may also contain surrounding ordinary structured
        // work and physical alignment. Prefer a point that leaves half
        // the target window as headroom, then an estimated in-window
        // point, before an estimated oversized point. Unknown remains a
        // peer of the preferred class, and every class still reaches the
        // Pareto/backfill path: this is ordering, never legality.
        auto resourceHeadroomClass =
            [targetSPMWindowBytes](std::optional<uint64_t> estimate) {
              if (!estimate || *estimate <= targetSPMWindowBytes / 2)
                return 0u;
              if (*estimate <= targetSPMWindowBytes)
                return 1u;
              return 2u;
            };
        return std::make_tuple(
                   resourceHeadroomClass(leftEstimate.estimatedPeakLiveBytes),
                   leftEstimate.estimatedComputeScalarOps,
                   leftEstimate.structuredWorkUnitUpperBound,
                   leftEstimate.estimatedPeakLiveBytes,
                   leftEstimate.outputTileCount,
                   leftEstimate.reductionTileCount,
                   left->getIdentity().stableKey) <
               std::make_tuple(
                   resourceHeadroomClass(rightEstimate.estimatedPeakLiveBytes),
                   rightEstimate.estimatedComputeScalarOps,
                   rightEstimate.structuredWorkUnitUpperBound,
                   rightEstimate.estimatedPeakLiveBytes,
                   rightEstimate.outputTileCount,
                   rightEstimate.reductionTileCount,
                   right->getIdentity().stableKey);
      });
      if (points.size() > kMaximumCoordinatedStructuralFrontierStates) {
        StructuredImplementationAlternativePoints retained;
        std::vector<bool> selected(points.size(), false);
        llvm::SmallSet<int64_t, 8> partitionCoverage;
        llvm::SmallVector<StructuredAlternativeTileShape, 8> outputTileCoverage;
        llvm::SmallVector<llvm::SmallVector<int64_t, 2>, 8>
            reductionTileCoverage;
        auto retain = [&](size_t index) {
          if (selected[index] ||
              retained.size() >= kMaximumCoordinatedStructuralFrontierStates)
            return;
          const StructuredAlternativeParameters &parameters =
              points[index]->getParameters();
          if (!llvm::is_contained(outputTileCoverage,
                                  parameters.outputTileSizes))
            outputTileCoverage.push_back(parameters.outputTileSizes);
          if (!llvm::is_contained(reductionTileCoverage,
                                  parameters.reductionTileSizes))
            reductionTileCoverage.push_back(parameters.reductionTileSizes);
          selected[index] = true;
          retained.push_back(std::move(points[index]));
        };
        for (size_t index = 0;
             index < points.size() &&
             retained.size() < kMaximumCoordinatedStructuralFrontierStates;
             ++index)
          if (partitionCoverage
                  .insert(points[index]->getParameters().parallelPartitionCount)
                  .second)
            retain(index);
        // A provider point is a product of independent typed parameter axes.
        // Keeping only the smallest structural-work values can otherwise fill
        // the whole beam with large tiles from one end of an axis before any
        // resource-feasible smaller tile is materialized.  Preserve one
        // stable representative of every declared output and reduction tile
        // vector before filling the remaining slots by the common ordering.
        // These vectors stay opaque to the coordinator: coverage does not
        // infer algorithm semantics, estimate target resources, or select a
        // winner.
        for (size_t index = 0;
             index < points.size() &&
             retained.size() < kMaximumCoordinatedStructuralFrontierStates;
             ++index) {
          if (selected[index])
            continue;
          const StructuredAlternativeTileShape &tileSizes =
              points[index]->getParameters().outputTileSizes;
          if (!llvm::is_contained(outputTileCoverage, tileSizes)) {
            outputTileCoverage.push_back(tileSizes);
            retain(index);
          }
        }
        for (size_t index = 0;
             index < points.size() &&
             retained.size() < kMaximumCoordinatedStructuralFrontierStates;
             ++index) {
          if (selected[index])
            continue;
          const llvm::SmallVector<int64_t, 2> &tileSizes =
              points[index]->getParameters().reductionTileSizes;
          if (!llvm::is_contained(reductionTileCoverage, tileSizes)) {
            reductionTileCoverage.push_back(tileSizes);
            retain(index);
          }
        }
        for (size_t index = 0;
             index < points.size() &&
             retained.size() < kMaximumCoordinatedStructuralFrontierStates;
             ++index)
          retain(index);
        statistics.structuralCoverageBeamPruned +=
            points.size() - retained.size();
        points = std::move(retained);
      }
      std::vector<
          std::shared_ptr<const StructuredImplementationAlternativePoint>>
          sharedPoints;
      sharedPoints.reserve(points.size());
      for (std::unique_ptr<StructuredImplementationAlternativePoint> &point :
           points)
        sharedPoints.emplace_back(std::move(point));

      // Let every semantic provider point compete through ordinary lowering
      // before adding non-identity physical siblings. Point-major Cartesian
      // order would let the three residency actions for a few early points
      // consume the shared structural beam and starve later parameter points.
      constexpr CandidateTileResidencyAction postActions[] = {
          CandidateTileResidencyAction::KeepSingleRegion,
          CandidateTileResidencyAction::SelectiveSpill,
          CandidateTileResidencyAction::SplitAtExplicitDDRBoundary};
      for (CandidateTileResidencyAction postAction : postActions) {
        for (const std::shared_ptr<
                 const StructuredImplementationAlternativePoint> &sharedPoint :
             sharedPoints) {
          if (!ledger.tryConsumeGeneration(
                  CoordinatedWorkKind::StructuredExpansion)) {
            statistics.structuralBudgetExhausted = true;
            break;
          }
          proposals.push_back(makeImplementationStructuralProposal(
              nextStableOrdinal++, provider->getStableKey(), sharedPoint,
              postAction, config.rankCount));
          ++statistics.implementationAlternativeProposals;
        }
        if (statistics.structuralBudgetExhausted)
          break;
      }
      if (statistics.structuralBudgetExhausted)
        break;
    }
  }

  if (!statistics.structuralBudgetExhausted) {
    std::string topologyFailure;
    mlir::FailureOr<CandidateTraversalConnectionTopology> topology =
        getCompleteRankCandidateConnectionTopology(sourceModule,
                                                   &topologyFailure);
    if (mlir::failed(topology)) {
      sourceModule.emitError()
          << "cannot derive structured connection topology"
          << (topologyFailure.empty() ? "" : ": ") << topologyFailure;
      return mlir::failure();
    }
    statistics.structuredConnections = topology->connectionCount;
    statistics.usedGeneralDAGBeam = topology->requiresGeneralDAGBeam;
    if (topology->domains.size() != topology->connectionCount) {
      sourceModule.emitError(
          "structured connection domains do not cover current SSA edges");
      return mlir::failure();
    }

    if (topology->connectionCount != 0) {
      const size_t stateLimit = topology->requiresGeneralDAGBeam
                                    ? kMaximumCoordinatedConnectionDAGBeamStates
                                    : kMaximumCoordinatedConnectionDPStates;
      auto estimate = [&](llvm::ArrayRef<CandidateTraversalConnectionChoice>
                              choices,
                          int64_t ordinal) {
        ConnectionProposalState state;
        state.choices.assign(choices.begin(), choices.end());
        state.stableSemanticOrdinal = ordinal;
        llvm::ArrayRef<CandidateTraversalConnectionDomain> domains =
            llvm::ArrayRef<CandidateTraversalConnectionDomain>(
                topology->domains)
                .take_front(choices.size());
        state.futureVersionSignature =
            getFutureVersionSignature(domains, choices);
        auto add = [&](uint64_t &target, uint64_t value) {
          std::optional<uint64_t> result = checkedAdd(target, value);
          if (!result) {
            target = std::numeric_limits<uint64_t>::max();
            state.estimateOverflow = true;
          } else {
            target = *result;
          }
        };
        for (auto [choice, domain] : llvm::zip_equal(choices, domains)) {
          llvm::ArrayRef<int64_t> producerTileSizes =
              choice.producerTileSizes.empty()
                  ? llvm::ArrayRef<int64_t>(domain.producerResultShape)
                  : llvm::ArrayRef<int64_t>(choice.producerTileSizes);
          llvm::ArrayRef<int64_t> consumerTileSizes =
              choice.consumerTileSizes.empty()
                  ? llvm::ArrayRef<int64_t>(domain.consumerResultShape)
                  : llvm::ArrayRef<int64_t>(choice.consumerTileSizes);
          std::optional<ConnectionTileWork> producer =
              estimateConnectionTileWork(domain.producerResultShape,
                                         producerTileSizes);
          std::optional<ConnectionTileWork> consumer =
              estimateConnectionTileWork(domain.consumerResultShape,
                                         consumerTileSizes);
          if (!producer || !consumer) {
            state.estimateOverflow = true;
            state.estimatedPeakResidentBytes =
                std::numeric_limits<uint64_t>::max();
            state.estimatedDDRReadBytes = std::numeric_limits<uint64_t>::max();
            state.estimatedDDRWriteBytes = std::numeric_limits<uint64_t>::max();
            state.estimatedDDRIssueWork = std::numeric_limits<uint64_t>::max();
            state.estimatedTraversalTrips =
                std::numeric_limits<uint64_t>::max();
            state.estimatedTileUnderutilizationElements =
                std::numeric_limits<uint64_t>::max();
            state.estimatedComputeElements =
                std::numeric_limits<uint64_t>::max();
            continue;
          }
          auto multiply = [&](uint64_t left, uint64_t right) {
            std::optional<uint64_t> result = checkedMultiply(left, right);
            if (!result) {
              state.estimateOverflow = true;
              return std::numeric_limits<uint64_t>::max();
            }
            return *result;
          };
          const uint64_t producerPaddedBytes = multiply(
              producer->paddedElements, domain.producerResultElementBytes);
          const uint64_t producerLogicalBytes = multiply(
              producer->logicalElements, domain.producerResultElementBytes);
          const uint64_t consumerPaddedBytes = multiply(
              consumer->paddedElements, domain.consumerResultElementBytes);
          // The consumer choice is in its result traversal domain. Before
          // actual TilingInterface replay, use that concrete padded traversal
          // work with the exact connected operand's element width as the
          // deterministic read-byte proxy. Actual Tile IR later replaces it
          // with loop-expanded movement facts.
          const uint64_t consumerDemandBytes = multiply(
              consumer->paddedElements, domain.consumerOperandElementBytes);
          const bool coupled =
              choice.action ==
              CandidateTraversalConnectionAction::CoupledResident;
          // Coupled producer demand is not independently selected. Until the
          // actual TilingInterface replay derives it exactly, use a bounded
          // shape estimate driven by this concrete consumer tile rather than a
          // synthetic producer tile choice.
          const uint64_t producerResidentElements =
              coupled ? std::min(producer->logicalElements,
                                 consumer->paddedElements)
                      : producer->paddedElements;
          const uint64_t producerResidentBytes =
              coupled ? std::min(producerLogicalBytes, consumerDemandBytes)
                      : producerPaddedBytes;
          uint64_t residentBytes = 0;
          switch (choice.action) {
          case CandidateTraversalConnectionAction::CoupledResident:
            add(residentBytes, producerResidentBytes);
            add(residentBytes, consumerPaddedBytes);
            break;
          case CandidateTraversalConnectionAction::SeparatedResident:
            add(residentBytes, producerPaddedBytes);
            add(residentBytes, consumerPaddedBytes);
            add(state.estimatedCoordinationPenalty, 1);
            break;
          case CandidateTraversalConnectionAction::SeparatedDDR:
          case CandidateTraversalConnectionAction::CrossRegion:
          case CandidateTraversalConnectionAction::SelectiveSpill: {
            residentBytes = std::max(producerPaddedBytes, consumerPaddedBytes);
            add(state.estimatedDDRWriteBytes, producerPaddedBytes);
            add(state.estimatedDDRReadBytes, consumerDemandBytes);
            add(state.estimatedDDRIssueWork, producer->traversalTrips);
            add(state.estimatedDDRIssueWork, consumer->traversalTrips);
            add(state.estimatedCoordinationPenalty,
                choice.action == CandidateTraversalConnectionAction::CrossRegion
                    ? 3
                    : 2);
            break;
          }
          }
          add(state.estimatedTraversalTrips, consumer->traversalTrips);
          add(state.estimatedTileUnderutilizationElements,
              consumer->underutilizedElements);
          add(state.estimatedComputeElements, consumer->paddedElements);
          if (coupled) {
            add(state.estimatedComputeElements, producerResidentElements);
          } else {
            add(state.estimatedTraversalTrips, producer->traversalTrips);
            add(state.estimatedTileUnderutilizationElements,
                producer->underutilizedElements);
            add(state.estimatedComputeElements, producer->paddedElements);
          }
          state.estimatedPeakResidentBytes =
              std::max(state.estimatedPeakResidentBytes, residentBytes);
        }
        return state;
      };
      auto stateLess = [](const ConnectionProposalState &left,
                          const ConnectionProposalState &right) {
        if (left.estimateOverflow != right.estimateOverflow)
          return !left.estimateOverflow;
        if (left.estimatedPeakResidentBytes != right.estimatedPeakResidentBytes)
          return left.estimatedPeakResidentBytes <
                 right.estimatedPeakResidentBytes;
        if (left.estimatedDDRReadBytes != right.estimatedDDRReadBytes)
          return left.estimatedDDRReadBytes < right.estimatedDDRReadBytes;
        if (left.estimatedDDRWriteBytes != right.estimatedDDRWriteBytes)
          return left.estimatedDDRWriteBytes < right.estimatedDDRWriteBytes;
        if (left.estimatedDDRIssueWork != right.estimatedDDRIssueWork)
          return left.estimatedDDRIssueWork < right.estimatedDDRIssueWork;
        if (left.estimatedTileUnderutilizationElements !=
            right.estimatedTileUnderutilizationElements)
          return left.estimatedTileUnderutilizationElements <
                 right.estimatedTileUnderutilizationElements;
        if (left.estimatedTraversalTrips != right.estimatedTraversalTrips)
          return left.estimatedTraversalTrips < right.estimatedTraversalTrips;
        if (left.estimatedComputeElements != right.estimatedComputeElements)
          return left.estimatedComputeElements < right.estimatedComputeElements;
        if (left.estimatedCoordinationPenalty !=
            right.estimatedCoordinationPenalty)
          return left.estimatedCoordinationPenalty <
                 right.estimatedCoordinationPenalty;
        if (left.futureVersionSignature != right.futureVersionSignature)
          return left.futureVersionSignature < right.futureVersionSignature;
        return left.stableSemanticOrdinal < right.stableSemanticOrdinal;
      };
      auto stateDominates = [](const ConnectionProposalState &left,
                               const ConnectionProposalState &right) {
        if (left.futureVersionSignature != right.futureVersionSignature)
          return false;
        if (left.estimateOverflow != right.estimateOverflow)
          return !left.estimateOverflow;
        if (left.estimateOverflow)
          return false;
        const bool noWorse =
            left.estimatedPeakResidentBytes <=
                right.estimatedPeakResidentBytes &&
            left.estimatedDDRReadBytes <= right.estimatedDDRReadBytes &&
            left.estimatedDDRWriteBytes <= right.estimatedDDRWriteBytes &&
            left.estimatedDDRIssueWork <= right.estimatedDDRIssueWork &&
            left.estimatedTraversalTrips <= right.estimatedTraversalTrips &&
            left.estimatedTileUnderutilizationElements <=
                right.estimatedTileUnderutilizationElements &&
            left.estimatedComputeElements <= right.estimatedComputeElements &&
            left.estimatedCoordinationPenalty <=
                right.estimatedCoordinationPenalty;
        const bool strictlyBetter =
            left.estimatedPeakResidentBytes <
                right.estimatedPeakResidentBytes ||
            left.estimatedDDRReadBytes < right.estimatedDDRReadBytes ||
            left.estimatedDDRWriteBytes < right.estimatedDDRWriteBytes ||
            left.estimatedDDRIssueWork < right.estimatedDDRIssueWork ||
            left.estimatedTraversalTrips < right.estimatedTraversalTrips ||
            left.estimatedTileUnderutilizationElements <
                right.estimatedTileUnderutilizationElements ||
            left.estimatedComputeElements < right.estimatedComputeElements ||
            left.estimatedCoordinationPenalty <
                right.estimatedCoordinationPenalty;
        return noWorse && strictlyBetter;
      };

      std::vector<ConnectionProposalState> current(1);
      current.front().futureVersionSignature = "connection-versions";

      for (unsigned edge = 0; edge < topology->connectionCount &&
                              !statistics.structuralBudgetExhausted;
           ++edge) {
        uint64_t rawLocalChoiceCount = 0;
        llvm::SmallVector<CandidateTraversalConnectionChoice, 8> localChoices =
            buildLocalConnectionChoicePool(topology->domains[edge],
                                           rawLocalChoiceCount);
        statistics.maximumConnectionLocalChoices = std::max<uint64_t>(
            statistics.maximumConnectionLocalChoices, localChoices.size());
        statistics.connectionLocalChoicesPruned +=
            rawLocalChoiceCount - localChoices.size();
        if (localChoices.empty()) {
          current.clear();
          break;
        }

        std::vector<ConnectionProposalState> next;
        llvm::ArrayRef<CandidateTraversalConnectionDomain> parentDomains =
            llvm::ArrayRef<CandidateTraversalConnectionDomain>(
                topology->domains)
                .take_front(edge);
        for (const ConnectionProposalState &parent : current) {
          uint64_t parentExpansions = 0;
          for (const CandidateTraversalConnectionChoice &localChoice :
               localChoices) {
            ++parentExpansions;
            if (!hasCompatibleSharedSeparatedVersion(
                    parentDomains, parent.choices, topology->domains[edge],
                    localChoice)) {
              ++statistics.connectionIncompatibleSharedVersionsPruned;
              continue;
            }
            llvm::SmallVector<CandidateTraversalConnectionChoice, 16> selected =
                parent.choices;
            selected.push_back(localChoice);
            if (llvm::any_of(next, [&](const ConnectionProposalState &state) {
                  return sameConnectionChoices(state.choices, selected);
                })) {
              ++statistics.connectionDPStatesMerged;
              continue;
            }
            if (!ledger.tryConsumeGeneration(
                    CoordinatedWorkKind::StructuredExpansion)) {
              statistics.structuralBudgetExhausted = true;
              break;
            }
            next.push_back(estimate(selected, nextStableOrdinal++));
            ++statistics.connectionProposalExpansions;
          }
          statistics.maximumConnectionExpansionsPerParent = std::max<uint64_t>(
              statistics.maximumConnectionExpansionsPerParent,
              parentExpansions);
          if (statistics.structuralBudgetExhausted)
            break;
        }
        if (statistics.structuralBudgetExhausted)
          break;

        llvm::sort(next, stateLess);
        std::vector<ConnectionProposalState> nondominated;
        llvm::StringSet<> paretoCoverage;
        llvm::ArrayRef<CandidateTraversalConnectionDomain> prefixDomains =
            llvm::ArrayRef<CandidateTraversalConnectionDomain>(
                topology->domains)
                .take_front(edge + 1);
        for (ConnectionProposalState &state : next) {
          const bool mandatoryDefault =
              isCanonicalCoupledFullPrefix(prefixDomains, state.choices);
          const bool preservesCoverage =
              paretoCoverage.insert(getConnectionCoverageClass(state.choices))
                  .second;
          const bool dominated = llvm::any_of(
              nondominated, [&](const ConnectionProposalState &retained) {
                return stateDominates(retained, state);
              });
          if (dominated && !mandatoryDefault && !preservesCoverage) {
            ++statistics.connectionDominatedStatesPruned;
            continue;
          }
          nondominated.push_back(std::move(state));
        }
        next = std::move(nondominated);
        if (next.size() > stateLimit) {
          std::vector<ConnectionProposalState> retained;
          std::vector<bool> selected(next.size(), false);
          llvm::StringSet<> coverage;
          auto retain = [&](size_t index) {
            if (selected[index] || retained.size() >= stateLimit)
              return;
            selected[index] = true;
            retained.push_back(next[index]);
          };
          auto baselineState =
              llvm::find_if(next, [&](const ConnectionProposalState &state) {
                return isCanonicalCoupledFullPrefix(prefixDomains,
                                                    state.choices);
              });
          if (baselineState != next.end())
            retain(static_cast<size_t>(baselineState - next.begin()));
          for (size_t index = 0;
               index < next.size() && retained.size() < stateLimit; ++index) {
            std::string key = getConnectionCoverageClass(next[index].choices);
            if (coverage.insert(key).second)
              retain(index);
          }
          for (size_t index = 0;
               index < next.size() && retained.size() < stateLimit; ++index)
            retain(index);
          statistics.connectionBeamStatesPruned +=
              next.size() - retained.size();
          next = std::move(retained);
          llvm::sort(next, stateLess);
        }
        statistics.maximumConnectionStates =
            std::max<uint64_t>(statistics.maximumConnectionStates, next.size());
        current = std::move(next);
      }

      for (ConnectionProposalState &state : current) {
        if (state.choices.size() != topology->connectionCount)
          continue;
        proposals.push_back(makeConnectionStructuralProposal(
            state.stableSemanticOrdinal, state.choices,
            state.estimatedPeakResidentBytes, state.estimatedDDRReadBytes,
            state.estimatedDDRWriteBytes, state.estimatedDDRIssueWork,
            state.estimatedTraversalTrips,
            state.estimatedTileUnderutilizationElements,
            state.estimatedComputeElements, state.estimatedCoordinationPenalty,
            state.futureVersionSignature, state.estimateOverflow));
      }
    }
  }

  llvm::sort(proposals, [](const CoordinatedStructuralProposal &left,
                           const CoordinatedStructuralProposal &right) {
    return left.stableSemanticOrdinal < right.stableSemanticOrdinal;
  });
  std::vector<CoordinatedStructuralProposal> pareto;
  llvm::StringSet<> canonicalKeys;
  for (CoordinatedStructuralProposal &proposal : proposals) {
    if (!canonicalKeys.insert(proposal.canonicalKey).second) {
      ++statistics.structuralEquivalentProposalsMerged;
      continue;
    }
    llvm::SmallVector<CoordinatedStructuredFrontierView, 16> existing;
    for (const CoordinatedStructuralProposal &state : pareto)
      existing.push_back(
          {state.stableSemanticOrdinal, state.reservedBaseline, &state.facts});
    mlir::FailureOr<CoordinatedStructuredInsertionPlan> insertion =
        planCoordinatedStructuredFrontierInsertion(
            {proposal.stableSemanticOrdinal, proposal.reservedBaseline,
             &proposal.facts},
            existing);
    if (mlir::failed(insertion))
      return mlir::failure();
    if (!insertion->retainCandidate) {
      ++statistics.structuralDominatedProposalsPruned;
      continue;
    }
    for (size_t index : llvm::reverse(insertion->eraseIndices)) {
      if (index < pareto.size() && !pareto[index].reservedBaseline) {
        pareto.erase(pareto.begin() + index);
        ++statistics.structuralDominatedProposalsPruned;
      }
    }
    pareto.push_back(std::move(proposal));
  }
  llvm::sort(pareto, [](const CoordinatedStructuralProposal &left,
                        const CoordinatedStructuralProposal &right) {
    return left.stableSemanticOrdinal < right.stableSemanticOrdinal;
  });

  if (pareto.size() > kMaximumCoordinatedStructuralFrontierStates) {
    std::vector<CoordinatedStructuralProposal> retained;
    std::vector<bool> selected(pareto.size(), false);
    llvm::StringSet<> coverage;
    auto retain = [&](size_t index) {
      if (selected[index] ||
          retained.size() >= kMaximumCoordinatedStructuralFrontierStates)
        return;
      selected[index] = true;
      retained.push_back(std::move(pareto[index]));
    };
    retain(0);
    for (size_t index = 1;
         index < pareto.size() &&
         retained.size() < kMaximumCoordinatedStructuralFrontierStates;
         ++index)
      if (coverage.insert(pareto[index].coverageClass).second)
        retain(index);
    for (size_t index = 1;
         index < pareto.size() &&
         retained.size() < kMaximumCoordinatedStructuralFrontierStates;
         ++index)
      retain(index);
    statistics.structuralCoverageBeamPruned += pareto.size() - retained.size();
    pareto = std::move(retained);
    llvm::sort(pareto, [](const CoordinatedStructuralProposal &left,
                          const CoordinatedStructuralProposal &right) {
      return left.stableSemanticOrdinal < right.stableSemanticOrdinal;
    });
  }

  // Admission order is itself a bounded coverage beam: the mandatory
  // baseline comes first, followed by the best stable representative of each
  // structural family, then remaining Pareto states. Reassign query-local
  // semantic ordinals after this deterministic ordering so serial reservation
  // and exact-failure backfill remain monotonic even when producer families
  // were derived at different times.
  std::vector<CoordinatedStructuralProposal> prioritized;
  std::vector<bool> selected(pareto.size(), false);
  llvm::StringSet<> admissionCoverage;
  auto retain = [&](size_t index) {
    if (selected[index])
      return;
    selected[index] = true;
    prioritized.push_back(std::move(pareto[index]));
  };
  retain(0);
  for (size_t index = 1; index < pareto.size(); ++index)
    if (admissionCoverage.insert(pareto[index].coverageClass).second)
      retain(index);
  for (size_t index = 1; index < pareto.size(); ++index)
    retain(index);
  pareto = std::move(prioritized);
  for (auto [ordinal, proposal] : llvm::enumerate(pareto)) {
    proposal.stableSemanticOrdinal = static_cast<int64_t>(ordinal);
    if (proposal.recipe.kind == CoordinatedStructuralRecipeKind::Traversal)
      proposal.recipe.traversal.stableSemanticOrdinal =
          static_cast<int64_t>(ordinal);
  }

  statistics.structuralProposalsDerived = pareto.size();
  statistics.structuralPendingPeak = pareto.size();
  statistics.structuralFrontierDigest = computeStructuralProposalDigest(pareto);
  return pareto;
}

} // namespace

struct CoordinatedDataflowSearchSession::Impl {
  struct ActualFactState {
    int64_t stableSemanticOrdinal = 0;
    bool reservedBaseline = false;
    CoordinatedStructuredFrontierFacts facts;
  };

  mlir::ModuleOp sourceModule;
  CoordinatedDataflowSearchConfig config;
  CoordinatedWorkLedger *ledger = nullptr;
  CoordinatedStructuredFrontierStatistics statistics;
  CoordinatedStructuredFrontierStatistics *externalStatistics = nullptr;
  std::vector<CoordinatedStructuralProposal> structuralFrontier;
  size_t nextProposal = 0;
  std::optional<int64_t> activeOrdinal;
  std::vector<int64_t> activeDominatedOrdinals;
  std::vector<ActualFactState> actualFacts;
  llvm::StringSet<> actualDigests;
  std::vector<std::string> actualAdmissions;
  CoordinatedWorkEstimate finalizationUpperBound;
  bool rankInvariant = false;
  bool stoppedForBudget = false;
  bool exactBackfillPending = false;
  bool materializationBackfillPending = false;

  void publish() {
    statistics.structuralPendingAtStop =
        structuralFrontier.size() -
        std::min(nextProposal, structuralFrontier.size());
    statistics.retainedStates = actualFacts.size();
    if (externalStatistics)
      *externalStatistics = statistics;
  }

  mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
  generateBaselineRank(int64_t logicalRank) {
    return materializeConservativeCompleteRankBaseline(sourceModule,
                                                       logicalRank);
  }

  mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
  materializeRank(const CoordinatedStructuralProposal &proposal,
                  int64_t logicalRank, std::string &failureReason) {
    failureReason.clear();
    switch (proposal.recipe.kind) {
    case CoordinatedStructuralRecipeKind::MandatoryBaseline: {
      auto generated = generateBaselineRank(logicalRank);
      if (mlir::failed(generated)) {
        failureReason = "mandatory baseline generation failed";
        return mlir::failure();
      }
      return std::move(*generated);
    }
    case CoordinatedStructuralRecipeKind::Traversal: {
      const StructuredTraversalProposal &traversal = proposal.recipe.traversal;
      if (traversal.cloneReservedBaseline) {
        auto generated = generateBaselineRank(logicalRank);
        if (mlir::failed(generated) ||
            !traversal.physicalLayoutProposalOrdinal) {
          failureReason = "layout proposal cannot replay the baseline recipe";
          return mlir::failure();
        }
        mlir::OwningOpRef<mlir::ModuleOp> module = std::move(*generated);
        if (mlir::failed(applyPhysicalLayoutProposal(
                *module, *traversal.physicalLayoutProposalOrdinal,
                /*result=*/nullptr, &failureReason)))
          return mlir::failure();
        return module;
      }
      return materializeCompleteRankCandidateTileProgram(
          sourceModule, logicalRank, traversal.tileSizes,
          traversal.reductionTileSizes, traversal.traversalKind,
          traversal.composition, traversal.residencyAction,
          traversal.boundaryMovementAction, traversal.loopMovementAction,
          &failureReason, traversal.selectedImplementation,
          traversal.physicalLayoutProposalOrdinal);
    }
    case CoordinatedStructuralRecipeKind::Connections:
      return materializeCompleteRankConnectionChoicesTileProgram(
          sourceModule, logicalRank, proposal.recipe.connections,
          CandidateBoundaryMovementAction::Staged,
          CandidateLoopMovementAction::AsConstructed, &failureReason);
    case CoordinatedStructuralRecipeKind::ImplementationAlternative: {
      if (!proposal.recipe.implementationAlternative) {
        failureReason = "implementation provider point is missing";
        return mlir::failure();
      }
      auto lowered = materializeStructuredImplementationAlternativeToTileRegion(
          sourceModule, *proposal.recipe.implementationAlternative, logicalRank,
          &failureReason);
      if (mlir::failed(lowered))
        return mlir::failure();
      if (proposal.recipe.implementationPostResidencyAction !=
          CandidateTileResidencyAction::KeepSingleRegion)
        return materializeCompleteRankTileResidencySibling(
            **lowered, proposal.recipe.implementationPostResidencyAction,
            &failureReason);
      return std::move(*lowered);
    }
    }
    llvm_unreachable("unknown coordinated structural recipe");
  }

  mlir::FailureOr<CoordinatedTileVariant>
  materialize(const CoordinatedStructuralProposal &proposal,
              ExecutableFinalizationReservation reservation) {
    CoordinatedTileVariant candidate;
    candidate.stableSemanticOrdinal = proposal.stableSemanticOrdinal;
    candidate.reservedBaseline = proposal.reservedBaseline;
    candidate.implementationAlternativeOrigin =
        proposal.recipe.kind ==
        CoordinatedStructuralRecipeKind::ImplementationAlternative;
    candidate.finalizationReservation = reservation;
    candidate.ranks.reserve(static_cast<size_t>(config.rankCount));
    ++statistics.actualCandidateAttempts;
    if (proposal.recipe.kind == CoordinatedStructuralRecipeKind::Connections)
      ++statistics.connectionActionMaterializations;
    if (proposal.recipe.kind ==
        CoordinatedStructuralRecipeKind::ImplementationAlternative)
      statistics.implementationAlternativeMaterializations +=
          rankInvariant ? 1 : static_cast<uint64_t>(config.rankCount);

    if (rankInvariant) {
      std::string failureReason;
      auto generated = materializeRank(proposal, 0, failureReason);
      if (mlir::failed(generated)) {
        if (!failureReason.empty())
          sourceModule.emitRemark()
              << "structural proposal " << proposal.stableSemanticOrdinal
              << " was unavailable: " << failureReason;
        return mlir::failure();
      }
      for (int64_t logicalRank = 0; logicalRank < config.rankCount;
           ++logicalRank) {
        mlir::OwningOpRef<mlir::ModuleOp> module =
            logicalRank == 0 ? std::move(*generated)
                             : mlir::cast<mlir::ModuleOp>(
                                   candidate.ranks.front().module->clone());
        candidate.ranks.emplace_back(logicalRank, std::move(module), nullptr);
      }
      return candidate;
    }

    std::vector<std::optional<mlir::OwningOpRef<mlir::ModuleOp>>> modules(
        static_cast<size_t>(config.rankCount));
    std::vector<std::string> failures(static_cast<size_t>(config.rankCount));
    unsigned workers = runBoundedRankPipelines(
        sourceModule.getContext(), static_cast<size_t>(config.rankCount),
        [&](size_t rank) {
          auto generated = materializeRank(proposal, static_cast<int64_t>(rank),
                                           failures[rank]);
          if (mlir::succeeded(generated))
            modules[rank].emplace(std::move(*generated));
        },
        static_cast<unsigned>(config.candidateParallelism));
    if (proposal.recipe.kind == CoordinatedStructuralRecipeKind::Connections)
      statistics.connectionWorkerCount =
          std::max<uint64_t>(statistics.connectionWorkerCount, workers);
    for (size_t rank = 0; rank < modules.size(); ++rank) {
      if (!modules[rank]) {
        if (!failures[rank].empty())
          sourceModule.emitRemark()
              << "structural proposal " << proposal.stableSemanticOrdinal
              << " was unavailable at logical rank " << rank << ": "
              << failures[rank];
        return mlir::failure();
      }
      candidate.ranks.emplace_back(static_cast<int64_t>(rank),
                                   std::move(*modules[rank]), nullptr);
    }
    return candidate;
  }
};

CoordinatedDataflowSearchSession::CoordinatedDataflowSearchSession(
    std::unique_ptr<Impl> impl)
    : impl(std::move(impl)) {}

CoordinatedDataflowSearchSession::~CoordinatedDataflowSearchSession() = default;
CoordinatedDataflowSearchSession::CoordinatedDataflowSearchSession(
    CoordinatedDataflowSearchSession &&) noexcept = default;
CoordinatedDataflowSearchSession &CoordinatedDataflowSearchSession::operator=(
    CoordinatedDataflowSearchSession &&) noexcept = default;

mlir::FailureOr<std::unique_ptr<CoordinatedDataflowSearchSession>>
CoordinatedDataflowSearchSession::create(
    mlir::ModuleOp sourceModule, const CoordinatedDataflowSearchConfig &config,
    CoordinatedWorkLedger &ledger,
    CoordinatedStructuredFrontierStatistics *statistics) {
  if (!sourceModule || config.rankCount <= 0 ||
      config.rankCount != ledger.getRankCount() ||
      config.candidateParallelism <= 0 ||
      config.maximumSuccessfulActualCandidates == 0 ||
      config.maximumSuccessfulActualCandidates >
          kMaximumCoordinatedSuccessfulActualCandidates) {
    if (sourceModule)
      sourceModule.emitError("invalid coordinated dataflow search contract");
    return mlir::failure();
  }
  auto state = std::make_unique<Impl>();
  state->sourceModule = sourceModule;
  state->config = config;
  state->ledger = &ledger;
  state->externalStatistics = statistics;
  state->rankInvariant = isCompleteRankTensorProgramRankInvariant(sourceModule);
  mlir::FailureOr<CoordinatedWorkEstimate> finalization =
      CoordinatedWorkLedger::getExecutableFinalizationUpperBound(
          config.rankCount);
  if (mlir::failed(finalization))
    return mlir::failure();
  state->finalizationUpperBound = *finalization;
  mlir::FailureOr<std::vector<CoordinatedStructuralProposal>> proposals =
      deriveStructuralFrontier(sourceModule, config, ledger, state->statistics);
  if (mlir::failed(proposals) || proposals->empty() ||
      !proposals->front().reservedBaseline)
    return mlir::failure();
  state->structuralFrontier = std::move(*proposals);
  state->publish();
  return std::unique_ptr<CoordinatedDataflowSearchSession>(
      new CoordinatedDataflowSearchSession(std::move(state)));
}

mlir::FailureOr<std::unique_ptr<CoordinatedTileVariant>>
CoordinatedDataflowSearchSession::admitNextActualCandidate() {
  if (!impl || impl->activeOrdinal)
    return mlir::failure();
  if (impl->statistics.successfulActualCandidates >=
          impl->config.maximumSuccessfulActualCandidates ||
      impl->stoppedForBudget ||
      impl->nextProposal >= impl->structuralFrontier.size()) {
    impl->publish();
    return std::unique_ptr<CoordinatedTileVariant>();
  }

  while (impl->nextProposal < impl->structuralFrontier.size()) {
    const CoordinatedStructuralProposal &proposal =
        impl->structuralFrontier[impl->nextProposal++];
    if (impl->config.reservedBaselineOnly && !proposal.reservedBaseline)
      break;

    ExecutableFinalizationReservation reservation;
    if (proposal.reservedBaseline) {
      reservation = impl->ledger->getMandatoryBaselineReservation();
    } else {
      std::optional<ExecutableFinalizationReservation> optionalReservation =
          impl->ledger->tryReserveExecutableFinalization(
              impl->finalizationUpperBound);
      if (!optionalReservation) {
        ++impl->statistics.finalizationAdmissionDenied;
        impl->stoppedForBudget = true;
        impl->statistics.structuralBudgetExhausted = true;
        impl->publish();
        return std::unique_ptr<CoordinatedTileVariant>();
      }
      reservation = *optionalReservation;
      uint64_t actualCloneCount = static_cast<uint64_t>(impl->config.rankCount);
      if (proposal.recipe.kind ==
          CoordinatedStructuralRecipeKind::ImplementationAlternative) {
        // The prepared structured artifact is the actual isolated clone and
        // is consumed in place by ordinary lowering. Only a non-identity
        // physical post-action creates another discardable sibling.
        std::optional<uint64_t> providerCloneCount = actualCloneCount;
        if (proposal.recipe.implementationPostResidencyAction !=
            CandidateTileResidencyAction::KeepSingleRegion)
          providerCloneCount = checkedAdd(
              actualCloneCount, impl->rankInvariant ? 1 : actualCloneCount);
        if (!providerCloneCount) {
          (void)impl->ledger->releaseExecutableFinalization(reservation);
          return mlir::failure();
        }
        actualCloneCount = *providerCloneCount;
      }
      if (!impl->ledger->tryConsumeGeneration(
              CoordinatedWorkKind::ActualTileClone, actualCloneCount)) {
        ++impl->statistics.generationAdmissionDenied;
        (void)impl->ledger->releaseExecutableFinalization(reservation);
        impl->stoppedForBudget = true;
        impl->statistics.structuralBudgetExhausted = true;
        impl->publish();
        return std::unique_ptr<CoordinatedTileVariant>();
      }
      impl->statistics.actualRankClones += actualCloneCount;
    }

    mlir::FailureOr<CoordinatedTileVariant> materialized =
        impl->materialize(proposal, reservation);
    if (mlir::failed(materialized)) {
      ++impl->statistics.candidateMaterializationFailures;
      impl->materializationBackfillPending = true;
      if (proposal.reservedBaseline ||
          mlir::failed(
              impl->ledger->releaseExecutableFinalization(reservation))) {
        impl->sourceModule.emitError(
            "mandatory or reservation-owned Tile materialization failed");
        return mlir::failure();
      }
      continue;
    }

    if (mlir::failed(verifyCoordinatedTileVariant(*materialized,
                                                  impl->config.rankCount))) {
      ++impl->statistics.candidateMaterializationFailures;
      impl->materializationBackfillPending = true;
      if (proposal.reservedBaseline ||
          mlir::failed(
              impl->ledger->releaseExecutableFinalization(reservation)))
        return mlir::failure();
      continue;
    }
    if (proposal.reservedBaseline) {
      CoordinatedWorkEstimate generation;
      generation.set(CoordinatedWorkKind::StructuredExpansion,
                     impl->rankInvariant
                         ? 1
                         : static_cast<uint64_t>(impl->config.rankCount));
      generation.set(CoordinatedWorkKind::ActualTileClone,
                     static_cast<uint64_t>(impl->config.rankCount));
      if (mlir::failed(
              impl->ledger->completeMandatoryBaselineGeneration(generation)))
        return mlir::failure();
      impl->statistics.actualRankClones +=
          static_cast<uint64_t>(impl->config.rankCount);
    }

    ++impl->statistics.materializedCandidates;
    std::string contentDigest =
        computeCoordinatedTileVariantContentDigest(*materialized);
    if (!impl->actualDigests.insert(contentDigest).second) {
      ++impl->statistics.equivalentCandidatesRejected;
      impl->materializationBackfillPending = true;
      if (proposal.reservedBaseline ||
          mlir::failed(
              impl->ledger->releaseExecutableFinalization(reservation)))
        return mlir::failure();
      continue;
    }
    mlir::FailureOr<CoordinatedStructuredFrontierFacts> facts =
        deriveCoordinatedStructuredFrontierFacts(*materialized);
    if (mlir::failed(facts)) {
      if (!proposal.reservedBaseline)
        (void)impl->ledger->releaseExecutableFinalization(reservation);
      return mlir::failure();
    }
    if (!proposal.reservedBaseline &&
        !canStillSatisfyCoordinatedProductionPromotion(
            *facts, impl->actualFacts.front().facts)) {
      ++impl->statistics.promotionIneligibleCandidatesRejected;
      impl->materializationBackfillPending = true;
      if (mlir::failed(
              impl->ledger->releaseExecutableFinalization(reservation)))
        return mlir::failure();
      continue;
    }
    llvm::SmallVector<CoordinatedStructuredFrontierView, 16> existing;
    for (const Impl::ActualFactState &state : impl->actualFacts)
      existing.push_back(
          {state.stableSemanticOrdinal, state.reservedBaseline, &state.facts});
    mlir::FailureOr<CoordinatedStructuredInsertionPlan> insertion =
        planCoordinatedStructuredFrontierInsertion(
            {proposal.stableSemanticOrdinal, proposal.reservedBaseline,
             &*facts},
            existing);
    if (mlir::failed(insertion))
      return mlir::failure();
    if (!insertion->retainCandidate) {
      bool equivalent = llvm::any_of(
          impl->actualFacts, [&](const Impl::ActualFactState &state) {
            return compareCoordinatedStructuredFrontierFacts(*facts,
                                                             state.facts) ==
                   CoordinatedStructuredFrontierOrder::Equivalent;
          });
      if (equivalent)
        ++impl->statistics.equivalentCandidatesRejected;
      else
        ++impl->statistics.dominatedCandidatesRejected;
      impl->materializationBackfillPending = true;
      if (proposal.reservedBaseline ||
          mlir::failed(
              impl->ledger->releaseExecutableFinalization(reservation)))
        return mlir::failure();
      continue;
    }
    impl->activeDominatedOrdinals.clear();
    for (size_t index : insertion->eraseIndices)
      if (index < impl->actualFacts.size() &&
          !impl->actualFacts[index].reservedBaseline)
        impl->activeDominatedOrdinals.push_back(
            impl->actualFacts[index].stableSemanticOrdinal);
    impl->actualFacts.push_back({proposal.stableSemanticOrdinal,
                                 proposal.reservedBaseline, std::move(*facts)});
    impl->actualAdmissions.push_back(
        (llvm::Twine(proposal.stableSemanticOrdinal) + ":" +
         llvm::Twine(proposal.reservedBaseline ? 1 : 0) + ":" + contentDigest)
            .str());
    impl->statistics.actualAdmissionDigest =
        computeActualAdmissionDigest(impl->actualAdmissions);
    ++impl->statistics.actualCandidateAdmissions;
    if (proposal.recipe.kind == CoordinatedStructuralRecipeKind::Connections)
      ++impl->statistics.retainedConnectionCandidates;
    if (proposal.recipe.kind ==
        CoordinatedStructuralRecipeKind::ImplementationAlternative)
      ++impl->statistics.retainedImplementationAlternativeCandidates;
    if (impl->exactBackfillPending) {
      ++impl->statistics.exactFailureBackfills;
      impl->exactBackfillPending = false;
    }
    if (impl->materializationBackfillPending) {
      ++impl->statistics.materializationFailureBackfills;
      impl->materializationBackfillPending = false;
    }
    impl->activeOrdinal = proposal.stableSemanticOrdinal;
    impl->statistics.peakLiveActualCandidates =
        std::max<uint64_t>(impl->statistics.peakLiveActualCandidates, 1);
    impl->publish();
    return std::make_unique<CoordinatedTileVariant>(std::move(*materialized));
  }
  impl->publish();
  return std::unique_ptr<CoordinatedTileVariant>();
}

mlir::LogicalResult CoordinatedDataflowSearchSession::completeActiveCandidate(
    int64_t stableSemanticOrdinal,
    CoordinatedActualCandidateDisposition disposition,
    llvm::StringRef failureGate) {
  (void)failureGate;
  if (!impl || !impl->activeOrdinal ||
      *impl->activeOrdinal != stableSemanticOrdinal)
    return mlir::failure();
  if (disposition == CoordinatedActualCandidateDisposition::ExactRejected) {
    impl->exactBackfillPending = true;
    llvm::erase_if(impl->actualFacts, [&](const Impl::ActualFactState &state) {
      return state.stableSemanticOrdinal == stableSemanticOrdinal &&
             !state.reservedBaseline;
    });
  } else {
    if (disposition == CoordinatedActualCandidateDisposition::ExactAccepted) {
      const size_t before = impl->actualFacts.size();
      llvm::erase_if(impl->actualFacts,
                     [&](const Impl::ActualFactState &state) {
                       return llvm::is_contained(impl->activeDominatedOrdinals,
                                                 state.stableSemanticOrdinal) &&
                              !state.reservedBaseline;
                     });
      impl->statistics.dominatedStatesErased +=
          before - impl->actualFacts.size();
    }
    ++impl->statistics.successfulActualCandidates;
  }
  impl->activeDominatedOrdinals.clear();
  impl->activeOrdinal.reset();
  impl->publish();
  return mlir::success();
}

bool CoordinatedDataflowSearchSession::exhausted() const {
  return !impl || (!impl->activeOrdinal &&
                   (impl->stoppedForBudget ||
                    impl->nextProposal >= impl->structuralFrontier.size() ||
                    impl->statistics.successfulActualCandidates >=
                        impl->config.maximumSuccessfulActualCandidates));
}

llvm::StringRef
CoordinatedDataflowSearchSession::getStructuralFrontierDigest() const {
  return impl ? llvm::StringRef(impl->statistics.structuralFrontierDigest)
              : llvm::StringRef();
}

llvm::StringRef
CoordinatedDataflowSearchSession::getActualAdmissionDigest() const {
  return impl ? llvm::StringRef(impl->statistics.actualAdmissionDigest)
              : llvm::StringRef();
}

mlir::FailureOr<CoordinatedTileFrontier> buildCoordinatedTileFrontier(
    mlir::ModuleOp sourceModule, const CoordinatedDataflowSearchConfig &config,
    CoordinatedWorkLedger &ledger,
    CoordinatedStructuredFrontierStatistics *structuredStatistics) {
  auto session = CoordinatedDataflowSearchSession::create(
      sourceModule, config, ledger, structuredStatistics);
  if (mlir::failed(session))
    return mlir::failure();
  CoordinatedTileFrontier frontier;
  while (frontier.size() < config.maximumSuccessfulActualCandidates) {
    auto next = (*session)->admitNextActualCandidate();
    if (mlir::failed(next))
      return mlir::failure();
    if (!*next)
      break;
    int64_t ordinal = (*next)->stableSemanticOrdinal;
    frontier.push_back(std::move(**next));
    if (mlir::failed((*session)->completeActiveCandidate(
            ordinal,
            CoordinatedActualCandidateDisposition::RetainedForCompatibility)))
      return mlir::failure();
  }
  if (frontier.empty() || !frontier.front().reservedBaseline)
    return mlir::failure();
  std::string digest = computeCoordinatedTileFrontierDigest(frontier);
  for (CoordinatedTileVariant &variant : frontier)
    variant.frontierDigest = digest;
  return frontier;
}

} // namespace wafer::compiler::detail
