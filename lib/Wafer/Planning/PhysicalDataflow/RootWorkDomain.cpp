//===- RootWorkDomain.cpp - Complete root and Tile work domain ----------===//

#include "Wafer/Planning/PhysicalDataflow/RootWorkDomain.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>

namespace wafer::compiler::detail {
namespace {

bool tileLess(TileId lhs, TileId rhs) {
  return lhs.getValue() < rhs.getValue();
}

analysis::BrokenRootRegionWork
invalidCursor(const analysis::RootRegionWorkId &site, llvm::StringRef detail) {
  return {analysis::BrokenRootRegionWorkReason::AssignmentProofMismatch, site,
          detail.str()};
}

} // namespace

RootWorkSiteOutcomeKind
classifyRootWorkOutcome(const analysis::RootRegionWorkOutcome &outcome) {
  if (std::holds_alternative<analysis::RootRegionWork>(outcome))
    return RootWorkSiteOutcomeKind::Work;
  if (std::holds_alternative<analysis::NoRootRegionWork>(outcome))
    return RootWorkSiteOutcomeKind::NoWork;
  if (std::holds_alternative<analysis::UnsupportedRootRegionWork>(outcome))
    return RootWorkSiteOutcomeKind::Unsupported;
  if (std::holds_alternative<analysis::RootRegionWorkLimitReached>(outcome))
    return RootWorkSiteOutcomeKind::Indeterminate;
  return RootWorkSiteOutcomeKind::CompilerBug;
}

RootWorkSuccessor
RootWorkDomain::fromOutcome(analysis::RootRegionWorkOutcome outcome) {
  switch (classifyRootWorkOutcome(outcome)) {
  case RootWorkSiteOutcomeKind::Work: {
    analysis::RootRegionWorkId id =
        std::get<analysis::RootRegionWork>(outcome).id;
    return {RootWorkSuccessorKind::Work,
            std::move(std::get<analysis::RootRegionWork>(outcome)),
            {},
            RootWorkCursor(std::move(id))};
  }
  case RootWorkSiteOutcomeKind::NoWork:
    return {RootWorkSuccessorKind::End};
  case RootWorkSiteOutcomeKind::Unsupported:
    return {RootWorkSuccessorKind::Unsupported,
            {},
            RootWorkDomainFailure(std::move(
                std::get<analysis::UnsupportedRootRegionWork>(outcome)))};
  case RootWorkSiteOutcomeKind::Indeterminate:
    return {RootWorkSuccessorKind::Indeterminate,
            {},
            RootWorkDomainFailure(std::move(
                std::get<analysis::RootRegionWorkLimitReached>(outcome)))};
  case RootWorkSiteOutcomeKind::CompilerBug:
    return {RootWorkSuccessorKind::CompilerBug,
            {},
            RootWorkDomainFailure(
                std::move(std::get<analysis::BrokenRootRegionWork>(outcome)))};
  }
  return {RootWorkSuccessorKind::CompilerBug,
          {},
          RootWorkDomainFailure(invalidCursor(
              {}, "root-work outcome has an unknown typed alternative"))};
}

mlir::FailureOr<RootWorkDomain> RootWorkDomain::create(
    const StructuredDAGAnalysis &dag, const SpatialAssignment &assignment,
    const analysis::ExactDemandProof &proof,
    llvm::ArrayRef<TileId> availableTiles, std::string *failureReason) {
  mlir::FailureOr<RootRegionWorkAnalysis> rootAnalysis =
      RootRegionWorkAnalysis::create(dag, assignment, proof, failureReason);
  if (mlir::failed(rootAnalysis))
    return mlir::failure();
  llvm::SmallVector<SemanticRootKey, 16> roots;
  roots.reserve(dag.getNodes().size());
  for (const StructuredDAGNode &node : dag.getNodes()) {
    const SemanticRootKey *root = rootAnalysis->getRoot(node.id);
    if (!root) {
      if (failureReason)
        *failureReason =
            "root-work domain has a node without semantic identity";
      return mlir::failure();
    }
    roots.push_back(*root);
  }
  llvm::sort(roots);
  if (roots.empty()) {
    if (failureReason)
      *failureReason = "root-work domain has no semantic roots";
    return mlir::failure();
  }
  if (std::adjacent_find(roots.begin(), roots.end()) != roots.end()) {
    if (failureReason)
      *failureReason = "root-work domain contains duplicate semantic roots";
    return mlir::failure();
  }

  llvm::SmallVector<TileId, 16> tiles(availableTiles.begin(),
                                      availableTiles.end());
  llvm::sort(tiles, tileLess);
  if (tiles.empty() ||
      std::adjacent_find(tiles.begin(), tiles.end()) != tiles.end()) {
    if (failureReason)
      *failureReason = "root-work domain requires unique available Tiles";
    return mlir::failure();
  }
  if (roots.size() > std::numeric_limits<size_t>::max() / tiles.size()) {
    if (failureReason)
      *failureReason = "root-work domain size is not representable";
    return mlir::failure();
  }
  return RootWorkDomain(std::move(*rootAnalysis), std::move(roots),
                        std::move(tiles));
}

bool RootWorkDomain::contains(const analysis::RootRegionWorkId &id) const {
  return llvm::is_contained(roots, id.root) &&
         llvm::is_contained(tiles, id.tile);
}

RootWorkSuccessor
RootWorkDomain::scanFrom(size_t linear,
                         const analysis::IndexRelationLimits &limits) const {
  const size_t pointCount = roots.size() * tiles.size();
  for (; linear < pointCount; ++linear) {
    const SemanticRootKey &root = roots[linear / tiles.size()];
    TileId tile = tiles[linear % tiles.size()];
    analysis::RootRegionWorkOutcome outcome =
        analysis.query(root, tile, limits);
    if (std::holds_alternative<analysis::NoRootRegionWork>(outcome))
      continue;
    return fromOutcome(std::move(outcome));
  }
  return {RootWorkSuccessorKind::End};
}

RootWorkSuccessor RootWorkDomain::getFirstWork(
    const analysis::IndexRelationLimits &limits) const {
  return scanFrom(0, limits);
}

RootWorkSuccessor
RootWorkDomain::getNextWork(const RootWorkCursor &currentCursor,
                            const analysis::IndexRelationLimits &limits) const {
  const analysis::RootRegionWorkId &current = currentCursor.getId();
  auto root = llvm::lower_bound(roots, current.root);
  auto tile = llvm::lower_bound(tiles, current.tile, tileLess);
  if (root == roots.end() || *root != current.root || tile == tiles.end() ||
      *tile != current.tile)
    return {RootWorkSuccessorKind::CompilerBug,
            {},
            RootWorkDomainFailure(invalidCursor(
                current, "root-work successor cursor is outside its domain"))};
  const size_t rootIndex = std::distance(roots.begin(), root);
  const size_t tileIndex = std::distance(tiles.begin(), tile);
  return scanFrom(rootIndex * tiles.size() + tileIndex + 1, limits);
}

RootWorkCollectionOutcome
collectRootWorks(const RootWorkDomain &domain,
                 const analysis::IndexRelationLimits &limits) {
  RootWorkCollection collection;
  RootWorkSuccessor current = domain.getFirstWork(limits);
  while (current.getKind() == RootWorkSuccessorKind::Work) {
    const RootWorkCursor *cursor = current.getCursor();
    if (!current.getWork() || !cursor)
      return invalidCursor({}, "root-work successor omitted its work value");
    RootWorkCursor nextCursor = *cursor;
    std::optional<analysis::RootRegionWork> work = current.takeWork();
    collection.works.push_back(std::move(*work));
    current = domain.getNextWork(nextCursor, limits);
  }
  if (current.getKind() == RootWorkSuccessorKind::End)
    return collection;
  const RootWorkDomainFailure *failure = current.getFailure();
  if (!failure)
    return invalidCursor({}, "root-work successor omitted its typed failure");
  return std::visit(
      [](const auto &value) -> RootWorkCollectionOutcome { return value; },
      *failure);
}

const RootWorkCollection *
getRootWorkCollection(const RootWorkCollectionOutcome &outcome) {
  return std::get_if<RootWorkCollection>(&outcome);
}

RootWorkCollection *getRootWorkCollection(RootWorkCollectionOutcome &outcome) {
  return std::get_if<RootWorkCollection>(&outcome);
}

} // namespace wafer::compiler::detail
