//===- MovementDomain.h - Explicit transfer realization domain -*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_MOVEMENTDOMAIN_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_MOVEMENTDOMAIN_H

#include "Wafer/Planning/PhysicalDataflow/CanonicalMovementPlan.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace wafer::compiler::detail {

enum class MovementDomainFailureKind : uint8_t {
  UnsupportedSemantics,
  BrokenContract,
};

struct MovementDomainFailure {
  MovementDomainFailureKind kind = MovementDomainFailureKind::BrokenContract;
  std::string detail;
};

enum class MovementSuccessorKind : uint8_t { Plan, End, CompilerBug };

class MovementCursor {
public:
  struct TreeChoice {
    size_t rootChoice = 0;
    uint64_t relayMask = 0;
    std::vector<size_t> parentChoices;
  };
  struct ClassChoice {
    /// Restricted-growth digits: zero is DDR; positive values are canonical
    /// peer-group labels in first-occurrence order.
    std::vector<uint32_t> membership;
    std::vector<TreeChoice> trees;
  };

private:
  std::vector<ClassChoice> classes;

  friend class MovementDomain;
};

struct EndpointTransferCapability {
  TileId source{0};
  TileId destination{0};

  friend bool operator==(const EndpointTransferCapability &lhs,
                         const EndpointTransferCapability &rhs) {
    return lhs.source == rhs.source && lhs.destination == rhs.destination;
  }
  friend bool operator<(const EndpointTransferCapability &lhs,
                        const EndpointTransferCapability &rhs) {
    return std::tuple(lhs.source.getValue(), lhs.destination.getValue()) <
           std::tuple(rhs.source.getValue(), rhs.destination.getValue());
  }
};

/// Immutable target facts consumed by the movement query. Current TX81 uses
/// an opaque end-to-end endpoint-transfer graph. Physical mesh links are not
/// endpoint routes and therefore do not appear here.
struct MovementTransportFacts {
  std::vector<TileId> availableTiles;
  std::vector<EndpointTransferCapability> endpointTransfers;
  bool ddrStagesSupported = true;
};

class MovementSuccessor {
public:
  MovementSuccessorKind getKind() const { return kind; }
  const MovementPlan *getPlan() const { return plan ? &*plan : nullptr; }
  const MovementCursor *getCursor() const {
    return cursor ? &*cursor : nullptr;
  }
  llvm::StringRef getDetail() const { return detail; }

private:
  MovementSuccessor(MovementSuccessorKind kind,
                    std::optional<MovementPlan> plan = {},
                    std::optional<MovementCursor> cursor = {},
                    std::string detail = {})
      : kind(kind), plan(std::move(plan)), cursor(std::move(cursor)),
        detail(std::move(detail)) {}

  MovementSuccessorKind kind;
  std::optional<MovementPlan> plan;
  std::optional<MovementCursor> cursor;
  std::string detail;

  friend class MovementDomain;
};

struct MovementDomainResult;

enum class MovementPlanActionKind : uint8_t {
  ExternalLoad,
  BoundaryTransfer,
  Gather,
};

/// Complete lazy realization domain for the finite payload requirements in
/// the current coordinate. Compatible payloads enumerate every destination
/// partition; each peer block enumerates every verifier-legal rooted
/// arborescence over any active relay subset. DDR remains an independent
/// sibling. Classic stars, chains and trees are proposals, not separate
/// legality rules.
class MovementDomain {
public:
  MovementSuccessor getFirstPlan() const;
  MovementSuccessor getNextPlan(const MovementCursor &cursor) const;
  bool contains(const MovementPlan &plan) const;
  std::vector<MovementPlan> getProposals() const;

  llvm::ArrayRef<MovementResourceDescription> getResources() const {
    return resources;
  }

private:
  struct ActionVariable {
    MovementPlanActionKind kind = MovementPlanActionKind::BoundaryTransfer;
    size_t planIndex = 0;
    MovementActionId action;
    TileId source{0};
    TileId destination{0};
    bool peerCapable = false;
  };

  struct ReuseClass {
    std::vector<ActionVariable> actions;
  };

  MovementDomain(MovementPlan base,
                 std::vector<MovementResourceDescription> resources,
                 MovementTransportFacts transport,
                 std::vector<ReuseClass> classes)
      : base(std::move(base)), resources(std::move(resources)),
        transport(std::move(transport)), classes(std::move(classes)) {}

  bool advanceClassChoice(size_t classIndex,
                          MovementCursor::ClassChoice &choice) const;
  std::optional<MovementPlan> buildPlan(const MovementCursor &cursor) const;
  std::optional<MovementCursor> getCursor(const MovementPlan &plan) const;

  MovementPlan base;
  std::vector<MovementResourceDescription> resources;
  MovementTransportFacts transport;
  std::vector<ReuseClass> classes;

  friend MovementDomainResult
  buildMovementDomain(const CanonicalMovementCoordinate &,
                      const RepresentationPlan &, llvm::ArrayRef<TileId>);
  friend MovementDomainResult
  buildMovementDomain(const CanonicalMovementCoordinate &,
                      const RepresentationPlan &,
                      const MovementTransportFacts &);
};

struct MovementDomainResult {
  std::optional<MovementDomain> domain;
  std::optional<MovementDomainFailure> failure;

  bool succeeded() const { return domain.has_value(); }
};

MovementDomainResult
buildMovementDomain(const CanonicalMovementCoordinate &canonical,
                    const RepresentationPlan &representations,
                    llvm::ArrayRef<TileId> availableTiles);

MovementDomainResult
buildMovementDomain(const CanonicalMovementCoordinate &canonical,
                    const RepresentationPlan &representations,
                    const MovementTransportFacts &transport);

MovementTransportFacts
buildOpaqueEndpointTransportFacts(llvm::ArrayRef<TileId> availableTiles);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_MOVEMENTDOMAIN_H
