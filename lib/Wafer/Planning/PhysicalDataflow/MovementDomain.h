//===- MovementDomain.h - Explicit transfer realization domain -*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_MOVEMENTDOMAIN_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_MOVEMENTDOMAIN_H

#include "Wafer/Planning/PhysicalDataflow/CanonicalMovementPlan.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>
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
private:
  struct Choice {
    MovementRealizationKind kind = MovementRealizationKind::DDRStage;
    std::vector<TileId> relays;
  };
  std::vector<Choice> choices;

  friend class MovementDomain;
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

/// Complete lazy realization domain for current unicast boundaries. DDR is
/// always retained; cross-Tile payloads additionally have one opaque
/// target-routed transfer and every explicit simple software-relay chain.
class MovementDomain {
public:
  MovementSuccessor getFirstPlan() const;
  MovementSuccessor getNextPlan(const MovementCursor &cursor) const;
  bool contains(const MovementPlan &plan) const;

  llvm::ArrayRef<MovementResourceDescription> getResources() const {
    return resources;
  }

private:
  struct Variable {
    bool gather = false;
    size_t planIndex = 0;
    TileId source{0};
    TileId destination{0};
    bool peerCapable = false;
    std::vector<TileId> availableRelays;
  };

  MovementDomain(MovementPlan base,
                 std::vector<MovementResourceDescription> resources,
                 std::vector<TileId> availableTiles,
                 std::vector<Variable> variables)
      : base(std::move(base)), resources(std::move(resources)),
        availableTiles(std::move(availableTiles)),
        variables(std::move(variables)) {}

  bool advanceChoice(size_t variable, MovementCursor::Choice &choice) const;
  std::optional<MovementPlan> buildPlan(const MovementCursor &cursor) const;
  std::optional<MovementCursor> getCursor(const MovementPlan &plan) const;

  MovementPlan base;
  std::vector<MovementResourceDescription> resources;
  std::vector<TileId> availableTiles;
  std::vector<Variable> variables;

  friend MovementDomainResult
  buildMovementDomain(const CanonicalMovementCoordinate &,
                      const RepresentationPlan &, llvm::ArrayRef<TileId>);
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

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_MOVEMENTDOMAIN_H
