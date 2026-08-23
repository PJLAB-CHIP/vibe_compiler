//===- RootWorkDomain.h - Complete root and Tile work domain -*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_ROOTWORKDOMAIN_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_ROOTWORKDOMAIN_H

#include "Wafer/Planning/PhysicalDataflow/RootRegionWorkAnalysis.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace wafer::compiler::detail {

enum class RootWorkSuccessorKind : uint8_t {
  Work,
  End,
  Unsupported,
  Indeterminate,
  CompilerBug,
};

enum class RootWorkSiteOutcomeKind : uint8_t {
  Work,
  NoWork,
  Unsupported,
  Indeterminate,
  CompilerBug,
};

using RootWorkDomainFailure = std::variant<analysis::UnsupportedRootRegionWork,
                                           analysis::RootRegionWorkLimitReached,
                                           analysis::BrokenRootRegionWork>;

RootWorkSiteOutcomeKind
classifyRootWorkOutcome(const analysis::RootRegionWorkOutcome &outcome);

/// Stable successor position created only by RootWorkDomain.
class RootWorkCursor {
public:
  const analysis::RootRegionWorkId &getId() const { return id; }

private:
  explicit RootWorkCursor(analysis::RootRegionWorkId id) : id(std::move(id)) {}

  analysis::RootRegionWorkId id;

  friend class RootWorkDomain;
};

class RootWorkSuccessor {
public:
  RootWorkSuccessorKind getKind() const { return kind; }
  const analysis::RootRegionWork *getWork() const {
    return work ? &*work : nullptr;
  }
  std::optional<analysis::RootRegionWork> takeWork() { return std::move(work); }
  const RootWorkDomainFailure *getFailure() const {
    return failure ? &*failure : nullptr;
  }
  const RootWorkCursor *getCursor() const {
    return cursor ? &*cursor : nullptr;
  }

private:
  RootWorkSuccessor(RootWorkSuccessorKind kind,
                    std::optional<analysis::RootRegionWork> work = {},
                    std::optional<RootWorkDomainFailure> failure = {},
                    std::optional<RootWorkCursor> cursor = {})
      : kind(kind), work(std::move(work)), failure(std::move(failure)),
        cursor(std::move(cursor)) {}

  RootWorkSuccessorKind kind;
  std::optional<analysis::RootRegionWork> work;
  std::optional<RootWorkDomainFailure> failure;
  std::optional<RootWorkCursor> cursor;

  friend class RootWorkDomain;
};

/// Lazy lexicographic domain of every nonempty `(semantic root, Tile)` work.
/// The domain borrows one unchanged DAG, assignment, and exact-demand proof;
/// they must outlive this object. It owns no IR and stores no RootRegionWork
/// point vector.
class RootWorkDomain {
public:
  static mlir::FailureOr<RootWorkDomain>
  create(const StructuredDAGAnalysis &dag, const SpatialAssignment &assignment,
         const analysis::ExactDemandProof &proof,
         llvm::ArrayRef<TileId> availableTiles,
         std::string *failureReason = nullptr);

  RootWorkSuccessor getFirstWork(const analysis::IndexRelationLimits &limits =
                                     analysis::IndexRelationLimits()) const;
  RootWorkSuccessor getNextWork(const RootWorkCursor &current,
                                const analysis::IndexRelationLimits &limits =
                                    analysis::IndexRelationLimits()) const;

  bool contains(const analysis::RootRegionWorkId &id) const;
  llvm::ArrayRef<SemanticRootKey> getRoots() const { return roots; }
  llvm::ArrayRef<TileId> getTiles() const { return tiles; }

private:
  RootWorkDomain(RootRegionWorkAnalysis analysis,
                 llvm::SmallVector<SemanticRootKey, 16> roots,
                 llvm::SmallVector<TileId, 16> tiles)
      : analysis(std::move(analysis)), roots(std::move(roots)),
        tiles(std::move(tiles)) {}

  RootWorkSuccessor scanFrom(size_t linear,
                             const analysis::IndexRelationLimits &limits) const;
  static RootWorkSuccessor fromOutcome(analysis::RootRegionWorkOutcome outcome);

  RootRegionWorkAnalysis analysis;
  llvm::SmallVector<SemanticRootKey, 16> roots;
  llvm::SmallVector<TileId, 16> tiles;
};

struct RootWorkCollection {
  std::vector<analysis::RootRegionWork> works;
};

using RootWorkCollectionOutcome =
    std::variant<RootWorkCollection, analysis::UnsupportedRootRegionWork,
                 analysis::RootRegionWorkLimitReached,
                 analysis::BrokenRootRegionWork>;

RootWorkCollectionOutcome
collectRootWorks(const RootWorkDomain &domain,
                 const analysis::IndexRelationLimits &limits =
                     analysis::IndexRelationLimits());

const RootWorkCollection *
getRootWorkCollection(const RootWorkCollectionOutcome &outcome);
RootWorkCollection *getRootWorkCollection(RootWorkCollectionOutcome &outcome);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_ROOTWORKDOMAIN_H
