//===- TemporalProposals.h - Feedback ordered integer choices -*- C++ -*-===//

#ifndef WAFER_DRIVER_PHYSICALDATAFLOW_TEMPORALPROPOSALS_H
#define WAFER_DRIVER_PHYSICALDATAFLOW_TEMPORALPROPOSALS_H

#include "Wafer/Analysis/Instr/CostModel.h"
#include "Wafer/Planning/PhysicalDataflow/TemporalDomain.h"
#include "llvm/ADT/DenseMap.h"

#include <array>
#include <cstdint>
#include <deque>
#include <optional>
#include <set>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace wafer::compiler::detail {

enum class TemporalProposalKind : uint8_t { Explore, Repair, Improve };

struct TemporalCoordinate {
  size_t domain, scope, iterator;

  friend bool operator<(TemporalCoordinate lhs, TemporalCoordinate rhs) {
    return std::tie(lhs.domain, lhs.scope, lhs.iterator) <
           std::tie(rhs.domain, rhs.scope, rhs.iterator);
  }
};

/// Orders explicit numeric choices in unchanged structural domains. Feedback
/// consists only of observed costs and current allocation-certificate scopes;
/// this object owns no candidate IR, memory facts or legality conclusions.
class TemporalProposals {
public:
  explicit TemporalProposals(std::vector<const TemporalDomain *> domains)
      : domains(std::move(domains)) {}

  void seed(const std::vector<TemporalChoice> &initial);
  bool startAt(const std::vector<TemporalChoice> &choices) {
    return append(choices, TemporalProposalKind::Explore, true);
  }
  /// Advances a lazy direction until a point is queued; a direction can also
  /// queue its exact relation-coordinated alternative. take() returns one.
  bool prepareNext(TemporalProposalKind kind);
  std::vector<TemporalChoice> take(TemporalProposalKind kind);
  /// Records an ordinary raw-domain point, unless already queued or visited.
  bool visitRaw(const std::vector<TemporalChoice> &choices);
  void observeAccepted(const std::vector<TemporalChoice> &choices,
                       const analysis::KnownSearchObjective &objective);
  bool observeCapacity(const std::vector<TemporalChoice> &choices,
                       const std::set<TemporalCoordinate> &affectedCoordinates,
                       bool prioritize = false);
  bool hasCapacityRoundInProgress() const {
    return !bestObjective && firstCapacityAnchor &&
           (!firstCapacityRoundComplete || !firstCapacityPending.empty());
  }
  uint64_t getPerformanceRound() const { return performanceRound; }

private:
  using Coordinate = TemporalCoordinate;
  struct ImprovementPoll {
    size_t anchor;
    std::vector<std::vector<std::pair<Coordinate, int>>> directions;
    size_t position = 0;
    size_t orderDomain = 0;
    size_t orderScope = 0;
    size_t orderPosition = 0;
  };
  struct AxisScale {
    int64_t step = 0;
    std::optional<int64_t> alignment;
    uint64_t rounds = 0;
  };
  enum class SeedVariant { Full, Kernel, Axis, Coupled, All };
  struct SeedFamily {
    std::vector<TemporalChoice> initial;
    std::vector<Coordinate> coordinates;
    std::vector<std::vector<Coordinate>> groups;
  };
  struct SeedPoint {
    size_t family;
    size_t level;
    SeedVariant variant;
    size_t axis = 0;
  };
  struct Entry {
    std::vector<TemporalChoice> choices;
    std::optional<analysis::KnownSearchObjective> bestObjective;
    std::set<Coordinate> capacityObserved;
    bool taken = false;
    uint64_t repairDepth = 0;
  };
  struct CapacityPoll {
    size_t anchor;
    std::vector<std::vector<Coordinate>> groups;
    std::vector<Coordinate> coordinates;
    size_t batch = 0;
    size_t single = 0;
    size_t pairFirst = 0;
    size_t pairSecond = 1;
    bool initialRound = false;
    bool remainingWholeDirection = false;
  };

  std::optional<size_t> find(const std::vector<TemporalChoice> &choices) const;
  bool append(std::vector<TemporalChoice> choices, TemporalProposalKind kind,
              bool preserveOrder = false);
  bool appendCapacityDirection();
  bool appendSeedPoint();
  bool advanceImprovementPoll();
  void initializeScales(const std::vector<TemporalChoice> &choices);
  void beginImprovementPoll();
  bool changeSize(std::vector<TemporalChoice> &choices, Coordinate coordinate,
                  int direction) const;
  std::optional<int64_t>
  getAlignment(const std::vector<TemporalChoice> &choices,
               Coordinate coordinate) const;
  bool complete(std::vector<TemporalChoice> &choices) const;
  void rankGroups(const std::vector<TemporalChoice> &choices,
                  std::vector<std::vector<Coordinate>> &groups,
                  bool shrinking) const;
  std::vector<Coordinate>
  coordinates(const std::vector<TemporalChoice> &choices) const;
  TemporalSizeInterval bounds(const std::vector<TemporalChoice> &choices,
                              Coordinate coordinate) const;
  static int64_t &value(std::vector<TemporalChoice> &choices,
                        Coordinate coordinate);

  std::vector<const TemporalDomain *> domains;
  std::vector<Entry> entries;
  std::unordered_map<size_t, std::vector<size_t>> entryIndex;
  std::array<std::deque<size_t>, 3> queues;
  std::deque<CapacityPoll> capacityPolls;
  bool preferFreshCapacity = true;
  std::optional<size_t> priorityCapacityAnchor;
  std::optional<ImprovementPoll> improvementPoll;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<AxisScale, 4>> scales;
  uint64_t performanceRound = 0;
  uint64_t performanceRounds = 0;
  bool performanceStarted = false;
  std::optional<analysis::KnownSearchObjective> bestObjective;
  std::optional<size_t> bestEntry;
  std::vector<SeedFamily> seedFamilies;
  std::deque<SeedPoint> seedPoints;
  std::optional<size_t> firstCapacityAnchor;
  bool firstCapacityRoundComplete = false;
  std::set<size_t> firstCapacityPending;
};

} // namespace wafer::compiler::detail

#endif // WAFER_DRIVER_PHYSICALDATAFLOW_TEMPORALPROPOSALS_H
