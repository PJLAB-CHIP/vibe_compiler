//===- TemporalProposals.h - Feedback ordered integer choices -*- C++ -*-===//

#ifndef WAFER_DRIVER_PHYSICALDATAFLOW_TEMPORALPROPOSALS_H
#define WAFER_DRIVER_PHYSICALDATAFLOW_TEMPORALPROPOSALS_H

#include "Wafer/Analysis/Instr/CostModel.h"
#include "Wafer/Planning/PhysicalDataflow/TemporalDomain.h"

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

enum class TemporalProposalKind : uint8_t { Explore, Repair, Improve, Fine };

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
  /// Prepares at most one distinct point from a lazy direction cursor.
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

private:
  using Coordinate = TemporalCoordinate;
  struct ImprovementPoll {
    size_t anchor;
    std::vector<Coordinate> coordinates;
    std::vector<std::vector<Coordinate>> groups;
    size_t batch = 0;
    size_t phase = 0;
    size_t single = 0;
    size_t pairFirst = 0;
    size_t pairSecond = 1;
    size_t pairDirection = 0;
    size_t orderDomain = 0;
    size_t orderScope = 0;
    size_t orderPosition = 0;
  };
  struct FinePoll {
    size_t anchor;
    std::vector<Coordinate> coordinates;
    size_t position = 0;
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
    TemporalProposalKind kind = TemporalProposalKind::Explore;
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
  bool advanceFinePoll();
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
  std::array<std::deque<size_t>, 4> queues;
  std::deque<CapacityPoll> capacityPolls;
  bool preferFreshCapacity = true;
  std::optional<size_t> priorityCapacityAnchor;
  std::deque<ImprovementPoll> improvementPolls;
  std::optional<analysis::KnownSearchObjective> bestObjective;
  std::optional<size_t> bestEntry;
  bool fineStarted = false;
  std::optional<FinePoll> finePoll;
  std::vector<SeedFamily> seedFamilies;
  std::deque<SeedPoint> seedPoints;
  std::optional<size_t> firstCapacityAnchor;
  bool firstCapacityRoundComplete = false;
  std::set<size_t> firstCapacityPending;
};

} // namespace wafer::compiler::detail

#endif // WAFER_DRIVER_PHYSICALDATAFLOW_TEMPORALPROPOSALS_H
