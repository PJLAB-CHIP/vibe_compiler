//===- TemporalProposals.h - Feedback ordered integer choices -*- C++ -*-===//

#ifndef WAFER_DRIVER_PHYSICALDATAFLOW_TEMPORALPROPOSALS_H
#define WAFER_DRIVER_PHYSICALDATAFLOW_TEMPORALPROPOSALS_H

#include "Wafer/Planning/PhysicalDataflow/TemporalDomain.h"

#include <array>
#include <cstdint>
#include <deque>
#include <optional>
#include <set>
#include <utility>
#include <vector>

namespace wafer::compiler::detail {

enum class TemporalProposalKind : uint8_t { Explore, Repair, Improve };

/// Orders explicit numeric choices in unchanged structural domains. Feedback
/// consists only of observed costs and current allocation-certificate scopes;
/// this object owns no candidate IR, memory facts or legality conclusions.
class TemporalProposals {
public:
  explicit TemporalProposals(std::vector<const TemporalDomain *> domains)
      : domains(std::move(domains)) {}

  void seed(const std::vector<TemporalChoice> &initial, size_t stratum = 0);
  bool empty(TemporalProposalKind kind) const;
  std::vector<TemporalChoice> take(TemporalProposalKind kind);
  /// Records an ordinary raw-domain point, unless already queued or visited.
  bool visitRaw(const std::vector<TemporalChoice> &choices);
  void observeAccepted(const std::vector<TemporalChoice> &choices,
                       uint64_t duration);
  bool observeCapacity(const std::vector<TemporalChoice> &choices,
                       const std::set<size_t> &affectedDomains);

private:
  struct Coordinate {
    size_t domain, scope, iterator;
  };
  struct Probe {
    std::vector<TemporalChoice> anchor;
    std::vector<Coordinate> coordinates;
    int direction = -1;
    int64_t distance = 1;
    std::optional<uint64_t> referenceDuration;
  };
  struct Entry {
    std::vector<TemporalChoice> choices;
    std::optional<Probe> probe;
    std::optional<uint64_t> bestDuration;
    std::set<size_t> capacityObserved;
  };

  std::optional<size_t> find(const std::vector<TemporalChoice> &choices) const;
  bool append(std::vector<TemporalChoice> choices, TemporalProposalKind kind,
              std::optional<Probe> probe = {});
  bool appendProbe(Probe probe, TemporalProposalKind kind);
  bool complete(std::vector<TemporalChoice> &choices) const;
  std::vector<Coordinate>
  coordinates(const std::vector<TemporalChoice> &choices) const;
  TemporalSizeInterval bounds(const std::vector<TemporalChoice> &choices,
                              Coordinate coordinate) const;
  static int64_t &value(std::vector<TemporalChoice> &choices,
                        Coordinate coordinate);
  static int64_t expandDistance(int64_t distance);
  void appendLayoutBoundaries(const std::vector<TemporalChoice> &choices,
                              Coordinate coordinate, uint64_t duration);

  std::vector<const TemporalDomain *> domains;
  std::vector<Entry> entries;
  std::array<std::deque<size_t>, 3> queues;
};

} // namespace wafer::compiler::detail

#endif // WAFER_DRIVER_PHYSICALDATAFLOW_TEMPORALPROPOSALS_H
