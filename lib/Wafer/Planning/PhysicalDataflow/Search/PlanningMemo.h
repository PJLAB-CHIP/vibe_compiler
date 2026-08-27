//===- PlanningMemo.h - Session-local typed query memo -------*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SEARCH_PLANNINGMEMO_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SEARCH_PLANNINGMEMO_H

#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <map>
#include <utility>

namespace wafer::compiler::detail {

class PlanningProfileSink;

/// Stable names for the pure, session-local query memos owned by physical
/// planning. Values are typed planning facts only; candidate IR and memory
/// planning results never enter these tables.
enum class PlanningMemoKind : unsigned {
  SpatialProposals,
  RootWork,
  RegionDomain,
  TemporalDomain,
  RepresentationDomain,
  MovementDomain,
  StorageDomain,
  InitialEventGraph,
  PostStructureEventGraph,
  ExecutionStructureDomain,
  StructureSpecificStorageDomain,
  ScheduleDomain,
  Count,
};

llvm::StringRef stringifyPlanningMemoKind(PlanningMemoKind kind);

void recordPlanningMemoLookup(PlanningProfileSink *profile,
                              PlanningMemoKind kind, bool hit);
void recordPlanningMemoEntries(PlanningProfileSink *profile,
                               PlanningMemoKind kind, size_t entries);

/// A full-key std::map with optional observation. Profiling never changes the
/// key, lookup, insertion, lifetime, or eviction behavior.
template <typename Key, typename Value> class PlanningMemo {
  using Map = std::map<Key, Value>;

public:
  using iterator = typename Map::iterator;
  using const_iterator = typename Map::const_iterator;

  PlanningMemo(PlanningMemoKind kind, PlanningProfileSink *profile = nullptr)
      : kind(kind), profile(profile) {}

  iterator find(const Key &key) {
    iterator found = values.find(key);
    recordPlanningMemoLookup(profile, kind, found != values.end());
    return found;
  }
  const_iterator find(const Key &key) const {
    const_iterator found = values.find(key);
    recordPlanningMemoLookup(profile, kind, found != values.end());
    return found;
  }

  size_t count(const Key &key) const {
    const bool found = values.count(key) != 0;
    recordPlanningMemoLookup(profile, kind, found);
    return found ? 1 : 0;
  }

  template <typename... Args>
  std::pair<iterator, bool> try_emplace(const Key &key, Args &&...args) {
    auto inserted = values.try_emplace(key, std::forward<Args>(args)...);
    if (inserted.second)
      recordPlanningMemoEntries(profile, kind, values.size());
    return inserted;
  }

  iterator end() { return values.end(); }
  const_iterator end() const { return values.end(); }
  size_t size() const { return values.size(); }

private:
  Map values;
  PlanningMemoKind kind;
  PlanningProfileSink *profile = nullptr;
};

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SEARCH_PLANNINGMEMO_H
