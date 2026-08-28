//===- PlanningMemo.cpp - Session-local typed query memo ---------------===//

#include "Wafer/Planning/PhysicalDataflow/Search/PlanningMemo.h"

#include "Wafer/Planning/PhysicalDataflow/Search/PlanningProfile.h"

namespace wafer::compiler::detail {

void recordPlanningMemoLookup(PlanningProfileSink *profile,
                              PlanningMemoKind kind, bool hit) {
  if (profile)
    profile->recordMemoLookup(kind, hit);
}

void recordPlanningMemoEntries(PlanningProfileSink *profile,
                               PlanningMemoKind kind, size_t entries) {
  if (profile)
    profile->recordMemoEntries(kind, entries);
}

} // namespace wafer::compiler::detail
