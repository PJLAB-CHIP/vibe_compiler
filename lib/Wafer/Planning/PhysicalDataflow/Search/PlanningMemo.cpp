//===- PlanningMemo.cpp - Session-local typed query memo ---------------===//

#include "Wafer/Planning/PhysicalDataflow/Search/PlanningMemo.h"

#include "Wafer/Planning/PhysicalDataflow/Search/PlanningProfile.h"

namespace wafer::compiler::detail {

llvm::StringRef stringifyPlanningMemoKind(PlanningMemoKind kind) {
  switch (kind) {
  case PlanningMemoKind::SpatialProposals:
    return "spatial-proposals";
  case PlanningMemoKind::RootWork:
    return "root-work";
  case PlanningMemoKind::RegionDomain:
    return "region-domain";
  case PlanningMemoKind::TemporalDomain:
    return "temporal-domain";
  case PlanningMemoKind::RepresentationDomain:
    return "representation-domain";
  case PlanningMemoKind::MovementDomain:
    return "movement-domain";
  case PlanningMemoKind::StorageDomain:
    return "storage-domain";
  case PlanningMemoKind::InitialEventGraph:
    return "initial-event-graph";
  case PlanningMemoKind::PostStructureEventGraph:
    return "post-structure-event-graph";
  case PlanningMemoKind::ExecutionStructureDomain:
    return "execution-structure-domain";
  case PlanningMemoKind::StructureSpecificStorageDomain:
    return "structure-specific-storage-domain";
  case PlanningMemoKind::ScheduleDomain:
    return "schedule-domain";
  case PlanningMemoKind::Count:
    break;
  }
  return "unknown";
}

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
