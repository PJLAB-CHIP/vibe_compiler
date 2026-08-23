//===- RegionDomain.h - Region execution and use domain ------*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_REGIONDOMAIN_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_REGIONDOMAIN_H

#include "Wafer/Planning/PhysicalDataflow/CanonicalRegionPlan.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace wafer::compiler::detail {

enum class RegionSuccessorKind : uint8_t { Plan, End, CompilerBug };

class RegionCursor {
private:
  llvm::SmallVector<llvm::SmallVector<uint32_t, 8>, 8> labels;
  llvm::SmallVector<uint8_t, 16> fragmentChoices;

  friend class RegionDomain;
};

class RegionSuccessor {
public:
  RegionSuccessorKind getKind() const { return kind; }
  const RegionPlan *getPlan() const { return plan ? &*plan : nullptr; }
  const RegionCursor *getCursor() const { return cursor ? &*cursor : nullptr; }
  llvm::StringRef getDetail() const { return detail; }

private:
  RegionSuccessor(RegionSuccessorKind kind, std::optional<RegionPlan> plan = {},
                  std::optional<RegionCursor> cursor = {},
                  std::string detail = {})
      : kind(kind), plan(std::move(plan)), cursor(std::move(cursor)),
        detail(std::move(detail)) {}

  RegionSuccessorKind kind;
  std::optional<RegionPlan> plan;
  std::optional<RegionCursor> cursor;
  std::string detail;

  friend class RegionDomain;
};

/// Complete lazy domain of connected root-work partitions and every current
/// External/Stored/Direct/Replica use choice. It owns only typed descriptors
/// derived from RootRegionWork and never owns source or candidate IR.
class RegionDomain {
public:
  static mlir::FailureOr<RegionDomain>
  create(llvm::ArrayRef<analysis::RootRegionWork> rootWorks,
         std::string *failureReason = nullptr);

  RegionSuccessor getFirstPlan() const;
  RegionSuccessor getNextPlan(const RegionCursor &cursor) const;
  bool contains(const RegionPlan &plan) const;

private:
  struct Component {
    TileId tile{0};
    llvm::SmallVector<analysis::RootRegionWorkId, 8> works;
    llvm::SmallVector<uint8_t, 64> potentialEdges;
  };

  struct LocalFragment {
    DemandFragmentId fragment;
    analysis::RootRegionWorkId producerWork;
    analysis::RootRegionWorkId consumerWork;
    ExecutionInstanceId producer;
    ExecutionInstanceId consumer;
    bool allowsRequiredLocal = false;
    bool allowsDirect = false;
    bool allowsReplica = false;
  };

  RegionDomain(std::vector<RegionGroupPlan> baseGroups,
               llvm::SmallVector<Component, 16> components,
               std::vector<LocalFragment> localFragments)
      : baseGroups(std::move(baseGroups)), components(std::move(components)),
        localFragments(std::move(localFragments)) {}

  llvm::SmallVector<llvm::SmallVector<uint32_t, 8>, 8> getFirstLabels() const;
  bool advanceLabels(
      llvm::SmallVectorImpl<llvm::SmallVector<uint32_t, 8>> &labels) const;
  std::vector<const LocalFragment *> getChoiceFragments(
      llvm::ArrayRef<llvm::SmallVector<uint32_t, 8>> labels) const;
  std::optional<RegionPlan>
  buildPlan(llvm::ArrayRef<llvm::SmallVector<uint32_t, 8>> labels,
            llvm::ArrayRef<uint8_t> choices) const;
  bool advanceChoices(llvm::ArrayRef<llvm::SmallVector<uint32_t, 8>> labels,
                      llvm::ArrayRef<const LocalFragment *> fragments,
                      llvm::SmallVectorImpl<uint8_t> &choices) const;
  RegionSuccessor findPlan(RegionCursor cursor, bool advanceCurrent) const;
  std::optional<RegionCursor> getCursor(const RegionPlan &plan) const;

  std::vector<RegionGroupPlan> baseGroups;
  llvm::SmallVector<Component, 16> components;
  std::vector<LocalFragment> localFragments;
};

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_REGIONDOMAIN_H
