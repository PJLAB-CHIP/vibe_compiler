//===- PhysicalVersionBuilder.h - Selected physical versions -*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_PHYSICALVERSIONBUILDER_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_PHYSICALVERSIONBUILDER_H

#include "Wafer/Planning/PhysicalDataflow/RepresentationDomain.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include <map>
#include <string>
#include <vector>

namespace wafer::compiler::detail {

struct PreparedRepresentationPlan {
  std::vector<PhysicalVersionPlan> primaryVersions;
  std::vector<PhysicalVersionPlan> derivedVersions;
  std::vector<PhysicalUseBinding> uses;
};

mlir::FailureOr<PreparedRepresentationPlan>
prepareRepresentationPlan(const RepresentationDomain &domain,
                          const RepresentationPlan &plan,
                          std::string *failureReason = nullptr);

/// Candidate-local binding table keyed only by the selected semantic version
/// ID. It never searches by source SSA value, layout, operation or first use.
class PhysicalVersionBuilder {
public:
  mlir::LogicalResult bind(const PhysicalVersionId &id, mlir::Value value,
                           std::string *failureReason = nullptr);
  mlir::Value lookup(const PhysicalVersionId &id) const;

private:
  std::map<PhysicalVersionId, mlir::Value> values;

  friend mlir::LogicalResult
  emitPreparedRepresentationVersions(const PreparedRepresentationPlan &,
                                     PhysicalVersionBuilder &,
                                     mlir::OpBuilder &, std::string *);
};

/// Emits all selected derived versions exactly once after a mutation-free
/// preflight of sources, layouts and target types. Primary values must already
/// be bound by boundary/execution owners.
mlir::LogicalResult
emitPreparedRepresentationVersions(const PreparedRepresentationPlan &prepared,
                                   PhysicalVersionBuilder &versions,
                                   mlir::OpBuilder &builder,
                                   std::string *failureReason = nullptr);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_PHYSICALVERSIONBUILDER_H
