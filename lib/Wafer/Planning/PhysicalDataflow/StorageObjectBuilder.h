//===- StorageObjectBuilder.h - Selected storage objects -----*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_STORAGEOBJECTBUILDER_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_STORAGEOBJECTBUILDER_H

#include "Wafer/Planning/PhysicalDataflow/StorageDomain.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include <map>
#include <string>
#include <vector>

namespace wafer::compiler::detail {

struct PreparedStorageObject {
  StorageObjectPlan plan;
  StorageResourceDescription resource;
  uint32_t multiplicity = 1;
};

struct PreparedStoragePlan {
  std::vector<PreparedStorageObject> objects;
  std::vector<PhysicalVersionStorageBinding> bindings;
  std::vector<SlotFamilyPlan> slotFamilies;
};

mlir::FailureOr<PreparedStoragePlan>
prepareStoragePlan(const StorageDomain &domain, const BufferPlan &plan,
                   std::string *failureReason = nullptr);

class StorageObjectBuilder {
public:
  mlir::Value lookup(const PhysicalVersionId &version,
                     uint64_t occurrence = 0) const;

private:
  std::map<StorageObjectId, std::vector<mlir::Value>> objects;
  std::map<PhysicalVersionId, StorageObjectId> bindings;

  friend mlir::LogicalResult
  emitPreparedStorageObjects(const PreparedStoragePlan &,
                             StorageObjectBuilder &, mlir::OpBuilder &,
                             std::string *);
};

/// Creates exact selected object multiplicities once. Slot selection is a
/// pure modulo lookup from an explicit occurrence supplied by the temporal or
/// execution-structure emitter; this builder never discovers loops or changes
/// multiplicity.
mlir::LogicalResult emitPreparedStorageObjects(
    const PreparedStoragePlan &prepared, StorageObjectBuilder &objects,
    mlir::OpBuilder &builder, std::string *failureReason = nullptr);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_STORAGEOBJECTBUILDER_H
