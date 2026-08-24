//===- StorageObjectBuilder.h - Selected storage objects -----*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_STORAGEOBJECTBUILDER_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_STORAGEOBJECTBUILDER_H

#include "Wafer/Planning/PhysicalDataflow/ExecutionStructurePlan.h"
#include "Wafer/Planning/PhysicalDataflow/StorageDomain.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"

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

struct SelectedStorageLifetimeBinding {
  StorageObjectId object;
  uint64_t occurrence = 0;
  mlir::Operation *definition = nullptr;
  std::vector<mlir::Operation *> uses;
  std::vector<mlir::Operation *> completions;
  mlir::Operation *release = nullptr;
};

struct RotatingStorageSelection {
  StorageObjectId object;
  uint32_t recurrenceAxis = 0;
  mlir::Value coordinate;
  mlir::Value value;
  std::vector<mlir::Operation *> selectorOperations;
};

mlir::FailureOr<PreparedStoragePlan>
prepareStoragePlan(const StorageDomain &domain, const BufferPlan &plan,
                   std::string *failureReason = nullptr);

class StorageObjectBuilder {
public:
  mlir::Value lookup(const StorageObjectId &object,
                     uint64_t occurrence = 0) const;
  mlir::FailureOr<mlir::Value>
  lookup(const StorageObjectId &object,
         llvm::ArrayRef<uint64_t> coordinates) const;
  mlir::Value lookup(const PhysicalVersionId &version,
                     uint64_t occurrence = 0) const;
  mlir::FailureOr<mlir::Value>
  lookup(const PhysicalVersionId &version,
         llvm::ArrayRef<uint64_t> coordinates) const;
  llvm::ArrayRef<mlir::Value> getSlots(const StorageObjectId &object) const;

  /// Builds a typed SSA selector for one dynamic occurrence coordinate. The
  /// selected family and axis must already be present in BufferPlan.
  mlir::FailureOr<RotatingStorageSelection>
  select(const StorageObjectId &object, uint32_t recurrenceAxis,
         mlir::Value coordinate, mlir::OpBuilder &builder,
         std::string *failureReason = nullptr) const;

private:
  std::map<StorageObjectId, std::vector<mlir::Value>> objects;
  std::map<PhysicalVersionId, StorageObjectId> bindings;
  std::map<StorageObjectId, SlotFamilyPlan> families;

  friend mlir::LogicalResult
  emitPreparedStorageObjects(const PreparedStoragePlan &,
                             StorageObjectBuilder &, mlir::OpBuilder &,
                             std::string *);
  friend mlir::LogicalResult
  verifyEmittedStorageObjects(const PreparedStoragePlan &,
                              const StorageObjectBuilder &, std::string *);
  friend mlir::LogicalResult
  verifyRotatingStorageSelections(const StorageObjectBuilder &,
                                  llvm::ArrayRef<RotatingStorageSelection>,
                                  std::string *);
};

/// Creates exact selected object multiplicities once. Slot selection is a
/// pure modulo lookup from an explicit occurrence supplied by the temporal or
/// execution-structure emitter; this builder never discovers loops or changes
/// multiplicity.
mlir::LogicalResult emitPreparedStorageObjects(
    const PreparedStoragePlan &prepared, StorageObjectBuilder &objects,
    mlir::OpBuilder &builder, std::string *failureReason = nullptr);

/// Verifies all-and-only object, slot and version bindings after selected
/// construction. Release/completion ordering is checked later against J's
/// actual events; this verifier never inserts a wait or deallocation.
mlir::LogicalResult
verifyEmittedStorageObjects(const PreparedStoragePlan &prepared,
                            const StorageObjectBuilder &objects,
                            std::string *failureReason = nullptr);

/// Normalizes the selected E/K steady-loop induction variable to the global
/// recurrence coordinate, including the already materialized prefix wave.
mlir::FailureOr<mlir::Value> buildSteadyOccurrenceCoordinate(
    const PipelinedExecutionStructure &pipeline, mlir::scf::ForOp steadyLoop,
    mlir::OpBuilder &builder, std::string *failureReason = nullptr);

mlir::LogicalResult verifyRotatingStorageSelections(
    const StorageObjectBuilder &objects,
    llvm::ArrayRef<RotatingStorageSelection> selections,
    std::string *failureReason = nullptr);

/// Verifies actual definition/use/completion/release order for explicitly
/// bound selected slots. Every operation must belong to the current IR and
/// directly reference the selected buffer or its async token.
mlir::LogicalResult verifySelectedStorageLifetimes(
    const StorageObjectBuilder &objects,
    llvm::ArrayRef<SelectedStorageLifetimeBinding> lifetimes,
    std::string *failureReason = nullptr);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_STORAGEOBJECTBUILDER_H
