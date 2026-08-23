//===- StorageObjectBuilder.cpp - Selected storage objects -----------===//

#include "Wafer/Planning/PhysicalDataflow/StorageObjectBuilder.h"

#include "Wafer/Analysis/PhysicalDataflow/PhysicalLayoutRelation.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "llvm/ADT/STLExtras.h"

#include <set>

namespace wafer::compiler::detail {
namespace {

void setFailure(std::string *failureReason, llvm::StringRef detail) {
  if (failureReason)
    *failureReason = detail.str();
}

mlir::LogicalResult fail(std::string *failureReason, llvm::StringRef detail) {
  setFailure(failureReason, detail);
  return mlir::failure();
}

} // namespace

mlir::FailureOr<PreparedStoragePlan>
prepareStoragePlan(const StorageDomain &domain, const BufferPlan &plan,
                   std::string *failureReason) {
  if (!domain.contains(plan)) {
    setFailure(failureReason, "selected storage plan is outside its domain");
    return mlir::failure();
  }
  std::map<StorageObjectId, uint32_t> multiplicities;
  for (const StorageObjectPlan &object : plan.storageObjects)
    if (!multiplicities.try_emplace(object.id, 1).second) {
      setFailure(failureReason, "selected storage plan has duplicate objects");
      return mlir::failure();
    }
  std::map<PhysicalVersionId, const PhysicalVersionStorageBinding *> bindings;
  for (const PhysicalVersionStorageBinding &binding : plan.versionBindings)
    if (!multiplicities.count(binding.object) ||
        !bindings.try_emplace(binding.version, &binding).second) {
      setFailure(failureReason,
                 "selected storage binding has a missing or duplicate object");
      return mlir::failure();
    }
  for (const SlotFamilyPlan &family : plan.slotFamilies) {
    if (family.multiplicity == 0 || family.id.members.empty()) {
      setFailure(failureReason, "selected slot family is malformed");
      return mlir::failure();
    }
    for (const PhysicalVersionId &member : family.id.members) {
      auto binding = bindings.find(member);
      if (binding == bindings.end()) {
        setFailure(failureReason, "selected slot family has an unbound member");
        return mlir::failure();
      }
      uint32_t &current = multiplicities[binding->second->object];
      if (current != 1 && current != family.multiplicity) {
        setFailure(failureReason,
                   "one storage object has conflicting slot multiplicities");
        return mlir::failure();
      }
      current = family.multiplicity;
    }
  }

  PreparedStoragePlan prepared;
  for (const StorageObjectPlan &object : plan.storageObjects) {
    const StorageResourceDescription *resource = domain.findResource(object.id);
    if (!resource || resource->residentDomain.getBoxes().size() != 1) {
      setFailure(failureReason,
                 "selected storage object is not one exact rectangle");
      return mlir::failure();
    }
    prepared.objects.push_back(
        {object, *resource, multiplicities.at(object.id)});
  }
  prepared.bindings = plan.versionBindings;
  llvm::sort(prepared.objects, [](const PreparedStorageObject &lhs,
                                  const PreparedStorageObject &rhs) {
    return lhs.plan < rhs.plan;
  });
  llvm::sort(prepared.bindings);
  return prepared;
}

mlir::Value StorageObjectBuilder::lookup(const PhysicalVersionId &version,
                                         uint64_t occurrence) const {
  auto binding = bindings.find(version);
  if (binding == bindings.end())
    return {};
  auto object = objects.find(binding->second);
  if (object == objects.end() || object->second.empty())
    return {};
  return object->second[occurrence % object->second.size()];
}

mlir::LogicalResult emitPreparedStorageObjects(
    const PreparedStoragePlan &prepared, StorageObjectBuilder &objects,
    mlir::OpBuilder &builder, std::string *failureReason) {
  struct Allocation {
    StorageObjectId object;
    mlir::MemRefType type;
    uint32_t multiplicity = 1;
  };
  std::vector<Allocation> allocations;
  std::set<StorageObjectId> seen;
  for (const PreparedStorageObject &object : prepared.objects) {
    if (!seen.insert(object.plan.id).second || object.multiplicity == 0 ||
        objects.objects.count(object.plan.id))
      return fail(failureReason,
                  "prepared storage object is duplicate or already emitted");
    const analysis::StaticRectangularIndexSet &box =
        object.resource.residentDomain.getBoxes().front();
    auto type = mlir::MemRefType::get(
        box.sizes, object.resource.elementType,
        mlir::MemRefLayoutAttrInterface{},
        MemoryAttr::get(object.resource.elementType.getContext(),
                        MemorySpace::SPM, object.resource.encoding));
    if (mlir::failed(analysis::PhysicalLayoutRelation::create(type)))
      return fail(failureReason,
                  "prepared storage object has an invalid physical type");
    allocations.push_back({object.plan.id, type, object.multiplicity});
  }
  for (const PhysicalVersionStorageBinding &binding : prepared.bindings) {
    if (!seen.count(binding.object) || objects.bindings.count(binding.version))
      return fail(failureReason,
                  "prepared storage binding is missing or duplicated");
  }

  for (const Allocation &allocation : allocations) {
    std::vector<mlir::Value> slots;
    for (uint32_t slot = 0; slot < allocation.multiplicity; ++slot)
      slots.push_back(builder.create<mlir::memref::AllocOp>(
          builder.getUnknownLoc(), allocation.type));
    objects.objects.emplace(allocation.object, std::move(slots));
  }
  for (const PhysicalVersionStorageBinding &binding : prepared.bindings)
    objects.bindings.emplace(binding.version, binding.object);
  return mlir::success();
}

} // namespace wafer::compiler::detail
