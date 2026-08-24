//===- StorageObjectBuilder.cpp - Selected storage objects -----------===//

#include "Wafer/Planning/PhysicalDataflow/StorageObjectBuilder.h"

#include "Wafer/Analysis/PhysicalDataflow/PhysicalLayoutRelation.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Async/IR/Async.h"
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
    if (family.multiplicity == 0 || family.id.objects.empty() ||
        family.occurrence.axisOccurrences.empty()) {
      setFailure(failureReason, "selected slot family is malformed");
      return mlir::failure();
    }
    for (const StorageObjectId &object : family.id.objects) {
      auto multiplicity = multiplicities.find(object);
      if (multiplicity == multiplicities.end()) {
        setFailure(failureReason,
                   "selected slot family has an unknown storage object");
        return mlir::failure();
      }
      uint32_t &current = multiplicity->second;
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
  prepared.slotFamilies = plan.slotFamilies;
  llvm::sort(prepared.objects, [](const PreparedStorageObject &lhs,
                                  const PreparedStorageObject &rhs) {
    return lhs.plan < rhs.plan;
  });
  llvm::sort(prepared.bindings);
  llvm::sort(prepared.slotFamilies);
  return prepared;
}

mlir::Value StorageObjectBuilder::lookup(const StorageObjectId &id,
                                         uint64_t occurrence) const {
  auto object = objects.find(id);
  if (object == objects.end() || object->second.empty())
    return {};
  return object->second[occurrence % object->second.size()];
}

mlir::FailureOr<mlir::Value>
StorageObjectBuilder::lookup(const StorageObjectId &id,
                             llvm::ArrayRef<uint64_t> coordinates) const {
  auto object = objects.find(id);
  if (object == objects.end() || object->second.empty())
    return mlir::failure();
  auto family = families.find(id);
  if (family == families.end())
    return coordinates.empty()
               ? mlir::FailureOr<mlir::Value>(object->second.front())
               : mlir::FailureOr<mlir::Value>(mlir::failure());
  const SlotFamilyPlan &plan = family->second;
  if (coordinates.size() != plan.occurrence.axisOccurrences.size() ||
      plan.multiplicity == 0 || object->second.size() != plan.multiplicity)
    return mlir::failure();
  for (auto [coordinate, count] :
       llvm::zip_equal(coordinates, plan.occurrence.axisOccurrences))
    if (count == 0 || coordinate >= count)
      return mlir::failure();
  uint64_t slot = 0;
  for (uint32_t axis : plan.rotationIterators) {
    if (axis >= coordinates.size())
      return mlir::failure();
    slot = (slot * (plan.occurrence.axisOccurrences[axis] % plan.multiplicity) +
            coordinates[axis] % plan.multiplicity) %
           plan.multiplicity;
  }
  return object->second[slot];
}

mlir::Value StorageObjectBuilder::lookup(const PhysicalVersionId &version,
                                         uint64_t occurrence) const {
  auto binding = bindings.find(version);
  return binding == bindings.end() ? mlir::Value{}
                                   : lookup(binding->second, occurrence);
}

mlir::FailureOr<mlir::Value>
StorageObjectBuilder::lookup(const PhysicalVersionId &version,
                             llvm::ArrayRef<uint64_t> coordinates) const {
  auto binding = bindings.find(version);
  return binding == bindings.end()
             ? mlir::FailureOr<mlir::Value>(mlir::failure())
             : lookup(binding->second, coordinates);
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
  std::map<StorageObjectId, SlotFamilyPlan> families;
  for (const SlotFamilyPlan &family : prepared.slotFamilies) {
    if (family.id.objects.empty() || family.multiplicity == 0 ||
        family.occurrence.axisOccurrences.empty())
      return fail(failureReason, "prepared slot family is malformed");
    llvm::SmallVector<uint32_t, 4> activeAxes;
    for (auto [axis, count] :
         llvm::enumerate(family.occurrence.axisOccurrences)) {
      if (count == 0)
        return fail(failureReason,
                    "prepared slot family has an empty occurrence axis");
      if (count > 1)
        activeAxes.push_back(static_cast<uint32_t>(axis));
    }
    llvm::SmallVector<uint32_t, 4> selectedAxes =
        family.rotationIterators;
    llvm::sort(selectedAxes);
    if ((family.multiplicity == 1 && !family.rotationIterators.empty()) ||
        (family.multiplicity > 1 &&
         (selectedAxes != activeAxes ||
          std::adjacent_find(selectedAxes.begin(), selectedAxes.end()) !=
              selectedAxes.end())))
      return fail(failureReason,
                  "prepared slot family has an invalid rotation relation");
    for (const StorageObjectId &object : family.id.objects) {
      auto allocation =
          llvm::find_if(allocations, [&](const Allocation &candidate) {
            return candidate.object == object;
          });
      if (allocation == allocations.end() ||
          allocation->multiplicity != family.multiplicity ||
          !families.try_emplace(object, family).second)
        return fail(failureReason,
                    "prepared slot family has an invalid object binding");
    }
  }
  for (const Allocation &allocation : allocations)
    if (allocation.multiplicity > 1 && !families.count(allocation.object))
      return fail(failureReason,
                  "multi-slot object has no selected slot family");

  for (const Allocation &allocation : allocations) {
    std::vector<mlir::Value> slots;
    for (uint32_t slot = 0; slot < allocation.multiplicity; ++slot)
      slots.push_back(builder.create<mlir::memref::AllocOp>(
          builder.getUnknownLoc(), allocation.type));
    objects.objects.emplace(allocation.object, std::move(slots));
  }
  for (const PhysicalVersionStorageBinding &binding : prepared.bindings)
    objects.bindings.emplace(binding.version, binding.object);
  objects.families = std::move(families);
  return mlir::success();
}

mlir::LogicalResult
verifyEmittedStorageObjects(const PreparedStoragePlan &prepared,
                            const StorageObjectBuilder &objects,
                            std::string *failureReason) {
  if (prepared.objects.size() != objects.objects.size() ||
      prepared.bindings.size() != objects.bindings.size())
    return fail(failureReason,
                "selected storage construction has incomplete inventories");
  for (const PreparedStorageObject &expected : prepared.objects) {
    auto actual = objects.objects.find(expected.plan.id);
    if (actual == objects.objects.end() || expected.multiplicity == 0 ||
        actual->second.size() != expected.multiplicity ||
        expected.resource.residentDomain.getBoxes().size() != 1)
      return fail(failureReason,
                  "selected storage construction has the wrong slot count");
    const analysis::StaticRectangularIndexSet &box =
        expected.resource.residentDomain.getBoxes().front();
    for (mlir::Value slot : actual->second) {
      auto type = mlir::dyn_cast<mlir::MemRefType>(slot.getType());
      MemoryAttr memory = type ? getWaferMemoryAttr(type) : MemoryAttr{};
      if (!type || !llvm::equal(type.getShape(), box.sizes) ||
          type.getElementType() != expected.resource.elementType || !memory ||
          memory.getSpace() != MemorySpace::SPM ||
          memory.getLayout() != expected.resource.encoding)
        return fail(failureReason,
                    "selected storage construction changed a physical type");
    }
  }
  for (const PhysicalVersionStorageBinding &binding : prepared.bindings) {
    auto actual = objects.bindings.find(binding.version);
    if (actual == objects.bindings.end() || actual->second != binding.object ||
        !objects.objects.count(binding.object))
      return fail(failureReason,
                  "selected storage construction changed a version binding");
  }
  for (const SlotFamilyPlan &family : prepared.slotFamilies)
    for (const StorageObjectId &object : family.id.objects) {
      auto actual = objects.objects.find(object);
      auto actualFamily = objects.families.find(object);
      if (actual == objects.objects.end() || family.multiplicity == 0 ||
          actual->second.size() != family.multiplicity ||
          actualFamily == objects.families.end() ||
          !(actualFamily->second == family))
        return fail(failureReason,
                    "selected slot family and emitted objects disagree");
    }
  return mlir::success();
}

mlir::LogicalResult verifySelectedStorageLifetimes(
    const StorageObjectBuilder &objects,
    llvm::ArrayRef<SelectedStorageLifetimeBinding> lifetimes,
    std::string *failureReason) {
  auto touches = [](mlir::Operation *operation, mlir::Value buffer) {
    return operation && (llvm::is_contained(operation->getOperands(), buffer) ||
                         llvm::is_contained(operation->getResults(), buffer));
  };
  std::set<std::pair<StorageObjectId, uint64_t>> seen;
  for (const SelectedStorageLifetimeBinding &lifetime : lifetimes) {
    mlir::Value buffer = objects.lookup(lifetime.object, lifetime.occurrence);
    if (!buffer || !lifetime.definition || lifetime.uses.empty() ||
        !lifetime.release ||
        !seen.emplace(lifetime.object, lifetime.occurrence).second ||
        !touches(lifetime.definition, buffer))
      return fail(failureReason,
                  "selected storage lifetime is missing a typed binding");
    mlir::Block *block = lifetime.definition->getBlock();
    if (!block || lifetime.release->getBlock() != block)
      return fail(failureReason,
                  "selected storage lifetime crosses an unbound block");
    auto allocation = buffer.getDefiningOp<mlir::memref::AllocOp>();
    auto release = mlir::dyn_cast<mlir::memref::DeallocOp>(lifetime.release);
    if (!allocation || allocation->getBlock() != block || !release ||
        release.getMemref() != buffer ||
        (allocation.getOperation() != lifetime.definition &&
         !allocation->isBeforeInBlock(lifetime.definition)) ||
        !lifetime.definition->isBeforeInBlock(lifetime.release))
      return fail(failureReason,
                  "selected storage allocation or release is misordered");
    for (mlir::Operation *use : lifetime.uses)
      if (!use || use->getBlock() != block || !touches(use, buffer) ||
          !lifetime.definition->isBeforeInBlock(use) ||
          !use->isBeforeInBlock(lifetime.release))
        return fail(failureReason,
                    "selected storage use is outside its actual lifetime");
    for (mlir::Operation *completion : lifetime.completions) {
      auto await = mlir::dyn_cast_or_null<mlir::async::AwaitOp>(completion);
      mlir::Operation *issue =
          await ? await.getOperand().getDefiningOp() : nullptr;
      if (!await || !issue || !llvm::is_contained(lifetime.uses, issue) ||
          completion->getBlock() != block ||
          !issue->isBeforeInBlock(completion) ||
          !completion->isBeforeInBlock(lifetime.release))
        return fail(failureReason,
                    "selected async use lacks matching completion before "
                    "release");
    }
  }
  return mlir::success();
}

} // namespace wafer::compiler::detail
