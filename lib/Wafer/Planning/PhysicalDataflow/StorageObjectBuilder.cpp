//===- StorageObjectBuilder.cpp - Selected storage objects -----------===//

#include "Wafer/Planning/PhysicalDataflow/StorageObjectBuilder.h"

#include "Wafer/Analysis/PhysicalDataflow/PhysicalLayoutRelation.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <limits>
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

llvm::ArrayRef<mlir::Value>
StorageObjectBuilder::getSlots(const StorageObjectId &id) const {
  auto object = objects.find(id);
  return object == objects.end() ? llvm::ArrayRef<mlir::Value>{}
                                 : llvm::ArrayRef<mlir::Value>(object->second);
}

mlir::FailureOr<RotatingStorageSelection>
StorageObjectBuilder::select(const StorageObjectId &id, uint32_t recurrenceAxis,
                             mlir::Value coordinate, mlir::OpBuilder &builder,
                             std::string *failureReason) const {
  auto object = objects.find(id);
  if (object == objects.end() || object->second.empty() || !coordinate ||
      (!mlir::isa<mlir::IndexType>(coordinate.getType()) &&
       !mlir::isa<mlir::IntegerType>(coordinate.getType()))) {
    setFailure(failureReason,
               "rotating storage selection has no object or coordinate");
    return mlir::failure();
  }
  RotatingStorageSelection selection;
  selection.object = id;
  selection.recurrenceAxis = recurrenceAxis;
  selection.coordinate = coordinate;
  selection.value = object->second.front();
  auto family = families.find(id);
  if (object->second.size() == 1) {
    if (family != families.end() && family->second.multiplicity != 1) {
      setFailure(failureReason,
                 "single storage slot disagrees with its selected family");
      return mlir::failure();
    }
    return selection;
  }
  if (family == families.end() ||
      family->second.multiplicity != object->second.size() ||
      family->second.rotationIterators !=
          llvm::SmallVector<uint32_t, 4>{recurrenceAxis} ||
      recurrenceAxis >= family->second.occurrence.axisOccurrences.size() ||
      family->second.occurrence.axisOccurrences[recurrenceAxis] <
          object->second.size()) {
    setFailure(failureReason,
               "rotating storage selection disagrees with its family");
    return mlir::failure();
  }
  mlir::Location loc = builder.getUnknownLoc();
  auto constant = [&](uint64_t value) -> mlir::Value {
    return builder.create<mlir::arith::ConstantOp>(
        loc, builder.getIntegerAttr(coordinate.getType(), value));
  };
  mlir::Value modulus = constant(object->second.size());
  auto remainder =
      builder.create<mlir::arith::RemUIOp>(loc, coordinate, modulus);
  selection.selectorOperations.push_back(remainder);
  for (uint64_t slot = 1; slot < object->second.size(); ++slot) {
    mlir::Value slotIndex = constant(slot);
    auto match = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, remainder, slotIndex);
    auto selected = builder.create<mlir::arith::SelectOp>(
        loc, match, object->second[slot], selection.value);
    selection.selectorOperations.push_back(match);
    selection.selectorOperations.push_back(selected);
    selection.value = selected;
  }
  return selection;
}

mlir::FailureOr<mlir::Value> buildSteadyOccurrenceCoordinate(
    const PipelinedExecutionStructure &pipeline, mlir::scf::ForOp steadyLoop,
    mlir::OpBuilder &builder, std::string *failureReason) {
  if (!steadyLoop || pipeline.iteration.prefixCount != 1 ||
      pipeline.iteration.recurrenceAxis >=
          pipeline.recurrence.axisOccurrences.size() ||
      pipeline.iteration.steadyTripCount == 0) {
    setFailure(failureReason,
               "steady occurrence coordinate has an invalid structure");
    return mlir::failure();
  }
  std::optional<int64_t> lower =
      mlir::getConstantIntValue(steadyLoop.getLowerBound());
  std::optional<int64_t> upper =
      mlir::getConstantIntValue(steadyLoop.getUpperBound());
  std::optional<int64_t> step = mlir::getConstantIntValue(steadyLoop.getStep());
  const __int128 difference =
      lower && upper ? static_cast<__int128>(*upper) - *lower : 0;
  if (!lower || !upper || !step || *step <= 0 || *upper <= *lower ||
      difference > std::numeric_limits<int64_t>::max()) {
    setFailure(failureReason,
               "steady occurrence coordinate requires static safe bounds");
    return mlir::failure();
  }
  const uint64_t tripCount =
      1 + static_cast<uint64_t>(difference - 1) / static_cast<uint64_t>(*step);
  if (tripCount != pipeline.iteration.steadyTripCount) {
    setFailure(failureReason,
               "steady occurrence coordinate trip count differs from K");
    return mlir::failure();
  }
  mlir::Location loc = steadyLoop.getLoc();
  mlir::Value normalized = steadyLoop.getInductionVar();
  if (*lower != 0) {
    auto lowerValue = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getIntegerAttr(normalized.getType(), *lower));
    normalized =
        builder.create<mlir::arith::SubIOp>(loc, normalized, lowerValue);
  }
  if (*step != 1) {
    auto stepValue = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getIntegerAttr(normalized.getType(), *step));
    normalized =
        builder.create<mlir::arith::DivUIOp>(loc, normalized, stepValue);
  }
  auto prefix = builder.create<mlir::arith::ConstantOp>(
      loc, builder.getIntegerAttr(normalized.getType(),
                                  pipeline.iteration.prefixCount));
  return builder.create<mlir::arith::AddIOp>(loc, normalized, prefix)
      .getResult();
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
    llvm::SmallVector<uint32_t, 4> selectedAxes = family.rotationIterators;
    llvm::sort(selectedAxes);
    if ((family.multiplicity == 1 && !family.rotationIterators.empty()) ||
        (family.multiplicity > 1 &&
         (selectedAxes.empty() ||
          std::adjacent_find(selectedAxes.begin(), selectedAxes.end()) !=
              selectedAxes.end() ||
          llvm::any_of(selectedAxes, [&](uint32_t axis) {
            return !llvm::is_contained(activeAxes, axis);
          }))))
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

mlir::LogicalResult verifyRotatingStorageSelections(
    const StorageObjectBuilder &objects,
    llvm::ArrayRef<RotatingStorageSelection> selections,
    std::string *failureReason) {
  llvm::DenseSet<mlir::Value> selectedValues;
  llvm::DenseSet<mlir::Operation *> selectorOperations;
  for (const RotatingStorageSelection &selection : selections) {
    llvm::ArrayRef<mlir::Value> slots = objects.getSlots(selection.object);
    auto family = objects.families.find(selection.object);
    if (slots.empty() || !selection.coordinate || !selection.value ||
        !selectedValues.insert(selection.value).second)
      return fail(failureReason,
                  "rotating storage selection has a stale or duplicate value");
    if (slots.size() == 1) {
      if (selection.value != slots.front() ||
          !selection.selectorOperations.empty() ||
          (family != objects.families.end() &&
           family->second.multiplicity != 1))
        return fail(failureReason,
                    "single-slot selection contains rotation scaffolding");
      continue;
    }
    if (family == objects.families.end() ||
        family->second.multiplicity != slots.size() ||
        family->second.rotationIterators !=
            llvm::SmallVector<uint32_t, 4>{selection.recurrenceAxis} ||
        selection.recurrenceAxis >=
            family->second.occurrence.axisOccurrences.size() ||
        family->second.occurrence.axisOccurrences[selection.recurrenceAxis] <
            slots.size())
      return fail(failureReason,
                  "rotating storage selection changed its family axis");
    if (selection.selectorOperations.size() != 1 + 2 * (slots.size() - 1))
      return fail(failureReason,
                  "rotating storage selector has the wrong operation count");
    auto remainder = mlir::dyn_cast_or_null<mlir::arith::RemUIOp>(
        selection.selectorOperations.front());
    if (!remainder || remainder.getLhs() != selection.coordinate ||
        mlir::getConstantIntValue(remainder.getRhs()) !=
            std::optional<int64_t>(static_cast<int64_t>(slots.size())))
      return fail(failureReason,
                  "rotating storage selector has the wrong modulo relation");
    mlir::Value previous = slots.front();
    mlir::Operation *previousOperation = remainder;
    for (size_t slot = 1; slot < slots.size(); ++slot) {
      auto match = mlir::dyn_cast_or_null<mlir::arith::CmpIOp>(
          selection.selectorOperations[2 * slot - 1]);
      auto selected = mlir::dyn_cast_or_null<mlir::arith::SelectOp>(
          selection.selectorOperations[2 * slot]);
      if (!match || !selected ||
          match.getPredicate() != mlir::arith::CmpIPredicate::eq ||
          match.getLhs() != remainder ||
          mlir::getConstantIntValue(match.getRhs()) !=
              std::optional<int64_t>(static_cast<int64_t>(slot)) ||
          selected.getCondition() != match ||
          selected.getTrueValue() != slots[slot] ||
          selected.getFalseValue() != previous ||
          match->getBlock() != remainder->getBlock() ||
          selected->getBlock() != remainder->getBlock() ||
          !previousOperation->isBeforeInBlock(match) ||
          !match->isBeforeInBlock(selected))
        return fail(failureReason,
                    "rotating storage selector chain differs from its slots");
      previous = selected;
      previousOperation = selected;
    }
    if (previous != selection.value)
      return fail(failureReason,
                  "rotating storage selector result differs from its chain");
    for (mlir::Operation *operation : selection.selectorOperations)
      if (!operation || !selectorOperations.insert(operation).second)
        return fail(failureReason,
                    "rotating storage selector operation is stale or reused");
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
