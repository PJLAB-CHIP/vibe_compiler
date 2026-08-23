//===- PhysicalVersionBuilder.cpp - Selected physical versions -------===//

#include "Wafer/Planning/PhysicalDataflow/PhysicalVersionBuilder.h"

#include "Wafer/Analysis/PhysicalDataflow/PhysicalLayoutRelation.h"

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

PhysicalVersionId getSourceVersion(const PhysicalVersionId &version) {
  const PhysicalVersionDerivationStep &step = version.derivation.back();
  if (step.kind == PhysicalVersionDerivationKind::AliasView &&
      step.sourceLogicalValue)
    return PhysicalVersionId{*step.sourceLogicalValue};
  PhysicalVersionId source = version;
  source.derivation.pop_back();
  return source;
}

} // namespace

mlir::FailureOr<PreparedRepresentationPlan>
prepareRepresentationPlan(const RepresentationDomain &domain,
                          const RepresentationPlan &plan,
                          std::string *failureReason) {
  if (!domain.contains(plan)) {
    setFailure(failureReason,
               "selected representation plan is outside the current domain");
    return mlir::failure();
  }
  std::map<PhysicalVersionId, const PhysicalVersionPlan *> versions;
  for (const PhysicalVersionPlan &version : plan.physicalVersions)
    if (!versions.try_emplace(version.id, &version).second) {
      setFailure(failureReason,
                 "selected representation has a duplicate version");
      return mlir::failure();
    }

  PreparedRepresentationPlan prepared;
  for (const PhysicalVersionPlan &version : plan.physicalVersions) {
    if (version.id.derivation.empty()) {
      prepared.primaryVersions.push_back(version);
      continue;
    }
    const PhysicalVersionDerivationStep &step = version.id.derivation.back();
    if (step.kind == PhysicalVersionDerivationKind::LayoutConversion &&
        step.sourceLogicalValue) {
      setFailure(failureReason,
                 "selected layout conversion changes logical identity");
      return mlir::failure();
    }
    if (step.kind == PhysicalVersionDerivationKind::AliasView &&
        !step.sourceLogicalValue) {
      setFailure(failureReason,
                 "selected alias version has no source logical value");
      return mlir::failure();
    }
    if (step.encoding != version.encoding) {
      setFailure(failureReason,
                 "selected conversion ID and version encoding disagree");
      return mlir::failure();
    }
    PhysicalVersionId source = getSourceVersion(version.id);
    auto sourcePlan = versions.find(source);
    if (sourcePlan == versions.end()) {
      setFailure(failureReason,
                 "selected conversion references a missing source version");
      return mlir::failure();
    }
    if (step.sourceEncoding != sourcePlan->second->encoding) {
      setFailure(failureReason,
                 "selected conversion ID and source encoding disagree");
      return mlir::failure();
    }
    if (step.kind == PhysicalVersionDerivationKind::LayoutConversion &&
        sourcePlan->second->encoding == version.encoding) {
      setFailure(failureReason,
                 "selected conversion does not change physical encoding");
      return mlir::failure();
    }
    prepared.derivedVersions.push_back(version);
  }
  llvm::sort(prepared.primaryVersions);
  llvm::sort(prepared.derivedVersions, [](const PhysicalVersionPlan &lhs,
                                          const PhysicalVersionPlan &rhs) {
    if (lhs.id.derivation.size() != rhs.id.derivation.size())
      return lhs.id.derivation.size() < rhs.id.derivation.size();
    return lhs.id < rhs.id;
  });
  prepared.uses = plan.uses;
  return prepared;
}

mlir::LogicalResult PhysicalVersionBuilder::bind(const PhysicalVersionId &id,
                                                 mlir::Value value,
                                                 std::string *failureReason) {
  if (!value)
    return fail(failureReason, "physical version binding has no SSA value");
  if (!values.try_emplace(id, value).second)
    return fail(failureReason, "physical version is bound more than once");
  return mlir::success();
}

mlir::Value PhysicalVersionBuilder::lookup(const PhysicalVersionId &id) const {
  auto found = values.find(id);
  return found == values.end() ? mlir::Value{} : found->second;
}

mlir::LogicalResult
emitPreparedRepresentationVersions(const PreparedRepresentationPlan &prepared,
                                   PhysicalVersionBuilder &versions,
                                   mlir::OpBuilder &builder,
                                   std::string *failureReason) {
  struct Emission {
    PhysicalVersionPlan version;
    PhysicalVersionId source;
    mlir::MemRefType targetType;
    bool alias = false;
  };
  std::vector<Emission> emissions;
  std::map<PhysicalVersionId, mlir::MemRefType> availableTypes;
  for (const PhysicalVersionPlan &version : prepared.primaryVersions) {
    mlir::Value value = versions.lookup(version.id);
    if (!value)
      return fail(failureReason, "primary physical version has no SPM binding");
    auto type = mlir::dyn_cast<mlir::MemRefType>(value.getType());
    MemoryAttr memory = type ? getWaferMemoryAttr(type) : MemoryAttr{};
    if (!type || !memory || memory.getSpace() != MemorySpace::SPM ||
        memory.getLayout() != version.encoding)
      return fail(failureReason,
                  "primary physical version has no matching SPM binding");
    availableTypes.emplace(version.id, type);
  }
  for (const PhysicalVersionPlan &version : prepared.derivedVersions) {
    if (versions.lookup(version.id))
      return fail(failureReason, "derived physical version is already bound");
    PhysicalVersionId source = getSourceVersion(version.id);
    auto sourceType = availableTypes.find(source);
    if (sourceType == availableTypes.end())
      return fail(failureReason,
                  "derived physical version source is not available");
    MemoryAttr sourceMemory = getWaferMemoryAttr(sourceType->second);
    if (!sourceMemory || sourceMemory.getSpace() != MemorySpace::SPM)
      return fail(failureReason,
                  "derived physical version source is not an SPM memref");
    const PhysicalVersionDerivationStep &step = version.id.derivation.back();
    if (step.sourceEncoding != sourceMemory.getLayout() ||
        step.encoding != version.encoding)
      return fail(failureReason,
                  "derived physical version path encoding is inconsistent");
    auto targetType = mlir::MemRefType::get(
        sourceType->second.getShape(), sourceType->second.getElementType(),
        mlir::MemRefLayoutAttrInterface{},
        MemoryAttr::get(sourceType->second.getContext(), MemorySpace::SPM,
                        version.encoding));
    const bool alias = step.kind == PhysicalVersionDerivationKind::AliasView;
    if ((!alias && sourceMemory.getLayout() == version.encoding) ||
        (alias && sourceMemory.getLayout() != version.encoding) ||
        mlir::failed(analysis::PhysicalLayoutRelation::create(targetType)))
      return fail(failureReason,
                  "derived physical version target encoding is invalid");
    emissions.push_back({version, std::move(source), targetType, alias});
    availableTypes.emplace(version.id, targetType);
  }

  for (const Emission &emission : emissions) {
    mlir::Value source = versions.lookup(emission.source);
    if (!source)
      return fail(failureReason,
                  "prepared physical version source disappeared");
    mlir::Value result = source;
    if (!emission.alias)
      result = builder
                   .create<LayoutMaterializeOp>(source.getLoc(),
                                                emission.targetType, source)
                   .getResult();
    if (mlir::failed(versions.bind(emission.version.id, result, failureReason)))
      return mlir::failure();
  }
  return mlir::success();
}

} // namespace wafer::compiler::detail
