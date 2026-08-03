//===- RuntimeLaunchContract.cpp - Typed runtime launch contract --------===//

#include "Wafer/Target/RuntimeLaunchContract.h"

#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"

#include <initializer_list>

namespace wafer {
namespace {

template <typename T>
llvm::Expected<T> invalidValue(llvm::StringRef kind,
                               llvm::StringRef canonicalSpelling) {
  return llvm::createStringError(llvm::errc::invalid_argument,
                                 "unknown %s '%s'", kind.str().c_str(),
                                 canonicalSpelling.str().c_str());
}

bool hasPhases(llvm::ArrayRef<RuntimeLaunchPhaseRole> actual,
               std::initializer_list<RuntimeLaunchPhaseRole> expected) {
  return actual == llvm::ArrayRef<RuntimeLaunchPhaseRole>(expected);
}

} // namespace

llvm::Expected<RuntimeLaunchContract> RuntimeLaunchContract::createKernel(
    KernelLaunchForm form, KernelEntryABI entryABI,
    llvm::ArrayRef<RuntimeLaunchPhaseRole> phases) {
  const bool valid = (form == KernelLaunchForm::PerRank &&
                      entryABI == KernelEntryABI::RankLocalPointerBlockV1 &&
                      hasPhases(phases, {RuntimeLaunchPhaseRole::Main})) ||
                     (form == KernelLaunchForm::Grid &&
                      (entryABI == KernelEntryABI::RankMajorPointerTableV1 ||
                       entryABI == KernelEntryABI::RankRowPointerTableV1) &&
                      hasPhases(phases, {RuntimeLaunchPhaseRole::Main})) ||
                     (form == KernelLaunchForm::Cluster &&
                      (entryABI == KernelEntryABI::RankMajorPointerTableV1 ||
                       entryABI == KernelEntryABI::RankRowPointerTableV1) &&
                      hasPhases(phases, {RuntimeLaunchPhaseRole::Prepare,
                                         RuntimeLaunchPhaseRole::Main}));
  if (!valid)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "kernel runtime launch form, entry ABI and ordered phases are "
        "incompatible");
  return RuntimeLaunchContract(KernelRuntimeLaunchContract{
      form, entryABI, std::vector<RuntimeLaunchPhaseRole>(phases)});
}

llvm::Expected<RuntimeLaunchContract> RuntimeLaunchContract::createModel(
    ModelEntryABI entryABI, llvm::ArrayRef<RuntimeLaunchPhaseRole> phases) {
  if (entryABI != ModelEntryABI::Tx81ModelBootParamV1 ||
      !hasPhases(phases, {RuntimeLaunchPhaseRole::Main}))
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "model runtime launch entry ABI and ordered phases are incompatible");
  return RuntimeLaunchContract(ModelRuntimeLaunchContract{
      entryABI, std::vector<RuntimeLaunchPhaseRole>(phases)});
}

RuntimeLaunchKind RuntimeLaunchContract::getKind() const {
  return std::holds_alternative<KernelRuntimeLaunchContract>(contract)
             ? RuntimeLaunchKind::Kernel
             : RuntimeLaunchKind::Model;
}

const KernelRuntimeLaunchContract *RuntimeLaunchContract::getKernel() const {
  return std::get_if<KernelRuntimeLaunchContract>(&contract);
}

const ModelRuntimeLaunchContract *RuntimeLaunchContract::getModel() const {
  return std::get_if<ModelRuntimeLaunchContract>(&contract);
}

llvm::ArrayRef<RuntimeLaunchPhaseRole>
RuntimeLaunchContract::getPhases() const {
  if (const auto *kernel = getKernel())
    return kernel->phases;
  return getModel()->phases;
}

bool operator==(const RuntimeLaunchContract &lhs,
                const RuntimeLaunchContract &rhs) {
  if (lhs.getKind() != rhs.getKind())
    return false;
  if (const auto *lhsKernel = lhs.getKernel()) {
    const auto *rhsKernel = rhs.getKernel();
    return rhsKernel && lhsKernel->form == rhsKernel->form &&
           lhsKernel->entryABI == rhsKernel->entryABI &&
           lhsKernel->phases == rhsKernel->phases;
  }
  const auto *lhsModel = lhs.getModel();
  const auto *rhsModel = rhs.getModel();
  return rhsModel && lhsModel->entryABI == rhsModel->entryABI &&
         lhsModel->phases == rhsModel->phases;
}

llvm::StringRef stringifyRuntimeLaunchKind(RuntimeLaunchKind kind) {
  switch (kind) {
  case RuntimeLaunchKind::Kernel:
    return "kernel";
  case RuntimeLaunchKind::Model:
    return "model";
  }
  llvm_unreachable("unknown runtime launch kind");
}

llvm::Expected<RuntimeLaunchKind>
parseRuntimeLaunchKind(llvm::StringRef canonicalSpelling) {
  if (canonicalSpelling == "kernel")
    return RuntimeLaunchKind::Kernel;
  if (canonicalSpelling == "model")
    return RuntimeLaunchKind::Model;
  return invalidValue<RuntimeLaunchKind>("runtime launch kind",
                                         canonicalSpelling);
}

llvm::StringRef stringifyKernelLaunchForm(KernelLaunchForm form) {
  switch (form) {
  case KernelLaunchForm::PerRank:
    return "per-rank";
  case KernelLaunchForm::Grid:
    return "grid";
  case KernelLaunchForm::Cluster:
    return "cluster";
  }
  llvm_unreachable("unknown kernel launch form");
}

llvm::Expected<KernelLaunchForm>
parseKernelLaunchForm(llvm::StringRef canonicalSpelling) {
  if (canonicalSpelling == "per-rank")
    return KernelLaunchForm::PerRank;
  if (canonicalSpelling == "grid")
    return KernelLaunchForm::Grid;
  if (canonicalSpelling == "cluster")
    return KernelLaunchForm::Cluster;
  return invalidValue<KernelLaunchForm>("kernel launch form",
                                        canonicalSpelling);
}

llvm::StringRef stringifyKernelEntryABI(KernelEntryABI entryABI) {
  switch (entryABI) {
  case KernelEntryABI::RankLocalPointerBlockV1:
    return "rank-local-pointer-block-v1";
  case KernelEntryABI::RankMajorPointerTableV1:
    return "rank-major-pointer-table-v1";
  case KernelEntryABI::RankRowPointerTableV1:
    return "rank-row-pointer-table-v1";
  }
  llvm_unreachable("unknown kernel entry ABI");
}

llvm::Expected<KernelEntryABI>
parseKernelEntryABI(llvm::StringRef canonicalSpelling) {
  if (canonicalSpelling == "rank-local-pointer-block-v1")
    return KernelEntryABI::RankLocalPointerBlockV1;
  if (canonicalSpelling == "rank-major-pointer-table-v1")
    return KernelEntryABI::RankMajorPointerTableV1;
  if (canonicalSpelling == "rank-row-pointer-table-v1")
    return KernelEntryABI::RankRowPointerTableV1;
  return invalidValue<KernelEntryABI>("kernel entry ABI", canonicalSpelling);
}

llvm::StringRef stringifyModelEntryABI(ModelEntryABI entryABI) {
  switch (entryABI) {
  case ModelEntryABI::Tx81ModelBootParamV1:
    return "tx81-model-bootparam-v1";
  }
  llvm_unreachable("unknown model entry ABI");
}

llvm::Expected<ModelEntryABI>
parseModelEntryABI(llvm::StringRef canonicalSpelling) {
  if (canonicalSpelling == "tx81-model-bootparam-v1")
    return ModelEntryABI::Tx81ModelBootParamV1;
  return invalidValue<ModelEntryABI>("model entry ABI", canonicalSpelling);
}

llvm::StringRef stringifyRuntimeLaunchPhaseRole(RuntimeLaunchPhaseRole phase) {
  switch (phase) {
  case RuntimeLaunchPhaseRole::Prepare:
    return "prepare";
  case RuntimeLaunchPhaseRole::Main:
    return "main";
  }
  llvm_unreachable("unknown runtime launch phase role");
}

llvm::Expected<RuntimeLaunchPhaseRole>
parseRuntimeLaunchPhaseRole(llvm::StringRef canonicalSpelling) {
  if (canonicalSpelling == "prepare")
    return RuntimeLaunchPhaseRole::Prepare;
  if (canonicalSpelling == "main")
    return RuntimeLaunchPhaseRole::Main;
  return invalidValue<RuntimeLaunchPhaseRole>("runtime launch phase role",
                                              canonicalSpelling);
}

bool isRuntimeLaunchContractCompatible(const RuntimeLaunchContract &launch,
                                       TargetProfileId targetProfile) {
  const auto *kernel = launch.getKernel();
  if (kernel && kernel->form == KernelLaunchForm::PerRank)
    return true;
  return targetProfile == TargetProfileId::waferTx81SingleCardKernelV1() ||
         targetProfile == TargetProfileId::waferTx81SingleCardKernelV3();
}

} // namespace wafer
