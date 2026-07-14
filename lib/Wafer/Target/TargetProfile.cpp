//===- TargetProfile.cpp - Closed Wafer target profile registry -----------===//

#include "Wafer/Target/TargetProfile.h"

#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"

namespace wafer {
namespace {

constexpr TargetProfileRecord kTargetProfiles[] = {{
    TargetProfileId::waferTx81SingleCardKernelV1(),
    "wafer-tx81-single-card-kernel-v1",
    TargetIdentityId::waferTx81SingleCard(),
    "wafer-tx81-single-card",
    KernelRuntimeABIId::waferTx81KernelV1(),
    "wafer-tx81-kernel-v1",
    "elf-riscv64",
}};

template <typename Id, typename GetId, typename GetSpelling>
llvm::Expected<Id> parseClosedId(llvm::StringRef spelling, llvm::StringRef kind,
                                 GetId getId, GetSpelling getSpelling) {
  for (const TargetProfileRecord &record : kTargetProfiles)
    if (getSpelling(record) == spelling)
      return getId(record);
  return llvm::createStringError(llvm::errc::invalid_argument,
                                 "unknown %s '%s'", kind.str().c_str(),
                                 spelling.str().c_str());
}

} // namespace

llvm::ArrayRef<TargetProfileRecord> getRegisteredTargetProfiles() {
  return kTargetProfiles;
}

llvm::Expected<TargetProfileId>
parseTargetProfileId(llvm::StringRef canonicalSpelling) {
  return parseClosedId<TargetProfileId>(
      canonicalSpelling, "target profile",
      [](const TargetProfileRecord &record) { return record.id; },
      [](const TargetProfileRecord &record) -> llvm::StringRef {
        return record.canonicalSpelling;
      });
}

llvm::Expected<TargetIdentityId>
parseTargetIdentityId(llvm::StringRef canonicalSpelling) {
  return parseClosedId<TargetIdentityId>(
      canonicalSpelling, "target identity",
      [](const TargetProfileRecord &record) { return record.targetIdentity; },
      [](const TargetProfileRecord &record) -> llvm::StringRef {
        return record.targetIdentitySpelling;
      });
}

llvm::Expected<KernelRuntimeABIId>
parseKernelRuntimeABIId(llvm::StringRef canonicalSpelling) {
  return parseClosedId<KernelRuntimeABIId>(
      canonicalSpelling, "kernel runtime ABI",
      [](const TargetProfileRecord &record) { return record.kernelRuntimeABI; },
      [](const TargetProfileRecord &record) -> llvm::StringRef {
        return record.kernelRuntimeABISpelling;
      });
}

const TargetProfileRecord &getTargetProfileRecord(TargetProfileId id) {
  for (const TargetProfileRecord &record : kTargetProfiles)
    if (record.id == id)
      return record;
  llvm_unreachable("closed TargetProfileId is not registered");
}

llvm::StringRef stringifyTargetProfileId(TargetProfileId id) {
  return getTargetProfileRecord(id).canonicalSpelling;
}

llvm::StringRef stringifyTargetIdentityId(TargetIdentityId id) {
  for (const TargetProfileRecord &record : kTargetProfiles)
    if (record.targetIdentity == id)
      return record.targetIdentitySpelling;
  llvm_unreachable("closed TargetIdentityId is not registered");
}

llvm::StringRef stringifyKernelRuntimeABIId(KernelRuntimeABIId id) {
  for (const TargetProfileRecord &record : kTargetProfiles)
    if (record.kernelRuntimeABI == id)
      return record.kernelRuntimeABISpelling;
  llvm_unreachable("closed KernelRuntimeABIId is not registered");
}

} // namespace wafer
