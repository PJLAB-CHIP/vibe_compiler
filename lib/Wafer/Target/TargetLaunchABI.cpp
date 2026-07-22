//===- TargetLaunchABI.cpp - Closed target launch ABI registry -----------===//

#include "Wafer/Target/TargetLaunchABI.h"

#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"

namespace wafer {
namespace {

constexpr TargetLaunchABIRecord kTargetLaunchABIs[] = {
    {TargetLaunchABIId::perRankPointerBlockV1(), "per-rank-pointer-block-v1"},
    {TargetLaunchABIId::tx81KernelGridPointerTableV1(),
     "tx81-kernel-grid-pointer-table-v1"},
    {TargetLaunchABIId::tx81ModelBootParamV1(), "tx81-model-bootparam-v1"},
};

} // namespace

llvm::ArrayRef<TargetLaunchABIRecord> getRegisteredTargetLaunchABIs() {
  return kTargetLaunchABIs;
}

llvm::Expected<TargetLaunchABIId>
parseTargetLaunchABIId(llvm::StringRef canonicalSpelling) {
  for (const TargetLaunchABIRecord &record : kTargetLaunchABIs)
    if (record.canonicalSpelling == canonicalSpelling)
      return record.id;
  return llvm::createStringError(llvm::errc::invalid_argument,
                                 "unknown target launch ABI '%s'",
                                 canonicalSpelling.str().c_str());
}

const TargetLaunchABIRecord &getTargetLaunchABIRecord(TargetLaunchABIId id) {
  for (const TargetLaunchABIRecord &record : kTargetLaunchABIs)
    if (record.id == id)
      return record;
  llvm_unreachable("closed TargetLaunchABIId is not registered");
}

llvm::StringRef stringifyTargetLaunchABIId(TargetLaunchABIId id) {
  return getTargetLaunchABIRecord(id).canonicalSpelling;
}

bool isTargetLaunchABICompatible(TargetLaunchABIId launchABI,
                                 TargetProfileId targetProfile) {
  if (launchABI == TargetLaunchABIId::perRankPointerBlockV1())
    return true;
  return targetProfile == TargetProfileId::waferTx81SingleCardKernelV1();
}

} // namespace wafer
