//===- TargetIdentity.cpp - Current Wafer target identity ----------------===//

#include "Wafer/Target/TargetIdentity.h"

#include "llvm/Support/Errc.h"

namespace wafer {

llvm::Expected<TargetIdentityId>
parseTargetIdentityId(llvm::StringRef canonicalSpelling) {
  if (canonicalSpelling == "wafer-tx81-single-card")
    return TargetIdentityId::waferTx81SingleCard();
  return llvm::createStringError(llvm::errc::invalid_argument,
                                 "unknown target identity '%s'",
                                 canonicalSpelling.str().c_str());
}

llvm::Expected<KernelRuntimeABIId>
parseKernelRuntimeABIId(llvm::StringRef canonicalSpelling) {
  if (canonicalSpelling == "wafer-tx81-kernel")
    return KernelRuntimeABIId::waferTx81Kernel();
  return llvm::createStringError(llvm::errc::invalid_argument,
                                 "unknown kernel runtime ABI '%s'",
                                 canonicalSpelling.str().c_str());
}

llvm::StringRef stringifyTargetIdentityId(TargetIdentityId id) {
  if (id == TargetIdentityId::waferTx81SingleCard())
    return "wafer-tx81-single-card";
  llvm_unreachable("unknown target identity");
}

llvm::StringRef stringifyKernelRuntimeABIId(KernelRuntimeABIId id) {
  if (id == KernelRuntimeABIId::waferTx81Kernel())
    return "wafer-tx81-kernel";
  llvm_unreachable("unknown kernel runtime ABI");
}

} // namespace wafer
