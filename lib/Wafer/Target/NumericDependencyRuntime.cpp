//===- NumericDependencyRuntime.cpp - Loaded numeric object conformance ===//

#include "Wafer/Target/NumericDependencyConformance.h"

#include "NumericDependencyConformanceInternal.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SHA256.h"

#include <limits>
#include <string>
#include <utility>

#if defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

namespace wafer {
using namespace numeric_dependency_conformance_internal;
namespace {

llvm::Expected<NumericLoadedObjectIdentity>
verifyLoadedObject(const NumericDependencyConformanceRecord &record,
                   NumericLoadedObjectIdentity identity,
                   llvm::StringRef realArtifactName,
                   llvm::StringRef loaderArtifactName) {
  const NumericDependencyArtifactIdentity *real =
      record.findArtifact(realArtifactName);
  const NumericDependencyArtifactIdentity *loader =
      record.findArtifact(loaderArtifactName);
  if (!real || !loader)
    return invalid(ErrorCode::ClosureMismatch,
                   "verified record lost loaded-object artifacts");
  if (!isLowerSHA256(identity.sha256))
    return invalid(ErrorCode::LoadedObjectMismatch,
                   "provider returned an invalid loaded-object SHA256");

  llvm::Expected<ManagedRoot> root = verifyManagedRoot(record.getManagedRoot());
  if (!root)
    return root.takeError();
  llvm::Expected<ManagedPath> loaded = resolveManagedPath(
      *root, identity.resolvedPath, RequiredFileType::RegularFile,
      "loaded " + stringifyNumericLoadedObjectKind(identity.kind),
      /*requireRelative=*/false);
  if (!loaded)
    return loaded.takeError();
  llvm::Expected<FileReadback> readback = readRegularFile(
      loaded->resolved, std::numeric_limits<uint64_t>::max(),
      "loaded " + stringifyNumericLoadedObjectKind(identity.kind));
  if (!readback)
    return readback.takeError();
  if (identity.sha256 != readback->digest)
    return invalid(ErrorCode::LoadedObjectMismatch,
                   "provider loaded-object digest does not match its file");
  if (loaded->resolved != real->resolvedPath &&
      loaded->resolved != loader->resolvedPath)
    return invalid(ErrorCode::LoadedObjectMismatch,
                   "loaded object is not the recorded real/loader artifact");
  if (readback->digest != real->sha256 || readback->digest != loader->sha256)
    return invalid(ErrorCode::LoadedObjectMismatch,
                   "loaded object content does not match managed record");
  identity.resolvedPath = loaded->resolved;
  return identity;
}

} // namespace

llvm::StringRef stringifyNumericLoadedObjectKind(NumericLoadedObjectKind kind) {
  switch (kind) {
  case NumericLoadedObjectKind::MPFR:
    return "mpfr";
  case NumericLoadedObjectKind::GMP:
    return "gmp";
  }
  llvm_unreachable("unknown loaded numeric object kind");
}

llvm::Expected<NumericLoadedObjectIdentity>
DladdrNumericLoadedObjectIdentityProvider::identify(
    NumericLoadedObjectKind kind) const {
  const void *symbol = kind == NumericLoadedObjectKind::MPFR ? mpfrSymbolAddress
                                                             : gmpSymbolAddress;
  if (!symbol)
    return invalid(ErrorCode::LoadedObjectUnavailable,
                   "representative " + stringifyNumericLoadedObjectKind(kind) +
                       " symbol address is null");
#if defined(__unix__) || defined(__APPLE__)
  Dl_info information{};
  if (dladdr(symbol, &information) == 0 || !information.dli_fname ||
      !*information.dli_fname)
    return invalid(ErrorCode::LoadedObjectUnavailable,
                   "dladdr could not resolve loaded " +
                       stringifyNumericLoadedObjectKind(kind));
  llvm::sys::fs::file_status status;
  if (std::error_code error = llvm::sys::fs::status(information.dli_fname,
                                                    status, /*follow=*/false))
    return invalid(ErrorCode::LoadedObjectUnavailable,
                   "dladdr path cannot be statted: " + error.message());
  if (status.type() == llvm::sys::fs::file_type::symlink_file)
    return invalid(ErrorCode::Symlink, "dladdr path is a symlink");
  if (status.type() != llvm::sys::fs::file_type::regular_file)
    return invalid(ErrorCode::FileType, "dladdr path is not a regular file");
  llvm::SmallString<256> resolved;
  if (std::error_code error =
          llvm::sys::fs::real_path(information.dli_fname, resolved))
    return invalid(ErrorCode::LoadedObjectUnavailable,
                   "dladdr path cannot be resolved: " + error.message());
  llvm::Expected<FileReadback> readback = readRegularFile(
      resolved, std::numeric_limits<uint64_t>::max(), "dladdr loaded object");
  if (!readback)
    return readback.takeError();
  return NumericLoadedObjectIdentity{kind, resolved.str().str(),
                                     std::move(readback->digest)};
#else
  return invalid(ErrorCode::LoadedObjectUnavailable,
                 "dladdr loaded-object readback is unavailable on this host");
#endif
}

llvm::Expected<NumericDependencyExecutionIdentity>
verifyNumericDependencyExecutionIdentity(
    const NumericDependencyConformanceRecord &record,
    const NumericLoadedObjectIdentityProvider &provider) {
  llvm::Expected<NumericLoadedObjectIdentity> mpfr =
      provider.identify(NumericLoadedObjectKind::MPFR);
  if (!mpfr)
    return mpfr.takeError();
  if (mpfr->kind != NumericLoadedObjectKind::MPFR)
    return invalid(ErrorCode::LoadedObjectMismatch,
                   "provider returned the wrong MPFR object kind");
  llvm::Expected<NumericLoadedObjectIdentity> gmp =
      provider.identify(NumericLoadedObjectKind::GMP);
  if (!gmp)
    return gmp.takeError();
  if (gmp->kind != NumericLoadedObjectKind::GMP)
    return invalid(ErrorCode::LoadedObjectMismatch,
                   "provider returned the wrong GMP object kind");

  llvm::Expected<NumericLoadedObjectIdentity> verifiedMPFR =
      verifyLoadedObject(record, std::move(*mpfr), "mpfr", "mpfr-soname");
  if (!verifiedMPFR)
    return verifiedMPFR.takeError();
  llvm::Expected<NumericLoadedObjectIdentity> verifiedGMP =
      verifyLoadedObject(record, std::move(*gmp), "gmp", "gmp-soname");
  if (!verifiedGMP)
    return verifiedGMP.takeError();

  llvm::SHA256 provenance;
  provenance.update("wafer-numeric-dependency-execution-v1");
  provenance.update(llvm::StringRef("\0", 1));
  provenance.update(record.getRecordSHA256());
  for (const NumericLoadedObjectIdentity *identity :
       {&*verifiedMPFR, &*verifiedGMP}) {
    provenance.update(llvm::StringRef("\0", 1));
    provenance.update(stringifyNumericLoadedObjectKind(identity->kind));
    provenance.update(llvm::StringRef("\0", 1));
    provenance.update(identity->resolvedPath);
    provenance.update(llvm::StringRef("\0", 1));
    provenance.update(identity->sha256);
  }
  return NumericDependencyExecutionIdentity(
      record.getRecordSHA256().str(),
      llvm::toHex(provenance.final(), /*LowerCase=*/true),
      std::move(*verifiedMPFR), std::move(*verifiedGMP));
}

} // namespace wafer
