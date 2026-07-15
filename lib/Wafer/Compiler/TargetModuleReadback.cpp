//===- TargetModuleReadback.cpp - Linked target module verification -----===//

#include "TargetArtifactInternal.h"

#include "Wafer/Support/TargetPolicy.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"
#include "llvm/TargetParser/Triple.h"

#include <string>

namespace wafer::compiler::detail {

llvm::Expected<TargetModuleReadback>
verifyTargetModule(llvm::StringRef path, llvm::StringRef entrySymbol,
                   TargetProfileId expectedProfile) {
  const TargetProfileRecord &profile = getTargetProfileRecord(expectedProfile);
  constexpr llvm::StringLiteral kDetectedRiscv64ELF = "elf-riscv64";
  if (profile.moduleFormat != kDetectedRiscv64ELF)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "target profile module format '%s' has no ELF readback verifier",
        profile.moduleFormat.str().c_str());
  if (!isRegularTargetFile(path))
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "target module is not a regular file");
  auto buffer = llvm::MemoryBuffer::getFile(path, /*IsText=*/false,
                                            /*RequiresNullTerminator=*/false);
  if (!buffer)
    return llvm::createStringError(buffer.getError(),
                                   "failed to read target module");
  llvm::StringRef bytes = (*buffer)->getBuffer();
  if (bytes.size() < 4 || !bytes.starts_with("\x7f"
                                             "ELF"))
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "target module is not ELF");

  auto object = llvm::object::ObjectFile::createObjectFile(path);
  if (!object)
    return object.takeError();
  if (!(*object).getBinary()->isELF() ||
      (*object).getBinary()->getBytesInAddress() != 8 ||
      (*object).getBinary()->makeTriple().getArch() != llvm::Triple::riscv64)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "target module is not RISC-V 64-bit ELF");
  bool foundEntry = false;
  bool foundNonFunctionEntry = false;
  bool foundNonExportedFunctionEntry = false;
  for (llvm::object::SymbolRef symbol : (*object).getBinary()->symbols()) {
    llvm::Expected<llvm::StringRef> name = symbol.getName();
    llvm::Expected<uint32_t> flags = symbol.getFlags();
    if (!name || !flags) {
      if (!name)
        llvm::consumeError(name.takeError());
      if (!flags)
        llvm::consumeError(flags.takeError());
      return llvm::createStringError(llvm::errc::invalid_argument,
                                     "target module symbol readback failed");
    }
    if (*name != entrySymbol ||
        (*flags & llvm::object::BasicSymbolRef::SF_Undefined))
      continue;
    llvm::Expected<llvm::object::SymbolRef::Type> type = symbol.getType();
    if (!type) {
      llvm::consumeError(type.takeError());
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "target module entry symbol type readback failed");
    }
    if (*type != llvm::object::SymbolRef::ST_Function) {
      foundNonFunctionEntry = true;
      continue;
    }
    if ((*flags & llvm::object::BasicSymbolRef::SF_Global) &&
        (*flags & llvm::object::BasicSymbolRef::SF_Exported)) {
      foundEntry = true;
      break;
    }
    foundNonExportedFunctionEntry = true;
  }
  if (!foundEntry && foundNonExportedFunctionEntry)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "target entry symbol is a function but is not externally visible");
  if (!foundEntry && foundNonFunctionEntry)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "target entry symbol is defined but is not a function");
  if (!foundEntry)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "target entry symbol is not defined");

  llvm::SHA256 hasher;
  hasher.update(bytes);
  return TargetModuleReadback{
      "sha256:" + llvm::toHex(hasher.final(), /*LowerCase=*/true),
      kDetectedRiscv64ELF.str()};
}

llvm::Expected<VerifiedTargetModule>
verifyLinkedTargetModuleForTesting(llvm::StringRef path,
                                   llvm::StringRef entrySymbol,
                                   TargetProfileId targetProfile) {
  llvm::Expected<TargetModuleReadback> readback =
      verifyTargetModule(path, entrySymbol, targetProfile);
  if (!readback)
    return readback.takeError();
  const TargetProfileRecord &profile = getTargetProfileRecord(targetProfile);
  return TargetArtifactBundleBuilder::makeModule(
      /*logicalRank=*/0, entrySymbol, llvm::sys::path::filename(path),
      readback->contentDigest, targetProfile, profile.targetIdentity,
      profile.kernelRuntimeABI, readback->moduleFormat,
      /*kernelABISlots=*/{});
}

} // namespace wafer::compiler::detail
