//===- TargetModuleReadback.cpp - Linked target module verification -----===//

#include "Wafer/CodeGen/LLVM/TargetCodeGenInternal.h"

#include "Wafer/Target/Core/TargetMemory.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"
#include "llvm/TargetParser/Triple.h"

#include <limits>
#include <optional>
#include <string>

namespace wafer::compiler::detail {
namespace {

llvm::Error verifyExportedFunction(const llvm::object::ObjectFile &object,
                                   llvm::StringRef expectedSymbol) {
  bool foundEntry = false;
  bool foundNonFunctionEntry = false;
  bool foundNonExportedFunctionEntry = false;
  for (llvm::object::SymbolRef symbol : object.symbols()) {
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
    if (*name != expectedSymbol ||
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
  return llvm::Error::success();
}

} // namespace

llvm::Expected<TargetModuleReadback>
verifyTargetModule(llvm::StringRef path,
                   llvm::ArrayRef<VerifiedTargetExport> expectedExports,
                   TargetIdentityId expectedTarget,
                   const RuntimeLaunchContract &expectedLaunch) {
  constexpr llvm::StringLiteral kDetectedRiscv64ELF = "elf-riscv64";
  if (expectedTarget != TargetIdentityId::waferTx81SingleCard() ||
      kCurrentTargetModuleFormat != kDetectedRiscv64ELF)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "target identity has no ELF readback verifier");
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
  if (expectedExports.empty())
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "target module export domain is empty");
  bool seenPrepare = false;
  bool seenMain = false;
  llvm::StringSet<> symbols;
  llvm::StringRef mainSymbol;
  for (const VerifiedTargetExport &targetExport : expectedExports) {
    bool *seen = targetExport.getRole() == TargetExportRole::Prepare
                     ? &seenPrepare
                     : &seenMain;
    if (*seen || targetExport.getSymbol().empty() ||
        !symbols.insert(targetExport.getSymbol()).second)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "target module exports have duplicate roles or symbols");
    *seen = true;
    if (targetExport.getRole() == TargetExportRole::Main)
      mainSymbol = targetExport.getSymbol();
    if (llvm::Error error = verifyExportedFunction(*(*object).getBinary(),
                                                   targetExport.getSymbol()))
      return std::move(error);
  }
  const bool hasPrepare = llvm::is_contained(expectedLaunch.getPhases(),
                                             RuntimeLaunchPhaseRole::Prepare);
  if (!seenMain || seenPrepare != hasPrepare ||
      expectedExports.size() != (hasPrepare ? 2u : 1u))
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "target module export roles do not match the runtime launch "
        "contract");

  llvm::SHA256 hasher;
  hasher.update(bytes);
  return TargetModuleReadback{
      "sha256:" + llvm::toHex(hasher.final(), /*LowerCase=*/true),
      kDetectedRiscv64ELF.str()};
}

llvm::Expected<VerifiedTargetModule> verifyLinkedTargetModuleForTesting(
    llvm::StringRef path, llvm::StringRef entrySymbol,
    TargetIdentityId targetIdentity,
    const RuntimeLaunchContract &runtimeLaunchContract) {
  std::vector<VerifiedTargetExport> exports;
  exports.push_back(LinkedTargetModulesBuilder::makeExport(
      TargetExportRole::Main, entrySymbol));
  llvm::Expected<TargetModuleReadback> readback =
      verifyTargetModule(path, exports, targetIdentity, runtimeLaunchContract);
  if (!readback)
    return readback.takeError();
  return LinkedTargetModulesBuilder::makeModule(
      TargetModuleId(0), llvm::sys::path::filename(path),
      readback->contentDigest, targetIdentity,
      KernelRuntimeABIId::waferTx81Kernel(), readback->moduleFormat,
      std::move(exports));
}

} // namespace wafer::compiler::detail
