//===- TargetModuleReadback.cpp - Linked target module verification -----===//

#include "TargetArtifactInternal.h"

#include "Wafer/Support/TargetPolicy.h"

#include "llvm/ADT/StringExtras.h"
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

llvm::Error verifyModelDynamicExport(const llvm::object::ObjectFile &object,
                                     llvm::StringRef entrySymbol) {
  std::optional<llvm::object::SectionRef> exportSection;
  for (llvm::object::SectionRef section : object.sections()) {
    llvm::Expected<llvm::StringRef> name = section.getName();
    if (!name)
      return name.takeError();
    if (*name != "ExportedDYNSYMTab")
      continue;
    if (exportSection || section.getSize() != 16)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "model target module dynamic export table is not one 16-byte "
          "record");
    exportSection = section;
  }
  if (!exportSection)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "model target module is missing ExportedDYNSYMTab");

  std::optional<uint64_t> entryAddress;
  std::optional<uint64_t> recordAddress;
  std::optional<uint64_t> nameAddress;
  bool namePayloadMatches = false;
  for (llvm::object::SymbolRef symbol : object.symbols()) {
    llvm::Expected<llvm::StringRef> symbolName = symbol.getName();
    llvm::Expected<uint32_t> flags = symbol.getFlags();
    if (!symbolName || !flags) {
      if (!symbolName)
        llvm::consumeError(symbolName.takeError());
      if (!flags)
        llvm::consumeError(flags.takeError());
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "model target module export symbol readback failed");
    }
    if ((*flags & llvm::object::BasicSymbolRef::SF_Undefined) ||
        (*symbolName != entrySymbol &&
         *symbolName != "__wafer_model_export_record" &&
         *symbolName != "__wafer_model_entry_name"))
      continue;
    llvm::Expected<uint64_t> address = symbol.getAddress();
    llvm::Expected<llvm::object::section_iterator> symbolSection =
        symbol.getSection();
    if (!address || !symbolSection) {
      if (!address)
        llvm::consumeError(address.takeError());
      if (!symbolSection)
        llvm::consumeError(symbolSection.takeError());
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "model target module export address readback failed");
    }
    if (*symbolSection == object.section_end())
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "model target module export symbol has no defining section");
    if (*symbolName == entrySymbol) {
      entryAddress = *address;
      continue;
    }

    llvm::Expected<llvm::StringRef> sectionName = (**symbolSection).getName();
    if (!sectionName)
      return sectionName.takeError();
    if (*symbolName == "__wafer_model_export_record") {
      if (*sectionName != "ExportedDYNSYMTab" ||
          *address != exportSection->getAddress())
        return llvm::createStringError(
            llvm::errc::invalid_argument,
            "model target module export record is outside its table");
      recordAddress = *address;
      continue;
    }

    llvm::Expected<llvm::StringRef> contents = (**symbolSection).getContents();
    if (!contents || *address < (**symbolSection).getAddress()) {
      if (!contents)
        llvm::consumeError(contents.takeError());
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "model target module export name payload is unreadable");
    }
    uint64_t offset = *address - (**symbolSection).getAddress();
    std::string expectedName = entrySymbol.str();
    expectedName.push_back('\0');
    if (offset <= contents->size() &&
        expectedName.size() <= contents->size() - offset &&
        contents->substr(offset, expectedName.size()) == expectedName)
      namePayloadMatches = true;
    nameAddress = *address;
  }
  if (!entryAddress || !recordAddress || !nameAddress || !namePayloadMatches)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "model target module dynamic export symbols are incomplete");

  bool foundEntryRelocation = false;
  bool foundNameRelocation = false;
  for (llvm::object::SectionRef section : object.sections()) {
    for (llvm::object::RelocationRef relocation : section.relocations()) {
      uint64_t offset = relocation.getOffset();
      if (offset != *recordAddress && offset != *recordAddress + 8)
        continue;
      llvm::object::ELFRelocationRef elfRelocation(relocation);
      llvm::Expected<int64_t> addend = elfRelocation.getAddend();
      if (!addend || *addend < 0) {
        if (!addend)
          llvm::consumeError(addend.takeError());
        return llvm::createStringError(
            llvm::errc::invalid_argument,
            "model target module export relocation has no valid addend");
      }
      uint64_t targetAddress = static_cast<uint64_t>(*addend);
      llvm::object::symbol_iterator relocationSymbol = relocation.getSymbol();
      if (relocationSymbol != object.symbol_end()) {
        llvm::Expected<uint64_t> symbolAddress = relocationSymbol->getAddress();
        if (!symbolAddress)
          return symbolAddress.takeError();
        if (*symbolAddress >
            std::numeric_limits<uint64_t>::max() - targetAddress)
          return llvm::createStringError(
              llvm::errc::invalid_argument,
              "model target module export relocation overflows uint64");
        targetAddress += *symbolAddress;
      }
      if (offset == *recordAddress && targetAddress == *entryAddress)
        foundEntryRelocation = true;
      if (offset == *recordAddress + 8 && targetAddress == *nameAddress)
        foundNameRelocation = true;
    }
  }
  if (!foundEntryRelocation || !foundNameRelocation)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "model target module export record does not relocate to its entry "
        "and name");
  return llvm::Error::success();
}

} // namespace

llvm::Expected<TargetModuleReadback>
verifyTargetModule(llvm::StringRef path, llvm::StringRef entrySymbol,
                   TargetProfileId expectedProfile,
                   TargetLaunchABIId expectedLaunchABI) {
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

  if (expectedLaunchABI == TargetLaunchABIId::tx81ModelBootParamV1())
    if (llvm::Error error =
            verifyModelDynamicExport(*(*object).getBinary(), entrySymbol))
      return std::move(error);

  llvm::SHA256 hasher;
  hasher.update(bytes);
  return TargetModuleReadback{
      "sha256:" + llvm::toHex(hasher.final(), /*LowerCase=*/true),
      kDetectedRiscv64ELF.str()};
}

llvm::Expected<VerifiedTargetModule> verifyLinkedTargetModuleForTesting(
    llvm::StringRef path, llvm::StringRef entrySymbol,
    TargetProfileId targetProfile, TargetLaunchABIId targetLaunchABI) {
  llvm::Expected<TargetModuleReadback> readback =
      verifyTargetModule(path, entrySymbol, targetProfile, targetLaunchABI);
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
