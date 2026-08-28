//===- TargetLLVMTranslation.cpp - Target conversion and translation ----===//

#include "Wafer/CodeGen/Target/TargetCodeGenInternal.h"
#include "Wafer/Driver/CompilationInternal.h"

#include "Wafer/Support/CompileWorkStatistics.h"

#include "Wafer/Support/CompileTiming.h"
#include "Wafer/Target/Core/TargetMemory.h"
#include "Wafer/Transforms/Passes.h"
#include "Wafer/Conversion/InstrToLLVM/InstrToLLVM.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <utility>

namespace wafer::compiler::detail {
namespace {

constexpr llvm::StringLiteral kTargetLLVMTriple = "riscv64-unknown-unknown-elf";
constexpr llvm::StringLiteral kTargetLLVMCardIdMetadata =
    "wafer.target.card_id";
constexpr llvm::StringLiteral kTargetLLVMTileIdMetadata =
    "wafer.target.tile_id";
constexpr llvm::StringLiteral kTargetLLVMLaunchSlotMetadata =
    "wafer.target.launch_slot";
constexpr llvm::StringLiteral kTargetLLVMEntryMetadata = "wafer.target.entry";
constexpr llvm::StringLiteral kTargetLLVMIdentityMetadata =
    "wafer.target.identity";
constexpr llvm::StringLiteral kTargetLLVMABIMetadata = "wafer.target.abi";
constexpr llvm::StringLiteral kTargetLLVMFormatMetadata =
    "wafer.target.module_format";
constexpr llvm::StringLiteral kTargetLLVMSlotsMetadata =
    "wafer.target.entry_arguments";

llvm::StringRef stringifyTileEntryArgumentKind(TileEntryArgumentKind kind) {
  switch (kind) {
  case TileEntryArgumentKind::ExternalInput:
    return "external-input";
  case TileEntryArgumentKind::TargetTensor:
    return "target-tensor";
  case TileEntryArgumentKind::ExternalOutput:
    return "external-output";
  case TileEntryArgumentKind::Workspace:
    return "workspace";
  case TileEntryArgumentKind::SharedWorkspace:
    return "card-workspace";
  case TileEntryArgumentKind::ProfileRecord:
    return "profile-record";
  case TileEntryArgumentKind::TransportStatus:
    return "transport-status";
  }
  llvm_unreachable("unknown tile entry argument kind");
}

llvm::StringRef
stringifyTileEntryArgumentAccess(TileEntryArgumentAccess access) {
  switch (access) {
  case TileEntryArgumentAccess::None:
    return "none";
  case TileEntryArgumentAccess::ReadOnly:
    return "read-only";
  case TileEntryArgumentAccess::WriteOnly:
    return "write-only";
  case TileEntryArgumentAccess::ReadWrite:
    return "read-write";
  }
  llvm_unreachable("unknown tile entry argument access");
}

llvm::Metadata *signedMetadata(llvm::LLVMContext &context, int64_t value) {
  return llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
      llvm::Type::getInt64Ty(context), static_cast<uint64_t>(value),
      /*IsSigned=*/true));
}

void addStringMetadata(llvm::Module &module, llvm::StringRef name,
                       llvm::StringRef value) {
  llvm::LLVMContext &context = module.getContext();
  module.getOrInsertNamedMetadata(name)->addOperand(
      llvm::MDNode::get(context, llvm::MDString::get(context, value)));
}

void addSignedMetadata(llvm::Module &module, llvm::StringRef name,
                       int64_t value) {
  llvm::LLVMContext &context = module.getContext();
  module.getOrInsertNamedMetadata(name)->addOperand(
      llvm::MDNode::get(context, signedMetadata(context, value)));
}

void attachTargetLLVMMetadata(llvm::Module &module,
                              const PreparedTile &prepared,
                              llvm::StringRef entrySymbol) {
  addSignedMetadata(module, kTargetLLVMCardIdMetadata,
                    prepared.cardId.getValue());
  addSignedMetadata(module, kTargetLLVMTileIdMetadata,
                    prepared.tileId.getValue());
  addSignedMetadata(module, kTargetLLVMLaunchSlotMetadata,
                    prepared.launchSlotId.getValue());
  addStringMetadata(module, kTargetLLVMEntryMetadata, entrySymbol);
  addStringMetadata(module, kTargetLLVMIdentityMetadata,
                    stringifyTargetIdentityId(prepared.targetIdentity));
  addStringMetadata(module, kTargetLLVMABIMetadata,
                    stringifyKernelRuntimeABIId(prepared.kernelRuntimeABI));
  addStringMetadata(module, kTargetLLVMFormatMetadata, prepared.moduleFormat);

  llvm::LLVMContext &context = module.getContext();
  llvm::NamedMDNode *slots =
      module.getOrInsertNamedMetadata(kTargetLLVMSlotsMetadata);
  for (const TileEntryArgument &slot : prepared.slots) {
    llvm::SmallVector<llvm::Metadata *, 13> fields = {
        signedMetadata(context, slot.ordinal),
        llvm::MDString::get(context, stringifyTileEntryArgumentKind(slot.kind)),
        signedMetadata(context, slot.resourceIndex),
        llvm::MDString::get(context, slot.name),
        llvm::MDString::get(context, stringifyLogicalFormat(slot.dtype)),
        llvm::MDString::get(context, stringifyMemLayout(slot.layout)),
        signedMetadata(context, slot.byteSize),
        signedMetadata(context, slot.alignment),
        llvm::MDString::get(context,
                            stringifyTileEntryArgumentAccess(slot.access)),
    };
    for (int64_t dimension : slot.shape)
      fields.push_back(signedMetadata(context, dimension));
    slots->addOperand(llvm::MDNode::get(context, fields));
  }
}

llvm::Expected<llvm::StringRef>
readSingleStringMetadata(const llvm::Module &module, llvm::StringRef name) {
  const llvm::NamedMDNode *named = module.getNamedMetadata(name);
  if (!named || named->getNumOperands() != 1)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "target LLVM metadata '%s' must contain exactly one row",
        name.str().c_str());
  const llvm::MDNode *row = named->getOperand(0);
  if (row->getNumOperands() != 1)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "target LLVM metadata '%s' row must contain one string",
        name.str().c_str());
  auto value = llvm::dyn_cast<llvm::MDString>(row->getOperand(0));
  if (!value)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "target LLVM metadata '%s' row is not a string", name.str().c_str());
  return value->getString();
}

llvm::Expected<int64_t> readSignedMetadataOperand(const llvm::MDNode &row,
                                                  unsigned index,
                                                  llvm::StringRef label) {
  if (index >= row.getNumOperands())
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "target LLVM %s is missing",
                                   label.str().c_str());
  auto constant =
      llvm::dyn_cast<llvm::ConstantAsMetadata>(row.getOperand(index));
  auto integer = constant
                     ? llvm::dyn_cast<llvm::ConstantInt>(constant->getValue())
                     : nullptr;
  if (!integer || integer->getBitWidth() != 64)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "target LLVM %s is not an i64",
                                   label.str().c_str());
  return integer->getSExtValue();
}

llvm::Expected<int64_t> readSingleSignedMetadata(const llvm::Module &module,
                                                 llvm::StringRef name) {
  const llvm::NamedMDNode *named = module.getNamedMetadata(name);
  if (!named || named->getNumOperands() != 1)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "target LLVM metadata '%s' must contain exactly one row",
        name.str().c_str());
  const llvm::MDNode *row = named->getOperand(0);
  if (row->getNumOperands() != 1)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "target LLVM metadata '%s' row must contain one integer",
        name.str().c_str());
  return readSignedMetadataOperand(*row, 0, name);
}

llvm::Expected<llvm::StringRef>
readStringMetadataOperand(const llvm::MDNode &row, unsigned index,
                          llvm::StringRef label) {
  if (index >= row.getNumOperands())
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "target LLVM %s is missing",
                                   label.str().c_str());
  auto value = llvm::dyn_cast<llvm::MDString>(row.getOperand(index));
  if (!value)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "target LLVM %s is not a string",
                                   label.str().c_str());
  return value->getString();
}

llvm::Error
verifyTargetLLVMSlotMetadata(const llvm::Module &module,
                             llvm::ArrayRef<TileEntryArgument> expectedSlots) {
  const llvm::NamedMDNode *slots =
      module.getNamedMetadata(kTargetLLVMSlotsMetadata);
  if (!slots || slots->getNumOperands() != expectedSlots.size())
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "target LLVM tile entry argument metadata "
                                   "does not cover the typed ABI");
  for (auto [index, expected] : llvm::enumerate(expectedSlots)) {
    const llvm::MDNode &row = *slots->getOperand(index);
    if (row.getNumOperands() != 9 + expected.shape.size())
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "target LLVM tile entry argument %zu has an invalid field count",
          index);
    llvm::Expected<int64_t> ordinal =
        readSignedMetadataOperand(row, 0, "tile entry argument ordinal");
    if (!ordinal)
      return ordinal.takeError();
    llvm::Expected<llvm::StringRef> kind =
        readStringMetadataOperand(row, 1, "tile entry argument kind");
    if (!kind)
      return kind.takeError();
    llvm::Expected<int64_t> resource =
        readSignedMetadataOperand(row, 2, "tile entry argument resource index");
    if (!resource)
      return resource.takeError();
    llvm::Expected<llvm::StringRef> name =
        readStringMetadataOperand(row, 3, "tile entry argument name");
    if (!name)
      return name.takeError();
    llvm::Expected<llvm::StringRef> dtype =
        readStringMetadataOperand(row, 4, "tile entry argument dtype");
    if (!dtype)
      return dtype.takeError();
    llvm::Expected<llvm::StringRef> layout =
        readStringMetadataOperand(row, 5, "tile entry argument layout");
    if (!layout)
      return layout.takeError();
    llvm::Expected<int64_t> byteSize =
        readSignedMetadataOperand(row, 6, "tile entry argument byte size");
    if (!byteSize)
      return byteSize.takeError();
    llvm::Expected<int64_t> alignment =
        readSignedMetadataOperand(row, 7, "tile entry argument alignment");
    if (!alignment)
      return alignment.takeError();
    llvm::Expected<llvm::StringRef> access =
        readStringMetadataOperand(row, 8, "tile entry argument access");
    if (!access)
      return access.takeError();
    if (*ordinal != expected.ordinal ||
        *kind != stringifyTileEntryArgumentKind(expected.kind) ||
        *resource != expected.resourceIndex || *name != expected.name ||
        *dtype != stringifyLogicalFormat(expected.dtype) ||
        *layout != stringifyMemLayout(expected.layout) ||
        *byteSize != expected.byteSize || *alignment != expected.alignment ||
        *access != stringifyTileEntryArgumentAccess(expected.access))
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "target LLVM tile entry argument %zu does not match the typed ABI",
          index);
    for (auto [dimensionIndex, expectedDimension] :
         llvm::enumerate(expected.shape)) {
      llvm::Expected<int64_t> dimension = readSignedMetadataOperand(
          row, 9 + dimensionIndex, "tile entry argument shape dimension");
      if (!dimension)
        return dimension.takeError();
      if (*dimension != expectedDimension)
        return llvm::createStringError(llvm::errc::invalid_argument,
                                       "target LLVM tile entry argument %zu "
                                       "shape does not match the typed "
                                       "ABI",
                                       index);
    }
  }
  return llvm::Error::success();
}

} // namespace

llvm::Error
verifyTargetLLVMModule(const llvm::Module &module, CardId expectedCardId,
                       TileId expectedTileId, LaunchSlotId expectedLaunchSlotId,
                       llvm::StringRef expectedEntrySymbol,
                       TargetIdentityId expectedTargetIdentity,
                       KernelRuntimeABIId expectedKernelRuntimeABI,
                       llvm::StringRef expectedModuleFormat,
                       llvm::ArrayRef<TileEntryArgument> expectedSlots) {
  std::string verifierOutput;
  llvm::raw_string_ostream verifierDiagnostics(verifierOutput);
  if (llvm::verifyModule(module, &verifierDiagnostics))
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "target LLVM module verification failed: %s",
                                   verifierOutput.c_str());
  std::string expectedIdentifier =
      llvm::formatv("wafer.target.card.{0}.tile.{1:D5}.launch.{2:D5}",
                    expectedCardId.getValue(), expectedTileId.getValue(),
                    expectedLaunchSlotId.getValue())
          .str();
  if (module.getModuleIdentifier() != expectedIdentifier)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "target LLVM module identifier does not match its Tile and "
        "launch slot");
  if (module.getTargetTriple() != kTargetLLVMTriple)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "target LLVM triple is not the closed target triple");

  llvm::Expected<int64_t> cardId =
      readSingleSignedMetadata(module, kTargetLLVMCardIdMetadata);
  if (!cardId)
    return cardId.takeError();
  llvm::Expected<int64_t> tileId =
      readSingleSignedMetadata(module, kTargetLLVMTileIdMetadata);
  if (!tileId)
    return tileId.takeError();
  llvm::Expected<int64_t> launchSlot =
      readSingleSignedMetadata(module, kTargetLLVMLaunchSlotMetadata);
  if (!launchSlot)
    return launchSlot.takeError();
  llvm::Expected<llvm::StringRef> entrySymbol =
      readSingleStringMetadata(module, kTargetLLVMEntryMetadata);
  if (!entrySymbol)
    return entrySymbol.takeError();
  llvm::Expected<llvm::StringRef> identitySpelling =
      readSingleStringMetadata(module, kTargetLLVMIdentityMetadata);
  if (!identitySpelling)
    return identitySpelling.takeError();
  llvm::Expected<llvm::StringRef> abiSpelling =
      readSingleStringMetadata(module, kTargetLLVMABIMetadata);
  if (!abiSpelling)
    return abiSpelling.takeError();
  llvm::Expected<llvm::StringRef> moduleFormat =
      readSingleStringMetadata(module, kTargetLLVMFormatMetadata);
  if (!moduleFormat)
    return moduleFormat.takeError();
  llvm::Expected<TargetIdentityId> identity =
      parseTargetIdentityId(*identitySpelling);
  if (!identity)
    return identity.takeError();
  llvm::Expected<KernelRuntimeABIId> abi =
      parseKernelRuntimeABIId(*abiSpelling);
  if (!abi)
    return abi.takeError();
  if (*cardId != expectedCardId.getValue() ||
      *tileId != expectedTileId.getValue() ||
      *launchSlot != expectedLaunchSlotId.getValue() ||
      *entrySymbol != expectedEntrySymbol ||
      *identity != expectedTargetIdentity || *abi != expectedKernelRuntimeABI ||
      *moduleFormat != expectedModuleFormat)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "target LLVM module metadata does not match the typed Tile "
        "identity");

  const llvm::Function *entry = module.getFunction(expectedEntrySymbol);
  if (!entry || entry->isDeclaration() || !entry->hasExternalLinkage())
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "target LLVM entry is missing, external, or not externally visible");
  llvm::FunctionType *type = entry->getFunctionType();
  if (type->isVarArg() || !type->getReturnType()->isVoidTy() ||
      type->getNumParams() != expectedSlots.size() ||
      !llvm::all_of(type->params(), [](llvm::Type *parameter) {
        return parameter->isIntegerTy(64);
      }))
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "target LLVM entry must be fixed void(i64...) with one parameter per "
        "typed tile entry argument");
  return verifyTargetLLVMSlotMetadata(module, expectedSlots);
}

mlir::LogicalResult lowerToTargetLLVM(PreparedTile &prepared) {
  wafer::support::recordCompileWork(
      wafer::support::CompileWorkKind::TargetLowering);
  TargetConversionRequest request{};
  request.defaultDDRArenaArgumentIndex = prepared.defaultDDRArenaArgumentIndex;
  request.cardId = prepared.cardId.getValue();
  request.tileId = prepared.tileId.getValue();
  request.transportStatusArgumentIndex = prepared.transportStatusArgumentIndex;
  request.transportPreparedBeforeEntry = prepared.transportPreparedBeforeEntry;
  request.profileRecordArgumentIndex = prepared.profileRecordArgumentIndex;
  return runPassPipeline(
      *prepared.module, "instr-to-target-llvm",
      [&](mlir::OpPassManager &manager) {
        manager.addPass(wafer::createLowerInstrToTargetLLVMPass(request));
      });
}

mlir::LogicalResult verifyLoweredKernelABI(PreparedTile &prepared,
                                           llvm::StringRef entrySymbol) {
  auto entry =
      prepared.module->lookupSymbol<mlir::LLVM::LLVMFuncOp>(entrySymbol);
  if (!entry || entry.isDeclaration())
    return prepared.module->emitError()
           << "target_abi_mismatch: lowered entry is missing or external";
  mlir::LLVM::LLVMFunctionType type = entry.getFunctionType();
  if (type.isVarArg() ||
      !mlir::isa<mlir::LLVM::LLVMVoidType>(type.getReturnType()) ||
      type.getNumParams() != prepared.slots.size() ||
      !llvm::all_of(type.getParams(), [](mlir::Type parameter) {
        return parameter.isInteger(64);
      }))
    return entry.emitError()
           << "target_abi_mismatch: lowered entry must be fixed void(i64...) "
              "with one parameter per typed tile entry argument";
  return mlir::success();
}

llvm::Expected<TargetLLVMModule>
translatePreparedTile(PreparedTile prepared, llvm::StringRef entrySymbol) {
  wafer::support::recordCompileWork(
      wafer::support::CompileWorkKind::TargetTranslation);
  auto llvmContext = std::make_unique<llvm::LLVMContext>();
  std::unique_ptr<llvm::Module> llvmModule = mlir::translateModuleToLLVMIR(
      *prepared.module, *llvmContext, "wafer_target_physical_tile");
  if (!llvmModule)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "target LLVM IR translation failed");
  llvmModule->setModuleIdentifier(
      llvm::formatv("wafer.target.card.{0}.tile.{1:D5}.launch.{2:D5}",
                    prepared.cardId.getValue(), prepared.tileId.getValue(),
                    prepared.launchSlotId.getValue())
          .str());
  llvmModule->setTargetTriple(kTargetLLVMTriple);
  if (llvm::Error error = instrumentProfileTargetModule(
          *llvmModule, entrySymbol, prepared.profileCapture))
    return std::move(error);
  attachTargetLLVMMetadata(*llvmModule, prepared, entrySymbol);
  if (llvm::Error error = verifyTargetLLVMModule(
          *llvmModule, prepared.cardId, prepared.tileId, prepared.launchSlotId,
          entrySymbol, prepared.targetIdentity, prepared.kernelRuntimeABI,
          prepared.moduleFormat, prepared.slots))
    return std::move(error);
  return TargetLLVMModulesBuilder::makeModule(
      prepared.cardId, prepared.tileId, prepared.launchSlotId, entrySymbol,
      prepared.targetIdentity, prepared.kernelRuntimeABI, prepared.moduleFormat,
      std::move(prepared.slots), std::move(llvmContext), std::move(llvmModule));
}

} // namespace wafer::compiler::detail
