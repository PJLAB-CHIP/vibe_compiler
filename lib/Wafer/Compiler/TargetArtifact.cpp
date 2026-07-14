//===- TargetArtifact.cpp - Typed all-rank target artifacts --------------===//

#include "TargetArtifactInternal.h"

#include "AcceptedCallClosure.h"

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/TargetPolicy.h"
#include "Wafer/Transforms/Passes.h"
#include "Wafer/Transforms/TargetConversion.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <cerrno>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#ifdef __linux__
#include <fcntl.h>
#include <linux/fs.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace wafer::compiler {

struct TargetLLVMModuleBundleBuilder {
  static TargetLLVMModule
  makeModule(int64_t logicalRank, llvm::StringRef entrySymbol,
             TargetProfileId targetProfile, TargetIdentityId targetIdentity,
             KernelRuntimeABIId kernelRuntimeABI, llvm::StringRef moduleFormat,
             std::vector<KernelABISlot> kernelABISlots,
             std::unique_ptr<llvm::LLVMContext> context,
             std::unique_ptr<llvm::Module> module) {
    return TargetLLVMModule(logicalRank, entrySymbol, targetProfile,
                            targetIdentity, kernelRuntimeABI, moduleFormat,
                            std::move(kernelABISlots), std::move(context),
                            std::move(module));
  }

  static TargetLLVMModuleBundle
  makeBundle(ExecutionConfig executionConfig,
             std::vector<TargetLLVMModule> modules) {
    return TargetLLVMModuleBundle(executionConfig, std::move(modules));
  }
};

struct TargetArtifactBundleBuilder {
  static VerifiedTargetModule
  makeModule(int64_t logicalRank, llvm::StringRef entrySymbol,
             llvm::StringRef relativePath, llvm::StringRef contentDigest,
             TargetProfileId targetProfile, TargetIdentityId targetIdentity,
             KernelRuntimeABIId kernelRuntimeABI, llvm::StringRef moduleFormat,
             std::vector<KernelABISlot> kernelABISlots) {
    return VerifiedTargetModule(logicalRank, entrySymbol, relativePath,
                                contentDigest, targetProfile, targetIdentity,
                                kernelRuntimeABI, moduleFormat,
                                std::move(kernelABISlots));
  }

  static TargetArtifactBundle
  makeBundle(llvm::StringRef rootDirectory, ExecutionConfig executionConfig,
             std::vector<VerifiedTargetModule> modules) {
    return TargetArtifactBundle(rootDirectory, executionConfig,
                                std::move(modules));
  }
};

TargetLLVMModule::TargetLLVMModule(int64_t logicalRank,
                                   llvm::StringRef entrySymbol,
                                   TargetProfileId targetProfile,
                                   TargetIdentityId targetIdentity,
                                   KernelRuntimeABIId kernelRuntimeABI,
                                   llvm::StringRef moduleFormat,
                                   std::vector<KernelABISlot> kernelABISlots,
                                   std::unique_ptr<llvm::LLVMContext> context,
                                   std::unique_ptr<llvm::Module> module)
    : logicalRank(logicalRank), entrySymbol(entrySymbol.str()),
      targetProfile(targetProfile), targetIdentity(targetIdentity),
      kernelRuntimeABI(kernelRuntimeABI), moduleFormat(moduleFormat.str()),
      kernelABISlots(std::move(kernelABISlots)), context(std::move(context)),
      module(std::move(module)) {}

TargetLLVMModule::~TargetLLVMModule() = default;
TargetLLVMModule::TargetLLVMModule(TargetLLVMModule &&) = default;
TargetLLVMModule &TargetLLVMModule::operator=(TargetLLVMModule &&) = default;

llvm::StringRef TargetLLVMModule::getModuleIdentifier() const {
  return module->getModuleIdentifier();
}

llvm::StringRef TargetLLVMModule::getTargetTriple() const {
  return module->getTargetTriple();
}

const llvm::Module &TargetLLVMModule::getModule() const { return *module; }

TargetLLVMModuleBundle::~TargetLLVMModuleBundle() = default;
TargetLLVMModuleBundle::TargetLLVMModuleBundle(TargetLLVMModuleBundle &&) =
    default;
TargetLLVMModuleBundle &
TargetLLVMModuleBundle::operator=(TargetLLVMModuleBundle &&) = default;

llvm::Expected<TargetToolchain>
TargetToolchain::create(llvm::StringRef pythonExecutable,
                        llvm::StringRef deviceLinkerScript) {
  if (pythonExecutable.empty())
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "Python executable must not be empty");
  if (deviceLinkerScript.empty())
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "device linker script must not be empty");
  return TargetToolchain(pythonExecutable, deviceLinkerScript);
}

namespace {

struct PreparedTargetRank {
  explicit PreparedTargetRank(const ExecutionConfig &executionConfig)
      : targetProfile(executionConfig.getTargetProfileId()),
        targetIdentity(getTargetProfileRecord(targetProfile).targetIdentity),
        kernelRuntimeABI(
            getTargetProfileRecord(targetProfile).kernelRuntimeABI),
        moduleFormat(getTargetProfileRecord(targetProfile).moduleFormat.str()) {
  }

  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::vector<KernelABISlot> slots;
  TargetProfileId targetProfile;
  TargetIdentityId targetIdentity;
  KernelRuntimeABIId kernelRuntimeABI;
  std::string moduleFormat;
  int64_t logicalRank = -1;
  int64_t defaultDDRArenaArgumentIndex = -1;
  int64_t transportStatusArgumentIndex = -1;
};

constexpr llvm::StringLiteral kTargetLLVMTriple = "riscv64-unknown-unknown-elf";
constexpr llvm::StringLiteral kTargetLLVMSchemaMetadata = "wafer.target.schema";
constexpr llvm::StringLiteral kTargetLLVMRankMetadata = "wafer.target.rank";
constexpr llvm::StringLiteral kTargetLLVMEntryMetadata = "wafer.target.entry";
constexpr llvm::StringLiteral kTargetLLVMProfileMetadata =
    "wafer.target.profile";
constexpr llvm::StringLiteral kTargetLLVMIdentityMetadata =
    "wafer.target.identity";
constexpr llvm::StringLiteral kTargetLLVMABIMetadata = "wafer.target.abi";
constexpr llvm::StringLiteral kTargetLLVMFormatMetadata =
    "wafer.target.module_format";
constexpr llvm::StringLiteral kTargetLLVMSlotsMetadata =
    "wafer.target.abi_slots";
constexpr llvm::StringLiteral kTargetLLVMSchema = "wafer-target-llvm-module-v1";

llvm::StringRef stringifyKernelABISlotRole(KernelABISlotRole role) {
  switch (role) {
  case KernelABISlotRole::UserInput:
    return "user-input";
  case KernelABISlotRole::Parameter:
    return "parameter";
  case KernelABISlotRole::Constant:
    return "constant";
  case KernelABISlotRole::Output:
    return "output";
  case KernelABISlotRole::Workspace:
    return "workspace";
  case KernelABISlotRole::TransportStatus:
    return "transport-status";
  }
  llvm_unreachable("unknown Kernel ABI slot role");
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
                              const PreparedTargetRank &prepared,
                              llvm::StringRef entrySymbol) {
  addStringMetadata(module, kTargetLLVMSchemaMetadata, kTargetLLVMSchema);
  addSignedMetadata(module, kTargetLLVMRankMetadata, prepared.logicalRank);
  addStringMetadata(module, kTargetLLVMEntryMetadata, entrySymbol);
  addStringMetadata(module, kTargetLLVMProfileMetadata,
                    stringifyTargetProfileId(prepared.targetProfile));
  addStringMetadata(module, kTargetLLVMIdentityMetadata,
                    stringifyTargetIdentityId(prepared.targetIdentity));
  addStringMetadata(module, kTargetLLVMABIMetadata,
                    stringifyKernelRuntimeABIId(prepared.kernelRuntimeABI));
  addStringMetadata(module, kTargetLLVMFormatMetadata, prepared.moduleFormat);

  llvm::LLVMContext &context = module.getContext();
  llvm::NamedMDNode *slots =
      module.getOrInsertNamedMetadata(kTargetLLVMSlotsMetadata);
  for (const KernelABISlot &slot : prepared.slots) {
    llvm::SmallVector<llvm::Metadata *, 12> fields = {
        signedMetadata(context, slot.ordinal),
        llvm::MDString::get(context, stringifyKernelABISlotRole(slot.role)),
        signedMetadata(context, slot.resourceIndex),
        llvm::MDString::get(context, slot.name),
        llvm::MDString::get(context, slot.dtype),
        signedMetadata(context, slot.byteSize),
        signedMetadata(context, slot.alignment),
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
                             llvm::ArrayRef<KernelABISlot> expectedSlots) {
  const llvm::NamedMDNode *slots =
      module.getNamedMetadata(kTargetLLVMSlotsMetadata);
  if (!slots || slots->getNumOperands() != expectedSlots.size())
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "target LLVM Kernel ABI slot metadata does not cover the typed ABI");
  for (auto [index, expected] : llvm::enumerate(expectedSlots)) {
    const llvm::MDNode &row = *slots->getOperand(index);
    if (row.getNumOperands() != 7 + expected.shape.size())
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "target LLVM Kernel ABI slot %zu has an invalid field count", index);
    llvm::Expected<int64_t> ordinal =
        readSignedMetadataOperand(row, 0, "Kernel ABI slot ordinal");
    if (!ordinal)
      return ordinal.takeError();
    llvm::Expected<llvm::StringRef> role =
        readStringMetadataOperand(row, 1, "Kernel ABI slot role");
    if (!role)
      return role.takeError();
    llvm::Expected<int64_t> resource =
        readSignedMetadataOperand(row, 2, "Kernel ABI resource index");
    if (!resource)
      return resource.takeError();
    llvm::Expected<llvm::StringRef> name =
        readStringMetadataOperand(row, 3, "Kernel ABI slot name");
    if (!name)
      return name.takeError();
    llvm::Expected<llvm::StringRef> dtype =
        readStringMetadataOperand(row, 4, "Kernel ABI slot dtype");
    if (!dtype)
      return dtype.takeError();
    llvm::Expected<int64_t> byteSize =
        readSignedMetadataOperand(row, 5, "Kernel ABI slot byte size");
    if (!byteSize)
      return byteSize.takeError();
    llvm::Expected<int64_t> alignment =
        readSignedMetadataOperand(row, 6, "Kernel ABI slot alignment");
    if (!alignment)
      return alignment.takeError();
    if (*ordinal != expected.ordinal ||
        *role != stringifyKernelABISlotRole(expected.role) ||
        *resource != expected.resourceIndex || *name != expected.name ||
        *dtype != expected.dtype || *byteSize != expected.byteSize ||
        *alignment != expected.alignment)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "target LLVM Kernel ABI slot %zu does not match the typed ABI",
          index);
    for (auto [dimensionIndex, expectedDimension] :
         llvm::enumerate(expected.shape)) {
      llvm::Expected<int64_t> dimension = readSignedMetadataOperand(
          row, 7 + dimensionIndex, "Kernel ABI slot shape dimension");
      if (!dimension)
        return dimension.takeError();
      if (*dimension != expectedDimension)
        return llvm::createStringError(
            llvm::errc::invalid_argument,
            "target LLVM Kernel ABI slot %zu shape does not match the typed "
            "ABI",
            index);
    }
  }
  return llvm::Error::success();
}

llvm::Error
verifyTargetLLVMModule(const llvm::Module &module, int64_t expectedLogicalRank,
                       llvm::StringRef expectedEntrySymbol,
                       TargetProfileId expectedProfile,
                       TargetIdentityId expectedTargetIdentity,
                       KernelRuntimeABIId expectedKernelRuntimeABI,
                       llvm::StringRef expectedModuleFormat,
                       llvm::ArrayRef<KernelABISlot> expectedSlots) {
  std::string verifierOutput;
  llvm::raw_string_ostream verifierDiagnostics(verifierOutput);
  if (llvm::verifyModule(module, &verifierDiagnostics))
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "target LLVM module verification failed: %s",
                                   verifierOutput.c_str());
  std::string expectedIdentifier =
      llvm::formatv("wafer.target.rank.{0:D5}", expectedLogicalRank).str();
  if (module.getModuleIdentifier() != expectedIdentifier)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "target LLVM module identifier does not match logical rank");
  if (module.getTargetTriple() != kTargetLLVMTriple)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "target LLVM triple is not the closed target triple");

  llvm::Expected<llvm::StringRef> schema =
      readSingleStringMetadata(module, kTargetLLVMSchemaMetadata);
  if (!schema)
    return schema.takeError();
  llvm::Expected<int64_t> logicalRank =
      readSingleSignedMetadata(module, kTargetLLVMRankMetadata);
  if (!logicalRank)
    return logicalRank.takeError();
  llvm::Expected<llvm::StringRef> entrySymbol =
      readSingleStringMetadata(module, kTargetLLVMEntryMetadata);
  if (!entrySymbol)
    return entrySymbol.takeError();
  llvm::Expected<llvm::StringRef> profileSpelling =
      readSingleStringMetadata(module, kTargetLLVMProfileMetadata);
  if (!profileSpelling)
    return profileSpelling.takeError();
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
  llvm::Expected<TargetProfileId> profile =
      parseTargetProfileId(*profileSpelling);
  if (!profile)
    return profile.takeError();
  llvm::Expected<TargetIdentityId> identity =
      parseTargetIdentityId(*identitySpelling);
  if (!identity)
    return identity.takeError();
  llvm::Expected<KernelRuntimeABIId> abi =
      parseKernelRuntimeABIId(*abiSpelling);
  if (!abi)
    return abi.takeError();
  if (*schema != kTargetLLVMSchema || *logicalRank != expectedLogicalRank ||
      *entrySymbol != expectedEntrySymbol || *profile != expectedProfile ||
      *identity != expectedTargetIdentity || *abi != expectedKernelRuntimeABI ||
      *moduleFormat != expectedModuleFormat)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "target LLVM module metadata does not match the typed rank identity");

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
        "typed Kernel ABI slot");
  return verifyTargetLLVMSlotMetadata(module, expectedSlots);
}

llvm::Error fail(llvm::raw_ostream &diagnostics, llvm::StringRef message) {
  diagnostics << "wafer-compile: " << message << "\n";
  return llvm::createStringError(llvm::errc::invalid_argument, "%s",
                                 message.str().c_str());
}

bool pathEntryExists(llvm::StringRef path) {
  llvm::sys::fs::file_type type =
      llvm::sys::fs::get_file_type(path, /*Follow=*/false);
  return type != llvm::sys::fs::file_type::file_not_found &&
         type != llvm::sys::fs::file_type::status_error;
}

bool isRegularFile(llvm::StringRef path) {
  return llvm::sys::fs::get_file_type(path, /*Follow=*/false) ==
         llvm::sys::fs::file_type::regular_file;
}

bool isExecutableFile(llvm::StringRef path) {
  return isRegularFile(path) &&
         !llvm::sys::fs::access(path, llvm::sys::fs::AccessMode::Execute);
}

bool checkedAdd(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0 || rhs > std::numeric_limits<int64_t>::max() - lhs)
    return false;
  result = lhs + rhs;
  return true;
}

KernelABISlotRole getKernelRole(ProgramResourceRole role) {
  switch (role) {
  case ProgramResourceRole::UserInput:
    return KernelABISlotRole::UserInput;
  case ProgramResourceRole::Parameter:
    return KernelABISlotRole::Parameter;
  case ProgramResourceRole::Constant:
    return KernelABISlotRole::Constant;
  case ProgramResourceRole::Output:
    return KernelABISlotRole::Output;
  }
  llvm_unreachable("unknown program resource role");
}

mlir::FailureOr<int64_t> getPhysicalBytes(mlir::Type type,
                                          mlir::Operation *anchor) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memrefType || !isWaferDDRMemRefType(memrefType)) {
    anchor->emitError()
        << "target_abi_mismatch: kernel resource is not a Wafer DDR memref";
    return mlir::failure();
  }
  std::optional<WaferPhysicalTensorInfo> info =
      computeWaferPhysicalTensorInfo(memrefType);
  if (!info || info->physicalBytes < 0) {
    anchor->emitError()
        << "target_abi_mismatch: cannot derive static resource byte size";
    return mlir::failure();
  }
  return info->physicalBytes;
}

mlir::Value resolveOutputAllocation(mlir::Value value) {
  llvm::SmallPtrSet<mlir::Operation *, 8> visited;
  while (value) {
    if (auto blockArgument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      mlir::Block *owner = blockArgument.getOwner();
      auto tileRegion =
          owner ? mlir::dyn_cast_or_null<TileRegionOp>(owner->getParentOp())
                : TileRegionOp{};
      if (!tileRegion || owner != &tileRegion.getBody().front() ||
          blockArgument.getArgNumber() >= tileRegion.getInputs().size())
        return value;
      value = tileRegion.getInputs()[blockArgument.getArgNumber()];
      continue;
    }

    mlir::Operation *definition = value.getDefiningOp();
    if (!definition || !visited.insert(definition).second)
      return value;
    if (mlir::isa<mlir::memref::AllocOp>(definition))
      return value;
    if (auto tileRegion = mlir::dyn_cast<TileRegionOp>(definition)) {
      auto result = mlir::dyn_cast<mlir::OpResult>(value);
      auto yield = mlir::dyn_cast<TileYieldOp>(
          tileRegion.getBody().front().getTerminator());
      if (!result || !yield ||
          result.getResultNumber() >= yield.getNumOperands())
        return value;
      value = yield.getOperand(result.getResultNumber());
      continue;
    }
    if (auto view = mlir::dyn_cast<mlir::ViewLikeOpInterface>(definition)) {
      value = view.getViewSource();
      continue;
    }
    return value;
  }
  return value;
}

mlir::FailureOr<PreparedTargetRank>
prepareTargetABI(const RankExecutable &rankExecutable,
                 const ExecutionConfig &executionConfig) {
  PreparedTargetRank prepared(executionConfig);
  prepared.module = rankExecutable.getModule().clone();
  prepared.logicalRank = rankExecutable.getLogicalRank();
  const int64_t defaultDDRAlignment =
      getDefaultWaferTargetPolicy().memory.ddrAlignmentBytes;

  llvm::Expected<detail::AcceptedCallClosure> closure =
      detail::analyzeAcceptedCallClosure(*prepared.module,
                                         rankExecutable.getEntrySymbol());
  if (!closure) {
    prepared.module->emitError()
        << "target_abi_mismatch: " << llvm::toString(closure.takeError());
    return mlir::failure();
  }
  mlir::func::FuncOp function = closure->entry;

  unsigned originalArgumentCount = function.getNumArguments();
  unsigned resultCount = function.getFunctionType().getNumResults();
  std::vector<const RankProgramBinding *> argumentBindings(
      originalArgumentCount, nullptr);
  std::vector<const RankProgramBinding *> outputBindings(resultCount, nullptr);
  for (const RankProgramBinding &binding :
       rankExecutable.getProgramBindings()) {
    std::vector<const RankProgramBinding *> &domain =
        binding.role == ProgramResourceRole::Output ? outputBindings
                                                    : argumentBindings;
    if (binding.index < 0 ||
        binding.index >= static_cast<int64_t>(domain.size()) ||
        domain[binding.index]) {
      function.emitError()
          << "target_abi_mismatch: resource bindings do not form an exact "
             "function boundary";
      return mlir::failure();
    }
    domain[binding.index] = &binding;
  }
  if (llvm::is_contained(argumentBindings, nullptr) ||
      llvm::is_contained(outputBindings, nullptr)) {
    function.emitError()
        << "target_abi_mismatch: resource bindings do not cover every "
           "argument and result";
    return mlir::failure();
  }

  prepared.slots.reserve(originalArgumentCount + resultCount + 1);
  auto appendSlot = [&](const RankProgramBinding &binding, mlir::Type type,
                        KernelABISlotRole role) -> mlir::LogicalResult {
    mlir::FailureOr<int64_t> byteSize = getPhysicalBytes(type, function);
    if (mlir::failed(byteSize))
      return mlir::failure();
    prepared.slots.push_back({static_cast<int64_t>(prepared.slots.size()), role,
                              binding.index, binding.name, binding.dtype,
                              binding.localShape, *byteSize,
                              defaultDDRAlignment});
    return mlir::success();
  };

  for (unsigned index = 0; index < originalArgumentCount; ++index)
    if (mlir::failed(appendSlot(*argumentBindings[index],
                                function.getArgument(index).getType(),
                                getKernelRole(argumentBindings[index]->role))))
      return mlir::failure();

  llvm::SmallVector<mlir::func::ReturnOp, 2> returns;
  function.walk([&](mlir::func::ReturnOp returnOp) {
    if (returnOp->getParentOfType<mlir::func::FuncOp>() == function)
      returns.push_back(returnOp);
  });
  if (returns.size() != 1 || returns.front().getNumOperands() != resultCount) {
    function.emitError()
        << "target_abi_mismatch: current output binding requires exactly one "
           "complete entry return";
    return mlir::failure();
  }

  for (unsigned index = 0; index < resultCount; ++index) {
    mlir::Value root =
        resolveOutputAllocation(returns.front().getOperand(index));
    auto allocation = root.getDefiningOp<mlir::memref::AllocOp>();
    if (!allocation || !isWaferDDRMemRefType(allocation.getType()) ||
        !allocation->getAttrOfType<DDROffsetAttr>(kWaferDDROffsetAttrName)) {
      returns.front().emitError()
          << "target_abi_mismatch: output #" << index
          << " is not backed by one accepted compiler-managed DDR root";
      return mlir::failure();
    }
    mlir::Type resultType = function.getFunctionType().getResult(index);
    if (allocation.getType() != resultType) {
      allocation.emitError()
          << "target_abi_mismatch: output root type differs from result type";
      return mlir::failure();
    }
    unsigned outputArgumentIndex = function.getNumArguments();
    function.insertArgument(outputArgumentIndex, resultType,
                            mlir::DictionaryAttr{}, function.getLoc());
    mlir::BlockArgument outputArgument =
        function.getArgument(outputArgumentIndex);
    allocation.getResult().replaceAllUsesWith(outputArgument);
    allocation.erase();
    if (mlir::failed(appendSlot(*outputBindings[index], resultType,
                                KernelABISlotRole::Output)))
      return mlir::failure();
  }

  int64_t arenaBytes = 0;
  int64_t arenaAlignment = defaultDDRAlignment;
  mlir::LogicalResult arenaValid = mlir::success();
  function.walk([&](mlir::memref::AllocOp allocation) {
    if (mlir::failed(arenaValid) || !isWaferDDRMemRefType(allocation.getType()))
      return;
    auto offset =
        allocation->getAttrOfType<DDROffsetAttr>(kWaferDDROffsetAttrName);
    std::optional<WaferPhysicalTensorInfo> info =
        computeWaferPhysicalTensorInfo(allocation.getType());
    int64_t end = 0;
    if (!offset || !info || info->physicalBytes < 0 ||
        !checkedAdd(offset.getOffset(), info->physicalBytes, end)) {
      arenaValid = allocation.emitError()
                   << "target_abi_mismatch: cannot derive default DDR arena "
                      "high-water bytes";
      return;
    }
    arenaBytes = std::max(arenaBytes, end);
    if (auto alignment = allocation.getAlignmentAttr())
      arenaAlignment = std::max(arenaAlignment, alignment.getInt());
  });
  if (mlir::failed(arenaValid))
    return mlir::failure();

  if (arenaBytes > 0) {
    prepared.defaultDDRArenaArgumentIndex = function.getNumArguments();
    function.insertArgument(prepared.defaultDDRArenaArgumentIndex,
                            mlir::IntegerType::get(function.getContext(), 64),
                            mlir::DictionaryAttr{}, function.getLoc());
    prepared.slots.push_back({static_cast<int64_t>(prepared.slots.size()),
                              KernelABISlotRole::Workspace,
                              0,
                              "default_ddr_arena",
                              "u8",
                              {arenaBytes},
                              arenaBytes,
                              arenaAlignment});
  }

  if (rankExecutable.getTransportContract() == TransportContract::DirectDTE) {
    prepared.transportStatusArgumentIndex = function.getNumArguments();
    function.insertArgument(prepared.transportStatusArgumentIndex,
                            mlir::IntegerType::get(function.getContext(), 64),
                            mlir::DictionaryAttr{}, function.getLoc());
    prepared.slots.push_back({static_cast<int64_t>(prepared.slots.size()),
                              KernelABISlotRole::TransportStatus,
                              0,
                              "direct_dte_status",
                              "u32",
                              {1},
                              4,
                              4});
  }

  if (mlir::failed(mlir::verify(*prepared.module)))
    return mlir::failure();
  return prepared;
}

mlir::LogicalResult lowerToTargetLLVM(PreparedTargetRank &prepared) {
  TargetConversionRequest request{
      prepared.targetProfile,
      prepared.defaultDDRArenaArgumentIndex,
      prepared.logicalRank,
      prepared.transportStatusArgumentIndex,
  };
  mlir::PassManager manager(prepared.module->getContext());
  manager.addPass(createLowerInstrToTargetLLVMPass(request));
  return manager.run(*prepared.module);
}

mlir::LogicalResult verifyLoweredKernelABI(PreparedTargetRank &prepared,
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
              "with one parameter per typed Kernel ABI slot";
  return mlir::success();
}

llvm::Expected<TargetLLVMModule>
translatePreparedTargetRank(PreparedTargetRank prepared,
                            llvm::StringRef entrySymbol) {
  auto llvmContext = std::make_unique<llvm::LLVMContext>();
  std::unique_ptr<llvm::Module> llvmModule = mlir::translateModuleToLLVMIR(
      *prepared.module, *llvmContext, "wafer_target_rank");
  if (!llvmModule)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "target LLVM IR translation failed");
  llvmModule->setModuleIdentifier(
      llvm::formatv("wafer.target.rank.{0:D5}", prepared.logicalRank).str());
  llvmModule->setTargetTriple(kTargetLLVMTriple);
  attachTargetLLVMMetadata(*llvmModule, prepared, entrySymbol);
  if (llvm::Error error = verifyTargetLLVMModule(
          *llvmModule, prepared.logicalRank, entrySymbol,
          prepared.targetProfile, prepared.targetIdentity,
          prepared.kernelRuntimeABI, prepared.moduleFormat, prepared.slots))
    return std::move(error);
  return TargetLLVMModuleBundleBuilder::makeModule(
      prepared.logicalRank, entrySymbol, prepared.targetProfile,
      prepared.targetIdentity, prepared.kernelRuntimeABI, prepared.moduleFormat,
      std::move(prepared.slots), std::move(llvmContext), std::move(llvmModule));
}

llvm::Error writeLLVMIR(const llvm::Module &module, llvm::StringRef path) {
  std::error_code error;
  llvm::raw_fd_ostream output(path, error, llvm::sys::fs::OF_Text);
  if (error)
    return llvm::createStringError(error, "failed to open target LLVM IR");
  module.print(output, nullptr);
  output.close();
  if (output.has_error())
    return llvm::createStringError(llvm::errc::io_error,
                                   "failed to write target LLVM IR");
  return llvm::Error::success();
}

llvm::Error runDeviceLink(const TargetToolchain &toolchain,
                          llvm::StringRef llvmIR, llvm::StringRef module,
                          llvm::StringRef object, llvm::StringRef crtObject) {
  std::string python = toolchain.getPythonExecutable().str();
  std::string script = toolchain.getDeviceLinkerScript().str();
  std::string llvmIRStorage = llvmIR.str();
  std::string moduleStorage = module.str();
  std::string objectStorage = object.str();
  std::string crtObjectStorage = crtObject.str();
  llvm::SmallVector<llvm::StringRef, 11> arguments = {python,
                                                      script,
                                                      "--llvm-ir",
                                                      llvmIRStorage,
                                                      "--output",
                                                      moduleStorage,
                                                      "--object-output",
                                                      objectStorage,
                                                      "--crt-object-output",
                                                      crtObjectStorage};
  int exitCode = llvm::sys::ExecuteAndWait(python, arguments);
  if (exitCode == 0)
    return llvm::Error::success();
  return llvm::createStringError(
      llvm::errc::io_error, "device link failed with exit code %d", exitCode);
}

struct TargetModuleReadback {
  std::string contentDigest;
  std::string moduleFormat;
};

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
  if (!isRegularFile(path))
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

bool publishDirectoryNoReplace(llvm::StringRef source,
                               llvm::StringRef destination,
                               llvm::raw_ostream &diagnostics) {
#ifdef __linux__
  std::string sourceStorage = source.str();
  std::string destinationStorage = destination.str();
  if (::syscall(SYS_renameat2, AT_FDCWD, sourceStorage.c_str(), AT_FDCWD,
                destinationStorage.c_str(), RENAME_NOREPLACE) == 0)
    return false;
  int errorNumber = errno;
  diagnostics << "wafer-compile: target_publication_failed: "
              << std::error_code(errorNumber, std::generic_category()).message()
              << "\n";
  return true;
#else
  (void)source;
  (void)destination;
  diagnostics << "wafer-compile: target_publication_failed: atomic no-replace "
                 "directory publication is unsupported on this host\n";
  return true;
#endif
}

} // namespace

llvm::Expected<TargetLLVMModuleBundle>
detail::compileExecutableBundleToTargetLLVMModulesImpl(
    const ExecutableBundle &executableBundle, llvm::raw_ostream &diagnostics,
    std::optional<int64_t> failAfterLogicalRank) {
  const std::vector<RankExecutable> &ranks =
      executableBundle.getRankExecutables();
  const ExecutionConfig &executionConfig =
      executableBundle.getExecutionConfig();
  if (ranks.size() != static_cast<size_t>(executionConfig.getRankCount()))
    return fail(diagnostics, "target LLVM rank domain is incomplete");

  const TargetProfileRecord &targetProfile =
      getTargetProfileRecord(executionConfig.getTargetProfileId());
  std::vector<PreparedTargetRank> preparedRanks;
  preparedRanks.reserve(ranks.size());
  for (auto [expectedRank, rank] : llvm::enumerate(ranks)) {
    if (rank.getLogicalRank() != static_cast<int64_t>(expectedRank))
      return fail(diagnostics, "target LLVM rank domain is not canonical");
    mlir::FailureOr<PreparedTargetRank> prepared =
        prepareTargetABI(rank, executionConfig);
    if (mlir::failed(prepared))
      return fail(diagnostics,
                  "target ABI preparation failed for logical rank " +
                      std::to_string(expectedRank));
    if (mlir::failed(lowerToTargetLLVM(*prepared)))
      return fail(diagnostics, "target lowering failed for logical rank " +
                                   std::to_string(expectedRank));
    if (mlir::failed(verifyLoweredKernelABI(*prepared, rank.getEntrySymbol())))
      return fail(diagnostics,
                  "target ABI verification failed for logical rank " +
                      std::to_string(expectedRank));
    if (prepared->logicalRank != static_cast<int64_t>(expectedRank) ||
        prepared->targetProfile != targetProfile.id ||
        prepared->targetIdentity != targetProfile.targetIdentity ||
        prepared->kernelRuntimeABI != targetProfile.kernelRuntimeABI ||
        prepared->moduleFormat != targetProfile.moduleFormat)
      return fail(diagnostics,
                  "prepared target profile readback failed for logical rank " +
                      std::to_string(expectedRank));
    preparedRanks.push_back(std::move(*prepared));
  }

  std::vector<TargetLLVMModule> modules;
  modules.reserve(ranks.size());
  for (auto [expectedRank, prepared] : llvm::enumerate(preparedRanks)) {
    llvm::Expected<TargetLLVMModule> translated = translatePreparedTargetRank(
        std::move(prepared), ranks[expectedRank].getEntrySymbol());
    if (!translated)
      return fail(diagnostics,
                  "target LLVM translation/readback failed for logical rank " +
                      std::to_string(expectedRank) + ": " +
                      llvm::toString(translated.takeError()));
    modules.push_back(std::move(*translated));
    if (failAfterLogicalRank &&
        static_cast<int64_t>(expectedRank) == *failAfterLogicalRank)
      return fail(diagnostics,
                  "test-only injected target failure after logical rank " +
                      std::to_string(expectedRank));
  }

  return TargetLLVMModuleBundleBuilder::makeBundle(executionConfig,
                                                   std::move(modules));
}

mlir::LogicalResult
detail::lowerTargetABIForTesting(const RankExecutable &rankExecutable,
                                 const ExecutionConfig &executionConfig) {
  mlir::FailureOr<PreparedTargetRank> prepared =
      prepareTargetABI(rankExecutable, executionConfig);
  if (mlir::failed(prepared) || mlir::failed(lowerToTargetLLVM(*prepared)))
    return mlir::failure();
  return verifyLoweredKernelABI(*prepared, rankExecutable.getEntrySymbol());
}

llvm::Error
detail::verifyTargetLLVMModuleForTesting(const TargetLLVMModule &targetModule) {
  return verifyTargetLLVMModule(
      targetModule.getModule(), targetModule.getLogicalRank(),
      targetModule.getEntrySymbol(), targetModule.getTargetProfileId(),
      targetModule.getTargetIdentityId(), targetModule.getKernelRuntimeABIId(),
      targetModule.getModuleFormat(), targetModule.getKernelABISlots());
}

llvm::Expected<VerifiedTargetModule>
detail::verifyLinkedTargetModuleForTesting(llvm::StringRef path,
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

llvm::Expected<TargetArtifactBundle>
compileTargetLLVMModuleBundleToTargetArtifacts(
    const TargetLLVMModuleBundle &targetLLVMModules,
    llvm::StringRef outputDirectory, const TargetToolchain &toolchain,
    llvm::raw_ostream &diagnostics) {
  if (targetLLVMModules.getModules().size() !=
      static_cast<size_t>(
          targetLLVMModules.getExecutionConfig().getRankCount()))
    return fail(diagnostics, "target LLVM bundle rank domain is incomplete");
  if (outputDirectory.empty())
    return fail(diagnostics, "target artifact directory must not be empty");
  if (pathEntryExists(outputDirectory))
    return fail(diagnostics,
                "refusing to replace existing target artifact directory");
  if (!isExecutableFile(toolchain.getPythonExecutable()))
    return fail(diagnostics, "configured Python executable is not executable");
  if (!isRegularFile(toolchain.getDeviceLinkerScript()))
    return fail(diagnostics,
                "configured device linker script is not a regular file");

  llvm::SmallString<256> outputParent(outputDirectory);
  llvm::sys::path::remove_filename(outputParent);
  if (outputParent.empty())
    outputParent = ".";
  if (std::error_code error = llvm::sys::fs::create_directories(outputParent))
    return fail(diagnostics,
                "failed to create target artifact parent: " + error.message());
  llvm::SmallString<256> stagingPrefix(outputParent);
  llvm::sys::path::append(stagingPrefix, ".wafer-target-artifacts-staging");
  llvm::SmallString<256> stagingRoot;
  if (std::error_code error =
          llvm::sys::fs::createUniqueDirectory(stagingPrefix, stagingRoot))
    return fail(diagnostics,
                "failed to create target artifact staging: " + error.message());
  auto cleanup = llvm::make_scope_exit(
      [&] { llvm::sys::fs::remove_directories(stagingRoot); });

  llvm::SmallString<256> modulesDirectory(stagingRoot);
  llvm::sys::path::append(modulesDirectory, "modules");
  llvm::SmallString<256> workDirectory(stagingRoot);
  llvm::sys::path::append(workDirectory, "work");
  if (std::error_code error =
          llvm::sys::fs::create_directories(modulesDirectory))
    return fail(diagnostics,
                "failed to create staged module directory: " + error.message());
  if (std::error_code error = llvm::sys::fs::create_directories(workDirectory))
    return fail(diagnostics,
                "failed to create target work directory: " + error.message());

  std::vector<VerifiedTargetModule> modules;
  modules.reserve(targetLLVMModules.getModules().size());
  for (auto [expectedRank, targetLLVMModule] :
       llvm::enumerate(targetLLVMModules.getModules())) {
    if (targetLLVMModule.getLogicalRank() != static_cast<int64_t>(expectedRank))
      return fail(diagnostics,
                  "target LLVM bundle rank domain is not canonical");
    std::string stem = (llvm::formatv("rank_{0:D5}", expectedRank)).str();
    llvm::SmallString<256> llvmIRPath(workDirectory);
    llvm::sys::path::append(llvmIRPath, stem + ".ll");
    llvm::SmallString<256> objectPath(workDirectory);
    llvm::sys::path::append(objectPath, stem + ".o");
    llvm::SmallString<256> crtObjectPath(workDirectory);
    llvm::sys::path::append(crtObjectPath, stem + ".wafer_crt.o");
    llvm::SmallString<256> modulePath(modulesDirectory);
    llvm::sys::path::append(modulePath, stem + ".so");
    if (llvm::Error error =
            writeLLVMIR(targetLLVMModule.getModule(), llvmIRPath))
      return fail(diagnostics, "target_module_verification_failed: " +
                                   llvm::toString(std::move(error)));
    if (llvm::Error error = runDeviceLink(toolchain, llvmIRPath, modulePath,
                                          objectPath, crtObjectPath))
      return fail(diagnostics, "target_module_verification_failed: " +
                                   llvm::toString(std::move(error)));
    llvm::Expected<TargetModuleReadback> moduleReadback =
        verifyTargetModule(modulePath, targetLLVMModule.getEntrySymbol(),
                           targetLLVMModule.getTargetProfileId());
    if (!moduleReadback)
      return fail(diagnostics, "target_module_verification_failed: " +
                                   llvm::toString(moduleReadback.takeError()));
    if (moduleReadback->moduleFormat != targetLLVMModule.getModuleFormat())
      return fail(diagnostics,
                  "target module format readback does not match prepared "
                  "target profile for logical rank " +
                      std::to_string(expectedRank));
    std::string relativePath = (llvm::Twine("modules/") + stem + ".so").str();
    modules.push_back(TargetArtifactBundleBuilder::makeModule(
        expectedRank, targetLLVMModule.getEntrySymbol(), relativePath,
        moduleReadback->contentDigest, targetLLVMModule.getTargetProfileId(),
        targetLLVMModule.getTargetIdentityId(),
        targetLLVMModule.getKernelRuntimeABIId(), moduleReadback->moduleFormat,
        targetLLVMModule.getKernelABISlots()));
  }

  if (std::error_code error = llvm::sys::fs::remove_directories(workDirectory))
    return fail(diagnostics,
                "failed to remove target work directory: " + error.message());
  if (publishDirectoryNoReplace(stagingRoot, outputDirectory, diagnostics))
    return llvm::createStringError(llvm::errc::io_error,
                                   "target artifact publication failed");
  cleanup.release();
  return TargetArtifactBundleBuilder::makeBundle(
      outputDirectory, targetLLVMModules.getExecutionConfig(),
      std::move(modules));
}

llvm::Expected<TargetArtifactBundle>
detail::compileExecutableBundleToTargetArtifactsImpl(
    const ExecutableBundle &executableBundle, llvm::StringRef outputDirectory,
    const TargetToolchain &toolchain, llvm::raw_ostream &diagnostics,
    std::optional<int64_t> failAfterLogicalRank) {
  llvm::Expected<TargetLLVMModuleBundle> targetLLVMModules =
      detail::compileExecutableBundleToTargetLLVMModulesImpl(
          executableBundle, diagnostics, failAfterLogicalRank);
  if (!targetLLVMModules)
    return targetLLVMModules.takeError();
  return compileTargetLLVMModuleBundleToTargetArtifacts(
      *targetLLVMModules, outputDirectory, toolchain, diagnostics);
}

llvm::Expected<TargetArtifactBundle> compileExecutableBundleToTargetArtifacts(
    const ExecutableBundle &executableBundle, llvm::StringRef outputDirectory,
    const TargetToolchain &toolchain, llvm::raw_ostream &diagnostics) {
  return detail::compileExecutableBundleToTargetArtifactsImpl(
      executableBundle, outputDirectory, toolchain, diagnostics, std::nullopt);
}

llvm::Expected<TargetLLVMModuleBundle>
compileExecutableBundleToTargetLLVMModules(
    const ExecutableBundle &executableBundle, llvm::raw_ostream &diagnostics) {
  return detail::compileExecutableBundleToTargetLLVMModulesImpl(
      executableBundle, diagnostics, std::nullopt);
}

} // namespace wafer::compiler
