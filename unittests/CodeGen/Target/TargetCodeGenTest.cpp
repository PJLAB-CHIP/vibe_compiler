//===- TargetCodeGenTest.cpp - Target code generation tests -------------===//

#include "Wafer/CodeGen/Executable/CardExecutableInternal.h"
#include "Wafer/CodeGen/Target/TargetCodeGenInternal.h"
#include "Wafer/Package/Writer/PackageInternal.h"
#include "Wafer/Program/ProgramData.h"

#include "Wafer/Target/Core/RuntimeLaunchContract.h"
#include "Wafer/Target/Core/TargetIdentity.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "gtest/gtest.h"

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#ifndef WAFER_TEST_PYTHON_EXECUTABLE
#error "WAFER_TEST_PYTHON_EXECUTABLE must name the configured Python"
#endif
#ifndef WAFER_TEST_DEVICE_LINKER_SCRIPT
#error "WAFER_TEST_DEVICE_LINKER_SCRIPT must name the device linker"
#endif
#ifndef WAFER_TEST_LLVM_CLANGXX
#error "WAFER_TEST_LLVM_CLANGXX must name a clang++ with RISC-V support"
#endif
#ifndef WAFER_TEST_TARGET_DATA_ENTRY_LLVM_IR
#error "WAFER_TEST_TARGET_DATA_ENTRY_LLVM_IR must name the target IR fixture"
#endif

namespace {

constexpr wafer::TargetIdentityId kTargetIdentity =
    wafer::TargetIdentityId::waferTx81SingleCard();

template <typename SemanticT>
const wafer::TargetCallDescriptor &getTargetCallDescriptor(SemanticT semantic) {
  return wafer::getTargetCallDescriptor(semantic);
}

static wafer::RuntimeLaunchContract makeKernelLaunch(
    wafer::KernelLaunchForm form,
    std::optional<wafer::KernelEntryABI> sharedEntryABI = std::nullopt) {
  constexpr std::array main{wafer::RuntimeLaunchPhaseRole::Main};
  constexpr std::array prepareMain{wafer::RuntimeLaunchPhaseRole::Prepare,
                                   wafer::RuntimeLaunchPhaseRole::Main};
  switch (form) {
  case wafer::KernelLaunchForm::Grid:
    return llvm::cantFail(wafer::RuntimeLaunchContract::createKernel(
        form,
        sharedEntryABI.value_or(wafer::KernelEntryABI::TileMajorPointerTable),
        main));
  case wafer::KernelLaunchForm::Cluster:
    return llvm::cantFail(wafer::RuntimeLaunchContract::createKernel(
        form,
        sharedEntryABI.value_or(wafer::KernelEntryABI::TileMajorPointerTable),
        prepareMain));
  }
  llvm_unreachable("unknown kernel launch form");
}

struct SharedKernelTransportScenario {
  wafer::KernelLaunchForm form;
  wafer::compiler::TransportContract transport;
};

constexpr std::array<SharedKernelTransportScenario, 2>
    kOrthogonalSharedKernelTransportScenarios = {{
        {wafer::KernelLaunchForm::Grid,
         wafer::compiler::TransportContract::DirectDTE},
        {wafer::KernelLaunchForm::Cluster,
         wafer::compiler::TransportContract::None},
    }};

static llvm::SmallString<256> pathInDirectory(llvm::StringRef directory,
                                              llvm::StringRef filename) {
  llvm::SmallString<256> path(directory);
  llvm::sys::path::append(path, filename);
  return path;
}

static int linkTargetModule(llvm::StringRef inputPath,
                            llvm::StringRef outputPath) {
  std::string python = WAFER_TEST_PYTHON_EXECUTABLE;
  std::string script = WAFER_TEST_DEVICE_LINKER_SCRIPT;
  std::string input = inputPath.str();
  std::string output = outputPath.str();
  std::string clang = WAFER_TEST_LLVM_CLANGXX;
  llvm::SmallVector<llvm::StringRef, 10> arguments = {
      python,     script, "--llvm-ir",      input,
      "--output", output, "--llvm-clangxx", clang};
  return llvm::sys::ExecuteAndWait(python, arguments);
}

static int linkTargetFixture(llvm::StringRef outputPath) {
  return linkTargetModule(WAFER_TEST_TARGET_DATA_ENTRY_LLVM_IR, outputPath);
}

static llvm::Expected<wafer::compiler::TargetToolchain> makeTestToolchain() {
  llvm::SmallString<256> pythonExecutable;
  llvm::SmallString<256> llvmClangXX;
  if (std::error_code error = llvm::sys::fs::real_path(
          WAFER_TEST_PYTHON_EXECUTABLE, pythonExecutable))
    return llvm::createStringError(error,
                                   "failed to resolve test Python executable");
  if (std::error_code error =
          llvm::sys::fs::real_path(WAFER_TEST_LLVM_CLANGXX, llvmClangXX))
    return llvm::createStringError(error, "failed to resolve test clang++");
  return wafer::compiler::TargetToolchain::create(
      pythonExecutable, WAFER_TEST_DEVICE_LINKER_SCRIPT, llvmClangXX,
      WAFER_TEST_TX8_DEPS_ROOT, WAFER_TEST_WAFER_INCLUDE_DIR,
      WAFER_TEST_WAFER_CRT_SOURCE, WAFER_TEST_WAFER_CRT_INCLUDE_DIR);
}

static wafer::compiler::TileEntryArgument
makeSlot(int64_t ordinal, wafer::compiler::TileEntryArgumentKind role,
         llvm::StringRef name) {
  if (role == wafer::compiler::TileEntryArgumentKind::TransportStatus)
    return {ordinal,
            role,
            0,
            name.str(),
            wafer::LogicalFormat::U32,
            wafer::MemLayout::Tensor,
            {1},
            WAFER_TX81_DIRECT_DTE_STATUS_STORAGE_BYTES,
            WAFER_TX81_DIRECT_DTE_STATUS_STORAGE_ALIGNMENT,
            wafer::compiler::TileEntryArgumentAccess::ReadWrite};
  const auto access =
      role == wafer::compiler::TileEntryArgumentKind::ExternalOutput
          ? wafer::compiler::TileEntryArgumentAccess::WriteOnly
          : wafer::compiler::TileEntryArgumentAccess::ReadOnly;
  return {ordinal,
          role,
          ordinal,
          name.str(),
          wafer::LogicalFormat::F32,
          wafer::MemLayout::Tensor,
          {4},
          16,
          64,
          access};
}

static llvm::Expected<wafer::compiler::TargetLLVMModules>
makeRuntimeLaunchModules(
    wafer::RuntimeLaunchContract launch,
    wafer::compiler::TransportContract transport =
        wafer::compiler::TransportContract::None,
    std::optional<int64_t> mismatchedSchemaTile = std::nullopt,
    std::optional<int64_t> mismatchedBodyTile = std::nullopt,
    size_t slotCount = 2, bool withCollidingClosure = false,
    bool withUnsupportedInlineAsm = false,
    std::optional<int64_t> renamedSlotTile = std::nullopt,
    bool permuteTiles = false, bool variableWorkspace = false) {
  const int64_t tileCount = 16;
  llvm::Expected<wafer::compiler::ExecutionConfig> config =
      wafer::compiler::ExecutionConfig::createForSingleCard(1);
  if (!config)
    return config.takeError();
  std::vector<wafer::compiler::TargetLLVMModule> modules;
  modules.reserve(tileCount);
  for (int64_t tile = 0; tile < tileCount; ++tile) {
    std::vector<wafer::compiler::TileEntryArgument> slots;
    slots.reserve(slotCount);
    for (size_t slot = 0; slot < slotCount; ++slot) {
      wafer::compiler::TileEntryArgumentKind role =
          variableWorkspace && slot + 1 == slotCount
              ? wafer::compiler::TileEntryArgumentKind::Workspace
          : transport == wafer::compiler::TransportContract::DirectDTE &&
                  slot + 1 == slotCount
              ? wafer::compiler::TileEntryArgumentKind::TransportStatus
          : slot + 1 == slotCount
              ? wafer::compiler::TileEntryArgumentKind::ExternalOutput
              : wafer::compiler::TileEntryArgumentKind::ExternalInput;
      llvm::StringRef name =
          role == wafer::compiler::TileEntryArgumentKind::ExternalOutput
              ? "output"
          : role == wafer::compiler::TileEntryArgumentKind::TransportStatus
              ? "direct_dte_status"
              : "input";
      slots.push_back(makeSlot(
          slot, role,
          renamedSlotTile == tile && slot == 0 ? "diagnostic_alias" : name));
      if (role == wafer::compiler::TileEntryArgumentKind::Workspace) {
        slots.back() = {static_cast<int64_t>(slot),
                        role,
                        0,
                        "default_ddr_arena",
                        wafer::LogicalFormat::U8,
                        wafer::MemLayout::Tensor,
                        {256 + 16 * tile},
                        256 + 16 * tile,
                        tile % 2 == 0 ? 16 : 32,
                        wafer::compiler::TileEntryArgumentAccess::ReadWrite};
      }
      if (mismatchedSchemaTile == tile && slot == 0)
        slots.back().dtype = wafer::LogicalFormat::F16;
    }
    auto context = std::make_unique<llvm::LLVMContext>();
    auto module = std::make_unique<llvm::Module>("launch-entry", *context);
    if (withUnsupportedInlineAsm && tile == 0)
      module->setModuleInlineAsm(".word 0");
    llvm::Type *i64 = llvm::Type::getInt64Ty(*context);
    llvm::Function *helper = nullptr;
    if (withCollidingClosure) {
      auto *state = new llvm::GlobalVariable(
          *module, i64, /*isConstant=*/false,
          llvm::GlobalValue::ExternalLinkage,
          llvm::ConstantInt::get(i64, static_cast<uint64_t>(tile)), "state");
      llvm::FunctionType *helperType = llvm::FunctionType::get(
          llvm::Type::getVoidTy(*context), {}, /*isVarArg=*/false);
      helper = llvm::Function::Create(
          helperType, llvm::GlobalValue::ExternalLinkage, "helper", *module);
      llvm::IRBuilder<> helperBuilder(
          llvm::BasicBlock::Create(*context, "entry", helper));
      llvm::LoadInst *load = helperBuilder.CreateLoad(i64, state);
      load->setVolatile(true);
      helperBuilder.CreateRetVoid();
    }
    llvm::SmallVector<llvm::Type *, 16> argumentTypes(slotCount, i64);
    llvm::FunctionType *type = llvm::FunctionType::get(
        llvm::Type::getVoidTy(*context), argumentTypes, /*isVarArg=*/false);
    llvm::Function *entry = llvm::Function::Create(
        type, llvm::GlobalValue::ExternalLinkage, "main", *module);
    llvm::IRBuilder<> builder(
        llvm::BasicBlock::Create(*context, "entry", entry));
    if (helper)
      builder.CreateCall(helper);
    if (mismatchedBodyTile == tile) {
      llvm::Value *address = builder.CreateIntToPtr(
          entry->getArg(0), llvm::PointerType::get(*context, 0));
      llvm::StoreInst *store =
          builder.CreateStore(llvm::ConstantInt::get(i64, 0), address);
      store->setVolatile(true);
    }
    builder.CreateRetVoid();
    modules.push_back(wafer::compiler::TargetLLVMModulesBuilder::makeModule(
        wafer::CardId(0),
        wafer::TileId(permuteTiles && tile < 2 ? 1 - tile : tile),
        wafer::LaunchSlotId(tile), "main", wafer::kCurrentTargetIdentity,
        wafer::kCurrentKernelRuntimeABI, wafer::kCurrentTargetModuleFormat,
        std::move(slots), std::move(context), std::move(module)));
  }
  return wafer::compiler::TargetLLVMModulesBuilder::makeModules(
      *config, std::move(launch), std::move(modules));
}

static wafer::compiler::TileEntryArgument
makeProfilerSlot(int64_t ordinal,
                 wafer::compiler::detail::ProfileCaptureKind capture) {
  const int64_t bytes = static_cast<int64_t>(
      wafer::compiler::detail::getProfileCaptureRecordBytes(capture));
  return {ordinal,
          wafer::compiler::TileEntryArgumentKind::ProfileRecord,
          0,
          "tx81_profiler_record",
          wafer::LogicalFormat::U8,
          wafer::MemLayout::Tensor,
          {bytes},
          bytes,
          WAFER_TX81_PROFILER_BUFFER_ALIGNMENT,
          wafer::compiler::TileEntryArgumentAccess::ReadWrite};
}

static llvm::Expected<wafer::compiler::TargetLLVMModules>
makeProfileRuntimeLaunchModules(
    wafer::RuntimeLaunchContract launch,
    wafer::compiler::detail::ProfileCaptureKind capture,
    wafer::compiler::TransportContract transport) {
  llvm::Expected<wafer::compiler::ExecutionConfig> config =
      wafer::compiler::ExecutionConfig::createForSingleCard(1);
  if (!config)
    return config.takeError();
  std::vector<wafer::compiler::TargetLLVMModule> modules;
  modules.reserve(16);
  for (int64_t tile = 0; tile < 16; ++tile) {
    std::vector<wafer::compiler::TileEntryArgument> slots;
    slots.push_back(makeSlot(
        0, wafer::compiler::TileEntryArgumentKind::ExternalInput, "input"));
    if (transport == wafer::compiler::TransportContract::DirectDTE)
      slots.push_back(
          makeSlot(1, wafer::compiler::TileEntryArgumentKind::TransportStatus,
                   "direct_dte_status"));
    slots.push_back(makeProfilerSlot(slots.size(), capture));

    auto context = std::make_unique<llvm::LLVMContext>();
    auto module =
        std::make_unique<llvm::Module>("profile-launch-entry", *context);
    llvm::Type *i64 = llvm::Type::getInt64Ty(*context);
    llvm::SmallVector<llvm::Type *, 4> argumentTypes(slots.size(), i64);
    llvm::Function *entry = llvm::Function::Create(
        llvm::FunctionType::get(llvm::Type::getVoidTy(*context), argumentTypes,
                                /*isVarArg=*/false),
        llvm::GlobalValue::ExternalLinkage, "main", *module);
    llvm::IRBuilder<> builder(
        llvm::BasicBlock::Create(*context, "entry", entry));
    builder.CreateRetVoid();
    if (llvm::Error error =
            wafer::compiler::detail::instrumentProfileTargetModule(
                *module, "main", capture))
      return std::move(error);
    modules.push_back(wafer::compiler::TargetLLVMModulesBuilder::makeModule(
        wafer::CardId(0), wafer::TileId(tile), wafer::LaunchSlotId(tile),
        "main", wafer::kCurrentTargetIdentity, wafer::kCurrentKernelRuntimeABI,
        wafer::kCurrentTargetModuleFormat, std::move(slots), std::move(context),
        std::move(module)));
  }
  return wafer::compiler::TargetLLVMModulesBuilder::makeModules(
      *config, std::move(launch), std::move(modules));
}

static llvm::Expected<wafer::compiler::CardExecutable>
makeProfileCardExecutable(const wafer::RuntimeLaunchContract &launch,
                          wafer::compiler::TransportContract transport) {
  llvm::Expected<wafer::compiler::ExecutionConfig> config =
      wafer::compiler::ExecutionConfig::createForSingleCard(1);
  if (!config)
    return config.takeError();
  auto context = std::make_shared<mlir::MLIRContext>();
  std::vector<wafer::compiler::TileExecutable> tiles;
  tiles.reserve(16);
  for (int64_t tileId = 0; tileId < 16; ++tileId) {
    mlir::OpBuilder builder(context.get());
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::ModuleOp::create(builder.getUnknownLoc());
    wafer::compiler::ProgramResourceBinding binding{};
    binding.role = wafer::compiler::ProgramResourceRole::UserInput;
    binding.programTensorId = {binding.role, 0};
    binding.index = 0;
    binding.programIndex = 0;
    binding.name = "input";
    binding.dtype = wafer::ProgramElementType::F32;
    binding.distribution = wafer::frontend::ProgramDistributionKind::Replicated;
    binding.globalShape = {4};
    binding.localShape = {4};
    binding.slice.partitionId = 0;
    binding.slice.replicaId = 0;
    binding.slice.offsets = {0};
    binding.slice.sizes = {4};
    binding.slice.strides = {1};
    tiles.push_back(wafer::compiler::CardExecutableBuilder::makeTileExecutable(
        wafer::CardId(0), wafer::TileId(tileId), wafer::LaunchSlotId(tileId),
        std::move(module), "main", {std::move(binding)}, transport));
  }
  return wafer::compiler::CardExecutableBuilder::makeCardExecutable(
      *config, launch, std::move(context), std::move(tiles),
      std::make_unique<wafer::compiler::ProgramDataHandoff>());
}

TEST(TargetCodeGenTest, PublicVerifiedModuleCannotBeForgedOrDefaulted) {
  static_assert(
      !std::is_default_constructible_v<wafer::compiler::VerifiedTargetModule>);
}

// Two parameter program tensors consumed through three target
// representations whose discovery order differs from the canonical
// descriptor order: per Tile the ordered schema is
// [bias-Tensor, weight-Cx, weight-Tensor], while the canonical placement
// sorts [weight-Tensor, weight-Cx, bias-Tensor]. The writer must re-issue
// canonical ids after the deterministic placement, rewrite every entry
// reference, materialize each representation exactly once through the
// shared physical codec (Cx block padding is canonical zero) and satisfy
// the strict canonical-layout verification.
TEST(TargetCodeGenTest,
     TwoSelectedRepresentationsRemapCanonicalIdsAndPadPhysically) {
  llvm::Expected<wafer::compiler::TargetToolchain> toolchain =
      makeTestToolchain();
  ASSERT_TRUE(static_cast<bool>(toolchain))
      << llvm::toString(toolchain.takeError());
  llvm::SmallString<256> temporaryDirectory;
  ASSERT_FALSE(llvm::sys::fs::createUniqueDirectory(
      "wafer-representation-package", temporaryDirectory));
  auto cleanup = llvm::make_scope_exit(
      [&]() { (void)llvm::sys::fs::remove_directories(temporaryDirectory); });

  // Two payload sources: weight f32{4} and bias f16{2}. Bias is selected as
  // BF16 so same-width conversion must change 1.0 from 0x3c00 to 0x3f80.
  const std::vector<uint8_t> weightPayload = {
      0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40,
      0x00, 0x00, 0x40, 0x40, 0x00, 0x00, 0x80, 0x40,
  };
  const std::vector<uint8_t> biasPayload = {
      0x00,
      0x3c,
      0x00,
      0x40,
  };
  auto writeNpy = [&](llvm::StringRef fileName, llvm::ArrayRef<uint8_t> payload,
                      llvm::StringRef npyDType, uint64_t elementBytes,
                      llvm::SmallVectorImpl<char> &pathOut) {
    const char magic[] = "\x93NUMPY";
    std::vector<uint8_t> npy(magic, magic + 6);
    npy.push_back(1);
    npy.push_back(0);
    const std::string header = "{'descr': '" + npyDType.str() +
                               "', 'fortran_order': False, 'shape': (" +
                               std::to_string(payload.size() / elementBytes) +
                               ",), }";
    npy.push_back(static_cast<uint8_t>(header.size() & 0xff));
    npy.push_back(static_cast<uint8_t>((header.size() >> 8) & 0xff));
    npy.insert(npy.end(), header.begin(), header.end());
    npy.insert(npy.end(), payload.begin(), payload.end());
    pathOut = temporaryDirectory;
    llvm::sys::path::append(pathOut, fileName);
    std::error_code error;
    llvm::raw_fd_ostream output(llvm::StringRef(pathOut.data(), pathOut.size()),
                                error);
    ASSERT_FALSE(error);
    output.write(reinterpret_cast<const char *>(npy.data()), npy.size());
    output.close();
  };
  llvm::SmallString<256> weightPath;
  writeNpy("weight.npy", weightPayload, "<f4", 4, weightPath);
  llvm::SmallString<256> biasPath;
  writeNpy("bias.npy", biasPayload, "<f2", 2, biasPath);
  auto handoff = std::make_unique<wafer::compiler::ProgramDataHandoff>(
      temporaryDirectory.str().str(), /*collectIOStatistics=*/true);
  llvm::Expected<wafer::compiler::SourceDataId> weightSource =
      handoff->establishSource(weightPath, "data/weight", nullptr);
  ASSERT_TRUE(static_cast<bool>(weightSource))
      << llvm::toString(weightSource.takeError());
  llvm::Expected<wafer::compiler::SourceDataId> biasSource =
      handoff->establishSource(biasPath, "data/bias", nullptr);
  ASSERT_TRUE(static_cast<bool>(biasSource))
      << llvm::toString(biasSource.takeError());
  llvm::Expected<wafer::compiler::ProgramDataRange> weightRange =
      wafer::compiler::ProgramDataRange::create(
          {wafer::compiler::ProgramResourceRole::Parameter, 0},
          wafer::ProgramElementType::F32, {4}, {4},
          wafer::frontend::ProgramDistributionKind::Replicated,
          wafer::compiler::ProgramDataRangeOrigin::OriginalSource, {0}, {4},
          {1}, *weightSource, handoff->getSource(*weightSource), nullptr);
  ASSERT_TRUE(static_cast<bool>(weightRange))
      << llvm::toString(weightRange.takeError());
  ASSERT_FALSE(static_cast<bool>(handoff->addRange(std::move(*weightRange))));
  llvm::Expected<wafer::compiler::ProgramDataRange> biasRange =
      wafer::compiler::ProgramDataRange::create(
          {wafer::compiler::ProgramResourceRole::Parameter, 1},
          wafer::ProgramElementType::F16, {2}, {2},
          wafer::frontend::ProgramDistributionKind::Replicated,
          wafer::compiler::ProgramDataRangeOrigin::OriginalSource, {0}, {2},
          {1}, *biasSource, handoff->getSource(*biasSource), nullptr);
  ASSERT_TRUE(static_cast<bool>(biasRange))
      << llvm::toString(biasRange.takeError());
  ASSERT_FALSE(static_cast<bool>(handoff->addRange(std::move(*biasRange))));

  llvm::Expected<wafer::compiler::ExecutionConfig> config =
      wafer::compiler::ExecutionConfig::createForSingleCard(1);
  ASSERT_TRUE(static_cast<bool>(config));
  wafer::RuntimeLaunchContract launch =
      makeKernelLaunch(wafer::KernelLaunchForm::Grid);
  std::vector<wafer::compiler::TargetLLVMModule> modules;
  modules.reserve(16);
  for (int64_t tile = 0; tile < 16; ++tile) {
    std::vector<wafer::compiler::TileEntryArgument> slots = {
        {0,
         wafer::compiler::TileEntryArgumentKind::TargetTensor,
         1,
         "bias",
         wafer::LogicalFormat::BF16,
         wafer::MemLayout::Tensor,
         {2},
         4,
         16,
         wafer::compiler::TileEntryArgumentAccess::ReadOnly},
        {1,
         wafer::compiler::TileEntryArgumentKind::TargetTensor,
         0,
         "weight",
         wafer::LogicalFormat::F32,
         wafer::MemLayout::Cx,
         {4},
         256,
         64,
         wafer::compiler::TileEntryArgumentAccess::ReadOnly},
        {2,
         wafer::compiler::TileEntryArgumentKind::TargetTensor,
         0,
         "weight",
         wafer::LogicalFormat::F32,
         wafer::MemLayout::Tensor,
         {4},
         16,
         64,
         wafer::compiler::TileEntryArgumentAccess::ReadOnly},
    };
    slots[0].targetTensorMaterialization =
        llvm::cantFail(wafer::TargetTensorMaterializationAction::create(
            wafer::LogicalFormat::F16, wafer::LogicalFormat::BF16,
            wafer::TargetConvertParameter::roundingMode(
                wafer::TargetRoundingMode::NearestEven)));
    for (size_t slot = 1; slot < slots.size(); ++slot)
      slots[slot].targetTensorMaterialization =
          llvm::cantFail(wafer::TargetTensorMaterializationAction::create(
              wafer::LogicalFormat::F32, wafer::LogicalFormat::F32,
              /*parameter=*/std::nullopt));
    slots.push_back(makeProfilerSlot(
        slots.size(), wafer::compiler::detail::ProfileCaptureKind::Count));
    auto context = std::make_unique<llvm::LLVMContext>();
    auto module =
        std::make_unique<llvm::Module>("representation-entry", *context);
    llvm::Type *i64 = llvm::Type::getInt64Ty(*context);
    llvm::SmallVector<llvm::Type *, 4> argumentTypes(slots.size(), i64);
    llvm::Function *entry = llvm::Function::Create(
        llvm::FunctionType::get(llvm::Type::getVoidTy(*context), argumentTypes,
                                /*isVarArg=*/false),
        llvm::GlobalValue::ExternalLinkage, "main", *module);
    llvm::IRBuilder<> builder(
        llvm::BasicBlock::Create(*context, "entry", entry));
    builder.CreateRetVoid();
    if (llvm::Error error =
            wafer::compiler::detail::instrumentProfileTargetModule(
                *module, "main",
                wafer::compiler::detail::ProfileCaptureKind::Count)) {
      ADD_FAILURE() << llvm::toString(std::move(error));
      return;
    }
    modules.push_back(wafer::compiler::TargetLLVMModulesBuilder::makeModule(
        wafer::CardId(0), wafer::TileId(tile), wafer::LaunchSlotId(tile),
        "main", wafer::kCurrentTargetIdentity, wafer::kCurrentKernelRuntimeABI,
        wafer::kCurrentTargetModuleFormat, std::move(slots), std::move(context),
        std::move(module)));
  }
  llvm::Expected<wafer::compiler::TargetLLVMModules> targetLLVM =
      wafer::compiler::TargetLLVMModulesBuilder::makeModules(
          *config, launch, std::move(modules));
  ASSERT_TRUE(static_cast<bool>(targetLLVM))
      << llvm::toString(targetLLVM.takeError());

  llvm::SmallString<256> linkDirectory = temporaryDirectory;
  llvm::sys::path::append(linkDirectory, "linked");
  std::string diagnosticsStorage;
  llvm::raw_string_ostream diagnostics(diagnosticsStorage);
  llvm::Expected<wafer::compiler::LinkedTargetModules> targetModules =
      wafer::compiler::detail::linkTargetLLVMModulesImpl(
          *targetLLVM, linkDirectory, *toolchain, diagnostics,
          wafer::compiler::detail::ProfileCaptureKind::Count);
  ASSERT_TRUE(static_cast<bool>(targetModules))
      << diagnosticsStorage << llvm::toString(targetModules.takeError());

  auto context = std::make_shared<mlir::MLIRContext>();
  std::vector<wafer::compiler::TileExecutable> tiles;
  tiles.reserve(16);
  for (int64_t tile = 0; tile < 16; ++tile) {
    mlir::OpBuilder builder(context.get());
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::ModuleOp::create(builder.getUnknownLoc());
    auto makeBinding = [](wafer::compiler::ProgramResourceRole role,
                          int64_t roleIndex, int64_t index,
                          wafer::ProgramElementType dtype,
                          llvm::ArrayRef<int64_t> shape) {
      wafer::compiler::ProgramResourceBinding binding{};
      binding.role = role;
      binding.programTensorId = {role, roleIndex};
      binding.index = index;
      binding.programIndex = index;
      binding.name = index == 0 ? "weight" : "bias";
      binding.dtype = dtype;
      binding.distribution =
          wafer::frontend::ProgramDistributionKind::Replicated;
      binding.globalShape.assign(shape.begin(), shape.end());
      binding.localShape.assign(shape.begin(), shape.end());
      binding.slice.partitionId = 0;
      binding.slice.replicaId = 0;
      binding.slice.offsets.assign(shape.size(), 0);
      binding.slice.sizes.assign(shape.begin(), shape.end());
      binding.slice.strides.assign(shape.size(), 1);
      return binding;
    };
    tiles.push_back(wafer::compiler::CardExecutableBuilder::makeTileExecutable(
        wafer::CardId(0), wafer::TileId(tile), wafer::LaunchSlotId(tile),
        std::move(module), "main",
        {makeBinding(wafer::compiler::ProgramResourceRole::Parameter, 0, 0,
                     wafer::ProgramElementType::F32, {4}),
         makeBinding(wafer::compiler::ProgramResourceRole::Parameter, 1, 1,
                     wafer::ProgramElementType::F16, {2})},
        wafer::compiler::TransportContract::None));
  }
  llvm::Expected<wafer::compiler::CardExecutable> executable =
      wafer::compiler::CardExecutableBuilder::makeCardExecutable(
          *config, launch, std::move(context), std::move(tiles),
          std::move(handoff));
  ASSERT_TRUE(static_cast<bool>(executable))
      << llvm::toString(executable.takeError());

  llvm::SmallString<256> packageDirectory = temporaryDirectory;
  llvm::sys::path::append(packageDirectory, "package");
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> package =
      wafer::compiler::detail::writePackage(temporaryDirectory, *executable,
                                            *targetModules, packageDirectory,
                                            diagnostics, std::nullopt);
  ASSERT_TRUE(static_cast<bool>(package))
      << diagnosticsStorage << llvm::toString(package.takeError());
  const wafer::runtime::PackageManifest &manifest = package->getManifest();

  // Program-tensor ids are discovery-ordered: bias (ordinal 0) is id 0 and
  // weight is id 1. The canonical placement therefore sorts
  // [bias-BF16-Tensor (id 0, offset 0, 4 bytes), weight-Tensor (id 1, offset
  // 64, 16 bytes), weight-Cx (id 2, offset 128, 256 bytes)].
  ASSERT_EQ(manifest.targetTensors.size(), 3u);
  EXPECT_EQ(manifest.targetTensors[0].programTensor.getValue(), 0u);
  EXPECT_EQ(manifest.targetTensors[0].layout,
            wafer::runtime::PackageMemLayout::Tensor);
  EXPECT_EQ(manifest.targetTensors[0].dtype, wafer::LogicalFormat::BF16);
  EXPECT_EQ(manifest.targetTensors[0].bytes, 4u);
  EXPECT_EQ(manifest.targetTensors[0].fileOffset, 0u);
  EXPECT_EQ(manifest.targetTensors[1].programTensor.getValue(), 1u);
  EXPECT_EQ(manifest.targetTensors[1].layout,
            wafer::runtime::PackageMemLayout::Tensor);
  EXPECT_EQ(manifest.targetTensors[1].bytes, 16u);
  EXPECT_EQ(manifest.targetTensors[1].fileOffset, 64u);
  EXPECT_EQ(manifest.targetTensors[2].programTensor.getValue(), 1u);
  EXPECT_EQ(manifest.targetTensors[2].layout,
            wafer::runtime::PackageMemLayout::Cx);
  EXPECT_EQ(manifest.targetTensors[2].bytes, 256u);
  EXPECT_EQ(manifest.targetTensors[2].fileOffset, 128u);
  EXPECT_EQ(manifest.programData.totalBytes, 384u);
  EXPECT_EQ(manifest.programData.baseAlignment, 64u);

  // Discovery order was [bias, Cx, Tensor]; every entry reference must
  // resolve to the canonical id.
  ASSERT_EQ(manifest.entries.size(), 16u);
  const uint64_t expectedReferences[3] = {0, 2, 1};
  for (const wafer::runtime::PackageEntrypointRecord &entry :
       manifest.entries) {
    ASSERT_EQ(entry.arguments.size(), 4u);
    for (size_t ordinal = 0; ordinal < 3; ++ordinal) {
      const auto *argument = std::get_if<wafer::runtime::TargetTensorArgument>(
          &entry.arguments[ordinal].reference);
      ASSERT_NE(argument, nullptr);
      EXPECT_EQ(argument->tensor.getValue(), expectedReferences[ordinal]);
    }
  }

  // Exact program-data bytes: converted bias BF16 at [0,4), weight payload at
  // [64,80), and blocked Cx weight at [128,144), with canonical zero padding.
  llvm::SmallString<256> programDataPath(packageDirectory);
  llvm::sys::path::append(programDataPath, "data", "program-data.bin");
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(programDataPath, /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false);
  ASSERT_TRUE(static_cast<bool>(buffer));
  llvm::StringRef content = (*buffer)->getBuffer();
  ASSERT_EQ(content.size(), 384u);
  const uint8_t expectedBiasBF16[] = {0x80, 0x3f, 0x00, 0x40};
  EXPECT_EQ(content.slice(0, 4),
            llvm::StringRef(reinterpret_cast<const char *>(expectedBiasBF16),
                            sizeof(expectedBiasBF16)));
  EXPECT_TRUE(content.slice(4, 64).find_first_not_of('\0') ==
              llvm::StringRef::npos);
  EXPECT_EQ(
      content.slice(64, 80),
      llvm::StringRef(reinterpret_cast<const char *>(weightPayload.data()),
                      weightPayload.size()));
  EXPECT_TRUE(content.slice(80, 128).find_first_not_of('\0') ==
              llvm::StringRef::npos);
  EXPECT_EQ(
      content.slice(128, 144),
      llvm::StringRef(reinterpret_cast<const char *>(weightPayload.data()),
                      weightPayload.size()));
  EXPECT_TRUE(content.slice(144, 384).find_first_not_of('\0') ==
              llvm::StringRef::npos);
  EXPECT_EQ(executable->getProgramDataHandoff()
                .getIOStatistics()
                .rangeMaterializations,
            3u);
}

TEST(TargetCodeGenTest, PackageSlotLegalityIgnoresDiagnosticNames) {
  wafer::compiler::ProgramResourceBinding binding{};
  binding.role = wafer::compiler::ProgramResourceRole::UserInput;
  binding.programTensorId = {binding.role, 0};
  binding.index = 0;
  binding.name = "frontend_name";
  binding.dtype = wafer::ProgramElementType::F32;
  binding.localShape = {4};
  wafer::compiler::TileEntryArgument userInput =
      makeSlot(0, wafer::compiler::TileEntryArgumentKind::ExternalInput,
               "lowered_diagnostic_alias");
  EXPECT_TRUE(wafer::compiler::detail::doesPackageSlotMatchProgramBinding(
      userInput, binding));
  binding.dtype = wafer::ProgramElementType::F16;
  EXPECT_TRUE(wafer::compiler::detail::doesPackageSlotMatchProgramBinding(
      userInput, binding));

  wafer::compiler::TileEntryArgument workspace{
      1,
      wafer::compiler::TileEntryArgumentKind::Workspace,
      0,
      "renamed_workspace",
      wafer::LogicalFormat::U8,
      wafer::MemLayout::Tensor,
      {64},
      64,
      64,
      wafer::compiler::TileEntryArgumentAccess::ReadWrite};
  EXPECT_TRUE(
      wafer::compiler::detail::isValidPackageCompilerManagedSlot(workspace));
  workspace.name = "another_workspace_label";
  EXPECT_TRUE(
      wafer::compiler::detail::isValidPackageCompilerManagedSlot(workspace));
  wafer::compiler::TileEntryArgument profileWorkspace{
      2,
      wafer::compiler::TileEntryArgumentKind::ProfileRecord,
      0,
      "diagnostic_profile_name",
      wafer::LogicalFormat::U8,
      wafer::MemLayout::Tensor,
      {WAFER_TX81_PROFILER_MIN_BUFFER_BYTES},
      WAFER_TX81_PROFILER_MIN_BUFFER_BYTES,
      WAFER_TX81_PROFILER_BUFFER_ALIGNMENT,
      wafer::compiler::TileEntryArgumentAccess::ReadWrite};
  EXPECT_TRUE(wafer::compiler::detail::isValidPackageCompilerManagedSlot(
      profileWorkspace));
  profileWorkspace.name = "another_diagnostic_profile_name";
  EXPECT_TRUE(wafer::compiler::detail::isValidPackageCompilerManagedSlot(
      profileWorkspace));
  std::vector<wafer::compiler::TileEntryArgument> captureSlots = {
      userInput, workspace, profileWorkspace};
  EXPECT_FALSE(static_cast<bool>(
      wafer::compiler::detail::verifyProfileCaptureTileEntryArguments(
          captureSlots, wafer::compiler::detail::ProfileCaptureKind::Count)));
  captureSlots.back().name = "renamed_without_changing_typed_identity";
  EXPECT_FALSE(static_cast<bool>(
      wafer::compiler::detail::verifyProfileCaptureTileEntryArguments(
          captureSlots, wafer::compiler::detail::ProfileCaptureKind::Count)));
  profileWorkspace.byteSize += 1;
  profileWorkspace.shape = {profileWorkspace.byteSize};
  EXPECT_FALSE(wafer::compiler::detail::isValidPackageCompilerManagedSlot(
      profileWorkspace));

  wafer::compiler::TileEntryArgument status =
      makeSlot(2, wafer::compiler::TileEntryArgumentKind::TransportStatus,
               "renamed_status");
  EXPECT_TRUE(
      wafer::compiler::detail::isValidPackageCompilerManagedSlot(status));
  status.alignment = 8;
  EXPECT_FALSE(
      wafer::compiler::detail::isValidPackageCompilerManagedSlot(status));
}

TEST(TargetCodeGenTest, CurrentEngineRegistryIncludesDirectDTEIssueAndWait) {
  llvm::ArrayRef<wafer::TargetCallDescriptor> descriptors =
      wafer::getTargetCallDescriptors();
  EXPECT_EQ(llvm::count_if(
                descriptors,
                [](const auto &descriptor) {
                  return wafer::getTargetCallTSMEngine(descriptor).has_value();
                }),
            106);
  auto siteKindCount = [&](wafer::runtime::ProfileTargetSiteKind kind) {
    return llvm::count_if(descriptors, [&](const auto &descriptor) {
      return wafer::runtime::getProfileTargetSiteKind(descriptor) == kind;
    });
  };
  EXPECT_EQ(siteKindCount(wafer::runtime::ProfileTargetSiteKind::NCCCommand),
            104);
  EXPECT_EQ(siteKindCount(wafer::runtime::ProfileTargetSiteKind::NCCCompletion),
            1);
  EXPECT_EQ(
      siteKindCount(wafer::runtime::ProfileTargetSiteKind::DirectDTEControl),
      5);
  EXPECT_EQ(
      siteKindCount(wafer::runtime::ProfileTargetSiteKind::DirectDTEIssue), 1);
  EXPECT_EQ(siteKindCount(wafer::runtime::ProfileTargetSiteKind::DirectDTEWait),
            1);
  EXPECT_EQ(descriptors.size(), 112u);
  EXPECT_EQ(wafer::getTargetCallTSMEngine(
                getTargetCallDescriptor(wafer::TargetCallBuiltin::RDMA)),
            wafer::TargetCallTSMEngine::RDMA);
  EXPECT_EQ(wafer::getTargetCallTSMEngine(
                getTargetCallDescriptor(wafer::TargetCallBuiltin::WDMA)),
            wafer::TargetCallTSMEngine::WDMA);
  EXPECT_EQ(wafer::getTargetCallTSMEngine(getTargetCallDescriptor(
                wafer::TargetCallBuiltin::GatherScatter)),
            wafer::TargetCallTSMEngine::TDMA);
  EXPECT_EQ(wafer::getTargetCallTSMEngine(
                getTargetCallDescriptor(wafer::TargetCallBuiltin::Gemm)),
            wafer::TargetCallTSMEngine::NE);
  EXPECT_EQ(wafer::getTargetCallTSMEngine(
                getTargetCallDescriptor(wafer::TargetCallBuiltin::Bit2FP)),
            wafer::TargetCallTSMEngine::CT);
  EXPECT_FALSE(wafer::getTargetCallTSMEngine(
      getTargetCallDescriptor(wafer::TargetCallBuiltin::NCCJoin)));
  EXPECT_EQ(wafer::getTargetCallTSMEngine(getTargetCallDescriptor(
                wafer::TargetCallBuiltin::DirectDTESendIssue)),
            wafer::TargetCallTSMEngine::DirectDTE);
  EXPECT_EQ(wafer::getTargetCallTSMEngine(getTargetCallDescriptor(
                wafer::TargetCallBuiltin::DirectDTEWait)),
            wafer::TargetCallTSMEngine::DirectDTE);
}

TEST(TargetCodeGenTest,
     CountAndTraceInstrumentationUseTheSameDenseTypedTargetSiteCollector) {
  llvm::LLVMContext context;
  llvm::Module module("profile-target", context);
  llvm::Type *voidType = llvm::Type::getVoidTy(context);
  llvm::Type *i64 = llvm::Type::getInt64Ty(context);
  llvm::Function *entry = llvm::Function::Create(
      llvm::FunctionType::get(voidType, {i64, i64}, /*isVarArg=*/false),
      llvm::GlobalValue::ExternalLinkage, "main", module);
  llvm::IRBuilder<> builder(llvm::BasicBlock::Create(context, "entry", entry));
  llvm::Value *dynamicA = builder.CreateAdd(
      entry->getArg(0), llvm::ConstantInt::get(i64, 1), "dynamic.a");
  builder.CreateAdd(entry->getArg(0), llvm::ConstantInt::get(i64, 2),
                    "dynamic.b");
  const wafer::TargetCallDescriptor &directDTEWait =
      getTargetCallDescriptor(wafer::TargetCallBuiltin::DirectDTEWait);

  auto emitTargetCall = [&](const wafer::TargetCallDescriptor &descriptor) {
    llvm::SmallVector<llvm::Type *, 32> types;
    llvm::SmallVector<llvm::Value *, 32> arguments;
    for (auto [argumentIndex, scalar] : llvm::enumerate(descriptor.arguments)) {
      unsigned width = scalar == wafer::TargetCallScalarType::I64 ? 64 : 32;
      llvm::Type *type = llvm::IntegerType::get(context, width);
      types.push_back(type);
      if (descriptor.symbol == directDTEWait.symbol && argumentIndex == 0)
        arguments.push_back(dynamicA);
      else
        arguments.push_back(llvm::ConstantInt::get(type, arguments.size() + 1));
    }
    llvm::Type *resultType =
        descriptor.result == wafer::TargetCallResultType::I64 ? i64 : voidType;
    llvm::FunctionType *type =
        llvm::FunctionType::get(resultType, types, /*isVarArg=*/false);
    llvm::FunctionCallee callee =
        module.getOrInsertFunction(descriptor.symbol, type);
    builder.CreateCall(callee, arguments);
  };
  emitTargetCall(getTargetCallDescriptor(wafer::TargetCallBuiltin::NCCJoin));
  emitTargetCall(
      getTargetCallDescriptor(wafer::TargetCallBuiltin::DirectDTEBegin));
  emitTargetCall(getTargetCallDescriptor(
      wafer::TargetCallBuiltin::DirectDTEBeginAfterPrepare));
  emitTargetCall(
      getTargetCallDescriptor(wafer::TargetCallBuiltin::DirectDTESendPrepare));
  emitTargetCall(
      getTargetCallDescriptor(wafer::TargetCallBuiltin::DirectDTERecvPrepare));
  emitTargetCall(getTargetCallDescriptor(wafer::TargetCallBuiltin::Bit2FP));
  emitTargetCall(getTargetCallDescriptor(wafer::TargetCallBuiltin::RDMA));
  emitTargetCall(
      getTargetCallDescriptor(wafer::TargetCallBuiltin::DirectDTESendIssue));
  emitTargetCall(directDTEWait);
  emitTargetCall(getTargetCallDescriptor(wafer::TargetCallBuiltin::Gemm));
  emitTargetCall(getTargetCallDescriptor(wafer::TargetCallBuiltin::WDMA));
  emitTargetCall(
      getTargetCallDescriptor(wafer::TargetCallBuiltin::GatherScatter));
  emitTargetCall(
      getTargetCallDescriptor(wafer::TargetCallBuiltin::DirectDTEFinish));
  builder.CreateRetVoid();

  llvm::Function *dead = llvm::Function::Create(
      llvm::FunctionType::get(voidType, {}, /*isVarArg=*/false),
      llvm::GlobalValue::InternalLinkage, "dead", module);
  llvm::IRBuilder<> deadBuilder(
      llvm::BasicBlock::Create(context, "entry", dead));
  const wafer::TargetCallDescriptor &wdma =
      getTargetCallDescriptor(wafer::TargetCallBuiltin::WDMA);
  llvm::SmallVector<llvm::Type *, 16> wdmaTypes;
  llvm::SmallVector<llvm::Value *, 16> wdmaArguments;
  for (wafer::TargetCallScalarType scalar : wdma.arguments) {
    unsigned width = scalar == wafer::TargetCallScalarType::I64 ? 64 : 32;
    llvm::Type *type = llvm::IntegerType::get(context, width);
    wdmaTypes.push_back(type);
    wdmaArguments.push_back(llvm::ConstantInt::get(type, 0));
  }
  deadBuilder.CreateCall(
      module.getOrInsertFunction(
          wdma.symbol,
          llvm::FunctionType::get(voidType, wdmaTypes, /*isVarArg=*/false)),
      wdmaArguments);
  deadBuilder.CreateRetVoid();

  auto before =
      wafer::compiler::detail::collectProfileTargetCallSites(module, "main");
  ASSERT_TRUE(static_cast<bool>(before)) << llvm::toString(before.takeError());
  std::unique_ptr<llvm::Module> productionModule = llvm::CloneModule(module);
  ASSERT_EQ(before->size(), 13u);
  EXPECT_EQ((*before)[0].siteId, 0u);
  EXPECT_EQ((*before)[0].siteKind,
            wafer::runtime::ProfileTargetSiteKind::NCCCompletion);
  EXPECT_FALSE((*before)[0].engine);
  EXPECT_EQ((*before)[1].siteId, 1u);
  EXPECT_EQ((*before)[1].siteKind,
            wafer::runtime::ProfileTargetSiteKind::DirectDTEControl);
  EXPECT_FALSE((*before)[1].engine);
  EXPECT_EQ((*before)[2].siteId, 2u);
  EXPECT_EQ((*before)[2].siteKind,
            wafer::runtime::ProfileTargetSiteKind::DirectDTEControl);
  EXPECT_FALSE((*before)[2].engine);
  EXPECT_EQ((*before)[3].siteId, 3u);
  EXPECT_EQ((*before)[3].siteKind,
            wafer::runtime::ProfileTargetSiteKind::DirectDTEControl);
  EXPECT_FALSE((*before)[3].engine);
  EXPECT_EQ((*before)[4].siteId, 4u);
  EXPECT_EQ((*before)[4].siteKind,
            wafer::runtime::ProfileTargetSiteKind::DirectDTEControl);
  EXPECT_FALSE((*before)[4].engine);
  EXPECT_EQ((*before)[5].siteId, 5u);
  EXPECT_EQ((*before)[5].siteKind,
            wafer::runtime::ProfileTargetSiteKind::NCCCommand);
  EXPECT_EQ((*before)[5].engine, wafer::TargetCallTSMEngine::CT);
  EXPECT_EQ((*before)[6].engine, wafer::TargetCallTSMEngine::RDMA);
  EXPECT_EQ((*before)[7].siteKind,
            wafer::runtime::ProfileTargetSiteKind::DirectDTEIssue);
  EXPECT_EQ((*before)[7].engine, wafer::TargetCallTSMEngine::DirectDTE);
  EXPECT_EQ((*before)[8].siteKind,
            wafer::runtime::ProfileTargetSiteKind::DirectDTEWait);
  EXPECT_EQ((*before)[8].engine, wafer::TargetCallTSMEngine::DirectDTE);
  EXPECT_EQ((*before)[9].engine, wafer::TargetCallTSMEngine::NE);
  EXPECT_EQ((*before)[10].engine, wafer::TargetCallTSMEngine::WDMA);
  EXPECT_EQ((*before)[11].engine, wafer::TargetCallTSMEngine::TDMA);
  EXPECT_EQ((*before)[12].siteKind,
            wafer::runtime::ProfileTargetSiteKind::DirectDTEControl);
  EXPECT_FALSE((*before)[12].engine);
  std::vector<std::string> correlationKeys;
  for (const auto &site : *before)
    correlationKeys.push_back(site.correlationKey);

  std::unique_ptr<llvm::Module> countModule =
      llvm::CloneModule(*productionModule);
  if (llvm::Error error =
          wafer::compiler::detail::instrumentProfileTargetModule(
              *countModule, "main",
              wafer::compiler::detail::ProfileCaptureKind::Count))
    FAIL() << llvm::toString(std::move(error));
  EXPECT_FALSE(static_cast<bool>(
      wafer::compiler::detail::verifyProfileTargetModuleInstrumentation(
          *countModule, "main",
          wafer::compiler::detail::ProfileCaptureKind::Count)));
  EXPECT_FALSE(static_cast<bool>(
      wafer::compiler::detail::verifyProfileTargetCallSitesMatch(
          *productionModule, "main", *countModule, "main")));
  llvm::Function *countSiteBegin =
      countModule->getFunction("wafer_tx81_profile_site_begin");
  llvm::Function *countSiteEnd =
      countModule->getFunction("wafer_tx81_profile_site_end");
  ASSERT_NE(countSiteBegin, nullptr);
  ASSERT_NE(countSiteEnd, nullptr);
  EXPECT_EQ(countSiteBegin->getNumUses(), before->size());
  EXPECT_EQ(countSiteEnd->getNumUses(), before->size());

  if (llvm::Error error =
          wafer::compiler::detail::instrumentProfileTargetModule(
              module, "main",
              wafer::compiler::detail::ProfileCaptureKind::Trace))
    FAIL() << llvm::toString(std::move(error));
  EXPECT_FALSE(static_cast<bool>(
      wafer::compiler::detail::verifyProfileTargetModuleInstrumentation(
          module, "main", wafer::compiler::detail::ProfileCaptureKind::Trace)));
  EXPECT_FALSE(static_cast<bool>(
      wafer::compiler::detail::verifyProfileTargetCallSitesMatch(
          *productionModule, "main", module, "main")));
  auto after =
      wafer::compiler::detail::collectProfileTargetCallSites(module, "main");
  ASSERT_TRUE(static_cast<bool>(after)) << llvm::toString(after.takeError());
  ASSERT_EQ(after->size(), 13u);
  for (auto [index, site] : llvm::enumerate(*after)) {
    EXPECT_EQ(site.correlationKey, correlationKeys[index]);
    EXPECT_NE(site.instructionOrdinal, (*before)[index].instructionOrdinal);
  }

  std::unique_ptr<llvm::Module> driftedTrace = llvm::CloneModule(module);
  llvm::Function *bit2FP = driftedTrace->getFunction(
      getTargetCallDescriptor(wafer::TargetCallBuiltin::Bit2FP).symbol);
  ASSERT_NE(bit2FP, nullptr);
  ASSERT_FALSE(bit2FP->user_empty());
  auto *driftedCall = llvm::dyn_cast<llvm::CallBase>(*bit2FP->user_begin());
  ASSERT_NE(driftedCall, nullptr);
  ASSERT_FALSE(driftedCall->arg_empty());
  auto *constant =
      llvm::dyn_cast<llvm::ConstantInt>(driftedCall->getArgOperand(0));
  ASSERT_NE(constant, nullptr);
  driftedCall->setArgOperand(
      0, llvm::ConstantInt::get(constant->getType(),
                                constant->getZExtValue() + 1));
  llvm::Error drift =
      wafer::compiler::detail::verifyProfileTargetCallSitesMatch(
          *productionModule, "main", *driftedTrace, "main");
  ASSERT_TRUE(static_cast<bool>(drift));
  EXPECT_NE(
      llvm::toString(std::move(drift)).find("differs from final production"),
      std::string::npos);

  std::unique_ptr<llvm::Module> dynamicallyDriftedTrace =
      llvm::CloneModule(module);
  llvm::Function *driftedWait =
      dynamicallyDriftedTrace->getFunction(directDTEWait.symbol);
  ASSERT_NE(driftedWait, nullptr);
  ASSERT_FALSE(driftedWait->user_empty());
  auto *driftedWaitCall =
      llvm::dyn_cast<llvm::CallBase>(*driftedWait->user_begin());
  ASSERT_NE(driftedWaitCall, nullptr);
  llvm::Function *driftedEntry = dynamicallyDriftedTrace->getFunction("main");
  ASSERT_NE(driftedEntry, nullptr);
  llvm::Instruction *replacement = nullptr;
  for (llvm::BasicBlock &block : *driftedEntry)
    for (llvm::Instruction &instruction : block)
      if (instruction.getName() == "dynamic.b")
        replacement = &instruction;
  ASSERT_NE(replacement, nullptr);
  driftedWaitCall->setArgOperand(0, replacement);
  llvm::Error dynamicDrift =
      wafer::compiler::detail::verifyProfileTargetCallSitesMatch(
          *productionModule, "main", *dynamicallyDriftedTrace, "main");
  ASSERT_TRUE(static_cast<bool>(dynamicDrift));
  EXPECT_NE(llvm::toString(std::move(dynamicDrift))
                .find("differs from final production"),
            std::string::npos);

  std::string text;
  llvm::raw_string_ostream output(text);
  module.print(output, nullptr);
  output.flush();
  EXPECT_NE(text.find("call void @wafer_tx81_profile_entry_begin_from_config("
                      "i64 %1)"),
            std::string::npos);
  EXPECT_NE(text.find("call void @wafer_tx81_profile_site_begin(i32 0)"),
            std::string::npos);
  EXPECT_NE(text.find("call void @wafer_tx81_profile_site_end(i32 12)"),
            std::string::npos);
  EXPECT_NE(text.find("call void @wafer_tx81_profile_entry_end()"),
            std::string::npos);

  llvm::Function *siteEnd = module.getFunction("wafer_tx81_profile_site_end");
  ASSERT_NE(siteEnd, nullptr);
  ASSERT_FALSE(siteEnd->user_empty());
  auto *missingEnd = llvm::dyn_cast<llvm::CallBase>(*siteEnd->user_begin());
  ASSERT_NE(missingEnd, nullptr);
  missingEnd->eraseFromParent();
  llvm::Error missingSiteError =
      wafer::compiler::detail::verifyProfileTargetModuleInstrumentation(
          module, "main", wafer::compiler::detail::ProfileCaptureKind::Trace);
  ASSERT_TRUE(static_cast<bool>(missingSiteError));
  EXPECT_NE(llvm::toString(std::move(missingSiteError))
                .find("does not cover the exact typed target-call site set"),
            std::string::npos);
}

TEST(TargetCodeGenTest,
     ProfileInstrumentationVerifierRejectsDeclarationsWithoutStructure) {
  llvm::LLVMContext context;
  llvm::Module module("profile-declarations-only", context);
  llvm::Type *voidType = llvm::Type::getVoidTy(context);
  llvm::Type *i64 = llvm::Type::getInt64Ty(context);
  llvm::Function *entry = llvm::Function::Create(
      llvm::FunctionType::get(voidType, {i64}, /*isVarArg=*/false),
      llvm::GlobalValue::ExternalLinkage, "main", module);
  llvm::IRBuilder<> builder(llvm::BasicBlock::Create(context, "entry", entry));
  builder.CreateRetVoid();
  llvm::Function::Create(
      llvm::FunctionType::get(voidType, {i64}, /*isVarArg=*/false),
      llvm::GlobalValue::ExternalLinkage,
      "wafer_tx81_profile_entry_begin_from_config", module);
  llvm::Function::Create(
      llvm::FunctionType::get(voidType, {}, /*isVarArg=*/false),
      llvm::GlobalValue::ExternalLinkage, "wafer_tx81_profile_entry_end",
      module);

  llvm::Error error =
      wafer::compiler::detail::verifyProfileTargetModuleInstrumentation(
          module, "main", wafer::compiler::detail::ProfileCaptureKind::Count);
  ASSERT_TRUE(static_cast<bool>(error));
  EXPECT_NE(llvm::toString(std::move(error))
                .find("must begin with exactly one record binding call"),
            std::string::npos);
}

TEST(TargetCodeGenTest, MultiTileLaunchValidationRejectsMismatchedSchemas) {
  llvm::Expected<wafer::compiler::TargetLLVMModules> mismatchedSchema =
      makeRuntimeLaunchModules(makeKernelLaunch(wafer::KernelLaunchForm::Grid),
                               wafer::compiler::TransportContract::None,
                               /*mismatchedSchemaTile=*/7);
  ASSERT_TRUE(static_cast<bool>(mismatchedSchema))
      << llvm::toString(mismatchedSchema.takeError());
  llvm::Error schemaError =
      wafer::compiler::detail::validateRuntimeLaunchContractDomainForTesting(
          *mismatchedSchema);
  ASSERT_TRUE(static_cast<bool>(schemaError));
  EXPECT_NE(llvm::toString(std::move(schemaError))
                .find("compatible per-Tile slot order"),
            std::string::npos);
}

TEST(TargetCodeGenTest,
     MultiTileLaunchValidationAcceptsTileLocalWorkspaceCapacity) {
  llvm::Expected<wafer::compiler::TargetLLVMModules> modules =
      makeRuntimeLaunchModules(makeKernelLaunch(wafer::KernelLaunchForm::Grid),
                               wafer::compiler::TransportContract::None,
                               std::nullopt, std::nullopt,
                               /*slotCount=*/2,
                               /*withCollidingClosure=*/false,
                               /*withUnsupportedInlineAsm=*/false, std::nullopt,
                               /*permuteTiles=*/false,
                               /*variableWorkspace=*/true);
  ASSERT_TRUE(static_cast<bool>(modules))
      << llvm::toString(modules.takeError());
  EXPECT_FALSE(static_cast<bool>(
      wafer::compiler::detail::validateRuntimeLaunchContractDomainForTesting(
          *modules)));
}

TEST(TargetCodeGenTest,
     RuntimeLaunchContractAndTransportSlotsRemainIndependentRecords) {
  llvm::Expected<wafer::compiler::TargetLLVMModules> gridWithTransport =
      makeRuntimeLaunchModules(makeKernelLaunch(wafer::KernelLaunchForm::Grid),
                               wafer::compiler::TransportContract::DirectDTE);
  ASSERT_TRUE(static_cast<bool>(gridWithTransport))
      << llvm::toString(gridWithTransport.takeError());
  EXPECT_FALSE(static_cast<bool>(
      wafer::compiler::detail::validateRuntimeLaunchContractDomainForTesting(
          *gridWithTransport)));
  EXPECT_EQ(
      llvm::count_if(
          gridWithTransport->getModules().front().getTileEntryArguments(),
          [](const wafer::compiler::TileEntryArgument &slot) {
            return slot.kind ==
                   wafer::compiler::TileEntryArgumentKind::TransportStatus;
          }),
      1);

  llvm::Expected<wafer::compiler::TargetLLVMModules> clusterWithoutTransport =
      makeRuntimeLaunchModules(
          makeKernelLaunch(wafer::KernelLaunchForm::Cluster),
          wafer::compiler::TransportContract::None);
  ASSERT_TRUE(static_cast<bool>(clusterWithoutTransport))
      << llvm::toString(clusterWithoutTransport.takeError());
  EXPECT_FALSE(static_cast<bool>(
      wafer::compiler::detail::validateRuntimeLaunchContractDomainForTesting(
          *clusterWithoutTransport)));
  EXPECT_EQ(
      llvm::count_if(
          clusterWithoutTransport->getModules().front().getTileEntryArguments(),
          [](const wafer::compiler::TileEntryArgument &slot) {
            return slot.kind ==
                   wafer::compiler::TileEntryArgumentKind::TransportStatus;
          }),
      0);
}

TEST(TargetCodeGenTest, KernelArgumentPacketLimitIsCheckedBeforeDeviceLink) {
  EXPECT_EQ(wafer::kTx81ClusterKernelArgumentBytesMax, 0x7d0u);

  auto verifySlotCount = [](wafer::RuntimeLaunchContract launch,
                            size_t slotCount) {
    llvm::Expected<wafer::compiler::TargetLLVMModules> targetLLVMModules =
        makeRuntimeLaunchModules(std::move(launch),
                                 wafer::compiler::TransportContract::None,
                                 std::nullopt, std::nullopt, slotCount);
    EXPECT_TRUE(static_cast<bool>(targetLLVMModules));
    if (!targetLLVMModules)
      return targetLLVMModules.takeError();
    return wafer::compiler::detail::
        validateRuntimeLaunchContractDomainForTesting(*targetLLVMModules);
  };

  llvm::Error rejectedEmptyGrid =
      verifySlotCount(makeKernelLaunch(wafer::KernelLaunchForm::Grid), 0);
  ASSERT_TRUE(static_cast<bool>(rejectedEmptyGrid));
  EXPECT_NE(llvm::toString(std::move(rejectedEmptyGrid))
                .find("at least one typed ABI slot"),
            std::string::npos);

  llvm::Error allowedGrid =
      verifySlotCount(makeKernelLaunch(wafer::KernelLaunchForm::Grid), 15);
  EXPECT_FALSE(static_cast<bool>(allowedGrid));
  llvm::Error rejectedGrid =
      verifySlotCount(makeKernelLaunch(wafer::KernelLaunchForm::Grid), 16);
  ASSERT_TRUE(static_cast<bool>(rejectedGrid));
  EXPECT_NE(llvm::toString(std::move(rejectedGrid)).find("packet limit"),
            std::string::npos);

  llvm::Error allowedCluster =
      verifySlotCount(makeKernelLaunch(wafer::KernelLaunchForm::Cluster), 15);
  EXPECT_FALSE(static_cast<bool>(allowedCluster));
  llvm::Error rejectedCluster =
      verifySlotCount(makeKernelLaunch(wafer::KernelLaunchForm::Cluster), 16);
  ASSERT_TRUE(static_cast<bool>(rejectedCluster));
  EXPECT_NE(llvm::toString(std::move(rejectedCluster)).find("packet limit"),
            std::string::npos);

  llvm::Error allowedGridRows = verifySlotCount(
      makeKernelLaunch(wafer::KernelLaunchForm::Grid,
                       wafer::KernelEntryABI::TileRowPointerTable),
      18);
  EXPECT_FALSE(static_cast<bool>(allowedGridRows));
  llvm::Error allowedClusterRows = verifySlotCount(
      makeKernelLaunch(wafer::KernelLaunchForm::Cluster,
                       wafer::KernelEntryABI::TileRowPointerTable),
      18);
  EXPECT_FALSE(static_cast<bool>(allowedClusterRows));
}

TEST(TargetCodeGenTest,
     KernelAggregationLoadsSelectedTileRowBeforeTypedSlotDispatch) {
  llvm::Expected<wafer::compiler::TargetLLVMModules> targetLLVMModules =
      makeRuntimeLaunchModules(
          makeKernelLaunch(wafer::KernelLaunchForm::Grid,
                           wafer::KernelEntryABI::TileRowPointerTable),
          wafer::compiler::TransportContract::None, std::nullopt, std::nullopt,
          /*slotCount=*/18);
  ASSERT_TRUE(static_cast<bool>(targetLLVMModules))
      << llvm::toString(targetLLVMModules.takeError());

  llvm::Expected<wafer::compiler::detail::OwnedTargetLLVMModule> aggregate =
      wafer::compiler::detail::buildKernelAggregateTargetModule(
          *targetLLVMModules);
  ASSERT_TRUE(static_cast<bool>(aggregate))
      << llvm::toString(aggregate.takeError());

  std::string ir;
  llvm::raw_string_ostream output(ir);
  aggregate->module->print(output, nullptr);
  output.flush();
  EXPECT_NE(ir.find("getelementptr inbounds i64, ptr %tile_row_pointers, i64 "
                    "15"),
            std::string::npos);
  EXPECT_NE(ir.find("%launch_slot.15.row = load i64"), std::string::npos);
  EXPECT_NE(
      ir.find(
          "%launch_slot.15.slots = inttoptr i64 %launch_slot.15.row to ptr"),
      std::string::npos);
  EXPECT_NE(
      ir.find("getelementptr inbounds i64, ptr %launch_slot.15.slots, i64 17"),
      std::string::npos);
}

TEST(TargetCodeGenTest, KernelAggregationDispatchesTileToExplicitLaunchSlot) {
  llvm::Expected<wafer::compiler::TargetLLVMModules> targetLLVMModules =
      makeRuntimeLaunchModules(
          makeKernelLaunch(wafer::KernelLaunchForm::Grid,
                           wafer::KernelEntryABI::TileRowPointerTable),
          wafer::compiler::TransportContract::None, std::nullopt, std::nullopt,
          /*slotCount=*/2,
          /*withCollidingClosure=*/false,
          /*withUnsupportedInlineAsm=*/false, /*renamedSlotTile=*/std::nullopt,
          /*permuteTiles=*/true);
  ASSERT_TRUE(static_cast<bool>(targetLLVMModules))
      << llvm::toString(targetLLVMModules.takeError());

  llvm::Expected<wafer::compiler::detail::OwnedTargetLLVMModule> aggregate =
      wafer::compiler::detail::buildKernelAggregateTargetModule(
          *targetLLVMModules);
  ASSERT_TRUE(static_cast<bool>(aggregate))
      << llvm::toString(aggregate.takeError());
  std::string ir;
  llvm::raw_string_ostream output(ir);
  aggregate->module->print(output, nullptr);
  output.flush();

  EXPECT_NE(ir.find("i32 1, label %tile.1.launch_slot.0"), std::string::npos);
  EXPECT_NE(ir.find("i32 0, label %tile.0.launch_slot.1"), std::string::npos);
  EXPECT_NE(ir.find("%launch_slot.0.row.address = getelementptr inbounds i64, "
                    "ptr %tile_row_pointers, i64 0"),
            std::string::npos);
  EXPECT_NE(ir.find("%launch_slot.1.row.address = getelementptr inbounds i64, "
                    "ptr %tile_row_pointers, i64 1"),
            std::string::npos);
  EXPECT_NE(ir.find("call void @__wafer_kernel_launch_slot_00000_main_body"),
            std::string::npos);
  EXPECT_NE(ir.find("call void @__wafer_kernel_launch_slot_00001_main_body"),
            std::string::npos);
}

TEST(TargetCodeGenTest,
     KernelAggregationImportsAllTilesAndBuildsTypedExportsDeterministically) {
  llvm::Expected<wafer::compiler::TargetLLVMModules> targetLLVMModules =
      makeRuntimeLaunchModules(
          makeKernelLaunch(wafer::KernelLaunchForm::Cluster),
          wafer::compiler::TransportContract::DirectDTE, std::nullopt,
          std::nullopt, /*slotCount=*/2, /*withCollidingClosure=*/true,
          /*withUnsupportedInlineAsm=*/false,
          /*renamedSlotTile=*/7);
  ASSERT_TRUE(static_cast<bool>(targetLLVMModules))
      << llvm::toString(targetLLVMModules.takeError());

  llvm::Expected<wafer::compiler::detail::OwnedTargetLLVMModule> first =
      wafer::compiler::detail::buildKernelAggregateTargetModule(
          *targetLLVMModules);
  ASSERT_TRUE(static_cast<bool>(first)) << llvm::toString(first.takeError());
  llvm::Expected<wafer::compiler::detail::OwnedTargetLLVMModule> second =
      wafer::compiler::detail::buildKernelAggregateTargetModule(
          *targetLLVMModules);
  ASSERT_TRUE(static_cast<bool>(second)) << llvm::toString(second.takeError());

  std::string firstIR;
  llvm::raw_string_ostream firstOutput(firstIR);
  first->module->print(firstOutput, nullptr);
  firstOutput.flush();
  std::string secondIR;
  llvm::raw_string_ostream secondOutput(secondIR);
  second->module->print(secondOutput, nullptr);
  secondOutput.flush();
  EXPECT_EQ(firstIR, secondIR);

  llvm::Function *prepare = first->module->getFunction(
      wafer::compiler::detail::kKernelPrepareExportSymbol);
  ASSERT_NE(prepare, nullptr);
  EXPECT_TRUE(prepare->hasExternalLinkage());
  unsigned prepareCallCount = 0;
  bool sawGetPid = false;
  bool sawInitTile = false;
  bool sawSyncInit = false;
  for (llvm::BasicBlock &block : *prepare)
    for (llvm::Instruction &instruction : block)
      if (auto *call = llvm::dyn_cast<llvm::CallBase>(&instruction)) {
        ++prepareCallCount;
        ASSERT_NE(call->getCalledFunction(), nullptr);
        llvm::StringRef callee = call->getCalledFunction()->getName();
        if (callee == "__get_pid") {
          sawGetPid = true;
          ASSERT_EQ(call->arg_size(), 1u);
          auto *dimension =
              llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(0));
          ASSERT_NE(dimension, nullptr);
          EXPECT_EQ(dimension->getZExtValue(), 0u);
        } else if (callee == "init_tile_id") {
          sawInitTile = true;
          ASSERT_EQ(call->arg_size(), 2u);
          EXPECT_EQ(call->getArgOperand(0)->getName(), "pid.x");
          auto *rowLength =
              llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(1));
          ASSERT_NE(rowLength, nullptr);
          EXPECT_EQ(rowLength->getZExtValue(), 4u);
        } else if (callee == "direct_sync_init") {
          sawSyncInit = true;
          ASSERT_EQ(call->arg_size(), 1u);
          auto *participantCount =
              llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(0));
          ASSERT_NE(participantCount, nullptr);
          EXPECT_EQ(participantCount->getZExtValue(), 16u);
        } else {
          ADD_FAILURE() << "unexpected cluster prepare call: " << callee.str();
        }
      }
  EXPECT_EQ(prepareCallCount, 3u);
  EXPECT_TRUE(sawGetPid);
  EXPECT_TRUE(sawInitTile);
  EXPECT_TRUE(sawSyncInit);

  llvm::Function *main = first->module->getFunction("main");
  ASSERT_NE(main, nullptr);
  EXPECT_TRUE(main->hasExternalLinkage());
  const llvm::SwitchInst *dispatch = nullptr;
  for (const llvm::BasicBlock &block : *main)
    if (const auto *candidate =
            llvm::dyn_cast<llvm::SwitchInst>(block.getTerminator()))
      dispatch = candidate;
  ASSERT_NE(dispatch, nullptr);
  EXPECT_EQ(dispatch->getNumCases(), 16u);
  EXPECT_EQ(llvm::count_if(first->module->functions(),
                           [](const llvm::Function &function) {
                             return function.getName().starts_with(
                                        "__wafer_kernel_launch_slot_") &&
                                    function.getName().ends_with("_main_body");
                           }),
            16u);
  EXPECT_TRUE(firstIR.find("i32 15, label %tile.15.launch_slot.15") !=
              std::string::npos);
  EXPECT_TRUE(firstIR.find("getelementptr inbounds i64, ptr %tile_major_slots, "
                           "i64 31") != std::string::npos);
  EXPECT_TRUE(firstIR.find("@__wafer_kernel_launch_slot_00015_main_body") !=
              std::string::npos);

  for (const wafer::compiler::TargetLLVMModule &source :
       targetLLVMModules->getModules()) {
    EXPECT_NE(source.getModule().getFunction("main"), nullptr);
    EXPECT_NE(source.getModule().getFunction("helper"), nullptr);
    EXPECT_EQ(source.getModule().getFunction(
                  "__wafer_kernel_launch_slot_00000_main_body"),
              nullptr);
  }
}

TEST(TargetCodeGenTest, KernelAggregationRejectsUnsupportedLinkConstructs) {
  llvm::Expected<wafer::compiler::TargetLLVMModules> targetLLVMModules =
      makeRuntimeLaunchModules(
          makeKernelLaunch(wafer::KernelLaunchForm::Cluster),
          wafer::compiler::TransportContract::DirectDTE, std::nullopt,
          std::nullopt, /*slotCount=*/2, /*withCollidingClosure=*/false,
          /*withUnsupportedInlineAsm=*/true);
  ASSERT_TRUE(static_cast<bool>(targetLLVMModules))
      << llvm::toString(targetLLVMModules.takeError());
  llvm::Expected<wafer::compiler::detail::OwnedTargetLLVMModule> aggregate =
      wafer::compiler::detail::buildKernelAggregateTargetModule(
          *targetLLVMModules);
  ASSERT_FALSE(static_cast<bool>(aggregate));
  EXPECT_NE(
      llvm::toString(aggregate.takeError()).find("module inline assembly"),
      std::string::npos);
}

TEST(TargetCodeGenTest, KernelGridLinkingAcceptsTileSpecializedModules) {
  llvm::Expected<wafer::compiler::TargetLLVMModules> targetLLVMModules =
      makeRuntimeLaunchModules(makeKernelLaunch(wafer::KernelLaunchForm::Grid),
                               wafer::compiler::TransportContract::None,
                               std::nullopt,
                               /*mismatchedBodyTile=*/1);
  ASSERT_TRUE(static_cast<bool>(targetLLVMModules))
      << llvm::toString(targetLLVMModules.takeError());
  llvm::SmallString<256> pythonExecutable;
  llvm::SmallString<256> llvmClangXX;
  ASSERT_FALSE(
      llvm::sys::fs::real_path(WAFER_TEST_PYTHON_EXECUTABLE, pythonExecutable));
  ASSERT_FALSE(llvm::sys::fs::real_path(WAFER_TEST_LLVM_CLANGXX, llvmClangXX));
  llvm::Expected<wafer::compiler::TargetToolchain> toolchain =
      wafer::compiler::TargetToolchain::create(
          pythonExecutable, WAFER_TEST_DEVICE_LINKER_SCRIPT, llvmClangXX,
          WAFER_TEST_TX8_DEPS_ROOT, WAFER_TEST_WAFER_INCLUDE_DIR,
          WAFER_TEST_WAFER_CRT_SOURCE, WAFER_TEST_WAFER_CRT_INCLUDE_DIR);
  ASSERT_TRUE(static_cast<bool>(toolchain))
      << llvm::toString(toolchain.takeError());

  llvm::SmallString<256> temporaryDirectory;
  ASSERT_FALSE(llvm::sys::fs::createUniqueDirectory(
      "wafer-grid-tile-specialization", temporaryDirectory));
  auto cleanup = llvm::make_scope_exit(
      [&]() { (void)llvm::sys::fs::remove_directories(temporaryDirectory); });
  llvm::SmallString<256> outputDirectory =
      pathInDirectory(temporaryDirectory, "targetModules");
  std::string diagnosticsStorage;
  llvm::raw_string_ostream diagnostics(diagnosticsStorage);
  llvm::Expected<wafer::compiler::LinkedTargetModules> result =
      wafer::compiler::linkTargetLLVMModules(
          *targetLLVMModules, outputDirectory, *toolchain, diagnostics);
  ASSERT_TRUE(static_cast<bool>(result))
      << diagnosticsStorage << llvm::toString(result.takeError());
  EXPECT_EQ(result->getRuntimeLaunchContract(),
            makeKernelLaunch(wafer::KernelLaunchForm::Grid));
  EXPECT_EQ(result->getModules().size(), 1u);
  EXPECT_EQ(result->getTileInterfaces().size(), 16u);
  EXPECT_TRUE(llvm::sys::fs::exists(outputDirectory));
}

TEST(TargetCodeGenTest,
     SharedKernelFormsLinkOneModuleAndCompleteTileInterfaces) {
  llvm::Expected<wafer::compiler::TargetToolchain> toolchain =
      makeTestToolchain();
  ASSERT_TRUE(static_cast<bool>(toolchain))
      << llvm::toString(toolchain.takeError());
  llvm::SmallString<256> temporaryDirectory;
  ASSERT_FALSE(llvm::sys::fs::createUniqueDirectory(
      "wafer-shared-module-linking", temporaryDirectory));
  auto cleanup = llvm::make_scope_exit(
      [&]() { (void)llvm::sys::fs::remove_directories(temporaryDirectory); });

  for (const SharedKernelTransportScenario &scenario :
       kOrthogonalSharedKernelTransportScenarios) {
    wafer::RuntimeLaunchContract launch = makeKernelLaunch(scenario.form);
    SCOPED_TRACE(
        (wafer::stringifyKernelLaunchForm(scenario.form) + ":" +
         (scenario.transport == wafer::compiler::TransportContract::DirectDTE
              ? "direct-dte"
              : "none"))
            .str());
    llvm::Expected<wafer::compiler::TargetLLVMModules> targetLLVMModules =
        makeRuntimeLaunchModules(launch, scenario.transport, std::nullopt,
                                 std::nullopt,
                                 /*slotCount=*/2,
                                 /*withCollidingClosure=*/false,
                                 /*withUnsupportedInlineAsm=*/false,
                                 /*renamedSlotTile=*/7);
    ASSERT_TRUE(static_cast<bool>(targetLLVMModules))
        << llvm::toString(targetLLVMModules.takeError());
    llvm::SmallString<256> outputDirectory = pathInDirectory(
        temporaryDirectory, scenario.form == wafer::KernelLaunchForm::Grid
                                ? "grid-targetModules"
                                : "cluster-targetModules");
    std::string diagnosticsStorage;
    llvm::raw_string_ostream diagnostics(diagnosticsStorage);
    llvm::Expected<wafer::compiler::LinkedTargetModules> targetModules =
        wafer::compiler::linkTargetLLVMModules(
            *targetLLVMModules, outputDirectory, *toolchain, diagnostics);
    ASSERT_TRUE(static_cast<bool>(targetModules))
        << diagnosticsStorage << llvm::toString(targetModules.takeError());
    ASSERT_EQ(targetModules->getModules().size(), 1u);
    ASSERT_EQ(targetModules->getTileInterfaces().size(), 16u);
    const wafer::compiler::VerifiedTargetModule &module =
        targetModules->getModules().front();
    EXPECT_EQ(module.getId().getValue(), 0u);
    EXPECT_EQ(module.getRelativePath(), "modules/module_00000.so");
    EXPECT_EQ(module.getExports().size(), launch.getPhases().size());
    EXPECT_EQ(module.getExports().back().getRole(),
              wafer::compiler::TargetExportRole::Main);
    EXPECT_EQ(module.getExports().back().getSymbol(), "main");
    if (module.getExports().size() == 2) {
      EXPECT_EQ(module.getExports().front().getRole(),
                wafer::compiler::TargetExportRole::Prepare);
      EXPECT_EQ(module.getExports().front().getSymbol(),
                wafer::compiler::detail::kKernelPrepareExportSymbol);
    }
    for (auto [launchSlot, tileInterface] :
         llvm::enumerate(targetModules->getTileInterfaces())) {
      EXPECT_EQ(tileInterface.getCardId(), wafer::CardId(0));
      EXPECT_EQ(tileInterface.getTileId(),
                wafer::TileId(static_cast<int64_t>(launchSlot)));
      EXPECT_EQ(tileInterface.getLaunchSlotId(),
                wafer::LaunchSlotId(static_cast<int64_t>(launchSlot)));
      EXPECT_EQ(tileInterface.getModuleId().getValue(), 0u);
      EXPECT_EQ(tileInterface.getTileEntryArguments().size(), 2u);
    }
    llvm::SmallString<256> modulePath(outputDirectory);
    llvm::sys::path::append(modulePath, module.getRelativePath());
    EXPECT_TRUE(llvm::sys::fs::is_regular_file(modulePath));
    llvm::SmallString<256> workPath(outputDirectory);
    llvm::sys::path::append(workPath, "work");
    EXPECT_FALSE(llvm::sys::fs::exists(workPath));
  }
}

TEST(TargetCodeGenTest,
     ProfileTraceCaptureCompilesAndPackagesEveryQualifiedKernelForm) {
  llvm::Expected<wafer::compiler::TargetToolchain> toolchain =
      makeTestToolchain();
  ASSERT_TRUE(static_cast<bool>(toolchain))
      << llvm::toString(toolchain.takeError());
  llvm::SmallString<256> temporaryDirectory;
  ASSERT_FALSE(llvm::sys::fs::createUniqueDirectory(
      "wafer-profile-runtime-launch", temporaryDirectory));
  auto cleanup = llvm::make_scope_exit(
      [&]() { (void)llvm::sys::fs::remove_directories(temporaryDirectory); });
  llvm::SmallString<256> sourceDirectory =
      pathInDirectory(temporaryDirectory, "source");
  ASSERT_FALSE(llvm::sys::fs::create_directories(sourceDirectory));

  constexpr wafer::compiler::detail::ProfileCaptureKind capture =
      wafer::compiler::detail::ProfileCaptureKind::Trace;
  for (const SharedKernelTransportScenario &scenario :
       kOrthogonalSharedKernelTransportScenarios) {
    wafer::RuntimeLaunchContract launch = makeKernelLaunch(scenario.form);
    SCOPED_TRACE(
        (wafer::stringifyKernelLaunchForm(scenario.form) + ":" +
         (scenario.transport == wafer::compiler::TransportContract::DirectDTE
              ? "direct-dte"
              : "none"))
            .str());
    llvm::Expected<wafer::compiler::TargetLLVMModules> targetLLVM =
        makeProfileRuntimeLaunchModules(launch, capture, scenario.transport);
    ASSERT_TRUE(static_cast<bool>(targetLLVM))
        << llvm::toString(targetLLVM.takeError());
    llvm::SmallString<256> outputDirectory = pathInDirectory(
        temporaryDirectory, (llvm::Twine("targetModules-") +
                             wafer::stringifyKernelLaunchForm(scenario.form))
                                .str());
    std::string diagnosticsStorage;
    llvm::raw_string_ostream diagnostics(diagnosticsStorage);
    llvm::Expected<wafer::compiler::LinkedTargetModules> targetModules =
        wafer::compiler::detail::linkTargetLLVMModulesImpl(
            *targetLLVM, outputDirectory, *toolchain, diagnostics, capture);
    ASSERT_TRUE(static_cast<bool>(targetModules))
        << diagnosticsStorage << llvm::toString(targetModules.takeError());
    EXPECT_EQ(targetModules->getModules().size(), 1u);
    ASSERT_EQ(targetModules->getTileInterfaces().size(), 16u);
    for (const wafer::compiler::VerifiedTargetTileInterface &tileInterface :
         targetModules->getTileInterfaces()) {
      ASSERT_FALSE(tileInterface.getTileEntryArguments().empty());
      const wafer::compiler::TileEntryArgument &profileSlot =
          tileInterface.getTileEntryArguments().back();
      EXPECT_EQ(profileSlot.kind,
                wafer::compiler::TileEntryArgumentKind::ProfileRecord);
      EXPECT_EQ(profileSlot.resourceIndex, 0);
      EXPECT_EQ(profileSlot.byteSize, WAFER_TX81_PROFILER_TRACE_BUFFER_BYTES);
      EXPECT_FALSE(static_cast<bool>(
          wafer::compiler::detail::verifyProfileCaptureTileEntryArguments(
              tileInterface.getTileEntryArguments(), capture)));
    }

    llvm::Expected<wafer::compiler::CardExecutable> executable =
        makeProfileCardExecutable(launch, scenario.transport);
    ASSERT_TRUE(static_cast<bool>(executable))
        << llvm::toString(executable.takeError());
    llvm::SmallString<256> packageDirectory = pathInDirectory(
        temporaryDirectory, (llvm::Twine("package-") +
                             wafer::stringifyKernelLaunchForm(scenario.form))
                                .str());
    llvm::Expected<wafer::runtime::VerifiedPackageManifest> package =
        wafer::compiler::detail::writePackage(sourceDirectory, *executable,
                                              *targetModules, packageDirectory,
                                              diagnostics, std::nullopt);
    ASSERT_TRUE(static_cast<bool>(package))
        << diagnosticsStorage << llvm::toString(package.takeError());
    const wafer::runtime::PackageManifest &manifest = package->getManifest();
    EXPECT_EQ(manifest.launch, launch);
    EXPECT_EQ(manifest.cardCount, 1);
    EXPECT_EQ(manifest.tileCount, 16);
    EXPECT_EQ(manifest.modules.size(), 1u);
    ASSERT_EQ(manifest.entries.size(), 16u);
    const wafer::runtime::ExternalPortRecord &sharedInput =
        manifest.inputs.front();
    ASSERT_EQ(manifest.inputs.size(), 1u);
    EXPECT_EQ(sharedInput.roleIndex, 0);
    wafer::runtime::RuntimeEnvironment noCardEnvironment{
        manifest.targetIdentity, manifest.runtimeABI, manifest.moduleFormat};
    const wafer::KernelRuntimeLaunchContract &kernel =
        manifest.launch.getKernel();
    noCardEnvironment.supportedKernelLaunchForms = {kernel.form};
    noCardEnvironment.supportedKernelEntryABIs = {kernel.entryABI};
    noCardEnvironment.supportsDirectDTE =
        scenario.transport == wafer::compiler::TransportContract::DirectDTE;
    noCardEnvironment.directDTEStatusABI =
        wafer::runtime::kDirectDTEStatusABI.str();
    noCardEnvironment.supportsHostWatchdog = true;
    std::vector<wafer::runtime::RuntimeInvocationBinding> noCardBindings;
    for (const wafer::runtime::ExternalPortRecord &port : manifest.inputs)
      noCardBindings.push_back({port.id, port.bytes, port.alignment});
    llvm::Expected<wafer::runtime::RuntimeInvocationPlan> noCardPlan =
        wafer::runtime::planRuntimeInvocation(*package, noCardBindings,
                                              noCardEnvironment);
    ASSERT_TRUE(static_cast<bool>(noCardPlan))
        << llvm::toString(noCardPlan.takeError());
    ASSERT_EQ(noCardPlan->tiles.size(), 16u);
    EXPECT_TRUE(llvm::all_of(noCardPlan->tiles, [&](const auto &tile) {
      return !tile.argumentAddresses.empty() &&
             tile.argumentAddresses.front().base ==
                 wafer::runtime::RuntimeArgumentAddressBase::Invocation;
    }));
    EXPECT_EQ(
        llvm::count_if(
            manifest.entries,
            [](const auto &entry) {
              return llvm::any_of(entry.arguments, [](const auto &argument) {
                if (const auto *profile =
                        std::get_if<wafer::runtime::ProfileRecordArgument>(
                            &argument.reference))
                  return profile->bytes ==
                         WAFER_TX81_PROFILER_TRACE_BUFFER_BYTES;
                return false;
              });
            }),
        16);
    for (const wafer::runtime::PackageEntrypointRecord &entry :
         manifest.entries) {
      ASSERT_FALSE(entry.arguments.empty());
      EXPECT_TRUE(std::holds_alternative<wafer::runtime::ExternalInputArgument>(
          entry.arguments.front().reference));
      EXPECT_TRUE(std::holds_alternative<wafer::runtime::ProfileRecordArgument>(
          entry.arguments.back().reference));
      EXPECT_EQ(std::get<wafer::runtime::ProfileRecordArgument>(
                    entry.arguments.back().reference)
                    .bytes,
                WAFER_TX81_PROFILER_TRACE_BUFFER_BYTES);
      EXPECT_EQ(
          std::holds_alternative<
              wafer::runtime::DirectDTETransportRequirements>(entry.transport),
          scenario.transport == wafer::compiler::TransportContract::DirectDTE);
    }
  }
}

TEST(TargetCodeGenTest, LinkedRiscvELFReadbackCarriesTypedProfileFacts) {
  llvm::SmallString<256> temporaryDirectory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "wafer-target-readback", temporaryDirectory);
  ASSERT_FALSE(error) << error.message();
  auto cleanup = llvm::make_scope_exit(
      [&]() { (void)llvm::sys::fs::remove_directories(temporaryDirectory); });

  llvm::SmallString<256> modulePath =
      pathInDirectory(temporaryDirectory, "kernel.so");
  ASSERT_EQ(linkTargetFixture(modulePath), 0);

  llvm::Expected<wafer::compiler::VerifiedTargetModule> module =
      wafer::compiler::detail::verifyLinkedTargetModuleForTesting(
          modulePath, "kernel_entry", kTargetIdentity,
          makeKernelLaunch(wafer::KernelLaunchForm::Grid));
  ASSERT_TRUE(static_cast<bool>(module)) << llvm::toString(module.takeError());
  EXPECT_EQ(module->getTargetIdentityId(), kTargetIdentity);
  EXPECT_EQ(module->getTargetIdentityId(), wafer::kCurrentTargetIdentity);
  EXPECT_EQ(module->getKernelRuntimeABIId(), wafer::kCurrentKernelRuntimeABI);
  EXPECT_EQ(module->getModuleFormat(), wafer::kCurrentTargetModuleFormat);
  EXPECT_EQ(module->getModuleFormat(), "elf-riscv64");
  EXPECT_TRUE(module->getContentDigest().starts_with("sha256:"));
  EXPECT_EQ(module->getContentDigest().size(), 71u);

  llvm::Expected<wafer::compiler::VerifiedTargetModule> missingEntry =
      wafer::compiler::detail::verifyLinkedTargetModuleForTesting(
          modulePath, "missing_entry", kTargetIdentity,
          makeKernelLaunch(wafer::KernelLaunchForm::Grid));
  ASSERT_FALSE(static_cast<bool>(missingEntry));
  EXPECT_NE(llvm::toString(missingEntry.takeError())
                .find("target entry symbol is not defined"),
            std::string::npos);

  llvm::Expected<wafer::compiler::VerifiedTargetModule> dataEntry =
      wafer::compiler::detail::verifyLinkedTargetModuleForTesting(
          modulePath, "data_entry", kTargetIdentity,
          makeKernelLaunch(wafer::KernelLaunchForm::Grid));
  ASSERT_FALSE(static_cast<bool>(dataEntry));
  EXPECT_NE(llvm::toString(dataEntry.takeError())
                .find("target entry symbol is defined but is not a function"),
            std::string::npos);

  llvm::Expected<wafer::compiler::VerifiedTargetModule> localEntry =
      wafer::compiler::detail::verifyLinkedTargetModuleForTesting(
          modulePath, "local_entry", kTargetIdentity,
          makeKernelLaunch(wafer::KernelLaunchForm::Grid));
  ASSERT_FALSE(static_cast<bool>(localEntry));
  EXPECT_NE(llvm::toString(localEntry.takeError())
                .find("target entry symbol is a function but is not externally "
                      "visible"),
            std::string::npos);
}

TEST(TargetCodeGenTest, ReadbackRejectsNonELFAndWrongArchitecture) {
  llvm::SmallString<256> temporaryDirectory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "wafer-target-readback-invalid", temporaryDirectory);
  ASSERT_FALSE(error) << error.message();
  auto cleanup = llvm::make_scope_exit(
      [&]() { (void)llvm::sys::fs::remove_directories(temporaryDirectory); });

  llvm::SmallString<256> nonELFPath =
      pathInDirectory(temporaryDirectory, "not-elf.so");
  {
    std::error_code outputError;
    llvm::raw_fd_ostream output(nonELFPath, outputError,
                                llvm::sys::fs::OF_None);
    ASSERT_FALSE(outputError) << outputError.message();
    output << "not an ELF file";
  }
  llvm::Expected<wafer::compiler::VerifiedTargetModule> nonELF =
      wafer::compiler::detail::verifyLinkedTargetModuleForTesting(
          nonELFPath, "kernel_entry", kTargetIdentity,
          makeKernelLaunch(wafer::KernelLaunchForm::Grid));
  ASSERT_FALSE(static_cast<bool>(nonELF));
  EXPECT_NE(llvm::toString(nonELF.takeError()).find("target module is not ELF"),
            std::string::npos);

  llvm::Expected<wafer::compiler::VerifiedTargetModule> wrongArchitecture =
      wafer::compiler::detail::verifyLinkedTargetModuleForTesting(
          "/bin/true", "kernel_entry", kTargetIdentity,
          makeKernelLaunch(wafer::KernelLaunchForm::Grid));
  ASSERT_FALSE(static_cast<bool>(wrongArchitecture));
  EXPECT_NE(llvm::toString(wrongArchitecture.takeError())
                .find("target module is not RISC-V 64-bit ELF"),
            std::string::npos);
}

} // namespace
