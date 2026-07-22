//===- TargetArtifactTest.cpp - Linked target readback tests -------------===//

#include "../../lib/Wafer/Compiler/PackageInternal.h"
#include "../../lib/Wafer/Compiler/TargetArtifactInternal.h"

#include "Wafer/Target/TargetProfile.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

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

constexpr wafer::TargetProfileId kProfile =
    wafer::TargetProfileId::waferTx81SingleCardKernelV1();

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
      pythonExecutable, WAFER_TEST_DEVICE_LINKER_SCRIPT, llvmClangXX);
}

static wafer::compiler::KernelABISlot
makeSlot(int64_t ordinal, wafer::compiler::KernelABISlotRole role,
         llvm::StringRef name) {
  if (role == wafer::compiler::KernelABISlotRole::TransportStatus)
    return {ordinal, role, 0, name.str(), "u32", wafer::MemLayout::Tensor,
            {1},
            WAFER_TX81_DIRECT_DTE_STATUS_V2_STORAGE_BYTES,
            WAFER_TX81_DIRECT_DTE_STATUS_V2_STORAGE_ALIGNMENT};
  return {ordinal, role, ordinal, name.str(), "f32", wafer::MemLayout::Tensor,
          {4},     16,   64};
}

static llvm::Expected<wafer::compiler::TargetLLVMModuleBundle>
makeLaunchABIBundle(wafer::TargetLaunchABIId launchABI,
                    std::optional<int64_t> mismatchedSchemaRank = std::nullopt,
                    std::optional<int64_t> mismatchedBodyRank = std::nullopt,
                    bool unsupportedModelRole = false, size_t slotCount = 2,
                    bool withCollidingClosure = false,
                    bool withUnsupportedInlineAsm = false,
                    std::optional<int64_t> renamedSlotRank = std::nullopt) {
  llvm::Expected<wafer::compiler::ExecutionConfig> config =
      wafer::compiler::ExecutionConfig::createForSingleCard(16, kProfile,
                                                            launchABI);
  if (!config)
    return config.takeError();
  const wafer::TargetProfileRecord &profile =
      wafer::getTargetProfileRecord(kProfile);
  std::vector<wafer::compiler::TargetLLVMModule> modules;
  modules.reserve(16);
  for (int64_t rank = 0; rank < 16; ++rank) {
    std::vector<wafer::compiler::KernelABISlot> slots;
    slots.reserve(slotCount);
    for (size_t slot = 0; slot < slotCount; ++slot) {
      wafer::compiler::KernelABISlotRole role =
          launchABI == wafer::TargetLaunchABIId::
                           tx81ClusterDirectDTEPrepareMainV1() &&
                  slot + 1 == slotCount
              ? wafer::compiler::KernelABISlotRole::TransportStatus
          : slot + 1 == slotCount
              ? wafer::compiler::KernelABISlotRole::Output
              : wafer::compiler::KernelABISlotRole::UserInput;
      if (unsupportedModelRole && slot == 0)
        role = wafer::compiler::KernelABISlotRole::Parameter;
      llvm::StringRef name =
          role == wafer::compiler::KernelABISlotRole::Output ? "output"
          : role == wafer::compiler::KernelABISlotRole::TransportStatus
              ? "direct_dte_status"
              : "input";
      slots.push_back(makeSlot(
          slot, role,
          renamedSlotRank == rank && slot == 0 ? "diagnostic_alias" : name));
      if (mismatchedSchemaRank == rank && slot == 0)
        slots.back().dtype = "f16";
    }
    auto context = std::make_unique<llvm::LLVMContext>();
    auto module = std::make_unique<llvm::Module>("launch-entry", *context);
    if (withUnsupportedInlineAsm && rank == 0)
      module->setModuleInlineAsm(".word 0");
    llvm::Type *i64 = llvm::Type::getInt64Ty(*context);
    llvm::Function *helper = nullptr;
    if (withCollidingClosure) {
      auto *state = new llvm::GlobalVariable(
          *module, i64, /*isConstant=*/false,
          llvm::GlobalValue::ExternalLinkage,
          llvm::ConstantInt::get(i64, static_cast<uint64_t>(rank)), "state");
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
    if (mismatchedBodyRank == rank) {
      llvm::Value *address = builder.CreateIntToPtr(
          entry->getArg(0), llvm::PointerType::get(*context, 0));
      llvm::StoreInst *store =
          builder.CreateStore(llvm::ConstantInt::get(i64, 0), address);
      store->setVolatile(true);
    }
    builder.CreateRetVoid();
    modules.push_back(
        wafer::compiler::TargetLLVMModuleBundleBuilder::makeModule(
            rank, "main", kProfile, profile.targetIdentity,
            profile.kernelRuntimeABI, profile.moduleFormat, std::move(slots),
            std::move(context), std::move(module)));
  }
  return wafer::compiler::TargetLLVMModuleBundleBuilder::makeBundle(
      *config, std::move(modules));
}

TEST(TargetArtifactTest, PublicVerifiedModuleCannotBeForgedOrDefaulted) {
  static_assert(
      !std::is_default_constructible_v<wafer::compiler::VerifiedTargetModule>);
}

TEST(TargetArtifactTest, PackageSlotLegalityIgnoresDiagnosticNames) {
  wafer::compiler::RankProgramBinding binding{};
  binding.role = wafer::compiler::ProgramResourceRole::UserInput;
  binding.index = 0;
  binding.name = "frontend_name";
  binding.dtype = "f32";
  binding.localShape = {4};
  wafer::compiler::KernelABISlot userInput =
      makeSlot(0, wafer::compiler::KernelABISlotRole::UserInput,
               "lowered_diagnostic_alias");
  EXPECT_TRUE(wafer::compiler::detail::doesPackageSlotMatchProgramBinding(
      userInput, binding));
  binding.dtype = "f16";
  EXPECT_FALSE(wafer::compiler::detail::doesPackageSlotMatchProgramBinding(
      userInput, binding));

  wafer::compiler::KernelABISlot workspace{
      1,    wafer::compiler::KernelABISlotRole::Workspace,
      0,    "renamed_workspace",
      "u8", wafer::MemLayout::Tensor,
      {64}, 64,
      64};
  EXPECT_TRUE(
      wafer::compiler::detail::isValidPackageCompilerManagedSlot(workspace));
  workspace.name = "another_workspace_label";
  EXPECT_TRUE(
      wafer::compiler::detail::isValidPackageCompilerManagedSlot(workspace));

  wafer::compiler::KernelABISlot status = makeSlot(
      2, wafer::compiler::KernelABISlotRole::TransportStatus, "renamed_status");
  EXPECT_TRUE(
      wafer::compiler::detail::isValidPackageCompilerManagedSlot(status));
  status.alignment = 8;
  EXPECT_FALSE(
      wafer::compiler::detail::isValidPackageCompilerManagedSlot(status));
}

TEST(TargetArtifactTest,
     DevicePublicationWrapsOrderedSlotsWithoutMutatingOwner) {
  llvm::LLVMContext context;
  llvm::Module module("device-entry", context);
  llvm::Type *i64 = llvm::Type::getInt64Ty(context);
  llvm::FunctionType *type = llvm::FunctionType::get(
      llvm::Type::getVoidTy(context), {i64, i64}, /*isVarArg=*/false);
  llvm::Function *entry = llvm::Function::Create(
      type, llvm::GlobalValue::ExternalLinkage, "main", module);
  llvm::IRBuilder<> builder(llvm::BasicBlock::Create(context, "entry", entry));
  builder.CreateRetVoid();

  llvm::SmallString<256> temporaryDirectory;
  ASSERT_FALSE(llvm::sys::fs::createUniqueDirectory("wafer-device-entry",
                                                    temporaryDirectory));
  auto cleanup = llvm::make_scope_exit(
      [&]() { (void)llvm::sys::fs::remove_directories(temporaryDirectory); });
  llvm::SmallString<256> outputPath =
      pathInDirectory(temporaryDirectory, "entry.ll");
  std::vector<wafer::compiler::KernelABISlot> slots = {
      makeSlot(0, wafer::compiler::KernelABISlotRole::UserInput, "input"),
      makeSlot(1, wafer::compiler::KernelABISlotRole::Output, "output")};
  if (llvm::Error error = wafer::compiler::detail::writeLLVMIR(
          module, "main", slots,
          wafer::TargetLaunchABIId::perRankPointerBlockV1(),
          /*logicalRank=*/0, /*rankCount=*/1, outputPath))
    FAIL() << llvm::toString(std::move(error));

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> output =
      llvm::MemoryBuffer::getFile(outputPath);
  ASSERT_TRUE(static_cast<bool>(output)) << output.getError().message();
  llvm::StringRef text = (*output)->getBuffer();
  EXPECT_TRUE(text.contains("define void @main(ptr %slots)"));
  EXPECT_TRUE(text.contains(
      "define internal void @wafer_device_entry_body(i64 %0, i64 %1)"));
  EXPECT_TRUE(text.contains("getelementptr inbounds i64, ptr %slots, i64 1"));
  EXPECT_TRUE(text.contains("call void @wafer_device_entry_body(i64"));

  EXPECT_EQ(module.getFunction("main"), entry);
  EXPECT_EQ(entry->arg_size(), 2u);
  EXPECT_EQ(module.getFunction("wafer_device_entry_body"), nullptr);

  slots.push_back(
      makeSlot(2, wafer::compiler::KernelABISlotRole::Output, "extra"));
  llvm::Error invalid = wafer::compiler::detail::writeLLVMIR(
      module, "main", slots, wafer::TargetLaunchABIId::perRankPointerBlockV1(),
      /*logicalRank=*/0, /*rankCount=*/1, outputPath);
  ASSERT_TRUE(static_cast<bool>(invalid));
  EXPECT_NE(llvm::toString(std::move(invalid))
                .find("does not match its typed ABI slots"),
            std::string::npos);
}

TEST(TargetArtifactTest, KernelGridEntrySelectsRankMajorSlotsWithPid) {
  llvm::LLVMContext context;
  llvm::Module module("kernel-grid-entry", context);
  llvm::Type *i64 = llvm::Type::getInt64Ty(context);
  llvm::FunctionType *type = llvm::FunctionType::get(
      llvm::Type::getVoidTy(context), {i64, i64}, /*isVarArg=*/false);
  llvm::Function *entry = llvm::Function::Create(
      type, llvm::GlobalValue::ExternalLinkage, "main", module);
  llvm::IRBuilder<> builder(llvm::BasicBlock::Create(context, "entry", entry));
  builder.CreateRetVoid();

  llvm::SmallString<256> temporaryDirectory;
  ASSERT_FALSE(llvm::sys::fs::createUniqueDirectory("wafer-kernel-grid-entry",
                                                    temporaryDirectory));
  auto cleanup = llvm::make_scope_exit(
      [&]() { (void)llvm::sys::fs::remove_directories(temporaryDirectory); });
  llvm::SmallString<256> outputPath =
      pathInDirectory(temporaryDirectory, "entry.ll");
  std::vector<wafer::compiler::KernelABISlot> slots = {
      makeSlot(0, wafer::compiler::KernelABISlotRole::UserInput, "input"),
      makeSlot(1, wafer::compiler::KernelABISlotRole::Output, "output")};
  if (llvm::Error error = wafer::compiler::detail::writeLLVMIR(
          module, "main", slots,
          wafer::TargetLaunchABIId::tx81KernelGridPointerTableV1(),
          /*logicalRank=*/0, /*rankCount=*/16, outputPath))
    FAIL() << llvm::toString(std::move(error));

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> output =
      llvm::MemoryBuffer::getFile(outputPath);
  ASSERT_TRUE(static_cast<bool>(output)) << output.getError().message();
  llvm::StringRef text = (*output)->getBuffer();
  EXPECT_TRUE(text.contains("declare i32 @__get_pid(i32)"));
  EXPECT_TRUE(text.contains("call i32 @__get_pid(i32 0)"));
  EXPECT_TRUE(text.contains("mul i64 %pid.x.i64, 2"));
  EXPECT_FALSE(text.contains("ExportedDYNSYMTab"));
}

TEST(TargetArtifactTest, ModelEntryLoadsRoleMajorRankDescriptorsAndExports) {
  llvm::LLVMContext context;
  llvm::Module module("model-entry", context);
  llvm::Type *i64 = llvm::Type::getInt64Ty(context);
  llvm::FunctionType *type =
      llvm::FunctionType::get(llvm::Type::getVoidTy(context), {i64, i64, i64},
                              /*isVarArg=*/false);
  llvm::Function *entry = llvm::Function::Create(
      type, llvm::GlobalValue::ExternalLinkage, "main", module);
  llvm::IRBuilder<> builder(llvm::BasicBlock::Create(context, "entry", entry));
  builder.CreateRetVoid();

  llvm::SmallString<256> temporaryDirectory;
  ASSERT_FALSE(llvm::sys::fs::createUniqueDirectory("wafer-model-entry",
                                                    temporaryDirectory));
  auto cleanup = llvm::make_scope_exit(
      [&]() { (void)llvm::sys::fs::remove_directories(temporaryDirectory); });
  llvm::SmallString<256> llvmIRPath =
      pathInDirectory(temporaryDirectory, "entry.ll");
  llvm::SmallString<256> modulePath =
      pathInDirectory(temporaryDirectory, "entry.so");
  std::vector<wafer::compiler::KernelABISlot> slots = {
      makeSlot(0, wafer::compiler::KernelABISlotRole::UserInput, "lhs"),
      makeSlot(1, wafer::compiler::KernelABISlotRole::Output, "output"),
      makeSlot(2, wafer::compiler::KernelABISlotRole::UserInput, "rhs")};
  if (llvm::Error error = wafer::compiler::detail::writeLLVMIR(
          module, "main", slots,
          wafer::TargetLaunchABIId::tx81ModelBootParamV1(),
          /*logicalRank=*/3, /*rankCount=*/16, llvmIRPath))
    FAIL() << llvm::toString(std::move(error));

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> output =
      llvm::MemoryBuffer::getFile(llvmIRPath);
  ASSERT_TRUE(static_cast<bool>(output)) << output.getError().message();
  llvm::StringRef text = (*output)->getBuffer();
  EXPECT_TRUE(text.contains("define void @main(ptr %boot_parameter)"));
  EXPECT_TRUE(
      text.contains("getelementptr inbounds i8, ptr %boot_parameter, i64 488"));
  EXPECT_TRUE(text.contains(
      "getelementptr inbounds i8, ptr %boot_parameter, i64 2576"));
  EXPECT_TRUE(
      text.contains("getelementptr inbounds i8, ptr %boot_parameter, i64 560"));
  EXPECT_TRUE(text.contains("section \"ExportedDYNSYMTab\""));
  EXPECT_TRUE(text.contains("@llvm.used"));

  ASSERT_EQ(linkTargetModule(llvmIRPath, modulePath), 0);
  llvm::Expected<wafer::compiler::VerifiedTargetModule> verified =
      wafer::compiler::detail::verifyLinkedTargetModuleForTesting(
          modulePath, "main", kProfile,
          wafer::TargetLaunchABIId::tx81ModelBootParamV1());
  ASSERT_TRUE(static_cast<bool>(verified))
      << llvm::toString(verified.takeError());

  std::vector<wafer::compiler::KernelABISlot> unsupportedSlots = slots;
  unsupportedSlots[1] =
      makeSlot(1, wafer::compiler::KernelABISlotRole::Parameter, "parameter");
  llvm::Error unsupported = wafer::compiler::detail::writeLLVMIR(
      module, "main", unsupportedSlots,
      wafer::TargetLaunchABIId::tx81ModelBootParamV1(),
      /*logicalRank=*/3, /*rankCount=*/16, llvmIRPath);
  ASSERT_TRUE(static_cast<bool>(unsupported));
  EXPECT_NE(llvm::toString(std::move(unsupported))
                .find("only supports rank-1..6 f32 user-input and output"),
            std::string::npos);
}

TEST(TargetArtifactTest, MultiTileLaunchPreflightRejectsSchemaAndModelRoles) {
  llvm::Expected<wafer::compiler::TargetLLVMModuleBundle> mismatchedSchema =
      makeLaunchABIBundle(
          wafer::TargetLaunchABIId::tx81KernelGridPointerTableV1(),
          /*mismatchedSchemaRank=*/7);
  ASSERT_TRUE(static_cast<bool>(mismatchedSchema))
      << llvm::toString(mismatchedSchema.takeError());
  llvm::Error schemaError =
      wafer::compiler::detail::validateTargetLaunchABIDomainForTesting(
          *mismatchedSchema);
  ASSERT_TRUE(static_cast<bool>(schemaError));
  EXPECT_NE(llvm::toString(std::move(schemaError))
                .find("identical all-rank slot schemas"),
            std::string::npos);

  llvm::Expected<wafer::compiler::TargetLLVMModuleBundle> unsupportedModel =
      makeLaunchABIBundle(wafer::TargetLaunchABIId::tx81ModelBootParamV1(),
                          std::nullopt, std::nullopt,
                          /*unsupportedModelRole=*/true);
  ASSERT_TRUE(static_cast<bool>(unsupportedModel))
      << llvm::toString(unsupportedModel.takeError());
  llvm::Error modelError =
      wafer::compiler::detail::validateTargetLaunchABIDomainForTesting(
          *unsupportedModel);
  ASSERT_TRUE(static_cast<bool>(modelError));
  EXPECT_NE(llvm::toString(std::move(modelError))
                .find("only supports rank-1..6 f32 user-input and output"),
            std::string::npos);
}

TEST(TargetArtifactTest, KernelArgumentPacketLimitIsCheckedBeforeDeviceLink) {
  EXPECT_EQ(wafer::kTx81ClusterKernelArgumentBytesMax, 0x7d0u);

  auto verifySlotCount = [](wafer::TargetLaunchABIId launchABI,
                            size_t slotCount) {
    llvm::Expected<wafer::compiler::TargetLLVMModuleBundle> bundle =
        makeLaunchABIBundle(launchABI, std::nullopt, std::nullopt,
                            /*unsupportedModelRole=*/false, slotCount);
    EXPECT_TRUE(static_cast<bool>(bundle));
    if (!bundle)
      return bundle.takeError();
    return wafer::compiler::detail::validateTargetLaunchABIDomainForTesting(
        *bundle);
  };

  llvm::Error rejectedEmptyGrid = verifySlotCount(
      wafer::TargetLaunchABIId::tx81KernelGridPointerTableV1(), 0);
  ASSERT_TRUE(static_cast<bool>(rejectedEmptyGrid));
  EXPECT_NE(llvm::toString(std::move(rejectedEmptyGrid))
                .find("at least one typed ABI slot"),
            std::string::npos);

  llvm::Error allowedGrid = verifySlotCount(
      wafer::TargetLaunchABIId::tx81KernelGridPointerTableV1(), 15);
  EXPECT_FALSE(static_cast<bool>(allowedGrid));
  llvm::Error rejectedGrid = verifySlotCount(
      wafer::TargetLaunchABIId::tx81KernelGridPointerTableV1(), 16);
  ASSERT_TRUE(static_cast<bool>(rejectedGrid));
  EXPECT_NE(llvm::toString(std::move(rejectedGrid)).find("packet limit"),
            std::string::npos);

  llvm::Error allowedCluster = verifySlotCount(
      wafer::TargetLaunchABIId::tx81ClusterDirectDTEPrepareMainV1(), 15);
  EXPECT_FALSE(static_cast<bool>(allowedCluster));
  llvm::Error rejectedCluster = verifySlotCount(
      wafer::TargetLaunchABIId::tx81ClusterDirectDTEPrepareMainV1(), 16);
  ASSERT_TRUE(static_cast<bool>(rejectedCluster));
  EXPECT_NE(llvm::toString(std::move(rejectedCluster)).find("packet limit"),
            std::string::npos);

  llvm::Error allowedPerRank =
      verifySlotCount(wafer::TargetLaunchABIId::perRankPointerBlockV1(), 251);
  EXPECT_FALSE(static_cast<bool>(allowedPerRank));
  llvm::Error rejectedPerRank =
      verifySlotCount(wafer::TargetLaunchABIId::perRankPointerBlockV1(), 252);
  ASSERT_TRUE(static_cast<bool>(rejectedPerRank));
  EXPECT_NE(llvm::toString(std::move(rejectedPerRank)).find("packet limit"),
            std::string::npos);
}

TEST(TargetArtifactTest,
     ClusterAggregationImportsAllRanksAndBuildsTypedExportsDeterministically) {
  llvm::Expected<wafer::compiler::TargetLLVMModuleBundle> bundle =
      makeLaunchABIBundle(
          wafer::TargetLaunchABIId::tx81ClusterDirectDTEPrepareMainV1(),
          std::nullopt, std::nullopt, /*unsupportedModelRole=*/false,
          /*slotCount=*/2, /*withCollidingClosure=*/true,
          /*withUnsupportedInlineAsm=*/false,
          /*renamedSlotRank=*/7);
  ASSERT_TRUE(static_cast<bool>(bundle)) << llvm::toString(bundle.takeError());

  llvm::Expected<wafer::compiler::detail::OwnedTargetLLVMModule> first =
      wafer::compiler::detail::buildClusterTargetModule(*bundle);
  ASSERT_TRUE(static_cast<bool>(first)) << llvm::toString(first.takeError());
  llvm::Expected<wafer::compiler::detail::OwnedTargetLLVMModule> second =
      wafer::compiler::detail::buildClusterTargetModule(*bundle);
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
      wafer::compiler::detail::kClusterPrepareExportSymbol);
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
          auto *rankCount =
              llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(0));
          ASSERT_NE(rankCount, nullptr);
          EXPECT_EQ(rankCount->getZExtValue(), 16u);
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
                                        "__wafer_cluster_rank_") &&
                                    function.getName().ends_with("_main_body");
                           }),
            16u);
  EXPECT_TRUE(firstIR.find("i32 15, label %pid.15") != std::string::npos);
  EXPECT_TRUE(firstIR.find("getelementptr inbounds i64, ptr %rank_major_slots, "
                           "i64 31") != std::string::npos);
  EXPECT_TRUE(firstIR.find("@__wafer_cluster_rank_00015_main_body") !=
              std::string::npos);

  for (const wafer::compiler::TargetLLVMModule &source : bundle->getModules()) {
    EXPECT_NE(source.getModule().getFunction("main"), nullptr);
    EXPECT_NE(source.getModule().getFunction("helper"), nullptr);
    EXPECT_EQ(
        source.getModule().getFunction("__wafer_cluster_rank_00000_main_body"),
        nullptr);
  }
}

TEST(TargetArtifactTest, ClusterAggregationRejectsUnsupportedLinkConstructs) {
  llvm::Expected<wafer::compiler::TargetLLVMModuleBundle> bundle =
      makeLaunchABIBundle(
          wafer::TargetLaunchABIId::tx81ClusterDirectDTEPrepareMainV1(),
          std::nullopt, std::nullopt, /*unsupportedModelRole=*/false,
          /*slotCount=*/2, /*withCollidingClosure=*/false,
          /*withUnsupportedInlineAsm=*/true);
  ASSERT_TRUE(static_cast<bool>(bundle)) << llvm::toString(bundle.takeError());
  llvm::Expected<wafer::compiler::detail::OwnedTargetLLVMModule> aggregate =
      wafer::compiler::detail::buildClusterTargetModule(*bundle);
  ASSERT_FALSE(static_cast<bool>(aggregate));
  EXPECT_NE(
      llvm::toString(aggregate.takeError()).find("module inline assembly"),
      std::string::npos);
}

TEST(TargetArtifactTest, KernelGridPublicationRejectsDifferentFinalModules) {
  llvm::Expected<wafer::compiler::TargetLLVMModuleBundle> bundle =
      makeLaunchABIBundle(
          wafer::TargetLaunchABIId::tx81KernelGridPointerTableV1(),
          std::nullopt, /*mismatchedBodyRank=*/1);
  ASSERT_TRUE(static_cast<bool>(bundle)) << llvm::toString(bundle.takeError());
  llvm::SmallString<256> pythonExecutable;
  llvm::SmallString<256> llvmClangXX;
  ASSERT_FALSE(
      llvm::sys::fs::real_path(WAFER_TEST_PYTHON_EXECUTABLE, pythonExecutable));
  ASSERT_FALSE(llvm::sys::fs::real_path(WAFER_TEST_LLVM_CLANGXX, llvmClangXX));
  llvm::Expected<wafer::compiler::TargetToolchain> toolchain =
      wafer::compiler::TargetToolchain::create(
          pythonExecutable, WAFER_TEST_DEVICE_LINKER_SCRIPT, llvmClangXX);
  ASSERT_TRUE(static_cast<bool>(toolchain))
      << llvm::toString(toolchain.takeError());

  llvm::SmallString<256> temporaryDirectory;
  ASSERT_FALSE(llvm::sys::fs::createUniqueDirectory(
      "wafer-grid-module-mismatch", temporaryDirectory));
  auto cleanup = llvm::make_scope_exit(
      [&]() { (void)llvm::sys::fs::remove_directories(temporaryDirectory); });
  llvm::SmallString<256> outputDirectory =
      pathInDirectory(temporaryDirectory, "artifacts");
  std::string diagnosticsStorage;
  llvm::raw_string_ostream diagnostics(diagnosticsStorage);
  llvm::Expected<wafer::compiler::TargetArtifactBundle> result =
      wafer::compiler::compileTargetLLVMModuleBundleToTargetArtifacts(
          *bundle, outputDirectory, *toolchain, diagnostics);
  ASSERT_FALSE(static_cast<bool>(result));
  llvm::consumeError(result.takeError());
  diagnostics.flush();
  EXPECT_NE(
      diagnosticsStorage.find("kernel-grid modules are not byte-identical"),
      std::string::npos)
      << diagnosticsStorage;
  EXPECT_FALSE(llvm::sys::fs::exists(outputDirectory));
}

TEST(TargetArtifactTest,
     SharedLaunchABIsPublishOneModuleAndCompleteRankInterfaces) {
  llvm::Expected<wafer::compiler::TargetToolchain> toolchain =
      makeTestToolchain();
  ASSERT_TRUE(static_cast<bool>(toolchain))
      << llvm::toString(toolchain.takeError());
  llvm::SmallString<256> temporaryDirectory;
  ASSERT_FALSE(llvm::sys::fs::createUniqueDirectory(
      "wafer-shared-module-publication", temporaryDirectory));
  auto cleanup = llvm::make_scope_exit(
      [&]() { (void)llvm::sys::fs::remove_directories(temporaryDirectory); });

  for (wafer::TargetLaunchABIId launchABI :
       {wafer::TargetLaunchABIId::tx81KernelGridPointerTableV1(),
        wafer::TargetLaunchABIId::tx81ClusterDirectDTEPrepareMainV1()}) {
    SCOPED_TRACE(wafer::stringifyTargetLaunchABIId(launchABI).str());
    llvm::Expected<wafer::compiler::TargetLLVMModuleBundle> bundle =
        makeLaunchABIBundle(launchABI, std::nullopt, std::nullopt,
                            /*unsupportedModelRole=*/false, /*slotCount=*/2,
                            /*withCollidingClosure=*/false,
                            /*withUnsupportedInlineAsm=*/false,
                            /*renamedSlotRank=*/7);
    ASSERT_TRUE(static_cast<bool>(bundle))
        << llvm::toString(bundle.takeError());
    llvm::SmallString<256> outputDirectory = pathInDirectory(
        temporaryDirectory,
        launchABI == wafer::TargetLaunchABIId::tx81KernelGridPointerTableV1()
            ? "grid-artifacts"
            : "cluster-artifacts");
    std::string diagnosticsStorage;
    llvm::raw_string_ostream diagnostics(diagnosticsStorage);
    llvm::Expected<wafer::compiler::TargetArtifactBundle> artifacts =
        wafer::compiler::compileTargetLLVMModuleBundleToTargetArtifacts(
            *bundle, outputDirectory, *toolchain, diagnostics);
    ASSERT_TRUE(static_cast<bool>(artifacts))
        << diagnosticsStorage << llvm::toString(artifacts.takeError());
    ASSERT_EQ(artifacts->getModules().size(), 1u);
    ASSERT_EQ(artifacts->getRankInterfaces().size(), 16u);
    const wafer::compiler::VerifiedTargetModule &module =
        artifacts->getModules().front();
    EXPECT_EQ(module.getId().getValue(), 0u);
    EXPECT_EQ(module.getRelativePath(), "modules/module_00000.so");
    EXPECT_EQ(
        module.getExports().size(),
        launchABI ==
                wafer::TargetLaunchABIId::tx81ClusterDirectDTEPrepareMainV1()
            ? 2u
            : 1u);
    EXPECT_EQ(module.getExports().back().getRole(),
              wafer::compiler::TargetExportRole::Main);
    EXPECT_EQ(module.getExports().back().getSymbol(), "main");
    if (module.getExports().size() == 2) {
      EXPECT_EQ(module.getExports().front().getRole(),
                wafer::compiler::TargetExportRole::Prepare);
      EXPECT_EQ(module.getExports().front().getSymbol(),
                wafer::compiler::detail::kClusterPrepareExportSymbol);
    }
    for (auto [logicalRank, rankInterface] :
         llvm::enumerate(artifacts->getRankInterfaces())) {
      EXPECT_EQ(rankInterface.getLogicalRank(),
                static_cast<int64_t>(logicalRank));
      EXPECT_EQ(rankInterface.getModuleId().getValue(), 0u);
      EXPECT_EQ(rankInterface.getKernelABISlots().size(), 2u);
    }
    llvm::SmallString<256> modulePath(outputDirectory);
    llvm::sys::path::append(modulePath, module.getRelativePath());
    EXPECT_TRUE(llvm::sys::fs::is_regular_file(modulePath));
    llvm::SmallString<256> workPath(outputDirectory);
    llvm::sys::path::append(workPath, "work");
    EXPECT_FALSE(llvm::sys::fs::exists(workPath));
  }
}

TEST(TargetArtifactTest, LinkedRiscvELFReadbackCarriesTypedProfileFacts) {
  llvm::SmallString<256> temporaryDirectory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "wafer-target-readback", temporaryDirectory);
  ASSERT_FALSE(error) << error.message();
  auto cleanup = llvm::make_scope_exit(
      [&]() { (void)llvm::sys::fs::remove_directories(temporaryDirectory); });

  llvm::SmallString<256> modulePath =
      pathInDirectory(temporaryDirectory, "kernel.so");
  ASSERT_EQ(linkTargetFixture(modulePath), 0);

  llvm::Expected<wafer::compiler::VerifiedTargetModule> missingModelExport =
      wafer::compiler::detail::verifyLinkedTargetModuleForTesting(
          modulePath, "kernel_entry", kProfile,
          wafer::TargetLaunchABIId::tx81ModelBootParamV1());
  ASSERT_FALSE(static_cast<bool>(missingModelExport));
  EXPECT_NE(llvm::toString(missingModelExport.takeError())
                .find("missing ExportedDYNSYMTab"),
            std::string::npos);

  llvm::Expected<wafer::compiler::VerifiedTargetModule> module =
      wafer::compiler::detail::verifyLinkedTargetModuleForTesting(
          modulePath, "kernel_entry", kProfile,
          wafer::TargetLaunchABIId::perRankPointerBlockV1());
  ASSERT_TRUE(static_cast<bool>(module)) << llvm::toString(module.takeError());
  const wafer::TargetProfileRecord &profile =
      wafer::getTargetProfileRecord(kProfile);
  EXPECT_EQ(module->getTargetProfileId(), kProfile);
  EXPECT_EQ(module->getTargetIdentityId(), profile.targetIdentity);
  EXPECT_EQ(module->getKernelRuntimeABIId(), profile.kernelRuntimeABI);
  EXPECT_EQ(module->getModuleFormat(), profile.moduleFormat);
  EXPECT_EQ(module->getModuleFormat(), "elf-riscv64");
  EXPECT_TRUE(module->getContentDigest().starts_with("sha256:"));
  EXPECT_EQ(module->getContentDigest().size(), 71u);

  llvm::Expected<wafer::compiler::VerifiedTargetModule> missingEntry =
      wafer::compiler::detail::verifyLinkedTargetModuleForTesting(
          modulePath, "missing_entry", kProfile,
          wafer::TargetLaunchABIId::perRankPointerBlockV1());
  ASSERT_FALSE(static_cast<bool>(missingEntry));
  EXPECT_NE(llvm::toString(missingEntry.takeError())
                .find("target entry symbol is not defined"),
            std::string::npos);

  llvm::Expected<wafer::compiler::VerifiedTargetModule> dataEntry =
      wafer::compiler::detail::verifyLinkedTargetModuleForTesting(
          modulePath, "data_entry", kProfile,
          wafer::TargetLaunchABIId::perRankPointerBlockV1());
  ASSERT_FALSE(static_cast<bool>(dataEntry));
  EXPECT_NE(llvm::toString(dataEntry.takeError())
                .find("target entry symbol is defined but is not a function"),
            std::string::npos);

  llvm::Expected<wafer::compiler::VerifiedTargetModule> localEntry =
      wafer::compiler::detail::verifyLinkedTargetModuleForTesting(
          modulePath, "local_entry", kProfile,
          wafer::TargetLaunchABIId::perRankPointerBlockV1());
  ASSERT_FALSE(static_cast<bool>(localEntry));
  EXPECT_NE(llvm::toString(localEntry.takeError())
                .find("target entry symbol is a function but is not externally "
                      "visible"),
            std::string::npos);
}

TEST(TargetArtifactTest, ReadbackRejectsNonELFAndWrongArchitecture) {
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
          nonELFPath, "kernel_entry", kProfile,
          wafer::TargetLaunchABIId::perRankPointerBlockV1());
  ASSERT_FALSE(static_cast<bool>(nonELF));
  EXPECT_NE(llvm::toString(nonELF.takeError()).find("target module is not ELF"),
            std::string::npos);

  llvm::Expected<wafer::compiler::VerifiedTargetModule> wrongArchitecture =
      wafer::compiler::detail::verifyLinkedTargetModuleForTesting(
          "/bin/true", "kernel_entry", kProfile,
          wafer::TargetLaunchABIId::perRankPointerBlockV1());
  ASSERT_FALSE(static_cast<bool>(wrongArchitecture));
  EXPECT_NE(llvm::toString(wrongArchitecture.takeError())
                .find("target module is not RISC-V 64-bit ELF"),
            std::string::npos);
}

} // namespace
