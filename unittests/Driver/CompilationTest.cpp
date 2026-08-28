//===- CompilationTest.cpp - Typed compiler request tests ----------------===//

#include "Wafer/Driver/Compilation.h"
#include "Wafer/CodeGen/TargetCodeGen.h"
#include "Wafer/Driver/CompilationResult.h"
#include "Wafer/Package/Manifest/PackageManifest.h"
#include "Wafer/Support/CompileTiming.h"
#include "Wafer/Target/Core/RuntimeLaunchContract.h"
#include "Wafer/Target/Core/TargetIdentity.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

TEST(CompilationTest, ExecutionConfigSeparatesPartitionsFromTiles) {
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(1);
  ASSERT_TRUE(static_cast<bool>(config));
  EXPECT_EQ(config->getNumPartitions(), 1);
  EXPECT_EQ(config->getTileCount(), 16);

  for (int64_t rejected :
       {std::numeric_limits<int64_t>::min(), int64_t{-1}, int64_t{0},
        int64_t{2}, int64_t{8}, int64_t{15}, int64_t{16}, int64_t{17},
        std::numeric_limits<int64_t>::max()}) {
    auto rejectedConfig =
        wafer::compiler::ExecutionConfig::createForSingleCard(rejected);
    ASSERT_FALSE(static_cast<bool>(rejectedConfig));
    EXPECT_FALSE(llvm::toString(rejectedConfig.takeError()).empty());
  }
}

TEST(CompilationTest, ExecutionConfigEqualityCoversTypedDomains) {
  auto first = wafer::compiler::ExecutionConfig::createForSingleCard(1);
  auto same = wafer::compiler::ExecutionConfig::createForSingleCard(1);
  ASSERT_TRUE(static_cast<bool>(first));
  ASSERT_TRUE(static_cast<bool>(same));
  EXPECT_EQ(*first, *same);
  EXPECT_EQ(first->getNumPartitions(), 1);
  EXPECT_EQ(first->getTileCount(), 16);
}

TEST(CompilationTest, CompilationRequestOwnsSourceAndHasNoImplicitDefaults) {
  static_assert(
      !std::is_default_constructible_v<wafer::compiler::ExecutionConfig>);
  static_assert(
      !std::is_default_constructible_v<wafer::compiler::CompilationRequest>);
  static_assert(
      !std::is_copy_constructible_v<wafer::compiler::CompilationRequest>);
  static_assert(
      std::is_move_constructible_v<wafer::compiler::CompilationRequest>);
  static_assert(
      !std::is_default_constructible_v<wafer::compiler::TileExecutable>);
  static_assert(!std::is_copy_constructible_v<wafer::compiler::TileExecutable>);
  static_assert(std::is_move_constructible_v<wafer::compiler::TileExecutable>);
  static_assert(
      !std::is_default_constructible_v<wafer::compiler::DeviceExecutable>);
  static_assert(
      !std::is_copy_constructible_v<wafer::compiler::DeviceExecutable>);
  static_assert(
      std::is_move_constructible_v<wafer::compiler::DeviceExecutable>);
  static_assert(
      !std::is_default_constructible_v<wafer::compiler::TargetLLVMModule>);
  static_assert(
      !std::is_copy_constructible_v<wafer::compiler::TargetLLVMModule>);
  static_assert(
      std::is_move_constructible_v<wafer::compiler::TargetLLVMModule>);
  static_assert(
      !std::is_default_constructible_v<wafer::compiler::TargetLLVMModules>);
  static_assert(
      !std::is_copy_constructible_v<wafer::compiler::TargetLLVMModules>);
  static_assert(
      std::is_move_constructible_v<wafer::compiler::TargetLLVMModules>);
  static_assert(
      !std::is_default_constructible_v<wafer::compiler::LinkedTargetModules>);
  static_assert(
      !std::is_copy_constructible_v<wafer::compiler::LinkedTargetModules>);
  static_assert(
      std::is_move_constructible_v<wafer::compiler::LinkedTargetModules>);
  static_assert(
      !std::is_default_constructible_v<wafer::compiler::ExecutablePackage>);
  static_assert(
      !std::is_copy_constructible_v<wafer::compiler::ExecutablePackage>);
  static_assert(
      std::is_move_constructible_v<wafer::compiler::ExecutablePackage>);
  static_assert(!std::is_default_constructible_v<
                wafer::compiler::ProfileInstrumentationProduct>);
  static_assert(!std::is_copy_constructible_v<
                wafer::compiler::ProfileInstrumentationProduct>);
  static_assert(std::is_move_constructible_v<
                wafer::compiler::ProfileInstrumentationProduct>);
  static_assert(
      !std::is_default_constructible_v<wafer::compiler::CompilationResult>);
  static_assert(
      !std::is_copy_constructible_v<wafer::compiler::CompilationResult>);
  static_assert(
      std::is_move_constructible_v<wafer::compiler::CompilationResult>);

  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(1);
  ASSERT_TRUE(static_cast<bool>(config));
  std::string source = "/tmp/source.program";
  auto request =
      wafer::compiler::CompilationRequest::create(source, std::move(*config));
  ASSERT_TRUE(static_cast<bool>(request));
  source.assign("/tmp/changed-after-request-construction.program");
  EXPECT_EQ(request->getSourceProgramDirectory(), "/tmp/source.program");
  EXPECT_EQ(request->getExecutionConfig().getNumPartitions(), 1);
  EXPECT_EQ(request->getExecutionConfig().getTileCount(), 16);
  EXPECT_EQ(request->getExecutionConfig().getTargetIdentityId(),
            wafer::TargetIdentityId::waferTx81SingleCard());
}

TEST(CompilationTest, CompilationRequestRejectsEmptySourceLocator) {
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(1);
  ASSERT_TRUE(static_cast<bool>(config));
  auto request = wafer::compiler::CompilationRequest::create("", *config);
  ASSERT_FALSE(static_cast<bool>(request));
  EXPECT_FALSE(llvm::toString(request.takeError()).empty());
}

TEST(CompilationTest, ProfileOptionsRequireCompleteTileKernelDomain) {
  auto fullCard = wafer::compiler::ExecutionConfig::createForSingleCard(1);
  ASSERT_TRUE(static_cast<bool>(fullCard));
  auto accepted = wafer::compiler::CompilationOptions::profile(*fullCard);
  ASSERT_TRUE(static_cast<bool>(accepted));
  EXPECT_TRUE(accepted->shouldProduceProfileInstrumentation());
  EXPECT_FALSE(accepted->shouldReportDetailedTiming());
  wafer::OptimizationConfig none = wafer::OptimizationConfig::none();
  auto acceptedNone =
      wafer::compiler::CompilationOptions::profile(*fullCard, none);
  ASSERT_TRUE(static_cast<bool>(acceptedNone));
  EXPECT_EQ(acceptedNone->getOptimizationConfig(), none);
  auto acceptedTimed = wafer::compiler::CompilationOptions::profile(
      *fullCard, none, wafer::compiler::CompilationTimingMode::Detailed);
  ASSERT_TRUE(static_cast<bool>(acceptedTimed));
  EXPECT_TRUE(acceptedTimed->shouldReportDetailedTiming());

  EXPECT_FALSE(wafer::compiler::CompilationOptions::standard()
                   .shouldProduceProfileInstrumentation());
  EXPECT_FALSE(wafer::compiler::CompilationOptions::standard()
                   .shouldReportDetailedTiming());
  EXPECT_TRUE(wafer::compiler::CompilationOptions::standard()
                  .getOptimizationConfig()
                  .isNone());
  EXPECT_TRUE(wafer::compiler::CompilationOptions::standard(
                  wafer::OptimizationConfig::search(),
                  wafer::compiler::CompilationTimingMode::Detailed)
                  .shouldReportDetailedTiming());
}

TEST(CompilationTest, OptimizationConfigHasExactlySearchAndNonePolicies) {
  wafer::OptimizationConfig search = wafer::OptimizationConfig::search();
  wafer::OptimizationConfig none = wafer::OptimizationConfig::none();

  EXPECT_TRUE(search.isSearch());
  EXPECT_FALSE(search.isNone());
  EXPECT_FALSE(none.isSearch());
  EXPECT_TRUE(none.isNone());
  EXPECT_NE(search, none);

  wafer::compiler::CompilationOptions options =
      wafer::compiler::CompilationOptions::standard(none);
  EXPECT_EQ(options.getOptimizationConfig(), none);
}

TEST(CompilationTest, DetailedTimingAggregatesInvocationLocalSpans) {
  std::string output;
  llvm::raw_string_ostream diagnostics(output);
  auto session =
      std::make_shared<wafer::support::CompileTimingSession>(diagnostics);
  {
    wafer::support::ScopedCompileTimingActivation activation(session);
    wafer::support::ScopedCompileTimingSpan span("stage", "test-pipeline",
                                                 "test-item", "request=7");
    span.markFailed();
  }
  std::thread worker([session] {
    wafer::support::ScopedCompileTimingActivation activation(session);
    wafer::support::ScopedCompileTimingSpan span("analysis", "test-worker",
                                                 "parallel-item");
  });
  worker.join();
  std::vector<std::thread> workers;
  workers.reserve(16);
  for (size_t workerIndex = 0; workerIndex < 16; ++workerIndex) {
    workers.emplace_back([session] {
      wafer::support::ScopedCompileTimingActivation activation(session);
      for (size_t iteration = 0; iteration < 64; ++iteration) {
        wafer::support::ScopedCompileTimingSpan span("analysis", "test-worker",
                                                     "sharded-parallel-item");
      }
    });
  }
  for (std::thread &parallelWorker : workers)
    parallelWorker.join();
  session->finishAndPrintSummary();
  EXPECT_NE(output.find("compile-timing-summary-begin"), std::string::npos);
  size_t failedRow = output.find("| stage | test-pipeline | test-item | 1 |");
  ASSERT_NE(failedRow, std::string::npos);
  size_t failedRowEnd = output.find('\n', failedRow);
  ASSERT_NE(failedRowEnd, std::string::npos);
  llvm::StringRef failedRowText(output.data() + failedRow,
                                failedRowEnd - failedRow);
  EXPECT_TRUE(failedRowText.ends_with("| 1 |"));
  EXPECT_NE(output.find("| analysis | test-worker | parallel-item | 1 |"),
            std::string::npos);
  EXPECT_NE(
      output.find("| analysis | test-worker | sharded-parallel-item | 1024 |"),
      std::string::npos);
  EXPECT_NE(output.find("compile-timing-summary-end transaction_wall_ms="),
            std::string::npos);
}

TEST(CompilationTest, InternalEntryRejectsProfileOptionsBeforeTransaction) {
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(1);
  ASSERT_TRUE(static_cast<bool>(config));
  auto request = wafer::compiler::CompilationRequest::create(
      "/tmp/never-touched.program", std::move(*config));
  ASSERT_TRUE(static_cast<bool>(request));
  auto toolchain = wafer::compiler::TargetToolchain::create(
      "python3", "wafer_device_link.py", "clang++",
      "/tmp/never-touched-tx8-deps", "/tmp/never-touched-include",
      "/tmp/never-touched-crt.c", "/tmp/never-touched-crt-include");
  ASSERT_TRUE(static_cast<bool>(toolchain))
      << llvm::toString(toolchain.takeError());
  auto profileOptions = wafer::compiler::CompilationOptions::profile(
      request->getExecutionConfig());
  ASSERT_TRUE(static_cast<bool>(profileOptions));
  llvm::Expected<wafer::compiler::CompiledProgram> rejected =
      wafer::compiler::compileProgramWithTargetLLVMModules(
          std::move(*request), "/tmp/never-touched-output", "helper",
          *toolchain, *profileOptions, llvm::errs());
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("profile"),
            std::string::npos);
}

TEST(CompilationTest, CompilationFailureClassifiesStageAndRendersLog) {
  for (wafer::compiler::CompilationStage stage : {
           wafer::compiler::CompilationStage::SourceVerification,
           wafer::compiler::CompilationStage::SpmdPartitioning,
           wafer::compiler::CompilationStage::TensorProgramPreparation,
           wafer::compiler::CompilationStage::ExecutableCompilation,
           wafer::compiler::CompilationStage::TargetCodeGeneration,
           wafer::compiler::CompilationStage::PackageAssembly,
           wafer::compiler::CompilationStage::PackageCommit,
       }) {
    wafer::compiler::CompilationFailure failure(stage);
    EXPECT_EQ(failure.getStage(), stage);
    std::string rendered;
    llvm::raw_string_ostream stream(rendered);
    failure.log(stream);
    EXPECT_EQ(rendered,
              std::string("compilation failed at the ") +
                  wafer::compiler::stringifyCompilationStage(stage).str() +
                  " stage");
    EXPECT_EQ(failure.convertToErrorCode(),
              llvm::errc::operation_not_permitted);
  }
}

TEST(CompilationTest, ExecutablePackageOwnsExactMemberSnapshots) {
  llvm::SmallString<256> temporaryDirectory;
  ASSERT_FALSE(llvm::sys::fs::createUniqueDirectory("wafer-package-members",
                                                    temporaryDirectory));
  auto cleanup = llvm::make_scope_exit(
      [&]() { (void)llvm::sys::fs::remove_directories(temporaryDirectory); });

  // Write the module and canonical empty program-data members.
  llvm::SmallString<256> modulesDirectory(temporaryDirectory);
  llvm::sys::path::append(modulesDirectory, "modules");
  ASSERT_FALSE(llvm::sys::fs::create_directories(modulesDirectory));
  std::string moduleBytes = "\x7f"
                            "ELF-wafer-test-module";
  moduleBytes.resize(32 * 1024, 'm');
  llvm::SmallString<256> modulePath(modulesDirectory);
  llvm::sys::path::append(modulePath, "tile_00000.so");
  {
    std::error_code error;
    llvm::raw_fd_ostream output(modulePath, error, llvm::sys::fs::OF_None);
    ASSERT_FALSE(error);
    output << moduleBytes;
    output.close();
    ASSERT_FALSE(output.has_error());
  }
  llvm::SmallString<256> dataDirectory(temporaryDirectory);
  llvm::sys::path::append(dataDirectory, "data");
  ASSERT_FALSE(llvm::sys::fs::create_directories(dataDirectory));
  llvm::SmallString<256> dataPath(dataDirectory);
  llvm::sys::path::append(dataPath, "program-data.bin");
  {
    std::error_code error;
    llvm::raw_fd_ostream output(dataPath, error, llvm::sys::fs::OF_None);
    ASSERT_FALSE(error);
  }
  llvm::SHA256 hasher;
  hasher.update(moduleBytes);
  std::string moduleDigest =
      "sha256:" + llvm::toHex(hasher.final(), /*LowerCase=*/true);

  wafer::RuntimeLaunchContract launch =
      llvm::cantFail(wafer::RuntimeLaunchContract::createKernel(
          wafer::KernelLaunchForm::Grid,
          wafer::KernelEntryABI::TileMajorPointerTable,
          {wafer::RuntimeLaunchPhaseRole::Main}));
  wafer::runtime::PackageManifest manifest(
      wafer::kCurrentTargetIdentity, wafer::kCurrentKernelRuntimeABI,
      std::move(launch), wafer::kCurrentTargetModuleFormat);
  manifest.program = wafer::runtime::ProgramId(0);
  manifest.cardCount = 1;
  manifest.tileCount = 16;
  manifest.programData = {
      "data/program-data.bin", 0, 1,
      "sha256:"
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"};
  manifest.modules = {
      {wafer::runtime::ModuleId(0),
       "modules/tile_00000.so",
       moduleDigest,
       wafer::kCurrentTargetModuleFormat.str(),
       {{wafer::runtime::PackageModuleExportRole::Main, "main"}}},
  };
  for (int64_t launchSlot = 0; launchSlot < 16; ++launchSlot) {
    wafer::runtime::PackageEntrypointRecord entry;
    entry.id = wafer::runtime::EntryId(launchSlot);
    entry.cardId = wafer::CardId(0);
    entry.tileId = wafer::TileId(launchSlot);
    entry.launchSlot = wafer::runtime::LaunchSlotId(launchSlot);
    entry.module = wafer::runtime::ModuleId(0);
    entry.completion =
        wafer::runtime::PackageEntryCompletionKind::ReturnAfterLocalDrain;
    entry.arguments.push_back({0, wafer::runtime::WorkspaceArgument{512, 256},
                               wafer::runtime::PackageAccessMode::ReadWrite});
    entry.transport = wafer::runtime::NoTransportRequirements{};
    manifest.entries.push_back(std::move(entry));
  }

  llvm::Expected<wafer::runtime::VerifiedPackageManifest> verified =
      wafer::runtime::verifyPackageManifest(std::move(manifest),
                                            temporaryDirectory);
  ASSERT_TRUE(static_cast<bool>(verified))
      << llvm::toString(verified.takeError());
  {
    llvm::SmallString<256> manifestPath(temporaryDirectory);
    llvm::sys::path::append(manifestPath,
                            wafer::runtime::kPackageManifestFileName);
    std::error_code error;
    llvm::raw_fd_ostream output(manifestPath, error, llvm::sys::fs::OF_Text);
    ASSERT_FALSE(error);
    output << wafer::runtime::serializeCanonicalPackageJson(*verified);
    output.close();
    ASSERT_FALSE(output.has_error());
  }

  auto countOpenDescriptors = []() -> size_t {
    size_t count = 0;
    std::error_code error;
    for (llvm::sys::fs::directory_iterator iterator("/proc/self/fd", error),
         end;
         iterator != end && !error; iterator.increment(error))
      ++count;
    EXPECT_FALSE(error);
    return count;
  };
  const size_t descriptorsBefore = countOpenDescriptors();
  llvm::Expected<wafer::runtime::ExecutablePackage> loaded =
      wafer::runtime::loadExecutablePackage(temporaryDirectory);
  ASSERT_TRUE(static_cast<bool>(loaded)) << llvm::toString(loaded.takeError());
  wafer::compiler::ExecutablePackage package = std::move(*loaded);
  EXPECT_EQ(countOpenDescriptors(), descriptorsBefore);

  const std::string canonicalManifest =
      wafer::runtime::serializeCanonicalPackageJson(
          package.getVerifiedManifest());

  // Overwrite the same inodes after binding. Owned snapshots must not observe
  // the writes (a retained read-only mmap would fail this contract).
  const std::string mutatedModule(moduleBytes.size(), 'x');
  for (const auto &mutation :
       std::vector<std::pair<llvm::StringRef, llvm::StringRef>>{
           {modulePath, mutatedModule}, {dataPath, "mutated-program-data"}}) {
    std::error_code error;
    llvm::raw_fd_ostream output(
        mutation.first, error, llvm::sys::fs::CD_OpenExisting,
        llvm::sys::fs::FA_Write, llvm::sys::fs::OF_None);
    ASSERT_FALSE(error);
    output << mutation.second;
    output.close();
    ASSERT_FALSE(output.has_error());
  }
  EXPECT_EQ(package.getModuleBuffers().front()->getBuffer(), moduleBytes);
  EXPECT_EQ(package.getModuleBuffers().front()->getBufferKind(),
            llvm::MemoryBuffer::MemoryBuffer_Malloc);
  EXPECT_EQ(package.getProgramDataBuffer().getBufferSize(), size_t{0});

  // Remove every path: the snapshots remain readable and unchanged.
  ASSERT_FALSE(llvm::sys::fs::remove_directories(temporaryDirectory));

  EXPECT_EQ(package.getRootDirectory(), temporaryDirectory.str().str());
  EXPECT_EQ(package.getModuleBuffers().size(), size_t{1});
  EXPECT_EQ(package.getModuleBuffers().front()->getBuffer(), moduleBytes);
  EXPECT_EQ(package.getProgramDataBuffer().getBufferSize(), size_t{0});
  // The owned manifest member carries exactly the canonical bytes of the
  // verified manifest this package was bound to.
  EXPECT_EQ(package.getManifestBuffer().getBuffer(), canonicalManifest);
}

} // namespace
