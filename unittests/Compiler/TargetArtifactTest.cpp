//===- TargetArtifactTest.cpp - Linked target readback tests -------------===//

#include "../../lib/Wafer/Compiler/TargetArtifactInternal.h"

#include "Wafer/Target/TargetProfile.h"
#include "Wafer/Support/OptimizationInvocation.h"
#include "Wafer/Support/OptimizationMechanism.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <string>
#include <type_traits>
#include <mutex>
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

class BackendInvocationRecorder final
    : public wafer::OptimizationInvocationRecorder {
public:
  bool beginInvocation(const wafer::InvocationPreparationV1 &preparation,
                       std::string *diagnostic = nullptr) override {
    std::lock_guard<std::mutex> lock(mutex);
    return journal.beginInvocation(preparation, diagnostic);
  }
  bool recordInvocationTerminal(
      const wafer::InvocationTelemetryV1 &telemetry,
      std::string *diagnostic = nullptr) override {
    std::lock_guard<std::mutex> lock(mutex);
    return journal.recordInvocationTerminal(telemetry, diagnostic);
  }
  std::vector<wafer::InvocationTelemetryV1> records() {
    std::lock_guard<std::mutex> lock(mutex);
    return journal.committedTerminals();
  }

  std::mutex mutex;
  wafer::OptimizationInvocationJournalV1 journal;
};

static wafer::OptimizationInvocationScopeContextV1 backendRecordingScope() {
  wafer::OptimizationInvocationScopeContextV1 scope;
  scope.scopeKind = wafer::InvocationScopeKindV1::ProductionCompile;
  scope.scopeDigest.fill(0x2a);
  return scope;
}

constexpr wafer::TargetProfileId kProfile =
    wafer::TargetProfileId::waferTx81SingleCardKernelV1();

static llvm::SmallString<256>
pathInDirectory(llvm::StringRef directory, llvm::StringRef filename) {
  llvm::SmallString<256> path(directory);
  llvm::sys::path::append(path, filename);
  return path;
}

static int linkTargetModule(llvm::StringRef outputPath) {
  std::string python = WAFER_TEST_PYTHON_EXECUTABLE;
  std::string script = WAFER_TEST_DEVICE_LINKER_SCRIPT;
  std::string input = WAFER_TEST_TARGET_DATA_ENTRY_LLVM_IR;
  std::string output = outputPath.str();
  std::string clang = WAFER_TEST_LLVM_CLANGXX;
  llvm::SmallVector<llvm::StringRef, 10> arguments = {
      python, script, "--llvm-ir", input, "--output", output,
      "--llvm-clangxx", clang};
  return llvm::sys::ExecuteAndWait(python, arguments);
}

TEST(TargetArtifactTest, PublicVerifiedModuleCannotBeForgedOrDefaulted) {
  static_assert(
      !std::is_default_constructible_v<wafer::compiler::VerifiedTargetModule>);
}

TEST(TargetArtifactTest, LinkedRiscvELFReadbackCarriesTypedProfileFacts) {
  llvm::SmallString<256> temporaryDirectory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "wafer-target-readback", temporaryDirectory);
  ASSERT_FALSE(error) << error.message();
  auto cleanup = llvm::make_scope_exit([&]() {
    (void)llvm::sys::fs::remove_directories(temporaryDirectory);
  });

  llvm::SmallString<256> modulePath =
      pathInDirectory(temporaryDirectory, "kernel.so");
  ASSERT_EQ(linkTargetModule(modulePath), 0);

  llvm::Expected<wafer::compiler::VerifiedTargetModule> module =
      wafer::compiler::detail::verifyLinkedTargetModuleForTesting(
          modulePath, "kernel_entry", kProfile);
  ASSERT_TRUE(static_cast<bool>(module))
      << llvm::toString(module.takeError());
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
          modulePath, "missing_entry", kProfile);
  ASSERT_FALSE(static_cast<bool>(missingEntry));
  EXPECT_NE(llvm::toString(missingEntry.takeError())
                .find("target entry symbol is not defined"),
            std::string::npos);

  llvm::Expected<wafer::compiler::VerifiedTargetModule> dataEntry =
      wafer::compiler::detail::verifyLinkedTargetModuleForTesting(
          modulePath, "data_entry", kProfile);
  ASSERT_FALSE(static_cast<bool>(dataEntry));
  EXPECT_NE(llvm::toString(dataEntry.takeError())
                .find("target entry symbol is defined but is not a function"),
            std::string::npos);

  llvm::Expected<wafer::compiler::VerifiedTargetModule> localEntry =
      wafer::compiler::detail::verifyLinkedTargetModuleForTesting(
          modulePath, "local_entry", kProfile);
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
  auto cleanup = llvm::make_scope_exit([&]() {
    (void)llvm::sys::fs::remove_directories(temporaryDirectory);
  });

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
          nonELFPath, "kernel_entry", kProfile);
  ASSERT_FALSE(static_cast<bool>(nonELF));
  EXPECT_NE(llvm::toString(nonELF.takeError()).find("target module is not ELF"),
            std::string::npos);

  llvm::Expected<wafer::compiler::VerifiedTargetModule> wrongArchitecture =
      wafer::compiler::detail::verifyLinkedTargetModuleForTesting(
          "/bin/true", "kernel_entry", kProfile);
  ASSERT_FALSE(static_cast<bool>(wrongArchitecture));
  EXPECT_NE(llvm::toString(wrongArchitecture.takeError())
                .find("target module is not RISC-V 64-bit ELF"),
            std::string::npos);
}

TEST(TargetArtifactTest, DeviceBackendTelemetryUsesActualExecutedArgv) {
  llvm::SmallString<256> temporaryDirectory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "wafer-target-action-observation", temporaryDirectory);
  ASSERT_FALSE(error) << error.message();
  auto cleanup = llvm::make_scope_exit([&]() {
    (void)llvm::sys::fs::remove_directories(temporaryDirectory);
  });

  llvm::Expected<wafer::compiler::TargetToolchain> toolchain =
      wafer::compiler::TargetToolchain::create(
          WAFER_TEST_PYTHON_EXECUTABLE, WAFER_TEST_DEVICE_LINKER_SCRIPT);
  ASSERT_TRUE(static_cast<bool>(toolchain))
      << llvm::toString(toolchain.takeError());
  auto recorder = std::make_shared<BackendInvocationRecorder>();
  wafer::ScopedOptimizationInvocationRecorder scoped(recorder,
                                                       backendRecordingScope());
  ASSERT_TRUE(scoped.installed());

  llvm::SmallString<256> modulePath =
      pathInDirectory(temporaryDirectory, "kernel.so");
  llvm::SmallString<256> objectPath =
      pathInDirectory(temporaryDirectory, "kernel.o");
  llvm::SmallString<256> crtPath =
      pathInDirectory(temporaryDirectory, "wafer_crt.o");
  llvm::Error linkError = wafer::compiler::detail::runDeviceLink(
      *toolchain, WAFER_TEST_TARGET_DATA_ENTRY_LLVM_IR, modulePath, objectPath,
      crtPath, /*logicalRank=*/0);
  ASSERT_FALSE(static_cast<bool>(linkError)) << llvm::toString(std::move(linkError));

  llvm::SmallString<256> rankOneModulePath =
      pathInDirectory(temporaryDirectory, "kernel-rank-one.so");
  llvm::SmallString<256> rankOneObjectPath =
      pathInDirectory(temporaryDirectory, "kernel-rank-one.o");
  llvm::SmallString<256> rankOneCRTPath =
      pathInDirectory(temporaryDirectory, "wafer-crt-rank-one.o");
  linkError = wafer::compiler::detail::runDeviceLink(
      *toolchain, WAFER_TEST_TARGET_DATA_ENTRY_LLVM_IR, rankOneModulePath,
      rankOneObjectPath, rankOneCRTPath, /*logicalRank=*/1);
  ASSERT_FALSE(static_cast<bool>(linkError))
      << llvm::toString(std::move(linkError));

  std::vector<wafer::InvocationTelemetryV1> records = recorder->records();
  ASSERT_EQ(records.size(), 6u);
  std::sort(records.begin(), records.end(), [](const auto &lhs, const auto &rhs) {
    return std::tie(lhs.identity.mechanismKey,
                    lhs.identity.invocationOrdinal) <
           std::tie(rhs.identity.mechanismKey,
                    rhs.identity.invocationOrdinal);
  });
  EXPECT_EQ(records[0].identity.mechanismKey,
            wafer::mechanism::DeviceObjectCompilation);
  EXPECT_EQ(records[2].identity.mechanismKey,
            wafer::mechanism::DeviceRuntimeCompilation);
  EXPECT_EQ(records[4].identity.mechanismKey,
            wafer::mechanism::DeviceGarbageCollectionLink);
  for (auto [index, record] : llvm::enumerate(records)) {
    EXPECT_EQ(record.identity.invocationOrdinal, index % 2);
    EXPECT_EQ(record.outcome, wafer::InvocationOutcome::Applied);
    ASSERT_EQ(record.backendActions.size(), 1u);
    EXPECT_EQ(record.backendActions.front().terminalStatus,
              wafer::BackendActionStatusV1::Success);
    ASSERT_FALSE(record.backendActions.front().argv.empty());
  }
  EXPECT_NE(std::find(records[0].backendActions[0].argv.begin(),
                      records[0].backendActions[0].argv.end(), "-O2"),
            records[0].backendActions[0].argv.end());
  EXPECT_NE(std::find(records[2].backendActions[0].argv.begin(),
                      records[2].backendActions[0].argv.end(), "-O2"),
            records[2].backendActions[0].argv.end());
  EXPECT_NE(std::find(records[4].backendActions[0].argv.begin(),
                      records[4].backendActions[0].argv.end(),
                      "-Wl,--gc-sections"),
            records[4].backendActions[0].argv.end());
}

} // namespace
