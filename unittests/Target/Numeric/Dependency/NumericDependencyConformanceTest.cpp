//===- NumericDependencyConformanceTest.cpp -------------------------------===//

#include "Wafer/Target/Numeric/Dependency/NumericDependencyConformance.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

namespace {

using wafer::NumericDependencyConformanceErrorCode;
using wafer::NumericDependencyConformanceRecord;
using wafer::NumericDependencyExecutionBinding;
using wafer::NumericLoadedObject;
using wafer::NumericLoadedObjectKind;
using wafer::NumericLoadedObjectProvider;

std::string digestBytes(llvm::StringRef bytes) {
  llvm::SHA256 hasher;
  hasher.update(bytes);
  return llvm::toHex(hasher.final(), /*LowerCase=*/true);
}

std::string digestFile(llvm::StringRef path) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path, /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false);
  EXPECT_TRUE(static_cast<bool>(buffer));
  return buffer ? digestBytes((*buffer)->getBuffer()) : std::string(64, '0');
}

template <typename T>
void expectError(llvm::Expected<T> result,
                 NumericDependencyConformanceErrorCode code) {
  if (result) {
    ADD_FAILURE() << "expected numeric dependency conformance failure";
    return;
  }
  bool handled = false;
  llvm::handleAllErrors(
      result.takeError(),
      [&](const wafer::NumericDependencyConformanceError &error) {
        handled = true;
        EXPECT_EQ(error.getCode(), code) << error.getDetail().str();
      });
  EXPECT_TRUE(handled);
}

class StaticLoadedObjectProvider final : public NumericLoadedObjectProvider {
public:
  StaticLoadedObjectProvider(NumericLoadedObject mpfr, NumericLoadedObject gmp)
      : mpfr(std::move(mpfr)), gmp(std::move(gmp)) {}

  llvm::Expected<NumericLoadedObject>
  getLoadedObject(NumericLoadedObjectKind kind) const override {
    return kind == NumericLoadedObjectKind::MPFR ? mpfr : gmp;
  }

private:
  NumericLoadedObject mpfr;
  NumericLoadedObject gmp;
};

class NumericDependencyConformanceTest : public testing::Test {
protected:
  void SetUp() override {
    ASSERT_FALSE(llvm::sys::fs::createUniqueDirectory(
        "wafer-numeric-conformance", root));
    recordPath = path("numeric-model-deps.json");
    generateCurrentPythonFixture();
    ASSERT_FALSE(HasFatalFailure());
    loadRecord();
  }

  void TearDown() override { llvm::sys::fs::remove_directories(root); }

  std::string path(llvm::StringRef relative) const {
    llvm::SmallString<256> result(root);
    llvm::sys::path::append(result, relative);
    return result.str().str();
  }

  void writeFile(llvm::StringRef relative, llvm::StringRef contents,
                 uint32_t mode = 0644) {
    const std::string destination = path(relative);
    llvm::SmallString<256> parent(destination);
    llvm::sys::path::remove_filename(parent);
    ASSERT_FALSE(llvm::sys::fs::create_directories(parent));
    std::error_code error;
    llvm::raw_fd_ostream output(destination, error, llvm::sys::fs::OF_None);
    ASSERT_FALSE(error) << error.message();
    output << contents;
    output.close();
    ASSERT_FALSE(llvm::sys::fs::setPermissions(
        destination, static_cast<llvm::sys::fs::perms>(mode)));
  }

  void generateCurrentPythonFixture() {
    llvm::SmallString<256> repository(WAFER_TEST_SOURCE_DIR);
    llvm::SmallString<256> producer(repository);
    llvm::sys::path::append(producer, "tools", "test_numeric_deps.py");
    ASSERT_TRUE(llvm::sys::fs::exists(producer))
        << "configured numeric dependency fixture producer is missing";

    llvm::ErrorOr<std::string> python = llvm::sys::findProgramByName("python3");
    ASSERT_TRUE(static_cast<bool>(python))
        << "python3 is required for the current producer fixture";
    const std::string program = *python;
    const std::string script =
        "import pathlib,sys;"
        "repo=pathlib.Path(sys.argv[1]);"
        "sys.path.insert(0,str(repo/'tools'));"
        "from test_numeric_deps import make_record,make_versions;"
        "root=pathlib.Path(sys.argv[2]);"
        "versions,archives=make_versions(root);"
        "make_record(root,versions,archives)";
    const std::vector<std::string> argumentStorage = {
        program, "-c", script, repository.str().str(), root.str().str()};
    llvm::SmallVector<llvm::StringRef, 8> arguments;
    for (const std::string &argument : argumentStorage)
      arguments.push_back(argument);
    std::string errorMessage;
    bool executionFailed = false;
    const int exitCode = llvm::sys::ExecuteAndWait(
        program, arguments, std::nullopt, {}, /*SecondsToWait=*/120,
        /*MemoryLimit=*/0, &errorMessage, &executionFailed);
    ASSERT_FALSE(executionFailed) << errorMessage;
    ASSERT_EQ(exitCode, 0) << errorMessage;
    ASSERT_TRUE(llvm::sys::fs::exists(recordPath));
  }

  void loadRecord() {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
        llvm::MemoryBuffer::getFile(recordPath, /*IsText=*/false,
                                    /*RequiresNullTerminator=*/false);
    ASSERT_TRUE(static_cast<bool>(buffer));
    llvm::Expected<llvm::json::Value> parsed =
        llvm::json::parse((*buffer)->getBuffer());
    ASSERT_TRUE(static_cast<bool>(parsed))
        << (parsed ? std::string() : llvm::toString(parsed.takeError()));
    record = std::move(*parsed);
    recordDigest = digestBytes((*buffer)->getBuffer());
    filePaths.clear();
    fileDigests.clear();
    llvm::json::Object *fileEntries = top().getObject("artifacts");
    ASSERT_NE(fileEntries, nullptr);
    for (auto &entry : *fileEntries) {
      llvm::json::Object *fileRecord = entry.second.getAsObject();
      ASSERT_NE(fileRecord, nullptr);
      std::optional<llvm::StringRef> relative = fileRecord->getString("path");
      std::optional<llvm::StringRef> digest = fileRecord->getString("sha256");
      ASSERT_TRUE(relative.has_value());
      ASSERT_TRUE(digest.has_value());
      filePaths[entry.first.str()] = path(*relative);
      fileDigests[entry.first.str()] = digest->str();
    }
  }

  std::string writeRecord() {
    const std::string text = llvm::formatv("{0:2}", record).str() + "\n";
    writeFile("numeric-model-deps.json", text);
    return digestBytes(text);
  }

  std::string writeRawRecord(llvm::StringRef text) {
    writeFile("numeric-model-deps.json", text);
    return digestBytes(text);
  }

  llvm::Expected<NumericDependencyConformanceRecord>
  read(llvm::StringRef expectedDigest) const {
    return wafer::readNumericDependencyConformanceRecord(root, recordPath,
                                                         expectedDigest);
  }

  llvm::json::Object &top() { return *record.getAsObject(); }

  llvm::json::Object &fileRecord(llvm::StringRef name) {
    llvm::json::Object *fileEntries = top().getObject("artifacts");
    EXPECT_NE(fileEntries, nullptr);
    llvm::json::Object *record = fileEntries->getObject(name);
    EXPECT_NE(record, nullptr);
    return *record;
  }

  llvm::json::Object &build() {
    llvm::json::Object *buildConfig = top().getObject("build");
    EXPECT_NE(buildConfig, nullptr);
    return *buildConfig;
  }

  llvm::json::Array &gates() {
    llvm::json::Object *conformance = top().getObject("conformance");
    EXPECT_NE(conformance, nullptr);
    llvm::json::Array *value = conformance->getArray("gates");
    EXPECT_NE(value, nullptr);
    return *value;
  }

  llvm::json::Object &gate(llvm::StringRef name) {
    for (llvm::json::Value &value : gates()) {
      llvm::json::Object *candidate = value.getAsObject();
      if (candidate && candidate->getString("name") == name)
        return *candidate;
    }
    ADD_FAILURE() << "missing fixture gate " << name.str();
    return *gates().front().getAsObject();
  }

  void tamperFileWithoutChangingSize(llvm::StringRef relative,
                                     uint32_t mode = 0644) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
        llvm::MemoryBuffer::getFile(path(relative), /*IsText=*/false,
                                    /*RequiresNullTerminator=*/false);
    ASSERT_TRUE(static_cast<bool>(buffer));
    std::string contents = (*buffer)->getBuffer().str();
    ASSERT_FALSE(contents.empty());
    contents[0] = contents[0] == 'X' ? 'Y' : 'X';
    writeFile(relative, contents, mode);
  }

  llvm::SmallString<256> root;
  std::string recordPath;
  llvm::json::Value record = nullptr;
  std::string recordDigest;
  std::map<std::string, std::string> filePaths;
  std::map<std::string, std::string> fileDigests;
};

TEST_F(NumericDependencyConformanceTest,
       ReadsPythonProducedCurrentRecordAndLoadedObjects) {
  llvm::Expected<NumericDependencyConformanceRecord> parsed =
      read(recordDigest);
  ASSERT_TRUE(static_cast<bool>(parsed))
      << (parsed ? std::string() : llvm::toString(parsed.takeError()));
  EXPECT_EQ(parsed->getSources().size(), 5u);
  EXPECT_EQ(parsed->getFiles().size(), 20u);
  EXPECT_EQ(parsed->getTools().size(), 10u);
  EXPECT_EQ(parsed->getEnvironments().size(), 4u);
  EXPECT_EQ(parsed->getLicenses().size(), 9u);
  EXPECT_EQ(parsed->getGates().size(), 23u);

  const wafer::NumericDependencyEnvironmentRecord *base =
      parsed->findEnvironment("base");
  ASSERT_NE(base, nullptr);
  EXPECT_NE(llvm::find_if(base->variables,
                          [](const auto &entry) {
                            return entry.first == "SOURCE_DATE_EPOCH" &&
                                   entry.second == "0";
                          }),
            base->variables.end());
  EXPECT_NE(llvm::find_if(base->variables,
                          [](const auto &entry) {
                            return entry.first == "TMPDIR" &&
                                   entry.second == "${NUMERIC_ROOT}/build/tmp";
                          }),
            base->variables.end());

  StaticLoadedObjectProvider provider(
      {NumericLoadedObjectKind::MPFR, filePaths["mpfr-soname"],
       fileDigests["mpfr-soname"]},
      {NumericLoadedObjectKind::GMP, filePaths["gmp"], fileDigests["gmp"]});
  llvm::Expected<NumericDependencyExecutionBinding> execution =
      wafer::bindNumericDependenciesToLoadedObjects(*parsed, provider);
  ASSERT_TRUE(static_cast<bool>(execution))
      << (execution ? std::string() : llvm::toString(execution.takeError()));
  EXPECT_EQ(execution->getRecordSHA256(), recordDigest);
  EXPECT_EQ(execution->getBindingSHA256().size(), 64u);
}

TEST_F(NumericDependencyConformanceTest, RejectsRecordDigestMismatch) {
  expectError(read(std::string(64, '0')),
              NumericDependencyConformanceErrorCode::RecordDigestMismatch);
}

TEST_F(NumericDependencyConformanceTest, RejectsTamperedFileHash) {
  tamperFileWithoutChangingSize("install/softfloat/lib/libsoftfloat.a");
  expectError(read(recordDigest),
              NumericDependencyConformanceErrorCode::DigestMismatch);
}

TEST_F(NumericDependencyConformanceTest, RejectsTamperedSourceTree) {
  tamperFileWithoutChangingSize("sources/SoftFloat-3e/source.txt");
  expectError(read(recordDigest),
              NumericDependencyConformanceErrorCode::SourceTreeMismatch);
}

TEST_F(NumericDependencyConformanceTest, RejectsEscapingFilePath) {
  fileRecord("m4")["path"] = "../outside";
  expectError(read(writeRecord()),
              NumericDependencyConformanceErrorCode::UnsafePath);
}

TEST_F(NumericDependencyConformanceTest, RejectsMisplacedManagedFile) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path("install/m4/bin/m4"), /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false);
  ASSERT_TRUE(static_cast<bool>(buffer));
  writeFile("install/m4/bin/m4-copy", (*buffer)->getBuffer(), 0755);
  fileRecord("m4")["path"] = "install/m4/bin/m4-copy";
  expectError(read(writeRecord()),
              NumericDependencyConformanceErrorCode::PolicyMismatch);
}

TEST_F(NumericDependencyConformanceTest, RejectsSymlinkFile) {
  const std::string original = path("install/m4/bin/m4");
  const std::string backing = path("install/m4/bin/m4-backing");
  ASSERT_FALSE(llvm::sys::fs::rename(original, backing));
  ASSERT_EQ(::symlink(backing.c_str(), original.c_str()), 0);
  expectError(read(recordDigest),
              NumericDependencyConformanceErrorCode::Symlink);
}

TEST_F(NumericDependencyConformanceTest, RejectsManagedRootAncestorSymlink) {
  const std::string rootName = llvm::sys::path::filename(root).str();
  llvm::SmallString<256> parent(root);
  llvm::sys::path::remove_filename(parent);
  const std::string alias = root.str().str() + "-ancestor-alias";
  ASSERT_EQ(::symlink(parent.c_str(), alias.c_str()), 0);
  llvm::SmallString<256> aliasRoot(alias);
  llvm::sys::path::append(aliasRoot, rootName);
  llvm::SmallString<256> aliasRecord(aliasRoot);
  llvm::sys::path::append(aliasRecord, "numeric-model-deps.json");
  expectError(wafer::readNumericDependencyConformanceRecord(
                  aliasRoot, aliasRecord, recordDigest),
              NumericDependencyConformanceErrorCode::Symlink);
  ASSERT_FALSE(llvm::sys::fs::remove(alias));
}

TEST_F(NumericDependencyConformanceTest, RejectsUnknownAndMissingFields) {
  top()["unknown"] = true;
  expectError(read(writeRecord()),
              NumericDependencyConformanceErrorCode::UnknownField);
  top().erase("unknown");
  top().erase("status");
  expectError(read(writeRecord()),
              NumericDependencyConformanceErrorCode::MissingField);
}

TEST_F(NumericDependencyConformanceTest, RejectsDuplicateJSONField) {
  std::string text = llvm::formatv("{0:2}", record).str();
  ASSERT_FALSE(text.empty());
  text.insert(1, "\"kind\":\"duplicate\",");
  expectError(read(writeRawRecord(text)),
              NumericDependencyConformanceErrorCode::DuplicateField);
}

TEST_F(NumericDependencyConformanceTest, RejectsBooleanIntegerFields) {
  gate("m4-build")["exit_code"] = false;
  expectError(read(writeRecord()),
              NumericDependencyConformanceErrorCode::TypeMismatch);
}

TEST_F(NumericDependencyConformanceTest,
       RejectsELFMetadataChangeEvenWhenPairsRemainCoherent) {
  for (llvm::StringRef name : {"mpfr", "mpfr-soname"}) {
    llvm::json::Object *elf = fileRecord(name).getObject("elf");
    ASSERT_NE(elf, nullptr);
    (*elf)["machine"] = "tampered-machine";
  }
  expectError(read(writeRecord()),
              NumericDependencyConformanceErrorCode::PolicyMismatch);
}

TEST_F(NumericDependencyConformanceTest, RejectsToolVersionChange) {
  llvm::json::Object *toolchain = build().getObject("toolchain");
  ASSERT_NE(toolchain, nullptr);
  llvm::json::Object *tools = toolchain->getObject("tools");
  ASSERT_NE(tools, nullptr);
  llvm::json::Object *cc = tools->getObject("cc");
  ASSERT_NE(cc, nullptr);
  std::optional<llvm::StringRef> firstLine =
      cc->getString("version_first_line");
  ASSERT_TRUE(firstLine.has_value());
  (*cc)["version_first_line"] = firstLine->str() + " changed";
  expectError(read(writeRecord()),
              NumericDependencyConformanceErrorCode::PolicyMismatch);
}

TEST_F(NumericDependencyConformanceTest, RejectsEnvironmentClosureChange) {
  llvm::json::Object *environments = build().getObject("environments");
  ASSERT_NE(environments, nullptr);
  llvm::json::Object *base = environments->getObject("base");
  ASSERT_NE(base, nullptr);
  base->erase("SOURCE_DATE_EPOCH");
  expectError(read(writeRecord()),
              NumericDependencyConformanceErrorCode::PolicyMismatch);
}

TEST_F(NumericDependencyConformanceTest, RejectsGateContractChange) {
  llvm::json::Array command;
  command.push_back("true");
  gate("mpfr-check")["command"] = std::move(command);
  expectError(read(writeRecord()),
              NumericDependencyConformanceErrorCode::PolicyMismatch);
}

TEST_F(NumericDependencyConformanceTest, RejectsMisplacedGateLog) {
  llvm::json::Object &logRecord = *gate("m4-build").getObject("log");
  std::optional<llvm::StringRef> oldPath = logRecord.getString("path");
  ASSERT_TRUE(oldPath.has_value());
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path(*oldPath), /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false);
  ASSERT_TRUE(static_cast<bool>(buffer));
  writeFile("conformance/misplaced.log", (*buffer)->getBuffer());
  logRecord["path"] = "conformance/misplaced.log";
  expectError(read(writeRecord()),
              NumericDependencyConformanceErrorCode::PolicyMismatch);
}

TEST_F(NumericDependencyConformanceTest, RejectsEmptyGateLog) {
  llvm::json::Object &logRecord = *gate("m4-build").getObject("log");
  std::optional<llvm::StringRef> relative = logRecord.getString("path");
  ASSERT_TRUE(relative.has_value());
  writeFile(*relative, "");
  logRecord["sha256"] = digestBytes("");
  logRecord["size"] = 0;
  logRecord["elf"] = nullptr;
  expectError(read(writeRecord()),
              NumericDependencyConformanceErrorCode::PolicyMismatch);
}

TEST_F(NumericDependencyConformanceTest, RejectsLicenseCopyRebinding) {
  const std::string relative = "install/licenses/mpfr-lesser.txt";
  const std::string wrong = "wrong-license-content\n";
  writeFile(relative, wrong);
  llvm::json::Object &licenseFile = fileRecord("license-mpfr-lesser");
  licenseFile["sha256"] = digestBytes(wrong);
  licenseFile["size"] = static_cast<int64_t>(wrong.size());
  licenseFile["elf"] = nullptr;
  llvm::json::Object *licenses = top().getObject("licenses");
  ASSERT_NE(licenses, nullptr);
  llvm::json::Object *license = licenses->getObject("license-mpfr-lesser");
  ASSERT_NE(license, nullptr);
  (*license)["sha256"] = digestBytes(wrong);
  expectError(read(writeRecord()),
              NumericDependencyConformanceErrorCode::DigestMismatch);
}

TEST_F(NumericDependencyConformanceTest, RejectsLoadedObjectDigestMismatch) {
  llvm::Expected<NumericDependencyConformanceRecord> parsed =
      read(recordDigest);
  ASSERT_TRUE(static_cast<bool>(parsed))
      << (parsed ? std::string() : llvm::toString(parsed.takeError()));
  StaticLoadedObjectProvider provider(
      {NumericLoadedObjectKind::MPFR, filePaths["mpfr"], std::string(64, '0')},
      {NumericLoadedObjectKind::GMP, filePaths["gmp"], fileDigests["gmp"]});
  expectError(wafer::bindNumericDependenciesToLoadedObjects(*parsed, provider),
              NumericDependencyConformanceErrorCode::LoadedObjectMismatch);
}

TEST_F(NumericDependencyConformanceTest, RejectsLoadedObjectOutsideClosure) {
  llvm::Expected<NumericDependencyConformanceRecord> parsed =
      read(recordDigest);
  ASSERT_TRUE(static_cast<bool>(parsed))
      << (parsed ? std::string() : llvm::toString(parsed.takeError()));
  const std::string outside = path("install/mpfr/include/mpfr.h");
  StaticLoadedObjectProvider provider(
      {NumericLoadedObjectKind::MPFR, outside, digestFile(outside)},
      {NumericLoadedObjectKind::GMP, filePaths["gmp"], fileDigests["gmp"]});
  expectError(wafer::bindNumericDependenciesToLoadedObjects(*parsed, provider),
              NumericDependencyConformanceErrorCode::LoadedObjectMismatch);
}

TEST_F(NumericDependencyConformanceTest, RejectsLoadedObjectWrongKind) {
  llvm::Expected<NumericDependencyConformanceRecord> parsed =
      read(recordDigest);
  ASSERT_TRUE(static_cast<bool>(parsed))
      << (parsed ? std::string() : llvm::toString(parsed.takeError()));
  StaticLoadedObjectProvider provider(
      {NumericLoadedObjectKind::GMP, filePaths["mpfr"], fileDigests["mpfr"]},
      {NumericLoadedObjectKind::GMP, filePaths["gmp"], fileDigests["gmp"]});
  expectError(wafer::bindNumericDependenciesToLoadedObjects(*parsed, provider),
              NumericDependencyConformanceErrorCode::LoadedObjectMismatch);
}

TEST_F(NumericDependencyConformanceTest, RejectsNullDladdrSymbol) {
  wafer::DladdrNumericLoadedObjectProvider provider(nullptr, nullptr);
  expectError(provider.getLoadedObject(NumericLoadedObjectKind::MPFR),
              NumericDependencyConformanceErrorCode::LoadedObjectUnavailable);
}

TEST_F(NumericDependencyConformanceTest,
       DladdrProviderReadsGenericLoadedObjectWithoutNumericHeaders) {
#if defined(__unix__) || defined(__APPLE__)
  const void *symbol =
      reinterpret_cast<const void *>(reinterpret_cast<uintptr_t>(&digestBytes));
  wafer::DladdrNumericLoadedObjectProvider provider(symbol, symbol);
  llvm::Expected<NumericLoadedObject> loadedObject =
      provider.getLoadedObject(NumericLoadedObjectKind::MPFR);
  ASSERT_TRUE(static_cast<bool>(loadedObject))
      << (loadedObject ? std::string()
                       : llvm::toString(loadedObject.takeError()));
  EXPECT_EQ(loadedObject->kind, NumericLoadedObjectKind::MPFR);
  EXPECT_TRUE(llvm::sys::path::is_absolute(loadedObject->resolvedPath));
  EXPECT_EQ(loadedObject->sha256.size(), 64u);
#else
  GTEST_SKIP() << "dladdr is unavailable on this host";
#endif
}

} // namespace
