//===- OptimizationQualificationArchiveStore.cpp - Immutable store ------===//

#include "Wafer/Support/OptimizationQualificationArchiveStore.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cerrno>
#include <set>
#include <system_error>

#ifdef __linux__
#include <fcntl.h>
#include <linux/fs.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace wafer {
namespace {

static bool fail(std::string *diagnostic, llvm::Twine message) {
  if (diagnostic)
    *diagnostic = message.str();
  return false;
}

static llvm::SmallString<256> child(llvm::StringRef parent,
                                    llvm::StringRef name) {
  llvm::SmallString<256> path(parent);
  llvm::sys::path::append(path, name);
  return path;
}

static bool pathExists(llvm::StringRef path) {
  llvm::sys::fs::file_status status;
  return !llvm::sys::fs::status(path, status, /*follow=*/false) &&
         status.type() != llvm::sys::fs::file_type::file_not_found;
}

static bool writeBytes(llvm::StringRef path, llvm::ArrayRef<uint8_t> bytes,
                       std::string *diagnostic) {
  std::error_code error;
  llvm::raw_fd_ostream output(path, error, llvm::sys::fs::OF_None);
  if (error)
    return fail(diagnostic,
                "failed to create archive member: " + error.message());
  output.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  output.close();
  if (output.has_error())
    return fail(diagnostic, "failed to write archive member");
  return true;
}

static bool readBytes(llvm::StringRef path, std::vector<uint8_t> &bytes,
                      std::string *diagnostic) {
  llvm::sys::fs::file_status status;
  if (std::error_code error =
          llvm::sys::fs::status(path, status, /*follow=*/false))
    return fail(diagnostic,
                "failed to stat archive member: " + error.message());
  if (status.type() != llvm::sys::fs::file_type::regular_file)
    return fail(diagnostic, "archive member is not a regular file");
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path, /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false);
  if (!buffer)
    return fail(diagnostic, "failed to read archive member: " +
                                buffer.getError().message());
  llvm::StringRef contents = (*buffer)->getBuffer();
  bytes.assign(contents.bytes_begin(), contents.bytes_end());
  return true;
}

static std::string mechanismFileName(MechanismKey key) {
  return std::to_string(key.semanticId) + ".bin";
}

static std::string digestFileName(const AdoptionDigest &digest) {
  return toHex(digest) + ".bin";
}

static bool createDirectory(llvm::StringRef path, std::string *diagnostic) {
  if (std::error_code error = llvm::sys::fs::create_directories(path))
    return fail(diagnostic,
                "failed to create archive directory: " + error.message());
  llvm::sys::fs::file_status status;
  if (std::error_code error =
          llvm::sys::fs::status(path, status, /*follow=*/false))
    return fail(diagnostic,
                "failed to stat archive directory: " + error.message());
  return status.type() == llvm::sys::fs::file_type::directory_file ||
         fail(diagnostic, "archive path is not a directory");
}

static bool fsyncPath(llvm::StringRef path, bool directory,
                      std::string *diagnostic) {
#ifdef __linux__
  int flags = O_RDONLY | O_CLOEXEC;
  if (directory)
    flags |= O_DIRECTORY;
  std::string storage = path.str();
  int descriptor = ::open(storage.c_str(), flags);
  if (descriptor < 0)
    return fail(diagnostic,
                "failed to open archive member for fsync: " +
                    std::error_code(errno, std::generic_category()).message());
  int result = ::fsync(descriptor);
  int errorNumber = errno;
  ::close(descriptor);
  if (result != 0)
    return fail(
        diagnostic,
        "failed to fsync archive member: " +
            std::error_code(errorNumber, std::generic_category()).message());
  return true;
#else
  (void)path;
  (void)directory;
  return fail(diagnostic,
              "qualification archive fsync is unsupported on this host");
#endif
}

static bool fsyncTree(llvm::StringRef root, std::string *diagnostic) {
  std::vector<std::string> directories = {root.str()};
  std::error_code error;
  for (llvm::sys::fs::recursive_directory_iterator
           iterator(root, error, /*follow_symlinks=*/false),
       end;
       iterator != end; iterator.increment(error)) {
    if (error)
      return fail(diagnostic,
                  "failed to walk archive tree: " + error.message());
    if (iterator->type() == llvm::sys::fs::file_type::directory_file) {
      directories.push_back(iterator->path());
      continue;
    }
    if (iterator->type() != llvm::sys::fs::file_type::regular_file)
      return fail(diagnostic, "archive tree contains a non-regular member");
    if (!fsyncPath(iterator->path(), false, diagnostic))
      return false;
  }
  if (error)
    return fail(diagnostic, "failed to walk archive tree: " + error.message());
  for (auto iterator = directories.rbegin(); iterator != directories.rend();
       ++iterator)
    if (!fsyncPath(*iterator, true, diagnostic))
      return false;
  return true;
}

static bool publishNoReplace(llvm::StringRef source,
                             llvm::StringRef destination,
                             std::string *diagnostic) {
#ifdef __linux__
  std::string sourceStorage = source.str();
  std::string destinationStorage = destination.str();
  if (::syscall(SYS_renameat2, AT_FDCWD, sourceStorage.c_str(), AT_FDCWD,
                destinationStorage.c_str(), RENAME_NOREPLACE) == 0)
    return true;
  return fail(diagnostic,
              "failed to seal qualification run without replacement: " +
                  std::error_code(errno, std::generic_category()).message());
#else
  (void)source;
  (void)destination;
  return fail(diagnostic,
              "atomic no-replace archive publication is unsupported");
#endif
}

static bool writeCompletedTree(llvm::StringRef directory,
                               const CompletedQualificationEvidenceV1 &value,
                               std::string *diagnostic) {
  llvm::SmallString<256> specs = child(directory, "specs");
  llvm::SmallString<256> invocations = child(directory, "invocations");
  llvm::SmallString<256> observations = child(directory, "observations");
  if (!createDirectory(specs, diagnostic) ||
      !createDirectory(invocations, diagnostic) ||
      !createDirectory(observations, diagnostic))
    return false;

  if (!writeBytes(child(directory, "input.bin"),
                  encodeAdoptionQualificationInputV1(value.input),
                  diagnostic) ||
      !writeBytes(child(directory, "run.bin"),
                  encodeAdoptionQualificationRunV1(value.run), diagnostic) ||
      !writeBytes(
          child(directory, "result-manifest.bin"),
          encodeAdoptionQualificationResultManifestV1(value.resultManifest),
          diagnostic) ||
      !writeBytes(child(directory, "run-terminal.bin"),
                  encodeAdoptionQualificationRunTerminalV1(value.runTerminal),
                  diagnostic))
    return false;
  if (value.publicationAttempt &&
      !writeBytes(
          child(directory, "publication-attempt.bin"),
          encodeOptimizationSetPublicationAttemptV1(*value.publicationAttempt),
          diagnostic))
    return false;
  if (value.optimizationBatch &&
      !writeBytes(
          child(directory, "optimization-batch.bin"),
          encodeOptimizationBatchObservationV1(*value.optimizationBatch),
          diagnostic))
    return false;

  for (const MechanismSpecBinding &binding : value.input.specBindings) {
    std::optional<AdoptionSpec> spec = lookupAdoptionSpec(binding.mechanismKey);
    if (!spec ||
        !writeBytes(child(specs, mechanismFileName(binding.mechanismKey)),
                    encodeAdoptionSpecV1(*spec), diagnostic))
      return false;
  }
  for (const InvocationTelemetryV1 &terminal : value.invocationTerminals) {
    AdoptionDigest id = digestInvocationIdentityV1(terminal.identity);
    if (!writeBytes(child(invocations, digestFileName(id)),
                    encodeInvocationTelemetryV1(terminal), diagnostic))
      return false;
  }
  for (const QualificationObservationV1 &observation : value.observations)
    if (!writeBytes(
            child(observations, mechanismFileName(observation.mechanismKey)),
            encodeQualificationObservationV1(observation), diagnostic))
      return false;
  return true;
}

static bool validateTreeMembers(llvm::StringRef directory,
                                const CompletedQualificationEvidenceV1 &value,
                                std::string *diagnostic) {
  std::set<std::string> expectedDirectories = {"specs", "invocations",
                                               "observations"};
  std::set<std::string> expectedFiles = {
      "input.bin", "run.bin", "result-manifest.bin", "run-terminal.bin"};
  if (value.publicationAttempt)
    expectedFiles.insert("publication-attempt.bin");
  if (value.optimizationBatch)
    expectedFiles.insert("optimization-batch.bin");
  for (const MechanismSpecBinding &binding : value.input.specBindings)
    expectedFiles.insert("specs/" + mechanismFileName(binding.mechanismKey));
  for (const InvocationTerminalBindingV1 &binding :
       value.resultManifest.invocationTerminals)
    expectedFiles.insert("invocations/" + digestFileName(binding.invocationId));
  for (const MechanismObservationBindingV1 &binding :
       value.resultManifest.observationBindings)
    expectedFiles.insert("observations/" +
                         mechanismFileName(binding.mechanismKey));

  std::set<std::string> actualDirectories;
  std::set<std::string> actualFiles;
  std::error_code error;
  for (llvm::sys::fs::recursive_directory_iterator
           iterator(directory, error, /*follow_symlinks=*/false),
       end;
       iterator != end; iterator.increment(error)) {
    if (error)
      return fail(diagnostic,
                  "failed to walk archive readback: " + error.message());
    llvm::StringRef path = iterator->path();
    if (!path.starts_with(directory))
      return fail(diagnostic, "archive iterator escaped its root");
    path = path.drop_front(directory.size());
    if (!path.empty() && llvm::sys::path::is_separator(path.front()))
      path = path.drop_front();
    if (iterator->type() == llvm::sys::fs::file_type::directory_file)
      actualDirectories.insert(path.str());
    else if (iterator->type() == llvm::sys::fs::file_type::regular_file)
      actualFiles.insert(path.str());
    else
      return fail(diagnostic, "archive readback contains a non-regular member");
  }
  if (error)
    return fail(diagnostic,
                "failed to walk archive readback: " + error.message());
  return (actualDirectories == expectedDirectories &&
          actualFiles == expectedFiles) ||
         fail(diagnostic,
              "archive members are not the all-and-only canonical set");
}

static bool readCompletedTree(llvm::StringRef directory,
                              CompletedQualificationEvidenceV1 &value,
                              std::string *diagnostic) {
  std::vector<uint8_t> bytes;
  CompletedQualificationEvidenceV1 parsed;
  if (!readBytes(child(directory, "input.bin"), bytes, diagnostic) ||
      !decodeCanonicalAdoptionQualificationInputV1(bytes, parsed.input,
                                                   diagnostic) ||
      !readBytes(child(directory, "run.bin"), bytes, diagnostic) ||
      !decodeCanonicalAdoptionQualificationRunV1(bytes, parsed.run,
                                                 diagnostic) ||
      !readBytes(child(directory, "result-manifest.bin"), bytes, diagnostic) ||
      !decodeCanonicalAdoptionQualificationResultManifestV1(
          bytes, parsed.resultManifest, diagnostic) ||
      !readBytes(child(directory, "run-terminal.bin"), bytes, diagnostic) ||
      !decodeCanonicalAdoptionQualificationRunTerminalV1(
          bytes, parsed.runTerminal, diagnostic))
    return false;

  bool optimization = parsed.input.optimizationProposalDigest.has_value();
  if (optimization) {
    OptimizationSetPublicationAttemptV1 attempt;
    if (!readBytes(child(directory, "publication-attempt.bin"), bytes,
                   diagnostic) ||
        !decodeCanonicalOptimizationSetPublicationAttemptV1(bytes, attempt,
                                                            diagnostic))
      return false;
    parsed.publicationAttempt = std::move(attempt);
    OptimizationBatchObservationV1 batch;
    if (!readBytes(child(directory, "optimization-batch.bin"), bytes,
                   diagnostic) ||
        !decodeCanonicalOptimizationBatchObservationV1(bytes, batch,
                                                       diagnostic))
      return false;
    parsed.optimizationBatch = std::move(batch);
  } else if (pathExists(child(directory, "publication-attempt.bin")) ||
             pathExists(child(directory, "optimization-batch.bin"))) {
    return fail(diagnostic,
                "general run archive contains optimization members");
  }

  llvm::SmallString<256> specs = child(directory, "specs");
  for (const MechanismSpecBinding &binding : parsed.input.specBindings) {
    if (!readBytes(child(specs, mechanismFileName(binding.mechanismKey)), bytes,
                   diagnostic) ||
        !validateCanonicalAdoptionSpecV1(bytes, diagnostic))
      return false;
    std::optional<AdoptionSpec> spec = lookupAdoptionSpec(binding.mechanismKey);
    if (!spec || digestAdoptionSpecV1(*spec) != binding.specDigest ||
        encodeAdoptionSpecV1(*spec) != bytes)
      return fail(diagnostic, "archived adoption spec readback mismatch");
  }

  llvm::SmallString<256> invocations = child(directory, "invocations");
  for (const InvocationTerminalBindingV1 &binding :
       parsed.resultManifest.invocationTerminals) {
    if (!readBytes(child(invocations, digestFileName(binding.invocationId)),
                   bytes, diagnostic))
      return false;
    InvocationTelemetryV1 terminal;
    if (!decodeCanonicalInvocationTelemetryV1(bytes, terminal, diagnostic) ||
        digestInvocationIdentityV1(terminal.identity) != binding.invocationId ||
        digestInvocationTelemetryV1(terminal) != binding.terminalDigest)
      return fail(diagnostic, "archived invocation terminal readback mismatch");
    parsed.invocationTerminals.push_back(std::move(terminal));
  }

  llvm::SmallString<256> observations = child(directory, "observations");
  for (const MechanismObservationBindingV1 &binding :
       parsed.resultManifest.observationBindings) {
    if (!readBytes(child(observations, mechanismFileName(binding.mechanismKey)),
                   bytes, diagnostic))
      return false;
    QualificationObservationV1 observation;
    if (!decodeCanonicalQualificationObservationV1(bytes, observation,
                                                   diagnostic) ||
        observation.mechanismKey != binding.mechanismKey ||
        digestQualificationObservationV1(observation) !=
            binding.observationDigest)
      return fail(diagnostic,
                  "archived qualification observation readback mismatch");
    parsed.observations.push_back(std::move(observation));
  }

  if (!validateCompletedQualificationEvidenceV1(parsed, diagnostic) ||
      !validateTreeMembers(directory, parsed, diagnostic))
    return false;
  value = std::move(parsed);
  return true;
}

static bool writeFileNoReplace(llvm::StringRef path,
                               llvm::ArrayRef<uint8_t> bytes,
                               std::string *diagnostic) {
#ifdef __linux__
  std::string storage = path.str();
  int descriptor =
      ::open(storage.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
  if (descriptor < 0)
    return fail(diagnostic,
                "failed to create no-replace archive record: " +
                    std::error_code(errno, std::generic_category()).message());
  size_t offset = 0;
  while (offset < bytes.size()) {
    ssize_t written =
        ::write(descriptor, bytes.data() + offset, bytes.size() - offset);
    if (written <= 0) {
      int errorNumber = errno;
      ::close(descriptor);
      llvm::sys::fs::remove(path);
      return fail(
          diagnostic,
          "failed to write no-replace archive record: " +
              std::error_code(errorNumber, std::generic_category()).message());
    }
    offset += static_cast<size_t>(written);
  }
  if (::fsync(descriptor) != 0) {
    int errorNumber = errno;
    ::close(descriptor);
    llvm::sys::fs::remove(path);
    return fail(
        diagnostic,
        "failed to fsync no-replace archive record: " +
            std::error_code(errorNumber, std::generic_category()).message());
  }
  ::close(descriptor);
  return true;
#else
  (void)path;
  (void)bytes;
  return fail(diagnostic, "no-replace qualification record is unsupported");
#endif
}

class ArchiveLock final {
public:
  ~ArchiveLock() {
#ifdef __linux__
    if (descriptor_ >= 0) {
      ::flock(descriptor_, LOCK_UN);
      ::close(descriptor_);
    }
#endif
  }

  bool acquire(llvm::StringRef path, std::string *diagnostic) {
#ifdef __linux__
    std::string storage = path.str();
    descriptor_ = ::open(storage.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    if (descriptor_ < 0 || ::flock(descriptor_, LOCK_EX) != 0)
      return fail(
          diagnostic,
          "failed to acquire qualification publication lock: " +
              std::error_code(errno, std::generic_category()).message());
    return true;
#else
    (void)path;
    return fail(diagnostic, "qualification publication lock is unsupported");
#endif
  }

private:
  int descriptor_ = -1;
};

static bool
validateQualifiedCandidate(const CompletedQualificationEvidenceV1 &chain,
                           const QualifiedOptimizationSetV1 &candidate,
                           std::string *diagnostic) {
  if (!validateCompletedQualificationEvidenceV1(chain, diagnostic) ||
      !chain.publicationAttempt || !chain.optimizationBatch ||
      chain.optimizationBatch->batchStatus !=
          OptimizationBatchStatusV1::Qualified ||
      !validateQualifiedOptimizationSetV1(candidate, diagnostic) ||
      candidate.proposalDigest != chain.optimizationBatch->proposalDigest ||
      candidate.batchObservationDigest !=
          digestOptimizationBatchObservationV1(*chain.optimizationBatch))
    return fail(diagnostic,
                "candidate set does not bind a Qualified completed batch");
  if (candidate.observationBindings.size() > chain.observations.size())
    return fail(diagnostic, "candidate observation domain is invalid");
  for (const MechanismObservationBindingV1 &binding :
       candidate.observationBindings) {
    auto iterator = llvm::find_if(
        chain.observations, [&](const QualificationObservationV1 &observation) {
          return observation.mechanismKey == binding.mechanismKey;
        });
    if (iterator == chain.observations.end() ||
        iterator->qualificationStatus != QualificationStatusV1::Qualified ||
        digestQualificationObservationV1(*iterator) !=
            binding.observationDigest ||
        !iterator->optimizationPublicationAttemptDigest ||
        *iterator->optimizationPublicationAttemptDigest !=
            digestOptimizationSetPublicationAttemptV1(
                *chain.publicationAttempt))
      return fail(diagnostic,
                  "candidate set observation is not exact and Qualified");
  }
  return true;
}

static bool writeSetTree(llvm::StringRef directory,
                         const CompletedQualificationEvidenceV1 &chain,
                         const QualifiedOptimizationSetV1 &candidate,
                         const OptimizationSetPublicationTerminalV1 &terminal,
                         std::string *diagnostic) {
  llvm::SmallString<256> specs = child(directory, "specs");
  llvm::SmallString<256> observations = child(directory, "observations");
  if (!createDirectory(specs, diagnostic) ||
      !createDirectory(observations, diagnostic) ||
      !writeBytes(child(directory, "qualified-set.bin"),
                  encodeQualifiedOptimizationSetV1(candidate), diagnostic) ||
      !writeBytes(child(directory, "proposal.bin"),
                  encodeOptimizationQualificationProposalV1(
                      getCurrentOptimizationQualificationProposal()),
                  diagnostic) ||
      !writeBytes(
          child(directory, "optimization-batch.bin"),
          encodeOptimizationBatchObservationV1(*chain.optimizationBatch),
          diagnostic) ||
      !writeBytes(child(directory, "publication-terminal.bin"),
                  encodeOptimizationSetPublicationTerminalV1(terminal),
                  diagnostic))
    return false;
  for (const MechanismObservationBindingV1 &binding :
       candidate.observationBindings) {
    std::optional<AdoptionSpec> spec = lookupAdoptionSpec(binding.mechanismKey);
    auto observation = llvm::find_if(
        chain.observations, [&](const QualificationObservationV1 &value) {
          return value.mechanismKey == binding.mechanismKey;
        });
    if (!spec || observation == chain.observations.end() ||
        !writeBytes(child(specs, mechanismFileName(binding.mechanismKey)),
                    encodeAdoptionSpecV1(*spec), diagnostic) ||
        !writeBytes(
            child(observations, mechanismFileName(binding.mechanismKey)),
            encodeQualificationObservationV1(*observation), diagnostic))
      return false;
  }
  return true;
}

static bool validateSetTreeMembers(llvm::StringRef directory,
                                   const QualifiedOptimizationSetV1 &candidate,
                                   std::string *diagnostic) {
  std::set<std::string> expectedDirectories = {"specs", "observations"};
  std::set<std::string> expectedFiles = {"qualified-set.bin", "proposal.bin",
                                         "optimization-batch.bin",
                                         "publication-terminal.bin"};
  for (const MechanismObservationBindingV1 &binding :
       candidate.observationBindings) {
    expectedFiles.insert("specs/" + mechanismFileName(binding.mechanismKey));
    expectedFiles.insert("observations/" +
                         mechanismFileName(binding.mechanismKey));
  }
  std::set<std::string> actualDirectories;
  std::set<std::string> actualFiles;
  std::error_code error;
  for (llvm::sys::fs::recursive_directory_iterator
           iterator(directory, error, /*follow_symlinks=*/false),
       end;
       iterator != end; iterator.increment(error)) {
    if (error)
      return fail(diagnostic,
                  "failed to walk qualified set: " + error.message());
    llvm::StringRef path = iterator->path();
    if (!path.starts_with(directory))
      return fail(diagnostic, "qualified-set iterator escaped its root");
    path = path.drop_front(directory.size());
    if (!path.empty() && llvm::sys::path::is_separator(path.front()))
      path = path.drop_front();
    if (iterator->type() == llvm::sys::fs::file_type::directory_file)
      actualDirectories.insert(path.str());
    else if (iterator->type() == llvm::sys::fs::file_type::regular_file)
      actualFiles.insert(path.str());
    else
      return fail(diagnostic, "qualified set contains a non-regular member");
  }
  return (actualDirectories == expectedDirectories &&
          actualFiles == expectedFiles) ||
         fail(diagnostic,
              "qualified set members are not all-and-only canonical");
}

static bool readSetTree(llvm::StringRef directory,
                        const CompletedQualificationEvidenceV1 &chain,
                        QualifiedOptimizationSetV1 &candidate,
                        OptimizationSetPublicationTerminalV1 &terminal,
                        std::string *diagnostic) {
  std::vector<uint8_t> bytes;
  if (!readBytes(child(directory, "qualified-set.bin"), bytes, diagnostic) ||
      !decodeCanonicalQualifiedOptimizationSetV1(bytes, candidate,
                                                 diagnostic) ||
      !readBytes(child(directory, "proposal.bin"), bytes, diagnostic) ||
      !validateCanonicalOptimizationQualificationProposalV1(bytes,
                                                            diagnostic) ||
      bytes != encodeOptimizationQualificationProposalV1(
                   getCurrentOptimizationQualificationProposal()) ||
      !readBytes(child(directory, "optimization-batch.bin"), bytes, diagnostic))
    return false;
  OptimizationBatchObservationV1 batch;
  if (!decodeCanonicalOptimizationBatchObservationV1(bytes, batch,
                                                     diagnostic) ||
      !chain.optimizationBatch ||
      digestOptimizationBatchObservationV1(batch) !=
          candidate.batchObservationDigest ||
      encodeOptimizationBatchObservationV1(batch) !=
          encodeOptimizationBatchObservationV1(*chain.optimizationBatch) ||
      !readBytes(child(directory, "publication-terminal.bin"), bytes,
                 diagnostic) ||
      !decodeCanonicalOptimizationSetPublicationTerminalV1(bytes, terminal,
                                                           diagnostic))
    return false;
  llvm::SmallString<256> specs = child(directory, "specs");
  llvm::SmallString<256> observations = child(directory, "observations");
  for (const MechanismObservationBindingV1 &binding :
       candidate.observationBindings) {
    if (!readBytes(child(specs, mechanismFileName(binding.mechanismKey)), bytes,
                   diagnostic) ||
        !validateCanonicalAdoptionSpecV1(bytes, diagnostic))
      return false;
    std::optional<AdoptionSpec> spec = lookupAdoptionSpec(binding.mechanismKey);
    if (!spec || encodeAdoptionSpecV1(*spec) != bytes ||
        !readBytes(child(observations, mechanismFileName(binding.mechanismKey)),
                   bytes, diagnostic))
      return false;
    QualificationObservationV1 observation;
    if (!decodeCanonicalQualificationObservationV1(bytes, observation,
                                                   diagnostic) ||
        observation.mechanismKey != binding.mechanismKey ||
        digestQualificationObservationV1(observation) !=
            binding.observationDigest)
      return fail(diagnostic, "qualified set observation readback mismatch");
  }
  if (!validateQualifiedCandidate(chain, candidate, diagnostic) ||
      !validateSetTreeMembers(directory, candidate, diagnostic))
    return false;
  return true;
}

static bool
persistPublicationTerminal(llvm::StringRef root,
                           const OptimizationSetPublicationTerminalV1 &terminal,
                           std::string *diagnostic) {
  llvm::SmallString<256> directory = child(root, "publication-terminals");
  if (!createDirectory(directory, diagnostic))
    return false;
  llvm::SmallString<256> path = child(
      directory, digestFileName(terminal.optimizationPublicationAttemptDigest));
  if (!writeFileNoReplace(path,
                          encodeOptimizationSetPublicationTerminalV1(terminal),
                          diagnostic))
    return false;
  return fsyncPath(directory, true, diagnostic);
}

static std::optional<ActiveQualifiedOptimizationSetRefV1>
readActiveRef(llvm::StringRef root, std::string *diagnostic, bool &valid) {
  valid = true;
  llvm::SmallString<256> path = child(root, "active-ref.bin");
  if (!pathExists(path))
    return std::nullopt;
  std::vector<uint8_t> bytes;
  ActiveQualifiedOptimizationSetRefV1 active;
  if (!readBytes(path, bytes, diagnostic) ||
      !decodeCanonicalActiveQualifiedOptimizationSetRefV1(bytes, active,
                                                          diagnostic)) {
    valid = false;
    return std::nullopt;
  }
  return active;
}

} // namespace

bool OptimizationQualificationArchiveStoreV1::publishCompletedRun(
    const CompletedQualificationEvidenceV1 &value,
    std::string *diagnostic) const {
  if (rootDirectory_.empty())
    return fail(diagnostic, "qualification archive root is empty");
  if (!validateCompletedQualificationEvidenceV1(value, diagnostic))
    return false;
  AdoptionDigest runDigest = digestAdoptionQualificationRunV1(value.run);
  llvm::SmallString<256> runs = child(rootDirectory_, "runs");
  llvm::SmallString<256> staging = child(rootDirectory_, ".staging");
  if (!createDirectory(runs, diagnostic) ||
      !createDirectory(staging, diagnostic))
    return false;
  llvm::SmallString<256> finalDirectory = child(runs, toHex(runDigest));
  if (pathExists(finalDirectory))
    return fail(diagnostic, "qualification run is already permanently sealed");

  llvm::SmallString<256> stagingPrefix = child(staging, "run");
  llvm::SmallString<256> stagingDirectory;
  if (std::error_code error =
          llvm::sys::fs::createUniqueDirectory(stagingPrefix, stagingDirectory))
    return fail(diagnostic,
                "failed to create qualification staging: " + error.message());
  auto cleanup = llvm::make_scope_exit(
      [&] { llvm::sys::fs::remove_directories(stagingDirectory); });
  if (!writeCompletedTree(stagingDirectory, value, diagnostic))
    return false;
  CompletedQualificationEvidenceV1 readback;
  if (!readCompletedTree(stagingDirectory, readback, diagnostic) ||
      digestAdoptionQualificationRunV1(readback.run) != runDigest)
    return fail(diagnostic, "qualification staging readback digest mismatch");
  if (!fsyncTree(stagingDirectory, diagnostic) ||
      !publishNoReplace(stagingDirectory, finalDirectory, diagnostic))
    return false;
  cleanup.release();
  if (!fsyncPath(runs, true, diagnostic))
    return false;
  return true;
}

bool OptimizationQualificationArchiveStoreV1::readCompletedRun(
    const AdoptionDigest &runDigest, CompletedQualificationEvidenceV1 &value,
    std::string *diagnostic) const {
  if (rootDirectory_.empty())
    return fail(diagnostic, "qualification archive root is empty");
  llvm::SmallString<256> directory =
      child(child(rootDirectory_, "runs"), toHex(runDigest));
  if (!readCompletedTree(directory, value, diagnostic))
    return false;
  return digestAdoptionQualificationRunV1(value.run) == runDigest ||
         fail(diagnostic, "qualification run directory digest mismatch");
}

bool OptimizationQualificationArchiveStoreV1::publishOptimizationSet(
    const AdoptionDigest &runDigest,
    const std::optional<QualifiedOptimizationSetV1> &candidateSet,
    OptimizationSetPublicationResultV1 &result, std::string *diagnostic) const {
  CompletedQualificationEvidenceV1 chain;
  if (!readCompletedRun(runDigest, chain, diagnostic) ||
      !chain.publicationAttempt || !chain.optimizationBatch)
    return fail(
        diagnostic,
        "optimization publication requires a completed optimization run");
  AdoptionDigest attemptDigest =
      digestOptimizationSetPublicationAttemptV1(*chain.publicationAttempt);
  AdoptionDigest batchDigest =
      digestOptimizationBatchObservationV1(*chain.optimizationBatch);
  auto reason = [](GlobalClosedReasonV1 value) {
    return ClosedReasonV1{getGlobalClosedReasonRefV1(value), std::nullopt};
  };

  if (chain.optimizationBatch->batchStatus ==
      OptimizationBatchStatusV1::Rejected) {
    if (candidateSet)
      return fail(diagnostic,
                  "Rejected batch cannot carry a candidate qualified set");
    result = {};
    result.terminal.optimizationPublicationAttemptDigest = attemptDigest;
    result.terminal.outcome =
        OptimizationSetPublicationOutcomeV1::RejectedEvidence;
    result.terminal.batchObservationDigest = batchDigest;
    result.terminal.closedReason =
        reason(GlobalClosedReasonV1::QualificationEvidenceRejected);
    return persistPublicationTerminal(rootDirectory_, result.terminal,
                                      diagnostic);
  }

  if (!candidateSet ||
      !validateQualifiedCandidate(chain, *candidateSet, diagnostic)) {
    result = {};
    result.terminal.optimizationPublicationAttemptDigest = attemptDigest;
    result.terminal.outcome =
        OptimizationSetPublicationOutcomeV1::PublicationFailed;
    result.terminal.batchObservationDigest = batchDigest;
    if (candidateSet &&
        !encodeQualifiedOptimizationSetV1(*candidateSet).empty())
      result.terminal.candidateSetDigest =
          digestQualifiedOptimizationSetV1(*candidateSet);
    result.terminal.closedReason =
        reason(GlobalClosedReasonV1::PublicationFailure);
    std::string validationDiagnostic = diagnostic ? *diagnostic : std::string();
    if (!persistPublicationTerminal(rootDirectory_, result.terminal,
                                    diagnostic))
      return false;
    if (diagnostic)
      *diagnostic = validationDiagnostic;
    return true;
  }

  AdoptionDigest setDigest = digestQualifiedOptimizationSetV1(*candidateSet);
  OptimizationSetPublicationTerminalV1 publishedTerminal;
  publishedTerminal.optimizationPublicationAttemptDigest = attemptDigest;
  publishedTerminal.outcome =
      OptimizationSetPublicationOutcomeV1::QualifiedPublished;
  publishedTerminal.batchObservationDigest = batchDigest;
  publishedTerminal.candidateSetDigest = setDigest;

  llvm::SmallString<256> sets = child(rootDirectory_, "sets");
  llvm::SmallString<256> staging = child(rootDirectory_, ".staging");
  if (!createDirectory(sets, diagnostic) ||
      !createDirectory(staging, diagnostic))
    return false;
  ArchiveLock lock;
  if (!lock.acquire(child(rootDirectory_, "active-ref.lock"), diagnostic))
    return false;
  bool currentValid = false;
  std::optional<ActiveQualifiedOptimizationSetRefV1> current =
      readActiveRef(rootDirectory_, diagnostic, currentValid);
  if (!currentValid)
    return false;
  if (current && current->optimizationPublicationAttemptDigest == attemptDigest)
    return fail(
        diagnostic,
        "optimization publication attempt is already permanently sealed");
  bool expectedMatches =
      (!current && !chain.publicationAttempt->expectedActiveRefDigest) ||
      (current && chain.publicationAttempt->expectedActiveRefDigest &&
       digestActiveQualifiedOptimizationSetRefV1(*current) ==
           *chain.publicationAttempt->expectedActiveRefDigest);
  if (!expectedMatches) {
    result = {};
    result.terminal.optimizationPublicationAttemptDigest = attemptDigest;
    result.terminal.outcome =
        OptimizationSetPublicationOutcomeV1::PublicationConflict;
    result.terminal.batchObservationDigest = batchDigest;
    result.terminal.candidateSetDigest = setDigest;
    result.terminal.closedReason =
        reason(GlobalClosedReasonV1::PublicationConflict);
    return persistPublicationTerminal(rootDirectory_, result.terminal,
                                      diagnostic);
  }
  if (current && current->generation == std::numeric_limits<uint64_t>::max()) {
    result = {};
    result.terminal.optimizationPublicationAttemptDigest = attemptDigest;
    result.terminal.outcome =
        OptimizationSetPublicationOutcomeV1::PublicationFailed;
    result.terminal.batchObservationDigest = batchDigest;
    result.terminal.candidateSetDigest = setDigest;
    result.terminal.closedReason =
        reason(GlobalClosedReasonV1::PublicationFailure);
    return persistPublicationTerminal(rootDirectory_, result.terminal,
                                      diagnostic);
  }

  ActiveQualifiedOptimizationSetRefV1 active;
  active.generation = current ? current->generation + 1 : 1;
  if (current)
    active.parentSetDigest = current->setDigest;
  active.setDigest = setDigest;
  active.qualificationRunDigest = runDigest;
  active.adoptionQualificationRunTerminalDigest =
      digestAdoptionQualificationRunTerminalV1(chain.runTerminal);
  active.optimizationPublicationAttemptDigest = attemptDigest;
  active.optimizationPublicationTerminalDigest =
      digestOptimizationSetPublicationTerminalV1(publishedTerminal);
  if (encodeActiveQualifiedOptimizationSetRefV1(active).empty())
    return fail(diagnostic, "prepared active qualified ref is invalid");

  llvm::SmallString<256> stagingPrefix = child(staging, "set");
  llvm::SmallString<256> stagingDirectory;
  if (std::error_code error =
          llvm::sys::fs::createUniqueDirectory(stagingPrefix, stagingDirectory))
    return fail(diagnostic,
                "failed to create qualified-set staging: " + error.message());
  auto cleanupSet = llvm::make_scope_exit(
      [&] { llvm::sys::fs::remove_directories(stagingDirectory); });
  if (!writeSetTree(stagingDirectory, chain, *candidateSet, publishedTerminal,
                    diagnostic))
    return false;
  QualifiedOptimizationSetV1 setReadback;
  OptimizationSetPublicationTerminalV1 terminalReadback;
  if (!readSetTree(stagingDirectory, chain, setReadback, terminalReadback,
                   diagnostic) ||
      digestQualifiedOptimizationSetV1(setReadback) != setDigest ||
      digestOptimizationSetPublicationTerminalV1(terminalReadback) !=
          active.optimizationPublicationTerminalDigest ||
      !fsyncTree(stagingDirectory, diagnostic))
    return fail(diagnostic, "qualified-set staging readback mismatch");

  llvm::SmallString<256> finalSet = child(sets, toHex(setDigest));
  if (!publishNoReplace(stagingDirectory, finalSet, diagnostic))
    return false;
  cleanupSet.release();
  auto cleanupPublishedSet = llvm::make_scope_exit(
      [&] { llvm::sys::fs::remove_directories(finalSet); });
  if (!fsyncPath(sets, true, diagnostic))
    return false;

  llvm::SmallString<256> activeStagingPrefix = child(staging, "active");
  llvm::SmallString<256> activeStagingDirectory;
  if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
          activeStagingPrefix, activeStagingDirectory))
    return fail(diagnostic,
                "failed to create active-ref staging: " + error.message());
  auto cleanupActive = llvm::make_scope_exit(
      [&] { llvm::sys::fs::remove_directories(activeStagingDirectory); });
  llvm::SmallString<256> stagedActive =
      child(activeStagingDirectory, "active-ref.bin");
  if (!writeBytes(stagedActive,
                  encodeActiveQualifiedOptimizationSetRefV1(active),
                  diagnostic) ||
      !fsyncPath(stagedActive, false, diagnostic) ||
      !fsyncPath(activeStagingDirectory, true, diagnostic))
    return false;
#ifdef __linux__
  llvm::SmallString<256> activePath = child(rootDirectory_, "active-ref.bin");
  std::string stagedStorage = stagedActive.str().str();
  std::string activeStorage = activePath.str().str();
  if (::rename(stagedStorage.c_str(), activeStorage.c_str()) != 0)
    return fail(diagnostic,
                "failed to atomically replace active qualified ref: " +
                    std::error_code(errno, std::generic_category()).message());
#else
  return fail(diagnostic,
              "atomic active-ref replacement is unsupported on this host");
#endif
  if (!fsyncPath(rootDirectory_, true, diagnostic))
    return false;
  cleanupPublishedSet.release();
  result = {publishedTerminal, active};
  return true;
}

bool OptimizationQualificationArchiveStoreV1::
    loadActiveQualifiedOptimizationSet(
        ActiveQualifiedOptimizationSelectionV1 &selection,
        std::string *diagnostic) const {
  bool activeValid = false;
  std::optional<ActiveQualifiedOptimizationSetRefV1> active =
      readActiveRef(rootDirectory_, diagnostic, activeValid);
  if (!activeValid || !active)
    return fail(diagnostic,
                "no valid active qualified optimization ref exists");
  CompletedQualificationEvidenceV1 chain;
  if (!readCompletedRun(active->qualificationRunDigest, chain, diagnostic) ||
      !chain.publicationAttempt || !chain.optimizationBatch ||
      digestAdoptionQualificationRunTerminalV1(chain.runTerminal) !=
          active->adoptionQualificationRunTerminalDigest ||
      digestOptimizationSetPublicationAttemptV1(*chain.publicationAttempt) !=
          active->optimizationPublicationAttemptDigest)
    return fail(diagnostic, "active ref run/publication chain is broken");
  llvm::SmallString<256> setDirectory =
      child(child(rootDirectory_, "sets"), toHex(active->setDigest));
  QualifiedOptimizationSetV1 candidate;
  OptimizationSetPublicationTerminalV1 terminal;
  if (!readSetTree(setDirectory, chain, candidate, terminal, diagnostic) ||
      digestQualifiedOptimizationSetV1(candidate) != active->setDigest ||
      terminal.outcome !=
          OptimizationSetPublicationOutcomeV1::QualifiedPublished ||
      terminal.optimizationPublicationAttemptDigest !=
          active->optimizationPublicationAttemptDigest ||
      !terminal.candidateSetDigest ||
      *terminal.candidateSetDigest != active->setDigest ||
      digestOptimizationSetPublicationTerminalV1(terminal) !=
          active->optimizationPublicationTerminalDigest)
    return fail(diagnostic, "active qualified set readback chain is broken");
  selection = {*active, std::move(candidate), std::move(terminal),
               std::move(chain)};
  return true;
}

} // namespace wafer
