//===- ProgramData.cpp - Transaction-owned program data ------------------===//

#include "Wafer/Driver/ProgramData/ProgramData.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace wafer::compiler {

namespace {

bool checkedMulU64(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

bool checkedAddU64(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (rhs > std::numeric_limits<uint64_t>::max() - lhs)
    return false;
  result = lhs + rhs;
  return true;
}

llvm::Error fail(ProgramDataFailureKind kind, llvm::StringRef locator,
                 llvm::StringRef detail, ProgramDataFailure *failure) {
  if (failure) {
    failure->kind = kind;
    failure->locator = locator.str();
    failure->detail = detail.str();
  }
  std::string message = (llvm::Twine(locator) + ": " + detail).str();
  return llvm::createStringError(llvm::errc::invalid_argument, "%s",
                                 message.c_str());
}

/// I/O-level failure of a transaction-private artifact. Unlike `fail` this
/// keeps the platform error category so callers can still distinguish
/// environment failures.
llvm::Error failIO(ProgramDataFailureKind kind, llvm::StringRef locator,
                   const llvm::Twine &detail, ProgramDataFailure *failure) {
  if (failure) {
    failure->kind = kind;
    failure->locator = locator.str();
    failure->detail = detail.str();
  }
  return llvm::createStringError(llvm::errc::io_error, "%s: %s",
                                 locator.str().c_str(), detail.str().c_str());
}

std::string hexEncode(llvm::ArrayRef<uint8_t> bytes) {
  std::string hex;
  hex.reserve(bytes.size() * 2);
  llvm::raw_string_ostream stream(hex);
  for (uint8_t byte : bytes)
    stream << llvm::format_hex_no_prefix(byte, 2);
  stream.flush();
  return hex;
}

/// Exact read of one byte span at an absolute file offset. Short reads are
/// an environment failure; callers classify it with `failIO`.
llvm::Error readFileSpan(llvm::sys::fs::file_t file, uint64_t offset,
                         llvm::MutableArrayRef<uint8_t> out,
                         ProgramDataIOStatistics *statistics = nullptr) {
  uint64_t done = 0;
  while (done < out.size()) {
    const size_t requested =
        std::min<size_t>(out.size() - done, kProgramDataReadWindowBytes);
    llvm::Expected<size_t> read = llvm::sys::fs::readNativeFileSlice(
        file,
        llvm::MutableArrayRef<char>(reinterpret_cast<char *>(out.data() + done),
                                    requested),
        offset + done);
    if (!read)
      return read.takeError();
    if (*read == 0)
      return llvm::createStringError(llvm::errc::io_error,
                                     "payload file ended unexpectedly");
    if (statistics) {
      ++statistics->readWindows;
      statistics->readBytes += *read;
      statistics->maximumReadWindowBytes =
          std::max<uint64_t>(statistics->maximumReadWindowBytes, *read);
    }
    done += *read;
  }
  return llvm::Error::success();
}

llvm::Expected<std::string>
computeFileDigestWithStatistics(llvm::StringRef path,
                                ProgramDataIOStatistics *statistics);

class ScopedFile {
public:
  explicit ScopedFile(llvm::sys::fs::file_t file) : file(file) {}
  ~ScopedFile() { close(); }
  ScopedFile(const ScopedFile &) = delete;
  ScopedFile &operator=(const ScopedFile &) = delete;

  llvm::sys::fs::file_t get() const { return file; }
  void close() {
    llvm::sys::fs::file_t openFile =
        std::exchange(file, llvm::sys::fs::kInvalidFile);
    if (openFile != llvm::sys::fs::kInvalidFile)
      llvm::sys::fs::closeFile(openFile);
  }
  llvm::sys::fs::file_t release() {
    return std::exchange(file, llvm::sys::fs::kInvalidFile);
  }

private:
  llvm::sys::fs::file_t file;
};

/// Buffer-backed adapter exposing one in-memory byte span through the
/// frontend payload-source seam. Used to parse the bounded NPY header region
/// before the owned copy is written.
class SpanPayloadSource final : public frontend::ProgramPayloadSource {
public:
  SpanPayloadSource(llvm::ArrayRef<uint8_t> span, uint64_t fileSize)
      : span(span), fileSize(fileSize) {}

  uint64_t getPayloadFileSize() const override { return fileSize; }

  llvm::Error
  readPayloadBytes(uint64_t offset,
                   llvm::MutableArrayRef<uint8_t> out) const override {
    if (offset > span.size() || out.size() > span.size() - offset)
      return llvm::createStringError(llvm::errc::invalid_argument,
                                     "payload header read is outside the "
                                     "bounded header span");
    std::memcpy(out.data(), span.data() + offset, out.size());
    return llvm::Error::success();
  }

private:
  llvm::ArrayRef<uint8_t> span;
  uint64_t fileSize;
};

std::optional<uint64_t> elementCount(llvm::ArrayRef<int64_t> shape) {
  uint64_t count = 1;
  for (int64_t dim : shape) {
    uint64_t next = 0;
    if (dim < 0 || !checkedMulU64(count, static_cast<uint64_t>(dim), next))
      return std::nullopt;
    count = next;
  }
  return count;
}

struct ValidatedPayload {
  ProgramElementType dtype;
  std::vector<int64_t> shape;
  uint64_t dataOffset = 0;
};

llvm::Expected<ValidatedPayload>
validatePayloadHeader(const frontend::ProgramPayloadSource &source,
                      llvm::StringRef locator, ProgramDataFailure *failure) {
  llvm::Expected<frontend::NpyPayloadHeader> header =
      frontend::parseNpyPayloadHeader(source, locator);
  if (!header) {
    llvm::consumeError(header.takeError());
    return fail(ProgramDataFailureKind::HeaderInvalid, locator,
                "npy payload header cannot be parsed", failure);
  }
  if (header->fortranOrder)
    return fail(ProgramDataFailureKind::UnsupportedEncoding, locator,
                "npy payload must be row-major", failure);

  auto decoded = frontend::decodeProgramNpyDescr(header->descr);
  if (!decoded)
    return fail(ProgramDataFailureKind::UnsupportedEncoding, locator,
                "npy payload has unsupported dtype", failure);
  std::optional<int64_t> admittedBytes =
      getProgramDTypeElementBytes(decoded->first);
  if (!admittedBytes)
    return fail(ProgramDataFailureKind::UnsupportedEncoding, locator,
                "npy payload dtype is not admitted at the program boundary",
                failure);
  if (decoded->second != static_cast<uint64_t>(*admittedBytes))
    return fail(ProgramDataFailureKind::UnsupportedEncoding, locator,
                "npy payload dtype width disagrees with the program boundary",
                failure);

  std::optional<uint64_t> elements = elementCount(header->shape);
  if (!elements)
    return fail(ProgramDataFailureKind::SizeOverflow, locator,
                "npy tensor element count is not representable", failure);
  uint64_t payloadBytes = 0;
  if (!checkedMulU64(*elements, decoded->second, payloadBytes))
    return fail(ProgramDataFailureKind::SizeOverflow, locator,
                "npy tensor byte count is not representable", failure);
  uint64_t expectedFileBytes = 0;
  if (!checkedAddU64(header->dataOffset, payloadBytes, expectedFileBytes))
    return fail(ProgramDataFailureKind::SizeOverflow, locator,
                "npy payload extent is not representable", failure);
  if (source.getPayloadFileSize() < expectedFileBytes)
    return fail(ProgramDataFailureKind::TruncatedPayload, locator,
                "npy payload file is smaller than the tensor payload", failure);
  if (source.getPayloadFileSize() != expectedFileBytes)
    return fail(ProgramDataFailureKind::TrailingPayload, locator,
                "npy payload file is larger than the exact tensor payload",
                failure);

  return ValidatedPayload{decoded->first, std::move(header->shape),
                          header->dataOffset};
}

} // namespace

llvm::StringRef stringifyProgramDataFailureKind(ProgramDataFailureKind kind) {
  switch (kind) {
  case ProgramDataFailureKind::MissingPayload:
    return "missing-payload";
  case ProgramDataFailureKind::HeaderInvalid:
    return "header-invalid";
  case ProgramDataFailureKind::UnsupportedEncoding:
    return "unsupported-encoding";
  case ProgramDataFailureKind::ShapeMismatch:
    return "shape-mismatch";
  case ProgramDataFailureKind::DTypeMismatch:
    return "dtype-mismatch";
  case ProgramDataFailureKind::TruncatedPayload:
    return "truncated-payload";
  case ProgramDataFailureKind::TrailingPayload:
    return "trailing-payload";
  case ProgramDataFailureKind::SizeOverflow:
    return "size-overflow";
  case ProgramDataFailureKind::DigestMismatch:
    return "digest-mismatch";
  case ProgramDataFailureKind::MissingRange:
    return "missing-range";
  case ProgramDataFailureKind::MaterializationIO:
    return "materialization-io";
  }
  llvm_unreachable("unknown program data failure kind");
}

std::optional<int64_t> getProgramDTypeElementBytes(ProgramElementType dtype) {
  if (dtype == ProgramElementType::Bool)
    return std::nullopt;
  return getProgramElementByteCount(dtype);
}

llvm::Expected<ProgramDataSource> ProgramDataSource::establish(
    llvm::StringRef path, llvm::StringRef ownedFilePath,
    llvm::StringRef locator, ProgramDataFailure *failure) {
  return establish(path, ownedFilePath, locator, failure, {});
}

llvm::Expected<ProgramDataSource> ProgramDataSource::establish(
    llvm::StringRef path, llvm::StringRef ownedFilePath,
    llvm::StringRef locator, ProgramDataFailure *failure,
    std::shared_ptr<ProgramDataIOStatistics> statistics) {
  llvm::Expected<llvm::sys::fs::file_t> inputFile =
      llvm::sys::fs::openNativeFileForRead(path);
  if (!inputFile)
    return fail(ProgramDataFailureKind::MissingPayload, locator,
                "failed to open payload: " +
                    llvm::toString(inputFile.takeError()),
                failure);
  if (statistics) {
    ++statistics->sourceOpens;
    ++statistics->fileOpens;
  }
  ScopedFile input(*inputFile);

  // Derive identity and size from the opened file, not from a path stat that
  // could name a different inode by the time the open completes.
  llvm::sys::fs::file_status sourceStatus;
  if (std::error_code error = llvm::sys::fs::status(input.get(), sourceStatus))
    return fail(ProgramDataFailureKind::MissingPayload, locator,
                "failed to stat opened payload: " + error.message(), failure);
  if (!llvm::sys::fs::is_regular_file(sourceStatus))
    return fail(ProgramDataFailureKind::MissingPayload, locator,
                "payload path is not a regular file", failure);
  const uint64_t sourceFileSize = sourceStatus.getSize();

  // The header region is bounded (frontend cap: 1 MiB). Parse and validate
  // it before any byte is written to the owned file, so malformed payloads
  // cost bounded I/O only.
  const size_t headerSpanBytes = static_cast<size_t>(
      std::min<uint64_t>(sourceFileSize, kProgramDataReadWindowBytes));
  llvm::SmallVector<uint8_t, 16> headerSpan(headerSpanBytes);
  if (llvm::Error readError =
          readFileSpan(input.get(), 0, headerSpan, statistics.get())) {
    llvm::consumeError(std::move(readError));
    return fail(ProgramDataFailureKind::MissingPayload, locator,
                "failed to read the payload header region", failure);
  }

  SpanPayloadSource headerSource(headerSpan, sourceFileSize);
  if (statistics)
    ++statistics->headerReads;
  llvm::Expected<ValidatedPayload> preliminary =
      validatePayloadHeader(headerSource, locator, failure);
  if (!preliminary)
    return preliminary.takeError();

  // Stream the file into transaction-private storage while hashing the exact
  // bytes written. The owned file, not the user path, is the source of all
  // later reads.
  std::error_code outputError;
  llvm::raw_fd_ostream output(ownedFilePath, outputError);
  if (outputError) {
    return failIO(ProgramDataFailureKind::MaterializationIO, locator,
                  "failed to create owned payload file: " +
                      outputError.message(),
                  failure);
  }
  if (statistics)
    ++statistics->fileOpens;

  auto abort = [&](llvm::StringRef detail) -> llvm::Error {
    output.close();
    llvm::sys::fs::remove(ownedFilePath);
    return failIO(ProgramDataFailureKind::MaterializationIO, locator, detail,
                  failure);
  };

  llvm::SHA256 hash;
  std::vector<uint8_t> window;
  uint64_t written = 0;
  while (written < sourceFileSize) {
    const size_t chunk = static_cast<size_t>(std::min<uint64_t>(
        sourceFileSize - written, kProgramDataReadWindowBytes));
    window.resize(chunk);
    if (llvm::Error readError =
            readFileSpan(input.get(), written, window, statistics.get()))
      return abort("failed to copy payload bytes: " +
                   llvm::toString(std::move(readError)));
    hash.update(window);
    output.write(reinterpret_cast<const char *>(window.data()), chunk);
    if (!output.has_error()) {
      written += chunk;
      continue;
    }
    return abort("failed to write owned payload file");
  }
  output.close();
  if (output.has_error())
    return abort("failed to close owned payload file");

  llvm::Expected<llvm::sys::fs::file_t> ownedFile =
      llvm::sys::fs::openNativeFileForRead(ownedFilePath);
  if (!ownedFile) {
    llvm::sys::fs::remove(ownedFilePath);
    return failIO(ProgramDataFailureKind::MaterializationIO, locator,
                  "failed to open owned payload file: " +
                      llvm::toString(ownedFile.takeError()),
                  failure);
  }
  if (statistics)
    ++statistics->fileOpens;
  ScopedFile owned(*ownedFile);
  llvm::sys::fs::file_status ownedStatus;
  if (std::error_code error = llvm::sys::fs::status(owned.get(), ownedStatus)) {
    owned.close();
    llvm::sys::fs::remove(ownedFilePath);
    return failIO(ProgramDataFailureKind::MaterializationIO, locator,
                  "failed to stat owned payload file: " + error.message(),
                  failure);
  }
  const uint64_t ownedFileSize = ownedStatus.getSize();
  const size_t ownedHeaderSpanBytes = static_cast<size_t>(
      std::min<uint64_t>(ownedFileSize, kProgramDataReadWindowBytes));
  llvm::SmallVector<uint8_t, 16> ownedHeaderSpan(ownedHeaderSpanBytes);
  if (llvm::Error readError =
          readFileSpan(owned.get(), 0, ownedHeaderSpan, statistics.get())) {
    owned.close();
    llvm::sys::fs::remove(ownedFilePath);
    return failIO(ProgramDataFailureKind::MaterializationIO, locator,
                  "failed to read owned payload header: " +
                      llvm::toString(std::move(readError)),
                  failure);
  }
  SpanPayloadSource ownedHeaderSource(ownedHeaderSpan, ownedFileSize);
  if (statistics)
    ++statistics->headerReads;
  llvm::Expected<ValidatedPayload> validated =
      validatePayloadHeader(ownedHeaderSource, locator, failure);
  if (!validated) {
    owned.close();
    llvm::sys::fs::remove(ownedFilePath);
    return validated.takeError();
  }

  // The digest carried by the source must describe the bytes that downstream
  // readers actually own, not merely the buffer presented to the write call.
  // Reuse the already-read header span as the first digest window, then stream
  // the remainder from the persistent owned descriptor.
  const std::string copiedDigest = hexEncode(hash.final());
  llvm::SHA256 ownedHash;
  ownedHash.update(ownedHeaderSpan);
  uint64_t digested = ownedHeaderSpan.size();
  while (digested < ownedFileSize) {
    const size_t chunk = static_cast<size_t>(std::min<uint64_t>(
        ownedFileSize - digested, kProgramDataReadWindowBytes));
    window.resize(chunk);
    if (llvm::Error readError =
            readFileSpan(owned.get(), digested, window, statistics.get())) {
      owned.close();
      llvm::sys::fs::remove(ownedFilePath);
      return failIO(ProgramDataFailureKind::MaterializationIO, locator,
                    "failed to digest owned payload: " +
                        llvm::toString(std::move(readError)),
                    failure);
    }
    ownedHash.update(window);
    digested += chunk;
  }
  const std::string ownedDigest = hexEncode(ownedHash.final());
  if (ownedDigest != copiedDigest) {
    owned.close();
    llvm::sys::fs::remove(ownedFilePath);
    return fail(ProgramDataFailureKind::DigestMismatch, locator,
                "owned payload digest disagrees with copied content", failure);
  }

  return ProgramDataSource(
      ownedFilePath.str(), locator.str(), std::move(validated->dtype),
      std::move(validated->shape), validated->dataOffset, ownedFileSize,
      ownedDigest, owned.release(), std::move(statistics));
}

ProgramDataSource::ProgramDataSource(ProgramDataSource &&other) noexcept
    : ownedFilePath(std::move(other.ownedFilePath)),
      locator(std::move(other.locator)), dtype(std::move(other.dtype)),
      shape(std::move(other.shape)), payloadOffset(other.payloadOffset),
      size(other.size), digest(std::move(other.digest)),
      ownedFile(std::exchange(other.ownedFile, llvm::sys::fs::kInvalidFile)),
      statistics(std::move(other.statistics)) {
  other.ownedFilePath.clear();
}

ProgramDataSource &
ProgramDataSource::operator=(ProgramDataSource &&other) noexcept {
  if (this == &other)
    return *this;
  releaseOwnedFile();
  ownedFilePath = std::move(other.ownedFilePath);
  locator = std::move(other.locator);
  dtype = std::move(other.dtype);
  shape = std::move(other.shape);
  payloadOffset = other.payloadOffset;
  size = other.size;
  digest = std::move(other.digest);
  ownedFile = std::exchange(other.ownedFile, llvm::sys::fs::kInvalidFile);
  statistics = std::move(other.statistics);
  other.ownedFilePath.clear();
  return *this;
}

ProgramDataSource::~ProgramDataSource() { releaseOwnedFile(); }

void ProgramDataSource::releaseOwnedFile() noexcept {
  llvm::sys::fs::file_t openFile =
      std::exchange(ownedFile, llvm::sys::fs::kInvalidFile);
  if (openFile != llvm::sys::fs::kInvalidFile)
    llvm::sys::fs::closeFile(openFile);
  if (!ownedFilePath.empty())
    llvm::sys::fs::remove(ownedFilePath);
  ownedFilePath.clear();
}

llvm::Error
ProgramDataSource::readPayloadBytes(uint64_t offset,
                                    llvm::MutableArrayRef<uint8_t> out) const {
  return readRange(offset, out);
}

llvm::Error
ProgramDataSource::readRange(uint64_t offset,
                             llvm::MutableArrayRef<uint8_t> out) const {
  if (offset > size || out.size() > size - offset)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "payload range is outside the owned file");
  if (ownedFile == llvm::sys::fs::kInvalidFile)
    return llvm::createStringError(llvm::errc::bad_file_descriptor,
                                   "owned payload file is closed");
  return readFileSpan(ownedFile, offset, out, statistics.get());
}

llvm::Expected<ProgramDataRange> ProgramDataRange::create(
    ProgramTensorId tensorId, ProgramElementType dtype,
    llvm::ArrayRef<int64_t> globalShape, llvm::ArrayRef<int64_t> localShape,
    frontend::ProgramDistributionKind distribution,
    ProgramDataRangeOrigin origin, llvm::ArrayRef<int64_t> sliceOffsets,
    llvm::ArrayRef<int64_t> sliceSizes, llvm::ArrayRef<int64_t> sliceStrides,
    SourceDataId sourceId, const ProgramDataSource &source,
    ProgramDataFailure *failure) {
  const std::string locator = source.getLocator().str();
  if (dtype != source.getDType())
    return fail(ProgramDataFailureKind::DTypeMismatch, locator,
                "program tensor dtype disagrees with payload source", failure);
  std::optional<int64_t> elementBytes = getProgramDTypeElementBytes(dtype);
  if (!elementBytes)
    return fail(ProgramDataFailureKind::UnsupportedEncoding, locator,
                "program tensor dtype is not admitted at the program "
                "boundary",
                failure);

  const size_t rank = globalShape.size();
  if (sliceOffsets.size() != rank || sliceSizes.size() != rank ||
      sliceStrides.size() != rank || localShape.size() != rank ||
      source.getShape().size() != rank)
    return fail(ProgramDataFailureKind::ShapeMismatch, locator,
                "program tensor rank is inconsistent", failure);

  // The source origin carries a shape proof: an original source stores the
  // global tensor, a materialized shard stores exactly the local tensor.
  if (origin == ProgramDataRangeOrigin::OriginalSource) {
    if (!llvm::equal(source.getShape(), globalShape))
      return fail(ProgramDataFailureKind::ShapeMismatch, locator,
                  "original payload source shape does not equal the global "
                  "program tensor shape",
                  failure);
  } else {
    if (!llvm::equal(source.getShape(), localShape))
      return fail(ProgramDataFailureKind::ShapeMismatch, locator,
                  "materialized shard source shape does not equal the local "
                  "program tensor shape",
                  failure);
    if (llvm::any_of(sliceOffsets, [](int64_t offset) { return offset != 0; }))
      return fail(ProgramDataFailureKind::ShapeMismatch, locator,
                  "materialized shard slice must start at the origin", failure);
    if (!llvm::equal(sliceSizes, localShape))
      return fail(ProgramDataFailureKind::ShapeMismatch, locator,
                  "materialized shard slice must cover the local tensor "
                  "exactly",
                  failure);
  }

  std::vector<int64_t> payloadShape(source.getShape());
  std::vector<uint64_t> payloadStrides(rank, 1);
  uint64_t stride = 1;
  for (size_t reverse = rank; reverse > 0; --reverse) {
    size_t dim = reverse - 1;
    payloadStrides[dim] = stride;
    uint64_t next = 0;
    if (!checkedMulU64(stride, static_cast<uint64_t>(payloadShape[dim]), next))
      return fail(ProgramDataFailureKind::SizeOverflow, locator,
                  "payload row stride is not representable", failure);
    stride = next;
  }

  uint64_t regionStartElement = 0;
  uint64_t regionElements = 1;
  bool contiguous = true;
  for (size_t dim = 0; dim < rank; ++dim) {
    if (sliceOffsets[dim] < 0 || sliceSizes[dim] < 0)
      return fail(ProgramDataFailureKind::ShapeMismatch, locator,
                  "program tensor slice has negative coordinates", failure);
    if (sliceStrides[dim] != 1)
      return fail(ProgramDataFailureKind::UnsupportedEncoding, locator,
                  "program tensor slice stride must be 1", failure);
    if (static_cast<uint64_t>(sliceOffsets[dim]) >
            static_cast<uint64_t>(globalShape[dim]) ||
        static_cast<uint64_t>(sliceSizes[dim]) >
            static_cast<uint64_t>(globalShape[dim]) -
                static_cast<uint64_t>(sliceOffsets[dim]))
      return fail(ProgramDataFailureKind::ShapeMismatch, locator,
                  "program tensor slice is outside the global tensor", failure);
    if (sliceSizes[dim] != localShape[dim])
      return fail(ProgramDataFailureKind::ShapeMismatch, locator,
                  "program tensor slice does not match the local shape",
                  failure);

    uint64_t start = 0;
    if (!checkedMulU64(static_cast<uint64_t>(sliceOffsets[dim]),
                       payloadStrides[dim], start) ||
        !checkedAddU64(regionStartElement, start, regionStartElement))
      return fail(ProgramDataFailureKind::SizeOverflow, locator,
                  "program tensor slice offset is not representable", failure);
    uint64_t elements = 0;
    if (!checkedMulU64(regionElements, static_cast<uint64_t>(sliceSizes[dim]),
                       elements))
      return fail(ProgramDataFailureKind::SizeOverflow, locator,
                  "program tensor region size is not representable", failure);
    regionElements = elements;
    if (dim > 0 &&
        (sliceOffsets[dim] != 0 || sliceSizes[dim] != payloadShape[dim]))
      contiguous = false;
  }

  uint64_t regionOffsetBytes = 0;
  uint64_t regionLength = 0;
  if (!checkedMulU64(regionStartElement, static_cast<uint64_t>(*elementBytes),
                     regionOffsetBytes) ||
      !checkedMulU64(regionElements, static_cast<uint64_t>(*elementBytes),
                     regionLength))
    return fail(ProgramDataFailureKind::SizeOverflow, locator,
                "program tensor region byte size is not representable",
                failure);
  const uint64_t sourcePayloadBytes =
      source.getSize() > source.getPayloadOffset()
          ? source.getSize() - source.getPayloadOffset()
          : 0;
  if (regionOffsetBytes > sourcePayloadBytes ||
      regionLength > sourcePayloadBytes - regionOffsetBytes)
    return fail(ProgramDataFailureKind::TruncatedPayload, locator,
                "program tensor region is outside the payload source", failure);

  return ProgramDataRange(tensorId, dtype, std::vector<int64_t>(globalShape),
                          std::vector<int64_t>(localShape), distribution,
                          std::vector<int64_t>(sliceOffsets),
                          std::vector<int64_t>(sliceSizes), sourceId,
                          regionOffsetBytes, regionLength,
                          std::move(payloadShape), contiguous);
}

llvm::Error
ProgramDataRange::materialize(const ProgramDataSource &source,
                              llvm::MutableArrayRef<uint8_t> out) const {
  if (out.size() != regionLength)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "materialization buffer does not match the "
                                   "program tensor region byte count");
  if (contiguous) {
    uint64_t fileOffset = 0;
    if (!checkedAddU64(source.getPayloadOffset(), regionOffset, fileOffset))
      return llvm::createStringError(llvm::errc::invalid_argument,
                                     "program tensor file offset overflows");
    return source.readRange(fileOffset, out);
  }

  std::optional<int64_t> elementBytes = getProgramDTypeElementBytes(dtype);
  if (!elementBytes)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "program tensor dtype is unsupported");
  const size_t rank = payloadShape.size();
  std::vector<uint64_t> payloadStrides(rank, 1);
  for (size_t reverse = rank; reverse > 1; --reverse) {
    size_t dim = reverse - 1;
    uint64_t stride = 0;
    if (payloadShape[dim] < 0 ||
        !checkedMulU64(payloadStrides[dim],
                       static_cast<uint64_t>(payloadShape[dim]), stride))
      return llvm::createStringError(llvm::errc::invalid_argument,
                                     "payload row stride overflows");
    payloadStrides[dim - 1] = stride;
  }

  std::optional<uint64_t> localElements = elementCount(localShape);
  uint64_t localBytes = 0;
  if (!localElements ||
      !checkedMulU64(*localElements, static_cast<uint64_t>(*elementBytes),
                     localBytes) ||
      localBytes != regionLength)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "program tensor region geometry is invalid");
  llvm::Error result = llvm::Error::success();
  std::vector<uint64_t> coordinate(rank, 0);
  uint64_t written = 0;
  for (uint64_t linear = 0; linear < *localElements && !result; ++linear) {
    uint64_t remaining = linear;
    uint64_t payloadLinear = 0;
    for (size_t reverse = rank; reverse > 0; --reverse) {
      size_t dim = reverse - 1;
      coordinate[dim] = remaining % static_cast<uint64_t>(localShape[dim]);
      remaining /= static_cast<uint64_t>(localShape[dim]);
      uint64_t payloadCoordinate = 0;
      uint64_t contribution = 0;
      if (!checkedAddU64(static_cast<uint64_t>(sliceOffsets[dim]),
                         coordinate[dim], payloadCoordinate) ||
          !checkedMulU64(payloadCoordinate, payloadStrides[dim],
                         contribution) ||
          !checkedAddU64(payloadLinear, contribution, payloadLinear)) {
        result = llvm::createStringError(llvm::errc::invalid_argument,
                                         "strided program tensor addressing "
                                         "overflows");
        break;
      }
    }
    if (result)
      break;
    uint64_t payloadByte = 0;
    if (!checkedMulU64(payloadLinear, static_cast<uint64_t>(*elementBytes),
                       payloadByte)) {
      result = llvm::createStringError(llvm::errc::invalid_argument,
                                       "strided program tensor addressing "
                                       "overflows");
      break;
    }
    uint64_t fileOffset = 0;
    if (!checkedAddU64(source.getPayloadOffset(), payloadByte, fileOffset)) {
      result = llvm::createStringError(llvm::errc::invalid_argument,
                                       "strided program tensor file offset "
                                       "overflows");
      break;
    }
    result = source.readRange(
        fileOffset, out.slice(written, static_cast<size_t>(*elementBytes)));
    written += static_cast<uint64_t>(*elementBytes);
  }
  return result;
}

llvm::Error
ProgramDataRange::materializeWindow(const ProgramDataSource &source,
                                    uint64_t regionByteOffset,
                                    llvm::MutableArrayRef<uint8_t> out) const {
  if (out.empty() && regionByteOffset <= regionLength)
    return llvm::Error::success();
  if (regionByteOffset > regionLength ||
      out.size() > regionLength - regionByteOffset)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "program tensor window exceeds the checked "
                                   "region byte count");
  if (contiguous) {
    uint64_t fileOffset = 0;
    if (!checkedAddU64(source.getPayloadOffset(), regionOffset, fileOffset) ||
        !checkedAddU64(fileOffset, regionByteOffset, fileOffset))
      return llvm::createStringError(llvm::errc::invalid_argument,
                                     "program tensor file offset overflows");
    return source.readRange(fileOffset, out);
  }

  std::optional<int64_t> elementBytes = getProgramDTypeElementBytes(dtype);
  if (!elementBytes)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "program tensor dtype is unsupported");
  if (regionByteOffset % static_cast<uint64_t>(*elementBytes) != 0 ||
      out.size() % static_cast<uint64_t>(*elementBytes) != 0)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "strided program tensor windows must be "
                                   "element-aligned");
  const uint64_t firstElement =
      regionByteOffset / static_cast<uint64_t>(*elementBytes);
  const uint64_t elementCountInWindow =
      out.size() / static_cast<uint64_t>(*elementBytes);
  if (elementCountInWindow == 0)
    return llvm::Error::success();

  const size_t rank = payloadShape.size();
  std::vector<uint64_t> payloadStrides(rank, 1);
  for (size_t reverse = rank; reverse > 1; --reverse) {
    size_t dim = reverse - 1;
    uint64_t stride = 0;
    if (payloadShape[dim] < 0 ||
        !checkedMulU64(payloadStrides[dim],
                       static_cast<uint64_t>(payloadShape[dim]), stride))
      return llvm::createStringError(llvm::errc::invalid_argument,
                                     "payload row stride overflows");
    payloadStrides[dim - 1] = stride;
  }
  llvm::Error result = llvm::Error::success();
  std::vector<uint64_t> coordinate(rank, 0);
  uint64_t written = 0;
  for (uint64_t linear = firstElement;
       linear < firstElement + elementCountInWindow && !result; ++linear) {
    uint64_t remaining = linear;
    uint64_t payloadLinear = 0;
    for (size_t reverse = rank; reverse > 0; --reverse) {
      size_t dim = reverse - 1;
      coordinate[dim] = remaining % static_cast<uint64_t>(localShape[dim]);
      remaining /= static_cast<uint64_t>(localShape[dim]);
      uint64_t payloadCoordinate = 0;
      uint64_t contribution = 0;
      if (!checkedAddU64(static_cast<uint64_t>(sliceOffsets[dim]),
                         coordinate[dim], payloadCoordinate) ||
          !checkedMulU64(payloadCoordinate, payloadStrides[dim],
                         contribution) ||
          !checkedAddU64(payloadLinear, contribution, payloadLinear)) {
        result = llvm::createStringError(llvm::errc::invalid_argument,
                                         "strided program tensor addressing "
                                         "overflows");
        break;
      }
    }
    if (result)
      break;
    uint64_t payloadByte = 0;
    if (!checkedMulU64(payloadLinear, static_cast<uint64_t>(*elementBytes),
                       payloadByte)) {
      result = llvm::createStringError(llvm::errc::invalid_argument,
                                       "strided program tensor addressing "
                                       "overflows");
      break;
    }
    uint64_t fileOffset = 0;
    if (!checkedAddU64(source.getPayloadOffset(), payloadByte, fileOffset)) {
      result = llvm::createStringError(llvm::errc::invalid_argument,
                                       "strided program tensor file offset "
                                       "overflows");
      break;
    }
    result = source.readRange(
        fileOffset, out.slice(written, static_cast<size_t>(*elementBytes)));
    written += static_cast<uint64_t>(*elementBytes);
  }
  return result;
}

llvm::Error ProgramDataHandoff::materializeRangeWindow(
    const ProgramDataRange &range, uint64_t regionByteOffset,
    llvm::MutableArrayRef<uint8_t> out) const {
  if (llvm::Error error = range.materializeWindow(
          getSource(range.getSourceId()), regionByteOffset, out))
    return error;
  return llvm::Error::success();
}

llvm::Expected<ProgramDataRangeMaterialization>
ProgramDataHandoff::beginRangeMaterialization(ProgramTensorId tensorId) const {
  if (!findRange(tensorId))
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "cannot begin materialization for an unknown program tensor range");
  if (ioStatistics)
    ++ioStatistics->rangeMaterializations;
  return ProgramDataRangeMaterialization(*this, tensorId);
}

ProgramDataRangeMaterialization::ProgramDataRangeMaterialization(
    ProgramDataRangeMaterialization &&other) noexcept
    : handoff(std::exchange(other.handoff, nullptr)), tensorId(other.tensorId) {
}

ProgramDataRangeMaterialization &ProgramDataRangeMaterialization::operator=(
    ProgramDataRangeMaterialization &&other) noexcept {
  if (this == &other)
    return *this;
  handoff = std::exchange(other.handoff, nullptr);
  tensorId = other.tensorId;
  return *this;
}

llvm::Error ProgramDataRangeMaterialization::materializeWindow(
    uint64_t regionByteOffset, llvm::MutableArrayRef<uint8_t> out) const {
  if (!handoff)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "cannot read through a moved-from program data materialization");
  const ProgramDataRange *range = handoff->findRange(tensorId);
  if (!range)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "program tensor range disappeared during materialization");
  return handoff->materializeRangeWindow(*range, regionByteOffset, out);
}

ProgramDataHandoff::ProgramDataHandoff(std::string storageParent,
                                       bool collectIOStatistics)
    : storageParent(std::move(storageParent)) {
  if (collectIOStatistics)
    ioStatistics = std::make_shared<ProgramDataIOStatistics>();
}

ProgramDataHandoff::ProgramDataHandoff(ProgramDataHandoff &&other) noexcept
    : storageParent(std::move(other.storageParent)),
      ownedDirectory(std::move(other.ownedDirectory)),
      nextOwnedFileOrdinal(other.nextOwnedFileOrdinal),
      ioStatistics(std::move(other.ioStatistics)),
      sources(std::move(other.sources)),
      candidates(std::move(other.candidates)), ranges(std::move(other.ranges)) {
  other.ownedDirectory.clear();
}

ProgramDataHandoff &
ProgramDataHandoff::operator=(ProgramDataHandoff &&other) noexcept {
  if (this == &other)
    return *this;
  releaseOwnedStorage();
  storageParent = std::move(other.storageParent);
  ownedDirectory = std::move(other.ownedDirectory);
  nextOwnedFileOrdinal = other.nextOwnedFileOrdinal;
  ioStatistics = std::move(other.ioStatistics);
  sources = std::move(other.sources);
  candidates = std::move(other.candidates);
  ranges = std::move(other.ranges);
  other.ownedDirectory.clear();
  return *this;
}

ProgramDataHandoff::~ProgramDataHandoff() { releaseOwnedStorage(); }

void ProgramDataHandoff::releaseOwnedStorage() noexcept {
  // Sources close their persistent handles and unlink their individual files
  // before the containing directory is removed.
  candidates.clear();
  sources.clear();
  if (!ownedDirectory.empty())
    llvm::sys::fs::remove_directories(ownedDirectory);
  ownedDirectory.clear();
}

llvm::Error
ProgramDataHandoff::ensureOwnedDirectory(llvm::StringRef locator,
                                         ProgramDataFailure *failure) {
  if (!ownedDirectory.empty())
    return llvm::Error::success();
  if (storageParent.empty())
    return fail(ProgramDataFailureKind::MaterializationIO, locator,
                "program data handoff has no storage parent", failure);
  llvm::SmallString<256> prefix(storageParent);
  llvm::sys::path::append(prefix, ".wafer-program-data");
  llvm::SmallString<256> created;
  if (std::error_code error =
          llvm::sys::fs::createUniqueDirectory(prefix, created))
    return failIO(ProgramDataFailureKind::MaterializationIO, locator,
                  "failed to create owned payload directory: " +
                      error.message(),
                  failure);
  ownedDirectory = created.str().str();
  return llvm::Error::success();
}

namespace {
/// Owned-file path for one source or candidate. The counter keeps names
/// unique inside the transaction.
std::string ownedPathFor(llvm::StringRef ownedDirectory, llvm::StringRef kind,
                         int64_t index) {
  llvm::SmallString<256> path(ownedDirectory);
  llvm::sys::path::append(path,
                          (llvm::Twine(kind) + "-" + llvm::Twine(index)).str());
  return path.str().str();
}
} // namespace

llvm::Expected<SourceDataId>
ProgramDataHandoff::establishSource(llvm::StringRef path,
                                    llvm::StringRef locator,
                                    ProgramDataFailure *failure) {
  if (llvm::Error error = ensureOwnedDirectory(locator, failure))
    return std::move(error);
  std::string ownedPath = ownedPathFor(
      ownedDirectory, "source", static_cast<int64_t>(nextOwnedFileOrdinal++));
  llvm::Expected<ProgramDataSource> source = ProgramDataSource::establish(
      path, ownedPath, locator, failure, ioStatistics);
  if (!source)
    return source.takeError();
  sources.push_back(std::make_unique<ProgramDataSource>(std::move(*source)));
  if (ioStatistics)
    ioStatistics->digestPasses += 2;
  return SourceDataId{static_cast<int64_t>(sources.size()) - 1};
}

llvm::Expected<const ProgramDataSource *>
ProgramDataHandoff::establishHelperOutput(llvm::StringRef path,
                                          llvm::StringRef locator,
                                          ProgramDataFailure *failure) {
  if (llvm::Error error = ensureOwnedDirectory(locator, failure))
    return std::move(error);
  std::string ownedPath =
      ownedPathFor(ownedDirectory, "candidate",
                   static_cast<int64_t>(nextOwnedFileOrdinal++));
  llvm::Expected<ProgramDataSource> source = ProgramDataSource::establish(
      path, ownedPath, locator, failure, ioStatistics);
  if (!source)
    return source.takeError();
  candidates.push_back(std::make_unique<ProgramDataSource>(std::move(*source)));
  if (ioStatistics) {
    ioStatistics->digestPasses += 2;
    ++ioStatistics->helperOutputReadbacks;
  }
  return candidates.back().get();
}

SourceDataId
ProgramDataHandoff::adoptCandidate(const ProgramDataSource *candidate) {
  assert(candidate && "adopted candidate must not be null");
  for (auto current = candidates.begin(); current != candidates.end();
       ++current) {
    if (current->get() != candidate)
      continue;
    sources.push_back(std::move(*current));
    candidates.erase(current);
    return SourceDataId{static_cast<int64_t>(sources.size()) - 1};
  }
  llvm_unreachable("adopted candidate must come from this handoff");
}

void ProgramDataHandoff::discardUnadoptedCandidates() { candidates.clear(); }

const ProgramDataSource *
ProgramDataHandoff::findByLocator(llvm::StringRef locator) const {
  for (const std::unique_ptr<ProgramDataSource> &candidate : candidates)
    if (candidate->getLocator() == locator)
      return candidate.get();
  for (const std::unique_ptr<ProgramDataSource> &source : sources)
    if (source->getLocator() == locator)
      return source.get();
  return nullptr;
}

llvm::Error ProgramDataHandoff::addRange(ProgramDataRange range) {
  if (!range.getSourceId().isValid() ||
      static_cast<size_t>(range.getSourceId().value) >= sources.size())
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "program data range references an unknown source");
  if (findRange(range.getTensorId()))
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "program tensor identity already owns a data range");
  ranges.push_back(std::move(range));
  return llvm::Error::success();
}

const ProgramDataSource &ProgramDataHandoff::getSource(SourceDataId id) const {
  assert(id.isValid() && static_cast<size_t>(id.value) < sources.size());
  return *sources[id.value];
}

const ProgramDataRange *
ProgramDataHandoff::findRange(ProgramTensorId tensorId) const {
  for (const ProgramDataRange &range : ranges)
    if (range.getTensorId() == tensorId)
      return &range;
  return nullptr;
}

llvm::Error
ProgramDataHandoff::materializeRange(const ProgramDataRange &range,
                                     llvm::MutableArrayRef<uint8_t> out) const {
  if (llvm::Error error =
          range.materialize(getSource(range.getSourceId()), out))
    return error;
  if (ioStatistics)
    ++ioStatistics->rangeMaterializations;
  return llvm::Error::success();
}

llvm::Error ProgramDataHandoff::materializeSourceToFile(
    SourceDataId id, llvm::StringRef path, ProgramDataFailure *failure) {
  const ProgramDataSource &source = getSource(id);
  const std::string locator = source.getLocator().str();
  llvm::SmallString<256> parent(path);
  llvm::sys::path::remove_filename(parent);
  if (std::error_code error = llvm::sys::fs::create_directories(parent))
    return failIO(ProgramDataFailureKind::MaterializationIO, locator,
                  "failed to create payload destination: " + error.message(),
                  failure);
  std::error_code error;
  llvm::raw_fd_ostream output(path, error);
  if (error)
    return failIO(ProgramDataFailureKind::MaterializationIO, locator,
                  "failed to open payload destination: " + error.message(),
                  failure);
  if (ioStatistics)
    ++ioStatistics->fileOpens;
  std::vector<uint8_t> window;
  const uint64_t size = source.getSize();
  uint64_t written = 0;
  while (written < size) {
    const uint64_t remaining = size - written;
    const size_t chunk = static_cast<size_t>(
        std::min<uint64_t>(remaining, kProgramDataReadWindowBytes));
    window.resize(chunk);
    if (llvm::Error readError = source.readRange(written, window)) {
      llvm::Error classified =
          failIO(ProgramDataFailureKind::MaterializationIO, locator,
                 "failed to read owned payload for materialization: " +
                     llvm::toString(std::move(readError)),
                 failure);
      output.close();
      return classified;
    }
    output.write(reinterpret_cast<const char *>(window.data()), chunk);
    if (!output.has_error()) {
      written += chunk;
      continue;
    }
    output.close();
    return failIO(ProgramDataFailureKind::MaterializationIO, locator,
                  "failed to write payload destination", failure);
  }
  output.close();
  if (output.has_error())
    return failIO(ProgramDataFailureKind::MaterializationIO, locator,
                  "failed to close payload destination", failure);

  llvm::Expected<std::string> digest =
      computeFileDigestWithStatistics(path, ioStatistics.get());
  if (!digest)
    return failIO(ProgramDataFailureKind::MaterializationIO, locator,
                  "failed to digest the materialized payload: " +
                      llvm::toString(digest.takeError()),
                  failure);
  if (*digest != source.getContentDigest())
    return fail(ProgramDataFailureKind::DigestMismatch, locator,
                "materialized payload digest disagrees with owned content",
                failure);
  if (ioStatistics) {
    ++ioStatistics->materializedFileWrites;
    ++ioStatistics->digestPasses;
    ioStatistics->materializedWriteBytes += size;
  }
  return llvm::Error::success();
}

llvm::Expected<bool> ProgramDataHandoff::verifyShardAgainstSource(
    const ProgramDataSource &shard, SourceDataId originalSourceId,
    llvm::ArrayRef<int64_t> offsets, llvm::ArrayRef<int64_t> sizes,
    ProgramDataFailure *failure) {
  const std::string locator = shard.getLocator().str();
  const ProgramDataSource &original = getSource(originalSourceId);
  llvm::Expected<std::string> shardRegionDigest = computePayloadRegionDigest(
      shard, std::vector<int64_t>(sizes.size(), 0), sizes, failure);
  llvm::Expected<std::string> originalRegionDigest =
      computePayloadRegionDigest(original, offsets, sizes, failure);
  if (!shardRegionDigest || !originalRegionDigest) {
    llvm::Error joined = llvm::Error::success();
    if (!shardRegionDigest)
      joined =
          llvm::joinErrors(std::move(joined), shardRegionDigest.takeError());
    if (!originalRegionDigest)
      joined =
          llvm::joinErrors(std::move(joined), originalRegionDigest.takeError());
    return std::move(joined);
  }
  (void)locator;
  if (ioStatistics)
    ioStatistics->digestPasses += 2;
  return *shardRegionDigest == *originalRegionDigest;
}

namespace {
llvm::Expected<std::string>
computeFileDigestWithStatistics(llvm::StringRef path,
                                ProgramDataIOStatistics *statistics) {
  llvm::Expected<llvm::sys::fs::file_t> file =
      llvm::sys::fs::openNativeFileForRead(path);
  if (!file)
    return file.takeError();
  if (statistics)
    ++statistics->fileOpens;
  ScopedFile input(*file);
  llvm::sys::fs::file_status status;
  if (std::error_code statusError = llvm::sys::fs::status(input.get(), status))
    return llvm::createStringError(
        statusError, "failed to stat file for digest: %s", path.str().c_str());
  const uint64_t size = status.getSize();
  llvm::SHA256 hash;
  std::vector<uint8_t> window;
  uint64_t consumed = 0;
  llvm::Error readError = llvm::Error::success();
  while (!readError && consumed < size) {
    const size_t chunk = static_cast<size_t>(
        std::min<uint64_t>(size - consumed, kProgramDataReadWindowBytes));
    window.resize(chunk);
    readError = readFileSpan(input.get(), consumed, window, statistics);
    if (!readError)
      hash.update(window);
    consumed += chunk;
  }
  if (readError)
    return std::move(readError);
  return hexEncode(hash.final());
}
} // namespace

llvm::Expected<std::string>
ProgramDataHandoff::computeFileDigest(llvm::StringRef path) {
  return computeFileDigestWithStatistics(path, nullptr);
}

llvm::Expected<std::string> computePayloadRegionDigest(
    const ProgramDataSource &source, llvm::ArrayRef<int64_t> offsets,
    llvm::ArrayRef<int64_t> sizes, ProgramDataFailure *failure) {
  const std::string locator = source.getLocator().str();
  const llvm::ArrayRef<int64_t> payloadShape = source.getShape();
  const size_t rank = payloadShape.size();
  if (offsets.size() != rank || sizes.size() != rank)
    return fail(ProgramDataFailureKind::ShapeMismatch, locator,
                "payload region rank is inconsistent", failure);
  std::optional<int64_t> elementBytes =
      getProgramDTypeElementBytes(source.getDType());
  if (!elementBytes)
    return fail(ProgramDataFailureKind::UnsupportedEncoding, locator,
                "payload dtype is unsupported", failure);

  std::vector<uint64_t> payloadStrides(rank, 1);
  for (size_t reverse = rank; reverse > 1; --reverse) {
    size_t dim = reverse - 1;
    uint64_t stride = 0;
    if (payloadShape[dim] < 0 ||
        !checkedMulU64(payloadStrides[dim],
                       static_cast<uint64_t>(payloadShape[dim]), stride))
      return fail(ProgramDataFailureKind::SizeOverflow, locator,
                  "payload row stride overflows", failure);
    payloadStrides[dim - 1] = stride;
  }

  bool contiguous = true;
  uint64_t startElement = 0;
  uint64_t regionElements = 1;
  for (size_t dim = 0; dim < rank; ++dim) {
    if (offsets[dim] < 0 || sizes[dim] < 0 ||
        static_cast<uint64_t>(offsets[dim]) >
            static_cast<uint64_t>(payloadShape[dim]) ||
        static_cast<uint64_t>(sizes[dim]) >
            static_cast<uint64_t>(payloadShape[dim]) -
                static_cast<uint64_t>(offsets[dim]))
      return fail(ProgramDataFailureKind::ShapeMismatch, locator,
                  "payload region is outside the tensor", failure);
    uint64_t start = 0;
    if (!checkedMulU64(static_cast<uint64_t>(offsets[dim]), payloadStrides[dim],
                       start) ||
        !checkedAddU64(startElement, start, startElement))
      return fail(ProgramDataFailureKind::SizeOverflow, locator,
                  "payload region offset overflows", failure);
    uint64_t elements = 0;
    if (!checkedMulU64(regionElements, static_cast<uint64_t>(sizes[dim]),
                       elements))
      return fail(ProgramDataFailureKind::SizeOverflow, locator,
                  "payload region size overflows", failure);
    regionElements = elements;
    if (dim > 0 && (offsets[dim] != 0 || sizes[dim] != payloadShape[dim]))
      contiguous = false;
  }

  // Every failure below is classified at its site so the tail can forward
  // the error unchanged; no error is re-wrapped into a different kind.
  auto readError = [&](llvm::Error error) -> llvm::Error {
    return failIO(ProgramDataFailureKind::MaterializationIO, locator,
                  "failed to read owned payload for region digest: " +
                      llvm::toString(std::move(error)),
                  failure);
  };

  llvm::Error result = llvm::Error::success();
  llvm::SHA256 hash;
  const uint64_t elementWidth = static_cast<uint64_t>(*elementBytes);
  if (contiguous) {
    uint64_t byteOffset = 0;
    if (!checkedMulU64(startElement, elementWidth, byteOffset))
      result = fail(ProgramDataFailureKind::SizeOverflow, locator,
                    "payload region byte offset overflows", failure);
    std::vector<uint8_t> window;
    uint64_t remaining = 0;
    if (!checkedMulU64(regionElements, elementWidth, remaining))
      result = fail(ProgramDataFailureKind::SizeOverflow, locator,
                    "payload region byte size overflows", failure);
    uint64_t consumed = 0;
    while (!result && consumed < remaining) {
      const size_t chunk = static_cast<size_t>(std::min<uint64_t>(
          remaining - consumed, kProgramDataReadWindowBytes));
      window.resize(chunk);
      uint64_t payloadOffset = 0;
      uint64_t fileOffset = 0;
      if (!checkedAddU64(byteOffset, consumed, payloadOffset) ||
          !checkedAddU64(source.getPayloadOffset(), payloadOffset,
                         fileOffset)) {
        result = fail(ProgramDataFailureKind::SizeOverflow, locator,
                      "payload region file offset overflows", failure);
        break;
      }
      if (llvm::Error error = source.readRange(fileOffset, window))
        result = readError(std::move(error));
      else
        hash.update(window);
      consumed += chunk;
    }
  } else {
    std::vector<uint64_t> coordinate(rank, 0);
    std::vector<uint8_t> window;
    const size_t elementChunk = std::max<size_t>(
        1, kProgramDataReadWindowBytes / static_cast<size_t>(elementWidth));
    uint64_t remainingElements = regionElements;
    for (uint64_t linear = 0; linear < regionElements && !result;) {
      const size_t batch = static_cast<size_t>(
          std::min<uint64_t>(remainingElements, elementChunk));
      window.resize(batch * static_cast<size_t>(elementWidth));
      size_t windowBytes = 0;
      for (size_t item = 0; item < batch; ++item, ++linear) {
        uint64_t current = linear;
        uint64_t payloadElement = 0;
        for (size_t reverse = rank; reverse > 0; --reverse) {
          size_t dim = reverse - 1;
          coordinate[dim] = current % static_cast<uint64_t>(sizes[dim]);
          current /= static_cast<uint64_t>(sizes[dim]);
          uint64_t payloadCoordinate = 0;
          uint64_t contribution = 0;
          if (!checkedAddU64(static_cast<uint64_t>(offsets[dim]),
                             coordinate[dim], payloadCoordinate) ||
              !checkedMulU64(payloadCoordinate, payloadStrides[dim],
                             contribution) ||
              !checkedAddU64(payloadElement, contribution, payloadElement)) {
            result = fail(ProgramDataFailureKind::SizeOverflow, locator,
                          "payload region addressing overflows", failure);
            break;
          }
        }
        if (result)
          break;
        uint64_t payloadByte = 0;
        if (!checkedMulU64(payloadElement, elementWidth, payloadByte)) {
          result = fail(ProgramDataFailureKind::SizeOverflow, locator,
                        "payload region addressing overflows", failure);
          break;
        }
        uint64_t fileOffset = 0;
        if (!checkedAddU64(source.getPayloadOffset(), payloadByte,
                           fileOffset)) {
          result = fail(ProgramDataFailureKind::SizeOverflow, locator,
                        "payload region file offset overflows", failure);
          break;
        }
        if (llvm::Error error = source.readRange(
                fileOffset, llvm::MutableArrayRef<uint8_t>(
                                window.data() + windowBytes,
                                static_cast<size_t>(elementWidth)))) {
          result = readError(std::move(error));
          break;
        }
        windowBytes += static_cast<size_t>(elementWidth);
      }
      if (!result)
        hash.update(llvm::ArrayRef<uint8_t>(window.data(), windowBytes));
      remainingElements -= batch;
    }
  }
  if (result)
    return result;
  return hexEncode(hash.final());
}

void ProgramDataIOStatistics::print(llvm::raw_ostream &stream) const {
  stream << "source_opens=" << sourceOpens << " file_opens=" << fileOpens
         << " read_windows=" << readWindows << " read_bytes=" << readBytes
         << " maximum_read_window_bytes=" << maximumReadWindowBytes
         << " header_reads=" << headerReads << " digest_passes=" << digestPasses
         << " helper_output_readbacks=" << helperOutputReadbacks
         << " range_materializations=" << rangeMaterializations
         << " materialized_file_writes=" << materializedFileWrites
         << " materialized_write_bytes=" << materializedWriteBytes;
}

} // namespace wafer::compiler
