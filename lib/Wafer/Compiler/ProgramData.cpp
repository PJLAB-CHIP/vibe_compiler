//===- ProgramData.cpp - Transaction-owned program data ------------------===//

#include "Wafer/Compiler/ProgramData.h"

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

/// Materialization window for streaming digest and file copies. Header
/// parsing is separately bounded by the frontend NPY header cap; this window
/// bounds all file I/O.
constexpr size_t kProgramDataFileWindowBytes = 1 << 20;

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
                         llvm::MutableArrayRef<uint8_t> out) {
  uint64_t done = 0;
  while (done < out.size()) {
    llvm::Expected<size_t> read = llvm::sys::fs::readNativeFileSlice(
        file,
        llvm::MutableArrayRef<char>(
            reinterpret_cast<char *>(out.data() + done),
            out.size() - done),
        offset + done);
    if (!read)
      return read.takeError();
    if (*read == 0)
      return llvm::createStringError(llvm::errc::io_error,
                                     "payload file ended unexpectedly");
    done += *read;
  }
  return llvm::Error::success();
}

/// Buffer-backed adapter exposing one in-memory byte span through the
/// frontend payload-source seam. Used to parse the bounded NPY header region
/// before the owned copy is written.
class SpanPayloadSource final : public frontend::ProgramPayloadSource {
public:
  SpanPayloadSource(llvm::ArrayRef<uint8_t> span, uint64_t fileSize)
      : span(span), fileSize(fileSize) {}

  uint64_t getPayloadFileSize() const override { return fileSize; }

  llvm::Error readPayloadBytes(uint64_t offset,
                               llvm::MutableArrayRef<uint8_t> out)
      const override {
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

} // namespace

llvm::StringRef
stringifyProgramDataFailureKind(ProgramDataFailureKind kind) {
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

std::optional<int64_t> getProgramDTypeElementBytes(llvm::StringRef dtype) {
  if (dtype == "i8" || dtype == "ui8")
    return 1;
  if (dtype == "i16" || dtype == "ui16")
    return 2;
  if (dtype == "f16" || dtype == "bf16")
    return 2;
  if (dtype == "i32" || dtype == "ui32")
    return 4;
  if (dtype == "f32" || dtype == "tf32")
    return 4;
  if (dtype == "i64" || dtype == "ui64")
    return 8;
  if (dtype == "f64")
    return 8;
  // i1 is not admitted at the program boundary: there is no boolean target
  // representation and conversion yet. NPY files carrying it still parse at
  // the source layer but fail establishment here.
  return std::nullopt;
}

llvm::Expected<ProgramDataSource>
ProgramDataSource::establish(llvm::StringRef path, llvm::StringRef ownedFilePath,
                             llvm::StringRef locator,
                             ProgramDataFailure *failure) {
  std::error_code statusError;
  llvm::sys::fs::file_status status;
  if ((statusError = llvm::sys::fs::status(path, status)))
    return fail(ProgramDataFailureKind::MissingPayload, locator,
                "failed to stat payload: " + statusError.message(), failure);
  if (!llvm::sys::fs::is_regular_file(status))
    return fail(ProgramDataFailureKind::MissingPayload, locator,
                "payload path is not a regular file", failure);
  const uint64_t sourceFileSize = status.getSize();

  llvm::Expected<llvm::sys::fs::file_t> inputFile =
      llvm::sys::fs::openNativeFileForRead(path);
  if (!inputFile)
    return fail(ProgramDataFailureKind::MissingPayload, locator,
                "failed to open payload: " +
                    llvm::toString(inputFile.takeError()),
                failure);

  // The header region is bounded (frontend cap: 1 MiB). Parse and validate
  // it before any byte is written to the owned file, so malformed payloads
  // cost bounded I/O only.
  const size_t headerSpanBytes = static_cast<size_t>(
      std::min<uint64_t>(sourceFileSize, kProgramDataFileWindowBytes));
  llvm::SmallVector<uint8_t, 16> headerSpan(headerSpanBytes);
  if (llvm::Error readError =
          readFileSpan(*inputFile, 0, headerSpan)) {
    llvm::consumeError(std::move(readError));
    llvm::sys::fs::closeFile(*inputFile);
    return fail(ProgramDataFailureKind::MissingPayload, locator,
                "failed to read the payload header region", failure);
  }

  SpanPayloadSource headerSource(headerSpan, sourceFileSize);
  llvm::Expected<frontend::NpyPayloadHeader> header =
      frontend::parseNpyPayloadHeader(headerSource, locator);
  if (!header) {
    llvm::consumeError(header.takeError());
    llvm::sys::fs::closeFile(*inputFile);
    return fail(ProgramDataFailureKind::HeaderInvalid, locator,
                "npy payload header cannot be parsed", failure);
  }
  if (header->fortranOrder) {
    llvm::sys::fs::closeFile(*inputFile);
    return fail(ProgramDataFailureKind::UnsupportedEncoding, locator,
                "npy payload must be row-major", failure);
  }

  auto decoded = frontend::decodeProgramNpyDescr(header->descr);
  if (!decoded) {
    llvm::sys::fs::closeFile(*inputFile);
    return fail(ProgramDataFailureKind::UnsupportedEncoding, locator,
                "npy payload has unsupported dtype", failure);
  }
  const std::string dtype = decoded->first.str();
  const uint64_t elementBytes = decoded->second;
  if (!getProgramDTypeElementBytes(dtype)) {
    llvm::sys::fs::closeFile(*inputFile);
    return fail(ProgramDataFailureKind::UnsupportedEncoding, locator,
                "npy payload dtype is not admitted at the program boundary",
                failure);
  }
  if (decoded->second != static_cast<uint64_t>(*getProgramDTypeElementBytes(dtype))) {
    llvm::sys::fs::closeFile(*inputFile);
    return fail(ProgramDataFailureKind::UnsupportedEncoding, locator,
                "npy payload dtype width disagrees with the program boundary",
                failure);
  }

  std::optional<uint64_t> elements = elementCount(header->shape);
  if (!elements) {
    llvm::sys::fs::closeFile(*inputFile);
    return fail(ProgramDataFailureKind::SizeOverflow, locator,
                "npy tensor element count is not representable", failure);
  }
  uint64_t payloadBytes = 0;
  if (!checkedMulU64(*elements, elementBytes, payloadBytes)) {
    llvm::sys::fs::closeFile(*inputFile);
    return fail(ProgramDataFailureKind::SizeOverflow, locator,
                "npy tensor byte count is not representable", failure);
  }
  uint64_t expectedFileBytes = 0;
  if (!checkedAddU64(header->dataOffset, payloadBytes, expectedFileBytes)) {
    llvm::sys::fs::closeFile(*inputFile);
    return fail(ProgramDataFailureKind::SizeOverflow, locator,
                "npy payload extent is not representable", failure);
  }
  if (sourceFileSize < expectedFileBytes) {
    llvm::sys::fs::closeFile(*inputFile);
    return fail(ProgramDataFailureKind::TruncatedPayload, locator,
                "npy payload file is smaller than the tensor payload",
                failure);
  }
  if (sourceFileSize != expectedFileBytes) {
    llvm::sys::fs::closeFile(*inputFile);
    return fail(ProgramDataFailureKind::TrailingPayload, locator,
                "npy payload file is larger than the exact tensor payload",
                failure);
  }

  // Stream the file into transaction-private storage while hashing the exact
  // bytes written. The owned file, not the user path, is the source of all
  // later reads.
  std::error_code outputError;
  llvm::raw_fd_ostream output(ownedFilePath, outputError);
  if (outputError) {
    llvm::sys::fs::closeFile(*inputFile);
    return failIO(ProgramDataFailureKind::MaterializationIO, locator,
                  "failed to create owned payload file: " +
                      outputError.message(),
                  failure);
  }

  auto abort = [&](llvm::StringRef detail) -> llvm::Error {
    output.close();
    llvm::sys::fs::closeFile(*inputFile);
    llvm::sys::fs::remove(ownedFilePath);
    return failIO(ProgramDataFailureKind::MaterializationIO, locator, detail,
                  failure);
  };

  llvm::SHA256 hash;
  std::vector<uint8_t> window;
  uint64_t written = 0;
  while (written < sourceFileSize) {
    const size_t chunk = static_cast<size_t>(std::min<uint64_t>(
        sourceFileSize - written, kProgramDataFileWindowBytes));
    window.resize(chunk);
    if (llvm::Error readError =
            readFileSpan(*inputFile, written, window))
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
  llvm::sys::fs::closeFile(*inputFile);

  llvm::ArrayRef<uint8_t> digest = hash.final();
  return ProgramDataSource(ownedFilePath.str(), locator.str(), std::move(dtype),
                           std::move(header->shape), header->dataOffset,
                           written, hexEncode(digest));
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
  llvm::Expected<llvm::sys::fs::file_t> file =
      llvm::sys::fs::openNativeFileForRead(ownedFilePath);
  if (!file)
    return file.takeError();
  llvm::Error result = readFileSpan(*file, offset, out);
  llvm::sys::fs::closeFile(*file);
  return result;
}

llvm::Expected<ProgramDataRange> ProgramDataRange::create(
    ProgramTensorId tensorId, llvm::StringRef dtype,
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
                  "materialized shard slice must start at the origin",
                  failure);
    if (!llvm::equal(sliceSizes, localShape))
      return fail(ProgramDataFailureKind::ShapeMismatch, locator,
                  "materialized shard slice must cover the local tensor "
                  "exactly",
                  failure);
  }

  std::vector<int64_t> payloadShape(source.getShape());
  std::vector<int64_t> payloadStrides(rank, 1);
  uint64_t stride = 1;
  for (size_t reverse = rank; reverse > 0; --reverse) {
    size_t dim = reverse - 1;
    payloadStrides[dim] = static_cast<int64_t>(stride);
    uint64_t next = 0;
    if (!checkedMulU64(stride,
                       static_cast<uint64_t>(payloadShape[dim]), next))
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
                  "program tensor slice is outside the global tensor",
                  failure);
    if (sliceSizes[dim] != localShape[dim])
      return fail(ProgramDataFailureKind::ShapeMismatch, locator,
                  "program tensor slice does not match the local shape",
                  failure);

    uint64_t start = 0;
    if (!checkedMulU64(static_cast<uint64_t>(sliceOffsets[dim]),
                       static_cast<uint64_t>(payloadStrides[dim]), start) ||
        !checkedAddU64(regionStartElement, start, regionStartElement))
      return fail(ProgramDataFailureKind::SizeOverflow, locator,
                  "program tensor slice offset is not representable", failure);
    uint64_t elements = 0;
    if (!checkedMulU64(regionElements,
                       static_cast<uint64_t>(sliceSizes[dim]), elements))
      return fail(ProgramDataFailureKind::SizeOverflow, locator,
                  "program tensor region size is not representable", failure);
    regionElements = elements;
    if (dim > 0 && (sliceOffsets[dim] != 0 ||
                    sliceSizes[dim] != payloadShape[dim]))
      contiguous = false;
  }

  uint64_t regionOffsetBytes = 0;
  uint64_t regionLength = 0;
  if (!checkedMulU64(regionStartElement,
                     static_cast<uint64_t>(*elementBytes), regionOffsetBytes) ||
      !checkedMulU64(regionElements, static_cast<uint64_t>(*elementBytes),
                     regionLength))
    return fail(ProgramDataFailureKind::SizeOverflow, locator,
                "program tensor region byte size is not representable",
                failure);
  const uint64_t sourcePayloadBytes = source.getSize() > source.getPayloadOffset()
                                          ? source.getSize() -
                                                source.getPayloadOffset()
                                          : 0;
  if (regionOffsetBytes > sourcePayloadBytes ||
      regionLength > sourcePayloadBytes - regionOffsetBytes)
    return fail(ProgramDataFailureKind::TruncatedPayload, locator,
                "program tensor region is outside the payload source",
                failure);

  return ProgramDataRange(tensorId, dtype.str(), std::vector<int64_t>(globalShape),
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
  if (contiguous)
    return source.readRange(source.getPayloadOffset() + regionOffset, out);

  std::optional<int64_t> elementBytes = getProgramDTypeElementBytes(dtype);
  if (!elementBytes)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "program tensor dtype is unsupported");
  const size_t rank = payloadShape.size();
  std::vector<int64_t> payloadStrides(rank, 1);
  for (size_t reverse = rank; reverse > 1; --reverse) {
    size_t dim = reverse - 1;
    int64_t stride = payloadStrides[dim] * payloadShape[dim];
    if (payloadStrides[dim] != 0 &&
        stride / payloadStrides[dim] != payloadShape[dim])
      return llvm::createStringError(llvm::errc::invalid_argument,
                                     "payload row stride overflows");
    payloadStrides[dim - 1] = stride;
  }

  std::optional<uint64_t> localElements = elementCount(localShape);
  if (!localElements ||
      *localElements * static_cast<uint64_t>(*elementBytes) != regionLength)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "program tensor region geometry is invalid");
  llvm::Expected<llvm::sys::fs::file_t> file =
      llvm::sys::fs::openNativeFileForRead(source.getOwnedFilePath());
  if (!file)
    return file.takeError();
  llvm::Error result = llvm::Error::success();
  std::vector<int64_t> coordinate(rank, 0);
  uint64_t written = 0;
  for (uint64_t linear = 0; linear < *localElements && !result; ++linear) {
    uint64_t remaining = linear;
    uint64_t payloadLinear = 0;
    for (size_t reverse = rank; reverse > 0; --reverse) {
      size_t dim = reverse - 1;
      coordinate[dim] = remaining % static_cast<uint64_t>(localShape[dim]);
      remaining /= static_cast<uint64_t>(localShape[dim]);
      int64_t payloadCoordinate = sliceOffsets[dim] + coordinate[dim];
      payloadLinear += static_cast<uint64_t>(payloadCoordinate) *
                       static_cast<uint64_t>(payloadStrides[dim]);
    }
    uint64_t payloadByte = 0;
    if (!checkedMulU64(payloadLinear,
                       static_cast<uint64_t>(*elementBytes), payloadByte)) {
      result = llvm::createStringError(llvm::errc::invalid_argument,
                                       "strided program tensor addressing "
                                       "overflows");
      break;
    }
    result = readFileSpan(*file, source.getPayloadOffset() + payloadByte,
                          out.slice(written, static_cast<size_t>(*elementBytes)));
    written += static_cast<uint64_t>(*elementBytes);
  }
  llvm::sys::fs::closeFile(*file);
  return result;
}

ProgramDataHandoff::ProgramDataHandoff(std::string scratchDirectory)
    : scratchDirectory(std::move(scratchDirectory)) {}

namespace {
/// Owned-file path for one source or candidate. The counter keeps names
/// unique inside the transaction.
std::string ownedPathFor(llvm::StringRef scratchDirectory,
                         llvm::StringRef kind, int64_t index) {
  llvm::SmallString<256> path(scratchDirectory);
  llvm::sys::path::append(path, "program-data-sources");
  llvm::sys::path::append(path,
                          (llvm::Twine(kind) + "-" + llvm::Twine(index)).str());
  return path.str().str();
}
} // namespace

llvm::Expected<SourceDataId>
ProgramDataHandoff::establishSource(llvm::StringRef path,
                                    llvm::StringRef locator,
                                    ProgramDataFailure *failure) {
  if (scratchDirectory.empty())
    return fail(ProgramDataFailureKind::MaterializationIO, locator,
                "program data handoff has no scratch directory", failure);
  llvm::SmallString<256> ownedDirectory(scratchDirectory);
  llvm::sys::path::append(ownedDirectory, "program-data-sources");
  if (std::error_code error =
          llvm::sys::fs::create_directories(ownedDirectory))
    return fail(ProgramDataFailureKind::MaterializationIO, locator,
                "failed to create owned payload directory: " + error.message(),
                failure);
  std::string ownedPath = ownedPathFor(
      scratchDirectory, "source", static_cast<int64_t>(sources.size()));
  llvm::Expected<ProgramDataSource> source =
      ProgramDataSource::establish(path, ownedPath, locator, failure);
  if (!source)
    return source.takeError();
  sources.push_back(
      std::make_unique<ProgramDataSource>(std::move(*source)));
  ++ioStatistics.sourceOpens;
  ++ioStatistics.headerReads;
  ++ioStatistics.digestPasses;
  return SourceDataId{static_cast<int64_t>(sources.size()) - 1};
}

llvm::Expected<const ProgramDataSource *>
ProgramDataHandoff::establishHelperOutput(llvm::StringRef path,
                                          llvm::StringRef locator,
                                          ProgramDataFailure *failure) {
  if (scratchDirectory.empty())
    return fail(ProgramDataFailureKind::MaterializationIO, locator,
                "program data handoff has no scratch directory", failure);
  llvm::SmallString<256> ownedDirectory(scratchDirectory);
  llvm::sys::path::append(ownedDirectory, "program-data-sources");
  if (std::error_code error =
          llvm::sys::fs::create_directories(ownedDirectory))
    return fail(ProgramDataFailureKind::MaterializationIO, locator,
                "failed to create owned payload directory: " + error.message(),
                failure);
  std::string ownedPath = ownedPathFor(
      scratchDirectory, "candidate", static_cast<int64_t>(candidates.size()));
  llvm::Expected<ProgramDataSource> source =
      ProgramDataSource::establish(path, ownedPath, locator, failure);
  if (!source)
    return source.takeError();
  candidates.push_back(
      std::make_unique<ProgramDataSource>(std::move(*source)));
  ++ioStatistics.sourceOpens;
  ++ioStatistics.headerReads;
  ++ioStatistics.digestPasses;
  ++ioStatistics.helperOutputReadbacks;
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

llvm::Error ProgramDataHandoff::materializeRange(
    const ProgramDataRange &range, llvm::MutableArrayRef<uint8_t> out) const {
  if (llvm::Error error = range.materialize(getSource(range.getSourceId()), out))
    return error;
  ++ioStatistics.rangeMaterializations;
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
  std::vector<uint8_t> window;
  const uint64_t size = source.getSize();
  uint64_t written = 0;
  while (written < size) {
    const uint64_t remaining = size - written;
    const size_t chunk = static_cast<size_t>(
        std::min<uint64_t>(remaining, kProgramDataFileWindowBytes));
    window.resize(chunk);
    if (llvm::Error readError = source.readRange(written, window)) {
      llvm::Error classified = failIO(
          ProgramDataFailureKind::MaterializationIO, locator,
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

  llvm::Expected<std::string> digest = computeFileDigest(path);
  if (!digest)
    return failIO(ProgramDataFailureKind::MaterializationIO, locator,
                  "failed to digest the materialized payload: " +
                      llvm::toString(digest.takeError()),
                  failure);
  if (*digest != source.getContentDigest())
    return fail(ProgramDataFailureKind::DigestMismatch, locator,
                "materialized payload digest disagrees with owned content",
                failure);
  ++ioStatistics.materializedFileWrites;
  ++ioStatistics.digestPasses;
  ioStatistics.materializedWriteBytes += size;
  return llvm::Error::success();
}

llvm::Expected<bool>
ProgramDataHandoff::verifyShardAgainstSource(
    const ProgramDataSource &shard, SourceDataId originalSourceId,
    llvm::ArrayRef<int64_t> offsets, llvm::ArrayRef<int64_t> sizes,
    ProgramDataFailure *failure) {
  const std::string locator = shard.getLocator().str();
  const ProgramDataSource &original = getSource(originalSourceId);
  llvm::Expected<std::string> shardRegionDigest = computePayloadRegionDigest(
      shard, std::vector<int64_t>(sizes.size(), 0), sizes, failure);
  llvm::Expected<std::string> originalRegionDigest = computePayloadRegionDigest(
      original, offsets, sizes, failure);
  if (!shardRegionDigest || !originalRegionDigest) {
    llvm::Error joined = llvm::Error::success();
    if (!shardRegionDigest)
      joined = llvm::joinErrors(std::move(joined),
                                shardRegionDigest.takeError());
    if (!originalRegionDigest)
      joined = llvm::joinErrors(std::move(joined),
                                originalRegionDigest.takeError());
    return std::move(joined);
  }
  (void)locator;
  ioStatistics.digestPasses += 2;
  return *shardRegionDigest == *originalRegionDigest;
}

llvm::Expected<std::string>
ProgramDataHandoff::computeFileDigest(llvm::StringRef path) {
  llvm::Expected<llvm::sys::fs::file_t> file =
      llvm::sys::fs::openNativeFileForRead(path);
  if (!file)
    return file.takeError();
  llvm::sys::fs::file_status status;
  std::error_code statusError;
  uint64_t size = 0;
  if ((statusError = llvm::sys::fs::status(path, status)))
    return llvm::createStringError(statusError,
                                   "failed to stat file for digest: %s",
                                   path.str().c_str());
  size = status.getSize();
  llvm::SHA256 hash;
  std::vector<uint8_t> window;
  uint64_t consumed = 0;
  llvm::Error readError = llvm::Error::success();
  while (!readError && consumed < size) {
    const size_t chunk = static_cast<size_t>(std::min<uint64_t>(
        size - consumed, kProgramDataFileWindowBytes));
    window.resize(chunk);
    readError = readFileSpan(*file, consumed, window);
    if (!readError)
      hash.update(window);
    consumed += chunk;
  }
  llvm::sys::fs::closeFile(*file);
  if (readError)
    return std::move(readError);
  return hexEncode(hash.final());
}

llvm::Expected<std::string>
computePayloadRegionDigest(const ProgramDataSource &source,
                           llvm::ArrayRef<int64_t> offsets,
                           llvm::ArrayRef<int64_t> sizes,
                           ProgramDataFailure *failure) {
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

  std::vector<int64_t> payloadStrides(rank, 1);
  for (size_t reverse = rank; reverse > 1; --reverse) {
    size_t dim = reverse - 1;
    int64_t stride = payloadStrides[dim] * payloadShape[dim];
    if (payloadStrides[dim] != 0 && stride / payloadStrides[dim] != payloadShape[dim])
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
    if (!checkedMulU64(static_cast<uint64_t>(offsets[dim]),
                       static_cast<uint64_t>(payloadStrides[dim]), start) ||
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

  llvm::Expected<llvm::sys::fs::file_t> file =
      llvm::sys::fs::openNativeFileForRead(source.getOwnedFilePath());
  if (!file)
    return failIO(ProgramDataFailureKind::MaterializationIO, locator,
                  "failed to open owned payload for region digest: " +
                      llvm::toString(file.takeError()),
                  failure);
  llvm::Error result = llvm::Error::success();
  llvm::SHA256 hash;
  const uint64_t elementWidth = static_cast<uint64_t>(*elementBytes);
  if (contiguous) {
    uint64_t byteOffset = 0;
    if (!checkedMulU64(startElement, elementWidth, byteOffset))
      result = fail(ProgramDataFailureKind::SizeOverflow, locator,
                    "payload region byte offset overflows", failure);
    std::vector<uint8_t> window;
    uint64_t remaining = regionElements * elementWidth;
    uint64_t consumed = 0;
    while (!result && consumed < remaining) {
      const size_t chunk = static_cast<size_t>(std::min<uint64_t>(
          remaining - consumed, kProgramDataFileWindowBytes));
      window.resize(chunk);
      if (llvm::Error error = readFileSpan(
              *file, source.getPayloadOffset() + byteOffset + consumed,
              window))
        result = readError(std::move(error));
      else
        hash.update(window);
      consumed += chunk;
    }
  } else {
    std::vector<int64_t> coordinate(rank, 0);
    std::vector<uint8_t> window;
    const size_t elementChunk = std::max<size_t>(
        1, kProgramDataFileWindowBytes / static_cast<size_t>(elementWidth));
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
          payloadElement +=
              static_cast<uint64_t>(offsets[dim] + coordinate[dim]) *
              static_cast<uint64_t>(payloadStrides[dim]);
        }
        uint64_t payloadByte = 0;
        if (!checkedMulU64(payloadElement, elementWidth, payloadByte)) {
          result = fail(ProgramDataFailureKind::SizeOverflow, locator,
                        "payload region addressing overflows", failure);
          break;
        }
        if (llvm::Error error = readFileSpan(
                *file, source.getPayloadOffset() + payloadByte,
                llvm::MutableArrayRef<uint8_t>(
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
  llvm::sys::fs::closeFile(*file);
  if (result)
    return result;
  return hexEncode(hash.final());
}

void ProgramDataIOStatistics::print(llvm::raw_ostream &stream) const {
  stream << "source_opens=" << sourceOpens
         << " header_reads=" << headerReads
         << " digest_passes=" << digestPasses
         << " helper_output_readbacks=" << helperOutputReadbacks
         << " range_materializations=" << rangeMaterializations
         << " materialized_file_writes=" << materializedFileWrites
         << " materialized_write_bytes=" << materializedWriteBytes;
}

} // namespace wafer::compiler
