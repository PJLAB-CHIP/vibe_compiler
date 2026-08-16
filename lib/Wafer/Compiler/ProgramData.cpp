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

/// Materialization window for streaming digest and file copies. Reads over
/// pinned in-memory sources use direct copies; this window bounds file I/O.
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

std::string hexEncode(llvm::ArrayRef<uint8_t> bytes) {
  std::string hex;
  hex.reserve(bytes.size() * 2);
  llvm::raw_string_ostream stream(hex);
  for (uint8_t byte : bytes)
    stream << llvm::format_hex_no_prefix(byte, 2);
  stream.flush();
  return hex;
}

/// Pinned-buffer adapter exposing one established MemoryBuffer through the
/// frontend payload-source seam used by NPY header parsing.
class PinnedBufferPayloadSource final : public frontend::ProgramPayloadSource {
public:
  explicit PinnedBufferPayloadSource(const llvm::MemoryBuffer &buffer)
      : buffer(buffer) {}

  uint64_t getPayloadFileSize() const override {
    return buffer.getBufferSize();
  }

  llvm::Error readPayloadBytes(uint64_t offset,
                               llvm::MutableArrayRef<uint8_t> out)
      const override {
    if (offset > buffer.getBufferSize() ||
        out.size() > buffer.getBufferSize() - offset)
      return llvm::createStringError(llvm::errc::invalid_argument,
                                     "payload read is outside the pinned file");
    std::memcpy(out.data(), buffer.getBufferStart() + offset, out.size());
    return llvm::Error::success();
  }

private:
  const llvm::MemoryBuffer &buffer;
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
  }
  llvm_unreachable("unknown program data failure kind");
}

std::optional<int64_t> getProgramDTypeElementBytes(llvm::StringRef dtype) {
  if (dtype == "i1" || dtype == "i8" || dtype == "ui8")
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
  return std::nullopt;
}

llvm::Expected<ProgramDataSource>
ProgramDataSource::establish(llvm::StringRef path, llvm::StringRef locator,
                            ProgramDataFailure *failure) {
  auto bufferOrError = llvm::MemoryBuffer::getFile(path);
  if (!bufferOrError)
    return fail(ProgramDataFailureKind::MissingPayload, locator,
                "failed to open payload: " +
                    bufferOrError.getError().message(),
                failure);

  PinnedBufferPayloadSource adapter(**bufferOrError);
  llvm::Expected<frontend::NpyPayloadHeader> header =
      frontend::parseNpyPayloadHeader(adapter, locator);
  if (!header)
    return fail(ProgramDataFailureKind::HeaderInvalid, locator,
                llvm::toString(header.takeError()), failure);
  if (header->fortranOrder)
    return fail(ProgramDataFailureKind::UnsupportedEncoding, locator,
                "npy payload must be row-major", failure);

  auto decoded = frontend::decodeProgramNpyDescr(header->descr);
  if (!decoded)
    return fail(ProgramDataFailureKind::UnsupportedEncoding, locator,
                "npy payload has unsupported dtype", failure);
  const std::string dtype = decoded->first.str();
  const uint64_t elementBytes = decoded->second;

  std::optional<uint64_t> elements = elementCount(header->shape);
  if (!elements)
    return fail(ProgramDataFailureKind::SizeOverflow, locator,
                "npy tensor element count is not representable", failure);
  uint64_t payloadBytes = 0;
  if (!checkedMulU64(*elements, elementBytes, payloadBytes))
    return fail(ProgramDataFailureKind::SizeOverflow, locator,
                "npy tensor byte count is not representable", failure);
  if (header->dataOffset > header->fileSize ||
      payloadBytes > header->fileSize - header->dataOffset)
    return fail(ProgramDataFailureKind::TruncatedPayload, locator,
                "npy payload file is smaller than the tensor payload",
                failure);
  if (payloadBytes != header->fileSize - header->dataOffset)
    return fail(ProgramDataFailureKind::TrailingPayload, locator,
                "npy payload file is larger than the exact tensor payload",
                failure);

  llvm::SHA256 hash;
  hash.update((*bufferOrError)->getBuffer());
  llvm::ArrayRef<uint8_t> digest = hash.final();
  return ProgramDataSource(std::move(*bufferOrError), locator.str(),
                           std::move(dtype), std::move(header->shape),
                           header->dataOffset, hexEncode(digest));
}

llvm::Error
ProgramDataSource::readPayloadBytes(uint64_t offset,
                                    llvm::MutableArrayRef<uint8_t> out) const {
  return readRange(offset, out);
}

llvm::Error
ProgramDataSource::readRange(uint64_t offset,
                             llvm::MutableArrayRef<uint8_t> out) const {
  if (offset > getSize() || out.size() > getSize() - offset)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "payload range is outside the pinned file");
  std::memcpy(out.data(), content->getBufferStart() + offset, out.size());
  return llvm::Error::success();
}

llvm::Expected<ProgramDataRange> ProgramDataRange::create(
    ProgramTensorId tensorId, llvm::StringRef dtype,
    llvm::ArrayRef<int64_t> globalShape, llvm::ArrayRef<int64_t> localShape,
    frontend::ProgramDistributionKind distribution,
    llvm::ArrayRef<int64_t> sliceOffsets, llvm::ArrayRef<int64_t> sliceSizes,
    llvm::ArrayRef<int64_t> sliceStrides, SourceDataId sourceId,
    const ProgramDataSource &source, ProgramDataFailure *failure) {
  const std::string locator = source.getLocator().str();
  if (dtype != source.getDType())
    return fail(ProgramDataFailureKind::DTypeMismatch, locator,
                "program tensor dtype disagrees with payload source", failure);
  std::optional<int64_t> elementBytes = getProgramDTypeElementBytes(dtype);
  if (!elementBytes)
    return fail(ProgramDataFailureKind::UnsupportedEncoding, locator,
                "program tensor dtype is unsupported", failure);

  const size_t rank = globalShape.size();
  if (sliceOffsets.size() != rank || sliceSizes.size() != rank ||
      sliceStrides.size() != rank || localShape.size() != rank ||
      source.getShape().size() != rank)
    return fail(ProgramDataFailureKind::ShapeMismatch, locator,
                "program tensor rank is inconsistent", failure);

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
  std::vector<int64_t> coordinate(rank, 0);
  uint64_t written = 0;
  for (uint64_t linear = 0; linear < *localElements; ++linear) {
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
                       static_cast<uint64_t>(*elementBytes), payloadByte))
      return llvm::createStringError(llvm::errc::invalid_argument,
                                     "strided program tensor addressing "
                                     "overflows");
    if (llvm::Error error = source.readRange(
            source.getPayloadOffset() + payloadByte,
            out.slice(written, static_cast<size_t>(*elementBytes))))
      return error;
    written += static_cast<uint64_t>(*elementBytes);
  }
  return llvm::Error::success();
}

llvm::Expected<SourceDataId>
ProgramDataHandoff::establishSource(llvm::StringRef path,
                                    llvm::StringRef locator,
                                    ProgramDataFailure *failure) {
  llvm::Expected<ProgramDataSource> source =
      ProgramDataSource::establish(path, locator, failure);
  if (!source)
    return source.takeError();
  sources.push_back(
      std::make_unique<ProgramDataSource>(std::move(*source)));
  ++ioStatistics.sourceOpens;
  ++ioStatistics.headerReads;
  ++ioStatistics.digestPasses;
  return SourceDataId{static_cast<int64_t>(sources.size()) - 1};
}

SourceDataId ProgramDataHandoff::addSource(ProgramDataSource source) {
  sources.push_back(std::make_unique<ProgramDataSource>(std::move(source)));
  ++ioStatistics.sourceOpens;
  ++ioStatistics.headerReads;
  ++ioStatistics.digestPasses;
  return SourceDataId{static_cast<int64_t>(sources.size()) - 1};
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
  llvm::SmallString<256> parent(path);
  llvm::sys::path::remove_filename(parent);
  if (std::error_code error = llvm::sys::fs::create_directories(parent))
    return llvm::createStringError(error,
                                   "failed to create payload destination: %s",
                                   path.str().c_str());
  std::error_code error;
  llvm::raw_fd_ostream output(path, error);
  if (error)
    return llvm::createStringError(error, "failed to open payload destination");
  std::vector<uint8_t> window;
  const uint64_t size = source.getSize();
  uint64_t written = 0;
  while (written < size) {
    const uint64_t remaining = size - written;
    const size_t chunk = static_cast<size_t>(
        std::min<uint64_t>(remaining, kProgramDataFileWindowBytes));
    window.resize(chunk);
    if (llvm::Error readError = source.readRange(written, window))
      return readError;
    output.write(reinterpret_cast<const char *>(window.data()), chunk);
    if (!output.has_error())
      written += chunk;
    else
      return llvm::createStringError(llvm::errc::io_error,
                                     "failed to write payload destination");
  }
  output.close();
  if (output.has_error())
    return llvm::createStringError(llvm::errc::io_error,
                                   "failed to close payload destination");

  llvm::Expected<std::string> digest = computeFileDigest(path);
  if (!digest)
    return digest.takeError();
  if (*digest != source.getContentDigest()) {
    if (failure) {
      failure->kind = ProgramDataFailureKind::DigestMismatch;
      failure->locator = path.str();
      failure->detail = "materialized payload digest disagrees with owned "
                        "content";
    }
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "materialized payload digest disagrees with owned content: %s",
        path.str().c_str());
  }
  ++ioStatistics.materializedFileWrites;
  ++ioStatistics.digestPasses;
  ioStatistics.materializedWriteBytes += size;
  return llvm::Error::success();
}

llvm::Expected<ShardVerification>
ProgramDataHandoff::verifyShardAgainstSource(
    llvm::StringRef shardPath, llvm::StringRef shardLocator,
    SourceDataId originalSourceId, llvm::ArrayRef<int64_t> offsets,
    llvm::ArrayRef<int64_t> sizes, ProgramDataFailure *failure) {
  ProgramDataFailure shardFailure;
  llvm::Expected<ProgramDataSource> shard =
      ProgramDataSource::establish(shardPath, shardLocator, &shardFailure);
  if (!shard) {
    if (failure)
      *failure = std::move(shardFailure);
    return shard.takeError();
  }
  const ProgramDataSource &original = getSource(originalSourceId);
  llvm::Expected<std::string> shardRegionDigest = computePayloadRegionDigest(
      *shard, std::vector<int64_t>(sizes.size(), 0), sizes);
  llvm::Expected<std::string> originalRegionDigest =
      computePayloadRegionDigest(original, offsets, sizes);
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
  ++ioStatistics.shardReadbacks;
  ++ioStatistics.headerReads;
  // Whole-file shard digest from establishment plus both region digests.
  ioStatistics.digestPasses += 3;
  return ShardVerification{*shardRegionDigest == *originalRegionDigest,
                           std::move(*shard)};
}

llvm::Expected<std::string>
ProgramDataHandoff::computeFileDigest(llvm::StringRef path) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> bufferOrError =
      llvm::MemoryBuffer::getFile(path);
  if (!bufferOrError)
    return llvm::createStringError(bufferOrError.getError(),
                                   "failed to open file for digest: %s",
                                   path.str().c_str());
  llvm::SHA256 hash;
  hash.update((*bufferOrError)->getBuffer());
  return hexEncode(hash.final());
}

llvm::Expected<std::string>
computePayloadRegionDigest(const ProgramDataSource &source,
                           llvm::ArrayRef<int64_t> offsets,
                           llvm::ArrayRef<int64_t> sizes) {
  const llvm::ArrayRef<int64_t> payloadShape = source.getShape();
  const size_t rank = payloadShape.size();
  if (offsets.size() != rank || sizes.size() != rank)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "payload region rank is inconsistent");
  std::optional<int64_t> elementBytes =
      getProgramDTypeElementBytes(source.getDType());
  if (!elementBytes)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "payload dtype is unsupported");

  std::vector<int64_t> payloadStrides(rank, 1);
  for (size_t reverse = rank; reverse > 1; --reverse) {
    size_t dim = reverse - 1;
    int64_t stride = payloadStrides[dim] * payloadShape[dim];
    if (payloadStrides[dim] != 0 && stride / payloadStrides[dim] != payloadShape[dim])
      return llvm::createStringError(llvm::errc::invalid_argument,
                                     "payload row stride overflows");
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
      return llvm::createStringError(llvm::errc::invalid_argument,
                                     "payload region is outside the tensor");
    uint64_t start = 0;
    if (!checkedMulU64(static_cast<uint64_t>(offsets[dim]),
                       static_cast<uint64_t>(payloadStrides[dim]), start) ||
        !checkedAddU64(startElement, start, startElement))
      return llvm::createStringError(llvm::errc::invalid_argument,
                                     "payload region offset overflows");
    uint64_t elements = 0;
    if (!checkedMulU64(regionElements, static_cast<uint64_t>(sizes[dim]),
                       elements))
      return llvm::createStringError(llvm::errc::invalid_argument,
                                     "payload region size overflows");
    regionElements = elements;
    if (dim > 0 && (offsets[dim] != 0 || sizes[dim] != payloadShape[dim]))
      contiguous = false;
  }

  llvm::SHA256 hash;
  const uint64_t elementWidth = static_cast<uint64_t>(*elementBytes);
  if (contiguous) {
    uint64_t byteOffset = 0;
    if (!checkedMulU64(startElement, elementWidth, byteOffset))
      return llvm::createStringError(llvm::errc::invalid_argument,
                                     "payload region byte offset overflows");
    std::vector<uint8_t> window;
    uint64_t remaining = regionElements * elementWidth;
    uint64_t consumed = 0;
    while (consumed < remaining) {
      const size_t chunk = static_cast<size_t>(std::min<uint64_t>(
          remaining - consumed, kProgramDataFileWindowBytes));
      window.resize(chunk);
      if (llvm::Error error = source.readRange(
              source.getPayloadOffset() + byteOffset + consumed, window))
        return error;
      hash.update(window);
      consumed += chunk;
    }
    return hexEncode(hash.final());
  }

  std::vector<int64_t> coordinate(rank, 0);
  std::vector<uint8_t> window;
  const size_t elementChunk = std::max<size_t>(
      1, kProgramDataFileWindowBytes / static_cast<size_t>(elementWidth));
  uint64_t remainingElements = regionElements;
  for (uint64_t linear = 0; linear < regionElements;) {
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
      if (!checkedMulU64(payloadElement, elementWidth, payloadByte))
        return llvm::createStringError(llvm::errc::invalid_argument,
                                       "payload region addressing overflows");
      if (llvm::Error error = source.readRange(
              source.getPayloadOffset() + payloadByte,
              llvm::MutableArrayRef<uint8_t>(
                  window.data() + windowBytes,
                  static_cast<size_t>(elementWidth))))
        return error;
      windowBytes += static_cast<size_t>(elementWidth);
    }
    remainingElements -= batch;
    hash.update(llvm::ArrayRef<uint8_t>(window.data(), windowBytes));
  }
  return hexEncode(hash.final());
}

void ProgramDataIOStatistics::print(llvm::raw_ostream &stream) const {
  stream << "source_opens=" << sourceOpens
         << " header_reads=" << headerReads
         << " digest_passes=" << digestPasses
         << " shard_readbacks=" << shardReadbacks
         << " range_materializations=" << rangeMaterializations
         << " materialized_file_writes=" << materializedFileWrites
         << " materialized_write_bytes=" << materializedWriteBytes;
}

} // namespace wafer::compiler
