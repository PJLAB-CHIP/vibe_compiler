//===- ProgramData.h - Transaction-owned program data ------------*- C++ -*-===//
//
// Transaction-owned parameter/constant payload ownership for one compilation.
// ProgramDataSource pins verified source bytes, ProgramDataRange describes the
// checked byte region one program tensor consumes, and ProgramDataHandoff
// owns the live set for the lifetime of a CardExecutable. Tile bindings
// reference identity and range only; they never copy payload bytes.
//
//===----------------------------------------------------------------------===//

#ifndef WAFER_COMPILER_PROGRAMDATA_H
#define WAFER_COMPILER_PROGRAMDATA_H

#include "Wafer/Compiler/Compilation.h"
#include "Wafer/Frontend/Program.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace wafer::compiler {

/// Dense identity of one transaction-owned payload source.
struct SourceDataId {
  int64_t value = -1;

  friend bool operator==(SourceDataId lhs, SourceDataId rhs) {
    return lhs.value == rhs.value;
  }
  friend bool operator!=(SourceDataId lhs, SourceDataId rhs) {
    return !(lhs == rhs);
  }
  bool isValid() const { return value >= 0; }
};

/// Classification of a program-data establishment or materialization failure.
/// Callers keep recoverable control flow; these kinds are never parsed from
/// diagnostic strings.
enum class ProgramDataFailureKind : uint8_t {
  /// Payload path is missing, not a regular file, or cannot be opened.
  MissingPayload,
  /// NPY magic, version, or header cannot be parsed.
  HeaderInvalid,
  /// Payload uses Fortran order or an unsupported dtype encoding.
  UnsupportedEncoding,
  /// Payload shape does not agree with typed metadata.
  ShapeMismatch,
  /// Payload dtype does not agree with typed metadata.
  DTypeMismatch,
  /// File or header is smaller than the exact payload extent.
  TruncatedPayload,
  /// File is larger than the exact payload extent.
  TrailingPayload,
  /// Checked byte arithmetic overflowed.
  SizeOverflow,
  /// A materialized artifact's content digest disagrees with the owned source.
  DigestMismatch,
  /// A referenced range is unavailable in the handoff.
  MissingRange,
};

struct ProgramDataFailure {
  ProgramDataFailureKind kind = ProgramDataFailureKind::MissingPayload;
  std::string locator;
  std::string detail;
};

llvm::StringRef
stringifyProgramDataFailureKind(ProgramDataFailureKind kind);

/// Element width in bytes for one admitted program-boundary dtype, or
/// std::nullopt for unsupported dtypes. This is the single dtype-width table
/// shared by payload ownership and program invocation materialization.
std::optional<int64_t> getProgramDTypeElementBytes(llvm::StringRef dtype);

/// Transaction-owned, content-pinned payload source. The pinned MemoryBuffer
/// and the whole-file digest prove that later reads observe the exact bytes
/// verified at establishment; user path changes afterwards cannot change this
/// compilation. Implements the frontend payload-source seam so directory
/// verification reads from owned content instead of reopening paths.
class ProgramDataSource final : public frontend::ProgramPayloadSource {
public:
  ProgramDataSource(ProgramDataSource &&) = default;
  ProgramDataSource &operator=(ProgramDataSource &&) = default;
  ProgramDataSource(const ProgramDataSource &) = delete;
  ProgramDataSource &operator=(const ProgramDataSource &) = delete;

  /// Opens, verifies, and digests one payload file. On failure `failure`
  /// (when non-null) carries the classification; the returned error string
  /// is the human-readable detail.
  static llvm::Expected<ProgramDataSource>
  establish(llvm::StringRef path, llvm::StringRef locator,
            ProgramDataFailure *failure = nullptr);

  /// Program-relative locator used only for diagnostics and provenance; it is
  /// never a data key.
  llvm::StringRef getLocator() const { return locator; }
  llvm::StringRef getDType() const { return dtype; }
  llvm::ArrayRef<int64_t> getShape() const { return shape; }
  uint64_t getPayloadOffset() const { return payloadOffset; }
  llvm::StringRef getContentDigest() const { return digest; }

  uint64_t getPayloadFileSize() const override { return getSize(); }
  llvm::Error readPayloadBytes(uint64_t offset,
                               llvm::MutableArrayRef<uint8_t> out)
      const override;

  /// Whole pinned file size in bytes.
  uint64_t getSize() const { return content->getBufferSize(); }

  /// Bounded copy of a source-relative byte span. The span must lie inside
  /// the pinned file.
  llvm::Error readRange(uint64_t offset, llvm::MutableArrayRef<uint8_t> out)
      const;

private:
  ProgramDataSource(std::unique_ptr<llvm::MemoryBuffer> content,
                    std::string locator, std::string dtype,
                    std::vector<int64_t> shape, uint64_t payloadOffset,
                    std::string digest)
      : content(std::move(content)), locator(std::move(locator)),
        dtype(std::move(dtype)), shape(std::move(shape)),
        payloadOffset(payloadOffset), digest(std::move(digest)) {}

  std::unique_ptr<llvm::MemoryBuffer> content;
  std::string locator;
  std::string dtype;
  std::vector<int64_t> shape;
  uint64_t payloadOffset;
  std::string digest;
};

/// Checked byte region one program tensor consumes inside one source. A range
/// never owns the file; it must not outlive its ProgramDataHandoff.
class ProgramDataRange {
public:
  /// Creates a checked range describing `payloadSlice` (element coordinates
  /// over the source payload tensor) for a program tensor. All arithmetic,
  /// rank, boundary, and stride facts are verified here.
  static llvm::Expected<ProgramDataRange>
  create(ProgramTensorId tensorId, llvm::StringRef dtype,
         llvm::ArrayRef<int64_t> globalShape,
         llvm::ArrayRef<int64_t> localShape,
         frontend::ProgramDistributionKind distribution,
         llvm::ArrayRef<int64_t> sliceOffsets,
         llvm::ArrayRef<int64_t> sliceSizes,
         llvm::ArrayRef<int64_t> sliceStrides, SourceDataId sourceId,
         const ProgramDataSource &source, ProgramDataFailure *failure);

  ProgramTensorId getTensorId() const { return tensorId; }
  llvm::StringRef getDType() const { return dtype; }
  llvm::ArrayRef<int64_t> getGlobalShape() const { return globalShape; }
  llvm::ArrayRef<int64_t> getLocalShape() const { return localShape; }
  frontend::ProgramDistributionKind getDistribution() const {
    return distribution;
  }
  llvm::ArrayRef<int64_t> getSliceOffsets() const { return sliceOffsets; }
  llvm::ArrayRef<int64_t> getSliceSizes() const { return sliceSizes; }
  SourceDataId getSourceId() const { return sourceId; }
  /// Total bytes of the local tensor this range describes.
  uint64_t getRegionLength() const { return regionLength; }

  /// Materializes the local tensor bytes into `out` (which must have exactly
  /// getRegionLength() bytes). Contiguous regions are one bounded copy;
  /// strided regions gather rows with checked element addressing.
  llvm::Error materialize(const ProgramDataSource &source,
                          llvm::MutableArrayRef<uint8_t> out) const;

private:
  ProgramDataRange(ProgramTensorId tensorId, std::string dtype,
                   std::vector<int64_t> globalShape,
                   std::vector<int64_t> localShape,
                   frontend::ProgramDistributionKind distribution,
                   std::vector<int64_t> sliceOffsets,
                   std::vector<int64_t> sliceSizes, SourceDataId sourceId,
                   uint64_t regionOffset, uint64_t regionLength,
                   std::vector<int64_t> payloadShape, bool contiguous)
      : tensorId(tensorId), dtype(std::move(dtype)),
        globalShape(std::move(globalShape)),
        localShape(std::move(localShape)), distribution(distribution),
        sliceOffsets(std::move(sliceOffsets)),
        sliceSizes(std::move(sliceSizes)), sourceId(sourceId),
        regionOffset(regionOffset), regionLength(regionLength),
        payloadShape(std::move(payloadShape)), contiguous(contiguous) {}

  ProgramTensorId tensorId;
  std::string dtype;
  std::vector<int64_t> globalShape;
  std::vector<int64_t> localShape;
  frontend::ProgramDistributionKind distribution;
  std::vector<int64_t> sliceOffsets;
  std::vector<int64_t> sliceSizes;
  SourceDataId sourceId;
  /// Byte offset of the region start relative to the source payload start.
  uint64_t regionOffset;
  uint64_t regionLength;
  /// Tensor shape of the payload stored in the referenced source.
  std::vector<int64_t> payloadShape;
  /// True when the region is one contiguous byte span of the payload.
  bool contiguous;
};

/// Explainable payload I/O ledger for one transaction. Counts are per-category
/// facts, not resource-identity counts.
struct ProgramDataIOStatistics {
  uint64_t sourceOpens = 0;
  uint64_t headerReads = 0;
  uint64_t digestPasses = 0;
  uint64_t shardReadbacks = 0;
  uint64_t rangeMaterializations = 0;
  uint64_t materializedFileWrites = 0;
  uint64_t materializedWriteBytes = 0;

  void print(llvm::raw_ostream &stream) const;
};

/// Result of verifying one SPMD helper shard output against an owned
/// pre-SPMD source region.
struct ShardVerification {
  /// True when the shard payload bytes equal the source region bytes:
  /// replication or a contiguous slice can keep referencing the original
  /// source instead of establishing a second owner for identical bytes.
  bool byteIdentical = false;
  /// The established shard content; only taken ownership of when the bytes
  /// differ.
  ProgramDataSource shard;
};

/// SHA-256 digest (lowercase hex) of one element region of a payload source.
/// The region is described by element coordinates over the source payload
/// tensor with unit strides; contiguous regions hash one span, strided
/// regions gather rows through a bounded window.
llvm::Expected<std::string>
computePayloadRegionDigest(const ProgramDataSource &source,
                           llvm::ArrayRef<int64_t> offsets,
                           llvm::ArrayRef<int64_t> sizes);

/// Move-only owner of all-and-only live payload sources and ranges of one
/// compilation transaction. It lives with the CardExecutable until the target
/// package stage has consumed the data (Q56).
class ProgramDataHandoff {
public:
  ProgramDataHandoff() = default;
  ProgramDataHandoff(ProgramDataHandoff &&) = default;
  ProgramDataHandoff &operator=(ProgramDataHandoff &&) = default;
  ProgramDataHandoff(const ProgramDataHandoff &) = delete;
  ProgramDataHandoff &operator=(const ProgramDataHandoff &) = delete;

  /// Establishes one payload source and takes ownership. Duplicate
  /// establishments are caller bugs; locators are diagnostics only.
  llvm::Expected<SourceDataId> establishSource(llvm::StringRef path,
                                               llvm::StringRef locator,
                                               ProgramDataFailure *failure);

  /// Takes ownership of an already established source (for helper shard
  /// output that must be verified before the dedup decision). Accounts the
  /// establishment I/O exactly once.
  SourceDataId addSource(ProgramDataSource source);

  /// Appends a checked range; the referenced source must already exist.
  llvm::Error addRange(ProgramDataRange range);

  const ProgramDataSource &getSource(SourceDataId id) const;
  const ProgramDataRange *findRange(ProgramTensorId tensorId) const;
  llvm::ArrayRef<ProgramDataRange> getRanges() const { return ranges; }
  size_t getSourceCount() const { return sources.size(); }

  /// Materializes one range into `out` and accounts the read. `out` must have
  /// exactly range.getRegionLength() bytes.
  llvm::Error materializeRange(const ProgramDataRange &range,
                               llvm::MutableArrayRef<uint8_t> out) const;

  /// Streams one owned source into a transaction path with bounded windows
  /// and verifies the written file digest against the owned content. This is
  /// the only payload-materialization path for the external SPMD helper
  /// boundary; it accounts file writes and bytes. `failure` classifies the
  /// digest verification result when set.
  llvm::Error materializeSourceToFile(SourceDataId id, llvm::StringRef path,
                                      ProgramDataFailure *failure = nullptr);

  /// Streaming digest of a file on disk (SHA-256, lowercase hex).
  static llvm::Expected<std::string> computeFileDigest(llvm::StringRef path);

  /// Establishes one helper shard output and proves whether its payload
  /// bytes equal the referenced owned source region. Accounts the shard
  /// open and the digest passes in the I/O ledger.
  llvm::Expected<ShardVerification>
  verifyShardAgainstSource(llvm::StringRef shardPath,
                           llvm::StringRef shardLocator,
                           SourceDataId originalSourceId,
                           llvm::ArrayRef<int64_t> offsets,
                           llvm::ArrayRef<int64_t> sizes,
                           ProgramDataFailure *failure = nullptr);

  const ProgramDataIOStatistics &getIOStatistics() const {
    return ioStatistics;
  }
  ProgramDataIOStatistics &getIOStatistics() { return ioStatistics; }

private:
  std::vector<std::unique_ptr<ProgramDataSource>> sources;
  std::vector<ProgramDataRange> ranges;
  /// Mutable so const handoff accessors can still account target-consumer
  /// materialization reads without weakening source/range ownership.
  mutable ProgramDataIOStatistics ioStatistics;
};

} // namespace wafer::compiler

#endif // WAFER_COMPILER_PROGRAMDATA_H
