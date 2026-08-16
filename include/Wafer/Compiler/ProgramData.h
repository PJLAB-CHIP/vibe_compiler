//===- ProgramData.h - Transaction-owned program data ------------*- C++ -*-===//
//
// Transaction-owned parameter/constant payload ownership for one compilation.
// ProgramDataSource owns an immutable transaction-private copy of one payload
// file; ProgramDataRange describes the checked byte region one program tensor
// consumes; ProgramDataHandoff owns the live set for the lifetime of a
// CardExecutable. Tile bindings reference identity and range only; they never
// copy payload bytes.
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

/// Classification of a program-data establishment, materialization, or
/// readback failure. Callers keep recoverable control flow; these kinds are
/// never parsed from diagnostic strings.
enum class ProgramDataFailureKind : uint8_t {
  /// Payload path is missing, not a regular file, or cannot be opened.
  MissingPayload,
  /// NPY magic, version, or header cannot be parsed.
  HeaderInvalid,
  /// Payload uses Fortran order or a dtype encoding not admitted at the
  /// program boundary.
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
  /// A transaction-private file or directory could not be created, written,
  /// read back, or closed.
  MaterializationIO,
};

struct ProgramDataFailure {
  ProgramDataFailureKind kind = ProgramDataFailureKind::MissingPayload;
  std::string locator;
  std::string detail;
};

llvm::StringRef
stringifyProgramDataFailureKind(ProgramDataFailureKind kind);

/// Element width in bytes for one dtype admitted at the program boundary, or
/// std::nullopt for dtypes without a target representation and conversion.
/// This is the admission table for program tensors and ranges. NPY source
/// decoding is a separate concern (frontend `decodeProgramNpyDescr`); a source
/// whose decoded dtype is not admitted here fails establishment.
std::optional<int64_t> getProgramDTypeElementBytes(llvm::StringRef dtype);

/// Where a range's source content comes from. The proof a range must carry
/// depends on this origin.
enum class ProgramDataRangeOrigin {
  /// The range references the original pre-SPMD payload source: the source
  /// tensor shape must equal the program tensor's global shape.
  OriginalSource,
  /// The range references a materialized helper shard: the source tensor
  /// shape must equal the local shard shape and the slice must cover it
  /// exactly from the origin.
  MaterializedShard,
};

/// Transaction-owned, content-immutable payload source. Establishment streams
/// the user file into a transaction-private file with a whole-file SHA-256
/// digest and verifies the NPY header and exact extent against the owned
/// bytes; the user path is never consulted again, so later in-place writes to
/// the original file cannot change this compilation. Implements the frontend
/// payload-source seam so directory verification reads from owned content
/// instead of reopening paths.
class ProgramDataSource final : public frontend::ProgramPayloadSource {
public:
  ProgramDataSource(ProgramDataSource &&) = default;
  ProgramDataSource &operator=(ProgramDataSource &&) = default;
  ProgramDataSource(const ProgramDataSource &) = delete;
  ProgramDataSource &operator=(const ProgramDataSource &) = delete;

  /// Opens, copies, verifies, and digests one payload file into
  /// `ownedFilePath` (a transaction-private path). On failure `failure`
  /// (when non-null) carries the classification; the returned error string
  /// is the human-readable detail. The owned file is removed again when
  /// establishment fails.
  static llvm::Expected<ProgramDataSource>
  establish(llvm::StringRef path, llvm::StringRef ownedFilePath,
            llvm::StringRef locator, ProgramDataFailure *failure = nullptr);

  /// Program-relative locator used for diagnostics, provenance, and resolver
  /// lookup; it is never a data key for range identity.
  llvm::StringRef getLocator() const { return locator; }
  llvm::StringRef getDType() const { return dtype; }
  llvm::ArrayRef<int64_t> getShape() const { return shape; }
  uint64_t getPayloadOffset() const { return payloadOffset; }
  llvm::StringRef getContentDigest() const { return digest; }
  llvm::StringRef getOwnedFilePath() const { return ownedFilePath; }

  uint64_t getPayloadFileSize() const override { return size; }

  std::optional<std::string> getOwnedContentDigest() const override {
    return digest;
  }

  /// Bounded read of a source-relative byte span from the owned file. The
  /// span must lie inside the owned file.
  llvm::Error readPayloadBytes(uint64_t offset,
                               llvm::MutableArrayRef<uint8_t> out)
      const override;

  /// Whole owned file size in bytes.
  uint64_t getSize() const { return size; }

  /// Bounded copy of a source-relative byte span from the owned file.
  llvm::Error readRange(uint64_t offset, llvm::MutableArrayRef<uint8_t> out)
      const;

private:
  ProgramDataSource(std::string ownedFilePath, std::string locator,
                    std::string dtype, std::vector<int64_t> shape,
                    uint64_t payloadOffset, uint64_t size, std::string digest)
      : ownedFilePath(std::move(ownedFilePath)), locator(std::move(locator)),
        dtype(std::move(dtype)), shape(std::move(shape)),
        payloadOffset(payloadOffset), size(size), digest(std::move(digest)) {}

  std::string ownedFilePath;
  std::string locator;
  std::string dtype;
  std::vector<int64_t> shape;
  uint64_t payloadOffset;
  uint64_t size;
  std::string digest;
};

/// Checked byte region one program tensor consumes inside one source. A range
/// never owns the file; it must not outlive its ProgramDataHandoff.
class ProgramDataRange {
public:
  /// Creates a checked range describing `sliceOffsets`/`sliceSizes` (element
  /// coordinates over the source payload tensor) for a program tensor. All
  /// arithmetic, rank, boundary, stride, and shape-origin facts are verified
  /// here: an `OriginalSource` range proves the source shape equals the
  /// global shape, a `MaterializedShard` range proves the source shape equals
  /// the local shape and is covered exactly from the origin.
  static llvm::Expected<ProgramDataRange>
  create(ProgramTensorId tensorId, llvm::StringRef dtype,
         llvm::ArrayRef<int64_t> globalShape,
         llvm::ArrayRef<int64_t> localShape,
         frontend::ProgramDistributionKind distribution,
         ProgramDataRangeOrigin origin, llvm::ArrayRef<int64_t> sliceOffsets,
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
  /// User payload files opened at establishment (canonical payloads and
  /// helper output readbacks).
  uint64_t sourceOpens = 0;
  /// Bounded NPY header parses: establishment plus every verification seam
  /// read of owned content.
  uint64_t headerReads = 0;
  /// Whole-content SHA-256 passes: establishment digests, shard region
  /// digests, and materialization readback digests.
  uint64_t digestPasses = 0;
  /// Helper output payloads read back at establishment (shards that the
  /// helper produced for the transaction).
  uint64_t helperOutputReadbacks = 0;
  /// Target-consumer range materializations (one per program tensor identity,
  /// shared across Tiles).
  uint64_t rangeMaterializations = 0;
  /// Helper-input payload files written from owned content.
  uint64_t materializedFileWrites = 0;
  /// Bytes written by helper-input materialization.
  uint64_t materializedWriteBytes = 0;

  void print(llvm::raw_ostream &stream) const;
};

/// SHA-256 digest (lowercase hex) of one element region of a payload source.
/// The region is described by element coordinates over the source payload
/// tensor with unit strides; contiguous regions hash one span, strided
/// regions gather rows through a bounded window. `failure` classifies every
/// recoverable failure when set.
llvm::Expected<std::string>
computePayloadRegionDigest(const ProgramDataSource &source,
                           llvm::ArrayRef<int64_t> offsets,
                           llvm::ArrayRef<int64_t> sizes,
                           ProgramDataFailure *failure = nullptr);

/// Move-only owner of all-and-only live payload sources and ranges of one
/// compilation transaction. It lives with the CardExecutable until the target
/// package stage has consumed the data (Q56). Owned files are stored under the
/// scratch directory the handoff was constructed with.
class ProgramDataHandoff {
public:
  /// `scratchDirectory` must be a transaction-private directory the handoff
  /// may create `program-data-sources/` inside.
  explicit ProgramDataHandoff(std::string scratchDirectory = {});
  ProgramDataHandoff(ProgramDataHandoff &&) = default;
  ProgramDataHandoff &operator=(ProgramDataHandoff &&) = default;
  ProgramDataHandoff(const ProgramDataHandoff &) = delete;
  ProgramDataHandoff &operator=(const ProgramDataHandoff &) = delete;

  /// Establishes one payload source and takes ownership. Duplicate
  /// establishments are caller bugs; locators are diagnostics only.
  llvm::Expected<SourceDataId> establishSource(llvm::StringRef path,
                                               llvm::StringRef locator,
                                               ProgramDataFailure *failure);

  /// Establishes one helper output payload into the candidate cache without
  /// taking source ownership. The cached candidate is transaction-owned and
  /// looked up by locator; `adoptCandidate` promotes a real partition into
  /// the source set without re-reading. Byte-identical helper outputs stay
  /// candidates only and are never part of the range identity space.
  llvm::Expected<const ProgramDataSource *>
  establishHelperOutput(llvm::StringRef path, llvm::StringRef locator,
                        ProgramDataFailure *failure);

  /// Promotes an established helper output candidate into the source set and
  /// returns its source identity. The candidate must come from this handoff's
  /// `establishHelperOutput`; its establishment I/O is not recounted.
  SourceDataId adoptCandidate(const ProgramDataSource *candidate);

  /// Returns the source or candidate carrying `locator`, or nullptr. Used by
  /// the verification resolver seam after establishment.
  const ProgramDataSource *findByLocator(llvm::StringRef locator) const;

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
  /// boundary; it accounts file writes and bytes. `failure` classifies every
  /// recoverable failure when set.
  llvm::Error materializeSourceToFile(SourceDataId id, llvm::StringRef path,
                                      ProgramDataFailure *failure = nullptr);

  /// Streaming digest of a file on disk (SHA-256, lowercase hex) using
  /// bounded windows; no whole-file mapping.
  static llvm::Expected<std::string> computeFileDigest(llvm::StringRef path);

  /// Proves whether one established helper shard's payload bytes equal the
  /// referenced owned source region. Both region digests are computed over
  /// owned content with bounded windows. Accounts the digest passes in the
  /// I/O ledger.
  llvm::Expected<bool>
  verifyShardAgainstSource(const ProgramDataSource &shard,
                           SourceDataId originalSourceId,
                           llvm::ArrayRef<int64_t> offsets,
                           llvm::ArrayRef<int64_t> sizes,
                           ProgramDataFailure *failure = nullptr);

  const ProgramDataIOStatistics &getIOStatistics() const {
    return ioStatistics;
  }
  ProgramDataIOStatistics &getIOStatistics() { return ioStatistics; }

private:
  std::string scratchDirectory;
  std::vector<std::unique_ptr<ProgramDataSource>> sources;
  std::vector<std::unique_ptr<ProgramDataSource>> candidates;
  std::vector<ProgramDataRange> ranges;
  /// Mutable so const handoff accessors can still account target-consumer
  /// materialization reads without weakening source/range ownership.
  mutable ProgramDataIOStatistics ioStatistics;
};

} // namespace wafer::compiler

#endif // WAFER_COMPILER_PROGRAMDATA_H
