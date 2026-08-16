//===- Program.h - Wafer frontend program verifier -----------*- C++ -*-===//

#ifndef WAFER_FRONTEND_PROGRAM_H
#define WAFER_FRONTEND_PROGRAM_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <vector>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace wafer::frontend {

enum class ProgramDistributionKind { Replicated, Partitioned };

struct ProgramPartitionSlice {
  int64_t partitionId = -1;
  int64_t replicaId = -1;
  std::vector<int64_t> offsets;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  std::string payloadPath;
};

struct ProgramBoundaryBinding {
  /// Function argument/result index used by the compiler ABI.
  int64_t index = -1;
  /// User-visible program-boundary position. For outputs this currently equals
  /// the function result index.
  int64_t programIndex = -1;
  ProgramDistributionKind distribution = ProgramDistributionKind::Replicated;
  std::vector<int64_t> globalShape;
  std::vector<int64_t> localShape;
  std::string dtype;
  std::vector<ProgramPartitionSlice> partitionSlices;
};

struct ProgramParameterBinding {
  int64_t argumentIndex = -1;
  std::string name;
  ProgramDistributionKind distribution = ProgramDistributionKind::Replicated;
  std::vector<int64_t> globalShape;
  std::vector<int64_t> localShape;
  std::string dtype;
  std::vector<ProgramPartitionSlice> partitionSlices;
};

struct ProgramConstantBinding {
  int64_t argumentIndex = -1;
  int64_t position = -1;
  std::vector<int64_t> shape;
  std::string dtype;
  std::string payloadPath;
};

struct FrontendProgramVerificationResult {
  unsigned programParameterCount = 0;
  unsigned programUserInputCount = 0;
  unsigned programConstantCount = 0;
  unsigned programParameterShardCount = 0;
  int64_t numPartitions = 0;
  std::vector<ProgramBoundaryBinding> distributedInputs;
  std::vector<ProgramBoundaryBinding> distributedOutputs;
  std::vector<ProgramParameterBinding> parameters;
  std::vector<ProgramConstantBinding> constants;
};

/// One input location entry of a program metadata file. Names and paths only
/// locate payloads inside the current container and form diagnostics; they
/// are never backend semantics.
struct ProgramInputLocator {
  std::string type;
  int64_t position = -1;
  std::string name;
};

/// Reads input locations from `functions/forward.meta`. The compiler uses
/// this to discover which program members are parameter/constant payloads
/// before establishing transaction ownership and taking the source snapshot.
llvm::Expected<std::vector<ProgramInputLocator>>
readProgramInputLocators(llvm::StringRef metaPath);

/// Owner-backed row-major tensor payload decoded by the same NPY parser used
/// by program-directory verification. Multi-byte elements use canonical
/// little-endian storage, independent of the host byte order.
struct NpyTensorPayload {
  std::string dtype;
  std::vector<int64_t> shape;
  std::vector<uint8_t> bytes;
};

llvm::Expected<NpyTensorPayload> loadNpyTensorPayload(llvm::StringRef path);

/// Transaction-owned stable payload content. Program-directory verification
/// reads header and extent facts through this interface instead of reopening
/// a payload path; the owner proves the bytes stay unchanged for the whole
/// compiler transaction.
class ProgramPayloadSource {
public:
  virtual ~ProgramPayloadSource() = default;

  /// Exact owned file size in bytes.
  virtual uint64_t getPayloadFileSize() const = 0;

  /// Bounded in-bounds read of [offset, offset + out.size()). Returning an
  /// Error describes an invalid or out-of-range read, not payload semantics.
  virtual llvm::Error readPayloadBytes(uint64_t offset,
                                       llvm::MutableArrayRef<uint8_t> out)
      const = 0;
};

/// Resolves program-relative payload locators ("data/<name>",
/// "constants/<position>") to transaction-owned sources. A null result means
/// the payload is unavailable and verification fails closed; the resolver
/// itself reports why.
class ProgramPayloadResolver {
public:
  virtual ~ProgramPayloadResolver() = default;
  virtual const ProgramPayloadSource *resolve(llvm::StringRef locator)
      const = 0;
};

/// Raw NPY header facts read from an owned payload source. Policy decisions
/// (order, dtype support, extent exactness) belong to the consumer.
struct NpyPayloadHeader {
  uint64_t fileSize = 0;
  uint64_t dataOffset = 0;
  std::string descr;
  bool fortranOrder = false;
  std::vector<int64_t> shape;
};

/// Parses the NPY header of an owned payload source with bounded reads. The
/// header must be complete and parseable; no shape/dtype/extent policy is
/// applied here.
llvm::Expected<NpyPayloadHeader>
parseNpyPayloadHeader(const ProgramPayloadSource &source,
                      llvm::StringRef displayName);

/// Decodes an NPY descr into the canonical program dtype and its element
/// width. Rejects unsupported encodings and non-canonical byte orders.
std::optional<std::pair<llvm::StringRef, uint64_t>>
decodeProgramNpyDescr(llvm::StringRef descr);

mlir::LogicalResult
verifyFrontendProgram(mlir::ModuleOp module, llvm::raw_ostream &diagnostics,
                      FrontendProgramVerificationResult *result = nullptr);

/// When `resolver` is non-null, parameter and captured-constant payloads are
/// verified against the resolver's owned content instead of program-directory
/// paths. A locator without a resolvable source fails closed.
mlir::LogicalResult
verifyProgramDirectory(mlir::ModuleOp module, llvm::StringRef programDir,
                       llvm::raw_ostream &diagnostics,
                       FrontendProgramVerificationResult *result = nullptr,
                       const ProgramPayloadResolver *resolver = nullptr);

} // namespace wafer::frontend

#endif // WAFER_FRONTEND_PROGRAM_H
