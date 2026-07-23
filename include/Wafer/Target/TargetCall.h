//===- TargetCall.h - Typed target call ABI and payloads -------*- C++ -*-===//

#ifndef WAFER_TARGET_TARGETCALL_H
#define WAFER_TARGET_TARGETCALL_H

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Target/TargetFormat.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace wafer {

namespace compiler {

enum class TargetDMADirection : uint8_t { Read, Write };

struct TargetStridedDMATransaction {
  TargetDMADirection direction;
  uint64_t source;
  uint64_t destination;
  uint32_t byteCount;
  uint32_t innerBytes;
  std::array<uint32_t, 3> strides;
  std::array<uint32_t, 3> iterations;
  LogicalFormat format;
};

struct TargetGatherScatterTransaction {
  uint64_t source;
  uint64_t destination;
  uint32_t byteCount;
  uint32_t innerBytes;
  std::array<uint32_t, 3> sourceStrides;
  std::array<uint32_t, 3> sourceIterations;
  std::array<uint32_t, 3> destinationStrides;
  std::array<uint32_t, 3> destinationIterations;
};

struct TargetMemsetTransaction {
  uint64_t destination;
  uint32_t value;
  uint32_t elementCount;
  LogicalFormat format;
};

struct TargetBit2FPTransaction {
  uint64_t source;
  uint64_t destination;
  uint32_t elementCount;
  LogicalFormat format;
};

struct TargetMaskMoveTransaction {
  uint64_t source;
  uint32_t mask;
  uint64_t destination;
  uint32_t elementCount;
  LogicalFormat format;
};

struct TargetGemmTransaction {
  uint64_t lhs;
  uint64_t rhs;
  uint64_t destination;
  uint32_t m;
  uint32_t k;
  uint32_t n;
  uint32_t batchCount;
  LogicalFormat format;
  GemmOrientation lhsOrientation = GemmOrientation::Normal;
  GemmOrientation rhsOrientation = GemmOrientation::Normal;
};

struct TargetElementwiseTransaction {
  InstrElementwiseKind kind;
  uint64_t lhs;
  std::optional<uint64_t> rhs;
  uint64_t destination;
  uint32_t elementCount;
  LogicalFormat format;
};

struct TargetReduceTransaction {
  InstrReduceKind kind;
  uint64_t source;
  uint64_t destination;
  uint32_t dimension;
  std::array<uint32_t, 4> nhwc;
  LogicalFormat format;
};

struct TargetConvertTransaction {
  InstrConvertKind kind;
  uint64_t source;
  uint64_t destination;
  uint32_t elementCount;
  std::optional<uint32_t> zeroPoint;
  std::optional<uint32_t> roundingMode;
};

struct TargetConvTransaction {
  InstrConvKind kind;
  uint64_t input;
  uint64_t weight;
  uint64_t destination;
  std::array<uint32_t, 4> inputShape;
  std::array<uint32_t, 4> weightShape;
  std::array<uint32_t, 4> outputShape;
  std::array<uint32_t, 4> pads;
  std::array<uint32_t, 4> unpads;
  std::array<uint32_t, 4> kernelStrides;
  std::array<uint32_t, 2> dilations;
  LogicalFormat format;
};

struct TargetPoolTransaction {
  InstrPoolKind kind;
  uint64_t input;
  uint64_t valueDestination;
  std::optional<uint64_t> indexDestination;
  std::array<uint32_t, 4> sourceShape;
  std::array<uint32_t, 4> destinationShape;
  std::array<uint32_t, 4> pads;
  std::array<uint32_t, 4> kernelStrides;
  LogicalFormat format;
};

struct TargetUnpoolTransaction {
  InstrUnpoolKind kind;
  uint64_t input;
  uint64_t destination;
  std::optional<uint32_t> indexAddress;
  std::array<uint32_t, 4> sourceShape;
  std::array<uint32_t, 4> destinationShape;
  std::array<uint32_t, 4> kernelStrides;
  LogicalFormat format;
};

enum class TargetTDMATransformKind : uint8_t { Pad, ImageToColumn };

struct TargetTDMATransformTransaction {
  TargetTDMATransformKind kind;
  uint64_t source;
  uint64_t destination;
  std::array<uint32_t, 4> sourceShape;
  std::array<uint32_t, 4> destinationShape;
  std::array<uint32_t, 4> pads;
  std::optional<std::array<uint32_t, 4>> kernelStrides;
  LogicalFormat format;
};

struct TargetPeripheralArgExtremaTransaction {
  InstrPeripheralKind kind;
  uint64_t source;
  uint64_t valueDestination;
  uint64_t indexDestination;
  uint32_t elementCount;
  LogicalFormat format;
};

struct TargetPeripheralBilinearTransaction {
  uint64_t source;
  uint64_t destination;
  uint32_t elementCount;
  LogicalFormat format;
  std::array<uint32_t, 4> sourceShape;
  std::array<uint32_t, 4> destinationShape;
};

struct TargetPeripheralLUTTransaction {
  InstrPeripheralKind kind;
  uint64_t source;
  uint64_t table;
  uint64_t destination;
  uint32_t elementCount;
  LogicalFormat format;
  uint32_t tableElementCount;
};

struct TargetPeripheralRandomTransaction {
  std::array<uint64_t, 2> sources;
  std::array<uint64_t, 3> destinations;
  uint32_t elementCount;
  LogicalFormat format;
};

struct TargetPeripheralElementMaskTransaction {
  uint64_t source;
  uint64_t destination;
  uint32_t elementCount;
  LogicalFormat format;
  uint32_t scale;
  uint32_t probability;
  uint32_t roundingMode;
};

struct TargetLocalFenceTransaction {};
struct TargetDirectDTEBeginTransaction {
  uint64_t statusAddress;
  uint32_t rankCount;
};
struct TargetDirectDTESendTransaction {
  uint64_t source;
  uint64_t remoteDestination;
  uint32_t byteCount;
  uint32_t localTile;
  uint32_t remoteTile;
  uint32_t remoteFSM;
  bool highPerformance;
};
struct TargetDirectDTEReceiveTransaction {
  uint64_t destination;
  uint32_t byteCount;
  uint32_t localTile;
  uint32_t remoteTile;
  uint32_t localFSM;
};
struct TargetDirectDTEWaitTransaction {
  uint64_t event;
};
struct TargetDirectDTEFinishTransaction {};

using TargetTransactionPayload = std::variant<
    TargetStridedDMATransaction, TargetGatherScatterTransaction,
    TargetMemsetTransaction, TargetBit2FPTransaction, TargetMaskMoveTransaction,
    TargetGemmTransaction, TargetElementwiseTransaction,
    TargetReduceTransaction, TargetConvertTransaction, TargetConvTransaction,
    TargetPoolTransaction, TargetUnpoolTransaction,
    TargetTDMATransformTransaction, TargetPeripheralArgExtremaTransaction,
    TargetPeripheralBilinearTransaction, TargetPeripheralLUTTransaction,
    TargetPeripheralRandomTransaction, TargetPeripheralElementMaskTransaction,
    TargetLocalFenceTransaction, TargetDirectDTEBeginTransaction,
    TargetDirectDTESendTransaction, TargetDirectDTEReceiveTransaction,
    TargetDirectDTEWaitTransaction, TargetDirectDTEFinishTransaction>;

} // namespace compiler

/// Fixed target calls that are not selected by an instruction-kind enum.
enum class TargetCallBuiltin : uint8_t {
  RDMA,
  WDMA,
  GatherScatter,
  Memset,
  Bit2FP,
  MaskMove,
  Gemm,
  GemmOrientedV2,
  TDMAPad,
  TDMAImg2Col,
  LocalFence,
  DirectDTEBegin,
  DirectDTEBeginAfterPrepare,
  DirectDTESendPrepare,
  DirectDTERecvPrepare,
  DirectDTEWait,
  DirectDTEFinish,
};

enum class TargetCallScalarType : uint8_t { I32, I64 };
enum class TargetCallResultType : uint8_t { Void, I64 };

/// A semantic identity owned by typed compiler enums, never reconstructed
/// from a symbol spelling by a consumer.
using TargetCallSemantic =
    std::variant<TargetCallBuiltin, InstrElementwiseKind, InstrReduceKind,
                 InstrConvertKind, InstrConvKind, InstrPoolKind,
                 InstrUnpoolKind, InstrPeripheralKind>;

/// Exact public target-call ABI. The strings and signature widths are a
/// compiler registry, not a serialized artifact or a packet description.
struct TargetCallDescriptor {
  std::string symbol;
  TargetCallResultType result;
  std::vector<TargetCallScalarType> arguments;
  TargetCallSemantic semantic;
};

struct TargetCallDecodeContext {
  TargetProfileId targetProfile;
  int64_t rankCount;
};

/// Returns the closed 111-call surface emitted by target LLVM lowering.
llvm::ArrayRef<TargetCallDescriptor> getTargetCallDescriptors();

const TargetCallDescriptor *findTargetCallDescriptor(llvm::StringRef symbol);
const TargetCallDescriptor *
findTargetCallDescriptor(const TargetCallSemantic &semantic);

const TargetCallDescriptor &getTargetCallDescriptor(TargetCallBuiltin call);
const TargetCallDescriptor &getTargetCallDescriptor(InstrElementwiseKind kind);
const TargetCallDescriptor &getTargetCallDescriptor(InstrReduceKind kind);
const TargetCallDescriptor &getTargetCallDescriptor(InstrConvertKind kind);
const TargetCallDescriptor &getTargetCallDescriptor(InstrConvKind kind);
const TargetCallDescriptor &getTargetCallDescriptor(InstrPoolKind kind);
const TargetCallDescriptor &getTargetCallDescriptor(InstrUnpoolKind kind);
const TargetCallDescriptor &getTargetCallDescriptor(InstrPeripheralKind kind);

/// Decodes one exact ABI argument vector into the descriptor's typed payload.
/// This is the only field-position factory shared by the JIT frontend and
/// downstream functional models.
llvm::Expected<compiler::TargetTransactionPayload>
decodeTargetCallPayload(const TargetCallDescriptor &descriptor,
                        const TargetCallDecodeContext &context,
                        llvm::ArrayRef<uint64_t> arguments);

} // namespace wafer

#endif // WAFER_TARGET_TARGETCALL_H
