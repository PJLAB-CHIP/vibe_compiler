//===- TargetCall.h - Typed target call ABI and payloads -------*- C++ -*-===//

#ifndef WAFER_TARGET_TARGETCALL_H
#define WAFER_TARGET_TARGETCALL_H

#include "Wafer/Target/Numeric/NumericSemantics.h"
#include "Wafer/Target/Core/TargetFormat.h"
#include "Wafer/Target/Core/TargetIdentity.h"
#include "Wafer/Target/Core/TargetOperation.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace wafer {

namespace target {

enum class TargetDMADirection : uint8_t { Read, Write };

struct TargetStridedDMACommand {
  TargetDMADirection direction;
  uint64_t source;
  uint64_t destination;
  uint32_t byteCount;
  uint32_t innerBytes;
  std::array<uint32_t, 3> strides;
  std::array<uint32_t, 3> iterations;
  LogicalFormat format;
};

struct TargetGatherScatterCommand {
  uint64_t source;
  uint64_t destination;
  uint32_t byteCount;
  uint32_t innerBytes;
  std::array<uint32_t, 3> sourceStrides;
  std::array<uint32_t, 3> sourceIterations;
  std::array<uint32_t, 3> destinationStrides;
  std::array<uint32_t, 3> destinationIterations;
};

struct TargetMemsetCommand {
  uint64_t destination;
  uint32_t value;
  uint32_t elementCount;
  LogicalFormat format;
};

struct TargetBit2FPCommand {
  uint64_t source;
  uint64_t destination;
  uint32_t elementCount;
  LogicalFormat format;
};

struct TargetMaskMoveCommand {
  uint64_t source;
  uint32_t mask;
  uint64_t destination;
  uint32_t elementCount;
  LogicalFormat format;
};

struct TargetGemmCommand {
  uint64_t lhs;
  uint64_t rhs;
  uint64_t destination;
  uint32_t m;
  uint32_t k;
  uint32_t n;
  uint32_t batchCount;
  LogicalFormat format;
  TargetGemmOrientation lhsOrientation = TargetGemmOrientation::Normal;
  TargetGemmOrientation rhsOrientation = TargetGemmOrientation::Normal;
};

struct TargetElementwiseCommand {
  NumericElementwiseOperation operation;
  uint64_t lhs;
  std::optional<uint64_t> rhs;
  uint64_t destination;
  uint32_t elementCount;
  LogicalFormat format;
};

struct TargetReduceCommand {
  NumericReduceOperation operation;
  uint64_t source;
  uint64_t destination;
  uint32_t dimension;
  std::array<uint32_t, 4> nhwc;
  LogicalFormat format;
};

struct TargetConvertCommand {
  TargetConvertOperation operation;
  uint64_t source;
  uint64_t destination;
  uint32_t elementCount;
  std::optional<uint32_t> zeroPoint;
  std::optional<uint32_t> roundingMode;
};

struct TargetConvCommand {
  TargetConvolutionOperation operation;
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

struct TargetPoolCommand {
  TargetPoolingOperation operation;
  uint64_t input;
  uint64_t valueDestination;
  std::optional<uint64_t> indexDestination;
  std::array<uint32_t, 4> sourceShape;
  std::array<uint32_t, 4> destinationShape;
  std::array<uint32_t, 4> pads;
  std::array<uint32_t, 4> kernelStrides;
  LogicalFormat format;
};

struct TargetUnpoolCommand {
  TargetUnpoolingOperation operation;
  uint64_t input;
  uint64_t destination;
  std::optional<uint32_t> indexAddress;
  std::array<uint32_t, 4> sourceShape;
  std::array<uint32_t, 4> destinationShape;
  std::array<uint32_t, 4> kernelStrides;
  LogicalFormat format;
};

enum class TargetTDMATransformKind : uint8_t { Pad, ImageToColumn };

struct TargetTDMATransformCommand {
  TargetTDMATransformKind kind;
  uint64_t source;
  uint64_t destination;
  std::array<uint32_t, 4> sourceShape;
  std::array<uint32_t, 4> destinationShape;
  std::array<uint32_t, 4> pads;
  std::optional<std::array<uint32_t, 4>> kernelStrides;
  LogicalFormat format;
};

struct TargetPeripheralArgExtremaCommand {
  TargetPeripheralOperation operation;
  uint64_t source;
  uint64_t valueDestination;
  uint64_t indexDestination;
  uint32_t elementCount;
  LogicalFormat format;
};

struct TargetPeripheralBilinearCommand {
  uint64_t source;
  uint64_t destination;
  uint32_t elementCount;
  LogicalFormat format;
  std::array<uint32_t, 4> sourceShape;
  std::array<uint32_t, 4> destinationShape;
};

struct TargetPeripheralLUTCommand {
  TargetPeripheralOperation operation;
  uint64_t source;
  uint64_t table;
  uint64_t destination;
  uint32_t elementCount;
  LogicalFormat format;
  uint32_t tableElementCount;
};

struct TargetPeripheralRandomCommand {
  std::array<uint64_t, 2> sources;
  std::array<uint64_t, 3> destinations;
  uint32_t elementCount;
  LogicalFormat format;
};

struct TargetPeripheralElementMaskCommand {
  uint64_t source;
  uint64_t destination;
  uint32_t elementCount;
  LogicalFormat format;
  uint32_t scale;
  uint32_t probability;
  uint32_t roundingMode;
};

struct TargetNCCJoinCommand {
  uint32_t participantMask;
};
struct TargetDirectDTEBeginCommand {
  uint64_t statusAddress;
  uint32_t participantCount;
};
struct TargetDirectDTESendCommand {
  uint64_t source;
  uint64_t remoteDestination;
  uint32_t byteCount;
  uint32_t localTile;
  uint32_t remoteTile;
  uint32_t remoteFSM;
  bool highPerformance;
};
struct TargetDirectDTESendIssueCommand {
  uint64_t event;
};
struct TargetDirectDTEReceiveCommand {
  uint64_t destination;
  uint32_t byteCount;
  uint32_t localTile;
  uint32_t remoteTile;
  uint32_t localFSM;
};
struct TargetDirectDTEWaitCommand {
  uint64_t event;
};
struct TargetDirectDTEFinishCommand {};

using TargetCommandPayload = std::variant<
    TargetStridedDMACommand, TargetGatherScatterCommand, TargetMemsetCommand,
    TargetBit2FPCommand, TargetMaskMoveCommand, TargetGemmCommand,
    TargetElementwiseCommand, TargetReduceCommand, TargetConvertCommand,
    TargetConvCommand, TargetPoolCommand, TargetUnpoolCommand,
    TargetTDMATransformCommand, TargetPeripheralArgExtremaCommand,
    TargetPeripheralBilinearCommand, TargetPeripheralLUTCommand,
    TargetPeripheralRandomCommand, TargetPeripheralElementMaskCommand,
    TargetNCCJoinCommand, TargetDirectDTEBeginCommand,
    TargetDirectDTESendCommand, TargetDirectDTESendIssueCommand,
    TargetDirectDTEReceiveCommand, TargetDirectDTEWaitCommand,
    TargetDirectDTEFinishCommand>;

} // namespace target

/// Fixed target calls that are not selected by an instruction-kind enum.
enum class TargetCallBuiltin : uint8_t {
  RDMA,
  WDMA,
  GatherScatter,
  Memset,
  Bit2FP,
  MaskMove,
  Gemm,
  GemmOriented,
  TDMAPad,
  TDMAImg2Col,
  NCCJoin,
  DirectDTEBegin,
  DirectDTEBeginAfterPrepare,
  DirectDTESendPrepare,
  DirectDTESendIssue,
  DirectDTERecvPrepare,
  DirectDTEWait,
  DirectDTEFinish,
};

enum class TargetCallScalarType : uint8_t { I32, I64 };
enum class TargetCallResultType : uint8_t { Void, I64 };

/// The real NCC engine reached by one registered target-call implementation.
/// Absence means that the call does not submit an NCC engine command. This
/// closed mapping is shared by profile instrumentation and static site-map
/// emission; consumers must not recover it from symbol spellings.
enum class TargetCallTSMEngine : uint8_t {
  CT,
  NE,
  RDMA,
  WDMA,
  TDMA,
  DirectDTE
};

/// Typed issue-domain metadata owned by the target-call registry. NCC engine
/// calls carry their worker in one exact registered argument position. Direct
/// DTE carries no NCC worker because its completion is represented by its
/// opaque event and wait sequence.
struct TargetCallIssueDomain {
  TargetCallTSMEngine engine;
  std::optional<size_t> nccWorkerArgument;
  TargetNCCCompletionBehavior completionBehavior =
      TargetNCCCompletionBehavior::None;
};

/// Semantic classification encoded by typed compiler enums, never
/// reconstructed from a symbol spelling by a consumer.
using TargetCallSemantic =
    std::variant<TargetCallBuiltin, NumericElementwiseOperation,
                 NumericReduceOperation, TargetConvertOperation,
                 TargetConvolutionOperation, TargetPoolingOperation,
                 TargetUnpoolingOperation, TargetPeripheralOperation>;

/// Exact public target-call ABI. The strings and signature widths are a
/// compiler registry, not a serialized module or a packet description.
struct TargetCallDescriptor {
  std::string symbol;
  TargetCallResultType result;
  std::vector<TargetCallScalarType> arguments;
  TargetCallSemantic semantic;
  std::optional<TargetCallIssueDomain> issueDomain;
};

struct TargetCallDecodeConfig {
  int64_t tileCount;
};

/// Returns the one closed target-call ABI surface consumed by compiler,
/// runtime, models, and profiling.
llvm::ArrayRef<TargetCallDescriptor> getTargetCallDescriptors();

const TargetCallDescriptor *findTargetCallDescriptor(llvm::StringRef symbol);
const TargetCallDescriptor *
findTargetCallDescriptor(const TargetCallSemantic &semantic);

std::optional<TargetCallTSMEngine>
getTargetCallTSMEngine(const TargetCallSemantic &semantic);
std::optional<TargetCallTSMEngine>
getTargetCallTSMEngine(const TargetCallDescriptor &descriptor);
llvm::StringRef stringifyTargetCallTSMEngine(TargetCallTSMEngine engine);

const TargetCallDescriptor &getTargetCallDescriptor(TargetCallBuiltin call);
const TargetCallDescriptor &
getTargetCallDescriptor(NumericElementwiseOperation operation);
const TargetCallDescriptor &
getTargetCallDescriptor(NumericReduceOperation operation);

const TargetCallDescriptor &
getTargetCallDescriptor(TargetConvertOperation operation);
const TargetCallDescriptor &
getTargetCallDescriptor(TargetConvolutionOperation operation);
const TargetCallDescriptor &
getTargetCallDescriptor(TargetPoolingOperation operation);
const TargetCallDescriptor &
getTargetCallDescriptor(TargetUnpoolingOperation operation);
const TargetCallDescriptor &
getTargetCallDescriptor(TargetPeripheralOperation operation);

/// Decodes one exact ABI argument vector into the descriptor's typed payload.
/// This is the only field-position factory shared by the JIT frontend and
/// downstream functional models.
llvm::Expected<target::TargetCommandPayload>
decodeTargetCallPayload(const TargetCallDescriptor &descriptor,
                        const TargetCallDecodeConfig &config,
                        llvm::ArrayRef<uint64_t> arguments);

/// Decodes the exact NCC worker carried by a registered issue call. Absence
/// means that the call does not issue an NCC command.
llvm::Expected<std::optional<TargetNCCWorker>>
decodeTargetCallNCCWorker(const TargetCallDescriptor &descriptor,
                          llvm::ArrayRef<uint64_t> arguments);

} // namespace wafer

#endif // WAFER_TARGET_TARGETCALL_H
