//===- ReferenceExecutorInternal.h - Invocation-local reference internals
//--===//

#ifndef WAFER_COMPILER_REFERENCEEXECUTORINTERNAL_H
#define WAFER_COMPILER_REFERENCEEXECUTORINTERNAL_H

#include "Wafer/Compiler/ReferenceExecutor.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace wafer::compiler {

struct ReferenceExecutionResultBuilder {
  static ReferenceExecutionResult
  make(int64_t logicalRank, std::vector<ReferenceOutputBinding> outputs) {
    return ReferenceExecutionResult(logicalRank, std::move(outputs));
  }
};

struct ReferenceMultiRankExecutionResultBuilder {
  static ReferenceMultiRankExecutionResult
  make(std::vector<ReferenceExecutionResult> rankResults,
       std::vector<ReferenceGlobalOutputBinding> globalOutputs) {
    return ReferenceMultiRankExecutionResult(std::move(rankResults),
                                             std::move(globalOutputs));
  }
};

struct ReferenceProgram::Impl {
  using ValueId = uint32_t;

  enum class CommandKind {
    Alloc,
    Dealloc,
    Cast,
    Constant,
    TileRegion,
    If,
    For,
    Call,
    RDMA,
    WDMA,
    GatherScatter,
    Convert,
    Gemm,
    Elementwise,
    Fill,
    LocalFence,
    DTESend,
    DTERecv,
    DTEWait,
    Branch,
    CondBranch,
    Return,
  };

  enum class NumericFormat {
    Int8,
    Int16,
    Int32,
    BFloat16,
    Float16,
    Float32,
    TF32
  };

  struct Scalar {
    std::optional<llvm::APInt> integer;
    std::optional<llvm::APFloat> floating;
    bool integerIsUnsigned = false;
  };

  struct BlockProgram;

  struct Command {
    CommandKind kind = CommandKind::LocalFence;
    ValueId result = 0;
    ValueId source = 0;
    ValueId dest = 0;
    ValueId lhs = 0;
    ValueId rhs = 0;
    ValueId scalar = 0;
    std::vector<ValueId> inputs;
    std::vector<ValueId> results;
    std::vector<ValueId> blockArguments;
    std::shared_ptr<BlockProgram> body;
    std::shared_ptr<BlockProgram> elseBody;
    ValueId condition = 0;
    ValueId lowerBound = 0;
    ValueId upperBound = 0;
    ValueId step = 0;
    ValueId inductionArgument = 0;
    std::vector<ValueId> iterInputs;
    std::vector<ValueId> iterArguments;
    std::vector<ValueId> trueInputs;
    std::vector<ValueId> falseInputs;
    uint32_t successor = 0;
    uint32_t trueSuccessor = 0;
    uint32_t falseSuccessor = 0;
    uint32_t callee = 0;
    mlir::MemRefType type;
    int64_t physicalBytes = 0;
    int64_t viewDelta = 0;
    int64_t acceptedOffset = 0;
    bool spmAllocation = false;
    Scalar scalarValue;
    std::vector<int64_t> sourceStrides;
    std::vector<int64_t> sourceIterations;
    std::vector<int64_t> destStrides;
    std::vector<int64_t> destIterations;
    std::optional<int64_t> sourceOffset;
    std::optional<int64_t> destOffset;
    uint64_t byteCount = 0;
    uint64_t innerBytes = 0;
    NumericFormat sourceFormat = NumericFormat::Float32;
    NumericFormat destFormat = NumericFormat::Float32;
    llvm::APFloat::roundingMode roundingMode =
        llvm::APFloat::rmNearestTiesToEven;
    bool stochasticRounding = false;
    int64_t m = 0;
    int64_t n = 0;
    int64_t k = 0;
    wafer::InstrElementwiseKind elementwiseKind =
        wafer::InstrElementwiseKind::Abs;
    int64_t peer = -1;
    int64_t messageCommunication = -1;
    wafer::DTEProtocolPhase messagePhase =
        wafer::DTEProtocolPhase::CollectivePermute;
    int64_t messageRound = -1;
    int64_t messagePayloadSlice = -1;
    int64_t remoteReceiverOffset = -1;
  };

  struct BlockProgram {
    std::vector<ValueId> arguments;
    std::vector<Command> commands;
  };

  struct ControlFlowProgram {
    std::vector<BlockProgram> blocks;
  };

  struct FunctionProgram {
    std::vector<ValueId> arguments;
    std::vector<mlir::MemRefType> argumentTypes;
    ControlFlowProgram body;
  };

  std::shared_ptr<mlir::MLIRContext> contextOwner;
  int64_t logicalRank = 0;
  std::vector<RankProgramBinding> programBindings;
  std::vector<FunctionProgram> functions;
  uint32_t entryFunction = 0;
  size_t projectedOperationCount = 0;
  bool usesStochasticRounding = false;
  TransportContract transportContract = TransportContract::None;
};

namespace reference_detail {

using NumericFormat = ReferenceProgram::Impl::NumericFormat;

struct Storage {
  std::vector<uint8_t> bytes;
};

struct BufferView {
  std::shared_ptr<Storage> storage;
  int64_t base = 0;
  int64_t viewOffset = 0;
  int64_t physicalBytes = 0;
  mlir::MemRefType type;
};

struct NumericValue {
  std::optional<llvm::APInt> integer;
  std::optional<llvm::APFloat> floating;
};

inline llvm::Error invalid(llvm::StringRef message) {
  return llvm::createStringError(llvm::errc::invalid_argument, "%s",
                                 message.str().c_str());
}

inline llvm::Error unsupported(llvm::StringRef message) {
  return invalid(("reference capability preflight failed: " + message).str());
}

std::optional<int64_t> getCompactByteCount(llvm::StringRef dtype,
                                           llvm::ArrayRef<int64_t> shape);
std::string elementDType(mlir::Type type);
std::optional<NumericFormat> getNumericFormat(mlir::Type type);
bool matchesNumericFormat(mlir::Type type, NumericFormat format);

llvm::Expected<int64_t> getAbsoluteOffset(const BufferView &buffer,
                                          llvm::ArrayRef<int64_t> indices,
                                          int64_t byteWidth);
llvm::Expected<NumericValue> readNumeric(const BufferView &buffer,
                                         llvm::ArrayRef<int64_t> indices,
                                         NumericFormat format);
llvm::Expected<llvm::APInt>
convertNumeric(const NumericValue &source, NumericFormat destFormat,
               llvm::APFloat::roundingMode roundingMode);
llvm::Expected<llvm::APInt> convertNumericStochastic(const NumericValue &source,
                                                     NumericFormat destFormat,
                                                     uint64_t randomBits);
llvm::Error writeNumericBits(const BufferView &buffer,
                             llvm::ArrayRef<int64_t> indices,
                             NumericFormat format, const llvm::APInt &bits);
llvm::Expected<float> readF32(const BufferView &buffer,
                              llvm::ArrayRef<int64_t> indices);
llvm::Error writeF32(const BufferView &buffer, llvm::ArrayRef<int64_t> indices,
                     float value);

llvm::Error projectReferenceProgram(ReferenceProgram::Impl &program,
                                    mlir::ModuleOp module,
                                    llvm::StringRef entrySymbol);
llvm::Expected<ReferenceExecutionResult>
interpretReferenceProgram(const ReferenceProgram::Impl &program,
                          llvm::ArrayRef<ReferenceInputBinding> inputs,
                          ReferenceExecutionOptions options);
llvm::Expected<ReferenceMultiRankExecutionResult> interpretReferencePrograms(
    llvm::ArrayRef<const ReferenceProgram::Impl *> programs,
    llvm::ArrayRef<ReferenceRankInvocation> invocations,
    ReferenceExecutionOptions options);

} // namespace reference_detail
} // namespace wafer::compiler

#endif // WAFER_COMPILER_REFERENCEEXECUTORINTERNAL_H
