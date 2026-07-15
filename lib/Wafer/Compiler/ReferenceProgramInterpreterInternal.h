//===- ReferenceProgramInterpreterInternal.h - Interpreter private API -*- C++
//-*-===//
#pragma once

#include "ReferenceExecutorInternal.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace wafer::compiler::reference_detail {

using Command = ReferenceProgram::Impl::Command;
using CommandKind = ReferenceProgram::Impl::CommandKind;
using Scalar = ReferenceProgram::Impl::Scalar;
using ValueId = ReferenceProgram::Impl::ValueId;

struct RuntimeValue {
  std::optional<BufferView> buffer;
  std::optional<Scalar> scalar;
};

enum class TransferKind { Return, Branch };

struct ControlTransfer {
  TransferKind kind = TransferKind::Return;
  uint32_t successor = 0;
  std::vector<ValueId> inputs;
};

struct TransportEventKey {
  int64_t source = -1;
  int64_t destination = -1;
  int64_t communication = -1;
  uint32_t phase = 0;
  int64_t round = -1;
  int64_t payloadSlice = -1;
  std::vector<int64_t> controlInstance;

  bool operator<(const TransportEventKey &other) const {
    return std::tie(source, destination, communication, phase, round,
                    payloadSlice, controlInstance) <
           std::tie(other.source, other.destination, other.communication,
                    other.phase, other.round, other.payloadSlice,
                    other.controlInstance);
  }
};

class TransportCoordinator {
public:
  virtual ~TransportCoordinator() = default;
  virtual llvm::Error issue(const TransportEventKey &key, bool isSend,
                            uint64_t byteCount, int64_t remoteReceiverOffset,
                            const BufferView &buffer) = 0;
  virtual bool isComplete(const TransportEventKey &key) const = 0;
};

struct TransportToken {
  TransportEventKey key;
  bool isSend = false;
};

class TransportBlockedError : public llvm::ErrorInfo<TransportBlockedError> {
public:
  static char ID;
  explicit TransportBlockedError(std::string message)
      : message(std::move(message)) {}
  void log(llvm::raw_ostream &stream) const override { stream << message; }
  std::error_code convertToErrorCode() const override {
    return llvm::inconvertibleErrorCode();
  }
  std::string message;
};

struct ProgramRunAttempt {
  std::optional<ReferenceExecutionResult> result;
  std::string blockedReason;
};

llvm::Error forEachLogicalIndex(
    llvm::ArrayRef<int64_t> shape,
    llvm::function_ref<llvm::Error(llvm::ArrayRef<int64_t>)> callback);

class ProgramInterpreter {
public:
  ProgramInterpreter(const ReferenceProgram::Impl &program,
                     ReferenceExecutionOptions options,
                     TransportCoordinator *transport = nullptr);

  llvm::Expected<ReferenceExecutionResult>
  run(llvm::ArrayRef<ReferenceInputBinding> inputs);

  llvm::Expected<ProgramRunAttempt>
  runUntilBlocked(llvm::ArrayRef<ReferenceInputBinding> inputs);

private:
  llvm::Error bindEntryArguments(llvm::ArrayRef<ReferenceInputBinding> inputs);

  llvm::Expected<BufferView> importTensor(const ReferenceTensor &tensor,
                                          mlir::MemRefType type);

  llvm::Expected<ReferenceTensor> exportTensor(const BufferView &view,
                                               llvm::StringRef dtype,
                                               llvm::ArrayRef<int64_t> shape);

  llvm::Expected<std::vector<ValueId>> executeControlFlow(
      const ReferenceProgram::Impl::ControlFlowProgram &controlFlow);

  llvm::Expected<std::vector<ValueId>>
  executeStructuredBlock(const ReferenceProgram::Impl::BlockProgram &block);

  llvm::Expected<ControlTransfer>
  executeBlock(const ReferenceProgram::Impl::BlockProgram &block);

  llvm::Expected<RuntimeValue> readValue(ValueId value) const;

  llvm::Expected<std::vector<RuntimeValue>>
  readValues(llvm::ArrayRef<ValueId> values) const;

  llvm::Error writeValues(llvm::ArrayRef<ValueId> destinations,
                          llvm::ArrayRef<RuntimeValue> values);

  llvm::Error assignValues(llvm::ArrayRef<ValueId> destinations,
                           llvm::ArrayRef<ValueId> sources);

  llvm::Expected<bool> lookupBoolean(ValueId value) const;

  llvm::Expected<int64_t> lookupSignedInteger(ValueId value) const;

  llvm::Expected<BufferView> allocate(const Command &command);

  llvm::Expected<BufferView> lookup(ValueId value) const;

  llvm::Error copyChunks(const BufferView &source, const BufferView &dest,
                         llvm::ArrayRef<int64_t> sourceOffsets,
                         llvm::ArrayRef<int64_t> destOffsets,
                         int64_t innerBytes);

  llvm::Expected<std::vector<int64_t>>
  descriptorOffsets(llvm::ArrayRef<int64_t> strides,
                    llvm::ArrayRef<int64_t> iterations, int64_t base = 0);

  llvm::Error executeRDMA(const Command &command);

  llvm::Error executeWDMA(const Command &command);

  llvm::Error executeGatherScatter(const Command &command);

  llvm::Error executeConvert(const Command &command);

  llvm::Error executeGemm(const Command &command);

  llvm::Error executeReduce(const Command &command);

  llvm::Error executeElementwise(const Command &command);

  llvm::Expected<float> convertScalarToF32(const Scalar &scalar);

  llvm::Error executeFill(const Command &command);

  llvm::Error executeBit2Fp(const Command &command);

  llvm::Error executeMaskMove(const Command &command);

  uint64_t nextStochasticBits();

  const ReferenceProgram::Impl &program;
  std::shared_ptr<Storage> spmArena;
  std::shared_ptr<Storage> ddrArena;
  ReferenceExecutionOptions options;
  uint64_t stochasticState;
  TransportCoordinator *transport = nullptr;
  llvm::DenseMap<ValueId, BufferView> buffers;
  llvm::DenseMap<ValueId, Scalar> scalars;
  llvm::DenseMap<ValueId, TransportToken> tokens;
  std::vector<int64_t> controlInstance;

  static constexpr int64_t kIfControlMarker =
      std::numeric_limits<int64_t>::min();
  static constexpr int64_t kForControlMarker =
      std::numeric_limits<int64_t>::min() + 1;
};

llvm::Expected<ReferenceMultiRankExecutionResult>
interpretReferenceProgramsWithDTE(
    llvm::ArrayRef<const ReferenceProgram::Impl *> programs,
    llvm::ArrayRef<ReferenceRankInvocation> invocations,
    ReferenceExecutionOptions options);

} // namespace wafer::compiler::reference_detail
