//===- ExecutionCost.cpp - Loop-aware instruction costs --------*- C++ -*-===//

#include "Internal.h"

#include "Wafer/Analysis/StaticBufferRange.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/AsyncTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <array>
#include <optional>

namespace wafer::analysis::detail {
namespace {

static Quantity getElementCount(mlir::Value value) {
  auto type = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  if (!type || !type.hasStaticShape())
    return Quantity::unknown(ScheduleCostReason::UnknownPhysicalGeometry);
  Quantity count{1};
  for (int64_t dim : type.getShape()) {
    if (dim < 0)
      return Quantity::unknown(ScheduleCostReason::UnknownPhysicalGeometry);
    count = multiply(count, static_cast<uint64_t>(dim));
  }
  return count;
}

static Quantity product(llvm::ArrayRef<int64_t> values) {
  Quantity result{1};
  for (int64_t value : values) {
    if (value < 0) {
      result = multiply(
          result, Quantity::unsupported(
                      ScheduleCostReason::UnsupportedInstructionSemantics));
      continue;
    }
    result = multiply(result, static_cast<uint64_t>(value));
  }
  return result;
}

enum class ScalarClass { F16Bf16, F32, Other, Unsupported };

static ScalarClass classifyScalar(mlir::Type type) {
  if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(type))
    type = shaped.getElementType();
  if (type.isF16() || mlir::isa<mlir::BFloat16Type>(type))
    return ScalarClass::F16Bf16;
  if (type.isF32())
    return ScalarClass::F32;
  if (mlir::isa<mlir::IntegerType, mlir::FloatType>(type))
    return ScalarClass::Other;
  return ScalarClass::Unsupported;
}

static void addVectorCost(InstructionProgramCost &cost, mlir::Type type,
                          Quantity logicalOps, Quantity multiplicity) {
  Quantity total = multiply(logicalOps, multiplicity);
  switch (classifyScalar(type)) {
  case ScalarClass::F16Bf16:
    add(cost.compute.vectorF16Bf16LogicalOps, total);
    return;
  case ScalarClass::F32:
    add(cost.compute.vectorF32LogicalOps, total);
    return;
  case ScalarClass::Other:
    add(cost.compute.vectorOtherLogicalOps, total);
    return;
  case ScalarClass::Unsupported:
    add(cost.compute.vectorOtherLogicalOps, total);
    degrade(cost.compute.vectorOtherLogicalOps,
            ScheduleCostKnowledge::Unsupported,
            ScheduleCostReason::UnsupportedComputeType);
    return;
  }
}

static void addNPUCost(InstructionProgramCost &cost, mlir::Type type,
                       Quantity logicalOps, Quantity multiplicity) {
  Quantity total = multiply(logicalOps, multiplicity);
  switch (classifyScalar(type)) {
  case ScalarClass::F16Bf16:
    add(cost.compute.npuF16Bf16LogicalOps, total);
    return;
  case ScalarClass::F32:
  case ScalarClass::Other:
    add(cost.compute.npuOtherLogicalOps, total);
    return;
  case ScalarClass::Unsupported:
    add(cost.compute.npuOtherLogicalOps, total);
    degrade(cost.compute.npuOtherLogicalOps, ScheduleCostKnowledge::Unsupported,
            ScheduleCostReason::UnsupportedComputeType);
    return;
  }
}

static std::optional<NoCCollectiveKind>
classifyCollective(DTEProtocolPhase phase) {
  switch (phase) {
  case DTEProtocolPhase::CollectivePermute:
    return NoCCollectiveKind::CollectivePermute;
  case DTEProtocolPhase::AllToAll:
    return NoCCollectiveKind::AllToAll;
  case DTEProtocolPhase::AllGatherDirect:
  case DTEProtocolPhase::AllGatherRing:
    return NoCCollectiveKind::AllGather;
  case DTEProtocolPhase::ReduceScatterDirect:
  case DTEProtocolPhase::ReduceScatterRing:
    return NoCCollectiveKind::ReduceScatter;
  case DTEProtocolPhase::AllReduceRing:
  case DTEProtocolPhase::AllReduceTreeReduce:
  case DTEProtocolPhase::AllReduceTreeBroadcast:
    return NoCCollectiveKind::AllReduce;
  case DTEProtocolPhase::PeerDataflow:
    // Peer-resident dataflow is an exact point-to-point transfer rather than
    // an instance of one of the five collective families. Its payload still
    // contributes to aggregate transmit/receive, event/instruction, and
    // whole-card minimum-hop link-byte demand.
    return std::nullopt;
  }
  llvm_unreachable("unhandled DTE protocol phase");
}

static void markDirectionalNoCUnknown(InstructionProgramCost &cost) {
  for (ScheduleCostMetric &metric : cost.noc.directionalTransmitBytes)
    degrade(metric, ScheduleCostKnowledge::Unknown,
            ScheduleCostReason::UnresolvedNoCRoute);
}

static void collectResourceCost(mlir::Operation *op,
                                InstructionProgramCost &cost,
                                Quantity multiplicity) {
  auto addBytes = [&](ScheduleCostMetric &metric, int64_t bytes) {
    if (bytes < 0) {
      degrade(metric, ScheduleCostKnowledge::Unknown,
              ScheduleCostReason::UnknownResourceBytes);
      return;
    }
    add(metric, multiply(Quantity{static_cast<uint64_t>(bytes)}, multiplicity));
  };
  auto physicalBytes = [](mlir::Value value) -> int64_t {
    auto type = mlir::dyn_cast<mlir::MemRefType>(value.getType());
    if (!type)
      return -1;
    std::optional<WaferPhysicalTensorInfo> info =
        computeWaferPhysicalTensorInfo(type);
    return info ? info->physicalBytes : -1;
  };

  if (auto rdma = mlir::dyn_cast<InstrRDMAOp>(op)) {
    addBytes(cost.ddrReadBytes, rdma.getByteCount());
    addBytes(cost.spmMovementBytes, rdma.getByteCount());
    return;
  }
  if (auto wdma = mlir::dyn_cast<InstrWDMAOp>(op)) {
    addBytes(cost.ddrWriteBytes, wdma.getByteCount());
    addBytes(cost.spmMovementBytes, wdma.getByteCount());
    return;
  }
  if (auto gatherScatter = mlir::dyn_cast<InstrGatherScatterOp>(op)) {
    addBytes(cost.spmMovementBytes, gatherScatter.getByteCount());
    addBytes(cost.gatherScatterBytes, gatherScatter.getByteCount());
    return;
  }
  if (auto dataMove = mlir::dyn_cast<InstrTDMADataMoveOp>(op)) {
    addBytes(cost.spmMovementBytes, physicalBytes(dataMove.getDest()));
    return;
  }
  if (auto maskMove = mlir::dyn_cast<InstrMaskMoveOp>(op)) {
    addBytes(cost.spmMovementBytes, physicalBytes(maskMove.getDest()));
  }
}

static mlir::Type getVectorInputType(InstrElementwiseOp op) {
  if (!op.getInputs().empty())
    return op.getInputs().front().getType();
  return op.getDest().getType();
}

static void collectComputeCost(mlir::Operation *op,
                               InstructionProgramCost &cost,
                               Quantity multiplicity) {
  if (auto elementwise = mlir::dyn_cast<InstrElementwiseOp>(op)) {
    addVectorCost(cost, getVectorInputType(elementwise),
                  getElementCount(elementwise.getDest()), multiplicity);
    return;
  }
  if (auto bit2fp = mlir::dyn_cast<InstrBit2FpOp>(op)) {
    addVectorCost(cost, bit2fp.getDest().getType(),
                  getElementCount(bit2fp.getDest()), multiplicity);
    return;
  }
  if (auto reduce = mlir::dyn_cast<InstrReduceOp>(op)) {
    Quantity input = getElementCount(reduce.getInput());
    Quantity output = getElementCount(reduce.getDest());
    Quantity logicalOps;
    if (input.knowledge != ScheduleCostKnowledge::Known)
      logicalOps = input;
    else if (output.knowledge != ScheduleCostKnowledge::Known)
      logicalOps = output;
    else if (input.value < output.value)
      logicalOps = Quantity::unsupported(
          ScheduleCostReason::UnsupportedInstructionSemantics);
    else if (reduce.getKind() == InstrReduceKind::Avg)
      // R inputs require R-1 reductions and one division per output.
      logicalOps = input;
    else
      logicalOps = Quantity{input.value - output.value};
    addVectorCost(cost, reduce.getInput().getType(), logicalOps, multiplicity);
    return;
  }
  if (auto convert = mlir::dyn_cast<InstrConvertOp>(op)) {
    // Mixed-endpoint converts cannot be assigned to one arithmetic rate.
    add(cost.compute.vectorOtherLogicalOps,
        multiply(getElementCount(convert.getDest()), multiplicity));
    return;
  }
  if (auto gemm = mlir::dyn_cast<InstrGemmOp>(op)) {
    Quantity logicalOps{2};
    for (int64_t dim : {gemm.getM(), gemm.getK(), gemm.getN()}) {
      if (dim < 0) {
        logicalOps =
            multiply(logicalOps,
                     Quantity::unsupported(
                         ScheduleCostReason::UnsupportedInstructionSemantics));
        continue;
      }
      logicalOps = multiply(logicalOps, static_cast<uint64_t>(dim));
    }
    int64_t batch = gemm.getBatchCount().value_or(1);
    if (batch < 0)
      logicalOps = multiply(
          logicalOps, Quantity::unsupported(
                          ScheduleCostReason::UnsupportedInstructionSemantics));
    else
      logicalOps = multiply(logicalOps, static_cast<uint64_t>(batch));
    addNPUCost(cost, gemm.getDest().getType(), logicalOps, multiplicity);
    return;
  }
  if (auto conv = mlir::dyn_cast<InstrConvOp>(op)) {
    llvm::ArrayRef<int64_t> output = conv.getOutputShape();
    llvm::ArrayRef<int64_t> weight = conv.getWeightShape();
    Quantity logicalOps = product(output);
    if (conv.getKind() != InstrConvKind::Conv || weight.size() != 4)
      logicalOps = multiply(
          logicalOps, Quantity::unsupported(
                          ScheduleCostReason::UnsupportedInstructionSemantics));
    else {
      logicalOps = multiply(logicalOps, product(weight.take_front(3)));
      logicalOps = multiply(logicalOps, 2);
    }
    addNPUCost(cost, conv.getDest().getType(), logicalOps, multiplicity);
    return;
  }

  // Fill and movement/DTE instructions have no arithmetic logical-op cost.
  if (mlir::isa<InstrFillOp, InstrMaskMoveOp, InstrRDMAOp, InstrWDMAOp,
                InstrGatherScatterOp, InstrTDMADataMoveOp, InstrDTESendOp,
                InstrDTERecvOp, InstrDTEWaitOp>(op))
    return;

  auto instruction = mlir::dyn_cast<WaferInstructionOpInterface>(op);
  if (!instruction)
    return;
  if (instruction.getInstructionFamily() == InstrFamily::NE)
    degrade(cost.compute.npuOtherLogicalOps, ScheduleCostKnowledge::Unsupported,
            ScheduleCostReason::UnsupportedInstructionSemantics);
  else if (instruction.getInstructionFamily() == InstrFamily::CT)
    degrade(cost.compute.vectorOtherLogicalOps,
            ScheduleCostKnowledge::Unsupported,
            ScheduleCostReason::UnsupportedInstructionSemantics);
}

static void collectNoCCost(mlir::Operation *op, InstructionProgramCost &cost,
                           Quantity multiplicity) {
  if (auto send = mlir::dyn_cast<InstrDTESendOp>(op)) {
    Quantity bytes =
        send.getBytes() < 0
            ? Quantity::unsupported(
                  ScheduleCostReason::UnsupportedInstructionSemantics)
            : Quantity{static_cast<uint64_t>(send.getBytes())};
    Quantity total = multiply(bytes, multiplicity);
    add(cost.noc.aggregateTransmitBytes, total);
    std::optional<NoCCollectiveKind> kind =
        classifyCollective(send.getMessage().getPhase());
    if (kind)
      add(cost.noc.collectiveTransmitBytes[static_cast<size_t>(*kind)], total);
    markDirectionalNoCUnknown(cost);
    return;
  }
  if (auto recv = mlir::dyn_cast<InstrDTERecvOp>(op)) {
    Quantity bytes =
        recv.getBytes() < 0
            ? Quantity::unsupported(
                  ScheduleCostReason::UnsupportedInstructionSemantics)
            : Quantity{static_cast<uint64_t>(recv.getBytes())};
    add(cost.noc.aggregateReceiveBytes, multiply(bytes, multiplicity));
    return;
  }
  if (auto wait = mlir::dyn_cast<InstrDTEWaitOp>(op))
    add(cost.noc.waitedEventCount,
        multiply(Quantity{wait.getTokens().size()}, multiplicity));
}

static uint64_t countParticipants(uint32_t participantMask) {
  uint64_t count = 0;
  while (participantMask != 0U) {
    count += participantMask & 1U;
    participantMask >>= 1U;
  }
  return count;
}

static bool isInstructionProgramOperation(mlir::Operation *op) {
  return mlir::isa<WaferInstructionOpInterface, SyncNCCJoinOp>(op);
}

static bool hasFollowingExecutableWork(mlir::Operation *op) {
  if (op->getParentOfType<mlir::scf::ForOp>())
    return true;
  for (mlir::Operation *current = op; current != nullptr;
       current = current->getParentOp()) {
    for (mlir::Operation *next = current->getNextNode(); next != nullptr;
         next = next->getNextNode()) {
      if (!next->hasTrait<mlir::OpTrait::IsTerminator>())
        return true;
    }
    if (mlir::isa_and_nonnull<mlir::func::FuncOp>(current->getParentOp()))
      return false;
  }
  return false;
}

static void addExecutionCount(InstructionExecutionCount &count,
                              ExecutionMultiplicity multiplicity,
                              uint64_t weight = 1,
                              bool countStaticSite = true) {
  if (countStaticSite)
    add(count.staticSites, Quantity{weight});
  add(count.exactExecutions, multiply(multiplicity.exact, weight));
  add(count.lowerBound, multiply(multiplicity.lowerBound, weight));
  add(count.upperBound, multiply(multiplicity.upperBound, weight));
}

static void collectNCCDrainWork(mlir::Operation *op,
                                InstructionProgramWork &work,
                                ExecutionMultiplicity multiplicity,
                                bool countStaticSite) {
  NCCCompletionContract contract = getNCCCompletionContract(op);
  bool isSteadyState =
      static_cast<bool>(op->getParentOfType<mlir::scf::ForOp>());
  bool isNonTerminal = hasFollowingExecutableWork(op);
  uint64_t participantWaits = countParticipants(contract.participantMask);
  if (contract.behavior == LocalInstructionCompletion::ParticipantJoin) {
    addExecutionCount(work.nccJoins, multiplicity, 1, countStaticSite);
    if (isSteadyState)
      addExecutionCount(work.steadyStateNCCJoins, multiplicity, 1,
                        countStaticSite);
    if (isNonTerminal)
      addExecutionCount(work.nonTerminalNCCJoins, multiplicity, 1,
                        countStaticSite);
    addExecutionCount(work.nccParticipantWaits, multiplicity, participantWaits,
                      countStaticSite);
    if (isSteadyState)
      addExecutionCount(work.steadyStateNCCParticipantWaits, multiplicity,
                        participantWaits, countStaticSite);
    if (isNonTerminal)
      addExecutionCount(work.nonTerminalNCCParticipantWaits, multiplicity,
                        participantWaits, countStaticSite);
    return;
  }
  if (contract.behavior != LocalInstructionCompletion::SynchronousWriteback)
    return;
  addExecutionCount(work.intrinsicNCCDrains, multiplicity, 1, countStaticSite);
  addExecutionCount(work.nccParticipantWaits, multiplicity, participantWaits,
                    countStaticSite);
  if (isSteadyState)
    addExecutionCount(work.steadyStateNCCParticipantWaits, multiplicity,
                      participantWaits, countStaticSite);
  if (isNonTerminal)
    addExecutionCount(work.nonTerminalNCCParticipantWaits, multiplicity,
                      participantWaits, countStaticSite);
}

static void collectInstructionWork(mlir::Operation *op,
                                   InstructionProgramWork &work,
                                   ExecutionMultiplicity multiplicity,
                                   bool countStaticSite) {
  addExecutionCount(work.instructions, multiplicity, 1, countStaticSite);
  for (mlir::Value result : op->getResults())
    if (mlir::isa<mlir::async::TokenType>(result.getType()))
      addExecutionCount(work.asynchronousEvents, multiplicity, 1,
                        countStaticSite);

  if (mlir::isa<InstrGatherScatterOp>(op))
    addExecutionCount(work.gatherScatterOperations, multiplicity, 1,
                      countStaticSite);
  if (mlir::isa<InstrDTEWaitOp>(op))
    addExecutionCount(work.dteWaitOperations, multiplicity, 1, countStaticSite);
  if (mlir::isa<InstrDTESendOp>(op))
    addExecutionCount(work.dteSendOperations, multiplicity, 1, countStaticSite);
  if (mlir::isa<InstrDTERecvOp>(op))
    addExecutionCount(work.dteReceiveOperations, multiplicity, 1,
                      countStaticSite);
  if (auto send = mlir::dyn_cast<InstrDTESendOp>(op)) {
    if (classifyCollective(send.getMessage().getPhase()))
      addExecutionCount(work.collectiveDTEIssues, multiplicity, 1,
                        countStaticSite);
    else
      addExecutionCount(work.peerDTEIssues, multiplicity, 1, countStaticSite);
  } else if (auto recv = mlir::dyn_cast<InstrDTERecvOp>(op)) {
    if (classifyCollective(recv.getMessage().getPhase()))
      addExecutionCount(work.collectiveDTEIssues, multiplicity, 1,
                        countStaticSite);
    else
      addExecutionCount(work.peerDTEIssues, multiplicity, 1, countStaticSite);
  }

  if (auto instruction = mlir::dyn_cast<WaferInstructionOpInterface>(op)) {
    InstructionExecutionCount *family = nullptr;
    switch (instruction.getInstructionFamily()) {
    case InstrFamily::RDMA:
      family = &work.rdmaIssues;
      break;
    case InstrFamily::WDMA:
      family = &work.wdmaIssues;
      break;
    case InstrFamily::TDMA:
      family = &work.tdmaIssues;
      break;
    case InstrFamily::CT:
      family = &work.ctIssues;
      break;
    case InstrFamily::NE:
      family = &work.neIssues;
      break;
    case InstrFamily::DTE:
      family = &work.dteOperations;
      break;
    }
    addExecutionCount(*family, multiplicity, 1, countStaticSite);
  }
  collectNCCDrainWork(op, work, multiplicity, countStaticSite);
}

static void collectInstructionCost(mlir::Operation *op,
                                   InstructionProgramCost &cost,
                                   ExecutionMultiplicity multiplicity,
                                   bool countStaticSite) {
  if (multiplicity.exact.knowledge == ScheduleCostKnowledge::Known &&
      multiplicity.exact.value == 0)
    return;
  collectInstructionWork(op, cost.work, multiplicity, countStaticSite);
  collectResourceCost(op, cost, multiplicity.exact);
  collectComputeCost(op, cost, multiplicity.exact);
  collectNoCCost(op, cost, multiplicity.exact);
}

enum class ConstantIndexKnowledge { Known, Unknown, Overflow };

struct ConstantIndex {
  ConstantIndexKnowledge knowledge = ConstantIndexKnowledge::Unknown;
  int64_t value = 0;
};

/// Evaluate the side-effect-free integer arithmetic that canonical IR
/// transformations use to rebuild static loop bounds. In particular, SCF
/// software pipelining rewrites a direct constant upper bound as
/// `upper - maxStage * step`. Treating that expression as dynamic would make
/// exact cost reject a semantically static candidate.
static ConstantIndex
evaluateConstantIndex(mlir::Value value,
                      llvm::DenseSet<mlir::Value> &activeValues) {
  if (std::optional<int64_t> constant = mlir::getConstantIntValue(value))
    return {ConstantIndexKnowledge::Known, *constant};
  if (!mlir::isa<mlir::IndexType, mlir::IntegerType>(value.getType()) ||
      !activeValues.insert(value).second)
    return {};

  auto evaluateBinary = [&](mlir::Value lhsValue, mlir::Value rhsValue,
                            auto checkedOperation) -> ConstantIndex {
    ConstantIndex lhs = evaluateConstantIndex(lhsValue, activeValues);
    ConstantIndex rhs = evaluateConstantIndex(rhsValue, activeValues);
    if (lhs.knowledge == ConstantIndexKnowledge::Overflow ||
        rhs.knowledge == ConstantIndexKnowledge::Overflow)
      return {ConstantIndexKnowledge::Overflow};
    if (lhs.knowledge != ConstantIndexKnowledge::Known ||
        rhs.knowledge != ConstantIndexKnowledge::Known)
      return {};
    int64_t result = 0;
    if (checkedOperation(lhs.value, rhs.value, result))
      return {ConstantIndexKnowledge::Overflow};
    return {ConstantIndexKnowledge::Known, result};
  };

  ConstantIndex result;
  if (auto add = value.getDefiningOp<mlir::arith::AddIOp>()) {
    result = evaluateBinary(add.getLhs(), add.getRhs(),
                            [](int64_t lhs, int64_t rhs, int64_t &folded) {
                              return llvm::AddOverflow(lhs, rhs, folded);
                            });
  } else if (auto sub = value.getDefiningOp<mlir::arith::SubIOp>()) {
    result = evaluateBinary(sub.getLhs(), sub.getRhs(),
                            [](int64_t lhs, int64_t rhs, int64_t &folded) {
                              return llvm::SubOverflow(lhs, rhs, folded);
                            });
  } else if (auto mul = value.getDefiningOp<mlir::arith::MulIOp>()) {
    result = evaluateBinary(mul.getLhs(), mul.getRhs(),
                            [](int64_t lhs, int64_t rhs, int64_t &folded) {
                              return llvm::MulOverflow(lhs, rhs, folded);
                            });
  }
  activeValues.erase(value);
  return result;
}

static Quantity getTripCount(mlir::scf::ForOp loop) {
  llvm::DenseSet<mlir::Value> activeValues;
  ConstantIndex lower =
      evaluateConstantIndex(loop.getLowerBound(), activeValues);
  ConstantIndex upper =
      evaluateConstantIndex(loop.getUpperBound(), activeValues);
  ConstantIndex step = evaluateConstantIndex(loop.getStep(), activeValues);
  if (lower.knowledge == ConstantIndexKnowledge::Overflow ||
      upper.knowledge == ConstantIndexKnowledge::Overflow ||
      step.knowledge == ConstantIndexKnowledge::Overflow)
    return Quantity::overflow();
  if (lower.knowledge != ConstantIndexKnowledge::Known ||
      upper.knowledge != ConstantIndexKnowledge::Known ||
      step.knowledge != ConstantIndexKnowledge::Known)
    return Quantity::unknown(ScheduleCostReason::DynamicLoopTripCount);
  if (step.value <= 0)
    return Quantity::unsupported(ScheduleCostReason::InvalidLoopStep);
  if (upper.value <= lower.value)
    return Quantity{0};
  uint64_t distance =
      static_cast<uint64_t>(upper.value) - static_cast<uint64_t>(lower.value);
  uint64_t positiveStep = static_cast<uint64_t>(step.value);
  return Quantity{distance / positiveStep +
                  static_cast<uint64_t>(distance % positiveStep != 0)};
}

class ProgramWalker {
public:
  ProgramWalker(
      llvm::function_ref<void(mlir::Operation *, ExecutionMultiplicity)>
          onInstruction,
      llvm::function_ref<void()> onUnsupportedControlFlow)
      : onInstruction(onInstruction),
        onUnsupportedControlFlow(onUnsupportedControlFlow) {}

  void walkRoot(mlir::Operation *root) {
    if (auto module = mlir::dyn_cast<mlir::ModuleOp>(root)) {
      for (mlir::func::FuncOp function : module.getOps<mlir::func::FuncOp>()) {
        if (function.isPrivate())
          continue;
        activeFunctions.insert(function.getOperation());
        walkOperation(function.getOperation(), ExecutionMultiplicity{});
        activeFunctions.erase(function.getOperation());
      }
      return;
    }
    if (mlir::isa<mlir::func::FuncOp>(root))
      activeFunctions.insert(root);
    walkOperation(root, ExecutionMultiplicity{});
  }

private:
  static ExecutionMultiplicity multiplyAll(ExecutionMultiplicity multiplicity,
                                           Quantity factor) {
    multiplicity.exact = multiply(multiplicity.exact, factor);
    multiplicity.lowerBound = multiply(multiplicity.lowerBound, factor);
    multiplicity.upperBound = multiply(multiplicity.upperBound, factor);
    return multiplicity;
  }

  static ExecutionMultiplicity
  enterIndeterminateLoop(ExecutionMultiplicity multiplicity,
                         Quantity tripCount) {
    multiplicity.exact = multiply(multiplicity.exact, tripCount);
    multiplicity.lowerBound = Quantity{0};
    multiplicity.upperBound = multiply(multiplicity.upperBound, tripCount);
    return multiplicity;
  }

  static ExecutionMultiplicity
  enterConditionalBranch(ExecutionMultiplicity multiplicity) {
    multiplicity.exact =
        multiply(multiplicity.exact,
                 Quantity::unknown(ScheduleCostReason::ConditionalControlFlow));
    multiplicity.lowerBound = Quantity{0};
    // One branch cannot execute more often than its parent. Summing the two
    // branch-local upper bounds later is conservative for every work kind.
    return multiplicity;
  }

  static ExecutionMultiplicity
  enterUnsupportedRegion(ExecutionMultiplicity multiplicity) {
    Quantity unsupported =
        Quantity::unsupported(ScheduleCostReason::UnsupportedControlFlow);
    multiplicity.exact = multiply(multiplicity.exact, unsupported);
    multiplicity.lowerBound = Quantity{0};
    multiplicity.upperBound = multiply(multiplicity.upperBound, unsupported);
    return multiplicity;
  }

  void walkRegions(mlir::Operation *op, ExecutionMultiplicity multiplicity) {
    for (mlir::Region &region : op->getRegions())
      walkRegion(region, multiplicity);
  }

  void walkRegion(mlir::Region &region, ExecutionMultiplicity multiplicity) {
    if (!region.empty() && !region.hasOneBlock())
      multiplicity = enterUnsupportedRegion(multiplicity);
    for (mlir::Block &block : region)
      for (mlir::Operation &op : block)
        walkOperation(&op, multiplicity);
  }

  void walkOperation(mlir::Operation *op, ExecutionMultiplicity multiplicity) {
    if (multiplicity.exact.knowledge == ScheduleCostKnowledge::Known &&
        multiplicity.exact.value == 0)
      return;
    if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
      Quantity tripCount = getTripCount(loop);
      if (tripCount.knowledge == ScheduleCostKnowledge::Known)
        walkRegion(loop.getRegion(), multiplyAll(multiplicity, tripCount));
      else
        walkRegion(loop.getRegion(),
                   enterIndeterminateLoop(multiplicity, tripCount));
      return;
    }
    if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
      std::optional<int64_t> condition = mlir::getConstantIntValue(
          mlir::getAsOpFoldResult(ifOp.getCondition()));
      if (condition) {
        mlir::Region &selected =
            *condition ? ifOp.getThenRegion() : ifOp.getElseRegion();
        walkRegion(selected, multiplicity);
      } else {
        ExecutionMultiplicity conditional =
            enterConditionalBranch(multiplicity);
        walkRegion(ifOp.getThenRegion(), conditional);
        walkRegion(ifOp.getElseRegion(), conditional);
      }
      return;
    }

    if (isInstructionProgramOperation(op)) {
      onInstruction(op, multiplicity);
      return;
    }
    if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) {
      mlir::func::FuncOp callee =
          mlir::SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(
              call, call.getCalleeAttr());
      if (!callee || callee.isDeclaration() ||
          !activeFunctions.insert(callee.getOperation()).second) {
        onUnsupportedControlFlow();
        return;
      }
      walkRegion(callee.getBody(), multiplicity);
      activeFunctions.erase(callee.getOperation());
      return;
    }
    if (op->getNumRegions() == 0)
      return;
    if (mlir::isa<mlir::ModuleOp, mlir::func::FuncOp, TileRegionOp>(op)) {
      walkRegions(op, multiplicity);
      return;
    }

    walkRegions(op, enterUnsupportedRegion(multiplicity));
  }

  llvm::function_ref<void(mlir::Operation *, ExecutionMultiplicity)>
      onInstruction;
  llvm::function_ref<void()> onUnsupportedControlFlow;
  llvm::DenseSet<mlir::Operation *> activeFunctions;
};

} // namespace

void walkInstructionProgramWork(
    mlir::Operation *root,
    llvm::function_ref<void(mlir::Operation *, ExecutionMultiplicity)>
        onInstruction,
    llvm::function_ref<void()> onUnsupportedControlFlow) {
  ProgramWalker(onInstruction, onUnsupportedControlFlow).walkRoot(root);
}

void walkInstructionProgram(
    mlir::Operation *root,
    llvm::function_ref<void(mlir::Operation *, Quantity)> onInstruction,
    llvm::function_ref<void()> onUnsupportedControlFlow) {
  auto adapt = [&](mlir::Operation *operation,
                   ExecutionMultiplicity multiplicity) {
    onInstruction(operation, multiplicity.exact);
  };
  walkInstructionProgramWork(root, adapt, onUnsupportedControlFlow);
}

void collectExecutionCost(mlir::Operation *root, InstructionProgramCost &cost) {
  llvm::DenseSet<mlir::Operation *> countedStaticSites;
  auto collect = [&](mlir::Operation *operation,
                     ExecutionMultiplicity multiplicity) {
    bool executes =
        multiplicity.exact.knowledge != ScheduleCostKnowledge::Known ||
        multiplicity.exact.value != 0;
    bool countStaticSite =
        executes && countedStaticSites.insert(operation).second;
    collectInstructionCost(operation, cost, multiplicity, countStaticSite);
  };
  auto markAllUnsupported = [&]() {
    auto mark = [](ScheduleCostMetric &metric) {
      degrade(metric, ScheduleCostKnowledge::Unsupported,
              ScheduleCostReason::UnsupportedControlFlow);
    };
    mark(cost.compute.npuF16Bf16LogicalOps);
    mark(cost.compute.npuOtherLogicalOps);
    mark(cost.compute.vectorF16Bf16LogicalOps);
    mark(cost.compute.vectorF32LogicalOps);
    mark(cost.compute.vectorOtherLogicalOps);
    mark(cost.ddrReadBytes);
    mark(cost.ddrWriteBytes);
    mark(cost.spmMovementBytes);
    mark(cost.gatherScatterBytes);
    mark(cost.noc.staticIssueSiteCount);
    mark(cost.noc.aggregateTransmitBytes);
    mark(cost.noc.aggregateReceiveBytes);
    mark(cost.noc.transmitMessageCount);
    mark(cost.noc.receiveMessageCount);
    mark(cost.noc.waitOperationCount);
    mark(cost.noc.waitedEventCount);
    for (ScheduleCostMetric &metric : cost.noc.directionalTransmitBytes)
      mark(metric);
    for (ScheduleCostMetric &metric : cost.noc.collectiveTransmitBytes)
      mark(metric);
    auto markCount = [&](InstructionExecutionCount &count) {
      mark(count.staticSites);
      mark(count.exactExecutions);
      mark(count.lowerBound);
      mark(count.upperBound);
    };
    for (InstructionWorkCountMember member : kInstructionWorkCountMembers)
      markCount(cost.work.*member);
  };
  walkInstructionProgramWork(root, collect, markAllUnsupported);

  auto exact = [](const InstructionExecutionCount &count) {
    return count.exactExecutions;
  };
  cost.instructionCount = exact(cost.work.instructions);
  cost.eventCount = exact(cost.work.asynchronousEvents);
  cost.nccJoinCount = exact(cost.work.nccJoins);
  cost.steadyStateNCCJoinCount = exact(cost.work.steadyStateNCCJoins);
  cost.nonTerminalNCCJoinCount = exact(cost.work.nonTerminalNCCJoins);
  cost.nccParticipantWaitCount = exact(cost.work.nccParticipantWaits);
  cost.steadyStateNCCParticipantWaitCount =
      exact(cost.work.steadyStateNCCParticipantWaits);
  cost.nonTerminalNCCParticipantWaitCount =
      exact(cost.work.nonTerminalNCCParticipantWaits);
  cost.intrinsicNCCDrainCount = exact(cost.work.intrinsicNCCDrains);
  cost.noc.staticIssueSiteCount = cost.work.dteSendOperations.staticSites;
  add(cost.noc.staticIssueSiteCount,
      Quantity{cost.work.dteReceiveOperations.staticSites.value,
               cost.work.dteReceiveOperations.staticSites.knowledge,
               cost.work.dteReceiveOperations.staticSites.reason});
  cost.noc.transmitMessageCount = exact(cost.work.dteSendOperations);
  cost.noc.receiveMessageCount = exact(cost.work.dteReceiveOperations);
  cost.noc.waitOperationCount = exact(cost.work.dteWaitOperations);
}

namespace {

struct BufferDependencyState {
  uint64_t lastWriterDepth = 0;
  uint64_t maximumReaderDepth = 0;
};

struct StructuralScheduleFacts {
  uint64_t maximumDependencyDepth = 0;
  uint64_t readyPriorityInversions = 0;
};

struct NCCWorkerScheduleState {
  /// Maximum depth among ordered issues that have not crossed a completion
  /// boundary. Independent engines on one worker are deliberately not chained
  /// here; the participant join consumes their maximum frontier.
  uint64_t pendingIssueDepth = 0;
  /// A participant join is an issue-order floor only for the workers that it
  /// names. Unrelated workers and Direct DTE remain independent.
  uint64_t completionDepth = 0;
};

static unsigned getStaticReadyPriority(mlir::Operation *operation) {
  auto instruction = mlir::dyn_cast<WaferInstructionOpInterface>(operation);
  if (!instruction)
    return 3;
  switch (instruction.getInstructionFamily()) {
  case InstrFamily::RDMA:
  case InstrFamily::WDMA:
  case InstrFamily::TDMA:
    return 0;
  case InstrFamily::DTE:
    return 1;
  case InstrFamily::CT:
  case InstrFamily::NE:
    return 2;
  }
  llvm_unreachable("unknown instruction family");
}

static bool hasUnsupportedDependencyControlFlow(mlir::func::FuncOp function) {
  bool unsupported = false;
  function.walk([&](mlir::Operation *operation) {
    if (mlir::isa<mlir::scf::ForOp, mlir::scf::IfOp, mlir::func::CallOp>(
            operation))
      unsupported = true;
  });
  return unsupported;
}

static std::optional<StructuralScheduleFacts>
analyzeFunctionStructuralSchedule(mlir::func::FuncOp function) {
  if (function.isDeclaration() || !function.getBody().hasOneBlock() ||
      hasUnsupportedDependencyControlFlow(function))
    return std::nullopt;

  llvm::DenseMap<mlir::Value, BufferDependencyState> buffers;
  llvm::DenseMap<mlir::Operation *, uint64_t> operationDepths;
  std::array<NCCWorkerScheduleState, kNCCWorkerCount> nccWorkers;
  uint64_t maximumDepth = 0;
  uint64_t readyPriorityInversions = 0;
  uint64_t seenReadyPriorities[3] = {};
  mlir::Block *readyBlock = nullptr;
  bool overflow = false;

  function.walk([&](mlir::Operation *operation) {
    if (overflow || !isInstructionProgramOperation(operation))
      return;

    auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
    if (!effects) {
      overflow = true;
      return;
    }

    uint64_t predecessorDepth = 0;
    for (mlir::Value operand : operation->getOperands()) {
      auto definition = operationDepths.find(operand.getDefiningOp());
      if (definition != operationDepths.end())
        predecessorDepth = std::max(predecessorDepth, definition->second);
    }

    llvm::DenseMap<mlir::Value, unsigned> accesses;
    llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 8> instances;
    effects.getEffects(instances);
    for (const auto &effect : instances) {
      mlir::Value value = effect.getValue();
      if (!value)
        continue;
      unsigned &flags = accesses[value];
      flags |=
          llvm::isa<mlir::MemoryEffects::Read>(effect.getEffect()) ? 1u : 0u;
      flags |=
          llvm::isa<mlir::MemoryEffects::Write>(effect.getEffect()) ? 2u : 0u;
    }
    for (auto [value, flags] : accesses) {
      const BufferDependencyState &state = buffers[value];
      if (flags & 1u)
        predecessorDepth = std::max(predecessorDepth, state.lastWriterDepth);
      if (flags & 2u)
        predecessorDepth = std::max({predecessorDepth, state.lastWriterDepth,
                                     state.maximumReaderDepth});
    }

    NCCCompletionContract completion = getNCCCompletionContract(operation);
    bool isCompletionBarrier =
        completion.behavior == LocalInstructionCompletion::ParticipantJoin ||
        completion.behavior == LocalInstructionCompletion::SynchronousWriteback;
    if (completion.behavior == LocalInstructionCompletion::OrderedPending &&
        completion.issueWorker) {
      const auto worker = static_cast<uint32_t>(*completion.issueWorker);
      if (worker >= nccWorkers.size()) {
        overflow = true;
        return;
      }
      predecessorDepth =
          std::max(predecessorDepth, nccWorkers[worker].completionDepth);
    } else if (isCompletionBarrier) {
      uint32_t participants = completion.participantMask;
      for (uint32_t worker = 0; worker < nccWorkers.size(); ++worker) {
        if ((participants & (uint32_t{1} << worker)) == 0)
          continue;
        predecessorDepth =
            std::max({predecessorDepth, nccWorkers[worker].pendingIssueDepth,
                      nccWorkers[worker].completionDepth});
      }
    }
    uint64_t depth = 0;
    if (!checkedAdd(predecessorDepth, 1, depth)) {
      overflow = true;
      return;
    }
    operationDepths[operation] = depth;
    maximumDepth = std::max(maximumDepth, depth);

    for (auto [value, flags] : accesses) {
      BufferDependencyState &state = buffers[value];
      if (flags & 2u) {
        state.lastWriterDepth = depth;
        state.maximumReaderDepth = 0;
      } else if (flags & 1u) {
        state.maximumReaderDepth = std::max(state.maximumReaderDepth, depth);
      }
    }
    if (completion.behavior == LocalInstructionCompletion::OrderedPending &&
        completion.issueWorker) {
      const auto worker = static_cast<uint32_t>(*completion.issueWorker);
      nccWorkers[worker].pendingIssueDepth =
          std::max(nccWorkers[worker].pendingIssueDepth, depth);
    } else if (isCompletionBarrier) {
      uint32_t participants = completion.participantMask;
      for (uint32_t worker = 0; worker < nccWorkers.size(); ++worker) {
        if ((participants & (uint32_t{1} << worker)) == 0)
          continue;
        nccWorkers[worker].pendingIssueDepth = 0;
        nccWorkers[worker].completionDepth = depth;
      }
    }

    if (operation->getBlock() != readyBlock || isCompletionBarrier) {
      readyBlock = operation->getBlock();
      std::fill_n(seenReadyPriorities, 3, 0);
    }
    if (!isCompletionBarrier) {
      unsigned priority = getStaticReadyPriority(operation);
      uint64_t precedingLowerPriority = 0;
      for (unsigned index = priority + 1; index < 3; ++index)
        if (!checkedAdd(precedingLowerPriority, seenReadyPriorities[index],
                        precedingLowerPriority)) {
          overflow = true;
          return;
        }
      if (!checkedAdd(readyPriorityInversions, precedingLowerPriority,
                      readyPriorityInversions)) {
        overflow = true;
        return;
      }
      if (!checkedAdd(seenReadyPriorities[priority], 1,
                      seenReadyPriorities[priority]))
        overflow = true;
    }
  });
  if (overflow)
    return std::nullopt;
  return StructuralScheduleFacts{maximumDepth, readyPriorityInversions};
}

} // namespace

void collectDataDependencyDepth(mlir::Operation *root,
                                InstructionProgramCost &cost) {
  bool sawFunction = false;
  bool unknown = false;
  root->walk([&](mlir::func::FuncOp function) {
    if (function.isDeclaration())
      return;
    sawFunction = true;
    std::optional<StructuralScheduleFacts> facts =
        analyzeFunctionStructuralSchedule(function);
    if (!facts) {
      unknown = true;
      return;
    }
    cost.dataDependencyDepth.value =
        std::max(cost.dataDependencyDepth.value, facts->maximumDependencyDepth);
    add(cost.readyOrderPriorityInversions,
        Quantity{facts->readyPriorityInversions});
  });
  if (!sawFunction || unknown) {
    degrade(cost.dataDependencyDepth, ScheduleCostKnowledge::Unknown,
            ScheduleCostReason::UnsupportedControlFlow);
    degrade(cost.readyOrderPriorityInversions, ScheduleCostKnowledge::Unknown,
            ScheduleCostReason::UnsupportedControlFlow);
  }
}

namespace {

static bool isSPMValue(mlir::Value value) {
  auto type = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  MemoryAttr memory = type ? getWaferMemoryAttr(type) : MemoryAttr{};
  return memory && memory.getSpace() == MemorySpace::SPM;
}

static bool hasRotatingSPMState(mlir::scf::ForOp loop) {
  mlir::Block::BlockArgListType iterArgs = loop.getRegionIterArgs();
  if (iterArgs.empty())
    return false;
  auto yield = mlir::cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
  for (auto [index, iterArg] : llvm::enumerate(iterArgs)) {
    if (!isSPMValue(iterArg))
      continue;
    mlir::Value yielded = yield.getOperand(index);
    if (yielded != iterArg && isSPMValue(yielded) &&
        llvm::is_contained(iterArgs, yielded))
      return true;
  }
  return false;
}

static bool instructionUsesOnlyF16Bf16ShapedValues(mlir::Operation *operation) {
  bool sawShaped = false;
  for (mlir::Value operand : operation->getOperands()) {
    auto shaped = mlir::dyn_cast<mlir::ShapedType>(operand.getType());
    if (!shaped)
      continue;
    sawShaped = true;
    mlir::Type elementType = shaped.getElementType();
    if (!elementType.isF16() && !mlir::isa<mlir::BFloat16Type>(elementType))
      return false;
  }
  return sawShaped;
}

static bool hasStaticPositiveTripCount(mlir::scf::ForOp loop) {
  Quantity tripCount = getTripCount(loop);
  return tripCount.knowledge == ScheduleCostKnowledge::Known &&
         tripCount.value > 0;
}

static bool
isQualifiedNCCOverlapWindow(mlir::scf::ForOp loop,
                            const TargetScheduleCostPolicy &policy) {
  if (policy.qualifiedOverlapFamilyMask == 0)
    return false;

  uint32_t familyMask = 0;
  for (mlir::Operation &operation : loop.getBody()->without_terminator()) {
    auto instruction = mlir::dyn_cast<WaferInstructionOpInterface>(&operation);
    if (!instruction)
      continue;
    NCCCompletionContract completion = getNCCCompletionContract(&operation);
    if (completion.behavior != LocalInstructionCompletion::OrderedPending ||
        !completion.issueWorker ||
        static_cast<uint32_t>(*completion.issueWorker) !=
            policy.qualifiedOverlapWorker ||
        !instructionUsesOnlyF16Bf16ShapedValues(&operation))
      return false;
    const uint32_t family =
        static_cast<uint32_t>(instruction.getInstructionFamily());
    if (family >= 32)
      return false;
    familyMask |= uint32_t{1} << family;
  }
  return familyMask == policy.qualifiedOverlapFamilyMask;
}

static bool
isDTEFootprintCompatibleWithOperation(mlir::Operation *operation,
                                      const StaticBufferRange &eventRange,
                                      bool eventWritesBuffer) {
  if (operation->getNumRegions() != 0)
    return false;
  if (mlir::isMemoryEffectFree(operation))
    return true;

  auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
  if (!effects)
    return false;
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 8> instances;
  effects.getEffects(instances);

  // Rootless resource effects describe an execution engine, not an operand
  // range. Every concrete memref operand must still have an operand-specific
  // effect so an omitted access cannot be mistaken for independence.
  for (mlir::Value operand : operation->getOperands()) {
    if (!mlir::isa<mlir::MemRefType>(operand.getType()))
      continue;
    if (llvm::none_of(instances, [&](const auto &effect) {
          return effect.getValue() == operand;
        }))
      return false;
  }

  for (const auto &effect : instances) {
    mlir::Value value = effect.getValue();
    if (!value || !mlir::isa<mlir::MemRefType>(value.getType()))
      continue;
    std::optional<StaticBufferRange> accessRange =
        resolveStaticBufferRange(value);
    if (!accessRange)
      return false;
    if (accessRange->root != eventRange.root ||
        staticByteRangesAreDisjoint(accessRange->bytes, eventRange.bytes))
      continue;
    const bool read = llvm::isa<mlir::MemoryEffects::Read>(effect.getEffect());
    // The Direct-DTE sender and compute may read the same bytes concurrently.
    // A receive owns a pending write, while any send-source write is a true
    // issue-to-wait hazard.
    if (eventWritesBuffer || !read)
      return false;
  }
  return true;
}

static bool
isQualifiedDirectDTEComputeWindow(mlir::Operation *issue,
                                  const TargetScheduleCostPolicy &policy) {
  if (policy.qualifiedDirectDTEOverlapFamilyMask == 0)
    return false;

  const uint32_t dteFamily = static_cast<uint32_t>(InstrFamily::DTE);
  if (dteFamily >= 32 || (policy.qualifiedDirectDTEOverlapFamilyMask &
                          (uint32_t{1} << dteFamily)) == 0)
    return false;

  mlir::Value token;
  mlir::Value buffer;
  bool eventWritesBuffer = false;
  if (auto send = mlir::dyn_cast<InstrDTESendOp>(issue)) {
    if (!send.getBinding())
      return false;
    token = send.getToken();
    buffer = send.getBuffer();
  } else if (auto recv = mlir::dyn_cast<InstrDTERecvOp>(issue)) {
    if (!recv.getBinding())
      return false;
    token = recv.getToken();
    buffer = recv.getBuffer();
    eventWritesBuffer = true;
  } else {
    return false;
  }
  if (!instructionUsesOnlyF16Bf16ShapedValues(issue) || !token.hasOneUse())
    return false;
  auto wait = mlir::dyn_cast<InstrDTEWaitOp>(*token.getUsers().begin());
  if (!wait || wait->getBlock() != issue->getBlock() ||
      !issue->isBeforeInBlock(wait))
    return false;
  std::optional<StaticBufferRange> eventRange =
      resolveStaticBufferRange(buffer);
  if (!eventRange)
    return false;

  bool hasQualifiedCompute = false;
  for (mlir::Operation *between = issue->getNextNode();
       between && between != wait.getOperation();
       between = between->getNextNode()) {
    if (!isDTEFootprintCompatibleWithOperation(between, *eventRange,
                                               eventWritesBuffer))
      return false;
    auto instruction = mlir::dyn_cast<WaferInstructionOpInterface>(between);
    if (!instruction || instruction.getInstructionFamily() == InstrFamily::DTE)
      continue;
    const uint32_t family =
        static_cast<uint32_t>(instruction.getInstructionFamily());
    hasQualifiedCompute |= family < 32 &&
                           (policy.qualifiedDirectDTEOverlapFamilyMask &
                            (uint32_t{1} << family)) != 0 &&
                           instructionUsesOnlyF16Bf16ShapedValues(between);
  }
  return hasQualifiedCompute;
}

} // namespace

void collectQualifiedOverlapWindows(mlir::Operation *root,
                                    InstructionProgramCost &cost,
                                    const TargetScheduleCostPolicy &policy) {
  bool unsupportedControlFlow = false;
  root->walk([&](mlir::Operation *operation) {
    if (mlir::isa<mlir::scf::IfOp, mlir::func::CallOp>(operation)) {
      unsupportedControlFlow = true;
      return;
    }
    if (mlir::isa<InstrDTESendOp, InstrDTERecvOp>(operation) &&
        isQualifiedDirectDTEComputeWindow(operation, policy))
      add(cost.directDTEComputeOverlapWindowCount, Quantity{1});

    auto loop = mlir::dyn_cast<mlir::scf::ForOp>(operation);
    if (!loop || !hasStaticPositiveTripCount(loop))
      return;
    if (!hasRotatingSPMState(loop))
      return;
    const bool ncc = isQualifiedNCCOverlapWindow(loop, policy);
    const bool directDTE = llvm::any_of(
        loop.getBody()->without_terminator(), [&](mlir::Operation &nested) {
          return mlir::isa<InstrDTESendOp, InstrDTERecvOp>(&nested) &&
                 isQualifiedDirectDTEComputeWindow(&nested, policy);
        });
    if (ncc || directDTE)
      add(cost.qualifiedOverlapWindowCount, Quantity{1});
  });
  if (unsupportedControlFlow) {
    degrade(cost.qualifiedOverlapWindowCount, ScheduleCostKnowledge::Unknown,
            ScheduleCostReason::UnsupportedControlFlow);
    degrade(cost.directDTEComputeOverlapWindowCount,
            ScheduleCostKnowledge::Unknown,
            ScheduleCostReason::UnsupportedControlFlow);
  }
}

} // namespace wafer::analysis::detail
