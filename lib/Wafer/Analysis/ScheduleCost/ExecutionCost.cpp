//===- ExecutionCost.cpp - Loop-aware instruction costs --------*- C++ -*-===//

#include "Internal.h"

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
    add(cost.noc.staticIssueSiteCount, Quantity{1});
    Quantity bytes =
        send.getBytes() < 0
            ? Quantity::unsupported(
                  ScheduleCostReason::UnsupportedInstructionSemantics)
            : Quantity{static_cast<uint64_t>(send.getBytes())};
    Quantity total = multiply(bytes, multiplicity);
    add(cost.noc.aggregateTransmitBytes, total);
    add(cost.noc.transmitMessageCount, multiplicity);
    std::optional<NoCCollectiveKind> kind =
        classifyCollective(send.getMessage().getPhase());
    if (kind)
      add(cost.noc.collectiveTransmitBytes[static_cast<size_t>(*kind)], total);
    markDirectionalNoCUnknown(cost);
    return;
  }
  if (auto recv = mlir::dyn_cast<InstrDTERecvOp>(op)) {
    add(cost.noc.staticIssueSiteCount, Quantity{1});
    Quantity bytes =
        recv.getBytes() < 0
            ? Quantity::unsupported(
                  ScheduleCostReason::UnsupportedInstructionSemantics)
            : Quantity{static_cast<uint64_t>(recv.getBytes())};
    add(cost.noc.aggregateReceiveBytes, multiply(bytes, multiplicity));
    add(cost.noc.receiveMessageCount, multiplicity);
    return;
  }
  if (auto wait = mlir::dyn_cast<InstrDTEWaitOp>(op)) {
    add(cost.noc.waitOperationCount, multiplicity);
    add(cost.noc.waitedEventCount,
        multiply(Quantity{wait.getTokens().size()}, multiplicity));
  }
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
  return mlir::isa<WaferInstructionOpInterface, SyncLocalFenceOp,
                   SyncNCCJoinOp>(op);
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

static void collectNCCDrainCost(mlir::Operation *op,
                                InstructionProgramCost &cost,
                                Quantity multiplicity) {
  NCCCompletionContract contract = getNCCCompletionContract(op);
  bool isSteadyState =
      static_cast<bool>(op->getParentOfType<mlir::scf::ForOp>());
  bool isNonTerminal = hasFollowingExecutableWork(op);
  Quantity participantWaits = multiply(
      Quantity{countParticipants(contract.participantMask)}, multiplicity);
  if (contract.behavior == LocalInstructionCompletion::ParticipantJoin) {
    add(cost.nccJoinCount, multiplicity);
    if (isSteadyState)
      add(cost.steadyStateNCCJoinCount, multiplicity);
    if (isNonTerminal)
      add(cost.nonTerminalNCCJoinCount, multiplicity);
    add(cost.nccParticipantWaitCount, participantWaits);
    if (isSteadyState)
      add(cost.steadyStateNCCParticipantWaitCount, participantWaits);
    if (isNonTerminal)
      add(cost.nonTerminalNCCParticipantWaitCount, participantWaits);
    return;
  }
  if (contract.behavior != LocalInstructionCompletion::SynchronousWriteback)
    return;
  add(cost.intrinsicNCCDrainCount, multiplicity);
  add(cost.nccParticipantWaitCount, participantWaits);
  if (isSteadyState)
    add(cost.steadyStateNCCParticipantWaitCount, participantWaits);
  if (isNonTerminal)
    add(cost.nonTerminalNCCParticipantWaitCount, participantWaits);
}

static void collectInstructionCost(mlir::Operation *op,
                                   InstructionProgramCost &cost,
                                   Quantity multiplicity) {
  if (multiplicity.knowledge == ScheduleCostKnowledge::Known &&
      multiplicity.value == 0)
    return;
  add(cost.instructionCount, multiply(Quantity{1}, multiplicity));
  for (mlir::Value result : op->getResults()) {
    if (mlir::isa<mlir::async::TokenType>(result.getType()))
      add(cost.eventCount, multiply(Quantity{1}, multiplicity));
  }
  collectResourceCost(op, cost, multiplicity);
  collectComputeCost(op, cost, multiplicity);
  collectNoCCost(op, cost, multiplicity);
  collectNCCDrainCost(op, cost, multiplicity);
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
      llvm::function_ref<void(mlir::Operation *, Quantity)> onInstruction,
      llvm::function_ref<void()> onUnsupportedControlFlow)
      : onInstruction(onInstruction),
        onUnsupportedControlFlow(onUnsupportedControlFlow) {}

  void walkRoot(mlir::Operation *root) {
    if (mlir::isa<mlir::func::FuncOp>(root))
      activeFunctions.insert(root);
    walkOperation(root, Quantity{1});
  }

private:
  void walkRegions(mlir::Operation *op, Quantity multiplicity) {
    for (mlir::Region &region : op->getRegions())
      walkRegion(region, multiplicity);
  }

  void walkRegion(mlir::Region &region, Quantity multiplicity) {
    if (!region.empty() && !region.hasOneBlock())
      multiplicity = multiply(
          multiplicity,
          Quantity::unsupported(ScheduleCostReason::UnsupportedControlFlow));
    for (mlir::Block &block : region)
      for (mlir::Operation &op : block)
        walkOperation(&op, multiplicity);
  }

  void walkOperation(mlir::Operation *op, Quantity multiplicity) {
    if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
      walkRegion(loop.getRegion(), multiply(multiplicity, getTripCount(loop)));
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
        Quantity conditional =
            Quantity::unknown(ScheduleCostReason::ConditionalControlFlow);
        walkRegion(ifOp.getThenRegion(), multiply(multiplicity, conditional));
        walkRegion(ifOp.getElseRegion(), multiply(multiplicity, conditional));
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

    walkRegions(op, multiply(multiplicity,
                             Quantity::unsupported(
                                 ScheduleCostReason::UnsupportedControlFlow)));
  }

  llvm::function_ref<void(mlir::Operation *, Quantity)> onInstruction;
  llvm::function_ref<void()> onUnsupportedControlFlow;
  llvm::DenseSet<mlir::Operation *> activeFunctions;
};

} // namespace

void walkInstructionProgram(
    mlir::Operation *root,
    llvm::function_ref<void(mlir::Operation *, Quantity)> onInstruction,
    llvm::function_ref<void()> onUnsupportedControlFlow) {
  ProgramWalker(onInstruction, onUnsupportedControlFlow).walkRoot(root);
}

void collectExecutionCost(mlir::Operation *root, InstructionProgramCost &cost) {
  auto collect = [&](mlir::Operation *operation, Quantity multiplicity) {
    collectInstructionCost(operation, cost, multiplicity);
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
    mark(cost.instructionCount);
    mark(cost.eventCount);
    mark(cost.nccJoinCount);
    mark(cost.steadyStateNCCJoinCount);
    mark(cost.nonTerminalNCCJoinCount);
    mark(cost.nccParticipantWaitCount);
    mark(cost.steadyStateNCCParticipantWaitCount);
    mark(cost.nonTerminalNCCParticipantWaitCount);
    mark(cost.intrinsicNCCDrainCount);
  };
  walkInstructionProgram(root, collect, markAllUnsupported);
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

} // namespace wafer::analysis::detail
