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
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <optional>

namespace wafer::analysis::detail {
namespace {

static Quantity getElementCount(mlir::Value value) {
  auto type = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  if (!type || !type.hasStaticShape())
    return Quantity::unavailable(
        ScheduleCostReason::UnavailablePhysicalGeometry);
  Quantity count{1};
  for (int64_t dim : type.getShape()) {
    if (dim < 0)
      return Quantity::unavailable(
          ScheduleCostReason::UnavailablePhysicalGeometry);
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

static ScalarClass classifyConvert(mlir::Type sourceType, mlir::Type destType) {
  ScalarClass source = classifyScalar(sourceType);
  ScalarClass dest = classifyScalar(destType);
  if (source == ScalarClass::Unsupported || dest == ScalarClass::Unsupported)
    return ScalarClass::Unsupported;
  // Conversion service is estimated at the slower modeled endpoint class.
  // This remains a typed point-model prior: integer and TF32-only pairs have
  // no calibrated vector rate and therefore stay in Other.
  if (source == ScalarClass::F32 || dest == ScalarClass::F32)
    return ScalarClass::F32;
  if (source == ScalarClass::F16Bf16 && dest == ScalarClass::F16Bf16)
    return ScalarClass::F16Bf16;
  return ScalarClass::Other;
}

static void addVectorCost(InstructionProgramCost &cost, ScalarClass scalarClass,
                          Quantity logicalOps, Quantity multiplicity) {
  Quantity total = multiply(logicalOps, multiplicity);
  switch (scalarClass) {
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

static void addVectorCost(InstructionProgramCost &cost, mlir::Type type,
                          Quantity logicalOps, Quantity multiplicity) {
  addVectorCost(cost, classifyScalar(type), logicalOps, multiplicity);
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

static void markDirectionalNoCUnavailable(InstructionProgramCost &cost) {
  for (ScheduleCostMetric &metric : cost.noc.directionalTransmitBytes)
    degrade(metric, ScheduleCostKnowledge::Unavailable,
            ScheduleCostReason::UnresolvedNoCRoute);
}

static void collectResourceCost(mlir::Operation *op,
                                InstructionProgramCost &cost,
                                Quantity multiplicity) {
  auto addBytes = [&](ScheduleCostMetric &metric, int64_t bytes) {
    if (bytes < 0) {
      degrade(metric, ScheduleCostKnowledge::Unavailable,
              ScheduleCostReason::UnavailableResourceBytes);
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
    addVectorCost(cost,
                  classifyConvert(convert.getSource().getType(),
                                  convert.getDest().getType()),
                  getElementCount(convert.getDest()), multiplicity);
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
    // Multiplication service follows the input format; F32 partial/output
    // storage does not change a low-precision GEMM into an F32-input GEMM.
    addNPUCost(cost, gemm.getLhs().getType(), logicalOps, multiplicity);
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
                InstrDTEBroadcastOp, InstrDTEScatterOp, InstrDTERecvOp,
                InstrDTEWaitOp>(op))
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
    Quantity bytes{static_cast<uint64_t>(send.getBytes())};
    Quantity total = multiply(bytes, multiplicity);
    add(cost.noc.aggregateTransmitBytes, total);
    markDirectionalNoCUnavailable(cost);
    return;
  }
  if (auto broadcast = mlir::dyn_cast<InstrDTEBroadcastOp>(op)) {
    Quantity bytes{static_cast<uint64_t>(broadcast.getBytes())};
    Quantity destinations{
        static_cast<uint64_t>(broadcast.getPeersAttr().size())};
    add(cost.noc.aggregateTransmitBytes,
        multiply(multiply(bytes, destinations), multiplicity));
    markDirectionalNoCUnavailable(cost);
    return;
  }
  if (auto scatter = mlir::dyn_cast<InstrDTEScatterOp>(op)) {
    Quantity bytes{static_cast<uint64_t>(scatter.getBytes())};
    Quantity destinations{static_cast<uint64_t>(scatter.getPeersAttr().size())};
    add(cost.noc.aggregateTransmitBytes,
        multiply(multiply(bytes, destinations), multiplicity));
    markDirectionalNoCUnavailable(cost);
    return;
  }
  if (auto recv = mlir::dyn_cast<InstrDTERecvOp>(op)) {
    Quantity bytes{static_cast<uint64_t>(recv.getBytes())};
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
      bool executable = isInstructionProgramOperation(next);
      next->walk([&](mlir::Operation *nested) {
        executable |= nested != next && isInstructionProgramOperation(nested);
      });
      if (executable)
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
  NCCOperationCompletion contract = getNCCOperationCompletion(op);
  bool isSteadyState =
      static_cast<bool>(op->getParentOfType<mlir::scf::ForOp>());
  bool isNonTerminal = hasFollowingExecutableWork(op);
  uint64_t participantWaits = countParticipants(contract.participantMask);
  if (contract.kind == NCCCompletionKind::ParticipantJoin) {
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
  if (contract.kind != NCCCompletionKind::SynchronousWriteback)
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
  if (mlir::isa<InstrDTESendOp, InstrDTEBroadcastOp, InstrDTEScatterOp>(op))
    addExecutionCount(work.dteSendOperations, multiplicity, 1, countStaticSite);
  if (mlir::isa<InstrDTERecvOp>(op))
    addExecutionCount(work.dteReceiveOperations, multiplicity, 1,
                      countStaticSite);
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

enum class ConstantIndexKnowledge { Known, Unavailable, Overflow };

struct ConstantIndex {
  ConstantIndexKnowledge knowledge = ConstantIndexKnowledge::Unavailable;
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
  if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    mlir::Block *owner = argument.getOwner();
    auto tileRegion =
        mlir::dyn_cast_or_null<TileRegionOp>(owner->getParentOp());
    unsigned index = argument.getArgNumber();
    if (tileRegion && owner == &tileRegion.getBody().front() &&
        index < tileRegion.getInputs().size())
      result =
          evaluateConstantIndex(tileRegion.getInputs()[index], activeValues);
  }
  if (result.knowledge == ConstantIndexKnowledge::Unavailable)
    if (auto opResult = mlir::dyn_cast<mlir::OpResult>(value)) {
      if (auto tileRegion = mlir::dyn_cast<TileRegionOp>(opResult.getOwner())) {
        auto yield = mlir::dyn_cast<TileYieldOp>(
            tileRegion.getBody().front().getTerminator());
        unsigned index = opResult.getResultNumber();
        if (yield && index < yield.getValues().size())
          result =
              evaluateConstantIndex(yield.getValues()[index], activeValues);
      }
    }
  if (result.knowledge == ConstantIndexKnowledge::Unavailable) {
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
  }
  if (result.knowledge == ConstantIndexKnowledge::Unavailable &&
      value.getType().isIndex()) {
    mlir::FailureOr<int64_t> constant =
        mlir::ValueBoundsConstraintSet::computeConstantBound(
            mlir::presburger::BoundType::EQ,
            mlir::ValueBoundsConstraintSet::Variable(value));
    if (mlir::succeeded(constant))
      result = {ConstantIndexKnowledge::Known, *constant};
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
    return Quantity::unavailable(ScheduleCostReason::DynamicLoopTripCount);
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
  enterUnavailableTripLoop(ExecutionMultiplicity multiplicity,
                           Quantity tripCount) {
    multiplicity.exact = multiply(multiplicity.exact, tripCount);
    multiplicity.lowerBound = Quantity{0};
    multiplicity.upperBound = multiply(multiplicity.upperBound, tripCount);
    return multiplicity;
  }

  static ExecutionMultiplicity
  enterConditionalBranch(ExecutionMultiplicity multiplicity) {
    multiplicity.exact = multiply(
        multiplicity.exact,
        Quantity::unavailable(ScheduleCostReason::ConditionalControlFlow));
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
                   enterUnavailableTripLoop(multiplicity, tripCount));
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

template <typename Callback>
static void forEachExactExecutionMetric(InstructionProgramCost &cost,
                                        Callback &&callback) {
  callback(cost.compute.npuF16Bf16LogicalOps);
  callback(cost.compute.npuOtherLogicalOps);
  callback(cost.compute.vectorF16Bf16LogicalOps);
  callback(cost.compute.vectorF32LogicalOps);
  callback(cost.compute.vectorOtherLogicalOps);
  callback(cost.ddrReadBytes);
  callback(cost.ddrWriteBytes);
  callback(cost.spmMovementBytes);
  callback(cost.gatherScatterBytes);
  callback(cost.noc.aggregateTransmitBytes);
  callback(cost.noc.aggregateReceiveBytes);
  callback(cost.noc.waitedEventCount);
  for (ScheduleCostMetric &metric : cost.noc.directionalTransmitBytes)
    callback(metric);
  for (InstructionWorkCountMember member : kInstructionWorkCountMembers)
    callback((cost.work.*member).exactExecutions);
}

template <typename Callback>
static void zipExactExecutionMetrics(InstructionProgramCost &result,
                                     const InstructionProgramCost &lhs,
                                     const InstructionProgramCost &rhs,
                                     Callback &&callback) {
  callback(result.compute.npuF16Bf16LogicalOps,
           lhs.compute.npuF16Bf16LogicalOps, rhs.compute.npuF16Bf16LogicalOps);
  callback(result.compute.npuOtherLogicalOps, lhs.compute.npuOtherLogicalOps,
           rhs.compute.npuOtherLogicalOps);
  callback(result.compute.vectorF16Bf16LogicalOps,
           lhs.compute.vectorF16Bf16LogicalOps,
           rhs.compute.vectorF16Bf16LogicalOps);
  callback(result.compute.vectorF32LogicalOps, lhs.compute.vectorF32LogicalOps,
           rhs.compute.vectorF32LogicalOps);
  callback(result.compute.vectorOtherLogicalOps,
           lhs.compute.vectorOtherLogicalOps,
           rhs.compute.vectorOtherLogicalOps);
  callback(result.ddrReadBytes, lhs.ddrReadBytes, rhs.ddrReadBytes);
  callback(result.ddrWriteBytes, lhs.ddrWriteBytes, rhs.ddrWriteBytes);
  callback(result.spmMovementBytes, lhs.spmMovementBytes, rhs.spmMovementBytes);
  callback(result.gatherScatterBytes, lhs.gatherScatterBytes,
           rhs.gatherScatterBytes);
  callback(result.noc.aggregateTransmitBytes, lhs.noc.aggregateTransmitBytes,
           rhs.noc.aggregateTransmitBytes);
  callback(result.noc.aggregateReceiveBytes, lhs.noc.aggregateReceiveBytes,
           rhs.noc.aggregateReceiveBytes);
  callback(result.noc.waitedEventCount, lhs.noc.waitedEventCount,
           rhs.noc.waitedEventCount);
  for (size_t index = 0; index < result.noc.directionalTransmitBytes.size();
       ++index)
    callback(result.noc.directionalTransmitBytes[index],
             lhs.noc.directionalTransmitBytes[index],
             rhs.noc.directionalTransmitBytes[index]);
  for (InstructionWorkCountMember member : kInstructionWorkCountMembers)
    callback((result.work.*member).exactExecutions,
             (lhs.work.*member).exactExecutions,
             (rhs.work.*member).exactExecutions);
}

static Quantity asQuantity(const ScheduleCostMetric &metric) {
  return {metric.value, metric.knowledge, metric.reason};
}

static void addExactExecutionCost(InstructionProgramCost &result,
                                  const InstructionProgramCost &increment) {
  zipExactExecutionMetrics(
      result, result, increment,
      [](ScheduleCostMetric &sum, const ScheduleCostMetric &,
         const ScheduleCostMetric &value) { add(sum, asQuantity(value)); });
}

static void scaleExactExecutionCost(InstructionProgramCost &cost,
                                    Quantity factor) {
  forEachExactExecutionMetric(cost, [&](ScheduleCostMetric &metric) {
    Quantity scaled = multiply(asQuantity(metric), factor);
    metric = {};
    add(metric, scaled);
  });
}

static ScheduleCostMetric
mergeConditionalExactMetric(const ScheduleCostMetric &thenMetric,
                            const ScheduleCostMetric &elseMetric) {
  if (thenMetric.value == elseMetric.value &&
      thenMetric.knowledge == elseMetric.knowledge &&
      thenMetric.reason == elseMetric.reason)
    return thenMetric;
  if (thenMetric.isKnown() && elseMetric.isKnown()) {
    ScheduleCostMetric result;
    degrade(result, ScheduleCostKnowledge::Unavailable,
            ScheduleCostReason::ConditionalControlFlow);
    return result;
  }

  const ScheduleCostMetric *selected = &thenMetric;
  if (getKnowledgeSeverity(elseMetric.knowledge) >
      getKnowledgeSeverity(thenMetric.knowledge))
    selected = &elseMetric;
  ScheduleCostMetric result;
  degrade(result, selected->knowledge, selected->reason);
  return result;
}

static InstructionProgramCost
mergeConditionalExactCost(const InstructionProgramCost &thenCost,
                          const InstructionProgramCost &elseCost) {
  InstructionProgramCost result;
  zipExactExecutionMetrics(
      result, thenCost, elseCost,
      [](ScheduleCostMetric &merged, const ScheduleCostMetric &thenMetric,
         const ScheduleCostMetric &elseMetric) {
        merged = mergeConditionalExactMetric(thenMetric, elseMetric);
      });
  return result;
}

static void markExactExecutionCostUnsupported(InstructionProgramCost &cost) {
  forEachExactExecutionMetric(cost, [](ScheduleCostMetric &metric) {
    degrade(metric, ScheduleCostKnowledge::Unsupported,
            ScheduleCostReason::UnsupportedControlFlow);
  });
}

/// Evaluates only dynamically executed cost dimensions. Unlike ProgramWalker,
/// this evaluator joins a dynamic `scf.if` dimension exactly when both
/// mutually exclusive paths contribute the same value to that dimension.
/// Static-site counts and conservative bounds remain owned by ProgramWalker.
class PathInvariantExactCostEvaluator {
public:
  explicit PathInvariantExactCostEvaluator(
      llvm::function_ref<bool(mlir::Operation *)> includeOperation)
      : includeOperation(includeOperation) {}

  InstructionProgramCost evaluateRoot(mlir::Operation *root) {
    InstructionProgramCost result;
    if (auto module = mlir::dyn_cast<mlir::ModuleOp>(root)) {
      for (mlir::func::FuncOp function : module.getOps<mlir::func::FuncOp>()) {
        if (function.isPrivate())
          continue;
        addExactExecutionCost(result, evaluateFunction(function));
      }
      return result;
    }
    if (auto function = mlir::dyn_cast<mlir::func::FuncOp>(root)) {
      return evaluateFunction(function);
    }
    evaluateOperation(root, result);
    return result;
  }

private:
  InstructionProgramCost evaluateRegion(mlir::Region &region) {
    InstructionProgramCost result;
    bool unsupported = !region.empty() && !region.hasOneBlock();
    for (mlir::Block &block : region)
      for (mlir::Operation &operation : block)
        evaluateOperation(&operation, result);
    if (unsupported)
      scaleExactExecutionCost(
          result,
          Quantity::unsupported(ScheduleCostReason::UnsupportedControlFlow));
    return result;
  }

  void evaluateOperation(mlir::Operation *operation,
                         InstructionProgramCost &result) {
    if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(operation)) {
      InstructionProgramCost body = evaluateRegion(loop.getRegion());
      scaleExactExecutionCost(body, getTripCount(loop));
      addExactExecutionCost(result, body);
      return;
    }
    if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(operation)) {
      std::optional<int64_t> condition = mlir::getConstantIntValue(
          mlir::getAsOpFoldResult(ifOp.getCondition()));
      if (condition) {
        mlir::Region &selected =
            *condition ? ifOp.getThenRegion() : ifOp.getElseRegion();
        addExactExecutionCost(result, evaluateRegion(selected));
        return;
      }
      InstructionProgramCost thenCost = evaluateRegion(ifOp.getThenRegion());
      InstructionProgramCost elseCost = evaluateRegion(ifOp.getElseRegion());
      addExactExecutionCost(result,
                            mergeConditionalExactCost(thenCost, elseCost));
      return;
    }
    if (isInstructionProgramOperation(operation)) {
      if (!includeOperation(operation))
        return;
      InstructionProgramCost instruction;
      collectInstructionCost(operation, instruction, ExecutionMultiplicity{},
                             /*countStaticSite=*/false);
      addExactExecutionCost(result, instruction);
      return;
    }
    if (auto call = mlir::dyn_cast<mlir::func::CallOp>(operation)) {
      mlir::func::FuncOp callee =
          mlir::SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(
              call, call.getCalleeAttr());
      if (!callee || callee.isDeclaration()) {
        markExactExecutionCostUnsupported(result);
        return;
      }
      addExactExecutionCost(result, evaluateFunction(callee));
      return;
    }
    if (operation->getNumRegions() == 0)
      return;
    if (mlir::isa<mlir::ModuleOp, mlir::func::FuncOp, TileRegionOp>(
            operation)) {
      for (mlir::Region &region : operation->getRegions())
        addExactExecutionCost(result, evaluateRegion(region));
      return;
    }

    InstructionProgramCost nested;
    for (mlir::Region &region : operation->getRegions())
      addExactExecutionCost(nested, evaluateRegion(region));
    scaleExactExecutionCost(
        nested,
        Quantity::unsupported(ScheduleCostReason::UnsupportedControlFlow));
    addExactExecutionCost(result, nested);
  }

  InstructionProgramCost evaluateFunction(mlir::func::FuncOp function) {
    auto cached = functionCosts.find(function.getOperation());
    if (cached != functionCosts.end())
      return cached->second;

    InstructionProgramCost result;
    if (!activeFunctions.insert(function.getOperation()).second) {
      markExactExecutionCostUnsupported(result);
      return result;
    }
    result = evaluateRegion(function.getBody());
    activeFunctions.erase(function.getOperation());
    functionCosts.try_emplace(function.getOperation(), result);
    return result;
  }

  llvm::DenseSet<mlir::Operation *> activeFunctions;
  llvm::DenseMap<mlir::Operation *, InstructionProgramCost> functionCosts;
  llvm::function_ref<bool(mlir::Operation *)> includeOperation;
};

static void refinePathInvariantExactCost(
    mlir::Operation *root, InstructionProgramCost &cost,
    llvm::function_ref<bool(mlir::Operation *)> includeOperation) {
  InstructionProgramCost exact =
      PathInvariantExactCostEvaluator(includeOperation).evaluateRoot(root);
  zipExactExecutionMetrics(
      cost, cost, exact,
      [](ScheduleCostMetric &result, const ScheduleCostMetric &,
         const ScheduleCostMetric &refined) { result = refined; });

  // A known exact execution count is also the tight lower and upper bound.
  // Keep branch-local static sites distinct so the IR surface remains
  // auditable even when all runtime paths have equal cost.
  for (InstructionWorkCountMember member : kInstructionWorkCountMembers) {
    InstructionExecutionCount &count = cost.work.*member;
    if (!count.exactExecutions.isKnown())
      continue;
    count.lowerBound = count.exactExecutions;
    count.upperBound = count.exactExecutions;
  }
}

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

void collectExecutionCost(
    mlir::Operation *root, InstructionProgramCost &cost,
    llvm::function_ref<bool(mlir::Operation *)> includeOperation) {
  llvm::DenseSet<mlir::Operation *> countedStaticSites;
  auto collect = [&](mlir::Operation *operation,
                     ExecutionMultiplicity multiplicity) {
    if (!includeOperation(operation))
      return;
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
  refinePathInvariantExactCost(root, cost, includeOperation);

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

void collectExecutionCost(mlir::Operation *root, InstructionProgramCost &cost) {
  collectExecutionCost(root, cost, [](mlir::Operation *) { return true; });
}

} // namespace wafer::analysis::detail
