//===- ExecutionCost.cpp - Loop-aware instruction costs --------*- C++ -*-===//

#include "Internal.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Async/IR/AsyncTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

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

static NoCCollectiveKind classifyCollective(DTEProtocolPhase phase) {
  switch (phase) {
  case DTEProtocolPhase::CollectivePermute:
    return NoCCollectiveKind::CollectivePermute;
  case DTEProtocolPhase::AllToAll:
    return NoCCollectiveKind::AllToAll;
  case DTEProtocolPhase::AllGatherDirect:
  case DTEProtocolPhase::AllGatherRing:
    return NoCCollectiveKind::AllGather;
  case DTEProtocolPhase::ReduceScatterDirect:
    return NoCCollectiveKind::ReduceScatter;
  case DTEProtocolPhase::AllReduceRing:
  case DTEProtocolPhase::AllReduceTreeReduce:
  case DTEProtocolPhase::AllReduceTreeBroadcast:
    return NoCCollectiveKind::AllReduce;
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
    Quantity bytes =
        send.getBytes() < 0
            ? Quantity::unsupported(
                  ScheduleCostReason::UnsupportedInstructionSemantics)
            : Quantity{static_cast<uint64_t>(send.getBytes())};
    Quantity total = multiply(bytes, multiplicity);
    add(cost.noc.aggregateTransmitBytes, total);
    NoCCollectiveKind kind = classifyCollective(send.getMessage().getPhase());
    add(cost.noc.collectiveTransmitBytes[static_cast<size_t>(kind)], total);
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
  }
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
}

static Quantity getTripCount(mlir::scf::ForOp loop) {
  std::optional<int64_t> lower =
      mlir::getConstantIntValue(mlir::getAsOpFoldResult(loop.getLowerBound()));
  std::optional<int64_t> upper =
      mlir::getConstantIntValue(mlir::getAsOpFoldResult(loop.getUpperBound()));
  std::optional<int64_t> step =
      mlir::getConstantIntValue(mlir::getAsOpFoldResult(loop.getStep()));
  if (!lower || !upper || !step)
    return Quantity::unknown(ScheduleCostReason::DynamicLoopTripCount);
  if (*step <= 0)
    return Quantity::unsupported(ScheduleCostReason::InvalidLoopStep);
  if (*upper <= *lower)
    return Quantity{0};
  uint64_t distance =
      static_cast<uint64_t>(*upper) - static_cast<uint64_t>(*lower);
  uint64_t positiveStep = static_cast<uint64_t>(*step);
  return Quantity{distance / positiveStep +
                  static_cast<uint64_t>(distance % positiveStep != 0)};
}

class ProgramWalker {
public:
  explicit ProgramWalker(InstructionProgramCost &cost) : cost(cost) {}

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

    if (mlir::isa<WaferInstructionOpInterface, SyncLocalFenceOp>(op)) {
      collectInstructionCost(op, cost, multiplicity);
      return;
    }
    if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) {
      mlir::func::FuncOp callee =
          mlir::SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(
              call, call.getCalleeAttr());
      if (!callee || callee.isDeclaration() ||
          !activeFunctions.insert(callee.getOperation()).second) {
        markAllExecutionMetricsUnsupported();
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

  void markAllExecutionMetricsUnsupported() {
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
    mark(cost.noc.aggregateTransmitBytes);
    mark(cost.noc.aggregateReceiveBytes);
    for (ScheduleCostMetric &metric : cost.noc.directionalTransmitBytes)
      mark(metric);
    for (ScheduleCostMetric &metric : cost.noc.collectiveTransmitBytes)
      mark(metric);
    mark(cost.instructionCount);
    mark(cost.eventCount);
  }

  InstructionProgramCost &cost;
  llvm::DenseSet<mlir::Operation *> activeFunctions;
};

} // namespace

void collectExecutionCost(mlir::Operation *root, InstructionProgramCost &cost) {
  ProgramWalker(cost).walkRoot(root);
}

} // namespace wafer::analysis::detail
