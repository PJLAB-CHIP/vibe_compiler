//===- CostModel.cpp - Instruction program performance model ----------===//

#include "Wafer/Analysis/Instr/CostModel.h"
#include "Wafer/Analysis/ControlFlow/SingleExecutionRegionFlow.h"
#include "Wafer/IR/NCCCompletion.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/CompileTiming.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

namespace wafer::analysis {
namespace {

bool checkedAdd(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (rhs > std::numeric_limits<uint64_t>::max() - lhs)
    return false;
  result = lhs + rhs;
  return true;
}

bool checkedMultiply(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

std::optional<uint64_t> timeForWork(uint64_t work, uint64_t rate) {
  if (rate == 0)
    return std::nullopt;
  if (work == 0)
    return uint64_t{0};
  constexpr uint64_t picosecondsPerSecond = UINT64_C(1000000000000);
  const unsigned __int128 numerator =
      static_cast<unsigned __int128>(work) * picosecondsPerSecond;
  const unsigned __int128 duration =
      (numerator + static_cast<unsigned __int128>(rate) - 1) / rate;
  if (duration > std::numeric_limits<uint64_t>::max())
    return std::nullopt;
  return static_cast<uint64_t>(duration);
}

template <typename Accessor>
std::optional<uint64_t>
maximumTileMetric(const InstructionProgramAggregateCost &cost,
                  Accessor accessor, const ScheduleCostMetric &aggregate) {
  if (cost.tileCosts.empty())
    return aggregate.isKnown() ? std::optional<uint64_t>(aggregate.value)
                               : std::nullopt;
  uint64_t maximum = 0;
  for (const InstructionProgramCost &tile : cost.tileCosts) {
    const ScheduleCostMetric &metric = accessor(tile);
    if (!metric.isKnown())
      return std::nullopt;
    maximum = std::max(maximum, metric.value);
  }
  return maximum;
}

std::variant<uint64_t, SearchObjectiveUnknownReason>
deriveNCCControlTime(const ScheduleCostMetric &joins,
                     const ScheduleCostMetric &participants,
                     const SearchCostPolicy &policy) {
  if (!joins.isKnown() || !participants.isKnown())
    return SearchObjectiveUnknownReason::MetricUnavailable;
  uint64_t calls = 0, waits = 0, total = 0;
  if (!checkedMultiply(joins.value, policy.nccJoinPicosecondsEstimate, calls) ||
      !checkedMultiply(participants.value,
                       policy.nccParticipantWaitPicosecondsEstimate, waits) ||
      !checkedAdd(calls, waits, total))
    return SearchObjectiveUnknownReason::ArithmeticOverflow;
  return total;
}

std::array<uint64_t, 4>
asStorageArray(const SearchResourceDurations &durations) {
  return {durations.spmHighWaterBytes, durations.ddrHighWaterBytes,
          durations.spmBufferCount, durations.ddrBufferCount};
}

std::optional<uint64_t> movementServiceTime(
    const ScheduleCostMetric &allBytes, const ScheduleCostMetric &gsBytes,
    const ScheduleCostMetric &innerIterations, const SearchCostPolicy &policy) {
  if (!allBytes.isKnown() || !gsBytes.isKnown() || !innerIterations.isKnown() ||
      gsBytes.value > allBytes.value)
    return std::nullopt;
  auto other =
      timeForWork(allBytes.value - gsBytes.value,
                  policy.spmExplicitMovementBytesPerSecondPerTileEstimate);
  auto bytes = timeForWork(
      gsBytes.value, policy.spmExplicitMovementBytesPerSecondPerTileEstimate);
  uint64_t traversal, total;
  if (!other || !bytes ||
      !checkedMultiply(innerIterations.value,
                       policy.gatherScatterInnerIterationPicosecondsEstimate,
                       traversal) ||
      !checkedAdd(*other, std::max(*bytes, traversal), total))
    return std::numeric_limits<uint64_t>::max();
  return total;
}

} // namespace

mlir::FailureOr<SearchCostCohort>
SearchCostCohort::create(const SearchCostPolicy &policy,
                         std::string *failureReason) {
  const std::array<uint64_t, 16> rates{
      policy.unmodeledInstructionPicosecondsEstimate,
      policy.ddrNominalBytesPerSecond,
      policy.directionalNoCBytesPerSecond,
      policy.dteEndpointBytesPerSecondEstimate,
      policy.dteFirstMessagePicosecondsEstimate,
      policy.dteMessageStartupPicosecondsEstimate,
      policy.noCHopPicosecondsEstimate,
      policy.instructionFixedPicosecondsEstimate,
      policy.gatherScatterInnerIterationPicosecondsEstimate,
      policy.dteWaitedEventPicosecondsEstimate,
      policy.nccJoinPicosecondsEstimate,
      policy.nccParticipantWaitPicosecondsEstimate,
      policy.f16Bf16NPULogicalOpsPerSecondPerTile,
      policy.f16Bf16VectorLogicalOpsPerSecondPerTile,
      policy.f32VectorLogicalOpsPerSecondPerTile,
      policy.spmExplicitMovementBytesPerSecondPerTileEstimate};
  if (policy.profileIdentity == 0 || llvm::is_contained(rates, uint64_t{0})) {
    if (failureReason)
      *failureReason = policy.profileIdentity == 0
                           ? "search cost cohort requires a nonzero profile "
                             "identity"
                           : "search cost cohort requires positive rates for "
                             "every enabled resource term";
    return mlir::failure();
  }
  return SearchCostCohort(policy);
}

namespace {

SearchObjective
deriveResourceObjective(const InstructionProgramAggregateCost &cost,
                        const std::optional<SearchCostCohort> &cohort) {
  if (!cohort)
    return UnknownSearchObjective{SearchObjectiveUnknownReason::NoCohort};
  const SearchCostPolicy &policy = cohort->getPolicy();
  if (!cost.aggregateCompute.npuOtherLogicalOps.isKnown() ||
      !cost.aggregateCompute.vectorOtherLogicalOps.isKnown())
    return UnknownSearchObjective{
        SearchObjectiveUnknownReason::MetricUnavailable};
  if (cost.aggregateCompute.npuOtherLogicalOps.value != 0 ||
      cost.aggregateCompute.vectorOtherLogicalOps.value != 0)
    return UnknownSearchObjective{
        SearchObjectiveUnknownReason::UncalibratedWork};

  auto npu = maximumTileMetric(
      cost,
      [](const InstructionProgramCost &tile) -> const ScheduleCostMetric & {
        return tile.compute.npuF16Bf16LogicalOps;
      },
      cost.aggregateCompute.npuF16Bf16LogicalOps);
  auto vectorF16 = maximumTileMetric(
      cost,
      [](const InstructionProgramCost &tile) -> const ScheduleCostMetric & {
        return tile.compute.vectorF16Bf16LogicalOps;
      },
      cost.aggregateCompute.vectorF16Bf16LogicalOps);
  auto vectorF32 = maximumTileMetric(
      cost,
      [](const InstructionProgramCost &tile) -> const ScheduleCostMetric & {
        return tile.compute.vectorF32LogicalOps;
      },
      cost.aggregateCompute.vectorF32LogicalOps);
  auto spm = maximumTileMetric(
      cost,
      [](const InstructionProgramCost &tile) -> const ScheduleCostMetric & {
        return tile.spmMovementBytes;
      },
      cost.aggregateSPMMovementBytes);
  auto instructions = maximumTileMetric(
      cost,
      [](const InstructionProgramCost &tile) -> const ScheduleCostMetric & {
        return tile.instructionCount;
      },
      cost.aggregateInstructionCount);
  auto dteWaits = maximumTileMetric(
      cost,
      [](const InstructionProgramCost &tile) -> const ScheduleCostMetric & {
        return tile.noc.waitedEventCount;
      },
      cost.aggregateNoC.waitedEventCount);
  if (!npu || !vectorF16 || !vectorF32 || !spm || !instructions || !dteWaits ||
      !cost.aggregateDDRReadBytes.isKnown() ||
      !cost.aggregateDDRWriteBytes.isKnown() ||
      !cost.aggregateNoC.staticIssueSiteCount.isKnown() ||
      !cost.maximumTileNoCTransmitBytes.isKnown() ||
      !cost.maximumTileNoCTransmitMessageCount.isKnown() ||
      !cost.minimumHopMessageDemand.isKnown() ||
      !cost.maximumTileSPMHighWaterBytes.isKnown() ||
      !cost.maximumTileDDRHighWaterBytes.isKnown() ||
      !cost.aggregateCompilerOwnedSPMBufferCount.isKnown() ||
      !cost.aggregateCompilerOwnedDDRBufferCount.isKnown())
    return UnknownSearchObjective{
        SearchObjectiveUnknownReason::MetricUnavailable};

  uint64_t ddrBytes = 0;
  if (!checkedAdd(cost.aggregateDDRReadBytes.value,
                  cost.aggregateDDRWriteBytes.value, ddrBytes))
    return UnknownSearchObjective{
        SearchObjectiveUnknownReason::ArithmeticOverflow};
  uint64_t nocBytes = 0;
  if (cost.aggregateNoC.staticIssueSiteCount.value != 0) {
    if (!cost.modeledNoCRoute.peakDirectedLinkByteDemand.isKnown())
      return UnknownSearchObjective{
          SearchObjectiveUnknownReason::MetricUnavailable};
    nocBytes = cost.modeledNoCRoute.peakDirectedLinkByteDemand.value;
  }

  const uint64_t dteEndpointBytes = cost.maximumTileNoCTransmitBytes.value;
  const uint64_t dteMessageCount =
      cost.maximumTileNoCTransmitMessageCount.value;
  const uint64_t hopMessageDemand = cost.minimumHopMessageDemand.value;

  SearchResourceDurations durations;
  if (cost.tileCosts.empty()) {
    auto control =
        deriveNCCControlTime(cost.aggregateNCCJoinCount,
                             cost.aggregateNCCParticipantWaitCount, policy);
    if (const auto *reason =
            std::get_if<SearchObjectiveUnknownReason>(&control))
      return UnknownSearchObjective{*reason};
    durations.nccWaitControlPicoseconds = std::get<uint64_t>(control);
  } else {
    // Join and participant maxima need not belong to the same Tile.
    for (const InstructionProgramCost &tile : cost.tileCosts) {
      auto control = deriveNCCControlTime(tile.nccJoinCount,
                                          tile.nccParticipantWaitCount, policy);
      if (const auto *reason =
              std::get_if<SearchObjectiveUnknownReason>(&control))
        return UnknownSearchObjective{*reason};
      durations.nccWaitControlPicoseconds = std::max(
          durations.nccWaitControlPicoseconds, std::get<uint64_t>(control));
    }
  }
  if (dteMessageCount != 0 &&
      (!checkedMultiply(dteMessageCount - 1,
                        policy.dteMessageStartupPicosecondsEstimate,
                        durations.dteStartupPicoseconds) ||
       !checkedAdd(durations.dteStartupPicoseconds,
                   policy.dteFirstMessagePicosecondsEstimate,
                   durations.dteStartupPicoseconds)))
    return UnknownSearchObjective{
        SearchObjectiveUnknownReason::ArithmeticOverflow};
  auto assignTime = [&](uint64_t work, uint64_t rate, uint64_t &destination) {
    std::optional<uint64_t> duration = timeForWork(work, rate);
    if (!duration)
      return false;
    destination = *duration;
    return true;
  };
  if (!assignTime(*npu, policy.f16Bf16NPULogicalOpsPerSecondPerTile,
                  durations.neF16Bf16Picoseconds) ||
      !assignTime(*vectorF16, policy.f16Bf16VectorLogicalOpsPerSecondPerTile,
                  durations.vectorF16Bf16Picoseconds) ||
      !assignTime(*vectorF32, policy.f32VectorLogicalOpsPerSecondPerTile,
                  durations.vectorF32Picoseconds) ||
      !assignTime(ddrBytes, policy.ddrNominalBytesPerSecond,
                  durations.ddrPicoseconds) ||
      !assignTime(nocBytes, policy.directionalNoCBytesPerSecond,
                  durations.nocPicoseconds) ||
      !assignTime(dteEndpointBytes, policy.dteEndpointBytesPerSecondEstimate,
                  durations.dteEndpointPicoseconds) ||
      !checkedMultiply(hopMessageDemand, policy.noCHopPicosecondsEstimate,
                       durations.nocHopPicoseconds) ||
      !checkedMultiply(*instructions,
                       policy.instructionFixedPicosecondsEstimate,
                       durations.instructionControlPicoseconds) ||
      !checkedMultiply(*dteWaits, policy.dteWaitedEventPicosecondsEstimate,
                       durations.dteWaitControlPicoseconds))
    return UnknownSearchObjective{
        SearchObjectiveUnknownReason::ArithmeticOverflow};
  auto assignMovement = [&](const auto &bytes, const auto &gs,
                            const auto &iterations) {
    auto duration = movementServiceTime(bytes, gs, iterations, policy);
    if (!duration)
      return false;
    durations.spmMovementPicoseconds =
        std::max(durations.spmMovementPicoseconds, *duration);
    return true;
  };
  if (cost.tileCosts.empty()) {
    if (!assignMovement(cost.aggregateSPMMovementBytes,
                        cost.aggregateGatherScatterBytes,
                        cost.aggregateGatherScatterInnerIterations))
      return UnknownSearchObjective{
          SearchObjectiveUnknownReason::MetricUnavailable};
  } else {
    for (const auto &tile : cost.tileCosts)
      if (!assignMovement(tile.spmMovementBytes, tile.gatherScatterBytes,
                          tile.gatherScatterInnerIterations))
        return UnknownSearchObjective{
            SearchObjectiveUnknownReason::MetricUnavailable};
  }
  durations.spmHighWaterBytes = cost.maximumTileSPMHighWaterBytes.value;
  durations.ddrHighWaterBytes = cost.maximumTileDDRHighWaterBytes.value;
  durations.spmBufferCount = cost.aggregateCompilerOwnedSPMBufferCount.value;
  durations.ddrBufferCount = cost.aggregateCompilerOwnedDDRBufferCount.value;
  return KnownSearchObjective{durations, *cohort};
}

// A performance assumption, not a reconstructed schedule: sum service on each
// Tile before taking the maximum. Never assemble one Tile from unrelated peaks.
std::optional<uint64_t> localServiceTime(const InstructionProgramCost &tile,
                                         const SearchCostPolicy &policy,
                                         bool includeIssue = true) {
  auto movement =
      movementServiceTime(tile.spmMovementBytes, tile.gatherScatterBytes,
                          tile.gatherScatterInnerIterations, policy);
  if (!movement)
    return std::nullopt;
  uint64_t total = *movement;
  auto addWork = [&](uint64_t work, uint64_t rate) {
    auto duration = timeForWork(work, rate);
    return duration && checkedAdd(total, *duration, total);
  };
  auto addFixed = [&](uint64_t count, uint64_t duration) {
    uint64_t term;
    return checkedMultiply(count, duration, term) &&
           checkedAdd(total, term, total);
  };
  if (!addWork(tile.compute.npuF16Bf16LogicalOps.value,
               policy.f16Bf16NPULogicalOpsPerSecondPerTile) ||
      !addWork(tile.compute.vectorF16Bf16LogicalOps.value,
               policy.f16Bf16VectorLogicalOpsPerSecondPerTile) ||
      !addWork(tile.compute.vectorF32LogicalOps.value,
               policy.f32VectorLogicalOpsPerSecondPerTile) ||
      !addWork(tile.noc.aggregateTransmitBytes.value,
               policy.dteEndpointBytesPerSecondEstimate) ||
      (includeIssue && !addFixed(tile.instructionCount.value,
                                 policy.instructionFixedPicosecondsEstimate)) ||
      !addFixed(tile.noc.waitedEventCount.value,
                policy.dteWaitedEventPicosecondsEstimate) ||
      !addFixed(tile.nccJoinCount.value, policy.nccJoinPicosecondsEstimate) ||
      !addFixed(tile.nccParticipantWaitCount.value,
                policy.nccParticipantWaitPicosecondsEstimate))
    return std::nullopt;
  uint64_t messages = tile.noc.transmitMessageCount.value;
  if (messages &&
      (!addFixed(1, policy.dteFirstMessagePicosecondsEstimate) ||
       !addFixed(messages - 1, policy.dteMessageStartupPicosecondsEstimate)))
    return std::nullopt;
  return total;
}

uint64_t estimateInstructionCount(const InstructionExecutionCount &work,
                                  const ScheduleCostMetric &count) {
  if (count.isKnown())
    return count.value;
  if (work.upperBound.isKnown())
    return work.upperBound.value;
  // Static sites are an explicit one-visit estimate, not a dynamic work fact.
  return work.staticSites.isKnown() ? work.staticSites.value : 1;
}

// Current-IR endpoints and estimates live only for this immutable comparison.
// Repeated dynamic messages are handled locally rather than merged by name.
struct CompletionTimes {
  llvm::DenseMap<int64_t, uint64_t> publications;
  llvm::DenseMap<mlir::Operation *, mlir::Operation *> sendForReceive;
  llvm::DenseMap<mlir::Operation *, mlir::Operation *> receiveForSend;
  llvm::DenseMap<mlir::Operation *, uint64_t> sendFinished, receivePosted;
  bool changed = false;

  template <typename Map, typename Key>
  void update(Map &map, Key key, uint64_t value) {
    auto [entry, inserted] = map.try_emplace(key, value);
    changed |= inserted || entry->second != value;
    entry->second = value;
  }
};

bool executesOnce(mlir::Operation *operation) {
  for (auto *parent = operation->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (mlir::isa<mlir::func::FuncOp>(parent))
      return true;
    if (!getSingleExecutionRegionFlow(parent))
      return false;
  }
  return false;
}

/// A bounded service estimate of existing instructions, never a scheduler or
/// legality proof. Values and clocks are local to this read-only invocation.
class InstructionServiceEstimator {
public:
  InstructionServiceEstimator(const SearchCostPolicy &policy,
                              CompletionTimes &completionTimes)
      : policy(policy), completionTimes(completionTimes) {}

  std::optional<uint64_t> estimate(mlir::Operation *root) {
    engines = {};
    issue = dte = messages = work = localEstimates = 0;
    indices.clear();
    aliases.clear();
    storages.clear();
    hazards.clear();
    tokens.clear();
    coarse = false;
    auto module = mlir::dyn_cast_or_null<mlir::ModuleOp>(root);
    if (!module)
      return std::nullopt;
    unsigned entries = 0;
    for (auto function : module.getOps<mlir::func::FuncOp>()) {
      if (function.isPrivate())
        continue;
      if (++entries != 1 || !function.getBody().hasOneBlock() ||
          !execute(function.getBody().front()))
        return std::nullopt;
    }
    return entries == 1 ? std::optional<uint64_t>(maximum()) : std::nullopt;
  }

  bool usesCoarseEstimate() const { return coarse; }
  uint64_t getWorkCount() const { return work; }
  uint64_t getLocalEstimateCount() const { return localEstimates; }

private:
  struct Hazard {
    uint64_t read = 0, write = 0;
  };
  struct Storage {
    mlir::Value root;
    std::optional<std::pair<uint64_t, uint64_t>> spm;
  };

  uint64_t maximum() const {
    uint64_t result = std::max(issue, dte);
    for (const auto &worker : engines)
      for (uint64_t time : worker)
        result = std::max(result, time);
    for (const auto &token : tokens)
      result = std::max(result, token.second);
    return result;
  }

  std::optional<int64_t> index(mlir::Value value) {
    auto known = indices.find(value);
    if (known != indices.end())
      return known->second;
    if (auto constant = mlir::getConstantIntValue(value))
      return constant;
    auto *op = value.getDefiningOp();
    if (!op || ++work > 65536)
      return std::nullopt;
    if (mlir::isa<mlir::arith::IndexCastOp, mlir::arith::IndexCastUIOp>(op))
      return index(op->getOperand(0));
    if (op->getNumOperands() != 2)
      return std::nullopt;
    auto a = index(op->getOperand(0)), b = index(op->getOperand(1));
    if (!a || !b)
      return std::nullopt;
    __int128 result;
    if (mlir::isa<mlir::arith::AddIOp>(op))
      result = (__int128)*a + *b;
    else if (mlir::isa<mlir::arith::SubIOp>(op))
      result = (__int128)*a - *b;
    else if (mlir::isa<mlir::arith::MulIOp>(op))
      result = (__int128)*a * *b;
    else if (mlir::isa<mlir::arith::DivUIOp>(op) && *a >= 0 && *b > 0)
      result = *a / *b;
    else if (mlir::isa<mlir::arith::RemUIOp>(op) && *a >= 0 && *b > 0)
      result = *a % *b;
    else if (auto comparison = mlir::dyn_cast<mlir::arith::CmpIOp>(op)) {
      if (comparison.getPredicate() != mlir::arith::CmpIPredicate::eq)
        return std::nullopt;
      result = *a == *b;
    } else
      return std::nullopt;
    if (result < std::numeric_limits<int64_t>::min() ||
        result > std::numeric_limits<int64_t>::max())
      return std::nullopt;
    return static_cast<int64_t>(result);
  }

  std::optional<Storage> storage(mlir::Value value) {
    if (++work > 65536)
      return std::nullopt;
    auto mapped = aliases.find(value);
    if (mapped != aliases.end())
      return mapped->second;
    if (auto view = mlir::dyn_cast_or_null<mlir::ViewLikeOpInterface>(
            value.getDefiningOp()))
      return storage(view.getViewSource());
    if (auto select = value.getDefiningOp<mlir::arith::SelectOp>()) {
      auto condition = index(select.getCondition());
      if (!condition)
        return std::nullopt;
      return storage(*condition ? select.getTrueValue()
                                : select.getFalseValue());
    }
    if (auto alloc = value.getDefiningOp<mlir::memref::AllocOp>()) {
      if (isWaferSPMMemRefType(alloc.getType())) {
        auto offset =
            alloc->getAttrOfType<SPMOffsetAttr>(kWaferSPMOffsetAttrName);
        auto physical = computeWaferPhysicalTensorInfo(alloc.getType());
        if (!offset || offset.getOffset() < 0 || !physical ||
            physical->physicalBytes < 0)
          return std::nullopt;
        uint64_t end;
        if (!checkedAdd(offset.getOffset(), physical->physicalBytes, end))
          return std::nullopt;
        return Storage{value, std::make_pair(offset.getOffset(), end)};
      }
      return Storage{value, {}};
    }
    auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
    if (argument &&
        mlir::isa<mlir::func::FuncOp>(argument.getOwner()->getParentOp()))
      return Storage{value, {}};
    return std::nullopt;
  }

  bool bind(mlir::Value to, mlir::Value from) {
    if (mlir::isa<mlir::MemRefType>(to.getType())) {
      auto root = storage(from);
      if (!root)
        return false;
      aliases[to] = *root;
    } else if (to.getType().isIndex()) {
      auto value = index(from);
      if (!value)
        return false;
      indices[to] = *value;
    } else if (mlir::isa<mlir::async::TokenType>(to.getType())) {
      auto found = tokens.find(from);
      if (found == tokens.end())
        return false;
      tokens[to] = found->second;
    }
    return true;
  }

  bool shift(uint64_t delta,
             const llvm::DenseMap<mlir::Value, uint64_t> &previousTokens) {
    if (!checkedAdd(issue, delta, issue))
      return false;
    if (!checkedAdd(dte, delta, dte))
      return false;
    for (auto &entry : tokens)
      if (previousTokens.lookup(entry.first) != entry.second &&
          !checkedAdd(entry.second, delta, entry.second))
        return false;
    for (auto &worker : engines)
      for (auto &time : worker)
        if (!checkedAdd(time, delta, time))
          return false;
    for (auto &entry : hazards)
      if (!checkedAdd(entry.second.read, delta, entry.second.read) ||
          !checkedAdd(entry.second.write, delta, entry.second.write))
        return false;
    return true;
  }

  bool executeLoop(mlir::scf::ForOp loop) {
    auto lower = index(loop.getLowerBound()),
         upper = index(loop.getUpperBound()), step = index(loop.getStep());
    if (!lower || !upper || !step || *step <= 0)
      return false;
    for (auto [arg, initial] :
         llvm::zip_equal(loop.getRegionIterArgs(), loop.getInitArgs()))
      if (!bind(arg, initial))
        return false;
    __int128 distance = (__int128)*upper - *lower;
    if (distance > std::numeric_limits<int64_t>::max())
      return false;
    uint64_t count = distance <= 0 ? 0 : 1 + (uint64_t(distance) - 1) / *step;
    auto yield =
        mlir::cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
    // Extrapolating clocks cannot advance loop-carried index values. Execute
    // these loops within the ordinary work bound instead, so their results
    // remain usable by following loop bounds and slot selections.
    bool canSummarize = llvm::all_of(
        llvm::zip_equal(loop.getRegionIterArgs(), yield.getOperands()),
        [](auto pair) {
          auto [argument, result] = pair;
          return !argument.getType().isIndex() || argument == result;
        });
    llvm::SmallVector<Storage> thirdAliases;
    llvm::DenseMap<mlir::Value, uint64_t> thirdTokens;
    uint64_t third = 0;
    for (uint64_t iteration = 0; iteration < count; ++iteration) {
      indices[loop.getInductionVar()] = *lower + iteration * *step;
      if (!execute(*loop.getBody()))
        return false;
      // Results provide temporary SSA destinations, so swapping iter_args is
      // evaluated simultaneously rather than overwriting another source.
      for (auto [result, value] :
           llvm::zip_equal(loop.getResults(), yield.getOperands()))
        if (!bind(result, value))
          return false;
      for (auto [arg, result] :
           llvm::zip_equal(loop.getRegionIterArgs(), loop.getResults()))
        if (!bind(arg, result))
          return false;
      if (iteration == 2) {
        third = maximum();
        thirdTokens = tokens;
        for (auto argument : loop.getRegionIterArgs())
          if (mlir::isa<mlir::MemRefType>(argument.getType()))
            thirdAliases.push_back(aliases.lookup(argument));
      }
      if (iteration == 4 && count > 5 && canSummarize) {
        unsigned position = 0;
        for (auto argument : loop.getRegionIterArgs())
          if (mlir::isa<mlir::MemRefType>(argument.getType())) {
            const auto &before = thirdAliases[position++];
            const auto &after = aliases.lookup(argument);
            canSummarize &=
                before.root == after.root && before.spm == after.spm;
          }
        if (!canSummarize)
          continue;
        uint64_t periods = (count - 5) / 2, delta;
        if (!checkedMultiply(maximum() - third, periods, delta) ||
            !shift(delta, thirdTokens))
          return false;
        iteration += periods * 2;
      }
    }
    if (!count)
      for (auto [result, initial] :
           llvm::zip_equal(loop.getResults(), loop.getInitArgs()))
        if (!bind(result, initial))
          return false;
    return true;
  }

  bool executeInstruction(mlir::Operation *operation) {
    auto completion = getNCCOperationCompletion(operation);
    if (completion.kind == NCCCompletionKind::ParticipantJoin) {
      uint64_t participants = 0;
      for (unsigned worker = 0; worker < engines.size(); ++worker)
        if (completion.participantMask & (1u << worker)) {
          ++participants;
          for (uint64_t time : engines[worker])
            issue = std::max(issue, time);
        }
      uint64_t waits;
      return checkedMultiply(participants,
                             policy.nccParticipantWaitPicosecondsEstimate,
                             waits) &&
             checkedAdd(issue, waits, issue) &&
             checkedAdd(issue, policy.nccJoinPicosecondsEstimate, issue);
    }
    auto instruction = mlir::dyn_cast<WaferInstructionOpInterface>(operation);
    auto worker = getNCCIssueWorker(operation);
    if (!instruction || !worker ||
        completion.kind != NCCCompletionKind::OrderedAsynchronousIssue)
      return false;
    unsigned family = static_cast<unsigned>(instruction.getInstructionFamily());
    if (family >= 5)
      return false; // DTE retains the existing lifecycle estimate.
    auto found = services.find(operation);
    if (found == services.end()) {
      auto work =
          analyzeInstructionProgramCost(operation, getTargetMemoryPolicy());
      if (!work.ddrReadBytes.isKnown() || !work.ddrWriteBytes.isKnown() ||
          !work.spmMovementBytes.isKnown() ||
          !work.compute.npuF16Bf16LogicalOps.isKnown() ||
          !work.compute.vectorF16Bf16LogicalOps.isKnown() ||
          !work.compute.vectorF32LogicalOps.isKnown() ||
          !work.compute.npuOtherLogicalOps.isKnown() ||
          work.compute.npuOtherLogicalOps.value ||
          !work.compute.vectorOtherLogicalOps.isKnown() ||
          work.compute.vectorOtherLogicalOps.value)
        return false;
      auto service = localServiceTime(work, policy, /*includeIssue=*/false);
      uint64_t bytes, duration;
      if (!service ||
          !checkedAdd(work.ddrReadBytes.value, work.ddrWriteBytes.value, bytes))
        return false;
      auto ddr = timeForWork(bytes, policy.ddrNominalBytesPerSecond);
      if (!ddr || !checkedAdd(*service, *ddr, duration))
        return false;
      found = services.try_emplace(operation, duration).first;
    }
    auto effects = mlir::getEffectsRecursively(operation);
    if (!effects)
      return false;
    llvm::SmallVector<std::pair<Storage, bool>, 4> accesses;
    for (const auto &effect : *effects) {
      if (!effect.getValue() ||
          !mlir::isa<mlir::MemRefType>(effect.getValue().getType()))
        continue;
      auto root = storage(effect.getValue());
      if (!root)
        return false;
      accesses.emplace_back(
          *root, !mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect()));
    }
    uint64_t &engine = engines[static_cast<unsigned>(*worker)][family];
    uint64_t start = std::max(issue, engine);
    for (const auto &[root, write] : accesses)
      for (const auto &[key, hazard] : hazards) {
        const auto &other = storages.find(key)->second;
        bool overlap = root.root == other.root;
        if (root.spm && other.spm)
          overlap |= root.spm->first < other.spm->second &&
                     other.spm->first < root.spm->second;
        if (overlap)
          start = std::max(start, write ? std::max(hazard.read, hazard.write)
                                        : hazard.write);
      }
    if (!checkedAdd(start, found->second, engine) ||
        !checkedAdd(issue, policy.instructionFixedPicosecondsEstimate, issue))
      return false;
    for (const auto &[root, write] : accesses) {
      storages.try_emplace(root.root, root);
      auto &hazard = hazards[root.root];
      auto &time = write ? hazard.write : hazard.read;
      time = std::max(time, engine);
    }
    return true;
  }

  uint64_t hazardTime(const Storage &root, bool write) const {
    uint64_t time = 0;
    for (const auto &[key, hazard] : hazards) {
      const auto &other = storages.find(key)->second;
      bool overlap = root.root == other.root;
      if (root.spm && other.spm)
        overlap |= root.spm->first < other.spm->second &&
                   other.spm->first < root.spm->second;
      if (overlap)
        time = std::max(time, write ? std::max(hazard.read, hazard.write)
                                    : hazard.write);
    }
    return time;
  }

  void recordAccess(const Storage &root, bool write, uint64_t time) {
    storages.try_emplace(root.root, root);
    auto &hazard = hazards[root.root];
    auto &previous = write ? hazard.write : hazard.read;
    previous = std::max(previous, time);
  }

  bool executePublication(mlir::Value data, bool publish) {
    auto root = storage(data);
    if (!root)
      return false;
    auto argument = mlir::dyn_cast<mlir::BlockArgument>(root->root);
    if (!argument)
      return false;
    auto function =
        mlir::dyn_cast<mlir::func::FuncOp>(argument.getOwner()->getParentOp());
    auto binding = function
                       ? function.getArgAttrOfType<DDRBindingAttr>(
                             argument.getArgNumber(), kWaferDDRBindingAttrName)
                       : DDRBindingAttr{};
    if (!binding)
      return false;
    if (publish) {
      issue = std::max(issue, hazardTime(*root, /*write=*/false));
      if (!checkedAdd(issue, policy.instructionFixedPicosecondsEstimate, issue))
        return false;
      completionTimes.update(completionTimes.publications,
                             binding.getResourceId(), issue);
    } else {
      auto found = completionTimes.publications.find(binding.getResourceId());
      if (found == completionTimes.publications.end()) {
        coarse = true;
        if (!checkedAdd(issue, policy.unmodeledInstructionPicosecondsEstimate,
                        issue))
          return false;
      } else {
        issue = std::max(issue, found->second);
      }
      if (!checkedAdd(issue, policy.instructionFixedPicosecondsEstimate, issue))
        return false;
    }
    return true;
  }

  bool executeDTE(mlir::Operation *operation) {
    if (auto wait = mlir::dyn_cast<InstrDTEWaitOp>(operation)) {
      for (auto token : wait.getTokens()) {
        auto found = tokens.find(token);
        if (found == tokens.end())
          return false;
        issue = std::max(issue, found->second);
      }
      uint64_t control;
      return checkedMultiply(wait.getTokens().size(),
                             policy.dteWaitedEventPicosecondsEstimate,
                             control) &&
             checkedAdd(issue, control, issue) &&
             checkedAdd(issue, policy.instructionFixedPicosecondsEstimate,
                        issue);
    }
    auto send = mlir::dyn_cast<InstrDTESendOp>(operation);
    auto receive = mlir::dyn_cast<InstrDTERecvOp>(operation);
    if (!send && !receive)
      return false;
    auto root = storage(send ? send.getBuffer() : receive.getBuffer());
    if (!root)
      return false;
    uint64_t start = std::max(issue, hazardTime(*root, bool(receive)));
    uint64_t finish = start;
    const bool repeated = !executesOnce(operation);
    if (send) {
      start = std::max(start, dte);
      auto remote = completionTimes.receiveForSend.find(operation);
      if (!repeated && remote != completionTimes.receiveForSend.end())
        start = std::max(start,
                         completionTimes.receivePosted.lookup(remote->second));
      else
        coarse = true;
      auto bytes = timeForWork(send.getBytes(),
                               policy.dteEndpointBytesPerSecondEstimate);
      uint64_t startup = messages++ == 0
                             ? policy.dteFirstMessagePicosecondsEstimate
                             : policy.dteMessageStartupPicosecondsEstimate;
      if (!bytes || !checkedAdd(start, startup, finish) ||
          !checkedAdd(finish, *bytes, finish))
        return false;
      dte = finish;
      if (!repeated)
        completionTimes.update(completionTimes.sendFinished, operation, finish);
      tokens[send.getToken()] = finish;
    } else {
      auto remote = completionTimes.sendForReceive.find(operation);
      if (!repeated && remote != completionTimes.sendForReceive.end()) {
        completionTimes.update(completionTimes.receivePosted, operation, start);
        finish = std::max(start,
                          completionTimes.sendFinished.lookup(remote->second));
      } else {
        // The receive has no matched single-invocation remote timestamp.
        // Preserve local work/dependencies with an explicit lifecycle prior.
        coarse = true;
        auto bytes = timeForWork(receive.getBytes(),
                                 policy.dteEndpointBytesPerSecondEstimate);
        if (!bytes ||
            !checkedAdd(start, policy.dteMessageStartupPicosecondsEstimate,
                        finish) ||
            !checkedAdd(finish, *bytes, finish))
          return false;
      }
      tokens[receive.getToken()] = finish;
    }
    recordAccess(*root, bool(receive), finish);
    return checkedAdd(issue, policy.instructionFixedPicosecondsEstimate, issue);
  }

  void estimateLocally(mlir::Operation *operation, uint64_t prefix) {
    coarse = true;
    ++localEstimates;
    auto cost =
        analyzeInstructionProgramCost(operation, getTargetMemoryPolicy());
    auto service = localServiceTime(cost, policy);
    uint64_t bytes, duration = 0;
    if (!service || !cost.ddrReadBytes.isKnown() ||
        !cost.ddrWriteBytes.isKnown() || !cost.instructionCount.isKnown() ||
        !cost.compute.npuF16Bf16LogicalOps.isKnown() ||
        !cost.compute.vectorF16Bf16LogicalOps.isKnown() ||
        !cost.compute.vectorF32LogicalOps.isKnown() ||
        !cost.compute.npuOtherLogicalOps.isKnown() ||
        cost.compute.npuOtherLogicalOps.value ||
        !cost.compute.vectorOtherLogicalOps.isKnown() ||
        cost.compute.vectorOtherLogicalOps.value ||
        !checkedAdd(cost.ddrReadBytes.value, cost.ddrWriteBytes.value, bytes)) {
      uint64_t count = std::max(
          uint64_t{1}, estimateInstructionCount(cost.work.instructions,
                                                cost.instructionCount));
      if (!checkedMultiply(
              count, policy.unmodeledInstructionPicosecondsEstimate, duration))
        duration = std::numeric_limits<uint64_t>::max();
    } else {
      auto ddr = timeForWork(bytes, policy.ddrNominalBytesPerSecond);
      if (!ddr || !checkedAdd(*service, *ddr, duration))
        duration = std::numeric_limits<uint64_t>::max();
      if (!duration)
        duration = policy.unmodeledInstructionPicosecondsEstimate;
    }
    if (!checkedAdd(prefix, duration, issue))
      issue = std::numeric_limits<uint64_t>::max();
    dte = issue;
    for (auto &worker : engines)
      worker.fill(issue);
    hazards.clear();
    // A failed partial loop/region visit cannot leave guessed carry results.
    auto within = [&](mlir::Value value) {
      auto *owner = value.getParentRegion()->getParentOp();
      return owner == operation || operation->isAncestor(owner) ||
             value.getDefiningOp() == operation;
    };
    for (auto it = aliases.begin(); it != aliases.end();)
      if (within(it->first))
        aliases.erase(it++);
      else
        ++it;
    for (auto it = indices.begin(); it != indices.end();)
      if (within(it->first))
        indices.erase(it++);
      else
        ++it;
    for (auto it = tokens.begin(); it != tokens.end();)
      if (within(it->first))
        tokens.erase(it++);
      else
        ++it;
  }

  bool execute(mlir::Block &block) {
    for (mlir::Operation &operation : block.without_terminator()) {
      const uint64_t prefix = maximum();
      auto visit = [&]() -> bool {
        if (++work > 65536)
          return false;
        if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(operation)) {
          if (!executeLoop(loop))
            return false;
        } else if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(operation)) {
          auto condition = index(branch.getCondition());
          if (!condition)
            return false;
          auto &region =
              *condition ? branch.getThenRegion() : branch.getElseRegion();
          if (region.empty())
            return branch.getNumResults() == 0;
          if (!region.hasOneBlock() || !execute(region.front()))
            return false;
          for (auto [result, value] :
               llvm::zip_equal(branch.getResults(),
                               region.front().getTerminator()->getOperands()))
            if (!bind(result, value))
              return false;
        } else if (auto region = mlir::dyn_cast<TileRegionOp>(operation)) {
          if (!region.getBody().hasOneBlock())
            return false;
          auto &body = region.getBody().front();
          for (auto [argument, operand] :
               llvm::zip_equal(body.getArguments(), region.getInputs()))
            if (!bind(argument, operand))
              return false;
          if (!execute(body))
            return false;
          for (auto [result, value] : llvm::zip_equal(
                   region.getResults(), body.getTerminator()->getOperands()))
            if (!bind(result, value))
              return false;
        } else if (auto publication =
                       mlir::dyn_cast<SyncDDRPublishOp>(operation)) {
          return executePublication(publication.getData(), true);
        } else if (auto acquisition =
                       mlir::dyn_cast<SyncDDRAcquireOp>(operation)) {
          return executePublication(acquisition.getData(), false);
        } else if (mlir::isa<InstrDTESendOp, InstrDTERecvOp, InstrDTEWaitOp>(
                       operation)) {
          return executeDTE(&operation);
        } else if (mlir::isa<WaferInstructionOpInterface, SyncNCCJoinOp>(
                       operation)) {
          if (!executeInstruction(&operation))
            return false;
        } else if (!mlir::isMemoryEffectFree(&operation) &&
                   !mlir::isa<mlir::memref::AllocOp, mlir::memref::DeallocOp>(
                       operation)) {
          return false;
        }
        return true;
      };
      if (!visit())
        estimateLocally(&operation, prefix);
    }
    return true;
  }

  const SearchCostPolicy &policy;
  CompletionTimes &completionTimes;
  std::array<std::array<uint64_t, static_cast<unsigned>(InstrFamily::DTE)>,
             kNCCWorkerCount>
      engines{};
  uint64_t issue = 0, dte = 0, messages = 0, work = 0, localEstimates = 0;
  bool coarse = false;
  llvm::DenseMap<mlir::Value, int64_t> indices;
  llvm::DenseMap<mlir::Value, Storage> aliases, storages;
  llvm::DenseMap<mlir::Value, Hazard> hazards;
  llvm::DenseMap<mlir::Value, uint64_t> tokens;
  llvm::DenseMap<mlir::Operation *, uint64_t> services;
};

} // namespace

SearchObjective
deriveSearchObjective(const InstructionProgramAggregateCost &cost,
                      const std::optional<SearchCostCohort> &cohort,
                      llvm::ArrayRef<TileInstructionProgram> currentPrograms) {
  SearchObjective result = deriveResourceObjective(cost, cohort);
  auto *known = std::get_if<KnownSearchObjective>(&result);
  const bool resourcesAvailable = known != nullptr;
  if (!known) {
    auto reason = std::get<UnknownSearchObjective>(result).reason;
    if (reason == SearchObjectiveUnknownReason::NoCohort)
      return result;
    KnownSearchObjective coarse{{}, *cohort, 0, true};
    if (reason == SearchObjectiveUnknownReason::ArithmeticOverflow) {
      coarse.estimatedDurationPicoseconds =
          std::numeric_limits<uint64_t>::max();
      return coarse;
    }
    // Last-resort service prior when detailed work cannot be priced. Reuse the
    // explicit generic instruction service estimate; do not invent IR or return
    // zero for unavailable work. This is only a ranking estimate.
    uint64_t count = 0;
    if (cost.tileCosts.empty()) {
      count = estimateInstructionCount(cost.aggregateWork.instructions,
                                       cost.aggregateInstructionCount);
    } else {
      for (const auto &tile : cost.tileCosts)
        count =
            std::max(count, estimateInstructionCount(tile.work.instructions,
                                                     tile.instructionCount));
    }
    count = std::max(count, uint64_t{1});
    if (!checkedMultiply(
            count, cohort->getPolicy().unmodeledInstructionPicosecondsEstimate,
            coarse.estimatedDurationPicoseconds))
      coarse.estimatedDurationPicoseconds =
          std::numeric_limits<uint64_t>::max();
    result = coarse;
    known = std::get_if<KnownSearchObjective>(&result);
    uint64_t bytes;
    if (cost.aggregateDDRReadBytes.isKnown() &&
        cost.aggregateDDRWriteBytes.isKnown() &&
        checkedAdd(cost.aggregateDDRReadBytes.value,
                   cost.aggregateDDRWriteBytes.value, bytes))
      if (auto ddr =
              timeForWork(bytes, cohort->getPolicy().ddrNominalBytesPerSecond))
        known->durations.ddrPicoseconds = *ddr;
  }

  const auto &d = known->durations;
  if (resourcesAvailable) {
    unsigned __int128 local = 0;
    if (cost.tileCosts.empty()) {
      // Aggregate-only callers represent a single program arithmetic oracle.
      for (uint64_t term :
           {d.neF16Bf16Picoseconds, d.vectorF16Bf16Picoseconds,
            d.vectorF32Picoseconds, d.spmMovementPicoseconds,
            d.dteEndpointPicoseconds, d.dteStartupPicoseconds,
            d.instructionControlPicoseconds, d.dteWaitControlPicoseconds,
            d.nccWaitControlPicoseconds})
        local += term;
    } else {
      for (const auto &tile : cost.tileCosts) {
        auto time = localServiceTime(tile, cohort->getPolicy());
        if (!time) {
          local = std::numeric_limits<uint64_t>::max();
          known->usesCoarseEstimate = true;
          break;
        }
        local = std::max(local, static_cast<unsigned __int128>(*time));
      }
    }
    // DDR is card-shared. Endpoint and link serialize the same payload: only
    // charge link pressure exceeding endpoint service, rather than both in
    // full.
    unsigned __int128 total = local + d.ddrPicoseconds + d.nocHopPicoseconds;
    if (d.nocPicoseconds > d.dteEndpointPicoseconds)
      total += d.nocPicoseconds - d.dteEndpointPicoseconds;
    if (total > std::numeric_limits<uint64_t>::max()) {
      known->estimatedDurationPicoseconds =
          std::numeric_limits<uint64_t>::max();
      known->usesCoarseEstimate = true;
    } else {
      known->estimatedDurationPicoseconds = static_cast<uint64_t>(total);
    }
  }
  if (!currentPrograms.empty() &&
      currentPrograms.size() == cost.tileCosts.size()) {
    llvm::SmallVector<TileInstructionProgram> ordered(currentPrograms);
    llvm::sort(ordered, [](const auto &lhs, const auto &rhs) {
      return lhs.tileId.getValue() < rhs.tileId.getValue();
    });
    CompletionTimes completionTimes;
    struct SendEndpoint {
      uint64_t tile;
      InstrDTESendOp op;
    };
    llvm::SmallVector<SendEndpoint> sends;
    for (const auto &program : ordered)
      program.root->walk([&](InstrDTESendOp send) {
        if (executesOnce(send))
          sends.push_back(
              {static_cast<uint64_t>(program.tileId.getValue()), send});
      });
    for (const auto &program : ordered)
      program.root->walk([&](InstrDTERecvOp receive) {
        if (!executesOnce(receive))
          return;
        for (auto &endpoint : sends)
          if (endpoint.tile == receive.getPeer() &&
              endpoint.op.getPeer() ==
                  static_cast<uint64_t>(program.tileId.getValue()) &&
              endpoint.op.getMessage() == receive.getMessage()) {
            completionTimes.sendForReceive[receive] = endpoint.op;
            completionTimes.receiveForSend[endpoint.op] = receive;
          }
      });
    llvm::SmallVector<std::unique_ptr<InstructionServiceEstimator>> estimators;
    for (size_t index = 0; index < ordered.size(); ++index)
      estimators.push_back(std::make_unique<InstructionServiceEstimator>(
          cohort->getPolicy(), completionTimes));
    uint64_t critical = 0;
    bool complete = false, coarse = false;
    for (unsigned iteration = 0; iteration < 32; ++iteration) {
      support::addCompileCounter("cost", "completion-propagation-rounds", 1);
      completionTimes.changed = false;
      critical = 0;
      coarse = false;
      bool failed = false;
      for (auto [program, estimator] : llvm::zip(ordered, estimators)) {
        auto duration = estimator->estimate(program.root);
        support::addCompileCounter("cost", "current-ir-estimate-work",
                                   estimator->getWorkCount());
        support::addCompileCounter("cost", "local-coarse-scopes",
                                   estimator->getLocalEstimateCount());
        if (!duration) {
          failed = true;
          break;
        }
        coarse |= estimator->usesCoarseEstimate();
        critical = std::max(critical, *duration);
      }
      if (failed)
        break;
      if (!completionTimes.changed) {
        complete = true;
        break;
      }
    }
    if (complete) {
      uint64_t duration =
          std::max({critical, d.ddrPicoseconds, d.nocPicoseconds});
      if (!checkedAdd(duration, d.nocHopPicoseconds, duration)) {
        duration = std::numeric_limits<uint64_t>::max();
        coarse = true;
      }
      known->estimatedDurationPicoseconds = duration;
      known->usesCoarseEstimate |= coarse;
    } else {
      known->usesCoarseEstimate = true;
      wafer::support::addCompileCounter("cost",
                                        "completion-estimate-work-limit", 1);
    }
  }
  return result;
}

SearchObjectiveComparison compareSearchObjectives(const SearchObjective &lhs,
                                                  const SearchObjective &rhs) {
  const auto *left = std::get_if<KnownSearchObjective>(&lhs);
  const auto *right = std::get_if<KnownSearchObjective>(&rhs);
  // Missing/mixed profiles are a caller contract error, not a resource
  // tradeoff.
  if (!left || !right || !(left->cohort == right->cohort))
    return SearchObjectiveComparison::Incomparable;
  auto leftKey = std::make_pair(left->estimatedDurationPicoseconds,
                                asStorageArray(left->durations));
  auto rightKey = std::make_pair(right->estimatedDurationPicoseconds,
                                 asStorageArray(right->durations));
  if (leftKey < rightKey)
    return SearchObjectiveComparison::Better;
  if (rightKey < leftKey)
    return SearchObjectiveComparison::Worse;
  return SearchObjectiveComparison::Equivalent;
}

} // namespace wafer::analysis
