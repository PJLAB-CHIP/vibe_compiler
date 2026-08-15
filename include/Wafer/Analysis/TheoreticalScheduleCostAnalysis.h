//===- TheoreticalScheduleCostAnalysis.h - Numeric cost -------*- C++ -*-===//

#ifndef WAFER_ANALYSIS_THEORETICALSCHEDULECOSTANALYSIS_H
#define WAFER_ANALYSIS_THEORETICALSCHEDULECOSTANALYSIS_H

#include "Wafer/Analysis/ScheduleCostAnalysis.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <utility>

namespace wafer::analysis {

/// Versioned point-model inputs used by the theoretical duration estimate.
/// These flags are diagnostic only; they never establish a performance proof.
enum class StaticDurationAssumption : uint32_t {
  None = 0,
  DDROperatingPoint = 1u << 0,
  ComputePeakReference = 1u << 1,
  ModeledShortestPathRoute = 1u << 2,
  DTEEndpointRatePrior = 1u << 3,
  DTEMessageStartupPrior = 1u << 4,
  NoCHopPrior = 1u << 5,
  ControlIssuePrior = 1u << 6,
  ExplicitSchedulePlan = 1u << 7,
  BufferedPipelinePlan = 1u << 8,
  SPMServiceRatePrior = 1u << 9,
};

using StaticDurationAssumptionMask = uint32_t;

constexpr StaticDurationAssumptionMask
staticDurationAssumptionMask(StaticDurationAssumption assumption) {
  return static_cast<StaticDurationAssumptionMask>(assumption);
}

/// Independently enabled model terms. A term is enabled once for a complete
/// comparison cohort only when its parameter and required work collector are
/// available for every candidate. Disabled terms contribute nothing to every
/// candidate in that cohort; they are never converted into a guessed value.
enum class StaticDurationTerm : uint32_t {
  DDR = 1u << 0,
  NPUF16Bf16 = 1u << 1,
  VectorF16Bf16 = 1u << 2,
  VectorF32 = 1u << 3,
  NoCLink = 1u << 4,
  NoCTransmitEndpoint = 1u << 5,
  NoCReceiveEndpoint = 1u << 6,
  NoCMessageStartup = 1u << 7,
  NoCHop = 1u << 8,
  SPMMovement = 1u << 9,
  InstructionControl = 1u << 10,
  DTEWaitControl = 1u << 11,
  NCCWaitControl = 1u << 12,
};

using StaticDurationTermMask = uint32_t;

constexpr StaticDurationTermMask
staticDurationTermMask(StaticDurationTerm term) {
  return static_cast<StaticDurationTermMask>(term);
}

struct StaticDurationEstimate {
  uint64_t picoseconds = 0;
  StaticDurationAssumptionMask assumptions = 0;
};

/// Numeric theoretical resource cost for one card executable. All
/// fields are always numeric. `enabledTerms` is cohort-wide and therefore
/// identical for every estimate produced by one plural estimation call.
struct ProgramDurationEstimate {
  StaticDurationEstimate ddr;
  StaticDurationEstimate compute;
  StaticDurationEstimate noc;
  StaticDurationEstimate spm;
  StaticDurationEstimate control;
  StaticDurationEstimate makespan;
  StaticDurationTermMask enabledTerms = 0;
};

/// Resource classes whose already-estimated service durations are placed on
/// an explicit finite schedule. Control overhead is deliberately outside this
/// set: it is charged once after the resource plan and cannot be hidden by a
/// data-movement/compute overlap claim.
enum class StaticScheduleResource : uint8_t {
  DDR = 1u << 0,
  Compute = 1u << 1,
  NoC = 1u << 2,
  SPMMovement = 1u << 3,
};

using StaticScheduleResourceMask = uint8_t;

constexpr StaticScheduleResourceMask
staticScheduleResourceMask(StaticScheduleResource resource) {
  return static_cast<StaticScheduleResourceMask>(resource);
}

constexpr StaticScheduleResourceMask allStaticScheduleResources() {
  return staticScheduleResourceMask(StaticScheduleResource::DDR) |
         staticScheduleResourceMask(StaticScheduleResource::Compute) |
         staticScheduleResourceMask(StaticScheduleResource::NoC) |
         staticScheduleResourceMask(StaticScheduleResource::SPMMovement);
}

/// One finite resource-work item. The pointer refers to either a whole final
/// program cost used by the conservative baseline or a scheduler-owned
/// phase/wave cost slice. The pointee must outlive the plan and its estimation
/// call.
struct StaticScheduleWork {
  const CardInstructionProgramCost *cost = nullptr;
  StaticScheduleResourceMask resources = 0;
};

/// Work in one branch has data/effect order and therefore sums. Different
/// branches in one stage are independent and therefore the stage duration is
/// their maximum. Distinct work pointers let a structured-DAG scheduler represent
/// concurrent branches with different service demands.
struct StaticScheduleBranch {
  llvm::SmallVector<StaticScheduleWork, 4> dependentWork;
};

struct StaticScheduleStage {
  llvm::SmallVector<StaticScheduleBranch, 4> independentBranches;
};

/// One explicit buffered pipeline. Prologue and epilogue are finite dependent
/// stage sequences. The steady-state initiation interval is the maximum
/// branch service duration in `steady`, repeated `steadyWaveCount` times. Each
/// branch is one explicitly independent resource/physical lane; dependent work
/// sharing that lane stays in the same branch and sums before the maximum.
/// The wave count is the number of steady intervals, so a caller implementing
/// `prologue + (waves - 1) * II + epilogue` passes `waves - 1` here.
struct StaticBufferedPipeline {
  llvm::SmallVector<StaticScheduleStage, 2> prologue;
  StaticScheduleStage steady;
  uint64_t steadyWaveCount = 0;
  llvm::SmallVector<StaticScheduleStage, 2> epilogue;
};

struct StaticScheduleStep {
  enum class Kind : uint8_t { Stage, BufferedPipeline };

  static StaticScheduleStep forStage(StaticScheduleStage stage);
  static StaticScheduleStep
  forBufferedPipeline(StaticBufferedPipeline pipeline);

  Kind kind = Kind::Stage;
  StaticScheduleStage stage;
  StaticBufferedPipeline pipeline;
};

/// Validated finite schedule plan. Every resource class must occur in the
/// plan, even if its cohort-wide enabled terms produce zero service time. This
/// prevents a malformed plan from silently dropping known work. Repetition is
/// allowed only when stated explicitly by the plan (for example a steady wave
/// count); all timeline arithmetic is saturating.
class StaticSchedulePlan {
public:
  StaticSchedulePlan(const StaticSchedulePlan &) = default;
  StaticSchedulePlan(StaticSchedulePlan &&) = default;
  StaticSchedulePlan &operator=(const StaticSchedulePlan &) = default;
  StaticSchedulePlan &operator=(StaticSchedulePlan &&) = default;

  static std::optional<StaticSchedulePlan>
  create(const CardInstructionProgramCost &controlCost,
         llvm::ArrayRef<StaticScheduleStep> steps);

  /// Conservative baseline plan: DDR, compute, NoC, and explicit SPM movement
  /// are four dependent stages. It is a concrete plan construction, not a
  /// binary global scheduling policy.
  static StaticSchedulePlan
  getConservative(const CardInstructionProgramCost &cost);

  llvm::ArrayRef<StaticScheduleStep> getSteps() const { return steps; }
  const CardInstructionProgramCost &getControlCost() const {
    return *controlCost;
  }

private:
  explicit StaticSchedulePlan(
      const CardInstructionProgramCost &controlCost,
      llvm::SmallVector<StaticScheduleStep, 8> steps)
      : controlCost(&controlCost), steps(std::move(steps)) {}

  const CardInstructionProgramCost *controlCost;
  llvm::SmallVector<StaticScheduleStep, 8> steps;
};

/// Estimate explicit plans under one uniform enabled-term set. Pointer form
/// lets independently-owned structured-DAG candidates retain their finite plans and
/// phase costs without copies. A null plan returns an empty result.
llvm::SmallVector<ProgramDurationEstimate, 16>
estimateStaticSchedulePlanDurations(
    llvm::ArrayRef<const StaticSchedulePlan *> plans,
    const TargetScheduleCostPolicy &policy);

} // namespace wafer::analysis

#endif // WAFER_ANALYSIS_THEORETICALSCHEDULECOSTANALYSIS_H
