#include "../../lib/Wafer/Compiler/CompilationInternal.h"
#include "../../lib/Wafer/Compiler/NoCResidentDataflow.h"
#include "../../lib/Wafer/Compiler/ScheduledRankFinalization.h"
#include "../../lib/Wafer/Compiler/WholeVariantAttemptPlan.h"
#include "../../lib/Wafer/Compiler/WholeVariantCoordinator.h"
#include "../../lib/Wafer/Transforms/Scheduling/ScheduleTensorProgramInternal.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace {

using Candidate = wafer::compiler::detail::RankVariantCandidate;
using Frontier = wafer::compiler::detail::RankVariantFrontier;

static constexpr llvm::StringLiteral kLargeTiledSource = R"mlir(
module {
  wafer.target.topology @default {
    card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
    tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>
  }
  wafer.execution.mesh @default_mesh {
    axes = ["rank"], endpoints = array<i64>,
    policy = "all_available", shape = array<i64: 16>, topology = @default
  }
  func.func @main(%lhs: tensor<524288xf32>, %rhs: tensor<524288xf32>)
      -> tensor<524288xf32> {
    %add_out = tensor.empty() : tensor<524288xf32>
    %add = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
      ins(%lhs, %lhs : tensor<524288xf32>, tensor<524288xf32>)
      outs(%add_out : tensor<524288xf32>) {
    ^bb0(%lhs_value: f32, %rhs_value: f32, %unused: f32):
      %value = arith.addf %lhs_value, %rhs_value : f32
      linalg.yield %value : f32
    } -> tensor<524288xf32>
    %mul_out = tensor.empty() : tensor<524288xf32>
    %mul = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
      ins(%rhs, %rhs : tensor<524288xf32>, tensor<524288xf32>)
      outs(%mul_out : tensor<524288xf32>) {
    ^bb0(%lhs_value: f32, %rhs_value: f32, %unused: f32):
      %value = arith.mulf %lhs_value, %rhs_value : f32
      linalg.yield %value : f32
    } -> tensor<524288xf32>
    %result_out = tensor.empty() : tensor<524288xf32>
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
      ins(%add, %mul : tensor<524288xf32>, tensor<524288xf32>)
      outs(%result_out : tensor<524288xf32>) {
    ^bb0(%add_value: f32, %mul_value: f32, %unused: f32):
      %value = arith.addf %add_value, %mul_value : f32
      linalg.yield %value : f32
    } -> tensor<524288xf32>
    return %result : tensor<524288xf32>
  }
}
)mlir";

static unsigned countDTEIssues(mlir::ModuleOp module) {
  unsigned count = 0;
  module.walk([&](mlir::Operation *operation) {
    if (mlir::isa<wafer::InstrDTESendOp, wafer::InstrDTERecvOp>(operation))
      ++count;
  });
  return count;
}

static size_t countNCCIssueWorkers(mlir::ModuleOp module) {
  std::set<wafer::NCCWorker> workers;
  module.walk([&](mlir::Operation *operation) {
    if (std::optional<wafer::NCCWorker> worker =
            wafer::getNCCIssueWorker(operation))
      workers.insert(*worker);
  });
  return workers.size();
}

static wafer::frontend::FrontendProgramVerificationResult
makeReplicatedProgram() {
  auto makeBinding = [](int64_t argumentIndex) {
    wafer::frontend::ProgramBoundaryBinding binding;
    binding.index = argumentIndex;
    binding.programIndex = argumentIndex;
    binding.distribution = wafer::frontend::ProgramDistributionKind::Replicated;
    binding.globalShape = {524288};
    binding.localShape = {524288};
    binding.dtype = "f32";
    for (int64_t rank = 0; rank < 16; ++rank) {
      wafer::frontend::ProgramRankSlice slice;
      slice.logicalRank = rank;
      slice.replicaId = rank;
      slice.offsets = {0};
      slice.sizes = {524288};
      slice.strides = {1};
      binding.rankSlices.push_back(std::move(slice));
    }
    return binding;
  };

  wafer::frontend::FrontendProgramVerificationResult program;
  program.logicalRankCount = 16;
  program.programUserInputCount = 2;
  program.distributedInputs.push_back(makeBinding(0));
  program.distributedInputs.push_back(makeBinding(1));
  program.distributedOutputs.push_back(makeBinding(0));
  return program;
}

struct FixedSlotInventory {
  unsigned complete = 0;
  unsigned withDTE = 0;
};

static FixedSlotInventory
inventoryCompleteFixedSlots(const std::vector<Frontier> &frontiers) {
  FixedSlotInventory inventory;
  if (frontiers.empty())
    return inventory;
  for (const Candidate &first : frontiers.front()) {
    if (first.bufferingKind != wafer::RankBufferingKind::StaticFixedSlot ||
        first.bufferingPlanOrdinal == 0)
      continue;
    bool complete = true;
    bool everyRankHasDTE = true;
    for (const Frontier &frontier : frontiers) {
      auto match = llvm::find_if(frontier, [&](const Candidate &candidate) {
        return candidate.stableOrdinal == first.stableOrdinal &&
               candidate.artifactKind == first.artifactKind &&
               candidate.bufferingKind == first.bufferingKind &&
               candidate.bufferingPlanOrdinal == first.bufferingPlanOrdinal &&
               candidate.workerPlacementKind == first.workerPlacementKind &&
               candidate.workerPlacementPlanOrdinal ==
                   first.workerPlacementPlanOrdinal;
      });
      if (match == frontier.end() || !match->module) {
        complete = false;
        break;
      }
      everyRankHasDTE &= countDTEIssues(*match->module) > 0;
    }
    if (!complete)
      continue;
    ++inventory.complete;
    inventory.withDTE += everyRankHasDTE ? 1u : 0u;
  }
  return inventory;
}

TEST(NoCTiledBoundaryDataflowTest,
     BuildsCompleteDTEFixedSlotTupleFromActualLoopTiledFrontiers) {
  mlir::DialectRegistry registry;
  wafer::compiler::detail::registerCompilationDialects(registry);
  auto context = std::make_unique<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();

  std::vector<Frontier> frontiers(16);
  for (int64_t rank = 0; rank < 16; ++rank) {
    mlir::OwningOpRef<mlir::ModuleOp> source =
        mlir::parseSourceString<mlir::ModuleOp>(
            kLargeTiledSource, mlir::ParserConfig(context.get()));
    ASSERT_TRUE(source);
    wafer::TensorProgramSchedulingConfig scheduling;
    scheduling.logicalRank = rank;
    scheduling.candidateParallelism = 4;
    mlir::FailureOr<std::vector<wafer::ScheduledRankCandidate>> scheduled =
        wafer::tensor_program_scheduling::testing::
            buildLegacyScheduledRankCandidateFrontier(*source, scheduling);
    ASSERT_TRUE(mlir::succeeded(scheduled));
    auto finalized =
        wafer::compiler::detail::finalizeScheduledRankCandidateFrontier(
            std::move(*scheduled));
    ASSERT_TRUE(mlir::succeeded(finalized));
    for (auto &candidate : *finalized)
      frontiers[rank].push_back(
          {std::move(candidate.module), candidate.stableOrdinal,
           candidate.artifactKind, candidate.reservedBaseline,
           candidate.bufferingKind, candidate.bufferingPlanOrdinal,
           candidate.workerPlacementKind,
           candidate.workerPlacementPlanOrdinal});
  }

  const FixedSlotInventory before = inventoryCompleteFixedSlots(frontiers);
  const size_t candidateCountBefore = frontiers.front().size();
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      16, wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(config)) << llvm::toString(config.takeError());
  wafer::frontend::FrontendProgramVerificationResult program =
      makeReplicatedProgram();
  std::string failure;
  ASSERT_TRUE(mlir::succeeded(
      wafer::compiler::detail::appendNoCResidentDataflowCandidates(
          frontiers, program, *config, &failure)))
      << failure;
  const FixedSlotInventory after = inventoryCompleteFixedSlots(frontiers);
  const unsigned rankZeroDTECandidates =
      llvm::count_if(frontiers.front(), [](const Candidate &candidate) {
        return candidate.module && countDTEIssues(*candidate.module) > 0;
      });

  EXPECT_GT(before.complete, 0u);
  EXPECT_EQ(before.withDTE, 0u);
  EXPECT_GT(after.withDTE, 0u)
      << "fixed-slot inventory before=" << before.complete
      << ", before+dte=" << before.withDTE << ", after=" << after.complete
      << ", after+dte=" << after.withDTE
      << ", candidate-count-before=" << candidateCountBefore
      << ", candidate-count-after=" << frontiers.front().size()
      << ", rank0-dte-candidates=" << rankZeroDTECandidates;

  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  auto qualified = wafer::compiler::detail::selectAcceptedWholeVariant(
      frontiers, program, *config, diagnostics,
      wafer::compiler::detail::WholeVariantSelectionMode::
          QualifyStaticFixedSlot);
  EXPECT_TRUE(mlir::succeeded(qualified)) << diagnosticText;
  if (mlir::succeeded(qualified)) {
    EXPECT_TRUE(llvm::all_of(
        qualified->selectedBufferingKinds, [](wafer::RankBufferingKind kind) {
          return kind == wafer::RankBufferingKind::StaticFixedSlot;
        }));
    EXPECT_TRUE(llvm::all_of(qualified->ranks,
                             [](const wafer::compiler::RankExecutable &rank) {
                               return countDTEIssues(rank.getModule()) > 0;
                             }));
  }

  const unsigned rankZeroCombinedCandidates =
      llvm::count_if(frontiers.front(), [](const Candidate &candidate) {
        return candidate.module &&
               candidate.bufferingKind ==
                   wafer::RankBufferingKind::StaticFixedSlot &&
               candidate.workerPlacementKind ==
                   wafer::RankWorkerPlacementKind::DisjointComponents &&
               countDTEIssues(*candidate.module) > 0 &&
               countNCCIssueWorkers(*candidate.module) >= 2;
      });
  EXPECT_GT(rankZeroCombinedCandidates, 0u);

  std::vector<wafer::compiler::detail::RankVariantMetadataFrontier> metadata;
  metadata.reserve(frontiers.size());
  for (const Frontier &frontier : frontiers) {
    wafer::compiler::detail::RankVariantMetadataFrontier rankMetadata;
    for (const Candidate &candidate : frontier)
      rankMetadata.push_back(
          {candidate.stableOrdinal, candidate.artifactKind,
           candidate.reservedBaseline, candidate.bufferingKind,
           candidate.bufferingPlanOrdinal, candidate.workerPlacementKind,
           candidate.workerPlacementPlanOrdinal});
    metadata.push_back(std::move(rankMetadata));
  }
  wafer::compiler::detail::WholeVariantAttemptPlan attemptPlan =
      wafer::compiler::detail::buildWholeVariantAttemptPlan(metadata, 16);
  ASSERT_TRUE(attemptPlan.isValid());
  unsigned completeCombinedCandidates = 0;
  unsigned plannedCombinedCandidates = 0;
  std::optional<std::vector<size_t>> fullActualCombinedIndices;
  for (const auto &[rankZeroIndex, candidate] :
       llvm::enumerate(frontiers.front())) {
    if (!candidate.module ||
        candidate.bufferingKind != wafer::RankBufferingKind::StaticFixedSlot ||
        candidate.workerPlacementKind !=
            wafer::RankWorkerPlacementKind::DisjointComponents ||
        countDTEIssues(*candidate.module) == 0 ||
        countNCCIssueWorkers(*candidate.module) < 2)
      continue;
    std::vector<size_t> indices{rankZeroIndex};
    bool complete = true;
    for (const Frontier &frontier : llvm::drop_begin(frontiers)) {
      auto corresponding = llvm::find_if(frontier, [&](const Candidate &other) {
        return other.module && other.stableOrdinal == candidate.stableOrdinal &&
               other.artifactKind == candidate.artifactKind &&
               other.bufferingKind == candidate.bufferingKind &&
               other.bufferingPlanOrdinal == candidate.bufferingPlanOrdinal &&
               other.workerPlacementKind == candidate.workerPlacementKind &&
               other.workerPlacementPlanOrdinal ==
                   candidate.workerPlacementPlanOrdinal &&
               countDTEIssues(*other.module) > 0 &&
               countNCCIssueWorkers(*other.module) >= 2;
      });
      if (corresponding == frontier.end()) {
        complete = false;
        break;
      }
      indices.push_back(
          static_cast<size_t>(std::distance(frontier.begin(), corresponding)));
    }
    if (!complete)
      continue;
    ++completeCombinedCandidates;
    const bool planned =
        llvm::is_contained(attemptPlan.workerPlacedCandidateIndices, indices) &&
        llvm::is_contained(attemptPlan.optimizedCandidateIndices, indices);
    plannedCombinedCandidates += planned ? 1u : 0u;
    if (planned && !fullActualCombinedIndices)
      fullActualCombinedIndices = indices;
  }
  EXPECT_GT(completeCombinedCandidates, 0u);
  EXPECT_GT(plannedCombinedCandidates, 0u);
  ASSERT_TRUE(fullActualCombinedIndices);

  // Attempt-plan reachability is asserted above. Isolate the exact complete
  // current-IR tuple so any remaining failure identifies this compound
  // candidate's actual qualification or target gate.
  std::vector<Frontier> qualificationFrontiers(frontiers.size());
  for (size_t rank = 0; rank < frontiers.size(); ++rank) {
    auto baseline = llvm::find_if(frontiers[rank], [](const Candidate &slot) {
      return slot.module && slot.reservedBaseline;
    });
    ASSERT_NE(baseline, frontiers[rank].end());
    const Candidate &combined =
        frontiers[rank][(*fullActualCombinedIndices)[rank]];
    qualificationFrontiers[rank].push_back(
        {mlir::cast<mlir::ModuleOp>(baseline->module.get()->clone()),
         baseline->stableOrdinal, baseline->artifactKind,
         baseline->reservedBaseline, baseline->bufferingKind,
         baseline->bufferingPlanOrdinal, baseline->workerPlacementKind,
         baseline->workerPlacementPlanOrdinal});
    qualificationFrontiers[rank].push_back(
        {mlir::cast<mlir::ModuleOp>(combined.module.get()->clone()),
         combined.stableOrdinal, combined.artifactKind,
         combined.reservedBaseline, combined.bufferingKind,
         combined.bufferingPlanOrdinal, combined.workerPlacementKind,
         combined.workerPlacementPlanOrdinal});
  }
  std::string compoundDiagnosticText;
  llvm::raw_string_ostream compoundDiagnostics(compoundDiagnosticText);
  auto compound = wafer::compiler::detail::selectAcceptedWholeVariant(
      qualificationFrontiers, program, *config, compoundDiagnostics,
      wafer::compiler::detail::WholeVariantSelectionMode::
          QualifyNoCResidentFixedSlotWorker);
  ASSERT_TRUE(mlir::succeeded(compound)) << compoundDiagnosticText;
}

} // namespace
