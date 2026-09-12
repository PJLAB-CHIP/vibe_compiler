//===- CostModelTest.cpp - Instruction program performance model -------===//

#include "Wafer/Analysis/Instr/CostModel.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitWaferDialects.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <array>
#include <limits>
#include <utility>

namespace {

using namespace wafer::analysis;

// Unit rates form a bounded arithmetic oracle. Real-scale current-IR and
// product coverage lives in ScheduleCostAnalysis and compiler/no-card tests.
SearchCostPolicy unitCostPolicy() {
  SearchCostPolicy policy;
  policy.ddrNominalBytesPerSecond = UINT64_C(1000000000000);
  policy.directionalNoCBytesPerSecond = UINT64_C(1000000000000);
  policy.dteEndpointBytesPerSecondEstimate = UINT64_C(1000000000000);
  policy.dteMessageStartupPicosecondsEstimate = 1;
  policy.dteFirstMessagePicosecondsEstimate = 1;
  policy.noCHopPicosecondsEstimate = 1;
  policy.instructionFixedPicosecondsEstimate = 1;
  policy.dteWaitedEventPicosecondsEstimate = 1;
  policy.nccJoinPicosecondsEstimate = 1;
  policy.nccParticipantWaitPicosecondsEstimate = 1;
  policy.f16Bf16NPULogicalOpsPerSecondPerTile = UINT64_C(1000000000000);
  policy.f16Bf16VectorLogicalOpsPerSecondPerTile = UINT64_C(1000000000000);
  policy.f32VectorLogicalOpsPerSecondPerTile = UINT64_C(1000000000000);
  policy.spmExplicitMovementBytesPerSecondPerTileEstimate =
      UINT64_C(1000000000000);
  return policy;
}

void expectCoarse(const SearchObjective &objective, bool saturated = false) {
  const auto *known = std::get_if<KnownSearchObjective>(&objective);
  ASSERT_NE(known, nullptr);
  EXPECT_TRUE(known->usesCoarseEstimate);
  EXPECT_GT(known->estimatedDurationPicoseconds, 0u);
  if (saturated) {
    EXPECT_EQ(known->estimatedDurationPicoseconds,
              std::numeric_limits<uint64_t>::max());
  }
}

TEST(CostModelTest, ActualDependenciesAndJoinsChangeOverlapWithIdenticalWork) {
  mlir::DialectRegistry registry;
  wafer::registerWaferCoreDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto cohort = SearchCostCohort::create(unitCostPolicy());
  ASSERT_TRUE(mlir::succeeded(cohort));
  for (int64_t extent : {1024, 1025, 1031}) {
    for (int64_t trips : {1, 32, 33}) {
      SCOPED_TRACE(::testing::Message() << extent << "/" << trips);
      std::array<uint64_t, 6> times{};
      std::array<uint64_t, 6> serialized{};
      for (unsigned variant = 0; variant != 6; ++variant) {
        std::string type = "memref<2x" + std::to_string(extent) + "x64xf16";
        const std::string ddr = type + ", #wafer.memory<ddr, tensor>>";
        const std::string spm = type + ", #wafer.memory<spm, tensor>>";
        std::string text;
        llvm::raw_string_ostream out(text);
        out << "module { wafer.target.topology @topology {card_grid = "
               "array<i64: 1, 1>, "
            << "card_interconnect = \"mesh\", tile_grid = array<i64: 4, 4>, "
               "unavailable_tiles = array<i64>}\n"
            << "func.func @main(%" << (variant < 3 ? "input" : "external")
            << ": " << ddr << ") {\n";
        if (variant >= 3)
          out << "%result = wafer.tile.region(%external : " << ddr << ") -> ("
              << ddr << ") {\n^bb0(%input: " << ddr << "):\n";
        for (auto [name, offset] :
             {std::pair{"a", 65536}, {"b", 1048576}, {"c", 2097152}})
          out << "%" << name
              << " = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<"
              << offset << ">} : " << spm << "\n";
        out << "%c0 = arith.constant 0 : index\n%c1 = arith.constant 1 : "
               "index\n"
            << "%end = arith.constant " << trips << " : index\n"
            << "scf.for %iv = %c0 to %end step %c1 {\n";
        for (auto name : {"b", "a"})
          out << "wafer.instr.rdma %input to %" << name
              << " {byte_count = " << 256 * extent
              << " : i64, inner_bytes = " << 256 * extent
              << " : i64, src_strides = array<i64: 0, 0, 0>, "
              << "src_iterations = array<i64: 1, 1, 1>} : " << ddr << " to "
              << spm << "\n";
        if (variant % 3 == 2)
          out << "wafer.instr.ncc_join [0]\n";
        const auto *source = variant % 3 == 1 ? "a" : "b";
        out << "wafer.instr.elementwise #wafer.instr_elementwise_kind<add> %"
            << source << ", %" << source << " into %c : " << spm << ", " << spm
            << " into " << spm << "\n}\n"
            << "wafer.instr.ncc_join [0]\n";
        if (variant >= 3)
          out << "wafer.tile.yield %input : " << ddr << "\n}\n";
        out << "return\n}}\n";
        auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
        ASSERT_TRUE(module) << text;
        llvm::SmallVector<TileInstructionProgram> programs{
            {wafer::TileId(0), *module}};
        auto cost = analyzeInstructionProgramAggregateCost(
            programs, wafer::getTargetMemoryPolicy());
        auto actual = deriveSearchObjective(cost, *cohort, programs);
        auto aggregate = deriveSearchObjective(cost, *cohort);
        const auto *estimate = std::get_if<KnownSearchObjective>(&actual);
        const auto *serial = std::get_if<KnownSearchObjective>(&aggregate);
        ASSERT_TRUE(estimate && serial);
        EXPECT_FALSE(estimate->usesCoarseEstimate);
        times[variant] = estimate->estimatedDurationPicoseconds;
        serialized[variant] = serial->estimatedDurationPicoseconds;
      }
      EXPECT_EQ(serialized[0], serialized[1]);
      EXPECT_LT(times[0], times[1]);
      EXPECT_LT(times[0], times[2]);
      EXPECT_LE(times[0], serialized[0]);
      for (unsigned variant = 0; variant != 3; ++variant) {
        EXPECT_EQ(times[variant], times[variant + 3]);
        EXPECT_EQ(serialized[variant], serialized[variant + 3]);
      }
    }
  }
}

TEST(CostModelTest, LoopCarriedIndexRemainsExactForFollowingLoops) {
  mlir::DialectRegistry registry;
  wafer::registerWaferCoreDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto cohort = SearchCostCohort::create(unitCostPolicy());
  ASSERT_TRUE(mlir::succeeded(cohort));
  for (int64_t extent : {1024, 1025, 1031})
    for (int64_t trips : {32, 33}) {
      std::array<uint64_t, 2> times{};
      for (bool carried : {false, true}) {
        std::string type = "memref<2x" + std::to_string(extent) + "x64xf16";
        const std::string ddr = type + ", #wafer.memory<ddr, tensor>>";
        const std::string spm = type + ", #wafer.memory<spm, tensor>>";
        std::string text;
        llvm::raw_string_ostream out(text);
        out << "module { func.func @main(%input: " << ddr << ") {\n"
            << "%local = memref.alloc() {wafer.spm.offset = "
               "#wafer.spm_offset<65536>} : "
            << spm << "\n"
            << "%c0 = arith.constant 0 : index\n"
            << "%c1 = arith.constant 1 : index\n"
            << "%end = arith.constant " << trips << " : index\n";
        if (carried)
          out << "%count = scf.for %iv = %c0 to %end step %c1 "
                 "iter_args(%index = %c0) -> index {\n"
                 "%next = arith.addi %index, %c1 : index\n"
                 "scf.yield %next : index\n}\n";
        out << "scf.for %iv = %c0 to %" << (carried ? "count" : "end")
            << " step %c1 {\nwafer.instr.rdma %input to %local "
            << "{byte_count = " << 256 * extent
            << " : i64, inner_bytes = " << 256 * extent
            << " : i64, src_strides = array<i64: 0, 0, 0>, "
               "src_iterations = array<i64: 1, 1, 1>} : "
            << ddr << " to " << spm
            << "\n}\nwafer.instr.ncc_join [0]\nreturn\n}}\n";
        auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
        ASSERT_TRUE(module) << text;
        // Both execute the same static instruction workload. Only the SSA
        // spelling of the second loop's bound differs.
        auto constant = module->clone();
        mlir::OwningOpRef<mlir::ModuleOp> staticOwner(constant);
        if (carried) {
          auto function = *constant.getOps<mlir::func::FuncOp>().begin();
          auto loops = function.getOps<mlir::scf::ForOp>();
          auto first = *loops.begin();
          first.getResult(0).replaceAllUsesWith(first.getUpperBound());
          first.erase();
        }
        llvm::SmallVector<TileInstructionProgram> staticPrograms{
            {wafer::TileId(0), constant}};
        auto cost = analyzeInstructionProgramAggregateCost(
            staticPrograms, wafer::getTargetMemoryPolicy());
        llvm::SmallVector<TileInstructionProgram> programs{
            {wafer::TileId(0), *module}};
        auto objective = deriveSearchObjective(cost, *cohort, programs);
        const auto *estimate = std::get_if<KnownSearchObjective>(&objective);
        ASSERT_TRUE(estimate);
        EXPECT_FALSE(estimate->usesCoarseEstimate);
        times[carried] = estimate->estimatedDurationPicoseconds;
      }
      EXPECT_EQ(times[0], times[1]);
    }
}

TEST(CostModelTest, TwoActualSlotsOverlapIdenticalLoadsAndComputations) {
  mlir::DialectRegistry registry;
  wafer::registerWaferCoreDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto policy = unitCostPolicy();
  policy.f16Bf16VectorLogicalOpsPerSecondPerTile = UINT64_C(125000000000);
  auto cohort = SearchCostCohort::create(policy);
  ASSERT_TRUE(mlir::succeeded(cohort));
  for (int64_t extent : {1024, 1025, 1031})
    for (int64_t trips : {32, 33}) {
      SCOPED_TRACE(::testing::Message() << extent << "/" << trips);
      std::array<uint64_t, 2> times{}, bytes{}, operations{}, aggregate{};
      for (unsigned pipeline = 0; pipeline != 2; ++pipeline) {
        const std::string shape =
            "memref<2x" + std::to_string(extent) + "x64xf16";
        const std::string ddr = shape + ", #wafer.memory<ddr, tensor>>";
        const std::string spm = shape + ", #wafer.memory<spm, tensor>>";
        std::string text;
        llvm::raw_string_ostream out(text);
        out << "module { func.func @main(%input: " << ddr << ") {\n";
        for (auto [name, offset] :
             {std::pair{"a", 65536}, {"b", 1048576}, {"c", 2097152}})
          out << "%" << name << " = memref.alloc() {wafer.spm.offset = "
              << "#wafer.spm_offset<" << offset << ">} : " << spm << "\n";
        auto load = [&](llvm::StringRef to) {
          out << "wafer.instr.rdma %input to %" << to
              << " {byte_count = " << 256 * extent
              << " : i64, inner_bytes = " << 256 * extent
              << " : i64, src_strides = array<i64: 0, 0, 0>, "
                 "src_iterations = array<i64: 1, 1, 1>} : "
              << ddr << " to " << spm << "\n";
        };
        auto compute = [&](llvm::StringRef from) {
          out << "wafer.instr.elementwise <add> %" << from << ", %" << from
              << " into %c : " << spm << ", " << spm << " into " << spm << "\n";
        };
        out << "%c0 = arith.constant 0 : index\n%c1 = arith.constant 1 : "
               "index\n"
               "%c2 = arith.constant 2 : index\n%end = arith.constant "
            << trips - pipeline << " : index\n";
        if (pipeline)
          load("a");
        out << "scf.for %iv = %c0 to %end step %c1 {\n";
        if (pipeline) {
          out << "%rem = arith.remui %iv, %c2 : index\n"
                 "%even = arith.cmpi eq, %rem, %c0 : index\n"
                 "%current = arith.select %even, %a, %b : "
              << spm << "\n"
              << "%next = arith.select %even, %b, %a : " << spm << "\n";
          load("next");
          compute("current");
        } else {
          load("a");
          compute("a");
        }
        out << "}\n";
        if (pipeline)
          compute(trips % 2 ? "a" : "b");
        out << "wafer.instr.ncc_join [0]\nreturn\n}}\n";
        auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
        ASSERT_TRUE(module) << text;
        llvm::SmallVector<TileInstructionProgram> programs{
            {wafer::TileId(0), *module}};
        auto cost = analyzeInstructionProgramAggregateCost(
            programs, wafer::getTargetMemoryPolicy());
        auto objective = deriveSearchObjective(cost, *cohort, programs);
        const auto *known = std::get_if<KnownSearchObjective>(&objective);
        ASSERT_TRUE(known);
        EXPECT_FALSE(known->usesCoarseEstimate);
        times[pipeline] = known->estimatedDurationPicoseconds;
        bytes[pipeline] = cost.aggregateDDRReadBytes.value;
        operations[pipeline] =
            cost.aggregateCompute.vectorF16Bf16LogicalOps.value;
        aggregate[pipeline] =
            std::get<KnownSearchObjective>(deriveSearchObjective(cost, *cohort))
                .estimatedDurationPicoseconds;
      }
      EXPECT_EQ(bytes[0], bytes[1]);
      EXPECT_EQ(operations[0], operations[1]);
      EXPECT_EQ(aggregate[0], aggregate[1]);
      EXPECT_LT(times[1], times[0]);
    }
}

TEST(CostModelTest, ExplicitCohortDerivesResourceTermsUnknownAndOverflow) {
  std::string failureReason;
  SearchCostPolicy policy = unitCostPolicy();
  auto cohort = SearchCostCohort::create(policy, &failureReason);
  ASSERT_TRUE(mlir::succeeded(cohort)) << failureReason;
  policy.ddrNominalBytesPerSecond = 0;
  EXPECT_TRUE(mlir::failed(SearchCostCohort::create(policy, &failureReason)));
  policy = unitCostPolicy();
  policy.profileIdentity = 0;
  EXPECT_TRUE(mlir::failed(SearchCostCohort::create(policy, &failureReason)));

  InstructionProgramAggregateCost cost;
  cost.aggregateInstructionCount.value = 1;
  cost.aggregateDDRReadBytes.value = 2;
  cost.aggregateDDRWriteBytes.value = 3;
  cost.aggregateNoC.staticIssueSiteCount.value = 1;
  cost.modeledNoCRoute.peakDirectedLinkByteDemand.value = 4;
  cost.maximumTileNoCTransmitBytes.value = 10;
  cost.maximumTileNoCTransmitMessageCount.value = 2;
  cost.minimumHopMessageDemand.value = 3;
  SearchObjective known = deriveSearchObjective(cost, *cohort);
  const auto *knownValue = std::get_if<KnownSearchObjective>(&known);
  ASSERT_NE(knownValue, nullptr);
  EXPECT_EQ(knownValue->durations.instructionControlPicoseconds, 1u);
  EXPECT_EQ(knownValue->durations.ddrPicoseconds, 5u);
  EXPECT_EQ(knownValue->durations.nocPicoseconds, 4u);
  EXPECT_EQ(knownValue->durations.dteEndpointPicoseconds, 10u);
  EXPECT_EQ(knownValue->durations.dteStartupPicoseconds, 2u);
  EXPECT_EQ(knownValue->durations.nocHopPicoseconds, 3u);
  EXPECT_EQ(knownValue->estimatedDurationPicoseconds, 21u);
  EXPECT_TRUE(std::holds_alternative<UnknownSearchObjective>(
      deriveSearchObjective(cost, std::nullopt)));
  cost.minimumHopMessageDemand.knowledge = ScheduleCostKnowledge::Unavailable;
  expectCoarse(deriveSearchObjective(cost, *cohort));
  cost.minimumHopMessageDemand.knowledge = ScheduleCostKnowledge::Known;
  cost.aggregateInstructionCount.knowledge = ScheduleCostKnowledge::Unavailable;
  expectCoarse(deriveSearchObjective(cost, *cohort));
  cost.aggregateInstructionCount.knowledge = ScheduleCostKnowledge::Known;
  cost.aggregateInstructionCount.value = std::numeric_limits<uint64_t>::max();
  SearchCostPolicy overflowPolicy = unitCostPolicy();
  overflowPolicy.instructionFixedPicosecondsEstimate = 2;
  auto overflowCohort = *SearchCostCohort::create(overflowPolicy);
  expectCoarse(deriveSearchObjective(cost, overflowCohort), true);
}

TEST(CostModelTest, DTEStartupSeparatesFirstMessageAndChecksCohortAndOverflow) {
  // Bounded arithmetic oracle; actual multi-Tile IR is covered by the
  // communication candidate tests and the guarded board timing probe.
  SearchCostPolicy policy;
  auto cohort = *SearchCostCohort::create(policy);
  InstructionProgramAggregateCost cost;
  for (auto [count, expected] : std::array<std::pair<uint64_t, uint64_t>, 4>{
           {{0, 0}, {1, 13'000'000}, {15, 34'000'000}, {32, 59'500'000}}}) {
    SCOPED_TRACE(count);
    cost.maximumTileNoCTransmitMessageCount.value = count;
    SearchObjective objective = deriveSearchObjective(cost, cohort);
    const auto *known = std::get_if<KnownSearchObjective>(&objective);
    ASSERT_NE(known, nullptr);
    EXPECT_EQ(known->durations.dteStartupPicoseconds, expected);
  }
  SearchObjective reference = deriveSearchObjective(cost, cohort);
  ++policy.dteFirstMessagePicosecondsEstimate;
  auto otherCohort = *SearchCostCohort::create(policy);
  EXPECT_EQ(compareSearchObjectives(reference,
                                    deriveSearchObjective(cost, otherCohort)),
            SearchObjectiveComparison::Incomparable);
  policy.dteFirstMessagePicosecondsEstimate = 0;
  EXPECT_TRUE(mlir::failed(SearchCostCohort::create(policy)));

  policy = unitCostPolicy();
  policy.dteFirstMessagePicosecondsEstimate =
      std::numeric_limits<uint64_t>::max();
  cost.maximumTileNoCTransmitMessageCount.value = 2;
  expectCoarse(deriveSearchObjective(cost, *SearchCostCohort::create(policy)),
               true);
  policy = unitCostPolicy();
  policy.dteMessageStartupPicosecondsEstimate = 2;
  cost.maximumTileNoCTransmitMessageCount.value =
      std::numeric_limits<uint64_t>::max();
  expectCoarse(deriveSearchObjective(cost, *SearchCostCohort::create(policy)),
               true);
}
TEST(CostModelTest, NCCControlCountsCallsAndParticipantsOnTheSameTile) {
  SearchCostPolicy policy;
  auto cohort = *SearchCostCohort::create(policy);
  InstructionProgramAggregateCost cost;
  for (uint64_t participants : {2, 3}) {
    cost.aggregateNCCJoinCount.value = 1;
    cost.aggregateNCCParticipantWaitCount.value = participants;
    SearchObjective combined = deriveSearchObjective(cost, cohort);
    EXPECT_EQ(std::get<KnownSearchObjective>(combined)
                  .durations.nccWaitControlPicoseconds,
              participants == 2 ? 230'000u : 275'000u);
    cost.aggregateNCCJoinCount.value = participants;
    SearchObjective split = deriveSearchObjective(cost, cohort);
    EXPECT_EQ(std::get<KnownSearchObjective>(split)
                  .durations.nccWaitControlPicoseconds,
              participants == 2 ? 370'000u : 555'000u);
    EXPECT_EQ(compareSearchObjectives(combined, split),
              SearchObjectiveComparison::Better);
  }
  cost.tileCosts.resize(2);
  cost.tileCosts[0].nccJoinCount.value = 2;
  cost.tileCosts[0].nccParticipantWaitCount.value = 2;
  cost.tileCosts[1].nccJoinCount.value = 1;
  cost.tileCosts[1].nccParticipantWaitCount.value = 3;
  SearchObjective known = deriveSearchObjective(cost, cohort);
  EXPECT_EQ(
      std::get<KnownSearchObjective>(known).durations.nccWaitControlPicoseconds,
      370'000u);
  ++policy.nccJoinPicosecondsEstimate;
  EXPECT_EQ(compareSearchObjectives(
                known,
                deriveSearchObjective(cost, *SearchCostCohort::create(policy))),
            SearchObjectiveComparison::Incomparable);
  cost.tileCosts[0].nccJoinCount.knowledge = ScheduleCostKnowledge::Unavailable;
  expectCoarse(deriveSearchObjective(cost, cohort));
  cost.tileCosts[0].nccJoinCount.knowledge = ScheduleCostKnowledge::Known;
  cost.tileCosts[0].nccJoinCount.value = std::numeric_limits<uint64_t>::max();
  expectCoarse(deriveSearchObjective(cost, cohort), true);
}
TEST(CostModelTest, ResourceTradeoffsUseScalarTime) {
  auto cohort = *SearchCostCohort::create(unitCostPolicy());
  auto makeCost = [](uint64_t ne, uint64_t vector) {
    InstructionProgramAggregateCost cost;
    cost.aggregateInstructionCount.value = 10;
    cost.aggregateCompute.npuF16Bf16LogicalOps.value = ne;
    cost.aggregateCompute.vectorF16Bf16LogicalOps.value = vector;
    return cost;
  };

  SearchObjective neHeavy = deriveSearchObjective(makeCost(1024, 0), cohort);
  SearchObjective vectorHeavy =
      deriveSearchObjective(makeCost(0, 1024), cohort);
  EXPECT_EQ(compareSearchObjectives(neHeavy, vectorHeavy),
            SearchObjectiveComparison::Equivalent);

  SearchObjective lessOfBoth =
      deriveSearchObjective(makeCost(1024, 1024), cohort);
  SearchObjective moreOfBoth =
      deriveSearchObjective(makeCost(1025, 1031), cohort);
  EXPECT_EQ(compareSearchObjectives(lessOfBoth, moreOfBoth),
            SearchObjectiveComparison::Better);
  const auto *known = std::get_if<KnownSearchObjective>(&lessOfBoth);
  ASSERT_NE(known, nullptr);
  EXPECT_EQ(known->durations.neF16Bf16Picoseconds, 1024u);
  EXPECT_EQ(known->durations.vectorF16Bf16Picoseconds, 1024u);
  EXPECT_EQ(known->durations.instructionControlPicoseconds, 10u);
}

TEST(CostModelTest, DDRAndPeerCanBothWinWithTheSameProfile) {
  auto cohort = *SearchCostCohort::create(SearchCostPolicy{});
  InstructionProgramAggregateCost ddr, peer;
  peer.maximumTileNoCTransmitMessageCount.value = 1;
  peer.maximumTileNoCTransmitBytes.value = 2048;
  ddr.aggregateDDRReadBytes.value = 2048;
  EXPECT_EQ(compareSearchObjectives(deriveSearchObjective(ddr, cohort),
                                    deriveSearchObjective(peer, cohort)),
            SearchObjectiveComparison::Better);
  ddr.aggregateDDRReadBytes.value = 16 * 1024 * 1024;
  EXPECT_EQ(compareSearchObjectives(deriveSearchObjective(ddr, cohort),
                                    deriveSearchObjective(peer, cohort)),
            SearchObjectiveComparison::Worse);
}

TEST(CostModelTest, CombineWorkOnEachTileBeforeTakingTheMaximum) {
  auto cohort = *SearchCostCohort::create(unitCostPolicy());
  InstructionProgramAggregateCost split;
  split.tileCosts.resize(2);
  split.tileCosts[0].compute.npuF16Bf16LogicalOps.value = 1024;
  split.tileCosts[1].compute.vectorF32LogicalOps.value = 1031;
  auto joined = split;
  joined.tileCosts[0].compute.vectorF32LogicalOps.value = 1031;
  joined.tileCosts[1].compute.vectorF32LogicalOps.value = 0;
  auto left = deriveSearchObjective(split, cohort);
  auto right = deriveSearchObjective(joined, cohort);
  EXPECT_EQ(std::get<KnownSearchObjective>(left).estimatedDurationPicoseconds,
            1031u);
  EXPECT_EQ(std::get<KnownSearchObjective>(right).estimatedDurationPicoseconds,
            2055u);
  EXPECT_EQ(compareSearchObjectives(left, right),
            SearchObjectiveComparison::Better);
}

TEST(CostModelTest, SharedDDRAndPayloadSerializationAreCountedOnce) {
  auto cohort = *SearchCostCohort::create(unitCostPolicy());
  InstructionProgramAggregateCost cost;
  cost.tileCosts.resize(2);
  for (auto &tile : cost.tileCosts) {
    tile.noc.aggregateTransmitBytes.value = 1024;
    tile.ddrReadBytes.value = 1024;
  }
  cost.aggregateNoC.staticIssueSiteCount.value = 2;
  cost.aggregateDDRReadBytes.value = 2048;
  cost.maximumTileNoCTransmitBytes.value = 1024;
  cost.modeledNoCRoute.peakDirectedLinkByteDemand.value = 1031;
  auto result = deriveSearchObjective(cost, cohort);
  EXPECT_EQ(std::get<KnownSearchObjective>(result).estimatedDurationPicoseconds,
            3079u); // Shared DDR 2048 plus payload bottleneck 1031.
}

TEST(CostModelTest, MissingWorkStillRanksAndStorageCannotVetoFasterTime) {
  auto cohort = *SearchCostCohort::create(unitCostPolicy());
  InstructionProgramAggregateCost cost;
  cost.aggregateInstructionCount.knowledge = ScheduleCostKnowledge::Unavailable;
  cost.aggregateWork.instructions.upperBound.knowledge =
      ScheduleCostKnowledge::Unavailable;
  cost.aggregateWork.instructions.staticSites.value = 3;
  auto coarse = deriveSearchObjective(cost, cohort);
  expectCoarse(coarse);
  EXPECT_EQ(std::get<KnownSearchObjective>(coarse).estimatedDurationPicoseconds,
            39'000'000u);
  cost.aggregateWork.instructions.upperBound = {5};
  auto larger = deriveSearchObjective(cost, cohort);
  EXPECT_EQ(compareSearchObjectives(coarse, larger),
            SearchObjectiveComparison::Better);

  InstructionProgramAggregateCost fast, slow;
  fast.aggregateInstructionCount.value = 1024;
  fast.maximumTileSPMHighWaterBytes.value = 100000;
  slow.aggregateInstructionCount.value = 1025;
  EXPECT_EQ(compareSearchObjectives(deriveSearchObjective(fast, cohort),
                                    deriveSearchObjective(slow, cohort)),
            SearchObjectiveComparison::Better);
  fast.aggregateCompute.npuOtherLogicalOps.value = 1024;
  expectCoarse(deriveSearchObjective(fast, cohort));
}

} // namespace
