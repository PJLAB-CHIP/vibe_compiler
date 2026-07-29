//===- TargetSchedulingCapabilityTest.cpp - Scheduling contract tests ---===//

#include "Wafer/Target/TargetSchedulingCapability.h"

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/IR/WaferInterfaces.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/Error.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

using wafer::TargetSchedulingCapabilityRegistry;
using wafer::TargetSchedulingCapabilityState;
using wafer::TargetSchedulingCompletionKind;
using wafer::TargetSchedulingDrainEvidence;
using wafer::TargetSchedulingEngine;
using wafer::TargetSchedulingEngineMask;
using wafer::TargetSchedulingLegalityRow;
using wafer::TargetSchedulingMechanism;
using wafer::TargetSchedulingOverlapEvidence;
using wafer::TargetSchedulingProfitabilityRow;
using wafer::TargetSchedulingProfitabilityScope;
using wafer::TargetSchedulingWindowPredicate;
using wafer::TargetSchedulingWindowQuery;
using wafer::TargetSchedulingWorkerRelation;

constexpr wafer::TargetProfileId kV1 =
    wafer::TargetProfileId::waferTx81SingleCardKernelV1();
constexpr wafer::TargetProfileId kV2 =
    wafer::TargetProfileId::waferTx81SingleCardKernelV2();
constexpr wafer::TargetProfileId kV3 =
    wafer::TargetProfileId::waferTx81SingleCardKernelV3();

constexpr TargetSchedulingEngineMask engine(TargetSchedulingEngine value) {
  return wafer::targetSchedulingEngineBit(value);
}

constexpr TargetSchedulingEngineMask kCTRdma =
    engine(TargetSchedulingEngine::CT) | engine(TargetSchedulingEngine::RDMA);
constexpr TargetSchedulingEngineMask kCTRdmaWdma =
    kCTRdma | engine(TargetSchedulingEngine::WDMA);
constexpr TargetSchedulingEngineMask kCTDTE =
    engine(TargetSchedulingEngine::CT) | engine(TargetSchedulingEngine::DTE);

static TargetSchedulingWindowPredicate makePredicate(
    TargetSchedulingMechanism mechanism, TargetSchedulingEngineMask engines,
    TargetSchedulingWorkerRelation workerRelation,
    TargetSchedulingCompletionKind completion, uint64_t minimumBytes = 0,
    uint64_t maximumBytes = std::numeric_limits<uint64_t>::max()) {
  TargetSchedulingWindowPredicate predicate{mechanism, engines, workerRelation,
                                            completion};
  predicate.geometry.minimumPayloadBytes = minimumBytes;
  predicate.geometry.maximumPayloadBytes = maximumBytes;
  predicate.geometry.maximumVisibleBuffers = 32;
  predicate.geometry.maximumVisibleTokens = 32;
  return predicate;
}

static TargetSchedulingWindowPredicate makeSameWorkerFixed(
    TargetSchedulingEngineMask engines, uint64_t minimumBytes = 0,
    uint64_t maximumBytes = std::numeric_limits<uint64_t>::max()) {
  return makePredicate(TargetSchedulingMechanism::StaticFixedSlot, engines,
                       TargetSchedulingWorkerRelation::SameNCCWorker,
                       TargetSchedulingCompletionKind::SameWorkerIssueOrder,
                       minimumBytes, maximumBytes);
}

static TargetSchedulingWindowQuery
makeQuery(wafer::TargetProfileId profile, TargetSchedulingMechanism mechanism,
          TargetSchedulingEngineMask engines,
          TargetSchedulingWorkerRelation workerRelation,
          TargetSchedulingCompletionKind completion,
          uint64_t payloadBytes = 16) {
  TargetSchedulingWindowQuery query(profile, mechanism);
  query.engines = engines;
  query.workerRelation = workerRelation;
  query.completion = completion;
  query.maximumPayloadBytes = payloadBytes;
  query.visibleBufferCount = 2;
  query.visibleTokenCount = 0;
  return query;
}

template <typename T>
static std::string takeExpectedError(llvm::Expected<T> value) {
  if (value) {
    ADD_FAILURE() << "expected contract construction/query to fail";
    return {};
  }
  return llvm::toString(value.takeError());
}

TEST(TargetSchedulingCapabilityTest,
     CompilerShippedProfilesAreVersionedAndKeepUnknownNonIllegal) {
  auto v1 = wafer::getTargetSchedulingCapabilityRegistry(kV1);
  ASSERT_TRUE(static_cast<bool>(v1)) << llvm::toString(v1.takeError());
  auto v2 = wafer::getTargetSchedulingCapabilityRegistry(kV2);
  ASSERT_TRUE(static_cast<bool>(v2)) << llvm::toString(v2.takeError());
  auto v3 = wafer::getTargetSchedulingCapabilityRegistry(kV3);
  ASSERT_TRUE(static_cast<bool>(v3)) << llvm::toString(v3.takeError());
  EXPECT_EQ(v1->getContractVersion(), 1u);
  EXPECT_EQ(v2->getContractVersion(), 1u);
  EXPECT_EQ(v3->getContractVersion(), 3u);
  EXPECT_EQ(v1->getTargetProfile(), kV1);
  EXPECT_EQ(v2->getTargetProfile(), kV2);
  EXPECT_EQ(v3->getTargetProfile(), kV3);

  auto sameWorker =
      makeQuery(kV1, TargetSchedulingMechanism::StaticFixedSlot, kCTRdma,
                TargetSchedulingWorkerRelation::SameNCCWorker,
                TargetSchedulingCompletionKind::SameWorkerIssueOrder);
  auto sameDecision = v1->query(sameWorker);
  ASSERT_TRUE(static_cast<bool>(sameDecision))
      << llvm::toString(sameDecision.takeError());
  EXPECT_EQ(sameDecision->legality, TargetSchedulingCapabilityState::Supported);
  EXPECT_EQ(sameDecision->profitability.overlap,
            TargetSchedulingOverlapEvidence::QualifiedOverlap);

  auto crossV1 =
      makeQuery(kV1, TargetSchedulingMechanism::StaticFixedSlot, kCTRdma,
                TargetSchedulingWorkerRelation::CrossNCCWorkers,
                TargetSchedulingCompletionKind::ParticipantJoin);
  auto crossV1Decision = v1->query(crossV1);
  ASSERT_TRUE(static_cast<bool>(crossV1Decision))
      << llvm::toString(crossV1Decision.takeError());
  EXPECT_EQ(crossV1Decision->legality,
            TargetSchedulingCapabilityState::Unknown);
  EXPECT_TRUE(crossV1Decision->profitability.isEntirelyUnknown());

  auto crossV2 = crossV1;
  crossV2.targetProfile = kV2;
  auto crossV2Decision = v2->query(crossV2);
  ASSERT_TRUE(static_cast<bool>(crossV2Decision))
      << llvm::toString(crossV2Decision.takeError());
  EXPECT_EQ(crossV2Decision->legality,
            TargetSchedulingCapabilityState::Unknown);
  EXPECT_TRUE(crossV2Decision->profitability.isEntirelyUnknown());

  auto crossV3 = crossV1;
  crossV3.targetProfile = kV3;
  auto crossV3Decision = v3->query(crossV3);
  ASSERT_TRUE(static_cast<bool>(crossV3Decision))
      << llvm::toString(crossV3Decision.takeError());
  EXPECT_EQ(crossV3Decision->legality,
            TargetSchedulingCapabilityState::Supported);
  // A supported legality row does not manufacture profitability evidence.
  EXPECT_TRUE(crossV3Decision->profitability.isEntirelyUnknown());

  sameWorker.geometryKnown = false;
  auto unknownGeometry = v1->query(sameWorker);
  ASSERT_TRUE(static_cast<bool>(unknownGeometry))
      << llvm::toString(unknownGeometry.takeError());
  EXPECT_EQ(unknownGeometry->legality,
            TargetSchedulingCapabilityState::Unknown);
}

TEST(TargetSchedulingCapabilityTest,
     ExactPairEvidenceDoesNotLeakIntoExactGroups) {
  auto registry = wafer::getTargetSchedulingCapabilityRegistry(kV1);
  ASSERT_TRUE(static_cast<bool>(registry))
      << llvm::toString(registry.takeError());

  auto pair = registry->query(
      makeQuery(kV1, TargetSchedulingMechanism::StaticFixedSlot, kCTRdma,
                TargetSchedulingWorkerRelation::SameNCCWorker,
                TargetSchedulingCompletionKind::SameWorkerIssueOrder));
  ASSERT_TRUE(static_cast<bool>(pair)) << llvm::toString(pair.takeError());
  EXPECT_EQ(pair->profitability.overlap,
            TargetSchedulingOverlapEvidence::QualifiedOverlap);
  EXPECT_EQ(pair->profitability.drain,
            TargetSchedulingDrainEvidence::QualifiedDrainElision);

  auto group = registry->query(
      makeQuery(kV1, TargetSchedulingMechanism::StaticFixedSlot, kCTRdmaWdma,
                TargetSchedulingWorkerRelation::SameNCCWorker,
                TargetSchedulingCompletionKind::SameWorkerIssueOrder));
  ASSERT_TRUE(static_cast<bool>(group)) << llvm::toString(group.takeError());
  EXPECT_EQ(group->legality, TargetSchedulingCapabilityState::Supported);
  EXPECT_EQ(group->profitability.overlap,
            TargetSchedulingOverlapEvidence::Unknown);
  EXPECT_EQ(group->profitability.drain,
            TargetSchedulingDrainEvidence::QualifiedDrainElision);

  unsigned pairRows = 0;
  unsigned groupRows = 0;
  for (const TargetSchedulingProfitabilityRow &row :
       registry->getProfitabilityRows()) {
    pairRows += row.scope == TargetSchedulingProfitabilityScope::ExactPair;
    groupRows += row.scope == TargetSchedulingProfitabilityScope::ExactGroup;
  }
  EXPECT_EQ(pairRows, 10u);
  EXPECT_EQ(groupRows, 16u);
}

TEST(TargetSchedulingCapabilityTest,
     MixedDTEFamiliesRequireV3AndRemainProfitabilityUnknown) {
  auto v1 = wafer::getTargetSchedulingCapabilityRegistry(kV1);
  auto v2 = wafer::getTargetSchedulingCapabilityRegistry(kV2);
  auto v3 = wafer::getTargetSchedulingCapabilityRegistry(kV3);
  ASSERT_TRUE(static_cast<bool>(v1)) << llvm::toString(v1.takeError());
  ASSERT_TRUE(static_cast<bool>(v2)) << llvm::toString(v2.takeError());
  ASSERT_TRUE(static_cast<bool>(v3)) << llvm::toString(v3.takeError());

  for (TargetSchedulingMechanism mechanism :
       {TargetSchedulingMechanism::StaticFixedSlot,
        TargetSchedulingMechanism::DirectDTEOverlap}) {
    auto v1Query = makeQuery(
        kV1, mechanism, kCTDTE, TargetSchedulingWorkerRelation::MixedNCCAndDTE,
        TargetSchedulingCompletionKind::ParticipantJoinAndExactEvent);
    auto v1Decision = v1->query(v1Query);
    ASSERT_TRUE(static_cast<bool>(v1Decision))
        << llvm::toString(v1Decision.takeError());
    EXPECT_EQ(v1Decision->legality, TargetSchedulingCapabilityState::Unknown);

    auto v2Query = v1Query;
    v2Query.targetProfile = kV2;
    auto v2Decision = v2->query(v2Query);
    ASSERT_TRUE(static_cast<bool>(v2Decision))
        << llvm::toString(v2Decision.takeError());
    EXPECT_EQ(v2Decision->legality, TargetSchedulingCapabilityState::Unknown);
    EXPECT_TRUE(v2Decision->profitability.isEntirelyUnknown());

    auto v3Query = v1Query;
    v3Query.targetProfile = kV3;
    auto v3Decision = v3->query(v3Query);
    ASSERT_TRUE(static_cast<bool>(v3Decision))
        << llvm::toString(v3Decision.takeError());
    EXPECT_EQ(v3Decision->legality, TargetSchedulingCapabilityState::Supported);
    EXPECT_TRUE(v3Decision->profitability.isEntirelyUnknown());
  }
}

TEST(TargetSchedulingCapabilityTest,
     WorkerPlacementUsesExactV3PairEvidenceWithoutGeneralizing) {
  auto registry = wafer::getTargetSchedulingCapabilityRegistry(kV3);
  ASSERT_TRUE(static_cast<bool>(registry))
      << llvm::toString(registry.takeError());

  auto qualifiedPair = registry->query(
      makeQuery(kV3, TargetSchedulingMechanism::WorkerPlacement, kCTRdma,
                TargetSchedulingWorkerRelation::CrossNCCWorkers,
                TargetSchedulingCompletionKind::ParticipantJoin));
  ASSERT_TRUE(static_cast<bool>(qualifiedPair))
      << llvm::toString(qualifiedPair.takeError());
  EXPECT_EQ(qualifiedPair->legality,
            TargetSchedulingCapabilityState::Supported);
  EXPECT_EQ(qualifiedPair->profitability.overlap,
            TargetSchedulingOverlapEvidence::QualifiedOverlap);
  EXPECT_EQ(qualifiedPair->profitability.drain,
            TargetSchedulingDrainEvidence::Unknown);

  auto otherPair = registry->query(makeQuery(
      kV3, TargetSchedulingMechanism::WorkerPlacement,
      engine(TargetSchedulingEngine::NE) | engine(TargetSchedulingEngine::RDMA),
      TargetSchedulingWorkerRelation::CrossNCCWorkers,
      TargetSchedulingCompletionKind::ParticipantJoin));
  ASSERT_TRUE(static_cast<bool>(otherPair))
      << llvm::toString(otherPair.takeError());
  EXPECT_EQ(otherPair->legality, TargetSchedulingCapabilityState::Supported);
  EXPECT_TRUE(otherPair->profitability.isEntirelyUnknown());

  auto v1Pair =
      makeQuery(kV1, TargetSchedulingMechanism::WorkerPlacement, kCTRdma,
                TargetSchedulingWorkerRelation::CrossNCCWorkers,
                TargetSchedulingCompletionKind::ParticipantJoin);
  auto v1Registry = wafer::getTargetSchedulingCapabilityRegistry(kV1);
  ASSERT_TRUE(static_cast<bool>(v1Registry))
      << llvm::toString(v1Registry.takeError());
  auto v1Decision = v1Registry->query(v1Pair);
  ASSERT_TRUE(static_cast<bool>(v1Decision))
      << llvm::toString(v1Decision.takeError());
  EXPECT_EQ(v1Decision->legality, TargetSchedulingCapabilityState::Unknown);

  auto v2Pair = v1Pair;
  v2Pair.targetProfile = kV2;
  auto v2Registry = wafer::getTargetSchedulingCapabilityRegistry(kV2);
  ASSERT_TRUE(static_cast<bool>(v2Registry))
      << llvm::toString(v2Registry.takeError());
  auto v2Decision = v2Registry->query(v2Pair);
  ASSERT_TRUE(static_cast<bool>(v2Decision))
      << llvm::toString(v2Decision.takeError());
  EXPECT_EQ(v2Decision->legality, TargetSchedulingCapabilityState::Unknown);
}

TEST(TargetSchedulingCapabilityTest,
     RegistryRejectsDuplicateOverlappingAndMismatchedPredicates) {
  const TargetSchedulingLegalityRow supported{
      makeSameWorkerFixed(kCTRdma), TargetSchedulingCapabilityState::Supported};

  EXPECT_NE(takeExpectedError(TargetSchedulingCapabilityRegistry::create(
                                  kV1, 1, {supported, supported}, {}))
                .find("duplicate or overlapping"),
            std::string::npos);

  const TargetSchedulingLegalityRow firstRange{
      makeSameWorkerFixed(kCTRdma, 0, 16),
      TargetSchedulingCapabilityState::Supported};
  const TargetSchedulingLegalityRow overlappingRange{
      makeSameWorkerFixed(kCTRdma, 16, 32),
      TargetSchedulingCapabilityState::Supported};
  EXPECT_NE(takeExpectedError(TargetSchedulingCapabilityRegistry::create(
                                  kV1, 1, {firstRange, overlappingRange}, {}))
                .find("duplicate or overlapping"),
            std::string::npos);

  const TargetSchedulingLegalityRow mismatched{
      makePredicate(TargetSchedulingMechanism::StaticFixedSlot, kCTRdma,
                    TargetSchedulingWorkerRelation::SameNCCWorker,
                    TargetSchedulingCompletionKind::ParticipantJoin),
      TargetSchedulingCapabilityState::Supported};
  EXPECT_NE(takeExpectedError(TargetSchedulingCapabilityRegistry::create(
                                  kV1, 1, {mismatched}, {}))
                .find("mismatched"),
            std::string::npos);
}

TEST(TargetSchedulingCapabilityTest,
     ProfitabilityScopeAndLegalityCoverageFailClosed) {
  const TargetSchedulingLegalityRow pairLegality{
      makeSameWorkerFixed(kCTRdma), TargetSchedulingCapabilityState::Supported};
  const TargetSchedulingLegalityRow groupLegality{
      makeSameWorkerFixed(kCTRdmaWdma),
      TargetSchedulingCapabilityState::Supported};

  const TargetSchedulingProfitabilityRow pairWithGroupEngines{
      groupLegality.predicate, TargetSchedulingProfitabilityScope::ExactPair,
      TargetSchedulingOverlapEvidence::QualifiedOverlap,
      TargetSchedulingDrainEvidence::Unknown};
  EXPECT_NE(
      takeExpectedError(TargetSchedulingCapabilityRegistry::create(
                            kV1, 1, {groupLegality}, {pairWithGroupEngines}))
          .find("scope does not match"),
      std::string::npos);

  const TargetSchedulingProfitabilityRow groupWithPairEngines{
      pairLegality.predicate, TargetSchedulingProfitabilityScope::ExactGroup,
      TargetSchedulingOverlapEvidence::QualifiedOverlap,
      TargetSchedulingDrainEvidence::Unknown};
  EXPECT_NE(
      takeExpectedError(TargetSchedulingCapabilityRegistry::create(
                            kV1, 1, {pairLegality}, {groupWithPairEngines}))
          .find("scope does not match"),
      std::string::npos);

  const TargetSchedulingProfitabilityRow noEvidence{
      pairLegality.predicate, TargetSchedulingProfitabilityScope::ExactPair,
      TargetSchedulingOverlapEvidence::Unknown,
      TargetSchedulingDrainEvidence::Unknown};
  EXPECT_NE(takeExpectedError(TargetSchedulingCapabilityRegistry::create(
                                  kV1, 1, {pairLegality}, {noEvidence}))
                .find("no qualified evidence"),
            std::string::npos);

  const TargetSchedulingProfitabilityRow uncovered{
      makeSameWorkerFixed(engine(TargetSchedulingEngine::NE) |
                          engine(TargetSchedulingEngine::RDMA)),
      TargetSchedulingProfitabilityScope::ExactPair,
      TargetSchedulingOverlapEvidence::QualifiedOverlap,
      TargetSchedulingDrainEvidence::Unknown};
  EXPECT_NE(takeExpectedError(TargetSchedulingCapabilityRegistry::create(
                                  kV1, 1, {pairLegality}, {uncovered}))
                .find("lacks one supported covering legality"),
            std::string::npos);

  const TargetSchedulingLegalityRow unsupported{
      pairLegality.predicate, TargetSchedulingCapabilityState::Unsupported};
  const TargetSchedulingProfitabilityRow falselyQualified{
      pairLegality.predicate, TargetSchedulingProfitabilityScope::ExactPair,
      TargetSchedulingOverlapEvidence::QualifiedOverlap,
      TargetSchedulingDrainEvidence::Unknown};
  EXPECT_NE(takeExpectedError(TargetSchedulingCapabilityRegistry::create(
                                  kV1, 1, {unsupported}, {falselyQualified}))
                .find("lacks one supported covering legality"),
            std::string::npos);
}

TEST(TargetSchedulingCapabilityTest,
     QueryKeepsUnsupportedUnknownAndProfileMismatchDistinct) {
  const TargetSchedulingLegalityRow unsupported{
      makeSameWorkerFixed(kCTRdma),
      TargetSchedulingCapabilityState::Unsupported};
  auto registry =
      TargetSchedulingCapabilityRegistry::create(kV1, 9, {unsupported}, {});
  ASSERT_TRUE(static_cast<bool>(registry))
      << llvm::toString(registry.takeError());

  auto unsupportedDecision = registry->query(
      makeQuery(kV1, TargetSchedulingMechanism::StaticFixedSlot, kCTRdma,
                TargetSchedulingWorkerRelation::SameNCCWorker,
                TargetSchedulingCompletionKind::SameWorkerIssueOrder));
  ASSERT_TRUE(static_cast<bool>(unsupportedDecision))
      << llvm::toString(unsupportedDecision.takeError());
  EXPECT_EQ(unsupportedDecision->legality,
            TargetSchedulingCapabilityState::Unsupported);
  EXPECT_TRUE(unsupportedDecision->profitability.isEntirelyUnknown());

  auto missingDecision = registry->query(
      makeQuery(kV1, TargetSchedulingMechanism::StaticFixedSlot,
                engine(TargetSchedulingEngine::NE),
                TargetSchedulingWorkerRelation::SameNCCWorker,
                TargetSchedulingCompletionKind::SameWorkerIssueOrder));
  ASSERT_TRUE(static_cast<bool>(missingDecision))
      << llvm::toString(missingDecision.takeError());
  EXPECT_EQ(missingDecision->legality,
            TargetSchedulingCapabilityState::Unknown);

  auto wrongProfile =
      makeQuery(kV2, TargetSchedulingMechanism::StaticFixedSlot, kCTRdma,
                TargetSchedulingWorkerRelation::SameNCCWorker,
                TargetSchedulingCompletionKind::SameWorkerIssueOrder);
  EXPECT_NE(takeExpectedError(registry->query(wrongProfile))
                .find("query/profile mismatch"),
            std::string::npos);

  auto mismatchedQuery =
      makeQuery(kV1, TargetSchedulingMechanism::StaticFixedSlot, kCTRdma,
                TargetSchedulingWorkerRelation::SameNCCWorker,
                TargetSchedulingCompletionKind::ParticipantJoin);
  EXPECT_NE(
      takeExpectedError(registry->query(mismatchedQuery)).find("mismatched"),
      std::string::npos);
}

TEST(TargetSchedulingCapabilityTest,
     TypedIRAnalysisDerivesEnginesWorkersGeometryAndDTEEvents) {
  mlir::DialectRegistry dialects;
  dialects.insert<mlir::arith::ArithDialect, mlir::async::AsyncDialect,
                  mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                  wafer::WaferDialect>();
  mlir::MLIRContext context(dialects);
  context.loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @arbitrary_name(
      %input: memref<4xf16, #wafer.memory<ddr, tensor>>) {
    %buffer = memref.alloc()
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %zero = arith.constant 0.000000e+00 : f16
    wafer.instr.fill %buffer, %zero
        {worker = #wafer.ncc_worker<worker1>}
        : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.rdma %input to %buffer
        {byte_count = 8 : i64, inner_bytes = 8 : i64,
         src_strides = array<i64: 0, 0, 0>,
         src_iterations = array<i64: 1, 1, 1>,
         worker = #wafer.ncc_worker<worker2>}
        : memref<4xf16, #wafer.memory<ddr, tensor>>
       to memref<4xf16, #wafer.memory<spm, tensor>>
    %sent = wafer.instr.dte_send %buffer
        {peer = 0 : i64, bytes = 8 : i64,
         message = #wafer.dte_message<communication = 1, phase = peer_dataflow, round = 0, slice = 0>}
        : memref<4xf16, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %sent : !async.token
    return
  }
}
)mlir",
      &context);
  ASSERT_TRUE(module);

  auto query = wafer::analyzeTargetSchedulingWindow(
      *module, kV3, TargetSchedulingMechanism::DirectDTEOverlap);
  ASSERT_TRUE(static_cast<bool>(query)) << llvm::toString(query.takeError());
  EXPECT_EQ(query->engines, engine(TargetSchedulingEngine::TDMA) |
                                engine(TargetSchedulingEngine::RDMA) |
                                engine(TargetSchedulingEngine::DTE));
  EXPECT_EQ(query->workerRelation,
            TargetSchedulingWorkerRelation::MixedNCCAndDTE);
  EXPECT_EQ(query->completion,
            TargetSchedulingCompletionKind::ParticipantJoinAndExactEvent);
  EXPECT_EQ(query->maximumPayloadBytes, 8u);
  EXPECT_EQ(query->visibleBufferCount, 1u);
  EXPECT_EQ(query->visibleTokenCount, 1u);
  EXPECT_TRUE(query->geometryKnown);
}

TEST(TargetSchedulingCapabilityTest,
     SequentiallyJoinedWorkersDoNotClaimCrossWorkerOverlap) {
  mlir::DialectRegistry dialects;
  dialects.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect, wafer::WaferDialect>();
  mlir::MLIRContext context(dialects);
  context.loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @main() {
    %a = memref.alloc() : memref<4xf16, #wafer.memory<spm, tensor>>
    %b = memref.alloc() : memref<4xf16, #wafer.memory<spm, tensor>>
    %zero = arith.constant 0.000000e+00 : f16
    wafer.instr.fill %a, %zero
        {worker = #wafer.ncc_worker<worker0>}
        : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.ncc_join [0]
    wafer.instr.fill %b, %zero
        {worker = #wafer.ncc_worker<worker1>}
        : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.ncc_join [1]
    return
  }
}
)mlir",
      &context);
  ASSERT_TRUE(module);

  wafer::NCCWorkerWindowSummary windows =
      wafer::analyzeNCCWorkerWindows(*module);
  EXPECT_EQ(windows.issuedWorkerMask, UINT32_C(0x3));
  EXPECT_FALSE(windows.hasCrossWorkerWindow);

  auto query = wafer::analyzeTargetSchedulingWindow(
      *module, kV3, TargetSchedulingMechanism::WorkerPlacement);
  ASSERT_TRUE(static_cast<bool>(query)) << llvm::toString(query.takeError());
  EXPECT_EQ(query->workerRelation,
            TargetSchedulingWorkerRelation::SameNCCWorker);
  EXPECT_EQ(query->completion,
            TargetSchedulingCompletionKind::SameWorkerIssueOrder);

  auto registry = wafer::getTargetSchedulingCapabilityRegistry(kV3);
  ASSERT_TRUE(static_cast<bool>(registry))
      << llvm::toString(registry.takeError());
  auto decision = registry->query(*query);
  ASSERT_TRUE(static_cast<bool>(decision))
      << llvm::toString(decision.takeError());
  EXPECT_EQ(decision->legality, TargetSchedulingCapabilityState::Supported);
  EXPECT_TRUE(decision->profitability.isEntirelyUnknown());
}

} // namespace
