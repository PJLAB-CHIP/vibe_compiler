//===- SelectedRegionMaterializationTest.cpp --------------------------===//

#include "Wafer/Planning/PhysicalDataflow/SelectedRegionMaterialization.h"

#include "TestSupport/CodeGen/CardExecutableTestSupport.h"
#include "TestSupport/Planning/CanonicalPlanningTestSupport.h"
#include "Wafer/CodeGen/Executable/CardExecutableCompilation.h"
#include "Wafer/Planning/PhysicalDataflow/CompleteCandidateMaterialization.h"
#include "Wafer/Planning/PhysicalDataflow/RegionDomain.h"
#include "Wafer/Planning/PhysicalDataflow/TemporalDomain.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <string>

namespace {

using namespace wafer;
using namespace wafer::compiler::detail;

struct PreparedChain {
  std::shared_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::unique_ptr<CardProgramAnalysis> program;
  wafer::test::CanonicalPlanningPrefix prefix;
  RegionDomain domain;
};

frontend::FrontendProgramVerificationResult metadata(int64_t extent) {
  frontend::FrontendProgramVerificationResult result;
  result.numPartitions = 1;
  result.programUserInputCount = 1;
  result.distributedInputs = {compiler::testing::boundary(0, {2, extent, 64})};
  result.distributedOutputs = {compiler::testing::boundary(0, {2, extent, 64})};
  return result;
}

std::string source(int64_t extent) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  stream << R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {axes = ["card"], shape = array<i64: 1>}
  func.func @main(%input: tensor<2x)mlir"
         << extent << "x64xf16>) -> tensor<2x" << extent << "x64xf16> {\n"
         << "    %e0 = tensor.empty() : tensor<2x" << extent << "x64xf16>\n"
         << "    %producer = linalg.map ins(%input : tensor<2x" << extent
         << "x64xf16>) outs(%e0 : tensor<2x" << extent
         << "x64xf16>) (%value: f16) {\n"
         << "      %next = arith.addf %value, %value : f16\n"
         << "      linalg.yield %next : f16\n"
         << "    }\n"
         << "    %e1 = tensor.empty() : tensor<2x" << extent << "x64xf16>\n"
         << "    %consumer = linalg.map ins(%producer : tensor<2x" << extent
         << "x64xf16>) outs(%e1 : tensor<2x" << extent
         << "x64xf16>) (%value: f16) {\n"
         << "      %next = arith.mulf %value, %value : f16\n"
         << "      linalg.yield %next : f16\n"
         << "    }\n"
         << "    return %consumer : tensor<2x" << extent << "x64xf16>\n"
         << "  }\n}\n";
  return result;
}

std::string fanoutSource(int64_t extent) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  stream << R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {axes = ["card"], shape = array<i64: 1>}
  func.func @main(%input: tensor<2x)mlir"
         << extent << "x64xf16>) -> (tensor<2x" << extent
         << "x64xf16>, tensor<2x" << extent << "x64xf16>) {\n"
         << "    %e0 = tensor.empty() : tensor<2x" << extent << "x64xf16>\n"
         << "    %producer = linalg.map ins(%input : tensor<2x" << extent
         << "x64xf16>) outs(%e0 : tensor<2x" << extent
         << "x64xf16>) (%value: f16) {\n"
         << "      %next = arith.addf %value, %value : f16\n"
         << "      linalg.yield %next : f16\n"
         << "    }\n"
         << "    %e1 = tensor.empty() : tensor<2x" << extent << "x64xf16>\n"
         << "    %left = linalg.map ins(%producer : tensor<2x" << extent
         << "x64xf16>) outs(%e1 : tensor<2x" << extent
         << "x64xf16>) (%value: f16) {\n"
         << "      %next = arith.mulf %value, %value : f16\n"
         << "      linalg.yield %next : f16\n"
         << "    }\n"
         << "    %e2 = tensor.empty() : tensor<2x" << extent << "x64xf16>\n"
         << "    %right = linalg.map ins(%producer : tensor<2x" << extent
         << "x64xf16>) outs(%e2 : tensor<2x" << extent
         << "x64xf16>) (%value: f16) {\n"
         << "      %next = arith.subf %value, %value : f16\n"
         << "      linalg.yield %next : f16\n"
         << "    }\n"
         << "    return %left, %right : tensor<2x" << extent
         << "x64xf16>, tensor<2x" << extent << "x64xf16>\n"
         << "  }\n}\n";
  return result;
}

frontend::FrontendProgramVerificationResult fanoutMetadata(int64_t extent) {
  frontend::FrontendProgramVerificationResult result = metadata(extent);
  result.distributedOutputs.push_back(
      compiler::testing::boundary(1, {2, extent, 64}));
  return result;
}

mlir::FailureOr<PreparedChain>
prepare(llvm::StringRef sourceText,
        const frontend::FrontendProgramVerificationResult &programMetadata,
        std::string *failureReason) {
  mlir::DialectRegistry registry;
  compiler::detail::registerCompilationDialects(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      sourceText, mlir::ParserConfig(context.get()));
  if (!module)
    return mlir::failure();
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  auto program = compiler::detail::analyzeCardProgram(
      *module, programMetadata, compiler::testing::executionConfig(),
      diagnostics);
  if (mlir::failed(program)) {
    if (failureReason)
      *failureReason = diagnostics.str();
    return mlir::failure();
  }
  auto prefix = wafer::test::buildCanonicalPlanningPrefix(
      (*program)->dag, (*program)->availableTileIds, failureReason);
  if (mlir::failed(prefix))
    return mlir::failure();
  auto domain = RegionDomain::create(prefix->rootWorks, failureReason);
  if (mlir::failed(domain))
    return mlir::failure();
  return PreparedChain{std::move(context), std::move(module),
                       std::move(*program), std::move(*prefix),
                       std::move(*domain)};
}

mlir::FailureOr<PreparedChain> prepare(int64_t extent,
                                       std::string *failureReason) {
  const std::string program = source(extent);
  return prepare(program, metadata(extent), failureReason);
}

std::optional<RegionPlan> selectTileZeroChain(const RegionPlan &canonical,
                                              LocalUseDelivery delivery) {
  RegionPlan result = canonical;
  size_t consumerIndex = result.groups.size();
  size_t producerIndex = result.groups.size();
  ExternalUseBinding selectedBoundary;
  bool found = false;
  for (auto [index, group] : llvm::enumerate(result.groups)) {
    if (group.tile != TileId(0))
      continue;
    auto boundary = llvm::find_if(
        group.externalBindings, [](const ExternalUseBinding &candidate) {
          return candidate.fragment.source.kind ==
                     analysis::RootBoundaryKind::StructuredResult &&
                 candidate.fragment.ownerTile == TileId(0);
        });
    if (boundary == group.externalBindings.end())
      continue;
    consumerIndex = index;
    selectedBoundary = *boundary;
    found = true;
    break;
  }
  if (!found)
    return std::nullopt;
  for (auto [index, group] : llvm::enumerate(result.groups))
    if (group.tile == TileId(0) &&
        llvm::any_of(group.mandatoryRoots, [&](const auto &work) {
          return work.root == selectedBoundary.fragment.source.semantic;
        })) {
      producerIndex = index;
      break;
    }
  if (producerIndex == result.groups.size() || producerIndex == consumerIndex)
    return std::nullopt;

  RegionGroupPlan merged = result.groups[consumerIndex];
  const RegionGroupPlan &producerGroup = result.groups[producerIndex];
  merged.mandatoryRoots.append(producerGroup.mandatoryRoots.begin(),
                               producerGroup.mandatoryRoots.end());
  merged.executions.insert(merged.executions.end(),
                           producerGroup.executions.begin(),
                           producerGroup.executions.end());
  merged.externalBindings.insert(merged.externalBindings.end(),
                                 producerGroup.externalBindings.begin(),
                                 producerGroup.externalBindings.end());
  auto boundary = llvm::find(merged.externalBindings, selectedBoundary);
  if (boundary == merged.externalBindings.end())
    return std::nullopt;
  merged.externalBindings.erase(boundary);

  auto producer = llvm::find_if(
      merged.executions, [&](const ExecutionInstancePlan &execution) {
        const auto *root =
            std::get_if<RequiredRootExecution>(&execution.id.source);
        return root &&
               root->work.root == selectedBoundary.fragment.source.semantic;
      });
  auto consumer = llvm::find_if(
      merged.executions, [&](const ExecutionInstancePlan &execution) {
        const auto *root =
            std::get_if<RequiredRootExecution>(&execution.id.source);
        return root &&
               root->shard == selectedBoundary.fragment.use.destinationShard;
      });
  if (producer == merged.executions.end() ||
      consumer == merged.executions.end())
    return std::nullopt;
  if (delivery == LocalUseDelivery::DirectNestedValue)
    producer->placement =
        ExecutionInstancePlan::NestedUnder{consumer->id.source};
  merged.localBindings.push_back(
      {selectedBoundary.fragment, producer->id, delivery});
  llvm::sort(merged.mandatoryRoots);
  llvm::sort(merged.executions);
  llvm::sort(merged.localBindings);
  llvm::sort(merged.externalBindings);

  const size_t high = std::max(producerIndex, consumerIndex);
  const size_t low = std::min(producerIndex, consumerIndex);
  result.groups.erase(result.groups.begin() + high);
  result.groups.erase(result.groups.begin() + low);
  result.groups.push_back(std::move(merged));
  llvm::sort(result.groups,
             [](const RegionGroupPlan &lhs, const RegionGroupPlan &rhs) {
               if (lhs.tile != rhs.tile)
                 return lhs.tile.getValue() < rhs.tile.getValue();
               return lhs.mandatoryRoots < rhs.mandatoryRoots;
             });
  return result;
}

std::optional<RegionPlan> selectTileZeroReplica(const RegionPlan &canonical,
                                                LocalUseDelivery delivery) {
  RegionPlan result = canonical;
  RegionGroupPlan *consumerGroup = nullptr;
  const RegionGroupPlan *producerGroup = nullptr;
  ExternalUseBinding boundary;
  for (RegionGroupPlan &group : result.groups) {
    if (group.tile != TileId(0))
      continue;
    auto found = llvm::find_if(
        group.externalBindings, [](const ExternalUseBinding &candidate) {
          return candidate.fragment.source.kind ==
                     analysis::RootBoundaryKind::StructuredResult &&
                 candidate.fragment.ownerTile == TileId(0);
        });
    if (found == group.externalBindings.end())
      continue;
    consumerGroup = &group;
    boundary = *found;
    break;
  }
  if (!consumerGroup)
    return std::nullopt;
  for (const RegionGroupPlan &group : result.groups)
    if (&group != consumerGroup && group.tile == TileId(0) &&
        llvm::any_of(group.mandatoryRoots, [&](const auto &work) {
          return work.root == boundary.fragment.source.semantic;
        })) {
      producerGroup = &group;
      break;
    }
  if (!producerGroup)
    return std::nullopt;
  auto producer = llvm::find_if(
      producerGroup->executions, [&](const ExecutionInstancePlan &execution) {
        const auto *root =
            std::get_if<RequiredRootExecution>(&execution.id.source);
        return root && root->work.root == boundary.fragment.source.semantic;
      });
  auto consumer = llvm::find_if(
      consumerGroup->executions, [&](const ExecutionInstancePlan &execution) {
        const auto *root =
            std::get_if<RequiredRootExecution>(&execution.id.source);
        return root && root->shard == boundary.fragment.use.destinationShard;
      });
  if (producer == producerGroup->executions.end() ||
      consumer == consumerGroup->executions.end())
    return std::nullopt;
  auto external = llvm::find(consumerGroup->externalBindings, boundary);
  if (external == consumerGroup->externalBindings.end())
    return std::nullopt;
  consumerGroup->externalBindings.erase(external);
  ReplicaExecutionPlan replica;
  replica.id.producer = std::get<RequiredRootExecution>(producer->id.source);
  replica.id.fragment = boundary.fragment;
  replica.placement =
      delivery == LocalUseDelivery::StoredRegionValue
          ? ExecutionInstancePlan::Placement(ExecutionInstancePlan::TopLevel{})
          : ExecutionInstancePlan::Placement(
                ExecutionInstancePlan::NestedUnder{consumer->id.source});
  consumerGroup->replicas.push_back(replica);
  consumerGroup->localBindings.push_back(
      {boundary.fragment, replica.id, delivery});
  llvm::sort(consumerGroup->replicas);
  llvm::sort(consumerGroup->localBindings);
  llvm::sort(consumerGroup->externalBindings);
  return result;
}

std::optional<RegionPlan>
selectTileZeroSplitReplicas(const RegionPlan &canonical,
                            LocalUseDelivery delivery) {
  RegionPlan result = canonical;
  std::optional<SemanticRootKey> producerRoot;
  llvm::SmallVector<std::pair<size_t, ExternalUseBinding>, 2> consumers;
  for (auto [index, group] : llvm::enumerate(result.groups)) {
    if (group.tile != TileId(0))
      continue;
    for (const ExternalUseBinding &binding : group.externalBindings) {
      if (binding.fragment.source.kind !=
              analysis::RootBoundaryKind::StructuredResult ||
          binding.fragment.ownerTile != TileId(0))
        continue;
      if (!producerRoot)
        producerRoot = binding.fragment.source.semantic;
      if (binding.fragment.source.semantic != *producerRoot)
        continue;
      consumers.push_back({index, binding});
      break;
    }
  }
  if (!producerRoot || consumers.size() != 2)
    return std::nullopt;
  const RegionGroupPlan *producerGroup = nullptr;
  for (const RegionGroupPlan &group : result.groups)
    if (group.tile == TileId(0) &&
        llvm::any_of(group.mandatoryRoots, [&](const auto &work) {
          return work.root == *producerRoot;
        })) {
      producerGroup = &group;
      break;
    }
  if (!producerGroup)
    return std::nullopt;
  auto producer = llvm::find_if(
      producerGroup->executions, [&](const ExecutionInstancePlan &execution) {
        const auto *root =
            std::get_if<RequiredRootExecution>(&execution.id.source);
        return root && root->work.root == *producerRoot;
      });
  if (producer == producerGroup->executions.end())
    return std::nullopt;
  for (const auto &[groupIndex, binding] : consumers) {
    RegionGroupPlan &consumerGroup = result.groups[groupIndex];
    auto external = llvm::find(consumerGroup.externalBindings, binding);
    auto consumer = llvm::find_if(
        consumerGroup.executions, [&](const ExecutionInstancePlan &execution) {
          const auto *root =
              std::get_if<RequiredRootExecution>(&execution.id.source);
          return root && root->shard == binding.fragment.use.destinationShard;
        });
    if (external == consumerGroup.externalBindings.end() ||
        consumer == consumerGroup.executions.end())
      return std::nullopt;
    consumerGroup.externalBindings.erase(external);
    ReplicaExecutionPlan replica;
    replica.id.producer = std::get<RequiredRootExecution>(producer->id.source);
    replica.id.fragment = binding.fragment;
    replica.placement =
        delivery == LocalUseDelivery::StoredRegionValue
            ? ExecutionInstancePlan::Placement(
                  ExecutionInstancePlan::TopLevel{})
            : ExecutionInstancePlan::Placement(
                  ExecutionInstancePlan::NestedUnder{consumer->id.source});
    consumerGroup.replicas.push_back(replica);
    consumerGroup.localBindings.push_back(
        {binding.fragment, replica.id, delivery});
    llvm::sort(consumerGroup.replicas);
    llvm::sort(consumerGroup.localBindings);
    llvm::sort(consumerGroup.externalBindings);
  }
  return result;
}

std::string print(mlir::Operation *operation) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  operation->print(stream);
  return result;
}

TEST(SelectedRegionMaterializationTest,
     StoredAndDirectPlansConstructOneExactTileZeroRegion) {
  for (int64_t extent : {1024, 1025}) {
    for (LocalUseDelivery delivery : {LocalUseDelivery::StoredRegionValue,
                                      LocalUseDelivery::DirectNestedValue}) {
      SCOPED_TRACE(extent);
      SCOPED_TRACE(static_cast<unsigned>(delivery));
      std::string failureReason;
      auto prepared = prepare(extent, &failureReason);
      ASSERT_TRUE(mlir::succeeded(prepared)) << failureReason;
      const std::string before = print(prepared->module->getOperation());
      std::optional<RegionPlan> selected =
          selectTileZeroChain(prepared->prefix.regions, delivery);
      ASSERT_TRUE(selected);
      ASSERT_TRUE(prepared->domain.contains(*selected));
      TemporalDomainResult temporalDomain =
          buildTemporalDomain(*selected, prepared->prefix.rootWorks);
      ASSERT_TRUE(temporalDomain.succeeded());
      TemporalSuccessor temporal = temporalDomain.domain->getFirstPlan();
      ASSERT_EQ(temporal.getKind(), TemporalSuccessorKind::Plan);
      ASSERT_NE(temporal.getPlan(), nullptr);
      auto selectedSource = prepareSelectedRegionMaterializationSource(
          *prepared->module, *prepared->program, *selected,
          prepared->prefix.rootWorks, &failureReason);
      ASSERT_TRUE(mlir::succeeded(selectedSource)) << failureReason;
      auto groups = prepareSelectedRegionGroups(
          *prepared->program, *selected, prepared->prefix.rootWorks,
          *temporal.getPlan(), selectedSource->executionNodes, &failureReason);
      ASSERT_TRUE(mlir::succeeded(groups)) << failureReason;
      auto tileZero = llvm::find_if(*groups, [](const auto &group) {
        return !group.shards.empty() &&
               group.shards.front().tile == TileId(0) &&
               group.shards.size() == 2;
      });
      ASSERT_NE(tileZero, groups->end());
      EXPECT_EQ(tileZero->independentlyMaterializedNodes.size(),
                delivery == LocalUseDelivery::StoredRegionValue ? 2u : 1u);

      mlir::OwningOpRef<mlir::ModuleOp> card;
      StructuredMaterializationRelations relations;
      ASSERT_TRUE(mlir::succeeded(lowerStructuredNodeGroupsToCardModule(
          *selectedSource->module, CardId(0),
          prepared->program->availableTileIds, selectedSource->operationNodes,
          *groups, card, &relations, &failureReason)))
          << failureReason;
      ASSERT_TRUE(mlir::succeeded(mlir::verify(*card)));
      unsigned tileZeroRegions = 0;
      std::set<uint32_t> tileZeroNodes;
      card->walk([&](TileModuleOp tile) {
        if (tile.getTileIdAttr().getInt() != 0)
          return;
        tile.walk([&](TileRegionOp) { ++tileZeroRegions; });
      });
      for (const StructuredOperationEmissionRelation &relation :
           relations.operationEmissions)
        if (relation.operation &&
            relation.operation->getParentOfType<TileModuleOp>() &&
            relation.operation->getParentOfType<TileModuleOp>()
                    .getTileIdAttr()
                    .getInt() == 0)
          tileZeroNodes.insert(relation.structuredNodeId);
      EXPECT_EQ(tileZeroRegions, 1u);
      EXPECT_EQ(tileZeroNodes, (std::set<uint32_t>{0, 1}));
      EXPECT_EQ(print(prepared->module->getOperation()), before);

      PreparedAttentionDecomposition noAttention;
      CompleteCandidatePlan candidate{
          prepared->prefix.spatial,   prepared->prefix.demand,
          prepared->prefix.rootWorks, *selected,
          *temporal.getPlan(),        noAttention};
      std::string diagnosticsText;
      llvm::raw_string_ostream diagnostics(diagnosticsText);
      CandidateMaterializationStatistics materializationStatistics;
      auto materialized = materializeCardCandidate(
          *prepared->module, CardId(0), *prepared->program, candidate,
          &materializationStatistics, diagnostics);
      ASSERT_TRUE(mlir::succeeded(materialized)) << diagnostics.str();
      EXPECT_EQ(materializationStatistics.cardModuleMaterializations, 1u);
      std::string verificationFailure;
      ASSERT_TRUE(mlir::succeeded(verifyMaterializedCardCandidate(
          *materialized->module, materialized->assignment,
          prepared->program->dag, materialized->relations,
          prepared->program->availableTileIds, verificationFailure)))
          << verificationFailure;
      compiler::ProgramDataHandoff programData;
      CardExecutableCompilationResult compiled = compileCardModuleToExecutable(
          std::move(materialized->module), CardId(0),
          prepared->program->availableTileIds,
          /*selectedBufferingScopes=*/{}, materialized->relations,
          metadata(extent), compiler::testing::executionConfig(), diagnostics,
          programData, /*statistics=*/nullptr,
          /*tilePipelineParallelism=*/0,
          /*captureTileDataflowIRTrace=*/false);
      ASSERT_TRUE(compiled.isAccepted())
          << compiled.gate << ": " << compiled.detail << '\n'
          << diagnostics.str();
      EXPECT_EQ(print(prepared->module->getOperation()), before);
    }
  }
}

TEST(SelectedRegionMaterializationTest,
     NestedMainAndTailClassesDriveCompactParentAndChildLoops) {
  for (int64_t extent : {1024, 1025}) {
    for (bool replica : {false, true}) {
      SCOPED_TRACE(extent);
      SCOPED_TRACE(replica);
      std::string failureReason;
      auto prepared = prepare(extent, &failureReason);
      ASSERT_TRUE(mlir::succeeded(prepared)) << failureReason;
      const std::string before = print(prepared->module->getOperation());
      std::optional<RegionPlan> selected =
          replica ? selectTileZeroReplica(prepared->prefix.regions,
                                          LocalUseDelivery::DirectNestedValue)
                  : selectTileZeroChain(prepared->prefix.regions,
                                        LocalUseDelivery::DirectNestedValue);
      ASSERT_TRUE(selected);
      ASSERT_TRUE(prepared->domain.contains(*selected));
      auto group =
          llvm::find_if(selected->groups, [](const RegionGroupPlan &candidate) {
            return candidate.tile == TileId(0) &&
                   !candidate.localBindings.empty();
          });
      ASSERT_NE(group, selected->groups.end());
      const LocalUseBinding &binding = group->localBindings.front();
      const ExecutionInstancePlan::Placement *producerPlacement = nullptr;
      if (const auto *required =
              std::get_if<ExecutionInstanceId>(&binding.producer)) {
        auto plan = llvm::find_if(group->executions,
                                  [&](const ExecutionInstancePlan &candidate) {
                                    return candidate.id == *required;
                                  });
        ASSERT_NE(plan, group->executions.end());
        producerPlacement = &plan->placement;
      } else {
        const auto &replicaExecution =
            std::get<ReplicaExecutionId>(binding.producer);
        auto plan = llvm::find_if(group->replicas,
                                  [&](const ReplicaExecutionPlan &candidate) {
                                    return candidate.id == replicaExecution;
                                  });
        ASSERT_NE(plan, group->replicas.end());
        producerPlacement = &plan->placement;
      }
      ASSERT_NE(producerPlacement, nullptr);
      const auto *nested =
          std::get_if<ExecutionInstancePlan::NestedUnder>(producerPlacement);
      ASSERT_NE(nested, nullptr);
      const RegionExecutionId parentExecution =
          ExecutionInstanceId{nested->consumer};

      TemporalDomainResult temporalDomain =
          buildTemporalDomain(*selected, prepared->prefix.rootWorks);
      ASSERT_TRUE(temporalDomain.succeeded());
      TemporalSuccessor first = temporalDomain.domain->getFirstPlan();
      ASSERT_EQ(first.getKind(), TemporalSuccessorKind::Plan);
      ASSERT_NE(first.getPlan(), nullptr);
      TemporalPlan parentPrefix;
      bool selectedParent = false;
      for (const TemporalScopePlan &scope : first.getPlan()->scopes) {
        if (!isTopLevelScope(scope.id))
          break;
        parentPrefix.scopes.push_back(scope);
        if (!(scope.id.execution == parentExecution))
          continue;
        auto majorAxis =
            llvm::find_if(parentPrefix.scopes.back().iteratorTileSizes,
                          [](int64_t size) { return size >= 128; });
        ASSERT_NE(majorAxis,
                  parentPrefix.scopes.back().iteratorTileSizes.end());
        const uint32_t majorDimension = static_cast<uint32_t>(std::distance(
            parentPrefix.scopes.back().iteratorTileSizes.begin(), majorAxis));
        auto featureAxis = llvm::find(
            parentPrefix.scopes.back().iteratorTileSizes, int64_t{64});
        ASSERT_NE(featureAxis,
                  parentPrefix.scopes.back().iteratorTileSizes.end());
        const uint32_t featureDimension = static_cast<uint32_t>(std::distance(
            parentPrefix.scopes.back().iteratorTileSizes.begin(), featureAxis));
        *majorAxis = 32;
        *featureAxis = 16;
        parentPrefix.scopes.back().waveLoopOrder = {featureDimension,
                                                    majorDimension};
        selectedParent = true;
        break;
      }
      ASSERT_TRUE(selectedParent);
      TemporalSuccessor withNested =
          temporalDomain.domain->completePrefix(parentPrefix);
      ASSERT_EQ(withNested.getKind(), TemporalSuccessorKind::Plan)
          << withNested.getDetail().str();
      ASSERT_NE(withNested.getPlan(), nullptr);
      TemporalPlan selectedTemporal = *withNested.getPlan();
      unsigned nestedClasses = 0;
      for (TemporalScopePlan &scope : selectedTemporal.scopes) {
        const auto *invocation =
            std::get_if<NestedInvocationClassId>(&scope.id.invocation);
        if (!invocation || !(scope.id.execution == binding.producer) ||
            !(invocation->parent == parentExecution))
          continue;
        ++nestedClasses;
        llvm::SmallVector<uint32_t, 4> active;
        for (auto [dimension, size] :
             llvm::enumerate(scope.iteratorTileSizes)) {
          if (size >= 32)
            size = 16;
          else if (size >= 16)
            size = 8;
          if (size < invocation->producerExtents[dimension])
            active.push_back(static_cast<uint32_t>(dimension));
        }
        std::reverse(active.begin(), active.end());
        scope.waveLoopOrder = std::move(active);
      }
      EXPECT_EQ(nestedClasses, extent == 1024 ? 1u : 2u);
      ASSERT_TRUE(temporalDomain.domain->contains(selectedTemporal));
      TemporalPlan feedbackPrefix = selectedTemporal;
      const SemanticRootKey parentRoot =
          std::visit([](const auto &source) { return source.work.root; },
                     nested->consumer);
      auto refined = refineTemporalPlanFromActualSPMFeedback(
          feedbackPrefix, prepared->prefix.rootWorks,
          llvm::ArrayRef<SemanticRootKey>(&parentRoot, 1), &failureReason);
      ASSERT_TRUE(mlir::succeeded(refined)) << failureReason;
      ASSERT_TRUE(*refined);
      EXPECT_TRUE(llvm::all_of(feedbackPrefix.scopes, [](const auto &scope) {
        return isTopLevelScope(scope.id);
      }));
      TemporalSuccessor reclosed =
          temporalDomain.domain->completePrefix(feedbackPrefix);
      ASSERT_EQ(reclosed.getKind(), TemporalSuccessorKind::Plan)
          << reclosed.getDetail().str();
      ASSERT_NE(reclosed.getPlan(), nullptr);
      EXPECT_TRUE(temporalDomain.domain->contains(*reclosed.getPlan()));

      PreparedAttentionDecomposition noAttention;
      CompleteCandidatePlan candidate{
          prepared->prefix.spatial,   prepared->prefix.demand,
          prepared->prefix.rootWorks, *selected,
          selectedTemporal,           noAttention};
      std::string diagnosticsText;
      llvm::raw_string_ostream diagnostics(diagnosticsText);
      CandidateMaterializationStatistics statistics;
      auto materialized = materializeCardCandidate(
          *prepared->module, CardId(0), *prepared->program, candidate,
          &statistics, diagnostics);
      ASSERT_TRUE(mlir::succeeded(materialized)) << diagnostics.str();
      std::string verificationFailure;
      ASSERT_TRUE(mlir::succeeded(verifyMaterializedCardCandidate(
          *materialized->module, materialized->assignment,
          prepared->program->dag, materialized->relations,
          prepared->program->availableTileIds, verificationFailure)))
          << verificationFailure;
      CardMaterializationPlan corrupted = materialized->assignment;
      auto corruptedGroup = llvm::find_if(
          corrupted.selectedRegionGroups, [](const auto &candidate) {
            return !candidate.nestedTemporalTiles.empty();
          });
      ASSERT_NE(corruptedGroup, corrupted.selectedRegionGroups.end());
      auto corruptedSize = llvm::find(
          corruptedGroup->nestedTemporalTiles.front().iteratorTileSizes,
          int64_t{8});
      ASSERT_NE(
          corruptedSize,
          corruptedGroup->nestedTemporalTiles.front().iteratorTileSizes.end());
      *corruptedSize = 4;
      std::string corruptedFailure;
      EXPECT_TRUE(mlir::failed(verifyMaterializedCardCandidate(
          *materialized->module, corrupted, prepared->program->dag,
          materialized->relations, prepared->program->availableTileIds,
          corruptedFailure)));
      EXPECT_NE(corruptedFailure.find("nested temporal loops"),
                std::string::npos);

      unsigned parentLoops = 0;
      unsigned childLoops = 0;
      unsigned nestedChildLoops = 0;
      materialized->module->walk([&](mlir::scf::ForOp loop) {
        TileModuleOp tile = loop->getParentOfType<TileModuleOp>();
        std::optional<int64_t> step = mlir::getConstantIntValue(loop.getStep());
        if (!tile || tile.getTileIdAttr().getInt() != 0 || !step)
          return;
        if (*step == 32)
          ++parentLoops;
        if (*step != 8)
          return;
        ++childLoops;
        for (mlir::Operation *owner = loop->getParentOp(); owner;
             owner = owner->getParentOp()) {
          auto parent = mlir::dyn_cast<mlir::scf::ForOp>(owner);
          std::optional<int64_t> parentStep =
              parent ? mlir::getConstantIntValue(parent.getStep())
                     : std::nullopt;
          if (parentStep && *parentStep == 32) {
            ++nestedChildLoops;
            break;
          }
        }
      });
      EXPECT_GE(parentLoops, 1u);
      EXPECT_GE(childLoops, 2u);
      EXPECT_GE(nestedChildLoops, 1u);

      compiler::ProgramDataHandoff programData;
      CardExecutableCompilationResult compiled = compileCardModuleToExecutable(
          std::move(materialized->module), CardId(0),
          prepared->program->availableTileIds,
          /*selectedBufferingScopes=*/{}, materialized->relations,
          metadata(extent), compiler::testing::executionConfig(), diagnostics,
          programData, /*statistics=*/nullptr,
          /*tilePipelineParallelism=*/0,
          /*captureTileDataflowIRTrace=*/false);
      ASSERT_TRUE(compiled.isAccepted())
          << compiled.gate << ": " << compiled.detail << '\n'
          << diagnostics.str();
      EXPECT_EQ(statistics.cardModuleMaterializations, 1u);
      EXPECT_EQ(print(prepared->module->getOperation()), before);
    }
  }
}

TEST(SelectedRegionMaterializationTest,
     StoredAndDirectReplicasRemainExplicitThroughActualAdmission) {
  for (int64_t extent : {1024, 1025}) {
    for (LocalUseDelivery delivery : {LocalUseDelivery::StoredRegionValue,
                                      LocalUseDelivery::DirectNestedValue}) {
      SCOPED_TRACE(extent);
      SCOPED_TRACE(static_cast<unsigned>(delivery));
      std::string failureReason;
      auto prepared = prepare(extent, &failureReason);
      ASSERT_TRUE(mlir::succeeded(prepared)) << failureReason;
      const std::string before = print(prepared->module->getOperation());
      std::optional<RegionPlan> selected =
          selectTileZeroReplica(prepared->prefix.regions, delivery);
      ASSERT_TRUE(selected);
      ASSERT_TRUE(prepared->domain.contains(*selected));
      TemporalDomainResult temporalDomain =
          buildTemporalDomain(*selected, prepared->prefix.rootWorks);
      ASSERT_TRUE(temporalDomain.succeeded());
      TemporalSuccessor temporal = temporalDomain.domain->getFirstPlan();
      ASSERT_EQ(temporal.getKind(), TemporalSuccessorKind::Plan);
      ASSERT_NE(temporal.getPlan(), nullptr);
      auto selectedSource = prepareSelectedRegionMaterializationSource(
          *prepared->module, *prepared->program, *selected,
          prepared->prefix.rootWorks, &failureReason);
      ASSERT_TRUE(mlir::succeeded(selectedSource)) << failureReason;
      auto groups = prepareSelectedRegionGroups(
          *prepared->program, *selected, prepared->prefix.rootWorks,
          *temporal.getPlan(), selectedSource->executionNodes, &failureReason);
      ASSERT_TRUE(mlir::succeeded(groups)) << failureReason;
      auto replicaGroup = llvm::find_if(*groups, [](const auto &group) {
        return !group.shards.empty() &&
               group.shards.front().tile == TileId(0) &&
               group.shards.size() == 2;
      });
      ASSERT_NE(replicaGroup, groups->end());
      EXPECT_EQ(replicaGroup->independentlyMaterializedNodes.size(),
                delivery == LocalUseDelivery::StoredRegionValue ? 2u : 1u);

      mlir::OwningOpRef<mlir::ModuleOp> directCard;
      StructuredMaterializationRelations directRelations;
      ASSERT_TRUE(mlir::succeeded(lowerStructuredNodeGroupsToCardModule(
          *selectedSource->module, CardId(0),
          prepared->program->availableTileIds, selectedSource->operationNodes,
          *groups, directCard, &directRelations, &failureReason)))
          << failureReason;

      PreparedAttentionDecomposition noAttention;
      CompleteCandidatePlan candidate{
          prepared->prefix.spatial,   prepared->prefix.demand,
          prepared->prefix.rootWorks, *selected,
          *temporal.getPlan(),        noAttention};
      std::string diagnosticsText;
      llvm::raw_string_ostream diagnostics(diagnosticsText);
      CandidateMaterializationStatistics statistics;
      auto materialized = materializeCardCandidate(
          *prepared->module, CardId(0), *prepared->program, candidate,
          &statistics, diagnostics);
      ASSERT_TRUE(mlir::succeeded(materialized)) << diagnostics.str();
      EXPECT_EQ(statistics.cardModuleMaterializations, 1u);
      std::string verificationFailure;
      ASSERT_TRUE(mlir::succeeded(verifyMaterializedCardCandidate(
          *materialized->module, materialized->assignment,
          prepared->program->dag, materialized->relations,
          prepared->program->availableTileIds, verificationFailure)))
          << verificationFailure;
      compiler::ProgramDataHandoff programData;
      CardExecutableCompilationResult compiled = compileCardModuleToExecutable(
          std::move(materialized->module), CardId(0),
          prepared->program->availableTileIds,
          /*selectedBufferingScopes=*/{}, materialized->relations,
          metadata(extent), compiler::testing::executionConfig(), diagnostics,
          programData, /*statistics=*/nullptr,
          /*tilePipelineParallelism=*/0,
          /*captureTileDataflowIRTrace=*/false);
      ASSERT_TRUE(compiled.isAccepted())
          << compiled.gate << ": " << compiled.detail << '\n'
          << diagnostics.str();
      EXPECT_EQ(print(prepared->module->getOperation()), before);
    }
  }
}

TEST(SelectedRegionMaterializationTest,
     FanoutSplitReplicasKeepTwoCandidateLocalExecutionNodes) {
  for (int64_t extent : {1024, 1025}) {
    for (LocalUseDelivery delivery : {LocalUseDelivery::StoredRegionValue,
                                      LocalUseDelivery::DirectNestedValue}) {
      SCOPED_TRACE(extent);
      SCOPED_TRACE(static_cast<unsigned>(delivery));
      std::string failureReason;
      const std::string programText = fanoutSource(extent);
      auto prepared =
          prepare(programText, fanoutMetadata(extent), &failureReason);
      ASSERT_TRUE(mlir::succeeded(prepared)) << failureReason;
      const std::string before = print(prepared->module->getOperation());
      std::optional<RegionPlan> selected =
          selectTileZeroSplitReplicas(prepared->prefix.regions, delivery);
      ASSERT_TRUE(selected);
      ASSERT_TRUE(prepared->domain.contains(*selected));
      TemporalDomainResult temporalDomain =
          buildTemporalDomain(*selected, prepared->prefix.rootWorks);
      ASSERT_TRUE(temporalDomain.succeeded());
      TemporalSuccessor temporal = temporalDomain.domain->getFirstPlan();
      ASSERT_EQ(temporal.getKind(), TemporalSuccessorKind::Plan);
      ASSERT_NE(temporal.getPlan(), nullptr);
      auto selectedSource = prepareSelectedRegionMaterializationSource(
          *prepared->module, *prepared->program, *selected,
          prepared->prefix.rootWorks, &failureReason);
      ASSERT_TRUE(mlir::succeeded(selectedSource)) << failureReason;
      llvm::SmallVector<uint32_t, 2> replicaNodes;
      for (const SelectedRegionExecutionNode &execution :
           selectedSource->executionNodes)
        if (std::holds_alternative<ReplicaExecutionId>(execution.execution))
          replicaNodes.push_back(execution.structuredNodeId);
      llvm::sort(replicaNodes);
      ASSERT_EQ(replicaNodes.size(), 2u);
      EXPECT_NE(replicaNodes[0], replicaNodes[1]);

      auto groups = prepareSelectedRegionGroups(
          *prepared->program, *selected, prepared->prefix.rootWorks,
          *temporal.getPlan(), selectedSource->executionNodes, &failureReason);
      ASSERT_TRUE(mlir::succeeded(groups)) << failureReason;
      unsigned replicaGroupCount = 0;
      for (const StructuredNodeShardGroup &group : *groups) {
        if (group.shards.empty() || group.shards.front().tile != TileId(0) ||
            group.shards.size() != 2)
          continue;
        ++replicaGroupCount;
        EXPECT_EQ(group.independentlyMaterializedNodes.size(),
                  delivery == LocalUseDelivery::StoredRegionValue ? 2u : 1u);
      }
      EXPECT_EQ(replicaGroupCount, 2u);

      PreparedAttentionDecomposition noAttention;
      CompleteCandidatePlan candidate{
          prepared->prefix.spatial,   prepared->prefix.demand,
          prepared->prefix.rootWorks, *selected,
          *temporal.getPlan(),        noAttention};
      std::string diagnosticsText;
      llvm::raw_string_ostream diagnostics(diagnosticsText);
      CandidateMaterializationStatistics statistics;
      auto materialized = materializeCardCandidate(
          *prepared->module, CardId(0), *prepared->program, candidate,
          &statistics, diagnostics);
      ASSERT_TRUE(mlir::succeeded(materialized)) << diagnostics.str();
      std::string verificationFailure;
      ASSERT_TRUE(mlir::succeeded(verifyMaterializedCardCandidate(
          *materialized->module, materialized->assignment,
          prepared->program->dag, materialized->relations,
          prepared->program->availableTileIds, verificationFailure)))
          << verificationFailure;
      compiler::ProgramDataHandoff programData;
      CardExecutableCompilationResult compiled = compileCardModuleToExecutable(
          std::move(materialized->module), CardId(0),
          prepared->program->availableTileIds,
          /*selectedBufferingScopes=*/{}, materialized->relations,
          fanoutMetadata(extent), compiler::testing::executionConfig(),
          diagnostics, programData, /*statistics=*/nullptr,
          /*tilePipelineParallelism=*/0,
          /*captureTileDataflowIRTrace=*/false);
      ASSERT_TRUE(compiled.isAccepted())
          << compiled.gate << ": " << compiled.detail << '\n'
          << diagnostics.str();
      EXPECT_EQ(statistics.cardModuleMaterializations, 1u);
      EXPECT_EQ(print(prepared->module->getOperation()), before);
    }
  }
}

TEST(SelectedRegionMaterializationTest,
     InvalidPlanAndMidConstructionFailureLeaveSourceUnchanged) {
  std::string failureReason;
  auto prepared = prepare(/*extent=*/1025, &failureReason);
  ASSERT_TRUE(mlir::succeeded(prepared)) << failureReason;
  const std::string before = print(prepared->module->getOperation());
  std::optional<RegionPlan> selected = selectTileZeroChain(
      prepared->prefix.regions, LocalUseDelivery::DirectNestedValue);
  ASSERT_TRUE(selected);
  TemporalDomainResult temporalDomain =
      buildTemporalDomain(*selected, prepared->prefix.rootWorks);
  ASSERT_TRUE(temporalDomain.succeeded());
  TemporalSuccessor temporal = temporalDomain.domain->getFirstPlan();
  ASSERT_EQ(temporal.getKind(), TemporalSuccessorKind::Plan);
  ASSERT_NE(temporal.getPlan(), nullptr);
  auto selectedSource = prepareSelectedRegionMaterializationSource(
      *prepared->module, *prepared->program, *selected,
      prepared->prefix.rootWorks, &failureReason);
  ASSERT_TRUE(mlir::succeeded(selectedSource)) << failureReason;

  TemporalPlan missingNested = *temporal.getPlan();
  auto nestedScope = llvm::find_if(missingNested.scopes, [](const auto &scope) {
    return !isTopLevelScope(scope.id);
  });
  ASSERT_NE(nestedScope, missingNested.scopes.end());
  missingNested.scopes.erase(nestedScope);
  auto missingNestedGroups = prepareSelectedRegionGroups(
      *prepared->program, *selected, prepared->prefix.rootWorks, missingNested,
      selectedSource->executionNodes, &failureReason);
  EXPECT_TRUE(mlir::failed(missingNestedGroups));
  EXPECT_NE(failureReason.find("temporal traversal scope"), std::string::npos);
  EXPECT_EQ(print(prepared->module->getOperation()), before);

  RegionPlan inconsistent = *selected;
  auto localGroup =
      llvm::find_if(inconsistent.groups, [](const RegionGroupPlan &group) {
        return !group.localBindings.empty();
      });
  ASSERT_NE(localGroup, inconsistent.groups.end());
  localGroup->localBindings.front().delivery =
      LocalUseDelivery::StoredRegionValue;
  EXPECT_FALSE(prepared->domain.contains(inconsistent));
  auto rejected = prepareSelectedRegionGroups(
      *prepared->program, inconsistent, prepared->prefix.rootWorks,
      *temporal.getPlan(), selectedSource->executionNodes, &failureReason);
  EXPECT_TRUE(mlir::failed(rejected));
  EXPECT_EQ(print(prepared->module->getOperation()), before);

  auto groups = prepareSelectedRegionGroups(
      *prepared->program, *selected, prepared->prefix.rootWorks,
      *temporal.getPlan(), selectedSource->executionNodes, &failureReason);
  ASSERT_TRUE(mlir::succeeded(groups)) << failureReason;
  auto coupled =
      llvm::find_if(*groups, [](const StructuredNodeShardGroup &group) {
        return group.shards.size() == 2;
      });
  ASSERT_NE(coupled, groups->end());
  coupled->independentlyMaterializedNodes.push_back(
      std::numeric_limits<uint32_t>::max());
  mlir::OwningOpRef<mlir::ModuleOp> card;
  EXPECT_TRUE(mlir::failed(lowerStructuredNodeGroupsToCardModule(
      *selectedSource->module, CardId(0), prepared->program->availableTileIds,
      selectedSource->operationNodes, *groups, card, nullptr, &failureReason)));
  EXPECT_FALSE(card);
  EXPECT_EQ(print(prepared->module->getOperation()), before);
}

} // namespace
