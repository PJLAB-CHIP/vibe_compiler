//===- PhysicalDataflowInstrumentation.h - Read-only counters -*- C++ -*-===//

#ifndef WAFER_DRIVER_PHYSICALDATAFLOW_PHYSICALDATAFLOWINSTRUMENTATION_H
#define WAFER_DRIVER_PHYSICALDATAFLOW_PHYSICALDATAFLOWINSTRUMENTATION_H

#include "Wafer/Driver/ExecutableCompilation.h"
#include "Wafer/Support/CompileTiming.h"
#include "Wafer/Transforms/Tile/BoundaryMovement.h"
#include "Wafer/Transforms/Tile/LayoutOptimization.h"

#include "llvm/Support/FormatVariadic.h"

#include <algorithm>
#include <limits>

namespace wafer::compiler::detail {

inline void recordMetric(llvm::StringRef name,
                         const analysis::ScheduleCostMetric &metric) {
  wafer::support::addCompileCounter("accepted-instr", name,
                                    metric.isKnown() ? metric.value : 0);
  wafer::support::addCompileCounter("accepted-instr", (name + "-known").str(),
                                    metric.isKnown());
}

inline void
recordExecutionCount(llvm::StringRef name,
                     const analysis::InstructionExecutionCount &execution) {
  recordMetric((name + "-sites").str(), execution.staticSites);
  recordMetric((name + "-executions").str(), execution.exactExecutions);
}

inline void recordAcceptedPhysicalDataflowInstrumentation(
    const PhysicalDataflowIRInventory &inventory,
    const analysis::InstructionProgramAggregateCost &cost) {
  auto physicalCounter = [&](llvm::StringRef name, uint64_t value) {
    wafer::support::addCompileCounter("accepted-physical-ir", name, value);
  };
  physicalCounter("tile-modules", inventory.tileModules);
  physicalCounter("tile-regions", inventory.tileRegions);
  physicalCounter("nested-operations", inventory.nestedOperations);
  physicalCounter("nested-operations-min", inventory.minimumNestedOperations);
  physicalCounter("nested-operations-max", inventory.maximumNestedOperations);
  physicalCounter("tile-dataflow-operations", inventory.tileDataflowOperations);
  physicalCounter("tile-dataflow-operations-min",
                  inventory.minimumTileDataflowOperations);
  physicalCounter("tile-dataflow-operations-max",
                  inventory.maximumTileDataflowOperations);
  physicalCounter("singleton-dataflow-regions",
                  inventory.singletonDataflowRegions);
  for (const PhysicalTileIRInventory &tile : inventory.tiles) {
    const int64_t tileId = tile.tile.getValue();
    physicalCounter(llvm::formatv("tile-{0}-regions", tileId).str(),
                    tile.regions);
    physicalCounter(llvm::formatv("tile-{0}-nested-operations", tileId).str(),
                    tile.nestedOperations);
    physicalCounter(llvm::formatv("tile-{0}-dataflow-operations", tileId).str(),
                    tile.tileDataflowOperations);
  }

  wafer::support::addCompileCounter("accepted-instr", "tiles",
                                    cost.tileCosts.size());
  recordExecutionCount("instructions", cost.aggregateWork.instructions);
  recordExecutionCount("async-events", cost.aggregateWork.asynchronousEvents);
  recordExecutionCount("rdma", cost.aggregateWork.rdmaIssues);
  recordExecutionCount("wdma", cost.aggregateWork.wdmaIssues);
  recordExecutionCount("tdma", cost.aggregateWork.tdmaIssues);
  recordExecutionCount("ct", cost.aggregateWork.ctIssues);
  recordExecutionCount("ne", cost.aggregateWork.neIssues);
  recordExecutionCount("dte", cost.aggregateWork.dteOperations);
  recordExecutionCount("gather-scatter",
                       cost.aggregateWork.gatherScatterOperations);
  recordExecutionCount("dte-send", cost.aggregateWork.dteSendOperations);
  recordExecutionCount("dte-receive", cost.aggregateWork.dteReceiveOperations);
  recordExecutionCount("dte-wait", cost.aggregateWork.dteWaitOperations);
  recordExecutionCount("ncc-join", cost.aggregateWork.nccJoins);
  recordExecutionCount("steady-state-ncc-join",
                       cost.aggregateWork.steadyStateNCCJoins);
  recordExecutionCount("non-terminal-ncc-join",
                       cost.aggregateWork.nonTerminalNCCJoins);
  recordMetric("ne-f16-bf16-logical-ops",
               cost.aggregateCompute.npuF16Bf16LogicalOps);
  recordMetric("vector-f16-bf16-logical-ops",
               cost.aggregateCompute.vectorF16Bf16LogicalOps);
  recordMetric("vector-f32-logical-ops",
               cost.aggregateCompute.vectorF32LogicalOps);
  recordMetric("ddr-read-bytes", cost.aggregateDDRReadBytes);
  recordMetric("ddr-write-bytes", cost.aggregateDDRWriteBytes);
  recordMetric("spm-movement-bytes", cost.aggregateSPMMovementBytes);
  recordMetric("gather-scatter-bytes", cost.aggregateGatherScatterBytes);
  recordMetric("noc-transmit-bytes", cost.aggregateNoC.aggregateTransmitBytes);
  recordMetric("noc-receive-bytes", cost.aggregateNoC.aggregateReceiveBytes);
  recordMetric("spm-high-water-max", cost.maximumTileSPMHighWaterBytes);
  recordMetric("ddr-high-water-max", cost.maximumTileDDRHighWaterBytes);
  recordMetric("spm-buffer-count", cost.aggregateCompilerOwnedSPMBufferCount);
  recordMetric("ddr-buffer-count", cost.aggregateCompilerOwnedDDRBufferCount);

  bool allKnown = true;
  uint64_t sum = 0;
  uint64_t minimum = 0;
  uint64_t maximum = 0;
  for (auto [index, tile] : llvm::enumerate(cost.tileCosts)) {
    const analysis::ScheduleCostMetric &count =
        tile.work.instructions.exactExecutions;
    if (!count.isKnown()) {
      allKnown = false;
      continue;
    }
    if (index == 0) {
      minimum = count.value;
      maximum = count.value;
    } else {
      minimum = std::min(minimum, count.value);
      maximum = std::max(maximum, count.value);
    }
    if (count.value > std::numeric_limits<uint64_t>::max() - sum) {
      allKnown = false;
      continue;
    }
    sum += count.value;
  }
  wafer::support::addCompileCounter("accepted-instr",
                                    "per-tile-instructions-known", allKnown);
  if (allKnown) {
    wafer::support::addCompileCounter("accepted-instr",
                                      "per-tile-instructions-sum", sum);
    wafer::support::addCompileCounter("accepted-instr",
                                      "per-tile-instructions-min", minimum);
    wafer::support::addCompileCounter("accepted-instr",
                                      "per-tile-instructions-max", maximum);
  }
}

inline void
recordLayoutInstrumentation(const LayoutOptimizationStatistics &statistics) {
  auto counter = [&](llvm::StringRef name, uint64_t value) {
    wafer::support::addCompileCounter("layout", name, value);
  };
  counter("invocations", statistics.invocations);
  counter("solver-work", statistics.solverWork);
  counter("pbqp-variables", statistics.pbqpVariables);
  counter("pbqp-factors", statistics.pbqpFactors);
  counter("value-groups", statistics.valueGroups);
  counter("dominated-states-pruned", statistics.dominatedLayoutStatesPruned);
  counter("use-bindings", statistics.useBindings);
  counter("tuple-variables", statistics.tupleVariables);
  counter("conversion-activations", statistics.conversionActivations);
  counter("canonical-assignments", statistics.canonicalAssignmentsBuilt);
  counter("feasible-fallbacks", statistics.canonicalAssignmentFallbacks);
  counter("selected-materializations", statistics.selectedMaterializations);
  counter("materializations-before", statistics.layoutMaterializationsBefore);
  counter("materializations-after", statistics.layoutMaterializationsAfter);
  counter("unused-materializations-erased",
          statistics.unusedMaterializationsErased);
  counter("shared-materializations-reused",
          statistics.sharedMaterializationsReused);
  counter("bufferization-invocations", statistics.bufferizationInvocations);
  counter("output-destinations", statistics.outputDestinations);
  counter("output-subviews", statistics.outputSubviews);
  counter("boundary-source-views-elided", statistics.boundarySourceViewsElided);
  counter("necessary-copies", statistics.necessaryCopies);
  counter("redundant-publication-copies",
          statistics.redundantPublicationCopies);
}

inline void
recordMovementInstrumentation(const BoundaryMovementStatistics &statistics) {
  auto counter = [&](llvm::StringRef name, uint64_t value) {
    wafer::support::addCompileCounter("movement", name, value);
  };
  counter("ddr-loads", statistics.ddrLoads);
  counter("ddr-stores", statistics.ddrStores);
  counter("peer-sends", statistics.peerSends);
  counter("peer-receives", statistics.peerReceives);
  counter("peer-relay-sends", statistics.peerRelaySends);
  counter("fanout-groups", statistics.topologyFanoutGroups);
  counter("fanout-rounds", statistics.topologyFanoutRounds);
  counter("native-broadcast-groups", statistics.nativeBroadcastGroups);
  counter("native-broadcast-rounds", statistics.nativeBroadcastRounds);
  counter("native-scatter-groups", statistics.nativeScatterGroups);
  counter("native-scatter-rounds", statistics.nativeScatterRounds);
  counter("ring-components", statistics.ringComponents);
  counter("ring-rounds", statistics.ringRounds);
  counter("recursive-doubling-components",
          statistics.recursiveDoublingComponents);
  counter("recursive-doubling-rounds", statistics.recursiveDoublingRounds);
  counter("recursive-doubling-seed-copies",
          statistics.recursiveDoublingSeedCopies);
  counter("recursive-doubling-seed-donations",
          statistics.recursiveDoublingSeedDonations);
  counter("sparse-round-components", statistics.sparseRoundComponents);
  counter("sparse-rounds", statistics.sparseRounds);
  counter("no-cut-ddr-components", statistics.noCutDDRComponents);
  counter("cross-tile-ddr-stages", statistics.crossTileDDRStages);
  counter("inter-region-ddr-stages", statistics.interRegionDDRStages);
  counter("output-copies-removed", statistics.outputCopiesRemoved);
  counter("tensor-bridges-removed", statistics.tensorBridgesRemoved);
}

} // namespace wafer::compiler::detail

#endif // WAFER_DRIVER_PHYSICALDATAFLOW_PHYSICALDATAFLOWINSTRUMENTATION_H
