//===- WaferProfileCollection.cpp - Board profile collection -----------===//

#include "WaferProfileCollection.h"

#include "Wafer/ABI/Tx81ProfilerABI.h"
#include "Wafer/Runtime/Profile/ProfilerRecord.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

namespace wafer::runtime::cli {
namespace {

constexpr llvm::StringLiteral kAggregateNames[] = {
    "statistics_window", "fu", "ct", "ne", "rdma", "wdma", "tdma", "scalar"};
constexpr llvm::StringLiteral kEngineNames[] = {"CT",   "NE",   "RDMA",
                                                "WDMA", "TDMA", "DIRECT_DTE"};
constexpr uint32_t kExpectedTraceFlags =
    WAFER_TX81_PROFILER_RECORD_TRACE_ENABLED |
    WAFER_TX81_PROFILER_RECORD_ENTRY_BEGUN |
    WAFER_TX81_PROFILER_RECORD_ENTRY_ENDED |
    WAFER_TX81_PROFILER_RECORD_COMPLETE;

llvm::Error invalid(llvm::Twine message) {
  return llvm::createStringError(llvm::errc::invalid_argument, "%s",
                                 message.str().c_str());
}

struct ProfileCollectionData {
  const ProfiledPackage *profiledPackage = nullptr;
  BoardInvocationFilePlan executionPlan;
  const ProfileCapturePackage *countPackage = nullptr;
  const ProfileCapturePackage *tracePackage = nullptr;
  BoardInvocationFilePlan countPlan;
  BoardInvocationFilePlan tracePlan;
  std::vector<Tx81ProfilerRecord> count;
  std::vector<Tx81ProfilerRecord> trace;
};

/// The stable profile output identity is the external output port.
using SemanticOutputKey = PortId;

SemanticOutputKey semanticOutputKey(const ExternalPortRecord &port) {
  return port.id;
}

/// The exact typed output contract is the package's own port record.
using ExactOutputContract = ExternalPortRecord;

const ExactOutputContract &exactOutputContract(const ExternalPortRecord &port) {
  return port;
}

bool sameExactOutputContract(const ExactOutputContract &lhs,
                             const ExactOutputContract &rhs) {
  return lhs == rhs;
}

struct IndexedOutput {
  const ExternalPortRecord *port = nullptr;
  const BoardRuntimeOutput *output = nullptr;
  std::optional<BoardOutputComparisonKind> externalExpectedComparison;
};

llvm::Expected<std::map<SemanticOutputKey, IndexedOutput>>
indexValidatedOutputs(const PackageManifest &manifest,
                      const BoardInvocationFilePlan &plan,
                      llvm::ArrayRef<BoardRuntimeOutput> outputs) {
  if (llvm::Error error = validateBoardOutputs(outputs, plan))
    return std::move(error);

  std::map<SemanticOutputKey, IndexedOutput> indexed;
  for (const BoardRuntimeOutput &output : outputs) {
    const ExternalPortRecord *port = nullptr;
    for (const ExternalPortRecord &candidate : manifest.outputs)
      if (candidate.id == output.port) {
        port = &candidate;
        break;
      }
    if (!port)
      return invalid("profile output is not an external output port");
    const uint64_t portId = output.port.getValue();
    std::optional<BoardOutputComparisonKind> expectedComparison;
    if (plan.expectedBytes.contains(portId)) {
      auto comparison = plan.expectedComparisons.find(portId);
      if (comparison == plan.expectedComparisons.end())
        return invalid(
            "profile output external expected comparison is missing");
      expectedComparison = comparison->second;
    }
    if (!indexed
             .try_emplace(semanticOutputKey(*port),
                          IndexedOutput{port, &output, expectedComparison})
             .second)
      return invalid("profile output has a duplicate output port");
  }
  return indexed;
}

std::string hashOutputBytes(llvm::ArrayRef<uint8_t> bytes) {
  llvm::SHA256 hasher;
  hasher.update(bytes);
  return "sha256:" + llvm::toHex(hasher.final(), /*LowerCase=*/true);
}

std::string formatSemanticOutputKey(const SemanticOutputKey &key) {
  return (llvm::Twine("output_port=") + llvm::Twine(key.getValue())).str();
}

llvm::Error compareReferenceFile(llvm::StringRef path,
                                 llvm::ArrayRef<uint8_t> actual,
                                 const SemanticOutputKey &key) {
  uint64_t referenceSize = 0;
  if (std::error_code error = llvm::sys::fs::file_size(path, referenceSize))
    return llvm::createStringError(
        error, "failed to inspect staged profile output reference");
  if (referenceSize != actual.size())
    return invalid("staged profile output reference byte count changed for " +
                   formatSemanticOutputKey(key));

  llvm::Expected<llvm::sys::fs::file_t> file =
      llvm::sys::fs::openNativeFileForRead(path);
  if (!file)
    return file.takeError();
  auto close =
      llvm::make_scope_exit([&] { (void)llvm::sys::fs::closeFile(*file); });

  constexpr size_t kComparisonChunkBytes = 64 * 1024;
  std::array<char, kComparisonChunkBytes> buffer{};
  for (uint64_t offset = 0; offset < actual.size();) {
    const size_t wanted = static_cast<size_t>(
        std::min<uint64_t>(buffer.size(), actual.size() - offset));
    size_t filled = 0;
    while (filled < wanted) {
      llvm::Expected<size_t> read = llvm::sys::fs::readNativeFileSlice(
          *file, llvm::MutableArrayRef(buffer).slice(filled, wanted - filled),
          offset + filled);
      if (!read)
        return read.takeError();
      if (*read == 0)
        return invalid("staged profile output reference ended early for " +
                       formatSemanticOutputKey(key));
      filled += *read;
    }
    llvm::ArrayRef<uint8_t> reference(
        reinterpret_cast<const uint8_t *>(buffer.data()), wanted);
    llvm::ArrayRef<uint8_t> actualChunk = actual.slice(offset, wanted);
    if (reference != actualChunk) {
      auto mismatch = llvm::mismatch(reference, actualChunk);
      const uint64_t mismatchOffset =
          offset + static_cast<uint64_t>(mismatch.first - reference.begin());
      return invalid("profile output differs from the profiled-package "
                     "reference for " +
                     formatSemanticOutputKey(key) + " at byte " +
                     llvm::Twine(mismatchOffset));
    }
    offset += wanted;
  }
  return llvm::Error::success();
}

Tx81ProfilerCaptureKind toRuntimeCaptureKind(ProfileCaptureKind capture) {
  switch (capture) {
  case ProfileCaptureKind::Count:
    return Tx81ProfilerCaptureKind::Count;
  case ProfileCaptureKind::Trace:
    return Tx81ProfilerCaptureKind::Trace;
  }
  llvm_unreachable("unknown profile capture kind");
}

llvm::Expected<BoardInvocationFilePlan>
makeCapturePlan(const BoardInvocationFilePlan &primaryPlan,
                const PackageManifest &primaryManifest,
                const ProfileCapturePackage &capture) {
  llvm::Expected<BoardInvocationFilePlan> plan = remapBoardInvocationFilePlan(
      primaryPlan, primaryManifest, capture.getPackage().getManifest());
  if (!plan)
    return plan.takeError();

  plan->request.profilerRecordBytes.emplace();
  plan->request.profilerRecordBytes->reserve(WAFER_TX81_PROFILER_TILE_COUNT);
  for (uint32_t tile = 0; tile < WAFER_TX81_PROFILER_TILE_COUNT; ++tile) {
    llvm::Expected<std::vector<uint8_t>> image = buildTx81ProfilerLaunchImage(
        capture.getRecordBytes(), tile,
        toRuntimeCaptureKind(capture.getCaptureKind()));
    if (!image)
      return image.takeError();
    plan->request.profilerRecordBytes->push_back(std::move(*image));
  }
  return plan;
}

llvm::Expected<std::vector<Tx81ProfilerRecord>>
decodeProfilerOutputs(const ProfileCapturePackage &capture,
                      const BoardRuntimeInvocationResult &result) {
  if (result.profilerOutputs.size() != WAFER_TX81_PROFILER_TILE_COUNT)
    return invalid("board profiler readback is not all-and-only 16 tiles");

  std::vector<Tx81ProfilerRecord> records;
  records.reserve(WAFER_TX81_PROFILER_TILE_COUNT);
  for (const BoardRuntimeProfilerOutput &output : result.profilerOutputs) {
    if (output.bytes.size() != capture.getRecordBytes())
      return invalid("board profiler readback has an unexpected byte count");
    llvm::Expected<Tx81ProfilerRecord> decoded =
        decodeTx81ProfilerRecord(output.bytes);
    if (!decoded)
      return decoded.takeError();
    if (decoded->header.tile_id !=
        static_cast<uint32_t>(output.launchSlot.getValue()))
      return invalid("board profiler record and launch slot disagree");
    records.push_back(std::move(*decoded));
  }
  if (llvm::Error error = verifyTx81ProfilerTileDomain(records))
    return std::move(error);
  llvm::sort(records,
             [](const Tx81ProfilerRecord &lhs, const Tx81ProfilerRecord &rhs) {
               return lhs.header.tile_id < rhs.header.tile_id;
             });

  for (const Tx81ProfilerRecord &record : records) {
    const bool trace =
        (record.header.flags & WAFER_TX81_PROFILER_RECORD_TRACE_ENABLED) != 0;
    const bool count =
        (record.header.flags & WAFER_TX81_PROFILER_RECORD_COUNT_ONLY) != 0;
    if ((capture.getCaptureKind() == ProfileCaptureKind::Count &&
         (!count || trace)) ||
        (capture.getCaptureKind() == ProfileCaptureKind::Trace &&
         (!trace || count)))
      return invalid("board profiler record capture kind disagrees with its "
                     "verified package");
  }
  return records;
}

llvm::StringRef stringifyEventEngine(uint8_t engine) {
  if (engine >= std::size(kEngineNames))
    llvm_unreachable("decoded profiler event has an unknown engine");
  return kEngineNames[engine];
}

llvm::StringRef stringifyEventKind(uint8_t kind) {
  switch (kind) {
  case WAFER_TX81_PROFILER_EVENT_NCC_COMMAND:
    return "ncc-command";
  case WAFER_TX81_PROFILER_EVENT_NCC_COMPLETION_WAIT:
    return "ncc-completion-wait";
  case WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_WAIT:
    return "direct-dte-wait";
  case WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_ISSUE:
    return "direct-dte-issue";
  case WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_PEER_READY_WAIT:
    return "direct-dte-peer-ready-wait";
  case WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_SETUP_ISSUE:
    return "direct-dte-setup-issue";
  case WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_COMPLETION_WAIT:
    return "direct-dte-completion-wait";
  case WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_CLEANUP:
    return "direct-dte-cleanup";
  case WAFER_TX81_PROFILER_EVENT_TARGET_SITE:
    return "target-site";
  }
  llvm_unreachable("decoded profiler event has an unknown kind");
}

bool engineMatches(ProfileTSMEngine expected, uint8_t actual) {
  switch (expected) {
  case ProfileTSMEngine::CT:
    return actual == WAFER_TX81_PROFILER_ENGINE_CT;
  case ProfileTSMEngine::NE:
    return actual == WAFER_TX81_PROFILER_ENGINE_NE;
  case ProfileTSMEngine::RDMA:
    return actual == WAFER_TX81_PROFILER_ENGINE_RDMA;
  case ProfileTSMEngine::WDMA:
    return actual == WAFER_TX81_PROFILER_ENGINE_WDMA;
  case ProfileTSMEngine::TDMA:
    return actual == WAFER_TX81_PROFILER_ENGINE_TDMA;
  case ProfileTSMEngine::DirectDTE:
    return actual == WAFER_TX81_PROFILER_ENGINE_DIRECT_DTE;
  }
  llvm_unreachable("unknown profile site engine");
}

bool eventMatchesSite(const ProfileTargetCallSite &site,
                      const WaferTx81ProfilerTSMCallEvent &event) {
  switch (event.kind) {
  case WAFER_TX81_PROFILER_EVENT_NCC_COMMAND:
    return site.siteKind == ProfileTargetSiteKind::NCCCommand && site.engine &&
           engineMatches(*site.engine, event.engine);
  case WAFER_TX81_PROFILER_EVENT_NCC_COMPLETION_WAIT:
    return (site.siteKind == ProfileTargetSiteKind::NCCCommand ||
            site.siteKind == ProfileTargetSiteKind::NCCCompletion) &&
           event.engine == WAFER_TX81_PROFILER_ENGINE_NONE;
  case WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_WAIT:
  case WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_COMPLETION_WAIT:
  case WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_CLEANUP:
    return site.siteKind == ProfileTargetSiteKind::DirectDTEWait &&
           site.engine && engineMatches(*site.engine, event.engine);
  case WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_ISSUE:
  case WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_PEER_READY_WAIT:
  case WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_SETUP_ISSUE:
    return site.siteKind == ProfileTargetSiteKind::DirectDTEIssue &&
           site.engine && engineMatches(*site.engine, event.engine);
  case WAFER_TX81_PROFILER_EVENT_TARGET_SITE:
    return event.engine == WAFER_TX81_PROFILER_ENGINE_NONE;
  }
  return false;
}

llvm::Error
validateTraceSites(const VerifiedProfileInstrumentation &instrumentation,
                   const ProfileCollectionData &collection) {
  if (collection.trace.size() != WAFER_TX81_PROFILER_TILE_COUNT)
    return invalid("profile trace tile domain is incomplete");
  llvm::ArrayRef<ProfileTileSiteMap> siteMap = instrumentation.getSiteMap();
  if (siteMap.size() != WAFER_TX81_PROFILER_TILE_COUNT)
    return invalid("profile trace has no complete typed site map");
  for (uint32_t launchSlot = 0; launchSlot < WAFER_TX81_PROFILER_TILE_COUNT;
       ++launchSlot) {
    const ProfileTileSiteMap &tileMap = siteMap[launchSlot];
    const int64_t tileId = tileMap.tileId.getValue();
    if (tileMap.cardId != CardId(0) || tileId < 0 ||
        tileId >= WAFER_TX81_PROFILER_TILE_COUNT ||
        tileMap.launchSlot != LaunchSlotId(launchSlot))
      return invalid(
          "profile typed trace/site-map Tile domain is not canonical");
    const Tx81ProfilerRecord &trace = collection.trace[tileId];
    if (trace.header.tile_id != static_cast<uint32_t>(tileId))
      return invalid("profile trace record disagrees with its Tile site map");
    for (const WaferTx81ProfilerTSMCallEvent &event : trace.events) {
      if (!isTx81ProfilerSiteValid(event))
        return invalid("profile trace event has no typed primary site");
      if (event.site_id >= tileMap.sites.size())
        return invalid(
            "profile trace event expectedOutputs an unknown typed site");
      const ProfileTargetCallSite &site = tileMap.sites[event.site_id];
      if (site.siteId != event.site_id || !eventMatchesSite(site, event))
        return invalid("profile trace event conflicts with its typed site");
    }
  }
  return llvm::Error::success();
}

bool sameDevice(const BoardDeviceInfo &lhs, const BoardDeviceInfo &rhs) {
  if (lhs.deviceId != rhs.deviceId ||
      lhs.runtimeVersion != rhs.runtimeVersion ||
      lhs.tileCount != rhs.tileCount || lhs.name != rhs.name ||
      lhs.pciBusId != rhs.pciBusId ||
      lhs.runtimeLibraryDigest != rhs.runtimeLibraryDigest ||
      lhs.tiles.size() != rhs.tiles.size())
    return false;
  return llvm::equal(lhs.tiles, rhs.tiles, [](const auto &a, const auto &b) {
    return a.tileId == b.tileId && a.launchSlot == b.launchSlot &&
           a.available == b.available && a.physicalX == b.physicalX &&
           a.physicalY == b.physicalY;
  });
}

llvm::Error validateTopology(const BoardDeviceInfo &device) {
  if (device.tileCount != WAFER_TX81_PROFILER_TILE_COUNT ||
      device.tiles.size() != WAFER_TX81_PROFILER_TILE_COUNT)
    return invalid("qualified profile device does not expose 16 tiles");
  std::array<bool, WAFER_TX81_PROFILER_TILE_COUNT> seen{};
  std::array<bool, WAFER_TX81_PROFILER_TILE_COUNT> seenLaunchSlots{};
  std::vector<std::pair<uint32_t, uint32_t>> coordinates;
  for (const BoardDeviceInfo::Tile &tile : device.tiles) {
    if (tile.tileId.getValue() < 0 ||
        tile.tileId.getValue() >= WAFER_TX81_PROFILER_TILE_COUNT ||
        seen[tile.tileId.getValue()] || !tile.launchSlot.isValid() ||
        tile.launchSlot.getValue() >= WAFER_TX81_PROFILER_TILE_COUNT ||
        seenLaunchSlots[tile.launchSlot.getValue()] || !tile.available)
      return invalid("qualified profile device tile domain is invalid");
    seen[tile.tileId.getValue()] = true;
    seenLaunchSlots[tile.launchSlot.getValue()] = true;
    if (llvm::is_contained(coordinates,
                           std::pair(tile.physicalX, tile.physicalY)))
      return invalid("qualified profile device has duplicate Tile identity "
                     "coordinates");
    coordinates.emplace_back(tile.physicalX, tile.physicalY);
  }
  return llvm::Error::success();
}

std::string formatSitePosition(const ProfileTargetCallSite &site) {
  if (!site.functionOrdinal && !site.blockOrdinal && !site.instructionOrdinal)
    return {};
  std::string position;
  llvm::raw_string_ostream output(position);
  bool first = true;
  auto append = [&](llvm::StringRef name, std::optional<uint64_t> value) {
    if (!value)
      return;
    if (!first)
      output << ",";
    output << name << "=" << *value;
    first = false;
  };
  append("function", site.functionOrdinal);
  append("block", site.blockOrdinal);
  append("instruction", site.instructionOrdinal);
  return output.str();
}

bool aggregateStable(const WaferTx81ProfilerRecordHeader &header,
                     uint32_t index) {
  const uint32_t bit = UINT32_C(1) << index;
  return (header.pmu_before.stable_mask & bit) != 0 &&
         (header.pmu_after.stable_mask & bit) != 0 &&
         (header.pmu_recovery.stable_mask & bit) != 0;
}

bool enableStable(const WaferTx81ProfilerRecordHeader &header) {
  return header.pmu_before.enable == header.pmu_after.enable &&
         header.pmu_after.enable == header.pmu_recovery.enable;
}

bool pmuEnabled(const WaferTx81ProfilerRecordHeader &header) {
  return enableStable(header) && header.pmu_before.enable != 0;
}

template <typename T>
void emitCounter(llvm::json::OStream &json, T start, T end, T recovery,
                 bool stable, bool enabled) {
  json.object([&] {
    json.attribute("start", start);
    json.attribute("end", end);
    json.attribute("recovery", recovery);
    json.attribute("stable", stable);
    json.attribute("enabled", enabled);
  });
}

void emitRuntimeLaunch(llvm::json::OStream &json,
                       const RuntimeLaunchContract &launch) {
  json.attribute("kind", "kernel");
  const KernelRuntimeLaunchContract &kernel = launch.getKernel();
  json.attribute("form", stringifyKernelLaunchForm(kernel.form));
  json.attribute("entry_abi", stringifyKernelEntryABI(kernel.entryABI));
  json.attributeArray("phases", [&] {
    for (RuntimeLaunchPhaseRole phase : launch.getPhases())
      json.value(stringifyRuntimeLaunchPhaseRole(phase));
  });
}

void emitProfileExperiment(llvm::json::OStream &json,
                           const ProfileCollectionData &collection,
                           llvm::ArrayRef<ProfileTileSiteMap> siteMap) {
  json.object([&] {
    json.attributeArray("clock", [&] {
      for (const ProfileTileSiteMap &tile : siteMap)
        json.object([&] {
          json.attribute("card_id", tile.cardId.getValue());
          json.attribute("tile_id", tile.tileId.getValue());
          json.attribute("launch_slot", tile.launchSlot.getValue());
          json.attribute("slope", 1.0);
          json.attribute("offset", 0.0);
          json.attribute("uncertainty", 0.0);
          json.attribute("round_trips", int64_t(0));
          json.attribute("valid", false);
          json.attribute("monotonic", true);
        });
    });
    json.attributeObject("trace", [&] {
      json.attribute("complete", true);
      json.attributeArray("tiles", [&] {
        for (const ProfileTileSiteMap &tile : siteMap) {
          const Tx81ProfilerRecord &record =
              collection.trace[tile.tileId.getValue()];
          json.object([&] {
            json.attribute("card_id", tile.cardId.getValue());
            json.attribute("tile_id", tile.tileId.getValue());
            json.attribute("launch_slot", tile.launchSlot.getValue());
            json.attribute("entry_begin_cycle",
                           record.header.entry_begin_cycle);
            json.attribute("entry_end_cycle", record.header.entry_end_cycle);
            json.attribute("capacity", int64_t(record.header.event_capacity));
            json.attribute("count", int64_t(record.header.event_count));
            json.attribute(
                "counted_event_count",
                collection.count[record.header.tile_id].header.next_sequence);
            json.attribute("next_sequence", record.header.next_sequence);
            json.attribute("dropped_event_count",
                           int64_t(record.header.dropped_event_count));
            json.attribute("record_flags", int64_t(record.header.flags));
            json.attribute("trace_state", int64_t(record.header.trace_state));
            json.attribute("overflow", false);
            json.attributeArray("events", [&] {
              for (const WaferTx81ProfilerTSMCallEvent &event : record.events)
                json.object([&] {
                  json.attribute("sequence", event.sequence);
                  json.attribute("site_id", int64_t(event.site_id));
                  json.attribute("sub_index", int64_t(event.sub_index));
                  if (event.engine == WAFER_TX81_PROFILER_ENGINE_NONE)
                    json.attribute("engine", llvm::json::Value(nullptr));
                  else
                    json.attribute("engine",
                                   stringifyEventEngine(event.engine));
                  json.attribute("kind", stringifyEventKind(event.kind));
                  json.attribute("observed_begin_cycle",
                                 event.observed_begin_cycle);
                  json.attribute("observed_end_cycle",
                                 event.observed_end_cycle);
                  json.attribute("counter_delta", event.counter_delta);
                  json.attribute("site_begin_cycle", event.site_begin_cycle);
                  json.attribute("site_end_cycle", event.site_end_cycle);
                  json.attribute("operation_begin_cycle",
                                 event.operation_begin_cycle);
                  json.attribute("operation_end_cycle",
                                 event.operation_end_cycle);
                  json.attribute("observation_count",
                                 int64_t(event.observation_count));
                  json.attribute("observed_span_valid",
                                 isTx81ProfilerObservationSpanValid(event));
                  json.attribute("site_span_valid",
                                 isTx81ProfilerSiteSpanValid(event));
                  json.attribute("operation_span_valid",
                                 isTx81ProfilerOperationSpanValid(event));
                  json.attribute("positive_delta",
                                 isTx81ProfilerCounterDeltaPositive(event));
                  json.attribute("attribution_ambiguous",
                                 isTx81ProfilerSameEngineAmbiguous(event));
                  if (event.kind == WAFER_TX81_PROFILER_EVENT_NCC_COMMAND) {
                    json.attribute("ncc_counter_valid",
                                   isTx81ProfilerNCCCounterValid(event));
                    if (!isTx81ProfilerNCCCounterValid(event))
                      json.attribute("observation_status",
                                     "counter-unavailable");
                    else if (isTx81ProfilerSameEngineAmbiguous(event))
                      json.attribute("observation_status",
                                     "attribution-ambiguous");
                    else if (isTx81ProfilerCounterDeltaPositive(event))
                      json.attribute("observation_status",
                                     "engine-delta-bounded");
                    else
                      json.attribute("observation_status", "counter-no-change");
                  } else {
                    json.attribute("ncc_counter_valid",
                                   llvm::json::Value(nullptr));
                    json.attribute("observation_status",
                                   llvm::json::Value(nullptr));
                  }
                  if (isTx81ProfilerWorkerValid(event))
                    json.attribute("worker",
                                   int64_t(getTx81ProfilerWorker(event)));
                  else
                    json.attribute("worker", llvm::json::Value(nullptr));
                  if (isTx81ProfilerLocalWait(event))
                    json.attribute("wait_scope", "local");
                  else if (isTx81ProfilerWorkerWait(event))
                    json.attribute("wait_scope", "worker");
                  else
                    json.attribute("wait_scope", llvm::json::Value(nullptr));
                  if (event.kind ==
                          WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_ISSUE ||
                      event.kind == WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_WAIT)
                    json.attribute("dte_counter_valid",
                                   isTx81ProfilerDirectDTECounterValid(event));
                  else
                    json.attribute("dte_counter_valid",
                                   llvm::json::Value(nullptr));
                  if (isTx81ProfilerDirectDTESend(event))
                    json.attribute("dte_role", "send");
                  else if (isTx81ProfilerDirectDTEReceive(event))
                    json.attribute("dte_role", "receive");
                  else
                    json.attribute("dte_role", llvm::json::Value(nullptr));
                });
            });
            const WaferTx81ProfilerCostSummary &cost =
                record.header.cost_summary;
            json.attributeObject("cost_summary", [&] {
              json.attribute("ncc_pmu_sample_cycles",
                             cost.ncc_pmu_sample_cycles);
              json.attribute("dte_pmu_sample_cycles",
                             cost.dte_pmu_sample_cycles);
              json.attribute("event_bookkeeping_cycles",
                             cost.event_bookkeeping_cycles);
              json.attribute("status_poll_cycles", cost.status_poll_cycles);
              json.attribute("site_hook_cycles", cost.site_hook_cycles);
              json.attribute("completion_loop_bookkeeping_cycles",
                             cost.completion_loop_bookkeeping_cycles);
              json.attribute("entry_setup_cycles", cost.entry_setup_cycles);
              json.attribute("entry_teardown_cycles",
                             cost.entry_teardown_cycles);
            });
          });
        }
      });
    });
    json.attributeObject("pmu", [&] {
      json.attributeArray("tiles", [&] {
        for (const ProfileTileSiteMap &tile : siteMap) {
          const Tx81ProfilerRecord &record =
              collection.trace[tile.tileId.getValue()];
          const WaferTx81ProfilerRecordHeader &header = record.header;
          const bool enabled = pmuEnabled(header);
          json.object([&] {
            json.attribute("card_id", tile.cardId.getValue());
            json.attribute("tile_id", tile.tileId.getValue());
            json.attribute("launch_slot", tile.launchSlot.getValue());
            json.attribute(
                "ncc_pmu_restore_verified",
                (header.summary_validity &
                 WAFER_TX81_PROFILER_SUMMARY_NCC_PMU_RESTORE_VERIFIED) != 0);
            json.attribute(
                "dte_pmu_restore_verified",
                (header.summary_validity &
                 WAFER_TX81_PROFILER_SUMMARY_DTE_PMU_RESTORE_VERIFIED) != 0);
            json.attributeObject("aggregates", [&] {
              for (uint32_t index = 0;
                   index < WAFER_TX81_PROFILER_PMU64_COUNTERS; ++index) {
                json.attributeBegin(kAggregateNames[index]);
                emitCounter(json, header.pmu_before.counters[index],
                            header.pmu_after.counters[index],
                            header.pmu_recovery.counters[index],
                            aggregateStable(header, index), enabled);
                json.attributeEnd();
              }
            });
            json.attributeArray("workers", [&] {
              for (uint32_t worker = 0; worker < WAFER_TX81_PROFILER_WORKERS;
                   ++worker)
                json.object([&] {
                  json.attribute("worker", int64_t(worker));
                  json.attributeArray("engines", [&] {
                    for (uint32_t engine = 0;
                         engine < WAFER_TX81_PROFILER_QUEUES; ++engine)
                      json.object([&] {
                        json.attribute("engine", kEngineNames[engine]);
                        json.attributeBegin("instructions");
                        emitCounter(
                            json,
                            header.pmu_before.instructions[worker][engine],
                            header.pmu_after.instructions[worker][engine],
                            header.pmu_recovery.instructions[worker][engine],
                            enableStable(header), enabled);
                        json.attributeEnd();
                        json.attributeBegin("blocking");
                        emitCounter(
                            json, header.pmu_before.blocking[worker][engine],
                            header.pmu_after.blocking[worker][engine],
                            header.pmu_recovery.blocking[worker][engine],
                            enableStable(header), enabled);
                        json.attributeEnd();
                      });
                  });
                });
            });
          });
        }
      });
    });
  });
}

llvm::Expected<std::string>
serializeEvidence(const VerifiedProfileInstrumentation &instrumentation,
                  llvm::StringRef runId, const BoardDeviceInfo &device,
                  llvm::ArrayRef<BoardProfileMeasurementSample> samples,
                  const ProfileCollectionData &collection,
                  const BoardProfileOutputValidator &outputValidation) {
  if (samples.size() != 1)
    return invalid("profile evidence inputs are incomplete");
  const PackageManifest &primaryManifest =
      collection.profiledPackage->getPackage().getManifest();
  llvm::ArrayRef<ProfileTileSiteMap> siteMap = instrumentation.getSiteMap();
  if (siteMap.size() != WAFER_TX81_PROFILER_TILE_COUNT)
    return invalid("profile evidence has no complete Tile site map");
  const std::string targetIdentity =
      stringifyTargetIdentityId(primaryManifest.targetIdentity).str();
  std::string storage;
  llvm::raw_string_ostream output(storage);
  llvm::json::OStream json(output, 2);
  json.object([&] {
    json.attribute("schema", "wafer.profile.evidence");
    json.attribute("run_id", runId);
    json.attributeObject("program", [&] {
      json.attribute("program_manifest_sha256",
                     collection.profiledPackage->getManifestDigest());
      json.attribute("target_identity", targetIdentity);
      json.attributeObject(
          "launch", [&] { emitRuntimeLaunch(json, primaryManifest.launch); });
      json.attribute("card_count", int64_t(1));
      json.attribute("tile_count", int64_t(WAFER_TX81_PROFILER_TILE_COUNT));
      json.attribute("site_correlation_basis", kProfileSiteCorrelationBasis);
      json.attribute("record_abi", kProfileRecordABI);
    });
    json.attributeBegin("static_cost_model");
    writeProfileStaticCostModel(
        json, collection.profiledPackage->getStaticCostModel());
    json.attributeEnd();
    json.attributeObject("output_validation", [&] {
      json.attribute("mode", stringifyBoardProfileOutputValidationMode(
                                 outputValidation.getMode()));
      json.attributeArray("resources", [&] {
        for (const BoardProfileOutputValidationResource &resource :
             outputValidation.getResources())
          json.object([&] {
            json.attribute("port", int64_t(resource.port.getValue()));
            json.attribute("role_index", resource.roleIndex);
            json.attribute("bytes", int64_t(resource.bytes));
            json.attribute("reference_sha256", resource.referenceSha256);
            if (resource.externalExpectedComparison)
              json.attribute("external_expected_comparison",
                             *resource.externalExpectedComparison ==
                                     BoardOutputComparisonKind::Exact
                                 ? "exact"
                                 : "relaxed-f16");
            else
              json.attribute("external_expected_comparison",
                             llvm::json::Value(nullptr));
            json.attribute("primary_output_validated",
                           resource.primaryOutputValidated);
            json.attribute("diagnostic_captures_match_primary",
                           resource.diagnosticCapturesMatchPrimary);
          });
      });
    });
    json.attributeArray("topology", [&] {
      for (uint32_t launchSlot = 0; launchSlot < WAFER_TX81_PROFILER_TILE_COUNT;
           ++launchSlot) {
        auto tile = llvm::find_if(device.tiles, [&](const auto &collection) {
          return collection.launchSlot == LaunchSlotId(launchSlot);
        });
        json.object([&] {
          json.attribute("card_id", int64_t(0));
          json.attribute("tile_id", tile->tileId.getValue());
          json.attribute("launch_slot", int64_t(launchSlot));
          json.attribute("x", int64_t(tile->physicalX));
          json.attribute("y", int64_t(tile->physicalY));
        });
      }
    });
    json.attributeObject("measurement", [&] {
      json.attributeArray("samples", [&] {
        for (const BoardProfileMeasurementSample &sample : samples)
          json.object([&] {
            json.attribute("sample_id", sample.id);
            json.attribute("sample_index", int64_t(sample.sampleIndex));
            json.attribute("device_elapsed_ns",
                           sample.deviceElapsedNanoseconds);
            json.attribute("device_timer_kind", sample.deviceTimerKind);
            json.attribute("host_submit_ns", sample.hostSubmitNanoseconds);
            json.attribute("host_launch_to_completion_ns",
                           sample.hostLaunchToCompletionNanoseconds);
            json.attribute("completion_observation_resolution_ns",
                           sample.completionObservationResolutionNanoseconds);
          });
      });
    });
    json.attributeArray("sites", [&] {
      for (const ProfileTileSiteMap &tile : siteMap)
        for (const ProfileTargetCallSite &site : tile.sites)
          json.object([&] {
            json.attribute("card_id", tile.cardId.getValue());
            json.attribute("tile_id", tile.tileId.getValue());
            json.attribute("launch_slot", tile.launchSlot.getValue());
            json.attribute("site_id", site.siteId);
            json.attribute("site_kind",
                           stringifyProfileTargetSiteKind(site.siteKind));
            json.attribute("correlation_key", site.correlationKey);
            if (site.engine)
              json.attribute("engine", stringifyProfileTSMEngine(*site.engine));
            else
              json.attribute("engine", llvm::json::Value(nullptr));
            json.attribute("target_call_ordinal", site.targetCallOrdinal);
            json.attribute("target_call_symbol", site.targetCallSymbol);
            std::string position = formatSitePosition(site);
            if (!position.empty())
              json.attribute("position", position);
          });
    });
    json.attributeObject("validity", [&] {
      json.attribute("environment", true);
      json.attribute("profile_instrumentation", true);
      json.attribute("measurement_basis", true);
    });
    json.attributeBegin("experiment");
    emitProfileExperiment(json, collection, siteMap);
    json.attributeEnd();
  });
  output << "\n";
  return output.str();
}

llvm::Error ensureRunsDirectory(llvm::StringRef instrumentationRoot,
                                llvm::SmallVectorImpl<char> &runs) {
  runs.assign(instrumentationRoot.begin(), instrumentationRoot.end());
  llvm::sys::path::append(runs, "runs");
  llvm::sys::fs::file_status status;
  std::error_code error = llvm::sys::fs::status(runs, status, /*follow=*/false);
  if (error == std::errc::no_such_file_or_directory) {
    error = llvm::sys::fs::create_directory(runs);
    if (error)
      return llvm::createStringError(error,
                                     "failed to create profile runs directory");
  } else if (error) {
    return llvm::createStringError(error,
                                   "failed to inspect profile runs directory");
  } else if (status.type() != llvm::sys::fs::file_type::directory_file) {
    return invalid("profile runs path is not a real directory");
  }
  llvm::SmallString<256> canonical;
  error = llvm::sys::fs::real_path(runs, canonical);
  if (error)
    return llvm::createStringError(error,
                                   "failed to resolve profile runs directory");
  llvm::SmallString<256> expected(instrumentationRoot);
  llvm::sys::path::append(expected, "runs");
  if (canonical != expected)
    return invalid(
        "profile runs directory escapes the verified instrumentation");
  runs.assign(canonical.begin(), canonical.end());
  error = llvm::sys::fs::setPermissions(runs, llvm::sys::fs::all_all);
  if (error)
    return llvm::createStringError(
        error, "failed to set profile runs directory permissions");
  return llvm::Error::success();
}

llvm::Error validateProfileReportMembers(llvm::StringRef directory) {
  size_t publicMemberCount = 0;
  std::error_code walkError;
  for (llvm::sys::fs::directory_iterator
           iterator(directory, walkError, /*follow_symlinks=*/false),
       end;
       iterator != end; iterator.increment(walkError)) {
    if (walkError)
      return llvm::createStringError(
          walkError, "failed to inspect profile report members");
    const llvm::StringRef name = llvm::sys::path::filename(iterator->path());
    if (iterator->type() != llvm::sys::fs::file_type::regular_file ||
        (name != "evidence.json" && name != "analysis.json" &&
         name != "index.html"))
      return invalid("profile report contains an unexpected file");
    ++publicMemberCount;
  }
  if (walkError)
    return llvm::createStringError(walkError,
                                   "failed to inspect profile report members");
  if (publicMemberCount != 3)
    return invalid("profile report file set is incomplete");
  return llvm::Error::success();
}

llvm::Error validateProfileEvidenceRunId(llvm::StringRef directory,
                                         llvm::StringRef expectedRunId) {
  llvm::SmallString<256> evidencePath(directory);
  llvm::sys::path::append(evidencePath, "evidence.json");
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> evidence =
      llvm::MemoryBuffer::getFile(evidencePath, /*IsText=*/true);
  if (!evidence)
    return llvm::createStringError(evidence.getError(),
                                   "failed to read managed profile evidence");
  llvm::Expected<llvm::json::Value> parsed =
      llvm::json::parse((*evidence)->getBuffer());
  if (!parsed)
    return invalid("managed profile evidence is not valid JSON: " +
                   llvm::toString(parsed.takeError()));
  llvm::json::Object *object = parsed->getAsObject();
  if (!object)
    return invalid("managed profile evidence root is not an object");
  std::optional<llvm::StringRef> runId = object->getString("run_id");
  if (!runId)
    return invalid("managed profile evidence is missing run_id");
  if (*runId != expectedRunId)
    return invalid("managed profile evidence run_id does not match its "
                   "directory basename");
  return llvm::Error::success();
}

llvm::Error validateManagedProfileRun(llvm::StringRef directory,
                                      llvm::StringRef expectedRunId) {
  if (llvm::Error reportError = validateProfileReportMembers(directory))
    return reportError;
  return validateProfileEvidenceRunId(directory, expectedRunId);
}

llvm::Expected<std::optional<std::string>>
resolveCurrentProfileRun(llvm::StringRef runsDirectory) {
  llvm::SmallString<256> current(runsDirectory);
  llvm::sys::path::append(current, "current");
  llvm::sys::fs::file_status status;
  std::error_code error =
      llvm::sys::fs::status(current, status, /*follow=*/false);
  if (error == std::errc::no_such_file_or_directory)
    return std::optional<std::string>();
  if (error)
    return llvm::createStringError(
        error, "failed to inspect current profile report entry");
  if (status.type() != llvm::sys::fs::file_type::symlink_file)
    return invalid("current profile report entry is not a managed link");

  llvm::SmallString<256> resolved;
  error = llvm::sys::fs::real_path(current, resolved);
  if (error)
    return llvm::createStringError(
        error, "failed to resolve current profile report entry");
  if (llvm::sys::path::parent_path(resolved) != runsDirectory ||
      !llvm::sys::path::filename(resolved).starts_with("run-") ||
      llvm::sys::fs::get_file_type(resolved, /*Follow=*/false) !=
          llvm::sys::fs::file_type::directory_file)
    return invalid("current profile report entry does not name a managed run");
  if (llvm::Error reportError = validateManagedProfileRun(
          resolved, llvm::sys::path::filename(resolved)))
    return std::move(reportError);
  return std::optional<std::string>(resolved.str().str());
}

llvm::Expected<std::string> findProfileResource(llvm::StringRef name) {
  static int executableAnchor = 0;
  std::string executable = llvm::sys::fs::getMainExecutable(
      /*argv0=*/nullptr, &executableAnchor);
  if (executable.empty())
    return invalid("failed to resolve wafer-run executable");
  llvm::SmallString<256> resource(llvm::sys::path::parent_path(executable));
  llvm::sys::path::append(resource, "..", "share", "wafer", name);
  llvm::SmallString<256> canonical;
  if (std::error_code error = llvm::sys::fs::real_path(resource, canonical))
    return llvm::createStringError(
        error, "failed to resolve installed profile resource: " + name);
  if (llvm::sys::fs::get_file_type(canonical, /*Follow=*/false) !=
      llvm::sys::fs::file_type::regular_file)
    return invalid("installed profile resource is not a regular file: " + name);
  return canonical.str().str();
}

llvm::Expected<std::pair<std::string, std::string>>
createStagingRun(llvm::StringRef instrumentationRoot) {
  llvm::SmallString<256> runs;
  if (llvm::Error error = ensureRunsDirectory(instrumentationRoot, runs))
    return std::move(error);
  for (unsigned attempt = 0; attempt < 32; ++attempt) {
    llvm::SmallString<256> model(runs);
    llvm::sys::path::append(model, ".staging-%%%%%%");
    llvm::SmallString<256> staging;
    if (std::error_code error =
            llvm::sys::fs::createUniqueDirectory(model, staging))
      return llvm::createStringError(
          error, "failed to create profile run staging directory");
    llvm::StringRef suffix = llvm::sys::path::filename(staging).drop_front(
        llvm::StringRef(".staging-").size());
    std::string runId = ("run-" + suffix).str();
    llvm::SmallString<256> finalPath(runs);
    llvm::sys::path::append(finalPath, runId);
    if (!llvm::sys::fs::exists(finalPath))
      return std::pair(runId, staging.str().str());
    (void)llvm::sys::fs::remove_directories(staging);
  }
  return invalid("failed to choose a unique profile run directory name");
}

struct PendingProfileReportRun {
  std::string runId;
  std::string stagingDirectory;
  std::string runDirectory;
  std::string runsDirectory;
  std::string currentEntry;
  std::optional<std::string> previousRunDirectory;
  std::string python;
  std::string reportScript;
};

llvm::Expected<PendingProfileReportRun>
createPendingProfileReportRun(llvm::StringRef instrumentationRoot) {
  llvm::ErrorOr<std::string> python = llvm::sys::findProgramByName("python3");
  if (!python)
    return llvm::createStringError(
        python.getError(),
        "failed to find python3 for the installed profile report");
  if (!llvm::sys::fs::can_execute(*python))
    return invalid("resolved python3 is not executable");
  llvm::Expected<std::string> report =
      findProfileResource("wafer_profile_report.py");
  if (!report)
    return report.takeError();
  llvm::Expected<std::string> schema =
      findProfileResource("wafer_profile_evidence.schema.json");
  if (!schema)
    return schema.takeError();
  (void)schema;

  llvm::SmallString<256> runs;
  if (llvm::Error error = ensureRunsDirectory(instrumentationRoot, runs))
    return std::move(error);
  llvm::Expected<std::optional<std::string>> previous =
      resolveCurrentProfileRun(runs);
  if (!previous)
    return previous.takeError();
  llvm::Expected<std::pair<std::string, std::string>> run =
      createStagingRun(instrumentationRoot);
  if (!run)
    return run.takeError();
  llvm::SmallString<256> runPath(instrumentationRoot);
  llvm::sys::path::append(runPath, "runs", run->first);
  llvm::SmallString<256> current(runs);
  llvm::sys::path::append(current, "current");
  return PendingProfileReportRun{
      run->first,       run->second,         runPath.str().str(),
      runs.str().str(), current.str().str(), std::move(*previous),
      *python,          std::move(*report)};
}

using RemoveManagedProfileRun =
    llvm::function_ref<std::error_code(llvm::StringRef)>;

llvm::Expected<std::string>
replaceCurrentProfileReport(PendingProfileReportRun &reportRun,
                            RemoveManagedProfileRun removeManagedRun) {
  for (llvm::StringRef name :
       {llvm::StringRef("evidence.json"), llvm::StringRef("analysis.json"),
        llvm::StringRef("index.html")}) {
    llvm::SmallString<256> path(reportRun.stagingDirectory);
    llvm::sys::path::append(path, name);
    if (llvm::sys::fs::get_file_type(path, /*Follow=*/false) !=
        llvm::sys::fs::file_type::regular_file)
      return invalid("profile report generator omitted " + name);
    if (std::error_code error =
            llvm::sys::fs::setPermissions(path, llvm::sys::fs::all_all))
      return llvm::createStringError(
          error, "failed to set profile report file permissions");
  }
  if (llvm::Error reportError = validateManagedProfileRun(
          reportRun.stagingDirectory, reportRun.runId))
    return std::move(reportError);
  if (std::error_code error = llvm::sys::fs::setPermissions(
          reportRun.stagingDirectory, llvm::sys::fs::all_all))
    return llvm::createStringError(
        error, "failed to set profile report directory permissions");

  if (std::error_code error = llvm::sys::fs::rename(reportRun.stagingDirectory,
                                                    reportRun.runDirectory))
    return llvm::createStringError(
        error, "failed to rename staged profile report directory");
  reportRun.stagingDirectory.clear();
  bool currentReplaced = false;
  llvm::SmallString<256> temporaryCurrent(reportRun.runsDirectory);
  llvm::sys::path::append(temporaryCurrent, ".current-" + reportRun.runId);
  auto rollbackNewRun = llvm::make_scope_exit([&] {
    (void)llvm::sys::fs::remove(temporaryCurrent);
    if (!currentReplaced)
      (void)llvm::sys::fs::remove_directories(reportRun.runDirectory);
  });
  if (std::error_code error =
          llvm::sys::fs::create_link(reportRun.runDirectory, temporaryCurrent))
    return llvm::createStringError(
        error, "failed to create atomic current profile report entry");

  llvm::Expected<std::optional<std::string>> observedPrevious =
      resolveCurrentProfileRun(reportRun.runsDirectory);
  if (!observedPrevious)
    return observedPrevious.takeError();
  if (*observedPrevious != reportRun.previousRunDirectory)
    return invalid("current profile report changed during this collection");
  if (std::error_code error =
          llvm::sys::fs::rename(temporaryCurrent, reportRun.currentEntry))
    return llvm::createStringError(
        error, "failed to replace the current profile report entry");
  currentReplaced = true;
  rollbackNewRun.release();

  if (reportRun.previousRunDirectory &&
      *reportRun.previousRunDirectory != reportRun.runDirectory) {
    llvm::StringRef previousRunId =
        llvm::sys::path::filename(*reportRun.previousRunDirectory);
    if (llvm::Error identityError = validateManagedProfileRun(
            *reportRun.previousRunDirectory, previousRunId))
      return std::move(identityError);
    if (std::error_code error =
            removeManagedRun(*reportRun.previousRunDirectory))
      return llvm::createStringError(
          error,
          "the current profile report was replaced, but the previous run "
          "could not be removed");
  }
  return reportRun.currentEntry;
}

llvm::Expected<std::string>
writeProfileReport(PendingProfileReportRun &reportRun,
                   const VerifiedProfileInstrumentation &instrumentation,
                   const BoardDeviceInfo &device,
                   llvm::ArrayRef<BoardProfileMeasurementSample> samples,
                   const ProfileCollectionData &collection,
                   const BoardProfileOutputValidator &outputValidation) {
  llvm::Expected<std::string> evidence =
      serializeEvidence(instrumentation, reportRun.runId, device, samples,
                        collection, outputValidation);
  if (!evidence)
    return evidence.takeError();
  llvm::SmallString<256> evidencePath(reportRun.stagingDirectory);
  llvm::sys::path::append(evidencePath, "evidence.json");
  std::error_code outputError;
  llvm::raw_fd_ostream output(evidencePath, outputError,
                              llvm::sys::fs::OF_Text);
  if (outputError)
    return llvm::createStringError(outputError,
                                   "failed to create profile evidence");
  output << *evidence;
  output.close();
  if (output.has_error())
    return llvm::createStringError(output.error(),
                                   "failed to write profile evidence");

  std::string evidenceArgument = evidencePath.str().str();
  std::string outputArgument = reportRun.stagingDirectory;
  llvm::SmallVector<llvm::StringRef, 6> arguments = {
      reportRun.python, "-B", reportRun.reportScript, evidenceArgument,
      "--output-directory", outputArgument};
  std::array<std::optional<llvm::StringRef>, 3> redirects = {
      std::nullopt, llvm::StringRef("/dev/null"), std::nullopt};
  std::string executionError;
  bool executionFailed = false;
  int exitCode = llvm::sys::ExecuteAndWait(
      reportRun.python, arguments, std::nullopt, redirects,
      /*SecondsToWait=*/0,
      /*MemoryLimit=*/0, &executionError, &executionFailed);
  if (executionFailed || exitCode != 0)
    return invalid(
        "profile report generation failed" +
        (executionError.empty()
             ? llvm::Twine(" with exit code ") + llvm::Twine(exitCode)
             : llvm::Twine(": ") + executionError));

  return replaceCurrentProfileReport(
      reportRun, [](llvm::StringRef runDirectory) {
        return llvm::sys::fs::remove_directories(runDirectory,
                                                 /*IgnoreErrors=*/false);
      });
}

} // namespace

#if defined(WAFER_PROFILE_COLLECTION_TESTING)
namespace testing {

llvm::Expected<ProfileReportWriteResult> writeProfileReportForTesting(
    llvm::StringRef instrumentationRoot,
    llvm::function_ref<llvm::Error(llvm::StringRef runId,
                                   llvm::StringRef stagingDirectory)>
        stage,
    llvm::function_ref<std::error_code(llvm::StringRef)> removeManagedRun) {
  llvm::SmallString<256> runs;
  if (llvm::Error error = ensureRunsDirectory(instrumentationRoot, runs))
    return std::move(error);
  llvm::Expected<std::optional<std::string>> previous =
      resolveCurrentProfileRun(runs);
  if (!previous)
    return previous.takeError();
  llvm::Expected<std::pair<std::string, std::string>> run =
      createStagingRun(instrumentationRoot);
  if (!run)
    return run.takeError();

  llvm::SmallString<256> finalPath(runs);
  llvm::sys::path::append(finalPath, run->first);
  llvm::SmallString<256> current(runs);
  llvm::sys::path::append(current, "current");
  PendingProfileReportRun reportRun{
      run->first,         run->second,         finalPath.str().str(),
      runs.str().str(),   current.str().str(), std::move(*previous),
      /*python=*/"",
      /*reportScript=*/""};
  auto cleanupStaging = llvm::make_scope_exit([&] {
    if (!reportRun.stagingDirectory.empty())
      (void)llvm::sys::fs::remove_directories(reportRun.stagingDirectory);
  });
  if (llvm::Error error = stage(reportRun.runId, reportRun.stagingDirectory))
    return std::move(error);
  llvm::Expected<std::string> currentEntry =
      replaceCurrentProfileReport(reportRun, removeManagedRun);
  if (!currentEntry)
    return currentEntry.takeError();
  return ProfileReportWriteResult{reportRun.runId, reportRun.runDirectory,
                                  std::move(*currentEntry)};
}

} // namespace testing
#endif

struct BoardProfileOutputValidator::Impl {
  struct ExpectedOutput {
    SemanticOutputKey key;
    ExactOutputContract contract;
    std::string path;
  };

  explicit Impl(std::string stagingDirectory)
      : stagingDirectory(std::move(stagingDirectory)) {}

  std::string stagingDirectory;
  BoardProfileOutputValidationMode mode =
      BoardProfileOutputValidationMode::SameSessionPrimary;
  bool primaryOutputsRecorded = false;
  bool referenceFilesRemoved = false;
  uint32_t diagnosticCaptureCount = 0;
  std::vector<BoardProfileOutputValidationResource> resources;
  std::vector<ExpectedOutput> expectedOutputs;
  std::map<SemanticOutputKey, size_t> outputIndices;
};

llvm::StringRef stringifyBoardProfileOutputValidationMode(
    BoardProfileOutputValidationMode mode) {
  switch (mode) {
  case BoardProfileOutputValidationMode::ExternalExpected:
    return "external-expected";
  case BoardProfileOutputValidationMode::Mixed:
    return "mixed";
  case BoardProfileOutputValidationMode::SameSessionPrimary:
    return "same-session-primary";
  }
  llvm_unreachable("unknown board profile output validation mode");
}

BoardProfileOutputValidator::BoardProfileOutputValidator(
    std::string stagingDirectory)
    : impl(std::make_unique<Impl>(std::move(stagingDirectory))) {}

BoardProfileOutputValidator::~BoardProfileOutputValidator() {
  if (!impl)
    return;
  for (const Impl::ExpectedOutput &reference : impl->expectedOutputs)
    if (!reference.path.empty())
      (void)llvm::sys::fs::remove(reference.path);
}

BoardProfileOutputValidator::BoardProfileOutputValidator(
    BoardProfileOutputValidator &&) = default;

BoardProfileOutputValidator &
BoardProfileOutputValidator::operator=(BoardProfileOutputValidator &&other) {
  if (this == &other)
    return *this;
  if (impl)
    for (const Impl::ExpectedOutput &reference : impl->expectedOutputs)
      if (!reference.path.empty())
        (void)llvm::sys::fs::remove(reference.path);
  impl = std::move(other.impl);
  return *this;
}

llvm::Error BoardProfileOutputValidator::recordPrimaryOutputs(
    const PackageManifest &manifest, const BoardInvocationFilePlan &plan,
    llvm::ArrayRef<BoardRuntimeOutput> outputs) {
  if (impl->primaryOutputsRecorded)
    return invalid(
        "profiled-package output reference is already primaryOutputsRecorded");
  if (impl->stagingDirectory.empty() ||
      llvm::sys::fs::get_file_type(impl->stagingDirectory,
                                   /*Follow=*/false) !=
          llvm::sys::fs::file_type::directory_file)
    return invalid("profile output reference staging directory is invalid");

  llvm::Expected<std::map<SemanticOutputKey, IndexedOutput>> indexed =
      indexValidatedOutputs(manifest, plan, outputs);
  if (!indexed)
    return indexed.takeError();
  if (indexed->empty())
    return invalid("profile collection has no writable output resource");

  std::vector<BoardProfileOutputValidationResource> resources;
  std::vector<Impl::ExpectedOutput> expectedOutputs;
  std::map<SemanticOutputKey, size_t> outputIndices;
  resources.reserve(indexed->size());
  expectedOutputs.reserve(indexed->size());
  size_t externalExpectedCount = 0;
  auto cleanup = llvm::make_scope_exit([&] {
    for (const Impl::ExpectedOutput &reference : expectedOutputs)
      if (!reference.path.empty())
        (void)llvm::sys::fs::remove(reference.path);
  });

  for (const auto &[key, output] : *indexed) {
    llvm::SmallString<256> model(impl->stagingDirectory);
    llvm::sys::path::append(model, ".output-reference-%%%%%%");
    int descriptor = -1;
    llvm::SmallString<256> referencePath;
    if (std::error_code error =
            llvm::sys::fs::createUniqueFile(model, descriptor, referencePath))
      return llvm::createStringError(
          error, "failed to create staged profile output reference");
    llvm::raw_fd_ostream referenceFile(descriptor, /*shouldClose=*/true);
    referenceFile.write(
        reinterpret_cast<const char *>(output.output->bytes.data()),
        output.output->bytes.size());
    referenceFile.close();
    if (referenceFile.has_error()) {
      const std::error_code error = referenceFile.error();
      (void)llvm::sys::fs::remove(referencePath);
      return llvm::createStringError(
          error, "failed to write staged profile output reference");
    }

    const size_t index = resources.size();
    outputIndices.emplace(key, index);
    expectedOutputs.push_back(
        {key, exactOutputContract(*output.port), referencePath.str().str()});
    resources.push_back({output.port->id, output.port->roleIndex,
                         output.port->bytes,
                         hashOutputBytes(output.output->bytes),
                         output.externalExpectedComparison, true, false});
    if (output.externalExpectedComparison)
      ++externalExpectedCount;
  }

  if (externalExpectedCount == resources.size())
    impl->mode = BoardProfileOutputValidationMode::ExternalExpected;
  else if (externalExpectedCount == 0)
    impl->mode = BoardProfileOutputValidationMode::SameSessionPrimary;
  else
    impl->mode = BoardProfileOutputValidationMode::Mixed;
  impl->resources = std::move(resources);
  impl->expectedOutputs = std::move(expectedOutputs);
  impl->outputIndices = std::move(outputIndices);
  impl->primaryOutputsRecorded = true;
  cleanup.release();
  return llvm::Error::success();
}

llvm::Error BoardProfileOutputValidator::compareWithPrimaryOutputs(
    const PackageManifest &manifest, const BoardInvocationFilePlan &plan,
    llvm::ArrayRef<BoardRuntimeOutput> outputs) {
  if (!impl->primaryOutputsRecorded)
    return invalid(
        "profiled-package output reference is not primaryOutputsRecorded");
  if (impl->referenceFilesRemoved)
    return invalid("profile output expectedOutputs were already removed");
  if (impl->diagnosticCaptureCount >= 2)
    return invalid("count and trace output equivalence is already validated");

  // External expected tensors are authoritative and are always checked before
  // the same-session comparison.
  llvm::Expected<std::map<SemanticOutputKey, IndexedOutput>> indexed =
      indexValidatedOutputs(manifest, plan, outputs);
  if (!indexed)
    return indexed.takeError();
  if (indexed->size() != impl->expectedOutputs.size())
    return invalid("profile output semantic resource domain changed");

  for (const auto &[key, output] : *indexed) {
    auto outputIndex = impl->outputIndices.find(key);
    if (outputIndex == impl->outputIndices.end())
      return invalid("profile output semantic resource domain changed");
    const size_t index = outputIndex->second;
    const Impl::ExpectedOutput &reference = impl->expectedOutputs[index];
    if (!sameExactOutputContract(reference.contract,
                                 exactOutputContract(*output.port)))
      return invalid("profile output typed port contract changed for " +
                     formatSemanticOutputKey(key));
    if (impl->resources[index].externalExpectedComparison !=
        output.externalExpectedComparison)
      return invalid("profile output external expected policy changed for " +
                     formatSemanticOutputKey(key));
    if (llvm::Error error =
            compareReferenceFile(reference.path, output.output->bytes, key))
      return error;
  }

  ++impl->diagnosticCaptureCount;
  if (impl->diagnosticCaptureCount == 2)
    for (BoardProfileOutputValidationResource &resource : impl->resources)
      resource.diagnosticCapturesMatchPrimary = true;
  return llvm::Error::success();
}

llvm::Error BoardProfileOutputValidator::validateDiagnosticOutputs(
    const PackageManifest &manifest, const BoardInvocationFilePlan &plan,
    llvm::ArrayRef<BoardRuntimeOutput> outputs) {
  return compareWithPrimaryOutputs(manifest, plan, outputs);
}

llvm::Error BoardProfileOutputValidator::removeReferenceFiles() {
  if (!impl->primaryOutputsRecorded)
    return invalid(
        "profiled-package output reference is not primaryOutputsRecorded");
  if (impl->diagnosticCaptureCount != 2)
    return invalid("count and trace output equivalence with the primary was "
                   "not validated");
  if (impl->referenceFilesRemoved)
    return invalid("profile output expectedOutputs were already removed");
  for (Impl::ExpectedOutput &reference : impl->expectedOutputs) {
    if (std::error_code error = llvm::sys::fs::remove(reference.path))
      return llvm::createStringError(
          error, "failed to remove staged profile output reference");
    reference.path.clear();
  }
  impl->referenceFilesRemoved = true;
  return llvm::Error::success();
}

BoardProfileOutputValidationMode BoardProfileOutputValidator::getMode() const {
  return impl->mode;
}

llvm::ArrayRef<BoardProfileOutputValidationResource>
BoardProfileOutputValidator::getResources() const {
  return impl->resources;
}

llvm::Expected<BoardProfileProtocolResult> runFixedBoardProfileProtocol(
    uint64_t traceCapacity,
    llvm::function_ref<llvm::Expected<BoardProfileProtocolObservation>(
        const BoardProfileProtocolStep &)>
        execute,
    llvm::function_ref<
        llvm::Error(llvm::ArrayRef<BoardProfileMeasurementSample>)>
        consumeMeasurements) {
  auto invoke = [&](BoardProfileProtocolLaunch launch)
      -> llvm::Expected<BoardProfileProtocolObservation> {
    return execute({launch});
  };

  llvm::Expected<BoardProfileProtocolObservation> primary =
      invoke(BoardProfileProtocolLaunch::Primary);
  if (!primary)
    return primary.takeError();
  if (!primary->deviceExecutionNanoseconds)
    return invalid("primary board device execution timing is unavailable");
  if (primary->hostSubmitNanoseconds > primary->launchToCompletionNanoseconds)
    return invalid("primary board host submit observation exceeds the "
                   "launch-to-completion envelope");

  BoardProfileProtocolResult result;
  result.samples.push_back({
      "primary",
      /*sampleIndex=*/0,
      *primary->deviceExecutionNanoseconds,
      "tx-stream-events",
      primary->hostSubmitNanoseconds,
      primary->launchToCompletionNanoseconds,
      primary->completionObservationResolutionNanoseconds,
  });
  result.primarySampleId = "primary";

  llvm::Expected<BoardProfileProtocolObservation> count =
      invoke(BoardProfileProtocolLaunch::Count);
  if (!count)
    return count.takeError();
  for (uint64_t sequence : count->countSequences)
    if (sequence > traceCapacity)
      return invalid("profile count exceeds the fixed trace package capacity");

  llvm::Expected<BoardProfileProtocolObservation> trace =
      invoke(BoardProfileProtocolLaunch::Trace);
  if (!trace)
    return trace.takeError();
  for (uint32_t tile = 0; tile < WAFER_TX81_PROFILER_TILE_COUNT; ++tile) {
    const BoardProfileTraceTileAudit &audit = trace->trace[tile];
    if (audit.countedEventCount != count->countSequences[tile] ||
        audit.countedEventCount != audit.nextSequence ||
        audit.nextSequence != audit.storedEventCount ||
        audit.droppedEventCount != 0 ||
        audit.recordFlags != kExpectedTraceFlags ||
        audit.traceState != WAFER_TX81_PROFILER_TRACE_COMPLETE)
      return invalid("profile trace audit differs from the count pass");
  }

  if (result.samples.size() != 1 || result.primarySampleId.empty())
    return invalid("primary profile measurement is incomplete");
  if (llvm::Error error = consumeMeasurements(result.samples))
    return std::move(error);
  return result;
}

llvm::Expected<BoardProfileCollectionResult>
runBoardProfileCollection(const VerifiedProfileInstrumentation &instrumentation,
                          const PackageManifest &primaryManifest,
                          const BoardInvocationFilePlan &primaryPlan,
                          BoardRuntimeDriver &driver) {
  if (primaryManifest.cardCount != 1 ||
      primaryManifest.tileCount != WAFER_TX81_PROFILER_TILE_COUNT)
    return invalid(
        "profile collection requires a complete one-card, 16-Tile package");

  ProfileCollectionData collection;
  collection.profiledPackage = &instrumentation.getProfiledPackage();

  llvm::Expected<BoardInvocationFilePlan> execution =
      remapBoardInvocationFilePlan(
          primaryPlan, primaryManifest,
          collection.profiledPackage->getPackage().getManifest());
  if (!execution)
    return execution.takeError();
  collection.executionPlan = std::move(*execution);
  collection.countPackage =
      instrumentation.findCapture(ProfileCaptureKind::Count);
  collection.tracePackage =
      instrumentation.findCapture(ProfileCaptureKind::Trace);
  if (!collection.countPackage || !collection.tracePackage)
    return invalid(
        "profile instrumentation has no complete capture package set");
  llvm::Expected<BoardInvocationFilePlan> count =
      makeCapturePlan(primaryPlan, primaryManifest, *collection.countPackage);
  if (!count)
    return count.takeError();
  collection.countPlan = std::move(*count);
  llvm::Expected<BoardInvocationFilePlan> trace =
      makeCapturePlan(primaryPlan, primaryManifest, *collection.tracePackage);
  if (!trace)
    return trace.takeError();
  collection.tracePlan = std::move(*trace);

  const uint64_t eventStorage = collection.tracePackage->getRecordBytes() -
                                WAFER_TX81_PROFILER_EVENTS_OFFSET -
                                WAFER_TX81_PROFILER_BUFFER_GUARD_BYTES;
  const uint64_t traceCapacity =
      eventStorage / WAFER_TX81_PROFILER_TSM_CALL_EVENT_BYTES;

  // Resolve the report generator and create its writable staging directory
  // before the first device/provider call.
  llvm::Expected<PendingProfileReportRun> reportRun =
      createPendingProfileReportRun(instrumentation.getRoot());
  if (!reportRun)
    return reportRun.takeError();
  auto cleanupStaging = llvm::make_scope_exit([&] {
    if (!reportRun->stagingDirectory.empty())
      (void)llvm::sys::fs::remove_directories(reportRun->stagingDirectory);
  });
  BoardProfileOutputValidator outputValidation(reportRun->stagingDirectory);

  std::optional<QualifiedBoardRuntimeSession> session;
  std::optional<BoardDeviceInfo> device;
  auto execute = [&](const ExecutablePackage &package,
                     const BoardInvocationFilePlan &plan, bool profilerExpected,
                     BoardCompletionObservationPolicy observationPolicy,
                     BoardDeviceTimingPolicy deviceTimingPolicy)
      -> llvm::Expected<BoardRuntimeInvocationResult> {
    BoardRuntimeInvocationRequest request = plan.request;
    request.completionObservationPolicy = observationPolicy;
    request.deviceTimingPolicy = deviceTimingPolicy;
    llvm::Expected<BoardRuntimeInvocationResult> result = [&]() {
      if (session)
        return executeBoardInvocationInSession(package, std::move(request),
                                               *session);
      llvm::Expected<
          std::pair<BoardRuntimeInvocationResult, QualifiedBoardRuntimeSession>>
          started = executeBoardInvocationAndStartSession(
              package, std::move(request), driver);
      if (!started)
        return llvm::Expected<BoardRuntimeInvocationResult>(
            started.takeError());
      session.emplace(std::move(started->second));
      return llvm::Expected<BoardRuntimeInvocationResult>(
          std::move(started->first));
    }();
    if (!result)
      return result.takeError();
    if (profilerExpected != !result->profilerOutputs.empty())
      return invalid("board invocation returned an unexpected profiler "
                     "readback domain");
    if (!device) {
      if (llvm::Error error = validateTopology(result->device))
        return std::move(error);
      device = result->device;
    } else if (!sameDevice(*device, result->device)) {
      return invalid("qualified board inventory changed within the profile "
                     "session");
    }
    return result;
  };

  std::optional<BoardRuntimeInvocationResult> finalResult;
  std::string profileRunDirectory;
  llvm::Expected<BoardProfileProtocolResult> protocol =
      runFixedBoardProfileProtocol(
          traceCapacity,
          [&](const BoardProfileProtocolStep &step)
              -> llvm::Expected<BoardProfileProtocolObservation> {
            BoardProfileProtocolObservation observation;
            switch (step.launch) {
            case BoardProfileProtocolLaunch::Primary: {
              llvm::Expected<BoardRuntimeInvocationResult> result = execute(
                  collection.profiledPackage->getPackage(),
                  collection.executionPlan,
                  /*profilerExpected=*/false,
                  BoardCompletionObservationPolicy::ProfileHighResolution,
                  BoardDeviceTimingPolicy::StreamEvents);
              if (!result)
                return result.takeError();
              if (llvm::Error error = outputValidation.recordPrimaryOutputs(
                      collection.profiledPackage->getPackage().getManifest(),
                      collection.executionPlan, result->outputs))
                return std::move(error);
              observation.deviceExecutionNanoseconds =
                  result->deviceExecutionNanoseconds;
              observation.hostSubmitNanoseconds = result->hostSubmitNanoseconds;
              observation.launchToCompletionNanoseconds =
                  result->launchToCompletionNanoseconds;
              observation.completionObservationResolutionNanoseconds =
                  result->completionObservationResolutionNanoseconds;
              finalResult = std::move(*result);
              return observation;
            }
            case BoardProfileProtocolLaunch::Count: {
              llvm::Expected<BoardRuntimeInvocationResult> result = execute(
                  collection.countPackage->getPackage(), collection.countPlan,
                  /*profilerExpected=*/true,
                  BoardCompletionObservationPolicy::Normal,
                  BoardDeviceTimingPolicy::Disabled);
              if (!result)
                return result.takeError();
              if (llvm::Error error =
                      outputValidation.validateDiagnosticOutputs(
                          collection.countPackage->getPackage().getManifest(),
                          collection.countPlan, result->outputs))
                return std::move(error);
              llvm::Expected<std::vector<Tx81ProfilerRecord>> records =
                  decodeProfilerOutputs(*collection.countPackage, *result);
              if (!records)
                return records.takeError();
              collection.count = std::move(*records);
              for (uint32_t tile = 0; tile < WAFER_TX81_PROFILER_TILE_COUNT;
                   ++tile)
                observation.countSequences[tile] =
                    collection.count[tile].header.next_sequence;
              return observation;
            }
            case BoardProfileProtocolLaunch::Trace: {
              llvm::Expected<BoardRuntimeInvocationResult> result = execute(
                  collection.tracePackage->getPackage(), collection.tracePlan,
                  /*profilerExpected=*/true,
                  BoardCompletionObservationPolicy::Normal,
                  BoardDeviceTimingPolicy::Disabled);
              if (!result)
                return result.takeError();
              if (llvm::Error error =
                      outputValidation.validateDiagnosticOutputs(
                          collection.tracePackage->getPackage().getManifest(),
                          collection.tracePlan, result->outputs))
                return std::move(error);
              llvm::Expected<std::vector<Tx81ProfilerRecord>> records =
                  decodeProfilerOutputs(*collection.tracePackage, *result);
              if (!records)
                return records.takeError();
              collection.trace = std::move(*records);
              if (llvm::Error error =
                      validateTraceSites(instrumentation, collection))
                return std::move(error);
              for (uint32_t tile = 0; tile < WAFER_TX81_PROFILER_TILE_COUNT;
                   ++tile) {
                const WaferTx81ProfilerRecordHeader &header =
                    collection.trace[tile].header;
                observation.trace[tile] = {
                    collection.count[tile].header.next_sequence,
                    header.next_sequence,
                    static_cast<uint64_t>(collection.trace[tile].events.size()),
                    header.dropped_event_count,
                    header.flags,
                    header.trace_state,
                };
              }
              return observation;
            }
            }
            llvm_unreachable("unknown board profile protocol launch");
          },
          [&](llvm::ArrayRef<BoardProfileMeasurementSample> samples)
              -> llvm::Error {
            if (!device)
              return invalid("profile collection produced no qualified device "
                             "inventory");
            if (llvm::Error error = outputValidation.removeReferenceFiles())
              return std::move(error);
            llvm::Expected<std::string> reportDirectory =
                writeProfileReport(*reportRun, instrumentation, *device,
                                   samples, collection, outputValidation);
            if (!reportDirectory)
              return reportDirectory.takeError();
            profileRunDirectory = std::move(*reportDirectory);
            return llvm::Error::success();
          });
  if (!protocol)
    return protocol.takeError();
  if (!finalResult)
    return invalid("profile collection produced no final output");
  cleanupStaging.release();

  return BoardProfileCollectionResult{
      std::move(*finalResult),
      std::move(collection.executionPlan),
      std::move(profileRunDirectory),
  };
}

} // namespace wafer::runtime::cli
