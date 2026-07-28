//===- WaferProfileCampaign.cpp - Automatic board profile run ----------===//

#include "WaferProfileCampaign.h"

#include "Wafer/ABI/Tx81ProfilerABI.h"
#include "Wafer/Runtime/ProfilerRecord.h"

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
    WAFER_TX81_PROFILER_RECORD_PUBLISHED;

llvm::Error invalid(llvm::Twine message) {
  return llvm::createStringError(llvm::errc::invalid_argument, "%s",
                                 message.str().c_str());
}

struct CandidateState {
  const ProfileVariantPackage *variant = nullptr;
  BoardInvocationFilePlan executionPlan;
  const ProfileCapturePackage *countPackage = nullptr;
  const ProfileCapturePackage *tracePackage = nullptr;
  BoardInvocationFilePlan countPlan;
  BoardInvocationFilePlan tracePlan;
  std::vector<Tx81ProfilerRecord> count;
  std::vector<Tx81ProfilerRecord> trace;
};

using SemanticOutputKey = std::tuple<int64_t, PackageResourceRole, int64_t>;

SemanticOutputKey semanticOutputKey(const PackageResourceRecord &resource) {
  return {resource.logicalRank, resource.role, resource.roleIndex};
}

struct ExactOutputContract {
  int64_t logicalRank = -1;
  PackageResourceRole role = PackageResourceRole::Output;
  int64_t roleIndex = -1;
  std::string dtype;
  std::vector<int64_t> shape;
  uint64_t bytes = 0;
  uint64_t alignment = 0;
  PackageAccessMode access = PackageAccessMode::WriteOnly;
  bool hostVisible = false;
};

ExactOutputContract exactOutputContract(const PackageResourceRecord &resource) {
  return {resource.logicalRank, resource.role,       resource.roleIndex,
          resource.type.dtype,  resource.type.shape, resource.bytes,
          resource.alignment,   resource.access,     resource.hostVisible};
}

bool sameExactOutputContract(const ExactOutputContract &lhs,
                             const ExactOutputContract &rhs) {
  return lhs.logicalRank == rhs.logicalRank && lhs.role == rhs.role &&
         lhs.roleIndex == rhs.roleIndex && lhs.dtype == rhs.dtype &&
         lhs.shape == rhs.shape && lhs.bytes == rhs.bytes &&
         lhs.alignment == rhs.alignment && lhs.access == rhs.access &&
         lhs.hostVisible == rhs.hostVisible;
}

struct IndexedOutput {
  const PackageResourceRecord *resource = nullptr;
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
    auto resource =
        llvm::find_if(manifest.resources, [&](const auto &candidate) {
          return candidate.id == output.resource;
        });
    if (resource == manifest.resources.end() || !resource->hostVisible ||
        resource->access == PackageAccessMode::ReadOnly)
      return invalid("profile output is not a host-visible writable resource");
    const uint64_t resourceId = output.resource.getValue();
    std::optional<BoardOutputComparisonKind> expectedComparison;
    if (plan.expectedBytes.contains(resourceId)) {
      auto comparison = plan.expectedComparisons.find(resourceId);
      if (comparison == plan.expectedComparisons.end())
        return invalid(
            "profile output external expected comparison is missing");
      expectedComparison = comparison->second;
    }
    if (!indexed
             .try_emplace(semanticOutputKey(*resource),
                          IndexedOutput{
                              &*resource,
                              &output,
                              expectedComparison,
                          })
             .second)
      return invalid("profile output has a duplicate semantic resource key");
  }
  return indexed;
}

std::string hashOutputBytes(llvm::ArrayRef<uint8_t> bytes) {
  llvm::SHA256 hasher;
  hasher.update(bytes);
  return "sha256:" + llvm::toHex(hasher.final(), /*LowerCase=*/true);
}

std::string formatSemanticOutputKey(const SemanticOutputKey &key) {
  auto [rank, role, roleIndex] = key;
  return (llvm::Twine("logical_rank=") + llvm::Twine(rank) +
          ",role=" + stringifyPackageResourceRole(role) +
          ",role_index=" + llvm::Twine(roleIndex))
      .str();
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
      return invalid("profile output differs from the production-artifact "
                     "reference for " +
                     formatSemanticOutputKey(key) + " at byte " +
                     llvm::Twine(mismatchOffset));
    }
    offset += wanted;
  }
  return llvm::Error::success();
}

bool isProfilerResource(const PackageResourceRecord &resource,
                        uint64_t recordBytes) {
  return resource.role == PackageResourceRole::Workspace &&
         resource.roleIndex == 1 && resource.type.dtype == "u8" &&
         resource.type.shape ==
             std::vector<int64_t>{static_cast<int64_t>(recordBytes)} &&
         resource.bytes == recordBytes &&
         resource.alignment == WAFER_TX81_PROFILER_BUFFER_ALIGNMENT &&
         resource.access == PackageAccessMode::ReadWrite &&
         !resource.hostVisible;
}

Tx81ProfilerCaptureKind toRuntimeCaptureKind(ProfileCaptureKind capture) {
  switch (capture) {
  case ProfileCaptureKind::Summary:
    return Tx81ProfilerCaptureKind::Summary;
  case ProfileCaptureKind::Count:
    return Tx81ProfilerCaptureKind::Count;
  case ProfileCaptureKind::Trace:
    return Tx81ProfilerCaptureKind::Trace;
  }
  llvm_unreachable("unknown profile capture kind");
}

llvm::Expected<BoardInvocationFilePlan>
makeCapturePlan(const BoardInvocationFilePlan &productionPlan,
                const PackageManifest &productionManifest,
                const ProfileCapturePackage &capture) {
  llvm::Expected<BoardInvocationFilePlan> plan = remapBoardInvocationFilePlan(
      productionPlan, productionManifest, capture.getPackage().getManifest());
  if (!plan)
    return plan.takeError();

  std::array<const PackageResourceRecord *, WAFER_TX81_PROFILER_TILE_COUNT>
      profilerResources{};
  for (const PackageResourceRecord &resource :
       capture.getPackage().getManifest().resources) {
    if (resource.role != PackageResourceRole::Workspace ||
        resource.roleIndex != 1)
      continue;
    if (!isProfilerResource(resource, capture.getRecordBytes()) ||
        resource.logicalRank < 0 ||
        resource.logicalRank >= WAFER_TX81_PROFILER_TILE_COUNT ||
        profilerResources[resource.logicalRank])
      return invalid("profile capture has an invalid typed profiler resource "
                     "domain");
    profilerResources[resource.logicalRank] = &resource;
  }
  for (uint32_t tile = 0; tile < WAFER_TX81_PROFILER_TILE_COUNT; ++tile) {
    const PackageResourceRecord *resource = profilerResources[tile];
    if (!resource)
      return invalid("profile capture omits a typed profiler resource");
    llvm::Expected<std::vector<uint8_t>> image = buildTx81ProfilerLaunchImage(
        capture.getRecordBytes(), tile,
        toRuntimeCaptureKind(capture.getCaptureKind()));
    if (!image)
      return image.takeError();
    plan->request.profilerBindings.push_back({resource->id, std::move(*image)});
  }
  return plan;
}

llvm::Expected<std::vector<Tx81ProfilerRecord>>
decodeProfilerOutputs(const ProfileCapturePackage &capture,
                      const BoardRuntimeInvocationResult &result) {
  const PackageManifest &manifest = capture.getPackage().getManifest();
  llvm::DenseMap<uint64_t, int64_t> resourceRanks;
  for (const PackageResourceRecord &resource : manifest.resources)
    if (isProfilerResource(resource, capture.getRecordBytes()))
      resourceRanks[resource.id.getValue()] = resource.logicalRank;
  if (resourceRanks.size() != WAFER_TX81_PROFILER_TILE_COUNT ||
      result.profilerOutputs.size() != WAFER_TX81_PROFILER_TILE_COUNT)
    return invalid("board profiler readback is not all-and-only 16 tiles");

  llvm::DenseSet<uint64_t> returned;
  std::vector<Tx81ProfilerRecord> records;
  records.reserve(WAFER_TX81_PROFILER_TILE_COUNT);
  for (const BoardRuntimeOutput &output : result.profilerOutputs) {
    auto rank = resourceRanks.find(output.resource.getValue());
    if (rank == resourceRanks.end() ||
        !returned.insert(output.resource.getValue()).second ||
        output.bytes.size() != capture.getRecordBytes())
      return invalid("board profiler readback has an unexpected resource "
                     "identity or byte count");
    llvm::Expected<Tx81ProfilerRecord> decoded =
        decodeTx81ProfilerRecord(output.bytes);
    if (!decoded)
      return decoded.takeError();
    if (decoded->header.tile_id != static_cast<uint32_t>(rank->second))
      return invalid("board profiler resource and record tile disagree");
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
    if ((capture.getCaptureKind() == ProfileCaptureKind::Summary &&
         (trace || count)) ||
        (capture.getCaptureKind() == ProfileCaptureKind::Count &&
         (!count || trace)) ||
        (capture.getCaptureKind() == ProfileCaptureKind::Trace &&
         (!trace || count)))
      return invalid("board profiler record capture kind disagrees with its "
                     "verified package");
  }
  return records;
}

const ProfileVariantSiteMap *
resolveSiteMap(const VerifiedProfileCompanion &companion,
               llvm::StringRef variantId) {
  return companion.findSiteMap(variantId);
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
  case WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_PEER_READY_WAIT:
  case WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_SETUP_ISSUE:
  case WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_COMPLETION_WAIT:
  case WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_CLEANUP:
    return site.siteKind == ProfileTargetSiteKind::DirectDTEWait &&
           site.engine &&
           engineMatches(*site.engine, event.engine);
  case WAFER_TX81_PROFILER_EVENT_TARGET_SITE:
    return event.engine == WAFER_TX81_PROFILER_ENGINE_NONE;
  }
  return false;
}

llvm::Error validateTraceSites(const VerifiedProfileCompanion &companion,
                               const CandidateState &candidate) {
  if (candidate.trace.size() != WAFER_TX81_PROFILER_TILE_COUNT)
    return invalid("profile trace tile domain is incomplete");
  const ProfileVariantSiteMap *siteMap =
      resolveSiteMap(companion, candidate.variant->getId());
  if (!siteMap || siteMap->ranks.size() != WAFER_TX81_PROFILER_TILE_COUNT)
    return invalid("profile trace has no complete typed site map");
  for (uint32_t tile = 0; tile < WAFER_TX81_PROFILER_TILE_COUNT; ++tile) {
    const Tx81ProfilerRecord &trace = candidate.trace[tile];
    const ProfileRankSiteMap &rankMap = siteMap->ranks[tile];
    if (trace.header.tile_id != tile || rankMap.logicalRank != tile)
      return invalid(
          "profile typed trace/site-map rank domain is not canonical");
    for (const WaferTx81ProfilerTSMCallEvent &event : trace.events) {
      if (!isTx81ProfilerSiteValid(event))
        return invalid("profile trace event has no final-artifact typed site");
      if (event.site_id >= rankMap.sites.size())
        return invalid("profile trace event references an unknown typed site");
      const ProfileTargetCallSite &site = rankMap.sites[event.site_id];
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
    return a.logicalIndex == b.logicalIndex && a.available == b.available &&
           a.physicalX == b.physicalX && a.physicalY == b.physicalY;
  });
}

llvm::Error validateTopology(const BoardDeviceInfo &device) {
  if (device.tileCount != WAFER_TX81_PROFILER_TILE_COUNT ||
      device.tiles.size() != WAFER_TX81_PROFILER_TILE_COUNT)
    return invalid("qualified profile device does not expose 16 tiles");
  std::array<bool, WAFER_TX81_PROFILER_TILE_COUNT> seen{};
  std::vector<std::pair<uint32_t, uint32_t>> coordinates;
  for (const BoardDeviceInfo::Tile &tile : device.tiles) {
    if (tile.logicalIndex >= WAFER_TX81_PROFILER_TILE_COUNT ||
        seen[tile.logicalIndex] || !tile.available)
      return invalid("qualified profile device tile domain is invalid");
    seen[tile.logicalIndex] = true;
    if (llvm::is_contained(coordinates,
                           std::pair(tile.physicalX, tile.physicalY)))
      return invalid("qualified profile device has duplicate physical tile "
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
  json.attribute("kind", stringifyRuntimeLaunchKind(launch.getKind()));
  if (const KernelRuntimeLaunchContract *kernel = launch.getKernel()) {
    json.attribute("form", stringifyKernelLaunchForm(kernel->form));
    json.attribute("entry_abi", stringifyKernelEntryABI(kernel->entryABI));
  } else {
    json.attribute("entry_abi",
                   stringifyModelEntryABI(launch.getModel()->entryABI));
  }
  json.attributeArray("phases", [&] {
    for (RuntimeLaunchPhaseRole phase : launch.getPhases())
      json.value(stringifyRuntimeLaunchPhaseRole(phase));
  });
}

void emitCandidateEvidence(llvm::json::OStream &json,
                           const CandidateState &candidate,
                           llvm::StringRef targetProfile,
                           const RuntimeLaunchContract &launch) {
  json.object([&] {
    json.attributeObject("artifact", [&] {
      json.attribute("digest", candidate.variant->getManifestDigest());
      json.attribute("target_profile", targetProfile);
      json.attributeObject("launch", [&] { emitRuntimeLaunch(json, launch); });
      json.attribute("execution_ranks",
                     int64_t(WAFER_TX81_PROFILER_TILE_COUNT));
    });
    json.attributeArray("clock", [&] {
      for (uint32_t tile = 0; tile < WAFER_TX81_PROFILER_TILE_COUNT; ++tile)
        json.object([&] {
          json.attribute("tile", int64_t(tile));
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
        for (const Tx81ProfilerRecord &record : candidate.trace)
          json.object([&] {
            json.attribute("tile", int64_t(record.header.tile_id));
            json.attribute("entry_begin_cycle",
                           record.header.entry_begin_cycle);
            json.attribute("entry_end_cycle", record.header.entry_end_cycle);
            json.attribute("capacity", int64_t(record.header.event_capacity));
            json.attribute("count", int64_t(record.header.event_count));
            json.attribute(
                "preflight_count",
                candidate.count[record.header.tile_id].header.next_sequence);
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
                      json.attribute("observation_status",
                                     "counter-no-change");
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
                  if (event.kind == WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_WAIT)
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
      });
    });
    json.attributeObject("pmu", [&] {
      json.attributeArray("tiles", [&] {
        for (const Tx81ProfilerRecord &record : candidate.trace) {
          const WaferTx81ProfilerRecordHeader &header = record.header;
          const bool enabled = pmuEnabled(header);
          json.object([&] {
            json.attribute("tile", int64_t(header.tile_id));
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
serializeEvidence(const VerifiedProfileCompanion &companion,
                  llvm::StringRef runId, const BoardDeviceInfo &device,
                  llvm::ArrayRef<BoardProfileMeasurementSample> samples,
                  const CandidateState &candidate,
                  const BoardProfileOutputValidationState &outputValidation) {
  if (samples.size() != 1)
    return invalid("profile evidence inputs are incomplete");
  const PackageManifest &productionManifest =
      candidate.variant->getPackage().getManifest();
  const std::string targetProfile =
      stringifyTargetProfileId(productionManifest.targetProfile).str();
  std::string storage;
  llvm::raw_string_ostream output(storage);
  llvm::json::OStream json(output, 2);
  json.object([&] {
    json.attribute("schema", "wafer.profile.evidence");
    json.attribute("schema_version", int64_t(7));
    json.attribute("run_id", runId);
    json.attributeObject("identity", [&] {
      json.attribute("production_manifest_sha256",
                     companion.getProductionManifestDigest());
      json.attribute("profile_companion_schema_version",
                     int64_t(companion.getSchemaVersion()));
      json.attribute("target_profile", targetProfile);
      json.attributeObject("launch", [&] {
        emitRuntimeLaunch(json, productionManifest.launch);
      });
      json.attribute("execution_ranks",
                     int64_t(WAFER_TX81_PROFILER_TILE_COUNT));
      json.attribute("site_correlation_basis", kProfileSiteCorrelationBasis);
      json.attribute("record_abi", kProfileRecordABI);
    });
    json.attributeObject("output_validation", [&] {
      json.attribute("mode", stringifyBoardProfileOutputValidationMode(
                                 outputValidation.getMode()));
      json.attributeArray("resources", [&] {
        for (const BoardProfileOutputValidationResource &resource :
             outputValidation.getResources())
          json.object([&] {
            json.attribute("logical_rank", resource.logicalRank);
            json.attribute("role", stringifyPackageResourceRole(resource.role));
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
            json.attribute("production_execution_validated",
                           resource.productionExecutionValidated);
            json.attribute("diagnostic_captures_match_primary",
                           resource.diagnosticCapturesMatchPrimary);
          });
      });
    });
    json.attributeArray("topology", [&] {
      for (uint32_t logical = 0; logical < WAFER_TX81_PROFILER_TILE_COUNT;
           ++logical) {
        auto tile = llvm::find_if(device.tiles, [&](const auto &candidate) {
          return candidate.logicalIndex == logical;
        });
        json.object([&] {
          json.attribute("tile", int64_t(logical));
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
      const ProfileVariantSiteMap *siteMap =
          resolveSiteMap(companion, candidate.variant->getId());
      for (const ProfileRankSiteMap &rank : siteMap->ranks)
        for (const ProfileTargetCallSite &site : rank.sites)
          json.object([&] {
            json.attribute("tile", rank.logicalRank);
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
      json.attribute("package_companion", true);
      json.attribute("measurement_basis", true);
    });
    json.attributeBegin("experiment");
    emitCandidateEvidence(json, candidate, targetProfile,
                          productionManifest.launch);
    json.attributeEnd();
  });
  output << "\n";
  return output.str();
}

llvm::Error ensureRunsDirectory(llvm::StringRef companionRoot,
                                llvm::SmallVectorImpl<char> &runs) {
  runs.assign(companionRoot.begin(), companionRoot.end());
  llvm::sys::path::append(runs, "runs");
  llvm::sys::fs::file_status status;
  std::error_code error = llvm::sys::fs::status(runs, status, /*follow=*/false);
  if (error == std::errc::no_such_file_or_directory) {
    if (error = llvm::sys::fs::create_directory(runs))
      return llvm::createStringError(error,
                                     "failed to create profile runs directory");
  } else if (error) {
    return llvm::createStringError(error,
                                   "failed to inspect profile runs directory");
  } else if (status.type() != llvm::sys::fs::file_type::directory_file) {
    return invalid("profile runs path is not a real directory");
  }
  llvm::SmallString<256> canonical;
  if (error = llvm::sys::fs::real_path(runs, canonical))
    return llvm::createStringError(error,
                                   "failed to resolve profile runs directory");
  llvm::SmallString<256> expected(companionRoot);
  llvm::sys::path::append(expected, "runs");
  if (canonical != expected)
    return invalid("profile runs directory escapes the verified companion");
  runs.assign(canonical.begin(), canonical.end());
  if (error = llvm::sys::fs::setPermissions(runs, llvm::sys::fs::all_all))
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
      return invalid("profile report contains an unexpected public artifact");
    ++publicMemberCount;
  }
  if (walkError)
    return llvm::createStringError(walkError,
                                   "failed to inspect profile report members");
  if (publicMemberCount != 3)
    return invalid("profile report public artifact domain is incomplete");
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
  if (error = llvm::sys::fs::real_path(current, resolved))
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
createStagingRun(llvm::StringRef companionRoot) {
  llvm::SmallString<256> runs;
  if (llvm::Error error = ensureRunsDirectory(companionRoot, runs))
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
  return invalid("failed to choose a unique profile run identity");
}

struct PreparedProfilePublication {
  std::string runId;
  std::string stagingDirectory;
  std::string finalDirectory;
  std::string runsDirectory;
  std::string currentEntry;
  std::optional<std::string> previousRunDirectory;
  std::string python;
  std::string reportScript;
};

llvm::Expected<PreparedProfilePublication>
prepareProfilePublication(llvm::StringRef companionRoot) {
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
  if (llvm::Error error = ensureRunsDirectory(companionRoot, runs))
    return std::move(error);
  llvm::Expected<std::optional<std::string>> previous =
      resolveCurrentProfileRun(runs);
  if (!previous)
    return previous.takeError();
  llvm::Expected<std::pair<std::string, std::string>> run =
      createStagingRun(companionRoot);
  if (!run)
    return run.takeError();
  llvm::SmallString<256> finalPath(companionRoot);
  llvm::sys::path::append(finalPath, "runs", run->first);
  llvm::SmallString<256> current(runs);
  llvm::sys::path::append(current, "current");
  return PreparedProfilePublication{
      run->first,       run->second,         finalPath.str().str(),
      runs.str().str(), current.str().str(), std::move(*previous),
      *python,          std::move(*report)};
}

using RemoveManagedProfileRun =
    llvm::function_ref<std::error_code(llvm::StringRef)>;

llvm::Expected<std::string>
activateProfilePublication(PreparedProfilePublication &publication,
                           RemoveManagedProfileRun removeManagedRun) {
  for (llvm::StringRef name :
       {llvm::StringRef("evidence.json"), llvm::StringRef("analysis.json"),
        llvm::StringRef("index.html")}) {
    llvm::SmallString<256> path(publication.stagingDirectory);
    llvm::sys::path::append(path, name);
    if (llvm::sys::fs::get_file_type(path, /*Follow=*/false) !=
        llvm::sys::fs::file_type::regular_file)
      return invalid("profile report generator omitted " + name);
    if (std::error_code error =
            llvm::sys::fs::setPermissions(path, llvm::sys::fs::all_all))
      return llvm::createStringError(
          error, "failed to set profile report artifact permissions");
  }
  if (llvm::Error reportError = validateManagedProfileRun(
          publication.stagingDirectory, publication.runId))
    return std::move(reportError);
  if (std::error_code error = llvm::sys::fs::setPermissions(
          publication.stagingDirectory, llvm::sys::fs::all_all))
    return llvm::createStringError(
        error, "failed to set profile report directory permissions");

  if (std::error_code error = llvm::sys::fs::rename(
          publication.stagingDirectory, publication.finalDirectory))
    return llvm::createStringError(error,
                                   "failed to publish profile run atomically");
  publication.stagingDirectory.clear();
  bool currentPublished = false;
  llvm::SmallString<256> temporaryCurrent(publication.runsDirectory);
  llvm::sys::path::append(temporaryCurrent, ".current-" + publication.runId);
  auto rollbackNewRun = llvm::make_scope_exit([&] {
    (void)llvm::sys::fs::remove(temporaryCurrent);
    if (!currentPublished)
      (void)llvm::sys::fs::remove_directories(publication.finalDirectory);
  });
  if (std::error_code error = llvm::sys::fs::create_link(
          publication.finalDirectory, temporaryCurrent))
    return llvm::createStringError(
        error, "failed to create atomic current profile report entry");

  llvm::Expected<std::optional<std::string>> observedPrevious =
      resolveCurrentProfileRun(publication.runsDirectory);
  if (!observedPrevious)
    return observedPrevious.takeError();
  if (*observedPrevious != publication.previousRunDirectory)
    return invalid("current profile report changed during this campaign");
  if (std::error_code error =
          llvm::sys::fs::rename(temporaryCurrent, publication.currentEntry))
    return llvm::createStringError(
        error, "failed to activate current profile report atomically");
  currentPublished = true;
  rollbackNewRun.release();

  if (publication.previousRunDirectory &&
      *publication.previousRunDirectory != publication.finalDirectory) {
    llvm::StringRef previousRunId =
        llvm::sys::path::filename(*publication.previousRunDirectory);
    if (llvm::Error identityError = validateManagedProfileRun(
            *publication.previousRunDirectory, previousRunId))
      return std::move(identityError);
    if (std::error_code error =
            removeManagedRun(*publication.previousRunDirectory))
      return llvm::createStringError(
          error,
          "profile report was activated, but the previous managed run could "
          "not be removed");
  }
  return publication.currentEntry;
}

llvm::Expected<std::string>
publishReport(PreparedProfilePublication &publication,
              const VerifiedProfileCompanion &companion,
              const BoardDeviceInfo &device,
              llvm::ArrayRef<BoardProfileMeasurementSample> samples,
              const CandidateState &candidate,
              const BoardProfileOutputValidationState &outputValidation) {
  llvm::Expected<std::string> evidence =
      serializeEvidence(companion, publication.runId, device, samples,
                        candidate, outputValidation);
  if (!evidence)
    return evidence.takeError();
  llvm::SmallString<256> evidencePath(publication.stagingDirectory);
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
  std::string outputArgument = publication.stagingDirectory;
  llvm::SmallVector<llvm::StringRef, 6> arguments = {
      publication.python, publication.reportScript, evidenceArgument,
      "--output-directory", outputArgument};
  std::array<std::optional<llvm::StringRef>, 3> redirects = {
      std::nullopt, llvm::StringRef("/dev/null"), std::nullopt};
  std::string executionError;
  bool executionFailed = false;
  int exitCode = llvm::sys::ExecuteAndWait(
      publication.python, arguments, std::nullopt, redirects,
      /*SecondsToWait=*/0,
      /*MemoryLimit=*/0, &executionError, &executionFailed);
  if (executionFailed || exitCode != 0)
    return invalid(
        "profile report generation failed" +
        (executionError.empty()
             ? llvm::Twine(" with exit code ") + llvm::Twine(exitCode)
             : llvm::Twine(": ") + executionError));

  return activateProfilePublication(publication,
                                    [](llvm::StringRef runDirectory) {
                                      return llvm::sys::fs::remove_directories(
                                          runDirectory, /*IgnoreErrors=*/false);
                                    });
}

} // namespace

#if defined(WAFER_PROFILE_CAMPAIGN_TESTING)
namespace testing {

llvm::Expected<ProfileReportPublicationResult> publishProfileReportForTesting(
    llvm::StringRef companionRoot,
    llvm::function_ref<llvm::Error(llvm::StringRef runId,
                                   llvm::StringRef stagingDirectory)>
        stage,
    llvm::function_ref<std::error_code(llvm::StringRef)> removeManagedRun) {
  llvm::SmallString<256> runs;
  if (llvm::Error error = ensureRunsDirectory(companionRoot, runs))
    return std::move(error);
  llvm::Expected<std::optional<std::string>> previous =
      resolveCurrentProfileRun(runs);
  if (!previous)
    return previous.takeError();
  llvm::Expected<std::pair<std::string, std::string>> run =
      createStagingRun(companionRoot);
  if (!run)
    return run.takeError();

  llvm::SmallString<256> finalPath(runs);
  llvm::sys::path::append(finalPath, run->first);
  llvm::SmallString<256> current(runs);
  llvm::sys::path::append(current, "current");
  PreparedProfilePublication publication{
      run->first,         run->second,         finalPath.str().str(),
      runs.str().str(),   current.str().str(), std::move(*previous),
      /*python=*/"",
      /*reportScript=*/""};
  auto cleanupStaging = llvm::make_scope_exit([&] {
    if (!publication.stagingDirectory.empty())
      (void)llvm::sys::fs::remove_directories(publication.stagingDirectory);
  });
  if (llvm::Error error =
          stage(publication.runId, publication.stagingDirectory))
    return std::move(error);
  llvm::Expected<std::string> activated =
      activateProfilePublication(publication, removeManagedRun);
  if (!activated)
    return activated.takeError();
  return ProfileReportPublicationResult{
      publication.runId, publication.finalDirectory, std::move(*activated)};
}

} // namespace testing
#endif

struct BoardProfileOutputValidationState::Impl {
  struct Reference {
    SemanticOutputKey key;
    ExactOutputContract contract;
    std::string path;
  };

  explicit Impl(std::string stagingDirectory)
      : stagingDirectory(std::move(stagingDirectory)) {}

  std::string stagingDirectory;
  BoardProfileOutputValidationMode mode =
      BoardProfileOutputValidationMode::SameSessionProduction;
  bool established = false;
  bool referencesReleased = false;
  uint32_t diagnosticCaptureCount = 0;
  std::vector<BoardProfileOutputValidationResource> resources;
  std::vector<Reference> references;
  std::map<SemanticOutputKey, size_t> referenceIndices;
};

llvm::StringRef stringifyBoardProfileOutputValidationMode(
    BoardProfileOutputValidationMode mode) {
  switch (mode) {
  case BoardProfileOutputValidationMode::ExternalExpected:
    return "external-expected";
  case BoardProfileOutputValidationMode::Mixed:
    return "mixed";
  case BoardProfileOutputValidationMode::SameSessionProduction:
    return "same-session-production";
  }
  llvm_unreachable("unknown board profile output validation mode");
}

BoardProfileOutputValidationState::BoardProfileOutputValidationState(
    std::string stagingDirectory)
    : impl(std::make_unique<Impl>(std::move(stagingDirectory))) {}

BoardProfileOutputValidationState::~BoardProfileOutputValidationState() {
  if (!impl)
    return;
  for (const Impl::Reference &reference : impl->references)
    if (!reference.path.empty())
      (void)llvm::sys::fs::remove(reference.path);
}

BoardProfileOutputValidationState::BoardProfileOutputValidationState(
    BoardProfileOutputValidationState &&) = default;

BoardProfileOutputValidationState &BoardProfileOutputValidationState::operator=(
    BoardProfileOutputValidationState &&other) {
  if (this == &other)
    return *this;
  if (impl)
    for (const Impl::Reference &reference : impl->references)
      if (!reference.path.empty())
        (void)llvm::sys::fs::remove(reference.path);
  impl = std::move(other.impl);
  return *this;
}

llvm::Error BoardProfileOutputValidationState::establishProductionReference(
    const PackageManifest &manifest, const BoardInvocationFilePlan &plan,
    llvm::ArrayRef<BoardRuntimeOutput> outputs) {
  if (impl->established)
    return invalid(
        "production-artifact output reference is already established");
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
    return invalid("profile campaign has no writable output resource");

  std::vector<BoardProfileOutputValidationResource> resources;
  std::vector<Impl::Reference> references;
  std::map<SemanticOutputKey, size_t> referenceIndices;
  resources.reserve(indexed->size());
  references.reserve(indexed->size());
  size_t externalExpectedCount = 0;
  auto cleanup = llvm::make_scope_exit([&] {
    for (const Impl::Reference &reference : references)
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
    referenceIndices.emplace(key, index);
    references.push_back({key, exactOutputContract(*output.resource),
                          referencePath.str().str()});
    resources.push_back({output.resource->logicalRank, output.resource->role,
                         output.resource->roleIndex, output.resource->bytes,
                         hashOutputBytes(output.output->bytes),
                         output.externalExpectedComparison, true, false});
    if (output.externalExpectedComparison)
      ++externalExpectedCount;
  }

  if (externalExpectedCount == resources.size())
    impl->mode = BoardProfileOutputValidationMode::ExternalExpected;
  else if (externalExpectedCount == 0)
    impl->mode = BoardProfileOutputValidationMode::SameSessionProduction;
  else
    impl->mode = BoardProfileOutputValidationMode::Mixed;
  impl->resources = std::move(resources);
  impl->references = std::move(references);
  impl->referenceIndices = std::move(referenceIndices);
  impl->established = true;
  cleanup.release();
  return llvm::Error::success();
}

llvm::Error BoardProfileOutputValidationState::validateAgainstReference(
    const PackageManifest &manifest, const BoardInvocationFilePlan &plan,
    llvm::ArrayRef<BoardRuntimeOutput> outputs) {
  if (!impl->established)
    return invalid("production-artifact output reference is not established");
  if (impl->referencesReleased)
    return invalid("profile output references were already removed");
  if (impl->diagnosticCaptureCount >= 2)
    return invalid("count and trace output equivalence is already validated");

  // External expected tensors are authoritative and are always checked before
  // the same-session comparison.
  llvm::Expected<std::map<SemanticOutputKey, IndexedOutput>> indexed =
      indexValidatedOutputs(manifest, plan, outputs);
  if (!indexed)
    return indexed.takeError();
  if (indexed->size() != impl->references.size())
    return invalid("profile output semantic resource domain changed");

  for (const auto &[key, output] : *indexed) {
    auto referenceIndex = impl->referenceIndices.find(key);
    if (referenceIndex == impl->referenceIndices.end())
      return invalid("profile output semantic resource domain changed");
    const size_t index = referenceIndex->second;
    const Impl::Reference &reference = impl->references[index];
    if (!sameExactOutputContract(reference.contract,
                                 exactOutputContract(*output.resource)))
      return invalid("profile output typed resource contract changed for " +
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

llvm::Error BoardProfileOutputValidationState::validateDiagnosticCapture(
    const PackageManifest &manifest, const BoardInvocationFilePlan &plan,
    llvm::ArrayRef<BoardRuntimeOutput> outputs) {
  return validateAgainstReference(manifest, plan, outputs);
}

llvm::Error BoardProfileOutputValidationState::finalizeAndRemoveReferences() {
  if (!impl->established)
    return invalid("production-artifact output reference is not established");
  if (impl->diagnosticCaptureCount != 2)
    return invalid("count and trace output equivalence with the primary was "
                   "not validated");
  if (impl->referencesReleased)
    return invalid("profile output references were already removed");
  for (Impl::Reference &reference : impl->references) {
    if (std::error_code error = llvm::sys::fs::remove(reference.path))
      return llvm::createStringError(
          error, "failed to remove staged profile output reference");
    reference.path.clear();
  }
  impl->referencesReleased = true;
  return llvm::Error::success();
}

BoardProfileOutputValidationMode
BoardProfileOutputValidationState::getMode() const {
  return impl->mode;
}

llvm::ArrayRef<BoardProfileOutputValidationResource>
BoardProfileOutputValidationState::getResources() const {
  return impl->resources;
}

llvm::Expected<BoardProfileProtocolResult> runFixedBoardProfileProtocol(
    uint64_t traceCapacity,
    llvm::function_ref<llvm::Expected<BoardProfileProtocolObservation>(
        const BoardProfileProtocolStep &)>
        execute,
    llvm::function_ref<
        llvm::Error(llvm::ArrayRef<BoardProfileMeasurementSample>)>
        finalize) {
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
  if (primary->hostSubmitNanoseconds >
      primary->launchToCompletionNanoseconds)
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
    if (audit.preflightCount != count->countSequences[tile] ||
        audit.preflightCount != audit.nextSequence ||
        audit.nextSequence != audit.storedEventCount ||
        audit.droppedEventCount != 0 ||
        audit.recordFlags != kExpectedTraceFlags ||
        audit.traceState != WAFER_TX81_PROFILER_TRACE_COMPLETE)
      return invalid("profile trace audit differs from count preflight");
  }

  if (result.samples.size() != 1 || result.primarySampleId.empty())
    return invalid("primary profile measurement is incomplete");
  if (llvm::Error error = finalize(result.samples))
    return std::move(error);
  return result;
}

llvm::Expected<BoardProfileCampaignResult>
runBoardProfileCampaign(const VerifiedProfileCompanion &companion,
                        const PackageManifest &productionManifest,
                        const BoardInvocationFilePlan &productionPlan,
                        BoardRuntimeDriver &driver) {
  if (productionManifest.rankCount != WAFER_TX81_PROFILER_TILE_COUNT)
    return invalid("profile campaign requires the complete 16-rank package");

  CandidateState candidate;
  candidate.variant = companion.findVariant(ProfileVariantRole::FinalArtifact);
  if (!candidate.variant)
    return invalid("profile companion has no production artifact");

  llvm::Expected<BoardInvocationFilePlan> execution =
      remapBoardInvocationFilePlan(
          productionPlan, productionManifest,
          candidate.variant->getPackage().getManifest());
  if (!execution)
    return execution.takeError();
  candidate.executionPlan = std::move(*execution);
  candidate.countPackage = companion.findCapture(candidate.variant->getId(),
                                                 ProfileCaptureKind::Count);
  candidate.tracePackage = companion.findCapture(candidate.variant->getId(),
                                                 ProfileCaptureKind::Trace);
  if (!candidate.countPackage || !candidate.tracePackage)
    return invalid("profile companion has no complete capture package set");
  llvm::Expected<BoardInvocationFilePlan> count = makeCapturePlan(
      productionPlan, productionManifest, *candidate.countPackage);
  if (!count)
    return count.takeError();
  candidate.countPlan = std::move(*count);
  llvm::Expected<BoardInvocationFilePlan> trace = makeCapturePlan(
      productionPlan, productionManifest, *candidate.tracePackage);
  if (!trace)
    return trace.takeError();
  candidate.tracePlan = std::move(*trace);

  const uint64_t eventStorage = candidate.tracePackage->getRecordBytes() -
                                WAFER_TX81_PROFILER_EVENTS_OFFSET -
                                WAFER_TX81_PROFILER_BUFFER_GUARD_BYTES;
  const uint64_t traceCapacity =
      eventStorage / WAFER_TX81_PROFILER_TSM_CALL_EVENT_BYTES;

  // All host publication dependencies and a writable atomic staging
  // directory are qualified before the first device/provider call.
  llvm::Expected<PreparedProfilePublication> publication =
      prepareProfilePublication(companion.getRoot());
  if (!publication)
    return publication.takeError();
  auto cleanupPublication = llvm::make_scope_exit([&] {
    if (!publication->stagingDirectory.empty())
      (void)llvm::sys::fs::remove_directories(publication->stagingDirectory);
  });
  BoardProfileOutputValidationState outputValidation(
      publication->stagingDirectory);

  std::optional<QualifiedBoardRuntimeSession> session;
  std::optional<BoardDeviceInfo> device;
  auto execute = [&](const VerifiedPackageManifest &package,
                     llvm::StringRef packageRoot,
                     const BoardInvocationFilePlan &plan, bool profilerExpected,
                     BoardCompletionObservationPolicy observationPolicy,
                     BoardDeviceTimingPolicy deviceTimingPolicy)
      -> llvm::Expected<BoardRuntimeInvocationResult> {
    BoardRuntimeInvocationRequest request = plan.request;
    request.completionObservationPolicy = observationPolicy;
    request.deviceTimingPolicy = deviceTimingPolicy;
    llvm::Expected<BoardRuntimeInvocationResult> result = [&]() {
      if (session)
        return executeBoardInvocationInSession(package, packageRoot,
                                               std::move(request), *session);
      llvm::Expected<
          std::pair<BoardRuntimeInvocationResult, QualifiedBoardRuntimeSession>>
          started = executeBoardInvocationAndStartSession(
              package, packageRoot, std::move(request), driver);
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
                  candidate.variant->getPackage(),
                  candidate.variant->getPackageDirectory(),
                  candidate.executionPlan,
                  /*profilerExpected=*/false,
                  BoardCompletionObservationPolicy::ProfileHighResolution,
                  BoardDeviceTimingPolicy::StreamEvents);
              if (!result)
                return result.takeError();
              if (llvm::Error error =
                      outputValidation.establishProductionReference(
                          candidate.variant->getPackage().getManifest(),
                          candidate.executionPlan, result->outputs))
                return std::move(error);
              observation.deviceExecutionNanoseconds =
                  result->deviceExecutionNanoseconds;
              observation.hostSubmitNanoseconds =
                  result->hostSubmitNanoseconds;
              observation.launchToCompletionNanoseconds =
                  result->launchToCompletionNanoseconds;
              observation.completionObservationResolutionNanoseconds =
                  result->completionObservationResolutionNanoseconds;
              finalResult = std::move(*result);
              return observation;
            }
            case BoardProfileProtocolLaunch::Count: {
              llvm::Expected<BoardRuntimeInvocationResult> result =
                  execute(candidate.countPackage->getPackage(),
                          candidate.countPackage->getPackageDirectory(),
                          candidate.countPlan,
                          /*profilerExpected=*/true,
                          BoardCompletionObservationPolicy::Normal,
                          BoardDeviceTimingPolicy::Disabled);
              if (!result)
                return result.takeError();
              if (llvm::Error error =
                      outputValidation.validateDiagnosticCapture(
                          candidate.countPackage->getPackage().getManifest(),
                          candidate.countPlan, result->outputs))
                return std::move(error);
              llvm::Expected<std::vector<Tx81ProfilerRecord>> records =
                  decodeProfilerOutputs(*candidate.countPackage, *result);
              if (!records)
                return records.takeError();
              candidate.count = std::move(*records);
              for (uint32_t tile = 0; tile < WAFER_TX81_PROFILER_TILE_COUNT;
                   ++tile)
                observation.countSequences[tile] =
                    candidate.count[tile].header.next_sequence;
              return observation;
            }
            case BoardProfileProtocolLaunch::Trace: {
              llvm::Expected<BoardRuntimeInvocationResult> result =
                  execute(candidate.tracePackage->getPackage(),
                          candidate.tracePackage->getPackageDirectory(),
                          candidate.tracePlan,
                          /*profilerExpected=*/true,
                          BoardCompletionObservationPolicy::Normal,
                          BoardDeviceTimingPolicy::Disabled);
              if (!result)
                return result.takeError();
              if (llvm::Error error =
                      outputValidation.validateDiagnosticCapture(
                          candidate.tracePackage->getPackage().getManifest(),
                          candidate.tracePlan, result->outputs))
                return std::move(error);
              llvm::Expected<std::vector<Tx81ProfilerRecord>> records =
                  decodeProfilerOutputs(*candidate.tracePackage, *result);
              if (!records)
                return records.takeError();
              candidate.trace = std::move(*records);
              if (llvm::Error error = validateTraceSites(companion, candidate))
                return std::move(error);
              for (uint32_t tile = 0; tile < WAFER_TX81_PROFILER_TILE_COUNT;
                   ++tile) {
                const WaferTx81ProfilerRecordHeader &header =
                    candidate.trace[tile].header;
                observation.trace[tile] = {
                    candidate.count[tile].header.next_sequence,
                    header.next_sequence,
                    static_cast<uint64_t>(candidate.trace[tile].events.size()),
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
              return invalid("profile campaign produced no qualified device "
                             "inventory");
            if (llvm::Error error =
                    outputValidation.finalizeAndRemoveReferences())
              return std::move(error);
            llvm::Expected<std::string> published =
                publishReport(*publication, companion, *device, samples,
                              candidate, outputValidation);
            if (!published)
              return published.takeError();
            profileRunDirectory = std::move(*published);
            return llvm::Error::success();
          });
  if (!protocol)
    return protocol.takeError();
  if (!finalResult)
    return invalid("profile campaign produced no final output");
  cleanupPublication.release();

  return BoardProfileCampaignResult{
      std::move(*finalResult),
      std::move(candidate.executionPlan),
      std::move(profileRunDirectory),
  };
}

} // namespace wafer::runtime::cli
