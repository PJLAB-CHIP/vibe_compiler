//===- StaticFixedSlotQualification.cpp - Fixed-slot attestation --------===//

#include "StaticFixedSlotQualification.h"

#include "AcceptedCallClosure.h"
#include "CompilationInternal.h"

#include "Wafer/Analysis/ScheduleCostAnalysis.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/IR/WaferInterfaces.h"
#include "Wafer/Runtime/PackageManifest.h"
#include "Wafer/Support/TargetPolicy.h"
#include "Wafer/Target/TargetProfile.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace wafer::compiler::detail {
namespace {

constexpr llvm::StringLiteral kAttestationSchema =
    "wafer-static-fixed-slot-qualification";
constexpr llvm::StringLiteral kActivationSchema =
    "wafer-static-fixed-slot-qualification-activation";
constexpr int64_t kSchemaVersion = 1;

struct SPMRootSummary {
  int64_t ordinal = 0;
  int64_t offset = 0;
  int64_t bytes = 0;
  int64_t end = 0;
};

struct RotationSummary {
  int64_t iterArg = 0;
  int64_t initialRoot = 0;
  int64_t nextIterArg = 0;
  int64_t nextRoot = 0;
};

struct StaticLoopSummary {
  int64_t ordinal = 0;
  int64_t lower = 0;
  int64_t upper = 0;
  int64_t step = 0;
  int64_t tripCount = 0;
  int64_t iterArgCount = 0;
  std::vector<RotationSummary> spmRotations;
};

struct EngineWorkerIssueSummary {
  std::string engine;
  int64_t worker = 0;
  int64_t count = 0;
};

struct DTEIssueSummary {
  int64_t ordinal = 0;
  std::string kind;
  int64_t peer = 0;
  int64_t bytes = 0;
  int64_t token = 0;
};

struct DTEWaitSummary {
  int64_t ordinal = 0;
  std::vector<int64_t> tokens;
};

struct ParticipantJoinSummary {
  int64_t ordinal = 0;
  std::vector<int64_t> participants;
  bool insideStaticLoop = false;
};

struct RankSummary {
  int64_t logicalRank = 0;
  std::string acceptedInstrDigest;
  std::vector<SPMRootSummary> spmRoots;
  std::vector<StaticLoopSummary> staticLoops;
  std::vector<EngineWorkerIssueSummary> engineWorkerIssues;
  std::vector<DTEIssueSummary> dteIssues;
  std::vector<DTEWaitSummary> dteWaits;
  std::vector<ParticipantJoinSummary> participantJoins;
  std::array<int64_t, 3> completionCounts = {};
  uint64_t directDTEComputeOverlapWindowCount = 0;
};

struct StaticFixedSlotIRSummary {
  std::vector<SPMRootSummary> spmRoots;
  std::vector<StaticLoopSummary> staticLoops;
};

static llvm::Error invalid(llvm::Twine message) {
  return llvm::createStringError(llvm::errc::invalid_argument, "%s",
                                 message.str().c_str());
}

static std::string digestBytes(llvm::StringRef bytes) {
  llvm::SHA256 hasher;
  hasher.update(bytes);
  return "sha256:" + llvm::toHex(hasher.final(), /*LowerCase=*/true);
}

static llvm::Expected<std::string> digestFile(llvm::StringRef path) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path, /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false);
  if (!buffer)
    return llvm::createStringError(buffer.getError(),
                                   "failed to read file for digest");
  return digestBytes((*buffer)->getBuffer());
}

static llvm::Expected<std::string>
digestPackageManifest(llvm::StringRef packageRoot) {
  llvm::SmallString<256> path(packageRoot);
  llvm::sys::path::append(path, runtime::kPackageManifestFileName);
  return digestFile(path);
}

static mlir::LogicalResult writeAndVerifyJSON(llvm::StringRef path,
                                              llvm::StringRef expected,
                                              llvm::raw_ostream &diagnostics) {
  std::error_code error;
  llvm::raw_fd_ostream output(path, error, llvm::sys::fs::OF_Text);
  if (error) {
    reject(diagnostics, "failed to create fixed-slot qualification metadata: " +
                            error.message());
    return mlir::failure();
  }
  output << expected;
  output.close();
  if (output.has_error()) {
    reject(diagnostics, "failed to write fixed-slot qualification metadata");
    return mlir::failure();
  }

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> readback =
      llvm::MemoryBuffer::getFile(path, /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false);
  if (!readback) {
    reject(diagnostics,
           "failed to read back fixed-slot qualification metadata: " +
               readback.getError().message());
    return mlir::failure();
  }
  if ((*readback)->getBuffer() != expected) {
    reject(diagnostics,
           "fixed-slot qualification metadata byte readback mismatch");
    return mlir::failure();
  }
  llvm::Expected<llvm::json::Value> parsed =
      llvm::json::parse((*readback)->getBuffer());
  if (!parsed || !parsed->getAsObject()) {
    if (!parsed)
      llvm::consumeError(parsed.takeError());
    reject(diagnostics,
           "fixed-slot qualification metadata JSON readback is invalid");
    return mlir::failure();
  }
  return mlir::success();
}

static llvm::Expected<int64_t> resolveDirectSPMRootOrdinal(
    mlir::Value value,
    const llvm::DenseMap<mlir::Operation *, int64_t> &rootOrdinals) {
  llvm::DenseSet<mlir::Value> visited;
  while (value && visited.insert(value).second) {
    if (auto allocation = value.getDefiningOp<mlir::memref::AllocOp>()) {
      auto found = rootOrdinals.find(allocation.getOperation());
      if (found == rootOrdinals.end())
        return invalid("SPM loop seed does not resolve to an accepted root");
      return found->second;
    }
    auto cast = value.getDefiningOp<mlir::memref::CastOp>();
    if (!cast || cast.getSource().getType() != value.getType())
      return invalid(
          "SPM loop seed is not a same-address accepted allocation value");
    value = cast.getSource();
  }
  return invalid("SPM loop seed has a cyclic or unsupported root");
}

static std::optional<int64_t> resolveNextLoopIterArg(mlir::Value value,
                                                     mlir::scf::ForOp loop) {
  llvm::DenseSet<mlir::Value> visited;
  while (value && visited.insert(value).second) {
    if (auto cast = value.getDefiningOp<mlir::memref::CastOp>()) {
      if (cast.getSource().getType() != value.getType())
        return std::nullopt;
      value = cast.getSource();
      continue;
    }
    auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
    if (!argument || argument.getOwner() != loop.getBody() ||
        argument.getArgNumber() == 0)
      return std::nullopt;
    return static_cast<int64_t>(argument.getArgNumber() - 1);
  }
  return std::nullopt;
}

/// Resolve one real loop-body effect/issue operand to the current loop's
/// storage-carrying iter arg. Unlike recurrence edges, consumption may be
/// through a partial ViewLike alias: touching any proven subview is sufficient
/// evidence that the rotating slot participates in actual work.
static std::optional<int64_t>
resolveConsumedLoopIterArg(mlir::Value value, mlir::scf::ForOp loop) {
  llvm::DenseSet<mlir::Value> visited;
  while (value && visited.insert(value).second) {
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      if (argument.getOwner() != loop.getBody() || argument.getArgNumber() == 0)
        return std::nullopt;
      return static_cast<int64_t>(argument.getArgNumber() - 1);
    }

    auto result = mlir::dyn_cast<mlir::OpResult>(value);
    mlir::Operation *definition = result ? result.getOwner() : nullptr;
    if (!definition || definition->getParentOp() != loop.getOperation())
      return std::nullopt;
    mlir::Value source;
    if (auto cast = mlir::dyn_cast_or_null<mlir::memref::CastOp>(definition)) {
      source = cast.getSource();
    } else if (auto view = mlir::dyn_cast_or_null<mlir::ViewLikeOpInterface>(
                   definition)) {
      source = view.getViewSource();
    } else {
      return std::nullopt;
    }

    auto sourceType = source
                          ? mlir::dyn_cast<mlir::MemRefType>(source.getType())
                          : mlir::MemRefType();
    auto resultType = mlir::dyn_cast<mlir::MemRefType>(value.getType());
    if (!sourceType || !resultType ||
        sourceType.getElementType() != resultType.getElementType() ||
        sourceType.getMemorySpace() != resultType.getMemorySpace())
      return std::nullopt;
    value = source;
  }
  return std::nullopt;
}

static llvm::Expected<int64_t>
deriveStaticTripCount(int64_t lower, int64_t upper, int64_t step) {
  if (step <= 0)
    return invalid("static fixed-slot loop step must be positive");
  if (upper <= lower)
    return int64_t{0};
  const __int128 distance =
      static_cast<__int128>(upper) - static_cast<__int128>(lower);
  const __int128 trips = (distance + static_cast<__int128>(step) - 1) / step;
  if (trips > std::numeric_limits<int64_t>::max())
    return invalid("static fixed-slot loop trip count overflows int64");
  return static_cast<int64_t>(trips);
}

struct RootedIterArgEdge {
  int64_t iterArg = 0;
  int64_t initialRoot = 0;
  int64_t nextIterArg = 0;
  int64_t nextRoot = 0;
};

static llvm::Expected<std::vector<RotationSummary>> deriveSPMRootCycles(
    mlir::scf::ForOp loop,
    const llvm::DenseMap<mlir::Operation *, int64_t> &rootOrdinals) {
  const int64_t iterArgCount =
      static_cast<int64_t>(loop.getNumRegionIterArgs());
  std::vector<std::optional<int64_t>> rootByIterArg(
      static_cast<size_t>(iterArgCount));
  for (auto [index, tuple] : llvm::enumerate(
           llvm::zip(loop.getInitArgs(), loop.getRegionIterArgs()))) {
    mlir::Value init = std::get<0>(tuple);
    mlir::BlockArgument iterArg = std::get<1>(tuple);
    if (!isWaferSPMMemRefType(iterArg.getType()))
      continue;
    if (init.getType() != iterArg.getType())
      return invalid(
          "fixed-slot qualification SPM iter-arg type is not invariant");
    llvm::Expected<int64_t> root =
        resolveDirectSPMRootOrdinal(init, rootOrdinals);
    if (!root)
      return root.takeError();
    rootByIterArg[index] = *root;
  }

  auto yield =
      mlir::dyn_cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
  if (!yield || yield.getOperands().size() != loop.getNumRegionIterArgs())
    return invalid("fixed-slot qualification loop has no exact iter-arg yield");

  std::map<int64_t, int64_t> rootSuccessors;
  std::map<int64_t, RootedIterArgEdge> representatives;
  for (int64_t index = 0; index < iterArgCount; ++index) {
    if (!rootByIterArg[static_cast<size_t>(index)])
      continue;
    const int64_t root = *rootByIterArg[static_cast<size_t>(index)];
    std::optional<int64_t> next =
        resolveNextLoopIterArg(yield.getOperand(index), loop);
    int64_t nextIterArg = index;
    int64_t nextRoot = root;
    if (next && *next >= 0 && *next < iterArgCount &&
        rootByIterArg[static_cast<size_t>(*next)]) {
      nextIterArg = *next;
      nextRoot = *rootByIterArg[static_cast<size_t>(*next)];
    } else {
      // A fixed slot may yield its original accepted allocation (or a
      // same-address cast) instead of spelling an identity region-argument
      // recurrence. This is fully auditable but is not a rotating edge.
      llvm::Expected<int64_t> yieldedRoot =
          resolveDirectSPMRootOrdinal(yield.getOperand(index), rootOrdinals);
      if (!yieldedRoot) {
        llvm::consumeError(yieldedRoot.takeError());
        return invalid(
            "fixed-slot qualification cannot prove SPM iter-arg recurrence");
      }
      if (*yieldedRoot != root)
        return invalid(
            "fixed-slot qualification SPM iter-arg changes to a different "
            "fixed root without an exact recurrence");
    }
    auto [successor, inserted] = rootSuccessors.try_emplace(root, nextRoot);
    if (!inserted && successor->second != nextRoot)
      return invalid(
          "fixed-slot qualification aliases give one SPM root conflicting "
          "recurrence successors");
    representatives.try_emplace(
        root, RootedIterArgEdge{index, root, nextIterArg, nextRoot});
  }

  std::vector<RotationSummary> rotations;
  llvm::DenseSet<int64_t> processed;
  for (const auto &[start, unused] : rootSuccessors) {
    (void)unused;
    if (processed.contains(start))
      continue;
    llvm::SmallVector<int64_t, 8> path;
    llvm::DenseMap<int64_t, size_t> positions;
    int64_t current = start;
    while (true) {
      if (processed.contains(current))
        break;
      auto repeated = positions.find(current);
      if (repeated != positions.end()) {
        llvm::ArrayRef<int64_t> cycle(path);
        cycle = cycle.drop_front(repeated->second);
        if (cycle.size() >= 2) {
          for (int64_t root : cycle) {
            const RootedIterArgEdge &edge = representatives.at(root);
            rotations.push_back({edge.iterArg, edge.initialRoot,
                                 edge.nextIterArg, edge.nextRoot});
          }
        }
        break;
      }
      auto successor = rootSuccessors.find(current);
      if (successor == rootSuccessors.end())
        break;
      positions[current] = path.size();
      path.push_back(current);
      current = successor->second;
    }
    for (int64_t root : path)
      processed.insert(root);
  }
  llvm::sort(rotations,
             [](const RotationSummary &lhs, const RotationSummary &rhs) {
               return lhs.initialRoot < rhs.initialRoot;
             });
  return rotations;
}

static llvm::Expected<std::string> stringifyEngine(InstrFamily family) {
  switch (family) {
  case InstrFamily::CT:
    return std::string("ct");
  case InstrFamily::NE:
    return std::string("ne");
  case InstrFamily::RDMA:
    return std::string("rdma");
  case InstrFamily::WDMA:
    return std::string("wdma");
  case InstrFamily::TDMA:
    return std::string("tdma");
  case InstrFamily::DTE:
    return std::string("dte");
  }
  return invalid("unknown typed instruction engine");
}

static llvm::Expected<StaticFixedSlotIRSummary>
deriveStaticFixedSlotIRSummary(const AcceptedCallClosure &closure) {
  StaticFixedSlotIRSummary summary;
  llvm::DenseMap<mlir::Operation *, int64_t> rootOrdinals;
  for (mlir::func::FuncOp function : closure.functions) {
    llvm::Error error = llvm::Error::success();
    function.walk([&](mlir::memref::AllocOp allocation) {
      if (error || !isWaferSPMMemRefType(allocation.getType()))
        return;
      if (allocation->getParentOfType<mlir::scf::ForOp>()) {
        error = invalid(
            "fixed-slot qualification cannot attest a loop-local SPM root");
        return;
      }
      auto acceptedOffset =
          allocation->getAttrOfType<SPMOffsetAttr>(kWaferSPMOffsetAttrName);
      std::optional<WaferPhysicalTensorInfo> physical =
          computeWaferPhysicalTensorInfo(allocation.getType());
      if (!acceptedOffset || acceptedOffset.getOffset() < 0 || !physical ||
          physical->physicalBytes <= 0 ||
          !allocation.getDynamicSizes().empty()) {
        error = invalid(
            "fixed-slot qualification requires a planned positive static SPM "
            "root");
        return;
      }
      const __int128 end = static_cast<__int128>(acceptedOffset.getOffset()) +
                           static_cast<__int128>(physical->physicalBytes);
      if (end > std::numeric_limits<int64_t>::max()) {
        error =
            invalid("fixed-slot qualification SPM root range overflows int64");
        return;
      }
      const int64_t ordinal = static_cast<int64_t>(summary.spmRoots.size());
      rootOrdinals[allocation.getOperation()] = ordinal;
      summary.spmRoots.push_back({ordinal, acceptedOffset.getOffset(),
                                  physical->physicalBytes,
                                  static_cast<int64_t>(end)});
    });
    if (error)
      return std::move(error);
  }
  if (summary.spmRoots.size() < 2)
    return invalid(
        "fixed-slot qualification requires at least two accepted SPM roots");
  const TargetMemoryPolicy memory = getDefaultWaferTargetPolicy().memory;
  llvm::SmallVector<const SPMRootSummary *, 8> orderedRoots;
  orderedRoots.reserve(summary.spmRoots.size());
  for (const SPMRootSummary &root : summary.spmRoots) {
    if (memory.spmAlignment <= 0 || root.offset < memory.spmBase ||
        root.end > memory.spmLimit || root.offset % memory.spmAlignment != 0)
      return invalid(
          "fixed-slot qualification SPM root violates the target arena or "
          "placement alignment");
    orderedRoots.push_back(&root);
  }
  llvm::sort(orderedRoots,
             [](const SPMRootSummary *lhs, const SPMRootSummary *rhs) {
               return std::tie(lhs->offset, lhs->end, lhs->ordinal) <
                      std::tie(rhs->offset, rhs->end, rhs->ordinal);
             });
  for (auto [previous, current] :
       llvm::zip(orderedRoots, llvm::drop_begin(orderedRoots)))
    if (previous->end > current->offset)
      return invalid(
          "fixed-slot qualification requires nonoverlapping SPM roots");

  bool sawSPMRootCycle = false;
  for (mlir::func::FuncOp function : closure.functions) {
    llvm::Error error = llvm::Error::success();
    function.walk([&](mlir::scf::ForOp loop) {
      if (error)
        return;
      std::optional<int64_t> lower =
          mlir::getConstantIntValue(loop.getLowerBound());
      std::optional<int64_t> upper =
          mlir::getConstantIntValue(loop.getUpperBound());
      std::optional<int64_t> step = mlir::getConstantIntValue(loop.getStep());
      if (!lower || !upper || !step) {
        error =
            invalid("fixed-slot qualification requires static SCF loop bounds");
        return;
      }
      llvm::Expected<int64_t> tripCount =
          deriveStaticTripCount(*lower, *upper, *step);
      if (!tripCount || *tripCount <= 0) {
        error = tripCount
                    ? invalid("fixed-slot qualification loop is never executed")
                    : tripCount.takeError();
        return;
      }

      StaticLoopSummary loopSummary;
      loopSummary.ordinal = static_cast<int64_t>(summary.staticLoops.size());
      loopSummary.lower = *lower;
      loopSummary.upper = *upper;
      loopSummary.step = *step;
      loopSummary.tripCount = *tripCount;
      loopSummary.iterArgCount =
          static_cast<int64_t>(loop.getNumRegionIterArgs());

      llvm::Expected<std::vector<RotationSummary>> rotations =
          deriveSPMRootCycles(loop, rootOrdinals);
      if (!rotations) {
        error = rotations.takeError();
        return;
      }
      loopSummary.spmRotations = std::move(*rotations);
      sawSPMRootCycle |= !loopSummary.spmRotations.empty();
      summary.staticLoops.push_back(std::move(loopSummary));
    });
    if (error)
      return std::move(error);
  }
  if (summary.staticLoops.empty() || !sawSPMRootCycle)
    return invalid(
        "fixed-slot qualification requires a statically proven rotating "
        "SPM loop");
  return summary;
}

static llvm::SmallVector<int64_t, 8>
getConsumedSPMRootCycle(llvm::ArrayRef<RotationSummary> rotations,
                        const llvm::DenseSet<int64_t> &consumedIterArgs) {
  llvm::DenseMap<int64_t, int64_t> successors;
  llvm::DenseMap<int64_t, int64_t> iterArgByRoot;
  for (const RotationSummary &rotation : rotations)
    if (!successors.try_emplace(rotation.initialRoot, rotation.nextRoot)
             .second ||
        !iterArgByRoot.try_emplace(rotation.initialRoot, rotation.iterArg)
             .second)
      return {};

  llvm::DenseSet<int64_t> completed;
  for (const RotationSummary &rotation : rotations) {
    const int64_t start = rotation.initialRoot;
    if (completed.contains(start))
      continue;
    llvm::SmallVector<int64_t, 8> path;
    llvm::DenseMap<int64_t, size_t> positions;
    int64_t current = start;
    while (true) {
      if (completed.contains(current))
        break;
      auto repeated = positions.find(current);
      if (repeated != positions.end()) {
        llvm::ArrayRef<int64_t> cycle(path);
        cycle = cycle.drop_front(repeated->second);
        if (cycle.size() >= 2 && llvm::any_of(cycle, [&](int64_t root) {
              return consumedIterArgs.contains(iterArgByRoot.lookup(root));
            }))
          return llvm::SmallVector<int64_t, 8>(cycle);
        break;
      }
      auto successor = successors.find(current);
      if (successor == successors.end())
        break;
      positions[current] = path.size();
      path.push_back(current);
      current = successor->second;
    }
    for (int64_t root : path)
      completed.insert(root);
  }
  return {};
}

static std::vector<RotationSummary> deriveWitnessSPMRootCycles(
    mlir::scf::ForOp loop,
    const llvm::DenseMap<mlir::Operation *, int64_t> &rootOrdinals,
    const llvm::DenseSet<int64_t> &consumedIterArgs) {
  const int64_t iterArgCount =
      static_cast<int64_t>(loop.getNumRegionIterArgs());
  std::vector<std::optional<int64_t>> rootByIterArg(
      static_cast<size_t>(iterArgCount));
  for (auto [index, tuple] : llvm::enumerate(
           llvm::zip(loop.getInitArgs(), loop.getRegionIterArgs()))) {
    mlir::Value init = std::get<0>(tuple);
    mlir::BlockArgument iterArg = std::get<1>(tuple);
    if (!isWaferSPMMemRefType(iterArg.getType()) ||
        init.getType() != iterArg.getType())
      continue;
    llvm::Expected<int64_t> root =
        resolveDirectSPMRootOrdinal(init, rootOrdinals);
    if (!root) {
      llvm::consumeError(root.takeError());
      continue;
    }
    rootByIterArg[index] = *root;
  }

  auto yield =
      mlir::dyn_cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
  if (!yield || yield.getOperands().size() != loop.getNumRegionIterArgs())
    return {};

  llvm::DenseMap<int64_t, int64_t> successors;
  llvm::DenseMap<int64_t, RootedIterArgEdge> representatives;
  llvm::DenseSet<int64_t> conflictedRoots;
  for (int64_t index = 0; index < iterArgCount; ++index) {
    if (!rootByIterArg[static_cast<size_t>(index)])
      continue;
    std::optional<int64_t> next =
        resolveNextLoopIterArg(yield.getOperand(index), loop);
    if (!next || *next < 0 || *next >= iterArgCount ||
        !rootByIterArg[static_cast<size_t>(*next)])
      continue;

    const int64_t root = *rootByIterArg[static_cast<size_t>(index)];
    const int64_t nextRoot = *rootByIterArg[static_cast<size_t>(*next)];
    auto [successor, inserted] = successors.try_emplace(root, nextRoot);
    if (!inserted && successor->second != nextRoot) {
      conflictedRoots.insert(root);
      continue;
    }
    auto [representative, representativeInserted] = representatives.try_emplace(
        root, RootedIterArgEdge{index, root, *next, nextRoot});
    if (!representativeInserted && consumedIterArgs.contains(index) &&
        !consumedIterArgs.contains(representative->second.iterArg))
      representative->second = {index, root, *next, nextRoot};
  }
  for (int64_t root : conflictedRoots) {
    successors.erase(root);
    representatives.erase(root);
  }

  std::vector<RotationSummary> rotations;
  llvm::DenseSet<int64_t> processed;
  for (const auto &[start, unused] : successors) {
    (void)unused;
    if (processed.contains(start))
      continue;
    llvm::SmallVector<int64_t, 8> path;
    llvm::DenseMap<int64_t, size_t> positions;
    int64_t current = start;
    while (true) {
      if (processed.contains(current))
        break;
      auto repeated = positions.find(current);
      if (repeated != positions.end()) {
        llvm::ArrayRef<int64_t> cycle(path);
        cycle = cycle.drop_front(repeated->second);
        if (cycle.size() >= 2)
          for (int64_t root : cycle) {
            auto representative = representatives.find(root);
            if (representative == representatives.end()) {
              rotations.clear();
              break;
            }
            const RootedIterArgEdge &edge = representative->second;
            rotations.push_back({edge.iterArg, edge.initialRoot,
                                 edge.nextIterArg, edge.nextRoot});
          }
        break;
      }
      auto successor = successors.find(current);
      if (successor == successors.end())
        break;
      positions[current] = path.size();
      path.push_back(current);
      current = successor->second;
    }
    for (int64_t root : path)
      processed.insert(root);
  }
  llvm::sort(rotations,
             [](const RotationSummary &lhs, const RotationSummary &rhs) {
               return lhs.initialRoot < rhs.initialRoot;
             });
  return rotations;
}

static bool isActualLoopEffectOrIssueOperand(mlir::Operation *operation,
                                             mlir::Value operand) {
  if (mlir::isa<WaferInstructionOpInterface>(operation))
    return true;
  auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
  if (!effects)
    return false;
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 4> instances;
  effects.getEffects(instances);
  return llvm::any_of(instances, [&](const auto &instance) {
    return instance.getValue() == operand;
  });
}

/// Selection needs one current-IR witness, not a closed inventory of every
/// unrelated allocation and loop in the accepted call closure. A witness is a
/// positive static loop whose planned SPM iter args form a multi-root cycle
/// with at least one cycle-carried iter arg consumed by a real effect/issue in
/// that loop. Rotation then exercises the other roots dynamically. The
/// publication companion deliberately derives the stronger complete summary
/// separately.
static llvm::Error
verifyStaticFixedSlotIRWitness(const AcceptedCallClosure &closure) {
  llvm::DenseMap<mlir::Operation *, int64_t> rootOrdinals;
  llvm::DenseMap<int64_t, mlir::memref::AllocOp> rootsByOrdinal;
  int64_t nextRootOrdinal = 0;
  for (mlir::func::FuncOp function : closure.functions)
    function.walk([&](mlir::memref::AllocOp allocation) {
      if (!isWaferSPMMemRefType(allocation.getType()) ||
          allocation->getParentOfType<mlir::scf::ForOp>())
        return;
      auto offset =
          allocation->getAttrOfType<SPMOffsetAttr>(kWaferSPMOffsetAttrName);
      std::optional<WaferPhysicalTensorInfo> physical =
          computeWaferPhysicalTensorInfo(allocation.getType());
      if (!offset || offset.getOffset() < 0 || !physical ||
          physical->physicalBytes <= 0 || !allocation.getDynamicSizes().empty())
        return;
      const __int128 end = static_cast<__int128>(offset.getOffset()) +
                           static_cast<__int128>(physical->physicalBytes);
      const TargetMemoryPolicy memory = getDefaultWaferTargetPolicy().memory;
      if (end > std::numeric_limits<int64_t>::max() ||
          memory.spmAlignment <= 0 || offset.getOffset() < memory.spmBase ||
          end > memory.spmLimit ||
          offset.getOffset() % memory.spmAlignment != 0)
        return;
      rootOrdinals[allocation.getOperation()] = nextRootOrdinal;
      rootsByOrdinal[nextRootOrdinal] = allocation;
      ++nextRootOrdinal;
    });

  bool sawUnconsumedRotation = false;
  for (mlir::func::FuncOp function : closure.functions) {
    bool foundWitness = false;
    function.walk([&](mlir::scf::ForOp loop) {
      if (foundWitness)
        return mlir::WalkResult::interrupt();
      std::optional<int64_t> lower =
          mlir::getConstantIntValue(loop.getLowerBound());
      std::optional<int64_t> upper =
          mlir::getConstantIntValue(loop.getUpperBound());
      std::optional<int64_t> step = mlir::getConstantIntValue(loop.getStep());
      if (!lower || !upper || !step)
        return mlir::WalkResult::advance();
      llvm::Expected<int64_t> tripCount =
          deriveStaticTripCount(*lower, *upper, *step);
      if (!tripCount) {
        llvm::consumeError(tripCount.takeError());
        return mlir::WalkResult::advance();
      }
      if (*tripCount <= 0)
        return mlir::WalkResult::advance();

      llvm::DenseSet<int64_t> consumedIterArgs;
      loop.walk([&](mlir::Operation *operation) {
        if (operation == loop.getOperation() ||
            operation->getParentOp() != loop.getOperation() ||
            mlir::isa<mlir::scf::YieldOp>(operation))
          return;
        for (mlir::Value operand : operation->getOperands()) {
          if (!isActualLoopEffectOrIssueOperand(operation, operand))
            continue;
          std::optional<int64_t> iterArg =
              resolveConsumedLoopIterArg(operand, loop);
          if (iterArg && *iterArg >= 0)
            consumedIterArgs.insert(*iterArg);
        }
      });

      std::vector<RotationSummary> rotations =
          deriveWitnessSPMRootCycles(loop, rootOrdinals, consumedIterArgs);
      if (rotations.empty())
        return mlir::WalkResult::advance();

      llvm::SmallVector<int64_t, 8> witnessedCycle =
          getConsumedSPMRootCycle(rotations, consumedIterArgs);
      if (witnessedCycle.empty() ||
          static_cast<uint64_t>(*tripCount) < witnessedCycle.size()) {
        sawUnconsumedRotation = true;
        return mlir::WalkResult::advance();
      }

      llvm::SmallVector<const SPMRootSummary *, 8> orderedRoots;
      std::vector<SPMRootSummary> witnessedRoots;
      for (int64_t rootOrdinal : witnessedCycle) {
        auto root = rootsByOrdinal.find(rootOrdinal);
        if (root == rootsByOrdinal.end())
          return mlir::WalkResult::advance();
        mlir::memref::AllocOp allocation = root->second;
        auto offset =
            allocation->getAttrOfType<SPMOffsetAttr>(kWaferSPMOffsetAttrName);
        std::optional<WaferPhysicalTensorInfo> physical =
            computeWaferPhysicalTensorInfo(allocation.getType());
        if (!offset || !physical)
          return mlir::WalkResult::advance();
        witnessedRoots.push_back(
            {rootOrdinal, offset.getOffset(), physical->physicalBytes,
             offset.getOffset() + physical->physicalBytes});
      }
      for (const SPMRootSummary &root : witnessedRoots)
        orderedRoots.push_back(&root);
      llvm::sort(orderedRoots,
                 [](const SPMRootSummary *lhs, const SPMRootSummary *rhs) {
                   return std::tie(lhs->offset, lhs->end, lhs->ordinal) <
                          std::tie(rhs->offset, rhs->end, rhs->ordinal);
                 });
      if (llvm::any_of(llvm::zip(orderedRoots, llvm::drop_begin(orderedRoots)),
                       [](const auto &pair) {
                         return std::get<0>(pair)->end >
                                std::get<1>(pair)->offset;
                       }))
        return mlir::WalkResult::advance();
      foundWitness = true;
      return mlir::WalkResult::interrupt();
    });
    if (foundWitness)
      return llvm::Error::success();
  }
  if (sawUnconsumedRotation)
    return invalid(
        "fixed-slot qualification rotating SPM cycle has no complete "
        "loop-local effect/issue consumption");
  return invalid(
      "fixed-slot qualification requires one positive static rotating SPM "
      "loop with current effect/issue consumption");
}

static llvm::Expected<RankSummary>
deriveRankSummary(const RankExecutable &rank, TargetProfileId targetProfile) {
  RankSummary summary;
  summary.logicalRank = rank.getLogicalRank();

  llvm::Expected<AcceptedCallClosure> closure =
      analyzeAcceptedCallClosure(rank.getModule(), rank.getEntrySymbol());
  if (!closure)
    return llvm::joinErrors(
        invalid("fixed-slot qualification accepted call closure is invalid"),
        closure.takeError());

  std::string moduleText;
  llvm::raw_string_ostream moduleStream(moduleText);
  rank.getModule().print(moduleStream);
  moduleStream << "\n";
  moduleStream.flush();
  summary.acceptedInstrDigest = digestBytes(moduleText);

  llvm::Expected<StaticFixedSlotIRSummary> fixedSlot =
      deriveStaticFixedSlotIRSummary(*closure);
  if (!fixedSlot)
    return fixedSlot.takeError();
  summary.spmRoots = std::move(fixedSlot->spmRoots);
  summary.staticLoops = std::move(fixedSlot->staticLoops);

  std::map<std::pair<int64_t, int64_t>, int64_t> issueCounts;
  llvm::DenseMap<mlir::Value, int64_t> tokenOrdinals;
  std::vector<bool> waitedTokens;
  for (mlir::func::FuncOp function : closure->functions) {
    llvm::Error error = llvm::Error::success();
    function.walk([&](mlir::Operation *operation) {
      if (error)
        return mlir::WalkResult::interrupt();

      if (mlir::isa<SyncLocalFenceOp>(operation)) {
        error = invalid(
            "fixed-slot qualification requires typed participant joins");
        return mlir::WalkResult::interrupt();
      }

      NCCCompletionContract completion = getNCCCompletionContract(operation);
      switch (completion.behavior) {
      case LocalInstructionCompletion::None:
        break;
      case LocalInstructionCompletion::OrderedPending:
        ++summary.completionCounts[0];
        break;
      case LocalInstructionCompletion::ParticipantJoin:
        ++summary.completionCounts[1];
        break;
      case LocalInstructionCompletion::SynchronousWriteback:
        ++summary.completionCounts[2];
        break;
      }

      if (completion.issueWorker) {
        auto instruction =
            mlir::dyn_cast<WaferInstructionOpInterface>(operation);
        if (!instruction) {
          error =
              invalid("typed NCC issue has no instruction-engine interface");
          return mlir::WalkResult::interrupt();
        }
        llvm::Expected<std::string> engine =
            stringifyEngine(instruction.getInstructionFamily());
        if (!engine || *engine == "dte") {
          if (!engine)
            error = engine.takeError();
          else
            error = invalid("Direct DTE cannot enter the NCC issue inventory");
          return mlir::WalkResult::interrupt();
        }
        int64_t worker = static_cast<int64_t>(*completion.issueWorker);
        if (worker < 0 || worker >= static_cast<int64_t>(kNCCWorkerCount)) {
          error =
              invalid("typed NCC issue worker is outside the target domain");
          return mlir::WalkResult::interrupt();
        }
        ++issueCounts[{static_cast<int64_t>(instruction.getInstructionFamily()),
                       worker}];
      }

      if (auto join = mlir::dyn_cast<SyncNCCJoinOp>(operation)) {
        ParticipantJoinSummary joinSummary;
        joinSummary.ordinal =
            static_cast<int64_t>(summary.participantJoins.size());
        joinSummary.participants.assign(join.getParticipants().begin(),
                                        join.getParticipants().end());
        joinSummary.insideStaticLoop =
            static_cast<bool>(join->getParentOfType<mlir::scf::ForOp>());
        summary.participantJoins.push_back(std::move(joinSummary));
      } else if (completion.behavior ==
                 LocalInstructionCompletion::ParticipantJoin) {
        error = invalid(
            "fixed-slot qualification found an untyped participant join");
        return mlir::WalkResult::interrupt();
      }

      auto recordDTEIssue = [&](auto issue,
                                llvm::StringRef kind) -> mlir::LogicalResult {
        if (!issue.getBinding())
          return mlir::failure();
        const uint64_t peer = issue.getPeer();
        const uint64_t bytes = issue.getBytes();
        if (peer > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            bytes > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
          return mlir::failure();
        int64_t token = static_cast<int64_t>(waitedTokens.size());
        if (!tokenOrdinals.try_emplace(issue.getToken(), token).second)
          return mlir::failure();
        waitedTokens.push_back(false);
        summary.dteIssues.push_back(
            {static_cast<int64_t>(summary.dteIssues.size()), kind.str(),
             static_cast<int64_t>(peer), static_cast<int64_t>(bytes), token});
        return mlir::success();
      };
      if (auto send = mlir::dyn_cast<InstrDTESendOp>(operation)) {
        if (mlir::failed(recordDTEIssue(send, "send"))) {
          error = invalid("fixed-slot qualification found an unbound DTE send");
          return mlir::WalkResult::interrupt();
        }
      } else if (auto recv = mlir::dyn_cast<InstrDTERecvOp>(operation)) {
        if (mlir::failed(recordDTEIssue(recv, "recv"))) {
          error =
              invalid("fixed-slot qualification found an unbound DTE receive");
          return mlir::WalkResult::interrupt();
        }
      } else if (auto wait = mlir::dyn_cast<InstrDTEWaitOp>(operation)) {
        DTEWaitSummary waitSummary;
        waitSummary.ordinal = static_cast<int64_t>(summary.dteWaits.size());
        for (mlir::Value tokenValue : wait.getTokens()) {
          auto found = tokenOrdinals.find(tokenValue);
          if (found == tokenOrdinals.end() ||
              waitedTokens[static_cast<size_t>(found->second)]) {
            error =
                invalid("fixed-slot qualification DTE wait has an unknown or "
                        "duplicate token");
            return mlir::WalkResult::interrupt();
          }
          waitedTokens[static_cast<size_t>(found->second)] = true;
          waitSummary.tokens.push_back(found->second);
        }
        summary.dteWaits.push_back(std::move(waitSummary));
      }
      return mlir::WalkResult::advance();
    });
    if (error)
      return std::move(error);
  }
  if (!llvm::all_of(waitedTokens, [](bool waited) { return waited; }))
    return invalid(
        "fixed-slot qualification found a DTE token without an exact wait");
  if (summary.completionCounts[1] !=
      static_cast<int64_t>(summary.participantJoins.size()))
    return invalid(
        "fixed-slot qualification participant completion inventory diverged");

  for (const auto &[key, count] : issueCounts) {
    llvm::Expected<std::string> engine =
        stringifyEngine(static_cast<InstrFamily>(key.first));
    if (!engine)
      return engine.takeError();
    summary.engineWorkerIssues.push_back({*engine, key.second, count});
  }
  if (summary.engineWorkerIssues.empty())
    return invalid("fixed-slot qualification has no typed engine issues");

  analysis::InstructionProgramCost cost =
      analysis::analyzeInstructionProgramCost(
          rank.getModule(),
          analysis::getTargetScheduleCostPolicy(targetProfile));
  if (cost.directDTEComputeOverlapWindowCount.isKnown())
    summary.directDTEComputeOverlapWindowCount =
        cost.directDTEComputeOverlapWindowCount.value;
  return summary;
}

static void writeRankSummary(llvm::json::OStream &json,
                             const RankSummary &rank) {
  json.object([&] {
    json.attribute("logical_rank", rank.logicalRank);
    json.attribute("accepted_instr_sha256", rank.acceptedInstrDigest);
    json.attribute("direct_dte_compute_overlap_window_count",
                   rank.directDTEComputeOverlapWindowCount);
    json.attributeArray("spm_alloc_roots", [&] {
      for (const SPMRootSummary &root : rank.spmRoots)
        json.object([&] {
          json.attribute("ordinal", root.ordinal);
          json.attribute("offset", root.offset);
          json.attribute("bytes", root.bytes);
          json.attribute("range_begin", root.offset);
          json.attribute("range_end", root.end);
        });
    });
    json.attributeArray("static_loops", [&] {
      for (const StaticLoopSummary &loop : rank.staticLoops)
        json.object([&] {
          json.attribute("ordinal", loop.ordinal);
          json.attribute("lower", loop.lower);
          json.attribute("upper", loop.upper);
          json.attribute("step", loop.step);
          json.attribute("trip_count", loop.tripCount);
          json.attribute("iter_arg_count", loop.iterArgCount);
          json.attributeArray("spm_iter_arg_rotations", [&] {
            for (const RotationSummary &rotation : loop.spmRotations)
              json.object([&] {
                json.attribute("iter_arg", rotation.iterArg);
                json.attribute("initial_root", rotation.initialRoot);
                json.attribute("next_iter_arg", rotation.nextIterArg);
                json.attribute("next_root", rotation.nextRoot);
              });
          });
        });
    });
    json.attributeArray("engine_worker_issues", [&] {
      for (const EngineWorkerIssueSummary &issue : rank.engineWorkerIssues)
        json.object([&] {
          json.attribute("engine", issue.engine);
          json.attribute("worker", issue.worker);
          json.attribute("count", issue.count);
        });
    });
    json.attributeObject("dte", [&] {
      json.attribute("token_count",
                     static_cast<int64_t>(rank.dteIssues.size()));
      json.attributeArray("issues", [&] {
        for (const DTEIssueSummary &issue : rank.dteIssues)
          json.object([&] {
            json.attribute("ordinal", issue.ordinal);
            json.attribute("kind", issue.kind);
            json.attribute("peer", issue.peer);
            json.attribute("bytes", issue.bytes);
            json.attribute("token", issue.token);
          });
      });
      json.attributeArray("waits", [&] {
        for (const DTEWaitSummary &wait : rank.dteWaits)
          json.object([&] {
            json.attribute("ordinal", wait.ordinal);
            json.attributeArray("tokens", [&] {
              for (int64_t token : wait.tokens)
                json.value(token);
            });
          });
      });
    });
    json.attributeObject("completion", [&] {
      json.attributeArray("participant_joins", [&] {
        for (const ParticipantJoinSummary &join : rank.participantJoins)
          json.object([&] {
            json.attribute("ordinal", join.ordinal);
            json.attributeArray("participants", [&] {
              for (int64_t participant : join.participants)
                json.value(participant);
            });
            json.attribute("inside_static_loop", join.insideStaticLoop);
          });
      });
      json.attributeArray("behaviors", [&] {
        constexpr std::array<llvm::StringLiteral, 3> names = {
            "ordered-pending", "participant-join", "synchronous-writeback"};
        for (auto [index, name] : llvm::enumerate(names))
          json.object([&] {
            json.attribute("kind", name);
            json.attribute("count", rank.completionCounts[index]);
          });
      });
    });
  });
}

static std::string serializeAttestation(llvm::StringRef manifestDigest,
                                        const ExecutableBundle &bundle,
                                        llvm::ArrayRef<RankSummary> ranks,
                                        bool directDTEComputeOverlap) {
  std::string storage;
  llvm::raw_string_ostream stream(storage);
  {
    llvm::json::OStream json(stream, /*IndentSize=*/2);
    json.object([&] {
      json.attribute("schema", kAttestationSchema);
      json.attribute("schema_version", kSchemaVersion);
      json.attribute("selection_kind", directDTEComputeOverlap
                                           ? "direct-dte-compute-overlap"
                                           : "static-fixed-slot");
      json.attribute("manifest_sha256", manifestDigest);
      json.attribute("accepted_instr_digest_basis",
                     "final-accepted-instr-module-text-v1");
      json.attributeObject("target", [&] {
        json.attribute("profile",
                       stringifyTargetProfileId(
                           bundle.getExecutionConfig().getTargetProfileId()));
        json.attribute("rank_count",
                       bundle.getExecutionConfig().getRankCount());
        json.attributeArray("logical_ranks", [&] {
          for (const RankSummary &rank : ranks)
            json.value(rank.logicalRank);
        });
      });
      json.attributeArray("ranks", [&] {
        for (const RankSummary &rank : ranks)
          writeRankSummary(json, rank);
      });
    });
  }
  stream << "\n";
  stream.flush();
  return storage;
}

static std::string serializeActivation(llvm::StringRef manifestDigest,
                                       llvm::StringRef attestationDigest) {
  std::string storage;
  llvm::raw_string_ostream stream(storage);
  {
    llvm::json::OStream json(stream, /*IndentSize=*/2);
    json.object([&] {
      json.attribute("schema", kActivationSchema);
      json.attribute("schema_version", kSchemaVersion);
      json.attribute("manifest_sha256", manifestDigest);
      json.attribute("attestation_sha256", attestationDigest);
    });
  }
  stream << "\n";
  stream.flush();
  return storage;
}

} // namespace

llvm::Error
verifyStaticFixedSlotQualificationEvidence(const RankExecutable &rank) {
  llvm::Expected<AcceptedCallClosure> closure =
      analyzeAcceptedCallClosure(rank.getModule(), rank.getEntrySymbol());
  if (!closure)
    return llvm::joinErrors(
        invalid("fixed-slot qualification accepted call closure is invalid"),
        closure.takeError());
  return verifyStaticFixedSlotIRWitness(*closure);
}

bool hasStaticFixedSlotQualificationEvidence(const RankExecutable &rank) {
  if (llvm::Error error = verifyStaticFixedSlotQualificationEvidence(rank)) {
    llvm::consumeError(std::move(error));
    return false;
  }
  return true;
}

llvm::Error verifyStaticFixedSlotCompanionEvidence(const RankExecutable &rank) {
  llvm::Expected<RankSummary> summary =
      deriveRankSummary(rank, TargetProfileId::waferTx81SingleCardKernelV3());
  if (!summary)
    return summary.takeError();
  return llvm::Error::success();
}

mlir::LogicalResult stageStaticFixedSlotQualificationCompanion(
    llvm::StringRef companionRoot, llvm::StringRef packageRoot,
    const ExecutableBundle &bundle, bool requireDirectDTEComputeOverlap,
    llvm::raw_ostream &diagnostics) {
  const auto &rankExecutables = bundle.getRankExecutables();
  if (rankExecutables.size() !=
      static_cast<size_t>(bundle.getExecutionConfig().getRankCount())) {
    reject(diagnostics,
           "fixed-slot qualification rank domain differs from final bundle");
    return mlir::failure();
  }

  std::vector<RankSummary> ranks;
  ranks.reserve(rankExecutables.size());
  const TargetProfileId targetProfile =
      bundle.getExecutionConfig().getTargetProfileId();
  for (auto [expectedRank, rank] : llvm::enumerate(rankExecutables)) {
    if (rank.getLogicalRank() != static_cast<int64_t>(expectedRank)) {
      reject(diagnostics,
             "fixed-slot qualification rank domain is not canonical");
      return mlir::failure();
    }
    llvm::Expected<RankSummary> summary =
        deriveRankSummary(rank, targetProfile);
    if (!summary) {
      reject(diagnostics, llvm::toString(summary.takeError()));
      return mlir::failure();
    }
    if (requireDirectDTEComputeOverlap &&
        summary->directDTEComputeOverlapWindowCount == 0) {
      reject(diagnostics,
             "Direct-DTE compute-overlap qualification rank has no explicit "
             "issue/compute/exact-wait window");
      return mlir::failure();
    }
    ranks.push_back(std::move(*summary));
  }

  llvm::Expected<std::string> manifestDigest =
      digestPackageManifest(packageRoot);
  if (!manifestDigest) {
    reject(diagnostics, llvm::toString(manifestDigest.takeError()));
    return mlir::failure();
  }
  if (createDirectory(companionRoot, diagnostics))
    return mlir::failure();

  const std::string attestation = serializeAttestation(
      *manifestDigest, bundle, ranks, requireDirectDTEComputeOverlap);
  llvm::SmallString<256> attestationPath(companionRoot);
  llvm::sys::path::append(attestationPath, "attestation.json");
  if (mlir::failed(
          writeAndVerifyJSON(attestationPath, attestation, diagnostics)))
    return mlir::failure();
  llvm::Expected<std::string> attestationDigest = digestFile(attestationPath);
  if (!attestationDigest) {
    reject(diagnostics, llvm::toString(attestationDigest.takeError()));
    return mlir::failure();
  }

  // Activation is the final write: its presence authenticates the exact
  // read-back attestation bytes and the exact staged package manifest bytes.
  const std::string activation =
      serializeActivation(*manifestDigest, *attestationDigest);
  llvm::SmallString<256> activationPath(companionRoot);
  llvm::sys::path::append(activationPath, "activation.json");
  return writeAndVerifyJSON(activationPath, activation, diagnostics);
}

} // namespace wafer::compiler::detail
