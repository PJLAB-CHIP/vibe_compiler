//===- RedundantTransferElimination.cpp - Exact storage coalescing -------===//

#include "Scheduling/RedundantTransferElimination.h"

#include "MemoryPlanning/LifetimeAnalysis.h"
#include "Wafer/Analysis/ControlFlow/SingleExecutionRegionFlow.h"
#include "Wafer/Analysis/Tile/TransferRealizability.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/CompileTiming.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#include <limits>
#include <memory>
#include <optional>

namespace wafer::tensor_program_scheduling {
namespace {

namespace mp = wafer::memory_planning::detail;

struct AccessRecord {
  mlir::Operation *operation = nullptr;
  int64_t beginEvent = 0;
  int64_t endEvent = 0;
  mp::PathCondition path = mp::PathCondition::root();
  bool reads = false;
  bool writes = false;
  bool directDTE = false;
};

struct ForwardingRecord {
  mlir::Operation *operation = nullptr;
  mp::ProgramPoint point;
};

struct AliasSummary {
  llvm::DenseSet<mlir::Value> values;
  llvm::SmallVector<AccessRecord, 8> accesses;
  llvm::SmallVector<mlir::memref::DeallocOp, 2> deallocations;
  llvm::SmallVector<ForwardingRecord, 8> forwardings;
  bool escaped = false;
};

static bool isUnitDescriptor(llvm::ArrayRef<int64_t> strides,
                             llvm::ArrayRef<int64_t> iterations) {
  return strides.size() == 3 && iterations.size() == 3 &&
         llvm::all_of(strides, [](int64_t value) { return value == 0; }) &&
         llvm::all_of(iterations, [](int64_t value) { return value == 1; });
}

static std::optional<int64_t> getPhysicalBytes(mlir::MemRefType type) {
  auto encoding = mlir::dyn_cast_or_null<WaferPhysicalEncodingAttrInterface>(
      type.getMemorySpace());
  if (!encoding || !type.hasStaticShape())
    return std::nullopt;
  mlir::FailureOr<int64_t> bytes = encoding.getPhysicalFootprintBytes(type);
  if (mlir::failed(bytes) || *bytes <= 0)
    return std::nullopt;
  return *bytes;
}

static bool hasOnlyStaticZeroOffsets(mlir::OffsetSizeAndStrideOpInterface op) {
  return llvm::all_of(op.getMixedOffsets(), [](mlir::OpFoldResult offset) {
    return mlir::getConstantIntValue(offset) == 0;
  });
}

static mlir::Value getBasePreservingViewSource(mlir::Operation *definition) {
  if (auto cast = mlir::dyn_cast<mlir::memref::CastOp>(definition))
    return cast.getSource();
  if (auto collapse = mlir::dyn_cast<mlir::memref::CollapseShapeOp>(definition))
    return collapse.getSrc();
  if (auto expand = mlir::dyn_cast<mlir::memref::ExpandShapeOp>(definition))
    return expand.getOutputShape().empty() ? expand.getSrc() : mlir::Value{};
  if (auto subview = mlir::dyn_cast<mlir::memref::SubViewOp>(definition))
    return hasOnlyStaticZeroOffsets(subview) ? subview.getSource()
                                             : mlir::Value{};
  if (auto reinterpret =
          mlir::dyn_cast<mlir::memref::ReinterpretCastOp>(definition))
    return hasOnlyStaticZeroOffsets(reinterpret) ? reinterpret.getSource()
                                                 : mlir::Value{};
  if (auto view = mlir::dyn_cast<mlir::memref::ViewOp>(definition)) {
    std::optional<int64_t> byteShift =
        mlir::getConstantIntValue(view.getByteShift());
    return byteShift == 0 ? view.getSource() : mlir::Value{};
  }
  return {};
}

static bool isCompleteContiguousCopy(InstrGatherScatterOp gather,
                                     mlir::MemRefType sourceType,
                                     mlir::MemRefType destType) {
  std::optional<int64_t> sourceBytes = getPhysicalBytes(sourceType);
  std::optional<int64_t> destBytes = getPhysicalBytes(destType);
  return sourceBytes && destBytes && *sourceBytes == *destBytes &&
         (!gather.getSrcOffsetAttr() ||
          gather.getSrcOffsetAttr().getInt() == 0) &&
         (!gather.getDstOffsetAttr() ||
          gather.getDstOffsetAttr().getInt() == 0) &&
         !gather.getSrcOffsetValue() && !gather.getDstOffsetValue() &&
         gather.getByteCountAttr().getInt() == *sourceBytes &&
         gather.getInnerBytesAttr().getInt() == *sourceBytes &&
         isUnitDescriptor(gather.getSrcStrides(), gather.getSrcIterations()) &&
         isUnitDescriptor(gather.getDstStrides(), gather.getDstIterations());
}

static mlir::Value resolveStorageRoot(mlir::Value value,
                                      llvm::DenseSet<mlir::Value> &active) {
  if (!value || !active.insert(value).second)
    return {};
  auto finish = [&](mlir::Value result) {
    active.erase(value);
    return result;
  };

  if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    if (mlir::Value entry =
            analysis::getSingleExecutionRegionEntryOperand(argument))
      return finish(resolveStorageRoot(entry, active));
    return finish(value);
  }

  auto result = mlir::dyn_cast<mlir::OpResult>(value);
  if (!result)
    return finish(value);
  mlir::Operation *definition = result.getOwner();
  if (mlir::isa<mlir::memref::AllocOp>(definition))
    return finish(value);
  if (mlir::isa<mlir::ViewLikeOpInterface>(definition)) {
    mlir::Value source = getBasePreservingViewSource(definition);
    return finish(source ? resolveStorageRoot(source, active) : mlir::Value{});
  }
  if (auto toTensor =
          mlir::dyn_cast<mlir::bufferization::ToTensorOp>(definition))
    return finish(resolveStorageRoot(toTensor.getMemref(), active));
  if (auto toMemref =
          mlir::dyn_cast<mlir::bufferization::ToMemrefOp>(definition))
    return finish(resolveStorageRoot(toMemref.getTensor(), active));
  if (mlir::Value exit = analysis::getSingleExecutionRegionExitOperand(result))
    return finish(resolveStorageRoot(exit, active));
  return finish(value);
}

static mlir::Value resolveStorageRoot(mlir::Value value) {
  llvm::DenseSet<mlir::Value> active;
  return resolveStorageRoot(value, active);
}

static bool
appendAliasSuccessors(mlir::Value value, mlir::OpOperand &use,
                      llvm::SmallVectorImpl<mlir::Value> &worklist,
                      llvm::SmallVectorImpl<ForwardingRecord> &forwardings,
                      const mp::StructuredTimeline &timeline) {
  mlir::Operation *owner = use.getOwner();
  auto recordForwarding = [&]() {
    std::optional<mp::ProgramPoint> point = timeline.lookup(owner);
    if (point)
      forwardings.push_back({owner, *point});
    return static_cast<bool>(point);
  };

  if (auto view = mlir::dyn_cast<mlir::ViewLikeOpInterface>(owner)) {
    if (view.getViewSource() != value || owner->getNumResults() != 1 ||
        !recordForwarding())
      return false;
    worklist.push_back(owner->getResult(0));
    return true;
  }
  if (auto toTensor = mlir::dyn_cast<mlir::bufferization::ToTensorOp>(owner)) {
    if (toTensor.getMemref() != value || !recordForwarding())
      return false;
    worklist.push_back(toTensor.getResult());
    return true;
  }
  if (auto toMemref = mlir::dyn_cast<mlir::bufferization::ToMemrefOp>(owner)) {
    if (toMemref.getTensor() != value || !recordForwarding())
      return false;
    worklist.push_back(toMemref.getResult());
    return true;
  }
  if (auto region = mlir::dyn_cast<TileRegionOp>(owner)) {
    unsigned index = use.getOperandNumber();
    if (region.getBody().empty() || index >= region.getInputs().size() ||
        index >= region.getBody().front().getNumArguments() ||
        !recordForwarding())
      return false;
    worklist.push_back(region.getBody().front().getArgument(index));
    return true;
  }
  if (auto yield = mlir::dyn_cast<TileYieldOp>(owner)) {
    auto region = mlir::dyn_cast_or_null<TileRegionOp>(yield->getParentOp());
    unsigned index = use.getOperandNumber();
    if (!region || index >= region.getNumResults() || !recordForwarding())
      return false;
    worklist.push_back(region.getResult(index));
    return true;
  }
  return false;
}

static std::optional<mp::ProgramPoint>
getDirectDTECompletionPoint(mlir::Operation *operation,
                            mp::ProgramPoint issuePoint,
                            const mp::StructuredTimeline &timeline) {
  if (!mlir::isa<InstrDTESendOp, InstrDTERecvOp>(operation))
    return std::nullopt;
  if (operation->getNumResults() != 1 || !operation->getResult(0).hasOneUse())
    return std::nullopt;
  mlir::Operation *user =
      (*operation->getResult(0).getUses().begin()).getOwner();
  auto wait = mlir::dyn_cast<InstrDTEWaitOp>(user);
  if (!wait || wait->getBlock() != operation->getBlock() ||
      !operation->isBeforeInBlock(wait))
    return std::nullopt;
  std::optional<mp::ProgramPoint> point = timeline.lookup(wait);
  if (!point || point->path != issuePoint.path)
    return std::nullopt;
  return point;
}

static AliasSummary collectAliases(mlir::Value root,
                                   const mp::StructuredTimeline &timeline) {
  AliasSummary summary;
  llvm::SmallVector<mlir::Value, 8> worklist{root};
  llvm::DenseMap<mlir::Operation *, unsigned> accessIndices;

  while (!worklist.empty()) {
    mlir::Value value = worklist.pop_back_val();
    if (!summary.values.insert(value).second)
      continue;

    for (mlir::OpOperand &use : value.getUses()) {
      mlir::Operation *owner = use.getOwner();
      if (appendAliasSuccessors(value, use, worklist, summary.forwardings,
                                timeline))
        continue;

      std::optional<mp::ProgramPoint> point = timeline.lookup(owner);
      if (!point) {
        summary.escaped = true;
        continue;
      }

      if (auto deallocation = mlir::dyn_cast<mlir::memref::DeallocOp>(owner)) {
        summary.deallocations.push_back(deallocation);
        auto [it, inserted] =
            accessIndices.try_emplace(owner, summary.accesses.size());
        if (inserted)
          summary.accesses.push_back({owner, point->event, point->event,
                                      point->path,
                                      /*reads=*/false, /*writes=*/false,
                                      /*directDTE=*/false});
        continue;
      }

      auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(owner);
      if (!effects) {
        summary.escaped = true;
        continue;
      }
      llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 8> instances;
      effects.getEffects(instances);
      bool reads = false;
      bool writes = false;
      bool recognized = false;
      for (const mlir::MemoryEffects::EffectInstance &effect : instances) {
        if (effect.getValue() != value)
          continue;
        recognized = true;
        reads |= mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect());
        writes |= mlir::isa<mlir::MemoryEffects::Write>(effect.getEffect());
      }
      if (!recognized || (!reads && !writes)) {
        summary.escaped = true;
        continue;
      }

      bool directDTE = mlir::isa<InstrDTESendOp, InstrDTERecvOp>(owner);
      int64_t endEvent = point->event;
      if (directDTE) {
        std::optional<mp::ProgramPoint> completion =
            getDirectDTECompletionPoint(owner, *point, timeline);
        if (!completion) {
          summary.escaped = true;
          continue;
        }
        endEvent = completion->event;
      }

      auto [it, inserted] =
          accessIndices.try_emplace(owner, summary.accesses.size());
      if (inserted) {
        summary.accesses.push_back({owner, point->event, endEvent, point->path,
                                    reads, writes, directDTE});
      } else {
        AccessRecord &access = summary.accesses[it->second];
        access.reads |= reads;
        access.writes |= writes;
      }
    }
  }
  return summary;
}

static bool preservesDirectDTEIsolation(const AliasSummary &source,
                                        const AliasSummary &dest) {
  llvm::SmallVector<const AccessRecord *, 16> accesses;
  for (const AccessRecord &access : source.accesses)
    accesses.push_back(&access);
  for (const AccessRecord &access : dest.accesses)
    accesses.push_back(&access);

  for (const AccessRecord *interval : accesses) {
    if (!interval->directDTE)
      continue;
    for (const AccessRecord *access : accesses) {
      if (access == interval)
        continue;
      if (access->beginEvent > interval->beginEvent &&
          access->beginEvent < interval->endEvent)
        return false;
    }
  }
  return true;
}

static bool hasOnlyPath(const AliasSummary &summary,
                        mp::PathCondition requiredPath) {
  return llvm::all_of(summary.accesses,
                      [&](const AccessRecord &access) {
                        return access.path == requiredPath;
                      }) &&
         llvm::all_of(summary.forwardings,
                      [&](const ForwardingRecord &forwarding) {
                        return forwarding.point.path == requiredPath;
                      });
}

static bool hasStaticPositiveTripCount(mlir::scf::ForOp loop) {
  std::optional<int64_t> lower =
      mlir::getConstantIntValue(mlir::getAsOpFoldResult(loop.getLowerBound()));
  std::optional<int64_t> upper =
      mlir::getConstantIntValue(mlir::getAsOpFoldResult(loop.getUpperBound()));
  std::optional<int64_t> step =
      mlir::getConstantIntValue(mlir::getAsOpFoldResult(loop.getStep()));
  if (!lower || !upper || !step || *step <= 0 || *lower >= *upper)
    return false;
  __int128 distance =
      static_cast<__int128>(*upper) - static_cast<__int128>(*lower);
  __int128 tripCount = (distance + static_cast<__int128>(*step) - 1) /
                       static_cast<__int128>(*step);
  return tripCount > 0;
}

static bool hasUnsupportedEnclosingControlFlow(mlir::scf::ForOp loop) {
  for (mlir::Operation *parent = loop->getParentOp(); parent;
       parent = parent->getParentOp())
    if (mlir::isa<mlir::scf::IfOp, mlir::scf::ForOp>(parent))
      return true;
  return false;
}

static bool isInSelectedLoop(mlir::Operation *operation,
                             mlir::scf::ForOp loop) {
  return operation && operation->getParentOfType<mlir::scf::ForOp>() == loop;
}

static bool isSupportedLoopPoint(mlir::Operation *operation,
                                 mp::PathCondition path, mlir::scf::ForOp loop,
                                 mp::PathCondition loopPath) {
  return path == loopPath && isInSelectedLoop(operation, loop);
}

static bool hasSupportedLoopSourceSnapshot(const AliasSummary &source,
                                           InstrGatherScatterOp gather,
                                           mlir::scf::ForOp loop,
                                           mp::PathCondition loopPath,
                                           int64_t copyEvent) {
  for (const ForwardingRecord &forwarding : source.forwardings) {
    if (forwarding.point.path == mp::PathCondition::root())
      continue;
    if (!isSupportedLoopPoint(forwarding.operation, forwarding.point.path, loop,
                              loopPath))
      return false;
  }
  for (const AccessRecord &access : source.accesses) {
    if (access.operation == gather.getOperation())
      continue;
    if (access.path != mp::PathCondition::root() &&
        !isSupportedLoopPoint(access.operation, access.path, loop, loopPath))
      return false;
    if (access.writes &&
        (access.beginEvent > copyEvent || access.endEvent > copyEvent))
      return false;
  }
  return true;
}

static bool hasSupportedReadOnlyLoopDestination(const AliasSummary &dest,
                                                InstrGatherScatterOp gather,
                                                mlir::scf::ForOp loop,
                                                mp::PathCondition loopPath,
                                                int64_t copyEvent) {
  for (const ForwardingRecord &forwarding : dest.forwardings)
    if (!isSupportedLoopPoint(forwarding.operation, forwarding.point.path, loop,
                              loopPath) ||
        forwarding.point.event <= copyEvent)
      return false;

  for (const AccessRecord &access : dest.accesses) {
    if (access.operation == gather.getOperation())
      continue;
    if (!isSupportedLoopPoint(access.operation, access.path, loop, loopPath) ||
        access.beginEvent <= copyEvent || access.writes)
      return false;
  }
  return true;
}

static std::optional<int64_t>
getRequiredAlignment(mlir::MemRefType type,
                     mlir::memref::AllocOp allocation = {}) {
  llvm::SmallVector<int64_t, 1> requirements;
  if (allocation) {
    std::optional<uint64_t> explicitAlignment = allocation.getAlignment();
    if (explicitAlignment) {
      if (*explicitAlignment == 0 ||
          *explicitAlignment >
              static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
        return std::nullopt;
      requirements.push_back(static_cast<int64_t>(*explicitAlignment));
    }
  }
  mlir::FailureOr<int64_t> alignment =
      computeWaferRequiredAlignmentBytes(type, requirements);
  if (mlir::failed(alignment))
    return std::nullopt;
  return *alignment;
}

static bool hasStaticZeroOffsetNonNegativeStrides(mlir::MemRefType type) {
  llvm::SmallVector<int64_t, 4> strides;
  int64_t offset = 0;
  if (mlir::failed(mlir::getStridesAndOffset(type, strides, offset)) ||
      offset != 0 || strides.size() != static_cast<size_t>(type.getRank()))
    return false;
  return llvm::none_of(strides, [](int64_t stride) {
    return stride == mlir::ShapedType::kDynamic || stride < 0;
  });
}

static mlir::Value getStaticFullTransferSource(mlir::Value source) {
  while (auto cast = source.getDefiningOp<mlir::memref::CastOp>()) {
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(cast.getSource().getType());
    auto resultType = mlir::dyn_cast<mlir::MemRefType>(cast.getType());
    if (!sourceType || !resultType ||
        sourceType.getShape() != resultType.getShape() ||
        sourceType.getElementType() != resultType.getElementType() ||
        sourceType.getMemorySpace() != resultType.getMemorySpace() ||
        getPhysicalBytes(sourceType) != getPhysicalBytes(resultType))
      break;
    source = cast.getSource();
  }
  return source;
}

static mlir::FailureOr<mlir::Value>
createReplacementView(InstrGatherScatterOp gather, mlir::Value source,
                      mlir::MemRefType sourceType, mlir::MemRefType destType) {
  if (source.getType() != sourceType)
    return mlir::failure();
  mlir::MemRefType replacementType =
      mlir::MemRefType::get(destType.getShape(), destType.getElementType(),
                            destType.getLayout(), sourceType.getMemorySpace());
  if (sourceType != destType &&
      (!hasStaticZeroOffsetNonNegativeStrides(sourceType) ||
       !hasStaticZeroOffsetNonNegativeStrides(destType)))
    return mlir::failure();
  if (sourceType == replacementType)
    return source;

  llvm::SmallVector<int64_t, 4> strides;
  int64_t offset = 0;
  if (mlir::failed(mlir::getStridesAndOffset(destType, strides, offset)) ||
      offset != 0 || llvm::any_of(strides, [](int64_t stride) {
        return stride == mlir::ShapedType::kDynamic || stride < 0;
      }))
    return mlir::failure();
  llvm::SmallVector<int64_t, 4> sizes(destType.getShape().begin(),
                                      destType.getShape().end());
  mlir::OpBuilder builder(gather);
  return builder
      .create<mlir::memref::ReinterpretCastOp>(gather.getLoc(), replacementType,
                                               source, offset, sizes, strides)
      .getResult();
}

static void eraseCreatedReplacement(mlir::Value replacement,
                                    mlir::Value transferSource) {
  if (replacement == transferSource)
    return;
  if (mlir::Operation *definition = replacement.getDefiningOp())
    definition->erase();
}

static bool tryElide(
    InstrGatherScatterOp gather, const mp::StructuredTimeline &timeline,
    llvm::function_ref<void(mlir::Value, mlir::Value)> notifyReplacement) {
  wafer::support::ScopedCompileTimingSpan totalTiming("optimization-phase",
                                                      "tryElide", "total");
  auto phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "analysis-phase", "tryElide", "eligibility");
  auto transferSourceType =
      mlir::dyn_cast<mlir::MemRefType>(gather.getSource().getType());
  auto destType = mlir::dyn_cast<mlir::MemRefType>(gather.getDest().getType());
  if (!transferSourceType || !destType ||
      !isWaferSPMMemRefType(transferSourceType) ||
      !isWaferSPMMemRefType(destType) ||
      !isCompleteContiguousCopy(gather, transferSourceType, destType))
    return false;

  mlir::Value transferSource = getStaticFullTransferSource(gather.getSource());
  auto sourceType = mlir::dyn_cast<mlir::MemRefType>(transferSource.getType());
  if (!sourceType)
    return false;
  mlir::Value sourceRoot = resolveStorageRoot(transferSource);
  mlir::Value destRoot = resolveStorageRoot(gather.getDest());
  if (!sourceRoot || !destRoot)
    return false;

  auto destAllocation = destRoot.getDefiningOp<mlir::memref::AllocOp>();
  if (!destAllocation || gather.getDest() != destRoot ||
      destAllocation->hasAttr(kWaferSPMOffsetAttrName) ||
      destAllocation->hasAttr(kWaferDDROffsetAttrName))
    return false;
  auto sourceRootType = mlir::dyn_cast<mlir::MemRefType>(sourceRoot.getType());
  auto sourceAllocation = sourceRoot.getDefiningOp<mlir::memref::AllocOp>();
  if (!sourceRootType || !sourceAllocation ||
      sourceAllocation->hasAttr(kWaferSPMOffsetAttrName) ||
      sourceAllocation->hasAttr(kWaferDDROffsetAttrName))
    return false;

  std::optional<mp::ProgramPoint> copyPoint =
      timeline.lookup(gather.getOperation());
  if (!copyPoint)
    return false;
  int64_t copyEvent = copyPoint->event;

  mlir::scf::ForOp loop = gather->getParentOfType<mlir::scf::ForOp>();
  bool loopBodyCandidate = static_cast<bool>(loop);
  if (loopBodyCandidate) {
    if (gather->getParentOp() != loop.getOperation() ||
        !hasStaticPositiveTripCount(loop) ||
        hasUnsupportedEnclosingControlFlow(loop) ||
        !loop.isDefinedOutsideOfLoop(sourceRoot) ||
        !loop.isDefinedOutsideOfLoop(destRoot))
      return false;
    mlir::func::FuncOp function = gather->getParentOfType<mlir::func::FuncOp>();
    if (!function)
      return false;
    mlir::DominanceInfo dominance(function);
    if (!dominance.dominates(sourceRoot, gather.getOperation()) ||
        !dominance.dominates(destRoot, gather.getOperation()) ||
        !dominance.dominates(transferSource, gather.getOperation()))
      return false;
  } else if (copyPoint->path != mp::PathCondition::root()) {
    return false;
  }

  bool exactSelfCopy = transferSource == gather.getDest();
  phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "analysis-phase", "tryElide", "collectAliases");
  AliasSummary sourceAliases = collectAliases(sourceRoot, timeline);
  AliasSummary destAliases = collectAliases(destRoot, timeline);
  if (sourceAliases.escaped || destAliases.escaped ||
      (!exactSelfCopy && (sourceAliases.values.contains(destRoot) ||
                          destAliases.values.contains(sourceRoot))) ||
      !sourceAliases.deallocations.empty() ||
      !destAliases.deallocations.empty() ||
      !preservesDirectDTEIsolation(sourceAliases, destAliases))
    return false;

  bool destinationMayWrite = false;
  if (loopBodyCandidate) {
    if (!hasSupportedLoopSourceSnapshot(sourceAliases, gather, loop,
                                        copyPoint->path, copyEvent) ||
        !hasSupportedReadOnlyLoopDestination(destAliases, gather, loop,
                                             copyPoint->path, copyEvent))
      return false;
  } else {
    if (!hasOnlyPath(sourceAliases, mp::PathCondition::root()) ||
        !hasOnlyPath(destAliases, mp::PathCondition::root()))
      return false;
    for (const ForwardingRecord &forwarding : destAliases.forwardings)
      if (forwarding.point.event < copyEvent)
        return false;

    for (const AccessRecord &access : destAliases.accesses) {
      if (access.operation == gather.getOperation())
        continue;
      if (access.beginEvent < copyEvent)
        return false;
      destinationMayWrite |= access.writes;
    }

    for (const AccessRecord &access : sourceAliases.accesses) {
      if (access.operation == gather.getOperation())
        continue;
      if (access.writes &&
          (access.beginEvent > copyEvent || access.endEvent > copyEvent))
        return false;
      if (destinationMayWrite &&
          (access.beginEvent > copyEvent || access.endEvent > copyEvent))
        return false;
    }
  }

  phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "analysis-phase", "tryElide", "proveStaticReshapeMetadataView");
  if (mlir::failed(
          analysis::TransferRealizability::proveStaticReshapeMetadataView(
              sourceType, destType, destinationMayWrite)))
    return false;
  mlir::MemRefType replacementType =
      mlir::MemRefType::get(destType.getShape(), destType.getElementType(),
                            destType.getLayout(), sourceType.getMemorySpace());
  if (mlir::failed(
          analysis::TransferRealizability::proveStaticReshapeMetadataView(
              sourceType, replacementType, destinationMayWrite)))
    return false;
  if (replacementType != destType)
    for (mlir::OpOperand &use : destRoot.getUses())
      if (use.getOwner() != gather.getOperation() &&
          !mlir::isa<mlir::memref::DeallocOp>(use.getOwner()) &&
          mlir::isa<WaferInstructionOpInterface>(use.getOwner()))
        return false;

  std::optional<int64_t> sourceAlignment =
      getRequiredAlignment(sourceRootType, sourceAllocation);
  std::optional<int64_t> transferSourceAlignment =
      getRequiredAlignment(sourceType);
  std::optional<int64_t> replacementAlignment =
      getRequiredAlignment(replacementType);
  std::optional<int64_t> destAlignment =
      getRequiredAlignment(destType, destAllocation);
  if (!sourceAlignment || !transferSourceAlignment || !replacementAlignment ||
      !destAlignment)
    return false;
  std::optional<int64_t> requiredAlignment = mp::combineAlignmentRequirements(
      *transferSourceAlignment, *replacementAlignment);
  if (requiredAlignment)
    requiredAlignment =
        mp::combineAlignmentRequirements(*requiredAlignment, *destAlignment);
  if (!requiredAlignment)
    return false;
  std::optional<int64_t> raisedSourceAlignment;
  if (*sourceAlignment % *requiredAlignment != 0) {
    raisedSourceAlignment =
        mp::combineAlignmentRequirements(*sourceAlignment, *requiredAlignment);
    if (!raisedSourceAlignment)
      return false;
  }

  phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "transformation-phase", "tryElide", "createReplacementView");
  mlir::FailureOr<mlir::Value> replacement =
      createReplacementView(gather, transferSource, sourceType, destType);
  if (mlir::failed(replacement))
    return false;

  llvm::SmallVector<mlir::OpOperand *, 8> uses;
  for (mlir::OpOperand &use : destRoot.getUses()) {
    if (use.getOwner() == gather.getOperation() ||
        mlir::isa<mlir::memref::DeallocOp>(use.getOwner()))
      continue;
    uses.push_back(&use);
  }
  mlir::func::FuncOp function = gather->getParentOfType<mlir::func::FuncOp>();
  if (!function) {
    eraseCreatedReplacement(*replacement, transferSource);
    return false;
  }
  if (replacement->getType() != destRoot.getType() &&
      llvm::any_of(uses, [&](mlir::OpOperand *use) {
        mlir::Operation *owner = use->getOwner();
        return !mlir::isa<WaferInstructionOpInterface, mlir::memref::LoadOp,
                          mlir::memref::StoreOp>(owner);
      })) {
    eraseCreatedReplacement(*replacement, transferSource);
    return false;
  }
  mlir::DominanceInfo dominance(function);
  if (llvm::any_of(uses, [&](mlir::OpOperand *use) {
        return !dominance.dominates(*replacement, use->getOwner());
      })) {
    eraseCreatedReplacement(*replacement, transferSource);
    return false;
  }
  for (mlir::OpOperand *use : uses)
    use->set(*replacement);
  notifyReplacement(gather.getDest(), *replacement);
  if (destRoot != gather.getDest())
    notifyReplacement(destRoot, *replacement);
  if (raisedSourceAlignment)
    sourceAllocation->setAttr(
        "alignment",
        mlir::IntegerAttr::get(mlir::IntegerType::get(gather.getContext(), 64),
                               *raisedSourceAlignment));
  gather.erase();
  if (destAllocation->use_empty())
    destAllocation.erase();
  return true;
}

} // namespace

unsigned elideRedundantFullBufferTransfers(
    mlir::ModuleOp module,
    llvm::function_ref<void(mlir::Value, mlir::Value)> notifyReplacement) {
  wafer::support::ScopedCompileTimingSpan timing(
      "optimization", "full-buffer-transfer-elision",
      "elideRedundantFullBufferTransfers");
  bool hasAssignedPlacement = false;
  {
    wafer::support::ScopedCompileTimingSpan placementCheckTiming(
        "analysis-phase", "elideRedundantFullBufferTransfers",
        "check-placement");
    module.walk([&](mlir::memref::AllocOp allocation) {
      hasAssignedPlacement |= allocation->hasAttr(kWaferSPMOffsetAttrName) ||
                              allocation->hasAttr(kWaferDDROffsetAttrName);
    });
  }
  if (hasAssignedPlacement)
    return 0;

  unsigned eliminated = 0;
  while (true) {
    llvm::SmallVector<InstrGatherScatterOp, 8> candidates;
    {
      wafer::support::ScopedCompileTimingSpan collectTiming(
          "analysis-phase", "elideRedundantFullBufferTransfers",
          "collect-candidates");
      module.walk(
          [&](InstrGatherScatterOp gather) { candidates.push_back(gather); });
    }
    // Candidate rejection does not mutate IR.  Reuse the exact structured
    // timeline for every candidate in the same function until a rewrite is
    // applied; the successful rewrite ends this iteration and therefore
    // invalidates the cache before the next fixed-point step.
    llvm::DenseMap<mlir::Operation *, std::unique_ptr<mp::StructuredTimeline>>
        timelines;
    llvm::DenseSet<mlir::Operation *> unavailableTimelines;
    bool changed = false;
    for (InstrGatherScatterOp gather : candidates) {
      mlir::func::FuncOp function =
          gather->getParentOfType<mlir::func::FuncOp>();
      if (!gather->getBlock() || !function)
        continue;
      mlir::Operation *functionOperation = function.getOperation();
      if (unavailableTimelines.contains(functionOperation))
        continue;
      auto timelineIt = timelines.find(functionOperation);
      if (timelineIt == timelines.end()) {
        mlir::FailureOr<mp::StructuredTimeline> timeline;
        {
          wafer::support::ScopedCompileTimingSpan timelineTiming(
              "analysis-phase", "elideRedundantFullBufferTransfers",
              "StructuredTimeline::build");
          timeline = mp::StructuredTimeline::build(functionOperation);
        }
        if (mlir::failed(timeline)) {
          unavailableTimelines.insert(functionOperation);
          continue;
        }
        timelineIt = timelines
                         .try_emplace(functionOperation,
                                      std::make_unique<mp::StructuredTimeline>(
                                          std::move(*timeline)))
                         .first;
      }
      bool elided = false;
      {
        wafer::support::ScopedCompileTimingSpan elideTiming(
            "optimization-phase", "elideRedundantFullBufferTransfers",
            "tryElide");
        elided = tryElide(gather, *timelineIt->second, notifyReplacement);
      }
      if (!elided)
        continue;
      ++eliminated;
      changed = true;
      break;
    }
    if (!changed) {
      if (mlir::failed(mlir::verify(module))) {
        module.emitError(
            "full-buffer transfer elimination produced invalid IR");
        return 0;
      }
      return eliminated;
    }
  }
}

unsigned elideRedundantFullBufferTransfers(mlir::ModuleOp module) {
  return elideRedundantFullBufferTransfers(
      module, [](mlir::Value, mlir::Value) {});
}

} // namespace wafer::tensor_program_scheduling
