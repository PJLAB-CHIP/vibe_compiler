//===- NoCIntermediateDataflow.cpp - Typed intermediate handoff ---------===//

#include "NoCIntermediateDataflow.h"

#include "Wafer/Compiler/GlobalTileRelation.h"
#include "Wafer/IR/Common/OpVerifierUtils.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/CompileTiming.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>

namespace wafer::compiler::detail {
namespace {

enum class BoundaryKind {
  Input,
  Parameter,
};

struct BoundaryDescriptor {
  BoundaryKind kind = BoundaryKind::Input;
  int64_t argumentIndex = -1;
  int64_t programIndex = -1;
  llvm::ArrayRef<int64_t> globalShape;
  llvm::ArrayRef<int64_t> localShape;
  llvm::ArrayRef<frontend::ProgramRankSlice> rankSlices;
};

struct BoundaryTileIdentity {
  const BoundaryDescriptor *boundary = nullptr;
  const frontend::ProgramRankSlice *rankSlice = nullptr;
  ResolvedBoundaryTileView view;
};

struct IntermediateSpillCut {
  int64_t logicalRank = -1;
  mlir::memref::AllocOp spillRoot;
  InstrWDMAOp producerStore;
  InstrRDMAOp consumerLoad;
  mlir::Value producerBuffer;
  mlir::Value consumerBuffer;
  int64_t bytes = 0;
  llvm::SmallVector<mlir::memref::DeallocOp, 2> spillDeallocations;
  llvm::SmallVector<mlir::memref::CastOp, 4> spillCasts;
};

static const frontend::ProgramRankSlice *
findRankSlice(llvm::ArrayRef<frontend::ProgramRankSlice> slices, int64_t rank) {
  const frontend::ProgramRankSlice *result = nullptr;
  for (const frontend::ProgramRankSlice &slice : slices) {
    if (slice.logicalRank != rank)
      continue;
    if (result)
      return nullptr;
    result = &slice;
  }
  return result;
}

static mlir::Value resolveTransparentDataflowAlias(mlir::Value value) {
  llvm::DenseSet<mlir::Value> seen;
  while (value && seen.insert(value).second) {
    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      mlir::Block *owner = blockArg.getOwner();
      auto tileRegion =
          owner ? mlir::dyn_cast_or_null<TileRegionOp>(owner->getParentOp())
                : TileRegionOp();
      if (!tileRegion || tileRegion.getBody().empty() ||
          owner != &tileRegion.getBody().front() ||
          blockArg.getArgNumber() >= tileRegion.getInputs().size())
        break;
      value = tileRegion.getInputs()[blockArg.getArgNumber()];
      continue;
    }
    auto result = mlir::dyn_cast<mlir::OpResult>(value);
    auto tileRegion = result ? mlir::dyn_cast<TileRegionOp>(result.getOwner())
                             : TileRegionOp();
    if (tileRegion && !tileRegion.getBody().empty() &&
        result.getResultNumber() <
            tileRegion.getBody().front().getTerminator()->getNumOperands()) {
      value = tileRegion.getBody().front().getTerminator()->getOperand(
          result.getResultNumber());
      continue;
    }
    if (auto cast = value.getDefiningOp<mlir::memref::CastOp>()) {
      auto sourceType =
          mlir::dyn_cast<mlir::MemRefType>(cast.getSource().getType());
      auto resultType =
          mlir::dyn_cast<mlir::MemRefType>(cast.getResult().getType());
      if (!sourceType || !resultType ||
          sourceType.getShape() != resultType.getShape() ||
          sourceType.getElementType() != resultType.getElementType() ||
          sourceType.getMemorySpace() != resultType.getMemorySpace())
        break;
      value = cast.getSource();
      continue;
    }
    break;
  }
  return value;
}

static llvm::SmallVector<BoundaryDescriptor, 8> getBoundaryDescriptors(
    const frontend::FrontendProgramVerificationResult &program) {
  llvm::SmallVector<BoundaryDescriptor, 8> boundaries;
  for (const frontend::ProgramBoundaryBinding &binding :
       program.distributedInputs)
    boundaries.push_back({BoundaryKind::Input, binding.index,
                          binding.programIndex, binding.globalShape,
                          binding.localShape, binding.rankSlices});
  for (const frontend::ProgramParameterBinding &binding : program.parameters)
    boundaries.push_back({BoundaryKind::Parameter, binding.argumentIndex,
                          binding.argumentIndex, binding.globalShape,
                          binding.localShape, binding.rankSlices});
  return boundaries;
}

static std::optional<BoundaryTileIdentity>
resolveBoundaryTileIdentity(mlir::Value value, int64_t rank,
                            llvm::ArrayRef<BoundaryDescriptor> boundaries) {
  auto equivalent = [](const BoundaryTileIdentity &lhs,
                       const BoundaryTileIdentity &rhs) {
    if (!lhs.boundary || !rhs.boundary || !lhs.rankSlice || !rhs.rankSlice ||
        lhs.boundary->kind != rhs.boundary->kind ||
        lhs.boundary->programIndex != rhs.boundary->programIndex ||
        lhs.view.leafType != rhs.view.leafType)
      return false;
    return compareRankBoundaryTileViews(
               lhs.view, *lhs.rankSlice, rhs.view, *rhs.rankSlice,
               lhs.boundary->globalShape,
               lhs.boundary->localShape) == StaticTileRelation::Equivalent;
  };

  std::optional<BoundaryTileIdentity> identity;
  for (const BoundaryDescriptor &boundary : boundaries) {
    std::optional<ResolvedBoundaryTileView> resolved =
        resolveBoundaryTileView(value, boundary.localShape);
    if (!resolved ||
        static_cast<int64_t>(resolved->argumentIndex) != boundary.argumentIndex)
      continue;
    const frontend::ProgramRankSlice *slice =
        findRankSlice(boundary.rankSlices, rank);
    if (!slice)
      return std::nullopt;
    if (!hashRankBoundaryTileView(*resolved, *slice, boundary.globalShape,
                                  boundary.localShape))
      return std::nullopt;
    BoundaryTileIdentity current{&boundary, slice, std::move(*resolved)};
    if (identity && !equivalent(*identity, current))
      return std::nullopt;
    identity = std::move(current);
  }
  return identity;
}

static bool equivalentBoundaryTileIdentity(const BoundaryTileIdentity &lhs,
                                           const BoundaryTileIdentity &rhs) {
  if (!lhs.boundary || !rhs.boundary || !lhs.rankSlice || !rhs.rankSlice ||
      lhs.boundary->kind != rhs.boundary->kind ||
      lhs.boundary->programIndex != rhs.boundary->programIndex ||
      lhs.view.leafType != rhs.view.leafType ||
      lhs.boundary->globalShape != rhs.boundary->globalShape ||
      lhs.boundary->localShape != rhs.boundary->localShape)
    return false;
  return compareRankBoundaryTileViews(lhs.view, *lhs.rankSlice, rhs.view,
                                      *rhs.rankSlice, lhs.boundary->globalShape,
                                      lhs.boundary->localShape) ==
         StaticTileRelation::Equivalent;
}

static std::optional<int64_t> getPhysicalBytes(mlir::Type type) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memrefType || !memrefType.hasStaticShape())
    return std::nullopt;
  std::optional<WaferPhysicalTensorInfo> info =
      computeWaferPhysicalTensorInfo(memrefType);
  return info && info->physicalBytes > 0
             ? std::optional<int64_t>(info->physicalBytes)
             : std::nullopt;
}

static bool isUnitDescriptor(llvm::ArrayRef<int64_t> strides,
                             llvm::ArrayRef<int64_t> iterations) {
  return strides.size() == 3 && iterations.size() == 3 &&
         llvm::all_of(strides, [](int64_t value) { return value == 0; }) &&
         llvm::all_of(iterations, [](int64_t value) { return value == 1; });
}

static bool isCompleteStore(InstrWDMAOp store, int64_t bytes) {
  return (!store.getSrcOffsetAttr() ||
          store.getSrcOffsetAttr().getInt() == 0) &&
         (!store.getDstOffsetAttr() ||
          store.getDstOffsetAttr().getInt() == 0) &&
         store.getByteCountAttr().getInt() == bytes &&
         store.getInnerBytesAttr().getInt() == bytes &&
         isUnitDescriptor(store.getDstStrides(), store.getDstIterations());
}

static bool isCompleteLoad(InstrRDMAOp load, int64_t bytes) {
  return (!load.getSrcOffsetAttr() || load.getSrcOffsetAttr().getInt() == 0) &&
         (!load.getDstOffsetAttr() || load.getDstOffsetAttr().getInt() == 0) &&
         load.getByteCountAttr().getInt() == bytes &&
         load.getInnerBytesAttr().getInt() == bytes &&
         isUnitDescriptor(load.getSrcStrides(), load.getSrcIterations());
}

static bool isTransparentCast(mlir::memref::CastOp cast, mlir::Value source) {
  auto sourceType = mlir::dyn_cast<mlir::MemRefType>(source.getType());
  auto resultType =
      mlir::dyn_cast<mlir::MemRefType>(cast.getResult().getType());
  return cast.getSource() == source && sourceType && resultType &&
         sourceType.getShape() == resultType.getShape() &&
         sourceType.getElementType() == resultType.getElementType() &&
         sourceType.getMemorySpace() == resultType.getMemorySpace();
}

static bool hasOnlyCutUses(mlir::Value value, InstrWDMAOp store,
                           InstrRDMAOp load,
                           llvm::DenseSet<mlir::Value> &visited) {
  if (!visited.insert(value).second)
    return true;
  for (mlir::OpOperand &use : value.getUses()) {
    mlir::Operation *owner = use.getOwner();
    if (owner == store.getOperation() && use.getOperandNumber() == 1)
      continue;
    if (owner == load.getOperation() && use.getOperandNumber() == 0)
      continue;
    if (mlir::isa<mlir::memref::DeallocOp>(owner))
      continue;
    if (auto region = mlir::dyn_cast<TileRegionOp>(owner)) {
      unsigned index = use.getOperandNumber();
      if (index >= region.getInputs().size() || region.getBody().empty() ||
          index >= region.getBody().front().getNumArguments() ||
          !hasOnlyCutUses(region.getBody().front().getArgument(index), store,
                          load, visited))
        return false;
      continue;
    }
    if (auto yield = mlir::dyn_cast<TileYieldOp>(owner)) {
      auto region = yield->getParentOfType<TileRegionOp>();
      unsigned index = use.getOperandNumber();
      if (!region || index >= region.getNumResults() ||
          !hasOnlyCutUses(region.getResult(index), store, load, visited))
        return false;
      continue;
    }
    if (auto cast = mlir::dyn_cast<mlir::memref::CastOp>(owner)) {
      if (use.getOperandNumber() != 0 || !isTransparentCast(cast, value) ||
          !hasOnlyCutUses(cast.getResult(), store, load, visited))
        return false;
      continue;
    }
    return false;
  }
  return true;
}

static llvm::SmallVector<IntermediateSpillCut, 4>
collectIntermediateSpillCuts(mlir::ModuleOp module, int64_t rank) {
  llvm::SmallVector<mlir::memref::AllocOp, 8> roots;
  module.walk([&](mlir::memref::AllocOp allocation) {
    if (wafer::detail::hasWaferMemorySpace(allocation.getType(),
                                           MemorySpace::DDR))
      roots.push_back(allocation);
  });

  llvm::SmallVector<IntermediateSpillCut, 4> cuts;
  for (mlir::memref::AllocOp root : roots) {
    llvm::SmallVector<InstrWDMAOp, 2> stores;
    llvm::SmallVector<InstrRDMAOp, 2> loads;
    module.walk([&](InstrWDMAOp store) {
      if (resolveTransparentDataflowAlias(store.getDest()) == root.getResult())
        stores.push_back(store);
    });
    module.walk([&](InstrRDMAOp load) {
      if (resolveTransparentDataflowAlias(load.getSource()) == root.getResult())
        loads.push_back(load);
    });
    if (stores.size() != 1 || loads.size() != 1)
      continue;

    InstrWDMAOp store = stores.front();
    InstrRDMAOp load = loads.front();
    if (store->getBlock() != load->getBlock() || !store->isBeforeInBlock(load))
      continue;
    std::optional<int64_t> sourceBytes =
        getPhysicalBytes(store.getSource().getType());
    std::optional<int64_t> spillBytes = getPhysicalBytes(root.getType());
    std::optional<int64_t> consumerBytes =
        getPhysicalBytes(load.getDest().getType());
    if (!sourceBytes || !spillBytes || !consumerBytes ||
        *sourceBytes != *spillBytes || *sourceBytes != *consumerBytes ||
        store.getSource().getType() != load.getDest().getType() ||
        !isCompleteStore(store, *sourceBytes) ||
        !isCompleteLoad(load, *sourceBytes))
      continue;

    llvm::DenseSet<mlir::Value> visited;
    if (!hasOnlyCutUses(root.getResult(), store, load, visited))
      continue;
    IntermediateSpillCut cut{
        rank,           root,        store, load, store.getSource(),
        load.getDest(), *sourceBytes};
    llvm::DenseSet<mlir::Operation *> cleanupOperations;
    for (mlir::Value alias : visited) {
      for (mlir::OpOperand &use : alias.getUses()) {
        mlir::Operation *owner = use.getOwner();
        if (!cleanupOperations.insert(owner).second)
          continue;
        if (auto deallocation = mlir::dyn_cast<mlir::memref::DeallocOp>(owner))
          cut.spillDeallocations.push_back(deallocation);
        else if (auto cast = mlir::dyn_cast<mlir::memref::CastOp>(owner);
                 cast && isTransparentCast(cast, alias))
          cut.spillCasts.push_back(cast);
      }
    }
    cuts.push_back(std::move(cut));
  }
  return cuts;
}

static bool writesValue(mlir::Operation *operation, mlir::Value value) {
  auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
  if (!effects)
    return false;
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 8> instances;
  effects.getEffects(instances);
  return llvm::any_of(instances, [&](const auto &instance) {
    return instance.getValue() == value &&
           mlir::isa<mlir::MemoryEffects::Write>(instance.getEffect());
  });
}

static bool readsValue(mlir::Operation *operation, mlir::Value value) {
  auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
  if (!effects)
    return false;
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 8> instances;
  effects.getEffects(instances);
  return llvm::any_of(instances, [&](const auto &instance) {
    return instance.getValue() == value &&
           mlir::isa<mlir::MemoryEffects::Read>(instance.getEffect());
  });
}

static bool isPureOverwrite(mlir::Operation *operation, mlir::Value value) {
  return writesValue(operation, value) && !readsValue(operation, value);
}

static bool
isProvablePureAliasWriter(mlir::Operation *operation,
                          const llvm::DenseSet<mlir::Value> &aliasSet) {
  bool writesAlias = llvm::any_of(aliasSet, [&](mlir::Value candidate) {
    return writesValue(operation, candidate);
  });
  bool readsAlias = llvm::any_of(aliasSet, [&](mlir::Value candidate) {
    return readsValue(operation, candidate);
  });
  return writesAlias && !readsAlias;
}

/// Erase one private SPM producer closure after its only observable snapshot
/// has been replaced by a peer edge.  This is intentionally stricter than
/// ordinary DCE: every alias use must be a transparent cast, one pure
/// instruction writer, or a deallocation.  The walk then recursively removes
/// newly-dead private input buffers (for example the RDMA feeding a redundant
/// producer).  Direct-DTE and WDMA are never removed by this helper.
static bool eraseDeadPrivateProducerValue(mlir::Value value,
                                          llvm::DenseSet<mlir::Value> &active) {
  value = resolveTransparentDataflowAlias(value);
  auto allocation = value.getDefiningOp<mlir::memref::AllocOp>();
  if (!allocation ||
      !wafer::detail::hasWaferMemorySpace(allocation.getType(),
                                          MemorySpace::SPM) ||
      !active.insert(value).second)
    return false;
  auto finish = [&](bool erased) {
    active.erase(value);
    return erased;
  };

  llvm::SmallVector<mlir::Value, 4> aliases{value};
  llvm::DenseSet<mlir::Value> aliasSet;
  aliasSet.insert(value);
  llvm::SmallVector<mlir::memref::CastOp, 4> casts;
  llvm::SmallVector<mlir::memref::DeallocOp, 2> deallocations;
  // Discover the complete transparent-alias closure before interpreting any
  // instruction operand role. This avoids making the proof depend on use-list
  // order when a writer reads through a cast of its destination.
  for (size_t index = 0; index < aliases.size(); ++index) {
    mlir::Value alias = aliases[index];
    for (mlir::OpOperand &use : alias.getUses()) {
      mlir::Operation *operation = use.getOwner();
      if (auto cast = mlir::dyn_cast<mlir::memref::CastOp>(operation)) {
        if (use.getOperandNumber() != 0 || !isTransparentCast(cast, alias))
          return finish(false);
        if (aliasSet.insert(cast.getResult()).second) {
          casts.push_back(cast);
          aliases.push_back(cast.getResult());
        }
      }
    }
  }

  mlir::Operation *writer = nullptr;
  for (mlir::Value alias : aliases) {
    for (mlir::OpOperand &use : alias.getUses()) {
      mlir::Operation *operation = use.getOwner();
      if (auto cast = mlir::dyn_cast<mlir::memref::CastOp>(operation);
          cast && use.getOperandNumber() == 0 &&
          isTransparentCast(cast, alias) && aliasSet.contains(cast.getResult()))
        continue;
      if (auto deallocation =
              mlir::dyn_cast<mlir::memref::DeallocOp>(operation)) {
        deallocations.push_back(deallocation);
        continue;
      }
      if (!isProvablePureAliasWriter(operation, aliasSet) ||
          (writer && writer != operation))
        return finish(false);
      writer = operation;
    }
  }
  auto instruction = writer
                         ? mlir::dyn_cast<WaferInstructionOpInterface>(writer)
                         : WaferInstructionOpInterface();
  if (!instruction || instruction.getInstructionFamily() == InstrFamily::DTE ||
      instruction.getInstructionFamily() == InstrFamily::WDMA ||
      llvm::any_of(writer->getResults(),
                   [](mlir::Value result) { return !result.use_empty(); }))
    return finish(false);

  // The writer may read other private values, but it must not write any value
  // outside this exact alias set.
  auto effects = mlir::cast<mlir::MemoryEffectOpInterface>(writer);
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 8> instances;
  effects.getEffects(instances);
  for (const auto &instance : instances) {
    if (!mlir::isa<mlir::MemoryEffects::Write>(instance.getEffect()))
      continue;
    mlir::Value written = instance.getValue();
    // Resource-only issue effects (for example the compute engine) have no
    // SSA value and disappear with the writer itself. Only a write to another
    // concrete memory value would make the producer closure observable.
    if (written && !aliasSet.contains(written))
      return finish(false);
  }

  llvm::SmallVector<mlir::Value, 4> inputs;
  llvm::DenseSet<mlir::Value> seenInputs;
  for (mlir::Value operand : writer->getOperands())
    if (mlir::isa<mlir::MemRefType>(operand.getType()) &&
        !aliasSet.contains(operand) && seenInputs.insert(operand).second)
      inputs.push_back(operand);

  for (mlir::memref::DeallocOp deallocation : deallocations)
    if (deallocation->getBlock())
      deallocation.erase();
  writer->erase();
  for (mlir::memref::CastOp cast : llvm::reverse(casts))
    if (cast->getBlock() && cast.getResult().use_empty())
      cast.erase();
  if (!allocation.getResult().use_empty())
    return finish(false);
  // Remove the handle from the recursion guard before destroying its defining
  // operation; DenseSet operations on a dangling MLIR Value are invalid.
  active.erase(value);
  allocation.erase();

  for (mlir::Value input : inputs)
    eraseDeadPrivateProducerValue(input, active);
  return true;
}

static void eraseDeadPrivateProducerValue(mlir::Value value) {
  llvm::DenseSet<mlir::Value> active;
  eraseDeadPrivateProducerValue(value, active);
}

static bool
hasUnsafeProducerUseAfterStore(mlir::Value value, InstrWDMAOp store,
                               mlir::Block *block,
                               llvm::DenseSet<mlir::Value> &visited) {
  if (!visited.insert(value).second)
    return false;
  for (mlir::OpOperand &use : value.getUses()) {
    mlir::Operation *operation = use.getOwner();
    if (operation == store.getOperation() && value == store.getSource() &&
        use.getOperandNumber() == 0)
      continue;
    if (operation->getBlock() != block)
      return true;

    if (auto view = mlir::dyn_cast<mlir::ViewLikeOpInterface>(operation);
        view && view.getViewSource() == value) {
      for (mlir::Value result : operation->getResults())
        if (mlir::isa<mlir::MemRefType>(result.getType()) &&
            hasUnsafeProducerUseAfterStore(result, store, block, visited))
          return true;
      continue;
    }

    if (!store->isBeforeInBlock(operation))
      continue;
    if (writesValue(operation, value) ||
        mlir::isa<mlir::memref::DeallocOp>(operation))
      return true;
    if (!readsValue(operation, value) && !mlir::isMemoryEffectFree(operation))
      return true;
  }
  return false;
}

static bool canReuseProducerBufferLocally(const IntermediateSpillCut &cut) {
  InstrWDMAOp store = cut.producerStore;
  InstrRDMAOp load = cut.consumerLoad;
  if (store->getBlock() != load->getBlock() || !store->isBeforeInBlock(load) ||
      cut.producerBuffer.getType() != cut.consumerBuffer.getType() ||
      !cut.consumerBuffer.getDefiningOp<mlir::memref::AllocOp>())
    return false;

  mlir::Block *block = load->getBlock();
  for (mlir::OpOperand &use : cut.consumerBuffer.getUses()) {
    mlir::Operation *operation = use.getOwner();
    if (operation == load.getOperation() && use.getOperandNumber() == 1)
      continue;
    if (operation->getBlock() != block || !load->isBeforeInBlock(operation) ||
        !readsValue(operation, cut.consumerBuffer) ||
        writesValue(operation, cut.consumerBuffer))
      return false;
  }
  // Reusing the producer buffer extends its live range through every local
  // consumer. Follow typed view aliases as well as the root value; a write,
  // free, unknown observer, or cross-block use after the spill snapshot makes
  // local reuse unsafe.
  llvm::DenseSet<mlir::Value> visited;
  return !hasUnsafeProducerUseAfterStore(cut.producerBuffer, store, block,
                                         visited);
}

static mlir::Operation *findLastWriterBefore(mlir::Value value,
                                             mlir::Operation *before) {
  if (!before)
    return nullptr;
  mlir::Operation *writer = nullptr;
  for (mlir::OpOperand &use : value.getUses()) {
    mlir::Operation *operation = use.getOwner();
    if (operation == before || operation->getBlock() != before->getBlock() ||
        !operation->isBeforeInBlock(before) || !writesValue(operation, value))
      continue;
    if (!writer || writer->isBeforeInBlock(operation))
      writer = operation;
  }
  return writer;
}

class ProducedValueEquivalence {
public:
  ProducedValueEquivalence(llvm::ArrayRef<BoundaryDescriptor> boundaries,
                           int64_t lhsRank, int64_t rhsRank)
      : boundaries(boundaries), lhsRank(lhsRank), rhsRank(rhsRank) {}

  bool prove(mlir::Value lhs, mlir::Operation *lhsBefore, mlir::Value rhs,
             mlir::Operation *rhsBefore) {
    if (!lhs || !rhs || lhs.getType() != rhs.getType())
      return false;
    if (lhs == rhs)
      return true;

    auto key = std::make_tuple(lhs, lhsBefore, rhs, rhsBefore);
    if (!active.insert(key).second)
      return false;
    auto finish = [&](bool result) {
      active.erase(key);
      return result;
    };

    std::optional<BoundaryTileIdentity> lhsBoundary =
        resolveBoundaryTileIdentity(lhs, lhsRank, boundaries);
    std::optional<BoundaryTileIdentity> rhsBoundary =
        resolveBoundaryTileIdentity(rhs, rhsRank, boundaries);
    if (lhsBoundary || rhsBoundary)
      return finish(lhsBoundary && rhsBoundary &&
                    equivalentBoundaryTileIdentity(*lhsBoundary, *rhsBoundary));

    mlir::Value lhsResolved = resolveTransparentDataflowAlias(lhs);
    mlir::Value rhsResolved = resolveTransparentDataflowAlias(rhs);
    if (lhsResolved != lhs || rhsResolved != rhs)
      return finish(prove(lhsResolved, lhsBefore, rhsResolved, rhsBefore));

    mlir::Operation *lhsWriter = findLastWriterBefore(lhs, lhsBefore);
    mlir::Operation *rhsWriter = findLastWriterBefore(rhs, rhsBefore);
    if (lhsWriter || rhsWriter) {
      if (!lhsWriter || !rhsWriter)
        return finish(false);
      auto lhsInstruction =
          mlir::dyn_cast<WaferInstructionOpInterface>(lhsWriter);
      auto rhsInstruction =
          mlir::dyn_cast<WaferInstructionOpInterface>(rhsWriter);
      if (!lhsInstruction || !rhsInstruction ||
          lhsInstruction.getInstructionFamily() == InstrFamily::DTE ||
          rhsInstruction.getInstructionFamily() == InstrFamily::DTE ||
          lhsInstruction.getInstructionFamily() == InstrFamily::WDMA ||
          rhsInstruction.getInstructionFamily() == InstrFamily::WDMA)
        return finish(false);

      bool equivalent = mlir::OperationEquivalence::isEquivalentTo(
          lhsWriter, rhsWriter,
          [&](mlir::Value lhsOperand,
              mlir::Value rhsOperand) -> mlir::LogicalResult {
            if (lhsOperand == lhs && rhsOperand == rhs) {
              if (isPureOverwrite(lhsWriter, lhs) &&
                  isPureOverwrite(rhsWriter, rhs))
                return mlir::success();
              return mlir::success(prove(lhs, lhsWriter, rhs, rhsWriter));
            }
            return mlir::success(
                prove(lhsOperand, lhsWriter, rhsOperand, rhsWriter));
          },
          /*markEquivalent=*/nullptr,
          mlir::OperationEquivalence::IgnoreLocations);
      return finish(equivalent);
    }

    mlir::Operation *lhsDefinition = lhs.getDefiningOp();
    mlir::Operation *rhsDefinition = rhs.getDefiningOp();
    if (!lhsDefinition || !rhsDefinition ||
        !mlir::isMemoryEffectFree(lhsDefinition) ||
        !mlir::isMemoryEffectFree(rhsDefinition))
      return finish(false);
    bool equivalent = mlir::OperationEquivalence::isEquivalentTo(
        lhsDefinition, rhsDefinition,
        [&](mlir::Value lhsOperand,
            mlir::Value rhsOperand) -> mlir::LogicalResult {
          return mlir::success(
              prove(lhsOperand, lhsDefinition, rhsOperand, rhsDefinition));
        },
        /*markEquivalent=*/nullptr,
        mlir::OperationEquivalence::IgnoreLocations);
    return finish(equivalent);
  }

private:
  llvm::ArrayRef<BoundaryDescriptor> boundaries;
  int64_t lhsRank = -1;
  int64_t rhsRank = -1;
  llvm::DenseSet<std::tuple<mlir::Value, mlir::Operation *, mlir::Value,
                            mlir::Operation *>>
      active;
};

struct RequiredOutputPublisher {
  int64_t logicalRank = -1;
  const frontend::ProgramBoundaryBinding *binding = nullptr;
  const frontend::ProgramRankSlice *rankSlice = nullptr;
  InstrWDMAOp publisher;
  mlir::Value source;
  mlir::Value destinationRoot;
  StaticTileRegion globalTile;
  int64_t bytes = 0;
};

struct RequiredOutputGroup {
  llvm::SmallVector<RequiredOutputPublisher, 16> publishers;
};

static mlir::Value stripTransparentMemRefCasts(mlir::Value value) {
  llvm::DenseSet<mlir::Value> seen;
  while (value && seen.insert(value).second) {
    auto cast = value.getDefiningOp<mlir::memref::CastOp>();
    if (!cast || !isTransparentCast(cast, cast.getSource()))
      break;
    value = cast.getSource();
  }
  return value;
}

static std::optional<unsigned> getRequiredOutputOrdinal(mlir::Value destination,
                                                        size_t outputCount) {
  destination = stripTransparentMemRefCasts(destination);
  auto argument = mlir::dyn_cast<mlir::BlockArgument>(destination);
  if (!argument)
    return std::nullopt;
  mlir::Block *owner = argument.getOwner();
  auto tileRegion =
      owner ? mlir::dyn_cast_or_null<TileRegionOp>(owner->getParentOp())
            : TileRegionOp();
  if (!tileRegion || tileRegion.getBody().empty() ||
      owner != &tileRegion.getBody().front() ||
      argument.getArgNumber() >= tileRegion.getInputs().size() ||
      outputCount == 0 || tileRegion.getInputs().size() < outputCount)
    return std::nullopt;
  const size_t outputBase = tileRegion.getInputs().size() - outputCount;
  if (argument.getArgNumber() < outputBase)
    return std::nullopt;
  return static_cast<unsigned>(argument.getArgNumber() - outputBase);
}

static bool isFinalRequiredOutputWriter(mlir::ModuleOp module,
                                        InstrWDMAOp publisher,
                                        mlir::Value destinationRoot) {
  bool valid = true;
  module.walk([&](mlir::Operation *operation) {
    if (!valid || operation == publisher.getOperation())
      return;
    auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
    if (!effects)
      return;
    llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 8> instances;
    effects.getEffects(instances);
    bool writesOutput = llvm::any_of(instances, [&](const auto &instance) {
      return mlir::isa<mlir::MemoryEffects::Write>(instance.getEffect()) &&
             instance.getValue() &&
             resolveTransparentDataflowAlias(instance.getValue()) ==
                 destinationRoot;
    });
    if (!writesOutput)
      return;
    // Earlier initialization in the same structured occurrence is harmless;
    // a sibling/nested occurrence or later writer means this WDMA is not the
    // required final publisher.
    valid = operation->getBlock() == publisher->getBlock() &&
            operation->isBeforeInBlock(publisher);
  });
  return valid;
}

static std::optional<RequiredOutputPublisher>
findRequiredOutputPublisher(mlir::ModuleOp module, int64_t logicalRank,
                            const frontend::ProgramBoundaryBinding &binding,
                            size_t outputCount) {
  if (binding.index < 0 || binding.index >= static_cast<int64_t>(outputCount) ||
      binding.programIndex < 0 ||
      binding.distribution != frontend::ProgramDistributionKind::Replicated)
    return std::nullopt;
  const frontend::ProgramRankSlice *rankSlice =
      findRankSlice(binding.rankSlices, logicalRank);
  if (!rankSlice)
    return std::nullopt;

  llvm::SmallVector<InstrWDMAOp, 2> stores;
  module.walk([&](InstrWDMAOp store) {
    std::optional<unsigned> ordinal =
        getRequiredOutputOrdinal(store.getDest(), outputCount);
    if (ordinal && *ordinal == static_cast<unsigned>(binding.index))
      stores.push_back(store);
  });
  if (stores.size() != 1)
    return std::nullopt;
  InstrWDMAOp publisher = stores.front();

  std::optional<int64_t> sourceBytes =
      getPhysicalBytes(publisher.getSource().getType());
  std::optional<int64_t> destinationBytes =
      getPhysicalBytes(publisher.getDest().getType());
  auto sourceType =
      mlir::dyn_cast<mlir::MemRefType>(publisher.getSource().getType());
  auto destinationType =
      mlir::dyn_cast<mlir::MemRefType>(publisher.getDest().getType());
  if (!sourceBytes || !destinationBytes || *sourceBytes != *destinationBytes ||
      !sourceType || !destinationType ||
      !wafer::detail::hasWaferMemorySpace(sourceType, MemorySpace::SPM) ||
      !wafer::detail::hasWaferMemorySpace(destinationType, MemorySpace::DDR) ||
      sourceType.getShape() != destinationType.getShape() ||
      sourceType.getElementType() != destinationType.getElementType() ||
      destinationType.getShape() !=
          llvm::ArrayRef<int64_t>(binding.localShape) ||
      !isCompleteStore(publisher, *sourceBytes))
    return std::nullopt;

  StaticTileRegion localTile;
  localTile.offsets.assign(binding.localShape.size(), 0);
  localTile.sizes = binding.localShape;
  localTile.strides.assign(binding.localShape.size(), 1);
  llvm::Expected<StaticTileRegion> globalTile = mapRankLocalTileToGlobal(
      *rankSlice, binding.globalShape, binding.localShape, localTile);
  if (!globalTile) {
    llvm::consumeError(globalTile.takeError());
    return std::nullopt;
  }

  mlir::Value destinationRoot =
      resolveTransparentDataflowAlias(publisher.getDest());
  if (!destinationRoot ||
      !isFinalRequiredOutputWriter(module, publisher, destinationRoot))
    return std::nullopt;

  return RequiredOutputPublisher{logicalRank,
                                 &binding,
                                 rankSlice,
                                 publisher,
                                 publisher.getSource(),
                                 destinationRoot,
                                 std::move(*globalTile),
                                 *sourceBytes};
}

static llvm::SmallVector<RequiredOutputGroup, 4> collectRequiredOutputGroups(
    llvm::MutableArrayRef<mlir::ModuleOp> modules,
    const frontend::FrontendProgramVerificationResult &program,
    llvm::ArrayRef<BoundaryDescriptor> boundaries) {
  llvm::SmallVector<RequiredOutputGroup, 4> groups;
  static constexpr size_t kRequiredOutputGroupLimit = 4;
  for (const frontend::ProgramBoundaryBinding &binding :
       program.distributedOutputs) {
    if (groups.size() >= kRequiredOutputGroupLimit)
      break;
    RequiredOutputGroup group;
    for (auto [rank, module] : llvm::enumerate(modules)) {
      std::optional<RequiredOutputPublisher> publisher =
          findRequiredOutputPublisher(module, static_cast<int64_t>(rank),
                                      binding,
                                      program.distributedOutputs.size());
      if (!publisher) {
        group.publishers.clear();
        break;
      }
      if (!group.publishers.empty()) {
        const RequiredOutputPublisher &anchor = group.publishers.front();
        if (publisher->bytes != anchor.bytes ||
            publisher->source.getType() != anchor.source.getType() ||
            compareStaticTiles(publisher->globalTile, anchor.globalTile) !=
                StaticTileRelation::Equivalent ||
            !haveEquivalentStructuredOperationPaths(publisher->publisher,
                                                    anchor.publisher)) {
          group.publishers.clear();
          break;
        }
        ProducedValueEquivalence equivalence(boundaries, anchor.logicalRank,
                                             publisher->logicalRank);
        if (!equivalence.prove(anchor.source, anchor.publisher,
                               publisher->source, publisher->publisher)) {
          group.publishers.clear();
          break;
        }
      }
      group.publishers.push_back(std::move(*publisher));
    }
    if (group.publishers.size() != modules.size() ||
        group.publishers.size() < 2)
      continue;
    // The program member ordinal is stable across processes; LLVM hashes are
    // deliberately not used to choose a physical owner.
    const size_t selected =
        static_cast<size_t>(binding.programIndex) % group.publishers.size();
    std::rotate(group.publishers.begin(), group.publishers.begin() + selected,
                group.publishers.end());
    groups.push_back(std::move(group));
  }
  return groups;
}

static void materializeRequiredOutputGroup(const RequiredOutputGroup &group,
                                           int64_t communicationId,
                                           NoCFanoutKind kind) {
  const RequiredOutputPublisher &owner = group.publishers.front();
  InstrWDMAOp ownerPublisher = owner.publisher;
  auto message =
      DTEMessageAttr::get(ownerPublisher.getContext(), communicationId,
                          DTEProtocolPhase::PeerDataflow, /*round=*/2,
                          /*payloadSlice=*/owner.binding->programIndex);

  mlir::OpBuilder ownerBuilder(ownerPublisher);
  if (kind == NoCFanoutKind::Direct) {
    for (const RequiredOutputPublisher &peer :
         llvm::drop_begin(group.publishers)) {
      auto send = ownerBuilder.create<CommPeerSendOp>(
          ownerPublisher.getLoc(),
          ownerBuilder.getType<mlir::async::TokenType>(), owner.source,
          ownerBuilder.getI64IntegerAttr(peer.logicalRank),
          ownerBuilder.getI64IntegerAttr(owner.bytes), message);
      ownerBuilder.create<mlir::async::AwaitOp>(ownerPublisher.getLoc(),
                                                send.getToken());
    }
  } else {
    const RequiredOutputPublisher &peer = group.publishers[1];
    auto send = ownerBuilder.create<CommPeerSendOp>(
        ownerPublisher.getLoc(), ownerBuilder.getType<mlir::async::TokenType>(),
        owner.source, ownerBuilder.getI64IntegerAttr(peer.logicalRank),
        ownerBuilder.getI64IntegerAttr(owner.bytes), message);
    ownerBuilder.create<mlir::async::AwaitOp>(ownerPublisher.getLoc(),
                                              send.getToken());
  }

  for (size_t index = 1; index < group.publishers.size(); ++index) {
    const RequiredOutputPublisher &peer = group.publishers[index];
    InstrWDMAOp peerPublisher = peer.publisher;
    mlir::OpBuilder builder(peerPublisher);
    auto received = builder.create<mlir::memref::AllocOp>(
        peerPublisher.getLoc(),
        mlir::cast<mlir::MemRefType>(peer.source.getType()));
    const int64_t source = kind == NoCFanoutKind::Direct
                               ? owner.logicalRank
                               : group.publishers[index - 1].logicalRank;
    auto recv = builder.create<CommPeerRecvOp>(
        peerPublisher.getLoc(), builder.getType<mlir::async::TokenType>(),
        received.getResult(), builder.getI64IntegerAttr(source),
        builder.getI64IntegerAttr(peer.bytes), message);
    auto await = builder.create<mlir::async::AwaitOp>(peerPublisher.getLoc(),
                                                      recv.getToken());
    if (kind == NoCFanoutKind::ReceiveForward &&
        index + 1 < group.publishers.size()) {
      builder.setInsertionPointAfter(await);
      auto send = builder.create<CommPeerSendOp>(
          peerPublisher.getLoc(), builder.getType<mlir::async::TokenType>(),
          received.getResult(),
          builder.getI64IntegerAttr(group.publishers[index + 1].logicalRank),
          builder.getI64IntegerAttr(peer.bytes), message);
      builder.create<mlir::async::AwaitOp>(peerPublisher.getLoc(),
                                           send.getToken());
    }
    peerPublisher->setOperand(0, received.getResult());
    eraseDeadPrivateProducerValue(peer.source);
  }
}

static bool
proveEquivalentProducer(const IntermediateSpillCut &lhs,
                        const IntermediateSpillCut &rhs,
                        llvm::ArrayRef<BoundaryDescriptor> boundaries) {
  if (lhs.bytes != rhs.bytes ||
      lhs.producerBuffer.getType() != rhs.producerBuffer.getType() ||
      lhs.consumerBuffer.getType() != rhs.consumerBuffer.getType() ||
      !haveEquivalentStructuredOperationPaths(lhs.producerStore,
                                              rhs.producerStore) ||
      !haveEquivalentStructuredOperationPaths(lhs.consumerLoad,
                                              rhs.consumerLoad))
    return false;
  ProducedValueEquivalence equivalence(boundaries, lhs.logicalRank,
                                       rhs.logicalRank);
  return equivalence.prove(lhs.producerBuffer, lhs.producerStore,
                           rhs.producerBuffer, rhs.producerStore);
}

struct IntermediateGroup {
  llvm::SmallVector<IntermediateSpillCut *, 16> cuts;
  bool ambiguous = false;
};

static llvm::SmallVector<IntermediateGroup, 4> buildIntermediateGroups(
    llvm::MutableArrayRef<llvm::SmallVector<IntermediateSpillCut, 4>>
        cutsByRank,
    llvm::ArrayRef<BoundaryDescriptor> boundaries) {
  llvm::SmallVector<IntermediateGroup, 8> semanticGroups;
  for (auto &rankCuts : cutsByRank) {
    for (IntermediateSpillCut &cut : rankCuts) {
      auto group = llvm::find_if(
          semanticGroups, [&](const IntermediateGroup &candidate) {
            return proveEquivalentProducer(*candidate.cuts.front(), cut,
                                           boundaries);
          });
      if (group == semanticGroups.end()) {
        semanticGroups.push_back(IntermediateGroup{{&cut}});
        continue;
      }
      if (llvm::any_of(group->cuts, [&](const IntermediateSpillCut *member) {
            return member->logicalRank == cut.logicalRank;
          })) {
        // Two equivalent cuts on one rank have no unique typed occurrence to
        // replace. Keep this whole equivalence class on the DDR path.
        group->ambiguous = true;
        continue;
      }
      group->cuts.push_back(&cut);
    }
  }

  llvm::SmallVector<IntermediateGroup, 4> groups;
  static constexpr size_t kIntermediateGroupLimit = 4;
  for (IntermediateGroup &group : semanticGroups) {
    if (groups.size() >= kIntermediateGroupLimit)
      break;
    if (group.ambiguous || group.cuts.size() < 2)
      continue;
    groups.push_back(std::move(group));
  }
  llvm::SmallVector<IntermediateGroup, 4> reusableGroups;
  llvm::DenseMap<int64_t, unsigned> ownerUseCounts;
  for (auto [groupOrdinal, group] : llvm::enumerate(groups)) {
    llvm::SmallVector<size_t, 16> reusableOwners;
    for (auto [index, cut] : llvm::enumerate(group.cuts))
      if (canReuseProducerBufferLocally(*cut))
        reusableOwners.push_back(index);
    if (reusableOwners.empty())
      continue;

    unsigned minimumUseCount = std::numeric_limits<unsigned>::max();
    for (size_t index : reusableOwners)
      minimumUseCount =
          std::min(minimumUseCount,
                   ownerUseCounts.lookup(group.cuts[index]->logicalRank));
    llvm::SmallVector<size_t, 16> leastUsedOwners;
    for (size_t index : reusableOwners)
      if (ownerUseCounts.lookup(group.cuts[index]->logicalRank) ==
          minimumUseCount)
        leastUsedOwners.push_back(index);

    // Balance independent semantic groups over the currently least-used
    // eligible ranks. Stable group discovery order breaks equal-load ties, so
    // owner choice and package bytes are reproducible across processes.
    const size_t selected =
        leastUsedOwners[groupOrdinal % leastUsedOwners.size()];
    ++ownerUseCounts[group.cuts[selected]->logicalRank];
    std::rotate(group.cuts.begin(), group.cuts.begin() + selected,
                group.cuts.end());
    reusableGroups.push_back(std::move(group));
  }
  return reusableGroups;
}

static void eraseProvablyDeadSpillStorage(IntermediateSpillCut &cut) {
  for (mlir::memref::DeallocOp deallocation : cut.spillDeallocations)
    if (deallocation->getBlock())
      deallocation.erase();

  bool changed = true;
  while (changed) {
    changed = false;
    for (mlir::memref::CastOp cast : cut.spillCasts) {
      if (!cast->getBlock() || !cast.getResult().use_empty())
        continue;
      cast.erase();
      changed = true;
    }
  }
  if (cut.spillRoot->getBlock() && cut.spillRoot.getResult().use_empty())
    cut.spillRoot.erase();
}

static void materializeOwnerLocalReuse(IntermediateSpillCut &owner) {
  owner.consumerLoad.erase();
  owner.consumerBuffer.replaceAllUsesWith(owner.producerBuffer);
  if (auto allocation =
          owner.consumerBuffer.getDefiningOp<mlir::memref::AllocOp>();
      allocation && allocation.getResult().use_empty())
    allocation.erase();
  owner.producerStore.erase();
  eraseProvablyDeadSpillStorage(owner);
}

static void materializeIntermediateGroup(const IntermediateGroup &group,
                                         int64_t communicationId,
                                         NoCFanoutKind kind) {
  IntermediateSpillCut &owner = *group.cuts.front();
  auto message = DTEMessageAttr::get(
      owner.producerStore.getContext(), communicationId,
      DTEProtocolPhase::PeerDataflow, /*round=*/1, /*payloadSlice=*/0);

  mlir::OpBuilder ownerBuilder(owner.producerStore);
  ownerBuilder.setInsertionPointAfter(owner.producerStore);
  if (kind == NoCFanoutKind::Direct) {
    for (IntermediateSpillCut *peer : llvm::drop_begin(group.cuts)) {
      auto send = ownerBuilder.create<CommPeerSendOp>(
          owner.producerStore.getLoc(),
          ownerBuilder.getType<mlir::async::TokenType>(), owner.producerBuffer,
          ownerBuilder.getI64IntegerAttr(peer->logicalRank),
          ownerBuilder.getI64IntegerAttr(owner.bytes), message);
      ownerBuilder.create<mlir::async::AwaitOp>(owner.producerStore.getLoc(),
                                                send.getToken());
    }
  } else {
    IntermediateSpillCut &peer = *group.cuts[1];
    auto send = ownerBuilder.create<CommPeerSendOp>(
        owner.producerStore.getLoc(),
        ownerBuilder.getType<mlir::async::TokenType>(), owner.producerBuffer,
        ownerBuilder.getI64IntegerAttr(peer.logicalRank),
        ownerBuilder.getI64IntegerAttr(owner.bytes), message);
    ownerBuilder.create<mlir::async::AwaitOp>(owner.producerStore.getLoc(),
                                              send.getToken());
  }

  for (size_t index = 1; index < group.cuts.size(); ++index) {
    IntermediateSpillCut &cut = *group.cuts[index];
    const int64_t source = kind == NoCFanoutKind::Direct
                               ? owner.logicalRank
                               : group.cuts[index - 1]->logicalRank;
    mlir::OpBuilder builder(cut.consumerLoad);
    auto recv = builder.create<CommPeerRecvOp>(
        cut.consumerLoad.getLoc(), builder.getType<mlir::async::TokenType>(),
        cut.consumerBuffer, builder.getI64IntegerAttr(source),
        builder.getI64IntegerAttr(cut.bytes), message);
    auto await = builder.create<mlir::async::AwaitOp>(cut.consumerLoad.getLoc(),
                                                      recv.getToken());
    if (kind == NoCFanoutKind::ReceiveForward &&
        index + 1 < group.cuts.size()) {
      builder.setInsertionPointAfter(await);
      auto send = builder.create<CommPeerSendOp>(
          cut.consumerLoad.getLoc(), builder.getType<mlir::async::TokenType>(),
          cut.consumerBuffer,
          builder.getI64IntegerAttr(group.cuts[index + 1]->logicalRank),
          builder.getI64IntegerAttr(cut.bytes), message);
      builder.create<mlir::async::AwaitOp>(cut.consumerLoad.getLoc(),
                                           send.getToken());
    }
    cut.consumerLoad.erase();
    cut.producerStore.erase();
    eraseProvablyDeadSpillStorage(cut);
    eraseDeadPrivateProducerValue(cut.producerBuffer);
  }
  materializeOwnerLocalReuse(owner);
}

} // namespace

unsigned materializeNoCIntermediateHandoffs(
    llvm::MutableArrayRef<mlir::ModuleOp> modules,
    const frontend::FrontendProgramVerificationResult &program,
    int64_t &communicationId, NoCFanoutKind kind) {
  wafer::support::ScopedCompileTimingSpan timing(
      "transformation", "materializeNoCIntermediateHandoffs", "total");
  if (modules.size() < 2 || communicationId < 0)
    return 0;
  llvm::SmallVector<BoundaryDescriptor, 8> boundaries =
      getBoundaryDescriptors(program);

  llvm::SmallVector<llvm::SmallVector<IntermediateSpillCut, 4>, 16> cutsByRank;
  auto phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "analysis-phase", "materializeNoCIntermediateHandoffs",
      "collectIntermediateSpillCuts");
  cutsByRank.reserve(modules.size());
  for (auto [rank, module] : llvm::enumerate(modules))
    cutsByRank.push_back(
        collectIntermediateSpillCuts(module, static_cast<int64_t>(rank)));
  phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "analysis-phase", "materializeNoCIntermediateHandoffs",
      "buildIntermediateGroups");
  llvm::SmallVector<IntermediateGroup, 4> groups =
      buildIntermediateGroups(cutsByRank, boundaries);
  const uint64_t availableIds =
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) -
      static_cast<uint64_t>(communicationId) + uint64_t{1};
  if (groups.empty() || static_cast<uint64_t>(groups.size()) > availableIds)
    return 0;

  phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "transformation-phase", "materializeNoCIntermediateHandoffs",
      "materializeIntermediateGroup");
  for (const IntermediateGroup &group : groups) {
    materializeIntermediateGroup(group, communicationId, kind);
    communicationId = communicationId == std::numeric_limits<int64_t>::max()
                          ? -1
                          : communicationId + 1;
  }
  return static_cast<unsigned>(groups.size());
}

unsigned materializeNoCOutputPublications(
    llvm::MutableArrayRef<mlir::ModuleOp> modules,
    const frontend::FrontendProgramVerificationResult &program,
    int64_t &communicationId, NoCFanoutKind kind) {
  wafer::support::ScopedCompileTimingSpan timing(
      "transformation", "materializeNoCOutputPublications", "total");
  if (modules.size() < 2 || communicationId < 0 ||
      modules.size() != static_cast<size_t>(program.logicalRankCount) ||
      program.distributedOutputs.empty())
    return 0;
  llvm::SmallVector<BoundaryDescriptor, 8> boundaries =
      getBoundaryDescriptors(program);
  auto phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "analysis-phase", "materializeNoCOutputPublications",
      "collectRequiredOutputGroups");
  llvm::SmallVector<RequiredOutputGroup, 4> groups =
      collectRequiredOutputGroups(modules, program, boundaries);
  const uint64_t availableIds =
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) -
      static_cast<uint64_t>(communicationId) + uint64_t{1};
  if (groups.empty() || static_cast<uint64_t>(groups.size()) > availableIds)
    return 0;

  phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "transformation-phase", "materializeNoCOutputPublications",
      "materializeRequiredOutputGroup");
  for (const RequiredOutputGroup &group : groups) {
    materializeRequiredOutputGroup(group, communicationId, kind);
    communicationId = communicationId == std::numeric_limits<int64_t>::max()
                          ? -1
                          : communicationId + 1;
  }
  return static_cast<unsigned>(groups.size());
}

} // namespace wafer::compiler::detail
