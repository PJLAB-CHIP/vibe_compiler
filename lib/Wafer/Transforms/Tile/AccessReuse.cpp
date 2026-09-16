//===- AccessReuse.cpp - Materialize selected read reuse ------------------===//
#include "AccessReuse.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include <limits>

namespace wafer::compiler::detail {
namespace {

llvm::SmallVector<mlir::OpFoldResult>
indexAttrs(mlir::IRRewriter &rewriter, llvm::ArrayRef<int64_t> values) {
  llvm::SmallVector<mlir::OpFoldResult> result;
  for (auto value : values)
    result.push_back(rewriter.getIndexAttr(value));
  return result;
}

llvm::SmallVector<mlir::OpFoldResult> evaluateMap(mlir::IRRewriter &rewriter,
                                                  mlir::Location location,
                                                  mlir::AffineMap map,
                                                  mlir::ValueRange arguments) {
  llvm::SmallVector<mlir::OpFoldResult> result;
  for (unsigned axis = 0; axis < map.getNumResults(); ++axis) {
    auto expression =
        mlir::simplifyAffineExpr(map.getResult(axis), map.getNumDims(), 0);
    if (auto constant = mlir::dyn_cast<mlir::AffineConstantExpr>(expression))
      result.push_back(rewriter.getIndexAttr(constant.getValue()));
    else
      result.push_back(
          rewriter
              .create<mlir::affine::AffineApplyOp>(
                  location,
                  mlir::AffineMap::get(map.getNumDims(), 0, expression),
                  arguments)
              .getResult());
  }
  return result;
}

mlir::Value indexValue(mlir::IRRewriter &rewriter, mlir::Location location,
                       mlir::OpFoldResult value) {
  if (auto constant = mlir::getConstantIntValue(value))
    return rewriter.create<mlir::arith::ConstantIndexOp>(location, *constant);
  return mlir::cast<mlir::Value>(value);
}

llvm::SmallVector<mlir::OpFoldResult>
subtractOffsets(mlir::IRRewriter &rewriter, mlir::Location location,
                llvm::ArrayRef<mlir::OpFoldResult> lhs,
                llvm::ArrayRef<mlir::OpFoldResult> rhs) {
  llvm::SmallVector<mlir::OpFoldResult> result;
  for (auto [a, b] : llvm::zip_equal(lhs, rhs)) {
    auto x = mlir::getConstantIntValue(a), y = mlir::getConstantIntValue(b);
    if (x && y)
      result.push_back(rewriter.getIndexAttr(*x - *y));
    else if (a == b)
      result.push_back(rewriter.getIndexAttr(0));
    else {
      // Compose before creating the relative subview: independently evaluated
      // origins can denote the same outer IV. Interval analysis of two
      // separate values cannot recover that correlation after lowering.
      auto d0 = rewriter.getAffineDimExpr(0);
      auto d1 = rewriter.getAffineDimExpr(1);
      result.push_back(mlir::affine::makeComposedFoldedAffineApply(
          rewriter, location, mlir::AffineMap::get(2, 0, d0 - d1),
          llvm::ArrayRef<mlir::OpFoldResult>{a, b}));
    }
  }
  return result;
}

mlir::Value slice(mlir::IRRewriter &rewriter, mlir::Location location,
                  mlir::Value source,
                  llvm::ArrayRef<mlir::OpFoldResult> offsets,
                  llvm::ArrayRef<int64_t> sizes) {
  return rewriter
      .create<mlir::memref::SubViewOp>(
          location, source, offsets, indexAttrs(rewriter, sizes),
          indexAttrs(rewriter, llvm::SmallVector<int64_t>(sizes.size(), 1)))
      .getResult();
}

mlir::Value allocate(mlir::IRRewriter &rewriter, mlir::Location location,
                     llvm::ArrayRef<int64_t> sizes, mlir::Type element) {
  auto type = mlir::MemRefType::get(
      sizes, element, mlir::MemRefLayoutAttrInterface{},
      MemoryAttr::get(rewriter.getContext(), MemorySpace::SPM,
                      MemLayout::Tensor));
  return rewriter.create<mlir::memref::AllocOp>(location, type).getResult();
}

struct ResidentWindow {
  mlir::Value buffer;
  llvm::SmallVector<mlir::OpFoldResult> offsets;
  std::optional<StorageLoadOp> load;
};

ResidentWindow materializeWindow(mlir::IRRewriter &rewriter,
                                 analysis::ScopedReadAccess window,
                                 const ResidentWindow *outer = nullptr) {
  rewriter.setInsertionPoint(window.scope);
  auto location = window.scope.getLoc();
  ResidentWindow result;
  result.offsets =
      evaluateMap(rewriter, location, window.origin, window.parameters);
  result.buffer = allocate(
      rewriter, location, window.sizes,
      mlir::cast<mlir::MemRefType>(window.source.getType()).getElementType());
  auto offsets = outer ? subtractOffsets(rewriter, location, result.offsets,
                                         outer->offsets)
                       : result.offsets;
  auto source = slice(rewriter, location, outer ? outer->buffer : window.source,
                      offsets, window.sizes);
  if (outer)
    rewriter.create<MoveCopyIntoOp>(location, source, result.buffer);
  else
    result.load =
        rewriter.create<StorageLoadOp>(location, source, result.buffer);
  return result;
}

void replaceRead(mlir::IRRewriter &rewriter, const analysis::ReadAccess &read,
                 const ResidentWindow &resident) {
  auto load = read.load;
  rewriter.setInsertionPoint(load);
  llvm::SmallVector<mlir::Value> indices;
  for (auto loop : read.loops)
    indices.push_back(loop.loop.getInductionVar());
  auto offsets =
      evaluateMap(rewriter, load.getLoc(), read.coordinates, indices);
  auto relative =
      subtractOffsets(rewriter, load.getLoc(), offsets, resident.offsets);
  auto source = slice(rewriter, load.getLoc(), resident.buffer, relative,
                      read.sourceType.getShape());
  rewriter.create<MoveCopyIntoOp>(load.getLoc(), source, load.getDest());
  rewriter.eraseOp(load);
}

void materializeSliding(mlir::IRRewriter &rewriter,
                        analysis::SlidingReadAccess window) {
  auto load = window.access.load;
  auto loop = window.scope;
  auto location = load.getLoc();
  llvm::SmallVector<int64_t> shape(window.access.sourceType.getShape());
  auto overlapShape = shape;
  overlapShape[window.axis] -= window.shift;
  auto addedShape = shape;
  addedShape[window.axis] = window.shift;
  const unsigned count = window.access.loops.size();
  llvm::SmallVector<mlir::AffineExpr> initialDimensions;
  llvm::SmallVector<mlir::Value> outerIVs;
  for (unsigned i = 0; i + 1 < count; ++i) {
    initialDimensions.push_back(
        mlir::getAffineDimExpr(i, rewriter.getContext()));
    auto domain = window.access.loops[i];
    outerIVs.push_back(domain.loop.getInductionVar());
  }
  initialDimensions.push_back(mlir::getAffineConstantExpr(
      window.access.loops.back().lower, rewriter.getContext()));
  auto initialMap = window.access.coordinates.replaceDimsAndSymbols(
      initialDimensions, {}, count - 1, 0);
  rewriter.setInsertionPoint(loop);
  auto buffer = allocate(rewriter, location, shape,
                         window.access.sourceType.getElementType());
  auto scratch = allocate(rewriter, location, overlapShape,
                          window.access.sourceType.getElementType());
  auto firstOffsets = evaluateMap(rewriter, location, initialMap, outerIVs);
  auto first =
      slice(rewriter, location, window.access.base, firstOffsets, shape);
  rewriter.create<StorageLoadOp>(location, first, buffer);
  rewriter.setInsertionPointToStart(loop.getBody());
  auto later = rewriter.create<mlir::arith::CmpIOp>(
      location, mlir::arith::CmpIPredicate::ne, loop.getInductionVar(),
      loop.getLowerBound());
  auto update = rewriter.create<mlir::scf::IfOp>(location, later, false);
  rewriter.setInsertionPointToStart(update.thenBlock());
  llvm::SmallVector<int64_t> zero(shape.size(), 0), oldOffsets = zero;
  oldOffsets[window.axis] = window.shift;
  auto old = slice(rewriter, location, buffer, indexAttrs(rewriter, oldOffsets),
                   overlapShape);
  rewriter.create<MoveCopyIntoOp>(location, old, scratch);
  auto kept = slice(rewriter, location, buffer, indexAttrs(rewriter, zero),
                    overlapShape);
  rewriter.create<MoveCopyIntoOp>(location, scratch, kept);
  llvm::SmallVector<mlir::Value> indices(outerIVs);
  indices.push_back(loop.getInductionVar());
  auto sourceOffsets =
      evaluateMap(rewriter, location, window.access.coordinates, indices);
  sourceOffsets[window.axis] =
      rewriter
          .create<mlir::arith::AddIOp>(
              location,
              indexValue(rewriter, location, sourceOffsets[window.axis]),
              rewriter.create<mlir::arith::ConstantIndexOp>(
                  location, overlapShape[window.axis]))
          .getResult();
  auto added =
      slice(rewriter, location, window.access.base, sourceOffsets, addedShape);
  auto tailOffsets = zero;
  tailOffsets[window.axis] = overlapShape[window.axis];
  auto tail = slice(rewriter, location, buffer,
                    indexAttrs(rewriter, tailOffsets), addedShape);
  rewriter.create<StorageLoadOp>(location, added, tail);
  rewriter.setInsertionPoint(load);
  rewriter.create<MoveCopyIntoOp>(location, buffer, load.getDest());
  rewriter.eraseOp(load);
}

} // namespace

AccessReuseResult
materializeAccessReuse(mlir::ModuleOp module,
                       StructuredMaterializationRelations &relations,
                       const AccessReuseChoice &choice) {
  AccessReuseResult result;
  if (!module || mlir::failed(verifyTileModuleCollection(module)) ||
      mlir::failed(checkStructuredBufferRelationsCurrent(module, relations))) {
    result.failure = AccessReuseFailureKind::BrokenContract;
    result.detail = "access reuse requires verified current buffer owners";
    return result;
  }
  if (choice.actions.empty()) {
    result.failure = AccessReuseFailureKind::Unsupported;
    result.detail = "no selected access reuse";
    return result;
  }
  auto facts = analysis::analyzeAccessReuse(module, {}, 0);
  struct PreparedAction {
    const AccessReuseAction *choice;
    std::optional<analysis::ScopedReadAccess> outer, inner;
    std::optional<analysis::SlidingReadAccess> sliding;
  };
  llvm::SmallVector<PreparedAction, 4> prepared;
  llvm::DenseSet<mlir::Operation *> selected;
  for (const auto &action : choice.actions) {
    PreparedAction current{&action, {}, {}, {}};
    if (action.kind == AccessReuseKind::Peer &&
        (action.reads.size() < 2 ||
         !llvm::any_of(facts.peers, [&](const auto &group) {
           return group.reads.size() == action.reads.size() &&
                  llvm::all_of(llvm::zip_equal(group.reads, action.reads),
                               [](const auto &pair) {
                                 return std::get<0>(pair).load ==
                                        std::get<1>(pair);
                               });
         }))) {
      result.failure = AccessReuseFailureKind::Unsupported;
      result.detail =
          "selected reads have no current exact peer reuse relation";
      return result;
    }
    if (action.kind != AccessReuseKind::Peer) {
      auto read = llvm::find_if(facts.reads, [&](const auto &read) {
        return !action.reads.empty() && read.load == action.reads.front();
      });
      if (read == facts.reads.end() || !read->base || !action.scope ||
          !llvm::any_of(read->loops, [&](const auto &loop) {
            return loop.loop == action.scope;
          })) {
        result.failure = AccessReuseFailureKind::Unsupported;
        result.detail = "selected reuse source or scope is no longer available";
        return result;
      }
      auto scoped =
          analysis::queryScopedAccessReuse(facts, read->base, action.scope);
      if (!scoped.access || scoped.access->reads != action.reads) {
        result.failure =
            scoped.status == analysis::IndexRelationStatus::ResourceExhausted
                ? AccessReuseFailureKind::Indeterminate
                : AccessReuseFailureKind::Unsupported;
        result.detail = "selected scope has no exact current read window";
        return result;
      }
      current.outer = std::move(scoped.access);
      if (action.kind == AccessReuseKind::TwoLevel) {
        auto innerRead = llvm::find_if(facts.reads, [&](const auto &r) {
          return r.base == read->base &&
                 llvm::any_of(r.loops, [&](const auto &l) {
                   return l.loop == action.innerScope;
                 });
        });
        auto outerScope = action.scope;
        if (innerRead == facts.reads.end() ||
            !outerScope->isAncestor(action.innerScope)) {
          result.failure = AccessReuseFailureKind::Unsupported;
          result.detail = "two-level reuse requires two nested current scopes";
          return result;
        }
        auto inner = analysis::queryScopedAccessReuse(facts, read->base,
                                                      action.innerScope);
        if (!inner.access) {
          result.failure =
              inner.status == analysis::IndexRelationStatus::ResourceExhausted
                  ? AccessReuseFailureKind::Indeterminate
                  : AccessReuseFailureKind::Unsupported;
          result.detail = "inner reuse scope has no exact current window";
          return result;
        }
        current.inner = std::move(inner.access);
      } else if (action.kind == AccessReuseKind::Sliding) {
        auto sliding = analysis::querySlidingAccessReuse(facts, read->load);
        if (!sliding.access) {
          result.failure =
              sliding.status == analysis::IndexRelationStatus::ResourceExhausted
                  ? AccessReuseFailureKind::Indeterminate
                  : AccessReuseFailureKind::Unsupported;
          result.detail =
              "selected read is not a fixed single-axis sliding window";
          return result;
        }
        current.sliding = std::move(sliding.access);
      }
    }
    for (auto read : action.reads)
      if (!selected.insert(read).second) {
        result.failure = AccessReuseFailureKind::BrokenContract;
        result.detail = "access reuse choices overlap the same current read";
        return result;
      }
    prepared.push_back(std::move(current));
  }
  int64_t next = 0;
  module.walk([&](mlir::Operation *operation) {
    for (auto attribute : operation->getAttrs())
      attribute.getValue().walk([&](DTEMessageAttr message) {
        next = std::max(next, message.getCommunicationId());
      });
  });
  if (choice.actions.size() >=
      uint64_t(std::numeric_limits<int64_t>::max() - next)) {
    result.failure = AccessReuseFailureKind::Unsupported;
    result.detail = "access reuse exhausts message identities";
    return result;
  }
  mlir::IRRewriter rewriter(module.getContext());
  llvm::DenseSet<mlir::Operation *> residentLoads;
  for (const auto &item : prepared) {
    const auto &action = *item.choice;
    if (item.sliding) {
      materializeSliding(rewriter, *item.sliding);
      ++result.slidingWindows;
      ++result.replacedLoads;
      continue;
    }
    if (item.outer) {
      auto outer = materializeWindow(rewriter, *item.outer);
      if (outer.load && action.shareWindow)
        residentLoads.insert(outer.load->getOperation());
      std::optional<ResidentWindow> inner;
      if (item.inner) {
        inner = materializeWindow(rewriter, *item.inner, &outer);
        ++result.twoLevelWindows;
      }
      ++result.residentWindows;
      for (auto load : item.outer->reads) {
        auto read = llvm::find_if(
            facts.reads, [&](const auto &read) { return read.load == load; });
        const bool useInner =
            item.inner && llvm::is_contained(item.inner->reads, load);
        replaceRead(rewriter, *read, useInner ? *inner : outer);
        ++result.replacedLoads;
      }
      continue;
    }
    auto donor = action.reads.front();
    auto physical = computeWaferPhysicalTensorInfo(
        mlir::cast<mlir::MemRefType>(donor.getDest().getType()));
    ++next;
    mlir::Operation *lastSend = donor;
    for (auto [index, endpoint] :
         llvm::enumerate(llvm::ArrayRef(action.reads).drop_front())) {
      StorageLoadOp receiver = endpoint;
      auto message = DTEMessageAttr::get(module.getContext(), next, 0, index);
      rewriter.setInsertionPointAfter(lastSend);
      lastSend = rewriter.create<CommPeerSendOp>(
          donor.getLoc(), donor.getDest(),
          receiver->getParentOfType<TileModuleOp>().getTileId(),
          physical->physicalBytes, message);
      rewriter.setInsertionPoint(receiver);
      rewriter.create<CommPeerRecvOp>(
          receiver.getLoc(), receiver.getDest(),
          donor->getParentOfType<TileModuleOp>().getTileId(),
          physical->physicalBytes, message);
      rewriter.eraseOp(receiver);
      ++result.movement.peerSends;
      ++result.movement.peerReceives;
    }
  }
  if (!residentLoads.empty()) {
    // The combined choice already requested peer supply for its resident
    // windows. Re-query the actual created loads, without predicting SSA or
    // recovering them by name/ordinal. No new strategy is selected here.
    auto residentFacts = analysis::analyzeAccessReuse(module, {}, 0);
    for (const auto &group : residentFacts.peers) {
      if (!llvm::all_of(group.reads, [&](const auto &read) {
            return residentLoads.contains(read.load);
          }))
        continue;
      if (next == std::numeric_limits<int64_t>::max()) {
        result.failure = AccessReuseFailureKind::Unsupported;
        result.detail = "access reuse exhausts message identities";
        return result;
      }
      auto donor = group.reads.front().load;
      mlir::Operation *lastSend = donor;
      ++next;
      for (auto [index, access] :
           llvm::enumerate(llvm::ArrayRef(group.reads).drop_front())) {
        auto receiver = access.load;
        auto message = DTEMessageAttr::get(module.getContext(), next, 0, index);
        rewriter.setInsertionPointAfter(lastSend);
        lastSend = rewriter.create<CommPeerSendOp>(
            donor.getLoc(), donor.getDest(),
            receiver->getParentOfType<TileModuleOp>().getTileId(),
            access.physicalBytes, message);
        rewriter.setInsertionPoint(receiver);
        rewriter.create<CommPeerRecvOp>(
            receiver.getLoc(), receiver.getDest(),
            donor->getParentOfType<TileModuleOp>().getTileId(),
            access.physicalBytes, message);
        rewriter.eraseOp(receiver);
        ++result.movement.peerSends;
        ++result.movement.peerReceives;
      }
    }
  }
  rebuildCurrentBufferOwnerRelations(module, relations);
  if (mlir::failed(verifyPhysicalTileDataflow(module)) ||
      mlir::failed(checkStructuredBufferRelationsCurrent(module, relations))) {
    result.failure = AccessReuseFailureKind::CompilerFailure;
    result.detail = "access reuse produced invalid physical dataflow";
  }
  return result;
}

} // namespace wafer::compiler::detail
