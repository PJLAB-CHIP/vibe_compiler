//===- OnlineAttentionStateOrientation.cpp - Private state reindexing -----===//

#include "OnlineAttentionStateOrientation.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/CompileTiming.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

namespace wafer::compiler::detail {
namespace {

// This is a closed online-state rewrite, not a graph-wide transpose search.
// Every value here exists in the same current IR epoch. No buffers or layouts
// are predicted; the subsequent layout and memory stages own those choices.
struct StateComponent {
  llvm::SetVector<mlir::Value> values;
  llvm::SetVector<mlir::Operation *> operations;
  llvm::SmallVector<mlir::OpOperand *> boundaries;
  llvm::SmallVector<mlir::scf::ForOp> loops;
  llvm::SmallVector<int64_t> permutation;
  llvm::DenseSet<mlir::Value> protectedValues;
  llvm::DenseMap<mlir::Operation *, mlir::RankedTensorType> privateInits;

  bool add(mlir::Value value) {
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
    if (!type || !type.hasStaticShape() ||
        type.getRank() != static_cast<int64_t>(permutation.size()) ||
        protectedValues.contains(value))
      return false;
    values.insert(value);
    return true;
  }

  bool addLoop(mlir::scf::ForOp loop, unsigned index) {
    operations.insert(loop);
    if (!llvm::is_contained(loops, loop))
      loops.push_back(loop);
    auto yield =
        mlir::cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
    return index < loop.getNumRegionIterArgs() &&
           add(loop.getInitArgs()[index]) &&
           add(loop.getRegionIterArgs()[index]) && add(loop.getResult(index)) &&
           add(yield.getOperand(index));
  }

  bool addBranch(mlir::scf::IfOp branch, unsigned index) {
    if (branch.getElseRegion().empty())
      return false;
    operations.insert(branch);
    return add(branch.getResult(index)) &&
           add(branch.thenYield().getOperand(index)) &&
           add(branch.elseYield().getOperand(index));
  }

  bool addOnline(LinalgExtOnlineAttentionOp online) {
    operations.insert(online);
    auto roles = online.getIterationRoles();
    if (mlir::failed(roles))
      return false;
    // All steps of one state, including KV tails, must agree on the same
    // accumulator coordinate order. Never infer it from equal dimensions.
    auto map = online.getAccumulatorMap();
    llvm::SmallVector<mlir::AffineExpr> expected;
    for (auto dimensions : {llvm::ArrayRef<unsigned>(roles->batch),
                            llvm::ArrayRef<unsigned>(roles->valueOutput),
                            llvm::ArrayRef<unsigned>(roles->query)})
      for (unsigned dimension : dimensions)
        expected.push_back(
            mlir::getAffineDimExpr(dimension, online.getContext()));
    if (expected.size() != permutation.size())
      return false;
    for (auto [index, axis] : llvm::enumerate(permutation))
      if (map.getResult(axis) != expected[index])
        return false;
    llvm::SmallVector<mlir::AffineExpr> rows;
    for (auto expression : expected)
      if (!llvm::is_contained(
              roles->valueOutput,
              mlir::cast<mlir::AffineDimExpr>(expression).getPosition()))
        rows.push_back(expression);
    if (llvm::ArrayRef<mlir::AffineExpr>(rows) !=
        online.getMaximumMap().getResults())
      return false;
    return add(online.getAccumulator()) && add(online.getUpdatedAccumulator());
  }

  bool addGeneric(mlir::linalg::GenericOp generic, unsigned operand) {
    if (!generic.hasPureTensorSemantics() || generic.getNumDpsInits() != 1 ||
        llvm::any_of(generic.getIteratorTypesArray(), [](auto kind) {
          return kind != mlir::utils::IteratorType::parallel;
        }))
      return false;
    auto maps = generic.getIndexingMapsArray();
    if (maps[operand] != maps.back() || !maps.back().isPermutation())
      return false;
    operations.insert(generic);
    // Merge inputs with the output map are the same state coordinate space.
    // Row coefficients retain their original compact maps and values.
    for (auto [index, value] : llvm::enumerate(generic->getOperands())) {
      if (index == static_cast<size_t>(generic.getNumDpsInputs()) &&
          generic.getBody()->getArgument(index).use_empty() &&
          !llvm::any_of(llvm::enumerate(generic.getDpsInputs()),
                        [&](auto input) {
                          return input.value() == value &&
                                 maps[input.index()] == maps.back();
                        })) {
        auto type = mlir::cast<mlir::RankedTensorType>(value.getType());
        if (!type.hasStaticShape() ||
            type.getRank() != static_cast<int64_t>(permutation.size()))
          return false;
        privateInits.try_emplace(generic, type);
      } else if (maps[index] == maps.back() && !add(value)) {
        return false;
      }
    }
    return add(generic.getResult(0));
  }

  bool addExtract(mlir::tensor::ExtractSliceOp slice) {
    operations.insert(slice);
    return add(slice.getSource()) && add(slice.getResult());
  }

  bool addInsert(mlir::tensor::InsertSliceOp slice) {
    operations.insert(slice);
    return add(slice.getSource()) && add(slice.getDest()) &&
           add(slice.getResult());
  }

  bool close(mlir::Value initial) {
    if (!add(initial))
      return false;
    for (unsigned index = 0; index < values.size(); ++index) {
      mlir::Value value = values[index];
      if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
        auto loop = mlir::dyn_cast<mlir::scf::ForOp>(
            argument.getOwner()->getParentOp());
        if (!loop || argument.getArgNumber() == 0 ||
            !addLoop(loop, argument.getArgNumber() - 1))
          return false;
      } else {
        mlir::Operation *producer = value.getDefiningOp();
        unsigned result = mlir::cast<mlir::OpResult>(value).getResultNumber();
        operations.insert(producer);
        if (mlir::isa<mlir::tensor::EmptyOp>(producer)) {
          // Static shape was checked by add().
        } else if (auto fill = mlir::dyn_cast<mlir::linalg::FillOp>(producer)) {
          if (!add(fill.getOutputs()[result]))
            return false;
        } else if (auto online =
                       mlir::dyn_cast<LinalgExtOnlineAttentionOp>(producer)) {
          if (result != 0 || !addOnline(online))
            return false;
        } else if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(producer)) {
          if (!addLoop(loop, result))
            return false;
        } else if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(producer)) {
          if (!addBranch(branch, result))
            return false;
        } else if (auto generic =
                       mlir::dyn_cast<mlir::linalg::GenericOp>(producer)) {
          if (result != 0 || !addGeneric(generic, generic.getNumDpsInputs()))
            return false;
        } else if (auto slice =
                       mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(producer)) {
          if (!addExtract(slice))
            return false;
        } else if (auto slice =
                       mlir::dyn_cast<mlir::tensor::InsertSliceOp>(producer)) {
          if (!addInsert(slice))
            return false;
        } else {
          return false;
        }
      }
      for (mlir::OpOperand &use : value.getUses()) {
        mlir::Operation *consumer = use.getOwner();
        if (auto online =
                mlir::dyn_cast<LinalgExtOnlineAttentionOp>(consumer)) {
          if (&use != online.getDpsInitOperand(0) || !addOnline(online))
            return false;
        } else if (auto fill = mlir::dyn_cast<mlir::linalg::FillOp>(consumer)) {
          operations.insert(fill);
          if (use.getOperandNumber() != fill.getNumDpsInputs() ||
              !add(fill.getResult(0)))
            return false;
        } else if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(consumer)) {
          if (use.getOperandNumber() < 3 ||
              !addLoop(loop, use.getOperandNumber() - 3))
            return false;
        } else if (auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(consumer)) {
          if (auto loop =
                  mlir::dyn_cast<mlir::scf::ForOp>(yield->getParentOp())) {
            if (!addLoop(loop, use.getOperandNumber()))
              return false;
          } else if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(
                         yield->getParentOp())) {
            if (!addBranch(branch, use.getOperandNumber()))
              return false;
          } else {
            return false;
          }
        } else if (auto generic =
                       mlir::dyn_cast<mlir::linalg::GenericOp>(consumer)) {
          if (!addGeneric(generic, use.getOperandNumber()))
            return false;
        } else if (auto slice =
                       mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(consumer)) {
          if (!addExtract(slice))
            return false;
        } else if (auto slice =
                       mlir::dyn_cast<mlir::tensor::InsertSliceOp>(consumer)) {
          // Publication does not make its larger destination part of the
          // private state. A recurrence insertion will independently be
          // reached through its destination or its loop-carried result.
          if (&use == &slice->getOpOperand(0))
            boundaries.push_back(&use);
          else if (!addInsert(slice))
            return false;
        } else {
          // Restore only outside the KV recurrence. Extra observations inside
          // it would introduce a per-step transpose, so reject the choice.
          boundaries.push_back(&use);
        }
      }
    }
    llvm::erase_if(boundaries, [&](mlir::OpOperand *use) {
      return mlir::isa<mlir::tensor::InsertSliceOp>(use->getOwner()) &&
             operations.contains(use->getOwner());
    });
    return llvm::all_of(boundaries, [&](mlir::OpOperand *use) {
      return llvm::none_of(loops, [&](mlir::scf::ForOp loop) {
        return loop->isAncestor(use->getOwner());
      });
    });
  }

  mlir::AffineMap permute(mlir::AffineMap map) const {
    llvm::SmallVector<mlir::AffineExpr> results;
    for (int64_t axis : permutation)
      results.push_back(map.getResult(axis));
    return mlir::AffineMap::get(map.getNumDims(), map.getNumSymbols(), results,
                                map.getContext());
  }

  void apply(mlir::IRRewriter &rewriter) {
    llvm::SmallVector<int64_t> inverse(permutation.size());
    for (auto [index, axis] : llvm::enumerate(permutation)) {
      inverse[axis] = index;
    }
    // Preflight is complete. Update the whole closed component in one epoch;
    // its transient type mismatches are not visible to another analysis.
    for (mlir::Operation *operation : operations) {
      mlir::Value privateInit;
      if (auto found = privateInits.find(operation);
          found != privateInits.end()) {
        rewriter.setInsertionPoint(operation);
        llvm::SmallVector<int64_t> newShape;
        for (int64_t axis : permutation)
          newShape.push_back(found->second.getDimSize(axis));
        privateInit = rewriter.create<mlir::tensor::EmptyOp>(
            operation->getLoc(), newShape, found->second.getElementType());
      }
      rewriter.modifyOpInPlace(operation, [&] {
        auto change = [&](mlir::Value value) {
          if (values.contains(value)) {
            auto type = mlir::cast<mlir::RankedTensorType>(value.getType());
            llvm::SmallVector<int64_t> newShape;
            for (int64_t axis : permutation)
              newShape.push_back(type.getDimSize(axis));
            value.setType(type.clone(newShape));
          }
        };
        for (mlir::Value result : operation->getResults())
          change(result);
        if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(operation))
          for (mlir::Value argument : loop.getRegionIterArgs())
            change(argument);
        if (auto online =
                mlir::dyn_cast<LinalgExtOnlineAttentionOp>(operation)) {
          auto maps = online.getIndexingMapsArray();
          // The interface appends empty maps for position operands; these
          // synthetic interface entries are not part of indexing_maps.
          maps.resize(online.getIndexingMaps().size());
          maps[online.getAccumulatorMapIndex()] =
              permute(online.getAccumulatorMap());
          online.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(maps));
        } else if (auto generic =
                       mlir::dyn_cast<mlir::linalg::GenericOp>(operation)) {
          auto maps = generic.getIndexingMapsArray();
          if (privateInit)
            generic.getDpsInitOperand(0)->set(privateInit);
          for (auto [index, operand] : llvm::enumerate(generic->getOperands()))
            if (values.contains(operand) || operand == privateInit)
              maps[index] = permute(maps[index]);
          generic.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(maps));
        } else if (auto slice = mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(
                       operation)) {
          permuteSlice(slice);
        } else if (auto slice =
                       mlir::dyn_cast<mlir::tensor::InsertSliceOp>(operation)) {
          permuteSlice(slice);
        }
      });
    }
    llvm::DenseMap<mlir::Value, mlir::Value> restored;
    for (mlir::OpOperand *use : boundaries) {
      mlir::Value value = use->get();
      if (auto found = restored.find(value); found != restored.end()) {
        rewriter.modifyOpInPlace(use->getOwner(),
                                 [&] { use->set(found->second); });
        continue;
      }
      rewriter.setInsertionPointAfterValue(value);
      auto type = mlir::cast<mlir::RankedTensorType>(value.getType());
      llvm::SmallVector<int64_t> shape;
      for (int64_t axis : inverse)
        shape.push_back(type.getDimSize(axis));
      auto output = rewriter.create<mlir::tensor::EmptyOp>(
          use->getOwner()->getLoc(), shape, type.getElementType());
      auto transpose = rewriter.create<mlir::linalg::TransposeOp>(
          use->getOwner()->getLoc(), value, output, inverse);
      restored[value] = transpose->getResult(0);
      rewriter.modifyOpInPlace(use->getOwner(),
                               [&] { use->set(transpose->getResult(0)); });
    }
  }

  template <typename Slice> void permuteSlice(Slice slice) {
    auto reorder = [&](llvm::ArrayRef<mlir::OpFoldResult> mixed,
                       mlir::MutableOperandRange dynamic, auto setStatic) {
      llvm::SmallVector<mlir::OpFoldResult> ordered;
      for (int64_t axis : permutation)
        ordered.push_back(mixed[axis]);
      llvm::SmallVector<mlir::Value> dynamicValues;
      llvm::SmallVector<int64_t> staticValues;
      mlir::dispatchIndexOpFoldResults(ordered, dynamicValues, staticValues);
      dynamic.assign(dynamicValues);
      setStatic(staticValues);
    };
    reorder(slice.getMixedOffsets(), slice.getOffsetsMutable(),
            [&](llvm::ArrayRef<int64_t> values) {
              slice.setStaticOffsets(values);
            });
    reorder(
        slice.getMixedSizes(), slice.getSizesMutable(),
        [&](llvm::ArrayRef<int64_t> values) { slice.setStaticSizes(values); });
    reorder(slice.getMixedStrides(), slice.getStridesMutable(),
            [&](llvm::ArrayRef<int64_t> values) {
              slice.setStaticStrides(values);
            });
  }
};
} // namespace

mlir::LogicalResult orientOnlineAttentionAccumulators(
    mlir::ModuleOp module, const StructuredMaterializationRelations &relations,
    mlir::IRRewriter &rewriter) {
  llvm::SmallVector<LinalgExtOnlineAttentionOp> onlineOps;
  module.walk([&](LinalgExtOnlineAttentionOp op) { onlineOps.push_back(op); });
  for (auto online : onlineOps) {
    auto roles = online.getIterationRoles();
    auto extents = online.getStaticLoopRanges();
    if (mlir::failed(roles) || llvm::none_of(roles->query, [&](unsigned axis) {
          return extents[axis] >= 64;
        }))
      continue;
    StateComponent component;
    auto map = online.getAccumulatorMap();
    for (auto dimensions : {llvm::ArrayRef<unsigned>(roles->batch),
                            llvm::ArrayRef<unsigned>(roles->valueOutput),
                            llvm::ArrayRef<unsigned>(roles->query)})
      for (unsigned axis : dimensions) {
        auto expression = mlir::getAffineDimExpr(axis, online.getContext());
        auto found = llvm::find(map.getResults(), expression);
        if (found == map.getResults().end())
          return mlir::failure();
        component.permutation.push_back(found - map.getResults().begin());
      }
    if (llvm::all_of(llvm::enumerate(component.permutation), [](auto item) {
          return item.index() == static_cast<unsigned>(item.value());
        }))
      continue;
    for (const auto &output : relations.structuralOutputs)
      component.protectedValues.insert(output.endpoint);
    for (const auto &boundary : relations.boundaryRelations) {
      component.protectedValues.insert(boundary.sourceEndpoint);
      component.protectedValues.insert(boundary.destinationEndpoint);
    }
    if (!component.close(online.getAccumulator()))
      continue;
    component.apply(rewriter);
    support::addCompileCounter("online-attention",
                               "transposed-state-components", 1);
  }
  return mlir::verify(module);
}
} // namespace wafer::compiler::detail
