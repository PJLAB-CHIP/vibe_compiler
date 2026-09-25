//===- TensorAssemblyMaterialization.cpp - Selected tensor demand --------===//

#include "TensorAssemblyMaterialization.h"
#include "Wafer/Transforms/Linalg/StructuredTiling.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/MathExtras.h"

#include <functional>
#include <map>
#include <vector>

namespace wafer::compiler::detail {

namespace {

struct BlockProofs {
  llvm::SmallVector<analysis::TensorSubsetBlockResult, 4> sources;
  std::optional<unsigned> splitAxis;
};

// The cache contains proofs for static shapes and all admissible origins.
// It has no SSA output, allocation, instruction or resource inventory.
class SubsetMaterializer {
public:
  SubsetMaterializer(mlir::RewriterBase &rewriter,
                     mlir::tensor::ExtractSliceOp read,
                     const analysis::TensorSubsetDemand &demand,
                     analysis::IndexRelationWork &work)
      : rewriter(rewriter), read(read), demand(demand), work(work),
        location(read.getLoc()) {}

  bool prepare(llvm::ArrayRef<int64_t> shape) {
    if (!work.charge())
      return false;
    std::vector<int64_t> key(shape.begin(), shape.end());
    if (proofs.count(key))
      return true;
    BlockProofs block;
    llvm::SmallBitVector blocking(shape.size());
    bool unconditional = false;
    bool unit = llvm::all_of(shape, [](int64_t size) { return size == 1; });
    for (const auto &source : demand.sources) {
      auto proof =
          analysis::queryTensorSubsetBlock(demand, source, shape, work);
      using Status = analysis::TensorSubsetBlockStatus;
      if (proof.status != Status::Copy && proof.status != Status::Subdivide) {
        status = proof.status == Status::ResourceExhausted
                     ? TensorSubsetMaterializationStatus::ResourceExhausted
                 : proof.status == Status::BrokenContract
                     ? TensorSubsetMaterializationStatus::BrokenContract
                     : TensorSubsetMaterializationStatus::Unsupported;
        detail = proof.detail;
        return false;
      }
      if (unit && proof.status != Status::Copy) {
        status = TensorSubsetMaterializationStatus::BrokenContract;
        detail = "exact subset relation has no ordered unit-block image";
        return false;
      }
      for (unsigned axis : proof.blockingAxes)
        blocking.set(axis);
      unconditional |= proof.block && proof.block->guards.empty();
      block.sources.push_back(std::move(proof));
      if (unconditional)
        break;
    }
    bool covered = unit || unconditional;
    // Interior origins range over every point of a smaller template. While a
    // nonunit axis still blocks a source proof, continue its constructive
    // subdivision instead of repeatedly projecting all source complements.
    // Check the initial tile and the shapes with no such blocking axis; this
    // retains aligned whole tiles and complete Tensor/NCx rows.
    bool checkCoverage = true;
    for (unsigned axis = 0; axis < shape.size(); ++axis)
      if (blocking.test(axis) && shape[axis] > 1 &&
          shape[axis] < demand.shape[axis])
        checkCoverage = false;
    if (!covered && checkCoverage &&
        llvm::all_of(block.sources,
                     [](const auto &source) { return bool(source.block); })) {
      llvm::SmallVector<int64_t> point(shape.size(), 1);
      if (!prepare(point))
        return false;
      const auto &pointProofs =
          proofs.at(std::vector<int64_t>(point.begin(), point.end()));
      // Unit blocks inherit the exact point partition. When every current
      // block guard is already required by that source's unit guard, the
      // larger block has the same complete partition. Independent batch and
      // contiguous channel axes need no repeated Boolean projection.
      covered = pointProofs.sources.size() == block.sources.size();
      if (covered)
        for (auto [whole, single] :
             llvm::zip_equal(block.sources, pointProofs.sources)) {
          covered &=
              single.block &&
              llvm::all_of(whole.block->guards, [&](auto guard) {
                return llvm::any_of(single.block->guards, [&](auto pointGuard) {
                  return guard.set == pointGuard.set &&
                         guard.complement == pointGuard.complement;
                });
              });
        }
    }
    if (!covered && checkCoverage &&
        llvm::all_of(block.sources, [](const auto &source) {
          return source.block.has_value();
        })) {
      llvm::SmallVector<analysis::IndexDomainCondition> domain{
          {demand.domain, false}};
      domain.append(demand.scopeConditions);
      llvm::SmallVector<mlir::AffineExpr> originBounds;
      llvm::SmallVector<bool> inequalities;
      for (unsigned axis = 0; axis < shape.size(); ++axis) {
        originBounds.push_back(
            rewriter.getAffineConstantExpr(demand.shape[axis] - shape[axis]) -
            rewriter.getAffineDimExpr(demand.parameters.size() + axis));
        inequalities.push_back(false);
      }
      if (!originBounds.empty())
        domain.push_back(
            {mlir::IntegerSet::get(demand.parameters.size() + shape.size(), 0,
                                   originBounds, inequalities),
             false});
      llvm::SmallVector<llvm::ArrayRef<analysis::IndexDomainCondition>>
          alternatives;
      for (const auto &source : block.sources)
        alternatives.push_back(source.block->guards);
      auto closure =
          analysis::proveIndexDomainCovered(domain, alternatives, work);
      if (closure.status == analysis::IndexRelationStatus::ResourceExhausted ||
          closure.status == analysis::IndexRelationStatus::Invalid) {
        status =
            closure.status == analysis::IndexRelationStatus::ResourceExhausted
                ? TensorSubsetMaterializationStatus::ResourceExhausted
                : TensorSubsetMaterializationStatus::BrokenContract;
        detail = closure.reason;
        return false;
      }
      covered = closure.isProvenTrue();
    }
    if (!covered) {
      for (unsigned axis = 0; axis < shape.size(); ++axis)
        if (shape[axis] > 1 && blocking.test(axis)) {
          block.splitAxis = axis;
          break;
        }
      if (!block.splitAxis)
        for (unsigned axis = 0; axis < shape.size(); ++axis)
          if (shape[axis] > 1) {
            block.splitAxis = axis;
            break;
          }
      auto child = llvm::to_vector(shape);
      unsigned axis = *block.splitAxis;
      child[axis] /= 2;
      if (!prepare(child))
        return false;
      if (shape[axis] % 2) {
        child[axis] = 1;
        if (!prepare(child))
          return false;
      }
    }
    proofs.emplace(std::move(key), std::move(block));
    return true;
  }

  // Reserve an upper bound on emitted operations before the first mutation.
  // Every recursive occurrence is charged, including repeated odd-tail
  // templates. Affine expression trees are represented by one affine.apply;
  // their analysis work has already been charged by the block query.
  bool reserveEmission(llvm::ArrayRef<int64_t> shape) {
    if (!work.charge(64 + 8 * shape.size()))
      return false;
    const auto &block =
        proofs.at(std::vector<int64_t>(shape.begin(), shape.end()));
    for (const auto &source : block.sources) {
      if (!source.block)
        continue;
      if (!work.charge(16 + 8 * source.block->sourceSizes.size()))
        return false;
      for (auto guard : source.block->guards)
        if (!work.charge(4 + 3 * guard.set.getNumConstraints()))
          return false;
    }
    if (!block.splitAxis)
      return true;
    unsigned axis = *block.splitAxis;
    auto child = llvm::to_vector(shape);
    child[axis] /= 2;
    if (!reserveEmission(child))
      return false;
    if (shape[axis] % 2) {
      child[axis] = 1;
      if (!reserveEmission(child))
        return false;
    }
    return true;
  }

  mlir::OpFoldResult index(mlir::AffineExpr expression,
                           llvm::ArrayRef<mlir::OpFoldResult> origin) {
    llvm::SmallVector<mlir::OpFoldResult> operands;
    for (const auto &parameter : demand.parameters)
      operands.push_back(parameter.induction);
    operands.append(origin.begin(), origin.end());
    return mlir::affine::makeComposedFoldedAffineApply(
        rewriter, location,
        mlir::AffineMap::get(operands.size(), 0, expression,
                             rewriter.getContext()),
        operands);
  }

  mlir::Value value(mlir::OpFoldResult index) {
    if (auto dynamic = mlir::dyn_cast<mlir::Value>(index))
      return dynamic;
    return rewriter.create<mlir::arith::ConstantIndexOp>(
        location,
        mlir::cast<mlir::IntegerAttr>(mlir::cast<mlir::Attribute>(index))
            .getInt());
  }

  mlir::Value guard(const analysis::TensorSubsetBlock &block,
                    llvm::ArrayRef<mlir::OpFoldResult> origin) {
    mlir::Value result;
    auto combine = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
      if (!a)
        return b;
      if (auto constant = mlir::getConstantIntValue(a))
        return *constant ? b : a;
      if (auto constant = mlir::getConstantIntValue(b))
        return *constant ? a : b;
      return rewriter.createOrFold<mlir::arith::AndIOp>(location, a, b);
    };
    auto zero = rewriter.create<mlir::arith::ConstantIndexOp>(location, 0);
    for (auto condition : block.guards) {
      mlir::Value conjunction;
      for (auto [expression, equality] : llvm::zip_equal(
               condition.set.getConstraints(), condition.set.getEqFlags())) {
        auto coordinate = value(index(expression, origin));
        auto predicate = equality ? mlir::arith::CmpIPredicate::eq
                                  : mlir::arith::CmpIPredicate::sge;
        auto test = rewriter.createOrFold<mlir::arith::CmpIOp>(
            location, predicate, coordinate, zero);
        conjunction = combine(conjunction, test);
      }
      if (!conjunction)
        conjunction =
            rewriter.create<mlir::arith::ConstantIntOp>(location, 1, 1);
      if (condition.complement) {
        auto one = rewriter.create<mlir::arith::ConstantIntOp>(location, 1, 1);
        conjunction = rewriter.createOrFold<mlir::arith::XOrIOp>(
            location, conjunction, one);
      }
      result = combine(result, conjunction);
    }
    return result;
  }

  mlir::Value copy(unsigned source, const analysis::TensorSubsetBlock &block,
                   llvm::ArrayRef<int64_t> shape,
                   llvm::ArrayRef<mlir::OpFoldResult> origin,
                   mlir::Value destination) {
    llvm::SmallVector<mlir::OpFoldResult> offsets, sizes, strides;
    for (auto expression : block.sourceOffsets.getResults())
      offsets.push_back(index(expression, origin));
    for (int64_t extent : block.sourceSizes) {
      sizes.push_back(rewriter.getIndexAttr(extent));
      strides.push_back(rewriter.getIndexAttr(1));
    }
    auto slice = rewriter.create<mlir::tensor::ExtractSliceOp>(
        location, demand.sources[source].source, offsets, sizes, strides);
    result.sourceReads.push_back(slice);
    auto type = mlir::RankedTensorType::get(
        shape, read.getType().getElementType(), read.getType().getEncoding());
    auto value = reshapeStaticTensorTile(rewriter, location, slice, type);
    if (!destination)
      return value;
    sizes.clear();
    strides.clear();
    for (int64_t extent : shape) {
      sizes.push_back(rewriter.getIndexAttr(extent));
      strides.push_back(rewriter.getIndexAttr(1));
    }
    return rewriter.create<mlir::tensor::InsertSliceOp>(
        location, value, destination, origin, sizes, strides);
  }

  mlir::Value emit(llvm::ArrayRef<int64_t> shape,
                   llvm::ArrayRef<mlir::OpFoldResult> origin,
                   mlir::Value destination) {
    ++templates;
    const auto &block =
        proofs.at(std::vector<int64_t>(shape.begin(), shape.end()));
    std::function<mlir::Value(unsigned)> branch =
        [&](unsigned source) -> mlir::Value {
      if (source < block.sources.size()) {
        const auto &proof = block.sources[source];
        if (!proof.block)
          return branch(source + 1);
        // Unit blocks inherit the exact DAG partition; larger closed blocks
        // have a bounded proof that their guards cover the origin domain.
        // After excluding earlier guards the final guard is therefore true.
        bool lastSource =
            !block.splitAxis && source + 1 == block.sources.size();
        if (proof.block->guards.empty())
          return copy(source, *proof.block, shape, origin, destination);
        auto condition = guard(*proof.block, origin);
        if (auto constant = mlir::getConstantIntValue(condition)) {
          // Coverage proves this final edge unreachable when its guard is
          // false. Keep the destination without emitting an invalid static
          // source slice into that unreachable edge.
          if (!*constant && lastSource)
            return destination;
          return *constant
                     ? copy(source, *proof.block, shape, origin, destination)
                     : branch(source + 1);
        }
        if (lastSource)
          return copy(source, *proof.block, shape, origin, destination);
        auto conditional = rewriter.create<mlir::scf::IfOp>(
            location, destination.getType(), condition, true);
        {
          mlir::OpBuilder::InsertionGuard restore(rewriter);
          rewriter.setInsertionPointToStart(
              &conditional.getThenRegion().front());
          auto written = copy(source, *proof.block, shape, origin, destination);
          rewriter.create<mlir::scf::YieldOp>(location, written);
          rewriter.setInsertionPointToStart(
              &conditional.getElseRegion().front());
          auto remainder = branch(source + 1);
          rewriter.create<mlir::scf::YieldOp>(location, remainder);
        }
        return conditional.getResult(0);
      }
      unsigned axis = *block.splitAxis;
      int64_t half = shape[axis] / 2;
      auto child = llvm::to_vector(shape);
      child[axis] = half;
      auto zero = rewriter.create<mlir::arith::ConstantIndexOp>(location, 0);
      auto two = rewriter.create<mlir::arith::ConstantIndexOp>(location, 2);
      auto one = rewriter.create<mlir::arith::ConstantIndexOp>(location, 1);
      auto loop = rewriter.create<mlir::scf::ForOp>(
          location, zero, two, one, mlir::ValueRange{destination},
          [&](mlir::OpBuilder &, mlir::Location, mlir::Value induction,
              mlir::ValueRange carried) {
            auto next = llvm::to_vector(origin);
            auto originExpr = rewriter.getAffineDimExpr(0) +
                              rewriter.getAffineDimExpr(1) * half;
            next[axis] = mlir::affine::makeComposedFoldedAffineApply(
                rewriter, location, originExpr, {origin[axis], induction});
            auto written = emit(child, next, carried.front());
            rewriter.create<mlir::scf::YieldOp>(location, written);
          });
      mlir::Value written = loop.getResult(0);
      if (shape[axis] % 2) {
        child[axis] = 1;
        auto next = llvm::to_vector(origin);
        next[axis] = mlir::affine::makeComposedFoldedAffineApply(
            rewriter, location, rewriter.getAffineDimExpr(0) + 2 * half,
            {origin[axis]});
        written = emit(child, next, written);
      }
      return written;
    };
    return branch(0);
  }

  mlir::RewriterBase &rewriter;
  mlir::tensor::ExtractSliceOp read;
  const analysis::TensorSubsetDemand &demand;
  analysis::IndexRelationWork &work;
  mlir::Location location;
  std::map<std::vector<int64_t>, BlockProofs> proofs;
  MaterializedTensorAssemblyRead result;
  uint64_t templates = 0;
  TensorSubsetMaterializationStatus status =
      TensorSubsetMaterializationStatus::Unsupported;
  std::string detail;
};

} // namespace

analysis::TensorSubsetDemandResult
queryTensorSubsetSlice(mlir::tensor::ExtractSliceOp read,
                       analysis::IndexRelationWork &work) {
  using Status = analysis::TensorAssemblyStatus;
  auto reject = [](Status status, llvm::StringRef detail) {
    return analysis::TensorSubsetDemandResult{status, std::nullopt,
                                              detail.str()};
  };
  if (!read)
    return reject(Status::BrokenContract, "subset query requires a read");
  llvm::SmallVector<int64_t> sizes;
  for (auto size : read.getMixedSizes()) {
    auto constant = mlir::getConstantIntValue(size);
    if (!constant || *constant <= 0)
      return reject(Status::Unsupported,
                    "subset sizes must be positive constants");
    sizes.push_back(*constant);
  }
  if (llvm::any_of(
          read.getMixedStrides(),
          [](auto stride) { return mlir::getConstantIntValue(stride) != 1; }) ||
      !read.getType().hasStaticShape() ||
      !mlir::computeRankReductionMask(sizes, read.getType().getShape()))
    return reject(Status::Unsupported,
                  "subset needs exact static unit-stride geometry");
  return analysis::queryTensorSubsetDemand(
      read.getSource(), read.getMixedOffsets(), sizes, read, work);
}

TensorSubsetMaterializationResult
materializeTensorSubsetRead(mlir::RewriterBase &rewriter,
                            mlir::tensor::ExtractSliceOp read,
                            const analysis::IndexRelationLimits &limits,
                            mlir::Operation *insertionPoint) {
  using Status = TensorSubsetMaterializationStatus;
  analysis::IndexRelationWork work(limits);
  auto reject = [&](Status status, llvm::StringRef detail) {
    return TensorSubsetMaterializationResult{
        status, std::nullopt, work.getConsumed(), 0, detail.str()};
  };
  auto query = queryTensorSubsetSlice(read, work);
  if (!query.isExact())
    return reject(
        query.status == analysis::TensorAssemblyStatus::ResourceExhausted
            ? Status::ResourceExhausted
        : query.status == analysis::TensorAssemblyStatus::BrokenContract
            ? Status::BrokenContract
            : Status::Unsupported,
        query.detail);
  const auto &demand = *query.demand;
  const auto &sizes = demand.shape;
  if (demand.sources.empty())
    return {Status::Exact,
            MaterializedTensorAssemblyRead{read.getResult(), {}},
            work.getConsumed(),
            0,
            {}};
  // A complete proved demand of one splat bit pattern has a compact literal
  // result even when its source coordinates cross reshape boundaries. This
  // uses the same source/definedness proof and never evaluates scalar math.
  mlir::DenseElementsAttr splat;
  bool uniform = llvm::all_of(demand.sources, [&](const auto &source) {
    mlir::DenseElementsAttr literal;
    if (!mlir::matchPattern(source.source, mlir::m_Constant(&literal)) ||
        !literal.isSplat())
      return false;
    auto compact = literal.resizeSplat(read.getType());
    if (splat && splat != compact)
      return false;
    splat = compact;
    return true;
  });
  if (uniform) {
    if (!work.charge())
      return reject(Status::ResourceExhausted, "splat emission budget");
    mlir::OpBuilder::InsertionGuard restore(rewriter);
    rewriter.setInsertionPoint(read);
    auto value = rewriter.create<mlir::arith::ConstantOp>(
        read.getLoc(), read.getType(), splat);
    return {Status::Exact,
            MaterializedTensorAssemblyRead{value, {}},
            work.getConsumed(),
            1,
            {}};
  }
  SubsetMaterializer materializer(rewriter, read, demand, work);
  auto staticBlocks = analysis::queryStaticTensorSubsetBlocks(demand, work);
  if (staticBlocks.status == analysis::IndexRelationStatus::ResourceExhausted ||
      staticBlocks.status == analysis::IndexRelationStatus::Invalid)
    return reject(staticBlocks.status ==
                          analysis::IndexRelationStatus::ResourceExhausted
                      ? Status::ResourceExhausted
                      : Status::BrokenContract,
                  staticBlocks.detail);
  bool staticCopy = staticBlocks.status == analysis::IndexRelationStatus::Exact;
  if (staticCopy) {
    for (const auto &block : staticBlocks.blocks)
      if (!work.charge(64 + 8 * (sizes.size() + block.copy.sourceSizes.size())))
        return reject(Status::ResourceExhausted,
                      "static subset emission budget");
  } else if (!materializer.prepare(sizes) ||
             !materializer.reserveEmission(sizes))
    return reject(
        work.isExhausted() ? Status::ResourceExhausted : materializer.status,
        work.isExhausted() ? "tensor subset exceeded cumulative emission budget"
                           : materializer.detail);
  if (!insertionPoint)
    insertionPoint = read;
  if (insertionPoint != read) {
    if (!insertionPoint->isProperAncestor(read))
      return reject(Status::BrokenContract,
                    "subset placement is not an enclosing loop");
    for (auto *block = read->getBlock(); block != insertionPoint->getBlock();) {
      auto loop =
          mlir::dyn_cast_or_null<mlir::scf::ForOp>(block->getParentOp());
      if (!loop)
        return reject(Status::Unsupported,
                      "subset placement crosses a conditional scope");
      auto lower = mlir::getConstantIntValue(loop.getLowerBound());
      auto upper = mlir::getConstantIntValue(loop.getUpperBound());
      if (!lower || !upper || *lower >= *upper)
        return reject(Status::Unsupported,
                      "subset placement crosses a possibly empty loop");
      block = loop->getBlock();
    }
    mlir::DominanceInfo dominance;
    for (const auto &source : demand.sources)
      if (!dominance.properlyDominates(source.source, insertionPoint))
        return reject(Status::Unsupported,
                      "subset source does not dominate its placement");
    for (const auto &[key, proofs] : materializer.proofs)
      for (const auto &source : proofs.sources) {
        if (!source.block)
          continue;
        for (auto [i, parameter] : llvm::enumerate(demand.parameters)) {
          bool used = llvm::any_of(
              source.block->sourceOffsets.getResults(),
              [&](mlir::AffineExpr e) { return e.isFunctionOfDim(i); });
          for (auto guard : source.block->guards)
            used |= llvm::any_of(
                guard.set.getConstraints(),
                [&](mlir::AffineExpr e) { return e.isFunctionOfDim(i); });
          if (used &&
              !dominance.properlyDominates(parameter.induction, insertionPoint))
            return reject(Status::Unsupported,
                          "subset index does not dominate its placement");
        }
      }
  }
  mlir::OpBuilder::InsertionGuard restore(rewriter);
  rewriter.setInsertionPoint(insertionPoint);
  llvm::SmallVector<mlir::OpFoldResult> origin(sizes.size(),
                                               rewriter.getIndexAttr(0));
  mlir::Value destination;
  bool needsDestination = staticCopy ? staticBlocks.blocks.size() != 1 : true;
  if (!staticCopy) {
    const auto &whole = materializer.proofs.at(
        std::vector<int64_t>(sizes.begin(), sizes.end()));
    needsDestination = whole.sources.size() != 1 ||
                       !whole.sources.front().block ||
                       !whole.sources.front().block->guards.empty();
  }
  if (needsDestination)
    destination = rewriter.create<mlir::tensor::EmptyOp>(
        read.getLoc(), sizes, read.getType().getElementType(),
        read.getType().getEncoding());
  mlir::Value value = destination;
  if (staticCopy) {
    for (const auto &block : staticBlocks.blocks) {
      llvm::SmallVector<mlir::OpFoldResult> offset;
      for (int64_t coordinate : block.destination.offsets)
        offset.push_back(rewriter.getIndexAttr(coordinate));
      value = materializer.copy(block.source, block.copy,
                                block.destination.sizes, offset, value);
      ++materializer.templates;
    }
  } else {
    value = materializer.emit(sizes, origin, destination);
  }
  materializer.result.value =
      reshapeStaticTensorTile(rewriter, read.getLoc(), value, read.getType());
  if (mlir::failed(mlir::verify(materializer.result.value.getDefiningOp())))
    return reject(Status::CompilerFailure,
                  "generated tensor subset violates its verified contract");
  return {Status::Exact,
          std::move(materializer.result),
          work.getConsumed(),
          materializer.templates,
          {}};
}

std::optional<int64_t> resolveStaticIndex(mlir::OpFoldResult value) {
  if (auto attribute = llvm::dyn_cast_if_present<mlir::IntegerAttr>(
          value.dyn_cast<mlir::Attribute>()))
    return attribute.getInt();
  mlir::Value dynamic = value.dyn_cast<mlir::Value>();
  if (!dynamic || !dynamic.getType().isIndex())
    return std::nullopt;
  mlir::FailureOr<int64_t> constant =
      mlir::ValueBoundsConstraintSet::computeConstantBound(
          mlir::presburger::BoundType::EQ,
          mlir::ValueBoundsConstraintSet::Variable(dynamic));
  if (mlir::succeeded(constant))
    return *constant;
  std::function<std::optional<int64_t>(mlir::Value, unsigned)> resolve =
      [&](mlir::Value current, unsigned depth) -> std::optional<int64_t> {
    if (!current || depth > 8)
      return std::nullopt;
    if (std::optional<int64_t> constant = mlir::getConstantIntValue(current))
      return constant;
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(current)) {
      auto loop = argument.getOwner()
                      ? mlir::dyn_cast_or_null<mlir::scf::ForOp>(
                            argument.getOwner()->getParentOp())
                      : mlir::scf::ForOp{};
      if (!loop || argument != loop.getInductionVar())
        return std::nullopt;
      std::optional<int64_t> lower =
          mlir::getConstantIntValue(loop.getLowerBound());
      std::optional<int64_t> upper =
          mlir::getConstantIntValue(loop.getUpperBound());
      std::optional<int64_t> step = mlir::getConstantIntValue(loop.getStep());
      if (!lower || !upper || !step || *step <= 0 || *upper <= *lower ||
          *upper - *lower > *step)
        return std::nullopt;
      return lower;
    }
    mlir::Operation *definition = current.getDefiningOp();
    mlir::AffineMap map;
    mlir::ValueRange operands;
    bool takeMinimum = false;
    bool takeMaximum = false;
    if (auto apply =
            mlir::dyn_cast_or_null<mlir::affine::AffineApplyOp>(definition)) {
      map = apply.getAffineMap();
      operands = apply.getMapOperands();
    } else if (auto minimum = mlir::dyn_cast_or_null<mlir::affine::AffineMinOp>(
                   definition)) {
      map = minimum.getAffineMap();
      operands = minimum.getMapOperands();
      takeMinimum = true;
    } else if (auto maximum = mlir::dyn_cast_or_null<mlir::affine::AffineMaxOp>(
                   definition)) {
      map = maximum.getAffineMap();
      operands = maximum.getMapOperands();
      takeMaximum = true;
    } else {
      return std::nullopt;
    }
    llvm::SmallVector<mlir::Attribute, 4> attributes;
    for (mlir::Value operand : operands) {
      std::optional<int64_t> operandValue = resolve(operand, depth + 1);
      if (!operandValue)
        return std::nullopt;
      attributes.push_back(mlir::IntegerAttr::get(
          mlir::IndexType::get(current.getContext()), *operandValue));
    }
    llvm::SmallVector<mlir::Attribute, 4> folded;
    if (mlir::failed(map.constantFold(attributes, folded)) || folded.empty())
      return std::nullopt;
    std::optional<int64_t> result;
    for (mlir::Attribute attribute : folded) {
      auto integer = mlir::dyn_cast<mlir::IntegerAttr>(attribute);
      if (!integer)
        return std::nullopt;
      if (!result)
        result = integer.getInt();
      else if (takeMinimum)
        result = std::min(*result, integer.getInt());
      else if (takeMaximum)
        result = std::max(*result, integer.getInt());
      else
        return std::nullopt;
    }
    return result;
  };
  return resolve(dynamic, 0);
}

} // namespace wafer::compiler::detail
