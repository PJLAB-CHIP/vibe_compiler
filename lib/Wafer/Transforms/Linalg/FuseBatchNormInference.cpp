//===- FuseBatchNormInference.cpp - Exact scalar fusion ------------------===//

#include "Wafer/Support/CompileTiming.h"
#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace wafer {
#define GEN_PASS_DEF_FUSEBATCHNORMINFERENCEPASS
#include "Wafer/Transforms/WaferTransformPasses.h.inc"

namespace {

// All reads and operation handles belong to this one unchanged matching epoch.
// Coordinates are those of the observable result, not a future tile/buffer.
struct Read {
  mlir::Value value;
  mlir::AffineMap map;
  bool operator==(const Read &other) const {
    return value == other.value && map == other.map;
  }
};

struct ScalarNode {
  Read result;
  mlir::linalg::GenericOp owner;
  mlir::Operation *scalar;
  llvm::SmallVector<Read, 2> operands;
};

bool isParallelMap(mlir::linalg::GenericOp op) {
  return op && op.hasPureTensorSemantics() && op.getNumDpsInits() == 1 &&
         op->getNumResults() == 1 &&
         op.getNumParallelLoops() == op.getNumLoops() &&
         op.getRegionOutputArgs().front().use_empty() &&
         op.getIndexingMapMatchingResult(op->getResult(0)).isPermutation();
}

bool isPassthrough(mlir::Operation *operation) {
  if (mlir::isa<mlir::tensor::CastOp, mlir::tensor::ExpandShapeOp,
                mlir::tensor::CollapseShapeOp>(operation))
    return true;
  auto op = mlir::dyn_cast<mlir::linalg::GenericOp>(operation);
  return isParallelMap(op) && op.getNumDpsInputs() == 1 &&
         op.getBody()->without_terminator().empty() &&
         mlir::cast<mlir::linalg::YieldOp>(op.getBody()->getTerminator())
                 .getValues()
                 .front() == op.getRegionInputArgs().front();
}

class Matcher {
public:
  explicit Matcher(mlir::linalg::GenericOp root)
      : root(root),
        type(mlir::cast<mlir::RankedTensorType>(root.getResult(0).getType())) {}

  std::optional<Read> normalize(Read read) {
    for (unsigned depth = 0; depth < 32; ++depth) {
      auto input = mlir::dyn_cast<mlir::RankedTensorType>(read.value.getType());
      if (!input || !input.hasStaticShape() || !read.map ||
          read.map.getNumResults() != input.getRank())
        return std::nullopt;
      auto *op = read.value.getDefiningOp();
      if (!op || op->getBlock() != root->getBlock())
        return read;
      if (auto cast = mlir::dyn_cast<mlir::tensor::CastOp>(op)) {
        auto source =
            mlir::cast<mlir::RankedTensorType>(cast.getSource().getType());
        if (source.getShape() != input.getShape())
          return std::nullopt;
        read.value = cast.getSource();
        continue;
      }
      if (auto expand = mlir::dyn_cast<mlir::tensor::ExpandShapeOp>(op)) {
        llvm::SmallVector<mlir::AffineExpr> coordinates;
        for (const auto &group : expand.getReassociationIndices()) {
          mlir::AffineExpr coordinate =
              mlir::getAffineConstantExpr(0, op->getContext());
          bool found = false;
          for (int64_t axis : group)
            if (input.getDimSize(axis) != 1) {
              if (found)
                return read;
              coordinate = read.map.getResult(axis);
              found = true;
            }
          coordinates.push_back(coordinate);
        }
        read = {expand.getSrc(),
                mlir::AffineMap::get(type.getRank(), 0, coordinates,
                                     op->getContext())};
        continue;
      }
      if (auto collapse = mlir::dyn_cast<mlir::tensor::CollapseShapeOp>(op)) {
        auto source =
            mlir::cast<mlir::RankedTensorType>(collapse.getSrc().getType());
        if (!source.hasStaticShape())
          return std::nullopt;
        llvm::SmallVector<mlir::AffineExpr> coordinates(
            source.getRank(), mlir::getAffineConstantExpr(0, op->getContext()));
        for (auto [axis, group] :
             llvm::enumerate(collapse.getReassociationIndices())) {
          bool found = false;
          for (int64_t sourceAxis : group)
            if (source.getDimSize(sourceAxis) != 1) {
              if (found)
                return read;
              coordinates[sourceAxis] = read.map.getResult(axis);
              found = true;
            }
        }
        read = {collapse.getSrc(),
                mlir::AffineMap::get(type.getRank(), 0, coordinates,
                                     op->getContext())};
        continue;
      }
      auto generic = mlir::dyn_cast<mlir::linalg::GenericOp>(op);
      if (!generic || !isPassthrough(op))
        return read;
      auto loops =
          mlir::inversePermutation(
              generic.getIndexingMapMatchingResult(generic->getResult(0)))
              .compose(read.map);
      read = {generic.getDpsInputs().front(),
              generic.getIndexingMapsArray().front().compose(loops)};
    }
    return std::nullopt;
  }

  std::optional<ScalarNode> inspect(Read read) {
    auto normalized = normalize(read);
    if (!normalized)
      return std::nullopt;
    auto op = normalized->value.getDefiningOp<mlir::linalg::GenericOp>();
    if (!isParallelMap(op) || op->getBlock() != root->getBlock())
      return std::nullopt;
    auto &body = *op.getBody();
    if (std::distance(body.begin(), body.end()) != 2)
      return std::nullopt;
    auto *scalar = &body.front();
    if (scalar->getNumResults() != 1 || scalar->getNumRegions() ||
        !mlir::isMemoryEffectFree(scalar) ||
        mlir::cast<mlir::linalg::YieldOp>(body.getTerminator())
                .getValues()
                .front() != scalar->getResult(0))
      return std::nullopt;
    ScalarNode node{*normalized, op, scalar, {}};
    auto loops = mlir::inversePermutation(
                     op.getIndexingMapMatchingResult(op->getResult(0)))
                     .compose(normalized->map);
    for (auto operand : scalar->getOperands()) {
      auto argument = mlir::dyn_cast<mlir::BlockArgument>(operand);
      if (!argument || argument.getOwner() != &body ||
          argument.getArgNumber() >= op.getNumDpsInputs())
        return std::nullopt;
      unsigned index = argument.getArgNumber();
      node.operands.push_back(
          {op.getDpsInputs()[index],
           op.getIndexingMapsArray()[index].compose(loops)});
    }
    return node;
  }

  template <typename Op> std::optional<ScalarNode> match(Read read) {
    auto node = inspect(read);
    return node && mlir::isa<Op>(node->scalar) ? node : std::nullopt;
  }

  std::optional<Read> input(Read read) {
    if (auto cast = match<mlir::arith::ExtFOp>(read))
      read = cast->operands.front();
    return normalize(read);
  }

  std::optional<unsigned> feature(Read read) const {
    auto shape = mlir::cast<mlir::RankedTensorType>(read.value.getType());
    std::optional<unsigned> result;
    for (auto [axis, expression] : llvm::enumerate(read.map.getResults())) {
      if (shape.getDimSize(axis) == 1)
        continue;
      auto dimension = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
      if (!dimension || (result && *result != dimension.getPosition()))
        return std::nullopt;
      result = dimension.getPosition();
    }
    if (!result || shape.getNumElements() != type.getDimSize(*result))
      return std::nullopt;
    return result;
  }

  bool isSplat(Read read, std::optional<double> wanted = std::nullopt) {
    auto value = normalize(read);
    if (!value)
      return false;
    auto constant = value->value.getDefiningOp<mlir::arith::ConstantOp>();
    auto dense =
        constant
            ? mlir::dyn_cast<mlir::DenseFPElementsAttr>(constant.getValue())
            : mlir::DenseFPElementsAttr{};
    return dense && dense.isSplat() &&
           (!wanted ||
            dense.getSplatValue<llvm::APFloat>().isExactlyValue(*wanted));
  }

  bool matchVariance(Read read, unsigned axis,
                     llvm::SmallVectorImpl<Read> &leaves) {
    auto add = match<mlir::arith::AddFOp>(read);
    if (!add)
      return false;
    for (unsigned side : {0u, 1u}) {
      auto variance = input(add->operands[side]);
      auto epsilon = normalize(add->operands[1 - side]);
      if (!variance || !epsilon || feature(*variance) != axis)
        continue;
      auto epsilonType =
          mlir::cast<mlir::RankedTensorType>(epsilon->value.getType());
      if (epsilonType.getNumElements() != 1 && !isSplat(*epsilon))
        continue;
      leaves.push_back(*variance);
      leaves.push_back(*epsilon);
      return true;
    }
    return false;
  }

  bool matchNormalized(Read read, llvm::SmallVectorImpl<Read> &leaves,
                       unsigned &axis) {
    auto node = inspect(read);
    if (!node ||
        !mlir::isa<mlir::arith::MulFOp, mlir::arith::DivFOp>(node->scalar))
      return false;
    for (unsigned side : {0u, 1u}) {
      if (side && mlir::isa<mlir::arith::DivFOp>(node->scalar))
        break;
      auto center = match<mlir::arith::SubFOp>(node->operands[side]);
      if (!center)
        continue;
      auto activation = input(center->operands[0]);
      auto mean = input(center->operands[1]);
      if (!activation || !mean || !activation->map.isPermutation())
        continue;
      auto featureAxis = feature(*mean);
      if (!featureAxis)
        continue;
      Read deviation = node->operands[1 - side];
      llvm::SmallVector<Read> candidate{*activation, *mean};
      if (mlir::isa<mlir::arith::MulFOp>(node->scalar)) {
        if (auto reciprocal = match<mlir::arith::DivFOp>(deviation)) {
          if (!isSplat(reciprocal->operands[0], 1.0))
            continue;
          candidate.push_back(*normalize(reciprocal->operands[0]));
          deviation = reciprocal->operands[1];
        } else if (auto rsqrt = match<mlir::math::RsqrtOp>(deviation)) {
          if (!matchVariance(rsqrt->operands[0], *featureAxis, candidate))
            continue;
          llvm::append_range(leaves, candidate);
          axis = *featureAxis;
          return true;
        } else {
          continue;
        }
      }
      auto sqrt = match<mlir::math::SqrtOp>(deviation);
      if (!sqrt || !matchVariance(sqrt->operands[0], *featureAxis, candidate))
        continue;
      llvm::append_range(leaves, candidate);
      axis = *featureAxis;
      return true;
    }
    return false;
  }

  bool recognize() {
    if (!type.hasStaticShape() || type.getRank() < 3 || !isParallelMap(root))
      return false;
    Read value{root.getResult(0), mlir::AffineMap::getMultiDimIdentityMap(
                                      type.getRank(), root.getContext())};
    output = value;
    if (auto relu = match<mlir::arith::MaximumFOp>(value)) {
      if (isSplat(relu->operands[1], 0.0)) {
        leaves.push_back(*normalize(relu->operands[1]));
        value = relu->operands[0];
      } else if (isSplat(relu->operands[0], 0.0)) {
        leaves.push_back(*normalize(relu->operands[0]));
        value = relu->operands[1];
      } else {
        return false;
      }
    }
    if (auto cast = match<mlir::arith::TruncFOp>(value))
      value = cast->operands[0];
    auto offset = match<mlir::arith::AddFOp>(value);
    if (!offset)
      return false;
    for (unsigned addSide : {0u, 1u}) {
      auto scale = match<mlir::arith::MulFOp>(offset->operands[addSide]);
      if (!scale)
        continue;
      for (unsigned mulSide : {0u, 1u}) {
        llvm::SmallVector<Read> candidate;
        unsigned axis;
        if (!matchNormalized(scale->operands[mulSide], candidate, axis))
          continue;
        auto gamma = input(scale->operands[1 - mulSide]);
        auto beta = input(offset->operands[1 - addSide]);
        if (!gamma || !beta || feature(*gamma) != axis ||
            feature(*beta) != axis)
          continue;
        candidate.push_back(*gamma);
        candidate.push_back(*beta);
        llvm::append_range(leaves, candidate);
        return collect(output) && closedUses();
      }
    }
    return false;
  }

  mlir::linalg::GenericOp materialize(mlir::IRRewriter &rewriter) {
    llvm::SmallVector<mlir::Value> operands;
    llvm::SmallVector<mlir::AffineMap> maps;
    for (Read read : inputs) {
      operands.push_back(read.value);
      maps.push_back(read.map);
    }
    maps.push_back(mlir::AffineMap::getMultiDimIdentityMap(type.getRank(),
                                                           root.getContext()));
    rewriter.setInsertionPoint(root);
    return rewriter.create<mlir::linalg::GenericOp>(
        root.getLoc(), mlir::TypeRange{type}, operands, root.getDpsInits(),
        maps,
        llvm::SmallVector<mlir::utils::IteratorType>(
            type.getRank(), mlir::utils::IteratorType::parallel),
        [&](mlir::OpBuilder &builder, mlir::Location loc,
            mlir::ValueRange arguments) {
          llvm::SmallVector<std::pair<Read, mlir::Value>> values;
          for (auto [index, read] : llvm::enumerate(inputs))
            values.emplace_back(read, arguments[index]);
          auto lookup = [&](Read read) {
            Read normalized = *normalize(read);
            auto found = llvm::find_if(values, [&](const auto &entry) {
              return entry.first == normalized;
            });
            assert(found != values.end());
            return found->second;
          };
          for (const ScalarNode &node : nodes) {
            mlir::IRMapping mapping;
            for (auto [original, operand] :
                 llvm::zip_equal(node.scalar->getOperands(), node.operands))
              mapping.map(original, lookup(operand));
            auto *cloned = builder.clone(*node.scalar, mapping);
            values.emplace_back(node.result, cloned->getResult(0));
          }
          builder.create<mlir::linalg::YieldOp>(loc, lookup(output));
        });
  }

  llvm::SmallVector<ScalarNode> nodes;

private:
  bool validInput(Read read) {
    auto shape = mlir::cast<mlir::RankedTensorType>(read.value.getType());
    if (!mlir::isa<mlir::FloatType>(shape.getElementType()) ||
        read.map.getNumSymbols())
      return false;
    for (auto [index, expression] : llvm::enumerate(read.map.getResults())) {
      if (auto dim = mlir::dyn_cast<mlir::AffineDimExpr>(expression)) {
        if (shape.getDimSize(index) != type.getDimSize(dim.getPosition()))
          return false;
      } else if (auto zero =
                     mlir::dyn_cast<mlir::AffineConstantExpr>(expression)) {
        if (zero.getValue() != 0 || shape.getDimSize(index) != 1)
          return false;
      } else {
        return false;
      }
    }
    return true;
  }

  bool collect(Read value) {
    auto read = normalize(value);
    if (!read || nodes.size() + inputs.size() >= 96)
      return false;
    if (llvm::is_contained(inputs, *read) ||
        llvm::any_of(nodes, [&](const ScalarNode &node) {
          return node.result == *read;
        }))
      return true;
    if (llvm::is_contained(leaves, *read)) {
      if (!validInput(*read))
        return false;
      inputs.push_back(*read);
      return true;
    }
    auto node = inspect(*read);
    if (!node)
      return false;
    for (Read operand : node->operands)
      if (!collect(operand))
        return false;
    nodes.push_back(*node);
    return true;
  }

  bool closedUses() const {
    llvm::DenseSet<mlir::Operation *> owners;
    for (const ScalarNode &node : nodes)
      if (!owners.insert(node.owner).second)
        return false; // A scalar occurrence is never duplicated over two maps.
    for (const ScalarNode &node : nodes) {
      if (node.owner == root)
        continue;
      llvm::SmallVector<mlir::Value> work{node.owner->getResult(0)};
      llvm::DenseSet<mlir::Value> seen;
      while (!work.empty()) {
        auto value = work.pop_back_val();
        if (!seen.insert(value).second)
          continue;
        if (seen.size() > 96)
          return false;
        for (mlir::OpOperand &use : value.getUses()) {
          auto *user = use.getOwner();
          if (owners.contains(user)) {
            if (!mlir::cast<mlir::linalg::GenericOp>(user).isDpsInput(&use))
              return false;
            continue;
          }
          if (!isPassthrough(user))
            return false;
          llvm::append_range(work, user->getResults());
        }
      }
    }
    return true;
  }

  mlir::linalg::GenericOp root;
  mlir::RankedTensorType type;
  Read output;
  llvm::SmallVector<Read> leaves;
  llvm::SmallVector<Read> inputs;
};

void eraseDeadClosure(mlir::IRRewriter &rewriter,
                      llvm::ArrayRef<mlir::Operation *> seeds) {
  llvm::SmallVector<mlir::Operation *> work(seeds);
  llvm::DenseSet<mlir::Operation *> queued(seeds.begin(), seeds.end());
  while (!work.empty()) {
    auto *op = work.pop_back_val();
    queued.erase(op);
    if (!mlir::isOpTriviallyDead(op))
      continue;
    for (auto input : op->getOperands())
      if (auto *producer = input.getDefiningOp();
          producer && queued.insert(producer).second)
        work.push_back(producer);
    rewriter.eraseOp(op);
  }
}

struct FuseBatchNormInferencePass final
    : impl::FuseBatchNormInferencePassBase<FuseBatchNormInferencePass> {
  void runOnOperation() final {
    auto function = getOperation();
    if (!function.getBody().hasOneBlock())
      return;
    llvm::SmallVector<mlir::linalg::GenericOp> roots;
    for (auto op : function.getBody().front().getOps<mlir::linalg::GenericOp>())
      if (op->getNumResults() == 1)
        roots.push_back(op);
    mlir::IRRewriter rewriter(&getContext());
    llvm::DenseSet<mlir::Operation *> consumed;
    llvm::SmallVector<mlir::Operation *> dead;
    for (auto root : llvm::reverse(roots)) {
      if (consumed.contains(root) || root.getResult(0).use_empty())
        continue;
      Matcher matcher(root);
      if (!matcher.recognize())
        continue;
      auto fused = matcher.materialize(rewriter);
      if (mlir::failed(mlir::verify(fused))) {
        rewriter.eraseOp(fused);
        signalPassFailure();
        return;
      }
      for (const ScalarNode &node : matcher.nodes)
        consumed.insert(node.owner);
      rewriter.replaceAllUsesWith(root.getResult(0), fused.getResult(0));
      dead.push_back(root);
      ++numFusedChains;
      support::addCompileCounter("normalization", "batch-norm-chains", 1);
    }
    // Original nodes remain live during the matching walk. Erase their dead
    // closure only after every original root handle has been consumed.
    eraseDeadClosure(rewriter, dead);
    if (mlir::failed(mlir::verify(function)))
      signalPassFailure();
  }
};
} // namespace
} // namespace wafer
