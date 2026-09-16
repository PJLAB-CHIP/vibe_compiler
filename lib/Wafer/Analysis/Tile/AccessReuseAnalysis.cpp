//===- AccessReuseAnalysis.cpp - Current read access reuse ----------------===//

#include "Wafer/Analysis/Tile/AccessReuseAnalysis.h"
#include "Wafer/Support/CompileTiming.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include <limits>
#include <map>
#include <optional>

namespace wafer::analysis {
namespace {
// Follow only explicit aliases. An escaping value, unknown effect or any
// second writer excludes the choice; this is not an alias-repair pass.
bool hasOnlyReads(mlir::Value value, mlir::Operation *writer,
                  llvm::DenseSet<mlir::Value> &visited) {
  if (!visited.insert(value).second)
    return true;
  for (mlir::OpOperand &use : value.getUses()) {
    mlir::Operation *op = use.getOwner();
    if (op == writer)
      continue;
    if (auto region = mlir::dyn_cast<TileRegionOp>(op)) {
      if (!hasOnlyReads(
              region.getBody().front().getArgument(use.getOperandNumber()),
              writer, visited))
        return false;
      continue;
    }
    if (auto view = mlir::dyn_cast<mlir::ViewLikeOpInterface>(op)) {
      if (view.getViewSource() != value)
        return false;
      for (mlir::Value result : op->getResults())
        if (!hasOnlyReads(result, writer, visited))
          return false;
      continue;
    }
    auto effects = mlir::getEffectsRecursively(op);
    if (!effects || op->hasTrait<mlir::OpTrait::IsTerminator>() ||
        op->getNumRegions())
      return false;
    for (mlir::Value result : op->getResults()) {
      if (!mlir::isa<mlir::ShapedType>(result.getType()))
        continue;
      // A materializing layout conversion or compute result has an explicit
      // allocation effect. It cannot alias this input. Unclassified results
      // remain excluded, and metadata aliases were followed above.
      if (!mlir::isa<mlir::MemRefType>(result.getType()) ||
          !llvm::any_of(*effects, [&](const auto &effect) {
            return effect.getValue() == result &&
                   mlir::isa<mlir::MemoryEffects::Allocate>(effect.getEffect());
          }))
        return false;
    }
    bool reads = false;
    for (const auto &effect : *effects) {
      if (effect.getValue() != value)
        continue;
      if (!mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect()))
        return false;
      reads = true;
    }
    if (!reads)
      return false;
  }
  return true;
}

mlir::BlockArgument getProgramSource(mlir::Value value) {
  while (true) {
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      if (auto region = mlir::dyn_cast<TileRegionOp>(
              argument.getOwner()->getParentOp())) {
        value = region.getInputs()[argument.getArgNumber()];
        continue;
      }
      return mlir::isa<mlir::func::FuncOp>(argument.getOwner()->getParentOp())
                 ? argument
                 : mlir::BlockArgument{};
    }
    if (auto view = value.getDefiningOp<mlir::memref::SubViewOp>()) {
      value = view.getSource();
      continue;
    }
    return {};
  }
}

// Compare actual address functions over the enclosing ordered iteration
// domain. The expression is query-local; no symbolic window survives mutation.
class AccessIndexing {
public:
  explicit AccessIndexing(llvm::ArrayRef<analysis::StaticLoopDomain> loops)
      : loops(loops) {}

  mlir::AffineExpr index(mlir::Value value) {
    auto found = expressions.find(value);
    if (found != expressions.end())
      return found->second;
    auto *context = value.getContext();
    mlir::AffineExpr result;
    if (auto constant = mlir::getConstantIntValue(value))
      result = mlir::getAffineConstantExpr(*constant, context);
    for (auto [position, domain] : llvm::enumerate(loops))
      if (mlir::scf::ForOp(domain.loop).getInductionVar() == value)
        result = mlir::getAffineDimExpr(position, context);
    if (!result) {
      if (auto add = value.getDefiningOp<mlir::arith::AddIOp>()) {
        auto a = index(add.getLhs()), b = index(add.getRhs());
        if (a && b)
          result = a + b;
      } else if (auto sub = value.getDefiningOp<mlir::arith::SubIOp>()) {
        auto a = index(sub.getLhs()), b = index(sub.getRhs());
        if (a && b)
          result = a - b;
      } else if (auto mul = value.getDefiningOp<mlir::arith::MulIOp>()) {
        auto a = index(mul.getLhs()), b = index(mul.getRhs());
        if (a && b &&
            (mlir::isa<mlir::AffineConstantExpr>(a) ||
             mlir::isa<mlir::AffineConstantExpr>(b)))
          result = a * b;
      } else if (auto apply =
                     value.getDefiningOp<mlir::affine::AffineApplyOp>()) {
        llvm::SmallVector<mlir::AffineExpr> operands;
        for (auto operand : apply.getOperands()) {
          auto expression = index(operand);
          if (!expression)
            return {};
          operands.push_back(expression);
        }
        auto map = apply.getAffineMap();
        result = map.getResult(0).replaceDimsAndSymbols(
            llvm::ArrayRef(operands).take_front(map.getNumDims()),
            llvm::ArrayRef(operands).drop_front(map.getNumDims()));
      }
    }
    if (result)
      result = mlir::simplifyAffineExpr(result, loops.size(), 0);
    expressions[value] = result;
    return result;
  }

  mlir::AffineExpr buffer(mlir::Value value) {
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      if (auto region =
              mlir::dyn_cast<TileRegionOp>(argument.getOwner()->getParentOp()))
        return buffer(region.getInputs()[argument.getArgNumber()]);
      auto type = mlir::dyn_cast<mlir::MemRefType>(argument.getType());
      llvm::SmallVector<int64_t> strides;
      int64_t offset;
      if (!type ||
          !mlir::isa<mlir::func::FuncOp>(argument.getOwner()->getParentOp()) ||
          mlir::failed(mlir::getStridesAndOffset(type, strides, offset)) ||
          mlir::ShapedType::isDynamic(offset))
        return {};
      return mlir::getAffineConstantExpr(offset, value.getContext());
    }
    auto view = value.getDefiningOp<mlir::memref::SubViewOp>();
    if (!view)
      return {};
    auto result = buffer(view.getSource());
    llvm::SmallVector<int64_t> strides;
    int64_t ignored;
    if (!result || mlir::failed(mlir::getStridesAndOffset(view.getSourceType(),
                                                          strides, ignored)))
      return {};
    for (auto [offset, stride] :
         llvm::zip_equal(view.getMixedOffsets(), strides)) {
      if (stride < 0)
        return {};
      mlir::AffineExpr expression;
      if (auto constant = mlir::getConstantIntValue(offset))
        expression = mlir::getAffineConstantExpr(*constant, value.getContext());
      else
        expression = index(mlir::cast<mlir::Value>(offset));
      if (!expression)
        return {};
      result = result + expression * stride;
    }
    return mlir::simplifyAffineExpr(result, loops.size(), 0);
  }

  std::optional<llvm::SmallVector<mlir::AffineExpr>>
  coordinates(mlir::Value value, mlir::Value &base) {
    auto type = mlir::dyn_cast<mlir::MemRefType>(value.getType());
    if (!type || !type.hasStaticShape())
      return std::nullopt;
    if (auto view = value.getDefiningOp<mlir::memref::SubViewOp>()) {
      if (view.getSourceType().getRank() != type.getRank() ||
          llvm::any_of(view.getStaticStrides(),
                       [](int64_t stride) { return stride != 1; }))
        return std::nullopt;
      auto parent = coordinates(view.getSource(), base);
      if (!parent)
        return std::nullopt;
      for (auto [axis, offset] : llvm::enumerate(view.getMixedOffsets())) {
        mlir::AffineExpr expression;
        if (auto constant = mlir::getConstantIntValue(offset))
          expression =
              mlir::getAffineConstantExpr(*constant, value.getContext());
        else
          expression = index(mlir::cast<mlir::Value>(offset));
        if (!expression)
          return std::nullopt;
        (*parent)[axis] = mlir::simplifyAffineExpr((*parent)[axis] + expression,
                                                   loops.size(), 0);
      }
      return parent;
    }
    if (!mlir::isa<mlir::BlockArgument>(value))
      return std::nullopt;
    base = value;
    return llvm::SmallVector<mlir::AffineExpr>(
        type.getRank(), mlir::getAffineConstantExpr(0, value.getContext()));
  }

private:
  llvm::ArrayRef<analysis::StaticLoopDomain> loops;
  llvm::DenseMap<mlir::Value, mlir::AffineExpr> expressions;
};

std::optional<ReadAccess> getReadAccess(StorageLoadOp load) {
  auto region = load->getParentOfType<TileRegionOp>();
  auto domains = analysis::getEnclosingStaticLoopDomains(load);
  if (!region || !domains ||
      !mlir::isa<mlir::func::FuncOp>(region->getParentOp()))
    return std::nullopt;
  auto argument = getProgramSource(load.getSource());
  if (!argument)
    return std::nullopt;
  auto function =
      mlir::cast<mlir::func::FuncOp>(argument.getOwner()->getParentOp());
  auto identity = function.getArgAttrOfType<ProgramArgumentAttr>(
      argument.getArgNumber(), kWaferProgramArgumentAttrName);
  if (!identity || identity.getIndex() != argument.getArgNumber())
    return std::nullopt;
  auto original = mlir::dyn_cast<mlir::MemRefType>(argument.getType());
  auto source = mlir::cast<mlir::MemRefType>(load.getSource().getType());
  auto payload = mlir::cast<mlir::MemRefType>(load.getDest().getType());
  if (!original || !original.getLayout().isIdentity() ||
      !isWaferDDRMemRefType(original) ||
      getWaferMemoryAttr(original).getLayout() != MemLayout::Tensor ||
      !source.hasStaticShape() || !payload.hasStaticShape() ||
      !load.getDest().getDefiningOp<mlir::memref::AllocOp>())
    return std::nullopt;
  llvm::SmallVector<int64_t> strides;
  int64_t offset;
  if (mlir::failed(mlir::getStridesAndOffset(source, strides, offset)) ||
      llvm::any_of(strides, [](int64_t stride) { return stride < 0; }))
    return std::nullopt;
  auto physical = computeWaferPhysicalTensorInfo(payload);
  if (!physical || physical->physicalBytes <= 0)
    return std::nullopt;
  llvm::DenseSet<mlir::Value> reads;
  if (!hasOnlyReads(load.getDest(), load, reads))
    return std::nullopt;
  AccessIndexing offsets(*domains);
  auto expression = offsets.buffer(load.getSource());
  if (!expression)
    return std::nullopt;
  ReadAccess access;
  access.load = load;
  access.tile = load->getParentOfType<TileModuleOp>();
  access.argument = identity.getIndex();
  access.sourceType = source;
  access.payloadType = payload;
  access.strides = std::move(strides);
  access.linearOffset = expression;
  access.loops = *domains;
  access.physicalBytes = physical->physicalBytes;
  if (auto coordinates = offsets.coordinates(load.getSource(), access.base))
    access.coordinates = mlir::AffineMap::get(access.loops.size(), 0,
                                              *coordinates, load.getContext());
  return access;
}

AccessReuseAnalysis collectReads(mlir::ModuleOp module) {
  llvm::SmallVector<TileModuleOp> tiles;
  for (auto tile : module.getOps<TileModuleOp>())
    tiles.push_back(tile);
  llvm::sort(tiles, [](TileModuleOp a, TileModuleOp b) {
    return std::make_pair(a.getCardId(), a.getTileId()) <
           std::make_pair(b.getCardId(), b.getTileId());
  });
  // A program argument denotes the same external resource across these
  // isolated Tile entries. A writer in any binding invalidates read-only
  // sharing, including Tiles that have no otherwise eligible load.
  std::map<std::pair<int64_t, int64_t>, bool> readOnlyInputs;
  bool completeBindings = true;
  for (auto tile : tiles)
    tile.walk([&](mlir::func::FuncOp function) {
      if (function.isExternal())
        return;
      bool unclassifiedEffects = false;
      function.walk([&](mlir::Operation *operation) {
        if (operation == function ||
            operation->hasTrait<mlir::OpTrait::HasRecursiveMemoryEffects>())
          return;
        auto effects = mlir::getEffectsRecursively(operation);
        if (!effects || llvm::any_of(*effects, [](const auto &effect) {
              return !effect.getValue() &&
                     effect.getResource() ==
                         mlir::SideEffects::DefaultResource::get() &&
                     mlir::isa<mlir::MemoryEffects::Write,
                               mlir::MemoryEffects::Free>(effect.getEffect());
            }))
          unclassifiedEffects = true;
      });
      for (auto argument : function.getArguments()) {
        auto identity = function.getArgAttrOfType<ProgramArgumentAttr>(
            argument.getArgNumber(), kWaferProgramArgumentAttrName);
        if (!identity) {
          if (mlir::isa<mlir::BaseMemRefType>(argument.getType()) &&
              !function.getArgAttrOfType<DDRBindingAttr>(
                  argument.getArgNumber(), kWaferDDRBindingAttrName))
            completeBindings = false;
          continue;
        }
        auto [entry, inserted] = readOnlyInputs.try_emplace(
            std::make_pair(tile.getCardId(), identity.getIndex()), true);
        llvm::DenseSet<mlir::Value> visited;
        entry->second = entry->second && !unclassifiedEffects &&
                        hasOnlyReads(argument, nullptr, visited);
      }
    });
  AccessReuseAnalysis result;
  if (!completeBindings) {
    result.status = IndexRelationStatus::Unsupported;
    result.detail = "access reuse requires complete typed source bindings";
    return result;
  }
  for (auto tile : tiles)
    tile.walk([&](StorageLoadOp load) {
      auto access = getReadAccess(load);
      if (!access || !readOnlyInputs.at({tile.getCardId(), access->argument}))
        return;
      result.reads.push_back(std::move(*access));
    });
  return result;
}

bool equivalentPeerAccess(const ReadAccess &a, const ReadAccess &b) {
  auto aTile = a.tile, bTile = b.tile;
  return a.argument == b.argument && a.sourceType == b.sourceType &&
         a.payloadType == b.payloadType && a.strides == b.strides &&
         a.linearOffset == b.linearOffset &&
         aTile.getCardId() == bTile.getCardId() &&
         a.loops.size() == b.loops.size() &&
         llvm::all_of(llvm::zip_equal(a.loops, b.loops), [](const auto &pair) {
           return std::get<0>(pair).bounds() == std::get<1>(pair).bounds();
         });
}

std::optional<uint64_t> product(uint64_t a, uint64_t b) {
  if (b && a > std::numeric_limits<uint64_t>::max() / b)
    return std::nullopt;
  return a * b;
}

std::optional<uint64_t> byteCount(llvm::ArrayRef<int64_t> shape,
                                  mlir::Type element) {
  if (!element.isIntOrFloat() || element.getIntOrFloatBitWidth() % 8)
    return std::nullopt;
  uint64_t size = element.getIntOrFloatBitWidth() / 8;
  for (int64_t extent : shape) {
    if (extent <= 0)
      return std::nullopt;
    auto next = product(size, extent);
    if (!next)
      return std::nullopt;
    size = *next;
  }
  return size;
}

// Extract only proved linear expressions. The image/equality proof below is
// still performed by IndexRelation; this adapter does not approximate holes,
// floor divisions, negative traversal, or an unknown SSA index.
struct LinearAccess {
  int64_t constant;
  llvm::SmallVector<int64_t, 4> coefficients;
};

std::optional<LinearAccess> linearAccess(mlir::AffineExpr expression,
                                         unsigned rank) {
  auto *context = expression.getContext();
  llvm::SmallVector<mlir::AffineExpr> values(
      rank, mlir::getAffineConstantExpr(0, context));
  auto evaluate = [&]() -> std::optional<int64_t> {
    auto value = mlir::simplifyAffineExpr(
        expression.replaceDimsAndSymbols(values, {}), 0, 0);
    auto constant = mlir::dyn_cast<mlir::AffineConstantExpr>(value);
    return constant ? std::optional<int64_t>(constant.getValue())
                    : std::nullopt;
  };
  auto constant = evaluate();
  if (!constant || *constant < 0)
    return std::nullopt;
  LinearAccess result{*constant, {}};
  mlir::AffineExpr reconstructed =
      mlir::getAffineConstantExpr(*constant, context);
  for (unsigned axis = 0; axis < rank; ++axis) {
    values[axis] = mlir::getAffineConstantExpr(1, context);
    auto one = evaluate();
    values[axis] = mlir::getAffineConstantExpr(0, context);
    int64_t coefficient;
    if (!one || llvm::SubOverflow(*one, *constant, coefficient) ||
        coefficient < 0)
      return std::nullopt;
    result.coefficients.push_back(coefficient);
    reconstructed =
        reconstructed + mlir::getAffineDimExpr(axis, context) * coefficient;
  }
  auto difference = mlir::dyn_cast<mlir::AffineConstantExpr>(
      mlir::simplifyAffineExpr(expression - reconstructed, rank, 0));
  if (!difference || difference.getValue() != 0)
    return std::nullopt;
  return result;
}

ScopedReadAccessResult queryScope(llvm::ArrayRef<const ReadAccess *> reads,
                                  mlir::Value source, mlir::scf::ForOp scope,
                                  const IndexRelationLimits &limits) {
  ScopedReadAccess result;
  result.source = source;
  result.scope = scope;
  auto sourceType = mlir::cast<mlir::MemRefType>(source.getType());
  int64_t ignoredOffset;
  if (mlir::failed(mlir::getStridesAndOffset(sourceType, result.sourceStrides,
                                             ignoredOffset)))
    return {};
  auto *context = scope.getContext();
  std::optional<mlir::presburger::PresburgerSet> united;
  llvm::SmallVector<mlir::AffineExpr> origins;
  for (const ReadAccess *record : reads) {
    const ReadAccess &read = *record;
    if (read.base != source || !scope->isAncestor(read.load))
      continue;
    if (!read.coordinates)
      return {};
    auto found = llvm::find_if(
        read.loops, [&](const auto &loop) { return loop.loop == scope; });
    if (found == read.loops.end())
      return {};
    const unsigned outerCount = found - read.loops.begin();
    llvm::SmallVector<mlir::Value> parameters;
    uint64_t scopeExecutions = 1, readExecutions = 1;
    llvm::SmallVector<int64_t> iterationShape;
    for (auto [index, loop] : llvm::enumerate(read.loops)) {
      int64_t trips = (loop.upper - loop.lower - 1) / loop.step + 1;
      auto &count = index < outerCount ? scopeExecutions : readExecutions;
      auto next = product(count, trips);
      if (!next)
        return {IndexRelationStatus::ResourceExhausted, {}};
      count = *next;
      if (index < outerCount)
        parameters.push_back(mlir::scf::ForOp(loop.loop).getInductionVar());
      else
        iterationShape.push_back(trips);
    }
    const unsigned innerCount = iterationShape.size();
    llvm::append_range(iterationShape, read.sourceType.getShape());
    llvm::SmallVector<mlir::AffineExpr> readOrigins, imageExpressions;
    llvm::SmallVector<mlir::AffineExpr> globalDimensions;
    for (unsigned i = 0; i < read.loops.size(); ++i)
      globalDimensions.push_back(i < outerCount
                                     ? mlir::getAffineDimExpr(i, context)
                                     : mlir::getAffineConstantExpr(0, context));
    auto globalOrigin =
        read.linearOffset.replaceDimsAndSymbols(globalDimensions, {});
    for (auto [axis, expression] :
         llvm::enumerate(read.coordinates.getResults())) {
      auto linear = linearAccess(expression, read.loops.size());
      if (!linear)
        return {};
      int64_t constantOffset;
      if (llvm::MulOverflow(linear->constant, result.sourceStrides[axis],
                            constantOffset))
        return {IndexRelationStatus::ResourceExhausted, {}};
      globalOrigin = globalOrigin - constantOffset;
      auto origin = mlir::getAffineConstantExpr(0, context);
      auto mapped = mlir::getAffineConstantExpr(linear->constant, context);
      int64_t maximum = linear->constant;
      for (auto [index, loop] : llvm::enumerate(read.loops)) {
        const int64_t coefficient = linear->coefficients[index];
        int64_t last = loop.lower +
                       ((loop.upper - loop.lower - 1) / loop.step) * loop.step;
        int64_t term;
        if (llvm::MulOverflow(coefficient, last, term) ||
            llvm::AddOverflow(maximum, term, maximum))
          return {IndexRelationStatus::ResourceExhausted, {}};
        if (index < outerCount)
          origin =
              origin + mlir::getAffineDimExpr(index, context) * coefficient;
        else {
          auto iv =
              mlir::getAffineDimExpr(index - outerCount, context) * loop.step +
              loop.lower;
          mapped = mapped + iv * coefficient;
        }
      }
      if (llvm::AddOverflow(maximum, read.sourceType.getDimSize(axis) - 1,
                            maximum) ||
          maximum >= sourceType.getDimSize(axis))
        return {};
      readOrigins.push_back(mlir::simplifyAffineExpr(origin, outerCount, 0));
      imageExpressions.push_back(
          mapped + mlir::getAffineDimExpr(innerCount + axis, context));
    }
    if (!result.reads.empty() &&
        (parameters != result.parameters || origins != readOrigins))
      return {};
    if (result.reads.empty()) {
      result.tile = read.tile;
      result.argument = read.argument;
      result.linearOffset =
          mlir::simplifyAffineExpr(globalOrigin, outerCount, 0);
      result.outerLoops.append(read.loops.begin(),
                               read.loops.begin() + outerCount);
    }
    auto relation = IndexRelation::fromAffineMap(
        mlir::AffineMap::get(iterationShape.size(), 0, imageExpressions,
                             context),
        iterationShape, sourceType.getShape(), limits);
    if (!relation.isExact())
      return {relation.status, {}};
    // The existing parametric tile-image query proves dense affine intervals
    // directly. Asking the generic Presburger image for block*step+element
    // repeatedly would rediscover this same tiling proof for every Tile.
    auto image = relation.get()->getRectangularTileImage(
        context, iterationShape, sourceType.getShape(), iterationShape, limits);
    if (!image.isExact())
      return {image.status, {}};
    auto evaluate =
        [&](mlir::AffineMap map,
            bool withOffsets) -> std::optional<llvm::SmallVector<int64_t>> {
      llvm::SmallVector<mlir::AffineExpr> arguments;
      if (withOffsets)
        arguments.assign(iterationShape.size(),
                         mlir::getAffineConstantExpr(0, context));
      for (auto size : iterationShape)
        arguments.push_back(mlir::getAffineConstantExpr(size, context));
      llvm::SmallVector<int64_t> values;
      for (auto expression : map.getResults()) {
        auto constant =
            mlir::dyn_cast<mlir::AffineConstantExpr>(mlir::simplifyAffineExpr(
                expression.replaceDimsAndSymbols(arguments, {}), 0, 0));
        if (!constant)
          return std::nullopt;
        values.push_back(constant.getValue());
      }
      return values;
    };
    auto offsets = evaluate(image.image->offsetMap, true);
    auto sizes = evaluate(image.image->sizeMap, false);
    if (!offsets || !sizes)
      return {};
    auto rectangle =
        IndexRelation::staticRectangularDomain(*offsets, *sizes, limits);
    if (!rectangle.isExact())
      return {rectangle.status, {}};
    if (!united)
      united = std::move(*rectangle.set);
    else
      united->unionInPlace(*rectangle.set);
    auto bytes =
        byteCount(read.sourceType.getShape(), sourceType.getElementType());
    auto total = bytes ? product(*bytes, readExecutions) : std::nullopt;
    auto payload = product(read.physicalBytes, readExecutions);
    if (!total || !payload ||
        result.readBytes > std::numeric_limits<uint64_t>::max() - *total ||
        result.payloadBytes > std::numeric_limits<uint64_t>::max() - *payload)
      return {IndexRelationStatus::ResourceExhausted, {}};
    result.readBytes += *total;
    result.payloadBytes += *payload;
    result.scopeExecutions = scopeExecutions;
    result.parameters = std::move(parameters);
    origins = std::move(readOrigins);
    result.reads.push_back(read.load);
  }
  if (!united)
    return {};
  IndexSetResult unionResult{IndexRelationStatus::Exact, std::move(united), {}};
  auto rectangle = unionResult.getExactStaticRectangularDomain(limits);
  if (!rectangle.isExact())
    return {rectangle.status, {}};
  result.sizes = std::move(rectangle.domain->sizes);
  auto bytes = byteCount(result.sizes, sourceType.getElementType());
  if (!bytes)
    return {IndexRelationStatus::ResourceExhausted, {}};
  result.windowBytes = *bytes;
  if (result.readBytes <= result.windowBytes)
    return {};
  for (auto [axis, offset] : llvm::enumerate(rectangle.domain->offsets))
    origins[axis] = origins[axis] + offset;
  for (auto [axis, offset] : llvm::enumerate(rectangle.domain->offsets)) {
    int64_t linear;
    if (llvm::MulOverflow(offset, result.sourceStrides[axis], linear))
      return {IndexRelationStatus::ResourceExhausted, {}};
    result.linearOffset = result.linearOffset + linear;
  }
  result.linearOffset = mlir::simplifyAffineExpr(result.linearOffset,
                                                 result.parameters.size(), 0);
  result.origin =
      mlir::AffineMap::get(result.parameters.size(), 0, origins, context);
  return {IndexRelationStatus::Exact, std::move(result)};
}

std::optional<SlidingReadAccess> getSlidingAccess(const ReadAccess &read) {
  auto load = read.load;
  auto loop = mlir::dyn_cast<mlir::scf::ForOp>(load->getParentOp());
  if (!loop || !read.coordinates || read.loops.empty())
    return std::nullopt;
  std::optional<unsigned> axis;
  int64_t shift = 0;
  for (auto [index, expression] :
       llvm::enumerate(read.coordinates.getResults())) {
    auto linear = linearAccess(expression, read.loops.size());
    if (!linear)
      return std::nullopt;
    int64_t coefficient = linear->coefficients.back();
    if (!coefficient)
      continue;
    if (axis || llvm::MulOverflow(coefficient, read.loops.back().step, shift) ||
        shift <= 0 || shift >= read.sourceType.getDimSize(index))
      return std::nullopt;
    axis = index;
  }
  if (!axis)
    return std::nullopt;
  return SlidingReadAccess{read, loop, *axis, shift};
}

void collectScopedAccesses(AccessReuseAnalysis &facts,
                           const IndexRelationLimits &limits,
                           uint64_t queryLimit) {
  llvm::SmallVector<std::pair<mlir::Value, mlir::scf::ForOp>, 16> queries;
  llvm::DenseSet<std::pair<mlir::Value, mlir::Operation *>> seen;
  llvm::DenseMap<mlir::Value, llvm::SmallVector<const ReadAccess *, 4>>
      bySource;
  for (const auto &read : facts.reads) {
    if (!read.base || !read.coordinates)
      continue;
    auto load = read.load;
    bySource[read.base].push_back(&read);
    auto region = load->getParentOfType<TileRegionOp>();
    for (const auto &domain : llvm::reverse(read.loops)) {
      auto scope = domain.loop;
      if (scope->getParentOfType<TileRegionOp>() != region)
        break;
      auto query = std::make_pair(read.base, scope);
      if (seen.insert({read.base, scope.getOperation()}).second)
        queries.push_back(query);
    }
    if (auto sliding = getSlidingAccess(read))
      facts.sliding.push_back(std::move(*sliding));
  }
  for (auto [source, scope] : queries) {
    if (facts.scopeQueries == queryLimit) {
      facts.indeterminateScopes += queries.size() - facts.scopeQueries;
      break;
    }
    ++facts.scopeQueries;
    auto query = queryScope(bySource.lookup(source), source, scope, limits);
    if (query.access)
      facts.scopes.push_back(std::move(*query.access));
    else if (query.status == IndexRelationStatus::ResourceExhausted)
      ++facts.indeterminateScopes;
    else
      ++facts.unsupportedScopes;
  }
  llvm::erase_if(facts.sliding, [&](const auto &window) {
    return !llvm::any_of(facts.scopes, [&](const auto &scope) {
      return scope.scope == window.scope &&
             scope.source == window.access.base && scope.reads.size() == 1 &&
             scope.reads.front() == window.access.load;
    });
  });
}

} // namespace

ScopedReadAccessResult
queryScopedAccessReuse(const AccessReuseAnalysis &facts, mlir::Value source,
                       mlir::scf::ForOp scope,
                       const IndexRelationLimits &limits) {
  llvm::SmallVector<const ReadAccess *, 4> reads;
  for (const auto &read : facts.reads)
    if (read.base == source)
      reads.push_back(&read);
  return queryScope(reads, source, scope, limits);
}

SlidingReadAccessResult
querySlidingAccessReuse(const AccessReuseAnalysis &facts, StorageLoadOp load,
                        const IndexRelationLimits &limits) {
  auto read = llvm::find_if(
      facts.reads, [&](const auto &read) { return read.load == load; });
  if (read == facts.reads.end() || !read->coordinates || read->loops.empty())
    return {};
  auto loop = mlir::dyn_cast<mlir::scf::ForOp>(load->getParentOp());
  if (!loop)
    return {};
  auto window = queryScopedAccessReuse(facts, read->base, loop, limits);
  if (!window.access || window.access->reads.size() != 1)
    return {window.status, {}};
  auto sliding = getSlidingAccess(*read);
  return {sliding ? IndexRelationStatus::Exact
                  : IndexRelationStatus::Unsupported,
          std::move(sliding)};
}

AccessReuseAnalysis analyzeAccessReuse(mlir::ModuleOp module,
                                       const IndexRelationLimits &limits,
                                       uint64_t scopeQueryLimit) {
  support::ScopedCompileTimingSpan timing("analysis-phase", "access-reuse",
                                          "current-reads");
  if (!module)
    return {IndexRelationStatus::Invalid,
            "access reuse requires a current module"};
  auto result = collectReads(module);
  if (result.status != IndexRelationStatus::Exact)
    return result;
  std::map<std::pair<int64_t, int64_t>, llvm::SmallVector<size_t, 4>>
      groupsBySource;
  for (const ReadAccess &access : result.reads) {
    auto tile = access.tile;
    auto &indices = groupsBySource[{tile.getCardId(), access.argument}];
    auto found = llvm::find_if(indices, [&](size_t index) {
      const auto &group = result.peers[index];
      return equivalentPeerAccess(group.reads.front(), access) &&
             !llvm::any_of(group.reads, [&](const auto &read) {
               return read.tile == access.tile;
             });
    });
    if (found == indices.end()) {
      indices.push_back(result.peers.size());
      result.peers.emplace_back();
      found = std::prev(indices.end());
    }
    result.peers[*found].reads.push_back(access);
  }
  llvm::erase_if(result.peers,
                 [](const auto &group) { return group.reads.size() < 2; });
  collectScopedAccesses(result, limits, scopeQueryLimit);
  return result;
}

} // namespace wafer::analysis
