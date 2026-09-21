//===- GatherScatterWork.cpp - Ordered GS coalescing and splitting --------===//

#include "GatherScatterWork.h"
#include "Wafer/Analysis/Instr/StaticIndexRange.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/CompileTiming.h"
#include "Wafer/Target/GatherScatter.h"
#include "Wafer/Transforms/Passes.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <array>
#include <numeric>

namespace wafer {
#define GEN_PASS_DEF_MATERIALIZEGATHERSCATTERWORKPASS
#include "Wafer/Transforms/WaferTransformPasses.h.inc"
} // namespace wafer

using namespace wafer;

namespace {
using Triple = std::array<int64_t, 3>;

struct Endpoint {
  Triple strides;
  Triple iterations;
};

struct Segment {
  int64_t inner;
  int64_t bytes;
  int64_t sourceOffset;
  int64_t destOffset;
  Endpoint source;
  Endpoint dest;
};

static Triple triple(llvm::ArrayRef<int64_t> values) {
  return {values[0], values[1], values[2]};
}

// No interchange: these rewrites preserve the linear address stream.
static Endpoint normalize(Endpoint endpoint) {
  llvm::SmallVector<std::pair<int64_t, int64_t>, 3> axes;
  for (unsigned i = 0; i < 3; ++i) {
    if (endpoint.iterations[i] == 1)
      continue;
    int64_t nextStride = 0;
    if (!axes.empty() &&
        !llvm::MulOverflow(axes.back().first, axes.back().second, nextStride) &&
        nextStride == endpoint.strides[i]) {
      axes.back().second *= endpoint.iterations[i];
    } else {
      axes.emplace_back(endpoint.strides[i], endpoint.iterations[i]);
    }
  }
  Endpoint result{{0, 0, 0}, {1, 1, 1}};
  for (auto [index, axis] : llvm::enumerate(axes)) {
    result.strides[index] = axis.first;
    result.iterations[index] = axis.second;
  }
  return result;
}

static int64_t offsetAt(const Endpoint &endpoint, int64_t ordinal) {
  int64_t offset = 0;
  for (unsigned axis = 0; axis < 3; ++axis) {
    offset += ordinal % endpoint.iterations[axis] * endpoint.strides[axis];
    ordinal /= endpoint.iterations[axis];
  }
  return offset;
}

struct SliceLevel {
  int64_t multiple;
  int64_t maximum;
  unsigned axis;
};

static llvm::SmallVector<SliceLevel, 3> sliceLevels(const Endpoint &endpoint,
                                                    int64_t ordinal) {
  llvm::SmallVector<SliceLevel, 3> result;
  int64_t prefix = 1;
  for (unsigned axis = 0; axis < 3; ++axis) {
    if (ordinal % prefix != 0)
      break;
    int64_t coordinate = ordinal / prefix % endpoint.iterations[axis];
    result.push_back(
        {prefix, prefix * (endpoint.iterations[axis] - coordinate), axis});
    prefix *= endpoint.iterations[axis];
  }
  return result;
}

static Endpoint slice(Endpoint endpoint, SliceLevel level, int64_t length) {
  endpoint.iterations[level.axis] = length / level.multiple;
  for (unsigned i = level.axis + 1; i < 3; ++i)
    endpoint.iterations[i] = 1;
  return normalize(endpoint);
}

static llvm::SmallVector<Segment> split(InstrGatherScatterOp op) {
  Endpoint source =
      normalize({triple(op.getSrcStrides()), triple(op.getSrcIterations())});
  Endpoint dest =
      normalize({triple(op.getDstStrides()), triple(op.getDstIterations())});
  int64_t inner = op.getInnerBytes();
  int64_t common =
      std::gcd(source.strides[0] == inner ? source.iterations[0] : 1,
               dest.strides[0] == inner ? dest.iterations[0] : 1);
  if (common > 1) {
    inner *= common;
    for (Endpoint *endpoint : {&source, &dest}) {
      endpoint->iterations[0] /= common;
      endpoint->strides[0] *= common;
      *endpoint = normalize(*endpoint);
    }
  }

  llvm::SmallVector<Segment> segments;
  const int64_t count = op.getByteCount() / inner;
  if (inner > target::kGatherScatterMaxPayloadBytes) {
    // Keep all pieces of an inner transfer before advancing either endpoint.
    // The verified uint32 payload bounds this expansion independently of the
    // iteration shapes. Adjacent identical pieces are looped during emission.
    for (int64_t ordinal = 0; ordinal < count; ++ordinal)
      for (int64_t byte = 0; byte < inner;) {
        int64_t size =
            std::min(inner - byte, target::kGatherScatterMaxPayloadBytes);
        segments.push_back({size,
                            size,
                            offsetAt(source, ordinal) + byte,
                            offsetAt(dest, ordinal) + byte,
                            {{0, 0, 0}, {1, 1, 1}},
                            {{0, 0, 0}, {1, 1, 1}}});
        byte += size;
      }
    return segments;
  }

  int64_t limit = std::min(target::kGatherScatterMaxInnerTransfers,
                           target::kGatherScatterMaxPayloadBytes / inner);
  for (int64_t ordinal = 0; ordinal < count;) {
    // A prefix of a rectangular stream consists of full faster axes and a
    // partial current axis. Intersect both endpoints' allowed prefix lengths.
    // This also handles different source/destination radix decompositions.
    int64_t best = 0;
    SliceLevel sourceLevel{}, destLevel{};
    for (SliceLevel lhs : sliceLevels(source, ordinal))
      for (SliceLevel rhs : sliceLevels(dest, ordinal)) {
        int64_t multiple = std::lcm(lhs.multiple, rhs.multiple);
        int64_t length = std::min({limit, lhs.maximum, rhs.maximum});
        length -= length % multiple;
        if (length > best) {
          best = length;
          sourceLevel = lhs;
          destLevel = rhs;
        }
      }
    // Both endpoints always admit one original inner transfer.
    assert(best > 0 && "verified GS must admit an ordered prefix");
    segments.push_back({inner, inner * best, offsetAt(source, ordinal),
                        offsetAt(dest, ordinal),
                        slice(source, sourceLevel, best),
                        slice(dest, destLevel, best)});
    ordinal += best;
  }
  return segments;
}

static bool sameStructure(const Segment &lhs, const Segment &rhs) {
  return lhs.inner == rhs.inner && lhs.bytes == rhs.bytes &&
         lhs.source.strides == rhs.source.strides &&
         lhs.source.iterations == rhs.source.iterations &&
         lhs.dest.strides == rhs.dest.strides &&
         lhs.dest.iterations == rhs.dest.iterations;
}

static bool haveDisjointEndpoints(InstrGatherScatterOp op,
                                  mlir::AliasAnalysis &aliases) {
  if (aliases.alias(op.getSource(), op.getDest()).isNo())
    return true;
  if (op.getSource() != op.getDest())
    return false;
  using namespace memory_planning::detail;
  auto range = [&](bool source) -> std::optional<StaticIndexRange> {
    mlir::Value dynamic =
        source ? op.getSrcOffsetValue() : op.getDstOffsetValue();
    mlir::IntegerAttr fixed =
        source ? op.getSrcOffsetAttr() : op.getDstOffsetAttr();
    StaticIndexRange result{fixed ? fixed.getInt() : 0,
                            fixed ? fixed.getInt() : 0, false};
    if (dynamic) {
      auto evaluated = evaluateNonNegativeStaticIndexRange(dynamic, op);
      if (!evaluated.succeeded() || evaluated.range.empty)
        return std::nullopt;
      result = evaluated.range;
    }
    auto strides = source ? op.getSrcStrides() : op.getDstStrides();
    auto iterations = source ? op.getSrcIterations() : op.getDstIterations();
    int64_t span = op.getInnerBytes();
    for (unsigned i = 0; i < 3; ++i)
      span += strides[i] * (iterations[i] - 1);
    if (llvm::AddOverflow(result.max, span, result.max))
      return std::nullopt;
    return result;
  };
  auto source = range(true), dest = range(false);
  return source && dest &&
         (source->max <= dest->min || dest->max <= source->min);
}

static void emit(mlir::IRRewriter &rewriter, InstrGatherScatterOp op,
                 llvm::ArrayRef<Segment> segments) {
  const auto loc = op.getLoc();
  auto constant = [&](int64_t value) -> mlir::Value {
    return rewriter.create<mlir::arith::ConstantIndexOp>(loc, value);
  };
  auto emitOne = [&](const Segment &segment, mlir::Value induction,
                     int64_t sourceStep, int64_t destStep) {
    auto offset = [&](mlir::Value base, mlir::IntegerAttr fixed, int64_t delta,
                      int64_t step) -> mlir::Value {
      if (fixed)
        delta += fixed.getInt();
      mlir::Value value = base;
      if (!value)
        value = constant(delta);
      else if (delta)
        value =
            rewriter.create<mlir::arith::AddIOp>(loc, value, constant(delta));
      if (induction && step) {
        mlir::Value scaled = rewriter.create<mlir::arith::MulIOp>(
            loc, induction, constant(step));
        value = rewriter.create<mlir::arith::AddIOp>(loc, value, scaled);
      }
      return value;
    };
    auto newOp = rewriter.create<InstrGatherScatterOp>(
        loc, op.getSource(), op.getDest(),
        offset(op.getSrcOffsetValue(), op.getSrcOffsetAttr(),
               segment.sourceOffset, sourceStep),
        offset(op.getDstOffsetValue(), op.getDstOffsetAttr(),
               segment.destOffset, destStep),
        segment.bytes, segment.inner, mlir::IntegerAttr{}, mlir::IntegerAttr{},
        segment.source.strides, segment.source.iterations, segment.dest.strides,
        segment.dest.iterations, op.getDdrResourceAttr(), op.getWorker());
    (void)newOp;
  };
  rewriter.setInsertionPoint(op);
  for (size_t first = 0; first < segments.size();) {
    size_t end = first + 1;
    int64_t sourceStep = 0, destStep = 0;
    if (end < segments.size() &&
        sameStructure(segments[first], segments[end])) {
      sourceStep = segments[end].sourceOffset - segments[first].sourceOffset;
      destStep = segments[end].destOffset - segments[first].destOffset;
      ++end;
      while (end < segments.size() &&
             sameStructure(segments[first], segments[end]) &&
             segments[end].sourceOffset - segments[end - 1].sourceOffset ==
                 sourceStep &&
             segments[end].destOffset - segments[end - 1].destOffset ==
                 destStep)
        ++end;
    }
    if (end - first > 1) {
      auto loop = rewriter.create<mlir::scf::ForOp>(
          loc, constant(0), constant(end - first), constant(1));
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(loop.getBody());
        emitOne(segments[first], loop.getInductionVar(), sourceStep, destStep);
      }
    } else {
      emitOne(segments[first], {}, 0, 0);
    }
    first = end;
  }
  rewriter.eraseOp(op);
}

struct MaterializeGatherScatterWorkPass
    : wafer::impl::MaterializeGatherScatterWorkPassBase<
          MaterializeGatherScatterWorkPass> {
  void runOnOperation() override {
    if (!materializeGatherScatterWork(getOperation()).succeeded())
      signalPassFailure();
  }
};
} // namespace

GatherScatterWorkResult
wafer::materializeGatherScatterWork(mlir::func::FuncOp function) {
  support::ScopedCompileTimingSpan timing(
      "instr-transform", "gather-scatter-work", "materialization");
  GatherScatterWorkResult result;
  llvm::SmallVector<InstrGatherScatterOp> work;
  function.walk([&](InstrGatherScatterOp op) {
    if (!target::isGatherScatterIssueBounded(op.getByteCount(),
                                             op.getInnerBytes()))
      work.push_back(op);
  });
  mlir::IRRewriter rewriter(function.getContext());
  for (InstrGatherScatterOp op : work) {
    llvm::SmallVector<Segment> segments = split(op);
    // Fresh for each current IR epoch. Coalescing into one instruction retains
    // the same snapshot and does not require a disjointness proof.
    mlir::AliasAnalysis aliases(function);
    if (segments.size() > 1 && !haveDisjointEndpoints(op, aliases)) {
      op.emitError(
          "unsupported_gather_scatter_alias: splitting requires proven "
          "disjoint source and destination accesses");
      result.failure = GatherScatterWorkFailure::UnsupportedAliasing;
      return result;
    }
    result.issuedSegments += segments.size();
    emit(rewriter, op, segments);
    ++result.rewrittenOperations;
  }
  if (mlir::failed(mlir::verify(function)))
    result.failure = GatherScatterWorkFailure::InvalidIR;
  support::addCompileCounter("movement", "bounded-gather-scatter-operations",
                             result.rewrittenOperations);
  support::addCompileCounter("movement", "gather-scatter-segments",
                             result.issuedSegments);
  return result;
}
