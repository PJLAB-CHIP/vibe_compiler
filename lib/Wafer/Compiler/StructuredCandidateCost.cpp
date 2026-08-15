//===- StructuredCandidateCost.cpp - Structured candidate costs --------===//

#include "RankCandidateSearch.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <map>
#include <set>

namespace wafer::compiler::detail {
namespace {

using Dimension = StructuredCandidateCostDimension;
using Knowledge = StructuredCandidateMetricKnowledge;
using Metric = StructuredCandidateMetric;

static size_t index(Dimension dimension) {
  return static_cast<size_t>(dimension);
}

static std::string digest(llvm::StringRef text) {
  llvm::SHA256 hasher;
  hasher.update(text);
  return llvm::toHex(hasher.final(), /*LowerCase=*/true);
}

static void appendType(llvm::raw_ostream &stream, mlir::Type type) {
  type.print(stream);
}

static void appendTypes(llvm::raw_ostream &stream, mlir::TypeRange types) {
  stream << '[';
  for (mlir::Type type : types) {
    appendType(stream, type);
    stream << ';';
  }
  stream << ']';
}

static unsigned getOperationOrdinal(mlir::Operation *operation) {
  mlir::Block *block = operation ? operation->getBlock() : nullptr;
  if (!block)
    return 0;
  unsigned ordinal = 0;
  for (mlir::Operation &candidate : *block) {
    if (&candidate == operation)
      return ordinal;
    ++ordinal;
  }
  llvm_unreachable("operation is not present in its parent block");
}

static unsigned getRegionOrdinal(mlir::Operation *parent,
                                 mlir::Region *region) {
  unsigned ordinal = 0;
  for (mlir::Region &candidate : parent->getRegions()) {
    if (&candidate == region)
      return ordinal;
    ++ordinal;
  }
  llvm_unreachable("region is not present in its parent operation");
}

static unsigned getBlockOrdinal(mlir::Region *region, mlir::Block *block) {
  unsigned ordinal = 0;
  for (mlir::Block &candidate : *region) {
    if (&candidate == block)
      return ordinal;
    ++ordinal;
  }
  llvm_unreachable("block is not present in its parent region");
}

static mlir::Value resolveInvariantValue(mlir::Value value) {
  while (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    mlir::Block *owner = argument.getOwner();
    auto tileRegion =
        owner ? mlir::dyn_cast_or_null<TileRegionOp>(owner->getParentOp())
              : TileRegionOp{};
    if (tileRegion && !tileRegion.getBody().empty() &&
        owner == &tileRegion.getBody().front() &&
        argument.getArgNumber() < tileRegion.getInputs().size()) {
      value = tileRegion.getInputs()[argument.getArgNumber()];
      continue;
    }
    auto loop =
        owner ? mlir::dyn_cast_or_null<mlir::scf::ForOp>(owner->getParentOp())
              : mlir::scf::ForOp{};
    if (!loop || owner != loop.getBody() || argument.getArgNumber() == 0)
      break;
    unsigned carriedIndex = argument.getArgNumber() - 1;
    auto yield =
        mlir::dyn_cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
    if (!yield || carriedIndex >= loop.getInitArgs().size() ||
        carriedIndex >= yield.getNumOperands() ||
        yield.getOperand(carriedIndex) != argument)
      break;
    value = loop.getInitArgs()[carriedIndex];
  }
  return value;
}

static std::optional<int64_t> getProvenConstantIndex(mlir::Value value) {
  value = resolveInvariantValue(value);
  if (std::optional<int64_t> constant = mlir::getConstantIntValue(value))
    return constant;
  if (!value || !value.getType().isIndex())
    return std::nullopt;
  mlir::FailureOr<int64_t> constant =
      mlir::ValueBoundsConstraintSet::computeConstantBound(
          mlir::presburger::BoundType::EQ,
          mlir::ValueBoundsConstraintSet::Variable(value));
  return mlir::succeeded(constant) ? std::optional<int64_t>(*constant)
                                   : std::nullopt;
}

static void appendSymbolicIndex(llvm::raw_ostream &stream, mlir::Value value) {
  value = resolveInvariantValue(value);
  if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    mlir::Operation *parent =
        argument.getOwner() ? argument.getOwner()->getParentOp() : nullptr;
    stream << "arg(";
    if (parent)
      stream << parent->getName().getStringRef();
    else
      stream << "detached";
    stream << '#' << argument.getArgNumber() << ':';
    appendType(stream, value.getType());
    stream << ')';
    return;
  }
  if (auto result = mlir::dyn_cast<mlir::OpResult>(value)) {
    mlir::Operation *definition = result.getOwner();
    stream << "result(" << definition->getName().getStringRef() << '@'
           << getOperationOrdinal(definition) << '#' << result.getResultNumber()
           << ':';
    appendType(stream, value.getType());
    stream << ')';
    return;
  }
  stream << "unknown(";
  appendType(stream, value.getType());
  stream << ')';
}

static void appendControlPath(llvm::raw_ostream &stream,
                              mlir::Operation *operation) {
  struct Frame {
    mlir::Operation *parent = nullptr;
    unsigned operationOrdinal = 0;
    unsigned regionOrdinal = 0;
    unsigned blockOrdinal = 0;
  };
  llvm::SmallVector<Frame, 8> frames;
  for (mlir::Operation *child = operation; child && child->getParentOp();) {
    mlir::Operation *parent = child->getParentOp();
    mlir::Block *block = child->getBlock();
    mlir::Region *region = block ? block->getParent() : nullptr;
    if (!region || region->getParentOp() != parent)
      llvm_unreachable("operation ancestry is not structurally nested");
    frames.push_back({parent, getOperationOrdinal(parent),
                      getRegionOrdinal(parent, region),
                      getBlockOrdinal(region, block)});
    child = parent;
  }
  for (const Frame &frame : llvm::reverse(frames)) {
    mlir::Operation *parent = frame.parent;
    stream << parent->getName().getStringRef() << '@' << frame.operationOrdinal
           << "/r" << frame.regionOrdinal << "b" << frame.blockOrdinal << '/';
    if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(parent)) {
      stream << "for(";
      for (mlir::Value bound :
           {loop.getLowerBound(), loop.getUpperBound(), loop.getStep()}) {
        if (std::optional<int64_t> value = getProvenConstantIndex(bound))
          stream << *value;
        else
          appendSymbolicIndex(stream, bound);
        stream << ',';
      }
      stream << ")/";
    }
    if (auto conditional = mlir::dyn_cast<mlir::scf::IfOp>(parent)) {
      stream << "if(";
      if (std::optional<int64_t> condition =
              mlir::getConstantIntValue(conditional.getCondition()))
        stream << *condition;
      else
        stream << '?';
      stream << ")/";
    }
  }
}

static bool isStaticallyUnreachable(mlir::Operation *operation) {
  for (mlir::Operation *child = operation; child && child->getParentOp();) {
    mlir::Operation *parent = child->getParentOp();
    if (auto conditional = mlir::dyn_cast<mlir::scf::IfOp>(parent)) {
      std::optional<int64_t> condition =
          mlir::getConstantIntValue(conditional.getCondition());
      if (condition) {
        mlir::Region *selected = *condition ? &conditional.getThenRegion()
                                            : &conditional.getElseRegion();
        if (child->getParentRegion() != selected)
          return true;
      }
    }
    child = parent;
  }
  return false;
}

static void appendTypedOperation(llvm::raw_ostream &stream,
                                 mlir::Operation *operation,
                                 bool includeAttributes) {
  appendControlPath(stream, operation);
  stream << operation->getName().getStringRef() << ':';
  appendTypes(stream, operation->getOperandTypes());
  stream << "->";
  appendTypes(stream, operation->getResultTypes());
  if (includeAttributes) {
    stream << '{';
    for (mlir::NamedAttribute attribute : operation->getAttrs()) {
      llvm::StringRef name = attribute.getName().getValue();
      if (name == mlir::SymbolTable::getSymbolAttrName() ||
          name == kWaferSPMOffsetAttrName || name == kWaferDDROffsetAttrName)
        continue;
      // Symbol spelling and paths are not structured candidates facts. Calls
      // remain conservatively distinguished by the Unknown operation family
      // disposition below, rather than by a source-level callee name.
      if (mlir::isa<mlir::SymbolRefAttr>(attribute.getValue())) {
        stream << name << "=<symbol>;";
        continue;
      }
      stream << name << '=';
      attribute.getValue().print(stream);
      stream << ';';
    }
    stream << '}';
  }
  stream << '\n';
}

static bool isTileOperation(mlir::Operation *operation) {
  return mlir::isa<
      StorageLoadOp, StorageStoreOp, LayoutMaterializeOp, ComputeFillOp,
      ComputeConvertOp, ComputeGemmOp, ComputeElementwiseOp, ComputeReduceOp,
      MoveCopyOp, MoveExtractSliceOp, MoveInsertSliceOp, MoveTransposeOp,
      MoveBroadcastOp, ViewReshapeOp, CommPeerSendOp, CommPeerRecvOp,
      CommAllGatherOp, CommReduceScatterOp, CommAllReduceOp>(operation);
}

static bool isLocalMovement(mlir::Operation *operation) {
  return mlir::isa<LayoutMaterializeOp, MoveExtractSliceOp, MoveInsertSliceOp,
                   MoveCopyOp, MoveTransposeOp, MoveBroadcastOp>(operation);
}

static bool isTileCompute(mlir::Operation *operation) {
  return mlir::isa<ComputeFillOp, ComputeConvertOp, ComputeGemmOp,
                   ComputeElementwiseOp, ComputeReduceOp>(operation);
}

static bool isTileCommunication(mlir::Operation *operation) {
  return mlir::isa<CommPeerSendOp, CommPeerRecvOp, CommAllGatherOp,
                   CommReduceScatterOp, CommAllReduceOp>(operation);
}

static std::optional<uint64_t> getStaticTripCount(mlir::scf::ForOp loop) {
  std::optional<int64_t> lower = getProvenConstantIndex(loop.getLowerBound());
  std::optional<int64_t> upper = getProvenConstantIndex(loop.getUpperBound());
  std::optional<int64_t> step = getProvenConstantIndex(loop.getStep());
  if (!lower || !upper || !step || *step <= 0)
    return std::nullopt;
  if (*upper <= *lower)
    return 0;
  uint64_t distance = static_cast<uint64_t>(*upper - *lower);
  uint64_t positiveStep = static_cast<uint64_t>(*step);
  return 1 + (distance - 1) / positiveStep;
}

struct ExecutionMultiplicity {
  std::optional<uint64_t> exact = 1;
  Knowledge knowledge = Knowledge::Known;
  std::string disposition;
};

static ExecutionMultiplicity
getExecutionMultiplicity(mlir::Operation *operation) {
  ExecutionMultiplicity result;
  std::string control;
  llvm::raw_string_ostream stream(control);
  for (mlir::Operation *child = operation; child && child->getParentOp();) {
    mlir::Operation *parent = child->getParentOp();
    if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(parent)) {
      std::optional<uint64_t> tripCount = getStaticTripCount(loop);
      if (!tripCount) {
        result.exact.reset();
        result.knowledge = Knowledge::Unknown;
        appendControlPath(stream, operation);
        stream << "dynamic-loop";
        break;
      }
      if (*tripCount != 0 &&
          *result.exact > std::numeric_limits<uint64_t>::max() / *tripCount) {
        result.exact.reset();
        result.knowledge = Knowledge::Overflow;
        appendControlPath(stream, operation);
        stream << "loop-trip-product-overflow";
        break;
      }
      *result.exact *= *tripCount;
    } else if (auto conditional = mlir::dyn_cast<mlir::scf::IfOp>(parent)) {
      if (!mlir::getConstantIntValue(conditional.getCondition())) {
        result.exact.reset();
        result.knowledge = Knowledge::Unknown;
        appendControlPath(stream, operation);
        stream << "dynamic-conditional-control";
        break;
      }
    }
    child = parent;
  }
  stream.flush();
  if (!result.exact)
    result.disposition = digest(control);
  return result;
}

static std::optional<uint64_t> getCompactBytes(mlir::Type type) {
  auto memref = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memref)
    return std::nullopt;
  std::optional<WaferPhysicalTensorInfo> info =
      computeWaferPhysicalTensorInfo(memref);
  if (!info || info->compactBytes < 0)
    return std::nullopt;
  return static_cast<uint64_t>(info->compactBytes);
}

static std::optional<uint64_t> getPhysicalBytes(mlir::Type type) {
  auto memref = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memref)
    return std::nullopt;
  std::optional<WaferPhysicalTensorInfo> info =
      computeWaferPhysicalTensorInfo(memref);
  if (!info || info->physicalBytes < 0)
    return std::nullopt;
  return static_cast<uint64_t>(info->physicalBytes);
}

static std::optional<mlir::RankedTensorType>
getLogicalWaferTensorType(mlir::Type type) {
  auto memref = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memref)
    return std::nullopt;
  std::optional<WaferPhysicalTensorInfo> info =
      computeWaferPhysicalTensorInfo(memref);
  if (!info)
    return std::nullopt;
  return info->logicalTensorType;
}

static std::optional<uint64_t> getStaticElementCount(mlir::Type type) {
  std::optional<mlir::RankedTensorType> tensor =
      getLogicalWaferTensorType(type);
  if (!tensor || !tensor->hasStaticShape())
    return std::nullopt;
  uint64_t count = 1;
  for (int64_t extent : tensor->getShape()) {
    if (extent < 0 ||
        (extent != 0 && count > std::numeric_limits<uint64_t>::max() /
                                    static_cast<uint64_t>(extent)))
      return std::nullopt;
    count *= static_cast<uint64_t>(extent);
  }
  return count;
}

static std::optional<uint64_t>
checkedProduct(std::initializer_list<uint64_t> factors) {
  uint64_t product = 1;
  for (uint64_t factor : factors) {
    if (factor != 0 && product > std::numeric_limits<uint64_t>::max() / factor)
      return std::nullopt;
    product *= factor;
  }
  return product;
}

static std::optional<uint64_t>
getStaticComputeLogicalWork(mlir::Operation *operation) {
  if (mlir::isa<ComputeFillOp>(operation))
    return 0;
  if (auto convert = mlir::dyn_cast<ComputeConvertOp>(operation))
    return getStaticElementCount(convert.getResult().getType());
  if (auto elementwise = mlir::dyn_cast<ComputeElementwiseOp>(operation)) {
    // Constant-predicate select can become a pure movement before instruction
    // lowering. Keep that route Unknown here rather than charging arithmetic
    // work that may not exist in the final accepted IR.
    if (elementwise.getKind() == ComputeElementwiseKind::Select)
      return std::nullopt;
    return getStaticElementCount(elementwise.getResult().getType());
  }
  if (auto reduce = mlir::dyn_cast<ComputeReduceOp>(operation)) {
    std::optional<uint64_t> input =
        getStaticElementCount(reduce.getInput().getType());
    std::optional<uint64_t> output =
        getStaticElementCount(reduce.getResult().getType());
    if (!input || !output || *input < *output)
      return std::nullopt;
    return reduce.getKind() == ComputeReduceKind::Avg ? input
                                                      : *input - *output;
  }
  auto gemm = mlir::dyn_cast<ComputeGemmOp>(operation);
  if (!gemm)
    return std::nullopt;
  std::optional<mlir::RankedTensorType> lhs =
      getLogicalWaferTensorType(gemm.getLhs().getType());
  std::optional<mlir::RankedTensorType> rhs =
      getLogicalWaferTensorType(gemm.getRhs().getType());
  if (!lhs || !rhs)
    return std::nullopt;

  int64_t lhsMDim = 0;
  int64_t lhsKDim = 1;
  int64_t rhsNDim = 1;
  uint64_t batch = 1;
  if (lhs->getRank() == 2 && rhs->getRank() == 2) {
    lhsMDim = gemm.getLhsOrientation().value_or(GemmOrientation::Normal) ==
                      GemmOrientation::Normal
                  ? 0
                  : 1;
    lhsKDim = 1 - lhsMDim;
    rhsNDim = gemm.getRhsOrientation().value_or(GemmOrientation::Normal) ==
                      GemmOrientation::Normal
                  ? 1
                  : 0;
  } else {
    auto getIndex = [&](llvm::StringRef name) -> std::optional<int64_t> {
      auto value = gemm->getAttrOfType<mlir::IntegerAttr>(name);
      return value ? std::optional<int64_t>(value.getInt()) : std::nullopt;
    };
    std::optional<int64_t> mDim = getIndex("lhs_m_dim");
    std::optional<int64_t> kDim = getIndex("lhs_contracting_dim");
    std::optional<int64_t> nDim = getIndex("rhs_n_dim");
    std::optional<int64_t> batchCount = getIndex("batch_count");
    if (!mDim || !kDim || !nDim || !batchCount || *batchCount < 0)
      return std::nullopt;
    lhsMDim = *mDim;
    lhsKDim = *kDim;
    rhsNDim = *nDim;
    batch = static_cast<uint64_t>(*batchCount);
  }
  if (lhsMDim < 0 || lhsKDim < 0 || rhsNDim < 0 || lhsMDim >= lhs->getRank() ||
      lhsKDim >= lhs->getRank() || rhsNDim >= rhs->getRank())
    return std::nullopt;
  int64_t m = lhs->getDimSize(lhsMDim);
  int64_t k = lhs->getDimSize(lhsKDim);
  int64_t n = rhs->getDimSize(rhsNDim);
  if (m < 0 || k < 0 || n < 0)
    return std::nullopt;
  return checkedProduct({2, static_cast<uint64_t>(m), static_cast<uint64_t>(k),
                         static_cast<uint64_t>(n), batch});
}

static std::optional<uint64_t>
getStaticComputeUnderutilization(mlir::Operation *operation) {
  mlir::Type resultType;
  if (auto fill = mlir::dyn_cast<ComputeFillOp>(operation))
    resultType = fill.getDest().getType();
  else if (operation->getNumResults() == 1)
    resultType = operation->getResult(0).getType();
  else
    return std::nullopt;

  auto memref = mlir::dyn_cast<mlir::MemRefType>(resultType);
  if (!memref)
    return std::nullopt;
  std::optional<WaferPhysicalTensorInfo> info =
      computeWaferPhysicalTensorInfo(memref);
  std::optional<uint64_t> logical = getStaticElementCount(resultType);
  if (!info || !logical || info->physicalElements < 0 ||
      static_cast<uint64_t>(info->physicalElements) < *logical)
    return std::nullopt;
  return static_cast<uint64_t>(info->physicalElements) - *logical;
}

static std::optional<uint64_t>
getEstimatedCommunicationPhases(mlir::Operation *operation) {
  if (mlir::isa<CommPeerSendOp, CommPeerRecvOp>(operation))
    return 1;
  if (!mlir::isa<CommAllGatherOp, CommReduceScatterOp, CommAllReduceOp>(
          operation))
    return std::nullopt;
  auto groupSize = operation->getAttrOfType<mlir::IntegerAttr>("group_size");
  if (!groupSize || groupSize.getInt() <= 0)
    return std::nullopt;
  uint64_t rounds = static_cast<uint64_t>(groupSize.getInt() - 1);
  if (!mlir::isa<CommAllReduceOp>(operation))
    return rounds;
  return checkedProduct({2, rounds});
}

struct MetricAccumulator {
  uint64_t value = 0;
  Knowledge knowledge = Knowledge::Known;
  std::string dispositionSource;
  std::set<std::string> estimateModels;

  bool addContribution(uint64_t amount, uint64_t multiplier,
                       llvm::StringRef identity) {
    if (knowledge == Knowledge::Overflow)
      return false;
    if (multiplier != 0 &&
        amount > std::numeric_limits<uint64_t>::max() / multiplier) {
      markOverflow(identity);
      return false;
    }
    uint64_t contribution = amount * multiplier;
    if (contribution > std::numeric_limits<uint64_t>::max() - value) {
      markOverflow(identity);
      return false;
    }
    value += contribution;
    return true;
  }

  void add(uint64_t amount, uint64_t multiplier, llvm::StringRef identity) {
    dispositionSource.append(identity);
    dispositionSource.push_back(':');
    dispositionSource.append(llvm::Twine(amount).str());
    dispositionSource.push_back('x');
    dispositionSource.append(llvm::Twine(multiplier).str());
    dispositionSource.push_back('\n');
    (void)addContribution(amount, multiplier, identity);
  }

  void addEstimate(uint64_t amount, uint64_t multiplier,
                   llvm::StringRef identity, llvm::StringRef model) {
    if (!addContribution(amount, multiplier, identity))
      return;
    if (knowledge == Knowledge::Known)
      knowledge = Knowledge::Estimated;
    if (knowledge == Knowledge::Estimated)
      estimateModels.insert(model.str());
  }

  void markUnknown(llvm::StringRef identity) {
    if (knowledge == Knowledge::Overflow || knowledge == Knowledge::Unsupported)
      return;
    knowledge = Knowledge::Unknown;
    dispositionSource.append(identity);
    dispositionSource.push_back('\n');
  }

  void markOverflow(llvm::StringRef identity) {
    knowledge = Knowledge::Overflow;
    dispositionSource.append(identity);
    dispositionSource.push_back('\n');
  }

  Metric finish() const {
    Metric result;
    result.value =
        knowledge == Knowledge::Known || knowledge == Knowledge::Estimated
            ? value
            : 0;
    result.knowledge = knowledge;
    if (knowledge == Knowledge::Estimated) {
      std::string models;
      for (const std::string &model : estimateModels) {
        models.append(model);
        models.push_back('\n');
      }
      result.disposition = digest(models);
    } else if (knowledge != Knowledge::Known) {
      result.disposition = digest(dispositionSource);
    }
    return result;
  }
};

struct RankFacts {
  std::array<Metric, kStructuredCandidateCostDimensionCount> selection;
  std::string futureLiveInterface;
  std::string physicalVersions;
  std::string loopReuseAndEffects;
  std::string collectiveAndPeerInterface;
};

static RankFacts deriveRankFacts(const CoordinatedRankTileModule &rank) {
  std::array<MetricAccumulator, kStructuredCandidateCostDimensionCount>
      accumulators;
  std::string futureText;
  std::string physicalText;
  std::string effectText;
  std::string communicationText;
  std::string allTileText;
  std::string ssaText;
  llvm::raw_string_ostream future(futureText);
  llvm::raw_string_ostream physical(physicalText);
  llvm::raw_string_ostream effects(effectText);
  llvm::raw_string_ostream communication(communicationText);
  llvm::raw_string_ostream allTile(allTileText);
  llvm::raw_string_ostream ssa(ssaText);
  llvm::DenseMap<mlir::Value, uint64_t> valueOrdinals;
  llvm::DenseMap<mlir::Operation *, uint64_t> tileDependencyDepths;
  uint64_t nextValueOrdinal = 0;
  uint64_t criticalPathLowerBound = 0;
  bool sawCompute = false;
  auto getValueOrdinal = [&](mlir::Value value) {
    auto [entry, inserted] = valueOrdinals.try_emplace(value, nextValueOrdinal);
    if (inserted)
      ++nextValueOrdinal;
    return entry->second;
  };

  future << "rank-domain-entry\n";
  for (mlir::func::FuncOp function :
       rank.module.get().getOps<mlir::func::FuncOp>()) {
    if (function.isExternal())
      continue;
    future << "function:";
    appendTypes(future, function.getArgumentTypes());
    future << "->";
    appendTypes(future, function.getResultTypes());
    future << '\n';
    for (mlir::BlockArgument argument : function.getArguments())
      (void)getValueOrdinal(argument);
  }

  unsigned regionOrdinal = 0;
  rank.module.get().walk([&](mlir::Operation *operation) {
    if (isStaticallyUnreachable(operation))
      return;
    for (mlir::Region &region : operation->getRegions())
      for (mlir::Block &block : region)
        for (mlir::BlockArgument argument : block.getArguments())
          (void)getValueOrdinal(argument);
    for (mlir::Value result : operation->getResults())
      (void)getValueOrdinal(result);

    if (mlir::isa<TileRegionOp>(operation) ||
        operation->getParentOfType<TileRegionOp>()) {
      appendControlPath(ssa, operation);
      ssa << operation->getName().getStringRef() << ":operands=";
      for (mlir::Value operand : operation->getOperands())
        ssa << getValueOrdinal(operand) << ',';
      ssa << ":results=";
      for (mlir::Value result : operation->getResults())
        ssa << getValueOrdinal(result) << ',';
      ssa << ':';
      appendTypes(ssa, operation->getOperandTypes());
      ssa << "->";
      appendTypes(ssa, operation->getResultTypes());
      ssa << '\n';
    }

    if (auto region = mlir::dyn_cast<TileRegionOp>(operation)) {
      future << "region:" << regionOrdinal << ':';
      appendTypes(future, region.getOperandTypes());
      future << "->";
      appendTypes(future, region.getResultTypes());
      future << '\n';
      physical << "region:" << regionOrdinal++ << ':';
      appendTypes(physical, region.getOperandTypes());
      physical << "->";
      appendTypes(physical, region.getResultTypes());
      physical << '\n';
      return;
    }
    if (auto allocation = mlir::dyn_cast<mlir::memref::AllocOp>(operation)) {
      physical << "alloc:";
      appendType(physical, allocation.getType());
      physical << ':';
      appendControlPath(physical, operation);
      physical << '\n';
      accumulators[index(Dimension::ResourcePressure)].addEstimate(
          1, 1, "memref.alloc", "static-buffer-sites");
      return;
    }
    if (!isTileOperation(operation))
      return;

    uint64_t dependencyDepth = 1;
    for (mlir::Value operand : operation->getOperands()) {
      mlir::Operation *definition =
          resolveInvariantValue(operand).getDefiningOp();
      auto found = tileDependencyDepths.find(definition);
      if (found != tileDependencyDepths.end() &&
          found->second < std::numeric_limits<uint64_t>::max())
        dependencyDepth = std::max(dependencyDepth, found->second + 1);
    }
    tileDependencyDepths[operation] = dependencyDepth;
    criticalPathLowerBound = std::max(criticalPathLowerBound, dependencyDepth);

    appendTypedOperation(allTile, operation, /*includeAttributes=*/true);
    if (isTileCommunication(operation))
      appendTypedOperation(communication, operation,
                           /*includeAttributes=*/true);

    // Completion depends on effect kind and control placement, not on source
    // spelling. Keep this signature exact and conservative until C3 turns the
    // obligation into Known final-IR waits and critical-path work.
    appendTypedOperation(effects, operation, /*includeAttributes=*/false);

    ExecutionMultiplicity multiplicity = getExecutionMultiplicity(operation);
    auto addKnownOrEstimated = [&](Dimension dimension,
                                   std::optional<uint64_t> amount,
                                   llvm::StringRef identity) {
      MetricAccumulator &accumulator = accumulators[index(dimension)];
      if (multiplicity.knowledge == Knowledge::Overflow) {
        accumulator.markOverflow(identity);
        return;
      }
      if (!amount) {
        accumulator.markUnknown(identity);
        return;
      }
      if (!multiplicity.exact) {
        std::string model = (llvm::Twine("static-site-dynamic-control:") +
                             multiplicity.disposition)
                                .str();
        accumulator.addEstimate(*amount, 1, identity, model);
        return;
      }
      accumulator.add(*amount, *multiplicity.exact, identity);
    };
    auto addEstimated = [&](Dimension dimension, std::optional<uint64_t> amount,
                            llvm::StringRef identity, llvm::StringRef model) {
      MetricAccumulator &accumulator = accumulators[index(dimension)];
      if (multiplicity.knowledge == Knowledge::Overflow) {
        accumulator.markOverflow(identity);
        return;
      }
      if (!amount) {
        accumulator.markUnknown(identity);
        return;
      }
      uint64_t modeledMultiplicity = multiplicity.exact.value_or(1);
      std::string modelKey = model.str();
      if (!multiplicity.exact) {
        modelKey.append(":static-site-dynamic-control:");
        modelKey.append(multiplicity.disposition);
      }
      accumulator.addEstimate(*amount, modeledMultiplicity, identity, modelKey);
    };

    std::optional<uint64_t> communicationPhases =
        getEstimatedCommunicationPhases(operation);
    std::optional<uint64_t> instructionSites = 1;
    if (mlir::isa<ViewReshapeOp>(operation)) {
      instructionSites = 0;
    } else if (isTileCommunication(operation)) {
      instructionSites = communicationPhases
                             ? std::optional<uint64_t>(
                                   std::max<uint64_t>(1, *communicationPhases))
                             : std::nullopt;
    }
    addEstimated(Dimension::InstrAggregateWork, instructionSites,
                 operation->getName().getStringRef(),
                 "tile-action-instruction-site-model");
    addEstimated(Dimension::CompletionMaximumRankWaitWork, instructionSites,
                 operation->getName().getStringRef(),
                 "tile-action-completion-obligation-model");
    if (mlir::isa<StorageLoadOp, StorageStoreOp>(operation) ||
        isLocalMovement(operation) || isTileCommunication(operation))
      addEstimated(Dimension::DescriptorPressure, instructionSites,
                   operation->getName().getStringRef(),
                   "tile-action-descriptor-site-model");

    if (isTileCompute(operation)) {
      sawCompute = true;
      std::optional<uint64_t> logicalWork =
          getStaticComputeLogicalWork(operation);
      llvm::StringRef identity = operation->getName().getStringRef();
      if (logicalWork) {
        addKnownOrEstimated(Dimension::ComputeAggregateWork, logicalWork,
                            identity);
        addKnownOrEstimated(Dimension::ComputeMaximumRankWork, logicalWork,
                            identity);
      } else if (auto select = mlir::dyn_cast<ComputeElementwiseOp>(operation);
                 select && select.getKind() == ComputeElementwiseKind::Select) {
        std::optional<uint64_t> selectWork =
            getStaticElementCount(select.getResult().getType());
        addEstimated(Dimension::ComputeAggregateWork, selectWork, identity,
                     "select-element-operation-proxy");
        addEstimated(Dimension::ComputeMaximumRankWork, selectWork, identity,
                     "select-element-operation-proxy");
      } else {
        addKnownOrEstimated(Dimension::ComputeAggregateWork, std::nullopt,
                            identity);
        addKnownOrEstimated(Dimension::ComputeMaximumRankWork, std::nullopt,
                            identity);
      }
      addKnownOrEstimated(Dimension::TileUnderutilization,
                          getStaticComputeUnderutilization(operation),
                          identity);
    }

    if (auto load = mlir::dyn_cast<StorageLoadOp>(operation)) {
      addKnownOrEstimated(Dimension::DDRAggregateReadBytes,
                          getCompactBytes(load.getSource().getType()),
                          "tile.load-bytes");
      addEstimated(Dimension::DDRMaximumRankIssueWork, 1, "tile.load",
                   "one-ddr-issue-per-tile-request");
    } else if (auto store = mlir::dyn_cast<StorageStoreOp>(operation)) {
      addKnownOrEstimated(Dimension::DDRAggregateWriteBytes,
                          getCompactBytes(store.getDest().getType()),
                          "tile.store-bytes");
      addEstimated(Dimension::DDRMaximumRankIssueWork, 1, "tile.store",
                   "one-ddr-issue-per-tile-request");
    }

    if (isLocalMovement(operation)) {
      mlir::Type movedType = operation->getNumResults() != 0
                                 ? operation->getResult(0).getType()
                                 : operation->getOperand(0).getType();
      std::optional<uint64_t> bytes = getPhysicalBytes(movedType);
      addKnownOrEstimated(Dimension::LocalAggregateMovementBytes, bytes,
                          operation->getName().getStringRef());
      addKnownOrEstimated(Dimension::LocalMaximumRankMovementBytes, bytes,
                          operation->getName().getStringRef());
      addKnownOrEstimated(Dimension::SPMAggregateMovementBytes, bytes,
                          operation->getName().getStringRef());
      addKnownOrEstimated(Dimension::SPMMaximumRankMovementBytes, bytes,
                          operation->getName().getStringRef());
    }

    if (isTileCommunication(operation)) {
      auto bytes = operation->getAttrOfType<mlir::IntegerAttr>("bytes");
      std::optional<uint64_t> payload;
      if (bytes && bytes.getInt() >= 0)
        payload = static_cast<uint64_t>(bytes.getInt());
      // Peer and collective request payloads are explicit logical bytes in
      // current Tile IR. Algorithm-specific routing remains owned by
      // executable evaluation;
      // the logical phase/link/endpoint models below provide only
      // comparable pre-Instr estimates and are replaced by final recost.
      addKnownOrEstimated(Dimension::NoCAggregatePayloadBytes, payload,
                          operation->getName().getStringRef());
      std::optional<uint64_t> linkWork;
      if (payload && communicationPhases) {
        if (mlir::isa<CommPeerRecvOp>(operation))
          linkWork = 0;
        else
          linkWork = checkedProduct({*payload, *communicationPhases});
      }
      addEstimated(Dimension::NoCLinkWork, linkWork,
                   operation->getName().getStringRef(),
                   "logical-one-hop-link-byte-model");
      addEstimated(Dimension::NoCEndpointWork, communicationPhases,
                   operation->getName().getStringRef(),
                   "logical-communication-phase-endpoint-model");
      addKnownOrEstimated(Dimension::AllRankCouplingWork, 1,
                          operation->getName().getStringRef());
    }
  });

  future.flush();
  physical.flush();
  effects.flush();
  communication.flush();
  allTile.flush();
  ssa.flush();

  if (!allTileText.empty()) {
    const std::string tileKey = digest(allTileText);
    accumulators[index(Dimension::CriticalPathLowerBound)].add(
        criticalPathLowerBound, 1, "tile-ssa-dependency-lower-bound");
    accumulators[index(Dimension::ResourcePressure)].addEstimate(
        0, 1, tileKey, "static-buffer-sites");
  }
  if (sawCompute) {
    // Current IR has no durable recompute role, while exact aggregate compute
    // above already charges every actual operation. Model the unlabelled
    // recompute portion as zero and keep the model identity explicit; final
    // Instr recost still charges every materialized compute instruction.
    accumulators[index(Dimension::RecomputeAggregateWork)].addEstimate(
        0, 1, "recompute-role-not-explicit-total-compute-exact",
        "no-explicit-recompute-role-model");
  }
  RankFacts result;
  for (size_t metricIndex = 0;
       metricIndex < kStructuredCandidateCostDimensionCount; ++metricIndex)
    result.selection[metricIndex] = accumulators[metricIndex].finish();
  result.futureLiveInterface = digest(futureText);
  result.physicalVersions = digest(physicalText + ssaText);
  result.loopReuseAndEffects = digest(effectText);
  result.collectiveAndPeerInterface = digest(communicationText);
  return result;
}

static Metric combineMetrics(llvm::ArrayRef<RankFacts> ranks,
                             Dimension dimension, bool maximum) {
  Metric result;
  bool allModeled = llvm::all_of(ranks, [&](const RankFacts &rank) {
    Knowledge knowledge = rank.selection[index(dimension)].knowledge;
    return knowledge == Knowledge::Known || knowledge == Knowledge::Estimated;
  });
  std::string disposition;
  if (!allModeled) {
    Knowledge combinedKnowledge = Knowledge::Known;
    for (auto [rankOrdinal, rank] : llvm::enumerate(ranks)) {
      const Metric &metric = rank.selection[index(dimension)];
      if (static_cast<unsigned>(metric.knowledge) >
          static_cast<unsigned>(combinedKnowledge))
        combinedKnowledge = metric.knowledge;
      disposition.append(llvm::Twine(rankOrdinal).str());
      disposition.push_back(':');
      disposition.append(
          llvm::Twine(static_cast<unsigned>(metric.knowledge)).str());
      disposition.push_back(':');
      disposition.append(llvm::Twine(metric.value).str());
      disposition.push_back(':');
      disposition.append(metric.disposition);
      disposition.push_back('\n');
    }
    result.knowledge = combinedKnowledge;
    result.disposition = digest(disposition);
    return result;
  }
  bool estimated = false;
  std::string models;
  for (const RankFacts &rank : ranks) {
    const Metric &metric = rank.selection[index(dimension)];
    if (metric.knowledge == Knowledge::Estimated) {
      estimated = true;
      models.append(metric.disposition);
      models.push_back('\n');
    }
    if (maximum) {
      result.value = std::max(result.value, metric.value);
    } else if (metric.value >
               std::numeric_limits<uint64_t>::max() - result.value) {
      result.value = 0;
      result.knowledge = Knowledge::Overflow;
      disposition.append("all-rank-reduction-overflow\n");
    } else {
      result.value += metric.value;
    }
  }
  if (result.knowledge == Knowledge::Overflow) {
    result.value = 0;
    result.disposition = digest(disposition);
  } else if (estimated) {
    result.knowledge = Knowledge::Estimated;
    result.disposition = digest(models);
  }
  return result;
}

static bool usesMaximumReduction(Dimension dimension) {
  switch (dimension) {
  case Dimension::DDRMaximumRankIssueWork:
  case Dimension::LocalMaximumRankMovementBytes:
  case Dimension::CompletionMaximumRankWaitWork:
  case Dimension::CriticalPathLowerBound:
  case Dimension::SPMMaximumRankMovementBytes:
  case Dimension::ComputeMaximumRankWork:
  case Dimension::TileUnderutilization:
  case Dimension::DescriptorPressure:
  case Dimension::ResourcePressure:
    return true;
  case Dimension::DDRAggregateReadBytes:
  case Dimension::DDRAggregateWriteBytes:
  case Dimension::LocalAggregateMovementBytes:
  case Dimension::NoCAggregatePayloadBytes:
  case Dimension::NoCLinkWork:
  case Dimension::NoCEndpointWork:
  case Dimension::SPMAggregateMovementBytes:
  case Dimension::ComputeAggregateWork:
  case Dimension::RecomputeAggregateWork:
  case Dimension::InstrAggregateWork:
  case Dimension::AllRankCouplingWork:
    return false;
  case Dimension::Count:
    break;
  }
  llvm_unreachable("invalid structured candidates dimension");
}

static std::string combineSignatures(
    llvm::ArrayRef<RankFacts> ranks,
    llvm::function_ref<llvm::StringRef(const RankFacts &)> select) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  for (auto [rankOrdinal, rank] : llvm::enumerate(ranks))
    stream << "rank:" << rankOrdinal << ':' << select(rank) << '\n';
  stream.flush();
  return digest(text);
}

enum class MetricOrder : uint8_t {
  Left,
  Right,
  Equivalent,
  Incomparable,
};

static MetricOrder compareMetric(const Metric &left, const Metric &right) {
  if (left.knowledge != right.knowledge ||
      left.disposition != right.disposition)
    return MetricOrder::Incomparable;
  if (left.knowledge != Knowledge::Known &&
      left.knowledge != Knowledge::Estimated)
    return MetricOrder::Equivalent;
  if (left.value < right.value)
    return MetricOrder::Left;
  if (right.value < left.value)
    return MetricOrder::Right;
  return MetricOrder::Equivalent;
}

struct CollectiveRecord {
  mlir::Operation *operation = nullptr;
  int64_t logicalRank = -1;
  int64_t localRank = -1;
  llvm::SmallVector<int64_t, 16> rankGroup;
};

struct PeerEndpointSet {
  mlir::Operation *anchor = nullptr;
  llvm::SmallVector<std::string, 4> sends;
  llvm::SmallVector<std::string, 4> receives;
};

static std::string getCollectiveSharedSignature(mlir::Operation *operation) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  appendControlPath(stream, operation);
  stream << operation->getName().getStringRef() << ':';
  appendTypes(stream, operation->getOperandTypes());
  stream << "->";
  appendTypes(stream, operation->getResultTypes());
  stream << '{';
  for (mlir::NamedAttribute attribute : operation->getAttrs()) {
    llvm::StringRef name = attribute.getName().getValue();
    if (name == "local_rank" || name == kWaferSPMOffsetAttrName ||
        name == kWaferDDROffsetAttrName)
      continue;
    stream << name << '=';
    attribute.getValue().print(stream);
    stream << ';';
  }
  stream << '}';
  stream.flush();
  return digest(text);
}

static std::string getPeerEndpointSignature(mlir::Operation *operation) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  appendControlPath(stream, operation);
  appendType(stream, operation->getOperand(0).getType());
  stream.flush();
  return digest(text);
}

static mlir::LogicalResult emitCommunicationError(mlir::Operation *operation,
                                                  llvm::StringRef message) {
  operation->emitError() << "coordinated structured communication: " << message;
  return mlir::failure();
}

static mlir::LogicalResult
verifyCommunicationControlPath(mlir::Operation *operation) {
  for (mlir::Operation *parent = operation->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (mlir::isa<mlir::ModuleOp, mlir::func::FuncOp, TileRegionOp>(parent))
      continue;
    if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(parent)) {
      // A non-constant structured loop remains a symbolic communication
      // occurrence. appendControlPath records the exact typed SSA origin of
      // all three bounds; participant/endpoint grouping below then requires
      // every rank to carry the same symbolic occurrence. It is never counted
      // as zero: the selection facts retain an Unknown multiplicity.
      (void)loop;
      continue;
    }
    if (auto conditional = mlir::dyn_cast<mlir::scf::IfOp>(parent)) {
      if (!mlir::getConstantIntValue(conditional.getCondition()))
        return emitCommunicationError(
            operation,
            "peer/collective endpoint has an unproved dynamic condition");
      continue;
    }
    return emitCommunicationError(
        operation, "peer/collective endpoint is nested in unsupported control");
  }
  return mlir::success();
}

} // namespace

mlir::FailureOr<StructuredCandidateCost>
deriveStructuredCandidateCost(const CoordinatedTileVariant &variant) {
  if (variant.ranks.empty())
    return mlir::failure();
  llvm::SmallVector<RankFacts, 8> ranks;
  ranks.reserve(variant.ranks.size());
  for (auto [expectedRank, rank] : llvm::enumerate(variant.ranks)) {
    if (!rank.module || rank.logicalRank != static_cast<int64_t>(expectedRank))
      return mlir::failure();
    ranks.push_back(deriveRankFacts(rank));
  }

  StructuredCandidateCost result;
  for (size_t metricIndex = 0;
       metricIndex < kStructuredCandidateCostDimensionCount; ++metricIndex) {
    Dimension dimension = static_cast<Dimension>(metricIndex);
    result.selection[metricIndex] =
        combineMetrics(ranks, dimension, usesMaximumReduction(dimension));
  }
  result.futureLiveInterface = combineSignatures(
      ranks, [](const RankFacts &rank) { return rank.futureLiveInterface; });
  result.physicalVersions = combineSignatures(
      ranks, [](const RankFacts &rank) { return rank.physicalVersions; });
  result.loopReuseAndEffects = combineSignatures(
      ranks, [](const RankFacts &rank) { return rank.loopReuseAndEffects; });
  result.collectiveAndPeerInterface =
      combineSignatures(ranks, [](const RankFacts &rank) {
        return rank.collectiveAndPeerInterface;
      });
  return result;
}

mlir::LogicalResult
verifyCoordinatedStructuredCommunication(const CoordinatedTileVariant &variant,
                                         int64_t expectedRankCount) {
  if (expectedRankCount <= 0 ||
      variant.ranks.size() != static_cast<size_t>(expectedRankCount))
    return mlir::failure();

  std::map<std::string, llvm::SmallVector<CollectiveRecord, 16>> collectives;
  std::map<std::string, PeerEndpointSet> peers;
  for (auto [expectedRank, rank] : llvm::enumerate(variant.ranks)) {
    if (!rank.module || rank.logicalRank != static_cast<int64_t>(expectedRank))
      return mlir::failure();
    std::map<std::string, uint64_t> collectiveOccurrences;
    mlir::LogicalResult rankResult = mlir::success();
    rank.module.get().walk([&](mlir::Operation *operation) {
      if (mlir::failed(rankResult))
        return mlir::WalkResult::interrupt();
      if (isStaticallyUnreachable(operation))
        return mlir::WalkResult::advance();
      if (mlir::isa<CommPeerSendOp, CommPeerRecvOp>(operation)) {
        if (mlir::failed(verifyCommunicationControlPath(operation))) {
          rankResult = mlir::failure();
          return mlir::WalkResult::interrupt();
        }
        auto peer = operation->getAttrOfType<mlir::IntegerAttr>("peer");
        auto bytes = operation->getAttrOfType<mlir::IntegerAttr>("bytes");
        mlir::Attribute message = operation->getAttr("message");
        if (!peer || !bytes || !message || peer.getInt() < 0 ||
            peer.getInt() >= expectedRankCount || bytes.getInt() <= 0) {
          rankResult = emitCommunicationError(
              operation, "peer endpoint is outside the complete rank domain");
          return mlir::WalkResult::interrupt();
        }
        int64_t source = mlir::isa<CommPeerSendOp>(operation) ? rank.logicalRank
                                                              : peer.getInt();
        int64_t destination = mlir::isa<CommPeerSendOp>(operation)
                                  ? peer.getInt()
                                  : rank.logicalRank;
        std::string keyText;
        llvm::raw_string_ostream key(keyText);
        key << source << "->" << destination << ":bytes=" << bytes.getInt()
            << ":message=";
        message.print(key);
        key.flush();
        PeerEndpointSet &endpoints = peers[digest(keyText)];
        if (!endpoints.anchor)
          endpoints.anchor = operation;
        std::string endpointSignature = getPeerEndpointSignature(operation);
        (mlir::isa<CommPeerSendOp>(operation) ? endpoints.sends
                                              : endpoints.receives)
            .push_back(std::move(endpointSignature));
        return mlir::WalkResult::advance();
      }
      if (!mlir::isa<CommAllGatherOp, CommReduceScatterOp, CommAllReduceOp>(
              operation))
        return mlir::WalkResult::advance();

      if (mlir::failed(verifyCommunicationControlPath(operation))) {
        rankResult = mlir::failure();
        return mlir::WalkResult::interrupt();
      }

      auto localRank =
          operation->getAttrOfType<mlir::IntegerAttr>("local_rank");
      auto groupSize =
          operation->getAttrOfType<mlir::IntegerAttr>("group_size");
      auto rankGroup =
          operation->getAttrOfType<mlir::DenseI64ArrayAttr>("rank_group");
      if (!localRank || !groupSize || !rankGroup || groupSize.getInt() <= 0 ||
          static_cast<size_t>(groupSize.getInt()) != rankGroup.size() ||
          localRank.getInt() < 0 || localRank.getInt() >= groupSize.getInt()) {
        rankResult = emitCommunicationError(
            operation, "collective parameters are not fully typed");
        return mlir::WalkResult::interrupt();
      }
      llvm::ArrayRef<int64_t> group = rankGroup.asArrayRef();
      if (llvm::any_of(group,
                       [&](int64_t member) {
                         return member < 0 || member >= expectedRankCount;
                       }) ||
          group[localRank.getInt()] != rank.logicalRank) {
        rankResult = emitCommunicationError(
            operation,
            "collective local rank does not identify this logical rank");
        return mlir::WalkResult::interrupt();
      }
      std::string shared = getCollectiveSharedSignature(operation);
      uint64_t occurrence = collectiveOccurrences[shared]++;
      std::string key =
          (llvm::Twine(shared) + ":occurrence=" + llvm::Twine(occurrence))
              .str();
      collectives[key].push_back(
          {operation, rank.logicalRank, localRank.getInt(),
           llvm::SmallVector<int64_t, 16>(group.begin(), group.end())});
      return mlir::WalkResult::advance();
    });
    if (mlir::failed(rankResult))
      return mlir::failure();
  }

  for (auto &[key, records] : collectives) {
    (void)key;
    if (records.empty())
      continue;
    llvm::ArrayRef<int64_t> group = records.front().rankGroup;
    if (records.size() != group.size())
      return emitCommunicationError(
          records.front().operation,
          "collective occurrence does not have all-and-only participants");
    llvm::SmallVector<bool, 16> observed(group.size(), false);
    for (const CollectiveRecord &record : records) {
      auto position = llvm::find(group, record.logicalRank);
      if (position == group.end())
        return emitCommunicationError(
            record.operation,
            "collective occurrence contains a non-participant rank");
      size_t groupIndex = static_cast<size_t>(position - group.begin());
      if (observed[groupIndex] ||
          record.localRank != static_cast<int64_t>(groupIndex))
        return emitCommunicationError(
            record.operation,
            "collective participant is duplicated or has mismatched local "
            "rank");
      observed[groupIndex] = true;
    }
    if (!llvm::all_of(observed, [](bool value) { return value; }))
      return emitCommunicationError(
          records.front().operation,
          "collective occurrence is missing a rank-group participant");
  }

  for (auto &[key, endpoints] : peers) {
    (void)key;
    if (endpoints.sends.size() != endpoints.receives.size())
      return emitCommunicationError(
          endpoints.anchor,
          "peer message has different static send and receive counts");
    llvm::sort(endpoints.sends);
    llvm::sort(endpoints.receives);
    if (endpoints.sends != endpoints.receives)
      return emitCommunicationError(
          endpoints.anchor,
          "peer message payload type or structured control path differs");
  }
  return mlir::success();
}

StructuredCandidateDominance
compareStructuredCandidateCost(const StructuredCandidateCost &left,
                               const StructuredCandidateCost &right) {
  if (left.futureLiveInterface != right.futureLiveInterface ||
      left.physicalVersions != right.physicalVersions ||
      left.loopReuseAndEffects != right.loopReuseAndEffects ||
      left.collectiveAndPeerInterface != right.collectiveAndPeerInterface)
    return StructuredCandidateDominance::Incomparable;

  bool leftBetter = false;
  bool rightBetter = false;
  for (auto [leftMetric, rightMetric] :
       llvm::zip_equal(left.selection, right.selection)) {
    switch (compareMetric(leftMetric, rightMetric)) {
    case MetricOrder::Left:
      leftBetter = true;
      break;
    case MetricOrder::Right:
      rightBetter = true;
      break;
    case MetricOrder::Equivalent:
      break;
    case MetricOrder::Incomparable:
      return StructuredCandidateDominance::Incomparable;
    }
    if (leftBetter && rightBetter)
      return StructuredCandidateDominance::Incomparable;
  }
  if (leftBetter)
    return StructuredCandidateDominance::LeftDominates;
  if (rightBetter)
    return StructuredCandidateDominance::RightDominates;
  return StructuredCandidateDominance::Equivalent;
}

bool canStillSatisfyCoordinatedProductionPromotion(
    const StructuredCandidateCost &candidate,
    const StructuredCandidateCost &baseline) {
  // These dimensions are exact physical movement bytes in Tile IR. Final
  // ready-order, worker placement, fixed-slot buffering and completion do not
  // remove or add the corresponding Tile movement operations. Production's
  // final promotion contract rejects a regression in any of them, regardless
  // of a calibrated DDR/NoC/compute estimate.
  constexpr Dimension invariantUncalibratedDimensions[] = {
      Dimension::LocalAggregateMovementBytes,
      Dimension::LocalMaximumRankMovementBytes,
      Dimension::SPMAggregateMovementBytes,
      Dimension::SPMMaximumRankMovementBytes,
  };
  bool hasInvariantRegression = false;
  for (Dimension dimension : invariantUncalibratedDimensions) {
    const Metric &candidateMetric = candidate.selection[index(dimension)];
    const Metric &baselineMetric = baseline.selection[index(dimension)];
    if (candidateMetric.knowledge == Knowledge::Known &&
        baselineMetric.knowledge == Knowledge::Known &&
        candidateMetric.value > baselineMetric.value)
      hasInvariantRegression = true;
  }
  if (!hasInvariantRegression)
    return true;

  // A local/SPM regression with no proved compensation cannot be promoted by
  // the current uncalibrated production policy. A candidate that already has
  // a Known improvement in another exact selection dimension is a real Pareto
  // trade-off, however, and must survive to final executable recost instead of
  // being discarded before DDR/compute/completion costs are available.
  for (size_t metricIndex = 0;
       metricIndex < kStructuredCandidateCostDimensionCount; ++metricIndex) {
    Dimension dimension = static_cast<Dimension>(metricIndex);
    if (llvm::is_contained(invariantUncalibratedDimensions, dimension))
      continue;
    const Metric &candidateMetric = candidate.selection[metricIndex];
    const Metric &baselineMetric = baseline.selection[metricIndex];
    if (candidateMetric.knowledge == Knowledge::Known &&
        baselineMetric.knowledge == Knowledge::Known &&
        candidateMetric.value < baselineMetric.value)
      return true;
  }
  return false;
}

mlir::FailureOr<StructuredParetoInsertion> planStructuredParetoInsertion(
    StructuredCandidateCostView candidate,
    llvm::ArrayRef<StructuredCandidateCostView> existing) {
  if (!candidate.facts || candidate.stableSemanticOrdinal < 0)
    return mlir::failure();

  StructuredParetoInsertion plan;
  plan.retainCandidate = true;
  for (auto [existingIndex, state] : llvm::enumerate(existing)) {
    if (!state.facts || state.stableSemanticOrdinal < 0)
      return mlir::failure();
    switch (compareStructuredCandidateCost(*candidate.facts, *state.facts)) {
    case StructuredCandidateDominance::LeftDominates:
      if (!state.reservedBaseline)
        plan.eraseIndices.push_back(existingIndex);
      break;
    case StructuredCandidateDominance::RightDominates:
      if (!candidate.reservedBaseline)
        plan.retainCandidate = false;
      break;
    case StructuredCandidateDominance::Equivalent:
      if (candidate.reservedBaseline && !state.reservedBaseline) {
        plan.eraseIndices.push_back(existingIndex);
      } else if (!candidate.reservedBaseline && state.reservedBaseline) {
        plan.retainCandidate = false;
      } else if (candidate.stableSemanticOrdinal <
                 state.stableSemanticOrdinal) {
        plan.eraseIndices.push_back(existingIndex);
      } else {
        plan.retainCandidate = false;
      }
      break;
    case StructuredCandidateDominance::Incomparable:
      break;
    }
    if (!plan.retainCandidate) {
      plan.eraseIndices.clear();
      return plan;
    }
  }
  llvm::sort(plan.eraseIndices);
  plan.eraseIndices.erase(
      std::unique(plan.eraseIndices.begin(), plan.eraseIndices.end()),
      plan.eraseIndices.end());
  return plan;
}

} // namespace wafer::compiler::detail
