//===- CandidateAnalysis.cpp - Tile candidate implementation
//-----------------===//

#include "Scheduling/ScheduleTensorProgramInternal.h"
#include "Wafer/Target/TargetCall.h"

namespace wafer::tensor_program_scheduling {

std::string getNearestSymbolName(mlir::Operation *op) {
  for (mlir::Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (auto name = parent->getAttrOfType<mlir::StringAttr>("sym_name"))
      return ("@" + name.getValue()).str();
  }
  return "@unknown";
}

void printI64List(llvm::ArrayRef<int64_t> values, llvm::raw_ostream &os) {
  os << "[";
  for (auto [index, value] : llvm::enumerate(values)) {
    if (index != 0)
      os << ",";
    os << value;
  }
  os << "]";
}

static bool checkedAdd(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0)
    return false;
  if (rhs > std::numeric_limits<int64_t>::max() - lhs)
    return false;
  result = lhs + rhs;
  return true;
}

int64_t saturatingAdd(int64_t lhs, int64_t rhs) {
  int64_t result = 0;
  if (!checkedAdd(lhs, rhs, result))
    return std::numeric_limits<int64_t>::max();
  return result;
}

int64_t saturatingMul(int64_t lhs, int64_t rhs) {
  if (lhs < 0 || rhs < 0)
    return std::numeric_limits<int64_t>::max();
  if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)
    return std::numeric_limits<int64_t>::max();
  return lhs * rhs;
}

int64_t ceilDiv(int64_t numerator, int64_t denominator) {
  if (numerator <= 0)
    return 0;
  if (denominator <= 0)
    return std::numeric_limits<int64_t>::max();
  return (numerator + denominator - 1) / denominator;
}

static int64_t getStaticMemRefElementCount(mlir::Value value) {
  auto type = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  if (!type || !type.hasStaticShape())
    return 0;
  int64_t count = 1;
  for (int64_t dim : type.getShape())
    count = saturatingMul(count, dim);
  return count;
}

static int64_t getOptionalBatchCount(InstrGemmOp op) {
  if (auto attr = op.getBatchCountAttr())
    return attr.getInt();
  return 1;
}

CandidateStats
estimateStats(mlir::ModuleOp module,
              const analysis::TargetScheduleCostPolicy &scheduleCostPolicy) {
  CandidateStats stats;
  stats.program = analysis::analyzeInstructionProgramCost(module.getOperation(),
                                                          scheduleCostPolicy);
  return stats;
}

static mlir::FailureOr<llvm::SmallVector<int64_t, 4>>
getStaticTraversalShapeFromTypes(mlir::TypeRange resultTypes);

static mlir::FailureOr<llvm::SmallVector<int64_t, 4>>
getStaticTraversalShapeFromTypes(mlir::TypeRange resultTypes) {
  if (resultTypes.empty())
    return mlir::failure();

  auto firstType = mlir::dyn_cast<mlir::RankedTensorType>(resultTypes.front());
  if (!firstType || !firstType.hasStaticShape())
    return mlir::failure();
  llvm::SmallVector<int64_t, 4> shape(firstType.getShape().begin(),
                                      firstType.getShape().end());

  for (mlir::Type type : resultTypes.drop_front()) {
    auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(type);
    if (!resultType || !resultType.hasStaticShape())
      return mlir::failure();
    if (!std::equal(shape.begin(), shape.end(), resultType.getShape().begin(),
                    resultType.getShape().end()))
      return mlir::failure();
  }
  return shape;
}

mlir::FailureOr<llvm::SmallVector<int64_t, 4>>
getStaticTraversalShape(mlir::func::FuncOp task) {
  return getStaticTraversalShapeFromTypes(task.getResultTypes());
}

std::optional<int64_t> getElementByteWidth(mlir::Type type) {
  if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(type))
    type = shaped.getElementType();
  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(type))
    return std::max<int64_t>(1, floatType.getWidth() / 8);
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type))
    return std::max<int64_t>(1, (intType.getWidth() + 7) / 8);
  return std::nullopt;
}

std::optional<llvm::SmallVector<mlir::linalg::LinalgOp, 4>>
getYieldedRootLinalgOps(mlir::func::FuncOp task) {
  if (!task.getBody().hasOneBlock())
    return std::nullopt;
  auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(
      task.getBody().front().getTerminator());
  if (!returnOp || returnOp.getOperands().empty())
    return std::nullopt;
  llvm::SmallVector<mlir::linalg::LinalgOp, 4> roots;
  for (mlir::Value value : returnOp.getOperands()) {
    auto root =
        mlir::dyn_cast_or_null<mlir::linalg::LinalgOp>(value.getDefiningOp());
    if (!root)
      return std::nullopt;
    roots.push_back(root);
  }
  return roots;
}

namespace {

struct TraversalComputeRootInfo {
  llvm::SmallVector<mlir::linalg::LinalgOp, 4> roots;
  bool hasAllReduceWrapper = false;
};

/// Recognizes only the all-reduce wrapper that complete traversal can tile and
/// fuse without recovering semantic roles from names: one ranked tensor
/// input/out/result, a shape-preserving input and type-identical out/result, a
/// direct unique Linalg producer, and a direct task output destination. The
/// collective verifier owns any input-element promotion legality. Fanout
/// remains fail-closed because its additional live dataflow is not represented
/// by the estimator.
static mlir::linalg::LinalgOp getAllReduceTraversalComputeRoot(
    mlir::func::FuncOp task, mlir::Value yieldedValue, unsigned outputIndex) {
  auto allReduce = mlir::dyn_cast_or_null<LinalgExtCollectiveAllReduceOp>(
      yieldedValue.getDefiningOp());
  if (!allReduce || allReduce->getBlock() != &task.getBody().front() ||
      allReduce.getInputs().size() != 1 || allReduce.getOuts().size() != 1 ||
      allReduce->getNumResults() != 1 || allReduce.getResult(0) != yieldedValue)
    return {};

  mlir::Value input = allReduce.getInputs().front();
  mlir::Value output = allReduce.getOuts().front();
  auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
  auto outputType = mlir::dyn_cast<mlir::RankedTensorType>(output.getType());
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(yieldedValue.getType());
  auto outputArgument = mlir::dyn_cast<mlir::BlockArgument>(output);
  unsigned outputArgumentBase = task.getNumArguments() - task.getNumResults();
  if (!inputType || !outputType || !resultType ||
      inputType.getShape() != resultType.getShape() ||
      outputType != resultType || !outputArgument ||
      outputArgument.getOwner() != &task.getBody().front() ||
      outputArgument.getArgNumber() != outputArgumentBase + outputIndex ||
      !output.hasOneUse() || !yieldedValue.hasOneUse())
    return {};

  auto producer =
      mlir::dyn_cast_or_null<mlir::linalg::LinalgOp>(input.getDefiningOp());
  if (!producer || producer->getBlock() != &task.getBody().front() ||
      producer->getNumResults() != 1 || producer->getResult(0) != input ||
      !input.hasOneUse())
    return {};
  return producer;
}

static std::optional<TraversalComputeRootInfo>
getTraversalComputeRootInfo(mlir::func::FuncOp task) {
  if (!task || !task.getBody().hasOneBlock())
    return std::nullopt;
  if (task.getNumArguments() < task.getNumResults())
    return std::nullopt;
  auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(
      task.getBody().front().getTerminator());
  if (!returnOp || returnOp.getOperands().empty())
    return std::nullopt;

  TraversalComputeRootInfo info;
  for (auto [outputIndex, value] : llvm::enumerate(returnOp.getOperands())) {
    if (auto root = mlir::dyn_cast_or_null<mlir::linalg::LinalgOp>(
            value.getDefiningOp())) {
      info.roots.push_back(root);
      continue;
    }
    mlir::linalg::LinalgOp root = getAllReduceTraversalComputeRoot(
        task, value, static_cast<unsigned>(outputIndex));
    if (!root)
      return std::nullopt;
    info.roots.push_back(root);
    info.hasAllReduceWrapper = true;
  }
  return info;
}

} // namespace

std::optional<llvm::SmallVector<mlir::linalg::LinalgOp, 4>>
getTraversalComputeRootLinalgOps(mlir::func::FuncOp task) {
  std::optional<TraversalComputeRootInfo> info =
      getTraversalComputeRootInfo(task);
  if (!info)
    return std::nullopt;
  return std::move(info->roots);
}

CandidateTraversalRootCapability
getTaskTraversalRootCapability(mlir::func::FuncOp task) {
  if (!task || !task.getBody().hasOneBlock())
    return CandidateTraversalRootCapability::Unsupported;
  auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(
      task.getBody().front().getTerminator());
  if (!returnOp || returnOp.getOperands().empty())
    return CandidateTraversalRootCapability::Unsupported;

  CandidateTraversalRootCapability taskCapability =
      CandidateTraversalRootCapability::Tiled;
  for (mlir::Value value : returnOp.getOperands()) {
    CandidateTraversalRootCapability rootCapability =
        classifyCandidateTraversalRoot(value.getDefiningOp());
    if (rootCapability == CandidateTraversalRootCapability::Unsupported)
      return rootCapability;
    if (rootCapability == CandidateTraversalRootCapability::FullTraversalOnly)
      taskCapability = rootCapability;
  }
  return taskCapability;
}

std::string getStandaloneTaskModuleText(mlir::func::FuncOp task) {
  std::string text;
  llvm::raw_string_ostream os(text);
  task->getParentOfType<mlir::ModuleOp>().print(os);
  return os.str();
}

mlir::func::FuncOp findSingleSelectionTask(mlir::ModuleOp module) {
  return structured_scheduler::findSingleTaskFunction(module);
}

static bool hasReductionIterator(mlir::linalg::LinalgOp op) {
  for (mlir::utils::IteratorType iteratorType : op.getIteratorTypesArray()) {
    if (iteratorType == mlir::utils::IteratorType::reduction)
      return true;
  }
  return false;
}

static mlir::FailureOr<llvm::SmallVector<int64_t, 2>>
getStaticRootReductionRangesImpl(
    std::optional<llvm::SmallVector<mlir::linalg::LinalgOp, 4>> roots) {
  if (!roots)
    return llvm::SmallVector<int64_t, 2>{};

  std::optional<llvm::SmallVector<int64_t, 2>> commonReductionRanges;
  for (mlir::linalg::LinalgOp root : *roots) {
    if (!mlir::isa<
            mlir::linalg::MatmulOp, mlir::linalg::MatmulTransposeAOp,
            mlir::linalg::MatmulTransposeBOp, mlir::linalg::BatchMatmulOp,
            mlir::linalg::BatchMatmulTransposeAOp,
            mlir::linalg::BatchMatmulTransposeBOp, mlir::linalg::GenericOp>(
            root.getOperation()) ||
        !hasReductionIterator(root))
      continue;

    llvm::SmallVector<int64_t, 4> loopRanges = root.getStaticLoopRanges();
    if (llvm::any_of(loopRanges, [](int64_t range) {
          return mlir::ShapedType::isDynamic(range);
        }))
      return mlir::failure();

    llvm::SmallVector<int64_t, 2> reductionRanges;
    for (auto [index, iteratorType] :
         llvm::enumerate(root.getIteratorTypesArray())) {
      if (iteratorType == mlir::utils::IteratorType::reduction)
        reductionRanges.push_back(loopRanges[index]);
    }
    if (reductionRanges.empty())
      continue;
    if (!commonReductionRanges) {
      commonReductionRanges = reductionRanges;
      continue;
    }
    if (*commonReductionRanges != reductionRanges)
      return llvm::SmallVector<int64_t, 2>{};
  }
  if (!commonReductionRanges)
    return llvm::SmallVector<int64_t, 2>{};
  return *commonReductionRanges;
}

mlir::FailureOr<llvm::SmallVector<int64_t, 2>>
getStaticRootReductionRanges(mlir::func::FuncOp task,
                             CandidateTileTraversalKind traversalKind) {
  return getStaticRootReductionRangesImpl(
      traversalKind == CandidateTileTraversalKind::PartialReduction
          ? getTraversalComputeRootLinalgOps(task)
          : getYieldedRootLinalgOps(task));
}

std::optional<std::string>
getReductionSplitLegalityFailure(mlir::func::FuncOp task,
                                 CandidateTileTraversalKind traversalKind) {
  std::optional<llvm::SmallVector<mlir::linalg::LinalgOp, 4>> roots =
      traversalKind == CandidateTileTraversalKind::PartialReduction
          ? getTraversalComputeRootLinalgOps(task)
          : getYieldedRootLinalgOps(task);
  if (!roots)
    return std::nullopt;

  for (mlir::linalg::LinalgOp root : *roots) {
    if (!hasReductionIterator(root))
      continue;
    std::string failureReason;
    if (mlir::failed(verifyCandidateReductionSplitNumericLegality(
            root, &failureReason))) {
      if (failureReason.empty())
        return "candidate reduction split failed numeric legality";
      return failureReason;
    }
  }
  return std::nullopt;
}

static std::optional<int64_t>
estimateRootMinimumSPMBytes(mlir::linalg::LinalgOp root,
                            const CandidateSpec &candidate) {
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(root->getResult(0).getType());
  if (!resultType || !resultType.hasStaticShape())
    return std::nullopt;
  std::optional<int64_t> elementBytes = getElementByteWidth(resultType);
  if (!elementBytes)
    return std::nullopt;

  if (mlir::isa<mlir::linalg::MatmulOp>(root.getOperation()) &&
      candidate.tileSizes.size() == 2 && root.getNumDpsInputs() == 2) {
    auto lhsType = mlir::dyn_cast<mlir::RankedTensorType>(
        root.getDpsInputOperand(0)->get().getType());
    auto rhsType = mlir::dyn_cast<mlir::RankedTensorType>(
        root.getDpsInputOperand(1)->get().getType());
    if (!lhsType || !rhsType || lhsType.getRank() != 2 ||
        rhsType.getRank() != 2 || !lhsType.hasStaticShape() ||
        !rhsType.hasStaticShape())
      return std::nullopt;
    int64_t m = candidate.tileSizes[0];
    int64_t n = candidate.tileSizes[1];
    int64_t k = lhsType.getDimSize(1);
    if (!candidate.reductionSplitSizes.empty())
      k = candidate.reductionSplitSizes.front();
    int64_t elements = saturatingAdd(saturatingMul(m, k), saturatingMul(k, n));
    elements = saturatingAdd(elements, saturatingMul(m, n));
    if (candidate.traversalKind ==
            CandidateTileTraversalKind::PartialReduction &&
        !candidate.reductionSplitSizes.empty()) {
      // PartialReductionOpInterface materializes actual [M, N, K-split]
      // tensors before its merge reduction. The matmul expansion has four
      // simultaneously live tensors at its multiply point: the initialized
      // accumulator, both broadcast operands, and the multiplication result.
      // None can alias another there under the value/effect contract, so this
      // is a sound rejection lower bound. Applying it before materialization
      // prevents an impossible partial from spending minutes in layout
      // movement lowering only to fail the unchanged exact SPM planner later.
      const int64_t partialElements =
          saturatingMul(saturatingMul(saturatingMul(m, n), k), 4);
      elements = std::max(elements, partialElements);
    }
    return saturatingMul(elements, *elementBytes);
  }

  int64_t tileElements = 1;
  for (int64_t size : candidate.tileSizes)
    tileElements = saturatingMul(tileElements, size);
  return saturatingMul(tileElements, *elementBytes);
}

static std::optional<int64_t> estimateRequiredSPMLowerBoundBytes(
    std::optional<llvm::SmallVector<mlir::linalg::LinalgOp, 4>> roots,
    const CandidateSpec &candidate) {
  if (!roots)
    return std::nullopt;

  int64_t required = 0;
  for (mlir::linalg::LinalgOp root : *roots) {
    std::optional<int64_t> rootBytes =
        estimateRootMinimumSPMBytes(root, candidate);
    if (!rootBytes)
      return std::nullopt;
    required = std::max(required, *rootBytes);
  }
  return required;
}

bool failsCheapSPMBound(mlir::func::FuncOp task, const CandidateSpec &candidate,
                        int64_t spmBase, int64_t spmLimit) {
  std::optional<int64_t> lowerBoundBytes = estimateRequiredSPMLowerBoundBytes(
      getTraversalComputeRootLinalgOps(task), candidate);
  if (!lowerBoundBytes || spmLimit <= spmBase)
    return false;
  return *lowerBoundBytes > (spmLimit - spmBase);
}

static std::optional<int64_t>
getAlignedPhysicalSPMBytes(llvm::ArrayRef<int64_t> shape,
                           mlir::Type elementType, MemLayout layout,
                           int64_t spmAlignment) {
  if (spmAlignment <= 0 ||
      llvm::any_of(shape, [](int64_t size) { return size <= 0; }))
    return std::nullopt;
  auto memory =
      MemoryAttr::get(elementType.getContext(), MemorySpace::SPM, layout);
  auto type = mlir::MemRefType::get(shape, elementType,
                                    mlir::MemRefLayoutAttrInterface{}, memory);
  std::optional<WaferPhysicalTensorInfo> physical =
      computeWaferPhysicalTensorInfo(type);
  mlir::FailureOr<int64_t> requiredAlignment =
      computeWaferRequiredAlignmentBytes(type, {spmAlignment});
  if (!physical || physical->physicalBytes < 0 ||
      mlir::failed(requiredAlignment) ||
      physical->physicalBytes >
          std::numeric_limits<int64_t>::max() - (*requiredAlignment - 1))
    return std::nullopt;
  return ((physical->physicalBytes + *requiredAlignment - 1) /
          *requiredAlignment) *
         *requiredAlignment;
}

static bool isDirectTaskArgument(mlir::func::FuncOp task, mlir::Value value) {
  auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
  return argument && argument.getOwner() == &task.getBody().front();
}

/// Returns the rank-independent required-live SPM bound for a task whose
/// returned values are direct, shape-preserving multi-rank all-reduces.
///
/// A tree reduction's internal rank must hold the local input, receive tile,
/// and accumulator tile at the same time.  Leaf ranks can sometimes alias
/// those roots and therefore pass exact rank-local SPM planning with a larger
/// tile.  Candidate legality must nevertheless use the worst collective role:
/// otherwise peers select different traversal steps and no longer execute the
/// same DTE message instances.
static std::optional<int64_t>
estimateDirectAllReduceRequiredLiveBytes(mlir::func::FuncOp task,
                                         const CandidateSpec &candidate,
                                         int64_t spmAlignment) {
  if (!task || !task.getBody().hasOneBlock() ||
      !candidate.reductionSplitSizes.empty() || candidate.tileSizes.empty())
    return std::nullopt;
  auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(
      task.getBody().front().getTerminator());
  if (!returnOp || returnOp.getOperands().empty())
    return std::nullopt;

  int64_t largestTileBytes = 0;
  for (mlir::Value returned : returnOp.getOperands()) {
    auto allReduce = returned.getDefiningOp<LinalgExtCollectiveAllReduceOp>();
    if (!allReduce || allReduce.getInputs().size() != 1 ||
        allReduce.getOuts().size() != 1 || allReduce->getNumResults() != 1 ||
        allReduce.getResult(0) != returned ||
        !isDirectTaskArgument(task, allReduce.getInputs().front()) ||
        !isDirectTaskArgument(task, allReduce.getOuts().front()))
      return std::nullopt;

    int64_t groupSize = 0;
    if (auto rankGroup = allReduce.getRankGroupAttr()) {
      groupSize = static_cast<int64_t>(rankGroup.asArrayRef().size());
    } else if (auto rankGroups = allReduce.getRankGroupsAttr()) {
      auto type = mlir::dyn_cast<mlir::RankedTensorType>(rankGroups.getType());
      if (!type || type.getRank() != 2)
        return std::nullopt;
      groupSize = type.getDimSize(1);
    }
    if (groupSize <= 1)
      return std::nullopt;

    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(
        allReduce.getInputs().front().getType());
    auto outputType = mlir::dyn_cast<mlir::RankedTensorType>(
        allReduce.getOuts().front().getType());
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(returned.getType());
    if (!inputType || !outputType || !resultType || inputType != outputType ||
        inputType != resultType || !resultType.hasStaticShape() ||
        static_cast<int64_t>(candidate.tileSizes.size()) !=
            resultType.getRank())
      return std::nullopt;
    for (auto [tile, extent] :
         llvm::zip(candidate.tileSizes, resultType.getShape()))
      if (tile <= 0 || tile > extent)
        return std::nullopt;

    std::optional<int64_t> tileBytes = getAlignedPhysicalSPMBytes(
        candidate.tileSizes, resultType.getElementType(), MemLayout::Tensor,
        spmAlignment);
    if (!tileBytes)
      return std::nullopt;
    largestTileBytes = std::max(largestTileBytes, *tileBytes);
  }

  return saturatingMul(largestTileBytes, 3);
}

/// Recognizes the exact rank-2 passthrough transpose producer generated by the
/// structured frontend for a transposed GEMM weight. The returned type is the
/// direct task-boundary tensor; the generic result is the matmul RHS.
static mlir::RankedTensorType
getDirectPassthroughTransposeInputType(mlir::func::FuncOp task,
                                       mlir::linalg::MatmulOp matmul) {
  auto transpose =
      matmul.getInputs()[1].getDefiningOp<mlir::linalg::GenericOp>();
  if (!transpose || transpose.getNumDpsInputs() != 1 ||
      transpose.getNumDpsInits() != 1 || transpose->getNumResults() != 1 ||
      !transpose.getRegion().hasOneBlock() ||
      !llvm::all_of(transpose.getIteratorTypesArray(), [](auto iteratorType) {
        return iteratorType == mlir::utils::IteratorType::parallel;
      }))
    return {};

  mlir::Value input = transpose.getDpsInputs().front();
  if (!isDirectTaskArgument(task, input) ||
      !transpose.getDpsInits().front().getDefiningOp<mlir::tensor::EmptyOp>() ||
      !transpose->getResult(0).hasOneUse() ||
      *transpose->getResult(0).getUsers().begin() != matmul.getOperation())
    return {};

  auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(transpose->getResult(0).getType());
  if (!inputType || !resultType || inputType.getRank() != 2 ||
      resultType.getRank() != 2 || !inputType.hasStaticShape() ||
      !resultType.hasStaticShape() ||
      inputType.getElementType() != resultType.getElementType() ||
      inputType.getDimSize(0) != resultType.getDimSize(1) ||
      inputType.getDimSize(1) != resultType.getDimSize(0))
    return {};

  llvm::SmallVector<mlir::AffineMap, 2> maps = transpose.getIndexingMapsArray();
  if (maps.size() != 2 || !maps[1].isIdentity() || maps[0].getNumDims() != 2 ||
      maps[0].getNumResults() != 2)
    return {};
  auto first = mlir::dyn_cast<mlir::AffineDimExpr>(maps[0].getResult(0));
  auto second = mlir::dyn_cast<mlir::AffineDimExpr>(maps[0].getResult(1));
  if (!first || !second || first.getPosition() != 1 ||
      second.getPosition() != 0)
    return {};

  mlir::Block &body = transpose.getRegion().front();
  auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
  if (!body.without_terminator().empty() || !yield ||
      yield.getValues().size() != 1 ||
      yield.getValues().front() != body.getArgument(0))
    return {};
  return inputType;
}

static std::optional<int64_t> estimateTargetSPMWorkingSetBytesImpl(
    mlir::func::FuncOp task, const CandidateSpec &candidate,
    int64_t spmAlignment, bool includeFusedTransposeSource,
    bool includeAllReduceTileBuffers) {
  // The named matmul lowering has a closed target-layout contract when the LHS
  // is a direct task boundary and the RHS is either direct or produced by one
  // exact rank-2 passthrough transpose. Direct operands contribute Tensor and
  // Cx roots, while the result contributes Cx and output Tensor roots. For a
  // fused transpose, conservatively count both its direct source Tensor and
  // its Tensor result in addition to the RHS Cx root. This is a safe upper
  // bound even when lifetime packing aliases the two Tensor roots.
  //
  // Other producer chains can introduce unmodeled materialized views, so
  // decline to estimate them rather than understate their demand.
  std::optional<TraversalComputeRootInfo> rootInfo =
      getTraversalComputeRootInfo(task);
  if (!rootInfo || rootInfo->roots.size() != 1 ||
      candidate.tileSizes.size() != 2 ||
      !candidate.reductionSplitSizes.empty() || !task.getBody().hasOneBlock())
    return std::nullopt;
  auto matmul = mlir::dyn_cast<mlir::linalg::MatmulOp>(
      rootInfo->roots.front().getOperation());
  if (!matmul)
    return std::nullopt;

  if (!isDirectTaskArgument(task, matmul.getInputs()[0]))
    return std::nullopt;

  auto lhsType =
      mlir::dyn_cast<mlir::RankedTensorType>(matmul.getInputs()[0].getType());
  auto rhsType =
      mlir::dyn_cast<mlir::RankedTensorType>(matmul.getInputs()[1].getType());
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(matmul.getResult(0).getType());
  if (!lhsType || !rhsType || !resultType || lhsType.getRank() != 2 ||
      rhsType.getRank() != 2 || resultType.getRank() != 2 ||
      !lhsType.hasStaticShape() || !rhsType.hasStaticShape() ||
      !resultType.hasStaticShape() ||
      lhsType.getDimSize(0) != resultType.getDimSize(0) ||
      lhsType.getDimSize(1) != rhsType.getDimSize(0) ||
      rhsType.getDimSize(1) != resultType.getDimSize(1))
    return std::nullopt;

  int64_t m = candidate.tileSizes[0];
  int64_t n = candidate.tileSizes[1];
  int64_t k = lhsType.getDimSize(1);
  if (m <= 0 || n <= 0 || k <= 0 || m > resultType.getDimSize(0) ||
      n > resultType.getDimSize(1))
    return std::nullopt;

  mlir::RankedTensorType transposeInputType;
  bool directRhs = isDirectTaskArgument(task, matmul.getInputs()[1]);
  if (!directRhs) {
    transposeInputType = getDirectPassthroughTransposeInputType(task, matmul);
    if (!transposeInputType)
      return std::nullopt;
  }

  struct BufferShape {
    llvm::SmallVector<int64_t, 2> shape;
    mlir::Type elementType;
    MemLayout layout;
  };
  llvm::SmallVector<BufferShape, 8> buffers = {
      {{m, k}, lhsType.getElementType(), MemLayout::Tensor},
      {{m, k}, lhsType.getElementType(), MemLayout::Cx},
  };
  if (directRhs) {
    buffers.push_back({{k, n}, rhsType.getElementType(), MemLayout::Tensor});
  } else {
    if (includeFusedTransposeSource)
      buffers.push_back(
          {{n, k}, transposeInputType.getElementType(), MemLayout::Tensor});
    buffers.push_back({{k, n}, rhsType.getElementType(), MemLayout::Tensor});
  }
  buffers.push_back({{k, n}, rhsType.getElementType(), MemLayout::Cx});
  buffers.push_back({{m, n}, resultType.getElementType(), MemLayout::Cx});
  buffers.push_back({{m, n}, resultType.getElementType(), MemLayout::Tensor});
  // Complete all-reduce traversal additionally materializes shape-preserving
  // input/out/result Tensor tiles around the fused producer. Count three
  // aligned tile roots as a conservative search-direction inventory. This is
  // deliberately excluded from the required-live rejection bound below: the
  // exact collective lifetimes remain owned by complete materialization and
  // SPM planning rather than this queue-ordering estimate.
  if (includeAllReduceTileBuffers && rootInfo->hasAllReduceWrapper) {
    for (int64_t index = 0; index < 3; ++index)
      buffers.push_back(
          {{m, n}, resultType.getElementType(), MemLayout::Tensor});
  }

  int64_t workingSetBytes = 0;
  for (const BufferShape &buffer : buffers) {
    std::optional<int64_t> bytes = getAlignedPhysicalSPMBytes(
        buffer.shape, buffer.elementType, buffer.layout, spmAlignment);
    if (!bytes ||
        *bytes > std::numeric_limits<int64_t>::max() - workingSetBytes)
      return std::nullopt;
    workingSetBytes += *bytes;
  }
  return workingSetBytes;
}

std::optional<int64_t>
estimateTargetSPMWorkingSetBytes(mlir::func::FuncOp task,
                                 const CandidateSpec &candidate,
                                 int64_t spmAlignment) {
  // Include all seven roots for the fused-transpose case. This conservative
  // inventory is suitable for directing search even if lifetime packing can
  // alias the source Tensor after the transpose completes.
  return estimateTargetSPMWorkingSetBytesImpl(
      task, candidate, spmAlignment,
      /*includeFusedTransposeSource=*/true,
      /*includeAllReduceTileBuffers=*/true);
}

std::optional<int64_t>
estimateSearchSPMWorkingSetBytes(mlir::func::FuncOp task,
                                 const CandidateSpec &candidate,
                                 int64_t spmAlignment) {
  if (std::optional<int64_t> targetWorkingSet =
          estimateTargetSPMWorkingSetBytes(task, candidate, spmAlignment))
    return targetWorkingSet;
  return estimateRequiredSPMLowerBoundBytes(
      getTraversalComputeRootLinalgOps(task), candidate);
}

std::optional<int64_t>
estimateTargetSPMRequiredLiveBytes(mlir::func::FuncOp task,
                                   const CandidateSpec &candidate,
                                   int64_t spmAlignment) {
  if (std::optional<int64_t> collectiveBytes =
          estimateDirectAllReduceRequiredLiveBytes(task, candidate,
                                                   spmAlignment))
    return collectiveBytes;

  // The source Tensor of a fused transpose need not overlap the later matmul
  // phase. The remaining six Tensor/Cx roots do: the transposed Tensor and Cx
  // RHS replace the direct RHS pair, and local issue completion keeps them
  // live with the LHS and result pairs. This is therefore a required-live
  // bound that may reject an impossible candidate before materialization.
  return estimateTargetSPMWorkingSetBytesImpl(
      task, candidate, spmAlignment,
      /*includeFusedTransposeSource=*/false,
      /*includeAllReduceTileBuffers=*/false);
}

static std::optional<std::string> getCheapTargetGeometryFailureImpl(
    std::optional<llvm::SmallVector<mlir::linalg::LinalgOp, 4>> roots,
    const CandidateSpec &candidate, llvm::ArrayRef<int64_t> reductionRanges) {
  if (!roots)
    return std::nullopt;

  constexpr int64_t maxTargetDimension = std::numeric_limits<uint16_t>::max();
  auto exceedsTargetDimension = [&](llvm::ArrayRef<int64_t> dimensions) {
    return llvm::any_of(dimensions, [&](int64_t dimension) {
      return dimension > maxTargetDimension;
    });
  };

  for (mlir::linalg::LinalgOp root : *roots) {
    bool isGemm =
        mlir::isa<mlir::linalg::MatmulOp, mlir::linalg::MatmulTransposeAOp,
                  mlir::linalg::MatmulTransposeBOp, mlir::linalg::BatchMatmulOp,
                  mlir::linalg::BatchMatmulTransposeAOp,
                  mlir::linalg::BatchMatmulTransposeBOp>(root.getOperation());
    bool isReduction = hasReductionIterator(root);
    if (!isGemm && !isReduction)
      continue;

    llvm::ArrayRef<int64_t> reductionSizes =
        candidate.reductionSplitSizes.empty()
            ? reductionRanges
            : llvm::ArrayRef<int64_t>(candidate.reductionSplitSizes);
    if (candidate.traversalKind ==
            CandidateTileTraversalKind::PartialReduction &&
        isGemm && !reductionSizes.empty()) {
      int64_t reductionTupleCount = 1;
      for (int64_t size : reductionSizes)
        reductionTupleCount = saturatingMul(reductionTupleCount, size);
      constexpr uint64_t budget = wafer::detail::kStaticTerminalOperationBudget;
      constexpr uint64_t maximumOrderedReductionTuples = (budget - 4) / 4;
      auto resultType =
          mlir::dyn_cast<mlir::RankedTensorType>(root->getResult(0).getType());
      // Mirror the exact native-reduce target tuple here so refinement reaches
      // a representable split without first expanding a guaranteed-illegal
      // partial tensor. The same CT format encoding registry is consumed by
      // target preflight.
      if (resultType && static_cast<uint64_t>(reductionTupleCount) >
                            maximumOrderedReductionTuples) {
        std::optional<LogicalFormat> format;
        if (mlir::isa<mlir::Float16Type>(resultType.getElementType()))
          format = LogicalFormat::F16;
        else if (mlir::isa<mlir::BFloat16Type>(resultType.getElementType()))
          format = LogicalFormat::BF16;
        else if (mlir::isa<mlir::Float32Type>(resultType.getElementType()))
          format = LogicalFormat::F32;
        const TargetFormatEncodingRecord *encoding =
            format ? findTargetFormatEncoding(TargetFormatEngine::CT, *format)
                   : nullptr;
        if (!encoding)
          return "static_terminal_budget_exceeded: partial-reduction merge "
                 "minimum terminal operation count exceeds 4096";
      }
    }
    // GEMM dimensions and a genuine local reduction are encoded through
    // target uint16_t shape fields. A sharded high-level reduction can,
    // however, leave a rank-local unit reduction whose physical program is
    // only elementwise compute plus collective DTE. In that case no CT
    // Reduce Data_Shape field exists, and applying its limit to the result
    // traversal axis incorrectly forces an otherwise legal extra loop.
    //
    // This is only a cheap rejection gate. The complete candidate still
    // materializes and passes the exact instruction/ABI preflight, including
    // the uint32_t element-count check for elementwise compute.
    const bool hasNonUnitLocalReduction =
        isReduction && llvm::any_of(reductionSizes, [](int64_t dimension) {
          return dimension > 1;
        });
    if (!isGemm && !hasNonUnitLocalReduction)
      continue;

    if (exceedsTargetDimension(candidate.tileSizes))
      return "target_abi_narrowing: candidate traversal dimension must fit "
             "uint16_t";

    if (exceedsTargetDimension(reductionSizes))
      return "target_abi_narrowing: candidate reduction dimension must fit "
             "uint16_t";
  }
  return std::nullopt;
}

std::optional<std::string> getCheapTargetGeometryFailure(
    mlir::func::FuncOp task, const CandidateSpec &candidate,
    llvm::ArrayRef<int64_t> reductionRanges) {
  return getCheapTargetGeometryFailureImpl(
      getTraversalComputeRootLinalgOps(task), candidate, reductionRanges);
}

static void addUnique(llvm::SmallVectorImpl<int64_t> &values, int64_t value,
                      int64_t dim) {
  if (value <= 0 || value > dim)
    return;
  if (!llvm::is_contained(values, value))
    values.push_back(value);
}

static int64_t absDiff(int64_t lhs, int64_t rhs) {
  return lhs > rhs ? lhs - rhs : rhs - lhs;
}

static int64_t chooseNextRefinementSize(int64_t dim, int64_t current,
                                        llvm::ArrayRef<int64_t> preferred) {
  if (current <= 1)
    return current;

  int64_t target = std::max<int64_t>(1, ceilDiv(current, 2));
  llvm::SmallVector<int64_t, 16> candidates;
  addUnique(candidates, target, dim);

  for (int64_t divisor = 2; divisor * divisor <= dim; ++divisor) {
    if (dim % divisor != 0)
      continue;
    addUnique(candidates, dim / divisor, dim);
    addUnique(candidates, divisor, dim);
  }

  for (int64_t value : preferred)
    addUnique(candidates, value, dim);
  addUnique(candidates, 1, dim);

  int64_t best = current;
  bool found = false;
  for (int64_t candidate : candidates) {
    if (candidate >= current)
      continue;
    if (!found) {
      best = candidate;
      found = true;
      continue;
    }

    int64_t candidateScore = absDiff(candidate, target);
    int64_t bestScore = absDiff(best, target);
    bool candidateDividesDim = dim % candidate == 0;
    bool bestDividesDim = dim % best == 0;
    if (candidateScore < bestScore ||
        (candidateScore == bestScore && candidateDividesDim &&
         !bestDividesDim) ||
        (candidateScore == bestScore && candidateDividesDim == bestDividesDim &&
         candidate > best))
      best = candidate;
  }

  return found ? best : current;
}

static llvm::SmallVector<int64_t, 8>
buildDimTileSizes(int64_t dim, llvm::ArrayRef<int64_t> preferred,
                  int64_t maxCandidatesPerDim) {
  llvm::SmallVector<int64_t, 8> values;
  addUnique(values, dim, dim);

  int64_t current = dim;
  while (maxCandidatesPerDim <= 0 ||
         static_cast<int64_t>(values.size()) < maxCandidatesPerDim) {
    int64_t next = chooseNextRefinementSize(dim, current, preferred);
    if (next >= current)
      break;
    addUnique(values, next, dim);
    current = next;
  }
  return values;
}

TileSizeOptions buildTileSizeOptions(llvm::ArrayRef<int64_t> traversalShape,
                                     llvm::ArrayRef<int64_t> reductionRanges,
                                     llvm::ArrayRef<int64_t> preferred,
                                     int64_t maxCandidatesPerDim,
                                     bool allowReductionSplits) {
  TileSizeOptions options;
  for (int64_t dim : traversalShape)
    options.traversal.push_back(
        buildDimTileSizes(dim, preferred, maxCandidatesPerDim));
  // Ordered reduction splitting is deliberately narrow: the conversion
  // proves one exact accumulator chain, while the same per-dimension menu and
  // global search budget keep the refinement finite.  Multi-axis reductions
  // remain fail-closed in materialization and are not advertised here.
  if (allowReductionSplits && reductionRanges.size() == 1)
    options.reduction.push_back(buildDimTileSizes(
        reductionRanges.front(), preferred, maxCandidatesPerDim));
  return options;
}

bool isFullFirstTile(llvm::ArrayRef<int64_t> traversalShape,
                     const TileInstance &tile) {
  if (tile.offsets.size() != traversalShape.size() ||
      tile.sizes.size() != traversalShape.size())
    return false;
  for (auto [shapeDim, values] :
       llvm::zip(traversalShape, llvm::zip(tile.offsets, tile.sizes))) {
    if (std::get<0>(values) != 0 || std::get<1>(values) != shapeDim)
      return false;
  }
  return true;
}

llvm::SmallVector<TileInstance, 8>
buildRepresentativeTiles(llvm::ArrayRef<int64_t> traversalShape,
                         llvm::ArrayRef<int64_t> tileSizes) {
  llvm::SmallVector<llvm::SmallVector<std::pair<int64_t, int64_t>, 2>, 4>
      perDim;
  for (auto [dim, tileSize] : llvm::zip(traversalShape, tileSizes)) {
    llvm::SmallVector<std::pair<int64_t, int64_t>, 2> reps;
    reps.push_back({0, std::min(dim, tileSize)});
    if (dim > tileSize) {
      int64_t tail = dim % tileSize;
      if (tail == 0)
        reps.push_back({dim - tileSize, tileSize});
      else
        reps.push_back({dim - tail, tail});
    }
    perDim.push_back(reps);
  }

  llvm::SmallVector<TileInstance, 8> tiles;
  llvm::SmallVector<int64_t, 4> offsets;
  llvm::SmallVector<int64_t, 4> sizes;
  std::function<void(unsigned)> build = [&](unsigned dim) {
    if (dim == perDim.size()) {
      for (const TileInstance &tile : tiles)
        if (tile.offsets == offsets && tile.sizes == sizes)
          return;
      tiles.push_back(TileInstance{llvm::SmallVector<int64_t, 4>(offsets),
                                   llvm::SmallVector<int64_t, 4>(sizes)});
      return;
    }
    for (auto [offset, size] : perDim[dim]) {
      offsets.push_back(offset);
      sizes.push_back(size);
      build(dim + 1);
      offsets.pop_back();
      sizes.pop_back();
    }
  };
  build(0);
  return tiles;
}

} // namespace wafer::tensor_program_scheduling
