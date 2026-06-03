//===- OpVerifierUtils.cpp - Wafer op verifier helpers ----------------===//

#include "OpVerifierUtils.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <limits>

namespace wafer::detail {

bool isSPMMemRef(mlir::Type type) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memrefType)
    return false;
  auto memorySpace = mlir::dyn_cast_or_null<wafer::MemorySpaceAttr>(
      memrefType.getMemorySpace());
  return memorySpace && memorySpace.getValue() == wafer::MemorySpace::SPM;
}

bool isSPMTileBuffer(mlir::Type type) {
  auto tileBufferType = mlir::dyn_cast<wafer::TileBufferType>(type);
  if (!tileBufferType)
    return false;
  auto memorySpace =
      mlir::cast<wafer::MemorySpaceAttr>(tileBufferType.getMemorySpace());
  return memorySpace.getValue() == wafer::MemorySpace::SPM;
}

wafer::MemLayoutAttr getTileBufferLayout(wafer::TileBufferType type) {
  return mlir::cast<wafer::MemLayoutAttr>(type.getLayout());
}

wafer::MemorySpaceAttr getTileBufferMemorySpace(wafer::TileBufferType type) {
  return mlir::cast<wafer::MemorySpaceAttr>(type.getMemorySpace());
}

mlir::RankedTensorType getTileBufferTensorType(wafer::TileBufferType type) {
  return mlir::cast<mlir::RankedTensorType>(type.getTensorType());
}

bool hasTileBufferLayout(wafer::TileBufferType type, wafer::MemLayout layout) {
  return getTileBufferLayout(type).getValue() == layout;
}

bool hasTileBufferMemorySpace(wafer::TileBufferType type,
                              wafer::MemorySpace memorySpace) {
  return getTileBufferMemorySpace(type).getValue() == memorySpace;
}

bool hasStaticMismatch(int64_t lhs, int64_t rhs) {
  return lhs != mlir::ShapedType::kDynamic &&
         rhs != mlir::ShapedType::kDynamic && lhs != rhs;
}

bool checkedMul(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0)
    return false;
  if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

bool checkedAdd(int64_t lhs, int64_t rhs, int64_t &result) {
  if ((rhs > 0 && lhs > std::numeric_limits<int64_t>::max() - rhs) ||
      (rhs < 0 && lhs < std::numeric_limits<int64_t>::min() - rhs))
    return false;
  result = lhs + rhs;
  return true;
}

std::optional<int64_t> getPhysicalTileId(int64_t cardY, int64_t cardX,
                                         int64_t tileY, int64_t tileX,
                                         int64_t cardXCount, int64_t tileYCount,
                                         int64_t tileXCount) {
  int64_t cardBase = 0;
  if (!checkedMul(cardY, cardXCount, cardBase))
    return std::nullopt;
  int64_t cardIndex = 0;
  if (!checkedAdd(cardBase, cardX, cardIndex))
    return std::nullopt;

  int64_t tileBase = 0;
  if (!checkedMul(cardIndex, tileYCount, tileBase))
    return std::nullopt;
  int64_t tileRow = 0;
  if (!checkedAdd(tileBase, tileY, tileRow))
    return std::nullopt;

  int64_t tileIdBase = 0;
  if (!checkedMul(tileRow, tileXCount, tileIdBase))
    return std::nullopt;
  int64_t tileId = 0;
  if (!checkedAdd(tileIdBase, tileX, tileId))
    return std::nullopt;
  return tileId;
}

static std::optional<int64_t> getElementBitWidth(mlir::Type elementType) {
  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(elementType))
    return floatType.getWidth();
  if (auto integerType = mlir::dyn_cast<mlir::IntegerType>(elementType))
    return integerType.getWidth();
  if (mlir::isa<mlir::IndexType>(elementType))
    return 64;
  if (auto complexType = mlir::dyn_cast<mlir::ComplexType>(elementType)) {
    std::optional<int64_t> elementBits =
        getElementBitWidth(complexType.getElementType());
    if (!elementBits)
      return std::nullopt;
    int64_t complexBits = 0;
    if (!checkedMul(*elementBits, 2, complexBits))
      return std::nullopt;
    return complexBits;
  }
  return std::nullopt;
}

std::optional<int64_t>
getCompactTensorByteSize(mlir::RankedTensorType tensorType) {
  if (!tensorType.hasStaticShape())
    return std::nullopt;

  int64_t elements = 1;
  for (int64_t dim : tensorType.getShape()) {
    int64_t next = 0;
    if (!checkedMul(elements, dim, next))
      return std::nullopt;
    elements = next;
  }

  std::optional<int64_t> elementBits =
      getElementBitWidth(tensorType.getElementType());
  if (!elementBits || *elementBits <= 0)
    return std::nullopt;

  int64_t totalBits = 0;
  if (!checkedMul(elements, *elementBits, totalBits))
    return std::nullopt;
  return totalBits / 8 + (totalBits % 8 == 0 ? 0 : 1);
}

std::optional<int64_t> getCompactByteSize(mlir::Type type) {
  if (auto tileBufferType = mlir::dyn_cast<TileBufferType>(type))
    return getCompactTensorByteSize(getTileBufferTensorType(tileBufferType));
  if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type))
    return getCompactTensorByteSize(tensorType);
  return std::nullopt;
}

int64_t getCompactByteSizeOrUnknown(mlir::Type type) {
  std::optional<int64_t> bytes = getCompactByteSize(type);
  return bytes ? *bytes : -1;
}

void appendLayoutRequirement(
    llvm::SmallVectorImpl<wafer::WaferLayoutRequirement> &requirements,
    wafer::WaferValueRole role, unsigned index, wafer::TileBufferType type) {
  requirements.push_back({role, index, getTileBufferLayout(type).getValue(),
                          getTileBufferMemorySpace(type).getValue()});
}

void appendResourceEffect(
    llvm::SmallVectorImpl<wafer::WaferResourceEffect> &effects,
    wafer::WaferResourceKind resource, wafer::WaferResourceAccess access,
    wafer::WaferValueRole role, unsigned index, int64_t bytes) {
  effects.push_back({resource, access, role, index, bytes});
}

static mlir::Type getRequirementValueType(mlir::Operation *op,
                                          wafer::WaferValueRole role,
                                          unsigned index) {
  switch (role) {
  case wafer::WaferValueRole::Operand:
    if (index < op->getNumOperands())
      return op->getOperand(index).getType();
    return {};
  case wafer::WaferValueRole::Result:
    if (index < op->getNumResults())
      return op->getResult(index).getType();
    return {};
  case wafer::WaferValueRole::None:
    return {};
  }
  llvm_unreachable("unknown Wafer value role");
}

mlir::LogicalResult verifyLayoutRequirements(
    mlir::Operation *op,
    llvm::ArrayRef<wafer::WaferLayoutRequirement> requirements) {
  if (requirements.empty())
    return op->emitOpError("layout interface must expose at least one "
                           "operand/result layout requirement");

  for (const wafer::WaferLayoutRequirement &requirement : requirements) {
    mlir::Type valueType =
        getRequirementValueType(op, requirement.role, requirement.index);
    if (!valueType)
      return op->emitOpError("layout interface returned an invalid value "
                             "reference");

    auto tileBufferType = mlir::dyn_cast<TileBufferType>(valueType);
    if (!tileBufferType)
      return op->emitOpError("layout interface requirements must refer to "
                             "tile_buffer values");

    if (!hasTileBufferLayout(tileBufferType, requirement.layout))
      return op->emitOpError(
          "layout interface requirement does not match value mem_layout");
    if (!hasTileBufferMemorySpace(tileBufferType, requirement.memorySpace))
      return op->emitOpError(
          "layout interface requirement does not match value memory_space");
  }
  return mlir::success();
}

mlir::LogicalResult
verifyResourceEffects(mlir::Operation *op,
                      llvm::ArrayRef<wafer::WaferResourceEffect> effects) {
  if (effects.empty())
    return op->emitOpError(
        "resource interface must expose at least one effect");

  for (const wafer::WaferResourceEffect &effect : effects) {
    if (effect.bytes == 0 || effect.bytes < -1)
      return op->emitOpError("resource interface effect byte count must be "
                             "positive or unknown");
    if (effect.role == wafer::WaferValueRole::None)
      continue;
    if (!getRequirementValueType(op, effect.role, effect.index))
      return op->emitOpError("resource interface returned an invalid value "
                             "reference");
  }
  return mlir::success();
}

mlir::LogicalResult verifyCommP2P(mlir::Operation *op, mlir::Value buffer,
                                  mlir::IntegerAttr peer,
                                  mlir::IntegerAttr bytes,
                                  mlir::Type tokenType) {
  auto tileBufferType = mlir::dyn_cast<TileBufferType>(buffer.getType());
  if (!tileBufferType)
    return op->emitOpError("comm p2p buffer must be a tile_buffer");
  if (!hasTileBufferMemorySpace(tileBufferType, MemorySpace::SPM))
    return op->emitOpError("comm p2p buffer must use SPM memory space");
  if (!mlir::isa<mlir::async::TokenType>(tokenType))
    return op->emitOpError("comm p2p result must be an async token");
  if (peer.getInt() < 0)
    return op->emitOpError("comm peer must be non-negative");
  if (bytes.getInt() <= 0)
    return op->emitOpError("comm byte count must be positive");
  if (op->hasAttr(kWaferCommSlotAttrName)) {
    auto slot = op->getAttrOfType<mlir::IntegerAttr>(kWaferCommSlotAttrName);
    if (!slot)
      return op->emitOpError("comm slot must be an integer attr");
    if (slot.getInt() < 0)
      return op->emitOpError("comm slot must be non-negative");
  }

  mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return mlir::success();

  llvm::DenseSet<int64_t> activeTileIds;
  module.walk([&](PlacementMapOp placement) {
    int64_t logicalRankCount = placement.getLogicalRankCountAttr().getInt();
    int64_t cardXCount = placement.getCardXCountAttr().getInt();
    int64_t tileYCount = placement.getTileYCountAttr().getInt();
    int64_t tileXCount = placement.getTileXCountAttr().getInt();
    llvm::ArrayRef<int64_t> coords =
        placement.getPhysicalTileCoordsAttr().asArrayRef();
    if (logicalRankCount <= 0 ||
        static_cast<int64_t>(coords.size()) != logicalRankCount * 4)
      return;
    for (int64_t rank = 0; rank < logicalRankCount; ++rank) {
      int64_t base = rank * 4;
      std::optional<int64_t> tileId = getPhysicalTileId(
          coords[base], coords[base + 1], coords[base + 2], coords[base + 3],
          cardXCount, tileYCount, tileXCount);
      if (tileId)
        activeTileIds.insert(*tileId);
    }
  });
  if (!activeTileIds.empty() && !activeTileIds.contains(peer.getInt()))
    return op->emitOpError("comm peer must refer to an active placement tile");

  return mlir::success();
}

mlir::LogicalResult verifyCommWaitTokens(mlir::Operation *op,
                                         mlir::OperandRange tokens) {
  if (tokens.empty())
    return op->emitOpError("comm wait must have at least one token");
  for (mlir::Value token : tokens) {
    if (!mlir::isa<mlir::async::TokenType>(token.getType()))
      return op->emitOpError("comm wait operands must be async tokens");
  }
  return mlir::success();
}
bool hasAnyBatchedGemmAttrs(mlir::Operation *op) {
  return op->hasAttr("batch_count") || op->hasAttr("lhs_batch_dims") ||
         op->hasAttr("rhs_batch_dims") || op->hasAttr("result_batch_dims") ||
         op->hasAttr("lhs_m_dim") || op->hasAttr("lhs_contracting_dim") ||
         op->hasAttr("rhs_contracting_dim") || op->hasAttr("rhs_n_dim") ||
         op->hasAttr("result_m_dim") || op->hasAttr("result_n_dim");
}

static mlir::LogicalResult
readRequiredI64Attr(mlir::Operation *op, llvm::StringRef name, int64_t &value) {
  auto attr = op->getAttrOfType<mlir::IntegerAttr>(name);
  if (!attr)
    return op->emitOpError("GEMM batched form requires ") << name << " attr";
  value = attr.getInt();
  return mlir::success();
}

static mlir::LogicalResult
readRequiredDenseI64Attr(mlir::Operation *op, llvm::StringRef name,
                         llvm::SmallVectorImpl<int64_t> &values) {
  auto attr = op->getAttrOfType<mlir::DenseI64ArrayAttr>(name);
  if (!attr)
    return op->emitOpError("GEMM batched form requires ") << name << " attr";
  values.assign(attr.asArrayRef().begin(), attr.asArrayRef().end());
  return mlir::success();
}

static mlir::LogicalResult getBatchedGemmDimAttrs(mlir::Operation *op,
                                                  BatchedGemmDimAttrs &attrs) {
  if (mlir::failed(readRequiredI64Attr(op, "batch_count", attrs.batchCount)) ||
      mlir::failed(
          readRequiredDenseI64Attr(op, "lhs_batch_dims", attrs.lhsBatchDims)) ||
      mlir::failed(
          readRequiredDenseI64Attr(op, "rhs_batch_dims", attrs.rhsBatchDims)) ||
      mlir::failed(readRequiredDenseI64Attr(op, "result_batch_dims",
                                            attrs.resultBatchDims)) ||
      mlir::failed(readRequiredI64Attr(op, "lhs_m_dim", attrs.lhsMDim)) ||
      mlir::failed(readRequiredI64Attr(op, "lhs_contracting_dim",
                                       attrs.lhsContractingDim)) ||
      mlir::failed(readRequiredI64Attr(op, "rhs_contracting_dim",
                                       attrs.rhsContractingDim)) ||
      mlir::failed(readRequiredI64Attr(op, "rhs_n_dim", attrs.rhsNDim)) ||
      mlir::failed(readRequiredI64Attr(op, "result_m_dim", attrs.resultMDim)) ||
      mlir::failed(readRequiredI64Attr(op, "result_n_dim", attrs.resultNDim)))
    return mlir::failure();
  return mlir::success();
}

static mlir::LogicalResult
verifyDimsInRangeAndUnique(mlir::Operation *op, llvm::StringRef name,
                           llvm::ArrayRef<int64_t> dims, int64_t rank) {
  if (dims.empty())
    return op->emitOpError("GEMM ") << name << " must be non-empty";

  llvm::DenseSet<int64_t> seen;
  for (int64_t dim : dims) {
    if (dim < 0 || dim >= rank)
      return op->emitOpError("GEMM ")
             << name << " entries must be within tensor rank";
    if (!seen.insert(dim).second)
      return op->emitOpError("GEMM ") << name << " entries must be unique";
  }
  return mlir::success();
}

static mlir::LogicalResult
verifyDimsCoverRank(mlir::Operation *op, llvm::StringRef name,
                    llvm::ArrayRef<int64_t> batchDims, int64_t dim0,
                    int64_t dim1, int64_t rank) {
  llvm::DenseSet<int64_t> seen;
  for (int64_t dim : batchDims)
    seen.insert(dim);
  if (dim0 < 0 || dim0 >= rank || dim1 < 0 || dim1 >= rank)
    return op->emitOpError("GEMM ")
           << name << " dimension attrs must be within tensor rank";
  if (!seen.insert(dim0).second || !seen.insert(dim1).second)
    return op->emitOpError("GEMM ")
           << name << " dimension attrs must be unique";
  if (static_cast<int64_t>(seen.size()) != rank)
    return op->emitOpError("GEMM ")
           << name << " dimension attrs must cover tensor rank";
  return mlir::success();
}

mlir::LogicalResult verifyBatchedGemmTileContract(
    mlir::Operation *op, mlir::RankedTensorType lhsTensor,
    mlir::RankedTensorType rhsTensor, mlir::RankedTensorType resultTensor,
    BatchedGemmDimAttrs &attrs) {
  if (mlir::failed(getBatchedGemmDimAttrs(op, attrs)))
    return mlir::failure();

  int64_t rank = lhsTensor.getRank();
  if (rank < 3 || rhsTensor.getRank() != rank || resultTensor.getRank() != rank)
    return op->emitOpError(
        "GEMM batched form expects operands and result to have the same rank "
        "of at least 3");
  if (!lhsTensor.hasStaticShape() || !rhsTensor.hasStaticShape() ||
      !resultTensor.hasStaticShape())
    return op->emitOpError("GEMM batched form requires static tensor shapes");

  if (attrs.batchCount <= 0)
    return op->emitOpError("GEMM batch_count attr must be positive");
  if (attrs.lhsBatchDims.size() != attrs.rhsBatchDims.size() ||
      attrs.lhsBatchDims.size() != attrs.resultBatchDims.size())
    return op->emitOpError(
        "GEMM batch dimension attrs must have matching lengths");

  if (mlir::failed(verifyDimsInRangeAndUnique(op, "lhs_batch_dims",
                                              attrs.lhsBatchDims, rank)) ||
      mlir::failed(verifyDimsInRangeAndUnique(op, "rhs_batch_dims",
                                              attrs.rhsBatchDims, rank)) ||
      mlir::failed(verifyDimsInRangeAndUnique(op, "result_batch_dims",
                                              attrs.resultBatchDims, rank)) ||
      mlir::failed(verifyDimsCoverRank(op, "lhs", attrs.lhsBatchDims,
                                       attrs.lhsMDim, attrs.lhsContractingDim,
                                       rank)) ||
      mlir::failed(verifyDimsCoverRank(op, "rhs", attrs.rhsBatchDims,
                                       attrs.rhsContractingDim, attrs.rhsNDim,
                                       rank)) ||
      mlir::failed(verifyDimsCoverRank(op, "result", attrs.resultBatchDims,
                                       attrs.resultMDim, attrs.resultNDim,
                                       rank)))
    return mlir::failure();

  int64_t inferredBatchCount = 1;
  for (auto [lhsBatchDim, rhsBatchDim, resultBatchDim] : llvm::zip(
           attrs.lhsBatchDims, attrs.rhsBatchDims, attrs.resultBatchDims)) {
    if (hasStaticMismatch(lhsTensor.getDimSize(lhsBatchDim),
                          rhsTensor.getDimSize(rhsBatchDim)) ||
        hasStaticMismatch(lhsTensor.getDimSize(lhsBatchDim),
                          resultTensor.getDimSize(resultBatchDim)))
      return op->emitOpError("GEMM batch dimensions must have matching shapes");
    if (!checkedMul(inferredBatchCount, resultTensor.getDimSize(resultBatchDim),
                    inferredBatchCount))
      return op->emitOpError("GEMM batch_count is too large to verify");
  }

  if (attrs.batchCount != inferredBatchCount)
    return op->emitOpError(
        "GEMM batch_count attr must match product of result batch dimensions");

  if (hasStaticMismatch(lhsTensor.getDimSize(attrs.lhsMDim),
                        resultTensor.getDimSize(attrs.resultMDim)))
    return op->emitOpError(
        "GEMM lhs M dimension must match result M dimension");
  if (hasStaticMismatch(rhsTensor.getDimSize(attrs.rhsNDim),
                        resultTensor.getDimSize(attrs.resultNDim)))
    return op->emitOpError(
        "GEMM rhs N dimension must match result N dimension");
  if (hasStaticMismatch(lhsTensor.getDimSize(attrs.lhsContractingDim),
                        rhsTensor.getDimSize(attrs.rhsContractingDim)))
    return op->emitOpError("GEMM lhs K dimension must match rhs K dimension");

  return mlir::success();
}
static unsigned getElementwiseArity(ComputeElementwiseKind kind) {
  switch (kind) {
  case ComputeElementwiseKind::Add:
  case ComputeElementwiseKind::Sub:
  case ComputeElementwiseKind::Mul:
  case ComputeElementwiseKind::Div:
  case ComputeElementwiseKind::Max:
  case ComputeElementwiseKind::Min:
  case ComputeElementwiseKind::Eq:
  case ComputeElementwiseKind::Ne:
  case ComputeElementwiseKind::Lt:
  case ComputeElementwiseKind::Le:
  case ComputeElementwiseKind::Gt:
  case ComputeElementwiseKind::Ge:
    return 2;
  case ComputeElementwiseKind::Neg:
  case ComputeElementwiseKind::Recip:
  case ComputeElementwiseKind::Sqrt:
  case ComputeElementwiseKind::Rsqrt:
  case ComputeElementwiseKind::Exp:
  case ComputeElementwiseKind::Tanh:
    return 1;
  }
  llvm_unreachable("unknown compute elementwise kind");
}

static bool isRelationKind(ComputeElementwiseKind kind) {
  switch (kind) {
  case ComputeElementwiseKind::Eq:
  case ComputeElementwiseKind::Ne:
  case ComputeElementwiseKind::Lt:
  case ComputeElementwiseKind::Le:
  case ComputeElementwiseKind::Gt:
  case ComputeElementwiseKind::Ge:
    return true;
  default:
    return false;
  }
}

mlir::LogicalResult verifyElementwiseTileContract(mlir::Operation *op,
                                                  ComputeElementwiseKind kind,
                                                  mlir::ValueRange inputs,
                                                  mlir::Type resultType) {
  auto resultTileType = mlir::dyn_cast<TileBufferType>(resultType);
  if (!resultTileType)
    return op->emitOpError("expects tile_buffer result");

  unsigned expectedArity = getElementwiseArity(kind);
  if (inputs.size() != expectedArity)
    return op->emitOpError("elementwise kind expects ")
           << expectedArity << " operand(s), got " << inputs.size();

  if (!hasTileBufferMemorySpace(resultTileType, MemorySpace::SPM))
    return op->emitOpError("elementwise result must use SPM memory space");
  if (!hasTileBufferLayout(resultTileType, MemLayout::Tensor))
    return op->emitOpError("elementwise result must use tensor mem_layout");

  mlir::RankedTensorType resultTensor = getTileBufferTensorType(resultTileType);
  mlir::ArrayAttr indexingMaps =
      op->getAttrOfType<mlir::ArrayAttr>("indexing_maps");

  if (indexingMaps) {
    if (indexingMaps.size() != inputs.size() + 1)
      return op->emitOpError(
          "elementwise indexing map count must match operands plus result");

    auto resultMapAttr = mlir::dyn_cast<mlir::AffineMapAttr>(
        indexingMaps[indexingMaps.size() - 1]);
    if (!resultMapAttr)
      return op->emitOpError(
          "elementwise indexing_maps entries must be affine maps");
    mlir::AffineMap resultMap = resultMapAttr.getValue();
    if (resultMap.getNumDims() != resultTensor.getRank() ||
        resultMap.getNumSymbols() != 0 || !resultMap.isIdentity())
      return op->emitOpError(
          "elementwise result indexing map must be identity");
  }

  for (auto [index, input] : llvm::enumerate(inputs)) {
    auto inputTileType = mlir::dyn_cast<TileBufferType>(input.getType());
    if (!inputTileType)
      return op->emitOpError("expects tile_buffer operands");
    if (!hasTileBufferMemorySpace(inputTileType, MemorySpace::SPM))
      return op->emitOpError("elementwise operands must use SPM memory space");
    if (!hasTileBufferLayout(inputTileType, MemLayout::Tensor))
      return op->emitOpError("elementwise operands must use tensor mem_layout");
    mlir::RankedTensorType inputTensor = getTileBufferTensorType(inputTileType);
    if (isRelationKind(kind)) {
      if (!resultTensor.getElementType().isInteger(1))
        return op->emitOpError("relation result element type must be i1");
      if (index > 0) {
        auto firstInputType =
            mlir::cast<TileBufferType>(inputs.front().getType());
        mlir::RankedTensorType firstInputTensor =
            getTileBufferTensorType(firstInputType);
        if (inputTensor.getElementType() != firstInputTensor.getElementType())
          return op->emitOpError(
              "relation operand element types must match each other");
      }
    } else if (inputTensor.getElementType() != resultTensor.getElementType()) {
      return op->emitOpError(
          "elementwise operand element types must match result element type");
    }
    if (!indexingMaps && isRelationKind(kind)) {
      if (inputTensor.getShape() != resultTensor.getShape())
        return op->emitOpError(
            "relation operand shapes must match result shape");
      continue;
    }
    if (!indexingMaps && inputTensor != resultTensor) {
      return op->emitOpError(
          "elementwise operand tensor types must match result tensor type");
    }
    if (!indexingMaps)
      continue;

    auto inputMapAttr =
        mlir::dyn_cast<mlir::AffineMapAttr>(indexingMaps[index]);
    if (!inputMapAttr)
      return op->emitOpError(
          "elementwise indexing_maps entries must be affine maps");
    mlir::AffineMap inputMap = inputMapAttr.getValue();
    if (inputMap.getNumDims() != resultTensor.getRank() ||
        inputMap.getNumSymbols() != 0 ||
        inputMap.getNumResults() != inputTensor.getRank() ||
        !inputMap.isProjectedPermutation())
      return op->emitOpError(
          "elementwise input indexing maps must be projected permutations");

    for (auto [dim, expr] : llvm::enumerate(inputMap.getResults())) {
      auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expr);
      if (!dimExpr || dimExpr.getPosition() >= resultTensor.getRank())
        return op->emitOpError(
            "elementwise input indexing maps must use result dimensions");
      if (hasStaticMismatch(inputTensor.getDimSize(dim),
                            resultTensor.getDimSize(dimExpr.getPosition())))
        return op->emitOpError(
            "elementwise indexing map dimension must match tensor shape");
    }
  }

  return mlir::success();
}
mlir::LogicalResult verifyReduceTileContract(mlir::Operation *op,
                                             mlir::Value input,
                                             mlir::Type resultType) {
  auto inputTileType = mlir::dyn_cast<TileBufferType>(input.getType());
  if (!inputTileType)
    return op->emitOpError("expects tile_buffer input");
  auto resultTileType = mlir::dyn_cast<TileBufferType>(resultType);
  if (!resultTileType)
    return op->emitOpError("expects tile_buffer result");

  if (!hasTileBufferMemorySpace(inputTileType, MemorySpace::SPM) ||
      !hasTileBufferMemorySpace(resultTileType, MemorySpace::SPM))
    return op->emitOpError("reduce operands/results must use SPM memory space");

  mlir::RankedTensorType inputTensor = getTileBufferTensorType(inputTileType);
  mlir::RankedTensorType resultTensor = getTileBufferTensorType(resultTileType);
  MemLayout expectedInputLayout =
      inputTensor.getRank() > 2 ? MemLayout::NCx : MemLayout::Cx;
  MemLayout expectedResultLayout =
      resultTensor.getRank() > 2 ? MemLayout::NCx : MemLayout::Cx;
  if (!hasTileBufferLayout(inputTileType, expectedInputLayout) ||
      !hasTileBufferLayout(resultTileType, expectedResultLayout)) {
    if (inputTensor.getRank() > 2 || resultTensor.getRank() > 2)
      return op->emitOpError(
          "reduce rank > 2 operands/results must use ncx mem_layout");
    return op->emitOpError(
        "reduce rank <= 2 operands/results must use cx mem_layout");
  }

  if (inputTensor.getElementType() != resultTensor.getElementType())
    return op->emitOpError(
        "reduce input element type must match result element type");

  auto dimensions = op->getAttrOfType<mlir::DenseI64ArrayAttr>("dimensions");
  if (!dimensions)
    return op->emitOpError("requires reduce dimensions attr");
  llvm::ArrayRef<int64_t> dims = dimensions.asArrayRef();
  if (dims.empty())
    return op->emitOpError("reduce dimensions must be non-empty");
  if (static_cast<int64_t>(dims.size()) > inputTensor.getRank())
    return op->emitOpError("reduce dimensions cannot exceed input tensor rank");

  llvm::DenseSet<int64_t> reducedDims;
  for (int64_t dim : dims) {
    if (dim < 0 || dim >= inputTensor.getRank())
      return op->emitOpError(
          "reduce dimensions must be within input tensor rank");
    if (!reducedDims.insert(dim).second)
      return op->emitOpError("reduce dimensions must be unique");
  }

  if (resultTensor.getRank() !=
      inputTensor.getRank() - static_cast<int64_t>(dims.size()))
    return op->emitOpError(
        "reduce result rank must match input rank minus reduce dimensions");

  int64_t resultDim = 0;
  for (int64_t inputDim = 0; inputDim < inputTensor.getRank(); ++inputDim) {
    if (reducedDims.contains(inputDim))
      continue;
    if (hasStaticMismatch(inputTensor.getDimSize(inputDim),
                          resultTensor.getDimSize(resultDim)))
      return op->emitOpError(
          "reduce result shape must match non-reduced input dimensions");
    ++resultDim;
  }

  mlir::Attribute initValue = op->getAttr("init_value");
  if (initValue) {
    auto typedInit = mlir::dyn_cast<mlir::TypedAttr>(initValue);
    if (!typedInit || typedInit.getType() != inputTensor.getElementType())
      return op->emitOpError(
          "reduce init_value type must match input element type");
    return mlir::success();
  }

  if (op->getNumOperands() >= 2) {
    mlir::Type initType = op->getOperand(1).getType();
    if (initType != inputTensor.getElementType())
      return op->emitOpError(
          "reduce init operand type must match input element type");
    return mlir::success();
  }

  return op->emitOpError(
      "requires reduce init_value attr or scalar init operand");
}

} // namespace wafer::detail
