//===- WaferDialect.cpp - Wafer dialect implementation -------------------===//

#include "Wafer/Dialect/Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include <cstdint>
#include <limits>
#include <optional>

using namespace wafer;

#include "Wafer/Dialect/Wafer/IR/WaferEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Wafer/Dialect/Wafer/IR/WaferAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Wafer/Dialect/Wafer/IR/WaferTypes.cpp.inc"

#include "Wafer/Dialect/Wafer/IR/WaferOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Wafer/Dialect/Wafer/IR/WaferOps.cpp.inc"

mlir::LogicalResult
TileBufferType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                       mlir::Type tensorType, mlir::Attribute layout,
                       mlir::Attribute memorySpace) {
  if (!mlir::isa<mlir::RankedTensorType>(tensorType))
    return emitError() << "tile_buffer logical type must be a ranked tensor";
  if (!mlir::isa<MemLayoutAttr>(layout))
    return emitError() << "tile_buffer layout must be a wafer mem_layout attr";
  if (!mlir::isa<MemorySpaceAttr>(memorySpace))
    return emitError()
           << "tile_buffer memory space must be a wafer memory_space attr";
  return mlir::success();
}

static bool isSPMMemRef(mlir::Type type) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memrefType)
    return false;
  auto memorySpace = mlir::dyn_cast_or_null<wafer::MemorySpaceAttr>(
      memrefType.getMemorySpace());
  return memorySpace && memorySpace.getValue() == wafer::MemorySpace::SPM;
}

static bool isSPMTileBuffer(mlir::Type type) {
  auto tileBufferType = mlir::dyn_cast<wafer::TileBufferType>(type);
  if (!tileBufferType)
    return false;
  auto memorySpace =
      mlir::cast<wafer::MemorySpaceAttr>(tileBufferType.getMemorySpace());
  return memorySpace.getValue() == wafer::MemorySpace::SPM;
}

static wafer::MemLayoutAttr getTileBufferLayout(wafer::TileBufferType type) {
  return mlir::cast<wafer::MemLayoutAttr>(type.getLayout());
}

static wafer::MemorySpaceAttr
getTileBufferMemorySpace(wafer::TileBufferType type) {
  return mlir::cast<wafer::MemorySpaceAttr>(type.getMemorySpace());
}

static mlir::RankedTensorType
getTileBufferTensorType(wafer::TileBufferType type) {
  return mlir::cast<mlir::RankedTensorType>(type.getTensorType());
}

static bool hasTileBufferLayout(wafer::TileBufferType type,
                                wafer::MemLayout layout) {
  return getTileBufferLayout(type).getValue() == layout;
}

static bool hasTileBufferMemorySpace(wafer::TileBufferType type,
                                     wafer::MemorySpace memorySpace) {
  return getTileBufferMemorySpace(type).getValue() == memorySpace;
}

static bool hasStaticMismatch(int64_t lhs, int64_t rhs) {
  return lhs != mlir::ShapedType::kDynamic &&
         rhs != mlir::ShapedType::kDynamic && lhs != rhs;
}

static bool checkedMul(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0)
    return false;
  if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

static bool checkedAdd(int64_t lhs, int64_t rhs, int64_t &result) {
  if ((rhs > 0 && lhs > std::numeric_limits<int64_t>::max() - rhs) ||
      (rhs < 0 && lhs < std::numeric_limits<int64_t>::min() - rhs))
    return false;
  result = lhs + rhs;
  return true;
}

static std::optional<int64_t>
getPhysicalTileId(int64_t cardY, int64_t cardX, int64_t tileY, int64_t tileX,
                  int64_t cardXCount, int64_t tileYCount, int64_t tileXCount) {
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

static std::optional<int64_t>
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

static mlir::LogicalResult
verifyCommP2P(mlir::Operation *op, mlir::Value buffer, mlir::IntegerAttr peer,
              mlir::IntegerAttr bytes, mlir::Type tokenType) {
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
      std::optional<int64_t> tileId =
          getPhysicalTileId(coords[base], coords[base + 1], coords[base + 2],
                            coords[base + 3], cardXCount, tileYCount,
                            tileXCount);
      if (tileId)
        activeTileIds.insert(*tileId);
    }
  });
  if (!activeTileIds.empty() && !activeTileIds.contains(peer.getInt()))
    return op->emitOpError("comm peer must refer to an active placement tile");

  return mlir::success();
}

mlir::LogicalResult DdrExternalBindingOp::verify() {
  auto tensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(getValue().getType());
  if (!tensorType || !tensorType.hasStaticShape())
    return emitOpError(
        "DDR external binding value must be a static ranked tensor");

  int64_t bytes = getBytesAttr().getInt();
  if (bytes <= 0)
    return emitOpError("DDR external binding bytes must be positive");

  std::optional<int64_t> expectedBytes = getCompactTensorByteSize(tensorType);
  if (!expectedBytes)
    return emitOpError("DDR external binding compact tensor storage size is "
                       "not representable");
  if (bytes != *expectedBytes)
    return emitOpError(
               "DDR external binding bytes must match compact tensor storage "
               "size, got ")
           << bytes << " and expected " << *expectedBytes;

  if (getAlignmentAttr().getInt() <= 0)
    return emitOpError("DDR external binding alignment must be positive");

  bool readOnly = getReadOnlyAttr().getValue();
  if (getKindAttr().getValue() == DdrBindingKind::Input && !readOnly)
    return emitOpError("DDR external input binding must be read-only");
  if (getKindAttr().getValue() == DdrBindingKind::Output && readOnly)
    return emitOpError("DDR external output binding must be writable");

  return mlir::success();
}

static mlir::LogicalResult
verifyIssueOnlyWaitPolicy(mlir::Operation *op,
                          wafer::AbiWaitPolicyAttr policy) {
  if (policy.getValue() != wafer::AbiWaitPolicy::IssueOnly)
    return op->emitOpError(
        "C ABI skeleton ops must use issue_only wait policy");
  return mlir::success();
}

struct BatchedGemmDimAttrs {
  llvm::SmallVector<int64_t, 2> lhsBatchDims;
  llvm::SmallVector<int64_t, 2> rhsBatchDims;
  llvm::SmallVector<int64_t, 2> resultBatchDims;
  int64_t batchCount = 0;
  int64_t lhsMDim = -1;
  int64_t lhsContractingDim = -1;
  int64_t rhsContractingDim = -1;
  int64_t rhsNDim = -1;
  int64_t resultMDim = -1;
  int64_t resultNDim = -1;
};

static bool hasAnyBatchedGemmAttrs(mlir::Operation *op) {
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

static mlir::LogicalResult verifyBatchedGemmTileContract(
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

mlir::LogicalResult AbiRdma1DOp::verify() {
  if (mlir::failed(
          verifyIssueOnlyWaitPolicy(getOperation(), getWaitPolicyAttr())))
    return mlir::failure();

  auto sourceType =
      mlir::dyn_cast<mlir::RankedTensorType>(getSource().getType());
  auto resultType = mlir::dyn_cast<TileBufferType>(getResult().getType());
  if (!sourceType || !resultType)
    return emitOpError(
        "ABI RDMA expects ranked tensor source and tile_buffer result");

  if (resultType.getTensorType() != sourceType)
    return emitOpError(
        "ABI RDMA result tensor type must match source tensor type");
  if (!hasTileBufferMemorySpace(resultType, MemorySpace::SPM))
    return emitOpError("ABI RDMA result must use SPM memory space");
  if (!hasTileBufferLayout(resultType, MemLayout::Tensor))
    return emitOpError("ABI RDMA result must use tensor mem_layout");

  std::optional<int64_t> expectedBytes = getCompactTensorByteSize(sourceType);
  if (!expectedBytes)
    return emitOpError(
        "ABI RDMA compact tensor transfer size is not representable");
  if (getBytesAttr().getInt() != *expectedBytes)
    return emitOpError(
        "ABI RDMA byte count must match compact tensor transfer size");

  return mlir::success();
}

mlir::LogicalResult AbiWdma1DOp::verify() {
  if (mlir::failed(
          verifyIssueOnlyWaitPolicy(getOperation(), getWaitPolicyAttr())))
    return mlir::failure();

  auto sourceType = mlir::dyn_cast<TileBufferType>(getSource().getType());
  auto destType = mlir::dyn_cast<mlir::RankedTensorType>(getDest().getType());
  if (!sourceType || !destType)
    return emitOpError(
        "ABI WDMA expects tile_buffer source and ranked tensor dest");

  if (sourceType.getTensorType() != destType)
    return emitOpError(
        "ABI WDMA source tensor type must match dest tensor type");
  if (!hasTileBufferMemorySpace(sourceType, MemorySpace::SPM))
    return emitOpError("ABI WDMA source must use SPM memory space");
  if (!hasTileBufferLayout(sourceType, MemLayout::Tensor))
    return emitOpError("ABI WDMA source must use tensor mem_layout");

  std::optional<int64_t> expectedBytes = getCompactTensorByteSize(destType);
  if (!expectedBytes)
    return emitOpError(
        "ABI WDMA compact tensor transfer size is not representable");
  if (getBytesAttr().getInt() != *expectedBytes)
    return emitOpError(
        "ABI WDMA byte count must match compact tensor transfer size");

  return mlir::success();
}

mlir::LogicalResult AbiGemmOp::verify() {
  if (mlir::failed(
          verifyIssueOnlyWaitPolicy(getOperation(), getWaitPolicyAttr())))
    return mlir::failure();

  auto lhsType = mlir::dyn_cast<TileBufferType>(getLhs().getType());
  auto rhsType = mlir::dyn_cast<TileBufferType>(getRhs().getType());
  auto resultType = mlir::dyn_cast<TileBufferType>(getResult().getType());
  if (!lhsType || !rhsType || !resultType)
    return emitOpError("ABI GEMM expects tile_buffer operands and result");

  for (TileBufferType type : {lhsType, rhsType, resultType}) {
    if (!hasTileBufferMemorySpace(type, MemorySpace::SPM))
      return emitOpError("ABI GEMM tile buffers must use SPM memory space");
    if (!hasTileBufferLayout(type, MemLayout::Cx))
      return emitOpError("ABI GEMM tile buffers must use cx mem_layout");
  }

  mlir::RankedTensorType lhsTensor = getTileBufferTensorType(lhsType);
  mlir::RankedTensorType rhsTensor = getTileBufferTensorType(rhsType);
  mlir::RankedTensorType resultTensor = getTileBufferTensorType(resultType);

  if (lhsTensor.getElementType() != rhsTensor.getElementType() ||
      lhsTensor.getElementType() != resultTensor.getElementType())
    return emitOpError("ABI GEMM operand and result element types must match");

  if (lhsTensor.getRank() != 2 || rhsTensor.getRank() != 2 ||
      resultTensor.getRank() != 2) {
    BatchedGemmDimAttrs attrs;
    if (mlir::failed(verifyBatchedGemmTileContract(
            getOperation(), lhsTensor, rhsTensor, resultTensor, attrs)))
      return mlir::failure();

    if (getMAttr().getInt() != lhsTensor.getDimSize(attrs.lhsMDim) ||
        getKAttr().getInt() != lhsTensor.getDimSize(attrs.lhsContractingDim) ||
        getNAttr().getInt() != rhsTensor.getDimSize(attrs.rhsNDim))
      return emitOpError("ABI GEMM m/k/n attrs must match batched GEMM dims");
    return mlir::success();
  }

  if (hasAnyBatchedGemmAttrs(getOperation()))
    return emitOpError(
        "ABI GEMM rank-2 form must not carry batched GEMM attrs");

  if (hasStaticMismatch(lhsTensor.getDimSize(1), rhsTensor.getDimSize(0)))
    return emitOpError("ABI GEMM lhs K dimension must match rhs K dimension");
  if (hasStaticMismatch(lhsTensor.getDimSize(0), resultTensor.getDimSize(0)) ||
      hasStaticMismatch(rhsTensor.getDimSize(1), resultTensor.getDimSize(1)))
    return emitOpError("ABI GEMM result shape must be lhs M by rhs N");

  if (getMAttr().getInt() != lhsTensor.getDimSize(0) ||
      getKAttr().getInt() != lhsTensor.getDimSize(1) ||
      getNAttr().getInt() != rhsTensor.getDimSize(1))
    return emitOpError("ABI GEMM m/k/n attrs must match tile buffer shapes");

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

static mlir::LogicalResult
verifyElementwiseTileContract(mlir::Operation *op, ComputeElementwiseKind kind,
                              mlir::ValueRange inputs, mlir::Type resultType) {
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
    if (inputTensor.getElementType() != resultTensor.getElementType())
      return op->emitOpError(
          "elementwise operand element types must match result element type");
    if (!indexingMaps && inputTensor != resultTensor)
      return op->emitOpError(
          "elementwise operand tensor types must match result tensor type");
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

mlir::LogicalResult AbiElementwiseOp::verify() {
  if (mlir::failed(
          verifyIssueOnlyWaitPolicy(getOperation(), getWaitPolicyAttr())))
    return mlir::failure();

  return verifyElementwiseTileContract(getOperation(), getKindAttr().getValue(),
                                       getInputs(), getResult().getType());
}

static mlir::LogicalResult verifyReduceTileContract(mlir::Operation *op,
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
  if (!hasTileBufferLayout(inputTileType, MemLayout::Tensor) ||
      !hasTileBufferLayout(resultTileType, MemLayout::Tensor))
    return op->emitOpError(
        "reduce operands/results must use tensor mem_layout");

  mlir::RankedTensorType inputTensor = getTileBufferTensorType(inputTileType);
  mlir::RankedTensorType resultTensor = getTileBufferTensorType(resultTileType);
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
  if (!initValue)
    return op->emitOpError("requires reduce init_value attr");
  auto typedInit = mlir::dyn_cast<mlir::TypedAttr>(initValue);
  if (!typedInit || typedInit.getType() != inputTensor.getElementType())
    return op->emitOpError(
        "reduce init_value type must match input element type");

  return mlir::success();
}

mlir::LogicalResult AbiReduceOp::verify() {
  if (mlir::failed(
          verifyIssueOnlyWaitPolicy(getOperation(), getWaitPolicyAttr())))
    return mlir::failure();
  return verifyReduceTileContract(getOperation(), getInput(),
                                  getResult().getType());
}

mlir::LogicalResult GroupOp::verify() {
  if (getNumResults() != getOuts().size())
    return emitOpError("expected result count to match outs count, got ")
           << getNumResults() << " results and " << getOuts().size() << " outs";

  for (auto [index, resultAndOut] :
       llvm::enumerate(llvm::zip(getResults(), getOuts()))) {
    mlir::Type resultType = std::get<0>(resultAndOut).getType();
    mlir::Type outType = std::get<1>(resultAndOut).getType();
    if (resultType != outType)
      return emitOpError("result type ")
             << resultType << " does not match outs type " << outType
             << " at index " << index;
  }

  for (auto input : getInputs()) {
    if (isSPMMemRef(input.getType()))
      return emitOpError("does not accept SPM memref inputs");
  }

  if (getBody().empty())
    return emitOpError("expected non-empty body region");

  mlir::Block &block = getBody().front();
  size_t expectedBlockArgs = getInputs().size() + getOuts().size();
  if (block.getNumArguments() != expectedBlockArgs)
    return emitOpError("expected ")
           << expectedBlockArgs
           << " body block arguments matching group ins plus outs, got "
           << block.getNumArguments();

  unsigned blockArgIndex = 0;
  for (auto input : getInputs()) {
    mlir::Type blockArgType = block.getArgument(blockArgIndex).getType();
    if (blockArgType != input.getType())
      return emitOpError("body block argument type ")
             << blockArgType << " does not match input type " << input.getType()
             << " at index " << blockArgIndex;
    if (isSPMMemRef(blockArgType))
      return emitOpError("does not accept SPM memref body arguments");
    ++blockArgIndex;
  }
  for (auto out : getOuts()) {
    mlir::Type blockArgType = block.getArgument(blockArgIndex).getType();
    if (blockArgType != out.getType())
      return emitOpError("body block argument type ")
             << blockArgType << " does not match outs type " << out.getType()
             << " at index " << blockArgIndex;
    ++blockArgIndex;
  }

  auto yield = mlir::dyn_cast<GroupYieldOp>(block.getTerminator());
  if (!yield)
    return emitOpError("expected wafer.group_yield terminator");

  if (yield.getValues().size() != getNumResults())
    return emitOpError(
               "expected group_yield value count to match result count, got ")
           << yield.getValues().size() << " values and " << getNumResults()
           << " results";

  for (auto [index, yieldedAndResult] :
       llvm::enumerate(llvm::zip(yield.getValues(), getResults()))) {
    mlir::Type yieldedType = std::get<0>(yieldedAndResult).getType();
    mlir::Type resultType = std::get<1>(yieldedAndResult).getType();
    if (yieldedType != resultType)
      return emitOpError("group yield type ")
             << yieldedType << " does not match result type " << resultType
             << " at index " << index;
  }

  for (mlir::NamedAttribute attr : getOperation()->getAttrs()) {
    if (attr.getName() != getOperandSegmentSizesAttrName())
      return emitOpError("does not accept semantic attributes");
  }

  return mlir::success();
}

mlir::LogicalResult CommRecvOp::verify() {
  return verifyCommP2P(getOperation(), getBuffer(), getPeerAttr(),
                       getBytesAttr(), getToken().getType());
}

mlir::LogicalResult CommSendOp::verify() {
  return verifyCommP2P(getOperation(), getBuffer(), getPeerAttr(),
                       getBytesAttr(), getToken().getType());
}

mlir::LogicalResult CommWaitOp::verify() {
  if (getTokens().empty())
    return emitOpError("comm wait must have at least one token");
  for (mlir::Value token : getTokens()) {
    if (!mlir::isa<mlir::async::TokenType>(token.getType()))
      return emitOpError("comm wait operands must be async tokens");
  }
  return mlir::success();
}

mlir::LogicalResult ComputeGemmOp::verify() {
  auto lhsType = mlir::dyn_cast<TileBufferType>(getLhs().getType());
  auto rhsType = mlir::dyn_cast<TileBufferType>(getRhs().getType());
  auto resultType = mlir::dyn_cast<TileBufferType>(getResult().getType());
  if (!lhsType || !rhsType || !resultType)
    return emitOpError("expects tile_buffer operands and result");

  for (TileBufferType type : {lhsType, rhsType, resultType}) {
    if (!hasTileBufferMemorySpace(type, MemorySpace::SPM))
      return emitOpError("gemm tile buffers must use SPM memory space");
    if (!hasTileBufferLayout(type, MemLayout::Cx))
      return emitOpError("gemm tile buffers must use cx mem_layout");
  }

  mlir::RankedTensorType lhsTensor = getTileBufferTensorType(lhsType);
  mlir::RankedTensorType rhsTensor = getTileBufferTensorType(rhsType);
  mlir::RankedTensorType resultTensor = getTileBufferTensorType(resultType);

  if (lhsTensor.getElementType() != rhsTensor.getElementType() ||
      lhsTensor.getElementType() != resultTensor.getElementType())
    return emitOpError("gemm operand and result element types must match");

  if (lhsTensor.getRank() != 2 || rhsTensor.getRank() != 2 ||
      resultTensor.getRank() != 2) {
    BatchedGemmDimAttrs attrs;
    return verifyBatchedGemmTileContract(getOperation(), lhsTensor, rhsTensor,
                                         resultTensor, attrs);
  }

  if (hasAnyBatchedGemmAttrs(getOperation()))
    return emitOpError("gemm rank-2 form must not carry batched GEMM attrs");

  if (hasStaticMismatch(lhsTensor.getDimSize(1), rhsTensor.getDimSize(0)))
    return emitOpError("gemm lhs K dimension must match rhs K dimension");
  if (hasStaticMismatch(lhsTensor.getDimSize(0), resultTensor.getDimSize(0)) ||
      hasStaticMismatch(rhsTensor.getDimSize(1), resultTensor.getDimSize(1)))
    return emitOpError("gemm result shape must be lhs M by rhs N");

  return mlir::success();
}

mlir::LogicalResult ComputeElementwiseOp::verify() {
  return verifyElementwiseTileContract(getOperation(), getKindAttr().getValue(),
                                       getInputs(), getResult().getType());
}

mlir::LogicalResult ComputeReduceOp::verify() {
  return verifyReduceTileContract(getOperation(), getInput(),
                                  getResult().getType());
}

mlir::LogicalResult LayoutMaterializeOp::verify() {
  auto sourceType = mlir::dyn_cast<TileBufferType>(getSource().getType());
  auto resultType = mlir::dyn_cast<TileBufferType>(getResult().getType());
  if (!sourceType || !resultType)
    return emitOpError("expects tile_buffer source and result types");

  if (sourceType.getTensorType() != resultType.getTensorType())
    return emitOpError("layout materialize must preserve logical tensor type");

  if (getTileBufferMemorySpace(sourceType).getValue() !=
      getTileBufferMemorySpace(resultType).getValue())
    return emitOpError("layout materialize must preserve memory space");

  if (getTileBufferLayout(sourceType).getValue() ==
      getTileBufferLayout(resultType).getValue())
    return emitOpError("layout materialize must change mem_layout");

  return mlir::success();
}

mlir::LogicalResult LaunchOp::verify() {
  if (getPackageRefAttr().getValue().empty())
    return emitOpError("launch package_ref must be non-empty");
  if (getSpmBytesAttr().getInt() < 0 || getDdrBytesAttr().getInt() < 0)
    return emitOpError("launch resource byte summaries must be non-negative");

  if (getNumResults() != getOutputs().size())
    return emitOpError("launch result count must match output count");

  for (mlir::Value input : getInputs()) {
    if (!mlir::isa<mlir::RankedTensorType>(input.getType()))
      return emitOpError("launch boundary values must be ranked tensors");
  }
  for (mlir::Value output : getOutputs()) {
    if (!mlir::isa<mlir::RankedTensorType>(output.getType()))
      return emitOpError("launch boundary values must be ranked tensors");
  }

  for (auto [index, resultAndOutput] :
       llvm::enumerate(llvm::zip(getResults(), getOutputs()))) {
    mlir::Type resultType = std::get<0>(resultAndOutput).getType();
    mlir::Type outputType = std::get<1>(resultAndOutput).getType();
    if (!mlir::isa<mlir::RankedTensorType>(resultType))
      return emitOpError("launch boundary values must be ranked tensors");
    if (resultType != outputType)
      return emitOpError("launch result type must match output type at index ")
             << index;
  }

  return mlir::success();
}

mlir::LogicalResult PlacementMapOp::verify() {
  int64_t logicalRankCount = getLogicalRankCountAttr().getInt();
  if (logicalRankCount <= 0)
    return emitOpError("logical rank count must be positive");

  int64_t cardYCount = getCardYCountAttr().getInt();
  int64_t cardXCount = getCardXCountAttr().getInt();
  int64_t tileYCount = getTileYCountAttr().getInt();
  int64_t tileXCount = getTileXCountAttr().getInt();
  if (cardYCount <= 0 || cardXCount <= 0 || tileYCount <= 0 || tileXCount <= 0)
    return emitOpError("physical topology dimensions must be positive");

  int64_t expectedCoordEntries = 0;
  if (!checkedMul(logicalRankCount, 4, expectedCoordEntries))
    return emitOpError("logical rank count is too large to verify");

  llvm::ArrayRef<int64_t> coords = getPhysicalTileCoordsAttr().asArrayRef();
  if (static_cast<int64_t>(coords.size()) != expectedCoordEntries)
    return emitOpError("physical tile mapping must contain one 4D coordinate "
                       "per logical rank");

  int64_t cardCount = 0;
  int64_t rowCount = 0;
  int64_t totalTileCount = 0;
  if (!checkedMul(cardYCount, cardXCount, cardCount) ||
      !checkedMul(cardCount, tileYCount, rowCount) ||
      !checkedMul(rowCount, tileXCount, totalTileCount))
    return emitOpError("physical topology tile count is too large to verify");

  llvm::DenseSet<int64_t> badTileIds;
  for (int64_t badTileId : getBadTileIdsAttr().asArrayRef()) {
    if (badTileId < 0 || badTileId >= totalTileCount)
      return emitOpError("bad tile id must be within physical topology");
    badTileIds.insert(badTileId);
  }

  llvm::DenseSet<int64_t> usedTileIds;
  for (int64_t rank = 0; rank < logicalRankCount; ++rank) {
    int64_t base = rank * 4;
    int64_t cardY = coords[base];
    int64_t cardX = coords[base + 1];
    int64_t tileY = coords[base + 2];
    int64_t tileX = coords[base + 3];

    if (cardY < 0 || cardY >= cardYCount || cardX < 0 || cardX >= cardXCount ||
        tileY < 0 || tileY >= tileYCount || tileX < 0 || tileX >= tileXCount)
      return emitOpError("physical coordinate for logical rank ")
             << rank << " is outside target topology";

    std::optional<int64_t> tileId = getPhysicalTileId(
        cardY, cardX, tileY, tileX, cardXCount, tileYCount, tileXCount);
    if (!tileId)
      return emitOpError("physical tile id is too large to verify");

    if (badTileIds.contains(*tileId))
      return emitOpError("maps logical rank ")
             << rank << " to bad tile id " << *tileId;
    if (!usedTileIds.insert(*tileId).second)
      return emitOpError("maps multiple logical ranks to physical tile id ")
             << *tileId;
  }

  return mlir::success();
}

mlir::LogicalResult LoadTileOp::verify() {
  auto sourceType =
      mlir::dyn_cast<mlir::RankedTensorType>(getSource().getType());
  auto resultType = mlir::dyn_cast<TileBufferType>(getResult().getType());
  if (!sourceType || !resultType)
    return emitOpError("expects ranked tensor source and tile_buffer result");

  if (resultType.getTensorType() != sourceType)
    return emitOpError(
        "load_tile result tensor type must match source tensor type");
  if (!hasTileBufferMemorySpace(resultType, MemorySpace::SPM))
    return emitOpError("load_tile result must use SPM memory space");
  if (!hasTileBufferLayout(resultType, MemLayout::Tensor))
    return emitOpError("load_tile result must use tensor mem_layout");

  return mlir::success();
}

mlir::LogicalResult StoreTileOp::verify() {
  auto sourceType = mlir::dyn_cast<TileBufferType>(getSource().getType());
  auto destType = mlir::dyn_cast<mlir::RankedTensorType>(getDest().getType());
  if (!sourceType || !destType)
    return emitOpError("expects tile_buffer source and ranked tensor dest");

  if (sourceType.getTensorType() != destType)
    return emitOpError(
        "store_tile source tensor type must match dest tensor type");
  if (!hasTileBufferMemorySpace(sourceType, MemorySpace::SPM))
    return emitOpError("store_tile source must use SPM memory space");
  if (!hasTileBufferLayout(sourceType, MemLayout::Tensor))
    return emitOpError(
        "store_tile source must use tensor mem_layout for external writeback");

  return mlir::success();
}

mlir::LogicalResult TileRegionOp::verify() {
  if (getBody().empty())
    return emitOpError("expected non-empty body region");

  for (mlir::Type resultType : getResultTypes()) {
    if (isSPMTileBuffer(resultType))
      return emitOpError(
          "SPM tile buffers cannot cross wafer.tile_region boundaries");
  }
  for (auto input : getInputs()) {
    if (isSPMTileBuffer(input.getType()))
      return emitOpError(
          "SPM tile buffers cannot cross wafer.tile_region boundaries");
  }

  mlir::Block &block = getBody().front();
  if (block.getNumArguments() != getInputs().size())
    return emitOpError("expected ")
           << getInputs().size()
           << " body block arguments matching tile_region inputs, got "
           << block.getNumArguments();

  for (auto [index, inputAndArg] :
       llvm::enumerate(llvm::zip(getInputs(), block.getArguments()))) {
    mlir::Type inputType = std::get<0>(inputAndArg).getType();
    mlir::Type blockArgType = std::get<1>(inputAndArg).getType();
    if (blockArgType != inputType)
      return emitOpError("body block argument type ")
             << blockArgType << " does not match input type " << inputType
             << " at index " << index;
    if (isSPMTileBuffer(blockArgType))
      return emitOpError(
          "SPM tile buffers cannot cross wafer.tile_region boundaries");
  }

  auto yield = mlir::dyn_cast<TileYieldOp>(block.getTerminator());
  if (!yield)
    return emitOpError("expected wafer.tile_yield terminator");

  if (yield.getValues().size() != getNumResults())
    return emitOpError(
               "expected tile_yield value count to match result count, got ")
           << yield.getValues().size() << " values and " << getNumResults()
           << " results";

  for (auto [index, yieldedAndResult] :
       llvm::enumerate(llvm::zip(yield.getValues(), getResults()))) {
    mlir::Type yieldedType = std::get<0>(yieldedAndResult).getType();
    mlir::Type resultType = std::get<1>(yieldedAndResult).getType();
    if (isSPMTileBuffer(yieldedType))
      return emitOpError(
          "SPM tile buffers cannot cross wafer.tile_region boundaries");
    if (yieldedType != resultType)
      return emitOpError("tile_yield type ")
             << yieldedType << " does not match tile_region result type "
             << resultType << " at index " << index;
  }

  for (mlir::NamedAttribute attr : getOperation()->getAttrs())
    return emitOpError("does not accept semantic attributes");

  return mlir::success();
}

void WaferDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Wafer/Dialect/Wafer/IR/WaferAttrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Wafer/Dialect/Wafer/IR/WaferTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "Wafer/Dialect/Wafer/IR/WaferOps.cpp.inc"
      >();
}
