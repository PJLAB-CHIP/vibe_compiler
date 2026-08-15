//===- WaferIRVerification.cpp - Shared Wafer IR verification -------------===//

#include "WaferIRVerification.h"

#include "Wafer/IR/Target/PhysicalTopology.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLExtras.h"

#include <limits>

namespace wafer::detail {

bool isExternalDiscardableAttribute(mlir::Operation *op,
                                    mlir::NamedAttribute attribute) {
  if (std::optional<mlir::RegisteredOperationName> registered =
          op->getRegisteredInfo())
    if (llvm::is_contained(registered->getAttributeNames(),
                           attribute.getName()))
      return false;
  llvm::StringRef name = attribute.getName().getValue();
  auto [dialectNamespace, suffix] = name.split('.');
  return !suffix.empty() && !dialectNamespace.empty() &&
         dialectNamespace != WaferDialect::getDialectNamespace();
}

bool isSPMMemRef(mlir::Type type) { return wafer::isWaferSPMMemRefType(type); }

bool isSPMBuffer(mlir::Type type) { return isSPMMemRef(type); }

std::optional<wafer::MemLayout> getWaferLayout(mlir::Type type) {
  if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type)) {
    if (auto memory = wafer::getWaferMemoryAttr(memrefType))
      return memory.getLayout();
    return std::nullopt;
  }
  return std::nullopt;
}

std::optional<wafer::MemorySpace> getWaferMemorySpace(mlir::Type type) {
  if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type)) {
    if (auto memory = wafer::getWaferMemoryAttr(memrefType))
      return memory.getSpace();
    return std::nullopt;
  }
  return std::nullopt;
}

std::optional<mlir::RankedTensorType> getLogicalTensorType(mlir::Type type) {
  if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type)) {
    if (!wafer::getWaferMemoryAttr(memrefType))
      return std::nullopt;
    return mlir::RankedTensorType::get(memrefType.getShape(),
                                       memrefType.getElementType());
  }
  if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type))
    return tensorType;
  return std::nullopt;
}

bool hasWaferLayout(mlir::Type type, wafer::MemLayout layout) {
  std::optional<wafer::MemLayout> actual = getWaferLayout(type);
  return actual && *actual == layout;
}

bool hasWaferMemorySpace(mlir::Type type, wafer::MemorySpace memorySpace) {
  std::optional<wafer::MemorySpace> actual = getWaferMemorySpace(type);
  return actual && *actual == memorySpace;
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

mlir::FailureOr<std::optional<int64_t>>
getOptionalExecutionMeshPartitionCount(mlir::Operation *op) {
  mlir::ModuleOp moduleOp = op->getParentOfType<mlir::ModuleOp>();
  if (!moduleOp)
    return std::optional<int64_t>();

  ExecutionMeshOp meshOp;
  bool multipleMeshes = false;
  for (ExecutionMeshOp candidate : moduleOp.getOps<ExecutionMeshOp>()) {
    if (!meshOp) {
      meshOp = candidate;
      continue;
    }
    multipleMeshes = true;
    break;
  }
  if (multipleMeshes)
    return op->emitOpError(
        "execution mesh partition validation requires a unique direct "
        "module execution mesh");
  if (!meshOp)
    return std::optional<int64_t>();

  int64_t partitionCount = 1;
  for (int64_t dim : meshOp.getShapeAttr().asArrayRef()) {
    if (dim <= 0 || !checkedMul(partitionCount, dim, partitionCount))
      return op->emitOpError("execution mesh partition count is invalid");
  }
  return std::optional<int64_t>(partitionCount);
}

mlir::LogicalResult
verifyPartitionIdWithinExecutionMesh(mlir::Operation *op, int64_t partitionId,
                                     llvm::StringRef subject) {
  mlir::FailureOr<std::optional<int64_t>> partitionCount =
      getOptionalExecutionMeshPartitionCount(op);
  if (mlir::failed(partitionCount))
    return mlir::failure();
  if (!*partitionCount)
    return mlir::success();
  if (partitionId < 0 || partitionId >= **partitionCount)
    return op->emitOpError()
           << subject << " must be within execution mesh partition count";
  return mlir::success();
}

mlir::LogicalResult
verifyPartitionIdsWithinExecutionMesh(mlir::Operation *op,
                                      llvm::ArrayRef<int64_t> partitionIds,
                                      llvm::StringRef subject) {
  mlir::FailureOr<std::optional<int64_t>> partitionCount =
      getOptionalExecutionMeshPartitionCount(op);
  if (mlir::failed(partitionCount))
    return mlir::failure();
  if (!*partitionCount)
    return mlir::success();
  for (int64_t partitionId : partitionIds) {
    if (partitionId < 0 || partitionId >= **partitionCount)
      return op->emitOpError()
             << subject
             << " partition IDs must be within execution mesh partition "
                "count";
  }
  return mlir::success();
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
  if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type)) {
    std::optional<WaferPhysicalTensorInfo> info =
        wafer::computeWaferPhysicalTensorInfo(memrefType);
    if (info && info->physicalBytes >= 0)
      return info->physicalBytes;
    return std::nullopt;
  }
  if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type))
    return getCompactTensorByteSize(tensorType);
  return std::nullopt;
}

int64_t getCompactByteSizeOrUnknown(mlir::Type type) {
  std::optional<int64_t> bytes = getCompactByteSize(type);
  return bytes ? *bytes : -1;
}

mlir::LogicalResult verifyCanonicalConv2DGeometry(
    mlir::Operation *op, mlir::RankedTensorType input,
    mlir::RankedTensorType weight, mlir::RankedTensorType output,
    llvm::ArrayRef<int64_t> pads, llvm::ArrayRef<int64_t> unpads,
    llvm::ArrayRef<int64_t> strides, llvm::ArrayRef<int64_t> dilations,
    llvm::StringRef diagnosticPrefix) {
  auto error = [&](llvm::Twine message) -> mlir::LogicalResult {
    return op->emitOpError() << diagnosticPrefix << message;
  };
  if (input.getRank() != 4 || weight.getRank() != 4 || output.getRank() != 4)
    return error("ordinary convolution requires rank-4 input, weight and "
                 "output");
  if (!input.hasStaticShape() || !weight.hasStaticShape() ||
      !output.hasStaticShape())
    return error("ordinary convolution requires static geometry");
  if (pads.size() != 4 || unpads.size() != 4 || strides.size() != 2 ||
      dilations.size() != 2)
    return error("ordinary convolution geometry attribute lengths are "
                 "invalid");
  if (llvm::any_of(input.getShape(),
                   [](int64_t value) { return value <= 0; }) ||
      llvm::any_of(weight.getShape(),
                   [](int64_t value) { return value <= 0; }) ||
      llvm::any_of(output.getShape(),
                   [](int64_t value) { return value <= 0; }) ||
      llvm::any_of(pads, [](int64_t value) { return value < 0; }) ||
      llvm::any_of(unpads, [](int64_t value) { return value < 0; }) ||
      llvm::any_of(strides, [](int64_t value) { return value <= 0; }) ||
      llvm::any_of(dilations, [](int64_t value) { return value <= 0; }))
    return error("ordinary convolution dimensions and stride/dilation must be "
                 "positive and pad/unpad must be non-negative");

  // Canonical storage is input/output NHWC and weight XYOI.
  if (output.getDimSize(0) != input.getDimSize(0))
    return error("convolution batch dimensions must match");
  if (input.getDimSize(3) != weight.getDimSize(3))
    return error("convolution input channels must match the weight input "
                 "channels");
  if (output.getDimSize(3) != weight.getDimSize(2))
    return error("convolution output channels do not match the weight "
                 "relation");

  auto inferOutput = [&](int64_t inputSize, int64_t kernel, int64_t stride,
                         int64_t dilation, int64_t padBefore, int64_t padAfter,
                         int64_t unpadBefore, int64_t unpadAfter,
                         llvm::StringRef role) -> mlir::FailureOr<int64_t> {
    int64_t padded = 0;
    int64_t dilatedSpan = 0;
    int64_t effectiveKernel = 0;
    if (!checkedAdd(inputSize, padBefore, padded) ||
        !checkedAdd(padded, padAfter, padded) ||
        !checkedMul(kernel - 1, dilation, dilatedSpan) ||
        !checkedAdd(dilatedSpan, 1, effectiveKernel)) {
      (void)error(llvm::Twine("convolution ") + role +
                  " geometry overflows int64");
      return mlir::failure();
    }
    if (padded < effectiveKernel) {
      (void)error(llvm::Twine("convolution ") + role +
                  " kernel exceeds the padded input");
      return mlir::failure();
    }
    int64_t windowed = (padded - effectiveKernel) / stride + 1;
    int64_t totalUnpad = 0;
    if (!checkedAdd(unpadBefore, unpadAfter, totalUnpad) ||
        windowed <= totalUnpad) {
      (void)error(llvm::Twine("convolution ") + role +
                  " unpadding removes the complete output");
      return mlir::failure();
    }
    return windowed - totalUnpad;
  };

  mlir::FailureOr<int64_t> expectedH = inferOutput(
      input.getDimSize(1), weight.getDimSize(1), strides[0], dilations[0],
      pads[0], pads[1], unpads[0], unpads[1], "height");
  mlir::FailureOr<int64_t> expectedW = inferOutput(
      input.getDimSize(2), weight.getDimSize(0), strides[1], dilations[1],
      pads[2], pads[3], unpads[2], unpads[3], "width");
  if (mlir::failed(expectedH) || mlir::failed(expectedW))
    return mlir::failure();
  if (output.getDimSize(1) != *expectedH || output.getDimSize(2) != *expectedW)
    return error("convolution output spatial shape does not match "
                 "input/kernel/stride/dilation/pad/unpad");
  return mlir::success();
}

mlir::LogicalResult verifyDTEP2P(mlir::Operation *op, mlir::Value buffer,
                                 mlir::IntegerAttr peer,
                                 mlir::IntegerAttr bytes,
                                 mlir::Type tokenType) {
  if (!getLogicalTensorType(buffer.getType()))
    return op->emitOpError("DTE p2p buffer must be a Wafer buffer");
  if (!hasWaferMemorySpace(buffer.getType(), MemorySpace::SPM))
    return op->emitOpError("DTE p2p buffer must use SPM memory space");
  if (!mlir::isa<mlir::async::TokenType>(tokenType))
    return op->emitOpError("DTE p2p result must be an async token");
  if (peer.getInt() < 0)
    return op->emitOpError("DTE peer must be non-negative");
  if (static_cast<uint64_t>(peer.getInt()) >
      std::numeric_limits<uint32_t>::max())
    return op->emitOpError("target_abi_narrowing: DTE peer must fit uint32_t");
  if (bytes.getInt() <= 0)
    return op->emitOpError("DTE byte count must be positive");
  if (static_cast<uint64_t>(bytes.getInt()) >
      std::numeric_limits<uint32_t>::max())
    return op->emitOpError(
        "target_abi_narrowing: DTE byte count must fit uint32_t");
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(buffer.getType());
  std::optional<wafer::WaferPhysicalTensorInfo> info =
      memrefType ? wafer::computeWaferPhysicalTensorInfo(memrefType)
                 : std::nullopt;
  if (!info || info->physicalBytes < 0)
    return op->emitOpError(
        "target_geometry_mismatch: DTE buffer physical byte size must be "
        "statically known");
  if (bytes.getInt() > info->physicalBytes)
    return op->emitOpError(
        "target_geometry_mismatch: DTE byte count exceeds buffer physical "
        "byte size");
  return mlir::success();
}

mlir::LogicalResult verifyDTEWaitTokens(mlir::Operation *op,
                                        mlir::OperandRange tokens) {
  if (tokens.empty())
    return op->emitOpError("DTE wait must have at least one token");
  for (mlir::Value token : tokens) {
    if (!mlir::isa<mlir::async::TokenType>(token.getType()))
      return op->emitOpError("DTE wait operands must be async tokens");
  }
  return mlir::success();
}
template <typename GemmOp> static bool hasAnyBatchedGemmAttrsImpl(GemmOp op) {
  return op.getBatchCountAttr() || op.getLhsBatchDimsAttr() ||
         op.getRhsBatchDimsAttr() || op.getResultBatchDimsAttr() ||
         op.getLhsMDimAttr() || op.getLhsContractingDimAttr() ||
         op.getRhsContractingDimAttr() || op.getRhsNDimAttr() ||
         op.getResultMDimAttr() || op.getResultNDimAttr();
}

bool hasAnyBatchedGemmAttrs(mlir::Operation *op) {
  if (auto tile = mlir::dyn_cast<ComputeGemmOp>(op))
    return hasAnyBatchedGemmAttrsImpl(tile);
  if (auto instr = mlir::dyn_cast<InstrGemmOp>(op))
    return hasAnyBatchedGemmAttrsImpl(instr);
  return false;
}

template <typename GemmOp>
static mlir::LogicalResult
getBatchedGemmDimAttrsImpl(GemmOp op, BatchedGemmDimAttrs &attrs) {
  auto requireI64 = [&](mlir::IntegerAttr attr, llvm::StringRef name,
                        int64_t &value) -> mlir::LogicalResult {
    if (!attr)
      return op.emitOpError("GEMM batched form requires ") << name << " attr";
    value = attr.getInt();
    return mlir::success();
  };
  auto requireDense =
      [&](mlir::DenseI64ArrayAttr attr, llvm::StringRef name,
          llvm::SmallVectorImpl<int64_t> &values) -> mlir::LogicalResult {
    if (!attr)
      return op.emitOpError("GEMM batched form requires ") << name << " attr";
    values.assign(attr.asArrayRef().begin(), attr.asArrayRef().end());
    return mlir::success();
  };
  if (mlir::failed(requireI64(op.getBatchCountAttr(), "batch_count",
                              attrs.batchCount)) ||
      mlir::failed(requireDense(op.getLhsBatchDimsAttr(), "lhs_batch_dims",
                                attrs.lhsBatchDims)) ||
      mlir::failed(requireDense(op.getRhsBatchDimsAttr(), "rhs_batch_dims",
                                attrs.rhsBatchDims)) ||
      mlir::failed(requireDense(op.getResultBatchDimsAttr(),
                                "result_batch_dims", attrs.resultBatchDims)) ||
      mlir::failed(
          requireI64(op.getLhsMDimAttr(), "lhs_m_dim", attrs.lhsMDim)) ||
      mlir::failed(requireI64(op.getLhsContractingDimAttr(),
                              "lhs_contracting_dim",
                              attrs.lhsContractingDim)) ||
      mlir::failed(requireI64(op.getRhsContractingDimAttr(),
                              "rhs_contracting_dim",
                              attrs.rhsContractingDim)) ||
      mlir::failed(
          requireI64(op.getRhsNDimAttr(), "rhs_n_dim", attrs.rhsNDim)) ||
      mlir::failed(requireI64(op.getResultMDimAttr(), "result_m_dim",
                              attrs.resultMDim)) ||
      mlir::failed(
          requireI64(op.getResultNDimAttr(), "result_n_dim", attrs.resultNDim)))
    return mlir::failure();
  return mlir::success();
}

static mlir::LogicalResult getBatchedGemmDimAttrs(mlir::Operation *op,
                                                  BatchedGemmDimAttrs &attrs) {
  if (auto tile = mlir::dyn_cast<ComputeGemmOp>(op))
    return getBatchedGemmDimAttrsImpl(tile, attrs);
  if (auto instr = mlir::dyn_cast<InstrGemmOp>(op))
    return getBatchedGemmDimAttrsImpl(instr, attrs);
  return mlir::failure();
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
    GemmOrientation lhsOrientation, GemmOrientation rhsOrientation,
    BatchedGemmDimAttrs &attrs) {
  if (mlir::failed(getBatchedGemmDimAttrs(op, attrs)))
    return mlir::failure();

  int64_t rank = lhsTensor.getRank();
  if (rank != 3 || rhsTensor.getRank() != rank ||
      resultTensor.getRank() != rank)
    return op->emitOpError(
        "GEMM batched form expects operands and result to have rank exactly "
        "3");
  if (!lhsTensor.hasStaticShape() || !rhsTensor.hasStaticShape() ||
      !resultTensor.hasStaticShape())
    return op->emitOpError("GEMM batched form requires static tensor shapes");

  int64_t expectedLhsMDim = lhsOrientation == GemmOrientation::Normal ? 1 : 2;
  int64_t expectedLhsKDim = lhsOrientation == GemmOrientation::Normal ? 2 : 1;
  int64_t expectedRhsKDim = rhsOrientation == GemmOrientation::Normal ? 1 : 2;
  int64_t expectedRhsNDim = rhsOrientation == GemmOrientation::Normal ? 2 : 1;
  if (attrs.lhsBatchDims.size() != 1 || attrs.lhsBatchDims.front() != 0 ||
      attrs.rhsBatchDims.size() != 1 || attrs.rhsBatchDims.front() != 0 ||
      attrs.resultBatchDims.size() != 1 || attrs.resultBatchDims.front() != 0 ||
      attrs.lhsMDim != expectedLhsMDim ||
      attrs.lhsContractingDim != expectedLhsKDim ||
      attrs.rhsContractingDim != expectedRhsKDim ||
      attrs.rhsNDim != expectedRhsNDim || attrs.resultMDim != 1 ||
      attrs.resultNDim != 2)
    return op->emitOpError(
        "GEMM batched form dimension attrs must match the explicit stored "
        "operand orientations and canonical [B,M,N] result");

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
  case ComputeElementwiseKind::Select:
    return 3;
  case ComputeElementwiseKind::Neg:
  case ComputeElementwiseKind::Recip:
  case ComputeElementwiseKind::Sqrt:
  case ComputeElementwiseKind::Rsqrt:
  case ComputeElementwiseKind::Exp:
  case ComputeElementwiseKind::Ln:
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

static bool isSelectKind(ComputeElementwiseKind kind) {
  return kind == ComputeElementwiseKind::Select;
}

mlir::LogicalResult
verifyElementwiseTileContract(mlir::Operation *op, ComputeElementwiseKind kind,
                              mlir::ValueRange inputs, mlir::Type resultType,
                              mlir::ArrayAttr indexingMaps) {
  std::optional<mlir::RankedTensorType> resultTensor =
      getLogicalTensorType(resultType);
  if (!resultTensor)
    return op->emitOpError("expects Wafer buffer result");

  unsigned expectedArity = getElementwiseArity(kind);
  if (inputs.size() != expectedArity)
    return op->emitOpError("elementwise kind expects ")
           << expectedArity << " operand(s), got " << inputs.size();

  if (!hasWaferMemorySpace(resultType, MemorySpace::SPM))
    return op->emitOpError("elementwise result must use SPM memory space");
  std::optional<MemLayout> resultLayout = getWaferLayout(resultType);
  if (!resultLayout)
    return op->emitOpError("elementwise result must carry Wafer layout");

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
    if (resultMap.getNumDims() != resultTensor->getRank() ||
        resultMap.getNumSymbols() != 0 || !resultMap.isIdentity())
      return op->emitOpError(
          "elementwise result indexing map must be identity");
  }

  std::optional<mlir::RankedTensorType> firstInputTensor;
  for (auto [index, input] : llvm::enumerate(inputs)) {
    std::optional<mlir::RankedTensorType> inputTensor =
        getLogicalTensorType(input.getType());
    if (!inputTensor)
      return op->emitOpError("expects Wafer buffer operands");
    if (!hasWaferMemorySpace(input.getType(), MemorySpace::SPM))
      return op->emitOpError("elementwise operands must use SPM memory space");
    std::optional<MemLayout> inputLayout = getWaferLayout(input.getType());
    if (!inputLayout)
      return op->emitOpError("elementwise operands must carry Wafer layout");
    if (!indexingMaps && inputLayout != resultLayout)
      return op->emitOpError(
          "map-free elementwise operands must use the result layout family");
    if (!firstInputTensor)
      firstInputTensor = inputTensor;
    if (isSelectKind(kind)) {
      if (index == 0) {
        if (!inputTensor->getElementType().isInteger(1))
          return op->emitOpError("select predicate element type must be i1");
      } else if (inputTensor->getElementType() !=
                 resultTensor->getElementType()) {
        return op->emitOpError("select value operand element types must match "
                               "result element type");
      }
    } else if (isRelationKind(kind)) {
      if (!resultTensor->getElementType().isInteger(1))
        return op->emitOpError("relation result element type must be i1");
      if (index > 0) {
        if (inputTensor->getElementType() != firstInputTensor->getElementType())
          return op->emitOpError(
              "relation operand element types must match each other");
      }
    } else if (inputTensor->getElementType() !=
               resultTensor->getElementType()) {
      return op->emitOpError(
          "elementwise operand element types must match result element type");
    }
    if (!indexingMaps && isRelationKind(kind)) {
      if (inputTensor->getShape() != resultTensor->getShape())
        return op->emitOpError(
            "relation operand shapes must match result shape");
      continue;
    }
    if (!indexingMaps && isSelectKind(kind) && index == 0) {
      if (inputTensor->getShape() != resultTensor->getShape())
        return op->emitOpError(
            "select predicate shape must match result shape");
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
    if (inputMap.getNumDims() != resultTensor->getRank() ||
        inputMap.getNumSymbols() != 0 ||
        inputMap.getNumResults() != inputTensor->getRank() ||
        !inputMap.isProjectedPermutation())
      return op->emitOpError(
          "elementwise input indexing maps must be permutation-only maps");

    for (auto [dim, expr] : llvm::enumerate(inputMap.getResults())) {
      auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expr);
      if (!dimExpr || dimExpr.getPosition() >= resultTensor->getRank())
        return op->emitOpError(
            "elementwise input indexing maps must use result dimensions");
      if (hasStaticMismatch(inputTensor->getDimSize(dim),
                            resultTensor->getDimSize(dimExpr.getPosition())))
        return op->emitOpError(
            "elementwise indexing map dimension must match tensor shape");
    }
  }

  return mlir::success();
}
mlir::LogicalResult verifyReduceTileContract(mlir::Operation *op,
                                             mlir::Value input,
                                             mlir::Type resultType) {
  std::optional<mlir::RankedTensorType> inputTensor =
      getLogicalTensorType(input.getType());
  if (!inputTensor)
    return op->emitOpError("expects Wafer buffer input");
  std::optional<mlir::RankedTensorType> resultTensor =
      getLogicalTensorType(resultType);
  if (!resultTensor)
    return op->emitOpError("expects Wafer buffer result");

  if (!hasWaferMemorySpace(input.getType(), MemorySpace::SPM) ||
      !hasWaferMemorySpace(resultType, MemorySpace::SPM))
    return op->emitOpError("reduce operands/results must use SPM memory space");

  MemLayout expectedInputLayout =
      inputTensor->getRank() > 2 ? MemLayout::NCx : MemLayout::Cx;
  MemLayout expectedResultLayout =
      resultTensor->getRank() > 2 ? MemLayout::NCx : MemLayout::Cx;
  if (!hasWaferLayout(input.getType(), expectedInputLayout) ||
      !hasWaferLayout(resultType, expectedResultLayout)) {
    if (inputTensor->getRank() > 2 || resultTensor->getRank() > 2)
      return op->emitOpError(
          "reduce rank > 2 operands/results must use ncx layout");
    return op->emitOpError(
        "reduce rank <= 2 operands/results must use cx layout");
  }

  if (inputTensor->getElementType() != resultTensor->getElementType())
    return op->emitOpError(
        "reduce input element type must match result element type");

  for (int64_t dim : inputTensor->getShape())
    if (dim == mlir::ShapedType::kDynamic || dim <= 0)
      return op->emitOpError(
          "reduce input/result shapes must be static and positive");
  for (int64_t dim : resultTensor->getShape())
    if (dim == mlir::ShapedType::kDynamic || dim <= 0)
      return op->emitOpError(
          "reduce input/result shapes must be static and positive");

  auto reduce = mlir::cast<ComputeReduceOp>(op);
  mlir::DenseI64ArrayAttr dimensions = reduce.getDimensionsAttr();
  llvm::ArrayRef<int64_t> dims = dimensions.asArrayRef();
  if (dims.empty())
    return op->emitOpError("reduce dimensions must be non-empty");
  if (static_cast<int64_t>(dims.size()) > inputTensor->getRank())
    return op->emitOpError("reduce dimensions cannot exceed input tensor rank");

  llvm::DenseSet<int64_t> reducedDims;
  for (int64_t dim : dims) {
    if (dim < 0 || dim >= inputTensor->getRank())
      return op->emitOpError(
          "reduce dimensions must be within input tensor rank");
    if (!reducedDims.insert(dim).second)
      return op->emitOpError("reduce dimensions must be unique");
  }

  if (resultTensor->getRank() !=
      inputTensor->getRank() - static_cast<int64_t>(dims.size()))
    return op->emitOpError(
        "reduce result rank must match input rank minus reduce dimensions");

  int64_t resultDim = 0;
  for (int64_t inputDim = 0; inputDim < inputTensor->getRank(); ++inputDim) {
    if (reducedDims.contains(inputDim))
      continue;
    if (hasStaticMismatch(inputTensor->getDimSize(inputDim),
                          resultTensor->getDimSize(resultDim)))
      return op->emitOpError(
          "reduce result shape must match non-reduced input dimensions");
    ++resultDim;
  }

  mlir::TypedAttr initValue = reduce.getInitValueAttr();
  bool hasInitOperand = static_cast<bool>(reduce.getInit());
  if (hasInitOperand == static_cast<bool>(initValue))
    return op->emitOpError(
        "requires exactly one of scalar init operand or init_value attr");

  if (initValue) {
    if (initValue.getType() != inputTensor->getElementType())
      return op->emitOpError(
          "reduce init_value type must match input element type");
    return mlir::success();
  }

  mlir::Value init = reduce.getInit();
  if (init.getType() != inputTensor->getElementType())
    return op->emitOpError(
        "reduce init operand type must match input element type");
  auto constant = init.getDefiningOp<mlir::arith::ConstantOp>();
  if (!constant)
    return op->emitOpError(
        "reduce init operand must be defined by arith.constant");
  auto typedInit = mlir::dyn_cast<mlir::TypedAttr>(constant.getValue());
  if (!typedInit || typedInit.getType() != inputTensor->getElementType())
    return op->emitOpError(
        "reduce init constant type must match input element type");

  return mlir::success();
}

} // namespace wafer::detail
