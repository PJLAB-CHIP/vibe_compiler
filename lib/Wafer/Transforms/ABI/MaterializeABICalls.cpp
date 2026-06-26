//===- MaterializeABICalls.cpp - Materialize Wafer C ABI calls ------------===//

#include "Wafer/Transforms/Passes.h"

#include "Wafer/ABI/TileAbi.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>

namespace wafer {
#define GEN_PASS_DEF_MATERIALIZEABICALLSPASS
#include "Wafer/Transforms/WaferPasses.h.inc"

namespace {

constexpr llvm::StringLiteral kFailurePrefix = "abi_materialization_failure: ";

struct DdrAddress {
  mlir::Value base;
  int64_t byteOffset = 0;
};

struct ReturnOutputBinding {
  llvm::SmallVector<mlir::Value> aliases;
  unsigned abiIndex = 0;
};

struct WorkspaceRootBinding {
  mlir::Value root;
  int64_t offset = 0;
  int64_t bytes = 0;
};

struct GemmBatchAttrs {
  bool enabled = false;
  int64_t batchCount = 1;
  llvm::SmallVector<int64_t, 4> lhsBatchDims;
  llvm::SmallVector<int64_t, 4> rhsBatchDims;
  llvm::SmallVector<int64_t, 4> resultBatchDims;
};

static bool hasAnyGemmBatchAttr(mlir::Operation *op) {
  return op->hasAttr("batch_count") || op->hasAttr("lhs_batch_dims") ||
         op->hasAttr("rhs_batch_dims") || op->hasAttr("result_batch_dims") ||
         op->hasAttr("lhs_m_dim") || op->hasAttr("lhs_contracting_dim") ||
         op->hasAttr("rhs_contracting_dim") || op->hasAttr("rhs_n_dim") ||
         op->hasAttr("result_m_dim") || op->hasAttr("result_n_dim");
}

static mlir::FailureOr<int64_t> readRequiredI64Attr(mlir::Operation *op,
                                                    llvm::StringRef name,
                                                    llvm::StringRef role) {
  auto attr = op->getAttrOfType<mlir::IntegerAttr>(name);
  if (!attr)
    return op->emitError() << kFailurePrefix << role << " requires " << name
                           << " attr";
  return attr.getInt();
}

static mlir::FailureOr<llvm::SmallVector<int64_t, 4>>
readRequiredDenseI64Attr(mlir::Operation *op, llvm::StringRef name,
                         llvm::StringRef role) {
  auto attr = op->getAttrOfType<mlir::DenseI64ArrayAttr>(name);
  if (!attr)
    return op->emitError() << kFailurePrefix << role << " requires " << name
                           << " attr";
  return llvm::SmallVector<int64_t, 4>(attr.asArrayRef().begin(),
                                       attr.asArrayRef().end());
}

static void appendUnique(llvm::SmallVectorImpl<mlir::Value> &values,
                         mlir::Value value) {
  if (!llvm::is_contained(values, value))
    values.push_back(value);
}

static bool isOutputAlias(mlir::Value value,
                          llvm::ArrayRef<ReturnOutputBinding> outputBindings) {
  return llvm::any_of(outputBindings, [&](const ReturnOutputBinding &binding) {
    return llvm::is_contained(binding.aliases, value);
  });
}

static mlir::Value resolveTileRegionBoundaryValue(mlir::Value value) {
  while (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    mlir::Block *owner = blockArg.getOwner();
    if (!owner)
      return value;
    auto tileRegion =
        mlir::dyn_cast_or_null<TileRegionOp>(owner->getParentOp());
    if (!tileRegion || tileRegion.getBody().empty() ||
        owner != &tileRegion.getBody().front())
      return value;
    if (blockArg.getArgNumber() >= tileRegion.getInputs().size())
      return value;
    value = tileRegion.getInputs()[blockArg.getArgNumber()];
  }
  return value;
}

static mlir::Value getRootViewSource(mlir::Value value) {
  value = resolveTileRegionBoundaryValue(value);
  while (mlir::Operation *def = value.getDefiningOp()) {
    if (auto result = mlir::dyn_cast<mlir::OpResult>(value)) {
      if (auto tileRegion = mlir::dyn_cast<TileRegionOp>(result.getOwner())) {
        if (!tileRegion.getBody().empty()) {
          auto yield = mlir::dyn_cast_or_null<TileYieldOp>(
              tileRegion.getBody().front().getTerminator());
          if (yield && result.getResultNumber() < yield.getValues().size()) {
            mlir::Value yielded = resolveTileRegionBoundaryValue(
                yield.getValues()[result.getResultNumber()]);
            if (yielded != value) {
              value = yielded;
              continue;
            }
          }
        }
      }
    }

    auto viewLike = mlir::dyn_cast<mlir::ViewLikeOpInterface>(def);
    if (!viewLike)
      return value;
    mlir::Value source =
        resolveTileRegionBoundaryValue(viewLike.getViewSource());
    if (source == value)
      return value;
    value = source;
  }
  return value;
}

static void
collectReturnedDdrAliases(mlir::Value value,
                          llvm::SmallVectorImpl<mlir::Value> &aliases) {
  appendUnique(aliases, getRootViewSource(value));

  auto result = mlir::dyn_cast<mlir::OpResult>(value);
  if (!result)
    return;
  auto tileRegion = mlir::dyn_cast<TileRegionOp>(result.getOwner());
  if (!tileRegion || tileRegion.getBody().empty())
    return;

  auto yield = mlir::dyn_cast_or_null<TileYieldOp>(
      tileRegion.getBody().front().getTerminator());
  if (!yield || result.getResultNumber() >= yield.getValues().size())
    return;
  appendUnique(aliases,
               getRootViewSource(yield.getValues()[result.getResultNumber()]));
}

static mlir::FailureOr<int64_t> getElementByteWidth(mlir::Operation *op,
                                                    mlir::Type type) {
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
    unsigned width = intType.getWidth();
    if (width == 0 || width % 8 != 0)
      return op->emitError() << kFailurePrefix
                             << "memref element type must be byte-addressable";
    return static_cast<int64_t>(width / 8);
  }
  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(type)) {
    unsigned width = floatType.getWidth();
    if (width == 0 || width % 8 != 0)
      return op->emitError() << kFailurePrefix
                             << "memref element type must be byte-addressable";
    return static_cast<int64_t>(width / 8);
  }
  if (mlir::isa<mlir::IndexType>(type))
    return int64_t{8};
  return op->emitError() << kFailurePrefix
                         << "memref element type must be scalar int, float, "
                         << "or index";
}

static mlir::FailureOr<int64_t> getStaticByteOffset(mlir::Operation *op,
                                                    mlir::MemRefType type,
                                                    llvm::StringRef role) {
  llvm::SmallVector<int64_t> strides;
  int64_t offset = 0;
  if (mlir::failed(mlir::getStridesAndOffset(type, strides, offset)) ||
      strides.size() != static_cast<size_t>(type.getRank()))
    return op->emitError() << kFailurePrefix << role
                           << " view must have static strided layout";
  if (offset == mlir::ShapedType::kDynamic || offset < 0)
    return op->emitError() << kFailurePrefix << role
                           << " view must have static non-negative byte offset";

  mlir::FailureOr<int64_t> elementBytes =
      getElementByteWidth(op, type.getElementType());
  if (mlir::failed(elementBytes))
    return mlir::failure();
  if (offset > std::numeric_limits<int64_t>::max() / *elementBytes)
    return op->emitError() << kFailurePrefix << role
                           << " view byte offset overflows int64";
  return offset * *elementBytes;
}

static mlir::FailureOr<int64_t> getStaticElementCount(mlir::Operation *op,
                                                      mlir::Value value,
                                                      llvm::StringRef role) {
  auto type = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  if (!type)
    return op->emitError() << kFailurePrefix << role
                           << " operand must have memref type";
  if (!type.hasStaticShape())
    return op->emitError() << kFailurePrefix << role
                           << " operand must have static shape";
  int64_t count = 1;
  for (int64_t dim : type.getShape()) {
    if (dim < 0 || count > std::numeric_limits<int64_t>::max() / dim)
      return op->emitError()
             << kFailurePrefix << role << " element count overflows int64";
    count *= dim;
  }
  return count;
}

static mlir::FailureOr<std::array<int64_t, 4>>
getStaticShape4D(mlir::Operation *op, mlir::Value value, llvm::StringRef role) {
  auto type = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  if (!type)
    return op->emitError() << kFailurePrefix << role
                           << " operand must have memref type";
  if (!type.hasStaticShape())
    return op->emitError() << kFailurePrefix << role
                           << " operand must have static shape";
  if (type.getRank() > 4)
    return op->emitError() << kFailurePrefix << role
                           << " operand rank must be <= 4 for native CT ABI";

  std::array<int64_t, 4> shape = {1, 1, 1, 1};
  int64_t offset = 4 - type.getRank();
  for (auto [index, dim] : llvm::enumerate(type.getShape()))
    shape[offset + index] = dim;
  return shape;
}

static mlir::FailureOr<int64_t> getSpmOffset(mlir::Operation *op,
                                             mlir::Value value) {
  auto valueType = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  if (!valueType || !isWaferSPMMemRefType(valueType))
    return op->emitError() << kFailurePrefix
                           << "SPM operand must have Wafer SPM memref type";

  mlir::Value root = getRootViewSource(value);
  mlir::Operation *rootDef = root.getDefiningOp();
  if (!rootDef)
    return op->emitError() << kFailurePrefix
                           << "SPM memref has no accepted wafer.spm.offset";
  auto offsetAttr =
      rootDef->getAttrOfType<SPMOffsetAttr>(kWaferSPMOffsetAttrName);
  if (!offsetAttr)
    return op->emitError() << kFailurePrefix
                           << "SPM memref has no accepted wafer.spm.offset";

  mlir::FailureOr<int64_t> viewOffset =
      getStaticByteOffset(op, valueType, "SPM");
  if (mlir::failed(viewOffset))
    return mlir::failure();
  int64_t base = offsetAttr.getOffset();
  if (base < 0 || *viewOffset > std::numeric_limits<int64_t>::max() - base)
    return op->emitError() << kFailurePrefix
                           << "SPM byte offset overflows int64";
  int64_t offset = base + *viewOffset;
  if (offset > std::numeric_limits<uint32_t>::max())
    return op->emitError() << kFailurePrefix
                           << "SPM byte offset is not representable";
  return offset;
}

static mlir::FailureOr<GemmBatchAttrs> getGemmBatchAttrs(InstrGemmOp gemm) {
  GemmBatchAttrs attrs;
  if (!hasAnyGemmBatchAttr(gemm.getOperation()))
    return attrs;

  attrs.enabled = true;
  mlir::FailureOr<int64_t> batchCount =
      readRequiredI64Attr(gemm.getOperation(), "batch_count", "batched GEMM");
  mlir::FailureOr<llvm::SmallVector<int64_t, 4>> lhsBatchDims =
      readRequiredDenseI64Attr(gemm.getOperation(), "lhs_batch_dims",
                               "batched GEMM");
  mlir::FailureOr<llvm::SmallVector<int64_t, 4>> rhsBatchDims =
      readRequiredDenseI64Attr(gemm.getOperation(), "rhs_batch_dims",
                               "batched GEMM");
  mlir::FailureOr<llvm::SmallVector<int64_t, 4>> resultBatchDims =
      readRequiredDenseI64Attr(gemm.getOperation(), "result_batch_dims",
                               "batched GEMM");
  if (mlir::failed(batchCount) || mlir::failed(lhsBatchDims) ||
      mlir::failed(rhsBatchDims) || mlir::failed(resultBatchDims))
    return mlir::failure();
  if (*batchCount <= 0)
    return gemm.emitError()
           << kFailurePrefix << "batched GEMM batch_count must be positive";
  if (lhsBatchDims->size() != rhsBatchDims->size() ||
      lhsBatchDims->size() != resultBatchDims->size())
    return gemm.emitError()
           << kFailurePrefix
           << "batched GEMM batch dimension attrs must have matching lengths";

  attrs.batchCount = *batchCount;
  attrs.lhsBatchDims = std::move(*lhsBatchDims);
  attrs.rhsBatchDims = std::move(*rhsBatchDims);
  attrs.resultBatchDims = std::move(*resultBatchDims);
  return attrs;
}

static mlir::FailureOr<llvm::SmallVector<int64_t, 4>>
getBatchShape(mlir::Operation *op, mlir::MemRefType type,
              llvm::ArrayRef<int64_t> dims, llvm::StringRef role) {
  llvm::SmallVector<int64_t, 4> shape;
  shape.reserve(dims.size());
  for (int64_t dim : dims) {
    if (dim < 0 || dim >= type.getRank())
      return op->emitError()
             << kFailurePrefix << role << " batch dim is out of range";
    int64_t size = type.getDimSize(dim);
    if (mlir::ShapedType::isDynamic(size) || size <= 0)
      return op->emitError() << kFailurePrefix << role
                             << " batch dim must have static positive size";
    shape.push_back(size);
  }
  return shape;
}

static mlir::FailureOr<llvm::SmallVector<int64_t, 4>>
delinearizeBatchIndex(mlir::Operation *op, int64_t linearIndex,
                      llvm::ArrayRef<int64_t> batchShape) {
  llvm::SmallVector<int64_t, 4> indices(batchShape.size(), 0);
  int64_t remaining = linearIndex;
  for (int64_t dim = static_cast<int64_t>(batchShape.size()) - 1; dim >= 0;
       --dim) {
    int64_t size = batchShape[dim];
    if (size <= 0)
      return op->emitError()
             << kFailurePrefix << "batched GEMM batch shape is invalid";
    indices[dim] = remaining % size;
    remaining /= size;
  }
  if (remaining != 0)
    return op->emitError() << kFailurePrefix
                           << "batched GEMM batch index is out of range";
  return indices;
}

static mlir::FailureOr<int64_t>
getBatchPhysicalByteOffset(mlir::Operation *op, mlir::MemRefType type,
                           llvm::ArrayRef<int64_t> batchDims,
                           llvm::ArrayRef<int64_t> batchIndices,
                           llvm::StringRef role) {
  if (batchDims.size() != batchIndices.size())
    return op->emitError()
           << kFailurePrefix << role
           << " batch dims and indices must have matching lengths";

  llvm::SmallVector<int64_t, 4> logicalIndices(type.getRank(), 0);
  for (auto [dim, index] : llvm::zip(batchDims, batchIndices)) {
    if (dim < 0 || dim >= type.getRank())
      return op->emitError()
             << kFailurePrefix << role << " batch dim is out of range";
    logicalIndices[dim] = index;
  }

  std::optional<int64_t> offset =
      wafer::computeWaferPhysicalElementByteOffset(type, logicalIndices);
  if (!offset)
    return op->emitError()
           << kFailurePrefix << role
           << " batch physical byte offset is not representable";
  return *offset;
}

static mlir::FailureOr<int64_t> addSpmByteOffsets(mlir::Operation *op,
                                                  int64_t base, int64_t delta,
                                                  llvm::StringRef role) {
  if (delta < 0 || base < 0 ||
      delta > std::numeric_limits<int64_t>::max() - base)
    return op->emitError() << kFailurePrefix << role
                           << " SPM byte offset overflows int64";
  int64_t offset = base + delta;
  if (offset > std::numeric_limits<uint32_t>::max())
    return op->emitError() << kFailurePrefix << role
                           << " SPM byte offset is not representable";
  return offset;
}

static mlir::FailureOr<DdrAddress>
getDdrAddress(mlir::Operation *op, mlir::Value value,
              const llvm::DenseMap<mlir::Value, mlir::Value> &externalBases) {
  auto valueType = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  if (!valueType || !isWaferDDRMemRefType(valueType))
    return op->emitError() << kFailurePrefix
                           << "DDR operand must have Wafer DDR memref type";

  mlir::Value root = getRootViewSource(value);
  auto it = externalBases.find(root);
  if (it == externalBases.end())
    return op->emitError()
           << kFailurePrefix
           << "DDR operand must resolve to an external function argument or "
           << "returned output binding";

  mlir::FailureOr<int64_t> viewOffset =
      getStaticByteOffset(op, valueType, "DDR");
  if (mlir::failed(viewOffset))
    return mlir::failure();
  return DdrAddress{it->second, *viewOffset};
}

static mlir::LogicalResult collectWorkspaceRootBindings(
    mlir::func::FuncOp func, llvm::ArrayRef<ReturnOutputBinding> outputBindings,
    llvm::SmallVectorImpl<WorkspaceRootBinding> &workspaceRoots,
    int64_t &workspaceBytes) {
  workspaceRoots.clear();
  workspaceBytes = 0;

  mlir::WalkResult result = func.walk([&](mlir::memref::AllocOp alloc) {
    mlir::Value root = alloc.getResult();
    auto type = mlir::dyn_cast<mlir::MemRefType>(root.getType());
    if (!type || !isWaferDDRMemRefType(type) ||
        isOutputAlias(root, outputBindings))
      return mlir::WalkResult::advance();

    auto offsetAttr =
        alloc->getAttrOfType<DDROffsetAttr>(kWaferDDROffsetAttrName);
    if (!offsetAttr)
      return mlir::WalkResult::advance();

    std::optional<WaferPhysicalTensorInfo> info =
        wafer::computeWaferPhysicalTensorInfo(type);
    if (!info || info->physicalBytes <= 0) {
      alloc.emitError()
          << kFailurePrefix
          << "workspace DDR allocation requires static positive physical size";
      return mlir::WalkResult::interrupt();
    }
    int64_t offset = offsetAttr.getOffset();
    if (offset < 0 ||
        info->physicalBytes > std::numeric_limits<int64_t>::max() - offset) {
      alloc.emitError() << kFailurePrefix
                        << "workspace DDR allocation range overflows";
      return mlir::WalkResult::interrupt();
    }
    int64_t end = offset + info->physicalBytes;
    workspaceBytes = std::max(workspaceBytes, end);
    workspaceRoots.push_back(
        WorkspaceRootBinding{root, offset, info->physicalBytes});
    return mlir::WalkResult::advance();
  });

  return result.wasInterrupted() ? mlir::failure() : mlir::success();
}

static mlir::Value createI32Constant(mlir::OpBuilder &builder,
                                     mlir::Location loc, int64_t value) {
  return builder.create<mlir::arith::ConstantIntOp>(loc, value, 32);
}

static mlir::Value createI64Constant(mlir::OpBuilder &builder,
                                     mlir::Location loc, int64_t value) {
  return builder.create<mlir::arith::ConstantIntOp>(loc, value, 64);
}

static mlir::Value createDdrAddressValue(mlir::OpBuilder &builder,
                                         mlir::Location loc,
                                         DdrAddress address) {
  if (address.byteOffset == 0)
    return address.base;
  mlir::Value offset = createI64Constant(builder, loc, address.byteOffset);
  return builder.create<mlir::arith::AddIOp>(loc, address.base, offset);
}

static mlir::Value createSpmOffsetValue(mlir::OpBuilder &builder,
                                        mlir::Location loc, int64_t offset) {
  return createI32Constant(builder, loc, offset);
}

static mlir::FailureOr<abi::DataFormat> getDataFormat(mlir::Operation *op,
                                                      mlir::Type type) {
  if (mlir::isa<mlir::Float16Type>(type))
    return abi::DataFormat::FP16;
  if (mlir::isa<mlir::BFloat16Type>(type))
    return abi::DataFormat::BF16;
  if (mlir::isa<mlir::Float32Type>(type))
    return abi::DataFormat::FP32;
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
    switch (intType.getWidth()) {
    case 1:
      return abi::DataFormat::BOOL;
    case 8:
      return abi::DataFormat::INT8;
    case 16:
      return abi::DataFormat::INT16;
    case 32:
      return abi::DataFormat::INT32;
    case 64:
      return abi::DataFormat::INT64;
    default:
      break;
    }
  }
  return op->emitError() << kFailurePrefix
                         << "memref element type has no TX8 Data_Format";
}

static mlir::FailureOr<abi::DataFormat> getDataFormat(mlir::Operation *op,
                                                      mlir::Value value) {
  auto type = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  if (!type)
    return op->emitError() << kFailurePrefix
                           << "ABI operand must have memref type";
  return getDataFormat(op, type.getElementType());
}

static bool isUnaryElementwiseKind(ComputeElementwiseKind kind) {
  switch (kind) {
  case ComputeElementwiseKind::Neg:
  case ComputeElementwiseKind::Recip:
  case ComputeElementwiseKind::Sqrt:
  case ComputeElementwiseKind::Rsqrt:
  case ComputeElementwiseKind::Exp:
  case ComputeElementwiseKind::Tanh:
    return true;
  default:
    return false;
  }
}

static bool isBinaryElementwiseKind(ComputeElementwiseKind kind) {
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
    return true;
  default:
    return false;
  }
}

static mlir::FailureOr<uint64_t> getScalarConstantBits(mlir::Operation *op,
                                                       mlir::Value value,
                                                       llvm::StringRef role) {
  mlir::Attribute attr;
  if (!mlir::matchPattern(value, mlir::m_Constant(&attr)))
    return op->emitError() << kFailurePrefix << role
                           << " must be an arith.constant scalar";

  if (auto integerAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
    llvm::APInt intValue = integerAttr.getValue();
    if (intValue.getBitWidth() > 64)
      return op->emitError()
             << kFailurePrefix << role << " constant does not fit in 64 bits";
    return intValue.getZExtValue();
  }

  if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(attr)) {
    llvm::APInt bits = floatAttr.getValue().bitcastToAPInt();
    if (bits.getBitWidth() > 64)
      return op->emitError()
             << kFailurePrefix << role << " constant does not fit in 64 bits";
    return bits.getZExtValue();
  }

  return op->emitError() << kFailurePrefix << role
                         << " must be an integer or float scalar constant";
}

static mlir::LogicalResult verifyZeroInit(mlir::Operation *op, mlir::Value init,
                                          llvm::StringRef role) {
  if (!init)
    return mlir::success();

  mlir::Attribute attr;
  if (!mlir::matchPattern(init, mlir::m_Constant(&attr)))
    return op->emitError() << kFailurePrefix << role
                           << " init must be an arith.constant scalar";
  if (auto integerAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
    if (!integerAttr.getValue().isZero())
      return op->emitError() << kFailurePrefix << role
                             << " init must be zero for native CT reduce ABI";
    return mlir::success();
  }
  if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(attr)) {
    if (!floatAttr.getValue().isZero())
      return op->emitError() << kFailurePrefix << role
                             << " init must be zero for native CT reduce ABI";
    return mlir::success();
  }
  return op->emitError() << kFailurePrefix << role
                         << " init must be an integer or float scalar constant";
}

static mlir::FailureOr<int64_t>
getNativeReduceDim(mlir::Operation *op, int64_t inputRank,
                   llvm::ArrayRef<int64_t> dimensions) {
  if (dimensions.empty())
    return op->emitError() << kFailurePrefix
                           << "reduce dimensions must be non-empty";
  for (int64_t dim : dimensions) {
    if (dim < 0 || dim >= inputRank)
      return op->emitError()
             << kFailurePrefix << "reduce dimension is out of range";
  }

  auto isDims = [&](std::initializer_list<int64_t> expected) {
    return dimensions.size() == expected.size() &&
           llvm::equal(dimensions, expected);
  };

  int64_t c = inputRank - 1;
  int64_t w = inputRank - 2;
  int64_t h = inputRank - 3;
  int64_t n = inputRank - 4;
  if (isDims({c}))
    return int64_t{0};
  if (w >= 0 && isDims({w}))
    return int64_t{1};
  if (h >= 0 && isDims({h}))
    return int64_t{2};
  if (n >= 0 && isDims({n}))
    return int64_t{3};
  if (w >= 0 && isDims({w, c}))
    return int64_t{4};
  if (h >= 0 && isDims({h, w, c}))
    return int64_t{5};

  return op->emitError()
         << kFailurePrefix
         << "reduce dimensions must map to native C/W/H/N/HW/HWC dim";
}

static mlir::FailureOr<std::array<int64_t, 3>>
getDescriptorTriple(mlir::Operation *op, llvm::ArrayRef<int64_t> values,
                    llvm::StringRef role) {
  if (values.size() != 3)
    return op->emitError() << kFailurePrefix << role
                           << " descriptor must have three values";
  return std::array<int64_t, 3>{values[0], values[1], values[2]};
}

static mlir::LogicalResult validateDmaDescriptor(
    mlir::Operation *op, DdrAddress ddrAddress, int64_t spmOffset,
    uint64_t byteCount, uint64_t innerBytes, llvm::ArrayRef<int64_t> strides,
    llvm::ArrayRef<int64_t> iterations, abi::DataFormat format, bool isRdma) {
  if (ddrAddress.byteOffset < 0)
    return op->emitError() << kFailurePrefix
                           << "DDR byte offset must be non-negative";
  if (static_cast<uint64_t>(ddrAddress.byteOffset) >
      std::numeric_limits<uint64_t>::max() - abi::kDdrLowerBound)
    return op->emitError() << kFailurePrefix << "DDR address overflows";
  if (spmOffset < 0 ||
      spmOffset > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
    return op->emitError() << kFailurePrefix
                           << "SPM byte offset is not representable";

  mlir::FailureOr<std::array<int64_t, 3>> strideArray =
      getDescriptorTriple(op, strides, "DMA stride");
  if (mlir::failed(strideArray))
    return mlir::failure();
  mlir::FailureOr<std::array<int64_t, 3>> iterationArray =
      getDescriptorTriple(op, iterations, "DMA iteration");
  if (mlir::failed(iterationArray))
    return mlir::failure();

  abi::DmaDescriptor descriptor;
  std::string error;
  uint64_t staticDdrAddress =
      abi::kDdrLowerBound + static_cast<uint64_t>(ddrAddress.byteOffset);
  bool ok =
      isRdma
          ? abi::buildRdma(staticDdrAddress, static_cast<uint32_t>(spmOffset),
                           byteCount, innerBytes, *strideArray, *iterationArray,
                           descriptor, &error)
          : abi::buildWdma(static_cast<uint32_t>(spmOffset), staticDdrAddress,
                           byteCount, innerBytes, *strideArray, *iterationArray,
                           descriptor, &error);
  if (!ok)
    return op->emitError() << kFailurePrefix << error;

  abi::DmaRegisterPacket packet;
  ok = isRdma
           ? abi::buildRdmaRegisterPacket(descriptor, format, packet, &error)
           : abi::buildWdmaRegisterPacket(descriptor, format, packet, &error);
  if (!ok)
    return op->emitError() << kFailurePrefix << error;
  return mlir::success();
}

static mlir::LogicalResult validateGatherScatterDescriptor(
    mlir::Operation *op, int64_t sourceOffset, int64_t destOffset,
    uint64_t byteCount, uint64_t innerBytes, llvm::ArrayRef<int64_t> srcStrides,
    llvm::ArrayRef<int64_t> srcIterations, llvm::ArrayRef<int64_t> dstStrides,
    llvm::ArrayRef<int64_t> dstIterations) {
  if (sourceOffset < 0 ||
      sourceOffset > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
    return op->emitError() << kFailurePrefix
                           << "gather/scatter source offset is not "
                           << "representable";
  if (destOffset < 0 ||
      destOffset > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
    return op->emitError() << kFailurePrefix
                           << "gather/scatter dest offset is not "
                           << "representable";

  mlir::FailureOr<std::array<int64_t, 3>> srcStrideArray =
      getDescriptorTriple(op, srcStrides, "gather/scatter source stride");
  if (mlir::failed(srcStrideArray))
    return mlir::failure();
  mlir::FailureOr<std::array<int64_t, 3>> srcIterationArray =
      getDescriptorTriple(op, srcIterations, "gather/scatter source iteration");
  if (mlir::failed(srcIterationArray))
    return mlir::failure();
  mlir::FailureOr<std::array<int64_t, 3>> dstStrideArray =
      getDescriptorTriple(op, dstStrides, "gather/scatter dest stride");
  if (mlir::failed(dstStrideArray))
    return mlir::failure();
  mlir::FailureOr<std::array<int64_t, 3>> dstIterationArray =
      getDescriptorTriple(op, dstIterations, "gather/scatter dest iteration");
  if (mlir::failed(dstIterationArray))
    return mlir::failure();

  abi::GatherScatterDescriptor descriptor;
  std::string error;
  bool ok = abi::buildGatherScatter(
      static_cast<uint32_t>(sourceOffset), static_cast<uint32_t>(destOffset),
      byteCount, innerBytes, *srcStrideArray, *srcIterationArray,
      *dstStrideArray, *dstIterationArray, descriptor, &error);
  if (!ok)
    return op->emitError() << kFailurePrefix << error;

  abi::GatherScatterRegisterPacket packet;
  if (!abi::buildGatherScatterRegisterPacket(descriptor, packet, &error))
    return op->emitError() << kFailurePrefix << error;
  return mlir::success();
}

static mlir::LogicalResult
ensureAbiDeclaration(mlir::ModuleOp module, llvm::StringRef name,
                     llvm::ArrayRef<mlir::Type> inputs,
                     llvm::ArrayRef<mlir::Type> results) {
  mlir::FunctionType type =
      mlir::FunctionType::get(module.getContext(), inputs, results);
  if (auto existing = module.lookupSymbol<mlir::func::FuncOp>(name)) {
    if (existing.getFunctionType() != type)
      return existing.emitError() << kFailurePrefix << "ABI declaration @"
                                  << name << " has incompatible function type";
    return mlir::success();
  }

  mlir::OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPointToEnd(module.getBody());
  auto declaration =
      builder.create<mlir::func::FuncOp>(module.getLoc(), name, type);
  declaration.setPrivate();
  return mlir::success();
}

static mlir::LogicalResult ensureAbiDeclarations(mlir::ModuleOp module) {
  mlir::MLIRContext *ctx = module.getContext();
  mlir::Type i32 = mlir::IntegerType::get(ctx, 32);
  mlir::Type i64 = mlir::IntegerType::get(ctx, 64);

  llvm::SmallVector<mlir::Type> rdmaInputs = {i64, i32, i64, i64};
  rdmaInputs.append(6, i64);
  rdmaInputs.push_back(i32);
  if (mlir::failed(
          ensureAbiDeclaration(module, "wafer_rdma", rdmaInputs, {i32})))
    return mlir::failure();
  llvm::SmallVector<mlir::Type> wdmaInputs = {i32, i64, i64, i64};
  wdmaInputs.append(6, i64);
  wdmaInputs.push_back(i32);
  if (mlir::failed(
          ensureAbiDeclaration(module, "wafer_wdma", wdmaInputs, {i32})))
    return mlir::failure();
  if (mlir::failed(ensureAbiDeclaration(
          module, "wafer_gemm", {i32, i32, i32, i64, i64, i64, i32}, {i32})))
    return mlir::failure();
  llvm::SmallVector<mlir::Type> gatherScatterInputs = {i32, i32, i64, i64};
  gatherScatterInputs.append(12, i64);
  if (mlir::failed(ensureAbiDeclaration(module, "wafer_gather_scatter",
                                        gatherScatterInputs, {i32})))
    return mlir::failure();
  if (mlir::failed(ensureAbiDeclaration(module, "wafer_fill",
                                        {i32, i64, i64, i32}, {i32})))
    return mlir::failure();
  if (mlir::failed(ensureAbiDeclaration(module, "wafer_elementwise",
                                        {i32, i32, i32, i32, i64, i32, i32},
                                        {i32})))
    return mlir::failure();
  if (mlir::failed(ensureAbiDeclaration(
          module, "wafer_reduce", {i32, i32, i32, i32, i64, i64, i64, i64, i32},
          {i32})))
    return mlir::failure();
  if (mlir::failed(ensureAbiDeclaration(module, "wafer_convert",
                                        {i32, i32, i32, i32, i64}, {i32})))
    return mlir::failure();
  if (mlir::failed(ensureAbiDeclaration(module, "wafer_dte_send",
                                        {i32, i32, i64}, {i32})))
    return mlir::failure();
  if (mlir::failed(ensureAbiDeclaration(module, "wafer_dte_recv",
                                        {i32, i32, i64}, {i32})))
    return mlir::failure();
  if (mlir::failed(
          ensureAbiDeclaration(module, "wafer_dte_wait", {i32}, {i32})))
    return mlir::failure();
  if (mlir::failed(
          ensureAbiDeclaration(module, "wafer_local_fence", {}, {i32})))
    return mlir::failure();
  return mlir::success();
}

static mlir::Value emitAbiCall(mlir::OpBuilder &builder, mlir::Location loc,
                               llvm::StringRef callee,
                               mlir::ValueRange arguments, mlir::Value status) {
  auto call = builder.create<mlir::func::CallOp>(
      loc, callee, mlir::TypeRange{status.getType()}, arguments);
  return builder.create<mlir::arith::OrIOp>(loc, status, call.getResult(0));
}

static bool isSupportedAbiInstruction(mlir::Operation *op) {
  return mlir::isa<InstrRDMAOp, InstrWDMAOp, InstrGatherScatterOp, InstrFillOp,
                   InstrElementwiseOp, InstrReduceOp, InstrConvertOp,
                   InstrGemmOp, InstrDTESendOp, InstrDTERecvOp, InstrDTEWaitOp,
                   SyncLocalFenceOp>(op);
}

static bool isWaferInstructionOp(mlir::Operation *op) {
  return op->getName().getStringRef().starts_with("wafer.instr.");
}

static mlir::LogicalResult emitInstructionCall(
    mlir::Operation *op, mlir::OpBuilder &builder, mlir::Value &status,
    const llvm::DenseMap<mlir::Value, mlir::Value> &externalBases) {
  mlir::Location loc = op->getLoc();
  auto appendI64Constants = [&](llvm::ArrayRef<int64_t> values,
                                llvm::SmallVectorImpl<mlir::Value> &arguments) {
    for (int64_t value : values)
      arguments.push_back(createI64Constant(builder, loc, value));
  };

  if (auto rdma = mlir::dyn_cast<InstrRDMAOp>(op)) {
    mlir::FailureOr<DdrAddress> ddr =
        getDdrAddress(op, rdma.getSource(), externalBases);
    if (mlir::failed(ddr))
      return mlir::failure();
    mlir::FailureOr<int64_t> spm = getSpmOffset(op, rdma.getDest());
    if (mlir::failed(spm))
      return mlir::failure();
    mlir::FailureOr<abi::DataFormat> format =
        getDataFormat(op, rdma.getSource());
    if (mlir::failed(format))
      return mlir::failure();
    if (mlir::failed(validateDmaDescriptor(
            op, *ddr, *spm, rdma.getByteCount(), rdma.getInnerBytes(),
            rdma.getSrcStrides(), rdma.getSrcIterations(), *format,
            /*isRdma=*/true)))
      return mlir::failure();
    mlir::Value ddrValue = createDdrAddressValue(builder, loc, *ddr);
    mlir::Value spmValue = createSpmOffsetValue(builder, loc, *spm);
    llvm::SmallVector<mlir::Value> arguments = {
        ddrValue, spmValue,
        createI64Constant(builder, loc, rdma.getByteCount()),
        createI64Constant(builder, loc, rdma.getInnerBytes())};
    appendI64Constants(rdma.getSrcStrides(), arguments);
    appendI64Constants(rdma.getSrcIterations(), arguments);
    arguments.push_back(
        createI32Constant(builder, loc, static_cast<int64_t>(*format)));
    status = emitAbiCall(builder, loc, "wafer_rdma", arguments, status);
    return mlir::success();
  }

  if (auto wdma = mlir::dyn_cast<InstrWDMAOp>(op)) {
    mlir::FailureOr<int64_t> spm = getSpmOffset(op, wdma.getSource());
    if (mlir::failed(spm))
      return mlir::failure();
    mlir::FailureOr<DdrAddress> ddr =
        getDdrAddress(op, wdma.getDest(), externalBases);
    if (mlir::failed(ddr))
      return mlir::failure();
    mlir::FailureOr<abi::DataFormat> format =
        getDataFormat(op, wdma.getSource());
    if (mlir::failed(format))
      return mlir::failure();
    if (mlir::failed(validateDmaDescriptor(
            op, *ddr, *spm, wdma.getByteCount(), wdma.getInnerBytes(),
            wdma.getDstStrides(), wdma.getDstIterations(), *format,
            /*isRdma=*/false)))
      return mlir::failure();
    mlir::Value spmValue = createSpmOffsetValue(builder, loc, *spm);
    mlir::Value ddrValue = createDdrAddressValue(builder, loc, *ddr);
    llvm::SmallVector<mlir::Value> arguments = {
        spmValue, ddrValue,
        createI64Constant(builder, loc, wdma.getByteCount()),
        createI64Constant(builder, loc, wdma.getInnerBytes())};
    appendI64Constants(wdma.getDstStrides(), arguments);
    appendI64Constants(wdma.getDstIterations(), arguments);
    arguments.push_back(
        createI32Constant(builder, loc, static_cast<int64_t>(*format)));
    status = emitAbiCall(builder, loc, "wafer_wdma", arguments, status);
    return mlir::success();
  }

  if (auto gatherScatter = mlir::dyn_cast<InstrGatherScatterOp>(op)) {
    mlir::FailureOr<int64_t> source =
        getSpmOffset(op, gatherScatter.getSource());
    if (mlir::failed(source))
      return mlir::failure();
    mlir::FailureOr<int64_t> dest = getSpmOffset(op, gatherScatter.getDest());
    if (mlir::failed(dest))
      return mlir::failure();

    auto addOptionalByteOffset =
        [&](int64_t base, std::optional<uint64_t> localOffset,
            llvm::StringRef role) -> mlir::FailureOr<int64_t> {
      if (!localOffset)
        return base;
      if (*localOffset >
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
        return op->emitError()
               << kFailurePrefix << role << " byte offset overflows int64";
      int64_t signedOffset = static_cast<int64_t>(*localOffset);
      if (base < 0 || signedOffset > std::numeric_limits<int64_t>::max() - base)
        return op->emitError()
               << kFailurePrefix << role << " byte offset overflows int64";
      return base + signedOffset;
    };

    source = addOptionalByteOffset(*source, gatherScatter.getSrcOffset(),
                                   "gather/scatter source");
    if (mlir::failed(source))
      return mlir::failure();
    dest = addOptionalByteOffset(*dest, gatherScatter.getDstOffset(),
                                 "gather/scatter dest");
    if (mlir::failed(dest))
      return mlir::failure();
    if (mlir::failed(validateGatherScatterDescriptor(
            op, *source, *dest, gatherScatter.getByteCount(),
            gatherScatter.getInnerBytes(), gatherScatter.getSrcStrides(),
            gatherScatter.getSrcIterations(), gatherScatter.getDstStrides(),
            gatherScatter.getDstIterations())))
      return mlir::failure();

    llvm::SmallVector<mlir::Value> arguments = {
        createSpmOffsetValue(builder, loc, *source),
        createSpmOffsetValue(builder, loc, *dest),
        createI64Constant(builder, loc, gatherScatter.getByteCount()),
        createI64Constant(builder, loc, gatherScatter.getInnerBytes())};
    appendI64Constants(gatherScatter.getSrcStrides(), arguments);
    appendI64Constants(gatherScatter.getSrcIterations(), arguments);
    appendI64Constants(gatherScatter.getDstStrides(), arguments);
    appendI64Constants(gatherScatter.getDstIterations(), arguments);
    status =
        emitAbiCall(builder, loc, "wafer_gather_scatter", arguments, status);
    return mlir::success();
  }

  if (auto fill = mlir::dyn_cast<InstrFillOp>(op)) {
    mlir::FailureOr<int64_t> dest = getSpmOffset(op, fill.getDest());
    if (mlir::failed(dest))
      return mlir::failure();
    mlir::FailureOr<int64_t> elements =
        getStaticElementCount(op, fill.getDest(), "fill dest");
    if (mlir::failed(elements))
      return mlir::failure();
    mlir::FailureOr<abi::DataFormat> format = getDataFormat(op, fill.getDest());
    if (mlir::failed(format))
      return mlir::failure();
    mlir::FailureOr<uint64_t> valueBits =
        getScalarConstantBits(op, fill.getValue(), "fill value");
    if (mlir::failed(valueBits))
      return mlir::failure();
    if (*valueBits > std::numeric_limits<uint32_t>::max())
      return fill.emitError()
             << kFailurePrefix
             << "fill value bits must fit in the native memset operand";

    status = emitAbiCall(
        builder, loc, "wafer_fill",
        {createSpmOffsetValue(builder, loc, *dest),
         createI64Constant(builder, loc, static_cast<int64_t>(*valueBits)),
         createI64Constant(builder, loc, *elements),
         createI32Constant(builder, loc, static_cast<int64_t>(*format))},
        status);
    return mlir::success();
  }

  if (auto elementwise = mlir::dyn_cast<InstrElementwiseOp>(op)) {
    ComputeElementwiseKind kind = elementwise.getKindAttr().getValue();
    unsigned expectedInputs = isUnaryElementwiseKind(kind) ? 1 : 2;
    if (!isUnaryElementwiseKind(kind) && !isBinaryElementwiseKind(kind))
      return elementwise.emitError()
             << kFailurePrefix << "unsupported elementwise kind";
    if (elementwise.getInputs().size() != expectedInputs)
      return elementwise.emitError()
             << kFailurePrefix << "elementwise input arity does not match kind";

    mlir::FailureOr<int64_t> dest = getSpmOffset(op, elementwise.getDest());
    if (mlir::failed(dest))
      return mlir::failure();
    mlir::FailureOr<int64_t> src0 =
        getSpmOffset(op, elementwise.getInputs().front());
    if (mlir::failed(src0))
      return mlir::failure();
    int64_t src1Value = 0;
    if (expectedInputs == 2) {
      mlir::FailureOr<int64_t> src1 =
          getSpmOffset(op, elementwise.getInputs()[1]);
      if (mlir::failed(src1))
        return mlir::failure();
      src1Value = *src1;
    }

    mlir::FailureOr<int64_t> elements =
        getStaticElementCount(op, elementwise.getDest(), "elementwise dest");
    if (mlir::failed(elements))
      return mlir::failure();
    for (mlir::Value input : elementwise.getInputs()) {
      mlir::FailureOr<int64_t> inputElements =
          getStaticElementCount(op, input, "elementwise input");
      if (mlir::failed(inputElements))
        return mlir::failure();
      if (*inputElements != *elements)
        return elementwise.emitError()
               << kFailurePrefix
               << "elementwise input element count must match dest";
    }

    mlir::FailureOr<abi::DataFormat> inputFormat =
        getDataFormat(op, elementwise.getInputs().front());
    if (mlir::failed(inputFormat))
      return mlir::failure();
    mlir::FailureOr<abi::DataFormat> outputFormat =
        getDataFormat(op, elementwise.getDest());
    if (mlir::failed(outputFormat))
      return mlir::failure();

    status = emitAbiCall(
        builder, loc, "wafer_elementwise",
        {createI32Constant(builder, loc, static_cast<int64_t>(kind)),
         createSpmOffsetValue(builder, loc, *dest),
         createSpmOffsetValue(builder, loc, *src0),
         createSpmOffsetValue(builder, loc, src1Value),
         createI64Constant(builder, loc, *elements),
         createI32Constant(builder, loc, static_cast<int64_t>(*inputFormat)),
         createI32Constant(builder, loc, static_cast<int64_t>(*outputFormat))},
        status);
    return mlir::success();
  }

  if (auto reduce = mlir::dyn_cast<InstrReduceOp>(op)) {
    auto dimensionsAttr =
        reduce->getAttrOfType<mlir::DenseI64ArrayAttr>("dimensions");
    if (!dimensionsAttr)
      return reduce.emitError()
             << kFailurePrefix << "reduce requires dimensions attr";
    if (mlir::failed(verifyZeroInit(op, reduce.getInit(), "reduce")))
      return mlir::failure();

    mlir::FailureOr<int64_t> source = getSpmOffset(op, reduce.getInput());
    if (mlir::failed(source))
      return mlir::failure();
    mlir::FailureOr<int64_t> dest = getSpmOffset(op, reduce.getDest());
    if (mlir::failed(dest))
      return mlir::failure();
    auto inputType = mlir::cast<mlir::MemRefType>(reduce.getInput().getType());
    mlir::FailureOr<int64_t> nativeDim = getNativeReduceDim(
        op, inputType.getRank(), dimensionsAttr.asArrayRef());
    if (mlir::failed(nativeDim))
      return mlir::failure();
    mlir::FailureOr<std::array<int64_t, 4>> shape =
        getStaticShape4D(op, reduce.getInput(), "reduce input");
    if (mlir::failed(shape))
      return mlir::failure();
    mlir::FailureOr<abi::DataFormat> format =
        getDataFormat(op, reduce.getInput());
    if (mlir::failed(format))
      return mlir::failure();

    status = emitAbiCall(
        builder, loc, "wafer_reduce",
        {createI32Constant(builder, loc,
                           static_cast<int64_t>(reduce.getKind())),
         createSpmOffsetValue(builder, loc, *source),
         createSpmOffsetValue(builder, loc, *dest),
         createI32Constant(builder, loc, *nativeDim),
         createI64Constant(builder, loc, (*shape)[0]),
         createI64Constant(builder, loc, (*shape)[1]),
         createI64Constant(builder, loc, (*shape)[2]),
         createI64Constant(builder, loc, (*shape)[3]),
         createI32Constant(builder, loc, static_cast<int64_t>(*format))},
        status);
    return mlir::success();
  }

  if (auto convert = mlir::dyn_cast<InstrConvertOp>(op)) {
    mlir::FailureOr<int64_t> source = getSpmOffset(op, convert.getSource());
    if (mlir::failed(source))
      return mlir::failure();
    mlir::FailureOr<int64_t> dest = getSpmOffset(op, convert.getDest());
    if (mlir::failed(dest))
      return mlir::failure();
    mlir::FailureOr<int64_t> sourceElements =
        getStaticElementCount(op, convert.getSource(), "convert source");
    if (mlir::failed(sourceElements))
      return mlir::failure();
    mlir::FailureOr<int64_t> destElements =
        getStaticElementCount(op, convert.getDest(), "convert dest");
    if (mlir::failed(destElements))
      return mlir::failure();
    if (*sourceElements != *destElements)
      return convert.emitError()
             << kFailurePrefix
             << "convert source element count must match dest";
    mlir::FailureOr<abi::DataFormat> sourceFormat =
        getDataFormat(op, convert.getSource());
    if (mlir::failed(sourceFormat))
      return mlir::failure();
    mlir::FailureOr<abi::DataFormat> destFormat =
        getDataFormat(op, convert.getDest());
    if (mlir::failed(destFormat))
      return mlir::failure();

    status = emitAbiCall(
        builder, loc, "wafer_convert",
        {createI32Constant(builder, loc, static_cast<int64_t>(*sourceFormat)),
         createI32Constant(builder, loc, static_cast<int64_t>(*destFormat)),
         createSpmOffsetValue(builder, loc, *source),
         createSpmOffsetValue(builder, loc, *dest),
         createI64Constant(builder, loc, *sourceElements)},
        status);
    return mlir::success();
  }

  if (auto gemm = mlir::dyn_cast<InstrGemmOp>(op)) {
    mlir::FailureOr<int64_t> lhsBase = getSpmOffset(op, gemm.getLhs());
    if (mlir::failed(lhsBase))
      return mlir::failure();
    mlir::FailureOr<int64_t> rhsBase = getSpmOffset(op, gemm.getRhs());
    if (mlir::failed(rhsBase))
      return mlir::failure();
    mlir::FailureOr<int64_t> destBase = getSpmOffset(op, gemm.getDest());
    if (mlir::failed(destBase))
      return mlir::failure();
    mlir::FailureOr<abi::DataFormat> inputFormat =
        getDataFormat(op, gemm.getLhs());
    if (mlir::failed(inputFormat))
      return mlir::failure();
    mlir::FailureOr<abi::DataFormat> outputFormat =
        getDataFormat(op, gemm.getDest());
    if (mlir::failed(outputFormat))
      return mlir::failure();
    abi::GemmDescriptor descriptor;
    std::string error;
    if (!abi::buildGemm(gemm.getM(), gemm.getK(), gemm.getN(), descriptor,
                        &error))
      return gemm.emitError() << kFailurePrefix << error;

    mlir::FailureOr<GemmBatchAttrs> batchAttrs = getGemmBatchAttrs(gemm);
    if (mlir::failed(batchAttrs))
      return mlir::failure();

    auto lhsType = mlir::cast<mlir::MemRefType>(gemm.getLhs().getType());
    auto rhsType = mlir::cast<mlir::MemRefType>(gemm.getRhs().getType());
    auto destType = mlir::cast<mlir::MemRefType>(gemm.getDest().getType());
    llvm::SmallVector<int64_t, 4> batchShape;
    if (batchAttrs->enabled) {
      mlir::FailureOr<llvm::SmallVector<int64_t, 4>> shape = getBatchShape(
          op, destType, batchAttrs->resultBatchDims, "GEMM result");
      if (mlir::failed(shape))
        return mlir::failure();
      batchShape = std::move(*shape);
    }

    for (int64_t batch = 0; batch < batchAttrs->batchCount; ++batch) {
      int64_t lhs = *lhsBase;
      int64_t rhs = *rhsBase;
      int64_t dest = *destBase;
      if (batchAttrs->enabled) {
        mlir::FailureOr<llvm::SmallVector<int64_t, 4>> batchIndices =
            delinearizeBatchIndex(op, batch, batchShape);
        if (mlir::failed(batchIndices))
          return mlir::failure();
        mlir::FailureOr<int64_t> lhsDelta = getBatchPhysicalByteOffset(
            op, lhsType, batchAttrs->lhsBatchDims, *batchIndices, "GEMM lhs");
        mlir::FailureOr<int64_t> rhsDelta = getBatchPhysicalByteOffset(
            op, rhsType, batchAttrs->rhsBatchDims, *batchIndices, "GEMM rhs");
        mlir::FailureOr<int64_t> destDelta = getBatchPhysicalByteOffset(
            op, destType, batchAttrs->resultBatchDims, *batchIndices,
            "GEMM dest");
        if (mlir::failed(lhsDelta) || mlir::failed(rhsDelta) ||
            mlir::failed(destDelta))
          return mlir::failure();
        mlir::FailureOr<int64_t> lhsOffset =
            addSpmByteOffsets(op, lhs, *lhsDelta, "GEMM lhs");
        mlir::FailureOr<int64_t> rhsOffset =
            addSpmByteOffsets(op, rhs, *rhsDelta, "GEMM rhs");
        mlir::FailureOr<int64_t> destOffset =
            addSpmByteOffsets(op, dest, *destDelta, "GEMM dest");
        if (mlir::failed(lhsOffset) || mlir::failed(rhsOffset) ||
            mlir::failed(destOffset))
          return mlir::failure();
        lhs = *lhsOffset;
        rhs = *rhsOffset;
        dest = *destOffset;
      }

      abi::GemmRegisterPacket packet;
      if (!abi::buildGemmRegisterPacket(
              descriptor, static_cast<uint32_t>(lhs),
              static_cast<uint32_t>(rhs), static_cast<uint32_t>(dest),
              *inputFormat, *outputFormat, packet, &error))
        return gemm.emitError() << kFailurePrefix << error;
      status = emitAbiCall(
          builder, loc, "wafer_gemm",
          {createSpmOffsetValue(builder, loc, lhs),
           createSpmOffsetValue(builder, loc, rhs),
           createSpmOffsetValue(builder, loc, dest),
           createI64Constant(builder, loc, gemm.getM()),
           createI64Constant(builder, loc, gemm.getK()),
           createI64Constant(builder, loc, gemm.getN()),
           createI32Constant(builder, loc, static_cast<int64_t>(*inputFormat))},
          status);
    }
    return mlir::success();
  }

  if (auto send = mlir::dyn_cast<InstrDTESendOp>(op)) {
    mlir::FailureOr<int64_t> buffer = getSpmOffset(op, send.getBuffer());
    if (mlir::failed(buffer))
      return mlir::failure();
    if (send.getPeer() > std::numeric_limits<uint32_t>::max())
      return send.emitError()
             << kFailurePrefix << "DTE peer is not representable";
    status = emitAbiCall(builder, loc, "wafer_dte_send",
                         {createSpmOffsetValue(builder, loc, *buffer),
                          createI32Constant(builder, loc, send.getPeer()),
                          createI64Constant(builder, loc, send.getBytes())},
                         status);
    return mlir::success();
  }

  if (auto recv = mlir::dyn_cast<InstrDTERecvOp>(op)) {
    mlir::FailureOr<int64_t> buffer = getSpmOffset(op, recv.getBuffer());
    if (mlir::failed(buffer))
      return mlir::failure();
    if (recv.getPeer() > std::numeric_limits<uint32_t>::max())
      return recv.emitError()
             << kFailurePrefix << "DTE peer is not representable";
    status = emitAbiCall(builder, loc, "wafer_dte_recv",
                         {createSpmOffsetValue(builder, loc, *buffer),
                          createI32Constant(builder, loc, recv.getPeer()),
                          createI64Constant(builder, loc, recv.getBytes())},
                         status);
    return mlir::success();
  }

  if (auto wait = mlir::dyn_cast<InstrDTEWaitOp>(op)) {
    status = emitAbiCall(
        builder, loc, "wafer_dte_wait",
        {createI32Constant(builder, loc, wait.getTokens().size())}, status);
    return mlir::success();
  }

  if (mlir::isa<SyncLocalFenceOp>(op)) {
    status = emitAbiCall(builder, loc, "wafer_local_fence", {}, status);
    return mlir::success();
  }

  return op->emitError() << kFailurePrefix << "unsupported instruction op "
                         << op->getName().getStringRef();
}

static bool containsTileRegion(mlir::func::FuncOp func) {
  bool found = false;
  func.walk([&](TileRegionOp) { found = true; });
  return found;
}

static mlir::LogicalResult materializeFunction(mlir::ModuleOp module,
                                               mlir::func::FuncOp func) {
  std::string abiName = (func.getSymName() + "_abi").str();
  if (module.lookupSymbol(abiName))
    return func.emitError() << kFailurePrefix << "ABI entry symbol @" << abiName
                            << " already exists";

  mlir::MLIRContext *ctx = module.getContext();
  mlir::Type i32 = mlir::IntegerType::get(ctx, 32);
  mlir::Type i64 = mlir::IntegerType::get(ctx, 64);

  llvm::SmallVector<mlir::Type> abiInputTypes;
  llvm::SmallVector<unsigned> ddrArgOrdinals;
  mlir::FunctionType originalType = func.getFunctionType();
  for (auto [index, type] : llvm::enumerate(originalType.getInputs())) {
    if (!isWaferDDRMemRefType(type)) {
      return func.emitError()
             << kFailurePrefix
             << "ABI materialization currently accepts only external DDR "
             << "memref function arguments";
    }
    abiInputTypes.push_back(i64);
    ddrArgOrdinals.push_back(index);
  }

  llvm::SmallVector<mlir::Value> inputAliases;
  for (unsigned index : ddrArgOrdinals)
    appendUnique(inputAliases, func.getArgument(index));

  llvm::SmallVector<ReturnOutputBinding> outputBindings;
  if (originalType.getNumResults() != 0) {
    llvm::SmallVector<mlir::func::ReturnOp> returns;
    func.walk(
        [&](mlir::func::ReturnOp returnOp) { returns.push_back(returnOp); });
    if (returns.size() != 1)
      return func.emitError()
             << kFailurePrefix
             << "ABI materialization requires one return op for returned "
             << "outputs";
    mlir::func::ReturnOp returnOp = returns.front();
    if (returnOp.getNumOperands() != originalType.getNumResults())
      return returnOp.emitError()
             << kFailurePrefix << "return operand count must match function "
             << "result count";

    for (auto [index, type] : llvm::enumerate(originalType.getResults())) {
      if (!isWaferDDRMemRefType(type))
        return func.emitError()
               << kFailurePrefix
               << "ABI materialization currently accepts only returned DDR "
               << "memrefs";

      llvm::SmallVector<mlir::Value> aliases;
      collectReturnedDdrAliases(returnOp.getOperand(index), aliases);
      bool alreadyExternal = llvm::any_of(aliases, [&](mlir::Value alias) {
        return llvm::is_contained(inputAliases, alias);
      });
      if (alreadyExternal)
        continue;

      ReturnOutputBinding binding;
      binding.aliases = std::move(aliases);
      binding.abiIndex = abiInputTypes.size();
      abiInputTypes.push_back(i64);
      outputBindings.push_back(std::move(binding));
    }
  }

  llvm::SmallVector<WorkspaceRootBinding> workspaceRoots;
  int64_t workspaceBytes = 0;
  if (mlir::failed(collectWorkspaceRootBindings(
          func, outputBindings, workspaceRoots, workspaceBytes)))
    return mlir::failure();
  std::optional<unsigned> workspaceAbiIndex;
  if (!workspaceRoots.empty()) {
    workspaceAbiIndex = abiInputTypes.size();
    abiInputTypes.push_back(i64);
  }

  mlir::FunctionType abiType =
      mlir::FunctionType::get(ctx, abiInputTypes, {i32});
  mlir::OpBuilder moduleBuilder(module.getBodyRegion());
  moduleBuilder.setInsertionPoint(func);
  auto abiFunc =
      moduleBuilder.create<mlir::func::FuncOp>(func.getLoc(), abiName, abiType);

  mlir::Block *entry = abiFunc.addEntryBlock();
  mlir::OpBuilder builder(entry, entry->begin());
  llvm::DenseMap<mlir::Value, mlir::Value> externalBases;
  for (auto [abiIndex, originalIndex] : llvm::enumerate(ddrArgOrdinals))
    externalBases[func.getArgument(originalIndex)] =
        entry->getArgument(abiIndex);
  for (const ReturnOutputBinding &binding : outputBindings) {
    for (mlir::Value alias : binding.aliases)
      externalBases[alias] = entry->getArgument(binding.abiIndex);
  }
  if (workspaceAbiIndex) {
    mlir::Value workspaceBase = entry->getArgument(*workspaceAbiIndex);
    for (const WorkspaceRootBinding &binding : workspaceRoots) {
      if (binding.offset == 0) {
        externalBases[binding.root] = workspaceBase;
        continue;
      }
      mlir::Value offset =
          createI64Constant(builder, binding.root.getLoc(), binding.offset);
      externalBases[binding.root] = builder.create<mlir::arith::AddIOp>(
          binding.root.getLoc(), workspaceBase, offset);
    }
  }

  mlir::Value status = createI32Constant(builder, func.getLoc(), 0);

  llvm::SmallVector<TileRegionOp> regions;
  func.walk([&](TileRegionOp region) { regions.push_back(region); });

  for (TileRegionOp region : regions) {
    if (region.getBody().empty())
      return region.emitError()
             << kFailurePrefix << "tile region must have one entry block";
    for (mlir::Operation &bodyOp : region.getBody().front()) {
      if (mlir::isa<TileYieldOp>(bodyOp))
        continue;
      if (isSupportedAbiInstruction(&bodyOp)) {
        if (mlir::failed(
                emitInstructionCall(&bodyOp, builder, status, externalBases)))
          return mlir::failure();
        continue;
      }
      if (isWaferInstructionOp(&bodyOp))
        return emitInstructionCall(&bodyOp, builder, status, externalBases);
    }
  }

  builder.create<mlir::func::ReturnOp>(func.getLoc(), status);
  func.erase();
  return mlir::success();
}

struct MaterializeABICallsPass
    : public impl::MaterializeABICallsPassBase<MaterializeABICallsPass> {
  using impl::MaterializeABICallsPassBase<
      MaterializeABICallsPass>::MaterializeABICallsPassBase;

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    if (mlir::failed(ensureAbiDeclarations(module))) {
      signalPassFailure();
      return;
    }

    llvm::SmallVector<mlir::func::FuncOp> funcs;
    module.walk([&](mlir::func::FuncOp func) {
      if (containsTileRegion(func))
        funcs.push_back(func);
    });

    for (mlir::func::FuncOp func : funcs) {
      if (mlir::failed(materializeFunction(module, func))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace
} // namespace wafer
