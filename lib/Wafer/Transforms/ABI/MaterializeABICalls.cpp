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
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
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
    auto viewLike = mlir::dyn_cast<mlir::ViewLikeOpInterface>(def);
    if (!viewLike)
      return value;
    mlir::Value source = resolveTileRegionBoundaryValue(viewLike.getViewSource());
    if (source == value)
      return value;
    value = source;
  }
  return value;
}

static mlir::FailureOr<int64_t> getElementByteWidth(mlir::Operation *op,
                                                    mlir::Type type) {
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
    unsigned width = intType.getWidth();
    if (width == 0 || width % 8 != 0)
      return op->emitError()
             << kFailurePrefix << "memref element type must be byte-addressable";
    return static_cast<int64_t>(width / 8);
  }
  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(type)) {
    unsigned width = floatType.getWidth();
    if (width == 0 || width % 8 != 0)
      return op->emitError()
             << kFailurePrefix << "memref element type must be byte-addressable";
    return static_cast<int64_t>(width / 8);
  }
  if (mlir::isa<mlir::IndexType>(type))
    return int64_t{8};
  return op->emitError()
         << kFailurePrefix << "memref element type must be scalar int, float, "
         << "or index";
}

static mlir::FailureOr<int64_t>
getStaticByteOffset(mlir::Operation *op, mlir::MemRefType type,
                    llvm::StringRef role) {
  llvm::SmallVector<int64_t> strides;
  int64_t offset = 0;
  if (mlir::failed(mlir::getStridesAndOffset(type, strides, offset)) ||
      strides.size() != static_cast<size_t>(type.getRank()))
    return op->emitError()
           << kFailurePrefix << role
           << " view must have static strided layout";
  if (offset == mlir::ShapedType::kDynamic || offset < 0)
    return op->emitError()
           << kFailurePrefix << role
           << " view must have static non-negative byte offset";

  mlir::FailureOr<int64_t> elementBytes =
      getElementByteWidth(op, type.getElementType());
  if (mlir::failed(elementBytes))
    return mlir::failure();
  if (offset > std::numeric_limits<int64_t>::max() / *elementBytes)
    return op->emitError()
           << kFailurePrefix << role << " view byte offset overflows int64";
  return offset * *elementBytes;
}

static mlir::FailureOr<int64_t> getSpmOffset(mlir::Operation *op,
                                             mlir::Value value) {
  auto valueType = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  if (!valueType || !isWaferSPMMemRefType(valueType))
    return op->emitError()
           << kFailurePrefix << "SPM operand must have Wafer SPM memref type";

  mlir::Value root = getRootViewSource(value);
  mlir::Operation *rootDef = root.getDefiningOp();
  if (!rootDef)
    return op->emitError()
           << kFailurePrefix
           << "SPM memref has no accepted wafer.spm.offset";
  auto offsetAttr =
      rootDef->getAttrOfType<SPMOffsetAttr>(kWaferSPMOffsetAttrName);
  if (!offsetAttr)
    return op->emitError()
           << kFailurePrefix
           << "SPM memref has no accepted wafer.spm.offset";

  mlir::FailureOr<int64_t> viewOffset =
      getStaticByteOffset(op, valueType, "SPM");
  if (mlir::failed(viewOffset))
    return mlir::failure();
  int64_t base = offsetAttr.getOffset();
  if (base < 0 ||
      *viewOffset > std::numeric_limits<int64_t>::max() - base)
    return op->emitError()
           << kFailurePrefix << "SPM byte offset overflows int64";
  int64_t offset = base + *viewOffset;
  if (offset > std::numeric_limits<uint32_t>::max())
    return op->emitError()
           << kFailurePrefix << "SPM byte offset is not representable";
  return offset;
}

static mlir::FailureOr<DdrAddress>
getDdrAddress(mlir::Operation *op, mlir::Value value,
              const llvm::DenseMap<mlir::Value, mlir::Value> &externalBases) {
  auto valueType = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  if (!valueType || !isWaferDDRMemRefType(valueType))
    return op->emitError()
           << kFailurePrefix << "DDR operand must have Wafer DDR memref type";

  mlir::Value root = getRootViewSource(value);
  auto it = externalBases.find(root);
  if (it == externalBases.end())
    return op->emitError()
           << kFailurePrefix
           << "DDR operand must resolve to an external function argument";

  mlir::FailureOr<int64_t> viewOffset =
      getStaticByteOffset(op, valueType, "DDR");
  if (mlir::failed(viewOffset))
    return mlir::failure();
  return DdrAddress{it->second, *viewOffset};
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

static mlir::FailureOr<std::array<int64_t, 3>>
getDescriptorTriple(mlir::Operation *op, llvm::ArrayRef<int64_t> values,
                    llvm::StringRef role) {
  if (values.size() != 3)
    return op->emitError()
           << kFailurePrefix << role << " descriptor must have three values";
  return std::array<int64_t, 3>{values[0], values[1], values[2]};
}

static mlir::LogicalResult
validateDmaDescriptor(mlir::Operation *op, DdrAddress ddrAddress,
                      int64_t spmOffset, uint64_t byteCount,
                      uint64_t innerBytes, llvm::ArrayRef<int64_t> strides,
                      llvm::ArrayRef<int64_t> iterations, bool isRdma) {
  if (ddrAddress.byteOffset < 0)
    return op->emitError()
           << kFailurePrefix << "DDR byte offset must be non-negative";
  if (static_cast<uint64_t>(ddrAddress.byteOffset) >
      std::numeric_limits<uint64_t>::max() - abi::kDdrLowerBound)
    return op->emitError() << kFailurePrefix << "DDR address overflows";
  if (spmOffset < 0 ||
      spmOffset > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
    return op->emitError()
           << kFailurePrefix << "SPM byte offset is not representable";

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
  bool ok = isRdma ? abi::buildRdma(staticDdrAddress,
                                    static_cast<uint32_t>(spmOffset),
                                    byteCount, innerBytes, *strideArray,
                                    *iterationArray, descriptor, &error)
                   : abi::buildWdma(static_cast<uint32_t>(spmOffset),
                                    staticDdrAddress, byteCount, innerBytes,
                                    *strideArray, *iterationArray, descriptor,
                                    &error);
  if (!ok)
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
      return existing.emitError()
             << kFailurePrefix << "ABI declaration @" << name
             << " has incompatible function type";
    return mlir::success();
  }

  mlir::OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPointToEnd(module.getBody());
  auto declaration = builder.create<mlir::func::FuncOp>(
      module.getLoc(), name, type);
  declaration.setPrivate();
  return mlir::success();
}

static mlir::LogicalResult ensureAbiDeclarations(mlir::ModuleOp module) {
  mlir::MLIRContext *ctx = module.getContext();
  mlir::Type i32 = mlir::IntegerType::get(ctx, 32);
  mlir::Type i64 = mlir::IntegerType::get(ctx, 64);

  llvm::SmallVector<mlir::Type> rdmaInputs = {i64, i32, i64, i64};
  rdmaInputs.append(6, i64);
  if (mlir::failed(
          ensureAbiDeclaration(module, "wafer_rdma", rdmaInputs, {i32})))
    return mlir::failure();
  llvm::SmallVector<mlir::Type> wdmaInputs = {i32, i64, i64, i64};
  wdmaInputs.append(6, i64);
  if (mlir::failed(
          ensureAbiDeclaration(module, "wafer_wdma", wdmaInputs, {i32})))
    return mlir::failure();
  if (mlir::failed(ensureAbiDeclaration(module, "wafer_gemm",
                                        {i32, i32, i32, i64, i64, i64},
                                        {i32})))
    return mlir::failure();
  llvm::SmallVector<mlir::Type> gatherScatterInputs = {i32, i32, i64, i64};
  gatherScatterInputs.append(12, i64);
  if (mlir::failed(ensureAbiDeclaration(module, "wafer_gather_scatter",
                                        gatherScatterInputs, {i32})))
    return mlir::failure();
  if (mlir::failed(
          ensureAbiDeclaration(module, "wafer_local_fence", {}, {i32})))
    return mlir::failure();
  return mlir::success();
}

static mlir::Value emitAbiCall(mlir::OpBuilder &builder, mlir::Location loc,
                               llvm::StringRef callee,
                               mlir::ValueRange arguments,
                               mlir::Value status) {
  auto call = builder.create<mlir::func::CallOp>(
      loc, callee, mlir::TypeRange{status.getType()}, arguments);
  return builder.create<mlir::arith::OrIOp>(loc, status, call.getResult(0));
}

static bool isSupportedAbiInstruction(mlir::Operation *op) {
  return mlir::isa<InstrRDMAOp, InstrWDMAOp, InstrGatherScatterOp, InstrGemmOp,
                   SyncLocalFenceOp>(op);
}

static bool isWaferInstructionOp(mlir::Operation *op) {
  return op->getName().getStringRef().starts_with("wafer.instr.");
}

static mlir::LogicalResult
emitInstructionCall(mlir::Operation *op, mlir::OpBuilder &builder,
                    mlir::Value &status,
                    const llvm::DenseMap<mlir::Value, mlir::Value>
                        &externalBases) {
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
    if (mlir::failed(validateDmaDescriptor(
            op, *ddr, *spm, rdma.getByteCount(), rdma.getInnerBytes(),
            rdma.getSrcStrides(), rdma.getSrcIterations(), /*isRdma=*/true)))
      return mlir::failure();
    mlir::Value ddrValue = createDdrAddressValue(builder, loc, *ddr);
    mlir::Value spmValue = createSpmOffsetValue(builder, loc, *spm);
    llvm::SmallVector<mlir::Value> arguments = {
        ddrValue, spmValue, createI64Constant(builder, loc, rdma.getByteCount()),
        createI64Constant(builder, loc, rdma.getInnerBytes())};
    appendI64Constants(rdma.getSrcStrides(), arguments);
    appendI64Constants(rdma.getSrcIterations(), arguments);
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
    if (mlir::failed(validateDmaDescriptor(
            op, *ddr, *spm, wdma.getByteCount(), wdma.getInnerBytes(),
            wdma.getDstStrides(), wdma.getDstIterations(), /*isRdma=*/false)))
      return mlir::failure();
    mlir::Value spmValue = createSpmOffsetValue(builder, loc, *spm);
    mlir::Value ddrValue = createDdrAddressValue(builder, loc, *ddr);
    llvm::SmallVector<mlir::Value> arguments = {
        spmValue, ddrValue, createI64Constant(builder, loc, wdma.getByteCount()),
        createI64Constant(builder, loc, wdma.getInnerBytes())};
    appendI64Constants(wdma.getDstStrides(), arguments);
    appendI64Constants(wdma.getDstIterations(), arguments);
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
      if (base < 0 ||
          signedOffset > std::numeric_limits<int64_t>::max() - base)
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

    llvm::SmallVector<mlir::Value> arguments = {
        createSpmOffsetValue(builder, loc, *source),
        createSpmOffsetValue(builder, loc, *dest),
        createI64Constant(builder, loc, gatherScatter.getByteCount()),
        createI64Constant(builder, loc, gatherScatter.getInnerBytes())};
    appendI64Constants(gatherScatter.getSrcStrides(), arguments);
    appendI64Constants(gatherScatter.getSrcIterations(), arguments);
    appendI64Constants(gatherScatter.getDstStrides(), arguments);
    appendI64Constants(gatherScatter.getDstIterations(), arguments);
    status = emitAbiCall(builder, loc, "wafer_gather_scatter", arguments,
                         status);
    return mlir::success();
  }

  if (auto gemm = mlir::dyn_cast<InstrGemmOp>(op)) {
    mlir::FailureOr<int64_t> lhs = getSpmOffset(op, gemm.getLhs());
    if (mlir::failed(lhs))
      return mlir::failure();
    mlir::FailureOr<int64_t> rhs = getSpmOffset(op, gemm.getRhs());
    if (mlir::failed(rhs))
      return mlir::failure();
    mlir::FailureOr<int64_t> dest = getSpmOffset(op, gemm.getDest());
    if (mlir::failed(dest))
      return mlir::failure();
    status = emitAbiCall(
        builder, loc, "wafer_gemm",
        {createSpmOffsetValue(builder, loc, *lhs),
         createSpmOffsetValue(builder, loc, *rhs),
         createSpmOffsetValue(builder, loc, *dest),
         createI64Constant(builder, loc, gemm.getM()),
         createI64Constant(builder, loc, gemm.getK()),
         createI64Constant(builder, loc, gemm.getN())},
        status);
    return mlir::success();
  }

  if (mlir::isa<SyncLocalFenceOp>(op)) {
    status = emitAbiCall(builder, loc, "wafer_local_fence", {}, status);
    return mlir::success();
  }

  return op->emitError()
         << kFailurePrefix << "unsupported instruction op "
         << op->getName().getStringRef();
}

static bool containsTileRegion(mlir::func::FuncOp func) {
  bool found = false;
  func.walk([&](TileRegionOp) { found = true; });
  return found;
}

static mlir::LogicalResult
materializeFunction(mlir::ModuleOp module, mlir::func::FuncOp func) {
  std::string abiName = (func.getSymName() + "_abi").str();
  if (module.lookupSymbol(abiName))
    return func.emitError()
           << kFailurePrefix << "ABI entry symbol @" << abiName
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
