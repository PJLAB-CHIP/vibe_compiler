#include "Wafer/Conversion/StableHLOToLinalg/Pipelines.h"
#include "Wafer/Frontend/InitImporterDialects.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitWaferDialects.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/STLExtras.h"

#include "gtest/gtest.h"

#include <initializer_list>
#include <optional>

namespace {

template <typename OpT> OpT findSingleOp(mlir::ModuleOp module);

TEST(WaferInterfacesTest, InterfaceClassesAreGenerated) {
  SUCCEED() << "Wafer interface headers compile";
}

TEST(WaferInterfacesTest,
     VerifiesLongDDRRegionChainWithoutRepeatedStorageTraversal) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::memref::MemRefDialect>();
  wafer::registerWaferCoreDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  mlir::OpBuilder builder(&context);
  mlir::Location loc = builder.getUnknownLoc();
  auto ddrType = mlir::MemRefType::get(
      {4}, builder.getF16Type(), mlir::MemRefLayoutAttrInterface{},
      wafer::MemoryAttr::get(&context, wafer::MemorySpace::DDR,
                             wafer::MemLayout::Tensor));
  auto module = mlir::ModuleOp::create(loc);
  auto function = mlir::func::FuncOp::create(
      loc, "ddr_region_chain",
      builder.getFunctionType(mlir::TypeRange{ddrType},
                              mlir::TypeRange{ddrType}));
  module.getBody()->push_back(function);
  mlir::Block *entry = function.addEntryBlock();
  builder.setInsertionPointToEnd(entry);

  // Explicit DDR stage splitting produces long sequential chains. Each
  // TileRegion verifies its own yield; a later sibling must consume that
  // verified DDR boundary without reopening the complete producer prefix.
  mlir::Value current = entry->getArgument(0);
  for (unsigned index = 0; index < 1024; ++index) {
    auto region = builder.create<wafer::TileRegionOp>(
        loc, mlir::TypeRange{ddrType}, mlir::ValueRange{current});
    region.getBody().push_back(new mlir::Block());
    mlir::Block &body = region.getBody().front();
    for (mlir::Value input : region.getInputs())
      body.addArgument(input.getType(), loc);
    mlir::OpBuilder bodyBuilder = mlir::OpBuilder::atBlockEnd(&body);
    bodyBuilder.create<wafer::TileYieldOp>(loc, body.getArgument(0));
    current = region.getResult(0);
  }
  builder.create<mlir::func::ReturnOp>(loc, current);

  EXPECT_TRUE(mlir::succeeded(mlir::verify(module)));
}

TEST(WaferInterfacesTest, TileRegionExposesStandardRegionBranchFlow) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::memref::MemRefDialect>();
  wafer::registerWaferCoreDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @region_branch(%input: memref<4xf16, #wafer.memory<ddr, tensor>>)
      -> memref<4xf16, #wafer.memory<ddr, tensor>> {
    %result = wafer.tile.region(
        %input : memref<4xf16, #wafer.memory<ddr, tensor>>) ->
        (memref<4xf16, #wafer.memory<ddr, tensor>>) {
      ^bb0(%arg: memref<4xf16, #wafer.memory<ddr, tensor>>):
        wafer.tile.yield %arg
            : memref<4xf16, #wafer.memory<ddr, tensor>>
    }
    return %result : memref<4xf16, #wafer.memory<ddr, tensor>>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  wafer::TileRegionOp tileRegion;
  module->walk([&](wafer::TileRegionOp op) {
    EXPECT_FALSE(tileRegion);
    tileRegion = op;
  });
  ASSERT_TRUE(tileRegion);
  auto branch =
      mlir::dyn_cast<mlir::RegionBranchOpInterface>(tileRegion.getOperation());
  ASSERT_TRUE(branch);

  llvm::SmallVector<mlir::RegionSuccessor, 1> successors;
  branch.getSuccessorRegions(mlir::RegionBranchPoint::parent(), successors);
  ASSERT_EQ(successors.size(), 1u);
  EXPECT_EQ(successors.front().getSuccessor(), &tileRegion.getBody());
  ASSERT_EQ(branch.getEntrySuccessorOperands(successors.front()).size(), 1u);
  EXPECT_EQ(branch.getEntrySuccessorOperands(successors.front()).front(),
            tileRegion.getInputs().front());

  successors.clear();
  branch.getSuccessorRegions(tileRegion.getBody(), successors);
  ASSERT_EQ(successors.size(), 1u);
  EXPECT_FALSE(successors.front().getSuccessor());
  ASSERT_EQ(successors.front().getSuccessorInputs().size(), 1u);
  EXPECT_EQ(successors.front().getSuccessorInputs().front(),
            tileRegion.getResult(0));

  auto yield = mlir::cast<wafer::TileYieldOp>(
      tileRegion.getBody().front().getTerminator());
  auto terminator = mlir::dyn_cast<mlir::RegionBranchTerminatorOpInterface>(
      yield.getOperation());
  ASSERT_TRUE(terminator);
  ASSERT_EQ(
      terminator.getSuccessorOperands(mlir::RegionBranchPoint::parent()).size(),
      1u);
  EXPECT_EQ(terminator.getSuccessorOperands(mlir::RegionBranchPoint::parent())
                .front(),
            yield.getValues().front());
}

TEST(WaferInterfacesTest, TileAliasesUseStandardViewAndDestinationContracts) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::memref::MemRefDialect>();
  wafer::registerWaferCoreDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @alias_contract(
      %source: memref<2x2xf16, #wafer.memory<spm, tensor>>,
      %dest: memref<4xf16, #wafer.memory<spm, tensor>>) {
    %view = wafer.tile.reshape %source
        : memref<2x2xf16, #wafer.memory<spm, tensor>>
       -> memref<4xf16, #wafer.memory<spm, tensor>>
    wafer.tile.insert_slice %view into %dest
        {offsets = array<i64: 0>, sizes = array<i64: 4>,
         strides = array<i64: 1>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
          into memref<4xf16, #wafer.memory<spm, tensor>>
    return
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  auto view = findSingleOp<wafer::ViewReshapeOp>(*module);
  ASSERT_TRUE(view);
  auto viewLike =
      mlir::dyn_cast<mlir::ViewLikeOpInterface>(view.getOperation());
  ASSERT_TRUE(viewLike);
  EXPECT_EQ(viewLike.getViewSource(), view.getSource());

  auto insert = findSingleOp<wafer::MoveInsertSliceOp>(*module);
  ASSERT_TRUE(insert);
  auto dps =
      mlir::dyn_cast<mlir::DestinationStyleOpInterface>(insert.getOperation());
  ASSERT_TRUE(dps);
  ASSERT_TRUE(dps.hasPureBufferSemantics());
  ASSERT_EQ(dps.getNumDpsInputs(), 1);
  ASSERT_EQ(dps.getNumDpsInits(), 1);
  EXPECT_EQ(dps.getDpsInputs().front(), insert.getSource());
  EXPECT_EQ(dps.getDpsInits().front(), insert.getDest());
  EXPECT_EQ(insert->getNumResults(), 0u);
}

template <typename OpT> OpT findSingleOp(mlir::ModuleOp module) {
  OpT found;
  module.walk([&](OpT op) {
    EXPECT_FALSE(found);
    found = op;
  });
  return found;
}

wafer::MemLayout getMemoryLayout(mlir::Value value) {
  auto type = mlir::cast<mlir::MemRefType>(value.getType());
  return wafer::getWaferMemoryAttr(type).getLayout();
}

template <typename EffectT, typename ResourceT>
bool hasMemoryEffect(
    llvm::ArrayRef<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        effects) {
  return llvm::any_of(effects, [](const auto &effect) {
    return llvm::isa<EffectT>(effect.getEffect()) &&
           llvm::isa<ResourceT>(effect.getResource());
  });
}

template <typename EffectT>
bool hasValueMemoryEffect(
    llvm::ArrayRef<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        effects,
    mlir::Value value) {
  return llvm::any_of(effects, [&](const auto &effect) {
    return llvm::isa<EffectT>(effect.getEffect()) && effect.getValue() == value;
  });
}

llvm::SmallVector<mlir::OpFoldResult>
getIndexOpFoldResults(mlir::MLIRContext &context,
                      llvm::ArrayRef<int64_t> values) {
  return mlir::getAsIndexOpFoldResult(&context, values);
}

bool hasConstantIntValues(llvm::ArrayRef<mlir::OpFoldResult> values,
                          llvm::ArrayRef<int64_t> expected) {
  std::optional<llvm::SmallVector<int64_t>> constants =
      mlir::getConstantIntValues(values);
  return constants && llvm::equal(*constants, expected);
}

TEST(WaferInterfacesTest,
     TensorIndexingExternalModelsExposeStaticSourceSemantics) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::tensor::TensorDialect>();
  wafer::registerWaferCoreDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @tensor_indexing(%input: tensor<2x1025x128xf16>, %offset: index)
      -> tensor<2x1025x128xf16> {
    %zero = arith.constant 0.0 : f16
    %dest = tensor.empty() : tensor<2x1025x128xf16>
    %slice = tensor.extract_slice %input[0, 1, 0] [2, 1024, 128]
        [1, 1, 1] : tensor<2x1025x128xf16> to tensor<2x1024x128xf16>
    %inserted = tensor.insert_slice %slice into %dest[0, 1, 0]
        [2, 1024, 128] [1, 1, 1]
        : tensor<2x1024x128xf16> into tensor<2x1025x128xf16>
    %collapsed = tensor.collapse_shape %slice [[0, 1], [2]]
        : tensor<2x1024x128xf16> into tensor<2048x128xf16>
    %expanded = tensor.expand_shape %collapsed [[0, 1], [2]]
        output_shape [2, 1024, 128]
        : tensor<2048x128xf16> into tensor<2x1024x128xf16>
    %padded = tensor.pad %expanded low[0, 1, 0] high[0, 0, 0] {
      ^bb0(%batch: index, %sequence: index, %feature: index):
        tensor.yield %zero : f16
    } : tensor<2x1024x128xf16> to tensor<2x1025x128xf16>
    %cast = tensor.cast %padded
        : tensor<2x1025x128xf16> to tensor<?x1025x128xf16>
    %dynamic = tensor.extract_slice %input[0, %offset, 0]
        [2, 1024, 128] [1, 1, 1]
        : tensor<2x1025x128xf16> to tensor<2x1024x128xf16>
    return %inserted : tensor<2x1025x128xf16>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  auto describe = [](mlir::Operation *operation) {
    auto indexing =
        mlir::dyn_cast<wafer::WaferTensorIndexingOpInterface>(operation);
    if (!indexing)
      return mlir::FailureOr<wafer::TensorIndexingDescription>(mlir::failure());
    return indexing.getTensorIndexingDescription(0);
  };

  auto insert = findSingleOp<mlir::tensor::InsertSliceOp>(*module);
  auto insertDescription = describe(insert);
  ASSERT_TRUE(mlir::succeeded(insertDescription));
  EXPECT_EQ(insertDescription->kind,
            wafer::TensorIndexingTransformKind::InsertSlice);
  ASSERT_EQ(insertDescription->operands.size(), 2u);
  EXPECT_EQ(insertDescription->operands[0].role,
            wafer::TensorIndexingOperandRole::Source);
  EXPECT_EQ(insertDescription->operands[1].role,
            wafer::TensorIndexingOperandRole::Destination);
  EXPECT_TRUE(llvm::equal(insertDescription->operands[0].offsets,
                          llvm::ArrayRef<int64_t>{0, 1, 0}));
  EXPECT_TRUE(llvm::equal(insertDescription->operands[0].strides,
                          llvm::ArrayRef<int64_t>{1, 1, 1}));

  auto collapse = findSingleOp<mlir::tensor::CollapseShapeOp>(*module);
  auto expand = findSingleOp<mlir::tensor::ExpandShapeOp>(*module);
  auto pad = findSingleOp<mlir::tensor::PadOp>(*module);
  auto cast = findSingleOp<mlir::tensor::CastOp>(*module);
  auto collapseDescription = describe(collapse);
  auto expandDescription = describe(expand);
  auto castDescription = describe(cast);
  ASSERT_TRUE(mlir::succeeded(collapseDescription));
  ASSERT_TRUE(mlir::succeeded(expandDescription));
  ASSERT_TRUE(mlir::succeeded(castDescription));
  EXPECT_EQ(collapseDescription->kind,
            wafer::TensorIndexingTransformKind::CollapseShape);
  EXPECT_EQ(expandDescription->kind,
            wafer::TensorIndexingTransformKind::ExpandShape);
  auto padDescription = describe(pad);
  ASSERT_TRUE(mlir::succeeded(padDescription));
  EXPECT_EQ(padDescription->kind, wafer::TensorIndexingTransformKind::Pad);
  EXPECT_TRUE(llvm::equal(padDescription->operands.front().offsets,
                          llvm::ArrayRef<int64_t>{0, 1, 0}));
  EXPECT_EQ(castDescription->kind, wafer::TensorIndexingTransformKind::Cast);

  llvm::SmallVector<mlir::tensor::ExtractSliceOp, 2> extracts;
  module->walk([&](mlir::tensor::ExtractSliceOp extract) {
    extracts.push_back(extract);
  });
  ASSERT_EQ(extracts.size(), 2u);
  unsigned exactDescriptions = 0;
  unsigned rejectedDynamicDescriptions = 0;
  for (mlir::tensor::ExtractSliceOp extract : extracts) {
    auto description = describe(extract);
    if (mlir::succeeded(description)) {
      ++exactDescriptions;
      EXPECT_EQ(description->kind,
                wafer::TensorIndexingTransformKind::ExtractSlice);
      EXPECT_TRUE(llvm::equal(description->operands.front().offsets,
                              llvm::ArrayRef<int64_t>{0, 1, 0}));
    } else {
      ++rejectedDynamicDescriptions;
    }
  }
  EXPECT_EQ(exactDescriptions, 1u);
  EXPECT_EQ(rejectedDynamicDescriptions, 1u);

  auto empty = findSingleOp<mlir::tensor::EmptyOp>(*module);
  EXPECT_FALSE(
      mlir::isa<wafer::WaferTensorIndexingOpInterface>(empty.getOperation()));
}

struct ExpectedExtractSlice {
  mlir::Value source;
  llvm::SmallVector<int64_t> offsets;
  llvm::SmallVector<int64_t> sizes;
  llvm::SmallVector<int64_t> resultShape;
};

ExpectedExtractSlice expectSlice(mlir::Value source,
                                 std::initializer_list<int64_t> offsets,
                                 std::initializer_list<int64_t> sizes,
                                 std::initializer_list<int64_t> resultShape) {
  return {source, llvm::SmallVector<int64_t>(offsets),
          llvm::SmallVector<int64_t>(sizes),
          llvm::SmallVector<int64_t>(resultShape)};
}

template <typename OpT> void expectLinalgExtCollectiveInterfaces(OpT op) {
  auto dps =
      mlir::dyn_cast<mlir::DestinationStyleOpInterface>(op.getOperation());
  ASSERT_TRUE(dps);
  ASSERT_EQ(dps.getNumDpsInputs(), static_cast<int64_t>(op.getInputs().size()));
  ASSERT_EQ(dps.getNumDpsInits(), static_cast<int64_t>(op.getOuts().size()));
  llvm::SmallVector<mlir::Value> dpsInputs = dps.getDpsInputs();
  ASSERT_EQ(dpsInputs.size(), op.getInputs().size());
  for (auto [index, input] : llvm::enumerate(op.getInputs()))
    EXPECT_EQ(dpsInputs[index], input);
  for (auto [index, out] : llvm::enumerate(op.getOuts()))
    EXPECT_EQ(dps.getDpsInits()[index], out);

  auto collective = mlir::dyn_cast<wafer::WaferLinalgExtCollectiveOpInterface>(
      op.getOperation());
  ASSERT_TRUE(collective);

  auto tiling = mlir::dyn_cast<mlir::TilingInterface>(op.getOperation());
  ASSERT_TRUE(tiling);
  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op.getResults().front().getType());
  mlir::OpBuilder builder(op);
  llvm::SmallVector<mlir::Range> domain = tiling.getIterationDomain(builder);
  ASSERT_EQ(domain.size(), static_cast<size_t>(resultType.getRank()));
  llvm::SmallVector<mlir::utils::IteratorType> iterators =
      tiling.getLoopIteratorTypes();
  ASSERT_EQ(iterators.size(), static_cast<size_t>(resultType.getRank()));
  for (int64_t dim = 0; dim < resultType.getRank(); ++dim) {
    EXPECT_EQ(iterators[dim], mlir::utils::IteratorType::parallel);
    EXPECT_EQ(mlir::getConstantIntValue(domain[dim].offset), 0);
    EXPECT_EQ(mlir::getConstantIntValue(domain[dim].stride), 1);
    if (!mlir::ShapedType::isDynamic(resultType.getDimSize(dim)))
      EXPECT_EQ(mlir::getConstantIntValue(domain[dim].size),
                resultType.getDimSize(dim));
  }
}

template <typename OpT>
void expectTiledImplementation(
    OpT op, mlir::MLIRContext &context, llvm::ArrayRef<int64_t> offsetValues,
    llvm::ArrayRef<int64_t> sizeValues, llvm::ArrayRef<int64_t> expectedShape,
    llvm::ArrayRef<ExpectedExtractSlice> expectedSlices = {}) {
  auto tiling = mlir::dyn_cast<mlir::TilingInterface>(op.getOperation());
  ASSERT_TRUE(tiling);
  mlir::OpBuilder builder(op);
  llvm::SmallVector<mlir::OpFoldResult> offsets =
      getIndexOpFoldResults(context, offsetValues);
  llvm::SmallVector<mlir::OpFoldResult> sizes =
      getIndexOpFoldResults(context, sizeValues);

  mlir::FailureOr<mlir::TilingResult> tiled =
      tiling.getTiledImplementation(builder, offsets, sizes);
  ASSERT_TRUE(mlir::succeeded(tiled));
  ASSERT_EQ(tiled->tiledOps.size(), 1u);
  EXPECT_TRUE(mlir::isa<OpT>(tiled->tiledOps.front()));
  ASSERT_EQ(tiled->tiledValues.size(), op.getResults().size());
  auto tiledType = mlir::dyn_cast<mlir::RankedTensorType>(
      tiled->tiledValues.front().getType());
  ASSERT_TRUE(tiledType);
  EXPECT_EQ(tiledType.getShape(), expectedShape);

  llvm::SmallVector<mlir::OpFoldResult> resultOffsets;
  llvm::SmallVector<mlir::OpFoldResult> resultSizes;
  EXPECT_TRUE(mlir::succeeded(tiling.getResultTilePosition(
      builder, 0, offsets, sizes, resultOffsets, resultSizes)));
  EXPECT_TRUE(hasConstantIntValues(resultOffsets, offsetValues));
  EXPECT_TRUE(hasConstantIntValues(resultSizes, sizeValues));

  if (!expectedSlices.empty()) {
    ASSERT_EQ(tiled->generatedSlices.size(), expectedSlices.size());
    for (auto [sliceOp, expected] :
         llvm::zip(tiled->generatedSlices, expectedSlices)) {
      auto slice = mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(sliceOp);
      ASSERT_TRUE(slice);
      EXPECT_EQ(slice.getSource(), expected.source);
      EXPECT_TRUE(
          hasConstantIntValues(slice.getMixedOffsets(), expected.offsets));
      EXPECT_TRUE(hasConstantIntValues(slice.getMixedSizes(), expected.sizes));
      llvm::SmallVector<int64_t> unitStrides(expected.offsets.size(), 1);
      EXPECT_TRUE(hasConstantIntValues(slice.getMixedStrides(), unitStrides));
      auto sliceType = mlir::dyn_cast<mlir::RankedTensorType>(slice.getType());
      ASSERT_TRUE(sliceType);
      EXPECT_EQ(sliceType.getShape(),
                llvm::ArrayRef<int64_t>(expected.resultShape));
    }
  }
}

template <typename OpT>
void expectTiledImplementationFailure(OpT op, mlir::MLIRContext &context,
                                      llvm::ArrayRef<int64_t> offsetValues,
                                      llvm::ArrayRef<int64_t> sizeValues) {
  auto tiling = mlir::dyn_cast<mlir::TilingInterface>(op.getOperation());
  ASSERT_TRUE(tiling);
  mlir::OpBuilder builder(op);
  llvm::SmallVector<mlir::OpFoldResult> offsets =
      getIndexOpFoldResults(context, offsetValues);
  llvm::SmallVector<mlir::OpFoldResult> sizes =
      getIndexOpFoldResults(context, sizeValues);
  EXPECT_TRUE(
      mlir::failed(tiling.getTiledImplementation(builder, offsets, sizes)));
}

TEST(WaferInterfacesTest, LayoutResourceAndMemoryEffectsAreQueryable) {
  mlir::DialectRegistry registry;
  wafer::registerWaferCoreDialects(registry);

  mlir::MLIRContext context(registry);
  context.loadDialect<wafer::WaferDialect, mlir::async::AsyncDialect,
                      mlir::memref::MemRefDialect>();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @default {card_grid = array<i64: 1, 1>,
      card_interconnect = "mesh", tile_grid = array<i64: 4, 4>,
      unavailable_tiles = array<i64>}
  %arg = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x4xf32, #wafer.memory<ddr, tensor>>
  %tile = memref.alloc() : memref<4x4xf32, #wafer.memory<spm, tensor>>
  wafer.tile.load %arg into %tile
      : memref<4x4xf32, #wafer.memory<ddr, tensor>>
    into memref<4x4xf32, #wafer.memory<spm, tensor>>
  %cx = wafer.tile.materialize_layout %tile
      : memref<4x4xf32, #wafer.memory<spm, tensor>>
     -> memref<4x4xf32, #wafer.memory<spm, cx>>
  %mm = wafer.tile.gemm %cx, %cx
      : (memref<4x4xf32, #wafer.memory<spm, cx>>,
         memref<4x4xf32, #wafer.memory<spm, cx>>)
     -> memref<4x4xf32, #wafer.memory<spm, cx>>
  %send = wafer.instr.dte_send %tile {peer = 0 : i64, bytes = 64 : i64,
      message = #wafer.dte_message<communication = 0, round = 0, slice = 0>}
      : memref<4x4xf32, #wafer.memory<spm, tensor>> -> !async.token
  wafer.instr.dte_wait %send : !async.token
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(module);

  auto load = findSingleOp<wafer::StorageLoadOp>(*module);
  ASSERT_TRUE(load);

  auto memoryEffects =
      mlir::dyn_cast<mlir::MemoryEffectOpInterface>(load.getOperation());
  ASSERT_TRUE(memoryEffects);
  llvm::SmallVector<
      mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>, 4>
      mlirEffects;
  memoryEffects.getEffects(mlirEffects);
  EXPECT_TRUE(
      (hasMemoryEffect<mlir::MemoryEffects::Read, wafer::WaferDDRResource>(
          mlirEffects)));
  EXPECT_TRUE(
      (hasMemoryEffect<mlir::MemoryEffects::Write, wafer::WaferSPMResource>(
          mlirEffects)));
  EXPECT_TRUE(hasValueMemoryEffect<mlir::MemoryEffects::Read>(
      mlirEffects, load.getSource()));
  EXPECT_TRUE(hasValueMemoryEffect<mlir::MemoryEffects::Write>(mlirEffects,
                                                               load.getDest()));

  auto materialize = findSingleOp<wafer::LayoutMaterializeOp>(*module);
  ASSERT_TRUE(materialize);
  EXPECT_EQ(getMemoryLayout(materialize.getSource()), wafer::MemLayout::Tensor);
  EXPECT_EQ(getMemoryLayout(materialize.getResult()), wafer::MemLayout::Cx);

  auto gemm = findSingleOp<wafer::ComputeGemmOp>(*module);
  ASSERT_TRUE(gemm);
  EXPECT_EQ(getMemoryLayout(gemm.getLhs()), wafer::MemLayout::Cx);
  EXPECT_EQ(getMemoryLayout(gemm.getRhs()), wafer::MemLayout::Cx);
  EXPECT_EQ(getMemoryLayout(gemm.getResult()), wafer::MemLayout::Cx);

  auto send = findSingleOp<wafer::InstrDTESendOp>(*module);
  ASSERT_TRUE(send);
  auto sendInstruction =
      mlir::dyn_cast<wafer::WaferInstructionOpInterface>(send.getOperation());
  ASSERT_TRUE(sendInstruction);
  EXPECT_EQ(sendInstruction.getInstructionFamily(), wafer::InstrFamily::DTE);
  mlirEffects.clear();
  mlir::cast<mlir::MemoryEffectOpInterface>(send.getOperation())
      .getEffects(mlirEffects);
  EXPECT_TRUE(hasValueMemoryEffect<mlir::MemoryEffects::Read>(
      mlirEffects, send.getBuffer()));
  EXPECT_TRUE(
      (hasMemoryEffect<mlir::MemoryEffects::Write,
                       wafer::WaferCommunicationResource>(mlirEffects)));

  auto wait = findSingleOp<wafer::InstrDTEWaitOp>(*module);
  ASSERT_TRUE(wait);
  auto waitInstruction =
      mlir::dyn_cast<wafer::WaferInstructionOpInterface>(wait.getOperation());
  ASSERT_TRUE(waitInstruction);
  EXPECT_EQ(waitInstruction.getInstructionFamily(), wafer::InstrFamily::DTE);
  mlirEffects.clear();
  mlir::cast<mlir::MemoryEffectOpInterface>(wait.getOperation())
      .getEffects(mlirEffects);
  EXPECT_TRUE(
      (hasMemoryEffect<mlir::MemoryEffects::Read,
                       wafer::WaferCommunicationResource>(mlirEffects)));
}

TEST(WaferInterfacesTest, InstructionInterfacesExposeFamilyAndEffects) {
  mlir::DialectRegistry registry;
  wafer::registerWaferCoreDialects(registry);
  registry.insert<mlir::arith::ArithDialect>();

  mlir::MLIRContext context(registry);
  context.loadDialect<wafer::WaferDialect, mlir::arith::ArithDialect>();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  %ddr_in = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf16, #wafer.memory<ddr, tensor>>
  %ddr_out = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf16, #wafer.memory<ddr, tensor>>
  %tensor = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf16, #wafer.memory<spm, tensor>>
  %converted = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf32, #wafer.memory<spm, tensor>>
  %cx = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf16, #wafer.memory<spm, cx>>
  %reduce_out = "builtin.unrealized_conversion_cast"()
      : () -> memref<4xf16, #wafer.memory<spm, cx>>
  %lhs = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf16, #wafer.memory<spm, cx>>
  %rhs = "builtin.unrealized_conversion_cast"()
      : () -> memref<8x16xf16, #wafer.memory<spm, cx>>
  %gemm_out = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x16xf16, #wafer.memory<spm, cx>>
  %f16 = arith.constant 0.000000e+00 : f16

  wafer.instr.rdma %ddr_in to %tensor
      {byte_count = 64 : i64, inner_bytes = 64 : i64,
       src_strides = array<i64: 0, 0, 0>,
       src_iterations = array<i64: 1, 1, 1>}
      : memref<4x8xf16, #wafer.memory<ddr, tensor>>
     to memref<4x8xf16, #wafer.memory<spm, tensor>>
  wafer.instr.gather_scatter %tensor to %cx
      {byte_count = 64 : i64, inner_bytes = 16 : i64,
       src_strides = array<i64: 16, 0, 0>,
       src_iterations = array<i64: 4, 1, 1>,
       dst_strides = array<i64: 16, 0, 0>,
       dst_iterations = array<i64: 4, 1, 1>}
      : memref<4x8xf16, #wafer.memory<spm, tensor>>
     to memref<4x8xf16, #wafer.memory<spm, cx>>
  wafer.instr.fill %tensor, %f16
      : memref<4x8xf16, #wafer.memory<spm, tensor>>, f16
  wafer.instr.mask_move %tensor, %tensor into %tensor
      {worker = #wafer.ncc_worker<worker2>}
      : memref<4x8xf16, #wafer.memory<spm, tensor>>,
        memref<4x8xf16, #wafer.memory<spm, tensor>>
    into memref<4x8xf16, #wafer.memory<spm, tensor>>
  wafer.instr.elementwise #wafer.instr_elementwise_kind<add> %tensor, %tensor into %tensor
      : memref<4x8xf16, #wafer.memory<spm, tensor>>,
        memref<4x8xf16, #wafer.memory<spm, tensor>>
    into memref<4x8xf16, #wafer.memory<spm, tensor>>
  wafer.instr.reduce #wafer.instr_reduce_kind<sum> %cx into %reduce_out
      {dim = 0 : i64}
      : memref<4x8xf16, #wafer.memory<spm, cx>>
    into memref<4xf16, #wafer.memory<spm, cx>>
  wafer.instr.convert #wafer.instr_convert_kind<fp16_fp32> %tensor into %converted
      : memref<4x8xf16, #wafer.memory<spm, tensor>>
     to memref<4x8xf32, #wafer.memory<spm, tensor>>
  wafer.instr.gemm %lhs, %rhs into %gemm_out
      {m = 4 : i64, k = 8 : i64, n = 16 : i64}
      : memref<4x8xf16, #wafer.memory<spm, cx>>,
        memref<8x16xf16, #wafer.memory<spm, cx>>
    into memref<4x16xf16, #wafer.memory<spm, cx>>
  wafer.instr.wdma %tensor to %ddr_out
      {byte_count = 64 : i64, inner_bytes = 64 : i64,
       dst_strides = array<i64: 0, 0, 0>,
       dst_iterations = array<i64: 1, 1, 1>}
      : memref<4x8xf16, #wafer.memory<spm, tensor>>
     to memref<4x8xf16, #wafer.memory<ddr, tensor>>
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(module);

  auto rdma = findSingleOp<wafer::InstrRDMAOp>(*module);
  ASSERT_TRUE(rdma);
  auto rdmaInstruction =
      mlir::dyn_cast<wafer::WaferInstructionOpInterface>(rdma.getOperation());
  ASSERT_TRUE(rdmaInstruction);
  EXPECT_EQ(rdmaInstruction.getInstructionFamily(), wafer::InstrFamily::RDMA);
  EXPECT_EQ(rdmaInstruction.getInstructionFamily(), wafer::InstrFamily::RDMA);

  auto rdmaEffects =
      mlir::cast<mlir::MemoryEffectOpInterface>(rdma.getOperation());
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 8> effects;
  rdmaEffects.getEffects(effects);
  EXPECT_TRUE(
      (hasMemoryEffect<mlir::MemoryEffects::Read, wafer::WaferDDRResource>(
          effects)));
  EXPECT_TRUE(
      (hasMemoryEffect<mlir::MemoryEffects::Write, wafer::WaferSPMResource>(
          effects)));
  EXPECT_TRUE((
      hasMemoryEffect<mlir::MemoryEffects::Write, wafer::WaferMovementResource>(
          effects)));
  EXPECT_TRUE(hasValueMemoryEffect<mlir::MemoryEffects::Read>(
      effects, rdma.getSource()));
  EXPECT_TRUE(hasValueMemoryEffect<mlir::MemoryEffects::Write>(effects,
                                                               rdma.getDest()));

  auto gather = findSingleOp<wafer::InstrGatherScatterOp>(*module);
  ASSERT_TRUE(gather);
  auto gatherInstruction =
      mlir::dyn_cast<wafer::WaferInstructionOpInterface>(gather.getOperation());
  ASSERT_TRUE(gatherInstruction);
  EXPECT_EQ(gatherInstruction.getInstructionFamily(), wafer::InstrFamily::TDMA);

  auto fill = findSingleOp<wafer::InstrFillOp>(*module);
  ASSERT_TRUE(fill);
  auto fillInstruction =
      mlir::dyn_cast<wafer::WaferInstructionOpInterface>(fill.getOperation());
  ASSERT_TRUE(fillInstruction);
  EXPECT_EQ(fillInstruction.getInstructionFamily(), wafer::InstrFamily::TDMA);
  EXPECT_EQ(wafer::getNCCOperationCompletion(fill).kind,
            wafer::NCCCompletionKind::OrderedAsynchronousIssue);
  auto fillIssue =
      mlir::dyn_cast<wafer::WaferNCCIssueOpInterface>(fill.getOperation());
  ASSERT_TRUE(fillIssue);
  EXPECT_EQ(fillIssue.getIssueWorker(), wafer::NCCWorker::Worker0);
  effects.clear();
  mlir::cast<mlir::MemoryEffectOpInterface>(fill.getOperation())
      .getEffects(effects);
  EXPECT_TRUE((
      hasMemoryEffect<mlir::MemoryEffects::Write, wafer::WaferMovementResource>(
          effects)));
  EXPECT_FALSE(
      (hasMemoryEffect<mlir::MemoryEffects::Write, wafer::WaferComputeResource>(
          effects)));
  EXPECT_TRUE(hasValueMemoryEffect<mlir::MemoryEffects::Write>(effects,
                                                               fill.getDest()));

  auto maskMove = findSingleOp<wafer::InstrMaskMoveOp>(*module);
  ASSERT_TRUE(maskMove);
  auto maskMoveInstruction = mlir::dyn_cast<wafer::WaferInstructionOpInterface>(
      maskMove.getOperation());
  ASSERT_TRUE(maskMoveInstruction);
  EXPECT_EQ(maskMoveInstruction.getInstructionFamily(), wafer::InstrFamily::CT);
  EXPECT_EQ(wafer::getNCCOperationCompletion(maskMove).kind,
            wafer::NCCCompletionKind::OrderedAsynchronousIssue);
  auto maskMoveIssue =
      mlir::dyn_cast<wafer::WaferNCCIssueOpInterface>(maskMove.getOperation());
  ASSERT_TRUE(maskMoveIssue);
  EXPECT_EQ(maskMoveIssue.getIssueWorker(), wafer::NCCWorker::Worker2);
  effects.clear();
  mlir::cast<mlir::MemoryEffectOpInterface>(maskMove.getOperation())
      .getEffects(effects);
  EXPECT_TRUE(
      (hasMemoryEffect<mlir::MemoryEffects::Write, wafer::WaferComputeResource>(
          effects)));
  EXPECT_FALSE((
      hasMemoryEffect<mlir::MemoryEffects::Write, wafer::WaferMovementResource>(
          effects)));
  EXPECT_TRUE(hasValueMemoryEffect<mlir::MemoryEffects::Read>(
      effects, maskMove.getSource()));
  EXPECT_TRUE(hasValueMemoryEffect<mlir::MemoryEffects::Read>(
      effects, maskMove.getMask()));
  EXPECT_TRUE(hasValueMemoryEffect<mlir::MemoryEffects::Write>(
      effects, maskMove.getDest()));

  auto elementwise = findSingleOp<wafer::InstrElementwiseOp>(*module);
  ASSERT_TRUE(elementwise);
  auto elementwiseInstruction =
      mlir::dyn_cast<wafer::WaferInstructionOpInterface>(
          elementwise.getOperation());
  ASSERT_TRUE(elementwiseInstruction);
  EXPECT_EQ(elementwiseInstruction.getInstructionFamily(),
            wafer::InstrFamily::CT);

  auto gemm = findSingleOp<wafer::InstrGemmOp>(*module);
  ASSERT_TRUE(gemm);
  auto gemmInstruction =
      mlir::dyn_cast<wafer::WaferInstructionOpInterface>(gemm.getOperation());
  ASSERT_TRUE(gemmInstruction);
  EXPECT_EQ(gemmInstruction.getInstructionFamily(), wafer::InstrFamily::NE);

  auto wdma = findSingleOp<wafer::InstrWDMAOp>(*module);
  ASSERT_TRUE(wdma);
  auto wdmaInstruction =
      mlir::dyn_cast<wafer::WaferInstructionOpInterface>(wdma.getOperation());
  ASSERT_TRUE(wdmaInstruction);
  EXPECT_EQ(wdmaInstruction.getInstructionFamily(), wafer::InstrFamily::WDMA);
}

TEST(WaferInterfacesTest, TypedNCCOperationCompletionsSeparateIssueAndJoin) {
  mlir::DialectRegistry registry;
  wafer::registerWaferCoreDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadDialect<wafer::WaferDialect>();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  %input = "builtin.unrealized_conversion_cast"()
      : () -> memref<4xf16, #wafer.memory<spm, tensor>>
  %value = "builtin.unrealized_conversion_cast"()
      : () -> memref<1xf16, #wafer.memory<spm, tensor>>
  %index = "builtin.unrealized_conversion_cast"()
      : () -> memref<1xi32, #wafer.memory<spm, tensor>>
  %resized = "builtin.unrealized_conversion_cast"()
      : () -> memref<4xf16, #wafer.memory<spm, tensor>>
  wafer.instr.peripheral #wafer.instr_peripheral_kind<argmax>
      %input into %value, %index {elem_count = 4 : i64}
      : memref<4xf16, #wafer.memory<spm, tensor>>
    into memref<1xf16, #wafer.memory<spm, tensor>>,
         memref<1xi32, #wafer.memory<spm, tensor>>
  wafer.instr.peripheral #wafer.instr_peripheral_kind<argmin>
      %input into %value, %index {elem_count = 4 : i64}
      : memref<4xf16, #wafer.memory<spm, tensor>>
    into memref<1xf16, #wafer.memory<spm, tensor>>,
         memref<1xi32, #wafer.memory<spm, tensor>>
  wafer.instr.peripheral #wafer.instr_peripheral_kind<bilinear>
      %input into %resized
      {elem_count = 4 : i64,
       source_shape = array<i64: 1, 1, 1, 4>,
       dest_shape = array<i64: 1, 1, 1, 4>}
      : memref<4xf16, #wafer.memory<spm, tensor>>
    into memref<4xf16, #wafer.memory<spm, tensor>>
  wafer.instr.ncc_join [0, 2]
  wafer.instr.ncc_join [0]
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  wafer::InstrPeripheralOp argmax;
  wafer::InstrPeripheralOp argmin;
  wafer::InstrPeripheralOp bilinear;
  module->walk([&](wafer::InstrPeripheralOp op) {
    switch (op.getKindAttr().getValue()) {
    case wafer::InstrPeripheralKind::ArgMax:
      argmax = op;
      break;
    case wafer::InstrPeripheralKind::ArgMin:
      argmin = op;
      break;
    case wafer::InstrPeripheralKind::Bilinear:
      bilinear = op;
      break;
    default:
      break;
    }
  });
  llvm::SmallVector<wafer::SyncNCCJoinOp, 2> joins;
  module->walk([&](wafer::SyncNCCJoinOp op) { joins.push_back(op); });
  ASSERT_TRUE(argmax);
  ASSERT_TRUE(argmin);
  ASSERT_TRUE(bilinear);
  ASSERT_EQ(joins.size(), 2u);
  wafer::SyncNCCJoinOp multiWorkerJoin = joins[0];
  wafer::SyncNCCJoinOp workerZeroJoin = joins[1];

  EXPECT_TRUE(
      mlir::isa<wafer::WaferNCCCompletionOpInterface>(argmax.getOperation()));
  EXPECT_TRUE(
      mlir::isa<wafer::WaferNCCCompletionOpInterface>(argmin.getOperation()));
  EXPECT_TRUE(
      mlir::isa<wafer::WaferNCCCompletionOpInterface>(bilinear.getOperation()));
  EXPECT_TRUE(mlir::isa<wafer::WaferNCCCompletionOpInterface>(
      multiWorkerJoin.getOperation()));

  EXPECT_EQ(wafer::getNCCOperationCompletion(argmax).kind,
            wafer::NCCCompletionKind::SynchronousWriteback);
  EXPECT_EQ(wafer::getNCCOperationCompletion(argmin).kind,
            wafer::NCCCompletionKind::SynchronousWriteback);
  EXPECT_EQ(wafer::getNCCOperationCompletion(bilinear).kind,
            wafer::NCCCompletionKind::OrderedAsynchronousIssue);
  EXPECT_EQ(wafer::getNCCOperationCompletion(multiWorkerJoin).kind,
            wafer::NCCCompletionKind::ParticipantJoin);
  EXPECT_EQ(wafer::getNCCOperationCompletion(workerZeroJoin).kind,
            wafer::NCCCompletionKind::ParticipantJoin);

  wafer::NCCOperationCompletion argmaxContract =
      wafer::getNCCOperationCompletion(argmax);
  ASSERT_TRUE(argmaxContract.issueWorker);
  EXPECT_EQ(*argmaxContract.issueWorker, wafer::NCCWorker::Worker0);
  EXPECT_EQ(argmaxContract.participantMask, uint32_t{1});

  wafer::NCCOperationCompletion joinContract =
      wafer::getNCCOperationCompletion(multiWorkerJoin);
  EXPECT_FALSE(joinContract.issueWorker);
  EXPECT_EQ(joinContract.participantMask,
            (uint32_t{1} << 0) | (uint32_t{1} << 2));

  wafer::NCCOperationCompletion workerZeroContract =
      wafer::getNCCOperationCompletion(workerZeroJoin);
  EXPECT_FALSE(workerZeroContract.issueWorker);
  EXPECT_EQ(workerZeroContract.participantMask, uint32_t{1});
}

TEST(WaferInterfacesTest, AttentionExposesStructuredAndCoupledContracts) {
  mlir::DialectRegistry registry;
  wafer::registerWaferCoreDialects(registry);
  registry.insert<mlir::func::FuncDialect, mlir::tensor::TensorDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
#q = affine_map<(b, m, k1, k2, n) -> (b, m, k1)>
#k = affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>
#v = affine_map<(b, m, k1, k2, n) -> (b, k2, n)>
#s = affine_map<(b, m, k1, k2, n) -> ()>
#mask = affine_map<(b, m, k1, k2, n) -> (b, m, k2)>
#o = affine_map<(b, m, k1, k2, n) -> (b, m, n)>
module {
  func.func @attention(
      %query: tensor<2x3x4xf16>, %key: tensor<2x5x4xf16>,
      %value: tensor<2x5x6xf16>, %scale: f32,
      %mask_value: tensor<2x3x5xf16>) -> tensor<2x3x6xf16> {
    %out = tensor.empty() : tensor<2x3x6xf16>
    %result = wafer.linalg_ext.attention
        ins(%query, %key, %value, %scale, %mask_value :
            tensor<2x3x4xf16>, tensor<2x5x4xf16>, tensor<2x5x6xf16>, f32,
            tensor<2x3x5xf16>)
        outs(%out : tensor<2x3x6xf16>)
        algorithm(<flash_attention>)
        indexing_maps = [#q, #k, #v, #s, #mask, #o]
        -> tensor<2x3x6xf16>
    return %result : tensor<2x3x6xf16>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  auto attention = findSingleOp<wafer::LinalgExtAttentionOp>(*module);
  ASSERT_TRUE(attention);
  auto dps = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(
      attention.getOperation());
  ASSERT_TRUE(dps);
  EXPECT_TRUE(dps.hasPureTensorSemantics());
  ASSERT_EQ(dps.getNumDpsInputs(), 5);
  ASSERT_EQ(dps.getNumDpsInits(), 1);
  EXPECT_EQ(dps.getDpsInits().front(), attention.getOutput());

  auto tiling = mlir::dyn_cast<mlir::TilingInterface>(attention.getOperation());
  ASSERT_TRUE(tiling);
  mlir::OpBuilder builder(attention);
  llvm::SmallVector<mlir::Range> domain = tiling.getIterationDomain(builder);
  ASSERT_EQ(domain.size(), 5u);
  EXPECT_TRUE(hasConstantIntValues(
      llvm::map_to_vector(domain, [](mlir::Range range) { return range.size; }),
      {2, 3, 4, 5, 6}));
  llvm::SmallVector<mlir::utils::IteratorType> iterators =
      tiling.getLoopIteratorTypes();
  ASSERT_EQ(iterators.size(), 5u);
  EXPECT_EQ(iterators[0], mlir::utils::IteratorType::parallel);
  EXPECT_EQ(iterators[1], mlir::utils::IteratorType::parallel);
  EXPECT_EQ(iterators[2], mlir::utils::IteratorType::reduction);
  EXPECT_EQ(iterators[3], mlir::utils::IteratorType::reduction);
  EXPECT_EQ(iterators[4], mlir::utils::IteratorType::parallel);

  auto coupled = mlir::dyn_cast<wafer::WaferCoupledReductionOpInterface>(
      attention.getOperation());
  ASSERT_TRUE(coupled);
  wafer::CoupledReductionDescription description =
      coupled.getCoupledReductionDescription();
  EXPECT_TRUE(
      llvm::equal(description.reductionIterators, llvm::ArrayRef<unsigned>{3}));
  ASSERT_EQ(description.components.size(), 3u);
  EXPECT_EQ(description.components[0].kind,
            wafer::CoupledReductionComponentKind::Maximum);
  EXPECT_EQ(description.components[1].kind,
            wafer::CoupledReductionComponentKind::Sum);
  EXPECT_EQ(description.components[2].kind,
            wafer::CoupledReductionComponentKind::Accumulator);
  EXPECT_EQ(description.components[0].elementType, builder.getF32Type());
  EXPECT_EQ(description.components[1].elementType, builder.getF32Type());
  EXPECT_EQ(description.components[2].elementType, builder.getF16Type());
  EXPECT_EQ(description.components[0].indexingMap,
            mlir::AffineMap::get(
                5, 0,
                {builder.getAffineDimExpr(0), builder.getAffineDimExpr(1)},
                &context));
  EXPECT_EQ(description.components[2].indexingMap, attention.getOutputMap());

  auto reify = mlir::cast<mlir::ReifyRankedShapedTypeOpInterface>(
      attention.getOperation());
  mlir::ReifiedRankedShapedTypeDims reifiedShapes;
  ASSERT_TRUE(mlir::succeeded(reify.reifyResultShapes(builder, reifiedShapes)));
  ASSERT_EQ(reifiedShapes.size(), 1u);
  EXPECT_TRUE(hasConstantIntValues(reifiedShapes.front(), {2, 3, 6}));

  llvm::SmallVector<mlir::OpFoldResult> offsets =
      getIndexOpFoldResults(context, {1, 1, 0, 0, 2});
  llvm::SmallVector<mlir::OpFoldResult> sizes =
      getIndexOpFoldResults(context, {1, 2, 4, 5, 3});
  mlir::FailureOr<mlir::TilingResult> tiled =
      tiling.getTiledImplementation(builder, offsets, sizes);
  ASSERT_TRUE(mlir::succeeded(tiled));
  ASSERT_EQ(tiled->tiledOps.size(), 1u);
  auto tiledAttention =
      mlir::dyn_cast<wafer::LinalgExtAttentionOp>(tiled->tiledOps.front());
  ASSERT_TRUE(tiledAttention);
  EXPECT_EQ(mlir::cast<mlir::ShapedType>(tiledAttention.getQuery().getType())
                .getShape(),
            llvm::ArrayRef<int64_t>({1, 2, 4}));
  EXPECT_EQ(mlir::cast<mlir::ShapedType>(tiledAttention.getKey().getType())
                .getShape(),
            llvm::ArrayRef<int64_t>({1, 5, 4}));
  EXPECT_EQ(mlir::cast<mlir::ShapedType>(tiledAttention.getValue().getType())
                .getShape(),
            llvm::ArrayRef<int64_t>({1, 5, 3}));
  EXPECT_EQ(mlir::cast<mlir::ShapedType>(tiledAttention.getResult(0).getType())
                .getShape(),
            llvm::ArrayRef<int64_t>({1, 2, 3}));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(tiledAttention)));

  sizes[3] = builder.getIndexAttr(2);
  EXPECT_TRUE(
      mlir::failed(tiling.getTiledImplementation(builder, offsets, sizes)));

  llvm::SmallVector<mlir::OpFoldResult> resultOffsets =
      getIndexOpFoldResults(context, {0, 1, 2});
  llvm::SmallVector<mlir::OpFoldResult> resultSizes =
      getIndexOpFoldResults(context, {2, 2, 3});
  mlir::FailureOr<mlir::TilingResult> generated =
      tiling.generateResultTileValue(builder, 0, resultOffsets, resultSizes);
  ASSERT_TRUE(mlir::succeeded(generated));
  ASSERT_EQ(generated->tiledValues.size(), 1u);
  EXPECT_EQ(mlir::cast<mlir::RankedTensorType>(
                generated->tiledValues.front().getType())
                .getShape(),
            llvm::ArrayRef<int64_t>({2, 2, 3}));

  llvm::SmallVector<mlir::MemoryEffects::EffectInstance> effects;
  mlir::cast<mlir::MemoryEffectOpInterface>(attention.getOperation())
      .getEffects(effects);
  EXPECT_TRUE(effects.empty());
}

TEST(WaferInterfacesTest, AttentionBufferEffectsNameExactOperands) {
  mlir::DialectRegistry registry;
  wafer::registerWaferCoreDialects(registry);
  registry.insert<mlir::func::FuncDialect, mlir::memref::MemRefDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
#q = affine_map<(b, m, k1, k2, n) -> (b, m, k1)>
#k = affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>
#v = affine_map<(b, m, k1, k2, n) -> (b, k2, n)>
#s = affine_map<(b, m, k1, k2, n) -> ()>
#o = affine_map<(b, m, k1, k2, n) -> (b, m, n)>
module {
  func.func @attention(
      %query: memref<2x3x4xf16>, %key: memref<2x5x4xf16>,
      %value: memref<2x5x6xf16>, %scale: f16,
      %out: memref<2x3x6xf16>) {
    wafer.linalg_ext.attention
        ins(%query, %key, %value, %scale :
            memref<2x3x4xf16>, memref<2x5x4xf16>, memref<2x5x6xf16>, f16)
        outs(%out : memref<2x3x6xf16>)
        algorithm(<flash_attention>)
        indexing_maps = [#q, #k, #v, #s, #o]
    return
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  auto attention = findSingleOp<wafer::LinalgExtAttentionOp>(*module);
  ASSERT_TRUE(attention);

  llvm::SmallVector<mlir::MemoryEffects::EffectInstance> effects;
  mlir::cast<mlir::MemoryEffectOpInterface>(attention.getOperation())
      .getEffects(effects);
  EXPECT_TRUE(hasValueMemoryEffect<mlir::MemoryEffects::Read>(
      effects, attention.getQuery()));
  EXPECT_TRUE(hasValueMemoryEffect<mlir::MemoryEffects::Read>(
      effects, attention.getKey()));
  EXPECT_TRUE(hasValueMemoryEffect<mlir::MemoryEffects::Read>(
      effects, attention.getValue()));
  EXPECT_TRUE(hasValueMemoryEffect<mlir::MemoryEffects::Write>(
      effects, attention.getOutput()));
  EXPECT_FALSE(hasValueMemoryEffect<mlir::MemoryEffects::Read>(
      effects, attention.getOutput()));

  auto tiling = mlir::cast<mlir::TilingInterface>(attention.getOperation());
  mlir::OpBuilder builder(attention);
  llvm::SmallVector<mlir::OpFoldResult> offsets =
      getIndexOpFoldResults(context, {0, 1, 0, 0, 2});
  llvm::SmallVector<mlir::OpFoldResult> sizes =
      getIndexOpFoldResults(context, {2, 2, 4, 5, 3});
  mlir::FailureOr<mlir::TilingResult> tiled =
      tiling.getTiledImplementation(builder, offsets, sizes);
  ASSERT_TRUE(mlir::succeeded(tiled));
  ASSERT_EQ(tiled->tiledOps.size(), 1u);
  auto tiledAttention =
      mlir::dyn_cast<wafer::LinalgExtAttentionOp>(tiled->tiledOps.front());
  ASSERT_TRUE(tiledAttention);
  EXPECT_EQ(tiledAttention.getNumResults(), 0u);
  EXPECT_TRUE(
      mlir::isa<mlir::MemRefType>(tiledAttention.getOutput().getType()));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(tiledAttention)));
}

TEST(WaferInterfacesTest, LinalgExtCollectivesExposeLinalgExtStyleContracts) {
  mlir::DialectRegistry registry;
  wafer::registerWaferCoreDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::tensor::TensorDialect>();

  mlir::MLIRContext context(registry);
  context.loadDialect<wafer::WaferDialect, mlir::arith::ArithDialect,
                      mlir::tensor::TensorDialect>();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  %input = "builtin.unrealized_conversion_cast"() : () -> tensor<4xf32>
  %out = "builtin.unrealized_conversion_cast"() : () -> tensor<4xf32>
  %reduced = wafer.linalg_ext.collective.all_reduce
      ins(%input : tensor<4xf32>)
      outs(%out : tensor<4xf32>)
      {
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      wafer.linalg_ext.collective.yield %sum : f32
      } {channel_id = 7 : i64, partition_group = array<i64: 0, 1>}
      -> tensor<4xf32>

  %gathered_out = "builtin.unrealized_conversion_cast"() : () -> tensor<8xf32>
  %gathered = wafer.linalg_ext.collective.all_gather
      ins(%input : tensor<4xf32>)
      outs(%gathered_out : tensor<8xf32>)
      {axis = 0 : i64, channel_id = 9 : i64, partition_group = array<i64: 0, 1>}
      -> tensor<8xf32>

  %wide = "builtin.unrealized_conversion_cast"() : () -> tensor<8xf32>
  %scattered = wafer.linalg_ext.collective.reduce_scatter
      ins(%wide : tensor<8xf32>)
      outs(%out : tensor<4xf32>)
      {
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      wafer.linalg_ext.collective.yield %sum : f32
      } {axis = 0 : i64, channel_id = 10 : i64,
         partition_group = array<i64: 0, 1>}
      -> tensor<4xf32>

  %matrix = "builtin.unrealized_conversion_cast"() : () -> tensor<4x2xf32>
  %matrix_out = "builtin.unrealized_conversion_cast"() : () -> tensor<2x4xf32>
  %a2a = wafer.linalg_ext.collective.all_to_all
      ins(%matrix : tensor<4x2xf32>)
      outs(%matrix_out : tensor<2x4xf32>)
      {split_axis = 0 : i64, concat_axis = 1 : i64,
       split_count = 2 : i64, channel_id = 11 : i64,
       partition_group = array<i64: 0, 1>}
      -> tensor<2x4xf32>

  %permuted = wafer.linalg_ext.collective.collective_permute
      ins(%input : tensor<4xf32>)
      outs(%out : tensor<4xf32>)
      {channel_id = 12 : i64, source_target_pairs = array<i64: 0, 1, 1, 0>}
      -> tensor<4xf32>
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(module);

  auto allReduce = findSingleOp<wafer::LinalgExtCollectiveAllReduceOp>(*module);
  ASSERT_TRUE(allReduce);
  expectLinalgExtCollectiveInterfaces(allReduce);
  auto collective = mlir::dyn_cast<wafer::WaferLinalgExtCollectiveOpInterface>(
      allReduce.getOperation());
  ASSERT_TRUE(collective);
  EXPECT_EQ(collective.getCollectiveKind(),
            wafer::WaferLinalgExtCollectiveKind::AllReduce);
  EXPECT_EQ(allReduce.getChannelId(), 7);
  EXPECT_TRUE(llvm::equal(allReduce.getPartitionGroupAttr().asArrayRef(),
                          llvm::ArrayRef<int64_t>({0, 1})));
  EXPECT_FALSE(allReduce.getCombiner().empty());

  auto tiling = mlir::dyn_cast<mlir::TilingInterface>(allReduce.getOperation());
  ASSERT_TRUE(tiling);
  mlir::OpBuilder builder(allReduce);
  llvm::SmallVector<mlir::Range> domain = tiling.getIterationDomain(builder);
  ASSERT_EQ(domain.size(), 1u);
  EXPECT_EQ(mlir::getConstantIntValue(domain[0].offset), 0);
  EXPECT_EQ(mlir::getConstantIntValue(domain[0].size), 4);
  EXPECT_EQ(mlir::getConstantIntValue(domain[0].stride), 1);
  llvm::SmallVector<mlir::utils::IteratorType> iterators =
      tiling.getLoopIteratorTypes();
  ASSERT_EQ(iterators.size(), 1u);
  EXPECT_EQ(iterators[0], mlir::utils::IteratorType::parallel);
  expectTiledImplementation(
      allReduce, context, llvm::SmallVector<int64_t>{1},
      llvm::SmallVector<int64_t>{2}, llvm::SmallVector<int64_t>{2},
      llvm::SmallVector<ExpectedExtractSlice>{
          expectSlice(allReduce.getInputs()[0], {1}, {2}, {2}),
          expectSlice(allReduce.getOuts()[0], {1}, {2}, {2})});

  auto allGather = findSingleOp<wafer::LinalgExtCollectiveAllGatherOp>(*module);
  ASSERT_TRUE(allGather);
  expectLinalgExtCollectiveInterfaces(allGather);
  auto gatherCollective =
      mlir::dyn_cast<wafer::WaferLinalgExtCollectiveOpInterface>(
          allGather.getOperation());
  ASSERT_TRUE(gatherCollective);
  EXPECT_EQ(gatherCollective.getCollectiveKind(),
            wafer::WaferLinalgExtCollectiveKind::AllGather);
  EXPECT_EQ(allGather.getAxis(), 0);
  EXPECT_EQ(allGather.getChannelId(), 9);
  EXPECT_TRUE(llvm::equal(allGather.getPartitionGroupAttr().asArrayRef(),
                          llvm::ArrayRef<int64_t>({0, 1})));
  expectTiledImplementation(
      allGather, context, llvm::SmallVector<int64_t>{0},
      llvm::SmallVector<int64_t>{8}, llvm::SmallVector<int64_t>{8},
      llvm::SmallVector<ExpectedExtractSlice>{
          expectSlice(allGather.getInputs()[0], {0}, {4}, {4}),
          expectSlice(allGather.getOuts()[0], {0}, {8}, {8})});

  auto gatherTiling =
      mlir::dyn_cast<mlir::TilingInterface>(allGather.getOperation());
  ASSERT_TRUE(gatherTiling);
  llvm::SmallVector<int64_t> crossingOffsetValues{3};
  llvm::SmallVector<int64_t> crossingSizeValues{2};
  llvm::SmallVector<mlir::OpFoldResult> crossingOffsets =
      mlir::getAsIndexOpFoldResult(&context, crossingOffsetValues);
  llvm::SmallVector<mlir::OpFoldResult> crossingSizes =
      mlir::getAsIndexOpFoldResult(&context, crossingSizeValues);
  EXPECT_TRUE(mlir::failed(gatherTiling.getTiledImplementation(
      builder, crossingOffsets, crossingSizes)));

  auto reduceScatter =
      findSingleOp<wafer::LinalgExtCollectiveReduceScatterOp>(*module);
  ASSERT_TRUE(reduceScatter);
  expectLinalgExtCollectiveInterfaces(reduceScatter);
  auto reduceScatterCollective =
      mlir::dyn_cast<wafer::WaferLinalgExtCollectiveOpInterface>(
          reduceScatter.getOperation());
  ASSERT_TRUE(reduceScatterCollective);
  EXPECT_EQ(reduceScatterCollective.getCollectiveKind(),
            wafer::WaferLinalgExtCollectiveKind::ReduceScatter);
  EXPECT_EQ(reduceScatter.getAxis(), 0);
  EXPECT_EQ(reduceScatter.getChannelId(), 10);
  EXPECT_TRUE(llvm::equal(reduceScatter.getPartitionGroupAttr().asArrayRef(),
                          llvm::ArrayRef<int64_t>({0, 1})));
  EXPECT_FALSE(reduceScatter.getCombiner().empty());
  EXPECT_TRUE(mlir::isa<mlir::TilingInterface>(reduceScatter.getOperation()));
  expectTiledImplementation(
      reduceScatter, context, llvm::SmallVector<int64_t>{0},
      llvm::SmallVector<int64_t>{4}, llvm::SmallVector<int64_t>{4},
      llvm::SmallVector<ExpectedExtractSlice>{
          expectSlice(reduceScatter.getInputs()[0], {0}, {8}, {8}),
          expectSlice(reduceScatter.getOuts()[0], {0}, {4}, {4})});
  expectTiledImplementationFailure(reduceScatter, context,
                                   llvm::SmallVector<int64_t>{1},
                                   llvm::SmallVector<int64_t>{2});

  auto allToAll = findSingleOp<wafer::LinalgExtCollectiveAllToAllOp>(*module);
  ASSERT_TRUE(allToAll);
  expectLinalgExtCollectiveInterfaces(allToAll);
  auto allToAllCollective =
      mlir::dyn_cast<wafer::WaferLinalgExtCollectiveOpInterface>(
          allToAll.getOperation());
  ASSERT_TRUE(allToAllCollective);
  EXPECT_EQ(allToAllCollective.getCollectiveKind(),
            wafer::WaferLinalgExtCollectiveKind::AllToAll);
  EXPECT_EQ(allToAll.getSplitAxis(), 0);
  EXPECT_EQ(allToAll.getConcatAxis(), 1);
  EXPECT_EQ(allToAll.getSplitCount(), 2);
  EXPECT_EQ(allToAll.getChannelId(), 11);
  EXPECT_TRUE(llvm::equal(allToAll.getPartitionGroupAttr().asArrayRef(),
                          llvm::ArrayRef<int64_t>({0, 1})));
  EXPECT_TRUE(mlir::isa<mlir::TilingInterface>(allToAll.getOperation()));
  expectTiledImplementation(
      allToAll, context, llvm::SmallVector<int64_t>{0, 0},
      llvm::SmallVector<int64_t>{2, 4}, llvm::SmallVector<int64_t>{2, 4},
      llvm::SmallVector<ExpectedExtractSlice>{
          expectSlice(allToAll.getInputs()[0], {0, 0}, {4, 2}, {4, 2}),
          expectSlice(allToAll.getOuts()[0], {0, 0}, {2, 4}, {2, 4})});
  expectTiledImplementationFailure(allToAll, context,
                                   llvm::SmallVector<int64_t>{0, 1},
                                   llvm::SmallVector<int64_t>{2, 2});

  auto permute =
      findSingleOp<wafer::LinalgExtCollectiveCollectivePermuteOp>(*module);
  ASSERT_TRUE(permute);
  expectLinalgExtCollectiveInterfaces(permute);
  auto permuteCollective =
      mlir::dyn_cast<wafer::WaferLinalgExtCollectiveOpInterface>(
          permute.getOperation());
  ASSERT_TRUE(permuteCollective);
  EXPECT_EQ(permuteCollective.getCollectiveKind(),
            wafer::WaferLinalgExtCollectiveKind::CollectivePermute);
  EXPECT_EQ(permute.getChannelId(), 12);
  EXPECT_TRUE(llvm::equal(permute.getSourceTargetPairsAttr().asArrayRef(),
                          llvm::ArrayRef<int64_t>({0, 1, 1, 0})));
  EXPECT_TRUE(mlir::isa<mlir::TilingInterface>(permute.getOperation()));
  expectTiledImplementation(
      permute, context, llvm::SmallVector<int64_t>{1},
      llvm::SmallVector<int64_t>{2}, llvm::SmallVector<int64_t>{2},
      llvm::SmallVector<ExpectedExtractSlice>{
          expectSlice(permute.getInputs()[0], {1}, {2}, {2}),
          expectSlice(permute.getOuts()[0], {1}, {2}, {2})});
}

TEST(WaferInterfacesTest, LinalgExtCollectiveTilingHandlesNonTrivialShapes) {
  mlir::DialectRegistry registry;
  wafer::registerWaferCoreDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::tensor::TensorDialect>();

  mlir::MLIRContext context(registry);
  context.loadDialect<wafer::WaferDialect, mlir::arith::ArithDialect,
                      mlir::tensor::TensorDialect>();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  %ar_input = "builtin.unrealized_conversion_cast"() : () -> tensor<1024x4096xf32>
  %ar_out = "builtin.unrealized_conversion_cast"() : () -> tensor<1024x4096xf32>
  %ar = wafer.linalg_ext.collective.all_reduce
      ins(%ar_input : tensor<1024x4096xf32>)
      outs(%ar_out : tensor<1024x4096xf32>)
      {
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      wafer.linalg_ext.collective.yield %sum : f32
      } {channel_id = 17 : i64, partition_group = array<i64: 0, 1, 2, 3, 4, 5, 6, 7>}
      -> tensor<1024x4096xf32>

  %ag_input = "builtin.unrealized_conversion_cast"() : () -> tensor<64x512xf32>
  %ag_out = "builtin.unrealized_conversion_cast"() : () -> tensor<64x4096xf32>
  %ag = wafer.linalg_ext.collective.all_gather
      ins(%ag_input : tensor<64x512xf32>)
      outs(%ag_out : tensor<64x4096xf32>)
      {axis = 1 : i64, channel_id = 18 : i64,
       partition_group = array<i64: 0, 1, 2, 3, 4, 5, 6, 7>}
      -> tensor<64x4096xf32>

  %rs_input = "builtin.unrealized_conversion_cast"() : () -> tensor<256x4096xf32>
  %rs_out = "builtin.unrealized_conversion_cast"() : () -> tensor<256x1024xf32>
  %rs = wafer.linalg_ext.collective.reduce_scatter
      ins(%rs_input : tensor<256x4096xf32>)
      outs(%rs_out : tensor<256x1024xf32>)
      {
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      wafer.linalg_ext.collective.yield %sum : f32
      } {axis = 1 : i64, channel_id = 19 : i64,
         partition_group = array<i64: 0, 1, 2, 3>}
      -> tensor<256x1024xf32>

  %a2a_input = "builtin.unrealized_conversion_cast"() : () -> tensor<512x128x64xf32>
  %a2a_out = "builtin.unrealized_conversion_cast"() : () -> tensor<128x512x64xf32>
  %a2a = wafer.linalg_ext.collective.all_to_all
      ins(%a2a_input : tensor<512x128x64xf32>)
      outs(%a2a_out : tensor<128x512x64xf32>)
      {split_axis = 0 : i64, concat_axis = 1 : i64,
       split_count = 4 : i64, channel_id = 20 : i64,
       partition_group = array<i64: 0, 1, 2, 3>}
      -> tensor<128x512x64xf32>

  %cp_input = "builtin.unrealized_conversion_cast"() : () -> tensor<4x1024x4096xf32>
  %cp_out = "builtin.unrealized_conversion_cast"() : () -> tensor<4x1024x4096xf32>
  %cp = wafer.linalg_ext.collective.collective_permute
      ins(%cp_input : tensor<4x1024x4096xf32>)
      outs(%cp_out : tensor<4x1024x4096xf32>)
      {channel_id = 21 : i64,
       source_target_pairs = array<i64: 0, 1, 1, 2, 2, 3, 3, 0>}
      -> tensor<4x1024x4096xf32>
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(module);

  auto allReduce = findSingleOp<wafer::LinalgExtCollectiveAllReduceOp>(*module);
  ASSERT_TRUE(allReduce);
  expectLinalgExtCollectiveInterfaces(allReduce);
  expectTiledImplementation(
      allReduce, context, llvm::SmallVector<int64_t>{128, 256},
      llvm::SmallVector<int64_t>{64, 512}, llvm::SmallVector<int64_t>{64, 512},
      llvm::SmallVector<ExpectedExtractSlice>{
          expectSlice(allReduce.getInputs()[0], {128, 256}, {64, 512},
                      {64, 512}),
          expectSlice(allReduce.getOuts()[0], {128, 256}, {64, 512},
                      {64, 512})});

  auto allGather = findSingleOp<wafer::LinalgExtCollectiveAllGatherOp>(*module);
  ASSERT_TRUE(allGather);
  expectLinalgExtCollectiveInterfaces(allGather);
  expectTiledImplementation(
      allGather, context, llvm::SmallVector<int64_t>{16, 0},
      llvm::SmallVector<int64_t>{8, 4096}, llvm::SmallVector<int64_t>{8, 4096},
      llvm::SmallVector<ExpectedExtractSlice>{
          expectSlice(allGather.getInputs()[0], {16, 0}, {8, 512}, {8, 512}),
          expectSlice(allGather.getOuts()[0], {16, 0}, {8, 4096}, {8, 4096})});
  expectTiledImplementationFailure(allGather, context,
                                   llvm::SmallVector<int64_t>{16, 512},
                                   llvm::SmallVector<int64_t>{8, 1024});

  auto reduceScatter =
      findSingleOp<wafer::LinalgExtCollectiveReduceScatterOp>(*module);
  ASSERT_TRUE(reduceScatter);
  expectLinalgExtCollectiveInterfaces(reduceScatter);
  expectTiledImplementation(reduceScatter, context,
                            llvm::SmallVector<int64_t>{32, 0},
                            llvm::SmallVector<int64_t>{16, 1024},
                            llvm::SmallVector<int64_t>{16, 1024},
                            llvm::SmallVector<ExpectedExtractSlice>{
                                expectSlice(reduceScatter.getInputs()[0],
                                            {32, 0}, {16, 4096}, {16, 4096}),
                                expectSlice(reduceScatter.getOuts()[0], {32, 0},
                                            {16, 1024}, {16, 1024})});
  expectTiledImplementationFailure(reduceScatter, context,
                                   llvm::SmallVector<int64_t>{32, 128},
                                   llvm::SmallVector<int64_t>{16, 512});

  auto allToAll = findSingleOp<wafer::LinalgExtCollectiveAllToAllOp>(*module);
  ASSERT_TRUE(allToAll);
  expectLinalgExtCollectiveInterfaces(allToAll);
  expectTiledImplementation(allToAll, context,
                            llvm::SmallVector<int64_t>{0, 0, 16},
                            llvm::SmallVector<int64_t>{128, 512, 8},
                            llvm::SmallVector<int64_t>{128, 512, 8},
                            llvm::SmallVector<ExpectedExtractSlice>{
                                expectSlice(allToAll.getInputs()[0], {0, 0, 16},
                                            {512, 128, 8}, {512, 128, 8}),
                                expectSlice(allToAll.getOuts()[0], {0, 0, 16},
                                            {128, 512, 8}, {128, 512, 8})});
  expectTiledImplementationFailure(allToAll, context,
                                   llvm::SmallVector<int64_t>{0, 64, 16},
                                   llvm::SmallVector<int64_t>{128, 128, 8});

  auto permute =
      findSingleOp<wafer::LinalgExtCollectiveCollectivePermuteOp>(*module);
  ASSERT_TRUE(permute);
  expectLinalgExtCollectiveInterfaces(permute);
  expectTiledImplementation(
      permute, context, llvm::SmallVector<int64_t>{1, 128, 256},
      llvm::SmallVector<int64_t>{2, 64, 512},
      llvm::SmallVector<int64_t>{2, 64, 512},
      llvm::SmallVector<ExpectedExtractSlice>{
          expectSlice(permute.getInputs()[0], {1, 128, 256}, {2, 64, 512},
                      {2, 64, 512}),
          expectSlice(permute.getOuts()[0], {1, 128, 256}, {2, 64, 512},
                      {2, 64, 512})});
}

TEST(WaferInterfacesTest,
     LinalgExtCollectiveTilingHandlesDynamicNonAxisShapes) {
  mlir::DialectRegistry registry;
  wafer::registerWaferCoreDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::tensor::TensorDialect>();

  mlir::MLIRContext context(registry);
  context.loadDialect<wafer::WaferDialect, mlir::arith::ArithDialect,
                      mlir::tensor::TensorDialect>();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  %ar_input = "builtin.unrealized_conversion_cast"() : () -> tensor<?x4096xf32>
  %ar_out = "builtin.unrealized_conversion_cast"() : () -> tensor<?x4096xf32>
  %ar = wafer.linalg_ext.collective.all_reduce
      ins(%ar_input : tensor<?x4096xf32>)
      outs(%ar_out : tensor<?x4096xf32>)
      {
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      wafer.linalg_ext.collective.yield %sum : f32
      } {channel_id = 22 : i64, partition_group = array<i64: 0, 1, 2, 3>}
      -> tensor<?x4096xf32>

  %ag_input = "builtin.unrealized_conversion_cast"() : () -> tensor<?x512xf32>
  %ag_out = "builtin.unrealized_conversion_cast"() : () -> tensor<?x2048xf32>
  %ag = wafer.linalg_ext.collective.all_gather
      ins(%ag_input : tensor<?x512xf32>)
      outs(%ag_out : tensor<?x2048xf32>)
      {axis = 1 : i64, channel_id = 23 : i64,
       partition_group = array<i64: 0, 1, 2, 3>}
      -> tensor<?x2048xf32>
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(module);

  auto allReduce = findSingleOp<wafer::LinalgExtCollectiveAllReduceOp>(*module);
  ASSERT_TRUE(allReduce);
  expectLinalgExtCollectiveInterfaces(allReduce);
  expectTiledImplementation(
      allReduce, context, llvm::SmallVector<int64_t>{0, 1024},
      llvm::SmallVector<int64_t>{8, 512}, llvm::SmallVector<int64_t>{8, 512},
      llvm::SmallVector<ExpectedExtractSlice>{
          expectSlice(allReduce.getInputs()[0], {0, 1024}, {8, 512}, {8, 512}),
          expectSlice(allReduce.getOuts()[0], {0, 1024}, {8, 512}, {8, 512})});

  auto allGather = findSingleOp<wafer::LinalgExtCollectiveAllGatherOp>(*module);
  ASSERT_TRUE(allGather);
  expectLinalgExtCollectiveInterfaces(allGather);
  expectTiledImplementation(
      allGather, context, llvm::SmallVector<int64_t>{0, 0},
      llvm::SmallVector<int64_t>{8, 2048}, llvm::SmallVector<int64_t>{8, 2048},
      llvm::SmallVector<ExpectedExtractSlice>{
          expectSlice(allGather.getInputs()[0], {0, 0}, {8, 512}, {8, 512}),
          expectSlice(allGather.getOuts()[0], {0, 0}, {8, 2048}, {8, 2048})});
}

TEST(WaferInterfacesTest, StablehloPipelineProducedCollectiveCanBeTiled) {
#ifndef WAFER_ENABLE_STABLEHLO
  GTEST_SKIP() << "StableHLO importer dependencies are disabled";
#else
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect, mlir::math::MathDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
  wafer::registerWaferCoreDialects(registry);
  wafer::registerImporterDialects(registry);

  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @partitioned_all_gather(%input: tensor<64x512xf32>) -> tensor<64x4096xf32> {
    %0 = "stablehlo.all_gather"(%input) {
      all_gather_dim = 1 : i64,
      replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 41, type = 1>
    } : (tensor<64x512xf32>) -> tensor<64x4096xf32>
    return %0 : tensor<64x4096xf32>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context);
  wafer::buildStablehloToLinalgPipeline(pm);
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  auto allGather = findSingleOp<wafer::LinalgExtCollectiveAllGatherOp>(*module);
  ASSERT_TRUE(allGather);
  expectLinalgExtCollectiveInterfaces(allGather);
  auto collective = mlir::dyn_cast<wafer::WaferLinalgExtCollectiveOpInterface>(
      allGather.getOperation());
  ASSERT_TRUE(collective);
  EXPECT_EQ(collective.getCollectiveKind(),
            wafer::WaferLinalgExtCollectiveKind::AllGather);
  EXPECT_EQ(allGather.getAxis(), 1);
  EXPECT_EQ(allGather.getChannelId(), 41);
  EXPECT_TRUE(llvm::equal(allGather.getPartitionGroupAttr().asArrayRef(),
                          llvm::ArrayRef<int64_t>({0, 1, 2, 3, 4, 5, 6, 7})));
  expectTiledImplementation(
      allGather, context, llvm::SmallVector<int64_t>{16, 0},
      llvm::SmallVector<int64_t>{8, 4096}, llvm::SmallVector<int64_t>{8, 4096},
      llvm::SmallVector<ExpectedExtractSlice>{
          expectSlice(allGather.getInputs()[0], {16, 0}, {8, 512}, {8, 512}),
          expectSlice(allGather.getOuts()[0], {16, 0}, {8, 4096}, {8, 4096})});
#endif
}

} // namespace
