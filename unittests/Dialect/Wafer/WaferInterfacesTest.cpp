#include "Wafer/Frontend/InitImporterDialects.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitAll.h"
#include "Wafer/Pipelines/Pipelines.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/STLExtras.h"

#include "gtest/gtest.h"

#include <initializer_list>
#include <optional>

namespace {

TEST(WaferInterfacesTest, InterfaceClassesAreGenerated) {
  SUCCEED() << "Wafer interface headers compile";
}

template <typename OpT> OpT findSingleOp(mlir::ModuleOp module) {
  OpT found;
  module.walk([&](OpT op) {
    EXPECT_FALSE(found);
    found = op;
  });
  return found;
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

bool hasLayoutRequirement(llvm::ArrayRef<wafer::WaferLayoutRequirement> reqs,
                          wafer::WaferValueRole role, unsigned index,
                          wafer::MemLayout layout,
                          wafer::MemorySpace memorySpace) {
  return llvm::any_of(reqs, [&](const wafer::WaferLayoutRequirement &req) {
    return req.role == role && req.index == index && req.layout == layout &&
           req.memorySpace == memorySpace;
  });
}

bool hasResourceEffect(llvm::ArrayRef<wafer::WaferResourceEffect> effects,
                       wafer::WaferResourceKind resource,
                       wafer::WaferResourceAccess access,
                       wafer::WaferValueRole role, unsigned index,
                       int64_t bytes) {
  return llvm::any_of(effects, [&](const wafer::WaferResourceEffect &effect) {
    return effect.resource == resource && effect.access == access &&
           effect.role == role && effect.index == index &&
           effect.bytes == bytes;
  });
}

bool hasTilingDemand(llvm::ArrayRef<wafer::WaferTilingDemand> demands,
                     wafer::WaferTilingDemandKind kind, unsigned index,
                     mlir::Type type) {
  return llvm::any_of(demands, [&](const wafer::WaferTilingDemand &demand) {
    return demand.kind == kind && demand.index == index && demand.type == type;
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

template <typename OpT> void expectTensorCollectiveInterfaces(OpT op) {
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

  auto waferTiling =
      mlir::dyn_cast<wafer::WaferTilingInterface>(op.getOperation());
  ASSERT_TRUE(waferTiling);
  llvm::SmallVector<wafer::WaferTilingDemand, 8> demands;
  waferTiling.collectWaferTilingDemand(demands);
  for (auto [index, input] : llvm::enumerate(op.getInputs())) {
    EXPECT_TRUE(hasTilingDemand(demands, wafer::WaferTilingDemandKind::Input,
                                static_cast<unsigned>(index), input.getType()));
  }
  for (auto [index, out] : llvm::enumerate(op.getOuts())) {
    EXPECT_TRUE(hasTilingDemand(demands, wafer::WaferTilingDemandKind::Output,
                                static_cast<unsigned>(index), out.getType()));
  }
  for (auto [index, result] : llvm::enumerate(op.getResults())) {
    EXPECT_TRUE(hasTilingDemand(demands, wafer::WaferTilingDemandKind::Result,
                                static_cast<unsigned>(index),
                                result.getType()));
  }
  EXPECT_TRUE(mlir::succeeded(waferTiling.verifyWaferTilingContract()));

  auto collective = mlir::dyn_cast<wafer::WaferTensorCollectiveOpInterface>(
      op.getOperation());
  ASSERT_TRUE(collective);
  EXPECT_TRUE(
      mlir::succeeded(collective.verifyWaferTensorCollectiveContract()));

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
  wafer::registerAllDialects(registry);

  mlir::MLIRContext context(registry);
  context.loadDialect<wafer::WaferDialect, mlir::async::AsyncDialect>();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  %arg = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x4xf32, #wafer.memory<ddr, tensor>>
  %tile = wafer.tile.load %arg
      : memref<4x4xf32, #wafer.memory<ddr, tensor>>
     -> memref<4x4xf32, #wafer.memory<spm, tensor>>
  %cx = wafer.tile.materialize_layout %tile
      : memref<4x4xf32, #wafer.memory<spm, tensor>>
     -> memref<4x4xf32, #wafer.memory<spm, cx>>
  %mm = wafer.tile.gemm %cx, %cx
      : (memref<4x4xf32, #wafer.memory<spm, cx>>,
         memref<4x4xf32, #wafer.memory<spm, cx>>)
     -> memref<4x4xf32, #wafer.memory<spm, cx>>
  %send = wafer.tile.send %tile {peer = 0 : i64, bytes = 64 : i64}
      : memref<4x4xf32, #wafer.memory<spm, tensor>> -> !async.token
  wafer.tile.wait %send : !async.token
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(module);

  auto load = findSingleOp<wafer::StorageLoadOp>(*module);
  ASSERT_TRUE(load);

  auto loadLayout =
      mlir::dyn_cast<wafer::WaferLayoutOpInterface>(load.getOperation());
  ASSERT_TRUE(loadLayout);
  llvm::SmallVector<wafer::WaferLayoutRequirement, 4> layoutReqs;
  loadLayout.collectWaferLayoutRequirements(layoutReqs);
  EXPECT_TRUE(hasLayoutRequirement(layoutReqs, wafer::WaferValueRole::Result, 0,
                                   wafer::MemLayout::Tensor,
                                   wafer::MemorySpace::SPM));
  EXPECT_TRUE(mlir::succeeded(loadLayout.verifyWaferLayoutContract()));

  auto loadResources =
      mlir::dyn_cast<wafer::WaferResourceEffectInterface>(load.getOperation());
  ASSERT_TRUE(loadResources);
  llvm::SmallVector<wafer::WaferResourceEffect, 4> resourceEffects;
  loadResources.collectWaferResourceEffects(resourceEffects);
  EXPECT_TRUE(hasResourceEffect(resourceEffects, wafer::WaferResourceKind::DDR,
                                wafer::WaferResourceAccess::Read,
                                wafer::WaferValueRole::Operand, 0, 64));
  EXPECT_TRUE(hasResourceEffect(resourceEffects, wafer::WaferResourceKind::SPM,
                                wafer::WaferResourceAccess::Write,
                                wafer::WaferValueRole::Result, 0, 64));
  EXPECT_TRUE(
      mlir::succeeded(loadResources.verifyWaferResourceEffectContract()));

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

  auto materialize = findSingleOp<wafer::LayoutMaterializeOp>(*module);
  ASSERT_TRUE(materialize);
  EXPECT_FALSE(
      mlir::isa<wafer::WaferLayoutOpInterface>(materialize.getOperation()));
  auto materialization =
      mlir::dyn_cast<wafer::WaferLayoutMaterializationOpInterface>(
          materialize.getOperation());
  ASSERT_TRUE(materialization);
  llvm::SmallVector<wafer::WaferLayoutRequirement, 4> materializationReqs;
  materialization.collectWaferMaterializationLayouts(materializationReqs);
  EXPECT_TRUE(hasLayoutRequirement(
      materializationReqs, wafer::WaferValueRole::Operand, 0,
      wafer::MemLayout::Tensor, wafer::MemorySpace::SPM));
  EXPECT_TRUE(
      hasLayoutRequirement(materializationReqs, wafer::WaferValueRole::Result,
                           0, wafer::MemLayout::Cx, wafer::MemorySpace::SPM));

  auto gemm = findSingleOp<wafer::ComputeGemmOp>(*module);
  ASSERT_TRUE(gemm);
  auto gemmLayout =
      mlir::dyn_cast<wafer::WaferLayoutOpInterface>(gemm.getOperation());
  ASSERT_TRUE(gemmLayout);
  layoutReqs.clear();
  gemmLayout.collectWaferLayoutRequirements(layoutReqs);
  EXPECT_TRUE(hasLayoutRequirement(layoutReqs, wafer::WaferValueRole::Operand,
                                   0, wafer::MemLayout::Cx,
                                   wafer::MemorySpace::SPM));
  EXPECT_TRUE(hasLayoutRequirement(layoutReqs, wafer::WaferValueRole::Operand,
                                   1, wafer::MemLayout::Cx,
                                   wafer::MemorySpace::SPM));
  EXPECT_TRUE(hasLayoutRequirement(layoutReqs, wafer::WaferValueRole::Result, 0,
                                   wafer::MemLayout::Cx,
                                   wafer::MemorySpace::SPM));

  auto send = findSingleOp<wafer::CommSendOp>(*module);
  ASSERT_TRUE(send);
  auto sendLayout =
      mlir::dyn_cast<wafer::WaferLayoutOpInterface>(send.getOperation());
  ASSERT_TRUE(sendLayout);
  layoutReqs.clear();
  sendLayout.collectWaferLayoutRequirements(layoutReqs);
  EXPECT_TRUE(hasLayoutRequirement(layoutReqs, wafer::WaferValueRole::Operand,
                                   0, wafer::MemLayout::Tensor,
                                   wafer::MemorySpace::SPM));

  auto sendResources =
      mlir::dyn_cast<wafer::WaferResourceEffectInterface>(send.getOperation());
  ASSERT_TRUE(sendResources);
  resourceEffects.clear();
  sendResources.collectWaferResourceEffects(resourceEffects);
  EXPECT_TRUE(hasResourceEffect(resourceEffects, wafer::WaferResourceKind::SPM,
                                wafer::WaferResourceAccess::Read,
                                wafer::WaferValueRole::Operand, 0, 64));
  EXPECT_TRUE(hasResourceEffect(
      resourceEffects, wafer::WaferResourceKind::Communication,
      wafer::WaferResourceAccess::Issue, wafer::WaferValueRole::None, 0, 64));

  auto wait = findSingleOp<wafer::CommWaitOp>(*module);
  ASSERT_TRUE(wait);
  auto waitResources =
      mlir::dyn_cast<wafer::WaferResourceEffectInterface>(wait.getOperation());
  ASSERT_TRUE(waitResources);
  resourceEffects.clear();
  waitResources.collectWaferResourceEffects(resourceEffects);
  EXPECT_TRUE(hasResourceEffect(
      resourceEffects, wafer::WaferResourceKind::Communication,
      wafer::WaferResourceAccess::Wait, wafer::WaferValueRole::Operand, 0, -1));
}

TEST(WaferInterfacesTest, TilingAndTileLoadContractsAreQueryable) {
  mlir::DialectRegistry registry;
  wafer::registerAllDialects(registry);

  mlir::MLIRContext context(registry);
  context.loadDialect<wafer::WaferDialect>();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  %source = "builtin.unrealized_conversion_cast"() : () -> tensor<4xf32>
  %source_memref = "builtin.unrealized_conversion_cast"()
      : () -> memref<4xf32, #wafer.memory<ddr, tensor>>
  %loaded = wafer.tile.load %source_memref
      : memref<4xf32, #wafer.memory<ddr, tensor>>
     -> memref<4xf32, #wafer.memory<spm, tensor>>
  %0 = wafer.group ins(%source : tensor<4xf32>)
                    outs(%source : tensor<4xf32>) {
  ^bb0(%in: tensor<4xf32>, %out: tensor<4xf32>):
    wafer.group.yield %in : tensor<4xf32>
  } : tensor<4xf32>
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(module);

  auto group = findSingleOp<wafer::GroupOp>(*module);
  ASSERT_TRUE(group);
  auto tiling =
      mlir::dyn_cast<wafer::WaferTilingInterface>(group.getOperation());
  ASSERT_TRUE(tiling);
  llvm::SmallVector<wafer::WaferTilingDemand, 4> demands;
  tiling.collectWaferTilingDemand(demands);
  EXPECT_TRUE(llvm::any_of(demands, [](const wafer::WaferTilingDemand &demand) {
    return demand.kind == wafer::WaferTilingDemandKind::Input &&
           demand.index == 0;
  }));
  EXPECT_TRUE(llvm::any_of(demands, [](const wafer::WaferTilingDemand &demand) {
    return demand.kind == wafer::WaferTilingDemandKind::Output &&
           demand.index == 0;
  }));
  EXPECT_TRUE(llvm::any_of(demands, [](const wafer::WaferTilingDemand &demand) {
    return demand.kind == wafer::WaferTilingDemandKind::Result &&
           demand.index == 0;
  }));
  EXPECT_TRUE(mlir::succeeded(tiling.verifyWaferTilingContract()));

  auto load = findSingleOp<wafer::StorageLoadOp>(*module);
  ASSERT_TRUE(load);
  auto loadResources =
      mlir::dyn_cast<wafer::WaferResourceEffectInterface>(load.getOperation());
  ASSERT_TRUE(loadResources);
  llvm::SmallVector<wafer::WaferResourceEffect, 4> effects;
  loadResources.collectWaferResourceEffects(effects);
  EXPECT_TRUE(hasResourceEffect(effects, wafer::WaferResourceKind::DDR,
                                wafer::WaferResourceAccess::Read,
                                wafer::WaferValueRole::Operand, 0, 16));
  EXPECT_TRUE(hasResourceEffect(effects, wafer::WaferResourceKind::SPM,
                                wafer::WaferResourceAccess::Write,
                                wafer::WaferValueRole::Result, 0, 16));
}

TEST(WaferInterfacesTest, InstructionInterfacesExposeQueueAndEffects) {
  mlir::DialectRegistry registry;
  wafer::registerAllDialects(registry);
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
  wafer.instr.elementwise #wafer.elementwise_kind<add> %tensor, %tensor into %tensor
      : memref<4x8xf16, #wafer.memory<spm, tensor>>,
        memref<4x8xf16, #wafer.memory<spm, tensor>>
    into memref<4x8xf16, #wafer.memory<spm, tensor>>
  wafer.instr.reduce #wafer.reduce_kind<sum> %cx into %reduce_out, %f16 : f16
      {dimensions = array<i64: 1>}
      : memref<4x8xf16, #wafer.memory<spm, cx>>
    into memref<4xf16, #wafer.memory<spm, cx>>
  wafer.instr.convert %tensor into %tensor
      {src_dtype = f16, dst_dtype = f16}
      : memref<4x8xf16, #wafer.memory<spm, tensor>>
     to memref<4x8xf16, #wafer.memory<spm, tensor>>
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
  EXPECT_EQ(rdmaInstruction.getInstructionQueueFamily(),
            wafer::InstrQueue::RDMA);
  EXPECT_TRUE(mlir::succeeded(rdmaInstruction.verifyInstructionContract()));

  auto rdmaResources =
      mlir::dyn_cast<wafer::WaferResourceEffectInterface>(rdma.getOperation());
  ASSERT_TRUE(rdmaResources);
  llvm::SmallVector<wafer::WaferResourceEffect, 4> effects;
  rdmaResources.collectWaferResourceEffects(effects);
  EXPECT_TRUE(hasResourceEffect(effects, wafer::WaferResourceKind::DDR,
                                wafer::WaferResourceAccess::Read,
                                wafer::WaferValueRole::Operand, 0, 64));
  EXPECT_TRUE(hasResourceEffect(effects, wafer::WaferResourceKind::SPM,
                                wafer::WaferResourceAccess::Write,
                                wafer::WaferValueRole::Operand, 1, 64));
  EXPECT_TRUE(hasResourceEffect(effects, wafer::WaferResourceKind::Movement,
                                wafer::WaferResourceAccess::Issue,
                                wafer::WaferValueRole::None, 0, 64));

  auto gather = findSingleOp<wafer::InstrGatherScatterOp>(*module);
  ASSERT_TRUE(gather);
  auto gatherInstruction =
      mlir::dyn_cast<wafer::WaferInstructionOpInterface>(gather.getOperation());
  ASSERT_TRUE(gatherInstruction);
  EXPECT_EQ(gatherInstruction.getInstructionQueueFamily(),
            wafer::InstrQueue::TDMA);

  auto elementwise = findSingleOp<wafer::InstrElementwiseOp>(*module);
  ASSERT_TRUE(elementwise);
  auto elementwiseInstruction =
      mlir::dyn_cast<wafer::WaferInstructionOpInterface>(
          elementwise.getOperation());
  ASSERT_TRUE(elementwiseInstruction);
  EXPECT_EQ(elementwiseInstruction.getInstructionQueueFamily(),
            wafer::InstrQueue::CT);

  auto gemm = findSingleOp<wafer::InstrGemmOp>(*module);
  ASSERT_TRUE(gemm);
  auto gemmInstruction =
      mlir::dyn_cast<wafer::WaferInstructionOpInterface>(gemm.getOperation());
  ASSERT_TRUE(gemmInstruction);
  EXPECT_EQ(gemmInstruction.getInstructionQueueFamily(), wafer::InstrQueue::NE);

  auto wdma = findSingleOp<wafer::InstrWDMAOp>(*module);
  ASSERT_TRUE(wdma);
  auto wdmaInstruction =
      mlir::dyn_cast<wafer::WaferInstructionOpInterface>(wdma.getOperation());
  ASSERT_TRUE(wdmaInstruction);
  EXPECT_EQ(wdmaInstruction.getInstructionQueueFamily(),
            wafer::InstrQueue::WDMA);
}

TEST(WaferInterfacesTest, TensorCollectivesExposeLinalgExtStyleContracts) {
  mlir::DialectRegistry registry;
  wafer::registerAllDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::tensor::TensorDialect>();

  mlir::MLIRContext context(registry);
  context.loadDialect<wafer::WaferDialect, mlir::arith::ArithDialect,
                      mlir::tensor::TensorDialect>();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  %input = "builtin.unrealized_conversion_cast"() : () -> tensor<4xf32>
  %out = "builtin.unrealized_conversion_cast"() : () -> tensor<4xf32>
  %reduced = wafer.tensor.all_reduce
      ins(%input : tensor<4xf32>)
      outs(%out : tensor<4xf32>)
      {
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      wafer.tensor.yield %sum : f32
      } {channel_id = 7 : i64, rank_group = array<i64: 0, 1>}
      -> tensor<4xf32>

  %gathered_out = "builtin.unrealized_conversion_cast"() : () -> tensor<8xf32>
  %gathered = wafer.tensor.all_gather
      ins(%input : tensor<4xf32>)
      outs(%gathered_out : tensor<8xf32>)
      {axis = 0 : i64, channel_id = 9 : i64, rank_group = array<i64: 0, 1>}
      -> tensor<8xf32>

  %wide = "builtin.unrealized_conversion_cast"() : () -> tensor<8xf32>
  %scattered = wafer.tensor.reduce_scatter
      ins(%wide : tensor<8xf32>)
      outs(%out : tensor<4xf32>)
      {
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      wafer.tensor.yield %sum : f32
      } {axis = 0 : i64, channel_id = 10 : i64,
         rank_group = array<i64: 0, 1>}
      -> tensor<4xf32>

  %matrix = "builtin.unrealized_conversion_cast"() : () -> tensor<4x2xf32>
  %matrix_out = "builtin.unrealized_conversion_cast"() : () -> tensor<2x4xf32>
  %a2a = wafer.tensor.all_to_all
      ins(%matrix : tensor<4x2xf32>)
      outs(%matrix_out : tensor<2x4xf32>)
      {split_axis = 0 : i64, concat_axis = 1 : i64,
       split_count = 2 : i64, channel_id = 11 : i64,
       rank_group = array<i64: 0, 1>}
      -> tensor<2x4xf32>

  %permuted = wafer.tensor.collective_permute
      ins(%input : tensor<4xf32>)
      outs(%out : tensor<4xf32>)
      {channel_id = 12 : i64, source_target_pairs = array<i64: 0, 1, 1, 0>}
      -> tensor<4xf32>
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(module);

  auto allReduce = findSingleOp<wafer::TensorCollectiveAllReduceOp>(*module);
  ASSERT_TRUE(allReduce);
  expectTensorCollectiveInterfaces(allReduce);
  auto collective = mlir::dyn_cast<wafer::WaferTensorCollectiveOpInterface>(
      allReduce.getOperation());
  ASSERT_TRUE(collective);
  wafer::WaferTensorCollectiveInfo info;
  collective.collectWaferTensorCollectiveInfo(info);
  EXPECT_EQ(info.kind, wafer::WaferTensorCollectiveKind::AllReduce);
  EXPECT_TRUE(info.hasCombiner);
  EXPECT_TRUE(info.hasCommunicationEffect);
  EXPECT_TRUE(info.hasChannelId);
  EXPECT_EQ(info.channelId, 7);
  EXPECT_EQ(info.rankGroup, llvm::SmallVector<int64_t>({0, 1}));
  EXPECT_TRUE(
      mlir::succeeded(collective.verifyWaferTensorCollectiveContract()));

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

  auto allGather = findSingleOp<wafer::TensorCollectiveAllGatherOp>(*module);
  ASSERT_TRUE(allGather);
  expectTensorCollectiveInterfaces(allGather);
  auto gatherCollective =
      mlir::dyn_cast<wafer::WaferTensorCollectiveOpInterface>(
          allGather.getOperation());
  ASSERT_TRUE(gatherCollective);
  info = {};
  gatherCollective.collectWaferTensorCollectiveInfo(info);
  EXPECT_EQ(info.kind, wafer::WaferTensorCollectiveKind::AllGather);
  EXPECT_TRUE(info.hasAxis);
  EXPECT_EQ(info.axis, 0);
  EXPECT_EQ(info.channelId, 9);
  EXPECT_EQ(info.rankGroup, llvm::SmallVector<int64_t>({0, 1}));
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
      findSingleOp<wafer::TensorCollectiveReduceScatterOp>(*module);
  ASSERT_TRUE(reduceScatter);
  expectTensorCollectiveInterfaces(reduceScatter);
  auto reduceScatterCollective =
      mlir::dyn_cast<wafer::WaferTensorCollectiveOpInterface>(
          reduceScatter.getOperation());
  ASSERT_TRUE(reduceScatterCollective);
  info = {};
  reduceScatterCollective.collectWaferTensorCollectiveInfo(info);
  EXPECT_EQ(info.kind, wafer::WaferTensorCollectiveKind::ReduceScatter);
  EXPECT_TRUE(info.hasAxis);
  EXPECT_TRUE(info.hasCombiner);
  EXPECT_EQ(info.axis, 0);
  EXPECT_EQ(info.channelId, 10);
  EXPECT_EQ(info.rankGroup, llvm::SmallVector<int64_t>({0, 1}));
  EXPECT_TRUE(mlir::succeeded(
      reduceScatterCollective.verifyWaferTensorCollectiveContract()));
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

  auto allToAll = findSingleOp<wafer::TensorCollectiveAllToAllOp>(*module);
  ASSERT_TRUE(allToAll);
  expectTensorCollectiveInterfaces(allToAll);
  auto allToAllCollective =
      mlir::dyn_cast<wafer::WaferTensorCollectiveOpInterface>(
          allToAll.getOperation());
  ASSERT_TRUE(allToAllCollective);
  info = {};
  allToAllCollective.collectWaferTensorCollectiveInfo(info);
  EXPECT_EQ(info.kind, wafer::WaferTensorCollectiveKind::AllToAll);
  EXPECT_TRUE(info.hasSplitAxis);
  EXPECT_TRUE(info.hasConcatAxis);
  EXPECT_TRUE(info.hasSplitCount);
  EXPECT_EQ(info.splitAxis, 0);
  EXPECT_EQ(info.concatAxis, 1);
  EXPECT_EQ(info.splitCount, 2);
  EXPECT_EQ(info.channelId, 11);
  EXPECT_EQ(info.rankGroup, llvm::SmallVector<int64_t>({0, 1}));
  EXPECT_TRUE(mlir::succeeded(
      allToAllCollective.verifyWaferTensorCollectiveContract()));
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
      findSingleOp<wafer::TensorCollectiveCollectivePermuteOp>(*module);
  ASSERT_TRUE(permute);
  expectTensorCollectiveInterfaces(permute);
  auto permuteCollective =
      mlir::dyn_cast<wafer::WaferTensorCollectiveOpInterface>(
          permute.getOperation());
  ASSERT_TRUE(permuteCollective);
  info = {};
  permuteCollective.collectWaferTensorCollectiveInfo(info);
  EXPECT_EQ(info.kind, wafer::WaferTensorCollectiveKind::CollectivePermute);
  EXPECT_TRUE(info.hasCommunicationEffect);
  EXPECT_EQ(info.channelId, 12);
  EXPECT_EQ(info.sourceTargetPairs, llvm::SmallVector<int64_t>({0, 1, 1, 0}));
  EXPECT_TRUE(
      mlir::succeeded(permuteCollective.verifyWaferTensorCollectiveContract()));
  EXPECT_TRUE(mlir::isa<mlir::TilingInterface>(permute.getOperation()));
  expectTiledImplementation(
      permute, context, llvm::SmallVector<int64_t>{1},
      llvm::SmallVector<int64_t>{2}, llvm::SmallVector<int64_t>{2},
      llvm::SmallVector<ExpectedExtractSlice>{
          expectSlice(permute.getInputs()[0], {1}, {2}, {2}),
          expectSlice(permute.getOuts()[0], {1}, {2}, {2})});
}

TEST(WaferInterfacesTest, TensorCollectiveTilingHandlesNonTrivialShapes) {
  mlir::DialectRegistry registry;
  wafer::registerAllDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::tensor::TensorDialect>();

  mlir::MLIRContext context(registry);
  context.loadDialect<wafer::WaferDialect, mlir::arith::ArithDialect,
                      mlir::tensor::TensorDialect>();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  %ar_input = "builtin.unrealized_conversion_cast"() : () -> tensor<1024x4096xf32>
  %ar_out = "builtin.unrealized_conversion_cast"() : () -> tensor<1024x4096xf32>
  %ar = wafer.tensor.all_reduce
      ins(%ar_input : tensor<1024x4096xf32>)
      outs(%ar_out : tensor<1024x4096xf32>)
      {
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      wafer.tensor.yield %sum : f32
      } {channel_id = 17 : i64, rank_group = array<i64: 0, 1, 2, 3, 4, 5, 6, 7>}
      -> tensor<1024x4096xf32>

  %ag_input = "builtin.unrealized_conversion_cast"() : () -> tensor<64x512xf32>
  %ag_out = "builtin.unrealized_conversion_cast"() : () -> tensor<64x4096xf32>
  %ag = wafer.tensor.all_gather
      ins(%ag_input : tensor<64x512xf32>)
      outs(%ag_out : tensor<64x4096xf32>)
      {axis = 1 : i64, channel_id = 18 : i64,
       rank_group = array<i64: 0, 1, 2, 3, 4, 5, 6, 7>}
      -> tensor<64x4096xf32>

  %rs_input = "builtin.unrealized_conversion_cast"() : () -> tensor<256x4096xf32>
  %rs_out = "builtin.unrealized_conversion_cast"() : () -> tensor<256x1024xf32>
  %rs = wafer.tensor.reduce_scatter
      ins(%rs_input : tensor<256x4096xf32>)
      outs(%rs_out : tensor<256x1024xf32>)
      {
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      wafer.tensor.yield %sum : f32
      } {axis = 1 : i64, channel_id = 19 : i64,
         rank_group = array<i64: 0, 1, 2, 3>}
      -> tensor<256x1024xf32>

  %a2a_input = "builtin.unrealized_conversion_cast"() : () -> tensor<512x128x64xf32>
  %a2a_out = "builtin.unrealized_conversion_cast"() : () -> tensor<128x512x64xf32>
  %a2a = wafer.tensor.all_to_all
      ins(%a2a_input : tensor<512x128x64xf32>)
      outs(%a2a_out : tensor<128x512x64xf32>)
      {split_axis = 0 : i64, concat_axis = 1 : i64,
       split_count = 4 : i64, channel_id = 20 : i64,
       rank_group = array<i64: 0, 1, 2, 3>}
      -> tensor<128x512x64xf32>

  %cp_input = "builtin.unrealized_conversion_cast"() : () -> tensor<4x1024x4096xf32>
  %cp_out = "builtin.unrealized_conversion_cast"() : () -> tensor<4x1024x4096xf32>
  %cp = wafer.tensor.collective_permute
      ins(%cp_input : tensor<4x1024x4096xf32>)
      outs(%cp_out : tensor<4x1024x4096xf32>)
      {channel_id = 21 : i64,
       source_target_pairs = array<i64: 0, 1, 1, 2, 2, 3, 3, 0>}
      -> tensor<4x1024x4096xf32>
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(module);

  auto allReduce = findSingleOp<wafer::TensorCollectiveAllReduceOp>(*module);
  ASSERT_TRUE(allReduce);
  expectTensorCollectiveInterfaces(allReduce);
  expectTiledImplementation(
      allReduce, context, llvm::SmallVector<int64_t>{128, 256},
      llvm::SmallVector<int64_t>{64, 512}, llvm::SmallVector<int64_t>{64, 512},
      llvm::SmallVector<ExpectedExtractSlice>{
          expectSlice(allReduce.getInputs()[0], {128, 256}, {64, 512},
                      {64, 512}),
          expectSlice(allReduce.getOuts()[0], {128, 256}, {64, 512},
                      {64, 512})});

  auto allGather = findSingleOp<wafer::TensorCollectiveAllGatherOp>(*module);
  ASSERT_TRUE(allGather);
  expectTensorCollectiveInterfaces(allGather);
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
      findSingleOp<wafer::TensorCollectiveReduceScatterOp>(*module);
  ASSERT_TRUE(reduceScatter);
  expectTensorCollectiveInterfaces(reduceScatter);
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

  auto allToAll = findSingleOp<wafer::TensorCollectiveAllToAllOp>(*module);
  ASSERT_TRUE(allToAll);
  expectTensorCollectiveInterfaces(allToAll);
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
      findSingleOp<wafer::TensorCollectiveCollectivePermuteOp>(*module);
  ASSERT_TRUE(permute);
  expectTensorCollectiveInterfaces(permute);
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

TEST(WaferInterfacesTest, TensorCollectiveTilingHandlesDynamicNonAxisShapes) {
  mlir::DialectRegistry registry;
  wafer::registerAllDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::tensor::TensorDialect>();

  mlir::MLIRContext context(registry);
  context.loadDialect<wafer::WaferDialect, mlir::arith::ArithDialect,
                      mlir::tensor::TensorDialect>();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  %ar_input = "builtin.unrealized_conversion_cast"() : () -> tensor<?x4096xf32>
  %ar_out = "builtin.unrealized_conversion_cast"() : () -> tensor<?x4096xf32>
  %ar = wafer.tensor.all_reduce
      ins(%ar_input : tensor<?x4096xf32>)
      outs(%ar_out : tensor<?x4096xf32>)
      {
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      wafer.tensor.yield %sum : f32
      } {channel_id = 22 : i64, rank_group = array<i64: 0, 1, 2, 3>}
      -> tensor<?x4096xf32>

  %ag_input = "builtin.unrealized_conversion_cast"() : () -> tensor<?x512xf32>
  %ag_out = "builtin.unrealized_conversion_cast"() : () -> tensor<?x2048xf32>
  %ag = wafer.tensor.all_gather
      ins(%ag_input : tensor<?x512xf32>)
      outs(%ag_out : tensor<?x2048xf32>)
      {axis = 1 : i64, channel_id = 23 : i64,
       rank_group = array<i64: 0, 1, 2, 3>}
      -> tensor<?x2048xf32>
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(module);

  auto allReduce = findSingleOp<wafer::TensorCollectiveAllReduceOp>(*module);
  ASSERT_TRUE(allReduce);
  expectTensorCollectiveInterfaces(allReduce);
  expectTiledImplementation(
      allReduce, context, llvm::SmallVector<int64_t>{0, 1024},
      llvm::SmallVector<int64_t>{8, 512}, llvm::SmallVector<int64_t>{8, 512},
      llvm::SmallVector<ExpectedExtractSlice>{
          expectSlice(allReduce.getInputs()[0], {0, 1024}, {8, 512}, {8, 512}),
          expectSlice(allReduce.getOuts()[0], {0, 1024}, {8, 512}, {8, 512})});

  auto allGather = findSingleOp<wafer::TensorCollectiveAllGatherOp>(*module);
  ASSERT_TRUE(allGather);
  expectTensorCollectiveInterfaces(allGather);
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
  wafer::registerAllDialects(registry);
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

  auto allGather = findSingleOp<wafer::TensorCollectiveAllGatherOp>(*module);
  ASSERT_TRUE(allGather);
  expectTensorCollectiveInterfaces(allGather);
  auto collective = mlir::dyn_cast<wafer::WaferTensorCollectiveOpInterface>(
      allGather.getOperation());
  ASSERT_TRUE(collective);
  wafer::WaferTensorCollectiveInfo info;
  collective.collectWaferTensorCollectiveInfo(info);
  EXPECT_EQ(info.kind, wafer::WaferTensorCollectiveKind::AllGather);
  EXPECT_EQ(info.axis, 1);
  EXPECT_EQ(info.channelId, 41);
  EXPECT_EQ(info.rankGroup,
            llvm::SmallVector<int64_t>({0, 1, 2, 3, 4, 5, 6, 7}));
  expectTiledImplementation(
      allGather, context, llvm::SmallVector<int64_t>{16, 0},
      llvm::SmallVector<int64_t>{8, 4096}, llvm::SmallVector<int64_t>{8, 4096},
      llvm::SmallVector<ExpectedExtractSlice>{
          expectSlice(allGather.getInputs()[0], {16, 0}, {8, 512}, {8, 512}),
          expectSlice(allGather.getOuts()[0], {16, 0}, {8, 4096}, {8, 4096})});
#endif
}

} // namespace
