#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitAll.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"

#include "gtest/gtest.h"

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
void expectTiledImplementation(OpT op, mlir::MLIRContext &context,
                               llvm::ArrayRef<int64_t> offsetValues,
                               llvm::ArrayRef<int64_t> sizeValues,
                               llvm::ArrayRef<int64_t> expectedShape) {
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
  %arg = "builtin.unrealized_conversion_cast"() : () -> tensor<4x4xf32>
  %tile = wafer.load_tile %arg
      : tensor<4x4xf32>
     -> !wafer.tile_buffer<tensor<4x4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  %cx = wafer.layout.materialize %tile
      : !wafer.tile_buffer<tensor<4x4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
     -> !wafer.tile_buffer<tensor<4x4xf32>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
  %mm = wafer.compute.gemm %cx, %cx
      : (!wafer.tile_buffer<tensor<4x4xf32>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>,
         !wafer.tile_buffer<tensor<4x4xf32>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>)
     -> !wafer.tile_buffer<tensor<4x4xf32>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
  %send = wafer.comm.send %tile {peer = 0 : i64, bytes = 64 : i64}
      : !wafer.tile_buffer<tensor<4x4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>> -> !async.token
  wafer.comm.wait %send : !async.token
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(module);

  auto load = findSingleOp<wafer::LoadTileOp>(*module);
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

TEST(WaferInterfacesTest, TilingAndDDRContractsAreQueryable) {
  mlir::DialectRegistry registry;
  wafer::registerAllDialects(registry);

  mlir::MLIRContext context(registry);
  context.loadDialect<wafer::WaferDialect>();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  %source = "builtin.unrealized_conversion_cast"() : () -> tensor<4xf32>
  wafer.ddr.external_binding <input> %source
      {alignment = 256 : i64, bytes = 16 : i64, host_visible = true, read_only = true}
      : tensor<4xf32>
  %0 = wafer.group ins(%source : tensor<4xf32>)
                    outs(%source : tensor<4xf32>) {
  ^bb0(%in: tensor<4xf32>, %out: tensor<4xf32>):
    wafer.group_yield %in : tensor<4xf32>
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

  auto binding = findSingleOp<wafer::DdrExternalBindingOp>(*module);
  ASSERT_TRUE(binding);
  auto bindingResources = mlir::dyn_cast<wafer::WaferResourceEffectInterface>(
      binding.getOperation());
  ASSERT_TRUE(bindingResources);
  llvm::SmallVector<wafer::WaferResourceEffect, 4> effects;
  bindingResources.collectWaferResourceEffects(effects);
  EXPECT_TRUE(hasResourceEffect(effects, wafer::WaferResourceKind::DDR,
                                wafer::WaferResourceAccess::Read,
                                wafer::WaferValueRole::Operand, 0, 16));
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
  %reduced = wafer.tensor_collective.all_reduce
      ins(%input : tensor<4xf32>)
      outs(%out : tensor<4xf32>)
      {
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      wafer.tensor_collective.yield %sum : f32
      } {channel_id = 7 : i64, rank_group = array<i64: 0, 1>}
      -> tensor<4xf32>

  %gathered_out = "builtin.unrealized_conversion_cast"() : () -> tensor<8xf32>
  %gathered = wafer.tensor_collective.all_gather
      ins(%input : tensor<4xf32>)
      outs(%gathered_out : tensor<8xf32>)
      {axis = 0 : i64, channel_id = 9 : i64, rank_group = array<i64: 0, 1>}
      -> tensor<8xf32>

  %wide = "builtin.unrealized_conversion_cast"() : () -> tensor<8xf32>
  %scattered = wafer.tensor_collective.reduce_scatter
      ins(%wide : tensor<8xf32>)
      outs(%out : tensor<4xf32>)
      {
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      wafer.tensor_collective.yield %sum : f32
      } {axis = 0 : i64, channel_id = 10 : i64,
         rank_group = array<i64: 0, 1>}
      -> tensor<4xf32>

  %matrix = "builtin.unrealized_conversion_cast"() : () -> tensor<4x2xf32>
  %matrix_out = "builtin.unrealized_conversion_cast"() : () -> tensor<2x4xf32>
  %a2a = wafer.tensor_collective.all_to_all
      ins(%matrix : tensor<4x2xf32>)
      outs(%matrix_out : tensor<2x4xf32>)
      {split_axis = 0 : i64, concat_axis = 1 : i64,
       split_count = 2 : i64, channel_id = 11 : i64,
       rank_group = array<i64: 0, 1>}
      -> tensor<2x4xf32>

  %permuted = wafer.tensor_collective.collective_permute
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

  llvm::SmallVector<int64_t> offsetValues{1};
  llvm::SmallVector<int64_t> sizeValues{2};
  llvm::SmallVector<mlir::OpFoldResult> offsets =
      mlir::getAsIndexOpFoldResult(&context, offsetValues);
  llvm::SmallVector<mlir::OpFoldResult> sizes =
      mlir::getAsIndexOpFoldResult(&context, sizeValues);
  mlir::FailureOr<mlir::TilingResult> tiled =
      tiling.getTiledImplementation(builder, offsets, sizes);
  ASSERT_TRUE(mlir::succeeded(tiled));
  ASSERT_EQ(tiled->tiledOps.size(), 1u);
  EXPECT_TRUE(
      mlir::isa<wafer::TensorCollectiveAllReduceOp>(tiled->tiledOps.front()));
  ASSERT_EQ(tiled->tiledValues.size(), 1u);
  auto tiledType = mlir::dyn_cast<mlir::RankedTensorType>(
      tiled->tiledValues.front().getType());
  ASSERT_TRUE(tiledType);
  EXPECT_EQ(tiledType.getShape(), llvm::ArrayRef<int64_t>({2}));

  llvm::SmallVector<mlir::OpFoldResult> resultOffsets;
  llvm::SmallVector<mlir::OpFoldResult> resultSizes;
  EXPECT_TRUE(mlir::succeeded(tiling.getResultTilePosition(
      builder, 0, offsets, sizes, resultOffsets, resultSizes)));
  EXPECT_EQ(mlir::getConstantIntValues(resultOffsets),
            std::optional<llvm::SmallVector<int64_t>>({{1}}));
  EXPECT_EQ(mlir::getConstantIntValues(resultSizes),
            std::optional<llvm::SmallVector<int64_t>>({{2}}));

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
  expectTiledImplementation(allGather, context, llvm::SmallVector<int64_t>{0},
                            llvm::SmallVector<int64_t>{8},
                            llvm::SmallVector<int64_t>{8});

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
      llvm::SmallVector<int64_t>{4}, llvm::SmallVector<int64_t>{4});
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
  expectTiledImplementation(allToAll, context, llvm::SmallVector<int64_t>{0, 0},
                            llvm::SmallVector<int64_t>{2, 4},
                            llvm::SmallVector<int64_t>{2, 4});
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
  expectTiledImplementation(permute, context, llvm::SmallVector<int64_t>{1},
                            llvm::SmallVector<int64_t>{2},
                            llvm::SmallVector<int64_t>{2});
}

} // namespace
