#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitAll.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"

#include "gtest/gtest.h"

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

} // namespace
