//===- MovementTransferBuilderTest.cpp -------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/MovementTransferBuilder.h"

#include "Wafer/InitWaferDialects.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Verifier.h"

#include "gtest/gtest.h"

#include <memory>

namespace {

using namespace wafer;
using namespace wafer::analysis;
using namespace wafer::compiler::detail;

struct IRFixture {
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::func::FuncOp function;
  mlir::OpBuilder builder;
  llvm::SmallVector<mlir::Value, 3> buffers;
};

IRFixture makeIR(bool mismatchedLast = false) {
  mlir::DialectRegistry registry;
  registerWaferCoreDialects(registry);
  registry.insert<mlir::async::AsyncDialect, mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect>();
  auto context = std::make_unique<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(context.get()));
  mlir::OpBuilder moduleBuilder(module.getBodyRegion());
  auto function = moduleBuilder.create<mlir::func::FuncOp>(
      module.getLoc(), "movement",
      moduleBuilder.getFunctionType(mlir::TypeRange{}, mlir::TypeRange{}));
  mlir::Block *entry = function.addEntryBlock();
  mlir::OpBuilder builder = mlir::OpBuilder::atBlockBegin(entry);
  llvm::SmallVector<mlir::Value, 3> buffers;
  for (int64_t tile = 0; tile < 3; ++tile) {
    MemLayout layout =
        mismatchedLast && tile == 2 ? MemLayout::NTensor : MemLayout::Tensor;
    auto type = mlir::MemRefType::get(
        {2, 1025, 128}, builder.getF16Type(), mlir::MemRefLayoutAttrInterface{},
        MemoryAttr::get(context.get(), MemorySpace::SPM, layout));
    buffers.push_back(
        builder.create<mlir::memref::AllocOp>(module.getLoc(), type));
  }
  return {std::move(context), mlir::OwningOpRef<mlir::ModuleOp>(module),
          function, builder, std::move(buffers)};
}

MovementActionId makeAction() {
  SemanticRootKey root;
  RootRegionWorkId work{root, TileId(2)};
  LogicalShardId shard{root, {0}};
  DemandFragmentId fragment;
  fragment.source.kind = RootBoundaryKind::ProgramInput;
  fragment.use = {0, shard};
  return ExternalLoadId{BoundaryRegionValueId{work, fragment}};
}

TEST(MovementTransferBuilderTest,
     RelayEmitsMatchingTokensWithoutImmediateAwait) {
  IRFixture ir = makeIR();
  MovementRealization realization;
  realization.kind = MovementRealizationKind::SoftwareRelay;
  realization.hops = {{TileId(0), TileId(1)}, {TileId(1), TileId(2)}};
  llvm::SmallVector<MovementEndpointBinding, 3> endpoints;
  for (int64_t tile = 0; tile < 3; ++tile)
    endpoints.push_back({TileId(tile), ir.buffers[tile], &ir.builder});
  std::string failureReason;
  auto prepared =
      preparePeerTransfer(makeAction(), realization, endpoints, &failureReason);
  ASSERT_TRUE(mlir::succeeded(prepared)) << failureReason;
  auto emitted = emitPreparedPeerTransfer(
      *prepared, endpoints, /*communication=*/7, /*payload=*/3, &failureReason);
  ASSERT_TRUE(mlir::succeeded(emitted)) << failureReason;
  EXPECT_EQ(emitted->sendTokens.size(), 2u);
  EXPECT_EQ(emitted->receiveTokens.size(), 2u);
  ir.builder.create<mlir::func::ReturnOp>(ir.module->getLoc());
  unsigned sends = 0;
  unsigned receives = 0;
  unsigned awaits = 0;
  ir.module->walk([&](CommPeerSendOp) { ++sends; });
  ir.module->walk([&](CommPeerRecvOp) { ++receives; });
  ir.module->walk([&](mlir::async::AwaitOp) { ++awaits; });
  EXPECT_EQ(sends, 2u);
  EXPECT_EQ(receives, 2u);
  EXPECT_EQ(awaits, 0u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*ir.module)));
}

TEST(MovementTransferBuilderTest,
     PhysicalMismatchAndCycleFailBeforeAnyCommunicationMutation) {
  IRFixture ir = makeIR(/*mismatchedLast=*/true);
  MovementRealization realization;
  realization.kind = MovementRealizationKind::SoftwareRelay;
  realization.hops = {{TileId(0), TileId(1)}, {TileId(1), TileId(2)}};
  llvm::SmallVector<MovementEndpointBinding, 3> endpoints;
  for (int64_t tile = 0; tile < 3; ++tile)
    endpoints.push_back({TileId(tile), ir.buffers[tile], &ir.builder});
  std::string failureReason;
  EXPECT_TRUE(mlir::failed(preparePeerTransfer(makeAction(), realization,
                                               endpoints, &failureReason)));
  unsigned communicationOps = 0;
  ir.module->walk([&](CommPeerSendOp) { ++communicationOps; });
  ir.module->walk([&](CommPeerRecvOp) { ++communicationOps; });
  EXPECT_EQ(communicationOps, 0u);

  realization.hops = {{TileId(0), TileId(1)}, {TileId(1), TileId(0)}};
  EXPECT_TRUE(mlir::failed(preparePeerTransfer(makeAction(), realization,
                                               endpoints, &failureReason)));
  EXPECT_EQ(communicationOps, 0u);
}

} // namespace
