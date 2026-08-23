//===- StorageObjectBuilderTest.cpp ----------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/StorageObjectBuilder.h"

#include "Wafer/Analysis/PhysicalDataflow/IndexRelation.h"
#include "Wafer/InitWaferDialects.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "gtest/gtest.h"

#include <memory>

namespace {

using namespace wafer;
using namespace wafer::analysis;
using namespace wafer::compiler::detail;

ExactIndexSet makeDomain(llvm::ArrayRef<int64_t> sizes) {
  llvm::SmallVector<int64_t, 4> offsets(sizes.size(), 0);
  IndexSetResult set = IndexRelation::staticRectangularDomain(offsets, sizes);
  EXPECT_TRUE(set.isExact()) << set.reason;
  StaticRectangularIndexSet box{offsets, llvm::SmallVector<int64_t, 4>(sizes)};
  return ExactIndexSet(std::move(*set.set), ExactIndexSetForm::BoxUnion, {box});
}

PhysicalVersionId makeVersion(uint32_t result) {
  SemanticRootKey root;
  RootRegionWorkId work{root, TileId(0)};
  LogicalShardId shard{root, {0}};
  return PhysicalVersionId{ExecutionResultValueId{
      ExecutionInstanceId{RequiredRootExecution{work, shard}}, result}};
}

struct IRFixture {
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::func::FuncOp function;
  mlir::OpBuilder builder;
};

IRFixture makeIR() {
  mlir::DialectRegistry registry;
  registerWaferCoreDialects(registry);
  registry.insert<mlir::func::FuncDialect, mlir::memref::MemRefDialect>();
  auto context = std::make_unique<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(context.get()));
  mlir::OpBuilder moduleBuilder(module.getBodyRegion());
  auto function = moduleBuilder.create<mlir::func::FuncOp>(
      module.getLoc(), "storage",
      moduleBuilder.getFunctionType(mlir::TypeRange{}, mlir::TypeRange{}));
  mlir::Block *entry = function.addEntryBlock();
  return {std::move(context), mlir::OwningOpRef<mlir::ModuleOp>(module),
          function, mlir::OpBuilder::atBlockBegin(entry)};
}

TEST(StorageObjectBuilderTest,
     MultiSlotAndIdentityAliasUseExactSelectedObjects) {
  IRFixture ir = makeIR();
  PhysicalVersionId source = makeVersion(0);
  PhysicalVersionId alias = makeVersion(1);
  StorageObjectId object{StorageObjectOrigin(source)};
  PreparedStoragePlan prepared;
  prepared.objects.push_back(
      {{object, TileId(0)},
       StorageResourceDescription{object, makeDomain({2, 1025, 128}),
                                  ir.builder.getF16Type(), MemLayout::Tensor},
       5});
  prepared.bindings.push_back({source, object, StorageBindingKind::Fresh});
  prepared.bindings.push_back(
      {alias, object, StorageBindingKind::IdentityAlias});
  StorageObjectBuilder objects;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(emitPreparedStorageObjects(
      prepared, objects, ir.builder, &failureReason)))
      << failureReason;
  unsigned allocations = 0;
  ir.module->walk([&](mlir::memref::AllocOp) { ++allocations; });
  EXPECT_EQ(allocations, 5u);
  EXPECT_EQ(objects.lookup(source, 0), objects.lookup(source, 5));
  EXPECT_NE(objects.lookup(source, 0), objects.lookup(source, 1));
  EXPECT_EQ(objects.lookup(alias, 7), objects.lookup(source, 7));
}

TEST(StorageObjectBuilderTest,
     DuplicatePreparedObjectFailsBeforeAnyAllocationMutation) {
  IRFixture ir = makeIR();
  PhysicalVersionId version = makeVersion(0);
  StorageObjectId object{StorageObjectOrigin(version)};
  PreparedStorageObject preparedObject{
      {object, TileId(0)},
      StorageResourceDescription{object, makeDomain({2, 1031, 128}),
                                 ir.builder.getF16Type(), MemLayout::Tensor},
      1};
  PreparedStoragePlan prepared;
  prepared.objects = {preparedObject, preparedObject};
  prepared.bindings.push_back({version, object, StorageBindingKind::Fresh});
  StorageObjectBuilder objects;
  std::string failureReason;
  EXPECT_TRUE(mlir::failed(emitPreparedStorageObjects(
      prepared, objects, ir.builder, &failureReason)));
  unsigned allocations = 0;
  ir.module->walk([&](mlir::memref::AllocOp) { ++allocations; });
  EXPECT_EQ(allocations, 0u);
}

} // namespace
