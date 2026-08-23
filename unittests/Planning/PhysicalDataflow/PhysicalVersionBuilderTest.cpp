//===- PhysicalVersionBuilderTest.cpp --------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/PhysicalVersionBuilder.h"

#include "Wafer/InitWaferDialects.h"

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
  mlir::Value primary;
};

IRFixture makeIR(MemLayout layout) {
  mlir::DialectRegistry registry;
  registerWaferCoreDialects(registry);
  registry.insert<mlir::func::FuncDialect, mlir::memref::MemRefDialect>();
  auto context = std::make_unique<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(context.get()));
  mlir::OpBuilder moduleBuilder(module.getBodyRegion());
  auto function = moduleBuilder.create<mlir::func::FuncOp>(
      module.getLoc(), "versions",
      moduleBuilder.getFunctionType(mlir::TypeRange{}, mlir::TypeRange{}));
  mlir::Block *entry = function.addEntryBlock();
  mlir::OpBuilder builder = mlir::OpBuilder::atBlockBegin(entry);
  auto type = mlir::MemRefType::get(
      {2, 1031, 128}, builder.getF16Type(), mlir::MemRefLayoutAttrInterface{},
      MemoryAttr::get(context.get(), MemorySpace::SPM, layout));
  mlir::Value primary =
      builder.create<mlir::memref::AllocOp>(module.getLoc(), type);
  return {std::move(context), mlir::OwningOpRef<mlir::ModuleOp>(module),
          function, builder, primary};
}

PhysicalVersionId makePrimary() {
  SemanticRootKey root;
  RootRegionWorkId work{root, TileId(0)};
  LogicalShardId shard{root, {0}};
  return PhysicalVersionId{ExecutionResultValueId{
      ExecutionInstanceId{RequiredRootExecution{work, shard}}, 0}};
}

PhysicalVersionId derive(PhysicalVersionId source, MemLayout encoding) {
  const MemLayout sourceEncoding = source.derivation.empty()
                                       ? MemLayout::Tensor
                                       : source.derivation.back().encoding;
  source.derivation.push_back({PhysicalVersionDerivationKind::LayoutConversion,
                               sourceEncoding, encoding,
                               SharedRepresentationAnchor{}});
  return source;
}

TEST(PhysicalVersionBuilderTest,
     SharedAndChainedConversionsEmitEachSelectedDefinitionOnce) {
  IRFixture ir = makeIR(MemLayout::Tensor);
  PhysicalVersionId primary = makePrimary();
  PhysicalVersionId nTensor = derive(primary, MemLayout::NTensor);
  PhysicalVersionId cx = derive(nTensor, MemLayout::Cx);
  PreparedRepresentationPlan prepared;
  prepared.primaryVersions.push_back({primary, MemLayout::Tensor});
  prepared.derivedVersions.push_back({nTensor, MemLayout::NTensor});
  prepared.derivedVersions.push_back({cx, MemLayout::Cx});

  PhysicalVersionBuilder versions;
  std::string failureReason;
  ASSERT_TRUE(
      mlir::succeeded(versions.bind(primary, ir.primary, &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(emitPreparedRepresentationVersions(
      prepared, versions, ir.builder, &failureReason)))
      << failureReason;
  ir.builder.create<mlir::func::ReturnOp>(ir.module->getLoc());
  EXPECT_TRUE(versions.lookup(nTensor));
  EXPECT_TRUE(versions.lookup(cx));
  unsigned conversions = 0;
  ir.module->walk([&](LayoutMaterializeOp) { ++conversions; });
  EXPECT_EQ(conversions, 2u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*ir.module)));
}

TEST(PhysicalVersionBuilderTest,
     PreflightAndDuplicateBindingFailuresLeaveIRUnchanged) {
  IRFixture ir = makeIR(MemLayout::NTensor);
  PhysicalVersionId primary = makePrimary();
  PhysicalVersionId derived = derive(primary, MemLayout::Cx);
  PreparedRepresentationPlan prepared;
  prepared.primaryVersions.push_back({primary, MemLayout::Tensor});
  prepared.derivedVersions.push_back({derived, MemLayout::Cx});
  PhysicalVersionBuilder versions;
  std::string failureReason;
  ASSERT_TRUE(
      mlir::succeeded(versions.bind(primary, ir.primary, &failureReason)))
      << failureReason;
  EXPECT_TRUE(mlir::failed(emitPreparedRepresentationVersions(
      prepared, versions, ir.builder, &failureReason)));
  EXPECT_NE(failureReason.find("primary"), std::string::npos);
  unsigned conversions = 0;
  ir.module->walk([&](LayoutMaterializeOp) { ++conversions; });
  EXPECT_EQ(conversions, 0u);
  EXPECT_TRUE(mlir::failed(versions.bind(primary, ir.primary, &failureReason)));
  EXPECT_EQ(conversions, 0u);
}

TEST(PhysicalVersionBuilderTest,
     ExactIdentityAliasBindsSourceWithoutMaterializingCopy) {
  IRFixture ir = makeIR(MemLayout::Tensor);
  PhysicalVersionId source = makePrimary();
  const auto &execution =
      std::get<ExecutionResultValueId>(source.logicalValue).execution;
  const auto &required = std::get<RequiredRootExecution>(execution.source);
  PhysicalVersionId target{SupportRegionValueId{
      required.work, SupportValueId{required.work.root, 0}}};
  PhysicalVersionId alias = target;
  alias.derivation.push_back(
      {PhysicalVersionDerivationKind::AliasView, MemLayout::Tensor,
       MemLayout::Tensor, SharedRepresentationAnchor{}, source.logicalValue});
  PreparedRepresentationPlan prepared;
  prepared.primaryVersions.push_back({source, MemLayout::Tensor});
  prepared.primaryVersions.push_back({target, MemLayout::Tensor});
  prepared.derivedVersions.push_back({alias, MemLayout::Tensor});
  PhysicalVersionBuilder versions;
  std::string failureReason;
  ASSERT_TRUE(
      mlir::succeeded(versions.bind(source, ir.primary, &failureReason)))
      << failureReason;
  ASSERT_TRUE(
      mlir::succeeded(versions.bind(target, ir.primary, &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(emitPreparedRepresentationVersions(
      prepared, versions, ir.builder, &failureReason)))
      << failureReason;
  EXPECT_EQ(versions.lookup(alias), ir.primary);
  unsigned conversions = 0;
  ir.module->walk([&](LayoutMaterializeOp) { ++conversions; });
  EXPECT_EQ(conversions, 0u);
}

} // namespace
