#include "Wafer/Frontend/InitImporterDialects.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitAll.h"
#include "Wafer/Pipelines/Pipelines.h"

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
  wafer::registerAllDialects(registry);

  mlir::MLIRContext context(registry);
  context.loadDialect<wafer::WaferDialect, mlir::async::AsyncDialect,
                      mlir::memref::MemRefDialect>();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
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
      message = #wafer.dte_message<communication = 0, phase = collective_permute, round = 0, slice = 0>}
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
  EXPECT_EQ(wafer::classifyLocalInstructionCompletion(fill),
            wafer::LocalInstructionCompletion::OrderedPending);
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
  EXPECT_FALSE((
      hasMemoryEffect<mlir::MemoryEffects::Write, wafer::WaferComputeResource>(
          effects)));
  EXPECT_TRUE(hasValueMemoryEffect<mlir::MemoryEffects::Write>(
      effects, fill.getDest()));

  auto maskMove = findSingleOp<wafer::InstrMaskMoveOp>(*module);
  ASSERT_TRUE(maskMove);
  auto maskMoveInstruction = mlir::dyn_cast<wafer::WaferInstructionOpInterface>(
      maskMove.getOperation());
  ASSERT_TRUE(maskMoveInstruction);
  EXPECT_EQ(maskMoveInstruction.getInstructionFamily(), wafer::InstrFamily::CT);
  EXPECT_EQ(wafer::classifyLocalInstructionCompletion(maskMove),
            wafer::LocalInstructionCompletion::OrderedPending);
  auto maskMoveIssue =
      mlir::dyn_cast<wafer::WaferNCCIssueOpInterface>(maskMove.getOperation());
  ASSERT_TRUE(maskMoveIssue);
  EXPECT_EQ(maskMoveIssue.getIssueWorker(), wafer::NCCWorker::Worker2);
  effects.clear();
  mlir::cast<mlir::MemoryEffectOpInterface>(maskMove.getOperation())
      .getEffects(effects);
  EXPECT_TRUE((
      hasMemoryEffect<mlir::MemoryEffects::Write, wafer::WaferComputeResource>(
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

TEST(WaferInterfacesTest, TypedNCCCompletionContractsSeparateIssueAndJoin) {
  mlir::DialectRegistry registry;
  wafer::registerAllDialects(registry);
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

  EXPECT_EQ(wafer::classifyLocalInstructionCompletion(argmax),
            wafer::LocalInstructionCompletion::SynchronousWriteback);
  EXPECT_EQ(wafer::classifyLocalInstructionCompletion(argmin),
            wafer::LocalInstructionCompletion::SynchronousWriteback);
  EXPECT_EQ(wafer::classifyLocalInstructionCompletion(bilinear),
            wafer::LocalInstructionCompletion::OrderedPending);
  EXPECT_EQ(wafer::classifyLocalInstructionCompletion(multiWorkerJoin),
            wafer::LocalInstructionCompletion::ParticipantJoin);
  EXPECT_EQ(wafer::classifyLocalInstructionCompletion(workerZeroJoin),
            wafer::LocalInstructionCompletion::ParticipantJoin);

  wafer::NCCCompletionContract argmaxContract =
      wafer::getNCCCompletionContract(argmax);
  ASSERT_TRUE(argmaxContract.issueWorker);
  EXPECT_EQ(*argmaxContract.issueWorker, wafer::NCCWorker::Worker0);
  EXPECT_EQ(argmaxContract.participantMask, uint32_t{1});

  wafer::NCCCompletionContract joinContract =
      wafer::getNCCCompletionContract(multiWorkerJoin);
  EXPECT_FALSE(joinContract.issueWorker);
  EXPECT_EQ(joinContract.participantMask,
            (uint32_t{1} << 0) | (uint32_t{1} << 2));

  wafer::NCCCompletionContract workerZeroContract =
      wafer::getNCCCompletionContract(workerZeroJoin);
  EXPECT_FALSE(workerZeroContract.issueWorker);
  EXPECT_EQ(workerZeroContract.participantMask, uint32_t{1});
}

TEST(WaferInterfacesTest, LinalgExtCollectivesExposeLinalgExtStyleContracts) {
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
  %reduced = wafer.linalg_ext.collective.all_reduce
      ins(%input : tensor<4xf32>)
      outs(%out : tensor<4xf32>)
      {
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      wafer.linalg_ext.collective.yield %sum : f32
      } {channel_id = 7 : i64, rank_group = array<i64: 0, 1>}
      -> tensor<4xf32>

  %gathered_out = "builtin.unrealized_conversion_cast"() : () -> tensor<8xf32>
  %gathered = wafer.linalg_ext.collective.all_gather
      ins(%input : tensor<4xf32>)
      outs(%gathered_out : tensor<8xf32>)
      {axis = 0 : i64, channel_id = 9 : i64, rank_group = array<i64: 0, 1>}
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
         rank_group = array<i64: 0, 1>}
      -> tensor<4xf32>

  %matrix = "builtin.unrealized_conversion_cast"() : () -> tensor<4x2xf32>
  %matrix_out = "builtin.unrealized_conversion_cast"() : () -> tensor<2x4xf32>
  %a2a = wafer.linalg_ext.collective.all_to_all
      ins(%matrix : tensor<4x2xf32>)
      outs(%matrix_out : tensor<2x4xf32>)
      {split_axis = 0 : i64, concat_axis = 1 : i64,
       split_count = 2 : i64, channel_id = 11 : i64,
       rank_group = array<i64: 0, 1>}
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
  EXPECT_TRUE(llvm::equal(allReduce.getRankGroupAttr().asArrayRef(),
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
  EXPECT_TRUE(llvm::equal(allGather.getRankGroupAttr().asArrayRef(),
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
  EXPECT_TRUE(llvm::equal(reduceScatter.getRankGroupAttr().asArrayRef(),
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
  EXPECT_TRUE(llvm::equal(allToAll.getRankGroupAttr().asArrayRef(),
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
  %ar = wafer.linalg_ext.collective.all_reduce
      ins(%ar_input : tensor<1024x4096xf32>)
      outs(%ar_out : tensor<1024x4096xf32>)
      {
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      wafer.linalg_ext.collective.yield %sum : f32
      } {channel_id = 17 : i64, rank_group = array<i64: 0, 1, 2, 3, 4, 5, 6, 7>}
      -> tensor<1024x4096xf32>

  %ag_input = "builtin.unrealized_conversion_cast"() : () -> tensor<64x512xf32>
  %ag_out = "builtin.unrealized_conversion_cast"() : () -> tensor<64x4096xf32>
  %ag = wafer.linalg_ext.collective.all_gather
      ins(%ag_input : tensor<64x512xf32>)
      outs(%ag_out : tensor<64x4096xf32>)
      {axis = 1 : i64, channel_id = 18 : i64,
       rank_group = array<i64: 0, 1, 2, 3, 4, 5, 6, 7>}
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
         rank_group = array<i64: 0, 1, 2, 3>}
      -> tensor<256x1024xf32>

  %a2a_input = "builtin.unrealized_conversion_cast"() : () -> tensor<512x128x64xf32>
  %a2a_out = "builtin.unrealized_conversion_cast"() : () -> tensor<128x512x64xf32>
  %a2a = wafer.linalg_ext.collective.all_to_all
      ins(%a2a_input : tensor<512x128x64xf32>)
      outs(%a2a_out : tensor<128x512x64xf32>)
      {split_axis = 0 : i64, concat_axis = 1 : i64,
       split_count = 4 : i64, channel_id = 20 : i64,
       rank_group = array<i64: 0, 1, 2, 3>}
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
  %ar = wafer.linalg_ext.collective.all_reduce
      ins(%ar_input : tensor<?x4096xf32>)
      outs(%ar_out : tensor<?x4096xf32>)
      {
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      wafer.linalg_ext.collective.yield %sum : f32
      } {channel_id = 22 : i64, rank_group = array<i64: 0, 1, 2, 3>}
      -> tensor<?x4096xf32>

  %ag_input = "builtin.unrealized_conversion_cast"() : () -> tensor<?x512xf32>
  %ag_out = "builtin.unrealized_conversion_cast"() : () -> tensor<?x2048xf32>
  %ag = wafer.linalg_ext.collective.all_gather
      ins(%ag_input : tensor<?x512xf32>)
      outs(%ag_out : tensor<?x2048xf32>)
      {axis = 1 : i64, channel_id = 23 : i64,
       rank_group = array<i64: 0, 1, 2, 3>}
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
  EXPECT_TRUE(llvm::equal(allGather.getRankGroupAttr().asArrayRef(),
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
