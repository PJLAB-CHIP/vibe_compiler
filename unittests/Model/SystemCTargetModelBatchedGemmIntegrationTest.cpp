//===- SystemCTargetModelBatchedGemmIntegrationTest.cpp -----------------===//

#include "Wafer/Model/SystemCTargetModel.h"

#include "Wafer/Compiler/TargetCallFrontend.h"
#include "Wafer/InitAll.h"
#include "Wafer/Target/PhysicalTensorCodec.h"

#include "Wafer/Compiler/ExecutableBundleInternal.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "gtest/gtest.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace wafer;
using namespace wafer::compiler;
using namespace wafer::model;

frontend::ProgramBoundaryBinding
singleRankBoundary(int64_t index, llvm::ArrayRef<int64_t> shape) {
  frontend::ProgramBoundaryBinding binding;
  binding.index = index;
  binding.programIndex = index;
  binding.distribution = frontend::ProgramDistributionKind::Replicated;
  binding.globalShape.assign(shape.begin(), shape.end());
  binding.localShape.assign(shape.begin(), shape.end());
  binding.dtype = "f16";
  frontend::ProgramRankSlice slice;
  slice.logicalRank = 0;
  slice.replicaId = 0;
  slice.offsets.assign(shape.size(), 0);
  slice.sizes.assign(shape.begin(), shape.end());
  slice.strides.assign(shape.size(), 1);
  binding.rankSlices.push_back(std::move(slice));
  return binding;
}

std::shared_ptr<mlir::MLIRContext> createCompilerContext() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::cf::ControlFlowDialect, mlir::func::FuncDialect,
                  mlir::LLVM::LLVMDialect, mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
  registerAllDialects(registry);
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  return context;
}

llvm::Expected<TargetLLVMModuleBundle>
buildBatchedGemmTargetBundle(std::string &diagnosticText) {
  auto context = createCompilerContext();
  auto tensorProgram = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @default {card_grid = array<i64: 1, 1>, card_interconnect = "mesh", tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh {axes = ["rank"], endpoints = array<i64: 0, 0, 0, 0>, policy = "explicit", shape = array<i64: 1>, topology = @default}
  func.func @main(%lhs: tensor<2x1x128xf16>,
                  %rhs: tensor<2x128x1xf16>) -> tensor<2x1x1xf16> {
    %empty = tensor.empty() : tensor<2x1x1xf16>
    %zero = arith.constant 0.0 : f16
    %init = linalg.fill ins(%zero : f16)
        outs(%empty : tensor<2x1x1xf16>) -> tensor<2x1x1xf16>
    %product = linalg.batch_matmul
        ins(%lhs, %rhs : tensor<2x1x128xf16>, tensor<2x128x1xf16>)
        outs(%init : tensor<2x1x1xf16>) -> tensor<2x1x1xf16>
    return %product : tensor<2x1x1xf16>
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  if (!tensorProgram)
    return llvm::createStringError("failed to parse batched GEMM model module");

  frontend::FrontendProgramVerificationResult program;
  program.logicalRankCount = 1;
  program.programUserInputCount = 2;
  program.distributedInputs = {singleRankBoundary(0, {2, 1, 128}),
                               singleRankBoundary(1, {2, 128, 1})};
  program.distributedOutputs = {singleRankBoundary(0, {2, 1, 1})};
  llvm::Expected<ExecutionConfig> config = ExecutionConfig::createForSingleCard(
      1, TargetProfileId::waferTx81SingleCardKernelV1(),
      RuntimeLaunchKind::Kernel);
  if (!config)
    return config.takeError();
  llvm::raw_string_ostream diagnostics(diagnosticText);
  llvm::Expected<ExecutableBundle> executable =
      compiler::detail::buildExecutableBundle(context, *tensorProgram,
                                              std::move(program), *config,
                                              diagnostics, std::nullopt);
  if (!executable)
    return executable.takeError();
  tensorProgram = nullptr;
  return compileExecutableBundleToTargetLLVMModules(*executable, diagnostics);
}

class RecordingTargetSink final : public TargetTransactionSink {
public:
  llvm::Error begin(const TargetCallInvocationDescriptor &) override {
    began = true;
    return llvm::Error::success();
  }

  llvm::Expected<uint64_t>
  issue(const TargetTransaction &transaction) override {
    transactions.push_back(transaction);
    return nextEvent++;
  }

  llvm::Error terminal(int64_t) override { return llvm::Error::success(); }
  llvm::Error prepareCommit() override { return llvm::Error::success(); }
  void commit() override { committed = true; }
  void abort(llvm::StringRef) override { aborted = true; }

  bool began = false;
  bool committed = false;
  bool aborted = false;
  uint64_t nextEvent = 1;
  std::vector<TargetTransaction> transactions;
};

TEST(SystemCTargetModelBatchedGemmIntegrationTest,
     PreservesImplicitNCxStorageFromSourceThroughTargetModel) {
  std::string diagnostics;
  llvm::Expected<TargetLLVMModuleBundle> bundle =
      buildBatchedGemmTargetBundle(diagnostics);
  ASSERT_TRUE(static_cast<bool>(bundle))
      << diagnostics << llvm::toString(bundle.takeError());
  ASSERT_EQ(bundle->getModules().size(), 1u);

  const TargetLLVMModule &module = bundle->getModules().front();
  TargetCallRankArguments rankArguments{0, {}};
  std::vector<TargetModelInputBinding> inputs;
  NumericTensorKey lhsKey = llvm::cantFail(NumericTensorKey::create(
      LogicalFormat::F16, MemLayout::Tensor, {2, 1, 128}));
  NumericTensorKey rhsKey = llvm::cantFail(NumericTensorKey::create(
      LogicalFormat::F16, MemLayout::Tensor, {2, 128, 1}));
  NumericTensorKey outputKey = llvm::cantFail(NumericTensorKey::create(
      LogicalFormat::F16, MemLayout::Tensor, {2, 1, 1}));

  std::vector<RawLogicalValue> lhs(2 * 128, {LogicalFormat::F16, UINT64_C(0)});
  for (size_t k = 0; k < 128; ++k) {
    lhs[k].bits = k < 64 ? UINT64_C(0x3c00) : UINT64_C(0x4000);
    lhs[128 + k].bits = k < 64 ? UINT64_C(0x4200) : UINT64_C(0x4400);
  }
  std::vector<RawLogicalValue> rhs(2 * 128,
                                   {LogicalFormat::F16, UINT64_C(0x3c00)});
  const std::array<const NumericTensorKey *, 2> inputKeys{&lhsKey, &rhsKey};
  const std::array<llvm::ArrayRef<RawLogicalValue>, 2> logicalInputs{lhs, rhs};
  size_t inputIndex = 0;
  for (const KernelABISlot &slot : module.getKernelABISlots()) {
    const uint64_t base =
        UINT64_C(0x100000) +
        static_cast<uint64_t>(slot.ordinal) * UINT64_C(0x10000);
    rankArguments.slots.push_back(base);
    if (slot.role != KernelABISlotRole::UserInput)
      continue;
    ASSERT_LT(inputIndex, logicalInputs.size());
    std::vector<uint8_t> bytes = llvm::cantFail(packPhysicalTensorLogicalValues(
        *inputKeys[inputIndex], logicalInputs[inputIndex], UINT8_C(0)));
    ASSERT_EQ(bytes.size(), static_cast<uint64_t>(slot.byteSize));
    inputs.push_back({0, slot.ordinal, std::move(bytes)});
    ++inputIndex;
  }
  ASSERT_EQ(inputIndex, logicalInputs.size());
  std::vector<TargetCallRankArguments> arguments{std::move(rankArguments)};

  // Prove that candidate selection did not split the batch into batch-1 calls.
  // The numeric execution below uses a separately prepared frontend from the
  // same target module and ABI slot values.
  RecordingTargetSink recordingSink;
  llvm::Expected<TargetCallExecutionResult> decoded =
      executeTargetCallFrontend(*bundle, arguments, recordingSink);
  ASSERT_TRUE(static_cast<bool>(decoded))
      << llvm::toString(decoded.takeError());
  EXPECT_TRUE(recordingSink.began);
  EXPECT_TRUE(recordingSink.committed);
  EXPECT_FALSE(recordingSink.aborted);
  std::vector<const TargetGemmTransaction *> gemms;
  for (const TargetTransaction &transaction : recordingSink.transactions)
    if (const auto *gemm =
            std::get_if<TargetGemmTransaction>(&transaction.payload))
      gemms.push_back(gemm);
  ASSERT_EQ(gemms.size(), 1u);
  EXPECT_EQ(gemms.front()->batchCount, 2u);
  EXPECT_EQ(gemms.front()->m, 1u);
  EXPECT_EQ(gemms.front()->k, 128u);
  EXPECT_EQ(gemms.front()->n, 1u);

  llvm::Expected<TargetCallExecutable> frontend =
      prepareTargetCallFrontend(*bundle, arguments);
  ASSERT_TRUE(static_cast<bool>(frontend))
      << llvm::toString(frontend.takeError());
  llvm::Expected<TargetModelResult> result =
      executeSystemCTargetModel(std::move(*frontend), inputs,
                                TargetModelKernelBudget::create(
                                    FormalNumericWorkBudget::create(
                                        /*maximumScalarEvaluations=*/4096,
                                        /*maximumFusedMultiplyAdds=*/4096),
                                    /*maximumMovementBytes=*/UINT64_C(1) << 20,
                                    /*maximumMovementSegments=*/4096));
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_EQ(result->outputs.size(), 1u);
  llvm::Expected<std::vector<RawLogicalValue>> output =
      unpackPhysicalTensorLogicalValues(outputKey, result->outputs[0].bytes);
  ASSERT_TRUE(static_cast<bool>(output)) << llvm::toString(output.takeError());
  ASSERT_EQ(output->size(), 2u);
  // 64 * 1 + 64 * 2 = 192; 64 * 3 + 64 * 4 = 448. Both are
  // exactly representable in f16. A Cx/NCx block-order mismatch instead mixes
  // the two batches at the 64-channel boundary.
  EXPECT_EQ((*output)[0].bits, UINT64_C(0x5a00));
  EXPECT_EQ((*output)[1].bits, UINT64_C(0x5f00));
}

} // namespace

extern "C" int sc_main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
