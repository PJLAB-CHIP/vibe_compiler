//===- SystemCTargetModelOrientedGemmIntegrationTest.cpp ----------------===//

#include "Wafer/Model/SystemCTargetModel.h"

#include "Wafer/Compiler/TargetCallFrontend.h"
#include "Wafer/InitAll.h"
#include "Wafer/Target/PhysicalTensorCodec.h"

#include "../../lib/Wafer/Compiler/ExecutableBundleInternal.h"

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

#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
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

frontend::ProgramBoundaryBinding boundary(int64_t index) {
  frontend::ProgramBoundaryBinding binding;
  binding.index = index;
  binding.programIndex = index;
  binding.distribution = frontend::ProgramDistributionKind::Replicated;
  binding.globalShape = {2, 2};
  binding.localShape = {2, 2};
  binding.dtype = "f16";
  frontend::ProgramRankSlice slice;
  slice.logicalRank = 0;
  slice.replicaId = 0;
  slice.offsets = {0, 0};
  slice.sizes = {2, 2};
  slice.strides = {1, 1};
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

llvm::Error rewriteGemmAsOrientationChain(TargetLLVMModuleBundle &bundle) {
  if (bundle.getModules().size() != 1)
    return llvm::createStringError("expected one target module");
  TargetLLVMModule &targetModule =
      const_cast<TargetLLVMModule &>(bundle.getModules().front());
  llvm::Module &module = const_cast<llvm::Module &>(targetModule.getModule());
  llvm::Function *entry = module.getFunction(targetModule.getEntrySymbol());
  if (!entry)
    return llvm::createStringError("missing target entry");

  llvm::CallInst *gemm = nullptr;
  bool hasWdma = false;
  for (llvm::Instruction &instruction : llvm::instructions(entry)) {
    auto *call = llvm::dyn_cast<llvm::CallInst>(&instruction);
    if (!call || !call->getCalledFunction())
      continue;
    llvm::StringRef name = call->getCalledFunction()->getName();
    if (name == "wafer_tx81_gemm") {
      if (gemm)
        return llvm::createStringError("expected one GEMM call");
      gemm = call;
    } else if (name == "wafer_tx81_wdma") {
      hasWdma = true;
    }
  }
  if (!gemm || !hasWdma)
    return llvm::createStringError("missing GEMM or WDMA call");

  llvm::Value *lhs = gemm->getArgOperand(0);
  llvm::Value *rhs = gemm->getArgOperand(1);
  llvm::Value *destination = gemm->getArgOperand(2);

  llvm::IRBuilder<> builder(gemm);
  llvm::Type *i64 = builder.getInt64Ty();
  llvm::Type *i32 = builder.getInt32Ty();
  llvm::FunctionType *functionType = llvm::FunctionType::get(
      builder.getVoidTy(), {i64, i64, i64, i32, i32, i32, i32, i32, i32, i32},
      false);
  llvm::FunctionCallee oriented =
      module.getOrInsertFunction("wafer_tx81_gemm_oriented_v2", functionType);
  auto emit = [&](llvm::Value *callLhs, llvm::Value *callRhs,
                  llvm::Value *callDestination, GemmOrientation lhsOrientation,
                  GemmOrientation rhsOrientation) {
    builder.CreateCall(
        oriented, {callLhs, callRhs, callDestination, gemm->getArgOperand(3),
                   gemm->getArgOperand(4), gemm->getArgOperand(5),
                   gemm->getArgOperand(6), gemm->getArgOperand(7),
                   builder.getInt32(static_cast<uint32_t>(lhsOrientation)),
                   builder.getInt32(static_cast<uint32_t>(rhsOrientation))});
  };
  emit(lhs, rhs, destination, GemmOrientation::Normal, GemmOrientation::Normal);
  emit(lhs, rhs, destination, GemmOrientation::Normal,
       GemmOrientation::Transpose);
  emit(lhs, rhs, destination, GemmOrientation::Transpose,
       GemmOrientation::Normal);
  emit(lhs, rhs, destination, GemmOrientation::Transpose,
       GemmOrientation::Transpose);
  gemm->eraseFromParent();
  return llvm::Error::success();
}

llvm::Expected<TargetLLVMModuleBundle>
buildOrientedGemmTargetBundle(std::string &diagnosticText) {
  auto context = createCompilerContext();
  auto tensorProgram = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @default {card_grid = array<i64: 1, 1>, card_interconnect = "mesh", tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh {axes = ["rank"], endpoints = array<i64: 0, 0, 0, 0>, policy = "explicit", shape = array<i64: 1>, topology = @default}
  func.func @main(%lhs: tensor<2x2xf16>, %rhs: tensor<2x2xf16>) -> tensor<2x2xf16> {
    %empty = tensor.empty() : tensor<2x2xf16>
    %zero = arith.constant 0.0 : f16
    %init = linalg.fill ins(%zero : f16)
        outs(%empty : tensor<2x2xf16>) -> tensor<2x2xf16>
    %product = linalg.matmul
        ins(%lhs, %rhs : tensor<2x2xf16>, tensor<2x2xf16>)
        outs(%init : tensor<2x2xf16>) -> tensor<2x2xf16>
    return %product : tensor<2x2xf16>
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  if (!tensorProgram)
    return llvm::createStringError("failed to parse oriented GEMM module");

  frontend::FrontendProgramVerificationResult program;
  program.logicalRankCount = 1;
  program.programUserInputCount = 2;
  program.distributedInputs = {boundary(0), boundary(1)};
  program.distributedOutputs = {boundary(0)};
  llvm::Expected<ExecutionConfig> config = ExecutionConfig::createForSingleCard(
      1, TargetProfileId::waferTx81SingleCardKernelV2(),
      TargetLaunchABIId::perRankPointerBlockV1());
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
  llvm::Expected<TargetLLVMModuleBundle> bundle =
      compileExecutableBundleToTargetLLVMModules(*executable, diagnostics);
  if (!bundle)
    return bundle.takeError();
  if (llvm::Error error = rewriteGemmAsOrientationChain(*bundle))
    return std::move(error);
  return std::move(*bundle);
}

class RecordingTargetSink final : public TargetTransactionSink {
public:
  llvm::Error begin(const TargetCallInvocationDescriptor &) override {
    return llvm::Error::success();
  }
  llvm::Expected<uint64_t>
  issue(const TargetTransaction &transaction) override {
    transactions.push_back(transaction);
    return nextEvent++;
  }
  llvm::Error terminal(int64_t) override { return llvm::Error::success(); }
  llvm::Error prepareCommit() override { return llvm::Error::success(); }
  void commit() override {}
  void abort(llvm::StringRef) override {}

  uint64_t nextEvent = 1;
  std::vector<TargetTransaction> transactions;
};

TEST(SystemCTargetModelOrientedGemmIntegrationTest,
     ExecutesNNNTTNTTThroughVersionedTargetABI) {
  std::string diagnostics;
  llvm::Expected<TargetLLVMModuleBundle> bundle =
      buildOrientedGemmTargetBundle(diagnostics);
  ASSERT_TRUE(static_cast<bool>(bundle))
      << diagnostics << llvm::toString(bundle.takeError());
  const TargetLLVMModule &module = bundle->getModules().front();

  NumericTensorKey tensorKey = llvm::cantFail(NumericTensorKey::create(
      LogicalFormat::F16, NumericTensorLayout::Tensor, {2, 2}));
  const std::array<std::vector<RawLogicalValue>, 2> logicalInputs{
      {{{LogicalFormat::F16, UINT64_C(0x3c00)},
        {LogicalFormat::F16, UINT64_C(0x4000)},
        {LogicalFormat::F16, UINT64_C(0x4200)},
        {LogicalFormat::F16, UINT64_C(0x4400)}},
       {{LogicalFormat::F16, UINT64_C(0x3c00)},
        {LogicalFormat::F16, UINT64_C(0x4000)},
        {LogicalFormat::F16, UINT64_C(0)},
        {LogicalFormat::F16, UINT64_C(0x3c00)}}}};
  TargetCallRankArguments rankArguments{0, {}};
  std::vector<TargetModelInputBinding> inputs;
  size_t inputIndex = 0;
  for (const KernelABISlot &slot : module.getKernelABISlots()) {
    rankArguments.slots.push_back(UINT64_C(0x100000) +
                                  static_cast<uint64_t>(slot.ordinal) *
                                      UINT64_C(0x10000));
    if (slot.role != KernelABISlotRole::UserInput)
      continue;
    ASSERT_LT(inputIndex, logicalInputs.size());
    std::vector<uint8_t> bytes = llvm::cantFail(packPhysicalTensorLogicalValues(
        tensorKey, logicalInputs[inputIndex++], UINT8_C(0)));
    ASSERT_EQ(bytes.size(), static_cast<uint64_t>(slot.byteSize));
    inputs.push_back({0, slot.ordinal, std::move(bytes)});
  }
  ASSERT_EQ(inputIndex, logicalInputs.size());
  std::vector<TargetCallRankArguments> arguments{rankArguments};

  RecordingTargetSink recording;
  llvm::Expected<TargetCallExecutionResult> decoded =
      executeTargetCallFrontend(*bundle, arguments, recording);
  ASSERT_TRUE(static_cast<bool>(decoded))
      << llvm::toString(decoded.takeError());
  std::vector<std::pair<GemmOrientation, GemmOrientation>> orientations;
  for (const TargetTransaction &transaction : recording.transactions)
    if (const auto *gemm =
            std::get_if<TargetGemmTransaction>(&transaction.payload))
      orientations.emplace_back(gemm->lhsOrientation, gemm->rhsOrientation);
  EXPECT_EQ(orientations,
            (std::vector<std::pair<GemmOrientation, GemmOrientation>>{
                {GemmOrientation::Normal, GemmOrientation::Normal},
                {GemmOrientation::Normal, GemmOrientation::Transpose},
                {GemmOrientation::Transpose, GemmOrientation::Normal},
                {GemmOrientation::Transpose, GemmOrientation::Transpose}}));

  llvm::Expected<TargetCallExecutable> frontend =
      prepareTargetCallFrontend(*bundle, arguments);
  ASSERT_TRUE(static_cast<bool>(frontend))
      << llvm::toString(frontend.takeError());
  llvm::Expected<TargetModelResult> result = executeSystemCTargetModel(
      std::move(*frontend), inputs,
      TargetModelKernelBudget::create(
          FormalNumericWorkBudget::create(/*maximumScalarEvaluations=*/4096,
                                          /*maximumFusedMultiplyAdds=*/4096),
          /*maximumMovementBytes=*/4096,
          /*maximumMovementSegments=*/256));
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_EQ(result->outputs.size(), 1u);
  llvm::Expected<std::vector<RawLogicalValue>> output =
      unpackPhysicalTensorLogicalValues(tensorKey, result->outputs[0].bytes);
  ASSERT_TRUE(static_cast<bool>(output)) << llvm::toString(output.takeError());
  const std::array<uint64_t, 4> expected{UINT64_C(0x4700), UINT64_C(0x4200),
                                         UINT64_C(0x4900), UINT64_C(0x4400)};
  ASSERT_EQ(output->size(), expected.size());
  for (size_t index = 0; index < expected.size(); ++index)
    EXPECT_EQ((*output)[index].bits, expected[index]);
}

} // namespace

extern "C" int sc_main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
