//===- SystemCTargetModelTestSupport.cpp - Source DTE test support --------===//

#include "SystemCTargetModelTestSupport.h"

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

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace wafer::model::test {
namespace {

frontend::ProgramBoundaryBinding partitionedBoundary(int64_t index) {
  frontend::ProgramBoundaryBinding binding;
  binding.index = index;
  binding.programIndex = index;
  binding.distribution = frontend::ProgramDistributionKind::Partitioned;
  binding.globalShape = {64};
  binding.localShape = {4};
  binding.dtype = "f32";
  for (int64_t rank = 0; rank < 16; ++rank) {
    frontend::ProgramRankSlice slice;
    slice.logicalRank = rank;
    slice.replicaId = 0;
    slice.offsets = {rank * 4};
    slice.sizes = {4};
    slice.strides = {1};
    binding.rankSlices.push_back(std::move(slice));
  }
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

std::vector<RawLogicalValue> makeRankValues(int64_t logicalRank) {
  std::vector<RawLogicalValue> values;
  values.reserve(4);
  for (uint64_t index = 0; index < 4; ++index) {
    // Exact finite f32 values in [1, 1.5), unique across the 64 elements.
    const uint64_t element = static_cast<uint64_t>(logicalRank) * 4 + index;
    values.push_back(
        {LogicalFormat::F32, UINT64_C(0x3f800000) + (element << 15)});
  }
  return values;
}

} // namespace

llvm::Expected<compiler::TargetLLVMModuleBundle>
buildDirectDTETargetBundle(std::string &diagnosticText) {
  auto context = createCompilerContext();
  auto tensorProgram = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @default {card_grid = array<i64: 1, 1>, card_interconnect = "mesh", tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh {axes = ["rank"], endpoints = array<i64>, policy = "all_available", shape = array<i64: 16>, topology = @default}
  func.func @main(%input: tensor<4xf32>) -> tensor<4xf32> {
    %out = tensor.empty() : tensor<4xf32>
    %permuted = wafer.linalg_ext.collective.collective_permute
        ins(%input : tensor<4xf32>) outs(%out : tensor<4xf32>)
        {source_target_pairs = array<i64: 0, 1, 1, 0, 2, 3, 3, 2,
                                          4, 5, 5, 4, 6, 7, 7, 6,
                                          8, 9, 9, 8, 10, 11, 11, 10,
                                          12, 13, 13, 12, 14, 15, 15, 14>,
         channel_id = 91 : i64} -> tensor<4xf32>
    return %permuted : tensor<4xf32>
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  if (!tensorProgram)
    return llvm::createStringError("failed to parse Direct-DTE model module");

  frontend::FrontendProgramVerificationResult program;
  program.logicalRankCount = 16;
  program.programUserInputCount = 1;
  program.distributedInputs = {partitionedBoundary(0)};
  program.distributedOutputs = {partitionedBoundary(0)};
  llvm::Expected<compiler::ExecutionConfig> config =
      compiler::ExecutionConfig::createForSingleCard(
          16, TargetProfileId::waferTx81SingleCardKernelV1(),
          TargetLaunchABIId::perRankPointerBlockV1());
  if (!config)
    return config.takeError();
  llvm::raw_string_ostream diagnostics(diagnosticText);
  llvm::Expected<compiler::ExecutableBundle> executable =
      compiler::detail::buildExecutableBundle(context, *tensorProgram,
                                              std::move(program), *config,
                                              diagnostics, std::nullopt);
  if (!executable)
    return executable.takeError();
  tensorProgram = nullptr;
  return compiler::compileExecutableBundleToTargetLLVMModules(*executable,
                                                              diagnostics);
}

llvm::Expected<DirectDTEInvocationData>
buildDirectDTEInvocationData(const compiler::TargetLLVMModuleBundle &bundle) {
  if (bundle.getModules().size() != 16)
    return llvm::createStringError(
        "Direct-DTE test bundle must contain exactly 16 ranks");

  DirectDTEInvocationData result;
  result.arguments.reserve(16);
  result.inputBytesByRank.resize(16);
  NumericTensorKey tensorKey = llvm::cantFail(NumericTensorKey::create(
      LogicalFormat::F32, NumericTensorLayout::Tensor, {4}));
  for (const compiler::TargetLLVMModule &module : bundle.getModules()) {
    const int64_t rank = module.getLogicalRank();
    if (rank < 0 || rank >= 16)
      return llvm::createStringError(
          "Direct-DTE test bundle rank is outside the canonical domain");
    compiler::TargetCallRankArguments arguments{rank, {}};
    size_t userInputCount = 0;
    for (const compiler::KernelABISlot &slot : module.getKernelABISlots()) {
      if (slot.ordinal < 0 || slot.byteSize <= 0 || slot.byteSize >= 0x10000)
        return llvm::createStringError(
            "Direct-DTE test slot is outside its synthetic DDR stride");
      const uint64_t base =
          UINT64_C(0x10000000) +
          static_cast<uint64_t>(rank) * UINT64_C(0x100000) +
          static_cast<uint64_t>(slot.ordinal) * UINT64_C(0x10000);
      arguments.slots.push_back(base);
      if (slot.role == compiler::KernelABISlotRole::UserInput) {
        ++userInputCount;
        llvm::Expected<std::vector<uint8_t>> bytes =
            packPhysicalTensorLogicalValues(tensorKey, makeRankValues(rank),
                                            UINT8_C(0));
        if (!bytes)
          return bytes.takeError();
        if (bytes->size() != static_cast<uint64_t>(slot.byteSize))
          return llvm::createStringError(
              "Direct-DTE test input bytes disagree with the typed ABI slot");
        result.inputBytesByRank[static_cast<size_t>(rank)] = *bytes;
        result.inputBindings.push_back({rank, slot.ordinal, std::move(*bytes)});
      } else if (slot.role == compiler::KernelABISlotRole::Parameter ||
                 slot.role == compiler::KernelABISlotRole::Constant) {
        return llvm::createStringError(
            "Direct-DTE source vertical unexpectedly gained a read-only slot");
      }
    }
    if (userInputCount != 1)
      return llvm::createStringError(
          "Direct-DTE source vertical must have one user input per rank");
    result.arguments.push_back(std::move(arguments));
  }
  return result;
}

} // namespace wafer::model::test
