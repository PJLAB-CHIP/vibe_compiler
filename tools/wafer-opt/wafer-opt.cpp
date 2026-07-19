//===- wafer-opt.cpp - Wafer optimizer driver ----------------------------===//

#include "Wafer/InitAll.h"
#include "Wafer/Pipelines/Pipelines.h"
#include "Wafer/Transforms/Passes.h"
#include "Wafer/Transforms/StructuredOptimization.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"

#ifdef WAFER_ENABLE_STABLEHLO
#include "stablehlo/dialect/Register.h"
#endif

#ifdef WAFER_ENABLE_SHARDY
#include "shardy/dialect/sdy/ir/register.h"
#include "shardy/dialect/sdy/transforms/passes.h"
#endif

namespace {

void registerWaferOptDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::arith::ArithDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::cf::ControlFlowDialect, mlir::func::FuncDialect,
                  mlir::LLVM::LLVMDialect, mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
  wafer::registerAllDialects(registry);
#ifdef WAFER_ENABLE_STABLEHLO
  mlir::stablehlo::registerAllDialects(registry);
#endif
#ifdef WAFER_ENABLE_SHARDY
  mlir::sdy::registerAllDialects(registry);
#endif
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerTilingInterfaceExternalModels(registry);
  mlir::func::registerInlinerExtension(registry);
  mlir::LLVM::registerInlinerInterface(registry);
}

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);
  mlir::DialectRegistry registry;
  registerWaferOptDialects(registry);
  mlir::registerTransformsPasses();
  mlir::arith::registerArithPasses();
  mlir::bufferization::registerBufferizationPasses();
  mlir::registerLinalgPasses();
  mlir::registerSCFPasses();
  mlir::tensor::registerTensorPasses();
  wafer::registerWaferTransformPasses();
  wafer::registerWaferPipelines();
#ifdef WAFER_ENABLE_SHARDY
  mlir::sdy::registerAllSdyPassesAndPipelines();
#endif

  std::string inputFilename;
  std::string outputFilename;
  std::tie(inputFilename, outputFilename) = mlir::registerAndParseCLIOptions(
      argc, argv, "Wafer optimizer driver\n", registry);

  auto baseConfig = std::make_shared<mlir::MlirOptMainConfig>(
      mlir::MlirOptMainConfig::createFromCLOptions());
  mlir::MlirOptMainConfig config = *baseConfig;
  config.setPassPipelineSetupFn(
      [baseConfig](mlir::PassManager &manager) -> mlir::LogicalResult {
        if (mlir::failed(baseConfig->setupPassPipeline(manager)))
          return mlir::failure();
        manager.addInstrumentation(
            wafer::createDebugOptimizationInvocationInstrumentation());
        return mlir::success();
      });

  auto input = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (!input) {
    llvm::errs() << "failed to open input: " << input.getError().message()
                 << '\n';
    return EXIT_FAILURE;
  }
  std::error_code outputError;
  llvm::ToolOutputFile output(outputFilename, outputError,
                              static_cast<llvm::sys::fs::OpenFlags>(0));
  if (outputError) {
    llvm::errs() << "failed to open output: " << outputError.message() << '\n';
    return EXIT_FAILURE;
  }
  mlir::LogicalResult result =
      mlir::MlirOptMain(output.os(), std::move(*input), registry, config);
  if (mlir::succeeded(result))
    output.keep();
  return mlir::asMainReturnCode(result);
}
