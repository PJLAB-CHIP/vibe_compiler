//===- wafer-opt.cpp - Wafer optimizer driver ----------------------------===//

#include "Wafer/Frontend/InitImporterDialects.h"
#include "Wafer/Frontend/Program.h"
#include "Wafer/InitAll.h"
#include "Wafer/Pipelines/Pipelines.h"
#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <string>

#ifdef WAFER_ENABLE_SHARDY
#include "shardy/dialect/sdy/transforms/passes.h"
#endif

#ifndef WAFER_XLA_SPMD_PARTITIONER_HELPER
#define WAFER_XLA_SPMD_PARTITIONER_HELPER ""
#endif

namespace {

void registerWaferOptDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::arith::ArithDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
  wafer::registerAllDialects(registry);
  wafer::registerImporterDialects(registry);
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::func::registerInlinerExtension(registry);
}

bool hasWaferProgramPipelineRequest(int argc, char **argv) {
  for (int i = 1; i < argc; ++i) {
    llvm::StringRef arg(argv[i]);
    if (arg == "--program-pipeline" || arg.starts_with("--program-pipeline="))
      return true;
  }
  return false;
}

#if defined(WAFER_ENABLE_STABLEHLO) && defined(WAFER_ENABLE_SHARDY)
mlir::OwningOpRef<mlir::ModuleOp> parseModule(llvm::StringRef filename,
                                              mlir::MLIRContext &context) {
  mlir::ParserConfig config(&context);
  return mlir::parseSourceFile<mlir::ModuleOp>(filename, config);
}

mlir::OwningOpRef<mlir::ModuleOp> parseAndVerifyStableHLOProgramDir(
    llvm::StringRef programPath, mlir::MLIRContext &context,
    wafer::frontend::FrontendProgramVerificationResult *result = nullptr) {
  llvm::SmallString<256> mlirPath(programPath);
  llvm::sys::path::append(mlirPath, "functions", "forward.mlir");

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(mlirPath, context);
  if (!module)
    return {};
  if (mlir::failed(mlir::verify(*module)))
    return {};

  if (mlir::failed(wafer::frontend::verifyAndMaterializeStableHLOProgramDir(
          *module, programPath, llvm::errs(), result)))
    return {};

  return module;
}

std::string programFile(llvm::StringRef programPath,
                        llvm::ArrayRef<llvm::StringRef> components) {
  llvm::SmallString<256> path(programPath);
  for (llvm::StringRef component : components)
    llvm::sys::path::append(path, component);
  return path.str().str();
}

bool createDirectory(llvm::StringRef path) {
  if (std::error_code error = llvm::sys::fs::create_directories(path)) {
    llvm::errs() << "wafer-opt: failed to create directory '" << path
                 << "': " << error.message() << "\n";
    return true;
  }
  return false;
}

bool copyFile(llvm::StringRef from, llvm::StringRef to) {
  if (std::error_code error = llvm::sys::fs::copy_file(from, to)) {
    llvm::errs() << "wafer-opt: failed to copy '" << from << "' to '" << to
                 << "': " << error.message() << "\n";
    return true;
  }
  return false;
}

bool copyDirectoryIfPresent(llvm::StringRef from, llvm::StringRef to) {
  if (!llvm::sys::fs::is_directory(from))
    return false;

  if (createDirectory(to))
    return true;

  std::error_code error;
  for (llvm::sys::fs::recursive_directory_iterator it(from, error), end;
       it != end; it.increment(error)) {
    if (error) {
      llvm::errs() << "wafer-opt: failed to walk directory '" << from
                   << "': " << error.message() << "\n";
      return true;
    }

    llvm::StringRef source = it->path();
    llvm::StringRef relative = source;
    relative.consume_front(from);
    if (relative.starts_with(llvm::sys::path::get_separator()))
      relative = relative.drop_front();

    llvm::SmallString<256> destination(to);
    llvm::sys::path::append(destination, relative);

    llvm::sys::fs::file_status status;
    if (std::error_code statusError = llvm::sys::fs::status(source, status)) {
      llvm::errs() << "wafer-opt: failed to stat '" << source
                   << "': " << statusError.message() << "\n";
      return true;
    }
    if (llvm::sys::fs::is_directory(status)) {
      if (createDirectory(destination))
        return true;
      continue;
    }
    if (llvm::sys::fs::is_regular_file(status)) {
      llvm::SmallString<256> parent(destination);
      llvm::sys::path::remove_filename(parent);
      if (createDirectory(parent))
        return true;
      if (copyFile(source, destination))
        return true;
    }
  }

  if (error) {
    llvm::errs() << "wafer-opt: failed to walk directory '" << from
                 << "': " << error.message() << "\n";
    return true;
  }
  return false;
}

bool writeProgramModule(mlir::ModuleOp module, llvm::StringRef programDir) {
  std::string functionsDir =
      programFile(programDir, {llvm::StringRef("functions")});
  if (createDirectory(functionsDir))
    return true;

  std::string mlirPath =
      programFile(programDir, {llvm::StringRef("functions"),
                               llvm::StringRef("forward.mlir")});
  std::error_code error;
  llvm::raw_fd_ostream os(mlirPath, error, llvm::sys::fs::OF_Text);
  if (error) {
    llvm::errs() << "wafer-opt: failed to write '" << mlirPath
                 << "': " << error.message() << "\n";
    return true;
  }
  module->print(os);
  os << "\n";
  os.close();
  if (os.has_error()) {
    llvm::errs() << "wafer-opt: failed to close '" << mlirPath << "'\n";
    return true;
  }

  return false;
}

bool stageProgramWithPropagatedModule(mlir::ModuleOp module,
                                      llvm::StringRef inputProgramDir,
                                      llvm::StringRef stagedProgramDir) {
  if (copyDirectoryIfPresent(inputProgramDir, stagedProgramDir))
    return true;
  return writeProgramModule(module, stagedProgramDir);
}

struct WaferProgramPipelineOptions {
  std::string pipelineName;
  std::string inputProgramDir;
  std::string outputProgramDir;
  int64_t defaultTileCount = 16;
};

bool parseIntegerOption(llvm::StringRef optionName, llvm::StringRef value,
                        int64_t &result) {
  if (value.getAsInteger(10, result)) {
    llvm::errs() << "wafer-opt: invalid " << optionName << " value: " << value
                 << "\n";
    return true;
  }
  return false;
}

bool parseWaferProgramPipelineOptions(int argc, char **argv,
                                      WaferProgramPipelineOptions &options) {
  for (int i = 1; i < argc; ++i) {
    llvm::StringRef arg(argv[i]);
    if (arg == "--program-pipeline") {
      if (i + 1 >= argc) {
        llvm::errs() << "wafer-opt: missing --program-pipeline value\n";
        return true;
      }
      options.pipelineName = argv[++i];
      continue;
    }

    constexpr llvm::StringRef programPipelinePrefix = "--program-pipeline=";
    if (arg.starts_with(programPipelinePrefix)) {
      options.pipelineName = arg.drop_front(programPipelinePrefix.size()).str();
      continue;
    }

    if (arg == "--input-program-dir") {
      if (i + 1 >= argc) {
        llvm::errs() << "wafer-opt: missing --input-program-dir value\n";
        return true;
      }
      options.inputProgramDir = argv[++i];
      continue;
    }

    constexpr llvm::StringRef inputProgramDirPrefix = "--input-program-dir=";
    if (arg.starts_with(inputProgramDirPrefix)) {
      options.inputProgramDir =
          arg.drop_front(inputProgramDirPrefix.size()).str();
      continue;
    }

    if (arg == "--output-program-dir") {
      if (i + 1 >= argc) {
        llvm::errs() << "wafer-opt: missing --output-program-dir value\n";
        return true;
      }
      options.outputProgramDir = argv[++i];
      continue;
    }

    constexpr llvm::StringRef outputProgramDirPrefix = "--output-program-dir=";
    if (arg.starts_with(outputProgramDirPrefix)) {
      options.outputProgramDir =
          arg.drop_front(outputProgramDirPrefix.size()).str();
      continue;
    }

    if (arg == "--default-tile-count") {
      if (i + 1 >= argc) {
        llvm::errs() << "wafer-opt: missing --default-tile-count value\n";
        return true;
      }
      if (parseIntegerOption("--default-tile-count", argv[++i],
                             options.defaultTileCount))
        return true;
      continue;
    }

    constexpr llvm::StringRef defaultTileCountPrefix = "--default-tile-count=";
    if (arg.starts_with(defaultTileCountPrefix)) {
      if (parseIntegerOption("--default-tile-count",
                             arg.drop_front(defaultTileCountPrefix.size()),
                             options.defaultTileCount))
        return true;
      continue;
    }

    llvm::errs() << "wafer-opt: unknown program pipeline argument: " << arg
                 << "\n";
    return true;
  }

  if (options.pipelineName.empty()) {
    llvm::errs() << "wafer-opt: --program-pipeline requires a pipeline name\n";
    return true;
  }
  if (options.inputProgramDir.empty()) {
    llvm::errs() << "wafer-opt: --program-pipeline requires "
                    "--input-program-dir\n";
    return true;
  }
  if (options.outputProgramDir.empty()) {
    llvm::errs() << "wafer-opt: --program-pipeline requires "
                    "--output-program-dir\n";
    return true;
  }
  if (options.pipelineName != "stablehlo-spmd" &&
      options.pipelineName != "stablehlo-spmd-to-linalg" &&
      options.pipelineName != "stablehlo-spmd-to-group") {
    llvm::errs() << "wafer-opt: unknown Wafer program pipeline: "
                 << options.pipelineName << "\n";
    return true;
  }

  return false;
}

std::string resolveSpmdPartitionerHelperPath() {
  if (const char *envPath = std::getenv("WAFER_XLA_SPMD_PARTITIONER_HELPER"))
    return envPath;
  return WAFER_XLA_SPMD_PARTITIONER_HELPER;
}

int runStableHLOSPMDStage(llvm::StringRef inputProgramDir,
                          llvm::StringRef outputProgramDir,
                          llvm::StringRef helperPath, int64_t defaultTileCount,
                          mlir::MLIRContext &context) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      parseAndVerifyStableHLOProgramDir(inputProgramDir, context);
  if (!module)
    return 1;

  mlir::PassManager pm(&context);
  wafer::buildStablehloShardingPropagationPipeline(pm, defaultTileCount);
  if (mlir::failed(pm.run(*module)))
    return 1;

  llvm::SmallString<256> tempPrefix(outputProgramDir);
  llvm::sys::path::remove_filename(tempPrefix);
  if (tempPrefix.empty())
    tempPrefix = ".";
  llvm::sys::path::append(tempPrefix, "wafer-spmd-propagated");

  llvm::SmallString<256> stagedProgramDir;
  if (std::error_code error =
          llvm::sys::fs::createUniqueDirectory(tempPrefix, stagedProgramDir)) {
    llvm::errs() << "wafer-opt: failed to create temporary propagated program "
                    "directory: "
                 << error.message() << "\n";
    return 1;
  }

  if (stageProgramWithPropagatedModule(*module, inputProgramDir,
                                       stagedProgramDir)) {
    llvm::sys::fs::remove_directories(stagedProgramDir);
    return 1;
  }

  std::string helper = helperPath.str();
  std::string staged = stagedProgramDir.str().str();
  std::string output = outputProgramDir.str();
  std::string logicalRankCount = std::to_string(defaultTileCount);
  llvm::SmallVector<llvm::StringRef, 8> args = {
      helper,           "--input-program-dir",
      staged,           "--output-program-dir",
      output,           "--entry-function",
      "forward",        "--logical-rank-count",
      logicalRankCount,
  };
  int exitCode = llvm::sys::ExecuteAndWait(helper, args);
  llvm::sys::fs::remove_directories(stagedProgramDir);
  if (exitCode != 0) {
    llvm::errs() << "wafer-opt: XLA SPMD partition helper failed";
    if (exitCode > 0)
      llvm::errs() << " with exit code " << exitCode;
    llvm::errs() << "\n";
    return 1;
  }

  wafer::frontend::FrontendProgramVerificationResult result;
  mlir::OwningOpRef<mlir::ModuleOp> outputModule =
      parseAndVerifyStableHLOProgramDir(outputProgramDir, context, &result);
  if (!outputModule)
    return 1;
  if (writeProgramModule(*outputModule, outputProgramDir))
    return 1;

  return 0;
}

int runStableHLOToLinalgStage(llvm::StringRef programDir,
                              mlir::MLIRContext &context) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      parseAndVerifyStableHLOProgramDir(programDir, context);
  if (!module)
    return 1;

  mlir::PassManager pm(&context);
  wafer::buildStablehloToLinalgPipeline(pm);
  if (mlir::failed(pm.run(*module)))
    return 1;

  if (writeProgramModule(*module, programDir))
    return 1;

  mlir::OwningOpRef<mlir::ModuleOp> loweredModule =
      parseAndVerifyStableHLOProgramDir(programDir, context);
  if (!loweredModule)
    return 1;

  return 0;
}

int runFormLogicalGroupsStage(llvm::StringRef programDir,
                              mlir::MLIRContext &context) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      parseAndVerifyStableHLOProgramDir(programDir, context);
  if (!module)
    return 1;

  mlir::PassManager pm(&context);
  wafer::buildFormLogicalGroupsPipeline(pm);
  if (mlir::failed(pm.run(*module)))
    return 1;

  if (writeProgramModule(*module, programDir))
    return 1;

  mlir::OwningOpRef<mlir::ModuleOp> groupedModule =
      parseAndVerifyStableHLOProgramDir(programDir, context);
  if (!groupedModule)
    return 1;

  return 0;
}
#endif

int runWaferProgramPipeline(int argc, char **argv) {
#ifndef WAFER_ENABLE_STABLEHLO
  llvm::errs()
      << "wafer-opt: StableHLO frontend dependencies are disabled in this "
         "build\n";
  return 1;
#elif !defined(WAFER_ENABLE_SHARDY)
  llvm::errs() << "wafer-opt: SPMD partitioner dependencies are disabled in "
                  "this build\n";
  return 1;
#else
  WaferProgramPipelineOptions options;
  if (parseWaferProgramPipelineOptions(argc, argv, options))
    return 1;

  mlir::DialectRegistry registry;
  registerWaferOptDialects(registry);

  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  std::string spmdPartitionerHelperPath = resolveSpmdPartitionerHelperPath();
  if (spmdPartitionerHelperPath.empty()) {
    llvm::errs() << "wafer-opt: no XLA SPMD partitioner helper configured; "
                    "set WAFER_XLA_SPMD_PARTITIONER_HELPER at CMake "
                    "configure time\n";
    return 1;
  }

  if (runStableHLOSPMDStage(options.inputProgramDir, options.outputProgramDir,
                            spmdPartitionerHelperPath, options.defaultTileCount,
                            context))
    return 1;

  bool needsLinalgStage = options.pipelineName == "stablehlo-spmd-to-linalg" ||
                          options.pipelineName == "stablehlo-spmd-to-group";
  if (needsLinalgStage &&
      runStableHLOToLinalgStage(options.outputProgramDir, context))
    return 1;

  if (options.pipelineName == "stablehlo-spmd-to-group" &&
      runFormLogicalGroupsStage(options.outputProgramDir, context))
    return 1;

  llvm::outs() << "wafer-opt: completed Wafer program pipeline "
               << options.pipelineName << ": " << options.outputProgramDir
               << "\n";
  return 0;
#endif
}

} // namespace

int main(int argc, char **argv) {
  if (hasWaferProgramPipelineRequest(argc, argv))
    return runWaferProgramPipeline(argc, argv);

  mlir::DialectRegistry registry;
  registerWaferOptDialects(registry);
  mlir::registerTransformsPasses();
  wafer::registerWaferTransformPasses();
  wafer::registerWaferPipelines();
#ifdef WAFER_ENABLE_SHARDY
  mlir::sdy::registerAllSdyPassesAndPipelines();
#endif

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Wafer optimizer driver\n", registry));
}
