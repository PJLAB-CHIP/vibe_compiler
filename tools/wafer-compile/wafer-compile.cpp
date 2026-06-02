//===- wafer-compile.cpp - Wafer compiler driver -------------------------===//

#ifdef WAFER_ENABLE_STABLEHLO
#include "Wafer/Frontend/InitImporterDialects.h"
#include "Wafer/Frontend/Program.h"
#include "Wafer/InitAll.h"
#include "Wafer/Pipelines/Pipelines.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#endif

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>

namespace {

void printHelp() {
  llvm::outs() << "wafer-compile\n";
#ifndef WAFER_ENABLE_STABLEHLO
  llvm::outs()
      << "  StableHLO frontend dependencies are disabled in this build\n";
#elif !defined(WAFER_ENABLE_SHARDY)
  llvm::outs()
      << "  SPMD partitioner dependencies are disabled in this build\n";
#else
  llvm::outs() << "  --partition-stablehlo-program <program-dir>\n"
               << "    --output-program-dir <program-dir>\n"
               << "    --xla-spmd-partitioner-helper <path>\n"
               << "    [--default-tile-count=16]\n";
#endif
}

#if defined(WAFER_ENABLE_STABLEHLO) && defined(WAFER_ENABLE_SHARDY)
mlir::OwningOpRef<mlir::ModuleOp> parseModule(llvm::StringRef filename,
                                              mlir::MLIRContext &context) {
  mlir::ParserConfig config(&context);
  return mlir::parseSourceFile<mlir::ModuleOp>(filename, config);
}

void registerToolDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect, mlir::math::MathDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
  wafer::registerAllDialects(registry);
  wafer::registerImporterDialects(registry);
  mlir::func::registerInlinerExtension(registry);
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

  if (mlir::failed(wafer::frontend::verifyStableHLOProgramDir(
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
    llvm::errs() << "wafer-compile: failed to create directory '" << path
                 << "': " << error.message() << "\n";
    return true;
  }
  return false;
}

bool copyFile(llvm::StringRef from, llvm::StringRef to) {
  if (std::error_code error = llvm::sys::fs::copy_file(from, to)) {
    llvm::errs() << "wafer-compile: failed to copy '" << from << "' to '" << to
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
      llvm::errs() << "wafer-compile: failed to walk directory '" << from
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
      llvm::errs() << "wafer-compile: failed to stat '" << source
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
    llvm::errs() << "wafer-compile: failed to walk directory '" << from
                 << "': " << error.message() << "\n";
    return true;
  }
  return false;
}

bool writePropagatedProgramDir(mlir::ModuleOp module,
                               llvm::StringRef inputProgramDir,
                               llvm::StringRef stagedProgramDir) {
  std::string stagedFunctions =
      programFile(stagedProgramDir, {llvm::StringRef("functions")});
  if (createDirectory(stagedFunctions))
    return true;

  std::string stagedMlir =
      programFile(stagedProgramDir, {llvm::StringRef("functions"),
                                     llvm::StringRef("forward.mlir")});
  std::error_code error;
  llvm::raw_fd_ostream os(stagedMlir, error, llvm::sys::fs::OF_Text);
  if (error) {
    llvm::errs() << "wafer-compile: failed to write '" << stagedMlir
                 << "': " << error.message() << "\n";
    return true;
  }
  module->print(os);
  os << "\n";
  os.close();
  if (os.has_error()) {
    llvm::errs() << "wafer-compile: failed to close '" << stagedMlir << "'\n";
    return true;
  }

  if (copyFile(
          programFile(inputProgramDir, {llvm::StringRef("functions"),
                                        llvm::StringRef("forward.meta")}),
          programFile(stagedProgramDir, {llvm::StringRef("functions"),
                                         llvm::StringRef("forward.meta")})))
    return true;

  return copyDirectoryIfPresent(
      programFile(inputProgramDir, {llvm::StringRef("data")}),
      programFile(stagedProgramDir, {llvm::StringRef("data")}));
}

int partitionStableHLOProgram(llvm::StringRef programPath,
                              llvm::StringRef outputProgramDirPath,
                              llvm::StringRef helperPath,
                              int64_t defaultTileCount) {
  if (outputProgramDirPath.empty()) {
    llvm::errs() << "wafer-compile: --partition-stablehlo-program requires "
                    "--output-program-dir\n";
    return 1;
  }
  if (helperPath.empty()) {
    llvm::errs() << "wafer-compile: --partition-stablehlo-program requires "
                    "--xla-spmd-partitioner-helper\n";
    return 1;
  }

  mlir::DialectRegistry registry;
  registerToolDialects(registry);

  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      parseAndVerifyStableHLOProgramDir(programPath, context);
  if (!module)
    return 1;

  mlir::PassManager pm(&context);
  wafer::buildStablehloShardingPropagationPipeline(pm, defaultTileCount);
  if (mlir::failed(pm.run(*module)))
    return 1;

  llvm::SmallString<256> tempPrefix(outputProgramDirPath);
  llvm::sys::path::remove_filename(tempPrefix);
  if (tempPrefix.empty())
    tempPrefix = ".";
  llvm::sys::path::append(tempPrefix, "wafer-spmd-propagated");

  llvm::SmallString<256> stagedProgramDir;
  if (std::error_code error =
          llvm::sys::fs::createUniqueDirectory(tempPrefix, stagedProgramDir)) {
    llvm::errs()
        << "wafer-compile: failed to create temporary propagated program "
           "directory: "
        << error.message() << "\n";
    return 1;
  }

  if (writePropagatedProgramDir(*module, programPath, stagedProgramDir)) {
    llvm::sys::fs::remove_directories(stagedProgramDir);
    return 1;
  }

  std::string helper = helperPath.str();
  std::string staged = stagedProgramDir.str().str();
  std::string output = outputProgramDirPath.str();
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
    llvm::errs() << "wafer-compile: XLA SPMD partition helper failed";
    if (exitCode > 0)
      llvm::errs() << " with exit code " << exitCode;
    llvm::errs() << "\n";
    return 1;
  }

  wafer::frontend::FrontendProgramVerificationResult result;
  mlir::OwningOpRef<mlir::ModuleOp> outputModule =
      parseAndVerifyStableHLOProgramDir(outputProgramDirPath, context, &result);
  if (!outputModule)
    return 1;

  llvm::outs() << "wafer-compile: partitioned StableHLO program directory "
                  "written: "
               << outputProgramDirPath << "\n";
  return 0;
}
#endif

} // namespace

int main(int argc, char **argv) {
  if (argc == 2 && std::string(argv[1]) == "--help") {
    printHelp();
    return 0;
  }

#ifndef WAFER_ENABLE_STABLEHLO
  llvm::errs()
      << "wafer-compile: StableHLO frontend dependencies are disabled in this "
         "build\n";
  return 1;
#elif !defined(WAFER_ENABLE_SHARDY)
  llvm::errs()
      << "wafer-compile: SPMD partitioner dependencies are disabled in this "
         "build\n";
  return 1;
#else
  std::string partitionProgramDir;
  std::string partitionOutputProgramDir;
  std::string spmdPartitionerHelperPath;
  int64_t defaultTileCount = 16;
  for (int i = 1; i < argc; ++i) {
    llvm::StringRef arg(argv[i]);
    if (arg == "--partition-stablehlo-program") {
      if (i + 1 >= argc) {
        llvm::errs() << "wafer-compile: missing --partition-stablehlo-program "
                        "directory\n";
        return 1;
      }
      partitionProgramDir = argv[++i];
      continue;
    }

    constexpr llvm::StringRef partitionProgramPrefix =
        "--partition-stablehlo-program=";
    if (arg.starts_with(partitionProgramPrefix)) {
      partitionProgramDir = arg.drop_front(partitionProgramPrefix.size()).str();
      continue;
    }

    if (arg == "--output-program-dir") {
      if (i + 1 >= argc) {
        llvm::errs() << "wafer-compile: missing --output-program-dir value\n";
        return 1;
      }
      partitionOutputProgramDir = argv[++i];
      continue;
    }

    constexpr llvm::StringRef outputProgramDirPrefix = "--output-program-dir=";
    if (arg.starts_with(outputProgramDirPrefix)) {
      partitionOutputProgramDir =
          arg.drop_front(outputProgramDirPrefix.size()).str();
      continue;
    }

    if (arg == "--xla-spmd-partitioner-helper") {
      if (i + 1 >= argc) {
        llvm::errs()
            << "wafer-compile: missing --xla-spmd-partitioner-helper value\n";
        return 1;
      }
      spmdPartitionerHelperPath = argv[++i];
      continue;
    }

    constexpr llvm::StringRef spmdHelperPrefix =
        "--xla-spmd-partitioner-helper=";
    if (arg.starts_with(spmdHelperPrefix)) {
      spmdPartitionerHelperPath = arg.drop_front(spmdHelperPrefix.size()).str();
      continue;
    }

    if (arg == "--default-tile-count") {
      if (i + 1 >= argc) {
        llvm::errs() << "wafer-compile: missing --default-tile-count value\n";
        return 1;
      }
      llvm::StringRef value(argv[++i]);
      if (value.getAsInteger(10, defaultTileCount)) {
        llvm::errs() << "wafer-compile: invalid --default-tile-count value: "
                     << value << "\n";
        return 1;
      }
      continue;
    }

    constexpr llvm::StringRef defaultTileCountPrefix = "--default-tile-count=";
    if (arg.starts_with(defaultTileCountPrefix)) {
      llvm::StringRef value = arg.drop_front(defaultTileCountPrefix.size());
      if (value.getAsInteger(10, defaultTileCount)) {
        llvm::errs() << "wafer-compile: invalid --default-tile-count value: "
                     << value << "\n";
        return 1;
      }
      continue;
    }

    llvm::errs() << "wafer-compile: unknown argument: " << arg << "\n";
    printHelp();
    return 1;
  }

  if (!partitionProgramDir.empty())
    return partitionStableHLOProgram(
        partitionProgramDir, partitionOutputProgramDir,
        spmdPartitionerHelperPath, defaultTileCount);

  llvm::errs() << "wafer-compile: unknown or incomplete arguments\n";
  printHelp();
  return 1;
#endif
}
