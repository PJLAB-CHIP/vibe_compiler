//===- wafer-compile-stablehlo.cpp - Wafer StableHLO compiler entry -------===//

#ifdef WAFER_ENABLE_STABLEHLO
#include "Wafer/Frontend/InitImporterDialects.h"
#include "Wafer/Frontend/Program.h"
#include "Wafer/InitAll.h"

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
#endif

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace {

void printHelp() {
  llvm::outs() << "wafer-compile-stablehlo\n";
#ifdef WAFER_ENABLE_STABLEHLO
  llvm::outs() << "  --verify-frontend-program <mlir-file>\n"
               << "  --verify-stablehlo-program <program-dir>\n";
#else
  llvm::outs()
      << "  StableHLO frontend dependencies are disabled in this build\n";
#endif
}

#ifdef WAFER_ENABLE_STABLEHLO
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

int verifyFrontendProgramFile(llvm::StringRef filename) {
  mlir::DialectRegistry registry;
  registerToolDialects(registry);

  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(filename, context);
  if (!module)
    return 1;
  if (mlir::failed(mlir::verify(*module)))
    return 1;

  wafer::frontend::FrontendProgramVerificationResult result;
  if (mlir::failed(wafer::frontend::verifyFrontendProgram(*module, llvm::errs(),
                                                          &result)))
    return 1;

  llvm::outs() << "wafer-compile-stablehlo: verified frontend program\n";
  return 0;
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

int verifyStableHLOProgramDir(llvm::StringRef programPath) {
  mlir::DialectRegistry registry;
  registerToolDialects(registry);

  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  wafer::frontend::FrontendProgramVerificationResult result;
  mlir::OwningOpRef<mlir::ModuleOp> module =
      parseAndVerifyStableHLOProgramDir(programPath, context, &result);
  if (!module)
    return 1;

  llvm::outs() << "wafer-compile-stablehlo: verified StableHLO program "
                  "directory parameters: "
               << result.programParameterCount << "\n";
  if (result.programConstantCount)
    llvm::outs()
        << "wafer-compile-stablehlo: verified StableHLO program directory "
           "constants: "
        << result.programConstantCount << "\n";
  if (result.programParameterShardCount)
    llvm::outs()
        << "wafer-compile-stablehlo: verified StableHLO program directory "
           "parameter shards: "
        << result.programParameterShardCount << "\n";
  llvm::outs() << "wafer-compile-stablehlo: verified StableHLO program\n";
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
      << "wafer-compile-stablehlo: StableHLO frontend dependencies are "
         "disabled in this build\n";
  return 1;
#else
  std::string verifyFilename;
  std::string programPath;
  for (int i = 1; i < argc; ++i) {
    llvm::StringRef arg(argv[i]);
    if (arg == "--verify-frontend-program") {
      if (i + 1 >= argc) {
        llvm::errs()
            << "wafer-compile-stablehlo: missing --verify-frontend-program "
               "filename\n";
        return 1;
      }
      verifyFilename = argv[++i];
      continue;
    }

    constexpr llvm::StringRef verifyPrefix = "--verify-frontend-program=";
    if (arg.starts_with(verifyPrefix)) {
      verifyFilename = arg.drop_front(verifyPrefix.size()).str();
      continue;
    }

    if (arg == "--verify-stablehlo-program") {
      if (i + 1 >= argc) {
        llvm::errs() << "wafer-compile-stablehlo: missing "
                        "--verify-stablehlo-program directory\n";
        return 1;
      }
      programPath = argv[++i];
      continue;
    }

    constexpr llvm::StringRef programPrefix = "--verify-stablehlo-program=";
    if (arg.starts_with(programPrefix)) {
      programPath = arg.drop_front(programPrefix.size()).str();
      continue;
    }

    llvm::errs() << "wafer-compile-stablehlo: unknown argument: " << arg
                 << "\n";
    printHelp();
    return 1;
  }

  unsigned actionCount = 0;
  if (!verifyFilename.empty())
    ++actionCount;
  if (!programPath.empty())
    ++actionCount;
  if (actionCount > 1) {
    llvm::errs() << "wafer-compile-stablehlo: choose exactly one action\n";
    printHelp();
    return 1;
  }

  if (!verifyFilename.empty())
    return verifyFrontendProgramFile(verifyFilename);
  if (!programPath.empty())
    return verifyStableHLOProgramDir(programPath);
  llvm::errs() << "wafer-compile-stablehlo: unknown or incomplete arguments\n";
  printHelp();
  return 1;
#endif
}
