//===- wafer-import-model.cpp - Wafer importer smoke tool ----------------===//

#ifdef WAFER_ENABLE_STABLEHLO
#include "Wafer/Frontend/Artifact.h"
#include "Wafer/Frontend/InitImporterDialects.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#endif

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace {

void printHelp() {
  llvm::outs() << "wafer-import-model\n";
#ifdef WAFER_ENABLE_STABLEHLO
  llvm::outs() << "  --emit-static-smoke-artifact\n"
               << "  --verify-import-result <mlir-file> [--sidecar "
                  "<sidecar-json>]\n";
#else
  llvm::outs() << "  importer dependencies are disabled in this build\n";
#endif
}

#ifdef WAFER_ENABLE_STABLEHLO
void emitStaticSmokeArtifact() {
  llvm::outs()
      << "module {\n"
      << "  func.func @wafer_import_static_smoke(%arg0: tensor<2x4xf32>, "
         "%arg1: tensor<2x4xf32>) -> tensor<2x4xf32> {\n"
      << "    %0 = stablehlo.add %arg0, %arg1 : tensor<2x4xf32>\n"
      << "    return %0 : tensor<2x4xf32>\n"
      << "  }\n"
      << "}\n";
}

int verifyImportResult(llvm::StringRef filename, llvm::StringRef sidecarPath) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect>();
  wafer::registerImporterDialects(registry);

  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  mlir::ParserConfig config(&context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(filename, config);
  if (!module)
    return 1;
  if (mlir::failed(mlir::verify(*module)))
    return 1;

  wafer::frontend::ArtifactVerificationResult result;
  if (mlir::failed(wafer::frontend::verifyFrontendArtifact(
          *module, sidecarPath, llvm::errs(), &result)))
    return 1;

  if (result.sidecarConstantCount != 0) {
    llvm::outs() << "wafer-import-model: verified frontend sidecar constants: "
                 << result.sidecarConstantCount << "\n";
  }
  llvm::outs() << "wafer-import-model: verified frontend artifact\n";
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
  llvm::errs() << "wafer-import-model: importer dependencies are disabled in "
                  "this build\n";
  return 1;
#else
  if (argc == 2 && std::string(argv[1]) == "--emit-static-smoke-artifact") {
    emitStaticSmokeArtifact();
    return 0;
  }

  std::string verifyFilename;
  std::string sidecarPath;
  for (int i = 1; i < argc; ++i) {
    llvm::StringRef arg(argv[i]);
    if (arg == "--verify-import-result") {
      if (i + 1 >= argc) {
        llvm::errs() << "wafer-import-model: missing --verify-import-result "
                        "filename\n";
        return 1;
      }
      verifyFilename = argv[++i];
      continue;
    }

    constexpr llvm::StringRef verifyPrefix = "--verify-import-result=";
    if (arg.starts_with(verifyPrefix)) {
      verifyFilename = arg.drop_front(verifyPrefix.size()).str();
      continue;
    }

    if (arg == "--sidecar") {
      if (i + 1 >= argc) {
        llvm::errs() << "wafer-import-model: missing --sidecar filename\n";
        return 1;
      }
      sidecarPath = argv[++i];
      continue;
    }

    constexpr llvm::StringRef sidecarPrefix = "--sidecar=";
    if (arg.starts_with(sidecarPrefix)) {
      sidecarPath = arg.drop_front(sidecarPrefix.size()).str();
      continue;
    }

    llvm::errs() << "wafer-import-model: unknown argument: " << arg << "\n";
    printHelp();
    return 1;
  }

  if (!verifyFilename.empty())
    return verifyImportResult(verifyFilename, sidecarPath);

  llvm::errs() << "wafer-import-model: unknown or incomplete arguments\n";
  printHelp();
  return 1;
#endif
}
