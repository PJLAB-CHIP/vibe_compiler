//===- wafer-import-model.cpp - Wafer importer reference artifact tool ----------------===//

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
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace {

void printHelp() {
  llvm::outs() << "wafer-import-model\n";
#ifdef WAFER_ENABLE_STABLEHLO
  llvm::outs() << "  --emit-static-reference-artifact\n"
               << "  --verify-import-result <mlir-file>\n"
               << "  --verify-stablehlo-bundle <bundle-dir>\n";
#else
  llvm::outs() << "  importer dependencies are disabled in this build\n";
#endif
}

#ifdef WAFER_ENABLE_STABLEHLO
void emitStaticReferenceArtifact() {
  llvm::outs()
      << "module {\n"
      << "  func.func @wafer_import_static_reference(%arg0: tensor<2x4xf32>, "
         "%arg1: tensor<2x4xf32>) -> tensor<2x4xf32> {\n"
      << "    %0 = stablehlo.add %arg0, %arg1 : tensor<2x4xf32>\n"
      << "    return %0 : tensor<2x4xf32>\n"
      << "  }\n"
      << "}\n";
}

mlir::OwningOpRef<mlir::ModuleOp>
parseModule(llvm::StringRef filename, mlir::MLIRContext &context) {
  mlir::ParserConfig config(&context);
  return mlir::parseSourceFile<mlir::ModuleOp>(filename, config);
}

int verifyImportResult(llvm::StringRef filename) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect>();
  wafer::registerImporterDialects(registry);

  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(filename, context);
  if (!module)
    return 1;
  if (mlir::failed(mlir::verify(*module)))
    return 1;

  wafer::frontend::ArtifactVerificationResult result;
  if (mlir::failed(wafer::frontend::verifyFrontendArtifact(
          *module, llvm::errs(), &result)))
    return 1;

  llvm::outs() << "wafer-import-model: verified frontend artifact\n";
  return 0;
}

int verifyStableHLOBundle(llvm::StringRef bundlePath) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect>();
  wafer::registerImporterDialects(registry);

  llvm::SmallString<256> mlirPath(bundlePath);
  llvm::sys::path::append(mlirPath, "functions", "forward.mlir");

  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(mlirPath, context);
  if (!module)
    return 1;
  if (mlir::failed(mlir::verify(*module)))
    return 1;

  wafer::frontend::ArtifactVerificationResult result;
  if (mlir::failed(wafer::frontend::verifyStableHLOBundle(
          *module, bundlePath, llvm::errs(), &result)))
    return 1;

  llvm::outs() << "wafer-import-model: verified StableHLO bundle parameters: "
               << result.bundleParameterCount << "\n";
  if (result.bundleParameterShardBindingCount)
    llvm::outs() << "wafer-import-model: verified StableHLO bundle "
                    "parameter shard bindings: "
                 << result.bundleParameterShardBindingCount << "\n";
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
  if (argc == 2 && std::string(argv[1]) == "--emit-static-reference-artifact") {
    emitStaticReferenceArtifact();
    return 0;
  }

  std::string verifyFilename;
  std::string bundlePath;
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

    if (arg == "--verify-stablehlo-bundle") {
      if (i + 1 >= argc) {
        llvm::errs() << "wafer-import-model: missing "
                        "--verify-stablehlo-bundle directory\n";
        return 1;
      }
      bundlePath = argv[++i];
      continue;
    }

    constexpr llvm::StringRef bundlePrefix = "--verify-stablehlo-bundle=";
    if (arg.starts_with(bundlePrefix)) {
      bundlePath = arg.drop_front(bundlePrefix.size()).str();
      continue;
    }

    llvm::errs() << "wafer-import-model: unknown argument: " << arg << "\n";
    printHelp();
    return 1;
  }

  if (!verifyFilename.empty())
    return verifyImportResult(verifyFilename);
  if (!bundlePath.empty())
    return verifyStableHLOBundle(bundlePath);

  llvm::errs() << "wafer-import-model: unknown or incomplete arguments\n";
  printHelp();
  return 1;
#endif
}
