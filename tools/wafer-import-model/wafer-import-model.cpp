//===- wafer-import-model.cpp - Wafer importer artifact tool -------------===//

#ifdef WAFER_ENABLE_STABLEHLO
#include "Wafer/Frontend/Artifact.h"
#include "Wafer/Frontend/InitImporterDialects.h"
#include "Wafer/InitAll.h"
#include "Wafer/Pipelines/Pipelines.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
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
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace {

void printHelp() {
  llvm::outs() << "wafer-import-model\n";
#ifdef WAFER_ENABLE_STABLEHLO
  llvm::outs() << "  --emit-static-reference-artifact\n"
               << "  --verify-import-result <mlir-file>\n"
               << "  --verify-stablehlo-bundle <bundle-dir>\n"
               << "  --compile-stablehlo-bundle-to-cabi <bundle-dir>\n"
               << "    [--target=wafer] [--tile-mapping=single]\n";
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
}

int verifyImportResult(llvm::StringRef filename) {
  mlir::DialectRegistry registry;
  registerToolDialects(registry);

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

mlir::OwningOpRef<mlir::ModuleOp> parseAndVerifyStableHLOBundle(
    llvm::StringRef bundlePath, mlir::MLIRContext &context,
    wafer::frontend::ArtifactVerificationResult *result = nullptr) {
  llvm::SmallString<256> mlirPath(bundlePath);
  llvm::sys::path::append(mlirPath, "functions", "forward.mlir");

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(mlirPath, context);
  if (!module)
    return {};
  if (mlir::failed(mlir::verify(*module)))
    return {};

  if (mlir::failed(wafer::frontend::verifyStableHLOBundle(
          *module, bundlePath, llvm::errs(), result)))
    return {};

  return module;
}

int verifyStableHLOBundle(llvm::StringRef bundlePath) {
  mlir::DialectRegistry registry;
  registerToolDialects(registry);

  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  wafer::frontend::ArtifactVerificationResult result;
  mlir::OwningOpRef<mlir::ModuleOp> module =
      parseAndVerifyStableHLOBundle(bundlePath, context, &result);
  if (!module)
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

int compileStableHLOBundleToCAbi(llvm::StringRef bundlePath,
                                 llvm::StringRef target,
                                 llvm::StringRef tileMapping) {
  mlir::DialectRegistry registry;
  registerToolDialects(registry);

  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      parseAndVerifyStableHLOBundle(bundlePath, context);
  if (!module)
    return 1;

  mlir::PassManager pm(&context);
  wafer::buildStablehloToCAbiPipeline(pm, target, tileMapping);
  if (mlir::failed(pm.run(*module)))
    return 1;

  module->print(llvm::outs());
  llvm::outs() << "\n";
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
  std::string compileBundlePath;
  std::string target = "wafer";
  std::string tileMapping = "single";
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

    if (arg == "--compile-stablehlo-bundle-to-cabi") {
      if (i + 1 >= argc) {
        llvm::errs() << "wafer-import-model: missing "
                        "--compile-stablehlo-bundle-to-cabi directory\n";
        return 1;
      }
      compileBundlePath = argv[++i];
      continue;
    }

    constexpr llvm::StringRef compileBundlePrefix =
        "--compile-stablehlo-bundle-to-cabi=";
    if (arg.starts_with(compileBundlePrefix)) {
      compileBundlePath = arg.drop_front(compileBundlePrefix.size()).str();
      continue;
    }

    if (arg == "--target") {
      if (i + 1 >= argc) {
        llvm::errs() << "wafer-import-model: missing --target value\n";
        return 1;
      }
      target = argv[++i];
      continue;
    }

    constexpr llvm::StringRef targetPrefix = "--target=";
    if (arg.starts_with(targetPrefix)) {
      target = arg.drop_front(targetPrefix.size()).str();
      continue;
    }

    if (arg == "--tile-mapping") {
      if (i + 1 >= argc) {
        llvm::errs() << "wafer-import-model: missing --tile-mapping value\n";
        return 1;
      }
      tileMapping = argv[++i];
      continue;
    }

    constexpr llvm::StringRef tileMappingPrefix = "--tile-mapping=";
    if (arg.starts_with(tileMappingPrefix)) {
      tileMapping = arg.drop_front(tileMappingPrefix.size()).str();
      continue;
    }

    llvm::errs() << "wafer-import-model: unknown argument: " << arg << "\n";
    printHelp();
    return 1;
  }

  unsigned actionCount = 0;
  if (!verifyFilename.empty())
    ++actionCount;
  if (!bundlePath.empty())
    ++actionCount;
  if (!compileBundlePath.empty())
    ++actionCount;
  if (actionCount > 1) {
    llvm::errs() << "wafer-import-model: choose exactly one action\n";
    printHelp();
    return 1;
  }

  if (!verifyFilename.empty())
    return verifyImportResult(verifyFilename);
  if (!bundlePath.empty())
    return verifyStableHLOBundle(bundlePath);
  if (!compileBundlePath.empty())
    return compileStableHLOBundleToCAbi(compileBundlePath, target, tileMapping);

  llvm::errs() << "wafer-import-model: unknown or incomplete arguments\n";
  printHelp();
  return 1;
#endif
}
