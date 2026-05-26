//===- wafer-import-model.cpp - Wafer importer smoke tool ----------------===//

#ifdef WAFER_ENABLE_STABLEHLO
#include "Wafer/Frontend/InitImporterDialects.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
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
               << "  --verify-import-result <mlir-file>\n";
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

bool rejectImportMarker(mlir::ModuleOp module, llvm::StringRef attrName,
                        llvm::StringRef message) {
  mlir::Attribute marker = module->getAttr(attrName);
  if (!marker)
    return false;

  if (auto boolMarker = mlir::dyn_cast<mlir::BoolAttr>(marker);
      boolMarker && !boolMarker.getValue())
    return false;

  llvm::errs() << "wafer-import-model: " << message << " rejected";
  if (auto stringMarker = mlir::dyn_cast<mlir::StringAttr>(marker))
    llvm::errs() << ": " << stringMarker.getValue();
  else
    llvm::errs() << ": " << marker;
  llvm::errs() << "\n";
  return true;
}

bool hasUnboundedDynamicShape(mlir::Type type) {
  auto shapedType = mlir::dyn_cast<mlir::ShapedType>(type);
  return shapedType && !shapedType.hasStaticShape();
}

bool rejectUnboundedDynamicShapes(mlir::ModuleOp module) {
  bool rejected = false;
  module.walk([&](mlir::func::FuncOp func) {
    auto rejectType = [&](mlir::Type type) {
      if (!hasUnboundedDynamicShape(type))
        return false;
      llvm::errs() << "wafer-import-model: unbounded dynamic shape rejected "
                      "in func.func @"
                   << func.getSymName() << ": " << type << "\n";
      rejected = true;
      return true;
    };

    for (mlir::Type input : func.getFunctionType().getInputs()) {
      if (rejectType(input))
        return mlir::WalkResult::interrupt();
    }
    for (mlir::Type result : func.getFunctionType().getResults()) {
      if (rejectType(result))
        return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  return rejected;
}

int verifyImportResult(llvm::StringRef filename) {
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

  bool rejected = false;
  rejected |=
      rejectImportMarker(*module, "wafer.import.graph_break", "graph break");
  rejected |= rejectImportMarker(*module, "wafer.import.eager_fallback",
                                 "eager fallback");
  rejected |= rejectUnboundedDynamicShapes(*module);
  if (rejected)
    return 1;

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

  if (argc == 3 && std::string(argv[1]) == "--verify-import-result")
    return verifyImportResult(argv[2]);

  constexpr llvm::StringRef verifyPrefix = "--verify-import-result=";
  if (argc == 2) {
    llvm::StringRef arg(argv[1]);
    if (arg.starts_with(verifyPrefix))
      return verifyImportResult(arg.drop_front(verifyPrefix.size()));
  }

  llvm::errs() << "wafer-import-model: unknown or incomplete arguments\n";
  printHelp();
  return 1;
#endif
}
