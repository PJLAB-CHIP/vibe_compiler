//===- wafer-verify-program.cpp - Program-directory advisory verifier ---===//

#include "Wafer/Frontend/InitImporterDialects.h"
#include "Wafer/Frontend/StableHLO/ProgramIngestion.h"
#include "Wafer/InitWaferDialects.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace {

void printHelp() {
  llvm::outs() << "usage: wafer-verify-program --program-dir <directory>\n";
}

} // namespace

int main(int argc, char **argv) {
  if (argc == 2 && llvm::StringRef(argv[1]) == "--help") {
    printHelp();
    return 0;
  }
  llvm::StringRef programDirectory;
  if (argc == 3 && llvm::StringRef(argv[1]) == "--program-dir")
    programDirectory = argv[2];
  else if (argc == 2 && llvm::StringRef(argv[1]).starts_with("--program-dir="))
    programDirectory = llvm::StringRef(argv[1]).drop_front(14);
  else {
    llvm::errs() << "wafer-verify-program: expected exactly --program-dir\n";
    printHelp();
    return 1;
  }

  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect>();
  wafer::registerWaferCoreDialects(registry);
  wafer::registerImporterDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  mlir::ScopedDiagnosticHandler handler(
      &context, [](mlir::Diagnostic &diagnostic) {
        llvm::errs() << "wafer-verify-program: ";
        diagnostic.print(llvm::errs());
        llvm::errs() << '\n';
        return mlir::success();
      });

  std::string diagnostics;
  llvm::raw_string_ostream stream(diagnostics);
  auto program = wafer::frontend::ingestStableHLOProgramDirectory(
      programDirectory, context, stream);
  if (mlir::failed(program)) {
    stream.flush();
    llvm::StringRef remaining(diagnostics);
    while (!remaining.empty()) {
      auto [line, rest] = remaining.split('\n');
      if (!line.empty())
        llvm::errs() << "wafer-verify-program: " << line << '\n';
      remaining = rest;
    }
    return 1;
  }
  llvm::outs() << "wafer-verify-program: verified portable StableHLO program"
               << " parameters=" << program->metadata.programParameterCount
               << " constants=" << program->metadata.programConstantCount
               << '\n';
  return 0;
}
