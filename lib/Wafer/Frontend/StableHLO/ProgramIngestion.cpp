//===- ProgramIngestion.cpp - Portable StableHLO source ingestion --------===//

#include "Wafer/Frontend/StableHLO/ProgramIngestion.h"

#include "stablehlo/dialect/Serialization.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

namespace wafer::frontend {

mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
deserializeStableHLOProgramDirectory(llvm::StringRef programDirectory,
                                     mlir::MLIRContext &context,
                                     llvm::raw_ostream &diagnostics) {
  for (llvm::StringRef retired : {llvm::StringRef("forward.mlir"),
                                  llvm::StringRef("forward.bytecode")}) {
    llvm::SmallString<256> retiredPath(programDirectory);
    llvm::sys::path::append(retiredPath, "functions", retired);
    llvm::sys::fs::file_status status;
    if (!llvm::sys::fs::status(retiredPath, status) &&
        llvm::sys::fs::is_regular_file(status)) {
      diagnostics << "portable StableHLO program rejected: retired source IR "
                     "member remains: functions/"
                  << retired << '\n';
      return mlir::failure();
    }
  }
  llvm::SmallString<256> path(programDirectory);
  llvm::sys::path::append(path, "functions", "forward.stablehlo.bc");
  auto buffer = llvm::MemoryBuffer::getFile(path, /*IsText=*/false,
                                            /*RequiresNullTerminator=*/false);
  if (!buffer) {
    diagnostics << "portable StableHLO program rejected: failed to read '"
                << path << "': " << buffer.getError().message() << '\n';
    return mlir::failure();
  }
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::stablehlo::deserializePortableArtifact(
          (*buffer)->getBuffer(), &context);
  if (!module || mlir::failed(mlir::verify(*module))) {
    diagnostics << "portable StableHLO program rejected: artifact is not "
                   "compatible with the pinned StableHLO reader\n";
    return mlir::failure();
  }
  return std::move(module);
}

mlir::LogicalResult
verifyStableHLOSourceModule(mlir::ModuleOp module,
                            llvm::raw_ostream &diagnostics) {
  if (!module)
    return mlir::failure();
  if (mlir::failed(verifyFrontendProgram(module, diagnostics)))
    return mlir::failure();

  mlir::Operation *illegal = nullptr;
  module.walk([&](mlir::Operation *operation) {
    llvm::StringRef dialect = operation->getName().getDialectNamespace();
    if (dialect != "builtin" && dialect != "func" && dialect != "stablehlo") {
      illegal = operation;
      return mlir::WalkResult::interrupt();
    }
    if (operation->getName().getStringRef() == "stablehlo.custom_call") {
      illegal = operation;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (illegal) {
    diagnostics << "portable StableHLO program rejected: operation '"
                << illegal->getName()
                << "' is not legal in a production source program\n";
    return mlir::failure();
  }

  mlir::func::FuncOp function;
  unsigned functionCount = 0;
  module.walk([&](mlir::func::FuncOp candidate) {
    ++functionCount;
    function = candidate;
  });
  if (functionCount != 1 || !function || function.isExternal()) {
    diagnostics << "portable StableHLO program rejected: expected one defined "
                   "entry function\n";
    return mlir::failure();
  }
  for (mlir::Type type : function.getFunctionType().getInputs())
    if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(type);
        shaped && (!shaped.hasRank() || !shaped.hasStaticShape())) {
      diagnostics << "portable StableHLO program rejected: dynamic input "
                   "boundary is unsupported\n";
      return mlir::failure();
    }
  for (mlir::Type type : function.getFunctionType().getResults())
    if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(type);
        shaped && (!shaped.hasRank() || !shaped.hasStaticShape())) {
      diagnostics << "portable StableHLO program rejected: dynamic output "
                   "boundary is unsupported\n";
      return mlir::failure();
    }
  return mlir::success();
}

mlir::FailureOr<VerifiedStableHLOProgram> ingestStableHLOProgramDirectory(
    llvm::StringRef programDirectory, mlir::MLIRContext &context,
    llvm::raw_ostream &diagnostics, const ProgramPayloadResolver *resolver) {
  auto module = deserializeStableHLOProgramDirectory(programDirectory, context,
                                                     diagnostics);
  if (mlir::failed(module) ||
      mlir::failed(verifyStableHLOSourceModule(**module, diagnostics)))
    return mlir::failure();
  VerifiedStableHLOProgram result;
  if (mlir::failed(verifyProgramDirectory(**module, programDirectory,
                                          diagnostics, &result.metadata,
                                          resolver)))
    return mlir::failure();
  result.module = std::move(*module);
  return result;
}

} // namespace wafer::frontend
