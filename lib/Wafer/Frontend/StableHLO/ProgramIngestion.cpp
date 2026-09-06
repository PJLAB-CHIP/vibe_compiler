//===- ProgramIngestion.cpp - Portable StableHLO source ingestion --------===//

#include "Wafer/Frontend/StableHLO/ProgramIngestion.h"

#include "stablehlo/dialect/Serialization.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>

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
  unsigned publicFunctionCount = 0;
  module.walk([&](mlir::func::FuncOp candidate) {
    if (!candidate.isPrivate()) {
      ++publicFunctionCount;
      function = candidate;
    }
  });
  if (publicFunctionCount != 1 || !function || function.isExternal()) {
    diagnostics << "portable StableHLO program rejected: expected one defined "
                   "public entry function\n";
    return mlir::failure();
  }

  // The portable boundary permits a single public entry plus private helper
  // functions, but helpers are deliberately a pure, finite call closure.  Do
  // this check while the source module is still authoritative; the inliner
  // later removes calls before the tensor pipeline's single-function stages.
  llvm::DenseMap<mlir::Operation *, unsigned> visitState;
  std::function<mlir::LogicalResult(mlir::func::FuncOp)> verifyHelper =
      [&](mlir::func::FuncOp helper) -> mlir::LogicalResult {
    if (helper.isExternal()) {
      helper.emitError("private source helper must have a body");
      return mlir::failure();
    }
    unsigned &state = visitState[helper.getOperation()];
    if (state == 1) {
      helper.emitError("recursive private source helper call is unsupported");
      return mlir::failure();
    }
    if (state == 2)
      return mlir::success();
    state = 1;
    mlir::LogicalResult result = mlir::success();
    helper.walk([&](mlir::Operation *operation) {
      if (mlir::failed(result) || operation == helper.getOperation())
        return mlir::WalkResult::interrupt();
      if (auto call = mlir::dyn_cast<mlir::func::CallOp>(operation)) {
        mlir::func::FuncOp callee =
            module.lookupSymbol<mlir::func::FuncOp>(call.getCallee());
        if (!callee || callee.isExternal()) {
          operation->emitError("private source helper calls an unresolved or "
                               "external function");
          result = mlir::failure();
          return mlir::WalkResult::interrupt();
        }
        if (!callee.isPrivate()) {
          operation->emitError("private source helper may only call a private "
                               "helper");
          result = mlir::failure();
          return mlir::WalkResult::interrupt();
        }
        result = verifyHelper(callee);
        return mlir::failed(result) ? mlir::WalkResult::interrupt()
                                    : mlir::WalkResult::advance();
      }
      if (mlir::isa<mlir::CallOpInterface>(operation)) {
        operation->emitError("indirect calls are unsupported in a private "
                             "source helper");
        result = mlir::failure();
        return mlir::WalkResult::interrupt();
      }
      if (mlir::isa<mlir::func::ReturnOp>(operation))
        return mlir::WalkResult::advance();
      if (!mlir::isMemoryEffectFree(operation)) {
        operation->emitError("private source helper must be pure");
        result = mlir::failure();
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
    if (mlir::succeeded(result))
      state = 2;
    return result;
  };
  for (mlir::func::FuncOp candidate : module.getOps<mlir::func::FuncOp>())
    if (candidate.isPrivate() && mlir::failed(verifyHelper(candidate)))
      return mlir::failure();
  mlir::LogicalResult callClosure = mlir::success();
  module.walk([&](mlir::func::CallOp call) {
    if (mlir::failed(callClosure))
      return mlir::WalkResult::interrupt();
    mlir::func::FuncOp callee =
        module.lookupSymbol<mlir::func::FuncOp>(call.getCallee());
    if (!callee || callee.isExternal() || callee == function) {
      call.emitError("source call must resolve to a defined non-recursive "
                     "function in this module");
      callClosure = mlir::failure();
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (mlir::failed(callClosure))
    return mlir::failure();

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
