//===- MaterializeExecutionMesh.cpp - Materialize Wafer execution mesh ----===//

#include "Wafer/Transforms/Passes.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <memory>

namespace wafer {
#define GEN_PASS_DEF_MATERIALIZEEXECUTIONMESHPASS
#include "Wafer/Transforms/WaferPasses.h.inc"

namespace {

static mlir::FailureOr<llvm::SmallVector<int64_t, 8>>
parseOptionalI64List(llvm::StringRef text, llvm::StringRef optionName,
                     mlir::Operation *anchor) {
  llvm::SmallVector<int64_t, 8> values;
  text = text.trim();
  if (text.empty())
    return values;

  if (text.starts_with(",") || text.ends_with(",")) {
    anchor->emitError() << "execution_mesh_failure: invalid empty entry in "
                        << optionName;
    return mlir::failure();
  }

  llvm::StringRef rest = text;
  while (!rest.empty()) {
    auto split = rest.split(',');
    llvm::StringRef part = split.first.trim();
    if (part.empty()) {
      anchor->emitError() << "execution_mesh_failure: invalid empty entry in "
                          << optionName;
      return mlir::failure();
    }
    int64_t value = 0;
    if (part.getAsInteger(10, value)) {
      anchor->emitError() << "execution_mesh_failure: invalid integer in "
                          << optionName << ": " << part;
      return mlir::failure();
    }
    values.push_back(value);
    rest = split.second;
  }
  return values;
}

static mlir::FailureOr<llvm::SmallVector<llvm::StringRef, 4>>
parseOptionalStringList(llvm::StringRef text, llvm::StringRef optionName,
                        mlir::Operation *anchor) {
  llvm::SmallVector<llvm::StringRef, 4> values;
  text = text.trim();
  if (text.empty())
    return values;

  if (text.starts_with(",") || text.ends_with(",")) {
    anchor->emitError() << "execution_mesh_failure: invalid empty entry in "
                        << optionName;
    return mlir::failure();
  }

  llvm::StringRef rest = text;
  while (!rest.empty()) {
    auto split = rest.split(',');
    llvm::StringRef part = split.first.trim();
    if (part.empty()) {
      anchor->emitError() << "execution_mesh_failure: invalid empty entry in "
                          << optionName;
      return mlir::failure();
    }
    values.push_back(part);
    rest = split.second;
  }
  return values;
}

static mlir::LogicalResult
materializeExecutionMesh(mlir::ModuleOp moduleOp, llvm::StringRef meshName,
                         llvm::StringRef axesOption,
                         llvm::StringRef shapeOption) {
  if (meshName.empty())
    return moduleOp.emitError()
           << "execution_mesh_failure: mesh symbol name must not be empty";

  bool hasSameMesh = false;
  moduleOp.walk([&](ExecutionMeshOp meshOp) {
    if (meshOp.getSymName() == meshName)
      hasSameMesh = true;
  });
  if (hasSameMesh)
    return moduleOp.emitError()
           << "execution_mesh_failure: module already contains "
              "wafer.execution.mesh @"
           << meshName;

  mlir::FailureOr<llvm::SmallVector<int64_t, 8>> parsedShape =
      parseOptionalI64List(shapeOption, "shape", moduleOp);
  if (mlir::failed(parsedShape))
    return mlir::failure();
  if (parsedShape->empty())
    return moduleOp.emitError()
           << "execution_mesh_failure: logical mesh shape must be explicit";

  mlir::FailureOr<llvm::SmallVector<llvm::StringRef, 4>> parsedAxes =
      parseOptionalStringList(axesOption, "axes", moduleOp);
  if (mlir::failed(parsedAxes))
    return mlir::failure();
  if (parsedAxes->empty())
    return moduleOp.emitError()
           << "execution_mesh_failure: logical mesh axes must not be empty";

  mlir::OpBuilder builder(moduleOp.getContext());
  builder.setInsertionPointToStart(moduleOp.getBody());
  builder.create<ExecutionMeshOp>(
      moduleOp.getLoc(), builder.getStringAttr(meshName),
      builder.getStrArrayAttr(*parsedAxes),
      mlir::DenseI64ArrayAttr::get(builder.getContext(), *parsedShape));

  return mlir::success();
}

struct MaterializeExecutionMeshPass
    : public impl::MaterializeExecutionMeshPassBase<
          MaterializeExecutionMeshPass> {
  using impl::MaterializeExecutionMeshPassBase<
      MaterializeExecutionMeshPass>::MaterializeExecutionMeshPassBase;

  void runOnOperation() final {
    if (mlir::failed(materializeExecutionMesh(getOperation(), meshName, axes,
                                              shape))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

} // namespace wafer
