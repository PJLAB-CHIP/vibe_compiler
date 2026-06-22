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
#include <limits>
#include <memory>

namespace wafer {
#define GEN_PASS_DEF_MATERIALIZEEXECUTIONMESHPASS
#include "Wafer/Transforms/WaferPasses.h.inc"

namespace {

static bool checkedMul(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0)
    return false;
  if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

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

static mlir::FailureOr<int64_t>
computeEndpointCount(mlir::ModuleOp moduleOp, TargetTopologyOp topologyOp) {
  llvm::ArrayRef<int64_t> cardGrid = topologyOp.getCardGridAttr().asArrayRef();
  llvm::ArrayRef<int64_t> tileGrid = topologyOp.getTileGridAttr().asArrayRef();

  int64_t cardCount = 0;
  int64_t rowCount = 0;
  int64_t endpointCount = 0;
  if (!checkedMul(cardGrid[0], cardGrid[1], cardCount) ||
      !checkedMul(cardCount, tileGrid[0], rowCount) ||
      !checkedMul(rowCount, tileGrid[1], endpointCount)) {
    moduleOp.emitError()
        << "execution_mesh_failure: endpoint count is too large";
    return mlir::failure();
  }

  int64_t unavailableCount =
      static_cast<int64_t>(topologyOp.getUnavailableTilesAttr().size() / 4);
  return endpointCount - unavailableCount;
}

static mlir::LogicalResult materializeExecutionMesh(
    mlir::ModuleOp moduleOp, llvm::StringRef meshName,
    llvm::StringRef topologyName, llvm::StringRef policy,
    llvm::StringRef axesOption, llvm::StringRef shapeOption,
    llvm::StringRef endpointsOption) {
  if (meshName.empty())
    return moduleOp.emitError()
           << "execution_mesh_failure: mesh symbol name must not be empty";
  if (topologyName.empty())
    return moduleOp.emitError()
           << "execution_mesh_failure: topology symbol name must not be empty";

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

  TargetTopologyOp topologyOp =
      moduleOp.lookupSymbol<TargetTopologyOp>(topologyName);
  if (!topologyOp)
    return moduleOp.emitError()
           << "execution_mesh_failure: referenced target topology not found: "
           << topologyName;

  if (policy != "all_available" && policy != "explicit")
    return moduleOp.emitError()
           << "execution_mesh_failure: policy must be all_available or "
              "explicit";

  mlir::FailureOr<llvm::SmallVector<int64_t, 8>> endpoints =
      parseOptionalI64List(endpointsOption, "endpoints", moduleOp);
  if (mlir::failed(endpoints))
    return mlir::failure();
  if (endpoints->size() % 4 != 0)
    return moduleOp.emitError()
           << "execution_mesh_failure: endpoints must contain "
              "card_y/card_x/tile_y/tile_x tuples";
  if (policy == "all_available" && !endpoints->empty())
    return moduleOp.emitError()
           << "execution_mesh_failure: all_available policy must not carry "
              "explicit endpoints";
  if (policy == "explicit" && endpoints->empty())
    return moduleOp.emitError()
           << "execution_mesh_failure: explicit policy requires endpoints";

  int64_t rankCount = 0;
  if (policy == "all_available") {
    mlir::FailureOr<int64_t> availableCount =
        computeEndpointCount(moduleOp, topologyOp);
    if (mlir::failed(availableCount))
      return mlir::failure();
    rankCount = *availableCount;
  } else {
    rankCount = static_cast<int64_t>(endpoints->size() / 4);
  }

  mlir::FailureOr<llvm::SmallVector<int64_t, 8>> parsedShape =
      parseOptionalI64List(shapeOption, "shape", moduleOp);
  if (mlir::failed(parsedShape))
    return mlir::failure();
  if (parsedShape->empty())
    parsedShape->push_back(rankCount);

  mlir::FailureOr<llvm::SmallVector<llvm::StringRef, 4>> parsedAxes =
      parseOptionalStringList(axesOption, "axes", moduleOp);
  if (mlir::failed(parsedAxes))
    return mlir::failure();
  if (parsedAxes->empty()) {
    if (parsedShape->size() != 1)
      return moduleOp.emitError()
             << "execution_mesh_failure: axes must be specified when shape "
                "rank is greater than one";
    parsedAxes->push_back("rank");
  }

  mlir::OpBuilder builder(moduleOp.getContext());
  builder.setInsertionPointAfter(topologyOp);
  builder.create<ExecutionMeshOp>(
      moduleOp.getLoc(), builder.getStringAttr(meshName),
      mlir::FlatSymbolRefAttr::get(builder.getContext(), topologyName),
      builder.getStrArrayAttr(*parsedAxes),
      mlir::DenseI64ArrayAttr::get(builder.getContext(), *parsedShape),
      builder.getStringAttr(policy),
      mlir::DenseI64ArrayAttr::get(builder.getContext(), *endpoints));

  return mlir::success();
}

struct MaterializeExecutionMeshPass
    : public impl::MaterializeExecutionMeshPassBase<
          MaterializeExecutionMeshPass> {
  using impl::MaterializeExecutionMeshPassBase<
      MaterializeExecutionMeshPass>::MaterializeExecutionMeshPassBase;

  void runOnOperation() final {
    if (mlir::failed(materializeExecutionMesh(
            getOperation(), meshName, topologyName, policy, axes, shape,
            endpoints))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

} // namespace wafer
