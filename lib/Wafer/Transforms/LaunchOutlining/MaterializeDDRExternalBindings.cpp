//===- MaterializeDDRExternalBindings.cpp - DDR binding demand ops -------===//

#include "Wafer/Transforms/Passes.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <limits>
#include <optional>

namespace wafer {
namespace {

constexpr int64_t kDefaultDDRExternalAlignment = 256;

struct ExternalBindingDemand {
  mlir::Value value;
  wafer::DdrBindingKind kind;
  int64_t bytes;
};

static bool checkedMul(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0)
    return false;
  if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

static std::optional<int64_t> getElementBitWidth(mlir::Type elementType) {
  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(elementType))
    return floatType.getWidth();
  if (auto integerType = mlir::dyn_cast<mlir::IntegerType>(elementType))
    return integerType.getWidth();
  if (mlir::isa<mlir::IndexType>(elementType))
    return 64;
  if (auto complexType = mlir::dyn_cast<mlir::ComplexType>(elementType)) {
    std::optional<int64_t> elementBits =
        getElementBitWidth(complexType.getElementType());
    if (!elementBits)
      return std::nullopt;
    int64_t complexBits = 0;
    if (!checkedMul(*elementBits, 2, complexBits))
      return std::nullopt;
    return complexBits;
  }
  return std::nullopt;
}

static std::optional<int64_t>
getCompactTensorByteSize(mlir::RankedTensorType tensorType) {
  if (!tensorType.hasStaticShape())
    return std::nullopt;

  int64_t elements = 1;
  for (int64_t dim : tensorType.getShape()) {
    int64_t next = 0;
    if (!checkedMul(elements, dim, next))
      return std::nullopt;
    elements = next;
  }

  std::optional<int64_t> elementBits =
      getElementBitWidth(tensorType.getElementType());
  if (!elementBits || *elementBits <= 0)
    return std::nullopt;

  int64_t totalBits = 0;
  if (!checkedMul(elements, *elementBits, totalBits))
    return std::nullopt;
  return totalBits / 8 + (totalBits % 8 == 0 ? 0 : 1);
}

static std::optional<mlir::Value>
resolveTileRegionBoundaryValue(wafer::TileRegionOp tileRegion,
                               mlir::Value bodyValue) {
  auto blockArgument = mlir::dyn_cast<mlir::BlockArgument>(bodyValue);
  if (!blockArgument)
    return std::nullopt;
  if (blockArgument.getOwner() != &tileRegion.getBody().front())
    return std::nullopt;
  if (blockArgument.getArgNumber() >= tileRegion.getInputs().size())
    return std::nullopt;
  return tileRegion.getInputs()[blockArgument.getArgNumber()];
}

static bool hasDemand(llvm::ArrayRef<ExternalBindingDemand> demands,
                      mlir::Value value, wafer::DdrBindingKind kind) {
  return llvm::any_of(demands, [&](const ExternalBindingDemand &demand) {
    return demand.value == value && demand.kind == kind;
  });
}

static bool hasExistingBinding(mlir::ModuleOp module, mlir::Value value,
                               wafer::DdrBindingKind kind) {
  bool found = false;
  module.walk([&](wafer::DdrExternalBindingOp binding) {
    if (binding.getValue() == value && binding.getKindAttr().getValue() == kind)
      found = true;
  });
  return found;
}

static mlir::LogicalResult
appendDemand(mlir::Operation *op, mlir::Value value, wafer::DdrBindingKind kind,
             llvm::SmallVectorImpl<ExternalBindingDemand> &demands) {
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
  if (!tensorType || !tensorType.hasStaticShape())
    return op->emitOpError(
        "cannot materialize DDR external binding for non-static tensor");

  std::optional<int64_t> bytes = getCompactTensorByteSize(tensorType);
  if (!bytes || *bytes <= 0)
    return op->emitOpError(
        "cannot materialize DDR external binding with unrepresentable compact "
        "storage size");

  if (hasDemand(demands, value, kind))
    return mlir::success();
  demands.push_back(ExternalBindingDemand{value, kind, *bytes});
  return mlir::success();
}

static mlir::LogicalResult collectExternalBindingDemands(
    wafer::TileRegionOp tileRegion,
    llvm::SmallVectorImpl<ExternalBindingDemand> &demands) {
  bool failed = false;
  tileRegion.walk([&](wafer::LoadTileOp load) {
    std::optional<mlir::Value> boundary =
        resolveTileRegionBoundaryValue(tileRegion, load.getSource());
    if (!boundary)
      return;
    if (mlir::failed(appendDemand(load.getOperation(), *boundary,
                                  wafer::DdrBindingKind::Input, demands)))
      failed = true;
  });
  if (failed)
    return mlir::failure();

  tileRegion.walk([&](wafer::StoreTileOp store) {
    std::optional<mlir::Value> boundary =
        resolveTileRegionBoundaryValue(tileRegion, store.getDest());
    if (!boundary)
      return;
    if (mlir::failed(appendDemand(store.getOperation(), *boundary,
                                  wafer::DdrBindingKind::Output, demands)))
      failed = true;
  });

  return mlir::failure(failed);
}

static void materializeBindings(mlir::ModuleOp module,
                                wafer::TileRegionOp tileRegion,
                                llvm::ArrayRef<ExternalBindingDemand> demands) {
  mlir::OpBuilder builder(tileRegion);
  mlir::MLIRContext *context = tileRegion.getContext();
  for (const ExternalBindingDemand &demand : demands) {
    if (hasExistingBinding(module, demand.value, demand.kind))
      continue;

    bool readOnly = demand.kind == wafer::DdrBindingKind::Input;
    builder.create<wafer::DdrExternalBindingOp>(
        tileRegion.getLoc(),
        wafer::DdrBindingKindAttr::get(context, demand.kind), demand.value,
        builder.getI64IntegerAttr(demand.bytes),
        builder.getI64IntegerAttr(kDefaultDDRExternalAlignment),
        builder.getBoolAttr(readOnly), builder.getBoolAttr(true));
  }
}

struct MaterializeDDRExternalBindingsPass
    : public mlir::PassWrapper<MaterializeDDRExternalBindingsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      MaterializeDDRExternalBindingsPass)

  llvm::StringRef getArgument() const final {
    return "wafer-materialize-ddr-external-bindings";
  }

  llvm::StringRef getDescription() const final {
    return "materialize launch-visible DDR external binding demands";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<wafer::WaferDialect>();
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    llvm::SmallVector<wafer::TileRegionOp> tileRegions;
    module.walk([&](wafer::TileRegionOp tileRegion) {
      tileRegions.push_back(tileRegion);
    });

    for (wafer::TileRegionOp tileRegion : tileRegions) {
      llvm::SmallVector<ExternalBindingDemand> demands;
      if (mlir::failed(collectExternalBindingDemands(tileRegion, demands))) {
        signalPassFailure();
        return;
      }
      materializeBindings(module, tileRegion, demands);
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createMaterializeDDRExternalBindingsPass() {
  return std::make_unique<MaterializeDDRExternalBindingsPass>();
}

} // namespace wafer
