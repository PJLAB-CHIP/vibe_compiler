//===- WaferInterfaces.cpp - Wafer operation interfaces ------------------===//

#include "Wafer/IR/WaferInterfaces.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>

#include "Wafer/IR/WaferInterfaces.cpp.inc"

namespace wafer {
bool CoupledReductionDescription::hasReplicatedComponent(
    unsigned iterationDimension) const {
  return llvm::any_of(
      components, [&](const CoupledReductionComponent &component) {
        return !component.indexingMap.isFunctionOfDim(iterationDimension);
      });
}

namespace {

mlir::FailureOr<llvm::SmallVector<int64_t, 4>>
resolveStaticValues(llvm::ArrayRef<mlir::OpFoldResult> values) {
  llvm::SmallVector<int64_t, 4> resolved;
  resolved.reserve(values.size());
  for (mlir::OpFoldResult value : values) {
    std::optional<int64_t> constant = mlir::getConstantIntValue(value);
    if (!constant)
      return mlir::failure();
    resolved.push_back(*constant);
  }
  return resolved;
}

TensorIndexingOperandDescription
describeOperand(mlir::OpOperand &operand, TensorIndexingOperandRole role,
                llvm::ArrayRef<int64_t> offsets = {},
                llvm::ArrayRef<int64_t> strides = {}) {
  TensorIndexingOperandDescription description;
  description.operand = operand.getOperandNumber();
  description.role = role;
  description.offsets.assign(offsets.begin(), offsets.end());
  description.strides.assign(strides.begin(), strides.end());
  return description;
}

mlir::FailureOr<TensorIndexingDescription>
describeUnary(unsigned result, TensorIndexingTransformKind kind,
              mlir::OpOperand &source) {
  if (result != 0)
    return mlir::failure();
  TensorIndexingDescription description;
  description.kind = kind;
  description.result = result;
  description.operands.push_back(
      describeOperand(source, TensorIndexingOperandRole::Source));
  return description;
}

struct ExpandShapeIndexingModel final
    : WaferTensorIndexingOpInterface::ExternalModel<
          ExpandShapeIndexingModel, mlir::tensor::ExpandShapeOp> {
  mlir::FailureOr<TensorIndexingDescription>
  getTensorIndexingDescription(mlir::Operation *operation,
                               unsigned result) const {
    auto expand = mlir::cast<mlir::tensor::ExpandShapeOp>(operation);
    return describeUnary(result, TensorIndexingTransformKind::ExpandShape,
                         expand.getSrcMutable());
  }
};

struct CollapseShapeIndexingModel final
    : WaferTensorIndexingOpInterface::ExternalModel<
          CollapseShapeIndexingModel, mlir::tensor::CollapseShapeOp> {
  mlir::FailureOr<TensorIndexingDescription>
  getTensorIndexingDescription(mlir::Operation *operation,
                               unsigned result) const {
    auto collapse = mlir::cast<mlir::tensor::CollapseShapeOp>(operation);
    return describeUnary(result, TensorIndexingTransformKind::CollapseShape,
                         collapse.getSrcMutable());
  }
};

struct ExtractSliceIndexingModel final
    : WaferTensorIndexingOpInterface::ExternalModel<
          ExtractSliceIndexingModel, mlir::tensor::ExtractSliceOp> {
  mlir::FailureOr<TensorIndexingDescription>
  getTensorIndexingDescription(mlir::Operation *operation,
                               unsigned result) const {
    if (result != 0)
      return mlir::failure();
    auto extract = mlir::cast<mlir::tensor::ExtractSliceOp>(operation);
    auto offsets = resolveStaticValues(extract.getMixedOffsets());
    auto strides = resolveStaticValues(extract.getMixedStrides());
    if (mlir::failed(offsets) || mlir::failed(strides))
      return mlir::failure();
    TensorIndexingDescription description;
    description.kind = TensorIndexingTransformKind::ExtractSlice;
    description.result = result;
    description.operands.push_back(
        describeOperand(extract.getSourceMutable(),
                        TensorIndexingOperandRole::Source, *offsets, *strides));
    return description;
  }
};

struct InsertSliceIndexingModel final
    : WaferTensorIndexingOpInterface::ExternalModel<
          InsertSliceIndexingModel, mlir::tensor::InsertSliceOp> {
  mlir::FailureOr<TensorIndexingDescription>
  getTensorIndexingDescription(mlir::Operation *operation,
                               unsigned result) const {
    if (result != 0)
      return mlir::failure();
    auto insert = mlir::cast<mlir::tensor::InsertSliceOp>(operation);
    auto offsets = resolveStaticValues(insert.getMixedOffsets());
    auto strides = resolveStaticValues(insert.getMixedStrides());
    if (mlir::failed(offsets) || mlir::failed(strides))
      return mlir::failure();
    TensorIndexingDescription description;
    description.kind = TensorIndexingTransformKind::InsertSlice;
    description.result = result;
    description.operands.push_back(
        describeOperand(insert.getSourceMutable(),
                        TensorIndexingOperandRole::Source, *offsets, *strides));
    description.operands.push_back(describeOperand(
        insert.getDestMutable(), TensorIndexingOperandRole::Destination));
    return description;
  }
};

struct PadIndexingModel final
    : WaferTensorIndexingOpInterface::ExternalModel<PadIndexingModel,
                                                    mlir::tensor::PadOp> {
  mlir::FailureOr<TensorIndexingDescription>
  getTensorIndexingDescription(mlir::Operation *operation,
                               unsigned result) const {
    if (result != 0)
      return mlir::failure();
    auto pad = mlir::cast<mlir::tensor::PadOp>(operation);
    auto offsets = resolveStaticValues(pad.getMixedLowPad());
    if (mlir::failed(offsets))
      return mlir::failure();
    llvm::SmallVector<int64_t, 4> strides(offsets->size(), 1);
    TensorIndexingDescription description;
    description.kind = TensorIndexingTransformKind::Pad;
    description.result = result;
    description.operands.push_back(
        describeOperand(pad.getSourceMutable(),
                        TensorIndexingOperandRole::Source, *offsets, strides));
    return description;
  }
};

struct CastIndexingModel final
    : WaferTensorIndexingOpInterface::ExternalModel<CastIndexingModel,
                                                    mlir::tensor::CastOp> {
  mlir::FailureOr<TensorIndexingDescription>
  getTensorIndexingDescription(mlir::Operation *operation,
                               unsigned result) const {
    auto cast = mlir::cast<mlir::tensor::CastOp>(operation);
    return describeUnary(result, TensorIndexingTransformKind::Cast,
                         cast.getSourceMutable());
  }
};

} // namespace

void registerWaferTensorIndexingExternalModels(
    mlir::DialectRegistry &registry) {
  registry.addExtension(+[](mlir::MLIRContext *context,
                            mlir::tensor::TensorDialect *) {
    mlir::tensor::ExpandShapeOp::attachInterface<ExpandShapeIndexingModel>(
        *context);
    mlir::tensor::CollapseShapeOp::attachInterface<CollapseShapeIndexingModel>(
        *context);
    mlir::tensor::ExtractSliceOp::attachInterface<ExtractSliceIndexingModel>(
        *context);
    mlir::tensor::InsertSliceOp::attachInterface<InsertSliceIndexingModel>(
        *context);
    mlir::tensor::PadOp::attachInterface<PadIndexingModel>(*context);
    mlir::tensor::CastOp::attachInterface<CastIndexingModel>(*context);
  });
}

} // namespace wafer
