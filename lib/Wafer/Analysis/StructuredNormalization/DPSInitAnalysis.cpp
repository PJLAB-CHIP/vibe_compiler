//===- DPSInitAnalysis.cpp - Recompute destination-init facts -----------===//

#include "Wafer/Analysis/DPSInitAnalysis.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"

namespace wafer {
namespace {

static mlir::Value stripStaticTensorViews(mlir::Value value,
                                          uint64_t &inspectedNodes) {
  while (mlir::Operation *definition = value.getDefiningOp()) {
    ++inspectedNodes;
    if (auto cast = mlir::dyn_cast<mlir::tensor::CastOp>(definition)) {
      value = cast.getSource();
      continue;
    }
    if (auto slice =
            mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(definition)) {
      if (!slice.getSourceType().hasStaticShape() ||
          !slice.getResultType().hasStaticShape())
        break;
      value = slice.getSource();
      continue;
    }
    if (auto collapse =
            mlir::dyn_cast<mlir::tensor::CollapseShapeOp>(definition)) {
      if (!collapse.getSrcType().hasStaticShape() ||
          !collapse.getResultType().hasStaticShape())
        break;
      value = collapse.getSrc();
      continue;
    }
    if (auto expand =
            mlir::dyn_cast<mlir::tensor::ExpandShapeOp>(definition)) {
      if (!expand.getSrcType().hasStaticShape() ||
          !expand.getResultType().hasStaticShape())
        break;
      value = expand.getSrc();
      continue;
    }
    break;
  }
  return value;
}

} // namespace

DPSInitFacts analyzeDPSInit(mlir::linalg::LinalgOp operation,
                            unsigned initIndex) {
  DPSInitFacts facts;
  if (initIndex >= operation.getNumDpsInits())
    return facts;

  mlir::OpOperand *initOperand = operation.getDpsInitOperand(initIndex);
  facts.readState = operation.payloadUsesValueFromOperand(initOperand)
                        ? InitReadState::Read
                        : InitReadState::Unread;
  mlir::Value root =
      stripStaticTensorViews(initOperand->get(), facts.inspectedNodes);
  facts.sourceRoot = root;

  if (root.getDefiningOp<mlir::tensor::EmptyOp>()) {
    facts.origin = InitOrigin::Undefined;
    return facts;
  }
  if (auto fill = root.getDefiningOp<mlir::linalg::FillOp>()) {
    facts.origin = InitOrigin::ExactSplat;
    facts.splatScalar = fill.getInputs().front();
    mlir::Attribute constant;
    if (mlir::matchPattern(facts.splatScalar, mlir::m_Constant(&constant)))
      facts.exactSplatValue = constant;
    return facts;
  }

  mlir::Attribute constant;
  if (mlir::matchPattern(root, mlir::m_Constant(&constant))) {
    if (auto elements = mlir::dyn_cast<mlir::DenseElementsAttr>(constant);
        elements && elements.isSplat()) {
      facts.origin = InitOrigin::ExactSplat;
      facts.exactSplatValue = elements.getSplatValue<mlir::Attribute>();
      return facts;
    }
  }
  facts.origin = InitOrigin::ExistingValue;
  return facts;
}

} // namespace wafer
