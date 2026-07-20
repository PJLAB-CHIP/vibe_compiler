//===- StructuredOpInterfaceModels.cpp - Source implementation models ----===//

#include "Wafer/Transforms/PhysicalDataflow.h"

#include "Wafer/Analysis/PhysicalDataflow/IndexRelation.h"
#include "Wafer/IR/WaferInterfaces.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/DialectRegistry.h"
#include "llvm/ADT/STLExtras.h"

namespace wafer {
namespace {

static bool isOne(mlir::Attribute attribute) {
  if (!attribute)
    return false;
  if (auto value = mlir::dyn_cast<mlir::FloatAttr>(attribute))
    return value.getValueAsDouble() == 1.0;
  if (auto value = mlir::dyn_cast<mlir::IntegerAttr>(attribute))
    return value.getValue().isOne();
  if (auto elements = mlir::dyn_cast<mlir::DenseElementsAttr>(attribute)) {
    if (!elements.isSplat())
      return false;
    if (mlir::isa<mlir::FloatType>(elements.getElementType()))
      return elements.getSplatValue<mlir::APFloat>().isExactlyValue(1.0);
    if (mlir::isa<mlir::IntegerType>(elements.getElementType()))
      return elements.getSplatValue<mlir::APInt>().isOne();
  }
  return false;
}

static mlir::Attribute getConstantAttribute(mlir::linalg::GenericOp generic,
                                            mlir::Value value) {
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantOp>())
    return constant.getValue();
  auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
  if (!argument || argument.getOwner() != generic.getBody() ||
      argument.getArgNumber() >= generic.getNumDpsInputs())
    return {};
  mlir::Value input = generic.getDpsInputs()[argument.getArgNumber()];
  if (auto constant = input.getDefiningOp<mlir::arith::ConstantOp>())
    return constant.getValue();
  return {};
}

static bool hasReciprocalExpression(mlir::linalg::GenericOp generic) {
  return llvm::any_of(
      generic.getBody()->without_terminator(), [&](mlir::Operation &operation) {
        auto div = mlir::dyn_cast<mlir::arith::DivFOp>(operation);
        return div && isOne(getConstantAttribute(generic, div.getLhs()));
      });
}

/// The reciprocal-via-division alternative is pointwise only when every
/// structured input relation is exactly represented by the current static
/// indexing maps. This is the first production consumer of IndexRelation; a
/// sound bound or unsupported relation keeps the baseline implementation.
static bool hasExactPointwiseIndexRelations(mlir::linalg::GenericOp generic) {
  if (generic.getNumDpsInits() != 1 || generic->getNumResults() != 1 ||
      llvm::any_of(generic.getIteratorTypesArray(), [](auto iterator) {
        return iterator != mlir::utils::IteratorType::parallel;
      }))
    return false;

  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(generic->getResult(0).getType());
  if (!resultType || !resultType.hasStaticShape())
    return false;
  llvm::SmallVector<mlir::AffineMap, 4> maps = generic.getIndexingMapsArray();
  if (maps.size() != generic.getNumDpsInputs() + 1 || !maps.back().isIdentity())
    return false;

  for (auto [input, map] : llvm::zip(generic.getDpsInputs(), maps)) {
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!inputType || !inputType.hasStaticShape())
      return false;
    analysis::IndexRelationResult relation =
        analysis::IndexRelation::fromAffineMap(map, resultType.getShape(),
                                               inputType.getShape());
    if (!relation.isExact())
      return false;
  }
  return true;
}

template <typename ConcreteModel>
static mlir::LogicalResult
materializeChecked(mlir::Operation *operation,
                   const TargetImplementationCandidate &candidate,
                   WaferTargetImplementationMaterializer &materializer,
                   mlir::OpBuilder &builder, const ConcreteModel &model) {
  llvm::SmallVector<TargetImplementationCandidate, 2> candidates;
  model.collectTargetImplementationCandidates(
      operation, WaferTargetCapabilities{}, candidates);
  if (!llvm::is_contained(candidates, candidate))
    return mlir::failure();
  return materializer.materializeTargetImplementation(operation, candidate,
                                                      builder);
}

template <typename ConcreteModel, typename ConcreteOp,
          TargetImplementationKind Kind>
struct DefaultStructuredOpModel
    : public WaferTargetImplementationOpInterface::ExternalModel<ConcreteModel,
                                                                 ConcreteOp> {
  void collectTargetImplementationCandidates(
      mlir::Operation *, const WaferTargetCapabilities &,
      llvm::SmallVectorImpl<TargetImplementationCandidate> &candidates) const {
    candidates.push_back(TargetImplementationCandidate{Kind});
  }

  mlir::LogicalResult materializeSelectedTargetImplementation(
      mlir::Operation *operation,
      const TargetImplementationCandidate &candidate,
      WaferTargetImplementationMaterializer &materializer,
      mlir::OpBuilder &builder) const {
    return materializeChecked(operation, candidate, materializer, builder,
                              *static_cast<const ConcreteModel *>(this));
  }
};

struct FillTargetImplementationModel
    : public DefaultStructuredOpModel<FillTargetImplementationModel,
                                      mlir::linalg::FillOp,
                                      TargetImplementationKind::Fill> {};

struct MatmulTargetImplementationModel
    : public DefaultStructuredOpModel<MatmulTargetImplementationModel,
                                      mlir::linalg::MatmulOp,
                                      TargetImplementationKind::Gemm> {};

struct BatchMatmulTargetImplementationModel
    : public DefaultStructuredOpModel<BatchMatmulTargetImplementationModel,
                                      mlir::linalg::BatchMatmulOp,
                                      TargetImplementationKind::BatchGemm> {};

struct GenericTargetImplementationModel
    : public WaferTargetImplementationOpInterface::ExternalModel<
          GenericTargetImplementationModel, mlir::linalg::GenericOp> {
  void collectTargetImplementationCandidates(
      mlir::Operation *operation, const WaferTargetCapabilities &capabilities,
      llvm::SmallVectorImpl<TargetImplementationCandidate> &candidates) const {
    auto generic = mlir::cast<mlir::linalg::GenericOp>(operation);
    candidates.push_back(
        TargetImplementationCandidate{TargetImplementationKind::Generic});
    if (capabilities.supportsElementwiseReciprocal &&
        capabilities.supportsElementwiseDivision &&
        hasReciprocalExpression(generic) &&
        hasExactPointwiseIndexRelations(generic))
      candidates.push_back(TargetImplementationCandidate{
          TargetImplementationKind::GenericReciprocalViaDivision});
  }

  mlir::LogicalResult materializeSelectedTargetImplementation(
      mlir::Operation *operation,
      const TargetImplementationCandidate &candidate,
      WaferTargetImplementationMaterializer &materializer,
      mlir::OpBuilder &builder) const {
    return materializeChecked(operation, candidate, materializer, builder,
                              *this);
  }
};

} // namespace

void registerTargetImplementationExternalModels(
    mlir::DialectRegistry &registry) {
  registry.addExtension(+[](mlir::MLIRContext *context,
                            mlir::linalg::LinalgDialect *) {
    mlir::linalg::FillOp::attachInterface<FillTargetImplementationModel>(
        *context);
    mlir::linalg::MatmulOp::attachInterface<MatmulTargetImplementationModel>(
        *context);
    mlir::linalg::BatchMatmulOp::attachInterface<
        BatchMatmulTargetImplementationModel>(*context);
    mlir::linalg::GenericOp::attachInterface<GenericTargetImplementationModel>(
        *context);
  });
}

} // namespace wafer
