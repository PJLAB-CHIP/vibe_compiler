//===- ComputeImplementation.cpp - Structured compute choices ---------===//

#include "Wafer/Planning/Search/ComputeImplementation.h"

#include "Wafer/Analysis/PhysicalDataflow/IndexRelation.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

namespace wafer::compiler::detail {
namespace {

bool isOne(mlir::Attribute attribute) {
  if (auto value = mlir::dyn_cast_or_null<mlir::FloatAttr>(attribute))
    return value.getValueAsDouble() == 1.0;
  if (auto value = mlir::dyn_cast_or_null<mlir::IntegerAttr>(attribute))
    return value.getValue().isOne();
  if (auto elements =
          mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(attribute)) {
    if (!elements.isSplat())
      return false;
    if (mlir::isa<mlir::FloatType>(elements.getElementType()))
      return elements.getSplatValue<mlir::APFloat>().isExactlyValue(1.0);
    if (mlir::isa<mlir::IntegerType>(elements.getElementType()))
      return elements.getSplatValue<mlir::APInt>().isOne();
  }
  return false;
}

mlir::Attribute getConstantAttribute(mlir::Value value) {
  llvm::DenseSet<mlir::Value> visited;
  while (value && visited.insert(value).second) {
    if (auto constant = value.getDefiningOp<mlir::arith::ConstantOp>())
      return constant.getValue();
    if (auto slice = value.getDefiningOp<mlir::tensor::ExtractSliceOp>()) {
      value = slice.getSource();
      continue;
    }
    if (auto expand = value.getDefiningOp<mlir::tensor::ExpandShapeOp>()) {
      value = expand.getSrc();
      continue;
    }
    if (auto collapse = value.getDefiningOp<mlir::tensor::CollapseShapeOp>()) {
      value = collapse.getSrc();
      continue;
    }
    if (auto cast = value.getDefiningOp<mlir::tensor::CastOp>()) {
      value = cast.getSource();
      continue;
    }
    return {};
  }
  return {};
}

mlir::Attribute getConstantAttribute(mlir::linalg::GenericOp generic,
                                     mlir::Value value) {
  if (mlir::Attribute attribute = getConstantAttribute(value))
    return attribute;
  auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
  if (!argument || argument.getOwner() != generic.getBody() ||
      argument.getArgNumber() >= generic.getNumDpsInputs())
    return {};
  return getConstantAttribute(generic.getDpsInputs()[argument.getArgNumber()]);
}

bool hasExactPointwiseRelations(mlir::linalg::GenericOp generic) {
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
    if (!inputType || !inputType.hasStaticShape() ||
        !analysis::IndexRelation::fromAffineMap(map, resultType.getShape(),
                                                inputType.getShape())
             .isExact())
      return false;
  }
  return true;
}

bool supportsReciprocal(mlir::Operation *operation) {
  auto generic = mlir::dyn_cast_or_null<mlir::linalg::GenericOp>(operation);
  if (!generic || !hasExactPointwiseRelations(generic))
    return false;
  mlir::arith::DivFOp reciprocal;
  for (mlir::Operation &payload : generic.getBody()->without_terminator()) {
    auto div = mlir::dyn_cast<mlir::arith::DivFOp>(payload);
    if (!div)
      continue;
    if (reciprocal)
      return false;
    reciprocal = div;
  }
  return reciprocal &&
         isOne(getConstantAttribute(generic, reciprocal.getLhs()));
}

} // namespace

mlir::FailureOr<CardComputeImplementationDomain>
CardComputeImplementationDomain::create(const CardProgramAnalysis &program,
                                        std::string *failureReason) {
  llvm::SmallVector<NodeDomain, 16> nodes;
  for (const StructuredDAGNode &node : program.dag.getNodes()) {
    if (!node.operation) {
      if (failureReason)
        *failureReason = "compute implementation node has no operation";
      return mlir::failure();
    }
    nodes.push_back({node.id, supportsReciprocal(node.operation)});
  }
  return CardComputeImplementationDomain(std::move(nodes));
}

CardComputeImplementationAssignment
CardComputeImplementationDomain::getFirstAssignment() const {
  CardComputeImplementationAssignment result;
  for (const NodeDomain &node : nodes)
    result.nodes.push_back(
        {node.node, StructuredComputeImplementation::Natural});
  return result;
}

bool CardComputeImplementationDomain::contains(
    const CardComputeImplementationAssignment &assignment) const {
  if (assignment.nodes.size() != nodes.size())
    return false;
  for (auto [domain, choice] : llvm::zip_equal(nodes, assignment.nodes)) {
    if (choice.node != domain.node)
      return false;
    if (choice.implementation == StructuredComputeImplementation::Reciprocal &&
        !domain.supportsReciprocal)
      return false;
    if (choice.implementation != StructuredComputeImplementation::Natural &&
        choice.implementation != StructuredComputeImplementation::Reciprocal)
      return false;
  }
  return true;
}

mlir::FailureOr<std::optional<CardComputeImplementationAssignment>>
CardComputeImplementationDomain::getNextAssignment(
    const CardComputeImplementationAssignment &assignment) const {
  if (!contains(assignment))
    return mlir::failure();
  CardComputeImplementationAssignment next = assignment;
  for (size_t reverse = 0; reverse < nodes.size(); ++reverse) {
    const size_t index = nodes.size() - reverse - 1;
    if (!nodes[index].supportsReciprocal ||
        next.nodes[index].implementation ==
            StructuredComputeImplementation::Reciprocal)
      continue;
    next.nodes[index].implementation =
        StructuredComputeImplementation::Reciprocal;
    for (size_t reset = index + 1; reset < nodes.size(); ++reset)
      next.nodes[reset].implementation =
          StructuredComputeImplementation::Natural;
    return std::optional<CardComputeImplementationAssignment>(std::move(next));
  }
  return std::optional<CardComputeImplementationAssignment>{};
}

} // namespace wafer::compiler::detail
