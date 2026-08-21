//===- PhysicalRepresentation.cpp - Selected value layouts -----------===//

#include "Wafer/Planning/Search/PhysicalRepresentation.h"

#include "Wafer/Analysis/PhysicalDataflow/PhysicalLayoutRelation.h"
#include "Wafer/Analysis/Structured/StructuredOperationTileFootprint.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>

namespace wafer::compiler::detail {
namespace {

const TemporalNodeAssignment *
findTemporal(const CardTemporalAssignment &assignment,
             StructuredDAGNodeID node) {
  auto found = llvm::find_if(assignment.nodes, [&](const auto &candidate) {
    return candidate.node == node;
  });
  return found == assignment.nodes.end() ? nullptr : &*found;
}

std::optional<analysis::StaticRectangularIndexSet>
findShard(const ExecutionShard *shard) {
  if (!shard)
    return std::nullopt;
  analysis::StaticRectangularIndexSet rectangle;
  for (const IteratorInterval &interval : shard->iterationDomain) {
    rectangle.offsets.push_back(interval.offset);
    rectangle.sizes.push_back(interval.size);
  }
  return rectangle;
}

llvm::SmallVector<MemLayout, 4>
getLegalLayouts(mlir::RankedTensorType tensorType,
                llvm::ArrayRef<int64_t> shape) {
  llvm::SmallVector<MemLayout, 4> result;
  if (!tensorType ||
      llvm::any_of(shape, [](int64_t size) { return size <= 0; }))
    return result;
  for (MemLayout layout :
       {MemLayout::Tensor, MemLayout::NTensor, MemLayout::Cx, MemLayout::NCx}) {
    auto type = mlir::MemRefType::get(
        shape, tensorType.getElementType(), mlir::MemRefLayoutAttrInterface{},
        MemoryAttr::get(tensorType.getContext(), MemorySpace::SPM, layout));
    if (mlir::succeeded(analysis::PhysicalLayoutRelation::create(type)))
      result.push_back(layout);
  }
  return result;
}

void intersectLayouts(llvm::SmallVectorImpl<MemLayout> &layouts,
                      llvm::ArrayRef<MemLayout> required) {
  llvm::erase_if(layouts, [&](MemLayout layout) {
    return !llvm::is_contained(required, layout);
  });
}

} // namespace

mlir::FailureOr<CardPhysicalRepresentationDomain>
CardPhysicalRepresentationDomain::create(
    const CardProgramAnalysis &program,
    const SpatialAssignment &spatial,
    const analysis::ExactDemandProof &demand,
    const CoupledRegionDomain &coupledDomain,
    const CoupledRegionAssignment &coupledAssignment,
    const CardTemporalDomain &temporalDomain,
    const CardTemporalAssignment &temporalAssignment,
    std::string *failureReason) {
  auto fail = [&](llvm::StringRef message)
      -> mlir::FailureOr<CardPhysicalRepresentationDomain> {
    if (failureReason)
      *failureReason = message.str();
    return mlir::failure();
  };
  mlir::FailureOr<StructuredDemandView> view = StructuredDemandView::create(
      program.dag, spatial, demand, failureReason);
  if (mlir::failed(view))
    return mlir::failure();
  if (!coupledDomain.contains(coupledAssignment))
    return fail("physical representation received a stale coupled assignment");
  if (!temporalDomain.contains(temporalAssignment))
    return fail("physical representation received a stale temporal assignment");

  llvm::SmallVector<ValueDomain, 32> values;
  for (const CoupledRegionGroup &group : coupledAssignment.groups) {
    for (StructuredDAGNodeID node : group.nodes) {
      const StructuredDAGNode *dagNode = program.dag.getNode(node);
      const TemporalNodeAssignment *temporal =
          findTemporal(temporalAssignment, node);
      std::optional<analysis::StaticRectangularIndexSet> shard =
          findShard(view->getShard(node, group.tile));
      if (!dagNode || !dagNode->operation || !temporal || !shard ||
          shard->sizes.size() != temporal->iteratorTileSizes.size())
        return fail("physical representation lost one node shard");
      llvm::SmallVector<int64_t, 4> leafSizes;
      for (auto [extent, tile] :
           llvm::zip_equal(shard->sizes, temporal->iteratorTileSizes)) {
        if (extent <= 0 || tile <= 0)
          return fail("physical representation has an invalid temporal leaf");
        leafSizes.push_back(std::min(extent, tile));
      }

      for (auto [operandNumber, operand] :
           llvm::enumerate(dagNode->operation->getOperands())) {
        auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(operand.getType());
        if (!tensor)
          continue;
        if (auto linalg =
                mlir::dyn_cast<mlir::linalg::LinalgOp>(dagNode->operation);
            linalg && !linalg.payloadUsesValueFromOperand(
                          &dagNode->operation->getOpOperand(operandNumber)))
          continue;
        auto shape = getStructuredOperandTileShape(
            dagNode->operation, static_cast<unsigned>(operandNumber),
            leafSizes);
        if (!shape)
          return fail("physical operand layout has no exact tile shape");
        llvm::SmallVector<MemLayout, 4> layouts =
            getLegalLayouts(tensor, *shape);
        if (layouts.empty())
          return fail("physical operand layout domain is empty");
        values.push_back(ValueDomain{
            group.tile, node, PhysicalValueRole::Operand,
            static_cast<unsigned>(operandNumber), std::move(layouts)});
      }
      for (unsigned resultNumber = 0;
           resultNumber < dagNode->operation->getNumResults(); ++resultNumber) {
        auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(
            dagNode->operation->getResult(resultNumber).getType());
        if (!tensor)
          continue;
        auto shape = getStructuredResultTileShape(dagNode->operation,
                                                  resultNumber, leafSizes);
        if (!shape)
          return fail("physical result layout has no exact tile shape");
        llvm::SmallVector<MemLayout, 4> layouts =
            getLegalLayouts(tensor, *shape);
        if (view->hasSpatialReduction(node)) {
          llvm::SmallVector<MemLayout, 4> partialLayouts =
              getLegalLayouts(tensor, shard->sizes);
          intersectLayouts(layouts, partialLayouts);
          const SemanticRootKey *root = view->getRoot(node);
          const bool isMergeTile = root && llvm::any_of(
              demand.reductionMerges,
              [&](const analysis::ReductionMergeRequirement &merge) {
                return merge.group.root == *root &&
                       merge.mergeTile == group.tile;
              });
          if (isMergeTile) {
            llvm::SmallVector<MemLayout, 4> mergedLayouts =
                getLegalLayouts(tensor, tensor.getShape());
            intersectLayouts(layouts, mergedLayouts);
          }
        }
        if (layouts.empty())
          return fail("physical result layout domain is empty");
        values.push_back(ValueDomain{group.tile, node,
                                     PhysicalValueRole::Result, resultNumber,
                                     std::move(layouts)});
      }
    }
  }
  llvm::sort(values, [](const ValueDomain &lhs, const ValueDomain &rhs) {
    return std::tuple(lhs.tile.getValue(), lhs.node,
                      static_cast<uint8_t>(lhs.role), lhs.index) <
           std::tuple(rhs.tile.getValue(), rhs.node,
                      static_cast<uint8_t>(rhs.role), rhs.index);
  });
  return CardPhysicalRepresentationDomain(std::move(values));
}

CardPhysicalRepresentationAssignment
CardPhysicalRepresentationDomain::getFirstAssignment() const {
  CardPhysicalRepresentationAssignment result;
  for (const ValueDomain &value : values)
    result.values.push_back(
        PhysicalRepresentationChoice{value.tile, value.node, value.role,
                                     value.index, value.layouts.front()});
  return result;
}

bool CardPhysicalRepresentationDomain::contains(
    const CardPhysicalRepresentationAssignment &assignment) const {
  if (assignment.values.size() != values.size())
    return false;
  for (auto [choice, domain] : llvm::zip_equal(assignment.values, values))
    if (choice.tile != domain.tile || choice.node != domain.node ||
        choice.role != domain.role || choice.index != domain.index ||
        !llvm::is_contained(domain.layouts, choice.layout))
      return false;
  return true;
}

mlir::FailureOr<std::optional<CardPhysicalRepresentationAssignment>>
CardPhysicalRepresentationDomain::getNextAssignment(
    const CardPhysicalRepresentationAssignment &assignment) const {
  if (!contains(assignment))
    return mlir::failure();
  CardPhysicalRepresentationAssignment next = assignment;
  for (size_t reverse = 0; reverse < values.size(); ++reverse) {
    const size_t index = values.size() - reverse - 1;
    auto current = llvm::find(values[index].layouts, next.values[index].layout);
    if (current == values[index].layouts.end())
      return mlir::failure();
    ++current;
    if (current == values[index].layouts.end())
      continue;
    next.values[index].layout = *current;
    for (size_t reset = index + 1; reset < values.size(); ++reset)
      next.values[reset].layout = values[reset].layouts.front();
    return std::optional<CardPhysicalRepresentationAssignment>(std::move(next));
  }
  return std::optional<CardPhysicalRepresentationAssignment>{};
}

} // namespace wafer::compiler::detail
