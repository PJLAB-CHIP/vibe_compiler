//===- TensorAssemblySelection.cpp - Bind actual tensor read families ----===//
#include "TensorAssemblySelection.h"

#include "llvm/ADT/STLExtras.h"

namespace wafer::compiler::detail {
namespace {
template <typename T> bool sameSet(const T &a, const T &b) {
  return a.size() == b.size() && llvm::all_of(a, [&](const auto &value) {
           return llvm::is_contained(b, value);
         });
}
} // namespace

TensorChoiceBindings TensorChoiceBindings::capture(mlir::Operation *parent) {
  TensorChoiceBindings result;
  parent->walk([&](mlir::Operation *operation) {
    for (auto value : operation->getResults())
      if (mlir::isa<mlir::RankedTensorType>(value.getType()))
        result.anchors[value].push_back(value);
    for (auto &region : operation->getRegions())
      for (auto &block : region)
        for (auto value : block.getArguments())
          if (mlir::isa<mlir::RankedTensorType>(value.getType()))
            result.anchors[value].push_back(value);
  });
  return result;
}

mlir::FailureOr<TensorChoiceBindings>
TensorChoiceBindings::clone(const mlir::IRMapping &mapping) const {
  TensorChoiceBindings result;
  for (const auto &[value, parents] : anchors) {
    auto mapped = mapping.lookupOrNull(value);
    if (!mapped)
      return mlir::failure();
    result.anchors[mapped] = parents;
  }
  return result;
}

void TensorChoiceBindings::remap(const mlir::IRMapping &mapping) {
  decltype(anchors) remapped;
  for (const auto &[value, parents] : anchors) {
    auto &destination = remapped[mapping.lookupOrDefault(value)];
    for (auto parent : parents)
      if (!llvm::is_contained(destination, parent))
        destination.push_back(parent);
  }
  anchors = std::move(remapped);
}

llvm::ArrayRef<mlir::Value>
TensorChoiceBindings::getAnchors(mlir::Value current) const {
  auto found = anchors.find(current);
  return found == anchors.end() ? llvm::ArrayRef<mlir::Value>{} : found->second;
}

void TensorChoiceBindings::notifyOperationReplaced(
    mlir::Operation *operation, mlir::ValueRange replacements) {
  for (auto [value, replacement] :
       llvm::zip_equal(operation->getResults(), replacements)) {
    auto found = anchors.find(value);
    if (found == anchors.end() || value == replacement)
      continue;
    auto parents = std::move(found->second);
    anchors.erase(found);
    if (!replacement)
      continue;
    auto &destination = anchors[replacement];
    for (auto parent : parents)
      if (!llvm::is_contained(destination, parent))
        destination.push_back(parent);
  }
}

void TensorChoiceBindings::notifyOperationErased(mlir::Operation *operation) {
  // Whole-region deletion can invalidate nested results and block arguments.
  operation->walk([&](mlir::Operation *nested) {
    for (auto value : nested->getResults())
      anchors.erase(value);
    for (auto &region : nested->getRegions())
      for (auto &block : region)
        notifyBlockErased(&block);
  });
}

void TensorChoiceBindings::notifyBlockErased(mlir::Block *block) {
  for (auto argument : block->getArguments())
    anchors.erase(argument);
}

bool operator==(const TensorAssemblySelection &a,
                const TensorAssemblySelection &b) {
  return sameSet(a.sources, b.sources) && sameSet(a.scopes, b.scopes) &&
         sameSet(a.reads, b.reads);
}

bool operator==(const TensorAssemblyIntent &a, const TensorAssemblyIntent &b) {
  return sameSet(a.selections, b.selections);
}

TensorAssemblyCapture
captureTensorAssemblyRead(mlir::tensor::ExtractSliceOp read,
                          const TensorChoiceBindings &bindings) {
  if (!read)
    return TensorAssemblyBindingUnavailable{"assembly read is absent"};
  TensorAssemblySelection result;
  llvm::append_range(result.sources, bindings.getAnchors(read.getSource()));
  if (result.sources.empty())
    return TensorAssemblyBindingUnavailable{
        "assembly source has no current correspondence to the live parent"};
  for (auto *operation = read->getParentOp(); operation;
       operation = operation->getParentOp()) {
    if (mlir::isa<TileRegionOp>(operation))
      break;
    auto coordinates = getIterationCoordinates(operation);
    if (coordinates && !llvm::is_contained(result.scopes, coordinates))
      result.scopes.push_back(coordinates);
  }
  if (result.scopes.empty()) {
    llvm::append_range(result.reads, bindings.getAnchors(read.getResult()));
    if (result.reads.empty())
      return TensorAssemblyBindingUnavailable{
          "assembly read has neither iteration coordinates nor a parent read "
          "correspondence"};
  }
  return result;
}

TensorAssemblyReadFamilies
groupTensorAssemblyReads(llvm::ArrayRef<mlir::tensor::ExtractSliceOp> reads,
                         const TensorChoiceBindings &bindings) {
  TensorAssemblyReadFamilies result;
  for (auto read : reads) {
    auto captured = captureTensorAssemblyRead(read, bindings);
    auto *selection = std::get_if<TensorAssemblySelection>(&captured);
    if (!selection) {
      result.unavailable.push_back(
          std::get<TensorAssemblyBindingUnavailable>(std::move(captured)));
      continue;
    }
    auto found = llvm::find_if(result.families, [&](const auto &family) {
      return family.selection == *selection;
    });
    if (found == result.families.end())
      result.families.push_back({std::move(*selection), {read}});
    else
      found->reads.push_back(read);
  }
  return result;
}
llvm::SmallVector<TensorAssemblyIntent, 4>
extendTensorAssemblyIntent(const TensorAssemblyIntent &current,
                           llvm::ArrayRef<TensorAssemblySelection> available) {
  // Discovery must support the complete proposed implementation. Retiling
  // can remove a previously selected family; extending it would manufacture
  // a combination absent from current IR.
  if (llvm::any_of(current.selections, [&](const auto &selection) {
        return !llvm::is_contained(available, selection);
      }))
    return {};
  TensorAssemblyIntent combined = current;
  for (const auto &selection : available)
    if (!llvm::is_contained(combined.selections, selection))
      combined.selections.push_back(selection);
  if (combined == current)
    return {};
  llvm::SmallVector<TensorAssemblyIntent, 4> result{combined};
  for (const auto &selection : available) {
    if (llvm::is_contained(current.selections, selection))
      continue;
    auto next = current;
    next.selections.push_back(selection);
    if (!llvm::is_contained(result, next))
      result.push_back(std::move(next));
  }
  return result;
}
} // namespace wafer::compiler::detail
