//===- TensorAssemblySelection.h - Current tensor read choices -*- C++ -*-===//
#ifndef WAFER_PLANNING_PHYSICALDATAFLOW_TENSORASSEMBLYSELECTION_H
#define WAFER_PLANNING_PHYSICALDATAFLOW_TENSORASSEMBLYSELECTION_H

#include "Wafer/IR/WaferDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <string>
#include <utility>
#include <variant>

namespace wafer::compiler::detail {

/// Choice correspondence to an immutable, live parent. Values are followed
/// only through clone mappings and actual SSA replacements. This is not a
/// source/alias/owner analysis and provides no materialization facts.
/// The caller drops these bindings before leaving the Tensor choice stage.
class TensorChoiceBindings : public mlir::RewriterBase::Listener {
public:
  TensorChoiceBindings() = default;
  TensorChoiceBindings(const TensorChoiceBindings &) = delete;
  TensorChoiceBindings &operator=(const TensorChoiceBindings &) = delete;
  TensorChoiceBindings(TensorChoiceBindings &&other) noexcept
      : anchors(std::move(other.anchors)) {}
  TensorChoiceBindings &operator=(TensorChoiceBindings &&other) noexcept {
    anchors = std::move(other.anchors);
    return *this;
  }
  static TensorChoiceBindings capture(mlir::Operation *parent);
  mlir::FailureOr<TensorChoiceBindings>
  clone(const mlir::IRMapping &mapping) const;
  void remap(const mlir::IRMapping &mapping);
  void clear() { anchors.clear(); }
  llvm::ArrayRef<mlir::Value> getAnchors(mlir::Value current) const;

  void notifyOperationReplaced(mlir::Operation *operation,
                               mlir::ValueRange replacements) override;
  void notifyOperationErased(mlir::Operation *operation) override;
  void notifyBlockErased(mlir::Block *block) override;

private:
  llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value, 2>> anchors;
};

/// One source snapshot and one complete family of current reads. Scope
/// coordinates group actual main/tail instances. Without tiling coordinates,
/// an explicit parent read anchor is required. Anchors borrow the immutable
/// parent; current read/loop handles never enter a persistent choice.
struct TensorAssemblySelection {
  llvm::SmallVector<mlir::Value, 2> sources;
  llvm::SmallVector<IterationCoordinatesAttr, 2> scopes;
  llvm::SmallVector<mlir::Value, 2> reads;
  friend bool operator==(const TensorAssemblySelection &a,
                         const TensorAssemblySelection &b);
};

struct TensorAssemblyIntent {
  llvm::SmallVector<TensorAssemblySelection, 4> selections;
  friend bool operator==(const TensorAssemblyIntent &a,
                         const TensorAssemblyIntent &b);
};

struct TensorAssemblyBindingUnavailable {
  std::string detail;
};
using TensorAssemblyCapture =
    std::variant<TensorAssemblySelection, TensorAssemblyBindingUnavailable>;

/// Identity capture is independent of generation capability. The caller must
/// query every current family member before selecting/materializing the group.
TensorAssemblyCapture
captureTensorAssemblyRead(mlir::tensor::ExtractSliceOp read,
                          const TensorChoiceBindings &bindings);

struct TensorAssemblyReadFamily {
  TensorAssemblySelection selection;
  llvm::SmallVector<mlir::tensor::ExtractSliceOp, 4> reads;
};
struct TensorAssemblyReadFamilies {
  llvm::SmallVector<TensorAssemblyReadFamily, 4> families;
  llvm::SmallVector<TensorAssemblyBindingUnavailable, 4> unavailable;
};
TensorAssemblyReadFamilies
groupTensorAssemblyReads(llvm::ArrayRef<mlir::tensor::ExtractSliceOp> reads,
                         const TensorChoiceBindings &bindings);

/// Extend a choice with available groups. Try the complete local alternative
/// first, then expose individual mixed shared/local extensions lazily.
llvm::SmallVector<TensorAssemblyIntent, 4>
extendTensorAssemblyIntent(const TensorAssemblyIntent &current,
                           llvm::ArrayRef<TensorAssemblySelection> available);

} // namespace wafer::compiler::detail
#endif
