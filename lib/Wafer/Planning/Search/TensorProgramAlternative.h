//===- TensorProgramAlternative.h - Structured program choices -*- C++ -*-===//
#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

namespace wafer::compiler::detail {

enum class TensorProgramAlternativeKind : uint8_t {
  Original,
  OnlineAttention,
  SplitKeyValueAttention,
};

/// One graph-level semantic assignment. K/V blocking belongs here only
/// because it changes the online recurrence graph; ordinary temporal tiling of
/// the resulting structured operations is not part of this assignment.
struct TensorProgramAlternativeAssignment {
  static TensorProgramAlternativeAssignment original();
  static TensorProgramAlternativeAssignment onlineAttention(
      int64_t keyValueBlockSize);
  static TensorProgramAlternativeAssignment splitKeyValueAttention(
      int64_t keyValueBlockSize, int64_t keyValuePartitionCount);

  TensorProgramAlternativeKind kind = TensorProgramAlternativeKind::Original;
  int64_t keyValueBlockSize = 0;
  int64_t keyValuePartitionCount = 1;

  friend bool operator==(const TensorProgramAlternativeAssignment &left,
                         const TensorProgramAlternativeAssignment &right) {
    return std::tie(left.kind, left.keyValuePartitionCount,
                    left.keyValueBlockSize) ==
           std::tie(right.kind, right.keyValuePartitionCount,
                    right.keyValueBlockSize);
  }
  friend bool operator<(const TensorProgramAlternativeAssignment &left,
                        const TensorProgramAlternativeAssignment &right) {
    return std::tie(left.kind, left.keyValuePartitionCount,
                    left.keyValueBlockSize) <
           std::tie(right.kind, right.keyValuePartitionCount,
                    right.keyValueBlockSize);
  }
};

/// Compact complete domain derived from one immutable TensorProgram. It does
/// not allocate a point vector: callers advance a typed assignment in constant
/// work and can stop under the owning search budget without losing siblings.
class TensorProgramAlternativeDomain {
public:
  static TensorProgramAlternativeDomain originalOnly();
  static TensorProgramAlternativeDomain attention(int64_t reductionExtent,
                                                   bool supportsSplit);

  TensorProgramAlternativeAssignment getFirstAssignment() const;
  mlir::FailureOr<std::optional<TensorProgramAlternativeAssignment>>
  getNextAssignment(
      const TensorProgramAlternativeAssignment &assignment) const;
  mlir::FailureOr<uint64_t> getAssignmentCount() const;
  bool contains(const TensorProgramAlternativeAssignment &assignment) const;

  int64_t getReductionExtent() const { return reductionExtent; }
  bool supportsSplitKeyValue() const { return splitKeyValue; }

private:
  TensorProgramAlternativeDomain(int64_t reductionExtent,
                                 bool splitKeyValue)
      : reductionExtent(reductionExtent), splitKeyValue(splitKeyValue) {}

  int64_t reductionExtent = 0;
  bool splitKeyValue = false;
};

mlir::FailureOr<TensorProgramAlternativeDomain>
getTensorProgramAlternativeDomain(mlir::ModuleOp source);

enum class TensorProgramAlternativeMaterializationKind : uint8_t {
  Materialized,
  InvalidAssignment,
  Indeterminate,
};

/// Typed result of constructing one actual alternative root. Querying never
/// clones. Materialization clones exactly one isolated source Module and either
/// returns that actual root or destroys it on failure; it never lowers,
/// evaluates, caches, or replays the clone.
class TensorProgramAlternativeMaterialization {
public:
  static TensorProgramAlternativeMaterialization
  materialized(mlir::OwningOpRef<mlir::ModuleOp> module);
  static TensorProgramAlternativeMaterialization
  invalidAssignment(std::string diagnostic);
  static TensorProgramAlternativeMaterialization
  indeterminate(std::string diagnostic);

  TensorProgramAlternativeMaterializationKind getKind() const { return kind; }
  llvm::StringRef getDiagnostic() const { return diagnostic; }
  mlir::OwningOpRef<mlir::ModuleOp> takeModule();

private:
  TensorProgramAlternativeMaterialization(
      TensorProgramAlternativeMaterializationKind kind,
      mlir::OwningOpRef<mlir::ModuleOp> module, std::string diagnostic)
      : kind(kind), module(std::move(module)),
        diagnostic(std::move(diagnostic)) {}

  TensorProgramAlternativeMaterializationKind kind;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::string diagnostic;
};

TensorProgramAlternativeMaterialization materializeTensorProgramAlternative(
    mlir::ModuleOp source,
    const TensorProgramAlternativeAssignment &assignment);

} // namespace wafer::compiler::detail
