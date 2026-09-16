//===- AccessReuseAnalysis.h - Current read access reuse --------*- C++ -*-===//

#ifndef WAFER_ANALYSIS_TILE_ACCESSREUSEANALYSIS_H
#define WAFER_ANALYSIS_TILE_ACCESSREUSEANALYSIS_H

#include "Wafer/Analysis/ControlFlow/StaticLoopDomain.h"
#include "Wafer/Analysis/Linalg/IndexRelation.h"
#include "Wafer/IR/WaferDialect.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <string>

namespace wafer::analysis {

/// Coordinates of an actual load, relative to its current region-local source
/// base. Map dimensions are the enclosing loop IVs in `loops`, in outer to
/// inner order. No allocation, placement, completion or capacity is predicted.
struct ReadAccess {
  StorageLoadOp load;
  TileModuleOp tile;
  int64_t argument = -1;
  mlir::MemRefType sourceType;
  mlir::MemRefType payloadType;
  llvm::SmallVector<int64_t> strides;
  mlir::AffineExpr linearOffset;
  llvm::SmallVector<StaticLoopDomain, 4> loops;
  int64_t physicalBytes = 0;
  mlir::Value base;
  mlir::AffineMap coordinates;
};

/// Equal current windows over matching static iteration domains. Each member
/// retains its actual Tile identity and load operation.
struct PeerAccessGroup {
  llvm::SmallVector<ReadAccess, 4> reads;
};

/// Exact union of current accesses inside one existing loop. `origin` is
/// parametric only in current SSA values defined outside that loop.
struct ScopedReadAccess {
  mlir::Value source;
  TileModuleOp tile;
  int64_t argument = -1;
  mlir::scf::ForOp scope;
  mlir::AffineMap origin;
  llvm::SmallVector<mlir::Value, 4> parameters;
  llvm::SmallVector<int64_t> sizes;
  llvm::SmallVector<int64_t> sourceStrides;
  mlir::AffineExpr linearOffset;
  llvm::SmallVector<StaticLoopDomain, 4> outerLoops;
  llvm::SmallVector<StorageLoadOp, 4> reads;
  uint64_t readBytes = 0;
  uint64_t payloadBytes = 0;
  uint64_t windowBytes = 0;
  uint64_t scopeExecutions = 0;
};

struct SlidingReadAccess {
  ReadAccess access;
  mlir::scf::ForOp scope;
  unsigned axis = 0;
  int64_t shift = 0;
};

struct ScopedReadAccessResult {
  IndexRelationStatus status = IndexRelationStatus::Unsupported;
  std::optional<ScopedReadAccess> access;
};

struct SlidingReadAccessResult {
  IndexRelationStatus status = IndexRelationStatus::Unsupported;
  std::optional<SlidingReadAccess> access;
};

/// Read-only facts for one unchanged candidate epoch. Mutation invalidates all
/// handles and derived relations. Different source resources are never merged
/// merely because their shapes or printed operations agree.
struct AccessReuseAnalysis {
  IndexRelationStatus status = IndexRelationStatus::Exact;
  std::string detail;
  llvm::SmallVector<ReadAccess, 16> reads;
  llvm::SmallVector<PeerAccessGroup, 4> peers;
  llvm::SmallVector<ScopedReadAccess, 8> scopes;
  llvm::SmallVector<SlidingReadAccess, 4> sliding;
  uint64_t scopeQueries = 0;
  uint64_t unsupportedScopes = 0;
  uint64_t indeterminateScopes = 0;
};

AccessReuseAnalysis analyzeAccessReuse(mlir::ModuleOp module,
                                       const IndexRelationLimits &limits = {},
                                       uint64_t scopeQueryLimit = 256);

ScopedReadAccessResult
queryScopedAccessReuse(const AccessReuseAnalysis &facts, mlir::Value source,
                       mlir::scf::ForOp scope,
                       const IndexRelationLimits &limits = {});

SlidingReadAccessResult
querySlidingAccessReuse(const AccessReuseAnalysis &facts, StorageLoadOp read,
                        const IndexRelationLimits &limits = {});

} // namespace wafer::analysis
#endif
