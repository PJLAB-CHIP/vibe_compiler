//===- WaferTensorProgramToCardModule.h - Card MPMD lowering -*- C++
//-*-===//

#ifndef WAFER_CONVERSION_WAFERTENSORPROGRAMTOCARDMODULE_H
#define WAFER_CONVERSION_WAFERTENSORPROGRAMTOCARDMODULE_H

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/DependentDataflow.h"
#include "Wafer/Target/TopologyIds.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace wafer {

/// Query-local spatial placement of one observable result. A present
/// `shardDimension` divides the result in `activeTileIds` order. A missing
/// dimension is the typed unpartitioned coordinate and requires exactly one
/// active Tile, which owns the complete result domain (including rank zero).
struct OutputTileMapping {
  unsigned outputIndex = 0;
  std::optional<unsigned> shardDimension;
  llvm::SmallVector<TileId, 16> activeTileIds;
  /// Static per-dimension temporal tile selected for this output.  Card
  /// materialization intersects it with each balanced spatial shard, so a
  /// smaller boundary shard becomes an ordinary finite tail class.  This is
  /// query-local input and is expressed by actual loops in the resulting IR.
  llvm::SmallVector<int64_t, 4> temporalTileSizes;
};

/// Complete physical placement consumed atomically by CardModule
/// materialization.  Every function result appears exactly once.  Different
/// results may use different dimensions and disjoint Tile sets; Tiles not
/// present in any output mapping receive a verifier-legal no-work entry.  This
/// object is never persisted in IR.
struct TileMapping {
  /// Explicit controller-owned lowering contract.  This query-local enum is
  /// consumed directly by CardModule materialization; it is not serialized
  /// into selected IR and is never recovered from edge actions or names.
  SpatialDataflowMaterializationMode materializationMode =
      SpatialDataflowMaterializationMode::JointDataflow;
  llvm::SmallVector<OutputTileMapping, 4> outputs;
  /// Per-structured-operation iterator tiles selected jointly with placement.
  /// Every scheduled source operation appears exactly once.  Reduction
  /// iterator sizes become typed accumulator traversal in the materialized
  /// Tile modules; parallel iterator sizes bound compact output traversal,
  /// while exact operand demand remains derived from consumer/result indexing
  /// relations.
  llvm::SmallVector<StructuredOpTemporalTile, 16> operationTemporalTiles;
  /// One exact action for every selected current-SSA producer/consumer edge.
  /// Demand, locality and remote fragments intentionally share this single
  /// carrier so CardModule materialization cannot observe a partial plan.
  llvm::SmallVector<SpatialEdgeStrategy, 16> edgeStrategies;
};

/// Materializes one explicit spatial mapping for a card-local structured
/// tensor program.
///
/// The input must contain one verified single-block functional tensor program,
/// one
/// physical target topology and one single-partition logical execution mesh.
/// Every static output domain is divided into balanced, nonempty contiguous
/// shards over the Tiles selected for that output. Each active Tile
/// owns the ordinary TilingInterface-driven TileRegion realization of only
/// its selected output roots and actual fused producer closure. Each selected
/// output is traversed by its own structured temporal loop nest and finite
/// tail classes; different outputs need not share a temporal tile shape.
/// Remaining output domains on that Tile take a typed no-store path; Tiles
/// with no selected output own a defined no-work entry. Every Tile entry has
/// the same full-card input/result ABI. Source arguments are preserved exactly;
/// private scheduling destinations are introduced and consumed inside this
/// conversion, leaving compiler-owned result roots for Tile memory
/// planning and exact target output binding.
///
/// The result is one owning module containing module-scope target topology and
/// logical mesh, one wafer.card.module, card-shared declarations exactly once,
/// and all-and-only available wafer.tile.module bodies. No search decision or
/// mapping side data is persisted. The source and `cardModule` output are
/// unchanged on failure.
mlir::LogicalResult lowerTensorProgramToCardModule(
    mlir::ModuleOp sourceModule, CardId cardId, const TileMapping &mapping,
    mlir::OwningOpRef<mlir::ModuleOp> &cardModule,
    std::string *failureReason = nullptr,
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes = {},
    StructuredMaterializationRelations *materializationRelations = nullptr,
    llvm::ArrayRef<llvm::SmallVector<uint32_t, 2>> observableOutputRootNodes =
        {});

struct TileRootShardRequest {
  TileId tileId{0};
  uint32_t targetRoot = 0;
};

struct TileRootShardMaterialization {
  TileId tileId{0};
  uint32_t targetRoot = 0;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  StructuredMaterializationRelations relations;
};

struct TileEntryMaterialization {
  TileId tileId{0};
  mlir::OwningOpRef<mlir::ModuleOp> module;
  StructuredMaterializationRelations relations;
};

class TileMaterializationSourceSession;

/// Mapping-local materialization session shared by narrow root/Tile probes and
/// the final CardModule assembly. Creation verifies the borrowed source,
/// target topology and complete mapping once and owns one immutable scheduling
/// clone. Every lowering method clones only the requested apply scope; the
/// source and session preparation remain unchanged on success or failure.
class TileMaterializationSession {
public:
  /// Builds one mapping-local apply session from an already-validated source
  /// session. This performs only coordinate-dependent validation and owns a
  /// private scheduling clone; source IR, topology and immutable operation
  /// identity are not revalidated.
  static mlir::FailureOr<std::unique_ptr<TileMaterializationSession>> create(
      const TileMaterializationSourceSession &sourceSession,
      const TileMapping &mapping, std::string *failureReason = nullptr,
      llvm::ArrayRef<llvm::SmallVector<uint32_t, 2>>
          observableOutputRootNodes = {});

  /// Convenience boundary for callers that own only one mapping. Multi-trial
  /// controllers should create a TileMaterializationSourceSession once and
  /// use the overload above for each coordinate.
  static mlir::FailureOr<std::unique_ptr<TileMaterializationSession>> create(
      mlir::ModuleOp sourceModule, CardId cardId, const TileMapping &mapping,
      std::string *failureReason = nullptr,
      llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes = {},
      llvm::ArrayRef<llvm::SmallVector<uint32_t, 2>>
          observableOutputRootNodes = {});

  ~TileMaterializationSession();
  TileMaterializationSession(TileMaterializationSession &&) noexcept;
  TileMaterializationSession &
  operator=(TileMaterializationSession &&) noexcept;
  TileMaterializationSession(const TileMaterializationSession &) = delete;
  TileMaterializationSession &
  operator=(const TileMaterializationSession &) = delete;

  mlir::LogicalResult lowerCardModule(
      mlir::OwningOpRef<mlir::ModuleOp> &cardModule,
      StructuredMaterializationRelations *materializationRelations = nullptr,
      std::string *failureReason = nullptr) const;

  mlir::LogicalResult lowerRootShards(
      llvm::ArrayRef<TileRootShardRequest> requests,
      llvm::SmallVectorImpl<TileRootShardMaterialization> &materializations,
      std::string *failureReason = nullptr) const;

  /// Materializes complete detached Tile entry functions after all root-local
  /// probes fit. This is the nearest isolated lifetime scope that can observe
  /// cross-region residency; it creates no CardModule/TileModule shell,
  /// sibling Tile or no-work wrapper.
  mlir::LogicalResult lowerTileEntries(
      llvm::ArrayRef<TileId> tileIds,
      llvm::SmallVectorImpl<TileEntryMaterialization> &materializations,
      std::string *failureReason = nullptr) const;

private:
  struct Impl;
  explicit TileMaterializationSession(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl;
};

/// Immutable source/target facts shared across every mapping coordinate in a
/// functional-legalization session. Creation verifies the borrowed source,
/// target topology, logical mesh, static output domains and structured node
/// identity exactly once. It does not own a mapping or mutable trial state.
class TileMaterializationSourceSession {
public:
  static mlir::FailureOr<std::unique_ptr<TileMaterializationSourceSession>>
  create(mlir::ModuleOp sourceModule, CardId cardId,
         llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes = {},
         std::string *failureReason = nullptr);

  ~TileMaterializationSourceSession();
  TileMaterializationSourceSession(TileMaterializationSourceSession &&) noexcept;
  TileMaterializationSourceSession &
  operator=(TileMaterializationSourceSession &&) noexcept;
  TileMaterializationSourceSession(
      const TileMaterializationSourceSession &) = delete;
  TileMaterializationSourceSession &
  operator=(const TileMaterializationSourceSession &) = delete;

private:
  friend class TileMaterializationSession;
  struct Impl;
  explicit TileMaterializationSourceSession(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl;
};

/// Materializes a batch of root/Tile shards for scoped feasibility probing.
/// Source verification, topology construction, scheduling clone and mapping
/// validation are performed once for the whole batch; each result owns one
/// detached Tile entry whose pull closure covers exactly the requested root
/// and its non-root support. Sibling roots, CardModule/TileModule shells and
/// no-work wrappers are never created. Results preserve request order and the
/// caller-owned output is unchanged on failure.
mlir::LogicalResult lowerTensorProgramToTileRootShards(
    mlir::ModuleOp sourceModule, CardId cardId, const TileMapping &mapping,
    llvm::ArrayRef<TileRootShardRequest> requests,
    llvm::SmallVectorImpl<TileRootShardMaterialization> &materializations,
    std::string *failureReason = nullptr,
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes = {},
    llvm::ArrayRef<llvm::SmallVector<uint32_t, 2>> observableOutputRootNodes =
        {});

} // namespace wafer

#endif // WAFER_CONVERSION_WAFERTENSORPROGRAMTOCARDMODULE_H
