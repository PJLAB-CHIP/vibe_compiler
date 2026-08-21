//===- WaferTensorProgramToTileRegion.h - Structured shard lowering -*- C++
//-*-===//

#ifndef WAFER_CONVERSION_WAFERTENSORPROGRAMTOTILEREGION_H
#define WAFER_CONVERSION_WAFERTENSORPROGRAMTOTILEREGION_H

#include "Wafer/Analysis/PhysicalDataflow/SpatialAssignment.h"
#include "Wafer/Target/Core/TopologyIds.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <string>

namespace mlir {
class OpBuilder;
class Operation;
class Value;
} // namespace mlir

namespace wafer {

enum class StructuredComputeImplementation : uint8_t {
  Natural,
  Reciprocal,
};

/// Query-local relation between a current structured operation and its DAG
/// node.  Callers remap `operation` with IRMapping whenever they clone an IR
/// scope; the relation is never encoded in Location or persisted in IR.
struct StructuredOperationNodeMapping {
  mlir::Operation *operation = nullptr;
  uint32_t structuredNodeId = 0;
};

/// One materialized buffer associated with a structured DAG node.  The buffer
/// is current-IR SSA and must be remapped or discarded with its owning IR.
struct StructuredOperationBufferRelation {
  uint32_t structuredNodeId = 0;
  mlir::Value buffer;
};

/// One physical compute operation emitted for a structured DAG node.  The
/// operation is current-IR SSA ownership evidence used while constructing
/// root-scoped TileRegions; it is never serialized or recovered from buffers.
struct StructuredOperationEmissionRelation {
  uint32_t structuredNodeId = 0;
  mlir::Operation *operation = nullptr;
};

/// One materialized buffer associated with an observable function result.
struct SpatialOutputBufferRelation {
  unsigned outputIndex = 0;
  mlir::Value buffer;
};

struct PartialReductionContributionBufferRelation {
  uint32_t structuredNodeId = 0;
  compiler::detail::ReductionGroupId group;
  unsigned resultIndex = 0;
  TileId sourceTile{0};
  mlir::Value buffer;
};

struct PartialReductionMergeInputBufferRelation {
  uint32_t structuredNodeId = 0;
  compiler::detail::ReductionGroupId group;
  unsigned resultIndex = 0;
  TileId sourceTile{0};
  mlir::Value buffer;
};

/// Exact current-IR relations emitted while lowering one or more TileRegions.
/// Result and operand buffers remain separate because allocation feedback for
/// an input demand refines the consumer node, while a result buffer describes
/// the producing node. No entry outlives or identifies a different IR epoch.
struct StructuredMaterializationRelations {
  llvm::SmallVector<StructuredOperationEmissionRelation, 16> operationEmissions;
  llvm::SmallVector<StructuredOperationBufferRelation, 16>
      operationResultBuffers;
  llvm::SmallVector<StructuredOperationBufferRelation, 16> operandBuffers;
  llvm::SmallVector<SpatialOutputBufferRelation, 4> outputBuffers;
  llvm::SmallVector<PartialReductionContributionBufferRelation, 8>
      partialReductionContributions;
  llvm::SmallVector<PartialReductionMergeInputBufferRelation, 8>
      partialReductionMergeInputs;

  void clear() {
    operationEmissions.clear();
    operationResultBuffers.clear();
    operandBuffers.clear();
    outputBuffers.clear();
    partialReductionContributions.clear();
    partialReductionMergeInputs.clear();
  }
};

/// The iteration-domain tile corresponding to one operand tile. This is a
/// transient analysis result derived from the current TilingInterface; callers
/// must not retain it across IR mutation.
struct OperandTileIterationDomain {
  llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
  llvm::SmallVector<mlir::OpFoldResult, 4> sizes;
};

/// The actual consumer implementation produced from one operand tile.
/// Ownership of operations remains with the caller's IR.
struct OperandTileMaterialization {
  OperandTileIterationDomain iterationDomain;
  llvm::SmallVector<mlir::Operation *, 2> tiledOperations;
  llvm::SmallVector<mlir::Value, 2> tiledValues;
  llvm::SmallVector<mlir::Operation *, 4> generatedSlices;
};

/// Maps an operand tile into the consumer iteration domain through
/// TilingInterface::getIterationDomainTileFromOperandTile.
mlir::FailureOr<OperandTileIterationDomain> mapOperandTileToIterationDomain(
    mlir::Operation *consumer, mlir::OpBuilder &builder, unsigned operandNumber,
    llvm::ArrayRef<mlir::OpFoldResult> offsets,
    llvm::ArrayRef<mlir::OpFoldResult> sizes, std::string *failureReason);

/// Materializes an actual consumer tile from an operand tile through
/// TilingInterface::getTiledImplementationFromOperandTile. The returned
/// iteration-domain relation is recomputed from the same current IR.
mlir::FailureOr<OperandTileMaterialization> materializeConsumerFromOperandTile(
    mlir::Operation *consumer, mlir::OpBuilder &builder, unsigned operandNumber,
    llvm::ArrayRef<mlir::OpFoldResult> offsets,
    llvm::ArrayRef<mlir::OpFoldResult> sizes, std::string *failureReason);

/// The actual partial-reduction and merge implementation produced through
/// PartialReductionOpInterface. Ownership of operations remains with the
/// caller's IR.
struct PartialReductionTileMaterialization {
  llvm::SmallVector<int, 2> reductionDimensions;
  llvm::SmallVector<mlir::Value, 2> initialValues;
  llvm::SmallVector<mlir::Operation *, 2> partialOperations;
  llvm::SmallVector<mlir::Value, 2> partialValues;
  llvm::SmallVector<mlir::Operation *, 4> generatedSlices;
  llvm::SmallVector<mlir::Operation *, 2> mergeOperations;
  llvm::SmallVector<mlir::Value, 2> mergedValues;
};

/// One output-domain shard selected for a Tile.  Output indices are
/// function-result indices.  A result omitted from a Tile's shard list is an
/// explicit no-work result on that Tile; its destination is yielded without a
/// store.  This query-local value is consumed atomically and never persisted.
struct SpatialOutputShard {
  unsigned outputIndex = 0;
  llvm::SmallVector<int64_t, 4> offsets;
  llvm::SmallVector<int64_t, 4> sizes;
  /// Static temporal tile shape for this spatial shard.  It has the same rank
  /// as `sizes`, is positive and no larger than the shard in any dimension.
  /// The lowering materializes steady scf.for traversal and finite static tail
  /// classes instead of expanding one operation per temporal wave.
  llvm::SmallVector<int64_t, 4> temporalTileSizes;
};

/// Query-local temporal tile selected for one current structured operation.
/// `iteratorTileSizes` is indexed by the operation's TilingInterface iterator
/// domain and therefore covers parallel and reduction iterators without
/// recovering semantic roles from operand positions.  The selected sizes are
/// consumed into actual loop/accumulator IR and are never persisted as a
/// schedule side channel.
struct StructuredOpTemporalTile {
  mlir::Operation *operation = nullptr;
  llvm::SmallVector<int64_t, 4> iteratorTileSizes;
  /// Permutation of exactly the iterator dimensions that produce multiple
  /// waves. Empty means canonical increasing order for legacy deterministic
  /// callers; selected search assignments always provide the explicit order.
  llvm::SmallVector<uint32_t, 4> waveLoopOrder;
};

/// Materializes a partial reduction for one iteration-domain tile and merges
/// it to the corresponding result tile. Numeric regrouping legality is checked
/// before any IR is created and fails closed when it cannot be established.
mlir::FailureOr<PartialReductionTileMaterialization>
materializePartialReductionTile(
    mlir::Operation *reduction, mlir::OpBuilder &builder,
    llvm::ArrayRef<mlir::OpFoldResult> iterationOffsets,
    llvm::ArrayRef<mlir::OpFoldResult> iterationSizes,
    std::string *failureReason);

/// As above, but merges into caller-provided result-tile destinations. This is
/// used to chain independently materialized reduction chunks through ordinary
/// SSA. Destination count and types are verified before interface mutation.
mlir::FailureOr<PartialReductionTileMaterialization>
materializePartialReductionTile(
    mlir::Operation *reduction, mlir::OpBuilder &builder,
    llvm::ArrayRef<mlir::OpFoldResult> iterationOffsets,
    llvm::ArrayRef<mlir::OpFoldResult> iterationSizes,
    mlir::ValueRange resultTileDestinations, std::string *failureReason);

/// Clones a module-preserving structured tensor program, materializes the
/// selected nonempty per-output shards and temporal traversal through
/// TilingInterface, and lowers their actual producer closures to TileRegion
/// IR. Outputs omitted from
/// `outputShards` perform no store on this Tile.  Direct target facts,
/// the card-partition execution mesh, and shared symbol declarations remain in
/// the private result module.  The source module and `module` output are
/// unchanged on failure.
///
/// `functionalArgumentCount` is the exact argument count before the owning
/// Tile materialization appended one private destination per result. This
/// helper verifies that relation instead of recovering it from argument
/// positions or types.
///
/// `currentLogicalPartition` identifies a card partition. A Tile ID
/// must never be passed through this parameter.
mlir::LogicalResult lowerSpatialOutputShardsToTileRegionModule(
    mlir::ModuleOp sourceModule, unsigned functionalArgumentCount,
    llvm::ArrayRef<SpatialOutputShard> outputShards,
    mlir::OwningOpRef<mlir::ModuleOp> &module, std::string *failureReason,
    int64_t currentLogicalPartition,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles = {},
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes = {},
    StructuredMaterializationRelations *materializationRelations = nullptr,
    bool requireOneStructuredRootPerRegion = false);

} // namespace wafer

#endif // WAFER_CONVERSION_WAFERTENSORPROGRAMTOTILEREGION_H
