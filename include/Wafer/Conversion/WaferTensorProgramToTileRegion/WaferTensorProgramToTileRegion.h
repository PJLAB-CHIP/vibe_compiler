//===- WaferTensorProgramToTileRegion.h - Tensor program lowering -*- C++
//-*-===//

#ifndef WAFER_CONVERSION_WAFERTENSORPROGRAMTOTILEREGION_H
#define WAFER_CONVERSION_WAFERTENSORPROGRAMTOTILEREGION_H

#include "Wafer/IR/WaferInterfaces.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <string>

namespace wafer {

/// Describes how a source operation may terminate a candidate traversal.
/// This is a structural materialization capability, not a profitability or
/// target-legality result.
enum class CandidateTraversalRootCapability {
  Tiled,
  FullTraversalOnly,
  Unsupported,
};

CandidateTraversalRootCapability
classifyCandidateTraversalRoot(mlir::Operation *operation);

/// Selects the tile seed used while materializing one discardable complete
/// candidate clone. This is transient transformation input: it is never
/// persisted in IR, rank-frontier metadata, or an artifact.
enum class CandidateTileTraversalKind : uint8_t {
  ResultDriven,
  OperandDriven,
};

/// Selects how one complete-rank candidate composes structured traversals.
/// This is invocation-local transformation input and is not persisted in IR.
enum class CompleteRankTraversalComposition : uint8_t {
  Coupled,
  Separated,
};

/// One atomic realization of a producer-result to consumer-op connection.
/// The vector of choices passed to complete-rank materialization is ordered by
/// current SSA/block traversal and is destroyed with that invocation.  The
/// selected realization is represented only by the resulting Tile IR.
enum class CandidateTraversalConnectionAction : uint8_t {
  CoupledResident,
  SeparatedResident,
  SeparatedDDR,
  CrossRegion,
  SelectiveSpill,
};

/// Complete transient parameters for one current-SSA connection.  The
/// producer vector is in the identified producer-result domain; the consumer
/// vector is in the consumer result traversal domain.  An empty vector selects
/// that side's full static result shape.  Coupled realization is driven only
/// by the consumer traversal: its producer vector must be empty and the exact
/// demand for the identified consumer operand is derived by TilingInterface
/// and the intervening typed SSA view relation.  Separated realizations may
/// select the two traversal vectors independently.  Nothing in this object is
/// persisted after the actual Tile clone is materialized.
struct CandidateTraversalConnectionChoice {
  CandidateTraversalConnectionAction action =
      CandidateTraversalConnectionAction::CoupledResident;
  llvm::SmallVector<int64_t, 4> producerTileSizes;
  llvm::SmallVector<int64_t, 4> consumerTileSizes;
};

/// Query-local identity and static domains for one exact
/// producer-result-to-consumer-operand SSA connection.  Operation ordinals are
/// positions in the queried standalone function block; together with the
/// result/operand numbers they provide a deterministic identity without using
/// symbols or operation names.  They are invalidated by IR mutation and are
/// never persisted as candidate metadata.
struct CandidateTraversalConnectionDomain {
  unsigned producerOperationOrdinal = 0;
  unsigned producerResultNumber = 0;
  unsigned consumerOperationOrdinal = 0;
  unsigned consumerOperandNumber = 0;
  /// Conservative byte-addressable storage width derived from each exact SSA
  /// type. Sub-byte integer widths occupy one storage byte for this structural
  /// estimate; unsupported element types make the topology query fail closed.
  uint64_t producerResultElementBytes = 0;
  uint64_t consumerOperandElementBytes = 0;
  uint64_t consumerResultElementBytes = 0;
  llvm::SmallVector<int64_t, 4> producerResultShape;
  llvm::SmallVector<int64_t, 4> consumerOperandShape;
  llvm::SmallVector<int64_t, 4> consumerResultShape;
};

struct CandidateTraversalConnectionTopology {
  unsigned connectionCount = 0;
  /// True when one producer result reaches more than one structured consumer
  /// operand; such a fanout can reconverge (including at two operands of the
  /// same operation) and therefore requires the general-DAG beam instead of
  /// the chain/tree DP table.
  bool requiresGeneralDAGBeam = false;
  /// Current-IR result domains in the same stable order as connection
  /// choices.  These are query-local analysis facts, not candidate metadata.
  llvm::SmallVector<CandidateTraversalConnectionDomain, 16> domains;
};

mlir::FailureOr<CandidateTraversalConnectionTopology>
getCompleteRankCandidateConnectionTopology(
    mlir::ModuleOp sourceModule, std::string *failureReason = nullptr);

/// Counts the finite structured producer-result to consumer-op connections in
/// one standalone complete-rank tensor program.  View chains are followed by
/// typed SSA; symbol names and operation spelling are not correspondence keys.
mlir::FailureOr<unsigned>
getCompleteRankCandidateConnectionCount(mlir::ModuleOp sourceModule,
                                        std::string *failureReason = nullptr);

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

/// Materializes a partial reduction for one iteration-domain tile and merges
/// it to the corresponding result tile. Numeric regrouping legality is checked
/// before any IR is created and fails closed when it cannot be established.
/// Transformation callers should invoke this on a discardable candidate clone
/// so a later interface failure remains atomic at candidate granularity.
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

namespace detail {

enum class CheckedStaticTileProductStatus { Success, InvalidInput, Overflow };

/// Computes product_i ceil(ranges[i] / tileSizes[i]) without signed or unsigned
/// overflow. Positive ranges and in-bounds positive tile sizes are required.
CheckedStaticTileProductStatus
checkedStaticTileProduct(llvm::ArrayRef<int64_t> ranges,
                         llvm::ArrayRef<int64_t> tileSizes, uint64_t &product);

/// Clones a standalone structured scheduling function into an owning module.
/// The function contract is: one entry block; entry arguments are read-only
/// inputs followed by output destinations; the number of output destinations
/// equals the function result count; func.return yields those result roots.
mlir::OwningOpRef<mlir::ModuleOp>
cloneTensorProgramToStandaloneModule(mlir::func::FuncOp function);

} // namespace detail

mlir::LogicalResult lowerTensorProgramToTileRegionModule(
    mlir::func::FuncOp function, mlir::OwningOpRef<mlir::ModuleOp> &module,
    std::string *failureReason, int64_t currentLogicalRank,
    std::optional<TargetImplementationKind> selectedAlternative = std::nullopt,
    bool useDirectMappedBoundaryTransfer = false);

/// Clones a module containing one complete rank function and lowers that
/// function to one outer Tile residency region while preserving module-level
/// target facts and referenced globals. This is the conservative complete-rank
/// decision boundary; source scheduling scopes are not lowering units.
mlir::LogicalResult lowerCompleteRankTensorProgramToTileRegionModule(
    mlir::ModuleOp sourceModule, mlir::OwningOpRef<mlir::ModuleOp> &module,
    std::string *failureReason, int64_t currentLogicalRank);

/// Clones a complete-rank module, materializes the requested actual structured
/// traversal using explicit tile sizes, and lowers the whole clone to one
/// unplaced Tile program. The composition request is consumed during this
/// call; no candidate descriptor or schedule side data survives in the IR.
mlir::LogicalResult lowerCompleteRankCandidateTensorProgramToTileRegionModule(
    mlir::ModuleOp sourceModule, llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    CandidateTileTraversalKind traversalKind,
    CompleteRankTraversalComposition composition,
    mlir::OwningOpRef<mlir::ModuleOp> &module, std::string *failureReason,
    int64_t currentLogicalRank,
    std::optional<TargetImplementationKind> selectedAlternative = std::nullopt,
    bool useDirectMappedBoundaryTransfer = false);

/// As above, but materializes one current-SSA-derived action for every
/// structured connection. Coupled edges are fused into the consumer traversal;
/// separated edges materialize a shared physical version in SPM or DDR. A
/// cross-region choice first materializes a DDR-clean boundary; region
/// partition itself remains a Tile-IR transformation after this conversion.
/// An empty tile vector selects each connected traversal's own full static
/// result shape; this is the independent-retile form for mixed-shape graphs.
mlir::LogicalResult lowerCompleteRankConnectionTensorProgramToTileRegionModule(
    mlir::ModuleOp sourceModule, llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<CandidateTraversalConnectionAction> connectionActions,
    mlir::OwningOpRef<mlir::ModuleOp> &module, std::string *failureReason,
    int64_t currentLogicalRank,
    std::optional<TargetImplementationKind> selectedAlternative = std::nullopt,
    bool useDirectMappedBoundaryTransfer = false);

/// Materializes explicit per-side tile parameters for every current-SSA
/// connection.  This is the joint-search entry point; choices are consumed
/// atomically and only their realized Tile IR survives.
mlir::LogicalResult
lowerCompleteRankConnectionChoicesTensorProgramToTileRegionModule(
    mlir::ModuleOp sourceModule,
    llvm::ArrayRef<CandidateTraversalConnectionChoice> connectionChoices,
    mlir::OwningOpRef<mlir::ModuleOp> &module, std::string *failureReason,
    int64_t currentLogicalRank,
    std::optional<TargetImplementationKind> selectedAlternative = std::nullopt,
    bool useDirectMappedBoundaryTransfer = false);

/// Verifies that replacing one structured reduction by more than one ordered
/// chunk, including neutral-initialized partials and chunk-result combines, is
/// permitted by the source IR's numeric semantics. This is a transformation
/// legality gate, not a target capability or profitability query.
mlir::LogicalResult
verifyCandidateReductionSplitNumericLegality(mlir::linalg::LinalgOp root,
                                             std::string *failureReason);

mlir::LogicalResult lowerCandidateTensorProgramToTileRegionModule(
    mlir::func::FuncOp function, llvm::ArrayRef<int64_t> candidateTileOffsets,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    mlir::OwningOpRef<mlir::ModuleOp> &module, std::string *failureReason,
    int64_t currentLogicalRank,
    std::optional<TargetImplementationKind> selectedAlternative = std::nullopt,
    bool useDirectMappedBoundaryTransfer = false);

/// Materializes a compact structured traversal of a standalone tensor
/// program and lowers it to tile-region IR.
mlir::LogicalResult lowerCompleteCandidateTensorProgramToTileRegionModule(
    mlir::func::FuncOp function, llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    mlir::OwningOpRef<mlir::ModuleOp> &module, std::string *failureReason,
    int64_t currentLogicalRank,
    std::optional<TargetImplementationKind> selectedAlternative = std::nullopt,
    bool useDirectMappedBoundaryTransfer = false,
    CandidateTileTraversalKind traversalKind =
        CandidateTileTraversalKind::ResultDriven);

} // namespace wafer

#endif // WAFER_CONVERSION_WAFERTENSORPROGRAMTOTILEREGION_H
