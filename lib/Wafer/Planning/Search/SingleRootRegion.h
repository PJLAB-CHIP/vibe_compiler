//===- SingleRootRegion.h - Selected placement region apply -*- C++ -*-===//

#pragma once

#include "Wafer/Analysis/Structured/CardProgramAnalysis.h"
#include "Wafer/Conversion/WaferTensorProgramToTileRegion/SingleRootTileRegion.h"

namespace wafer::compiler::detail {

struct CardSingleRootRegionMaterialization {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  StructuredMaterializationRelations relations;
};

/// Applies one exact-demand-satisfied logical shard trial to actual
/// single-root Card/Tile IR. This function performs no enumeration or
/// selection.
mlir::FailureOr<CardSingleRootRegionMaterialization>
materializeCardSingleRootRegions(mlir::ModuleOp tensorProgram,
                                 const CardProgramAnalysis &program,
                                 CardId cardId,
                                 const analysis::LogicalShardTrial &trial,
                                 std::string *failureReason = nullptr);

} // namespace wafer::compiler::detail
