//===- SingleRootRegion.h - Selected placement region apply -*- C++ -*-===//

#pragma once

#include "Wafer/Analysis/Structured/CardProgramAnalysis.h"
#include "Wafer/Conversion/WaferTensorProgramToTileRegion/SingleRootTileRegion.h"
#include "Wafer/Planning/PhysicalDataflow/RootRegionWorkAnalysis.h"

namespace wafer::compiler::detail {

struct CardSingleRootRegionMaterialization {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  StructuredMaterializationRelations relations;
};

/// Applies one exact-demand-satisfied spatial assignment to actual
/// single-root Card/Tile IR. This function performs no enumeration or
/// selection.
mlir::FailureOr<CardSingleRootRegionMaterialization>
materializeCardSingleRootRegions(mlir::ModuleOp tensorProgram,
                                 const CardProgramAnalysis &program,
                                 CardId cardId,
                                 const SpatialAssignment &spatial,
                                 const analysis::ExactDemandProof &demand,
                                 std::string *failureReason = nullptr);

} // namespace wafer::compiler::detail
