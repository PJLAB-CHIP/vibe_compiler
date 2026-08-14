//===- TileRegionLocalFit.cpp - Region-local SPM evaluation ------------===//

#include "TileRegionLocalFit.h"

#include "CardExecutableCompilation.h"

#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/Support/CompileTiming.h"
#include "Wafer/Support/TargetPolicy.h"
#include "Wafer/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <utility>

namespace wafer::compiler::detail {
namespace {

static TileRegionLocalFitResult
fail(TileRegionLocalFitStatus status, llvm::StringRef gate,
     llvm::StringRef detail, PhysicalTileFinalizationFailure failure = {}) {
  TileRegionLocalFitResult result;
  result.status = status;
  result.gate = gate.str();
  result.detail = detail.str();
  result.finalization = std::move(failure);
  return result;
}

static PhysicalTileFinalizationFailure
makeSPMFailure(const SPMMemoryPlanningFailure &spm) {
  PhysicalTileFinalizationFailure result;
  result.kind = PhysicalTileFinalizationFailureKind::SPMAllocation;
  result.spmCapacityOverflow =
      spm.kind == SPMMemoryPlanningFailureKind::CapacityOverflow;
  result.spmPlanningFailureKind = spm.kind;
  result.spmLargestDemandLocation = spm.largestDemandLocation;
  result.spmLargestDemandType = spm.largestDemandType;
  result.spmLargestDemandBytes = spm.largestDemandBytes;
  result.spmDemandCount = spm.demandCount;
  auto appendEvidence = [](const SPMMemoryPlanningFailure::DemandEvidence &from,
                           auto &to) {
    PhysicalTileFinalizationFailure::SPMDemandEvidence evidence{
        from.location, from.type, from.bytes, {}};
    evidence.userLocations.append(from.userLocations.begin(),
                                  from.userLocations.end());
    to.push_back(std::move(evidence));
  };
  for (const auto &demand : spm.largestDemands)
    appendEvidence(demand, result.spmLargestDemands);
  for (const auto &demand : spm.capacityConflictDemands)
    appendEvidence(demand, result.spmCapacityConflictDemands);
  for (const auto &demand : spm.individuallyOversizedDemands)
    appendEvidence(demand, result.spmIndividuallyOversizedDemands);
  return result;
}

class IsolatedTileRegionTransaction {
public:
  static mlir::FailureOr<std::unique_ptr<IsolatedTileRegionTransaction>>
  create(TileRegionOp source, std::string &detail) {
    auto transaction = std::make_unique<IsolatedTileRegionTransaction>();
    auto *scratchBlock = new mlir::Block();
    transaction->scratchRegion.push_back(scratchBlock);

    llvm::SmallVector<mlir::Location, 4> argumentLocations(
        source.getInputs().size(), source.getLoc());
    llvm::SmallVector<mlir::Type, 4> argumentTypes;
    argumentTypes.reserve(source.getInputs().size());
    for (mlir::Value input : source.getInputs())
      argumentTypes.push_back(input.getType());
    scratchBlock->addArguments(argumentTypes, argumentLocations);
    mlir::IRMapping mapping;
    for (auto [input, scratchArgument] :
         llvm::zip_equal(source.getInputs(), scratchBlock->getArguments()))
      mapping.map(input, scratchArgument);

    mlir::OpBuilder builder(source.getContext());
    builder.setInsertionPointToEnd(scratchBlock);
    transaction->region = mlir::cast<TileRegionOp>(
        builder.clone(*source.getOperation(), mapping));
    if (mlir::failed(mlir::verify(transaction->region))) {
      detail = "mapped TileRegion transaction is not verifier-legal";
      return mlir::failure();
    }
    return transaction;
  }

  TileRegionOp getRegion() const { return region; }

private:
  mlir::Region scratchRegion;
  TileRegionOp region;
};

} // namespace

TileRegionLocalFitResult
evaluateTileRegionLocalFit(TileRegionOp region,
                           TileRegionToInstrLoweringSession &loweringSession,
                           llvm::raw_ostream &diagnostics) {
  wafer::support::ScopedCompileTimingSpan timing(
      "stage", "tile-region-local-fit", "tile-region-local-fit");
  auto reportFailure = [&](TileRegionLocalFitResult result) {
    timing.markFailed();
    diagnostics << "wafer-compile: tile-region-local-fit outcome="
                << (result.isProvenExactRejection() ? "exact-rejection"
                                                    : "indeterminate")
                << " gate=" << result.gate << " detail=" << result.detail
                << '\n';
    return result;
  };

  std::string detail;
  if (!region)
    return reportFailure(fail(
        TileRegionLocalFitStatus::IndeterminateFailure, "local-fit-contract",
        "local-fit requires one materialized TileRegion"));
  bool hasCall = false;
  region.walk([&](mlir::CallOpInterface) { hasCall = true; });
  if (hasCall)
    return reportFailure(fail(
        TileRegionLocalFitStatus::IndeterminateFailure, "local-fit-contract",
        "TileRegion local-fit cannot prove an external call closure"));

  auto isolated = IsolatedTileRegionTransaction::create(region, detail);
  if (mlir::failed(isolated))
    return reportFailure(fail(TileRegionLocalFitStatus::IndeterminateFailure,
                              "local-fit-contract", detail));
  TileRegionOp isolatedRegion = (*isolated)->getRegion();
  if (mlir::failed(
          convertTileRegionToInstr(isolatedRegion, loweringSession, &detail)) ||
      containsTileDataflowOperations(isolatedRegion.getOperation()) ||
      mlir::failed(mlir::verify(isolatedRegion)))
    return reportFailure(fail(TileRegionLocalFitStatus::IndeterminateFailure,
                              "tile-region-to-instr", detail));

  const TargetMemoryPolicy memory = getDefaultWaferTargetPolicy().memory;
  SPMMemoryPlanningFailure spmFailure;
  if (mlir::failed(checkTileRegionSPMCapacity(
          isolatedRegion, memory.spmBase, memory.spmLimit, memory.spmAlignment,
          &spmFailure))) {
    PhysicalTileFinalizationFailure finalization = makeSPMFailure(spmFailure);
    const bool exact =
        spmFailure.kind == SPMMemoryPlanningFailureKind::CapacityOverflow;
    return reportFailure(
        fail(exact ? TileRegionLocalFitStatus::ProvenExactRejection
                   : TileRegionLocalFitStatus::IndeterminateFailure,
             "spm-allocation",
             exact ? "TileRegion failed an exact local SPM fit gate"
                   : "TileRegion local SPM fit is indeterminate",
             std::move(finalization)));
  }

  TileRegionLocalFitResult result;
  result.status = TileRegionLocalFitStatus::Fits;
  return result;
}

} // namespace wafer::compiler::detail
