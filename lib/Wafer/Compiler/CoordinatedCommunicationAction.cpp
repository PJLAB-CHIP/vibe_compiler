//===- CoordinatedCommunicationAction.cpp -------------------------------===//

#include "CoordinatedCommunicationAction.h"

#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/IR/Common/OpVerifierUtils.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Verifier.h"

#include <optional>
#include <utility>

namespace wafer::compiler::detail {

CoordinatedCommunicationActionPoint::CoordinatedCommunicationActionPoint(
    CoordinatedCommunicationActionPointIdentity identity)
    : identity(std::move(identity)) {}

mlir::LogicalResult verifyCoordinatedCommunicationActionDomain(
    llvm::ArrayRef<mlir::ModuleOp> currentCanonicalInstrModules,
    const frontend::FrontendProgramVerificationResult &program,
    std::string *failureReason) {
  auto fail = [&](llvm::StringRef reason) {
    if (failureReason)
      *failureReason = reason.str();
    return mlir::failure();
  };
  if (currentCanonicalInstrModules.empty() ||
      currentCanonicalInstrModules.size() !=
          static_cast<size_t>(program.logicalRankCount))
    return fail("invalid coordinated communication action domain");

  mlir::ModuleOp firstModule = currentCanonicalInstrModules.front();
  if (!firstModule)
    return fail("communication action domain contains a null parent");
  mlir::MLIRContext *context = firstModule.getContext();
  for (mlir::ModuleOp parent : currentCanonicalInstrModules) {
    if (!parent || parent.getContext() != context ||
        containsTileDataflowOperations(parent.getOperation()) ||
        mlir::failed(mlir::verify(parent)))
      return fail(
          "communication action requires verified canonical Instr parents");
    bool hasFinalizationOwnedFacts = false;
    parent.walk([&](mlir::Operation *operation) {
      if (auto allocation = mlir::dyn_cast<mlir::memref::AllocOp>(operation))
        hasFinalizationOwnedFacts |=
            allocation->hasAttr(kWaferSPMOffsetAttrName) ||
            allocation->hasAttr(kWaferDDROffsetAttrName);
      if (auto send = mlir::dyn_cast<InstrDTESendOp>(operation))
        hasFinalizationOwnedFacts |= send.getBinding().has_value();
      if (auto recv = mlir::dyn_cast<InstrDTERecvOp>(operation))
        hasFinalizationOwnedFacts |= recv.getBinding().has_value();
      if (std::optional<NCCWorker> worker = getNCCIssueWorker(operation))
        hasFinalizationOwnedFacts |= *worker != NCCWorker::Worker0;
    });
    if (hasFinalizationOwnedFacts)
      return fail("communication action requires unplaced canonical parents");
  }
  if (failureReason)
    failureReason->clear();
  return mlir::success();
}

mlir::FailureOr<CoordinatedCommunicationAction>
materializeCoordinatedCommunicationAction(
    llvm::ArrayRef<mlir::ModuleOp> currentCanonicalInstrModules,
    const frontend::FrontendProgramVerificationResult &program,
    const CoordinatedCommunicationActionPoint &point,
    std::string *failureReason) {
  auto fail = [&](llvm::StringRef reason) {
    if (failureReason)
      *failureReason = reason.str();
    return mlir::FailureOr<CoordinatedCommunicationAction>(mlir::failure());
  };

  const CoordinatedCommunicationActionPointIdentity &identity =
      point.getIdentity();
  if (identity.providerKey.empty())
    return fail("communication action point has no provider identity");
  std::string domainFailure;
  if (mlir::failed(verifyCoordinatedCommunicationActionDomain(
          currentCanonicalInstrModules, program, &domainFailure)))
    return fail(domainFailure);

  CoordinatedCommunicationAction action;
  action.identity = identity;
  action.rankModules.reserve(currentCanonicalInstrModules.size());
  llvm::SmallVector<mlir::ModuleOp, 16> clones;
  clones.reserve(currentCanonicalInstrModules.size());
  for (mlir::ModuleOp parent : currentCanonicalInstrModules) {
    action.rankModules.push_back(mlir::cast<mlir::ModuleOp>(parent->clone()));
    clones.push_back(*action.rankModules.back());
  }

  std::string localFailure;
  if (mlir::failed(point.materialize(clones, program, &localFailure)))
    return fail(
        localFailure.empty()
            ? llvm::StringRef("communication action point was unavailable")
            : llvm::StringRef(localFailure));

  std::string outputFailure;
  if (mlir::failed(verifyCoordinatedCommunicationActionDomain(clones, program,
                                                              &outputFailure)))
    return fail(
        outputFailure.empty()
            ? llvm::StringRef("communication action did not produce canonical "
                              "Instr IR")
            : llvm::StringRef(outputFailure));

  if (failureReason)
    failureReason->clear();
  return action;
}

} // namespace wafer::compiler::detail
