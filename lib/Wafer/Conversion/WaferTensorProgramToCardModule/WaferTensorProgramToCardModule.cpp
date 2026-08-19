//===- WaferTensorProgramToCardModule.cpp - Public conversion API ---===//

#include "Internal.h"

namespace wafer {

using namespace tensor_program_to_card_module;

struct TileMaterializationSession::Impl {
  mlir::ModuleOp sourceModule;
  CardId cardId{0};
  TileMapping mapping;
  TileMaterializationPreparation preparation;
};

struct TileMaterializationSourceSession::Impl {
  TileMaterializationSourcePreparation preparation;
};

TileMaterializationSourceSession::TileMaterializationSourceSession(
    std::unique_ptr<Impl> impl)
    : impl(std::move(impl)) {}

TileMaterializationSourceSession::~TileMaterializationSourceSession() = default;
TileMaterializationSourceSession::TileMaterializationSourceSession(
    TileMaterializationSourceSession &&) noexcept = default;
TileMaterializationSourceSession &TileMaterializationSourceSession::operator=(
    TileMaterializationSourceSession &&) noexcept = default;

mlir::FailureOr<std::unique_ptr<TileMaterializationSourceSession>>
TileMaterializationSourceSession::create(
    mlir::ModuleOp sourceModule, CardId cardId,
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes,
    std::string *failureReason) {
  mlir::FailureOr<TileMaterializationSourcePreparation> preparation =
      prepareTileMaterializationSource(sourceModule, cardId, operationNodes,
                                       failureReason);
  if (mlir::failed(preparation))
    return mlir::failure();
  auto state = std::make_unique<Impl>();
  state->preparation = std::move(*preparation);
  return std::unique_ptr<TileMaterializationSourceSession>(
      new TileMaterializationSourceSession(std::move(state)));
}

TileMaterializationSession::TileMaterializationSession(
    std::unique_ptr<Impl> impl)
    : impl(std::move(impl)) {}

TileMaterializationSession::~TileMaterializationSession() = default;
TileMaterializationSession::TileMaterializationSession(
    TileMaterializationSession &&) noexcept = default;
TileMaterializationSession &TileMaterializationSession::operator=(
    TileMaterializationSession &&) noexcept = default;

mlir::FailureOr<std::unique_ptr<TileMaterializationSession>>
TileMaterializationSession::create(
    const TileMaterializationSourceSession &sourceSession,
    const TileMapping &mapping, std::string *failureReason) {
  const TileMaterializationSourcePreparation &source =
      sourceSession.impl->preparation;
  auto state = std::make_unique<Impl>();
  state->sourceModule = source.sourceModule;
  state->cardId = source.cardId;
  state->mapping = mapping;
  mlir::FailureOr<TileMaterializationPreparation> preparation =
      prepareTileMaterialization(source, state->mapping, failureReason);
  if (mlir::failed(preparation))
    return mlir::failure();
  state->preparation = std::move(*preparation);
  return std::unique_ptr<TileMaterializationSession>(
      new TileMaterializationSession(std::move(state)));
}

mlir::FailureOr<std::unique_ptr<TileMaterializationSession>>
TileMaterializationSession::create(
    mlir::ModuleOp sourceModule, CardId cardId, const TileMapping &mapping,
    std::string *failureReason,
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes) {
  mlir::FailureOr<std::unique_ptr<TileMaterializationSourceSession>> source =
      TileMaterializationSourceSession::create(sourceModule, cardId,
                                               operationNodes, failureReason);
  if (mlir::failed(source))
    return mlir::failure();
  return create(**source, mapping, failureReason);
}

mlir::LogicalResult TileMaterializationSession::lowerCardModule(
    mlir::OwningOpRef<mlir::ModuleOp> &cardModule,
    StructuredMaterializationRelations *materializationRelations,
    std::string *failureReason,
    CardModuleMaterializationStatistics *statistics) const {
  return lowerPreparedTensorProgramToCardModule(
      impl->sourceModule, impl->cardId, impl->mapping, impl->preparation,
      cardModule, failureReason, materializationRelations, statistics);
}

mlir::LogicalResult lowerTensorProgramToCardModule(
    mlir::ModuleOp sourceModule, CardId cardId, const TileMapping &mapping,
    mlir::OwningOpRef<mlir::ModuleOp> &cardModule, std::string *failureReason,
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes,
    StructuredMaterializationRelations *materializationRelations) {
  mlir::FailureOr<std::unique_ptr<TileMaterializationSession>> session =
      TileMaterializationSession::create(sourceModule, cardId, mapping,
                                         failureReason, operationNodes);
  if (mlir::failed(session))
    return mlir::failure();
  return (*session)->lowerCardModule(cardModule, materializationRelations,
                                     failureReason);
}

} // namespace wafer
