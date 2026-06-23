//===- DTEOps.cpp - Wafer DTE instruction verifier implementation --------===//

#include "Wafer/IR/WaferDialect.h"

#include "OpVerifierUtils.h"

#include "llvm/ADT/STLExtras.h"

using namespace wafer;
using namespace wafer::detail;

namespace {

static void appendDTELayoutRequirement(
    llvm::SmallVectorImpl<WaferLayoutRequirement> &requirements,
    mlir::Type bufferType) {
  appendLayoutRequirement(requirements, WaferValueRole::Operand, 0,
                          bufferType);
}

static mlir::LogicalResult
verifyDTEPeer(mlir::Operation *op, mlir::IntegerAttr peerAttr) {
  return verifyLogicalRankWithinExecutionMesh(op, peerAttr.getInt(),
                                              "DTE peer logical rank");
}

static void appendDTEIssueEffect(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects, int64_t bytes) {
  appendResourceEffect(effects, WaferResourceKind::Communication,
                       WaferResourceAccess::Issue, WaferValueRole::None, 0,
                       bytes);
}

} // namespace

mlir::LogicalResult InstrDTERecvOp::verify() {
  if (mlir::failed(verifyDTEP2P(getOperation(), getBuffer(), getPeerAttr(),
                                getBytesAttr(), getToken().getType())))
    return mlir::failure();
  return verifyDTEPeer(getOperation(), getPeerAttr());
}

InstrFamily InstrDTERecvOp::getInstructionFamily() {
  return InstrFamily::DTE;
}

mlir::LogicalResult InstrDTERecvOp::verifyInstructionContract() {
  return verify();
}

void InstrDTERecvOp::collectWaferLayoutRequirements(
    llvm::SmallVectorImpl<WaferLayoutRequirement> &requirements) {
  appendDTELayoutRequirement(requirements, getBuffer().getType());
}

mlir::LogicalResult InstrDTERecvOp::verifyWaferLayoutContract() {
  llvm::SmallVector<WaferLayoutRequirement, 4> requirements;
  collectWaferLayoutRequirements(requirements);
  return verifyLayoutRequirements(getOperation(), requirements);
}

void InstrDTERecvOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  int64_t bytes = getBytesAttr().getInt();
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Write, WaferValueRole::Operand, 0,
                       bytes);
  appendDTEIssueEffect(effects, bytes);
}

mlir::LogicalResult InstrDTERecvOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 4> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
}

mlir::LogicalResult InstrDTESendOp::verify() {
  if (mlir::failed(verifyDTEP2P(getOperation(), getBuffer(), getPeerAttr(),
                                getBytesAttr(), getToken().getType())))
    return mlir::failure();
  return verifyDTEPeer(getOperation(), getPeerAttr());
}

InstrFamily InstrDTESendOp::getInstructionFamily() {
  return InstrFamily::DTE;
}

mlir::LogicalResult InstrDTESendOp::verifyInstructionContract() {
  return verify();
}

void InstrDTESendOp::collectWaferLayoutRequirements(
    llvm::SmallVectorImpl<WaferLayoutRequirement> &requirements) {
  appendDTELayoutRequirement(requirements, getBuffer().getType());
}

mlir::LogicalResult InstrDTESendOp::verifyWaferLayoutContract() {
  llvm::SmallVector<WaferLayoutRequirement, 4> requirements;
  collectWaferLayoutRequirements(requirements);
  return verifyLayoutRequirements(getOperation(), requirements);
}

void InstrDTESendOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  int64_t bytes = getBytesAttr().getInt();
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Read, WaferValueRole::Operand, 0,
                       bytes);
  appendDTEIssueEffect(effects, bytes);
}

mlir::LogicalResult InstrDTESendOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 4> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
}

mlir::LogicalResult InstrDTEWaitOp::verify() {
  return verifyDTEWaitTokens(getOperation(), getTokens());
}

InstrFamily InstrDTEWaitOp::getInstructionFamily() {
  return InstrFamily::DTE;
}

mlir::LogicalResult InstrDTEWaitOp::verifyInstructionContract() {
  return verify();
}

void InstrDTEWaitOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  for (auto [index, token] : llvm::enumerate(getTokens())) {
    (void)token;
    appendResourceEffect(effects, WaferResourceKind::Communication,
                         WaferResourceAccess::Wait, WaferValueRole::Operand,
                         index, -1);
  }
}

mlir::LogicalResult InstrDTEWaitOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 4> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
}
