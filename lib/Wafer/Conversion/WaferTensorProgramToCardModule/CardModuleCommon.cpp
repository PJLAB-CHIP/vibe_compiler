//===- CardModuleCommon.cpp - Shared CardModule lowering support -----===//

#include "Internal.h"

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/DependentDataflow.h"
#include "Wafer/Conversion/WaferTensorProgramToTileRegion/WaferTensorProgramToTileRegion.h"
#include "Wafer/IR/Target/TargetTopology.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/BoundedParallel.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <string>
#include <utility>

namespace wafer::tensor_program_to_card_module {

bool isCardSharedDeclaration(mlir::Operation &operation) {
  if (!mlir::isa<mlir::SymbolOpInterface>(&operation))
    return false;
  return llvm::all_of(operation.getRegions(),
                      [](mlir::Region &region) { return region.empty(); });
}

bool relationsBelongTo(mlir::Operation *root,
                       const StructuredMaterializationRelations &relations) {
  llvm::DenseSet<const void *> liveValues;
  root->walk([&](mlir::Operation *operation) {
    for (mlir::Value result : operation->getResults())
      liveValues.insert(result.getAsOpaquePointer());
    for (mlir::Region &region : operation->getRegions())
      for (mlir::Block &block : region)
        for (mlir::BlockArgument argument : block.getArguments())
          liveValues.insert(argument.getAsOpaquePointer());
  });
  auto allLive = [&](const auto &entries) {
    return llvm::all_of(entries, [&](const auto &entry) {
      return entry.buffer &&
             liveValues.contains(entry.buffer.getAsOpaquePointer());
    });
  };
  return allLive(relations.operationResultBuffers) &&
         allLive(relations.operandBuffers) && allLive(relations.outputBuffers);
}

mlir::FailureOr<mlir::func::FuncOp>
getSourceTensorProgram(mlir::ModuleOp module, std::string *failureReason) {
  mlir::func::FuncOp program;
  for (mlir::func::FuncOp function : module.getOps<mlir::func::FuncOp>()) {
    if (function.isExternal())
      continue;
    if (program)
      return failCardModuleValue<mlir::func::FuncOp>(
          failureReason,
          "card spatial mapping requires exactly one defined direct "
          "tensor-program function");
    program = function;
  }
  if (!program)
    return failCardModuleValue<mlir::func::FuncOp>(
        failureReason,
        "card spatial mapping requires exactly one defined direct "
        "tensor-program function");
  if (!program.getBody().hasOneBlock())
    return failCardModuleValue<mlir::func::FuncOp>(
        failureReason, "card spatial mapping requires a single-block tensor "
                       "program");
  if (program.getNumResults() == 0)
    return failCardModuleValue<mlir::func::FuncOp>(
        failureReason, "card spatial mapping requires tensor-program results");

  auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(
      program.getBody().front().getTerminator());
  if (!returnOp || returnOp.getNumOperands() != program.getNumResults())
    return failCardModuleValue<mlir::func::FuncOp>(
        failureReason,
        "card spatial mapping requires a complete functional return");
  for (unsigned index = 0; index < program.getNumResults(); ++index) {
    mlir::Type resultType = program.getResultTypes()[index];
    if (!mlir::isa<mlir::RankedTensorType>(resultType) ||
        returnOp.getOperand(index).getType() != resultType)
      return failCardModuleValue<mlir::func::FuncOp>(
          failureReason, "card spatial mapping requires ranked tensor results");
  }
  return program;
}

mlir::LogicalResult verifyLogicalMesh(mlir::ModuleOp module,
                                      std::string *failureReason) {
  llvm::SmallVector<ExecutionMeshOp, 2> meshes(
      module.getOps<ExecutionMeshOp>());
  if (meshes.size() != 1)
    return failCardModule(
        failureReason,
        "card spatial mapping requires exactly one direct logical "
        "execution mesh");

  int64_t partitionCount = 1;
  for (int64_t dimension : meshes.front().getShapeAttr().asArrayRef()) {
    if (dimension <= 0 ||
        partitionCount > std::numeric_limits<int64_t>::max() / dimension)
      return failCardModule(failureReason,
                            "logical execution mesh is not representable");
    partitionCount *= dimension;
  }
  if (partitionCount != 1)
    return failCardModule(
        failureReason,
        "card spatial mapping currently requires one logical card "
        "partition");
  return mlir::success();
}

mlir::FailureOr<llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4>>
getStaticOutputDomains(mlir::func::FuncOp program, std::string *failureReason) {
  llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4> domains;
  domains.reserve(program.getNumResults());
  for (mlir::Type type : program.getResultTypes()) {
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type);
    if (!tensorType || !tensorType.hasStaticShape())
      return failCardModuleValue<
          llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4>>(
          failureReason, "card spatial mapping requires static tensor-program "
                         "output shapes");
    if (llvm::any_of(tensorType.getShape(),
                     [](int64_t extent) { return extent <= 0; }))
      return failCardModuleValue<
          llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4>>(
          failureReason,
          "card spatial mapping requires nonempty static output domains");
    domains.emplace_back(tensorType.getShape());
  }
  return domains;
}

mlir::FailureOr<mlir::func::FuncOp>
takeLoweredTensorProgram(mlir::ModuleOp shardModule,
                         std::string *failureReason) {
  mlir::func::FuncOp program;
  for (mlir::func::FuncOp function : shardModule.getOps<mlir::func::FuncOp>()) {
    if (function.isExternal())
      continue;
    if (program)
      return failCardModuleValue<mlir::func::FuncOp>(
          failureReason,
          "spatial output lowering produced more than one executable "
          "function");
    program = function;
  }
  if (!program)
    return failCardModuleValue<mlir::func::FuncOp>(
        failureReason,
        "spatial output lowering produced no executable function");
  program->remove();
  return program;
}

mlir::FailureOr<mlir::func::FuncOp>
createNoWorkEntry(mlir::func::FuncOp sourceProgram,
                  std::string *failureReason) {
  // A no-work Tile needs the verified symbol/signature contract, not a copy
  // of the executable body that is immediately discarded. Preserve the op
  // shell and construct the only region state this module can contain.
  auto entry =
      mlir::cast<mlir::func::FuncOp>(sourceProgram->cloneWithoutRegions());
  mlir::Block *block = entry.addEntryBlock();
  for (mlir::Type resultType : entry.getResultTypes())
    entry.insertArgument(entry.getNumArguments(), resultType,
                         mlir::DictionaryAttr{}, entry.getLoc());
  mlir::OpBuilder builder(block, block->end());
  mlir::ValueRange outputs(block->getArguments());
  outputs = outputs.take_back(entry.getNumResults());
  builder.create<mlir::func::ReturnOp>(entry.getLoc(), outputs);
  if (mlir::failed(mlir::verify(entry))) {
    entry->destroy();
    return failCardModuleValue<mlir::func::FuncOp>(
        failureReason, "failed to create verifier-legal no-work entry");
  }
  return entry;
}

/// Removes the private Tile output destinations after TileRegion
/// materialization. Outputs remain full-card tensors: active Tiles write only
/// their selected subviews, while no-work Tiles perform no writes. Replacing
/// each internal destination by tensor.empty makes the eventual
/// compiler-owned allocation the single output root; target ABI preparation
/// later replaces that root with the user output pointer through its verified
/// result path. The source argument boundary is never inferred or changed.
mlir::LogicalResult removeTileOutputDestinations(
    mlir::func::FuncOp entry, unsigned sourceArgumentCount,
    StructuredMaterializationRelations &relations, std::string *failureReason,
    uint64_t boundaryArgumentCount) {
  const unsigned resultCount = entry.getNumResults();
  if (resultCount == 0 || entry.getNumArguments() != sourceArgumentCount +
                                                         resultCount +
                                                         boundaryArgumentCount)
    return failCardModule(
        failureReason, "Tile entry does not have the exact private scheduling "
                       "boundary");
  const unsigned outputBase = sourceArgumentCount;
  mlir::OpBuilder builder(&entry.getBody().front(),
                          entry.getBody().front().begin());
  llvm::BitVector eraseArguments(entry.getNumArguments());
  for (unsigned index = 0; index < resultCount; ++index) {
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(entry.getResultTypes()[index]);
    mlir::BlockArgument destination = entry.getArgument(outputBase + index);
    if (!resultType || !resultType.hasStaticShape() ||
        destination.getType() != resultType)
      return failCardModule(
          failureReason, "Tile scheduling destination is not a static tensor");
    auto empty = builder.create<mlir::tensor::EmptyOp>(
        entry.getLoc(), resultType.getShape(), resultType.getElementType());
    auto retarget = [&](auto &entries) {
      for (auto &relation : entries)
        if (relation.buffer == destination)
          relation.buffer = empty.getResult();
    };
    retarget(relations.operationResultBuffers);
    retarget(relations.operandBuffers);
    retarget(relations.outputBuffers);
    destination.replaceAllUsesWith(empty.getResult());
    eraseArguments.set(outputBase + index);
  }
  entry.eraseArguments(eraseArguments);
  if (mlir::failed(mlir::verify(entry))) {
    llvm::errs() << "--- entry IR on boundary verify failure ---\n";
    entry->getParentOfType<mlir::ModuleOp>().print(llvm::errs());
    return failCardModule(
        failureReason, "Tile functional result boundary is not verifier-legal");
  }
  return mlir::success();
}

mlir::LogicalResult verifyDirectSourceMembers(mlir::ModuleOp sourceModule,
                                              mlir::func::FuncOp sourceProgram,
                                              std::string *failureReason) {
  for (mlir::Operation &operation :
       sourceModule.getBody()->without_terminator()) {
    if (&operation == sourceProgram.getOperation() ||
        mlir::isa<TargetTopologyOp, ExecutionMeshOp>(operation) ||
        isCardSharedDeclaration(operation))
      continue;
    return failCardModule(
        failureReason,
        (llvm::Twine("card spatial mapping does not support direct "
                     "module operation '") +
         operation.getName().getStringRef() + "'")
            .str());
  }
  return mlir::success();
}

void cloneModuleFacts(mlir::ModuleOp sourceModule,
                      mlir::ModuleOp destinationModule) {
  mlir::OpBuilder builder(destinationModule.getBodyRegion());
  mlir::IRMapping mapping;
  for (TargetTopologyOp topology : sourceModule.getOps<TargetTopologyOp>())
    builder.clone(*topology.getOperation(), mapping);
  for (ExecutionMeshOp mesh : sourceModule.getOps<ExecutionMeshOp>())
    builder.clone(*mesh.getOperation(), mapping);
}

} // namespace wafer::tensor_program_to_card_module
