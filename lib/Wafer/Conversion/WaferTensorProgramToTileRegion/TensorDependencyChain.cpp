//===- TensorDependencyChain.cpp - Structured tensor edge chain ---------===//

#include "Internal.h"

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/DependentDataflow.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <functional>

using namespace wafer;
using namespace wafer::tensor_program_to_tile_region;

mlir::FailureOr<llvm::SmallVector<mlir::Operation *, 4>>
wafer::traceProducerToConsumerChain(mlir::Operation *producer,
                                    unsigned producerResult,
                                    mlir::Operation *consumer,
                                    unsigned consumerOperand,
                                    std::string *failureReason) {
  auto failChain = [&](llvm::StringRef message)
      -> mlir::FailureOr<llvm::SmallVector<mlir::Operation *, 4>> {
    setFailureReason(failureReason, message);
    return mlir::failure();
  };
  if (!producer || !consumer || producer->getBlock() != consumer->getBlock() ||
      producerResult >= producer->getNumResults() ||
      consumerOperand >= consumer->getNumOperands())
    return failChain(
        "edge strategy does not name one in-block structured dependency");

  mlir::Value source = producer->getResult(producerResult);
  mlir::Value current = consumer->getOperand(consumerOperand);
  llvm::DenseMap<mlir::Value, bool> sourceDependencyMemo;
  std::function<bool(mlir::Value)> dependsOnSource =
      [&](mlir::Value value) -> bool {
    if (value == source)
      return true;
    auto found = sourceDependencyMemo.find(value);
    if (found != sourceDependencyMemo.end())
      return found->second;
    mlir::Operation *operation = value.getDefiningOp();
    if (!operation || operation->getBlock() != producer->getBlock() ||
        (mlir::isa<mlir::TilingInterface>(operation) &&
         mlir::isa<mlir::DestinationStyleOpInterface>(operation))) {
      sourceDependencyMemo[value] = false;
      return false;
    }
    sourceDependencyMemo[value] = false;
    const bool result =
        llvm::any_of(operation->getOperands(), [&](mlir::Value operand) {
          return mlir::isa<mlir::RankedTensorType>(operand.getType()) &&
                 dependsOnSource(operand);
        });
    sourceDependencyMemo[value] = result;
    return result;
  };

  llvm::SmallVector<mlir::Operation *, 4> reverseChain;
  llvm::DenseSet<mlir::Value> visited;
  while (current != source) {
    if (!current || !visited.insert(current).second)
      return failChain(
          "structured dependency producer-to-consumer tensor chain is cyclic");
    mlir::Operation *operation = current.getDefiningOp();
    if (!operation || operation->getBlock() != producer->getBlock() ||
        operation == producer || !mlir::isMemoryEffectFree(operation))
      return failChain("structured dependency is not a pure in-block "
                       "producer-to-consumer tensor chain");
    if (mlir::isa<mlir::TilingInterface>(operation) &&
        mlir::isa<mlir::DestinationStyleOpInterface>(operation))
      return failChain(
          "structured dependency crosses another scheduled operation");

    mlir::Value tensorSource;
    for (mlir::Value operand : operation->getOperands()) {
      if (!mlir::isa<mlir::RankedTensorType>(operand.getType()))
        continue;
      if (!dependsOnSource(operand))
        continue;
      if (tensorSource)
        return failChain((llvm::Twine("structured dependency support op ") +
                          operation->getName().getStringRef() +
                          " has multiple paths from one producer")
                             .str());
      tensorSource = operand;
    }
    if (!tensorSource)
      return failChain(
          "structured dependency tensor transform has no tensor source");
    reverseChain.push_back(operation);
    current = tensorSource;
  }
  std::reverse(reverseChain.begin(), reverseChain.end());
  return reverseChain;
}
