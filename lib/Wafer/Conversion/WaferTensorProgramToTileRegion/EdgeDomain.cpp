//===- EdgeDomain.cpp - Selected edge structural domains ---------===//

#include "EdgeDomain.h"

#include "llvm/ADT/STLExtras.h"

namespace wafer::tensor_program_to_tile_region {
namespace {

mlir::LogicalResult failResult(std::string *failureReason,
                               llvm::StringRef message) {
  setFailureReason(failureReason, message);
  return mlir::failure();
}

} // namespace

mlir::LogicalResult validateEdge(mlir::Operation *producer,
                                 unsigned producerResult,
                                 mlir::Operation *consumer,
                                 unsigned consumerOperand, mlir::Block &body,
                                 std::string *failureReason) {
  if (!producer || !consumer || producer->getBlock() != &body ||
      consumer->getBlock() != &body ||
      producerResult >= producer->getNumResults() ||
      consumerOperand >= consumer->getNumOperands())
    return failResult(failureReason,
                      "edge strategy must name one current-SSA dependency");
  if (producerResult != 0 || producer->getNumResults() != 1 ||
      consumer->getNumResults() != 1)
    return failResult(
        failureReason,
        "edge strategy requires one tiled producer and consumer result");
  return mlir::success();
}

mlir::LogicalResult validateStaticDomain(mlir::RankedTensorType type,
                                         llvm::ArrayRef<int64_t> offsets,
                                         llvm::ArrayRef<int64_t> sizes,
                                         std::string *failureReason,
                                         llvm::StringRef subject) {
  if (!type || !type.hasStaticShape() ||
      offsets.size() != static_cast<size_t>(type.getRank()) ||
      sizes.size() != offsets.size() ||
      llvm::any_of(llvm::zip_equal(offsets, sizes, type.getShape()),
                   [](auto values) {
                     auto [offset, size, extent] = values;
                     return offset < 0 || size <= 0 || extent <= 0 ||
                            offset > extent - size;
                   }))
    return failResult(failureReason,
                      (subject + " has an invalid static domain").str());
  return mlir::success();
}

} // namespace wafer::tensor_program_to_tile_region
