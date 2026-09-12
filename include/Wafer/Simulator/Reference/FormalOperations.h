//===- FormalOperations.h - Validated formal operations -------*- C++ -*-===//

#ifndef WAFER_TARGET_FORMALOPERATIONS_H
#define WAFER_TARGET_FORMALOPERATIONS_H

#include "Wafer/Target/TargetOperation.h"
#include "Wafer/Target/PhysicalTensor/PhysicalTensorDescriptor.h"
#include "Wafer/Target/PhysicalTensor/NumericCodec.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace wafer {

struct FormalConvertOperation {
  uint16_t opcode;
  PhysicalTensorDescriptor source;
  PhysicalTensorDescriptor destination;
  std::optional<TargetConvertParameter> parameter;

  friend bool operator==(const FormalConvertOperation &lhs,
                         const FormalConvertOperation &rhs) {
    return lhs.opcode == rhs.opcode && lhs.source == rhs.source &&
           lhs.destination == rhs.destination && lhs.parameter == rhs.parameter;
  }
};

struct FormalElementwiseOperation {
  TargetElementwiseOperation operation;
  std::vector<PhysicalTensorDescriptor> inputs;
  PhysicalTensorDescriptor destination;

  friend bool operator==(const FormalElementwiseOperation &lhs,
                         const FormalElementwiseOperation &rhs) {
    return lhs.operation == rhs.operation && lhs.inputs == rhs.inputs &&
           lhs.destination == rhs.destination;
  }
};

/// Exact GEMM dimension mapping. Factories accept the mapping explicitly and
/// reject every non-canonical permutation; this keeps the command key
/// faithful to the terminal instruction fields without admitting alternate
/// predicate paths in capability matching.
struct FormalGemmGeometry {
  std::vector<uint64_t> lhsBatchDimensions;
  uint64_t lhsMDimension = 0;
  uint64_t lhsContractingDimension = 0;
  std::vector<uint64_t> rhsBatchDimensions;
  uint64_t rhsContractingDimension = 0;
  uint64_t rhsNDimension = 0;
  std::vector<uint64_t> destinationBatchDimensions;
  uint64_t destinationMDimension = 0;
  uint64_t destinationNDimension = 0;

  friend bool operator==(const FormalGemmGeometry &lhs,
                         const FormalGemmGeometry &rhs) {
    return lhs.lhsBatchDimensions == rhs.lhsBatchDimensions &&
           lhs.lhsMDimension == rhs.lhsMDimension &&
           lhs.lhsContractingDimension == rhs.lhsContractingDimension &&
           lhs.rhsBatchDimensions == rhs.rhsBatchDimensions &&
           lhs.rhsContractingDimension == rhs.rhsContractingDimension &&
           lhs.rhsNDimension == rhs.rhsNDimension &&
           lhs.destinationBatchDimensions == rhs.destinationBatchDimensions &&
           lhs.destinationMDimension == rhs.destinationMDimension &&
           lhs.destinationNDimension == rhs.destinationNDimension;
  }
};

llvm::Expected<FormalGemmGeometry>
getCanonicalFormalGemmGeometry(uint64_t rank);

struct FormalGemmOperation {
  PhysicalTensorDescriptor lhs;
  PhysicalTensorDescriptor rhs;
  PhysicalTensorDescriptor destination;
  uint16_t m;
  uint16_t k;
  uint16_t n;
  uint16_t batchCount;
  FormalGemmGeometry axes;
  TargetGemmOrientation lhsOrientation;
  TargetGemmOrientation rhsOrientation;
  std::optional<PhysicalTensorDescriptor> psum;

  friend bool operator==(const FormalGemmOperation &lhs,
                         const FormalGemmOperation &rhs) {
    return lhs.lhs == rhs.lhs && lhs.rhs == rhs.rhs &&
           lhs.destination == rhs.destination && lhs.m == rhs.m &&
           lhs.k == rhs.k && lhs.n == rhs.n &&
           lhs.batchCount == rhs.batchCount && lhs.axes == rhs.axes &&
           lhs.lhsOrientation == rhs.lhsOrientation &&
           lhs.rhsOrientation == rhs.rhsOrientation && lhs.psum == rhs.psum;
  }
};

struct FormalReduceOperation {
  TargetReduceOperation operation;
  PhysicalTensorDescriptor input;
  PhysicalTensorDescriptor destination;
  TargetReduceDimension dimension;

  friend bool operator==(const FormalReduceOperation &lhs,
                         const FormalReduceOperation &rhs) {
    return lhs.operation == rhs.operation && lhs.input == rhs.input &&
           lhs.destination == rhs.destination && lhs.dimension == rhs.dimension;
  }
};

llvm::Expected<FormalConvertOperation>
createFormalConvertOperation(uint16_t opcode, PhysicalTensorDescriptor source,
                             PhysicalTensorDescriptor destination,
                             std::optional<TargetConvertParameter> parameter);

llvm::Expected<FormalElementwiseOperation>
createFormalElementwiseOperation(TargetElementwiseOperation operation,
                                 std::vector<PhysicalTensorDescriptor> inputs,
                                 PhysicalTensorDescriptor destination);

llvm::Expected<FormalGemmOperation> createFormalGemmOperation(
    PhysicalTensorDescriptor lhs, PhysicalTensorDescriptor rhs,
    PhysicalTensorDescriptor destination, uint32_t m, uint32_t k, uint32_t n,
    uint32_t batchCount, FormalGemmGeometry geometry,
    TargetGemmOrientation lhsOrientation = TargetGemmOrientation::Normal,
    TargetGemmOrientation rhsOrientation = TargetGemmOrientation::Normal,
    std::optional<PhysicalTensorDescriptor> psum = std::nullopt);

llvm::Expected<FormalReduceOperation> createFormalReduceOperation(
    TargetReduceOperation operation, PhysicalTensorDescriptor input,
    PhysicalTensorDescriptor destination, TargetReduceDimension dimension);

} // namespace wafer

#endif // WAFER_TARGET_FORMALOPERATIONS_H
