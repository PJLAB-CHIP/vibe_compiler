//===- DistributedBoundary.cpp - Distributed boundary validation -----===//

#include "Wafer/Frontend/Program/ProgramInternal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <vector>

using namespace mlir;

namespace wafer::frontend::program_detail {

namespace {

wafer::frontend::ProgramPartitionSlice
getVerifiedPartitionSlice(const DistributedBoundaryPartition &partition) {
  return {partition.partitionId, partition.replicaId, partition.offsets,
          partition.sizes,       partition.strides,   {}};
}

bool verifyDistributedBoundaryCoverage(
    llvm::ArrayRef<DistributedBoundaryPartition> partitions,
    llvm::ArrayRef<int64_t> globalShape, llvm::StringRef distribution,
    llvm::raw_ostream &diagnostics) {
  if (distribution == "replicated") {
    for (const DistributedBoundaryPartition &partition : partitions) {
      if (!llvm::all_of(partition.offsets,
                        [](int64_t offset) { return offset == 0; }) ||
          !llvm::equal(partition.sizes, globalShape))
        return rejectProgramDirectory(
            "replicated distributed boundary partition must cover the full "
            "global tensor",
            diagnostics);
    }
    return false;
  }
  if (distribution != "partitioned")
    return rejectProgramDirectory(
        "distributed boundary distribution must be 'replicated' or "
        "'partitioned'",
        diagnostics);

  uint64_t globalElements = 1;
  for (int64_t dim : globalShape) {
    uint64_t next = 0;
    if (!checkedMulUint64(globalElements, static_cast<uint64_t>(dim), next))
      return rejectProgramDirectory(
          "distributed boundary global coverage size overflows uint64",
          diagnostics);
    globalElements = next;
  }

  uint64_t coveredElements = 0;
  for (const DistributedBoundaryPartition &partition : partitions) {
    uint64_t partitionElements = 1;
    for (int64_t size : partition.sizes) {
      uint64_t next = 0;
      if (!checkedMulUint64(partitionElements, static_cast<uint64_t>(size),
                            next))
        return rejectProgramDirectory(
            "distributed boundary partition coverage size overflows uint64",
            diagnostics);
      partitionElements = next;
    }
    if (partitionElements >
        std::numeric_limits<uint64_t>::max() - coveredElements)
      return rejectProgramDirectory(
          "distributed boundary coverage size overflows uint64", diagnostics);
    coveredElements += partitionElements;
  }

  for (size_t lhsIndex = 0; lhsIndex < partitions.size(); ++lhsIndex) {
    for (size_t rhsIndex = lhsIndex + 1; rhsIndex < partitions.size();
         ++rhsIndex) {
      const DistributedBoundaryPartition &lhs = partitions[lhsIndex];
      const DistributedBoundaryPartition &rhs = partitions[rhsIndex];
      bool overlaps = true;
      for (size_t dim = 0; dim < globalShape.size(); ++dim) {
        int64_t lhsEnd = lhs.offsets[dim] + lhs.sizes[dim];
        int64_t rhsEnd = rhs.offsets[dim] + rhs.sizes[dim];
        if (lhsEnd <= rhs.offsets[dim] || rhsEnd <= lhs.offsets[dim]) {
          overlaps = false;
          break;
        }
      }
      if (overlaps)
        return rejectProgramDirectory(
            "distributed boundary partition slices overlap", diagnostics);
    }
  }
  if (coveredElements != globalElements)
    return rejectProgramDirectory(
        "distributed boundary partition slices do not cover the global tensor",
        diagnostics);
  return false;
}

bool verifyDistributedBoundaryBinding(const DistributedBoundaryBinding &binding,
                                      RankedTensorType tensorType,
                                      const ProgramSignature &signature,
                                      int64_t numPartitions,
                                      llvm::raw_ostream &diagnostics) {
  if (!tensorType.hasStaticShape())
    return rejectProgramDirectory(
        "distributed boundary must refer to a static ranked tensor",
        diagnostics);
  if (!llvm::equal(binding.localShape, tensorType.getShape()) ||
      !llvm::equal(binding.localShape, signature.shape))
    return rejectProgramDirectory(
        "distributed boundary local_shape does not match module and metadata",
        diagnostics);
  if (binding.globalShape.size() != static_cast<size_t>(tensorType.getRank()))
    return rejectProgramDirectory(
        "distributed boundary global_shape rank does not match local tensor",
        diagnostics);

  std::string elementDtype = dtypeString(tensorType.getElementType());
  if (elementDtype.empty() ||
      normalizeProgramDtype(binding.dtype) != elementDtype ||
      normalizeProgramDtype(signature.dtype) != elementDtype)
    return rejectProgramDirectory(
        "distributed boundary dtype does not match module and metadata",
        diagnostics);
  if (binding.distribution != "replicated" &&
      binding.distribution != "partitioned")
    return rejectProgramDirectory(
        "distributed boundary distribution must be 'replicated' or "
        "'partitioned'",
        diagnostics);
  if (binding.partitions.size() != static_cast<size_t>(numPartitions))
    return rejectProgramDirectory(
        "distributed boundary partition count does not match num_partitions",
        diagnostics);

  std::vector<bool> seenPartitions(numPartitions, false);
  std::vector<bool> seenReplicaIds(numPartitions, false);
  std::vector<int64_t> maximumSizes(binding.globalShape.size(), 0);
  for (const DistributedBoundaryPartition &partition : binding.partitions) {
    if (partition.partitionId < 0 || partition.partitionId >= numPartitions)
      return rejectProgramDirectory(
          "distributed boundary partition_id is out of range", diagnostics);
    if (seenPartitions[partition.partitionId])
      return rejectProgramDirectory(
          "duplicate distributed boundary partition_id", diagnostics);
    seenPartitions[partition.partitionId] = true;

    if (binding.distribution == "partitioned") {
      if (partition.replicaId != 0)
        return rejectProgramDirectory(
            "partitioned distributed boundary replica_id must be 0",
            diagnostics);
    } else {
      if (partition.replicaId < 0 || partition.replicaId >= numPartitions)
        return rejectProgramDirectory(
            "replicated distributed boundary replica_id is out of range",
            diagnostics);
      if (seenReplicaIds[partition.replicaId])
        return rejectProgramDirectory(
            "duplicate replicated distributed boundary replica_id",
            diagnostics);
      seenReplicaIds[partition.replicaId] = true;
    }

    size_t tensorRank = binding.globalShape.size();
    if (partition.offsets.size() != tensorRank ||
        partition.sizes.size() != tensorRank ||
        partition.strides.size() != tensorRank)
      return rejectProgramDirectory(
          "distributed boundary slice rank does not match tensor rank",
          diagnostics);
    for (auto [dim, offset] : llvm::enumerate(partition.offsets)) {
      int64_t size = partition.sizes[dim];
      if (offset > binding.globalShape[dim] ||
          size > binding.globalShape[dim] - offset)
        return rejectProgramDirectory(
            "distributed boundary slice exceeds global_shape", diagnostics);
      if (size > binding.localShape[dim])
        return rejectProgramDirectory(
            "distributed boundary slice exceeds local_shape", diagnostics);
      if (partition.strides[dim] != 1)
        return rejectProgramDirectory("distributed boundary stride must be 1",
                                      diagnostics);
      maximumSizes[dim] = std::max(maximumSizes[dim], size);
    }
  }

  if (llvm::any_of(seenPartitions, [](bool seen) { return !seen; }))
    return rejectProgramDirectory(
        "distributed boundary partition domain is incomplete", diagnostics);
  if (binding.distribution == "replicated") {
    if (binding.globalShape != binding.localShape)
      return rejectProgramDirectory(
          "replicated distributed boundary global_shape and local_shape must "
          "match",
          diagnostics);
    if (llvm::any_of(seenReplicaIds, [](bool seen) { return !seen; }))
      return rejectProgramDirectory(
          "replicated distributed boundary replica_id domain is incomplete",
          diagnostics);
  } else if (maximumSizes != binding.localShape) {
    return rejectProgramDirectory(
        "partitioned distributed boundary slices do not explain local_shape",
        diagnostics);
  }
  return verifyDistributedBoundaryCoverage(binding.partitions,
                                           binding.globalShape,
                                           binding.distribution, diagnostics);
}

} // namespace

bool verifyDistributedBoundary(
    ModuleOp module, const ProgramMetadata &meta, func::FuncOp func,
    bool postSpmdMarker, llvm::raw_ostream &diagnostics,
    wafer::frontend::FrontendProgramVerificationResult *result) {
  if (!postSpmdMarker) {
    if (meta.distributedBoundary)
      return rejectProgramDirectory(
          "distributed_boundary requires a post-SPMD marker", diagnostics);
    return false;
  }
  if (!meta.distributedBoundary)
    return rejectProgramDirectory(
        "post-SPMD program directory is missing distributed_boundary",
        diagnostics);

  const DistributedBoundary &boundary = *meta.distributedBoundary;
  if (boundary.numPartitions <= 0)
    return rejectProgramDirectory(
        "distributed_boundary num_partitions must be positive", diagnostics);

  FailureOr<int64_t> meshPartitionCount =
      getSingleExecutionMeshPartitionCount(module, diagnostics);
  if (failed(meshPartitionCount))
    return true;
  if (*meshPartitionCount != boundary.numPartitions)
    return rejectProgramDirectory(
        "distributed_boundary num_partitions does not match execution mesh "
        "partition count",
        diagnostics);

  FunctionType functionType = func.getFunctionType();
  size_t expectedInputCount = llvm::count_if(
      meta.inputLocations, [](const ProgramInputLocation &location) {
        return location.type == "input_arg";
      });
  if (boundary.inputs.size() != expectedInputCount)
    return rejectProgramDirectory(
        "distributed_boundary inputs do not exactly cover input_arg "
        "locations",
        diagnostics);
  if (boundary.outputs.size() != functionType.getNumResults())
    return rejectProgramDirectory(
        "distributed_boundary outputs do not exactly cover function results",
        diagnostics);

  std::vector<bool> seenInputs(functionType.getNumInputs(), false);
  for (const DistributedBoundaryBinding &binding : boundary.inputs) {
    if (binding.index < 0 ||
        binding.index >= static_cast<int64_t>(functionType.getNumInputs()))
      return rejectProgramDirectory(
          "distributed boundary argument_index is out of range", diagnostics);
    if (meta.inputLocations[binding.index].type != "input_arg")
      return rejectProgramDirectory(
          "distributed boundary argument_index does not refer to input_arg",
          diagnostics);
    if (seenInputs[binding.index])
      return rejectProgramDirectory(
          "duplicate distributed boundary argument_index", diagnostics);
    seenInputs[binding.index] = true;
    auto tensorType =
        dyn_cast<RankedTensorType>(functionType.getInput(binding.index));
    if (!tensorType ||
        verifyDistributedBoundaryBinding(binding, tensorType,
                                         meta.inputSignatures[binding.index],
                                         boundary.numPartitions, diagnostics))
      return true;
  }
  for (auto [index, location] : llvm::enumerate(meta.inputLocations)) {
    if (location.type == "input_arg" && !seenInputs[index])
      return rejectProgramDirectory(
          "input_arg is missing distributed boundary metadata", diagnostics);
  }

  std::vector<bool> seenOutputs(functionType.getNumResults(), false);
  for (const DistributedBoundaryBinding &binding : boundary.outputs) {
    if (binding.index < 0 ||
        binding.index >= static_cast<int64_t>(functionType.getNumResults()))
      return rejectProgramDirectory(
          "distributed boundary result_index is out of range", diagnostics);
    if (seenOutputs[binding.index])
      return rejectProgramDirectory(
          "duplicate distributed boundary result_index", diagnostics);
    seenOutputs[binding.index] = true;
    auto tensorType =
        dyn_cast<RankedTensorType>(functionType.getResult(binding.index));
    if (!tensorType ||
        verifyDistributedBoundaryBinding(binding, tensorType,
                                         meta.outputSignatures[binding.index],
                                         boundary.numPartitions, diagnostics))
      return true;
  }
  if (llvm::any_of(seenOutputs, [](bool seen) { return !seen; }))
    return rejectProgramDirectory(
        "function result is missing distributed boundary metadata",
        diagnostics);
  if (result) {
    result->numPartitions = boundary.numPartitions;
    auto copyBindings = [&](llvm::ArrayRef<DistributedBoundaryBinding> source,
                            std::vector<wafer::frontend::ProgramBoundaryBinding>
                                &destination,
                            bool inputs) {
      destination.reserve(source.size());
      for (const DistributedBoundaryBinding &binding : source) {
        wafer::frontend::ProgramBoundaryBinding typed;
        typed.index = binding.index;
        typed.programIndex = inputs
                                 ? meta.inputLocations[binding.index].position
                                 : binding.index;
        typed.distribution = getVerifiedDistributionKind(binding.distribution);
        typed.globalShape = binding.globalShape;
        typed.localShape = binding.localShape;
        typed.dtype = llvm::cantFail(
            parseProgramElementType(normalizeProgramDtype(binding.dtype)));
        typed.partitionSlices.reserve(binding.partitions.size());
        for (const DistributedBoundaryPartition &partition : binding.partitions)
          typed.partitionSlices.push_back(getVerifiedPartitionSlice(partition));
        destination.push_back(std::move(typed));
      }
    };
    copyBindings(boundary.inputs, result->distributedInputs, true);
    copyBindings(boundary.outputs, result->distributedOutputs, false);
  }
  return false;
}

} // namespace wafer::frontend::program_detail
