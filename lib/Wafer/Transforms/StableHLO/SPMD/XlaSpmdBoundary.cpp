//===- XlaSpmdBoundary.cpp - Distributed boundary geometry ----------===//

#include "XlaSpmdPartitionerInternal.h"

#include "absl/strings/str_cat.h"
#include "tsl/platform/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <optional>
#include <utility>

namespace wafer::xla_spmd_helper {

std::vector<int64_t> shapeDims(const xla::Shape &shape) {
  return std::vector<int64_t>(shape.dimensions().begin(),
                              shape.dimensions().end());
}

std::vector<int64_t> zeros(size_t size) {
  return std::vector<int64_t>(size, 0);
}

std::vector<int64_t> ones(size_t size) { return std::vector<int64_t>(size, 1); }

static absl::StatusOr<DistributedBoundaryBinding>
makeDistributedBoundaryBinding(int64_t index, std::string dtype,
                               const xla::Shape &globalShape,
                               const xla::Shape &localShape,
                               const xla::HloSharding &sharding,
                               int64_t numPartitions) {
  if (!globalShape.IsArray() || !localShape.IsArray())
    return absl::InvalidArgumentError(
        "distributed function boundary must contain array shapes");
  if (sharding.IsTuple())
    return absl::InvalidArgumentError(
        "distributed tensor boundary has tuple sharding");
  if (sharding.IsManual() || sharding.IsManualSubgroup() ||
      sharding.IsUnknown() || sharding.IsShardGroup())
    return absl::InvalidArgumentError(
        "manual, unknown, and shard-group boundary shardings are unsupported");
  if (sharding.HasPartialReplication())
    return absl::InvalidArgumentError(
        "partial replication at the function boundary is unsupported");
  if (sharding.IsTileMaximal() && !sharding.IsReplicated())
    return absl::InvalidArgumentError(
        "single-device sharding at the function boundary is unsupported");
  TF_RETURN_IF_ERROR(sharding.Validate(globalShape, numPartitions));

  DistributedBoundaryBinding binding;
  binding.index = index;
  binding.dtype = std::move(dtype);
  binding.globalShape = shapeDims(globalShape);
  binding.localShape = shapeDims(localShape);
  bool replicated = sharding.IsReplicated();
  binding.distribution = replicated ? "replicated" : "partitioned";

  if (!replicated) {
    if (!sharding.IsTiled())
      return absl::InvalidArgumentError(
          "function boundary sharding is neither replicated nor tiled");
    if (sharding.TotalNumTiles() != numPartitions)
      return absl::InvalidArgumentError(absl::StrCat(
          "partitioned function boundary does not cover every partition: ",
          sharding.ToString()));
    for (int64_t partitionId = 0; partitionId < numPartitions;
         ++partitionId) {
      if (!sharding.UsesDevice(partitionId))
        return absl::InvalidArgumentError(absl::StrCat(
            "partitioned function boundary partition domain is not exact: ",
            sharding.ToString()));
    }
  }

  std::vector<int64_t> maxSizes(globalShape.rank(), 0);
  binding.partitions.reserve(numPartitions);
  for (int64_t partitionId = 0; partitionId < numPartitions; ++partitionId) {
    DistributedPartition partition;
    partition.partitionId = partitionId;
    partition.replicaId = replicated ? partitionId : 0;
    partition.strides = ones(globalShape.rank());
    if (replicated) {
      partition.offsets = zeros(globalShape.rank());
      partition.sizes = binding.globalShape;
    } else {
      partition.offsets =
          sharding.TileOffsetForDevice(globalShape, partitionId);
      std::vector<int64_t> limits =
          sharding.TileLimitForDevice(globalShape, partitionId);
      for (auto [offset, limit] : llvm::zip(partition.offsets, limits))
        partition.sizes.push_back(limit - offset);
    }
    if (partition.sizes.size() != binding.localShape.size())
      return absl::InvalidArgumentError(
          "distributed slice rank differs from local tensor rank");
    for (auto [dim, size] : llvm::enumerate(partition.sizes)) {
      if (size < 0 || size > binding.localShape[dim])
        return absl::InvalidArgumentError(
            "distributed slice does not fit the local tensor shape");
      maxSizes[dim] = std::max(maxSizes[dim], size);
    }
    binding.partitions.push_back(std::move(partition));
  }

  if (replicated) {
    if (binding.globalShape != binding.localShape)
      return absl::InvalidArgumentError(
          "replicated boundary tensor changed shape during SPMD partitioning");
  } else if (maxSizes != binding.localShape) {
    return absl::InvalidArgumentError(
        "partitioned boundary slices do not explain the local tensor shape");
  }
  return binding;
}

static absl::StatusOr<const xla::Shape *>
resultShape(const xla::Shape &rootShape, size_t resultCount, size_t index) {
  if (resultCount == 1 && !rootShape.IsTuple())
    return &rootShape;
  if (!rootShape.IsTuple() ||
      rootShape.tuple_shapes_size() != static_cast<int64_t>(resultCount) ||
      index >= resultCount || rootShape.tuple_shapes(index).IsTuple())
    return absl::InvalidArgumentError(
        "HLO result shape does not match the flat program boundary");
  return &rootShape.tuple_shapes(index);
}

static absl::StatusOr<xla::HloSharding>
resultSharding(const xla::HloInstruction &root, size_t resultCount,
               size_t index) {
  if (!root.has_sharding())
    return xla::HloSharding::Replicate();
  const xla::HloSharding &sharding = root.sharding();
  if (resultCount == 1 && !root.shape().IsTuple()) {
    if (sharding.IsTuple())
      return absl::InvalidArgumentError(
          "non-tuple HLO result has tuple sharding");
    return sharding;
  }
  if (!root.shape().IsTuple())
    return absl::InvalidArgumentError(
        "multiple program results require a tuple HLO root");
  if (sharding.IsTuple())
    return sharding.GetSubSharding(root.shape(), {static_cast<int64_t>(index)});
  if (sharding.IsReplicated())
    return xla::HloSharding::Replicate();
  return absl::InvalidArgumentError(
      "tuple HLO result has a non-tuple non-replicated sharding");
}

absl::StatusOr<DistributedBoundary> buildDistributedBoundary(
    const ProgramMetadata &meta, const xla::HloModule &distributedModule,
    const xla::HloModule &partitionedModule, int64_t numPartitions) {
  const xla::HloComputation *distributedEntry =
      distributedModule.entry_computation();
  const xla::HloComputation *partitionedEntry =
      partitionedModule.entry_computation();
  if (!distributedEntry || !partitionedEntry)
    return absl::InvalidArgumentError(
        "HLO module is missing entry computation");
  if (distributedEntry->num_parameters() !=
          static_cast<int64_t>(meta.inputLocations.size()) ||
      partitionedEntry->num_parameters() !=
          static_cast<int64_t>(meta.inputLocations.size()))
    return absl::InvalidArgumentError(
        "HLO parameters do not match program directory metadata");

  DistributedBoundary boundary;
  boundary.numPartitions = numPartitions;
  for (size_t index = 0; index < meta.inputLocations.size(); ++index) {
    if (meta.inputLocations[index].type != "input_arg")
      continue;
    const xla::HloInstruction *globalParameter =
        distributedEntry->parameter_instruction(index);
    const xla::HloInstruction *localParameter =
        partitionedEntry->parameter_instruction(index);
    if (!globalParameter || !localParameter)
      return absl::InvalidArgumentError(
          "HLO input_arg parameter instruction is missing");
    xla::HloSharding sharding = globalParameter->has_sharding()
                                    ? globalParameter->sharding()
                                    : xla::HloSharding::Replicate();
    TF_ASSIGN_OR_RETURN(DistributedBoundaryBinding binding,
                        makeDistributedBoundaryBinding(
                            static_cast<int64_t>(index),
                            meta.inputSignatures[index].dtype,
                            globalParameter->shape(), localParameter->shape(),
                            sharding, numPartitions));
    boundary.inputs.push_back(std::move(binding));
  }

  const xla::HloInstruction *globalRoot = distributedEntry->root_instruction();
  const xla::HloInstruction *localRoot = partitionedEntry->root_instruction();
  if (!globalRoot || !localRoot)
    return absl::InvalidArgumentError("HLO entry root is missing");
  for (size_t index = 0; index < meta.outputSignatures.size(); ++index) {
    TF_ASSIGN_OR_RETURN(
        const xla::Shape *globalShape,
        resultShape(globalRoot->shape(), meta.outputSignatures.size(), index));
    TF_ASSIGN_OR_RETURN(
        const xla::Shape *localShape,
        resultShape(localRoot->shape(), meta.outputSignatures.size(), index));
    TF_ASSIGN_OR_RETURN(
        xla::HloSharding sharding,
        resultSharding(*globalRoot, meta.outputSignatures.size(), index));
    TF_ASSIGN_OR_RETURN(DistributedBoundaryBinding binding,
                        makeDistributedBoundaryBinding(
                            static_cast<int64_t>(index),
                            meta.outputSignatures[index].dtype, *globalShape,
                            *localShape, sharding, numPartitions));
    boundary.outputs.push_back(std::move(binding));
  }
  return boundary;
}

} // namespace wafer::xla_spmd_helper
