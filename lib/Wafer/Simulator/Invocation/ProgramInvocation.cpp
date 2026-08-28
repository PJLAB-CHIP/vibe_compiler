//===- ProgramInvocation.cpp - Typed source invocation materialization --===//

#include "Wafer/Simulator/Invocation/ProgramInvocation.h"

#include "Wafer/Driver/ProgramData.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Path.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace wafer::compiler {
namespace {

llvm::Error invalid(llvm::StringRef message) {
  return llvm::createStringError(llvm::errc::invalid_argument, "%s",
                                 message.str().c_str());
}

std::optional<int64_t> elementCount(llvm::ArrayRef<int64_t> shape) {
  int64_t count = 1;
  for (int64_t dim : shape) {
    if (dim < 0 ||
        (dim != 0 && count > std::numeric_limits<int64_t>::max() / dim))
      return std::nullopt;
    count *= dim;
  }
  return count;
}

struct ProgramTensorDTypeInfo {
  int64_t elementBytes = 0;
  bool floating = false;
};

std::optional<ProgramTensorDTypeInfo>
getProgramTensorDTypeInfo(ProgramElementType dtype) {
  std::optional<int64_t> elementBytes = getProgramDTypeElementBytes(dtype);
  if (!elementBytes)
    return std::nullopt;
  return ProgramTensorDTypeInfo{*elementBytes,
                                isFloatingProgramElementType(dtype)};
}

llvm::Expected<ProgramTensor>
sliceProgramTensor(const ProgramTensor &global,
                   const ProgramResourceBinding &binding) {
  if (global.getDType() != binding.dtype ||
      global.getShape() != llvm::ArrayRef<int64_t>(binding.globalShape))
    return invalid("global program input disagrees with typed binding");
  const frontend::ProgramPartitionSlice &slice = binding.slice;
  const size_t rank = binding.globalShape.size();
  if (slice.offsets.size() != rank || slice.sizes.size() != rank ||
      slice.strides.size() != rank || slice.sizes != binding.localShape)
    return invalid("program input slice has inconsistent rank or shape");

  auto globalElements = elementCount(binding.globalShape);
  auto localElements = elementCount(binding.localShape);
  if (!globalElements || !localElements ||
      (*globalElements == 0 && !global.getBytes().empty()) ||
      (*globalElements != 0 &&
       global.getBytes().size() % static_cast<size_t>(*globalElements) != 0))
    return invalid("global program input byte geometry is invalid");
  const size_t elementBytes =
      *globalElements == 0
          ? 0
          : global.getBytes().size() / static_cast<size_t>(*globalElements);
  if (elementBytes != 0 &&
      static_cast<uint64_t>(*localElements) >
          std::numeric_limits<size_t>::max() / elementBytes)
    return invalid("local program input byte geometry is not representable");
  std::vector<uint8_t> local(static_cast<size_t>(*localElements) *
                             elementBytes);
  if (*localElements == 0)
    return ProgramTensor::create(binding.dtype, binding.localShape, local);

  std::vector<int64_t> coordinate(rank, 0);
  for (int64_t localLinear = 0; localLinear < *localElements; ++localLinear) {
    int64_t remaining = localLinear;
    for (size_t reverse = rank; reverse > 0; --reverse) {
      size_t dim = reverse - 1;
      coordinate[dim] = remaining % binding.localShape[dim];
      remaining /= binding.localShape[dim];
    }
    int64_t globalLinear = 0;
    for (size_t dim = 0; dim < rank; ++dim) {
      if (slice.offsets[dim] < 0 || slice.strides[dim] <= 0 ||
          coordinate[dim] >
              (std::numeric_limits<int64_t>::max() - slice.offsets[dim]) /
                  slice.strides[dim])
        return invalid("program input slice coordinate overflows");
      int64_t globalCoordinate =
          slice.offsets[dim] + coordinate[dim] * slice.strides[dim];
      if (globalCoordinate < 0 || globalCoordinate >= binding.globalShape[dim])
        return invalid("program input slice is outside global tensor");
      globalLinear = globalLinear * binding.globalShape[dim] + globalCoordinate;
    }
    std::memcpy(local.data() + static_cast<size_t>(localLinear) * elementBytes,
                global.getBytes().data() +
                    static_cast<size_t>(globalLinear) * elementBytes,
                elementBytes);
  }
  return ProgramTensor::create(binding.dtype, binding.localShape, local);
}

} // namespace

std::optional<int64_t>
computeProgramTensorByteCount(ProgramElementType dtype,
                              llvm::ArrayRef<int64_t> shape) {
  std::optional<ProgramTensorDTypeInfo> info = getProgramTensorDTypeInfo(dtype);
  if (!info)
    return std::nullopt;
  int64_t elementBytes = info->elementBytes;
  int64_t bytes = elementBytes;
  for (int64_t dim : shape) {
    if (dim < 0 ||
        (dim != 0 && bytes > std::numeric_limits<int64_t>::max() / dim))
      return std::nullopt;
    bytes *= dim;
  }
  return bytes;
}

bool isFloatingProgramTensorDType(ProgramElementType dtype) {
  std::optional<ProgramTensorDTypeInfo> info = getProgramTensorDTypeInfo(dtype);
  return info && info->floating;
}

llvm::Expected<ProgramTensor>
ProgramTensor::create(ProgramElementType dtype, llvm::ArrayRef<int64_t> shape,
                      llvm::ArrayRef<uint8_t> bytes) {
  std::optional<int64_t> expected = computeProgramTensorByteCount(dtype, shape);
  if (!expected)
    return invalid("program tensor has unsupported dtype or shape");
  if (*expected != static_cast<int64_t>(bytes.size()))
    return invalid("program tensor byte count disagrees with dtype and shape");
  return ProgramTensor(dtype, std::vector<int64_t>(shape),
                       std::vector<uint8_t>(bytes));
}

llvm::Expected<ProgramTensor>
ProgramTensor::share(ProgramElementType dtype, llvm::ArrayRef<int64_t> shape,
                     std::shared_ptr<const std::vector<uint8_t>> storage,
                     size_t offset) {
  std::optional<int64_t> expected = computeProgramTensorByteCount(dtype, shape);
  if (!expected)
    return invalid("program tensor has unsupported dtype or shape");
  if (!storage || *expected < 0 || offset > storage->size() ||
      static_cast<size_t>(*expected) > storage->size() - offset)
    return invalid("program tensor view is outside its shared storage");
  ProgramTensor tensor(dtype, std::vector<int64_t>(shape), {});
  tensor.sharedStorage = std::move(storage);
  tensor.sharedOffset = offset;
  tensor.sharedSize = static_cast<size_t>(*expected);
  return tensor;
}

llvm::ArrayRef<uint8_t> ProgramTensor::getBytes() const {
  if (!sharedStorage)
    return bytes;
  return llvm::ArrayRef<uint8_t>(sharedStorage->data() + sharedOffset,
                                 sharedSize);
}

llvm::Expected<ProgramTensor> ProgramTensor::loadNpy(llvm::StringRef path) {
  auto payload = frontend::loadNpyTensorPayload(path);
  if (!payload)
    return payload.takeError();
  return create(payload->dtype, payload->shape, payload->bytes);
}

llvm::Expected<std::vector<ProgramTileInvocation>> prepareProgramInvocations(
    const DeviceExecutable &deviceExecutable,
    llvm::ArrayRef<ProgramGlobalInputBinding> globalInputs) {
  if (deviceExecutable.getTileExecutables().empty())
    return invalid("program invocation executable domain must not be empty");
  const ProgramDataHandoff &handoff = deviceExecutable.getProgramDataHandoff();

  // One shared materialization per owned data range: every Tile binding for
  // the same program tensor references the same storage, so parameter and
  // constant payload bytes are read exactly once per range, never per Tile.
  std::map<ProgramTensorId, std::shared_ptr<const std::vector<uint8_t>>>
      materializedRanges;
  std::vector<ProgramTileInvocation> invocations;
  invocations.reserve(deviceExecutable.getTileExecutables().size());
  for (const TileExecutable &tile : deviceExecutable.getTileExecutables()) {
    if (tile.getCardId() != CardId(0))
      return invalid(
          "program invocation supports only the current single-card domain");
    ProgramTileInvocation invocation{
        tile.getCardId(), tile.getTileId(), tile.getLaunchSlotId(), {}};
    for (const ProgramResourceBinding &binding : tile.getProgramBindings()) {
      if (binding.role == ProgramResourceRole::Output)
        continue;
      llvm::Expected<ProgramTensor> tensor =
          [&]() -> llvm::Expected<ProgramTensor> {
        if (binding.role == ProgramResourceRole::UserInput) {
          const ProgramGlobalInputBinding *match = nullptr;
          for (const ProgramGlobalInputBinding &input : globalInputs)
            if (input.index == binding.programIndex) {
              if (match)
                return invalid("duplicate global program input index");
              match = &input;
            }
          if (!match)
            return invalid("missing global program input index");
          return sliceProgramTensor(match->tensor, binding);
        }

        const ProgramDataRange *range =
            handoff.findRange(binding.programTensorId);
        if (!range)
          return invalid("program tensor has no owned data range");
        if (range->getDType() != binding.dtype ||
            range->getLocalShape() !=
                llvm::ArrayRef<int64_t>(binding.localShape))
          return invalid("program data range disagrees with typed Tile "
                         "binding");
        auto existing = materializedRanges.find(binding.programTensorId);
        if (existing != materializedRanges.end())
          return ProgramTensor::share(binding.dtype, binding.localShape,
                                      existing->second, 0);
        if (range->getRegionLength() >
            static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
          return invalid("program data range is not host-representable");
        auto storage = std::make_shared<std::vector<uint8_t>>(
            static_cast<size_t>(range->getRegionLength()));
        if (llvm::Error error = handoff.materializeRange(*range, *storage))
          return invalid(llvm::toString(std::move(error)));
        materializedRanges.emplace(binding.programTensorId, storage);
        return ProgramTensor::share(binding.dtype, binding.localShape,
                                    std::move(storage), 0);
      }();
      if (!tensor)
        return tensor.takeError();
      invocation.inputs.push_back(
          {binding.role, binding.index, std::move(*tensor)});
    }
    invocations.push_back(std::move(invocation));
  }
  const auto &firstBindings =
      deviceExecutable.getTileExecutables().front().getProgramBindings();
  const size_t expectedGlobalInputs = static_cast<size_t>(
      llvm::count_if(firstBindings, [](const ProgramResourceBinding &binding) {
        return binding.role == ProgramResourceRole::UserInput;
      }));
  if (globalInputs.size() != expectedGlobalInputs)
    return invalid("unexpected global program input index");
  return invocations;
}

llvm::Expected<ProgramTensor>
sliceProgramTensorForBinding(const ProgramTensor &global,
                             const ProgramResourceBinding &binding) {
  return sliceProgramTensor(global, binding);
}

} // namespace wafer::compiler
