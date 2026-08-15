//===- ProgramInvocation.cpp - Typed source invocation materialization --===//

#include "Wafer/Compiler/ProgramInvocation.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Path.h"

#include <cstdint>
#include <cstring>
#include <limits>
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
getProgramTensorDTypeInfo(llvm::StringRef dtype) {
  if (dtype == "i8" || dtype == "ui8")
    return ProgramTensorDTypeInfo{1, false};
  if (dtype == "i16" || dtype == "ui16")
    return ProgramTensorDTypeInfo{2, false};
  if (dtype == "f16" || dtype == "bf16")
    return ProgramTensorDTypeInfo{2, true};
  if (dtype == "i32" || dtype == "ui32")
    return ProgramTensorDTypeInfo{4, false};
  if (dtype == "f32" || dtype == "tf32")
    return ProgramTensorDTypeInfo{4, true};
  if (dtype == "i64" || dtype == "ui64")
    return ProgramTensorDTypeInfo{8, false};
  if (dtype == "f64")
    return ProgramTensorDTypeInfo{8, true};
  return std::nullopt;
}

bool isSafeRelativePath(llvm::StringRef path) {
  if (path.empty() || llvm::sys::path::is_absolute(path) || path.contains('\\'))
    return false;
  for (llvm::sys::path::const_iterator current = llvm::sys::path::begin(path),
                                       end = llvm::sys::path::end(path);
       current != end; ++current)
    if (*current == "." || *current == ".." || current->empty())
      return false;
  return true;
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
computeProgramTensorByteCount(llvm::StringRef dtype,
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

bool isFloatingProgramTensorDType(llvm::StringRef dtype) {
  std::optional<ProgramTensorDTypeInfo> info = getProgramTensorDTypeInfo(dtype);
  return info && info->floating;
}

llvm::Expected<ProgramTensor>
ProgramTensor::create(llvm::StringRef dtype, llvm::ArrayRef<int64_t> shape,
                      llvm::ArrayRef<uint8_t> bytes) {
  std::optional<int64_t> expected = computeProgramTensorByteCount(dtype, shape);
  if (!expected)
    return invalid("program tensor has unsupported dtype or shape");
  if (*expected != static_cast<int64_t>(bytes.size()))
    return invalid("program tensor byte count disagrees with dtype and shape");
  return ProgramTensor(dtype.str(), std::vector<int64_t>(shape),
                       std::vector<uint8_t>(bytes));
}

llvm::Expected<ProgramTensor> ProgramTensor::loadNpy(llvm::StringRef path) {
  auto payload = frontend::loadNpyTensorPayload(path);
  if (!payload)
    return payload.takeError();
  return create(payload->dtype, payload->shape, payload->bytes);
}

llvm::Expected<std::vector<ProgramTileInvocation>> prepareProgramInvocations(
    const PhysicalTileExecutables &physicalTileExecutables,
    llvm::StringRef packageRoot,
    llvm::ArrayRef<ProgramGlobalInputBinding> globalInputs) {
  if (packageRoot.empty())
    return invalid("program invocation package root must not be empty");
  if (physicalTileExecutables.getPhysicalTileExecutables().empty())
    return invalid("program invocation executable domain must not be empty");
  std::vector<ProgramTileInvocation> invocations;
  invocations.reserve(
      physicalTileExecutables.getPhysicalTileExecutables().size());
  for (const PhysicalTileExecutable &tile :
       physicalTileExecutables.getPhysicalTileExecutables()) {
    if (tile.getPhysicalCardId() != PhysicalCardId(0))
      return invalid(
          "program invocation supports only the current single-card domain");
    ProgramTileInvocation invocation{tile.getPhysicalCardId(),
                                     tile.getPhysicalTileId(),
                                     tile.getLaunchSlotId(),
                                     {}};
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
        if (!isSafeRelativePath(binding.slice.payloadPath))
          return invalid("program payload path is not safe and relative");
        llvm::SmallString<256> payload(packageRoot);
        llvm::sys::path::append(payload, binding.slice.payloadPath);
        auto loaded = ProgramTensor::loadNpy(payload);
        if (!loaded)
          return loaded.takeError();
        if (loaded->getDType() != binding.dtype ||
            loaded->getShape() != llvm::ArrayRef<int64_t>(binding.localShape))
          return invalid("program payload disagrees with typed Tile binding");
        return loaded;
      }();
      if (!tensor)
        return tensor.takeError();
      invocation.inputs.push_back(
          {binding.role, binding.index, std::move(*tensor)});
    }
    invocations.push_back(std::move(invocation));
  }
  const auto &firstBindings =
      physicalTileExecutables.getPhysicalTileExecutables()
          .front()
          .getProgramBindings();
  if (globalInputs.size() !=
      llvm::count_if(firstBindings, [](const ProgramResourceBinding &binding) {
        return binding.role == ProgramResourceRole::UserInput;
      }))
    return invalid("unexpected global program input index");
  return invocations;
}

llvm::Expected<ProgramTensor>
sliceProgramTensorForBinding(const ProgramTensor &global,
                             const ProgramResourceBinding &binding) {
  return sliceProgramTensor(global, binding);
}

} // namespace wafer::compiler
