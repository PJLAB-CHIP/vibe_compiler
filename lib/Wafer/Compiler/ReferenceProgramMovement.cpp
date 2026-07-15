//===- ReferenceProgramMovement.cpp - Reference movement semantics ---===//

#include "ReferenceProgramInterpreterInternal.h"

#include "llvm/ADT/STLExtras.h"

#include <cstring>
#include <vector>

namespace wafer::compiler::reference_detail {

llvm::Error
ProgramInterpreter::copyChunks(const BufferView &source, const BufferView &dest,
                               llvm::ArrayRef<int64_t> sourceOffsets,
                               llvm::ArrayRef<int64_t> destOffsets,
                               int64_t innerBytes) {
  if (sourceOffsets.size() != destOffsets.size() || innerBytes < 0)
    return invalid("movement descriptor has inconsistent chunk counts");
  std::vector<uint8_t> payload(sourceOffsets.size() * innerBytes);
  for (auto [position, offset] : llvm::enumerate(sourceOffsets)) {
    if (offset < 0 || offset > source.physicalBytes - innerBytes)
      return invalid("movement source descriptor exceeds buffer extent");
    int64_t absolute = source.base + source.viewOffset + offset;
    if (absolute < 0 ||
        absolute >
            static_cast<int64_t>(source.storage->bytes.size()) - innerBytes)
      return invalid("movement source exceeds storage arena");
    std::memcpy(payload.data() + position * innerBytes,
                source.storage->bytes.data() + absolute, innerBytes);
  }
  for (auto [position, offset] : llvm::enumerate(destOffsets)) {
    if (offset < 0 || offset > dest.physicalBytes - innerBytes)
      return invalid("movement destination descriptor exceeds buffer extent");
    int64_t absolute = dest.base + dest.viewOffset + offset;
    if (absolute < 0 ||
        absolute >
            static_cast<int64_t>(dest.storage->bytes.size()) - innerBytes)
      return invalid("movement destination exceeds storage arena");
    std::memcpy(dest.storage->bytes.data() + absolute,
                payload.data() + position * innerBytes, innerBytes);
  }
  return llvm::Error::success();
}

llvm::Expected<std::vector<int64_t>>
ProgramInterpreter::descriptorOffsets(llvm::ArrayRef<int64_t> strides,
                                      llvm::ArrayRef<int64_t> iterations,
                                      int64_t base) {
  if (strides.size() != 3 || iterations.size() != 3 || base < 0)
    return invalid("movement descriptor must have three non-negative levels");
  std::vector<int64_t> offsets;
  for (int64_t i0 = 0; i0 < iterations[0]; ++i0)
    for (int64_t i1 = 0; i1 < iterations[1]; ++i1)
      for (int64_t i2 = 0; i2 < iterations[2]; ++i2) {
        if (strides[0] < 0 || strides[1] < 0 || strides[2] < 0)
          return invalid("movement descriptor contains a negative field");
        offsets.push_back(base + i0 * strides[0] + i1 * strides[1] +
                          i2 * strides[2]);
      }
  return offsets;
}

llvm::Error ProgramInterpreter::executeRDMA(const Command &command) {
  auto source = lookup(command.source);
  auto dest = lookup(command.dest);
  if (!source)
    return source.takeError();
  if (!dest)
    return dest.takeError();
  auto sourceOffsets =
      descriptorOffsets(command.sourceStrides, command.sourceIterations);
  if (!sourceOffsets)
    return sourceOffsets.takeError();
  std::vector<int64_t> destOffsets(sourceOffsets->size());
  for (auto [index, offset] : llvm::enumerate(destOffsets))
    offset = index * command.innerBytes;
  return copyChunks(*source, *dest, *sourceOffsets, destOffsets,
                    command.innerBytes);
}

llvm::Error ProgramInterpreter::executeWDMA(const Command &command) {
  auto source = lookup(command.source);
  auto dest = lookup(command.dest);
  if (!source)
    return source.takeError();
  if (!dest)
    return dest.takeError();
  auto destOffsets =
      descriptorOffsets(command.destStrides, command.destIterations);
  if (!destOffsets)
    return destOffsets.takeError();
  std::vector<int64_t> sourceOffsets(destOffsets->size());
  for (auto [index, offset] : llvm::enumerate(sourceOffsets))
    offset = index * command.innerBytes;
  return copyChunks(*source, *dest, sourceOffsets, *destOffsets,
                    command.innerBytes);
}

llvm::Error ProgramInterpreter::executeGatherScatter(const Command &command) {
  auto source = lookup(command.source);
  auto dest = lookup(command.dest);
  if (!source)
    return source.takeError();
  if (!dest)
    return dest.takeError();
  auto sourceOffsets =
      descriptorOffsets(command.sourceStrides, command.sourceIterations,
                        command.sourceOffset.value_or(0));
  auto destOffsets =
      descriptorOffsets(command.destStrides, command.destIterations,
                        command.destOffset.value_or(0));
  if (!sourceOffsets)
    return sourceOffsets.takeError();
  if (!destOffsets)
    return destOffsets.takeError();
  if (sourceOffsets->size() != destOffsets->size())
    return invalid("movement source/destination descriptors disagree");
  return copyChunks(*source, *dest, *sourceOffsets, *destOffsets,
                    command.innerBytes);
}

} // namespace wafer::compiler::reference_detail
