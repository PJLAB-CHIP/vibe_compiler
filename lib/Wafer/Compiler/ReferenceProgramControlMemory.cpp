//===- ReferenceProgramControlMemory.cpp - Control and memory state --===//

#include "ReferenceProgramInterpreterInternal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace wafer::compiler::reference_detail {

llvm::Error forEachLogicalIndex(
    llvm::ArrayRef<int64_t> shape,
    llvm::function_ref<llvm::Error(llvm::ArrayRef<int64_t>)> callback) {
  for (int64_t dim : shape)
    if (dim < 0)
      return invalid("reference executor requires static tensor shapes");
  if (llvm::is_contained(shape, int64_t{0}))
    return llvm::Error::success();
  llvm::SmallVector<int64_t> index(shape.size(), 0);
  if (shape.empty())
    return callback(index);
  while (true) {
    if (llvm::Error error = callback(index))
      return error;
    int64_t dim = static_cast<int64_t>(shape.size()) - 1;
    for (; dim >= 0; --dim) {
      if (++index[dim] < shape[dim])
        break;
      index[dim] = 0;
    }
    if (dim < 0)
      break;
  }
  return llvm::Error::success();
}

llvm::Error ProgramInterpreter::bindEntryArguments(
    llvm::ArrayRef<ReferenceInputBinding> inputs) {
  const auto &entry = program.functions[program.entryFunction];
  llvm::SmallVector<bool> used(inputs.size(), false);
  for (const RankProgramBinding &binding : program.programBindings) {
    if (binding.role == ProgramResourceRole::Output)
      continue;
    int64_t match = -1;
    for (auto [position, input] : llvm::enumerate(inputs)) {
      if (input.role == binding.role && input.index == binding.index) {
        if (match >= 0)
          return invalid("duplicate reference input binding");
        match = static_cast<int64_t>(position);
      }
    }
    if (match < 0)
      return invalid("missing reference input binding");
    used[match] = true;
    if (inputs[match].tensor.getDType() != binding.dtype ||
        inputs[match].tensor.getShape() !=
            llvm::ArrayRef<int64_t>(binding.localShape))
      return invalid(
          "reference input tensor disagrees with typed rank binding");
    auto imported =
        importTensor(inputs[match].tensor, entry.argumentTypes[binding.index]);
    if (!imported)
      return imported.takeError();
    buffers[entry.arguments[binding.index]] = std::move(*imported);
  }
  if (llvm::is_contained(used, false))
    return invalid("unexpected reference input binding");
  return llvm::Error::success();
}

llvm::Expected<BufferView>
ProgramInterpreter::importTensor(const ReferenceTensor &tensor,
                                 mlir::MemRefType type) {
  auto info = wafer::computeWaferPhysicalTensorInfo(type);
  if (!info || info->physicalBytes < 0 || info->elementBytes <= 0)
    return invalid("input has no static accepted physical layout");
  if (elementDType(type.getElementType()) != tensor.getDType() ||
      type.getShape() != tensor.getShape())
    return invalid("input tensor type disagrees with entry memref");
  BufferView view{std::make_shared<Storage>(), 0, 0, info->physicalBytes, type};
  view.storage->bytes.resize(info->physicalBytes, 0);
  int64_t linearByte = 0;
  if (llvm::Error error = forEachLogicalIndex(
          type.getShape(), [&](llvm::ArrayRef<int64_t> index) -> llvm::Error {
            auto offset = getAbsoluteOffset(view, index, info->elementBytes);
            if (!offset)
              return offset.takeError();
            std::memcpy(view.storage->bytes.data() + *offset,
                        tensor.getBytes().data() + linearByte,
                        info->elementBytes);
            linearByte += info->elementBytes;
            return llvm::Error::success();
          }))
    return std::move(error);
  return view;
}

llvm::Expected<ReferenceTensor>
ProgramInterpreter::exportTensor(const BufferView &view, llvm::StringRef dtype,
                                 llvm::ArrayRef<int64_t> shape) {
  auto bytes = getCompactByteCount(dtype, shape);
  if (!bytes)
    return invalid("output tensor has unsupported dtype or shape");
  if (elementDType(view.type.getElementType()) != dtype ||
      view.type.getShape() != shape)
    return invalid("entry result disagrees with typed output binding");
  std::vector<uint8_t> compact(*bytes);
  int64_t linearByte = 0;
  auto info = wafer::computeWaferPhysicalTensorInfo(view.type);
  if (!info || info->elementBytes <= 0)
    return invalid("output has no static accepted physical layout");
  if (llvm::Error error = forEachLogicalIndex(
          shape, [&](llvm::ArrayRef<int64_t> index) -> llvm::Error {
            auto offset = getAbsoluteOffset(view, index, info->elementBytes);
            if (!offset)
              return offset.takeError();
            std::memcpy(compact.data() + linearByte,
                        view.storage->bytes.data() + *offset,
                        info->elementBytes);
            linearByte += info->elementBytes;
            return llvm::Error::success();
          }))
    return std::move(error);
  return ReferenceTensor::create(dtype, shape, compact);
}

llvm::Expected<RuntimeValue>
ProgramInterpreter::readValue(ValueId value) const {
  auto buffer = buffers.find(value);
  auto scalar = scalars.find(value);
  if (buffer != buffers.end() && scalar != scalars.end())
    return invalid("projected value is both buffer and scalar");
  if (buffer != buffers.end())
    return RuntimeValue{buffer->second, std::nullopt};
  if (scalar != scalars.end())
    return RuntimeValue{std::nullopt, scalar->second};
  return invalid("projected runtime value is unavailable");
}

llvm::Expected<std::vector<RuntimeValue>>
ProgramInterpreter::readValues(llvm::ArrayRef<ValueId> values) const {
  std::vector<RuntimeValue> result;
  result.reserve(values.size());
  for (ValueId value : values) {
    auto runtimeValue = readValue(value);
    if (!runtimeValue)
      return runtimeValue.takeError();
    result.push_back(std::move(*runtimeValue));
  }
  return result;
}

llvm::Error
ProgramInterpreter::writeValues(llvm::ArrayRef<ValueId> destinations,
                                llvm::ArrayRef<RuntimeValue> values) {
  if (destinations.size() != values.size())
    return invalid("projected runtime value arity mismatch");
  for (auto [destination, value] : llvm::zip_equal(destinations, values)) {
    buffers.erase(destination);
    scalars.erase(destination);
    if (value.buffer)
      buffers[destination] = *value.buffer;
    else if (value.scalar)
      scalars[destination] = *value.scalar;
    else
      return invalid("projected runtime value has no representation");
  }
  return llvm::Error::success();
}

llvm::Error
ProgramInterpreter::assignValues(llvm::ArrayRef<ValueId> destinations,
                                 llvm::ArrayRef<ValueId> sources) {
  auto values = readValues(sources);
  if (!values)
    return values.takeError();
  return writeValues(destinations, *values);
}

llvm::Expected<bool> ProgramInterpreter::lookupBoolean(ValueId value) const {
  auto found = scalars.find(value);
  if (found == scalars.end() || !found->second.integer ||
      found->second.integer->getBitWidth() != 1)
    return invalid("projected branch condition is not an available i1");
  return !found->second.integer->isZero();
}

llvm::Expected<int64_t>
ProgramInterpreter::lookupSignedInteger(ValueId value) const {
  auto found = scalars.find(value);
  if (found == scalars.end() || !found->second.integer ||
      found->second.integer->getBitWidth() > 64)
    return invalid("projected loop bound is not an available integer");
  return found->second.integer->getSExtValue();
}

llvm::Expected<BufferView>
ProgramInterpreter::allocate(const Command &command) {
  if (command.acceptedOffset < 0 || command.physicalBytes < 0 ||
      command.physicalBytes >
          std::numeric_limits<int64_t>::max() - command.acceptedOffset)
    return invalid("accepted allocation range overflows");
  std::shared_ptr<Storage> arena = command.spmAllocation ? spmArena : ddrArena;
  int64_t end = command.acceptedOffset + command.physicalBytes;
  if (end > static_cast<int64_t>(arena->bytes.size()))
    arena->bytes.resize(end, 0);
  return BufferView{std::move(arena), command.acceptedOffset, 0,
                    command.physicalBytes, command.type};
}

llvm::Expected<BufferView> ProgramInterpreter::lookup(ValueId value) const {
  auto found = buffers.find(value);
  if (found == buffers.end())
    return invalid("instruction operand buffer is unavailable");
  return found->second;
}

} // namespace wafer::compiler::reference_detail
