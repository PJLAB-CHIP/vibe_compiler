//===- TargetModelInvocation.cpp - Typed source/model binding -----------===//

#include "Wafer/Simulator/Invocation/TargetModelInvocation.h"

#include "Wafer/Target/PhysicalTensor/PhysicalTensorCodec.h"
#include "Wafer/Target/PhysicalTensor/TargetTensorMaterialization.h"
#include "Wafer/Simulator/Reference/FormalOperations.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace wafer::model {
namespace {

llvm::Error invocationError(TargetModelInvocationErrorCode code,
                            const llvm::Twine &detail) {
  return llvm::make_error<TargetModelInvocationError>(code, detail.str());
}

std::optional<compiler::TileEntryArgumentKind>
getKernelRole(compiler::ProgramResourceRole role) {
  switch (role) {
  case compiler::ProgramResourceRole::UserInput:
    return compiler::TileEntryArgumentKind::ExternalInput;
  case compiler::ProgramResourceRole::Parameter:
  case compiler::ProgramResourceRole::Constant:
    return compiler::TileEntryArgumentKind::TargetTensor;
  case compiler::ProgramResourceRole::Output:
    return compiler::TileEntryArgumentKind::ExternalOutput;
  }
  llvm_unreachable("unknown program resource role");
}

bool isReadOnly(compiler::TileEntryArgumentKind kind) {
  return kind == compiler::TileEntryArgumentKind::ExternalInput ||
         kind == compiler::TileEntryArgumentKind::TargetTensor;
}

llvm::Expected<PhysicalTensorDescriptor>
makeTensorKey(const compiler::TileEntryArgument &slot) {
  const LogicalFormat format = slot.dtype;
  std::vector<uint64_t> shape;
  shape.reserve(slot.shape.size());
  for (int64_t dimension : slot.shape) {
    if (dimension < 0)
      return invocationError(
          TargetModelInvocationErrorCode::InvalidTileEntryArgument,
          "tile entry argument has a dynamic or negative dimension");
    shape.push_back(static_cast<uint64_t>(dimension));
  }
  std::optional<PhysicalTensorLayout> layout;
  switch (slot.layout) {
  case MemLayout::Tensor:
    layout = PhysicalTensorLayout::Tensor;
    break;
  case MemLayout::NTensor:
    layout = PhysicalTensorLayout::NTensor;
    break;
  case MemLayout::Cx:
    layout = PhysicalTensorLayout::Cx;
    break;
  case MemLayout::NCx:
    layout = PhysicalTensorLayout::NCx;
    break;
  }
  if (!layout)
    return invocationError(
        TargetModelInvocationErrorCode::InvalidTileEntryArgument,
        "tile entry argument layout is outside the closed enum");
  llvm::Expected<PhysicalTensorDescriptor> key =
      PhysicalTensorDescriptor::create(format, *layout, std::move(shape));
  if (!key)
    return invocationError(
        TargetModelInvocationErrorCode::InvalidTileEntryArgument,
        llvm::toString(key.takeError()));
  return key;
}

llvm::Expected<std::vector<RawLogicalValue>>
decodeCompactProgramTensor(const compiler::ProgramTensor &tensor,
                           LogicalFormat format) {
  const LogicalFormatDescriptor *descriptor =
      findLogicalFormatDescriptor(format);
  if (!descriptor || descriptor->bitpacked)
    return invocationError(
        TargetModelInvocationErrorCode::UnsupportedProgramTensor,
        "compact source invocation does not support bitpacked program tensors");
  std::optional<int64_t> expected = compiler::computeProgramTensorByteCount(
      tensor.getDType(), tensor.getShape());
  if (!expected || *expected != static_cast<int64_t>(tensor.getBytes().size()))
    return invocationError(
        TargetModelInvocationErrorCode::InvalidProgramInvocation,
        "program tensor compact byte geometry is invalid");
  const uint64_t elementBytes = descriptor->storageBits / 8;
  if (elementBytes == 0 || tensor.getBytes().size() % elementBytes != 0)
    return invocationError(
        TargetModelInvocationErrorCode::InvalidProgramInvocation,
        "program tensor storage width is inconsistent");
  const LogicalScalarCodecPolicy policy = getTargetTensorScalarCodecPolicy();
  std::vector<RawLogicalValue> values;
  values.reserve(tensor.getBytes().size() / elementBytes);
  for (uint64_t offset = 0; offset < tensor.getBytes().size();
       offset += elementBytes) {
    llvm::Expected<RawLogicalValue> value =
        readRawLogicalValue(format, tensor.getBytes(), offset * 8, policy);
    if (!value)
      return invocationError(
          TargetModelInvocationErrorCode::InvalidProgramInvocation,
          llvm::toString(value.takeError()));
    values.push_back(*value);
  }
  return values;
}

llvm::Expected<std::vector<uint8_t>>
encodeCompactProgramTensor(llvm::ArrayRef<RawLogicalValue> values,
                           LogicalFormat format) {
  const LogicalFormatDescriptor *descriptor =
      findLogicalFormatDescriptor(format);
  if (!descriptor || descriptor->bitpacked)
    return invocationError(
        TargetModelInvocationErrorCode::UnsupportedProgramTensor,
        "compact target output does not support bitpacked program tensors");
  const uint64_t elementBytes = descriptor->storageBits / 8;
  if (elementBytes == 0 ||
      values.size() > std::numeric_limits<size_t>::max() / elementBytes)
    return invocationError(TargetModelInvocationErrorCode::AddressOverflow,
                           "compact target output byte count overflows");
  std::vector<uint8_t> bytes(values.size() * elementBytes, 0);
  const LogicalScalarCodecPolicy policy = getTargetTensorScalarCodecPolicy();
  for (auto [index, value] : llvm::enumerate(values)) {
    if (value.format != format)
      return invocationError(
          TargetModelInvocationErrorCode::InvalidTileEntryArgument,
          "target output contains a mixed logical format");
    if (llvm::Error error = writeRawLogicalValue(
            value, bytes, static_cast<uint64_t>(index) * elementBytes * 8,
            policy))
      return invocationError(
          TargetModelInvocationErrorCode::InvalidTileEntryArgument,
          llvm::toString(std::move(error)));
  }
  return bytes;
}

bool checkedAlign(uint64_t value, uint64_t alignment, uint64_t &result) {
  if (alignment == 0)
    return false;
  const uint64_t remainder = value % alignment;
  const uint64_t padding = remainder == 0 ? 0 : alignment - remainder;
  if (padding > std::numeric_limits<uint64_t>::max() - value)
    return false;
  result = value + padding;
  return true;
}

bool haveSameResourceGeometry(const compiler::TileEntryArgument &lhs,
                              const compiler::TileEntryArgument &rhs) {
  return lhs.kind == rhs.kind && lhs.resourceIndex == rhs.resourceIndex &&
         lhs.dtype == rhs.dtype && lhs.layout == rhs.layout &&
         lhs.shape == rhs.shape && lhs.byteSize == rhs.byteSize &&
         lhs.alignment == rhs.alignment && lhs.access == rhs.access &&
         lhs.targetTensorMaterialization == rhs.targetTensorMaterialization;
}

bool haveSameBytesView(llvm::ArrayRef<uint8_t> lhs,
                       llvm::ArrayRef<uint8_t> rhs) {
  return (lhs.data() == rhs.data() && lhs.size() == rhs.size()) ||
         llvm::equal(lhs, rhs);
}

/// Invocation-local materialization of one card-owned read-only input. The
/// source view is non-owning; it is only compared while the invocation
/// inputs are alive.
struct MaterializedInput {
  TargetModelResourceId resource;
  std::vector<uint8_t> bytes;
  llvm::ArrayRef<uint8_t> sourceView;
};

} // namespace

llvm::StringRef
stringifyTargetModelInvocationErrorCode(TargetModelInvocationErrorCode code) {
  switch (code) {
  case TargetModelInvocationErrorCode::InvalidTileDomain:
    return "invalid-tile-domain";
  case TargetModelInvocationErrorCode::InvalidProgramInvocation:
    return "invalid-program-invocation";
  case TargetModelInvocationErrorCode::InvalidTileEntryArgument:
    return "invalid-kernel-abi-slot";
  case TargetModelInvocationErrorCode::UnsupportedProgramTensor:
    return "unsupported-program-tensor";
  case TargetModelInvocationErrorCode::AddressOverflow:
    return "address-overflow";
  case TargetModelInvocationErrorCode::FrontendPreparationFailure:
    return "frontend-preparation-failure";
  }
  llvm_unreachable("unknown target model invocation error code");
}

char TargetModelInvocationError::ID;

void TargetModelInvocationError::log(llvm::raw_ostream &stream) const {
  stream << "target model invocation "
         << stringifyTargetModelInvocationErrorCode(code) << ": " << detail;
}

std::error_code TargetModelInvocationError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

llvm::Expected<std::vector<uint8_t>>
encodeTargetModelProgramTensor(const compiler::ProgramTensor &tensor,
                               const compiler::TileEntryArgument &slot) {
  if (getTargetLogicalFormat(tensor.getDType()) != slot.dtype ||
      tensor.getShape() != llvm::ArrayRef<int64_t>(slot.shape))
    return invocationError(
        TargetModelInvocationErrorCode::InvalidProgramInvocation,
        "program tensor dtype/shape disagrees with tile entry argument");
  llvm::Expected<PhysicalTensorDescriptor> key = makeTensorKey(slot);
  if (!key)
    return key.takeError();
  llvm::Expected<std::vector<RawLogicalValue>> values =
      decodeCompactProgramTensor(tensor, key->getFormat());
  if (!values)
    return values.takeError();
  llvm::Expected<std::vector<uint8_t>> physical =
      packPhysicalTensorLogicalValues(*key, *values, UINT8_C(0));
  if (!physical)
    return invocationError(
        TargetModelInvocationErrorCode::InvalidProgramInvocation,
        llvm::toString(physical.takeError()));
  if (slot.byteSize < 0 ||
      physical->size() != static_cast<uint64_t>(slot.byteSize))
    return invocationError(
        TargetModelInvocationErrorCode::InvalidTileEntryArgument,
        "physical tensor bytes disagree with Kernel ABI");
  return physical;
}

llvm::Expected<compiler::ProgramTensor>
decodeTargetModelProgramTensor(const compiler::TileEntryArgument &slot,
                               llvm::ArrayRef<uint8_t> physicalBytes) {
  if (slot.byteSize < 0 ||
      physicalBytes.size() != static_cast<uint64_t>(slot.byteSize))
    return invocationError(
        TargetModelInvocationErrorCode::InvalidTileEntryArgument,
        "target output byte count disagrees with Kernel ABI");
  llvm::Expected<PhysicalTensorDescriptor> key = makeTensorKey(slot);
  if (!key)
    return key.takeError();
  llvm::Expected<std::vector<RawLogicalValue>> values =
      unpackPhysicalTensorLogicalValues(*key, physicalBytes);
  if (!values)
    return invocationError(
        TargetModelInvocationErrorCode::InvalidTileEntryArgument,
        llvm::toString(values.takeError()));
  llvm::Expected<std::vector<uint8_t>> compact =
      encodeCompactProgramTensor(*values, key->getFormat());
  if (!compact)
    return compact.takeError();
  llvm::Expected<compiler::ProgramTensor> tensor =
      compiler::ProgramTensor::create(getProgramElementType(slot.dtype),
                                      slot.shape, *compact);
  if (!tensor)
    return invocationError(
        TargetModelInvocationErrorCode::InvalidTileEntryArgument,
        llvm::toString(tensor.takeError()));
  return tensor;
}

llvm::Expected<PreparedTargetModelInvocation> prepareTargetModelInvocation(
    const compiler::DeviceExecutable &deviceExecutable,
    const compiler::TargetLLVMModules &targetLLVMModules,
    llvm::ArrayRef<compiler::ProgramTileInvocation> programInvocations) {
  const auto &tiles = deviceExecutable.getTileExecutables();
  const auto &modules = targetLLVMModules.getModules();
  if (deviceExecutable.getExecutionConfig() !=
          targetLLVMModules.getExecutionConfig() ||
      tiles.size() != modules.size() ||
      tiles.size() != programInvocations.size() ||
      tiles.size() != static_cast<size_t>(
                          deviceExecutable.getExecutionConfig().getTileCount()))
    return invocationError(TargetModelInvocationErrorCode::InvalidTileDomain,
                           "source, Tile executable, and target LLVM "
                           "launch domains are not identical");

  std::vector<compiler::TargetCallTileArguments> arguments;
  std::vector<TargetModelInputBinding> inputBindings;
  std::vector<MaterializedInput> materializedInputs;
  struct Allocation {
    TargetModelResourceId resource;
    compiler::TileEntryArgument slot;
    uint64_t base = 0;
    std::vector<int64_t> launchSlots;
  };
  std::vector<Allocation> allocations;
  arguments.reserve(tiles.size());
  uint64_t nextAddress = UINT64_C(0x100000000);
  for (size_t tileIndex = 0; tileIndex < tiles.size(); ++tileIndex) {
    const compiler::TileExecutable &tile = tiles[tileIndex];
    const compiler::TargetLLVMModule &module = modules[tileIndex];
    const compiler::ProgramTileInvocation &invocation =
        programInvocations[tileIndex];
    const LaunchSlotId expectedLaunchSlot(static_cast<int64_t>(tileIndex));
    if (tile.getLaunchSlotId() != expectedLaunchSlot ||
        module.getLaunchSlotId() != expectedLaunchSlot ||
        invocation.launchSlotId != expectedLaunchSlot ||
        module.getCardId() != tile.getCardId() ||
        module.getTileId() != tile.getTileId() ||
        invocation.cardId != tile.getCardId() ||
        invocation.tileId != tile.getTileId())
      return invocationError(
          TargetModelInvocationErrorCode::InvalidTileDomain,
          "Tile executable or target LLVM launch order is not "
          "canonical");

    compiler::TargetCallTileArguments tileArguments{
        tile.getCardId(), tile.getTileId(), tile.getLaunchSlotId(), {}};
    const auto &slots = module.getTileEntryArguments();
    tileArguments.slots.reserve(slots.size());
    std::vector<bool> consumedInputs(invocation.inputs.size(), false);
    for (auto [slotIndex, slot] : llvm::enumerate(slots)) {
      if (slot.ordinal != static_cast<int64_t>(slotIndex) ||
          slot.resourceIndex < 0 || slot.byteSize <= 0 || slot.alignment <= 0)
        return invocationError(
            TargetModelInvocationErrorCode::InvalidTileEntryArgument,
            "tile entry argument identity, byte size, or alignment is invalid");
      const TargetModelResourceId resource = getTargetModelResourceId(
          tile.getCardId(), tile.getTileId(), slot.kind, slot.resourceIndex);
      Allocation *allocation = nullptr;
      for (Allocation &candidate : allocations)
        if (candidate.resource == resource) {
          allocation = &candidate;
          break;
        }
      if (allocation) {
        if (llvm::is_contained(allocation->launchSlots,
                               tile.getLaunchSlotId().getValue()))
          return invocationError(
              TargetModelInvocationErrorCode::InvalidTileEntryArgument,
              "one Tile ABI references a physical resource more than once");
        if (!haveSameResourceGeometry(allocation->slot, slot))
          return invocationError(
              TargetModelInvocationErrorCode::InvalidTileEntryArgument,
              "tile entry arguments disagree on one physical resource");
        allocation->launchSlots.push_back(tile.getLaunchSlotId().getValue());
      } else {
        uint64_t base = 0;
        if (!checkedAlign(nextAddress, static_cast<uint64_t>(slot.alignment),
                          base) ||
            static_cast<uint64_t>(slot.byteSize) >
                std::numeric_limits<uint64_t>::max() - base)
          return invocationError(
              TargetModelInvocationErrorCode::AddressOverflow,
              "tile entry argument address domain overflows");
        allocations.push_back(
            {resource, slot, base, {tile.getLaunchSlotId().getValue()}});
        allocation = &allocations.back();
        nextAddress = base + static_cast<uint64_t>(slot.byteSize);
      }
      tileArguments.slots.push_back(allocation->base);

      if (!isReadOnly(slot.kind))
        continue;
      const compiler::ProgramInputBinding *input = nullptr;
      size_t inputIndex = 0;
      for (auto [candidateIndex, candidate] :
           llvm::enumerate(invocation.inputs)) {
        if (getKernelRole(candidate.role) == slot.kind &&
            candidate.index == slot.resourceIndex) {
          if (input)
            return invocationError(
                TargetModelInvocationErrorCode::InvalidProgramInvocation,
                "duplicate program input for one tile entry argument");
          input = &candidate;
          inputIndex = candidateIndex;
        }
      }
      if (!input)
        return invocationError(
            TargetModelInvocationErrorCode::InvalidProgramInvocation,
            "missing program input for one tile entry argument");
      if (consumedInputs[inputIndex])
        return invocationError(
            TargetModelInvocationErrorCode::InvalidProgramInvocation,
            "program input is referenced by multiple tile entry arguments");
      consumedInputs[inputIndex] = true;
      MaterializedInput *existing = nullptr;
      for (MaterializedInput &candidate : materializedInputs)
        if (candidate.resource == resource) {
          existing = &candidate;
          break;
        }
      if (existing) {
        // One card-owned resource referenced by several Tile ABI slots:
        // the identical partition-0 binding must present the same bytes.
        // Shared transaction materialization views agree by storage
        // identity; independently sliced inputs agree by byte comparison.
        // The target codec runs exactly once per card-shared resource.
        if (!haveSameBytesView(existing->sourceView, input->tensor.getBytes()))
          return invocationError(
              TargetModelInvocationErrorCode::InvalidProgramInvocation,
              "Tile invocations disagree on one card-shared input resource");
      } else {
        llvm::Expected<std::vector<uint8_t>> bytes =
            encodeTargetModelProgramTensor(input->tensor, slot);
        if (!bytes)
          return bytes.takeError();
        materializedInputs.push_back(
            {resource, std::move(*bytes), input->tensor.getBytes()});
      }
    }
    if (!llvm::all_of(consumedInputs, [](bool consumed) { return consumed; }))
      return invocationError(
          TargetModelInvocationErrorCode::InvalidProgramInvocation,
          "program invocation contains an input absent from the Kernel ABI");
    arguments.push_back(std::move(tileArguments));
  }

  for (const Allocation &allocation : allocations) {
    const bool sharedAcrossTiles = !allocation.resource.tileId.has_value();
    if ((sharedAcrossTiles && allocation.launchSlots.size() != tiles.size()) ||
        (!sharedAcrossTiles && allocation.launchSlots.size() != 1))
      return invocationError(
          TargetModelInvocationErrorCode::InvalidTileEntryArgument,
          "physical resource owner disagrees with its Tile ABI domain");
  }

  inputBindings.reserve(materializedInputs.size());
  for (MaterializedInput &materialized : materializedInputs)
    inputBindings.push_back(
        {materialized.resource, std::move(materialized.bytes)});
  llvm::Expected<compiler::TargetCallExecutable> executable =
      compiler::createTargetCallExecutable(targetLLVMModules, arguments);
  if (!executable)
    return invocationError(
        TargetModelInvocationErrorCode::FrontendPreparationFailure,
        llvm::toString(executable.takeError()));
  return PreparedTargetModelInvocation(std::move(*executable),
                                       std::move(inputBindings));
}

} // namespace wafer::model
