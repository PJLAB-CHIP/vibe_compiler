//===- TargetModelInvocation.cpp - Typed source/model binding -----------===//

#include "Wafer/Model/TargetModelInvocation.h"

#include "Wafer/Target/NumericSemantics.h"
#include "Wafer/Target/PhysicalTensorCodec.h"

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

std::optional<LogicalFormat> getLogicalFormat(llvm::StringRef dtype) {
  if (dtype == "i1")
    return LogicalFormat::Bool;
  if (dtype == "ui8")
    return LogicalFormat::U8;
  if (dtype == "ui16")
    return LogicalFormat::U16;
  if (dtype == "ui32")
    return LogicalFormat::U32;
  if (dtype == "ui64")
    return LogicalFormat::U64;
  llvm::Expected<LogicalFormat> parsed = parseLogicalFormat(dtype);
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return std::nullopt;
  }
  return *parsed;
}

NumericTensorLayout getNumericLayout(MemLayout layout) {
  switch (layout) {
  case MemLayout::Tensor:
    return NumericTensorLayout::Tensor;
  case MemLayout::NTensor:
    return NumericTensorLayout::NTensor;
  case MemLayout::Cx:
    return NumericTensorLayout::Cx;
  case MemLayout::NCx:
    return NumericTensorLayout::NCx;
  }
  llvm_unreachable("unknown Kernel ABI memory layout");
}

std::optional<compiler::KernelABISlotRole>
getKernelRole(compiler::ProgramResourceRole role) {
  switch (role) {
  case compiler::ProgramResourceRole::UserInput:
    return compiler::KernelABISlotRole::UserInput;
  case compiler::ProgramResourceRole::Parameter:
    return compiler::KernelABISlotRole::Parameter;
  case compiler::ProgramResourceRole::Constant:
    return compiler::KernelABISlotRole::Constant;
  case compiler::ProgramResourceRole::Output:
    return compiler::KernelABISlotRole::Output;
  }
  llvm_unreachable("unknown program resource role");
}

bool isReadOnly(compiler::KernelABISlotRole role) {
  return role == compiler::KernelABISlotRole::UserInput ||
         role == compiler::KernelABISlotRole::Parameter ||
         role == compiler::KernelABISlotRole::Constant;
}

llvm::Expected<NumericTensorKey>
makeTensorKey(const compiler::KernelABISlot &slot) {
  std::optional<LogicalFormat> format = getLogicalFormat(slot.dtype);
  if (!format)
    return invocationError(
        TargetModelInvocationErrorCode::UnsupportedProgramTensor,
        "Kernel ABI slot has no logical format: " + slot.dtype);
  std::vector<uint64_t> shape;
  shape.reserve(slot.shape.size());
  for (int64_t dimension : slot.shape) {
    if (dimension < 0)
      return invocationError(
          TargetModelInvocationErrorCode::InvalidKernelABISlot,
          "Kernel ABI slot has a dynamic or negative dimension");
    shape.push_back(static_cast<uint64_t>(dimension));
  }
  llvm::Expected<NumericTensorKey> key = NumericTensorKey::create(
      *format, getNumericLayout(slot.layout), std::move(shape));
  if (!key)
    return invocationError(TargetModelInvocationErrorCode::InvalidKernelABISlot,
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
        "compact source invocation does not admit bitpacked program tensors");
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
  const LogicalScalarCodecPolicy policy =
      getModelProfileRecord(ModelProfileId::formalDeterministicV1())
          .numericDecodePolicy;
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
                           llvm::StringRef dtype, LogicalFormat format) {
  const LogicalFormatDescriptor *descriptor =
      findLogicalFormatDescriptor(format);
  if (!descriptor || descriptor->bitpacked)
    return invocationError(
        TargetModelInvocationErrorCode::UnsupportedProgramTensor,
        "compact target output does not admit bitpacked program tensors");
  const uint64_t elementBytes = descriptor->storageBits / 8;
  if (elementBytes == 0 ||
      values.size() > std::numeric_limits<size_t>::max() / elementBytes)
    return invocationError(TargetModelInvocationErrorCode::AddressOverflow,
                           "compact target output byte count overflows");
  std::vector<uint8_t> bytes(values.size() * elementBytes, 0);
  const LogicalScalarCodecPolicy policy =
      getModelProfileRecord(ModelProfileId::formalDeterministicV1())
          .numericEncodePolicy;
  for (auto [index, value] : llvm::enumerate(values)) {
    if (value.format != format)
      return invocationError(
          TargetModelInvocationErrorCode::InvalidKernelABISlot,
          "target output contains a mixed logical format");
    if (llvm::Error error = writeRawLogicalValue(
            value, bytes, static_cast<uint64_t>(index) * elementBytes * 8,
            policy))
      return invocationError(
          TargetModelInvocationErrorCode::InvalidKernelABISlot,
          llvm::toString(std::move(error)));
  }
  (void)dtype;
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

} // namespace

llvm::StringRef
stringifyTargetModelInvocationErrorCode(TargetModelInvocationErrorCode code) {
  switch (code) {
  case TargetModelInvocationErrorCode::InvalidBundleDomain:
    return "invalid-bundle-domain";
  case TargetModelInvocationErrorCode::InvalidProgramInvocation:
    return "invalid-program-invocation";
  case TargetModelInvocationErrorCode::InvalidKernelABISlot:
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
                               const compiler::KernelABISlot &slot) {
  if (tensor.getDType() != slot.dtype ||
      tensor.getShape() != llvm::ArrayRef<int64_t>(slot.shape))
    return invocationError(
        TargetModelInvocationErrorCode::InvalidProgramInvocation,
        "program tensor dtype/shape disagrees with Kernel ABI slot");
  llvm::Expected<NumericTensorKey> key = makeTensorKey(slot);
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
    return invocationError(TargetModelInvocationErrorCode::InvalidKernelABISlot,
                           "physical tensor bytes disagree with Kernel ABI");
  return physical;
}

llvm::Expected<compiler::ProgramTensor>
decodeTargetModelProgramTensor(const compiler::KernelABISlot &slot,
                               llvm::ArrayRef<uint8_t> physicalBytes) {
  if (slot.byteSize < 0 ||
      physicalBytes.size() != static_cast<uint64_t>(slot.byteSize))
    return invocationError(
        TargetModelInvocationErrorCode::InvalidKernelABISlot,
        "target output byte count disagrees with Kernel ABI");
  llvm::Expected<NumericTensorKey> key = makeTensorKey(slot);
  if (!key)
    return key.takeError();
  llvm::Expected<std::vector<RawLogicalValue>> values =
      unpackPhysicalTensorLogicalValues(*key, physicalBytes);
  if (!values)
    return invocationError(TargetModelInvocationErrorCode::InvalidKernelABISlot,
                           llvm::toString(values.takeError()));
  llvm::Expected<std::vector<uint8_t>> compact =
      encodeCompactProgramTensor(*values, slot.dtype, key->getFormat());
  if (!compact)
    return compact.takeError();
  llvm::Expected<compiler::ProgramTensor> tensor =
      compiler::ProgramTensor::create(slot.dtype, slot.shape, *compact);
  if (!tensor)
    return invocationError(TargetModelInvocationErrorCode::InvalidKernelABISlot,
                           llvm::toString(tensor.takeError()));
  return tensor;
}

llvm::Expected<PreparedTargetModelInvocation> prepareTargetModelInvocation(
    const compiler::ExecutableBundle &executableBundle,
    const compiler::TargetLLVMModuleBundle &targetLLVMModuleBundle,
    llvm::ArrayRef<compiler::ProgramRankInvocation> programInvocations) {
  const auto &ranks = executableBundle.getRankExecutables();
  const auto &modules = targetLLVMModuleBundle.getModules();
  if (executableBundle.getExecutionConfig() !=
          targetLLVMModuleBundle.getExecutionConfig() ||
      ranks.size() != modules.size() ||
      ranks.size() != programInvocations.size() ||
      ranks.size() != static_cast<size_t>(
                          executableBundle.getExecutionConfig().getRankCount()))
    return invocationError(TargetModelInvocationErrorCode::InvalidBundleDomain,
                           "source, executable, and target LLVM rank domains "
                           "are not identical");

  std::vector<compiler::TargetCallRankArguments> arguments;
  std::vector<TargetModelInputBinding> inputBindings;
  arguments.reserve(ranks.size());
  uint64_t nextAddress = UINT64_C(0x100000000);
  for (size_t rankIndex = 0; rankIndex < ranks.size(); ++rankIndex) {
    const compiler::RankExecutable &rank = ranks[rankIndex];
    const compiler::TargetLLVMModule &module = modules[rankIndex];
    const compiler::ProgramRankInvocation &invocation =
        programInvocations[rankIndex];
    if (rank.getLogicalRank() != static_cast<int64_t>(rankIndex) ||
        module.getLogicalRank() != static_cast<int64_t>(rankIndex) ||
        invocation.logicalRank != static_cast<int64_t>(rankIndex))
      return invocationError(
          TargetModelInvocationErrorCode::InvalidBundleDomain,
          "source, executable, or target LLVM rank order is not canonical");

    compiler::TargetCallRankArguments rankArguments;
    rankArguments.logicalRank = static_cast<int64_t>(rankIndex);
    const auto &slots = module.getKernelABISlots();
    rankArguments.slots.reserve(slots.size());
    std::vector<bool> consumedInputs(invocation.inputs.size(), false);
    for (auto [slotIndex, slot] : llvm::enumerate(slots)) {
      if (slot.ordinal != static_cast<int64_t>(slotIndex) ||
          slot.byteSize < 0 || slot.alignment <= 0)
        return invocationError(
            TargetModelInvocationErrorCode::InvalidKernelABISlot,
            "Kernel ABI slot order, byte size, or alignment is invalid");
      uint64_t base = 0;
      if (!checkedAlign(nextAddress, static_cast<uint64_t>(slot.alignment),
                        base) ||
          static_cast<uint64_t>(slot.byteSize) >
              std::numeric_limits<uint64_t>::max() - base)
        return invocationError(TargetModelInvocationErrorCode::AddressOverflow,
                               "Kernel ABI address domain overflows");
      rankArguments.slots.push_back(base);
      nextAddress = base + static_cast<uint64_t>(slot.byteSize);

      if (!isReadOnly(slot.role))
        continue;
      const compiler::ProgramInputBinding *input = nullptr;
      size_t inputIndex = 0;
      for (auto [candidateIndex, candidate] :
           llvm::enumerate(invocation.inputs)) {
        if (getKernelRole(candidate.role) == slot.role &&
            candidate.index == slot.resourceIndex) {
          if (input)
            return invocationError(
                TargetModelInvocationErrorCode::InvalidProgramInvocation,
                "duplicate program input for one Kernel ABI slot");
          input = &candidate;
          inputIndex = candidateIndex;
        }
      }
      if (!input)
        return invocationError(
            TargetModelInvocationErrorCode::InvalidProgramInvocation,
            "missing program input for one Kernel ABI slot");
      if (consumedInputs[inputIndex])
        return invocationError(
            TargetModelInvocationErrorCode::InvalidProgramInvocation,
            "program input is referenced by multiple Kernel ABI slots");
      llvm::Expected<std::vector<uint8_t>> bytes =
          encodeTargetModelProgramTensor(input->tensor, slot);
      if (!bytes)
        return bytes.takeError();
      consumedInputs[inputIndex] = true;
      inputBindings.push_back(
          {static_cast<int64_t>(rankIndex), slot.ordinal, std::move(*bytes)});
    }
    if (!llvm::all_of(consumedInputs, [](bool consumed) { return consumed; }))
      return invocationError(
          TargetModelInvocationErrorCode::InvalidProgramInvocation,
          "program invocation contains an input absent from the Kernel ABI");
    arguments.push_back(std::move(rankArguments));
  }

  llvm::Expected<compiler::TargetCallExecutable> executable =
      compiler::prepareTargetCallFrontend(targetLLVMModuleBundle, arguments);
  if (!executable)
    return invocationError(
        TargetModelInvocationErrorCode::FrontendPreparationFailure,
        llvm::toString(executable.takeError()));
  return PreparedTargetModelInvocation(std::move(*executable),
                                       std::move(inputBindings));
}

} // namespace wafer::model
