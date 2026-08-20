//===- TargetModelTensorNumeric.cpp - Tensor numeric kernels ---------===//

#include "TargetModelKernelInternal.h"

#include "Wafer/Target/Core/TargetFormat.h"
#include "Wafer/Target/Layout/PhysicalTensorCodec.h"
#include "Wafer/Target/Layout/TargetTensorMaterialization.h"

#include "llvm/ADT/STLExtras.h"

#include <optional>
#include <utility>
#include <vector>

namespace wafer::model::kernel_detail {
namespace {

PhysicalTensorLayout getCountLayout(LogicalFormat format) {
  return format == LogicalFormat::Bool ? PhysicalTensorLayout::Tensor
                                       : PhysicalTensorLayout::Cx;
}

llvm::Expected<PhysicalTensorDescriptor>
makeTensor(LogicalFormat format, std::vector<uint64_t> shape) {
  llvm::Expected<PhysicalTensorDescriptor> key =
      PhysicalTensorDescriptor::create(format, getCountLayout(format),
                                       std::move(shape));
  if (!key)
    return kernelError(TargetModelKernelErrorCode::FormalNumericFailure,
                       llvm::toString(key.takeError()));
  return key;
}

TargetModelByteRead makeTensorRead(int64_t launchSlot, uint64_t address,
                                   const PhysicalTensorDescriptor &key) {
  return TargetModelByteRead{
      launchSlot, TargetModelAddressSpace::TileSPM, address,
      llvm::cantFail(getPhysicalTensorStorageBytes(key)), std::nullopt};
}

TargetModelCommandEffect
withReads(TargetModelCommandEffect effect,
          std::vector<TargetModelByteRead> pendingReads) {
  effect.pendingReads = std::move(pendingReads);
  return effect;
}

llvm::Expected<std::vector<RawLogicalValue>>
readTensor(const InvocationMemoryRegistry &memory, int64_t launchSlot,
           uint64_t address, const PhysicalTensorDescriptor &key) {
  llvm::Expected<uint64_t> bytes = getPhysicalTensorStorageBytes(key);
  if (!bytes)
    return kernelError(TargetModelKernelErrorCode::PhysicalCodecFailure,
                       llvm::toString(bytes.takeError()));
  llvm::Expected<std::vector<uint8_t>> storage = readSnapshot(
      memory, launchSlot, TargetModelAddressSpace::TileSPM, address, *bytes);
  if (!storage)
    return storage.takeError();
  llvm::Expected<std::vector<RawLogicalValue>> values =
      unpackPhysicalTensorLogicalValues(key, *storage);
  if (!values)
    return kernelError(TargetModelKernelErrorCode::PhysicalCodecFailure,
                       llvm::toString(values.takeError()));
  return values;
}

llvm::Expected<std::vector<uint8_t>>
readTensorStorage(const InvocationMemoryRegistry &memory, int64_t launchSlot,
                  uint64_t address, const PhysicalTensorDescriptor &key) {
  llvm::Expected<uint64_t> bytes = getPhysicalTensorStorageBytes(key);
  if (!bytes)
    return kernelError(TargetModelKernelErrorCode::PhysicalCodecFailure,
                       llvm::toString(bytes.takeError()));
  return readSnapshot(memory, launchSlot, TargetModelAddressSpace::TileSPM,
                      address, *bytes);
}

llvm::Expected<std::vector<uint8_t>>
packTensor(const InvocationMemoryRegistry &memory, int64_t launchSlot,
           uint64_t address, const PhysicalTensorDescriptor &key,
           llvm::ArrayRef<RawLogicalValue> values) {
  llvm::Expected<uint64_t> bytes = getPhysicalTensorStorageBytes(key);
  if (!bytes)
    return kernelError(TargetModelKernelErrorCode::PhysicalCodecFailure,
                       llvm::toString(bytes.takeError()));
  llvm::Expected<std::vector<uint8_t>> storage = readSnapshot(
      memory, launchSlot, TargetModelAddressSpace::TileSPM, address, *bytes);
  if (!storage)
    return storage.takeError();
  llvm::Expected<std::vector<uint8_t>> packed =
      packPhysicalTensorLogicalValues(key, values, *storage);
  if (!packed)
    return kernelError(TargetModelKernelErrorCode::PhysicalCodecFailure,
                       llvm::toString(packed.takeError()));
  return packed;
}

struct ManagedReferenceInput {
  uint64_t address;
  const PhysicalTensorDescriptor *key;
};

uint64_t getScalarEvaluations(const FormalConvertOperation &operation) {
  return operation.destination.getElementCount();
}
uint64_t getScalarEvaluations(const FormalElementwiseOperation &operation) {
  return operation.destination.getElementCount();
}
uint64_t getScalarEvaluations(const FormalReduceOperation &operation) {
  return operation.input.getElementCount();
}

TargetModelConvertRequest
makeManagedRequest(const FormalConvertOperation &operation,
                   TargetModelNumericOperands tensors) {
  return {operation, std::move(tensors)};
}
TargetModelElementwiseRequest
makeManagedRequest(const FormalElementwiseOperation &operation,
                   TargetModelNumericOperands tensors) {
  return {operation, std::move(tensors)};
}
TargetModelReduceRequest
makeManagedRequest(const FormalReduceOperation &operation,
                   TargetModelNumericOperands tensors) {
  return {operation, std::move(tensors)};
}

template <typename Operation>
llvm::Expected<std::optional<TargetModelCommandEffect>>
tryExecuteManagedReference(
    const compiler::TargetCommand &command,
    const InvocationMemoryRegistry &memory, const Operation &operation,
    llvm::ArrayRef<ManagedReferenceInput> inputDescriptors,
    uint64_t destinationAddress, const PhysicalTensorDescriptor &destinationKey,
    TargetModelKernelBudget budget, TargetModelExecutionPolicy policy) {
  if (policy.getTensorDispatchPolicy() ==
      TargetModelTensorDispatchPolicy::FormalOnly)
    return std::optional<TargetModelCommandEffect>();
  const TargetModelManagedReferenceBackend *backend =
      policy.getManagedReferenceBackend();
  if (!backend)
    return kernelError(
        TargetModelKernelErrorCode::ManagedReferenceBackendUnavailable,
        "managed-reference tensor policy has no functional backend");

  std::vector<TargetModelNumericTensor> inputs;
  inputs.reserve(inputDescriptors.size());
  std::vector<TargetModelByteRead> pendingReads;
  pendingReads.reserve(inputDescriptors.size());
  for (const ManagedReferenceInput &input : inputDescriptors) {
    llvm::Expected<std::vector<uint8_t>> storage = readTensorStorage(
        memory, command.launchSlotId.getValue(), input.address, *input.key);
    if (!storage)
      return storage.takeError();
    pendingReads.push_back(makeTensorRead(command.launchSlotId.getValue(),
                                          input.address, *input.key));
    inputs.push_back({*input.key, std::move(*storage)});
  }
  llvm::Expected<std::vector<uint8_t>> destinationStorage =
      readTensorStorage(memory, command.launchSlotId.getValue(),
                        destinationAddress, destinationKey);
  if (!destinationStorage)
    return destinationStorage.takeError();
  const size_t expectedStorageSize = destinationStorage->size();
  auto request = makeManagedRequest(
      operation,
      TargetModelNumericOperands{
          std::move(inputs), {destinationKey, std::move(*destinationStorage)}});
  llvm::Expected<TargetModelManagedReferenceResult> result =
      backend->execute(request, budget.getNumericBudget());
  if (!result)
    return kernelError(
        TargetModelKernelErrorCode::ManagedReferenceBackendFailure,
        llvm::toString(result.takeError()));

  const uint64_t expectedScalarEvaluations = getScalarEvaluations(operation);
  if (result->destination.key != destinationKey ||
      result->destination.storage.size() != expectedStorageSize ||
      result->evidence.scalarEvaluations != expectedScalarEvaluations ||
      result->evidence.environmentDigest.empty() ||
      result->evidence.implementation.empty())
    return kernelError(
        TargetModelKernelErrorCode::ManagedReferenceBackendFailure,
        "managed-reference backend returned incomplete or mismatched evidence");
  return std::optional<TargetModelCommandEffect>(withReads(
      TargetModelCommandEffect{
          {TargetModelByteWrite{
              command.launchSlotId.getValue(), TargetModelAddressSpace::TileSPM,
              destinationAddress, 1, std::move(result->destination.storage)}},
          result->flags,
          TargetModelControlAction::None,
          TargetModelNumericBackend::ManagedReference,
          {},
          std::move(result->evidence)},
      std::move(pendingReads)));
}

} // namespace

llvm::Expected<TargetModelCommandEffect>
executeElementwise(const compiler::TargetCommand &command,
                   const target::TargetElementwiseCommand &value,
                   const InvocationMemoryRegistry &memory,
                   TargetModelKernelBudget budget,
                   TargetModelExecutionPolicy policy) {
  const LogicalFormat destinationFormat =
      isTargetElementwiseRelation(value.operation) ? LogicalFormat::Bool
                                                   : value.format;
  llvm::Expected<PhysicalTensorDescriptor> inputKey =
      makeTensor(value.format, {value.elementCount});
  llvm::Expected<PhysicalTensorDescriptor> destinationKey =
      makeTensor(destinationFormat, {value.elementCount});
  if (!inputKey)
    return inputKey.takeError();
  if (!destinationKey)
    return destinationKey.takeError();
  std::vector<PhysicalTensorDescriptor> inputKeys;
  inputKeys.push_back(*inputKey);
  if (value.rhs)
    inputKeys.push_back(*inputKey);
  llvm::Expected<FormalElementwiseOperation> operation =
      createFormalElementwiseOperation(value.operation, std::move(inputKeys),
                                       *destinationKey);
  if (!operation)
    return kernelError(TargetModelKernelErrorCode::FormalNumericFailure,
                       llvm::toString(operation.takeError()));

  std::vector<ManagedReferenceInput> managedInputs{{value.lhs, &*inputKey}};
  if (value.rhs)
    managedInputs.push_back({*value.rhs, &*inputKey});
  llvm::Expected<std::optional<TargetModelCommandEffect>> managed =
      tryExecuteManagedReference(command, memory, *operation, managedInputs,
                                 value.destination, *destinationKey, budget,
                                 policy);
  if (!managed)
    return managed.takeError();
  if (*managed)
    return std::move(**managed);

  std::vector<std::vector<RawLogicalValue>> inputs;
  llvm::Expected<std::vector<RawLogicalValue>> lhs =
      readTensor(memory, command.launchSlotId.getValue(), value.lhs, *inputKey);
  if (!lhs)
    return lhs.takeError();
  inputs.push_back(std::move(*lhs));
  if (value.rhs) {
    llvm::Expected<std::vector<RawLogicalValue>> rhs = readTensor(
        memory, command.launchSlotId.getValue(), *value.rhs, *inputKey);
    if (!rhs)
      return rhs.takeError();
    inputs.push_back(std::move(*rhs));
  }
  std::vector<llvm::ArrayRef<RawLogicalValue>> views;
  for (const std::vector<RawLogicalValue> &input : inputs)
    views.push_back(input);
  FormalNumericExecutionContext localContext;
  llvm::Expected<FormalTensorNumericResult> result = executeFormalTensorNumeric(
      localContext, *operation, views, budget.getNumericBudget());
  if (!result)
    return kernelError(TargetModelKernelErrorCode::FormalNumericFailure,
                       llvm::toString(result.takeError()));
  llvm::Expected<std::vector<uint8_t>> packed =
      packTensor(memory, command.launchSlotId.getValue(), value.destination,
                 *destinationKey, result->values);
  if (!packed)
    return packed.takeError();
  std::vector<TargetModelByteRead> pendingReads{
      makeTensorRead(command.launchSlotId.getValue(), value.lhs, *inputKey)};
  if (value.rhs)
    pendingReads.push_back(
        makeTensorRead(command.launchSlotId.getValue(), *value.rhs, *inputKey));
  return withReads(
      TargetModelCommandEffect{
          {TargetModelByteWrite{command.launchSlotId.getValue(),
                                TargetModelAddressSpace::TileSPM,
                                value.destination, 1, std::move(*packed)}},
          result->flags,
          TargetModelControlAction::None,
          TargetModelNumericBackend::Formal},
      std::move(pendingReads));
}

llvm::Expected<TargetModelCommandEffect>
executeConvert(const compiler::TargetCommand &command,
               const target::TargetConvertCommand &value,
               const InvocationMemoryRegistry &memory,
               TargetModelKernelBudget budget,
               TargetModelExecutionPolicy policy) {
  const uint16_t opcode = value.operation.getOpcode();
  const TargetConvertRoute *route = findTargetConvertRoute(opcode);
  if (!route)
    return kernelError(
        TargetModelKernelErrorCode::FormalNumericFailure,
        "convert route is not registered for the current target");
  llvm::Expected<PhysicalTensorDescriptor> sourceKey =
      makeTensor(route->source, {value.elementCount});
  llvm::Expected<PhysicalTensorDescriptor> destinationKey =
      makeTensor(route->destination, {value.elementCount});
  if (!sourceKey)
    return sourceKey.takeError();
  if (!destinationKey)
    return destinationKey.takeError();
  llvm::Expected<FormalConvertOperation> operation =
      createFormalConvertOperation(opcode, *sourceKey, *destinationKey,
                                   value.parameter);
  if (!operation)
    return kernelError(TargetModelKernelErrorCode::FormalNumericFailure,
                       llvm::toString(operation.takeError()));
  const ManagedReferenceInput managedInput{value.source, &*sourceKey};
  llvm::Expected<std::optional<TargetModelCommandEffect>> managed =
      tryExecuteManagedReference(command, memory, *operation, managedInput,
                                 value.destination, *destinationKey, budget,
                                 policy);
  if (!managed)
    return managed.takeError();
  if (*managed)
    return std::move(**managed);
  llvm::Expected<std::vector<RawLogicalValue>> source = readTensor(
      memory, command.launchSlotId.getValue(), value.source, *sourceKey);
  if (!source)
    return source.takeError();
  std::vector<llvm::ArrayRef<RawLogicalValue>> views{*source};
  FormalNumericExecutionContext localContext;
  llvm::Expected<FormalTensorNumericResult> result = executeFormalTensorNumeric(
      localContext, *operation, views, budget.getNumericBudget());
  if (!result)
    return kernelError(TargetModelKernelErrorCode::FormalNumericFailure,
                       llvm::toString(result.takeError()));
  llvm::Expected<std::vector<uint8_t>> packed =
      packTensor(memory, command.launchSlotId.getValue(), value.destination,
                 *destinationKey, result->values);
  if (!packed)
    return packed.takeError();
  return withReads(
      TargetModelCommandEffect{
          {TargetModelByteWrite{command.launchSlotId.getValue(),
                                TargetModelAddressSpace::TileSPM,
                                value.destination, 1, std::move(*packed)}},
          result->flags,
          TargetModelControlAction::None,
          TargetModelNumericBackend::Formal},
      {makeTensorRead(command.launchSlotId.getValue(), value.source,
                      *sourceKey)});
}

llvm::Expected<TargetModelCommandEffect>
executeReduce(const compiler::TargetCommand &command,
              const target::TargetReduceCommand &value,
              const InvocationMemoryRegistry &memory,
              TargetModelKernelBudget budget,
              TargetModelExecutionPolicy policy) {
  const auto dimension = static_cast<TargetReduceDimension>(value.dimension);
  std::vector<uint64_t> inputShape(value.nhwc.begin(), value.nhwc.end());
  const std::vector<size_t> reducedDimensions =
      getTargetReduceLogicalDimensions(dimension, inputShape.size());
  if (reducedDimensions.empty())
    return kernelError(TargetModelKernelErrorCode::InvalidCommandField,
                       "reduce dimension is invalid for the fixed ABI shape");
  std::vector<uint64_t> destinationShape;
  for (size_t index = 0; index < inputShape.size(); ++index)
    if (!llvm::is_contained(reducedDimensions, index))
      destinationShape.push_back(inputShape[index]);
  const PhysicalTensorLayout destinationLayout = destinationShape.size() > 2
                                                     ? PhysicalTensorLayout::NCx
                                                     : PhysicalTensorLayout::Cx;
  llvm::Expected<PhysicalTensorDescriptor> inputKey =
      PhysicalTensorDescriptor::create(value.format, PhysicalTensorLayout::NCx,
                                       std::move(inputShape));
  llvm::Expected<PhysicalTensorDescriptor> destinationKey =
      PhysicalTensorDescriptor::create(value.format, destinationLayout,
                                       std::move(destinationShape));
  if (!inputKey || !destinationKey) {
    llvm::Error errors = llvm::Error::success();
    if (!inputKey)
      errors = llvm::joinErrors(std::move(errors), inputKey.takeError());
    if (!destinationKey)
      errors = llvm::joinErrors(std::move(errors), destinationKey.takeError());
    return kernelError(TargetModelKernelErrorCode::FormalNumericFailure,
                       llvm::toString(std::move(errors)));
  }
  llvm::Expected<FormalReduceOperation> operation = createFormalReduceOperation(
      value.operation, *inputKey, *destinationKey, dimension);
  if (!operation)
    return kernelError(TargetModelKernelErrorCode::FormalNumericFailure,
                       llvm::toString(operation.takeError()));
  const ManagedReferenceInput managedInput{value.source, &*inputKey};
  llvm::Expected<std::optional<TargetModelCommandEffect>> managed =
      tryExecuteManagedReference(command, memory, *operation, managedInput,
                                 value.destination, *destinationKey, budget,
                                 policy);
  if (!managed)
    return managed.takeError();
  if (*managed)
    return std::move(**managed);
  llvm::Expected<std::vector<RawLogicalValue>> input = readTensor(
      memory, command.launchSlotId.getValue(), value.source, *inputKey);
  if (!input)
    return input.takeError();
  std::vector<llvm::ArrayRef<RawLogicalValue>> views{*input};
  FormalNumericExecutionContext localContext;
  llvm::Expected<FormalTensorNumericResult> result = executeFormalTensorNumeric(
      localContext, *operation, views, budget.getNumericBudget());
  if (!result)
    return kernelError(TargetModelKernelErrorCode::FormalNumericFailure,
                       llvm::toString(result.takeError()));
  llvm::Expected<std::vector<uint8_t>> packed =
      packTensor(memory, command.launchSlotId.getValue(), value.destination,
                 *destinationKey, result->values);
  if (!packed)
    return packed.takeError();
  return withReads(
      TargetModelCommandEffect{
          {TargetModelByteWrite{command.launchSlotId.getValue(),
                                TargetModelAddressSpace::TileSPM,
                                value.destination, 1, std::move(*packed)}},
          result->flags,
          TargetModelControlAction::None,
          TargetModelNumericBackend::Formal},
      {makeTensorRead(command.launchSlotId.getValue(), value.source,
                      *inputKey)});
}

llvm::Expected<TargetModelCommandEffect>
executeGemm(const compiler::TargetCommand &command,
            const target::TargetGemmCommand &value,
            const InvocationMemoryRegistry &memory,
            TargetModelKernelBudget budget, TargetModelExecutionPolicy policy) {
  const bool batched = value.batchCount > 1;
  // The target call has no free layout field. Its storage contract is Cx for
  // rank-2 GEMM and NCx (including per-batch bank alignment) for batched GEMM.
  const PhysicalTensorLayout layout =
      batched ? PhysicalTensorLayout::NCx : PhysicalTensorLayout::Cx;
  std::vector<uint64_t> lhsShape =
      batched
          ? (value.lhsOrientation == TargetGemmOrientation::Normal
                 ? std::vector<uint64_t>{value.batchCount, value.m, value.k}
                 : std::vector<uint64_t>{value.batchCount, value.k, value.m})
          : (value.lhsOrientation == TargetGemmOrientation::Normal
                 ? std::vector<uint64_t>{value.m, value.k}
                 : std::vector<uint64_t>{value.k, value.m});
  std::vector<uint64_t> rhsShape =
      batched
          ? (value.rhsOrientation == TargetGemmOrientation::Normal
                 ? std::vector<uint64_t>{value.batchCount, value.k, value.n}
                 : std::vector<uint64_t>{value.batchCount, value.n, value.k})
          : (value.rhsOrientation == TargetGemmOrientation::Normal
                 ? std::vector<uint64_t>{value.k, value.n}
                 : std::vector<uint64_t>{value.n, value.k});
  std::vector<uint64_t> destinationShape =
      batched ? std::vector<uint64_t>{value.batchCount, value.m, value.n}
              : std::vector<uint64_t>{value.m, value.n};
  llvm::Expected<PhysicalTensorDescriptor> lhsKey =
      PhysicalTensorDescriptor::create(value.format, layout,
                                       std::move(lhsShape));
  llvm::Expected<PhysicalTensorDescriptor> rhsKey =
      PhysicalTensorDescriptor::create(value.format, layout,
                                       std::move(rhsShape));
  llvm::Expected<PhysicalTensorDescriptor> destinationKey =
      PhysicalTensorDescriptor::create(value.format, layout,
                                       std::move(destinationShape));
  if (!lhsKey || !rhsKey || !destinationKey) {
    llvm::Error errors = llvm::Error::success();
    if (!lhsKey)
      errors = llvm::joinErrors(std::move(errors), lhsKey.takeError());
    if (!rhsKey)
      errors = llvm::joinErrors(std::move(errors), rhsKey.takeError());
    if (!destinationKey)
      errors = llvm::joinErrors(std::move(errors), destinationKey.takeError());
    return kernelError(TargetModelKernelErrorCode::FormalNumericFailure,
                       llvm::toString(std::move(errors)));
  }
  const uint64_t rank = lhsKey->getShape().size();
  llvm::Expected<FormalGemmGeometry> axes =
      getCanonicalFormalGemmGeometry(rank);
  if (!axes)
    return kernelError(TargetModelKernelErrorCode::FormalNumericFailure,
                       llvm::toString(axes.takeError()));
  llvm::Expected<FormalGemmOperation> operation = createFormalGemmOperation(
      *lhsKey, *rhsKey, *destinationKey, value.m, value.k, value.n,
      value.batchCount, *axes, value.lhsOrientation, value.rhsOrientation);
  if (!operation)
    return kernelError(TargetModelKernelErrorCode::FormalNumericFailure,
                       llvm::toString(operation.takeError()));

  if (policy.getGemmDispatchPolicy() ==
          TargetModelGemmDispatchPolicy::BulkThenFormal &&
      value.lhsOrientation == TargetGemmOrientation::Normal &&
      value.rhsOrientation == TargetGemmOrientation::Normal) {
    const TargetModelBulkBackend *backend = policy.getBulkBackend();
    if (!backend)
      return kernelError(TargetModelKernelErrorCode::BulkBackendUnavailable,
                         "bulk-then-formal policy has no bulk backend");
    llvm::Expected<std::vector<uint8_t>> lhsStorage = readTensorStorage(
        memory, command.launchSlotId.getValue(), value.lhs, *lhsKey);
    llvm::Expected<std::vector<uint8_t>> rhsStorage = readTensorStorage(
        memory, command.launchSlotId.getValue(), value.rhs, *rhsKey);
    llvm::Expected<std::vector<uint8_t>> destinationStorage =
        readTensorStorage(memory, command.launchSlotId.getValue(),
                          value.destination, *destinationKey);
    if (!lhsStorage)
      return lhsStorage.takeError();
    if (!rhsStorage)
      return rhsStorage.takeError();
    if (!destinationStorage)
      return destinationStorage.takeError();
    TargetModelGemmRequest request{
        *operation,
        {{{*lhsKey, std::move(*lhsStorage)}, {*rhsKey, std::move(*rhsStorage)}},
         {*destinationKey, std::move(*destinationStorage)}}};
    llvm::Expected<std::optional<TargetModelBulkResult>> bulk =
        backend->tryExecute(request);
    if (!bulk)
      return kernelError(TargetModelKernelErrorCode::BulkBackendFailure,
                         llvm::toString(bulk.takeError()));
    if (*bulk) {
      TargetModelBulkResult result = std::move(**bulk);
      if (result.destination.key != *destinationKey ||
          result.destination.storage.size() !=
              request.tensors.destinationTemplate.storage.size())
        return kernelError(TargetModelKernelErrorCode::BulkBackendFailure,
                           "bulk backend returned a mismatched destination");
      if (result.evidence.matmulInvocations != 1 ||
          result.evidence.reorderInvocations > 1 ||
          result.evidence.formalFusedMultiplyAdds != 0 ||
          result.evidence.evidenceKind == TargetModelBulkEvidenceKind::None ||
          result.evidence.evidenceDigest.empty() ||
          result.evidence.implementation.empty())
        return kernelError(
            TargetModelKernelErrorCode::BulkBackendFailure,
            "bulk backend returned incomplete dispatch evidence");
      return withReads(
          TargetModelCommandEffect{
              {TargetModelByteWrite{command.launchSlotId.getValue(),
                                    TargetModelAddressSpace::TileSPM,
                                    value.destination, 1,
                                    std::move(result.destination.storage)}},
              result.flags,
              TargetModelControlAction::None,
              TargetModelNumericBackend::Bulk,
              std::move(result.evidence)},
          {makeTensorRead(command.launchSlotId.getValue(), value.lhs, *lhsKey),
           makeTensorRead(command.launchSlotId.getValue(), value.rhs,
                          *rhsKey)});
    }

    uint64_t outputCount = 0;
    uint64_t fusedMultiplyAdds = 0;
    uint64_t scalarEvaluations = 0;
    if (!checkedMultiply(value.batchCount, value.m, outputCount) ||
        !checkedMultiply(outputCount, value.n, outputCount) ||
        !checkedMultiply(outputCount, value.k, fusedMultiplyAdds) ||
        !checkedAdd(fusedMultiplyAdds, outputCount, scalarEvaluations))
      return kernelError(TargetModelKernelErrorCode::InvalidCommandField,
                         "GEMM formal work count overflows");
    const FormalNumericWorkBudget formalBudget = budget.getNumericBudget();
    if (scalarEvaluations > formalBudget.getMaximumScalarEvaluations() ||
        fusedMultiplyAdds > formalBudget.getMaximumFusedMultiplyAdds())
      return kernelError(
          TargetModelKernelErrorCode::BulkBackendUnavailable,
          "GEMM exceeds the formal budget and has no exact bulk qualification");
  }

  llvm::Expected<std::vector<RawLogicalValue>> lhs =
      readTensor(memory, command.launchSlotId.getValue(), value.lhs, *lhsKey);
  llvm::Expected<std::vector<RawLogicalValue>> rhs =
      readTensor(memory, command.launchSlotId.getValue(), value.rhs, *rhsKey);
  if (!lhs)
    return lhs.takeError();
  if (!rhs)
    return rhs.takeError();
  std::vector<llvm::ArrayRef<RawLogicalValue>> views{*lhs, *rhs};
  FormalNumericExecutionContext localContext;
  llvm::Expected<FormalTensorNumericResult> result = executeFormalTensorNumeric(
      localContext, *operation, views, budget.getNumericBudget());
  if (!result)
    return kernelError(TargetModelKernelErrorCode::FormalNumericFailure,
                       llvm::toString(result.takeError()));
  llvm::Expected<std::vector<uint8_t>> packed =
      packTensor(memory, command.launchSlotId.getValue(), value.destination,
                 *destinationKey, result->values);
  if (!packed)
    return packed.takeError();
  return withReads(
      TargetModelCommandEffect{
          {TargetModelByteWrite{command.launchSlotId.getValue(),
                                TargetModelAddressSpace::TileSPM,
                                value.destination, 1, std::move(*packed)}},
          result->flags,
          TargetModelControlAction::None,
          TargetModelNumericBackend::Formal},
      {makeTensorRead(command.launchSlotId.getValue(), value.lhs, *lhsKey),
       makeTensorRead(command.launchSlotId.getValue(), value.rhs, *rhsKey)});
}

llvm::Expected<TargetModelCommandEffect>
executeMemset(const compiler::TargetCommand &command,
              const target::TargetMemsetCommand &value,
              const InvocationMemoryRegistry &memory) {
  llvm::Expected<PhysicalTensorDescriptor> key =
      makeTensor(value.format, {value.elementCount});
  if (!key)
    return key.takeError();
  const LogicalScalarCodecPolicy policy = getTargetTensorScalarCodecPolicy();
  llvm::Expected<RawLogicalValue> scalar = makeRawLogicalValue(
      value.format, value.value, policy.nonCanonicalEncoding);
  if (!scalar)
    return kernelError(TargetModelKernelErrorCode::InvalidCommandField,
                       llvm::toString(scalar.takeError()));
  std::vector<RawLogicalValue> values(value.elementCount, *scalar);
  llvm::Expected<std::vector<uint8_t>> packed = packTensor(
      memory, command.launchSlotId.getValue(), value.destination, *key, values);
  if (!packed)
    return packed.takeError();
  return TargetModelCommandEffect{
      {TargetModelByteWrite{command.launchSlotId.getValue(),
                            TargetModelAddressSpace::TileSPM, value.destination,
                            1, std::move(*packed)}},
      {},
      TargetModelControlAction::None};
}

} // namespace wafer::model::kernel_detail
