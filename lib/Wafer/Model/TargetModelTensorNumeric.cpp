//===- TargetModelTensorNumeric.cpp - Tensor numeric kernels ---------===//

#include "TargetModelKernelInternal.h"

#include "Wafer/Target/PhysicalTensorCodec.h"
#include "Wafer/Target/TargetFormat.h"

#include "llvm/ADT/STLExtras.h"

#include <optional>
#include <utility>
#include <vector>

namespace wafer::model::kernel_detail {
namespace {

llvm::Expected<NumericElementwiseOperation>
getNumericOperation(InstrElementwiseKind kind) {
#define WAFER_ELEMENTWISE_CASE(NAME)                                           \
  case InstrElementwiseKind::NAME:                                             \
    return NumericElementwiseOperation::NAME
  switch (kind) {
    WAFER_ELEMENTWISE_CASE(Abs);
    WAFER_ELEMENTWISE_CASE(Recip);
    WAFER_ELEMENTWISE_CASE(Square);
    WAFER_ELEMENTWISE_CASE(Sqrt);
    WAFER_ELEMENTWISE_CASE(Rsqrt);
    WAFER_ELEMENTWISE_CASE(Neg);
    WAFER_ELEMENTWISE_CASE(Max);
    WAFER_ELEMENTWISE_CASE(Min);
    WAFER_ELEMENTWISE_CASE(Add);
    WAFER_ELEMENTWISE_CASE(Sub);
    WAFER_ELEMENTWISE_CASE(Mul);
    WAFER_ELEMENTWISE_CASE(Div);
    WAFER_ELEMENTWISE_CASE(Eq);
    WAFER_ELEMENTWISE_CASE(Ne);
    WAFER_ELEMENTWISE_CASE(Ge);
    WAFER_ELEMENTWISE_CASE(Gt);
    WAFER_ELEMENTWISE_CASE(Le);
    WAFER_ELEMENTWISE_CASE(Lt);
    WAFER_ELEMENTWISE_CASE(LogicNot);
    WAFER_ELEMENTWISE_CASE(LogicAnd);
    WAFER_ELEMENTWISE_CASE(LogicOr);
    WAFER_ELEMENTWISE_CASE(LogicXor);
    WAFER_ELEMENTWISE_CASE(Log2);
    WAFER_ELEMENTWISE_CASE(Ln);
    WAFER_ELEMENTWISE_CASE(Pow2);
    WAFER_ELEMENTWISE_CASE(Exp);
    WAFER_ELEMENTWISE_CASE(ExpLp);
    WAFER_ELEMENTWISE_CASE(Sin);
    WAFER_ELEMENTWISE_CASE(Cos);
    WAFER_ELEMENTWISE_CASE(Tanh);
    WAFER_ELEMENTWISE_CASE(Sigmoid);
    WAFER_ELEMENTWISE_CASE(Relu);
    WAFER_ELEMENTWISE_CASE(SatRelu);
    WAFER_ELEMENTWISE_CASE(LeakyRelu);
    WAFER_ELEMENTWISE_CASE(Softplus);
  }
#undef WAFER_ELEMENTWISE_CASE
  return kernelError(TargetModelKernelErrorCode::InvalidTransactionField,
                     "elementwise kind has no numeric operation mapping");
}

llvm::Expected<NumericReduceOperation>
getNumericOperation(InstrReduceKind kind) {
  switch (kind) {
  case InstrReduceKind::Sum:
    return NumericReduceOperation::Sum;
  case InstrReduceKind::Max:
    return NumericReduceOperation::Max;
  case InstrReduceKind::Min:
    return NumericReduceOperation::Min;
  case InstrReduceKind::Avg:
    return NumericReduceOperation::Avg;
  }
  return kernelError(TargetModelKernelErrorCode::InvalidTransactionField,
                     "reduce kind has no numeric operation mapping");
}

NumericTensorLayout getCountLayout(LogicalFormat format) {
  return format == LogicalFormat::Bool ? NumericTensorLayout::Tensor
                                       : NumericTensorLayout::Cx;
}

llvm::Expected<NumericTensorKey> makeTensor(LogicalFormat format,
                                            std::vector<uint64_t> shape) {
  llvm::Expected<NumericTensorKey> key = NumericTensorKey::create(
      format, getCountLayout(format), std::move(shape));
  if (!key)
    return kernelError(TargetModelKernelErrorCode::NumericResolutionFailure,
                       llvm::toString(key.takeError()));
  return key;
}

llvm::Expected<std::vector<RawLogicalValue>>
readTensor(const InvocationMemoryRegistry &memory, int64_t rank,
           uint64_t address, const NumericTensorKey &key) {
  llvm::Expected<uint64_t> bytes = getPhysicalTensorStorageBytes(key);
  if (!bytes)
    return kernelError(TargetModelKernelErrorCode::PhysicalCodecFailure,
                       llvm::toString(bytes.takeError()));
  llvm::Expected<std::vector<uint8_t>> storage = readSnapshot(
      memory, rank, TargetModelAddressSpace::RankSPM, address, *bytes);
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
readTensorStorage(const InvocationMemoryRegistry &memory, int64_t rank,
                  uint64_t address, const NumericTensorKey &key) {
  llvm::Expected<uint64_t> bytes = getPhysicalTensorStorageBytes(key);
  if (!bytes)
    return kernelError(TargetModelKernelErrorCode::PhysicalCodecFailure,
                       llvm::toString(bytes.takeError()));
  return readSnapshot(memory, rank, TargetModelAddressSpace::RankSPM, address,
                      *bytes);
}

llvm::Expected<std::vector<uint8_t>>
packTensor(const InvocationMemoryRegistry &memory, int64_t rank,
           uint64_t address, const NumericTensorKey &key,
           llvm::ArrayRef<RawLogicalValue> values) {
  llvm::Expected<uint64_t> bytes = getPhysicalTensorStorageBytes(key);
  if (!bytes)
    return kernelError(TargetModelKernelErrorCode::PhysicalCodecFailure,
                       llvm::toString(bytes.takeError()));
  llvm::Expected<std::vector<uint8_t>> storage = readSnapshot(
      memory, rank, TargetModelAddressSpace::RankSPM, address, *bytes);
  if (!storage)
    return storage.takeError();
  llvm::Expected<std::vector<uint8_t>> packed =
      packPhysicalTensorLogicalValues(key, values, *storage);
  if (!packed)
    return kernelError(TargetModelKernelErrorCode::PhysicalCodecFailure,
                       llvm::toString(packed.takeError()));
  return packed;
}

llvm::Expected<ResolvedNumericCommand>
resolveNumeric(NumericCommandKey command) {
  llvm::Expected<ResolvedNumericCommand> resolved = resolveNumericCommand(
      ModelProfileId::formalDeterministicV1(), std::move(command));
  if (!resolved)
    return kernelError(TargetModelKernelErrorCode::NumericResolutionFailure,
                       llvm::toString(resolved.takeError()));
  return resolved;
}

TargetProfileId
getNumericCompatibilityProfile(const InvocationMemoryRegistry &memory) {
  return getTargetProfileRecord(memory.getAddressPlan().getTargetProfile())
      .numericCompatibilityProfile;
}

struct ManagedReferenceInput {
  uint64_t address;
  const NumericTensorKey *key;
};

llvm::Expected<std::optional<TargetModelCommandEffect>>
tryExecuteManagedReference(
    const compiler::TargetTransaction &transaction,
    const InvocationMemoryRegistry &memory,
    const ResolvedNumericCommand &resolved,
    llvm::ArrayRef<ManagedReferenceInput> inputDescriptors,
    uint64_t destinationAddress, const NumericTensorKey &destinationKey,
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
  for (const ManagedReferenceInput &input : inputDescriptors) {
    llvm::Expected<std::vector<uint8_t>> storage = readTensorStorage(
        memory, transaction.logicalRank, input.address, *input.key);
    if (!storage)
      return storage.takeError();
    inputs.push_back({*input.key, std::move(*storage)});
  }
  llvm::Expected<std::vector<uint8_t>> destinationStorage = readTensorStorage(
      memory, transaction.logicalRank, destinationAddress, destinationKey);
  if (!destinationStorage)
    return destinationStorage.takeError();
  const size_t expectedStorageSize = destinationStorage->size();
  TargetModelNumericRequest request{
      resolved,
      std::move(inputs),
      {destinationKey, std::move(*destinationStorage)}};
  llvm::Expected<TargetModelManagedReferenceResult> result =
      backend->execute(request, budget.getNumericBudget());
  if (!result)
    return kernelError(
        TargetModelKernelErrorCode::ManagedReferenceBackendFailure,
        llvm::toString(result.takeError()));

  uint64_t expectedScalarEvaluations = 0;
  switch (resolved.getFamily()) {
  case NumericCommandFamily::CTConvert:
    expectedScalarEvaluations = destinationKey.getElementCount();
    break;
  case NumericCommandFamily::CTElementwise:
    expectedScalarEvaluations = destinationKey.getElementCount();
    break;
  case NumericCommandFamily::NativeCTReduce:
    expectedScalarEvaluations =
        resolved.getCommandKey().getNativeCTReduce()->input.getElementCount();
    break;
  case NumericCommandFamily::NEGemm:
    return kernelError(
        TargetModelKernelErrorCode::ManagedReferenceBackendFailure,
        "GEMM reached the non-GEMM managed-reference dispatch seam");
  }
  if (result->destination.key != destinationKey ||
      result->destination.storage.size() != expectedStorageSize ||
      result->evidence.scalarEvaluations != expectedScalarEvaluations ||
      result->evidence.environmentDigest.empty() ||
      result->evidence.implementation.empty())
    return kernelError(
        TargetModelKernelErrorCode::ManagedReferenceBackendFailure,
        "managed-reference backend returned incomplete or mismatched evidence");
  return std::optional<TargetModelCommandEffect>(TargetModelCommandEffect{
      {TargetModelByteWrite{
          transaction.logicalRank, TargetModelAddressSpace::RankSPM,
          destinationAddress, 1, std::move(result->destination.storage)}},
      result->flags,
      TargetModelControlAction::None,
      TargetModelNumericBackend::ManagedReference,
      {},
      std::move(result->evidence)});
}

} // namespace

llvm::Expected<TargetModelCommandEffect>
executeElementwise(const compiler::TargetTransaction &transaction,
                   const compiler::TargetElementwiseTransaction &value,
                   const InvocationMemoryRegistry &memory,
                   TargetModelKernelBudget budget,
                   TargetModelExecutionPolicy policy) {
  llvm::Expected<NumericElementwiseOperation> operation =
      getNumericOperation(value.kind);
  if (!operation)
    return operation.takeError();
  const LogicalFormat destinationFormat =
      isNumericElementwiseRelation(*operation) ? LogicalFormat::Bool
                                               : value.format;
  llvm::Expected<NumericTensorKey> inputKey =
      makeTensor(value.format, {value.elementCount});
  llvm::Expected<NumericTensorKey> destinationKey =
      makeTensor(destinationFormat, {value.elementCount});
  if (!inputKey)
    return inputKey.takeError();
  if (!destinationKey)
    return destinationKey.takeError();
  std::vector<NumericTensorKey> inputKeys;
  inputKeys.push_back(*inputKey);
  if (value.rhs)
    inputKeys.push_back(*inputKey);
  llvm::Expected<NumericCommandKey> command =
      NumericCommandKey::createCTElementwise(
          getNumericCompatibilityProfile(memory), *operation,
          std::move(inputKeys), *destinationKey);
  if (!command)
    return kernelError(TargetModelKernelErrorCode::NumericResolutionFailure,
                       llvm::toString(command.takeError()));
  llvm::Expected<ResolvedNumericCommand> resolved =
      resolveNumeric(std::move(*command));
  if (!resolved)
    return resolved.takeError();

  std::vector<ManagedReferenceInput> managedInputs{{value.lhs, &*inputKey}};
  if (value.rhs)
    managedInputs.push_back({*value.rhs, &*inputKey});
  llvm::Expected<std::optional<TargetModelCommandEffect>> managed =
      tryExecuteManagedReference(transaction, memory, *resolved, managedInputs,
                                 value.destination, *destinationKey, budget,
                                 policy);
  if (!managed)
    return managed.takeError();
  if (*managed)
    return std::move(**managed);

  std::vector<std::vector<RawLogicalValue>> inputs;
  llvm::Expected<std::vector<RawLogicalValue>> lhs =
      readTensor(memory, transaction.logicalRank, value.lhs, *inputKey);
  if (!lhs)
    return lhs.takeError();
  inputs.push_back(std::move(*lhs));
  if (value.rhs) {
    llvm::Expected<std::vector<RawLogicalValue>> rhs =
        readTensor(memory, transaction.logicalRank, *value.rhs, *inputKey);
    if (!rhs)
      return rhs.takeError();
    inputs.push_back(std::move(*rhs));
  }
  std::vector<llvm::ArrayRef<RawLogicalValue>> views;
  for (const std::vector<RawLogicalValue> &input : inputs)
    views.push_back(input);
  FormalNumericExecutionContext localContext;
  llvm::Expected<FormalTensorNumericResult> result = executeFormalTensorNumeric(
      localContext, *resolved, views, budget.getNumericBudget());
  if (!result)
    return kernelError(TargetModelKernelErrorCode::NumericResolutionFailure,
                       llvm::toString(result.takeError()));
  llvm::Expected<std::vector<uint8_t>> packed =
      packTensor(memory, transaction.logicalRank, value.destination,
                 *destinationKey, result->values);
  if (!packed)
    return packed.takeError();
  return TargetModelCommandEffect{
      {TargetModelByteWrite{transaction.logicalRank,
                            TargetModelAddressSpace::RankSPM, value.destination,
                            1, std::move(*packed)}},
      result->flags,
      TargetModelControlAction::None,
      TargetModelNumericBackend::Formal};
}

llvm::Expected<TargetModelCommandEffect>
executeConvert(const compiler::TargetTransaction &transaction,
               const compiler::TargetConvertTransaction &value,
               const InvocationMemoryRegistry &memory,
               TargetModelKernelBudget budget,
               TargetModelExecutionPolicy policy) {
  const uint16_t opcode = static_cast<uint16_t>(value.kind);
  const TargetConvertRoute *route = findTargetConvertRoute(
      memory.getAddressPlan().getTargetProfile(), opcode);
  if (!route)
    return kernelError(TargetModelKernelErrorCode::NumericResolutionFailure,
                       "convert route is not registered for target profile");
  llvm::Expected<NumericTensorKey> sourceKey =
      makeTensor(route->source, {value.elementCount});
  llvm::Expected<NumericTensorKey> destinationKey =
      makeTensor(route->destination, {value.elementCount});
  if (!sourceKey)
    return sourceKey.takeError();
  if (!destinationKey)
    return destinationKey.takeError();
  std::optional<NumericConvertParameter> parameter;
  if (value.zeroPoint)
    parameter = NumericConvertParameter::zeroPoint(*value.zeroPoint);
  if (value.roundingMode) {
    llvm::Expected<NumericRoundingMode> mode =
        parseNumericRoundingMode(static_cast<uint8_t>(*value.roundingMode));
    if (!mode)
      return kernelError(TargetModelKernelErrorCode::NumericResolutionFailure,
                         llvm::toString(mode.takeError()));
    parameter = NumericConvertParameter::roundingMode(*mode);
  }
  llvm::Expected<NumericCommandKey> command =
      NumericCommandKey::createCTConvert(getNumericCompatibilityProfile(memory),
                                         opcode, *sourceKey, *destinationKey,
                                         parameter);
  if (!command)
    return kernelError(TargetModelKernelErrorCode::NumericResolutionFailure,
                       llvm::toString(command.takeError()));
  llvm::Expected<ResolvedNumericCommand> resolved =
      resolveNumeric(std::move(*command));
  if (!resolved)
    return resolved.takeError();
  const ManagedReferenceInput managedInput{value.source, &*sourceKey};
  llvm::Expected<std::optional<TargetModelCommandEffect>> managed =
      tryExecuteManagedReference(transaction, memory, *resolved, managedInput,
                                 value.destination, *destinationKey, budget,
                                 policy);
  if (!managed)
    return managed.takeError();
  if (*managed)
    return std::move(**managed);
  llvm::Expected<std::vector<RawLogicalValue>> source =
      readTensor(memory, transaction.logicalRank, value.source, *sourceKey);
  if (!source)
    return source.takeError();
  std::vector<llvm::ArrayRef<RawLogicalValue>> views{*source};
  FormalNumericExecutionContext localContext;
  llvm::Expected<FormalTensorNumericResult> result = executeFormalTensorNumeric(
      localContext, *resolved, views, budget.getNumericBudget());
  if (!result)
    return kernelError(TargetModelKernelErrorCode::NumericResolutionFailure,
                       llvm::toString(result.takeError()));
  llvm::Expected<std::vector<uint8_t>> packed =
      packTensor(memory, transaction.logicalRank, value.destination,
                 *destinationKey, result->values);
  if (!packed)
    return packed.takeError();
  return TargetModelCommandEffect{
      {TargetModelByteWrite{transaction.logicalRank,
                            TargetModelAddressSpace::RankSPM, value.destination,
                            1, std::move(*packed)}},
      result->flags,
      TargetModelControlAction::None,
      TargetModelNumericBackend::Formal};
}

llvm::Expected<TargetModelCommandEffect>
executeReduce(const compiler::TargetTransaction &transaction,
              const compiler::TargetReduceTransaction &value,
              const InvocationMemoryRegistry &memory,
              TargetModelKernelBudget budget,
              TargetModelExecutionPolicy policy) {
  llvm::Expected<NumericReduceOperation> operation =
      getNumericOperation(value.kind);
  if (!operation)
    return operation.takeError();
  const auto dimension = static_cast<NativeCTReduceDimension>(value.dimension);
  std::vector<uint64_t> inputShape(value.nhwc.begin(), value.nhwc.end());
  const std::vector<size_t> reducedDimensions =
      getNativeCTReduceLogicalDimensions(dimension, inputShape.size());
  if (reducedDimensions.empty())
    return kernelError(TargetModelKernelErrorCode::InvalidTransactionField,
                       "reduce dimension is invalid for the fixed ABI shape");
  std::vector<uint64_t> destinationShape;
  for (size_t index = 0; index < inputShape.size(); ++index)
    if (!llvm::is_contained(reducedDimensions, index))
      destinationShape.push_back(inputShape[index]);
  const NumericTensorLayout destinationLayout = destinationShape.size() > 2
                                                    ? NumericTensorLayout::NCx
                                                    : NumericTensorLayout::Cx;
  llvm::Expected<NumericTensorKey> inputKey = NumericTensorKey::create(
      value.format, NumericTensorLayout::NCx, std::move(inputShape));
  llvm::Expected<NumericTensorKey> destinationKey = NumericTensorKey::create(
      value.format, destinationLayout, std::move(destinationShape));
  if (!inputKey || !destinationKey) {
    llvm::Error errors = llvm::Error::success();
    if (!inputKey)
      errors = llvm::joinErrors(std::move(errors), inputKey.takeError());
    if (!destinationKey)
      errors = llvm::joinErrors(std::move(errors), destinationKey.takeError());
    return kernelError(TargetModelKernelErrorCode::NumericResolutionFailure,
                       llvm::toString(std::move(errors)));
  }
  llvm::Expected<NumericCommandKey> command =
      NumericCommandKey::createNativeCTReduce(
          getNumericCompatibilityProfile(memory), *operation, *inputKey,
          *destinationKey, dimension);
  if (!command)
    return kernelError(TargetModelKernelErrorCode::NumericResolutionFailure,
                       llvm::toString(command.takeError()));
  llvm::Expected<ResolvedNumericCommand> resolved =
      resolveNumeric(std::move(*command));
  if (!resolved)
    return resolved.takeError();
  const ManagedReferenceInput managedInput{value.source, &*inputKey};
  llvm::Expected<std::optional<TargetModelCommandEffect>> managed =
      tryExecuteManagedReference(transaction, memory, *resolved, managedInput,
                                 value.destination, *destinationKey, budget,
                                 policy);
  if (!managed)
    return managed.takeError();
  if (*managed)
    return std::move(**managed);
  llvm::Expected<std::vector<RawLogicalValue>> input =
      readTensor(memory, transaction.logicalRank, value.source, *inputKey);
  if (!input)
    return input.takeError();
  std::vector<llvm::ArrayRef<RawLogicalValue>> views{*input};
  FormalNumericExecutionContext localContext;
  llvm::Expected<FormalTensorNumericResult> result = executeFormalTensorNumeric(
      localContext, *resolved, views, budget.getNumericBudget());
  if (!result)
    return kernelError(TargetModelKernelErrorCode::NumericResolutionFailure,
                       llvm::toString(result.takeError()));
  llvm::Expected<std::vector<uint8_t>> packed =
      packTensor(memory, transaction.logicalRank, value.destination,
                 *destinationKey, result->values);
  if (!packed)
    return packed.takeError();
  return TargetModelCommandEffect{
      {TargetModelByteWrite{transaction.logicalRank,
                            TargetModelAddressSpace::RankSPM, value.destination,
                            1, std::move(*packed)}},
      result->flags,
      TargetModelControlAction::None,
      TargetModelNumericBackend::Formal};
}

llvm::Expected<TargetModelCommandEffect>
executeGemm(const compiler::TargetTransaction &transaction,
            const compiler::TargetGemmTransaction &value,
            const InvocationMemoryRegistry &memory,
            TargetModelKernelBudget budget, TargetModelExecutionPolicy policy) {
  const bool batched = value.batchCount > 1;
  // The target call has no free layout field. Its storage contract is Cx for
  // rank-2 GEMM and NCx (including per-batch bank alignment) for batched GEMM.
  const NumericTensorLayout layout =
      batched ? NumericTensorLayout::NCx : NumericTensorLayout::Cx;
  std::vector<uint64_t> lhsShape =
      batched
          ? (value.lhsOrientation == GemmOrientation::Normal
                 ? std::vector<uint64_t>{value.batchCount, value.m, value.k}
                 : std::vector<uint64_t>{value.batchCount, value.k, value.m})
          : (value.lhsOrientation == GemmOrientation::Normal
                 ? std::vector<uint64_t>{value.m, value.k}
                 : std::vector<uint64_t>{value.k, value.m});
  std::vector<uint64_t> rhsShape =
      batched
          ? (value.rhsOrientation == GemmOrientation::Normal
                 ? std::vector<uint64_t>{value.batchCount, value.k, value.n}
                 : std::vector<uint64_t>{value.batchCount, value.n, value.k})
          : (value.rhsOrientation == GemmOrientation::Normal
                 ? std::vector<uint64_t>{value.k, value.n}
                 : std::vector<uint64_t>{value.n, value.k});
  std::vector<uint64_t> destinationShape =
      batched ? std::vector<uint64_t>{value.batchCount, value.m, value.n}
              : std::vector<uint64_t>{value.m, value.n};
  llvm::Expected<NumericTensorKey> lhsKey =
      NumericTensorKey::create(value.format, layout, std::move(lhsShape));
  llvm::Expected<NumericTensorKey> rhsKey =
      NumericTensorKey::create(value.format, layout, std::move(rhsShape));
  llvm::Expected<NumericTensorKey> destinationKey = NumericTensorKey::create(
      value.format, layout, std::move(destinationShape));
  if (!lhsKey || !rhsKey || !destinationKey) {
    llvm::Error errors = llvm::Error::success();
    if (!lhsKey)
      errors = llvm::joinErrors(std::move(errors), lhsKey.takeError());
    if (!rhsKey)
      errors = llvm::joinErrors(std::move(errors), rhsKey.takeError());
    if (!destinationKey)
      errors = llvm::joinErrors(std::move(errors), destinationKey.takeError());
    return kernelError(TargetModelKernelErrorCode::NumericResolutionFailure,
                       llvm::toString(std::move(errors)));
  }
  const uint64_t rank = lhsKey->getShape().size();
  llvm::Expected<NumericGemmAxes> axes = getCanonicalNumericGemmAxes(rank);
  if (!axes)
    return kernelError(TargetModelKernelErrorCode::NumericResolutionFailure,
                       llvm::toString(axes.takeError()));
  llvm::Expected<NumericCommandKey> command = NumericCommandKey::createNEGemm(
      getNumericCompatibilityProfile(memory), *lhsKey, *rhsKey, *destinationKey,
      value.m, value.k, value.n, value.batchCount, *axes, value.lhsOrientation,
      value.rhsOrientation);
  if (!command)
    return kernelError(TargetModelKernelErrorCode::NumericResolutionFailure,
                       llvm::toString(command.takeError()));
  llvm::Expected<ResolvedNumericCommand> resolved =
      resolveNumeric(std::move(*command));
  if (!resolved)
    return resolved.takeError();

  if (policy.getGemmDispatchPolicy() ==
          TargetModelGemmDispatchPolicy::PreferAdmitted &&
      value.lhsOrientation == GemmOrientation::Normal &&
      value.rhsOrientation == GemmOrientation::Normal) {
    const TargetModelBulkBackend *backend = policy.getBulkBackend();
    if (!backend)
      return kernelError(TargetModelKernelErrorCode::BulkBackendUnavailable,
                         "prefer-admitted policy has no bulk backend");
    llvm::Expected<std::vector<uint8_t>> lhsStorage =
        readTensorStorage(memory, transaction.logicalRank, value.lhs, *lhsKey);
    llvm::Expected<std::vector<uint8_t>> rhsStorage =
        readTensorStorage(memory, transaction.logicalRank, value.rhs, *rhsKey);
    llvm::Expected<std::vector<uint8_t>> destinationStorage = readTensorStorage(
        memory, transaction.logicalRank, value.destination, *destinationKey);
    if (!lhsStorage)
      return lhsStorage.takeError();
    if (!rhsStorage)
      return rhsStorage.takeError();
    if (!destinationStorage)
      return destinationStorage.takeError();
    TargetModelNumericRequest request{
        *resolved,
        {{*lhsKey, std::move(*lhsStorage)}, {*rhsKey, std::move(*rhsStorage)}},
        {*destinationKey, std::move(*destinationStorage)}};
    llvm::Expected<std::optional<TargetModelBulkResult>> bulk =
        backend->tryExecute(request);
    if (!bulk)
      return kernelError(TargetModelKernelErrorCode::BulkBackendFailure,
                         llvm::toString(bulk.takeError()));
    if (*bulk) {
      TargetModelBulkResult result = std::move(**bulk);
      if (result.destination.key != *destinationKey ||
          result.destination.storage.size() !=
              request.destinationTemplate.storage.size())
        return kernelError(TargetModelKernelErrorCode::BulkBackendFailure,
                           "bulk backend returned a mismatched destination");
      if (result.evidence.matmulInvocations != 1 ||
          result.evidence.reorderInvocations > 1 ||
          result.evidence.formalFusedMultiplyAdds != 0 ||
          result.evidence.provenanceKind ==
              TargetModelBulkProvenanceKind::None ||
          result.evidence.provenanceDigest.empty() ||
          result.evidence.implementation.empty())
        return kernelError(
            TargetModelKernelErrorCode::BulkBackendFailure,
            "bulk backend returned incomplete dispatch evidence");
      return TargetModelCommandEffect{
          {TargetModelByteWrite{
              transaction.logicalRank, TargetModelAddressSpace::RankSPM,
              value.destination, 1, std::move(result.destination.storage)}},
          result.flags,
          TargetModelControlAction::None,
          TargetModelNumericBackend::Bulk,
          std::move(result.evidence)};
    }

    uint64_t outputCount = 0;
    uint64_t fusedMultiplyAdds = 0;
    uint64_t scalarEvaluations = 0;
    if (!checkedMultiply(value.batchCount, value.m, outputCount) ||
        !checkedMultiply(outputCount, value.n, outputCount) ||
        !checkedMultiply(outputCount, value.k, fusedMultiplyAdds) ||
        !checkedAdd(fusedMultiplyAdds, outputCount, scalarEvaluations))
      return kernelError(TargetModelKernelErrorCode::InvalidTransactionField,
                         "GEMM formal work count overflows");
    const FormalNumericWorkBudget formalBudget = budget.getNumericBudget();
    if (scalarEvaluations > formalBudget.getMaximumScalarEvaluations() ||
        fusedMultiplyAdds > formalBudget.getMaximumFusedMultiplyAdds())
      return kernelError(
          TargetModelKernelErrorCode::BulkBackendUnavailable,
          "GEMM exceeds the formal budget and has no exact bulk admission");
  }

  llvm::Expected<std::vector<RawLogicalValue>> lhs =
      readTensor(memory, transaction.logicalRank, value.lhs, *lhsKey);
  llvm::Expected<std::vector<RawLogicalValue>> rhs =
      readTensor(memory, transaction.logicalRank, value.rhs, *rhsKey);
  if (!lhs)
    return lhs.takeError();
  if (!rhs)
    return rhs.takeError();
  std::vector<llvm::ArrayRef<RawLogicalValue>> views{*lhs, *rhs};
  FormalNumericExecutionContext localContext;
  llvm::Expected<FormalTensorNumericResult> result = executeFormalTensorNumeric(
      localContext, *resolved, views, budget.getNumericBudget());
  if (!result)
    return kernelError(TargetModelKernelErrorCode::NumericResolutionFailure,
                       llvm::toString(result.takeError()));
  llvm::Expected<std::vector<uint8_t>> packed =
      packTensor(memory, transaction.logicalRank, value.destination,
                 *destinationKey, result->values);
  if (!packed)
    return packed.takeError();
  return TargetModelCommandEffect{
      {TargetModelByteWrite{transaction.logicalRank,
                            TargetModelAddressSpace::RankSPM, value.destination,
                            1, std::move(*packed)}},
      result->flags,
      TargetModelControlAction::None,
      TargetModelNumericBackend::Formal};
}

llvm::Expected<TargetModelCommandEffect>
executeMemset(const compiler::TargetTransaction &transaction,
              const compiler::TargetMemsetTransaction &value,
              const InvocationMemoryRegistry &memory) {
  llvm::Expected<NumericTensorKey> key =
      makeTensor(value.format, {value.elementCount});
  if (!key)
    return key.takeError();
  const LogicalScalarCodecPolicy policy =
      getModelProfileRecord(ModelProfileId::formalDeterministicV1())
          .numericEncodePolicy;
  llvm::Expected<RawLogicalValue> scalar = makeRawLogicalValue(
      value.format, value.value, policy.nonCanonicalEncoding);
  if (!scalar)
    return kernelError(TargetModelKernelErrorCode::InvalidTransactionField,
                       llvm::toString(scalar.takeError()));
  std::vector<RawLogicalValue> values(value.elementCount, *scalar);
  llvm::Expected<std::vector<uint8_t>> packed = packTensor(
      memory, transaction.logicalRank, value.destination, *key, values);
  if (!packed)
    return packed.takeError();
  return TargetModelCommandEffect{
      {TargetModelByteWrite{transaction.logicalRank,
                            TargetModelAddressSpace::RankSPM, value.destination,
                            1, std::move(*packed)}},
      {},
      TargetModelControlAction::None};
}

} // namespace wafer::model::kernel_detail
