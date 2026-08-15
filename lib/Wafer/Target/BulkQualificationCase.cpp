//===- BulkQualificationCase.cpp - Qualification case generation ----===//

#include "BulkQualificationInternal.h"

#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace wafer {

using namespace bulk_qualification_detail;

namespace {

uint64_t splitMix64(uint64_t &state) {
  state += UINT64_C(0x9e3779b97f4a7c15);
  uint64_t value = state;
  value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31);
}

RawLogicalValue generatedValue(LogicalFormat format, uint64_t &state) {
  static constexpr uint32_t f32Values[] = {
      UINT32_C(0x00000000), UINT32_C(0x3e800000), UINT32_C(0xbe800000),
      UINT32_C(0x3f000000), UINT32_C(0xbf000000), UINT32_C(0x3f800000),
      UINT32_C(0xbf800000), UINT32_C(0x40000000), UINT32_C(0xc0000000),
  };
  static constexpr uint16_t f16Values[] = {
      UINT16_C(0x0000), UINT16_C(0x3400), UINT16_C(0xb400),
      UINT16_C(0x3800), UINT16_C(0xb800), UINT16_C(0x3c00),
      UINT16_C(0xbc00), UINT16_C(0x4000), UINT16_C(0xc000),
  };
  static constexpr uint16_t bf16Values[] = {
      UINT16_C(0x0000), UINT16_C(0x3e80), UINT16_C(0xbe80),
      UINT16_C(0x3f00), UINT16_C(0xbf00), UINT16_C(0x3f80),
      UINT16_C(0xbf80), UINT16_C(0x4000), UINT16_C(0xc000),
  };
  const size_t index = static_cast<size_t>(splitMix64(state) % 9);
  switch (format) {
  case LogicalFormat::F16:
    return {format, f16Values[index]};
  case LogicalFormat::BF16:
    return {format, bf16Values[index]};
  case LogicalFormat::F32:
    return {format, f32Values[index]};
  default:
    llvm_unreachable("qualification generator received unsupported format");
  }
}

std::vector<RawLogicalValue> generateValues(LogicalFormat format,
                                            uint64_t count, uint64_t &state) {
  std::vector<RawLogicalValue> values;
  values.reserve(static_cast<size_t>(count));
  for (uint64_t index = 0; index < count; ++index)
    values.push_back(generatedValue(format, state));
  return values;
}

} // namespace

llvm::Expected<BulkQualificationCase>
materializeBulkQualificationCase(BulkQualificationSpec spec,
                                 BulkNumericWorkBudget bulkBudget) {
  std::vector<uint64_t> lhsShape;
  std::vector<uint64_t> rhsShape;
  std::vector<uint64_t> destinationShape;
  const uint64_t rank = spec.getBatchCount() == 1 ? 2 : 3;
  if (rank == 3) {
    lhsShape.push_back(spec.getBatchCount());
    rhsShape.push_back(spec.getBatchCount());
    destinationShape.push_back(spec.getBatchCount());
  }
  lhsShape.insert(lhsShape.end(), {spec.getM(), spec.getK()});
  rhsShape.insert(rhsShape.end(), {spec.getK(), spec.getN()});
  destinationShape.insert(destinationShape.end(), {spec.getM(), spec.getN()});
  llvm::Expected<NumericTensorKey> lhs =
      NumericTensorKey::create(spec.getFormat(), spec.getLHSLayout(), lhsShape);
  llvm::Expected<NumericTensorKey> rhs =
      NumericTensorKey::create(spec.getFormat(), spec.getRHSLayout(), rhsShape);
  llvm::Expected<NumericTensorKey> destination = NumericTensorKey::create(
      spec.getFormat(), spec.getDestinationLayout(), destinationShape);
  if (llvm::Error error = takeExpectedErrors(lhs, rhs, destination))
    return error;
  llvm::Expected<NumericGemmAxes> axes = getCanonicalNumericGemmAxes(rank);
  if (!axes)
    return axes.takeError();
  llvm::Expected<NumericCommandKey> key = NumericCommandKey::createNEGemm(
      *lhs, *rhs, *destination, spec.getM(), spec.getK(), spec.getN(),
      spec.getBatchCount(), *axes);
  if (!key)
    return key.takeError();
  llvm::Expected<ResolvedNumericCommand> command = resolveNumericCommand(
      ModelProfileId::formalDeterministic(), std::move(*key));
  if (!command)
    return command.takeError();
  if (!command->isSupported())
    return invalid("qualification spec resolves to an unsupported command");

  llvm::Expected<uint64_t> lhsBytes = getBulkTensorPhysicalBytes(*lhs);
  llvm::Expected<uint64_t> rhsBytes = getBulkTensorPhysicalBytes(*rhs);
  llvm::Expected<uint64_t> destinationBytes =
      getBulkTensorPhysicalBytes(*destination);
  if (llvm::Error error =
          takeExpectedErrors(lhsBytes, rhsBytes, destinationBytes))
    return error;
  if (*lhsBytes > bulkBudget.getMaximumTotalBytes() ||
      *rhsBytes > bulkBudget.getMaximumTotalBytes() - *lhsBytes ||
      *destinationBytes >
          bulkBudget.getMaximumTotalBytes() - *lhsBytes - *rhsBytes)
    return invalid(
        "qualification physical tensors exceed the bulk byte budget");

  llvm::Expected<BulkTensorStorage> lhsStorage = [&]() {
    if (spec.hasExplicitPhysicalPayload())
      return BulkTensorStorage::create(
          *lhs, std::vector<uint8_t>(spec.getLHSPhysicalPayload().begin(),
                                     spec.getLHSPhysicalPayload().end()));
    uint64_t state = spec.getSeed();
    std::vector<RawLogicalValue> values =
        generateValues(spec.getFormat(), lhs->getElementCount(), state);
    return packBulkTensorLogicalValues(*lhs, values, UINT8_C(0xa5));
  }();
  llvm::Expected<BulkTensorStorage> rhsStorage = [&]() {
    if (spec.hasExplicitPhysicalPayload())
      return BulkTensorStorage::create(
          *rhs, std::vector<uint8_t>(spec.getRHSPhysicalPayload().begin(),
                                     spec.getRHSPhysicalPayload().end()));
    uint64_t state = spec.getSeed();
    (void)generateValues(spec.getFormat(), lhs->getElementCount(), state);
    std::vector<RawLogicalValue> values =
        generateValues(spec.getFormat(), rhs->getElementCount(), state);
    return packBulkTensorLogicalValues(*rhs, values, UINT8_C(0xa5));
  }();
  llvm::Expected<BulkTensorStorage> destinationTemplate = [&]() {
    if (spec.hasExplicitPhysicalPayload())
      return BulkTensorStorage::create(
          *destination,
          std::vector<uint8_t>(
              spec.getDestinationTemplatePhysicalPayload().begin(),
              spec.getDestinationTemplatePhysicalPayload().end()));
    std::vector<RawLogicalValue> destinationZeros(
        static_cast<size_t>(destination->getElementCount()),
        RawLogicalValue{spec.getFormat(), 0});
    return packBulkTensorLogicalValues(*destination, destinationZeros,
                                       UINT8_C(0x5a));
  }();
  if (llvm::Error error =
          takeExpectedErrors(lhsStorage, rhsStorage, destinationTemplate))
    return error;
  std::vector<BulkTensorStorage> inputs;
  inputs.push_back(std::move(*lhsStorage));
  inputs.push_back(std::move(*rhsStorage));
  return BulkQualificationCase(std::move(spec), std::move(*command),
                               std::move(inputs),
                               std::move(*destinationTemplate));
}

} // namespace wafer
