//===- TargetTensorMaterialization.cpp - Static data conversion ---------===//

#include "Wafer/Target/PhysicalTensor/TargetTensorMaterialization.h"

#include "Wafer/Target/PhysicalTensor/TargetScalarConversion.h"

#include "llvm/Support/Errc.h"

namespace wafer {

llvm::Expected<TargetTensorMaterializationAction>
TargetTensorMaterializationAction::create(
    LogicalFormat source, LogicalFormat destination,
    std::optional<TargetConvertParameter> parameter) {
  if (source == destination) {
    if (parameter)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "identity target tensor materialization cannot carry a parameter");
    return TargetTensorMaterializationAction(
        TargetTensorMaterializationKind::Identity, source, destination,
        std::nullopt, std::nullopt);
  }
  const TargetConvertRoute *route = findTargetConvertRoute(source, destination);
  if (!route)
    return llvm::createStringError(
        llvm::errc::not_supported,
        "target has no value-conversion route from %s to %s",
        stringifyLogicalFormat(source).str().c_str(),
        stringifyLogicalFormat(destination).str().c_str());
  auto operation = TargetConvertOperation::create(route->opcode);
  if (!operation)
    return operation.takeError();
  switch (route->parameterKind) {
  case TargetConvertParameterKind::None:
    if (parameter)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "parameter-free target tensor conversion received a parameter");
    break;
  case TargetConvertParameterKind::RoundingMode:
    if (!parameter || !parameter->getRoundingMode())
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "target tensor conversion requires an explicit rounding mode");
    break;
  case TargetConvertParameterKind::ZeroPoint:
    if (!parameter || !parameter->getZeroPoint())
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "target tensor conversion requires an explicit zero point");
    break;
  }
  return TargetTensorMaterializationAction(
      TargetTensorMaterializationKind::ValueConversion, source, destination,
      *operation, parameter);
}

LogicalScalarCodecPolicy getTargetTensorScalarCodecPolicy() {
  return {LogicalByteOrder::LittleEndian,
          LogicalBitOrder::LeastSignificantBitFirstWithinByte,
          NonCanonicalEncodingPolicy::Reject};
}

llvm::Expected<RawLogicalValue>
materializeTargetTensorValue(const TargetTensorMaterializationAction &action,
                             RawLogicalValue source) {
  if (source.format != action.getSourceFormat())
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "materialization source format mismatch");
  if (action.getKind() == TargetTensorMaterializationKind::Identity)
    return source;
  auto result = convertTargetScalar(*action.getConversion(),
                                    action.getParameter(), source);
  if (!result)
    return result.takeError();
  return result->value;
}

} // namespace wafer
