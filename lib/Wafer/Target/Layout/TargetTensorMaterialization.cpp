//===- TargetTensorMaterialization.cpp - Static data conversion ---------===//

#include "Wafer/Target/Layout/TargetTensorMaterialization.h"

#include "Wafer/Target/Numeric/TargetScalarConversion.h"

#include "llvm/Support/Errc.h"

namespace wafer {

std::optional<LogicalFormat> getTargetLogicalFormat(ProgramElementType type) {
  switch (type) {
  case ProgramElementType::Bool:
    return LogicalFormat::Bool;
  case ProgramElementType::I8:
    return LogicalFormat::I8;
  case ProgramElementType::U8:
    return LogicalFormat::U8;
  case ProgramElementType::I16:
    return LogicalFormat::I16;
  case ProgramElementType::U16:
    return LogicalFormat::U16;
  case ProgramElementType::I32:
    return LogicalFormat::I32;
  case ProgramElementType::U32:
    return LogicalFormat::U32;
  case ProgramElementType::I64:
    return LogicalFormat::I64;
  case ProgramElementType::U64:
    return LogicalFormat::U64;
  case ProgramElementType::F16:
    return LogicalFormat::F16;
  case ProgramElementType::BF16:
    return LogicalFormat::BF16;
  case ProgramElementType::F32:
    return LogicalFormat::F32;
  case ProgramElementType::TF32:
    return LogicalFormat::TF32;
  case ProgramElementType::F64:
    return std::nullopt;
  }
  llvm_unreachable("unknown program element type");
}

ProgramElementType getProgramElementType(LogicalFormat format) {
  switch (format) {
  case LogicalFormat::Bool:
    return ProgramElementType::Bool;
  case LogicalFormat::I8:
    return ProgramElementType::I8;
  case LogicalFormat::U8:
    return ProgramElementType::U8;
  case LogicalFormat::I16:
    return ProgramElementType::I16;
  case LogicalFormat::U16:
    return ProgramElementType::U16;
  case LogicalFormat::I32:
    return ProgramElementType::I32;
  case LogicalFormat::U32:
    return ProgramElementType::U32;
  case LogicalFormat::I64:
    return ProgramElementType::I64;
  case LogicalFormat::U64:
    return ProgramElementType::U64;
  case LogicalFormat::F16:
    return ProgramElementType::F16;
  case LogicalFormat::BF16:
    return ProgramElementType::BF16;
  case LogicalFormat::F32:
    return ProgramElementType::F32;
  case LogicalFormat::TF32:
    return ProgramElementType::TF32;
  }
  llvm_unreachable("unknown target logical format");
}

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
