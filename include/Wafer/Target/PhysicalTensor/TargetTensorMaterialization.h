//===- TargetTensorMaterialization.h - Static data conversion -*- C++ -*-===//

#ifndef WAFER_TARGET_TARGETTENSORMATERIALIZATION_H
#define WAFER_TARGET_TARGETTENSORMATERIALIZATION_H

#include "Wafer/Frontend/ProgramElementType.h"
#include "Wafer/Target/TargetOperation.h"
#include "Wafer/Target/PhysicalTensor/NumericCodec.h"

#include "llvm/Support/Error.h"

#include <optional>

namespace wafer {

/// Explicit relation between parsed program element types and current target
/// logical formats. Program/source code does not depend on target protocol
/// headers merely to carry its own closed type.
std::optional<LogicalFormat> getTargetLogicalFormat(ProgramElementType type);
ProgramElementType getProgramElementType(LogicalFormat format);

enum class TargetTensorMaterializationKind : uint8_t {
  Identity,
  ValueConversion,
};

class TargetTensorMaterializationAction {
public:
  TargetTensorMaterializationAction() = delete;

  static llvm::Expected<TargetTensorMaterializationAction>
  create(LogicalFormat source, LogicalFormat destination,
         std::optional<TargetConvertParameter> parameter);

  TargetTensorMaterializationKind getKind() const { return kind; }
  LogicalFormat getSourceFormat() const { return source; }
  LogicalFormat getDestinationFormat() const { return destination; }
  std::optional<TargetConvertOperation> getConversion() const {
    return conversion;
  }
  std::optional<TargetConvertParameter> getParameter() const {
    return parameter;
  }

  friend bool operator==(const TargetTensorMaterializationAction &lhs,
                         const TargetTensorMaterializationAction &rhs) {
    return lhs.kind == rhs.kind && lhs.source == rhs.source &&
           lhs.destination == rhs.destination &&
           lhs.conversion == rhs.conversion && lhs.parameter == rhs.parameter;
  }
  friend bool operator!=(const TargetTensorMaterializationAction &lhs,
                         const TargetTensorMaterializationAction &rhs) {
    return !(lhs == rhs);
  }

private:
  TargetTensorMaterializationAction(
      TargetTensorMaterializationKind kind, LogicalFormat source,
      LogicalFormat destination,
      std::optional<TargetConvertOperation> conversion,
      std::optional<TargetConvertParameter> parameter)
      : kind(kind), source(source), destination(destination),
        conversion(conversion), parameter(parameter) {}

  TargetTensorMaterializationKind kind;
  LogicalFormat source;
  LogicalFormat destination;
  std::optional<TargetConvertOperation> conversion;
  std::optional<TargetConvertParameter> parameter;
};

LogicalScalarCodecPolicy getTargetTensorScalarCodecPolicy();

/// Converts one canonical logical value according to an already validated
/// action. Identity preserves the exact value. Failure has no side effect.
llvm::Expected<RawLogicalValue>
materializeTargetTensorValue(const TargetTensorMaterializationAction &action,
                             RawLogicalValue source);

} // namespace wafer

#endif // WAFER_TARGET_TARGETTENSORMATERIALIZATION_H
