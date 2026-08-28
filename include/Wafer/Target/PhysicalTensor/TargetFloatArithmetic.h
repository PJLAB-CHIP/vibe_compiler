//===- TargetFloatArithmetic.h - Shared target float primitives -*- C++ -*-===//

#ifndef WAFER_TARGET_TARGETFLOATARITHMETIC_H
#define WAFER_TARGET_TARGETFLOATARITHMETIC_H

#include "Wafer/Target/TargetOperation.h"
#include "Wafer/Target/PhysicalTensor/NumericCodec.h"

#include "llvm/ADT/APFloat.h"

#include <cstdint>
#include <optional>

namespace wafer::target_numeric_detail {

/// Low-level deterministic floating primitives shared by static TargetTensor
/// conversion and the formal model. They do not select an operation, mutate
/// execution state, or carry model/qualification identity.
const llvm::fltSemantics *getFloatSemantics(LogicalFormat format);
llvm::APFloat decodeFloat(RawLogicalValue value);
std::optional<uint64_t> encodeFloat(const llvm::APFloat &value,
                                    LogicalFormat format);
uint64_t
canonicalPositiveQuietNaNBits(const LogicalFormatDescriptor &descriptor);
bool isNaNClass(LogicalValueClass valueClass);
bool isTinyAfterUnboundedPrecisionRounding(
    RawLogicalValue source, const LogicalFormatDescriptor &sourceDescriptor,
    const LogicalFormatDescriptor &destinationDescriptor,
    TargetRoundingMode mode);

} // namespace wafer::target_numeric_detail

#endif // WAFER_TARGET_TARGETFLOATARITHMETIC_H
