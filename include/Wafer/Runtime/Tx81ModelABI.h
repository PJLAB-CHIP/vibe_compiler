//===- Tx81ModelABI.h - Qualified TX81 model launch wire ABI ---*- C++ -*-===//

#ifndef WAFER_RUNTIME_TX81MODELABI_H
#define WAFER_RUNTIME_TX81MODELABI_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <vector>

namespace wafer::runtime {

/// Digest-qualified TX81 V5.6 model wire layout recovered from the public
/// runtime and matching Kcore firmware. This is deliberately not a generic
/// vendor ABI: callers must opt into this exact layout identity.
inline constexpr llvm::StringLiteral kTx81ModelBootParamV1 =
    "tx81-model-bootparam-v1";

enum class Tx81ModelTensorClass { Input, Output, Parameter };

/// One launch-visible tensor before canonical BootParam ordering. The builder
/// orders tensors by class (input, output, parameter), then logical rank and
/// ABI slot ordinal. Address is the device address returned by the qualified
/// TX allocation provider.
struct Tx81ModelTensorDescriptor {
  Tx81ModelTensorClass tensorClass = Tx81ModelTensorClass::Input;
  int64_t logicalRank = -1;
  uint64_t slotOrdinal = 0;
  uint64_t deviceAddress = 0;
  uint64_t bytes = 0;
  std::string dtype;
  std::vector<int64_t> shape;
};

struct Tx81ModelBootParamImage {
  std::vector<uint8_t> bytes;
  std::vector<Tx81ModelTensorDescriptor> canonicalTensors;
};

/// Builds the host bytes copied to the device before txLaunchModel. The first
/// implementation intentionally accepts only the parameter-free FP32 tensor
/// subset admitted by the closed model BootParam v1 contract. Unsupported
/// tensor contracts fail before any TX call.
llvm::Expected<Tx81ModelBootParamImage> buildTx81ModelBootParam(
    llvm::ArrayRef<Tx81ModelTensorDescriptor> tensors,
    uint64_t dynamicTLVDeviceAddress);

/// Builds the one-module type-7 D_DynMods payload. The module name must be the
/// exact invocation-local name registered by txLoadGraph.
llvm::Expected<std::vector<uint8_t>>
buildTx81DynlibRunModules(llvm::StringRef moduleName);

/// Builds the type-7 D_GraphTLV whose nested device address points at the
/// D_DynMods allocation.
llvm::Expected<std::vector<uint8_t>>
buildTx81DynlibRunTLV(uint64_t dynModsDeviceAddress);

} // namespace wafer::runtime

#endif // WAFER_RUNTIME_TX81MODELABI_H
