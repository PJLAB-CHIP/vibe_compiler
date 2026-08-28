//===- TargetModelInvocation.h - Typed source/model binding ----*- C++ -*-===//

#ifndef WAFER_MODEL_TARGETMODELINVOCATION_H
#define WAFER_MODEL_TARGETMODELINVOCATION_H

#include "Wafer/Simulator/Memory/TargetModelMemory.h"
#include "Wafer/Simulator/Invocation/ProgramInvocation.h"
#include "Wafer/Simulator/Invocation/TargetCallExecution.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace wafer::model {

enum class TargetModelInvocationErrorCode : uint8_t {
  InvalidTileDomain,
  InvalidProgramInvocation,
  InvalidTileEntryArgument,
  UnsupportedProgramTensor,
  AddressOverflow,
  FrontendPreparationFailure,
};

llvm::StringRef
stringifyTargetModelInvocationErrorCode(TargetModelInvocationErrorCode code);

class TargetModelInvocationError final
    : public llvm::ErrorInfo<TargetModelInvocationError> {
public:
  static char ID;

  TargetModelInvocationError(TargetModelInvocationErrorCode code,
                             std::string detail)
      : code(code), detail(std::move(detail)) {}

  TargetModelInvocationErrorCode getCode() const { return code; }
  llvm::StringRef getDetail() const { return detail; }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  TargetModelInvocationErrorCode code;
  std::string detail;
};

/// Complete owner-backed input to the SystemC execution boundary. Slot values
/// and read-only physical bytes are closed before the target-call frontend is
/// prepared; neither aliases source NPY storage.
class PreparedTargetModelInvocation {
public:
  PreparedTargetModelInvocation(PreparedTargetModelInvocation &&) = default;
  PreparedTargetModelInvocation &
  operator=(PreparedTargetModelInvocation &&) = default;
  PreparedTargetModelInvocation(const PreparedTargetModelInvocation &) = delete;
  PreparedTargetModelInvocation &
  operator=(const PreparedTargetModelInvocation &) = delete;

  compiler::TargetCallExecutable &getExecutable() { return executable; }
  /// Unique card-owned input allocations. Repeated Tile ABI views do not
  /// duplicate the binding or its physical bytes.
  llvm::ArrayRef<TargetModelInputBinding> getInputBindings() const {
    return inputBindings;
  }

private:
  friend llvm::Expected<PreparedTargetModelInvocation>
  prepareTargetModelInvocation(const compiler::DeviceExecutable &,
                               const compiler::TargetLLVMModules &,
                               llvm::ArrayRef<compiler::ProgramTileInvocation>);

  PreparedTargetModelInvocation(
      compiler::TargetCallExecutable executable,
      std::vector<TargetModelInputBinding> inputBindings)
      : executable(std::move(executable)),
        inputBindings(std::move(inputBindings)) {}

  compiler::TargetCallExecutable executable;
  std::vector<TargetModelInputBinding> inputBindings;
};

/// Converts one compact typed program tensor to or from the physical layout
/// carried by an exact tile entry argument. These helpers consume the shared
/// physical tensor codec and do not infer layout from byte size or a name.
llvm::Expected<std::vector<uint8_t>>
encodeTargetModelProgramTensor(const compiler::ProgramTensor &tensor,
                               const compiler::TileEntryArgument &slot);
llvm::Expected<compiler::ProgramTensor>
decodeTargetModelProgramTensor(const compiler::TileEntryArgument &slot,
                               llvm::ArrayRef<uint8_t> physicalBytes);

/// Exact card binding from the accepted source invocation to the target
/// LLVM fixed ABI. Every non-output program resource and every tile entry
/// argument is consumed exactly once before host JIT materialization is
/// returned.
llvm::Expected<PreparedTargetModelInvocation> prepareTargetModelInvocation(
    const compiler::DeviceExecutable &deviceExecutable,
    const compiler::TargetLLVMModules &targetLLVMModules,
    llvm::ArrayRef<compiler::ProgramTileInvocation> programInvocations);

} // namespace wafer::model

#endif // WAFER_MODEL_TARGETMODELINVOCATION_H
