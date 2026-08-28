//===- ProgramInvocation.h - Typed program invocation -------*- C++ -*-===//

#ifndef WAFER_PROGRAM_PROGRAMINVOCATION_H
#define WAFER_PROGRAM_PROGRAMINVOCATION_H

#include "Wafer/Driver/Compilation.h"
#include "Wafer/Program/ProgramElementType.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace wafer::compiler {

/// Checked compact program-boundary byte geometry. Boolean program tensors
/// are not currently admitted because their source NPY byte representation is
/// distinct from the target bitpacked representation.
std::optional<int64_t>
computeProgramTensorByteCount(ProgramElementType dtype,
                              llvm::ArrayRef<int64_t> shape);

/// Returns whether one admitted ProgramTensor dtype has floating-point
/// semantics. Keeping this classification with byte geometry prevents output
/// consumers from silently treating a newly admitted float as raw storage.
bool isFloatingProgramTensorDType(ProgramElementType dtype);

/// Owner-backed compact row-major tensor at a typed program boundary.
/// Multi-byte elements use canonical little-endian storage. This is source
/// invocation data, not an execution result or device storage. Tensors can
/// be shared views over one transaction-owned materialization; repeated
/// Tile bindings then never duplicate payload bytes.
class ProgramTensor {
public:
  static llvm::Expected<ProgramTensor> create(ProgramElementType dtype,
                                              llvm::ArrayRef<int64_t> shape,
                                              llvm::ArrayRef<uint8_t> bytes);
  static llvm::Expected<ProgramTensor> loadNpy(llvm::StringRef path);

  /// Non-owning view over shared materialized storage. The view covers
  /// [offset, offset + byteCount) of `storage`; the shared owner keeps the
  /// bytes alive for the lifetime of every view.
  static llvm::Expected<ProgramTensor>
  share(ProgramElementType dtype, llvm::ArrayRef<int64_t> shape,
        std::shared_ptr<const std::vector<uint8_t>> storage, size_t offset);

  ProgramElementType getDType() const { return dtype; }
  llvm::ArrayRef<int64_t> getShape() const { return shape; }
  llvm::ArrayRef<uint8_t> getBytes() const;

private:
  ProgramTensor(ProgramElementType dtype, std::vector<int64_t> shape,
                std::vector<uint8_t> bytes)
      : dtype(dtype), shape(std::move(shape)), bytes(std::move(bytes)) {}

  ProgramElementType dtype;
  std::vector<int64_t> shape;
  std::vector<uint8_t> bytes;
  std::shared_ptr<const std::vector<uint8_t>> sharedStorage;
  size_t sharedOffset = 0;
  size_t sharedSize = 0;
};

/// Exact non-output source resource supplied to one accepted Tile.
struct ProgramInputBinding {
  ProgramResourceRole role;
  int64_t index;
  ProgramTensor tensor;
};

/// Complete source invocation inputs for one Tile launch entry.
struct ProgramTileInvocation {
  CardId cardId;
  TileId tileId;
  LaunchSlotId launchSlotId;
  std::vector<ProgramInputBinding> inputs;
};

/// One complete logical user input before accepted partition slicing.
struct ProgramGlobalInputBinding {
  int64_t index;
  ProgramTensor tensor;
};

/// Builds all Tile-local source bindings from typed deviceExecutable
/// slices. User inputs are sliced from complete logical tensors;
/// parameters/constants are materialized exactly once per owned
/// ProgramDataRange through the device executable's program data handoff, and
/// every Tile view shares the same materialized storage. The function
/// performs no compute and is shared by independent execution consumers
/// without sharing their numeric kernels or schedulers.
llvm::Expected<std::vector<ProgramTileInvocation>> prepareProgramInvocations(
    const DeviceExecutable &deviceExecutable,
    llvm::ArrayRef<ProgramGlobalInputBinding> globalInputs);

/// Applies one already-verified program binding slice to a complete logical
/// tensor. Both user-input and expected-output consumers use this slicing
/// operation; it performs no reference or target compute.
llvm::Expected<ProgramTensor>
sliceProgramTensorForBinding(const ProgramTensor &global,
                             const ProgramResourceBinding &binding);

} // namespace wafer::compiler

#endif // WAFER_PROGRAM_PROGRAMINVOCATION_H
