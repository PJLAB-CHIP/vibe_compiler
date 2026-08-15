//===- ProgramInvocation.h - Typed program invocation -------*- C++ -*-===//

#ifndef WAFER_COMPILER_PROGRAMINVOCATION_H
#define WAFER_COMPILER_PROGRAMINVOCATION_H

#include "Wafer/Compiler/Compilation.h"

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
computeProgramTensorByteCount(llvm::StringRef dtype,
                              llvm::ArrayRef<int64_t> shape);

/// Returns whether one admitted ProgramTensor dtype has floating-point
/// semantics. Keeping this classification with byte geometry prevents output
/// consumers from silently treating a newly admitted float as raw storage.
bool isFloatingProgramTensorDType(llvm::StringRef dtype);

/// Owner-backed compact row-major tensor at a typed program boundary.
/// Multi-byte elements use canonical little-endian storage. This is source
/// invocation data, not an execution result or device storage.
class ProgramTensor {
public:
  static llvm::Expected<ProgramTensor> create(llvm::StringRef dtype,
                                              llvm::ArrayRef<int64_t> shape,
                                              llvm::ArrayRef<uint8_t> bytes);
  static llvm::Expected<ProgramTensor> loadNpy(llvm::StringRef path);

  llvm::StringRef getDType() const { return dtype; }
  llvm::ArrayRef<int64_t> getShape() const { return shape; }
  llvm::ArrayRef<uint8_t> getBytes() const { return bytes; }

private:
  ProgramTensor(std::string dtype, std::vector<int64_t> shape,
                std::vector<uint8_t> bytes)
      : dtype(std::move(dtype)), shape(std::move(shape)),
        bytes(std::move(bytes)) {}

  std::string dtype;
  std::vector<int64_t> shape;
  std::vector<uint8_t> bytes;
};

/// Exact non-output source resource supplied to one accepted physical Tile.
struct ProgramInputBinding {
  ProgramResourceRole role;
  int64_t index;
  ProgramTensor tensor;
};

/// Complete source invocation inputs for one physical Tile launch entry.
struct ProgramTileInvocation {
  PhysicalCardId physicalCardId;
  PhysicalTileId physicalTileId;
  LaunchSlotId launchSlotId;
  std::vector<ProgramInputBinding> inputs;
};

/// One complete logical user input before accepted partition slicing.
struct ProgramGlobalInputBinding {
  int64_t index;
  ProgramTensor tensor;
};

/// Builds all Tile-local source bindings from typed physicalTileExecutables
/// slices. User inputs are sliced from complete logical tensors;
/// parameters/constants are loaded from their already-verified package-relative
/// NPY payload paths. The function performs no compute and is shared by
/// independent execution consumers without sharing their numeric kernels or
/// schedulers.
llvm::Expected<std::vector<ProgramTileInvocation>> prepareProgramInvocations(
    const PhysicalTileExecutables &physicalTileExecutables,
    llvm::StringRef packageRoot,
    llvm::ArrayRef<ProgramGlobalInputBinding> globalInputs);

/// Applies one already-verified program binding slice to a complete logical
/// tensor. Both user-input and expected-output consumers use this slicing
/// operation; it performs no reference or target compute.
llvm::Expected<ProgramTensor>
sliceProgramTensorForBinding(const ProgramTensor &global,
                             const ProgramResourceBinding &binding);

} // namespace wafer::compiler

#endif // WAFER_COMPILER_PROGRAMINVOCATION_H
