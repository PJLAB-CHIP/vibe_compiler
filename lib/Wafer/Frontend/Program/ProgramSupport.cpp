//===- ProgramSupport.cpp - Frontend program shared support -------------===//

#include "ProgramInternal.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>
#include <optional>
#include <string>

using namespace mlir;

namespace wafer::frontend::program_detail {

bool rejectProgramDirectory(llvm::StringRef reason,
                            llvm::raw_ostream &diagnostics) {
  diagnostics << "program directory metadata rejected: " << reason << "\n";
  return true;
}

std::string dtypeString(Type elementType) {
  if (isa<Float32Type>(elementType))
    return "f32";
  if (isa<Float16Type>(elementType))
    return "f16";
  if (isa<BFloat16Type>(elementType))
    return "bf16";
  if (isa<Float64Type>(elementType))
    return "f64";
  if (auto integer = dyn_cast<IntegerType>(elementType))
    return ("i" + Twine(integer.getWidth())).str();
  return "";
}

std::string normalizeProgramDtype(llvm::StringRef dtype) {
  if (dtype == "float32")
    return "f32";
  if (dtype == "float16")
    return "f16";
  if (dtype == "bfloat16")
    return "bf16";
  if (dtype == "float64")
    return "f64";
  if (dtype == "int8")
    return "i8";
  if (dtype == "int16")
    return "i16";
  if (dtype == "int32")
    return "i32";
  if (dtype == "int64")
    return "i64";
  return dtype.str();
}

bool checkedMulUint64(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

std::optional<uint64_t> checkedRawByteSize(llvm::ArrayRef<int64_t> shape,
                                           Type elementType) {
  elementType = mlir::getElementTypeOrSelf(elementType);
  if (dtypeString(elementType).empty())
    return std::nullopt;

  unsigned bitWidth = elementType.getIntOrFloatBitWidth();
  if (bitWidth == 0 || bitWidth % 8 != 0)
    return std::nullopt;

  // A rank-zero tensor is a scalar and therefore has one element. A static
  // zero dimension makes the tensor empty; zero is a representable byte size,
  // not an overflow sentinel.
  uint64_t elements = 1;
  for (int64_t dim : shape) {
    uint64_t next = 0;
    if (dim < 0 ||
        !checkedMulUint64(elements, static_cast<uint64_t>(dim), next))
      return std::nullopt;
    elements = next;
  }

  uint64_t bytes = 0;
  if (!checkedMulUint64(elements, bitWidth / 8, bytes))
    return std::nullopt;
  return bytes;
}

std::string programPath(llvm::StringRef programDir,
                        llvm::ArrayRef<llvm::StringRef> components) {
  llvm::SmallString<256> path(programDir);
  for (llvm::StringRef component : components)
    llvm::sys::path::append(path, component);
  return path.str().str();
}

bool fileExists(llvm::StringRef path) {
  llvm::sys::fs::file_status status;
  if (std::error_code error = llvm::sys::fs::status(path, status))
    return false;
  return llvm::sys::fs::is_regular_file(status);
}

} // namespace wafer::frontend::program_detail
