//===- XlaSpmdFilesystem.cpp - Program payload filesystem I/O --------===//

#include "XlaSpmdPartitionerInternal.h"

#include "absl/strings/str_cat.h"
#include "llvm/ADT/STLExtras.h"

#include <fstream>
#include <iterator>
#include <system_error>

namespace wafer::xla_spmd_helper {

absl::StatusOr<std::string> readFile(const fs::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    return absl::NotFoundError(absl::StrCat("failed to open ", path.string()));
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

absl::Status writeFile(const fs::path &path, std::string_view content) {
  fs::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary);
  if (!output)
    return absl::InternalError(absl::StrCat("failed to write ", path.string()));
  output.write(content.data(), static_cast<std::streamsize>(content.size()));
  return absl::OkStatus();
}

absl::Status writeBytes(const fs::path &path,
                        const std::vector<uint8_t> &content) {
  fs::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary);
  if (!output)
    return absl::InternalError(absl::StrCat("failed to write ", path.string()));
  output.write(reinterpret_cast<const char *>(content.data()),
               static_cast<std::streamsize>(content.size()));
  return absl::OkStatus();
}

absl::Status copyConstantPayloads(const Options &options,
                                  const ProgramMetadata &meta) {
  std::vector<int64_t> copiedPositions;
  for (const InputLocation &location : meta.inputLocations) {
    if (location.type != "constant")
      continue;
    if (location.position < 0)
      return absl::InvalidArgumentError(
          "constant input location has negative position");
    if (llvm::is_contained(copiedPositions, location.position))
      continue;

    fs::path relative =
        fs::path("constants") / std::to_string(location.position);
    fs::path source = options.inputProgramDir / relative;
    fs::path destination = options.outputProgramDir / relative;
    if (!fs::is_regular_file(source)) {
      return absl::NotFoundError(
          absl::StrCat("missing constant data file: ", source.string()));
    }

    fs::create_directories(destination.parent_path());
    std::error_code error;
    fs::copy_file(source, destination, fs::copy_options::overwrite_existing,
                  error);
    if (error) {
      return absl::InternalError(
          absl::StrCat("failed to copy constant data file: ", source.string(),
                       " -> ", destination.string(), ": ", error.message()));
    }
    copiedPositions.push_back(location.position);
  }
  return absl::OkStatus();
}

} // namespace wafer::xla_spmd_helper
