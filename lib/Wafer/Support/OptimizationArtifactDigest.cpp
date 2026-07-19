//===- OptimizationArtifactDigest.cpp - Exact artifact snapshots --------===//

#include "Wafer/Support/OptimizationArtifactDigest.h"

#include "llvm/Support/SHA256.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace wafer {
namespace {

static void hashU64(llvm::SHA256 &hash, uint64_t value) {
  std::array<uint8_t, 8> bytes{};
  for (unsigned index = 0; index < bytes.size(); ++index)
    bytes[index] = static_cast<uint8_t>(value >> (56 - index * 8));
  hash.update(bytes);
}

static void startDomain(llvm::SHA256 &hash, llvm::StringRef domain) {
  hash.update(domain);
  const uint8_t separator = 0;
  hash.update(llvm::ArrayRef<uint8_t>(&separator, 1));
}

static bool hashRegularFileContents(llvm::SHA256 &hash,
                                    const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    return false;
  std::array<char, 64 * 1024> buffer{};
  while (stream) {
    stream.read(buffer.data(), buffer.size());
    std::streamsize count = stream.gcount();
    if (count > 0)
      hash.update(llvm::ArrayRef<uint8_t>(
          reinterpret_cast<const uint8_t *>(buffer.data()), count));
  }
  return stream.eof();
}

} // namespace

std::optional<OptimizationDigest>
digestOptimizationArtifactFileV1(llvm::StringRef path,
                                 llvm::StringRef domain) {
  std::filesystem::path file(path.str());
  std::error_code error;
  if (!std::filesystem::is_regular_file(file, error) || error)
    return std::nullopt;
  llvm::SHA256 hash;
  startDomain(hash, domain);
  uintmax_t size = std::filesystem::file_size(file, error);
  if (error)
    return std::nullopt;
  hashU64(hash, size);
  if (!hashRegularFileContents(hash, file))
    return std::nullopt;
  return hash.final();
}

std::optional<OptimizationDigest>
digestOptimizationArtifactDirectoryV1(llvm::StringRef directory,
                                      llvm::StringRef domain) {
  std::error_code error;
  std::filesystem::path root(directory.str());
  if (!std::filesystem::is_directory(root, error) || error)
    return std::nullopt;
  std::vector<std::filesystem::path> files;
  for (std::filesystem::recursive_directory_iterator iterator(root, error), end;
       !error && iterator != end; iterator.increment(error)) {
    if (iterator->is_symlink(error) || error)
      return std::nullopt;
    if (iterator->is_regular_file(error))
      files.push_back(std::filesystem::relative(iterator->path(), root, error));
    if (error)
      return std::nullopt;
  }
  if (error)
    return std::nullopt;
  std::sort(files.begin(), files.end());

  llvm::SHA256 hash;
  startDomain(hash, domain);
  hashU64(hash, files.size());
  for (const std::filesystem::path &relative : files) {
    std::string generic = relative.generic_string();
    hashU64(hash, generic.size());
    hash.update(generic);
    std::filesystem::path absolute = root / relative;
    uintmax_t size = std::filesystem::file_size(absolute, error);
    if (error)
      return std::nullopt;
    hashU64(hash, size);
    if (!hashRegularFileContents(hash, absolute))
      return std::nullopt;
  }
  return hash.final();
}

OptimizationDigest digestOptimizationBytesV1(llvm::StringRef domain,
                                              llvm::ArrayRef<uint8_t> bytes) {
  llvm::SHA256 hash;
  startDomain(hash, domain);
  hashU64(hash, bytes.size());
  hash.update(bytes);
  return hash.final();
}

} // namespace wafer
