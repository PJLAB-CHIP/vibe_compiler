//===- NumericDependencyELF.cpp - Managed files and ELF metadata -*- C++
//-*-===//

#include "NumericDependencyConformanceInternal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Path.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace wafer::numeric_dependency_conformance_internal {

bool isLowerHex(llvm::StringRef value) {
  return !value.empty() && llvm::all_of(value, [](char character) {
    return (character >= '0' && character <= '9') ||
           (character >= 'a' && character <= 'f');
  });
}

llvm::Expected<bool> hasELFMagic(llvm::StringRef path,
                                 const llvm::Twine &label) {
  const std::string pathStorage = path.str();
  int flags = O_RDONLY;
#ifdef O_CLOEXEC
  flags |= O_CLOEXEC;
#endif
#ifdef O_NOFOLLOW
  flags |= O_NOFOLLOW;
#endif
  const int descriptor = ::open(pathStorage.c_str(), flags);
  if (descriptor < 0)
    return invalid(ErrorCode::IO, label +
                                      " cannot be opened for ELF inspection: " +
                                      std::strerror(errno));
  char magic[4] = {};
  size_t offset = 0;
  while (offset != sizeof(magic)) {
    const ssize_t count =
        ::read(descriptor, magic + offset, sizeof(magic) - offset);
    if (count < 0) {
      if (errno == EINTR)
        continue;
      const int readError = errno;
      ::close(descriptor);
      return invalid(ErrorCode::IO, label +
                                        " cannot be read for ELF inspection: " +
                                        std::strerror(readError));
    }
    if (count == 0)
      break;
    offset += static_cast<size_t>(count);
  }
  ::close(descriptor);
  return offset == sizeof(magic) && magic[0] == '\x7f' && magic[1] == 'E' &&
         magic[2] == 'L' && magic[3] == 'F';
}

llvm::Expected<std::string> findELFHeader(llvm::StringRef output,
                                          llvm::StringRef name,
                                          const llvm::Twine &label) {
  llvm::SmallVector<llvm::StringRef, 64> lines;
  output.split(lines, '\n');
  const std::string prefix = (name + ":").str();
  for (llvm::StringRef line : lines) {
    line = line.trim();
    if (line.consume_front(prefix))
      return line.trim().str();
  }
  return invalid(ErrorCode::PolicyMismatch,
                 label + " readelf output is missing " + name);
}

std::vector<std::string> findELFDynamicValues(llvm::StringRef output,
                                              llvm::StringRef tag,
                                              bool splitPaths) {
  llvm::SmallVector<llvm::StringRef, 64> lines;
  output.split(lines, '\n');
  const std::string marker = ("(" + tag + ")").str();
  std::vector<std::string> values;
  for (llvm::StringRef line : lines) {
    const size_t markerPosition = line.find(marker);
    if (markerPosition == llvm::StringRef::npos)
      continue;
    const size_t open = line.find('[', markerPosition + marker.size());
    const size_t close = open == llvm::StringRef::npos
                             ? llvm::StringRef::npos
                             : line.find(']', open + 1);
    if (open == llvm::StringRef::npos || close == llvm::StringRef::npos)
      continue;
    llvm::StringRef value = line.slice(open + 1, close);
    if (!splitPaths) {
      values.push_back(value.str());
      continue;
    }
    llvm::SmallVector<llvm::StringRef, 8> paths;
    value.split(paths, ':', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    for (llvm::StringRef path : paths)
      values.push_back(path.str());
  }
  return values;
}

llvm::Expected<NumericDependencyELFRecord>
readELFRecord(llvm::StringRef path, llvm::StringRef readelfPath,
              const llvm::Twine &label) {
  const std::vector<std::string> arguments = {readelfPath.str(), "--wide",
                                              "--file-header",   "--notes",
                                              "--dynamic",       path.str()};
  const std::vector<std::string> environment = commandEnvironment(readelfPath);
  llvm::Expected<std::string> output = runAndCapture(
      readelfPath, arguments, environment, label + " readelf inspection");
  if (!output)
    return output.takeError();

  NumericDependencyELFRecord result;
  llvm::Expected<std::string> elfClass = findELFHeader(*output, "Class", label);
  if (!elfClass)
    return elfClass.takeError();
  llvm::Expected<std::string> data = findELFHeader(*output, "Data", label);
  if (!data)
    return data.takeError();
  llvm::Expected<std::string> type = findELFHeader(*output, "Type", label);
  if (!type)
    return type.takeError();
  llvm::Expected<std::string> machine =
      findELFHeader(*output, "Machine", label);
  if (!machine)
    return machine.takeError();
  result.elfClass = std::move(*elfClass);
  if (llvm::StringRef(*data).contains("little endian"))
    result.byteOrder = "little";
  else if (llvm::StringRef(*data).contains("big endian"))
    result.byteOrder = "big";
  else
    return invalid(ErrorCode::PolicyMismatch,
                   label + " readelf output has unknown byte order");
  result.type = llvm::StringRef(*type).ltrim().split(' ').first.str();
  result.machine = std::move(*machine);

  const size_t buildMarker = output->find("Build ID:");
  if (buildMarker != std::string::npos) {
    llvm::StringRef build =
        llvm::StringRef(*output)
            .drop_front(buildMarker + llvm::StringRef("Build ID:").size())
            .ltrim();
    build = build.take_while(llvm::isHexDigit);
    if (!build.empty()) {
      std::string lowered = build.str();
      llvm::transform(lowered, lowered.begin(), [](char character) {
        return static_cast<char>(
            std::tolower(static_cast<unsigned char>(character)));
      });
      result.buildId = std::move(lowered);
    }
  }
  std::vector<std::string> sonames =
      findELFDynamicValues(*output, "SONAME", /*splitPaths=*/false);
  if (!sonames.empty())
    result.soname = std::move(sonames.front());
  result.needed = findELFDynamicValues(*output, "NEEDED", /*splitPaths=*/false);
  result.rpath = findELFDynamicValues(*output, "RPATH", /*splitPaths=*/true);
  result.runpath =
      findELFDynamicValues(*output, "RUNPATH", /*splitPaths=*/true);
  return result;
}

bool sameELFRecord(const NumericDependencyELFRecord &lhs,
                   const NumericDependencyELFRecord &rhs) {
  return lhs.elfClass == rhs.elfClass && lhs.byteOrder == rhs.byteOrder &&
         lhs.type == rhs.type && lhs.machine == rhs.machine &&
         lhs.buildId == rhs.buildId && lhs.soname == rhs.soname &&
         lhs.needed == rhs.needed && lhs.rpath == rhs.rpath &&
         lhs.runpath == rhs.runpath;
}

llvm::Expected<NumericDependencyFileRecord>
parseFileRecord(llvm::StringRef name, const llvm::json::Value &value,
                const ManagedRoot &root, llvm::StringRef readelfPath) {
  llvm::Expected<const llvm::json::Object *> object =
      requireObject(value, "numeric dependency file " + name);
  if (!object)
    return object.takeError();
  if (llvm::Error error = requireExactKeys(
          **object, {"path", "sha256", "size", "file_type", "mode", "elf"},
          "numeric dependency file " + name))
    return error;

  llvm::Expected<std::string> relative =
      requireString(*(*object)->get("path"), "numeric dependency file path");
  if (!relative)
    return relative.takeError();
  llvm::Expected<std::string> digest = requireString(
      *(*object)->get("sha256"), "numeric dependency file SHA256");
  if (!digest)
    return digest.takeError();
  if (!isLowerSHA256(*digest))
    return invalid(ErrorCode::TypeMismatch,
                   "numeric dependency file SHA256 is not lowercase SHA-256");
  llvm::Expected<uint64_t> size =
      requireUnsigned(*(*object)->get("size"), "numeric dependency file size");
  if (!size)
    return size.takeError();
  llvm::Expected<std::string> fileType = requireString(
      *(*object)->get("file_type"), "numeric dependency file file_type");
  if (!fileType)
    return fileType.takeError();
  if (*fileType != "regular")
    return invalid(ErrorCode::FileType,
                   "numeric dependency file file_type is not regular");
  llvm::Expected<uint64_t> mode =
      requireUnsigned(*(*object)->get("mode"), "numeric dependency file mode");
  if (!mode)
    return mode.takeError();
  if (*mode > 07777)
    return invalid(ErrorCode::TypeMismatch,
                   "numeric dependency file mode is out of range");

  std::optional<NumericDependencyELFRecord> elf;
  const llvm::json::Value &elfValue = *(*object)->get("elf");
  if (!elfValue.getAsNull()) {
    llvm::Expected<const llvm::json::Object *> elfObject =
        requireObject(elfValue, "numeric dependency file ELF metadata");
    if (!elfObject)
      return elfObject.takeError();
    if (llvm::Error error = requireExactKeys(
            **elfObject,
            {"class", "byte_order", "type", "machine", "build_id", "soname",
             "needed", "rpath", "runpath"},
            "numeric dependency file ELF metadata"))
      return error;
    NumericDependencyELFRecord parsedELF;
#define READ_ELF_STRING(Field, Key)                                            \
  do {                                                                         \
    llvm::Expected<std::string> parsed = requireString(                        \
        *(*elfObject)->get(Key), "numeric dependency file ELF " Key);          \
    if (!parsed)                                                               \
      return parsed.takeError();                                               \
    parsedELF.Field = std::move(*parsed);                                      \
  } while (false)
    READ_ELF_STRING(elfClass, "class");
    READ_ELF_STRING(byteOrder, "byte_order");
    READ_ELF_STRING(type, "type");
    READ_ELF_STRING(machine, "machine");
#undef READ_ELF_STRING
    llvm::Expected<std::optional<std::string>> buildId = requireNullableString(
        *(*elfObject)->get("build_id"), "numeric dependency file ELF build_id");
    if (!buildId)
      return buildId.takeError();
    llvm::Expected<std::optional<std::string>> soname = requireNullableString(
        *(*elfObject)->get("soname"), "numeric dependency file ELF soname");
    if (!soname)
      return soname.takeError();
    parsedELF.buildId = std::move(*buildId);
    parsedELF.soname = std::move(*soname);
    if (parsedELF.buildId &&
        (parsedELF.buildId->empty() ||
         !llvm::all_of(*parsedELF.buildId, llvm::isHexDigit)))
      return invalid(ErrorCode::TypeMismatch,
                     "numeric dependency file ELF build_id is malformed");
    llvm::Expected<std::vector<std::string>> needed = requireStringArray(
        *(*elfObject)->get("needed"), "numeric dependency file ELF needed");
    if (!needed)
      return needed.takeError();
    llvm::Expected<std::vector<std::string>> rpath = requireStringArray(
        *(*elfObject)->get("rpath"), "numeric dependency file ELF rpath");
    if (!rpath)
      return rpath.takeError();
    llvm::Expected<std::vector<std::string>> runpath = requireStringArray(
        *(*elfObject)->get("runpath"), "numeric dependency file ELF runpath");
    if (!runpath)
      return runpath.takeError();
    parsedELF.needed = std::move(*needed);
    parsedELF.rpath = std::move(*rpath);
    parsedELF.runpath = std::move(*runpath);
    if (parsedELF.elfClass.empty() || parsedELF.byteOrder.empty() ||
        parsedELF.type.empty() || parsedELF.machine.empty())
      return invalid(ErrorCode::TypeMismatch,
                     "numeric dependency file ELF metadata is incomplete");
    elf = std::move(parsedELF);
  }

  llvm::Expected<ManagedPath> path = resolveManagedPath(
      root, *relative, RequiredFileType::RegularFile,
      "numeric dependency file " + name, /*requireRelative=*/true);
  if (!path)
    return path.takeError();
  llvm::Expected<FileReadback> readback =
      readRegularFile(path->resolved, std::numeric_limits<uint64_t>::max(),
                      "numeric dependency file " + name);
  if (!readback)
    return readback.takeError();
  if (readback->size != *size)
    return invalid(ErrorCode::SizeMismatch,
                   "numeric dependency file " + name + " size mismatch");
  if (readback->digest != *digest)
    return invalid(ErrorCode::DigestMismatch,
                   "numeric dependency file " + name + " SHA256 mismatch");
  if (readback->mode != *mode)
    return invalid(ErrorCode::PolicyMismatch,
                   "numeric dependency file " + name + " mode mismatch");

  llvm::Expected<bool> isELF =
      hasELFMagic(path->resolved, "numeric dependency file " + name);
  if (!isELF)
    return isELF.takeError();
  if (*isELF) {
    if (!elf)
      return invalid(ErrorCode::PolicyMismatch,
                     "numeric dependency file " + name +
                         " omits its actual ELF metadata");
    llvm::Expected<NumericDependencyELFRecord> actualELF = readELFRecord(
        path->resolved, readelfPath, "numeric dependency file " + name);
    if (!actualELF)
      return actualELF.takeError();
    if (!sameELFRecord(*elf, *actualELF))
      return invalid(ErrorCode::PolicyMismatch, "numeric dependency file " +
                                                    name +
                                                    " ELF metadata mismatch");
  } else if (elf) {
    return invalid(ErrorCode::PolicyMismatch,
                   "numeric dependency file " + name +
                       " records ELF metadata for a non-ELF file");
  }

  llvm::Expected<FileReadback> finalReadback =
      readRegularFile(path->resolved, std::numeric_limits<uint64_t>::max(),
                      "numeric dependency file " + name);
  if (!finalReadback)
    return finalReadback.takeError();
  if (finalReadback->digest != readback->digest ||
      finalReadback->size != readback->size ||
      finalReadback->mode != readback->mode)
    return invalid(ErrorCode::IO,
                   "numeric dependency file " + name +
                       " changed during ELF metadata inspection");
  return NumericDependencyFileRecord{name.str(),
                                     path->relative,
                                     path->resolved,
                                     std::move(*digest),
                                     *size,
                                     std::move(*fileType),
                                     static_cast<uint32_t>(*mode),
                                     std::move(elf)};
}

llvm::Error
validateSharedObjectPair(llvm::StringRef stem,
                         const NumericDependencyFileRecord &real,
                         const NumericDependencyFileRecord &loader) {
  llvm::StringRef realName = llvm::sys::path::filename(real.resolvedPath);
  llvm::StringRef loaderName = llvm::sys::path::filename(loader.resolvedPath);
  const std::string prefix = ("lib" + stem + ".so.").str();
  if (!realName.starts_with(prefix) || !loaderName.starts_with(prefix) ||
      real.resolvedPath == loader.resolvedPath)
    return invalid(ErrorCode::PolicyMismatch,
                   stem + " real/loader shared-object metadata is invalid");
  if (real.sha256 != loader.sha256 || real.size != loader.size)
    return invalid(ErrorCode::DigestMismatch,
                   stem + " real/loader shared-object content differs");
  if (real.mode != loader.mode || real.fileType != loader.fileType ||
      !real.elf || !loader.elf)
    return invalid(ErrorCode::PolicyMismatch,
                   stem + " real/loader ELF metadata is incomplete");
  const NumericDependencyELFRecord &lhs = *real.elf;
  const NumericDependencyELFRecord &rhs = *loader.elf;
  if (lhs.elfClass != rhs.elfClass || lhs.byteOrder != rhs.byteOrder ||
      lhs.type != rhs.type || lhs.machine != rhs.machine ||
      lhs.buildId != rhs.buildId || lhs.soname != rhs.soname ||
      lhs.needed != rhs.needed || lhs.rpath != rhs.rpath ||
      lhs.runpath != rhs.runpath || lhs.type != "DYN" ||
      lhs.byteOrder != "little" || !lhs.soname || *lhs.soname != loaderName ||
      !lhs.buildId || !isLowerHex(*lhs.buildId) ||
      !llvm::StringRef(*lhs.soname).starts_with(prefix))
    return invalid(ErrorCode::PolicyMismatch,
                   stem + " real/loader ELF metadata mismatch");
  return llvm::Error::success();
}

} // namespace wafer::numeric_dependency_conformance_internal
