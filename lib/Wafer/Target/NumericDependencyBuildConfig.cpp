//===- NumericDependencyBuildConfig.cpp - Build configuration -*- C++ -*-===//

#include "NumericDependencyConformanceInternal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"

#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace wafer::numeric_dependency_conformance_internal {

std::string sha256(llvm::StringRef bytes) {
  llvm::SHA256 hasher;
  hasher.update(bytes);
  return llvm::toHex(hasher.final(), /*LowerCase=*/true);
}

llvm::Expected<NumericDependencyToolRecord>
parseToolRecord(llvm::StringRef name, const llvm::json::Value &value) {
  llvm::Expected<const llvm::json::Object *> object =
      requireObject(value, "numeric tool " + name);
  if (!object)
    return object.takeError();
  if (llvm::Error error =
          requireExactKeys(**object,
                           {"path", "sha256", "size", "file_type", "mode",
                            "version_first_line", "version_output_sha256"},
                           "numeric tool " + name))
    return error;

  NumericDependencyToolRecord result;
  result.name = name.str();
#define READ_TOOL_STRING(Field, Key)                                           \
  do {                                                                         \
    llvm::Expected<std::string> parsed =                                       \
        requireString(*(*object)->get(Key), "numeric tool " + name + "." Key); \
    if (!parsed)                                                               \
      return parsed.takeError();                                               \
    result.Field = std::move(*parsed);                                         \
  } while (false)
  READ_TOOL_STRING(resolvedPath, "path");
  READ_TOOL_STRING(sha256, "sha256");
  READ_TOOL_STRING(fileType, "file_type");
  READ_TOOL_STRING(versionFirstLine, "version_first_line");
  READ_TOOL_STRING(versionOutputSHA256, "version_output_sha256");
#undef READ_TOOL_STRING
  llvm::Expected<uint64_t> size =
      requireUnsigned(*(*object)->get("size"), "numeric tool size");
  if (!size)
    return size.takeError();
  llvm::Expected<uint64_t> mode =
      requireUnsigned(*(*object)->get("mode"), "numeric tool mode");
  if (!mode)
    return mode.takeError();
  if (*mode > 07777 || result.fileType != "regular" ||
      !isLowerSHA256(result.sha256) ||
      !isLowerSHA256(result.versionOutputSHA256) ||
      llvm::StringRef(result.versionFirstLine).contains('\n') ||
      llvm::StringRef(result.versionFirstLine).contains('\r'))
    return invalid(ErrorCode::TypeMismatch,
                   "numeric tool record is malformed for " + name);
  llvm::Expected<std::string> resolved =
      resolveAbsoluteRegularPath(result.resolvedPath, "numeric tool " + name);
  if (!resolved)
    return resolved.takeError();
  llvm::Expected<FileReadback> readback = readRegularFile(
      *resolved, std::numeric_limits<uint64_t>::max(), "numeric tool " + name);
  if (!readback)
    return readback.takeError();
  if (readback->size != *size)
    return invalid(ErrorCode::SizeMismatch,
                   "numeric tool size mismatch for " + name);
  if (readback->digest != result.sha256)
    return invalid(ErrorCode::DigestMismatch,
                   "numeric tool SHA256 mismatch for " + name);
  if (readback->mode != *mode)
    return invalid(ErrorCode::PolicyMismatch,
                   "numeric tool mode mismatch for " + name);

  std::string versionOutput;
  if (name == "false") {
    versionOutput = "managed-false-command\n";
  } else {
    std::vector<std::string> arguments = {*resolved};
    if (name == "shell") {
      arguments.push_back("-c");
      arguments.push_back("printf 'managed-shell\\n'");
    } else {
      arguments.push_back("--version");
    }
    const std::vector<std::string> environment = commandEnvironment(*resolved);
    llvm::Expected<std::string> captured =
        runAndCapture(*resolved, arguments, environment,
                      "numeric tool version command " + name);
    if (!captured)
      return captured.takeError();
    versionOutput = std::move(*captured);
  }
  llvm::StringRef firstLine(versionOutput);
  const size_t lineEnd = firstLine.find_first_of("\r\n");
  if (lineEnd != llvm::StringRef::npos)
    firstLine = firstLine.take_front(lineEnd);
  if (result.versionFirstLine != firstLine ||
      result.versionOutputSHA256 != sha256(versionOutput))
    return invalid(ErrorCode::PolicyMismatch,
                   "numeric tool version mismatch for " + name);
  result.size = *size;
  result.mode = static_cast<uint32_t>(*mode);
  result.resolvedPath = std::move(*resolved);
  return result;
}

llvm::Expected<NumericDependencyEnvironmentRecord>
parseEnvironmentRecord(llvm::StringRef name, const llvm::json::Value &value) {
  llvm::Expected<const llvm::json::Object *> object =
      requireObject(value, "numeric environment " + name);
  if (!object)
    return object.takeError();
  NumericDependencyEnvironmentRecord result;
  result.name = name.str();
  llvm::StringSet<> keys;
  for (const auto &entry : **object) {
    llvm::StringRef key = entry.first;
    if (key.empty() || key.contains('=') || !keys.insert(entry.first).second)
      return invalid(ErrorCode::TypeMismatch,
                     "numeric environment variable name is invalid");
    llvm::Expected<std::string> environmentValue = requireString(
        entry.second, "numeric environment " + name + "." + entry.first.str());
    if (!environmentValue)
      return environmentValue.takeError();
    if (llvm::StringRef(*environmentValue).contains('\0'))
      return invalid(ErrorCode::TypeMismatch,
                     "numeric environment value contains NUL");
    result.variables.emplace_back(entry.first.str(),
                                  std::move(*environmentValue));
  }
  return result;
}

llvm::StringMap<std::string>
environmentMap(const NumericDependencyEnvironmentRecord &environment) {
  llvm::StringMap<std::string> result;
  for (const auto &entry : environment.variables)
    result.insert(entry);
  return result;
}

llvm::Error verifyEnvironment(const NumericDependencyEnvironmentRecord &actual,
                              const llvm::StringMap<std::string> &expected) {
  llvm::StringMap<std::string> values = environmentMap(actual);
  if (values.size() != expected.size())
    return invalid(ErrorCode::PolicyMismatch, "numeric environment " +
                                                  actual.name +
                                                  " variable closure mismatch");
  for (const auto &entry : expected) {
    auto iterator = values.find(entry.first());
    if (iterator == values.end() || iterator->second != entry.second)
      return invalid(ErrorCode::PolicyMismatch,
                     "numeric environment " + actual.name +
                         " does not match frozen policy");
  }
  return llvm::Error::success();
}

llvm::Expected<ParsedBuildConfig>
parseBuildConfig(const llvm::json::Value &value) {
  llvm::Expected<const llvm::json::Object *> build =
      requireObject(value, "numeric dependency build");
  if (!build)
    return build.takeError();
  if (llvm::Error error = requireExactKeys(
          **build,
          {"platform", "softfloat_specialization", "softfloat_thread_local",
           "softfloat_raise_flags", "configure_options", "linkage",
           "managed_m4", "pkg_config", "jobs", "toolchain", "environments",
           "mpfr_patches", "elf_identity_policy"},
          "numeric dependency build"))
    return error;

  ParsedBuildConfig parsedResult;
  NumericDependencyBuildConfig &result = parsedResult.build;
#define READ_BUILD_STRING(Field, Key)                                          \
  do {                                                                         \
    llvm::Expected<std::string> parsed =                                       \
        requireString(*(*build)->get(Key), "numeric build " Key);              \
    if (!parsed)                                                               \
      return parsed.takeError();                                               \
    result.Field = std::move(*parsed);                                         \
  } while (false)
  READ_BUILD_STRING(platform, "platform");
  READ_BUILD_STRING(softFloatSpecialization, "softfloat_specialization");
  READ_BUILD_STRING(softFloatThreadLocal, "softfloat_thread_local");
  READ_BUILD_STRING(softFloatRaiseFlags, "softfloat_raise_flags");
  READ_BUILD_STRING(managedM4RelativePath, "managed_m4");
  READ_BUILD_STRING(pkgConfigPolicy, "pkg_config");
  READ_BUILD_STRING(mpfrPatches, "mpfr_patches");
  READ_BUILD_STRING(elfValidationPolicy, "elf_identity_policy");
#undef READ_BUILD_STRING

  llvm::Expected<uint64_t> jobs =
      requireUnsigned(*(*build)->get("jobs"), "numeric build jobs");
  if (!jobs)
    return jobs.takeError();
  if (*jobs == 0)
    return invalid(ErrorCode::PolicyMismatch,
                   "numeric build jobs must be positive");
  result.jobs = *jobs;

  llvm::Expected<const llvm::json::Object *> toolchain =
      requireObject(*(*build)->get("toolchain"), "numeric toolchain");
  if (!toolchain)
    return toolchain.takeError();
  if (llvm::Error error = requireExactKeys(**toolchain, {"policy", "tools"},
                                           "numeric toolchain"))
    return error;
  llvm::Expected<std::string> toolchainPolicy =
      requireString(*(*toolchain)->get("policy"), "numeric toolchain policy");
  if (!toolchainPolicy)
    return toolchainPolicy.takeError();
  result.toolchainPolicy = std::move(*toolchainPolicy);
  if (result.toolchainPolicy != "wafer-host-numeric-build-environment-v1")
    return invalid(ErrorCode::PolicyMismatch,
                   "numeric toolchain policy mismatch");
  llvm::Expected<const llvm::json::Object *> tools =
      requireObject(*(*toolchain)->get("tools"), "numeric tools");
  if (!tools)
    return tools.takeError();
  llvm::StringSet<> allowedTools;
  for (llvm::StringRef name : kRequiredTools)
    allowedTools.insert(name);
  for (const auto &entry : **tools)
    if (!allowedTools.contains(entry.first))
      return invalid(ErrorCode::UnknownField,
                     "numeric tools has unknown field '" + entry.first.str() +
                         "'");
  for (llvm::StringRef name : kRequiredTools) {
    const llvm::json::Value *toolValue = (*tools)->get(name);
    if (!toolValue)
      return invalid(ErrorCode::MissingField,
                     "numeric tools is missing field '" + name + "'");
    llvm::Expected<NumericDependencyToolRecord> tool =
        parseToolRecord(name, *toolValue);
    if (!tool)
      return tool.takeError();
    parsedResult.tools.push_back(std::move(*tool));
  }

  llvm::Expected<const llvm::json::Object *> environments =
      requireObject(*(*build)->get("environments"), "numeric environments");
  if (!environments)
    return environments.takeError();
  llvm::StringSet<> allowedEnvironments;
  for (llvm::StringRef name : kRequiredEnvironments)
    allowedEnvironments.insert(name);
  for (const auto &entry : **environments)
    if (!allowedEnvironments.contains(entry.first))
      return invalid(ErrorCode::UnknownField,
                     "numeric environments has unknown field '" +
                         entry.first.str() + "'");
  for (llvm::StringRef name : kRequiredEnvironments) {
    const llvm::json::Value *environmentValue = (*environments)->get(name);
    if (!environmentValue)
      return invalid(ErrorCode::MissingField,
                     "numeric environments is missing field '" + name + "'");
    llvm::Expected<NumericDependencyEnvironmentRecord> environment =
        parseEnvironmentRecord(name, *environmentValue);
    if (!environment)
      return environment.takeError();
    parsedResult.environments.push_back(std::move(*environment));
  }

  llvm::Expected<const llvm::json::Object *> options = requireObject(
      *(*build)->get("configure_options"), "numeric configure options");
  if (!options)
    return options.takeError();
  if (llvm::Error error = requireExactKeys(**options, {"m4", "gmp", "mpfr"},
                                           "numeric configure options"))
    return error;
  llvm::Expected<std::vector<std::string>> m4Options = requireStringArray(
      *(*options)->get("m4"), "numeric m4 configure options");
  if (!m4Options)
    return m4Options.takeError();
  llvm::Expected<std::vector<std::string>> gmpOptions = requireStringArray(
      *(*options)->get("gmp"), "numeric GMP configure options");
  if (!gmpOptions)
    return gmpOptions.takeError();
  llvm::Expected<std::vector<std::string>> mpfrOptions = requireStringArray(
      *(*options)->get("mpfr"), "numeric MPFR configure options");
  if (!mpfrOptions)
    return mpfrOptions.takeError();
  result.m4ConfigureOptions = std::move(*m4Options);
  result.gmpConfigureOptions = std::move(*gmpOptions);
  result.mpfrConfigureOptions = std::move(*mpfrOptions);

  llvm::Expected<const llvm::json::Object *> linkage =
      requireObject(*(*build)->get("linkage"), "numeric linkage policy");
  if (!linkage)
    return linkage.takeError();
  if (llvm::Error error = requireExactKeys(
          **linkage, {"softfloat", "gmp", "mpfr"}, "numeric linkage policy"))
    return error;
#define READ_LINKAGE_STRING(Field, Key)                                        \
  do {                                                                         \
    llvm::Expected<std::string> parsed =                                       \
        requireString(*(*linkage)->get(Key), "numeric linkage " Key);          \
    if (!parsed)                                                               \
      return parsed.takeError();                                               \
    result.Field = std::move(*parsed);                                         \
  } while (false)
  READ_LINKAGE_STRING(softFloatLinkage, "softfloat");
  READ_LINKAGE_STRING(gmpLinkage, "gmp");
  READ_LINKAGE_STRING(mpfrLinkage, "mpfr");
#undef READ_LINKAGE_STRING

  if (llvm::Error error = expectString(result.platform, "Linux-x86_64-GCC",
                                       "numeric build platform"))
    return error;
  if (llvm::Error error =
          expectString(result.softFloatSpecialization, "ARM-VFPv2-defaultNaN",
                       "SoftFloat specialization"))
    return error;
  if (llvm::Error error = expectString(result.softFloatThreadLocal,
                                       "_Thread_local", "SoftFloat TLS"))
    return error;
  if (llvm::Error error =
          expectString(result.softFloatRaiseFlags, "non-trapping",
                       "SoftFloat raise-flags policy"))
    return error;
  if (llvm::Error error = expectStringArray(result.m4ConfigureOptions,
                                            {"--disable-dependency-tracking"},
                                            "m4 configure options"))
    return error;
  if (llvm::Error error = expectStringArray(
          result.gmpConfigureOptions,
          {"--enable-shared", "--disable-static", "--with-pic"},
          "GMP configure options"))
    return error;
  if (llvm::Error error = expectStringArray(
          result.mpfrConfigureOptions,
          {"--enable-shared", "--disable-static", "--enable-thread-safe",
           "--disable-dependency-tracking", "--with-gmp=${GMP_PREFIX}"},
          "MPFR configure options"))
    return error;
  if (llvm::Error error =
          expectString(result.softFloatLinkage, "static", "SoftFloat linkage"))
    return error;
  if (llvm::Error error =
          expectString(result.gmpLinkage, "shared-only", "GMP linkage"))
    return error;
  if (llvm::Error error =
          expectString(result.mpfrLinkage, "shared-only", "MPFR linkage"))
    return error;
  if (llvm::Error error = expectString(result.managedM4RelativePath,
                                       "install/m4/bin/m4", "managed m4 path"))
    return error;
  if (llvm::Error error =
          expectString(result.pkgConfigPolicy, "disabled", "pkg-config policy"))
    return error;
  if (llvm::Error error =
          expectString(result.mpfrPatches, "", "MPFR patch set"))
    return error;
  if (llvm::Error error = expectString(result.elfValidationPolicy,
                                       "sha256-build-id-soname-needed-rpath-v1",
                                       "ELF validation policy"))
    return error;

  auto findTool =
      [&](llvm::StringRef name) -> const NumericDependencyToolRecord * {
    auto iterator = llvm::find_if(parsedResult.tools,
                                  [&](const NumericDependencyToolRecord &tool) {
                                    return tool.name == name;
                                  });
    return iterator == parsedResult.tools.end() ? nullptr : &*iterator;
  };
  auto findEnvironment =
      [&](llvm::StringRef name) -> const NumericDependencyEnvironmentRecord * {
    auto iterator = llvm::find_if(
        parsedResult.environments,
        [&](const NumericDependencyEnvironmentRecord &environment) {
          return environment.name == name;
        });
    return iterator == parsedResult.environments.end() ? nullptr : &*iterator;
  };
  for (llvm::StringRef name : kRequiredTools)
    if (!findTool(name))
      return invalid(ErrorCode::ClosureMismatch,
                     "numeric tool closure changed during parsing");

  llvm::SmallVector<std::string, 10> toolDirectories;
  for (llvm::StringRef name : kRequiredTools) {
    std::string directory =
        llvm::sys::path::parent_path(findTool(name)->resolvedPath).str();
    if (!llvm::is_contained(toolDirectories, directory))
      toolDirectories.push_back(std::move(directory));
  }
  std::string toolPath;
  for (llvm::StringRef directory : toolDirectories) {
    if (!toolPath.empty())
      toolPath.push_back(':');
    toolPath.append(directory);
  }
  constexpr llvm::StringLiteral rootToken = "${NUMERIC_ROOT}";
  llvm::StringMap<std::string> base;
  base["AR"] = findTool("ar")->resolvedPath;
  base["CC"] = findTool("cc")->resolvedPath;
  base["CXX"] = findTool("cxx")->resolvedPath;
  base["HOME"] = (rootToken + "/build/home").str();
  base["LANG"] = "C";
  base["LC_ALL"] = "C";
  base["LD"] = findTool("ld")->resolvedPath;
  base["MAKE"] = findTool("make")->resolvedPath;
  base["NM"] = findTool("nm")->resolvedPath;
  base["PATH"] = toolPath;
  base["RANLIB"] = findTool("ranlib")->resolvedPath;
  base["SHELL"] = findTool("shell")->resolvedPath;
  base["SOURCE_DATE_EPOCH"] = "0";
  base["TMPDIR"] = (rootToken + "/build/tmp").str();
  base["TZ"] = "UTC";

  llvm::StringMap<std::string> managed = base;
  managed["M4"] = (rootToken + "/install/m4/bin/m4").str();
  managed["PATH"] =
      (rootToken + "/install/m4/bin:" + llvm::Twine(toolPath)).str();
  managed["PKG_CONFIG"] = findTool("false")->resolvedPath;
  llvm::StringMap<std::string> mpfr = managed;
  mpfr["CPPFLAGS"] = ("-I" + rootToken + "/install/gmp/include").str();
  mpfr["LDFLAGS"] = ("-L" + rootToken + "/install/gmp/lib -Wl,-rpath," +
                     rootToken + "/install/gmp/lib")
                        .str();
  mpfr["LD_LIBRARY_PATH"] = (rootToken + "/install/gmp/lib").str();
  llvm::StringMap<std::string> versionRuntime = managed;
  versionRuntime["LD_LIBRARY_PATH"] =
      (rootToken + "/install/mpfr/lib:" + rootToken + "/install/gmp/lib").str();

  const std::pair<llvm::StringRef, const llvm::StringMap<std::string> *>
      expectedEnvironments[] = {{"base", &base},
                                {"managed", &managed},
                                {"mpfr", &mpfr},
                                {"version-runtime", &versionRuntime}};
  for (const auto &expected : expectedEnvironments) {
    const NumericDependencyEnvironmentRecord *actual =
        findEnvironment(expected.first);
    if (!actual)
      return invalid(ErrorCode::ClosureMismatch,
                     "numeric environment closure changed during parsing");
    if (llvm::Error error = verifyEnvironment(*actual, *expected.second))
      return error;
  }
  return parsedResult;
}

} // namespace wafer::numeric_dependency_conformance_internal
