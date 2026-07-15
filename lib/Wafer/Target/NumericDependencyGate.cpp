//===- NumericDependencyGate.cpp - Conformance gate contracts -*- C++ -*-===//

#include "NumericDependencyConformanceInternal.h"

#include <string>
#include <utility>
#include <vector>

namespace wafer::numeric_dependency_conformance_internal {

struct GateContract {
  std::vector<std::string> command;
  std::string cwd;
  std::string environment;
};

llvm::Expected<GateContract>
expectedGateContract(const NumericDependencyConformanceRecord &record,
                     llvm::StringRef name) {
  constexpr llvm::StringLiteral root = "${NUMERIC_ROOT}";
  const NumericDependencyToolIdentity *make = record.findTool("make");
  const NumericDependencyToolIdentity *cc = record.findTool("cc");
  const NumericDependencySourceIdentity *m4Source = record.findSource("m4");
  const NumericDependencySourceIdentity *gmpSource = record.findSource("gmp");
  const NumericDependencySourceIdentity *mpfrSource = record.findSource("mpfr");
  const NumericDependencyArtifactIdentity *gmp = record.findArtifact("gmp");
  const NumericDependencyArtifactIdentity *mpfr = record.findArtifact("mpfr");
  if (!make || !cc || !m4Source || !gmpSource || !mpfrSource || !gmp || !mpfr)
    return invalid(ErrorCode::ClosureMismatch,
                   "gate contract inputs are incomplete");
  const std::string jobs =
      "-j" + std::to_string(record.getBuildIdentity().jobs);

  struct AutotoolsPolicy {
    llvm::StringRef dependency;
    const NumericDependencySourceIdentity *source;
    llvm::ArrayRef<std::string> options;
    llvm::StringRef environment;
  };
  const AutotoolsPolicy dependencies[] = {
      {"m4", m4Source, record.getBuildIdentity().m4ConfigureOptions, "base"},
      {"gmp", gmpSource, record.getBuildIdentity().gmpConfigureOptions,
       "managed"},
      {"mpfr", mpfrSource, record.getBuildIdentity().mpfrConfigureOptions,
       "mpfr"},
  };
  for (const AutotoolsPolicy &dependency : dependencies) {
    const std::string prefix = dependency.dependency.str() + "-";
    const std::string cwd = (root + "/build/" + dependency.dependency).str();
    if (name == prefix + "configure") {
      std::vector<std::string> command = {
          (root + "/" + dependency.source->sourceRelativePath + "/configure")
              .str(),
          ("--prefix=" + root + "/install/" + dependency.dependency).str()};
      for (const std::string &rawOption : dependency.options) {
        std::string option = rawOption;
        const llvm::StringRef marker = "${GMP_PREFIX}";
        const size_t position = option.find(marker.str());
        if (position != std::string::npos)
          option.replace(position, marker.size(),
                         (root + "/install/gmp").str());
        command.push_back(std::move(option));
      }
      return GateContract{std::move(command), cwd,
                          dependency.environment.str()};
    }
    if (name == prefix + "build")
      return GateContract{
          {make->resolvedPath, jobs}, cwd, dependency.environment.str()};
    if (name == prefix + "check")
      return GateContract{{make->resolvedPath, jobs, "check"},
                          cwd,
                          dependency.environment.str()};
    if (name == prefix + "install")
      return GateContract{
          {make->resolvedPath, "install"}, cwd, dependency.environment.str()};
  }

  if (name == "m4-version")
    return GateContract{
        {(root + "/install/m4/bin/m4").str(), "--version"}, root.str(), "base"};
  if (name == "softfloat-build")
    return GateContract{
        {make->resolvedPath, jobs,
         ("SOURCE_DIR=" + root + "/sources/SoftFloat-3e/source").str(),
         "SPECIALIZE_TYPE=ARM-VFPv2-defaultNaN"},
        (root + "/build/softfloat").str(),
        "managed"};
  if (name == "testfloat-build")
    return GateContract{
        {make->resolvedPath, jobs, "testsoftfloat",
         ("SOURCE_DIR=" + root + "/sources/TestFloat-3e/source").str(),
         ("SOFTFLOAT_INCLUDE_DIR=" + root +
          "/sources/SoftFloat-3e/source/include")
             .str(),
         ("SOFTFLOAT_LIB=" + root + "/build/softfloat/softfloat.a").str()},
        (root + "/build/testfloat").str(),
        "managed"};
  if (name == "softfloat-policy-compile")
    return GateContract{
        {cc->resolvedPath, "-std=c11", "-DTHREAD_LOCAL=_Thread_local",
         ("-I" + root + "/install/softfloat/include").str(),
         (root + "/build/softfloat/wafer-softfloat-policy-probe.c").str(),
         (root + "/install/softfloat/lib/libsoftfloat.a").str(), "-pthread",
         "-o", (root + "/build/softfloat/wafer-softfloat-policy-probe").str()},
        root.str(),
        "managed"};
  if (name == "softfloat-tls-default-nan")
    return GateContract{
        {(root + "/build/softfloat/wafer-softfloat-policy-probe").str()},
        root.str(),
        "managed"};

  llvm::StringRef selector;
  if (name == "testsoftfloat-f16-mulAdd")
    selector = "f16_mulAdd";
  else if (name == "testsoftfloat-f32-mulAdd")
    selector = "f32_mulAdd";
  else if (name == "testsoftfloat-all1")
    selector = "-all1";
  else if (name == "testsoftfloat-all2")
    selector = "-all2";
  if (!selector.empty())
    return GateContract{{(root + "/install/testfloat/bin/testsoftfloat").str(),
                         "-seed", "1", "-level", "1", "-errorstop",
                         "-rnear_even", "-tininessafter", selector.str()},
                        root.str(),
                        "managed"};

  if (name == "managed-version-thread-safe-compile")
    return GateContract{
        {cc->resolvedPath, ("-I" + root + "/install/gmp/include").str(),
         ("-I" + root + "/install/mpfr/include").str(),
         (root + "/build/managed-version-thread-safe.c").str(),
         (root + "/" + mpfr->relativePath).str(),
         (root + "/" + gmp->relativePath).str(),
         ("-Wl,-rpath," + root + "/install/mpfr/lib").str(),
         ("-Wl,-rpath," + root + "/install/gmp/lib").str(), "-pthread", "-o",
         (root + "/build/managed-version-thread-safe").str()},
        root.str(),
        "managed"};
  if (name == "managed-version-thread-safe")
    return GateContract{{(root + "/build/managed-version-thread-safe").str()},
                        root.str(),
                        "version-runtime"};
  return invalid(ErrorCode::ClosureMismatch,
                 "unknown numeric conformance gate " + name);
}

llvm::Error
validateGateContract(const NumericDependencyConformanceRecord &record,
                     const NumericDependencyConformanceGateIdentity &gate) {
  if (gate.exitCode != 0)
    return invalid(ErrorCode::PolicyMismatch,
                   "numeric conformance gate exit_code is not zero");
  llvm::Expected<GateContract> expected =
      expectedGateContract(record, gate.name);
  if (!expected)
    return expected.takeError();
  if (gate.command != expected->command || gate.cwd != expected->cwd ||
      gate.environment != expected->environment)
    return invalid(ErrorCode::PolicyMismatch,
                   "numeric conformance gate invocation mismatch for " +
                       gate.name);
  if (!record.findEnvironment(gate.environment))
    return invalid(ErrorCode::ClosureMismatch,
                   "numeric conformance gate references unknown environment");
  return llvm::Error::success();
}

} // namespace wafer::numeric_dependency_conformance_internal
