//===- NumericDependencyProcess.cpp - Conformance subprocesses -*- C++ -*-===//

#include "NumericDependencyConformanceInternal.h"

#include "llvm/Support/Path.h"

#include <cerrno>
#include <cstring>
#include <string>
#include <vector>

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

namespace wafer::numeric_dependency_conformance_internal {

llvm::Expected<std::string>
runAndCapture(llvm::StringRef program, llvm::ArrayRef<std::string> arguments,
              llvm::ArrayRef<std::string> environment,
              const llvm::Twine &label) {
  constexpr size_t maximumOutputBytes = 16 * 1024 * 1024;
  const std::string stableLabel = label.str();
  const std::string programStorage = program.str();
  if (arguments.empty() || arguments.front() != program)
    return invalid(ErrorCode::InvalidArgument,
                   stableLabel + " has an invalid argument vector");

  std::vector<char *> argumentPointers;
  argumentPointers.reserve(arguments.size() + 1);
  for (const std::string &argument : arguments)
    argumentPointers.push_back(const_cast<char *>(argument.c_str()));
  argumentPointers.push_back(nullptr);
  std::vector<char *> environmentPointers;
  environmentPointers.reserve(environment.size() + 1);
  for (const std::string &entry : environment)
    environmentPointers.push_back(const_cast<char *>(entry.c_str()));
  environmentPointers.push_back(nullptr);

  int descriptors[2];
  if (::pipe(descriptors) != 0)
    return invalid(ErrorCode::IO, stableLabel +
                                      " output pipe cannot be created: " +
                                      std::strerror(errno));
  const pid_t child = ::fork();
  if (child < 0) {
    ::close(descriptors[0]);
    ::close(descriptors[1]);
    return invalid(ErrorCode::IO, stableLabel + " process cannot be created: " +
                                      std::strerror(errno));
  }
  if (child == 0) {
    ::close(descriptors[0]);
    if (::dup2(descriptors[1], STDOUT_FILENO) < 0 ||
        ::dup2(descriptors[1], STDERR_FILENO) < 0)
      ::_exit(126);
    ::close(descriptors[1]);
    ::execve(programStorage.c_str(), argumentPointers.data(),
             environmentPointers.data());
    ::_exit(127);
  }

  ::close(descriptors[1]);
  std::string output;
  output.reserve(4096);
  std::vector<char> buffer(64 * 1024);
  bool exceededLimit = false;
  bool readFailed = false;
  int readError = 0;
  while (true) {
    const ssize_t count = ::read(descriptors[0], buffer.data(), buffer.size());
    if (count < 0) {
      if (errno == EINTR)
        continue;
      readFailed = true;
      readError = errno;
      break;
    }
    if (count == 0)
      break;
    if (static_cast<size_t>(count) > maximumOutputBytes - output.size()) {
      exceededLimit = true;
      break;
    }
    output.append(buffer.data(), static_cast<size_t>(count));
  }
  ::close(descriptors[0]);
  if (exceededLimit || readFailed)
    ::kill(child, SIGKILL);

  int status = 0;
  while (::waitpid(child, &status, 0) < 0) {
    if (errno == EINTR)
      continue;
    return invalid(
        ErrorCode::IO,
        stableLabel + " process cannot be waited for: " + std::strerror(errno));
  }
  if (readFailed)
    return invalid(ErrorCode::IO, stableLabel + " output cannot be read: " +
                                      std::strerror(readError));
  if (exceededLimit)
    return invalid(ErrorCode::ResourceLimit,
                   stableLabel + " output exceeds byte limit");
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
    return invalid(ErrorCode::PolicyMismatch,
                   stableLabel + " did not exit successfully");
  return output;
}

std::vector<std::string> commandEnvironment(llvm::StringRef program) {
  return {"LC_ALL=C", "LANG=C",
          ("PATH=" + llvm::sys::path::parent_path(program)).str()};
}

} // namespace wafer::numeric_dependency_conformance_internal
