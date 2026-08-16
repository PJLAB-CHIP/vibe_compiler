//===- WaferRunBoardIO.cpp - Board invocation file bindings -------------===//

#include "WaferRunBoardIO.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace wafer::runtime::cli {
namespace {

llvm::Expected<std::vector<uint8_t>> readRawFile(llvm::StringRef path,
                                                 uint64_t expectedBytes,
                                                 llvm::StringRef portLabel) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path, /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false);
  if (!buffer)
    return llvm::createStringError(buffer.getError(),
                                   "failed to read raw tensor file: " + path);
  llvm::StringRef bytes = (*buffer)->getBuffer();
  if (bytes.size() != expectedBytes)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "raw tensor file byte count does not match " + portLabel + ": " +
            path);
  return std::vector<uint8_t>(reinterpret_cast<const uint8_t *>(bytes.data()),
                              reinterpret_cast<const uint8_t *>(bytes.data()) +
                                  bytes.size());
}

llvm::Expected<llvm::DenseMap<uint64_t, std::string>>
indexPortFiles(llvm::ArrayRef<PortFile> files, llvm::StringRef option) {
  llvm::DenseMap<uint64_t, std::string> indexed;
  for (const PortFile &file : files) {
    if (file.path.empty())
      return llvm::createStringError(llvm::errc::invalid_argument,
                                     "empty raw file path for " + option);
    if (!indexed.try_emplace(file.portId, file.path).second)
      return llvm::createStringError(llvm::errc::invalid_argument,
                                     "duplicate port id for " + option);
  }
  return indexed;
}

llvm::Expected<std::string> normalizeOutputPath(llvm::StringRef path) {
  llvm::SmallString<256> absolute(path);
  if (std::error_code error = llvm::sys::fs::make_absolute(absolute))
    return llvm::createStringError(
        error, "failed to make raw output path absolute: " + path);

  llvm::StringRef filename = llvm::sys::path::filename(absolute);
  llvm::StringRef parent = llvm::sys::path::parent_path(absolute);
  if (filename.empty() || filename == "." || filename == ".." || parent.empty())
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "raw output path has no file name or parent");

  llvm::SmallString<256> canonicalParent;
  if (std::error_code error = llvm::sys::fs::real_path(parent, canonicalParent))
    return llvm::createStringError(
        error, "failed to resolve raw output parent directory: " + parent);

  llvm::SmallString<256> canonical(canonicalParent);
  llvm::sys::path::append(canonical, filename);
  if (llvm::sys::fs::get_file_type(canonical, /*Follow=*/false) ==
      llvm::sys::fs::file_type::directory_file)
    return llvm::createStringError(
        llvm::errc::is_a_directory,
        "raw output path names an existing directory: " + canonical);
  return canonical.str().str();
}

struct StagedRawFile {
  std::string target;
  llvm::SmallString<256> temporary;
};

uint16_t readLittleEndianU16(llvm::ArrayRef<uint8_t> bytes, size_t offset) {
  return static_cast<uint16_t>(bytes[offset]) |
         (static_cast<uint16_t>(bytes[offset + 1]) << 8);
}

bool isFiniteF16(uint16_t bits) { return ((bits >> 10) & 0x1f) != 0x1f; }

double decodeFiniteF16(uint16_t bits) {
  const bool negative = (bits & UINT16_C(0x8000)) != 0;
  const uint16_t exponent = (bits >> 10) & UINT16_C(0x1f);
  const uint16_t fraction = bits & UINT16_C(0x03ff);
  double value =
      exponent == 0
          ? std::ldexp(static_cast<double>(fraction), -24)
          : std::ldexp(static_cast<double>(UINT16_C(0x0400) + fraction),
                       static_cast<int>(exponent) - 25);
  return negative ? -value : value;
}

uint32_t orderedF16(uint16_t bits) {
  constexpr uint32_t sign = UINT32_C(0x8000);
  return (bits & UINT16_C(0x8000))
             ? sign - static_cast<uint32_t>(bits & UINT16_C(0x7fff))
             : sign + static_cast<uint32_t>(bits);
}

llvm::Error validateRelaxedF16Output(uint64_t portId,
                                     llvm::ArrayRef<uint8_t> expected,
                                     llvm::ArrayRef<uint8_t> actual) {
  for (size_t offset = 0; offset < expected.size(); offset += 2) {
    const uint16_t expectedBits = readLittleEndianU16(expected, offset);
    const uint16_t actualBits = readLittleEndianU16(actual, offset);
    const size_t element = offset / 2;
    if (!isFiniteF16(expectedBits) || !isFiniteF16(actualBits))
      return llvm::createStringError(
          llvm::errc::result_out_of_range,
          "relaxed f16 board output contains NaN or infinity for port " +
              std::to_string(portId) + " at element " +
              std::to_string(element));

    const double expectedValue = decodeFiniteF16(expectedBits);
    const double actualValue = decodeFiniteF16(actualBits);
    if (expectedValue == actualValue)
      continue;

    const double absoluteError = std::abs(expectedValue - actualValue);
    const double limit = kRelaxedF16AbsoluteTolerance +
                         kRelaxedF16RelativeTolerance * std::abs(expectedValue);
    const uint32_t ulpDistance =
        orderedF16(expectedBits) > orderedF16(actualBits)
            ? orderedF16(expectedBits) - orderedF16(actualBits)
            : orderedF16(actualBits) - orderedF16(expectedBits);
    if (absoluteError <= limit && ulpDistance <= kRelaxedF16MaximumUlp)
      continue;

    std::string detail;
    llvm::raw_string_ostream message(detail);
    message << "relaxed f16 board output differs for port " << portId
            << " at element " << element << ": expected=" << expectedValue
            << " actual=" << actualValue << " abs=" << absoluteError
            << " limit=" << limit << " ulp=" << ulpDistance;
    return llvm::createStringError(llvm::errc::result_out_of_range,
                                   message.str());
  }
  return llvm::Error::success();
}

llvm::Error writeRawFilesViaTemporaryFiles(
    llvm::ArrayRef<std::pair<llvm::StringRef, llvm::ArrayRef<uint8_t>>>
        captures) {
  std::vector<StagedRawFile> staged;
  staged.reserve(captures.size());
  auto discardStaged = [&]() {
    for (const StagedRawFile &file : staged)
      if (!file.temporary.empty())
        (void)llvm::sys::fs::remove(file.temporary);
  };

  for (auto [target, bytes] : captures) {
    llvm::SmallString<256> model(target);
    model += ".tmp-%%%%%%";
    int descriptor = -1;
    llvm::SmallString<256> temporary;
    std::error_code error =
        llvm::sys::fs::createUniqueFile(model, descriptor, temporary);
    if (error) {
      discardStaged();
      return llvm::createStringError(
          error, "failed to stage raw output file: " + target);
    }
    staged.push_back({target.str(), temporary});
    llvm::raw_fd_ostream output(descriptor, /*shouldClose=*/true);
    output.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    output.close();
    if (output.has_error()) {
      std::error_code outputError = output.error();
      discardStaged();
      return llvm::createStringError(
          outputError, "failed to write staged raw output file: " + target);
    }
  }

  for (StagedRawFile &file : staged) {
    if (std::error_code error =
            llvm::sys::fs::rename(file.temporary, file.target)) {
      discardStaged();
      return llvm::createStringError(
          error, "failed to replace raw output file: " + file.target);
    }
    file.temporary.clear();
  }
  return llvm::Error::success();
}

} // namespace

llvm::Expected<BoardInvocationFilePlan> prepareBoardInvocationFiles(
    const PackageManifest &manifest, BoardRuntimeInvocationRequest request,
    llvm::ArrayRef<PortFile> inputFiles, llvm::ArrayRef<PortFile> expectedFiles,
    llvm::ArrayRef<PortFile> outputFiles,
    llvm::ArrayRef<PortFile> relaxedF16ExpectedFiles) {
  llvm::Expected<llvm::DenseMap<uint64_t, std::string>> inputs =
      indexPortFiles(inputFiles, "--resource");
  if (!inputs)
    return inputs.takeError();
  llvm::Expected<llvm::DenseMap<uint64_t, std::string>> expected =
      indexPortFiles(expectedFiles, "--expected");
  if (!expected)
    return expected.takeError();
  llvm::Expected<llvm::DenseMap<uint64_t, std::string>> relaxedF16Expected =
      indexPortFiles(relaxedF16ExpectedFiles, "--expected-f16-relaxed");
  if (!relaxedF16Expected)
    return relaxedF16Expected.takeError();
  for (const auto &entry : *relaxedF16Expected)
    if (expected->contains(entry.first))
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "duplicate port id across --expected and --expected-f16-relaxed");
  llvm::Expected<llvm::DenseMap<uint64_t, std::string>> outputs =
      indexPortFiles(outputFiles, "--output");
  if (!outputs)
    return outputs.takeError();

  std::set<std::string> distinctOutputPaths;
  for (auto &entry : *outputs) {
    llvm::Expected<std::string> canonical = normalizeOutputPath(entry.second);
    if (!canonical)
      return canonical.takeError();
    entry.second = std::move(*canonical);
    if (!distinctOutputPaths.insert(entry.second).second)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "multiple --output ports use the same raw file path");
  }

  BoardInvocationFilePlan plan;
  plan.request = std::move(request);
  plan.outputPaths = *outputs;
  for (const ExternalPortRecord &port : manifest.inputs) {
    auto source = inputs->find(port.id.getValue());
    if (source == inputs->end())
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "--resource omits input port " +
              std::to_string(port.id.getValue()));
    llvm::Expected<std::vector<uint8_t>> loaded = readRawFile(
        source->second, port.bytes,
        (llvm::Twine("input port ") + llvm::Twine(port.id.getValue())).str());
    if (!loaded)
      return loaded.takeError();
    plan.request.bindings.push_back({port.id, std::move(*loaded)});
    inputs->erase(source);
  }
  for (const ExternalPortRecord &port : manifest.outputs) {
    const uint64_t portId = port.id.getValue();
    plan.outputBytes[portId] = port.bytes;
    auto reference = expected->find(portId);
    auto relaxedReference = relaxedF16Expected->find(portId);
    auto capture = outputs->find(portId);
    if (reference == expected->end() &&
        relaxedReference == relaxedF16Expected->end() &&
        capture == outputs->end())
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "--expected/--output omit output port " + std::to_string(portId));
    const bool useRelaxedF16 = relaxedReference != relaxedF16Expected->end();
    if (useRelaxedF16 && (port.dtype != "f16" || port.bytes % 2 != 0))
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "--expected-f16-relaxed requires an f16 output port");
    if (reference != expected->end() || useRelaxedF16) {
      const std::string &path =
          useRelaxedF16 ? relaxedReference->second : reference->second;
      llvm::Expected<std::vector<uint8_t>> loaded = readRawFile(
          path, port.bytes,
          (llvm::Twine("output port ") + llvm::Twine(portId)).str());
      if (!loaded)
        return loaded.takeError();
      plan.expectedBytes[portId] = std::move(*loaded);
      plan.expectedComparisons[portId] =
          useRelaxedF16 ? BoardOutputComparisonKind::RelaxedF16
                        : BoardOutputComparisonKind::Exact;
      if (useRelaxedF16)
        relaxedF16Expected->erase(relaxedReference);
      else
        expected->erase(reference);
    }
    if (capture != outputs->end())
      outputs->erase(capture);
  }
  if (!inputs->empty() || !expected->empty() ||
      !relaxedF16Expected->empty() || !outputs->empty())
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "board invocation contains port ids outside the package");
  return plan;
}

llvm::Error validateBoardOutputs(llvm::ArrayRef<BoardRuntimeOutput> outputs,
                                 const BoardInvocationFilePlan &plan) {
  llvm::DenseSet<uint64_t> returnedPorts;
  for (const BoardRuntimeOutput &output : outputs) {
    const uint64_t portId = output.port.getValue();
    if (!plan.outputBytes.contains(portId) ||
        !returnedPorts.insert(portId).second)
      return llvm::createStringError(
          llvm::errc::result_out_of_range,
          "board returned an unexpected or duplicate output port " +
              std::to_string(portId));
    if (output.bytes.size() != plan.outputBytes.lookup(portId))
      return llvm::createStringError(
          llvm::errc::result_out_of_range,
          "board returned a wrong-sized output port " +
              std::to_string(portId));
    auto reference = plan.expectedBytes.find(portId);
    if (reference == plan.expectedBytes.end())
      continue;
    auto comparison = plan.expectedComparisons.find(portId);
    if (comparison == plan.expectedComparisons.end())
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "board expected output has no typed comparison policy for port " +
              std::to_string(portId));
    if (comparison->second == BoardOutputComparisonKind::RelaxedF16) {
      if (llvm::Error error =
              validateRelaxedF16Output(portId, reference->second, output.bytes))
        return error;
      continue;
    }
    if (reference->second != output.bytes) {
      auto mismatch = llvm::mismatch(reference->second, output.bytes);
      size_t offset =
          static_cast<size_t>(mismatch.first - reference->second.begin());
      return llvm::createStringError(
          llvm::errc::result_out_of_range,
          "complete board output differs for port " + std::to_string(portId) +
              " at byte " + std::to_string(offset) + ": expected=0x" +
              llvm::utohexstr(*mismatch.first) + " actual=0x" +
              llvm::utohexstr(*mismatch.second));
    }
  }
  if (returnedPorts.size() != plan.outputBytes.size())
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "board result did not return all-and-only output ports");
  return llvm::Error::success();
}

llvm::Expected<BoardInvocationFilePlan>
remapBoardInvocationFilePlan(const BoardInvocationFilePlan &sourcePlan,
                             const PackageManifest &sourceManifest,
                             const PackageManifest &targetManifest) {
  if (sourcePlan.request.profilerRecordBytes)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "cannot remap a board invocation that already has profiler records");

  // Stable external port identity: the program-boundary role index within the
  // input/output domain. Port ids are dense per package and may differ.
  llvm::DenseMap<int64_t, uint64_t> inputByRoleIndex;
  for (const ExternalPortRecord &port : sourceManifest.inputs)
    inputByRoleIndex[port.roleIndex] = port.id.getValue();
  llvm::DenseMap<int64_t, uint64_t> outputByRoleIndex;
  for (const ExternalPortRecord &port : sourceManifest.outputs)
    outputByRoleIndex[port.roleIndex] = port.id.getValue();
  llvm::DenseMap<int64_t, uint64_t> targetInputByRoleIndex;
  for (const ExternalPortRecord &port : targetManifest.inputs) {
    auto source = inputByRoleIndex.find(port.roleIndex);
    if (source == inputByRoleIndex.end() || port.bytes == 0)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "profile package input port domain differs from the production "
          "package");
    targetInputByRoleIndex[port.roleIndex] = port.id.getValue();
  }
  llvm::DenseMap<int64_t, uint64_t> targetOutputByRoleIndex;
  for (const ExternalPortRecord &port : targetManifest.outputs) {
    auto source = outputByRoleIndex.find(port.roleIndex);
    if (source == outputByRoleIndex.end() || port.bytes == 0)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "profile package output port domain differs from the production "
          "package");
    targetOutputByRoleIndex[port.roleIndex] = port.id.getValue();
  }
  if (targetManifest.inputs.size() != sourceManifest.inputs.size() ||
      targetManifest.outputs.size() != sourceManifest.outputs.size())
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "profile package external port domain differs from the production "
        "package");

  auto remapInput = [&](uint64_t sourcePort) -> llvm::Expected<uint64_t> {
    if (sourcePort >= sourceManifest.inputs.size())
      return llvm::createStringError(llvm::errc::invalid_argument,
                                     "prepared board invocation references an "
                                     "unknown production input port");
    const int64_t roleIndex = sourceManifest.inputs[sourcePort].roleIndex;
    auto target = targetInputByRoleIndex.find(roleIndex);
    if (target == targetInputByRoleIndex.end())
      return llvm::createStringError(llvm::errc::invalid_argument,
                                     "production input port has no profile "
                                     "package counterpart");
    return target->second;
  };
  auto remapOutput = [&](uint64_t sourcePort) -> llvm::Expected<uint64_t> {
    if (sourcePort >= sourceManifest.outputs.size())
      return llvm::createStringError(llvm::errc::invalid_argument,
                                     "prepared board invocation references an "
                                     "unknown production output port");
    const int64_t roleIndex = sourceManifest.outputs[sourcePort].roleIndex;
    auto target = targetOutputByRoleIndex.find(roleIndex);
    if (target == targetOutputByRoleIndex.end())
      return llvm::createStringError(llvm::errc::invalid_argument,
                                     "production output port has no profile "
                                     "package counterpart");
    return target->second;
  };

  BoardInvocationFilePlan targetPlan;
  targetPlan.request.deviceId = sourcePlan.request.deviceId;
  targetPlan.request.completionTimeoutMilliseconds =
      sourcePlan.request.completionTimeoutMilliseconds;
  targetPlan.request.qualification = sourcePlan.request.qualification;
  targetPlan.request.bindings.reserve(sourcePlan.request.bindings.size());
  for (const BoardRuntimeBinding &binding : sourcePlan.request.bindings) {
    llvm::Expected<uint64_t> target = remapInput(binding.port.getValue());
    if (!target)
      return target.takeError();
    targetPlan.request.bindings.push_back({PortId(*target), binding.bytes});
  }

  auto remapVectorMap =
      [&](const llvm::DenseMap<uint64_t, std::vector<uint8_t>> &source,
          llvm::DenseMap<uint64_t, std::vector<uint8_t>> &target)
      -> llvm::Error {
    for (const auto &entry : source) {
      llvm::Expected<uint64_t> targetId = remapOutput(entry.first);
      if (!targetId)
        return targetId.takeError();
      target[*targetId] = entry.second;
    }
    return llvm::Error::success();
  };
  if (llvm::Error error =
          remapVectorMap(sourcePlan.expectedBytes, targetPlan.expectedBytes))
    return std::move(error);
  for (const auto &entry : sourcePlan.expectedComparisons) {
    llvm::Expected<uint64_t> targetId = remapOutput(entry.first);
    if (!targetId)
      return targetId.takeError();
    targetPlan.expectedComparisons[*targetId] = entry.second;
  }
  for (const auto &entry : sourcePlan.outputPaths) {
    llvm::Expected<uint64_t> targetId = remapOutput(entry.first);
    if (!targetId)
      return targetId.takeError();
    targetPlan.outputPaths[*targetId] = entry.second;
  }
  for (const auto &entry : sourcePlan.outputBytes) {
    llvm::Expected<uint64_t> targetId = remapOutput(entry.first);
    if (!targetId)
      return targetId.takeError();
    targetPlan.outputBytes[*targetId] = entry.second;
  }
  return targetPlan;
}

llvm::Error
validateAndWriteBoardOutputs(llvm::ArrayRef<BoardRuntimeOutput> outputs,
                             const BoardInvocationFilePlan &plan) {
  if (llvm::Error error = validateBoardOutputs(outputs, plan))
    return error;

  std::vector<std::pair<llvm::StringRef, llvm::ArrayRef<uint8_t>>> captures;
  captures.reserve(plan.outputPaths.size());
  for (const BoardRuntimeOutput &output : outputs) {
    auto capture = plan.outputPaths.find(output.port.getValue());
    if (capture != plan.outputPaths.end())
      captures.emplace_back(capture->second, output.bytes);
  }
  return writeRawFilesViaTemporaryFiles(captures);
}

} // namespace wafer::runtime::cli
