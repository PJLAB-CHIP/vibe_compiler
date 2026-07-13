//===- ReferenceInvocation.cpp - Typed rank invocation materialization ---===//

#include "Wafer/Compiler/ReferenceExecutor.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Path.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace wafer::compiler {
namespace {

llvm::Error invalid(llvm::StringRef message) {
  return llvm::createStringError(llvm::errc::invalid_argument, "%s",
                                 message.str().c_str());
}

std::optional<int64_t> elementCount(llvm::ArrayRef<int64_t> shape) {
  int64_t count = 1;
  for (int64_t dim : shape) {
    if (dim < 0 ||
        (dim != 0 && count > std::numeric_limits<int64_t>::max() / dim))
      return std::nullopt;
    count *= dim;
  }
  return count;
}

bool isSafeRelativePath(llvm::StringRef path) {
  if (path.empty() || llvm::sys::path::is_absolute(path) || path.contains('\\'))
    return false;
  for (llvm::sys::path::const_iterator current = llvm::sys::path::begin(path),
                                       end = llvm::sys::path::end(path);
       current != end; ++current)
    if (*current == "." || *current == ".." || current->empty())
      return false;
  return true;
}

llvm::Expected<ReferenceTensor>
sliceGlobalTensor(const ReferenceTensor &global,
                  const RankProgramBinding &binding) {
  if (global.getDType() != binding.dtype ||
      global.getShape() != llvm::ArrayRef<int64_t>(binding.globalShape))
    return invalid("global reference input disagrees with typed binding");
  const frontend::ProgramRankSlice &slice = binding.slice;
  const size_t rank = binding.globalShape.size();
  if (slice.offsets.size() != rank || slice.sizes.size() != rank ||
      slice.strides.size() != rank || slice.sizes != binding.localShape)
    return invalid("reference input slice has inconsistent rank or shape");

  auto globalElements = elementCount(binding.globalShape);
  auto localElements = elementCount(binding.localShape);
  if (!globalElements || !localElements ||
      (*globalElements == 0 && !global.getBytes().empty()) ||
      (*globalElements != 0 &&
       global.getBytes().size() % static_cast<size_t>(*globalElements) != 0))
    return invalid("global reference input byte geometry is invalid");
  const size_t elementBytes =
      *globalElements == 0
          ? 0
          : global.getBytes().size() / static_cast<size_t>(*globalElements);
  if (elementBytes != 0 &&
      static_cast<uint64_t>(*localElements) >
          std::numeric_limits<size_t>::max() / elementBytes)
    return invalid("local reference input byte geometry is not representable");
  std::vector<uint8_t> local(static_cast<size_t>(*localElements) *
                             elementBytes);
  if (*localElements == 0)
    return ReferenceTensor::create(binding.dtype, binding.localShape, local);

  std::vector<int64_t> coordinate(rank, 0);
  for (int64_t localLinear = 0; localLinear < *localElements; ++localLinear) {
    int64_t remaining = localLinear;
    for (size_t reverse = rank; reverse > 0; --reverse) {
      size_t dim = reverse - 1;
      coordinate[dim] = remaining % binding.localShape[dim];
      remaining /= binding.localShape[dim];
    }
    int64_t globalLinear = 0;
    for (size_t dim = 0; dim < rank; ++dim) {
      if (slice.offsets[dim] < 0 || slice.strides[dim] <= 0 ||
          coordinate[dim] >
              (std::numeric_limits<int64_t>::max() - slice.offsets[dim]) /
                  slice.strides[dim])
        return invalid("reference input slice coordinate overflows");
      int64_t globalCoordinate =
          slice.offsets[dim] + coordinate[dim] * slice.strides[dim];
      if (globalCoordinate < 0 || globalCoordinate >= binding.globalShape[dim])
        return invalid("reference input slice is outside global tensor");
      globalLinear = globalLinear * binding.globalShape[dim] + globalCoordinate;
    }
    std::memcpy(local.data() + static_cast<size_t>(localLinear) * elementBytes,
                global.getBytes().data() +
                    static_cast<size_t>(globalLinear) * elementBytes,
                elementBytes);
  }
  return ReferenceTensor::create(binding.dtype, binding.localShape, local);
}

} // namespace

llvm::Expected<std::vector<ReferenceRankInvocation>>
prepareReferenceInvocations(
    const ExecutableBundle &bundle, llvm::StringRef packageRoot,
    llvm::ArrayRef<ReferenceGlobalInputBinding> globalInputs) {
  if (packageRoot.empty())
    return invalid("reference package root must not be empty");
  if (bundle.getRankExecutables().empty())
    return invalid("reference executable domain must not be empty");
  std::vector<ReferenceRankInvocation> invocations;
  invocations.reserve(bundle.getRankExecutables().size());
  for (const RankExecutable &rank : bundle.getRankExecutables()) {
    ReferenceRankInvocation invocation;
    invocation.logicalRank = rank.getLogicalRank();
    for (const RankProgramBinding &binding : rank.getProgramBindings()) {
      if (binding.role == ProgramResourceRole::Output)
        continue;
      llvm::Expected<ReferenceTensor> tensor =
          [&]() -> llvm::Expected<ReferenceTensor> {
        if (binding.role == ProgramResourceRole::UserInput) {
          const ReferenceGlobalInputBinding *match = nullptr;
          for (const ReferenceGlobalInputBinding &input : globalInputs)
            if (input.index == binding.index) {
              if (match)
                return invalid("duplicate global reference input index");
              match = &input;
            }
          if (!match)
            return invalid("missing global reference input index");
          return sliceGlobalTensor(match->tensor, binding);
        }
        if (!isSafeRelativePath(binding.slice.payloadPath))
          return invalid("reference payload path is not safe and relative");
        llvm::SmallString<256> payload(packageRoot);
        llvm::sys::path::append(payload, binding.slice.payloadPath);
        auto loaded = ReferenceTensor::loadNpy(payload);
        if (!loaded)
          return loaded.takeError();
        if (loaded->getDType() != binding.dtype ||
            loaded->getShape() != llvm::ArrayRef<int64_t>(binding.localShape))
          return invalid("reference payload disagrees with typed rank binding");
        return loaded;
      }();
      if (!tensor)
        return tensor.takeError();
      invocation.inputs.push_back(
          {binding.role, binding.index, std::move(*tensor)});
    }
    invocations.push_back(std::move(invocation));
  }
  const auto &firstBindings =
      bundle.getRankExecutables().front().getProgramBindings();
  if (globalInputs.size() !=
      llvm::count_if(firstBindings, [](const RankProgramBinding &binding) {
        return binding.role == ProgramResourceRole::UserInput;
      }))
    return invalid("unexpected global reference input index");
  return invocations;
}

} // namespace wafer::compiler
