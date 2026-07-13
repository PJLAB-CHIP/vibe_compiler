//===- ReferenceExecutor.cpp - Accepted-rank semantic orchestration -----===//

#include "Wafer/Compiler/ReferenceExecutor.h"

#include "ReferenceExecutorInternal.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace wafer::compiler {

struct ReferenceProgramBuilder {
  static llvm::Expected<ReferenceProgram>
  prepare(const ExecutableBundle &bundle, int64_t logicalRank) {
    if (logicalRank < 0 ||
        logicalRank >= bundle.getExecutionConfig().getRankCount())
      return reference_detail::invalid(
          "reference logical rank is outside executable bundle");
    const RankExecutable *match = nullptr;
    for (const RankExecutable &rank : bundle.getRankExecutables()) {
      if (rank.getLogicalRank() != logicalRank)
        continue;
      if (match)
        return reference_detail::invalid(
            "executable bundle contains a duplicate logical rank");
      match = &rank;
    }
    if (!match)
      return reference_detail::invalid(
          "executable bundle is missing requested logical rank");
    if (match->getTransportContract() != TransportContract::None)
      return reference_detail::unsupported(
          "single-rank executor does not accept transport");

    auto impl = std::make_unique<ReferenceProgram::Impl>();
    impl->contextOwner = bundle.context;
    impl->logicalRank = logicalRank;
    impl->programBindings = match->getProgramBindings();
    if (llvm::Error error = reference_detail::projectReferenceProgram(
            *impl, match->getModule(), match->getEntrySymbol()))
      return std::move(error);
    return ReferenceProgram(std::move(impl));
  }
};

ReferenceProgram::ReferenceProgram(std::unique_ptr<Impl> impl)
    : impl(std::move(impl)) {}

ReferenceProgram::ReferenceProgram(ReferenceProgram &&) noexcept = default;
ReferenceProgram &
ReferenceProgram::operator=(ReferenceProgram &&) noexcept = default;
ReferenceProgram::~ReferenceProgram() = default;

int64_t ReferenceProgram::getLogicalRank() const {
  return impl ? impl->logicalRank : -1;
}

size_t ReferenceProgram::getProjectedOperationCount() const {
  return impl ? impl->projectedOperationCount : 0;
}

llvm::Expected<ReferenceTensor>
ReferenceTensor::create(llvm::StringRef dtype, llvm::ArrayRef<int64_t> shape,
                        llvm::ArrayRef<uint8_t> bytes) {
  std::optional<int64_t> expected =
      reference_detail::getCompactByteCount(dtype, shape);
  if (!expected)
    return reference_detail::invalid(
        "reference tensor has unsupported dtype or shape");
  if (*expected != static_cast<int64_t>(bytes.size()))
    return reference_detail::invalid(
        "reference tensor byte count disagrees with dtype and shape");
  return ReferenceTensor(dtype.str(), std::vector<int64_t>(shape),
                         std::vector<uint8_t>(bytes));
}

llvm::Expected<ReferenceProgram>
prepareReferenceRank(const ExecutableBundle &bundle, int64_t logicalRank) {
  return ReferenceProgramBuilder::prepare(bundle, logicalRank);
}

llvm::Expected<ReferenceExecutionResult>
executeReferenceProgram(const ReferenceProgram &program,
                        llvm::ArrayRef<ReferenceInputBinding> inputs,
                        ReferenceExecutionOptions options) {
  if (!program.impl)
    return reference_detail::invalid("reference program is moved-from");
  return reference_detail::interpretReferenceProgram(*program.impl, inputs,
                                                     options);
}

llvm::Expected<ReferenceExecutionResult>
executeReferenceRank(const ExecutableBundle &bundle, int64_t logicalRank,
                     llvm::ArrayRef<ReferenceInputBinding> inputs,
                     ReferenceExecutionOptions options) {
  auto program = prepareReferenceRank(bundle, logicalRank);
  if (!program)
    return program.takeError();
  return executeReferenceProgram(*program, inputs, options);
}

} // namespace wafer::compiler
