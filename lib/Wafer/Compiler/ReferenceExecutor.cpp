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
  prepare(const ExecutableBundle &bundle, int64_t logicalRank,
          TransportContract requiredTransport) {
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
    if (match->getTransportContract() != requiredTransport)
      return reference_detail::unsupported(
          requiredTransport == TransportContract::None
              ? "reference executor requires transport-free rank input"
              : "multi-rank executor requires accepted Direct DTE transport");

    auto impl = std::make_unique<ReferenceProgram::Impl>();
    impl->contextOwner = bundle.context;
    impl->logicalRank = logicalRank;
    impl->programBindings = match->getProgramBindings();
    impl->transportContract = match->getTransportContract();
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

llvm::Expected<std::vector<ReferenceRankInvocation>>
prepareReferenceInvocations(
    const ExecutableBundle &bundle, llvm::StringRef packageRoot,
    llvm::ArrayRef<ReferenceGlobalInputBinding> globalInputs) {
  return prepareProgramInvocations(bundle, packageRoot, globalInputs);
}

llvm::Expected<ReferenceProgram>
prepareReferenceRank(const ExecutableBundle &bundle, int64_t logicalRank) {
  return ReferenceProgramBuilder::prepare(bundle, logicalRank,
                                          TransportContract::None);
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

llvm::Expected<ReferenceMultiRankExecutionResult>
executeReferenceBundle(const ExecutableBundle &bundle,
                       llvm::ArrayRef<ReferenceRankInvocation> invocations,
                       ReferenceExecutionOptions options) {
  const int64_t rankCount = bundle.getExecutionConfig().getRankCount();
  if (rankCount <= 1)
    return reference_detail::invalid(
        "multi-rank reference execution requires more than one rank");
  if (bundle.getRankExecutables().size() != static_cast<size_t>(rankCount))
    return reference_detail::invalid(
        "multi-rank executable domain is incomplete");
  const TransportContract transport =
      bundle.getRankExecutables().front().getTransportContract();
  for (const RankExecutable &rank : bundle.getRankExecutables())
    if (rank.getTransportContract() != transport)
      return reference_detail::invalid(
          "multi-rank executable transport domain is not homogeneous");

  // Project the complete rank domain before importing any invocation tensor.
  // This preserves the existing capability-preflight atomicity contract.
  std::vector<ReferenceProgram> programs;
  programs.reserve(static_cast<size_t>(rankCount));
  for (int64_t rank = 0; rank < rankCount; ++rank) {
    auto program = ReferenceProgramBuilder::prepare(bundle, rank, transport);
    if (!program)
      return program.takeError();
    programs.push_back(std::move(*program));
  }
  std::vector<const ReferenceProgram::Impl *> projected;
  projected.reserve(programs.size());
  for (const ReferenceProgram &program : programs)
    projected.push_back(program.impl.get());
  return reference_detail::interpretReferencePrograms(projected, invocations,
                                                      options);
}

} // namespace wafer::compiler
