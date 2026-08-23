//===- SearchWork.h - Search work limits and counts ----------*- C++ -*-===//

#ifndef WAFER_COMPILER_SEARCH_SEARCHWORK_H
#define WAFER_COMPILER_SEARCH_SEARCHWORK_H

#include <cstdint>

namespace wafer::compiler::detail {

/// A required, invocation-local limit. The current production search always
/// has a finite complete-candidate evaluation budget.
struct SearchWorkBudget {
  static SearchWorkBudget bounded(uint64_t evaluations) {
    return {evaluations};
  }

  uint64_t maximumEvaluations = 0;
};

/// Search-local work counts. They are exposed only to an explicitly requested
/// diagnostic sink and never participate in legality or candidate identity.
struct SearchWorkCounts {
  uint64_t generated = 0;
  uint64_t evaluated = 0;
  uint64_t accepted = 0;
  uint64_t exactRejected = 0;
  uint64_t indeterminate = 0;
};

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_SEARCH_SEARCHWORK_H
