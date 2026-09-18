//===- CompilationQualification.h - Internal inspection choices --*- C++
//-*-===//
#ifndef WAFER_DRIVER_COMPILATIONQUALIFICATION_H
#define WAFER_DRIVER_COMPILATIONQUALIFICATION_H

#include "Wafer/Driver/PhysicalDataflow/BaselineCurrentIR.h"
#include "Wafer/Driver/PhysicalDataflow/UnifiedSearch.h"

#include <variant>

namespace wafer::compiler::detail {
struct SearchCandidateInspection {
  CandidateObserver observer;
};
// Selection adapters live in TestSupport. Product callers pass nullptr.
struct CompilationQualification {
  std::variant<CommunicationCandidateSelection, SearchCandidateInspection>
      choice;
};
} // namespace wafer::compiler::detail
#endif
