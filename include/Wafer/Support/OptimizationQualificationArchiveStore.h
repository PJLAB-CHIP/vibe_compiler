//===- OptimizationQualificationArchiveStore.h - Immutable store -*- C++
//-*-===//

#ifndef WAFER_SUPPORT_OPTIMIZATIONQUALIFICATIONARCHIVESTORE_H
#define WAFER_SUPPORT_OPTIMIZATIONQUALIFICATIONARCHIVESTORE_H

#include "Wafer/Support/OptimizationQualificationArchive.h"

#include <string>

namespace wafer {

struct OptimizationSetPublicationResultV1 {
  OptimizationSetPublicationTerminalV1 terminal;
  std::optional<ActiveQualifiedOptimizationSetRefV1> activeRef;
};

struct ActiveQualifiedOptimizationSelectionV1 {
  ActiveQualifiedOptimizationSetRefV1 activeRef;
  QualifiedOptimizationSetV1 qualifiedSet;
  OptimizationSetPublicationTerminalV1 publicationTerminal;
  CompletedQualificationEvidenceV1 completedRun;
};

/// File-backed immutable run archive. A completed run is first written to an
/// unreachable staging directory, parsed and validated from disk, fsynced, and
/// then published with no-replace semantics under its canonical run digest.
class OptimizationQualificationArchiveStoreV1 {
public:
  explicit OptimizationQualificationArchiveStoreV1(std::string rootDirectory)
      : rootDirectory_(std::move(rootDirectory)) {}

  bool publishCompletedRun(const CompletedQualificationEvidenceV1 &value,
                           std::string *diagnostic = nullptr) const;

  bool readCompletedRun(const AdoptionDigest &runDigest,
                        CompletedQualificationEvidenceV1 &value,
                        std::string *diagnostic = nullptr) const;

  /// Publishes a rejected batch terminal, or stages/readbacks a fully
  /// qualified immutable set and performs one expected-active CAS. Conflicts
  /// produce a typed terminal and never replace the winner's active ref.
  bool publishOptimizationSet(
      const AdoptionDigest &runDigest,
      const std::optional<QualifiedOptimizationSetV1> &candidateSet,
      OptimizationSetPublicationResultV1 &result,
      std::string *diagnostic = nullptr) const;

  /// Production read path: reads exactly active-ref.bin and its digest-named
  /// immutable set/run. Missing or mismatched members fail closed; no latest
  /// directory scan or registry fallback is performed.
  bool loadActiveQualifiedOptimizationSet(
      ActiveQualifiedOptimizationSelectionV1 &selection,
      std::string *diagnostic = nullptr) const;

  const std::string &rootDirectory() const { return rootDirectory_; }

private:
  std::string rootDirectory_;
};

} // namespace wafer

#endif // WAFER_SUPPORT_OPTIMIZATIONQUALIFICATIONARCHIVESTORE_H
