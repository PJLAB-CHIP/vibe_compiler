//===- CanonicalIRSnapshot.h - Structural transaction snapshot -*- C++ -*-===//

#ifndef WAFER_SUPPORT_CANONICALIRSNAPSHOT_H
#define WAFER_SUPPORT_CANONICALIRSNAPSHOT_H

#include "Wafer/Support/OptimizationAdoption.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mlir {
class Operation;
} // namespace mlir

namespace wafer {

enum class CanonicalIRSnapshotMode : uint8_t {
  SemanticStructure,
  MutationGuard,
};

struct CanonicalIRSnapshotV1 {
  uint16_t schemaVersion = 1;
  CanonicalIRSnapshotMode mode =
      CanonicalIRSnapshotMode::SemanticStructure;
  std::string rootOperationKind;
  std::vector<uint8_t> structuralBytes;
  AdoptionDigest sha256Digest{};
};

/// Encodes structure and SSA relations by canonical region/block/op/value
/// ordinals.  SemanticStructure excludes locations; MutationGuard includes the
/// complete location tree.  Unknown operations and any type, attribute or
/// property without an explicitly registered codec are rejected.
bool createCanonicalIRSnapshotV1(mlir::Operation *root,
                                 CanonicalIRSnapshotMode mode,
                                 CanonicalIRSnapshotV1 &snapshot,
                                 std::string *diagnostic = nullptr);

} // namespace wafer

#endif // WAFER_SUPPORT_CANONICALIRSNAPSHOT_H
