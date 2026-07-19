add_mlir_library(WaferSupport
  OptimizationArtifactDigest.cpp
  OptimizationMechanism.cpp
  OptimizationAdoption.cpp
  OptimizationInvocation.cpp
  OptimizationQualification.cpp
  OptimizationQualificationEvidence.cpp
  OptimizationQualificationArchive.cpp
  OptimizationQualificationArchiveStore.cpp
  CanonicalIRSnapshot.cpp

  LINK_LIBS PUBLIC
  WaferIR
  MLIRIR
  MLIRSupport
)
