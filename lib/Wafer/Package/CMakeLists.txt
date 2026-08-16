add_mlir_library(WaferPackageSupport
  PARTIAL_SOURCES_INTENDED

  PackageManifest.cpp
  PackageManifestJson.cpp
  PackageManifestReadback.cpp
  PackageManifestSerialization.cpp
  PackageManifestVerification.cpp
  ProfileInstrumentationModel.cpp

  LINK_LIBS PUBLIC
  LLVMSupport
  WaferTarget
)
