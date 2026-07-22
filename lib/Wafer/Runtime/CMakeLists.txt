add_mlir_library(WaferRuntime
  PARTIAL_SOURCES_INTENDED

  BoardRuntime.cpp
  PackageManifest.cpp
  PackageManifestJson.cpp
  PackageManifestReadback.cpp
  PackageManifestSerialization.cpp
  PackageManifestVerification.cpp
  RuntimeSessionPreflight.cpp
  Tx81ModelABI.cpp

  LINK_LIBS PUBLIC
  LLVMSupport
  WaferTarget
)
