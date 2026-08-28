set(LLVM_OPTIONAL_SOURCES
  Writer/Package.cpp
  Writer/WriteExecutablePackage.cpp
)

add_mlir_library(WaferPackage

  Manifest/PackageManifest.cpp
  Manifest/PackageManifestJson.cpp
  Manifest/PackageManifestReadback.cpp
  Manifest/PackageManifestSerialization.cpp
  Manifest/PackageManifestVerification.cpp
  Profile/ProfileInstrumentationModel.cpp

  LINK_LIBS PUBLIC
  LLVMSupport
  WaferPhysicalTensor
  WaferTarget
)
