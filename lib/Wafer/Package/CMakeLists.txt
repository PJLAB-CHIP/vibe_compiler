add_mlir_library(WaferPackageSupport
  PARTIAL_SOURCES_INTENDED

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

set(WAFER_PACKAGE_WRITER_SOURCES
  Package/Writer/Package.cpp
  Package/Writer/WriteExecutablePackage.cpp
  PARENT_SCOPE)
