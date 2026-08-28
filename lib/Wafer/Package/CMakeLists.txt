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

add_mlir_library(WaferPackageWriter
  Writer/Package.cpp

  LINK_LIBS PUBLIC
  LLVMSupport
  WaferAnalysis
  WaferCodeGen
  WaferPackage
  WaferPhysicalTensor
  WaferProgramData
  WaferTarget
  MLIRIR
  MLIRSupport
)

target_include_directories(obj.WaferPackageWriter PRIVATE
  ${PROJECT_SOURCE_DIR}/lib)
target_include_directories(WaferPackageWriter PRIVATE
  ${PROJECT_SOURCE_DIR}/lib)
