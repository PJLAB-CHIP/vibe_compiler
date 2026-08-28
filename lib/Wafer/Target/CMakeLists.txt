add_subdirectory(PhysicalTensor)

add_mlir_library(WaferTarget
  RuntimeLaunchContract.cpp
  TargetCall.cpp
  TargetCallDecoder.cpp
  TargetFormat.cpp
  TargetIdentity.cpp
  TargetOperation.cpp

  LINK_LIBS PUBLIC
  LLVMSupport
)
