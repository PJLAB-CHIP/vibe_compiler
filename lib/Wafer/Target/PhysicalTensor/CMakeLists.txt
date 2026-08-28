add_mlir_library(WaferPhysicalTensor
  NumericCodec.cpp
  PhysicalLayout.cpp
  PhysicalTensorCodec.cpp
  PhysicalTensorDescriptor.cpp
  TargetFloatArithmetic.cpp
  TargetScalarConversion.cpp
  TargetTensorMaterialization.cpp

  LINK_LIBS PUBLIC
  LLVMSupport
  WaferTarget
)
