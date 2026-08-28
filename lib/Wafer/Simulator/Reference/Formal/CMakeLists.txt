add_mlir_library(WaferFormalNumeric
  FormalNumericConvert.cpp
  FormalNumericElementwise.cpp
  FormalNumericGemm.cpp
  FormalNumericReduce.cpp
  FormalNumericSupport.cpp
  FormalNumericValidation.cpp
  FormalOperations.cpp

  LINK_LIBS PUBLIC
  WaferPhysicalTensor
  WaferTarget
  LLVMSupport
)
