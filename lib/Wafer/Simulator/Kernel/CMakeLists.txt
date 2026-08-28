if(WAFER_ENABLE_NUMERIC_MODEL_DEPS)
  add_mlir_library(WaferSimulatorKernel
    TargetModelCommandValidation.cpp
    TargetModelControl.cpp
    TargetModelKernel.cpp
    TargetModelMovement.cpp
    TargetModelTensorNumeric.cpp

    LINK_LIBS PUBLIC
    WaferFormalNumeric
    WaferReferenceNumeric
    WaferPhysicalTensor
    WaferTarget
    WaferSimulatorMemory
    LLVMSupport
  )

  target_include_directories(obj.WaferSimulatorKernel PRIVATE
    ${PROJECT_SOURCE_DIR}/lib)
  target_include_directories(WaferSimulatorKernel PRIVATE
    ${PROJECT_SOURCE_DIR}/lib)
endif()
