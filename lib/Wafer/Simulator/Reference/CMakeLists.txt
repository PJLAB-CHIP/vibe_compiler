if(WAFER_ENABLE_TARGET_NUMERIC_BACKEND)
  add_mlir_library(WaferTargetNumericBackend
    ManagedReferenceTargetModel.cpp
    TargetNumericBackend.cpp

    LINK_LIBS PUBLIC
    WaferOneDNNBackend
    WaferSimulatorKernel
    LLVMSupport
  )
endif()
