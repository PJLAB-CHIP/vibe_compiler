if(WAFER_ENABLE_SYSTEMC_MODEL)
  add_mlir_library(WaferSystemCSimulator
    SystemCTargetModel.cpp

    LINK_LIBS PUBLIC
    WaferSystemCBridge
    WaferSimulatorKernel
    WaferSimulatorInvocation
    WaferSimulatorMemory
    LLVMSupport
  )

  target_include_directories(obj.WaferSystemCSimulator PRIVATE
    ${PROJECT_SOURCE_DIR}/lib)
  target_include_directories(WaferSystemCSimulator PRIVATE
    ${PROJECT_SOURCE_DIR}/lib)
endif()
