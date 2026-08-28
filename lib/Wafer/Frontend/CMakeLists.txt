add_mlir_library(WaferFrontend
  DistributedBoundary.cpp
  DistributedSupport.cpp
  FunctionBoundary.cpp
  NpyPayload.cpp
  ParameterShards.cpp
  Program.cpp
  ProgramMetadata.cpp
  ProgramSupport.cpp

  LINK_LIBS PUBLIC
  LLVMSupport
  MLIRFuncDialect
  MLIRIR
  MLIRSupport
  WaferIR
)

target_include_directories(obj.WaferFrontend PRIVATE ${PROJECT_SOURCE_DIR}/lib)
target_include_directories(WaferFrontend PRIVATE ${PROJECT_SOURCE_DIR}/lib)

if(WAFER_ENABLE_IMPORTER_DEPS)
  add_mlir_library(WaferStableHLOProgram
    StableHLO/ProgramIngestion.cpp

    LINK_LIBS PUBLIC
    LLVMSupport
    MLIRFuncDialect
    MLIRIR
    MLIRSupport
    StablehloSerialization
    WaferFrontend
  )
endif()
