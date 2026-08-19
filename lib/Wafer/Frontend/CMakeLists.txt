add_mlir_library(WaferFrontend
  Distribution/DistributedBoundary.cpp
  Distribution/DistributedSupport.cpp
  Program/FunctionBoundary.cpp
  Program/NpyPayload.cpp
  Program/ParameterShards.cpp
  Program/Program.cpp
  Program/ProgramMetadata.cpp
  Program/ProgramSupport.cpp

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
