add_mlir_library(WaferCompiler
  Compilation.cpp
  ExecutableBundle.cpp
  Package.cpp
  TargetArtifact.cpp

  LINK_LIBS PUBLIC
  LLVMSupport
  LLVMCore
  LLVMObject
  WaferFrontend
  WaferIR
  WaferPipelines
  WaferRuntime
  WaferTransforms
  MLIRArithDialect
  MLIRArithTransforms
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRBuiltinToLLVMIRTranslation
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRFuncInlinerExtension
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRLLVMDialect
  MLIRLLVMIRTransforms
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRParser
  MLIRPass
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRSupport
  MLIRTensorDialect
  MLIRTensorTransforms
  MLIRTargetLLVMIRExport
  MLIRLLVMToLLVMIRTranslation
  MLIRTransforms
)

if(WAFER_ENABLE_IMPORTER_DEPS)
  target_compile_definitions(obj.WaferCompiler PRIVATE WAFER_ENABLE_STABLEHLO=1)
  target_compile_definitions(WaferCompiler PRIVATE WAFER_ENABLE_STABLEHLO=1)
  target_link_libraries(obj.WaferCompiler PRIVATE StablehloRegister)
  target_link_libraries(WaferCompiler PRIVATE StablehloRegister)
endif()

if(WAFER_ENABLE_SPMD_PARTITIONER_DEPS)
  target_compile_definitions(obj.WaferCompiler PRIVATE WAFER_ENABLE_SHARDY=1)
  target_compile_definitions(WaferCompiler PRIVATE WAFER_ENABLE_SHARDY=1)
  target_link_libraries(obj.WaferCompiler PRIVATE ShardySdyRegister)
  target_link_libraries(WaferCompiler PRIVATE ShardySdyRegister)
endif()
