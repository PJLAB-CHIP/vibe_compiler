add_subdirectory(StandaloneTileModules)
add_subdirectory(ProgramData)

add_mlir_library(WaferCompiler
  Compilation.cpp
  CompilationOrchestration.cpp
  CompilationStages.cpp
  CurrentIRExecutablePipeline.cpp
  DeviceExecutableConstruction.cpp
  ExecutableCompilation.cpp
  ExecutableLowering.cpp
  PackageAssembly.cpp
  PhysicalDataflow/ActualResultController.cpp
  PhysicalDataflow/BaselineCurrentIR.cpp
  PhysicalDataflow/SearchCurrentIR.cpp
  PhysicalDataflow/StructuredProgramAnalysis.cpp
  PhysicalDataflow/UnifiedSearch.cpp
  ProgramDirectoryTransaction.cpp
  ProgramResourceVerification.cpp
  SpmdCompilationBridge.cpp
  TensorProgramCompilation.cpp

  LINK_LIBS PUBLIC
  LLVMSupport
  LLVMCore
  LLVMBitReader
  LLVMBitWriter
  LLVMLinker
  LLVMObject
  WaferAnalysis
  WaferCodeGen
  WaferFrontend
  WaferIR
  WaferInstrToLLVM
  WaferLinalgTransforms
  WaferPackage
  WaferPackageWriter
  WaferPhysicalTensor
  WaferPlanning
  WaferProgramData
  WaferStableHLOPipelines
  WaferStandaloneTileModules
  WaferTarget
  WaferTileToInstr
  WaferTileToInstrPipelines
  WaferTransforms
  MLIRAffineToStandard
  MLIRAffineDialect
  MLIRArithDialect
  MLIRArithTransforms
  MLIRArithValueBoundsOpInterfaceImpl
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRBuiltinToLLVMIRTranslation
  MLIRBytecodeWriter
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
  MLIRTensorInferTypeOpInterfaceImpl
  MLIRTensorTransforms
  MLIRTargetLLVMIRExport
  MLIRLLVMToLLVMIRTranslation
  MLIRTransforms
  LLVMTarget
  LLVMTargetParser
)

target_include_directories(obj.WaferCompiler PRIVATE
  ${PROJECT_SOURCE_DIR}/lib
  ${PROJECT_SOURCE_DIR}/lib/Wafer/Transforms)
target_include_directories(WaferCompiler PRIVATE
  ${PROJECT_SOURCE_DIR}/lib
  ${PROJECT_SOURCE_DIR}/lib/Wafer/Transforms)

if(WAFER_ENABLE_IMPORTER_DEPS)
  target_compile_definitions(obj.WaferCompiler PRIVATE WAFER_ENABLE_STABLEHLO=1)
  target_compile_definitions(WaferCompiler PRIVATE WAFER_ENABLE_STABLEHLO=1)
  target_link_libraries(obj.WaferCompiler PRIVATE StablehloRegister)
  target_link_libraries(obj.WaferCompiler PRIVATE WaferStableHLOProgram)
  target_link_libraries(WaferCompiler PRIVATE StablehloRegister
                                              WaferStableHLOProgram)
endif()

if(WAFER_ENABLE_SPMD_PARTITIONER_DEPS)
  target_compile_definitions(obj.WaferCompiler PRIVATE WAFER_ENABLE_SHARDY=1)
  target_compile_definitions(WaferCompiler PRIVATE WAFER_ENABLE_SHARDY=1)
  target_link_libraries(obj.WaferCompiler PRIVATE ShardySdyRegister)
  target_link_libraries(WaferCompiler PRIVATE ShardySdyRegister)
endif()

if(WAFER_ENABLE_UNIT_TESTS)
  add_subdirectory(TestSupport)
endif()
