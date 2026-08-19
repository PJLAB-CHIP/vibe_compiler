set(WAFER_NATIVE_JIT_LIBS
  LLVM${LLVM_NATIVE_ARCH}CodeGen
  LLVM${LLVM_NATIVE_ARCH}Desc
  LLVM${LLVM_NATIVE_ARCH}Info)
foreach(_wafer_native_jit_lib IN LISTS WAFER_NATIVE_JIT_LIBS)
  if(NOT TARGET ${_wafer_native_jit_lib})
    message(FATAL_ERROR
      "target-call frontend requires native LLVM backend target "
      "${_wafer_native_jit_lib}")
  endif()
endforeach()

add_subdirectory(Baseline)
add_subdirectory(Search)
add_subdirectory(Planning)
add_subdirectory(Executable)
add_subdirectory(Pipeline)
add_subdirectory(Transport)
add_subdirectory(Target)
add_subdirectory(Package)
add_subdirectory(Program)

set(LLVM_OPTIONAL_SOURCES
  ${WAFER_COMPILER_PLANNING_OPTIONAL_SOURCES}
  ${WAFER_COMPILER_TRANSPORT_OPTIONAL_SOURCES})

add_mlir_library(WaferCompiler
  ${WAFER_COMPILER_PIPELINE_SOURCES}
  ${WAFER_COMPILER_BASELINE_SOURCES}
  ${WAFER_COMPILER_SEARCH_SOURCES}
  ${WAFER_COMPILER_PLANNING_SOURCES}
  ${WAFER_COMPILER_EXECUTABLE_SOURCES}
  ${WAFER_COMPILER_TRANSPORT_SOURCES}
  ${WAFER_COMPILER_TARGET_SOURCES}
  ${WAFER_COMPILER_PACKAGE_SOURCES}
  ${WAFER_COMPILER_PROGRAM_SOURCES}

  LINK_LIBS PUBLIC
  LLVMSupport
  LLVMCore
  LLVMBitReader
  LLVMBitWriter
  LLVMExecutionEngine
  LLVMJITLink
  LLVMLinker
  LLVMObject
  LLVMOrcJIT
  LLVMRuntimeDyld
  WaferFrontend
  WaferAnalysis
  WaferIR
  WaferPipelines
  WaferPackageSupport
  WaferTarget
  WaferCardModuleToTileModules
  WaferTensorProgramToCardModule
  WaferTensorProgramToTileRegion
  WaferTileRegionToInstr
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
  MLIRTensorTransforms
  MLIRTargetLLVMIRExport
  MLIRLLVMToLLVMIRTranslation
  MLIRTransforms
  LLVMTarget
  LLVMTargetParser
  ${WAFER_NATIVE_JIT_LIBS}
)

target_include_directories(obj.WaferCompiler PRIVATE ${PROJECT_SOURCE_DIR}/lib)
target_include_directories(WaferCompiler PRIVATE ${PROJECT_SOURCE_DIR}/lib)

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
