add_subdirectory(Target/Layout/Core)
add_subdirectory(IR)
add_subdirectory(Target)
add_subdirectory(Frontend)
add_subdirectory(Analysis)
add_subdirectory(Conversion)
add_subdirectory(Transforms)
add_subdirectory(Package)
add_subdirectory(Runtime)
add_subdirectory(Planning)
add_subdirectory(CodeGen)
add_subdirectory(Driver)
add_subdirectory(Program)

set(WAFER_NATIVE_JIT_LIBS
  LLVM${LLVM_NATIVE_ARCH}CodeGen
  LLVM${LLVM_NATIVE_ARCH}Desc
  LLVM${LLVM_NATIVE_ARCH}Info)
foreach(_wafer_native_jit_lib IN LISTS WAFER_NATIVE_JIT_LIBS)
  if(NOT TARGET ${_wafer_native_jit_lib})
    message(FATAL_ERROR
      "target-call execution requires native LLVM backend target "
      "${_wafer_native_jit_lib}")
  endif()
endforeach()

add_mlir_library(WaferCompiler
  ${WAFER_PLANNING_SOURCES}
  ${WAFER_CODEGEN_SOURCES}
  ${WAFER_DRIVER_SOURCES}
  ${WAFER_PROGRAM_SOURCES}
  ${WAFER_PACKAGE_WRITER_SOURCES}
  ${WAFER_TARGET_EXECUTION_SOURCES}
  ${WAFER_COMPILATION_TRANSFORM_SOURCES}

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
  WaferInstrToLLVM
  WaferPackageSupport
  WaferStandaloneTileModules
  WaferStableHLOPipelines
  WaferTarget
  WaferTargetTensorMaterialization
  WaferLinalgTransforms
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
# Canonical/full feasibility reuses the current private static-packing
# contract; it does not expose Transforms headers through WaferCompiler.
target_include_directories(obj.WaferCompiler PRIVATE
  ${PROJECT_SOURCE_DIR}/lib/Wafer/Transforms)
target_include_directories(WaferCompiler PRIVATE
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

add_subdirectory(Model)
