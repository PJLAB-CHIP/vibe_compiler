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

set(LLVM_OPTIONAL_SOURCES
  AttentionImplementationAlternative.cpp
  CoordinatedCommunicationAction.cpp
  RankCandidateSearch.cpp
  LowerRankInstrModules.cpp
  RankCandidateEvaluation.cpp
  StructuredCandidateDerivation.cpp
  StructuredCandidateCost.cpp
  RankCandidateSelection.cpp
  GlobalTileRelation.cpp
  NoCCommunicationAction.cpp
  NoCIntermediateRouting.cpp
  NoCPartialDataflow.cpp
  StaticFixedSlotQualification.cpp
  StructuredImplementationAlternative.cpp
  RankResourceCostValidation.cpp
)

add_mlir_library(WaferCompiler
  ExecutableCallClosure.cpp
  Compilation.cpp
  CompilationOrchestration.cpp
  CompilationStages.cpp
  CardExecutableCompilation.cpp
  CardExecutableLowering.cpp
  DirectDTETransport.cpp
  CardExecutable.cpp
  TensorProgramCompilation.cpp
  Package.cpp
  ProgramData.cpp
  ProgramDirectoryTransaction.cpp
  ProgramInvocation.cpp
  ProgramTensorComparison.cpp
  TileMemoryPlanning.cpp
  SpmdCompilationBridge.cpp
  StructuredBufferRelations.cpp
  TargetABIPreparation.cpp
  TargetLLVMModule.cpp
  TargetModuleLinking.cpp
  TargetKernelAggregate.cpp
  TargetDeviceLink.cpp
  TargetLLVMTranslation.cpp
  TargetModuleReadback.cpp
  WriteExecutablePackage.cpp
  TargetInstrumentation.cpp
  SPMCapacityEvaluation.cpp
  CompileCardExecutableLLVMModules.cpp
  SelectedBufferMaterialization.cpp
  StructuredDAGExactDemandQuery.cpp
  TargetCallFrontend.cpp
  StructuredDAGCandidateSchedule.cpp
  StructuredDAGPlacementEnumeration.cpp
  StructuredDAGEdgeDemandPlan.cpp
  StructuredDAGEdgeStrategyPlan.cpp
  StructuredDAGSchedule.cpp
  StructuredDAGSchedulePlan.cpp
  CardExecutableSynthesis.cpp
  DeterministicCardExecutableSynthesis.cpp
  ProgramResourceVerification.cpp

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
