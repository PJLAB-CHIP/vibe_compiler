add_mlir_library(WaferCodeGen
  DeviceExecutable.cpp
  ProgramElementTypeConversion.cpp
  LLVM/CompileDeviceExecutableLLVMModules.cpp
  LLVM/TargetABIPreparation.cpp
  LLVM/TargetDeviceLink.cpp
  LLVM/TargetInstrumentation.cpp
  LLVM/TargetKernelAggregate.cpp
  LLVM/TargetLLVMModule.cpp
  LLVM/TargetLLVMTranslation.cpp
  LLVM/TargetMemoryMapping.cpp
  LLVM/TargetModuleLinking.cpp
  LLVM/TargetModuleReadback.cpp
  LLVM/TileEntryArgumentCompatibility.cpp

  LINK_LIBS PUBLIC
  LLVMSupport
  LLVMCore
  LLVMAnalysis
  LLVMBitReader
  LLVMBitWriter
  LLVMLinker
  LLVMObject
  LLVMTargetParser
  WaferAnalysis
  WaferFrontend
  WaferIR
  WaferInstrToLLVM
  WaferPackage
  WaferPhysicalTensor
  WaferProgramData
  WaferTarget
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSupport
  MLIRTargetLLVMIRExport
  MLIRLLVMToLLVMIRTranslation
)

target_include_directories(obj.WaferCodeGen PRIVATE ${PROJECT_SOURCE_DIR}/lib)
target_include_directories(WaferCodeGen PRIVATE ${PROJECT_SOURCE_DIR}/lib)
