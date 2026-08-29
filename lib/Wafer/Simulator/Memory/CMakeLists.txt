add_mlir_library(WaferSimulatorMemory
  TargetModelMemory.cpp
  TargetModelStridedMemory.cpp

  LINK_LIBS PUBLIC
  WaferCodeGen
  WaferTarget
  LLVMSupport
  MLIRIR
)
