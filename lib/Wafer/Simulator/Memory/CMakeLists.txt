add_mlir_library(WaferSimulatorMemory
  TargetModelMemory.cpp
  TargetModelStridedMemory.cpp

  LINK_LIBS PUBLIC
  WaferTarget
  LLVMSupport
  MLIRIR
)
