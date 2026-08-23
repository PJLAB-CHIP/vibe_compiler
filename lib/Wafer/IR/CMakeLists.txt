add_mlir_dialect_library(WaferIR
  WaferDialect.cpp
  WaferInterfaces.cpp
  NCCCompletion.cpp
  Common/WaferIRVerification.cpp
  Instr/ComputeOps.cpp
  Instr/DTEOps.cpp
  Instr/InstructionVerification.cpp
  Instr/MovementOps.cpp
  Instr/PeripheralOps.cpp
  Instr/SyncOps.cpp
  LinalgExt/AttentionOps.cpp
  LinalgExt/CollectiveOps.cpp
  Program/ModuleOps.cpp
  Resource/SPMOps.cpp
  Target/TargetTopology.cpp
  Target/TopologyOps.cpp
  Tensor/TensorOps.cpp
  Tile/CommOps.cpp
  Tile/ComputeOps.cpp
  Tile/LayoutOps.cpp
  Tile/MoveOps.cpp
  Tile/TileRegionOps.cpp
  Tile/ViewOps.cpp

  DEPENDS
  WaferAttrsIncGen
  WaferInterfacesIncGen
  WaferOpsIncGen
  WaferPhysicalEncodingInterfacesIncGen
  WaferTypesIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRAsyncDialect
  MLIRBufferizationDialect
  MLIRBytecodeOpInterface
  MLIRControlFlowInterfaces
  MLIRDestinationStyleOpInterface
  MLIRFuncDialect
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRTensorDialect
  MLIRTilingInterface
  MLIRViewLikeInterface
  WaferTargetPhysicalLayout
)

target_include_directories(WaferIR PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Common)
if(TARGET obj.WaferIR)
  target_include_directories(obj.WaferIR PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Common)
endif()
