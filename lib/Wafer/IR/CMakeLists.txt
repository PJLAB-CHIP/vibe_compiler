add_mlir_dialect_library(WaferIR
  WaferDialect.cpp
  WaferInterfaces.cpp
  Common/OpVerifierUtils.cpp
  Instr/InstructionOps.cpp
  Instr/SyncOps.cpp
  Resource/PlacementOps.cpp
  Resource/ShardOps.cpp
  Resource/SPMOps.cpp
  Runtime/LaunchOps.cpp
  Tensor/GroupOps.cpp
  Tensor/TensorCollectiveOps.cpp
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
  WaferTypesIncGen

  LINK_LIBS PUBLIC
  MLIRAsyncDialect
  MLIRBytecodeOpInterface
  MLIRControlFlowInterfaces
  MLIRDestinationStyleOpInterface
  MLIRFuncDialect
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRTensorDialect
  MLIRTilingInterface
)

target_include_directories(WaferIR PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Common)
if(TARGET obj.WaferIR)
  target_include_directories(obj.WaferIR PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Common)
endif()
