add_mlir_dialect_library(WaferIR
  WaferDialect.cpp
  WaferInterfaces.cpp
  Ops/AbiOps.cpp
  Ops/CommOps.cpp
  Ops/ComputeOps.cpp
  Ops/DDROps.cpp
  Ops/GroupOps.cpp
  Ops/LaunchOps.cpp
  Ops/LayoutOps.cpp
  Ops/MoveOps.cpp
  Ops/OpVerifierUtils.cpp
  Ops/PlacementOps.cpp
  Ops/SPMOps.cpp
  Ops/SyncOps.cpp
  Ops/TensorCollectiveOps.cpp
  Ops/TileRegionOps.cpp
  Ops/ViewOps.cpp

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
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRTensorDialect
  MLIRTilingInterface
)
