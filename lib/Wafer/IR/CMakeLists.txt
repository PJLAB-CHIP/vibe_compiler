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
  Ops/OpVerifierUtils.cpp
  Ops/PlacementOps.cpp
  Ops/SPMOps.cpp
  Ops/SyncOps.cpp
  Ops/TileRegionOps.cpp

  DEPENDS
  WaferAttrsIncGen
  WaferInterfacesIncGen
  WaferOpsIncGen
  WaferTypesIncGen

  LINK_LIBS PUBLIC
  MLIRAsyncDialect
  MLIRBytecodeOpInterface
  MLIRControlFlowInterfaces
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRSupport
)
