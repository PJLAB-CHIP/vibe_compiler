add_mlir_library(WaferTransforms
  CheckNormSchedule.cpp
  CheckProjectionResidualSchedule.cpp
  CheckRootTileCandidates.cpp
  CheckSoftmaxSchedule.cpp
  CheckSPMAllocation.cpp
  CompactLayoutAssignment.cpp
  FormGroups.cpp
  LowerToCAbiSkeleton.cpp
  MaterializeDDRExternalBindings.cpp
  LowerStablehloDot.cpp
  LowerStablehloElementwise.cpp
  LowerStablehloReduce.cpp
  LowerStablehloShape.cpp
  MaterializeMultiTileNoComm.cpp
  MaterializeSingleTile.cpp
  NormalizeConstants.cpp
  Passes.cpp

  LINK_LIBS PUBLIC
  WaferIR
  MLIRArithDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRPass
  MLIRSupport
  MLIRTensorDialect
)

if(WAFER_ENABLE_IMPORTER_DEPS)
  target_compile_definitions(obj.WaferTransforms PRIVATE WAFER_ENABLE_STABLEHLO=1)
  target_compile_definitions(WaferTransforms PRIVATE WAFER_ENABLE_STABLEHLO=1)
  target_link_libraries(WaferTransforms PUBLIC StablehloOps)
endif()
