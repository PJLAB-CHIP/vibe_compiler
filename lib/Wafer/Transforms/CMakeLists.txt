add_mlir_library(WaferTransforms
  Group/FormGroupCandidates.cpp
  StableHLOToLinalg/NormalizeConstants.cpp
  StableHLOToLinalg/LowerStablehloDot.cpp
  StableHLOToLinalg/LowerStablehloElementwise.cpp
  StableHLOToLinalg/NormalizeStablehloCollectives.cpp
  StableHLOToLinalg/LowerStablehloReduce.cpp
  StableHLOToLinalg/LowerStablehloShape.cpp
  SPMD/ApplyDefaultSpmdSharding.cpp
  Passes.cpp

  LINK_LIBS PUBLIC
  WaferIR
  MLIRArithDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRPass
  MLIRSupport
  MLIRTensorDialect
)

target_include_directories(obj.WaferTransforms PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(WaferTransforms PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(WAFER_ENABLE_IMPORTER_DEPS)
  target_compile_definitions(obj.WaferTransforms PRIVATE WAFER_ENABLE_STABLEHLO=1)
  target_compile_definitions(WaferTransforms PRIVATE WAFER_ENABLE_STABLEHLO=1)
  target_link_libraries(obj.WaferTransforms PRIVATE StablehloOps)
  target_link_libraries(WaferTransforms PRIVATE StablehloOps)
endif()

if(WAFER_ENABLE_SPMD_PARTITIONER_DEPS)
  target_compile_definitions(obj.WaferTransforms PRIVATE WAFER_ENABLE_SHARDY=1)
  target_compile_definitions(WaferTransforms PRIVATE WAFER_ENABLE_SHARDY=1)
  target_link_libraries(obj.WaferTransforms PRIVATE ShardySdyDialect)
  target_link_libraries(WaferTransforms PRIVATE ShardySdyDialect)
endif()
