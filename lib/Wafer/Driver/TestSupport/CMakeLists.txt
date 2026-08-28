add_mlir_library(WaferCompilerTestSupport
  CompilerTesting.cpp

  LINK_LIBS PUBLIC
  LLVMSupport
  WaferCompiler
  WaferTransforms
)

target_include_directories(obj.WaferCompilerTestSupport PRIVATE
  ${PROJECT_SOURCE_DIR}/lib
  ${PROJECT_SOURCE_DIR}/unittests)
target_include_directories(WaferCompilerTestSupport PRIVATE
  ${PROJECT_SOURCE_DIR}/lib
  ${PROJECT_SOURCE_DIR}/unittests)
