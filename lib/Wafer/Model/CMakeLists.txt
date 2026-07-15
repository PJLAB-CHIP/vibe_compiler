add_mlir_library(WaferTargetModelCore
  PARTIAL_SOURCES_INTENDED
  TargetModelInvocation.cpp
  TargetModelMemory.cpp

  LINK_LIBS PUBLIC
  WaferCompiler
  WaferTarget
  LLVMSupport
)

if(WAFER_ENABLE_NUMERIC_MODEL_DEPS)
  add_mlir_library(WaferTargetFunctionalModel
    PARTIAL_SOURCES_INTENDED
    TargetModelKernel.cpp

    LINK_LIBS PUBLIC
    WaferNumericModel
    WaferTargetModelCore
    WaferTarget
    LLVMSupport
  )
endif()

if(WAFER_ENABLE_BULK_MODEL_DEPS)
  add_mlir_library(WaferTargetBulkModel
    PARTIAL_SOURCES_INTENDED
    TargetBulkModel.cpp

    LINK_LIBS PUBLIC
    WaferBulkModel
    WaferTargetFunctionalModel
    LLVMSupport
  )
endif()

if(WAFER_ENABLE_SYSTEMC_MODEL)
  # SystemC 3.0.2's public headers require RTTI and exceptions, while the
  # LLVM/MLIR libraries in this repository are built without RTTI.  Keep the
  # SystemC C++ ABI in a plain bridge that does not include any LLVM or Wafer
  # public header.  WaferSystemCModel itself retains the repository ABI.
  add_library(WaferSystemCBridge STATIC
    SystemCBridge.cpp
  )
  target_compile_features(WaferSystemCBridge PRIVATE cxx_std_17)
  set_target_properties(WaferSystemCBridge PROPERTIES
    POSITION_INDEPENDENT_CODE ON
  )
  target_link_libraries(WaferSystemCBridge PUBLIC SystemC::systemc)
  if(MSVC)
    target_compile_options(WaferSystemCBridge PRIVATE /GR /EHsc)
  else()
    target_compile_options(WaferSystemCBridge PRIVATE -frtti -fexceptions)
  endif()

  add_mlir_library(WaferSystemCModel
    PARTIAL_SOURCES_INTENDED
    SystemCTargetModel.cpp

    LINK_LIBS PUBLIC
    WaferCompiler
    WaferTargetFunctionalModel
    WaferTargetModelCore
    WaferSystemCBridge
    LLVMSupport
  )
endif()
