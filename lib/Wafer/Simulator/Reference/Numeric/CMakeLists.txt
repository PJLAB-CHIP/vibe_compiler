if(WAFER_ENABLE_NUMERIC_MODEL_DEPS)
  add_mlir_library(WaferReferenceNumeric
    FormalTensorNumeric.cpp
    MPFRNumeric.cpp
    SoftFloatOracle.cpp

    LINK_LIBS PUBLIC
    LLVMSupport
    WaferFormalNumeric
    WaferTarget
    WaferNumeric::GMP
    WaferNumeric::MPFR
    WaferNumeric::SoftFloat
  )

  target_compile_definitions(obj.WaferReferenceNumeric PRIVATE
    THREAD_LOCAL=thread_local)
  target_include_directories(obj.WaferReferenceNumeric PRIVATE
    ${PROJECT_SOURCE_DIR}/lib)
  target_include_directories(WaferReferenceNumeric PRIVATE
    ${PROJECT_SOURCE_DIR}/lib)
endif()
