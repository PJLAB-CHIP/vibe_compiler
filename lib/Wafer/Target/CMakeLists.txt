add_mlir_library(WaferTarget
  PARTIAL_SOURCES_INTENDED
  FormalNumeric.cpp
  NumericCodec.cpp
  NumericDependencyConformance.cpp
  NumericSemantics.cpp
  TargetFormat.cpp
  TargetProfile.cpp

  LINK_LIBS PUBLIC
  LLVMSupport
)

if(WAFER_ENABLE_NUMERIC_MODEL_DEPS)
  add_mlir_library(WaferNumericModel
    PARTIAL_SOURCES_INTENDED
    FormalTensorNumeric.cpp
    MPFRNumeric.cpp
    SoftFloatOracle.cpp

    LINK_LIBS PUBLIC
    WaferTarget
    WaferNumeric::GMP
    WaferNumeric::MPFR
    WaferNumeric::SoftFloat
  )

  # add_mlir_library compiles sources through its obj.<target> object library.
  # Imported-target usage requirements reach consumers of WaferNumericModel,
  # but LLVM's helper does not propagate the SoftFloat TLS declaration to that
  # intermediate object target.  Keep the adapter declaration identical to the
  # managed ARM-VFPv2-defaultNaN library that the dependency gate built.
  target_compile_definitions(obj.WaferNumericModel PRIVATE
    THREAD_LOCAL=thread_local)
endif()
