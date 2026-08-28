add_mlir_library(WaferTarget
  PARTIAL_SOURCES_INTENDED
  Numeric/NumericCodec.cpp
  Layout/PhysicalTensorDescriptor.cpp
  Layout/PhysicalTensorCodec.cpp
  Core/RuntimeLaunchContract.cpp
  Core/TargetCall.cpp
  Core/TargetCallDecoder.cpp
  Core/TargetFormat.cpp
  Core/TargetIdentity.cpp
  Core/TargetOperation.cpp

  LINK_LIBS PUBLIC
  LLVMSupport
  WaferTargetPhysicalLayout
)

add_mlir_library(WaferTargetScalarConversion
  PARTIAL_SOURCES_INTENDED
  Numeric/TargetFloatArithmetic.cpp
  Numeric/TargetScalarConversion.cpp

  LINK_LIBS PUBLIC
  WaferTarget
  LLVMSupport
)

add_mlir_library(WaferFormalNumeric
  PARTIAL_SOURCES_INTENDED
  Numeric/Formal/FormalNumericConvert.cpp
  Numeric/Formal/FormalNumericElementwise.cpp
  Numeric/Formal/FormalNumericGemm.cpp
  Numeric/Formal/FormalNumericReduce.cpp
  Numeric/Formal/FormalNumericSupport.cpp
  Numeric/Formal/FormalNumericValidation.cpp
  Numeric/Formal/FormalOperations.cpp

  LINK_LIBS PUBLIC
  WaferTargetScalarConversion
  WaferTarget
  LLVMSupport
)

add_mlir_library(WaferTargetTensorMaterialization
  PARTIAL_SOURCES_INTENDED
  Layout/TargetTensorMaterialization.cpp

  LINK_LIBS PUBLIC
  WaferTargetScalarConversion
  WaferTarget
  LLVMSupport
)

set(WAFER_TARGET_EXECUTION_SOURCES
  Target/Execution/TargetCallExecution.cpp
  PARENT_SCOPE)

if(WAFER_ENABLE_NUMERIC_MODEL_DEPS)
  add_mlir_library(WaferNumericModel
    PARTIAL_SOURCES_INTENDED
    Numeric/Formal/FormalTensorNumeric.cpp
    Numeric/Formal/MPFRNumeric.cpp
    Numeric/Formal/SoftFloatOracle.cpp

    LINK_LIBS PUBLIC
    WaferFormalNumeric
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

if(WAFER_ENABLE_TARGET_NUMERIC_BACKEND)
  add_mlir_library(WaferOneDNNBackend
    PARTIAL_SOURCES_INTENDED
    Numeric/Qualification/QualifiedOneDNNExecution.cpp
    Numeric/Qualification/OneDNNExecutionEnvironment.cpp
    Numeric/Qualification/OneDNNQualification.cpp
    Numeric/Qualification/OneDNNQualificationCalibration.cpp
    Numeric/Qualification/OneDNNQualificationCase.cpp
    Numeric/Qualification/OneDNNQualificationComparison.cpp
    Numeric/Qualification/OneDNNQualificationExecution.cpp
    Numeric/Qualification/OneDNNQualificationPolicy.cpp
    Numeric/Qualification/OneDNNQualificationValidation.cpp
    Numeric/Qualification/OneDNNTensorCodec.cpp
    Numeric/Qualification/OneDNNTensorNumericSupport.cpp
    Numeric/Qualification/OneDNNGemmAdapter.cpp

    LINK_LIBS PUBLIC
    WaferIR
    WaferNumericModel
    WaferOneDNN::oneDNN
    LLVMTargetParser
  )
  target_compile_definitions(obj.WaferOneDNNBackend PRIVATE
    WAFER_ONEDNN_DEPENDENCY_RECORD_SHA256="${WAFER_ONEDNN_DEPS_RECORD_SHA256}"
    WAFER_ONEDNN_LIBRARY_SHA256="${WAFER_ONEDNN_BACKEND_LIBRARY_SHA256}"
    WAFER_ONEDNN_VERSION="${WAFER_ONEDNN_BACKEND_VERSION}"
    WAFER_ONEDNN_COMMIT="${WAFER_ONEDNN_BACKEND_COMMIT}"
  )
  # oneDNN's C++ wrapper reports primitive creation/execution failures with
  # exceptions.  Keep exception support local to the managed backend adapter;
  # the rest of the compiler retains LLVM's no-exceptions policy.
  if(MSVC)
    target_compile_options(obj.WaferOneDNNBackend PRIVATE /EHsc)
  else()
    target_compile_options(obj.WaferOneDNNBackend PRIVATE -fexceptions)
  endif()
endif()
