add_mlir_library(WaferTarget
  PARTIAL_SOURCES_INTENDED
  Numeric/Formal/FormalNumericConvert.cpp
  Numeric/Formal/FormalNumericElementwise.cpp
  Numeric/Formal/FormalNumericGemm.cpp
  Numeric/Formal/FormalNumericReduce.cpp
  Numeric/Formal/FormalNumericSupport.cpp
  Numeric/Formal/FormalNumericValidation.cpp
  Numeric/NumericCapability.cpp
  Numeric/NumericCodec.cpp
  Numeric/NumericCommand.cpp
  Numeric/Dependency/NumericDependencyBuildConfig.cpp
  Numeric/Dependency/NumericDependencyConformance.cpp
  Numeric/Dependency/NumericDependencyELF.cpp
  Numeric/Dependency/NumericDependencyFilesystem.cpp
  Numeric/Dependency/NumericDependencyGate.cpp
  Numeric/Dependency/NumericDependencyManifest.cpp
  Numeric/Dependency/NumericDependencyProcess.cpp
  Numeric/Dependency/NumericDependencyRuntime.cpp
  Numeric/NumericProfiles.cpp
  Numeric/NumericSemanticsInternal.cpp
  Layout/PhysicalTensorCodec.cpp
  Core/RuntimeLaunchContract.cpp
  Core/TargetCall.cpp
  Core/TargetCallDecoder.cpp
  Core/TargetFormat.cpp
  Core/TargetIdentity.cpp
  Core/TargetOperation.cpp
  Core/TargetSchedulingCapability.cpp

  LINK_LIBS PUBLIC
  LLVMSupport
  WaferTargetPhysicalLayout
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

if(WAFER_ENABLE_BULK_MODEL_DEPS)
  add_mlir_library(WaferBulkModel
    PARTIAL_SOURCES_INTENDED
    Numeric/Qualification/QualifiedBulkExecution.cpp
    Numeric/Qualification/BulkExecutionEnvironment.cpp
    Numeric/Qualification/BulkQualification.cpp
    Numeric/Qualification/BulkQualificationCalibration.cpp
    Numeric/Qualification/BulkQualificationCase.cpp
    Numeric/Qualification/BulkQualificationComparison.cpp
    Numeric/Qualification/BulkQualificationExecution.cpp
    Numeric/Qualification/BulkQualificationPolicy.cpp
    Numeric/Qualification/BulkQualificationValidation.cpp
    Numeric/Qualification/BulkTensorCodec.cpp
    Numeric/Qualification/BulkTensorNumericSupport.cpp
    Numeric/Qualification/OneDNNGemmAdapter.cpp

    LINK_LIBS PUBLIC
    WaferIR
    WaferNumericModel
    WaferBulk::oneDNN
    LLVMTargetParser
  )
  target_compile_definitions(obj.WaferBulkModel PRIVATE
    WAFER_BULK_DEPENDENCY_RECORD_SHA256="${WAFER_BULK_MODEL_DEPS_RECORD_SHA256}"
    WAFER_BULK_ONEDNN_LIBRARY_SHA256="${WAFER_BULK_MODEL_ONEDNN_LIBRARY_SHA256}"
    WAFER_BULK_ONEDNN_VERSION="${WAFER_BULK_MODEL_ONEDNN_VERSION}"
    WAFER_BULK_ONEDNN_COMMIT="${WAFER_BULK_MODEL_ONEDNN_COMMIT}"
  )
  # oneDNN's C++ wrapper reports primitive creation/execution failures with
  # exceptions.  Keep exception support local to the managed backend adapter;
  # the rest of the compiler retains LLVM's no-exceptions policy.
  if(MSVC)
    target_compile_options(obj.WaferBulkModel PRIVATE /EHsc)
  else()
    target_compile_options(obj.WaferBulkModel PRIVATE -fexceptions)
  endif()
endif()
