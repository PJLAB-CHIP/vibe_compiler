add_mlir_library(WaferTarget
  PARTIAL_SOURCES_INTENDED
  FormalNumericConvert.cpp
  FormalNumericElementwise.cpp
  FormalNumericGemm.cpp
  FormalNumericReduce.cpp
  FormalNumericSupport.cpp
  FormalNumericValidation.cpp
  NumericCapability.cpp
  NumericCodec.cpp
  NumericCommand.cpp
  NumericDependencyBuildIdentity.cpp
  NumericDependencyConformance.cpp
  NumericDependencyELF.cpp
  NumericDependencyFilesystem.cpp
  NumericDependencyGate.cpp
  NumericDependencyManifest.cpp
  NumericDependencyProcess.cpp
  NumericDependencyRuntime.cpp
  NumericProfiles.cpp
  NumericSemanticsInternal.cpp
  PhysicalTensorCodec.cpp
  RuntimeLaunchContract.cpp
  TargetCall.cpp
  TargetCallDecoder.cpp
  TargetFormat.cpp
  TargetNumericCapability.cpp
  TargetProfile.cpp
  TargetSchedulingCapability.cpp

  LINK_LIBS PUBLIC
  LLVMSupport
  WaferIR
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

if(WAFER_ENABLE_BULK_MODEL_DEPS)
  add_mlir_library(WaferBulkModel
    PARTIAL_SOURCES_INTENDED
    BulkAdmission.cpp
    BulkExecutionEnvironment.cpp
    BulkQualification.cpp
    BulkQualificationCalibration.cpp
    BulkQualificationCase.cpp
    BulkQualificationComparison.cpp
    BulkQualificationExecution.cpp
    BulkQualificationPolicy.cpp
    BulkQualificationValidation.cpp
    BulkTensorCodec.cpp
    BulkTensorNumericSupport.cpp
    OneDNNGemmAdapter.cpp

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
