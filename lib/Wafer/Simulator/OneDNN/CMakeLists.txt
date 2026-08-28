if(WAFER_ENABLE_TARGET_NUMERIC_BACKEND)
  add_mlir_library(WaferOneDNNBackend
    QualifiedOneDNNExecution.cpp
    OneDNNExecutionEnvironment.cpp
    OneDNNQualification.cpp
    OneDNNQualificationCalibration.cpp
    OneDNNQualificationCase.cpp
    OneDNNQualificationComparison.cpp
    OneDNNQualificationExecution.cpp
    OneDNNQualificationPolicy.cpp
    OneDNNQualificationValidation.cpp
    OneDNNTensorCodec.cpp
    OneDNNTensorNumericSupport.cpp
    OneDNNGemmAdapter.cpp

    LINK_LIBS PUBLIC
    WaferIR
    WaferReferenceNumeric
    WaferOneDNN::oneDNN
    LLVMTargetParser
  )
  target_compile_definitions(obj.WaferOneDNNBackend PRIVATE
    WAFER_ONEDNN_DEPENDENCY_RECORD_SHA256="${WAFER_ONEDNN_DEPS_RECORD_SHA256}"
    WAFER_ONEDNN_LIBRARY_SHA256="${WAFER_ONEDNN_BACKEND_LIBRARY_SHA256}"
    WAFER_ONEDNN_VERSION="${WAFER_ONEDNN_BACKEND_VERSION}"
    WAFER_ONEDNN_COMMIT="${WAFER_ONEDNN_BACKEND_COMMIT}"
  )
  if(MSVC)
    target_compile_options(obj.WaferOneDNNBackend PRIVATE /EHsc)
  else()
    target_compile_options(obj.WaferOneDNNBackend PRIVATE -fexceptions)
  endif()
endif()
