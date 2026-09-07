add_executable(wafer-run
  WaferProfileCollection.cpp
  WaferRunBoardIO.cpp
  wafer-run.cpp
)
llvm_update_compile_flags(wafer-run)
set(_wafer_run_resource_directory
  "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/../share/wafer")
set(_wafer_run_report_resource
  "${_wafer_run_resource_directory}/wafer_profile_report.py")
set(_wafer_run_schema_resource
  "${_wafer_run_resource_directory}/wafer_profile_evidence.schema.json")
add_custom_command(
  OUTPUT
    "${_wafer_run_report_resource}"
    "${_wafer_run_schema_resource}"
  COMMAND ${CMAKE_COMMAND} -E make_directory
    "${_wafer_run_resource_directory}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/tools/wafer_profile_report.py"
    "${_wafer_run_report_resource}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/tools/wafer_profile_evidence.schema.json"
    "${_wafer_run_schema_resource}"
  DEPENDS
    "${CMAKE_SOURCE_DIR}/tools/wafer_profile_report.py"
    "${CMAKE_SOURCE_DIR}/tools/wafer_profile_evidence.schema.json"
  VERBATIM
)
add_custom_target(wafer-run-resources
  DEPENDS
    "${_wafer_run_report_resource}"
    "${_wafer_run_schema_resource}"
)
add_dependencies(wafer-run wafer-run-resources)
target_link_libraries(wafer-run
  PRIVATE
  LLVMSupport
  WaferRuntime
)
if(WAFER_ENABLE_BOARD_RUNTIME)
  target_sources(wafer-run PRIVATE TxBoardRuntime.cpp)
  target_compile_definitions(wafer-run PRIVATE
    WAFER_ENABLE_BOARD_RUNTIME=1
    WAFER_TX_RUNTIME_LIBRARY_PATH="${WAFER_TX_RUNTIME_LIBRARY_REAL_PATH}"
  )
  target_include_directories(wafer-run PRIVATE
    "${WAFER_TX_RUNTIME_INCLUDE_DIR}"
    "${CMAKE_SOURCE_DIR}/lib"
  )
  target_link_libraries(wafer-run PRIVATE ${CMAKE_DL_LIBS})
endif()

if(WAFER_ENABLE_PRODUCT_INSTALL)
  include(GNUInstallDirs)
  install(TARGETS wafer-run
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT Runtime)
  install(FILES
    "${CMAKE_SOURCE_DIR}/tools/wafer_profile_report.py"
    "${CMAKE_SOURCE_DIR}/tools/wafer_profile_evidence.schema.json"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/wafer"
    COMPONENT Runtime)
endif()
unset(_wafer_run_resource_directory)
unset(_wafer_run_report_resource)
unset(_wafer_run_schema_resource)
