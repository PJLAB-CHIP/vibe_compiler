if(WAFER_ENABLE_SYSTEMC_MODEL)
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
endif()
