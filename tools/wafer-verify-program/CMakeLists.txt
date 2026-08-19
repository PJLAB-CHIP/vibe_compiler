if(WAFER_ENABLE_IMPORTER_DEPS)
  add_executable(wafer-verify-program wafer-verify-program.cpp)
  llvm_update_compile_flags(wafer-verify-program)
  target_link_libraries(wafer-verify-program PRIVATE
    LLVMSupport
    MLIRFuncDialect
    MLIRIR
    MLIRSupport
    StablehloRegister
    WaferStableHLOProgram
  )
  include(GNUInstallDirs)
  install(TARGETS wafer-verify-program RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
