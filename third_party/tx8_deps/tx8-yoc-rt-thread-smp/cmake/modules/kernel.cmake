
include_guard(GLOBAL)

if(CONFIG_NO_TO_FIND_TARGET_TOOLS)
  message(DEBUG "no to find target tools")
  add_custom_target(asm)
  add_custom_target(compiler)
  add_custom_target(compiler-cpp)
  add_custom_target(linker)
else ()
  find_package(TargetTools)
endif ()
if(NOT DEFINED TX8FW_FIRMWARE_LIB)
  set(TX8FW_FIRMWARE_LIB kcorert)
endif ()

if(${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_BINARY_DIR})
  message(FATAL_ERROR "Source directory equals build directory.\
 In-source builds are not supported.\
 Please specify a build directory, e.g. cmake -Bbuild -H.")
endif()
include(${TX8FW_BASE}/tx8fw_install_info.cmake OPTIONAL)
if(IS_INSTALL_TX8FW)
  include(${TX8FW_BASE}/tx8fw_imported.cmake)
else ()
  add_subdirectory(${TX8FW_BASE} ${__build_dir})
endif ()

