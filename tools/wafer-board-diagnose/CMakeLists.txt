if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_library(wafer-board-register-reader SHARED register_reader.c)
  set_target_properties(wafer-board-register-reader PROPERTIES
    OUTPUT_NAME wafer_board_diagnose
    C_STANDARD 11
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/share/wafer/board-diagnose")
  target_compile_options(wafer-board-register-reader PRIVATE -Wall -Wextra -Werror)
  add_custom_command(
    OUTPUT "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/wafer-board-diagnose"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${CMAKE_CURRENT_SOURCE_DIR}/wafer-board-diagnose.py"
      "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/wafer-board-diagnose"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/wafer-board-diagnose.py"
    VERBATIM)
  add_custom_target(wafer-board-diagnose ALL
    DEPENDS "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/wafer-board-diagnose")
  add_dependencies(wafer-board-diagnose wafer-board-register-reader)
  if(WAFER_ENABLE_PRODUCT_INSTALL)
    include(GNUInstallDirs)
    install(PROGRAMS wafer-board-diagnose.py
      DESTINATION ${CMAKE_INSTALL_BINDIR} RENAME wafer-board-diagnose COMPONENT Runtime)
    install(TARGETS wafer-board-register-reader
      LIBRARY DESTINATION ${CMAKE_INSTALL_DATADIR}/wafer/board-diagnose COMPONENT Runtime)
  endif()
endif()
