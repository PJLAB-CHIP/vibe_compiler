


  zephyr_include_directories(.)
  zephyr_include_directories(libcpu/riscv/c908)

  file(GLOB TEMP_SOURCE_FILE_LIST ./*.c ./libcpu/riscv/c908/*.c)

  zephyr_sources(
          ${TEMP_SOURCE_FILE_LIST}
  )