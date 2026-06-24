#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "kcorert" for configuration "Release"
set_property(TARGET kcorert APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(kcorert PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "ASM;C;CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libkcorert.a"
  )

list(APPEND _cmake_import_check_targets kcorert )
list(APPEND _cmake_import_check_files_for_kcorert "${_IMPORT_PREFIX}/lib/libkcorert.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
