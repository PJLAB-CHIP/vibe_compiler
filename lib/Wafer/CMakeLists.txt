set(WAFER_NATIVE_JIT_LIBS
  LLVM${LLVM_NATIVE_ARCH}CodeGen
  LLVM${LLVM_NATIVE_ARCH}Desc
  LLVM${LLVM_NATIVE_ARCH}Info)
foreach(_wafer_native_jit_lib IN LISTS WAFER_NATIVE_JIT_LIBS)
  if(NOT TARGET ${_wafer_native_jit_lib})
    message(FATAL_ERROR
      "target-call execution requires native LLVM backend target "
      "${_wafer_native_jit_lib}")
  endif()
endforeach()

add_subdirectory(IR)
add_subdirectory(Target)
add_subdirectory(Frontend)
add_subdirectory(Analysis)
add_subdirectory(Conversion)
add_subdirectory(Transforms)
add_subdirectory(Planning)
add_subdirectory(CodeGen)
add_subdirectory(Package)
add_subdirectory(Runtime)
add_subdirectory(Driver)
add_subdirectory(Simulator)
