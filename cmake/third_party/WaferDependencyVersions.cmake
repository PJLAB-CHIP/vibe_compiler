set(WAFER_LLVM_VERSION "20.0.0git")
set(WAFER_LLVM_PACKAGE_VERSION "20.0.0git")
set(WAFER_LLVM_COMMIT "f0b3287297aeeddcf030e3c1b08d05a69ad465aa")
set(WAFER_LLVM_REPOSITORY "https://github.com/llvm/llvm-project.git")
set(WAFER_LLVM_LINUX_X64_URL "")

set(WAFER_STABLEHLO_TAG "v1.7.1")
set(WAFER_STABLEHLO_COMMIT "e51fd95e5b2c28861f22dc9d609fb2a7f002124e")
set(WAFER_STABLEHLO_REPOSITORY "https://github.com/openxla/stablehlo.git")

set(WAFER_SHARDY_COMMIT "1838e28554df4db156dbd448e1c28ae35baf6533")
set(WAFER_OPENXLA_XLA_SHARDY_BASE_COMMIT "1142aa484b8e11440f609859a6a55edfb733f869")
set(WAFER_SHARDY_REPOSITORY "https://github.com/openxla/shardy.git")

set(WAFER_OPENXLA_XLA_COMMIT "32ebd694c4d0442e241d76324ff1a721831366b4")
set(WAFER_OPENXLA_XLA_REPOSITORY "https://github.com/openxla/xla.git")

set(WAFER_GOOGLETEST_TAG "v1.15.2")
set(WAFER_GOOGLETEST_COMMIT "b514bdc898e2951020cbdca1304b75f5950d1f59")
set(WAFER_GOOGLETEST_REPOSITORY "https://github.com/google/googletest.git")

set(WAFER_PYTORCH_VERSION "2.5.0")
set(WAFER_TORCHVISION_VERSION "0.20.0")
set(WAFER_TORCH_XLA_PYTHON_VERSION "2.5.0")
set(WAFER_PYTORCH_XLA_COMMIT "396608c7105b3763874fe3800dfabdfa2b38a28a")
set(WAFER_PYTORCH_XLA_REPOSITORY "https://github.com/pytorch/xla.git")

set(WAFER_PYTHON_LIT_VERSION "18.1.8")

# Static memory packing dependency. Wafer checks in an audited, source-derived
# C++17 core port from this exact upstream commit; CMake never fetches it.
set(WAFER_MINIMALLOC_COMMIT "9f5cf810fec4494df473c23cffd0567989e81b69")
set(WAFER_MINIMALLOC_REPOSITORY "https://github.com/google/minimalloc.git")

# Functional-numeric model dependencies.  These are source archive pins rather
# than host package/SONAME requirements.  CMake never downloads these archives;
# tools/bootstrap_deps.py is the only supported fetch/build entry point.
set(WAFER_SOFTFLOAT_VERSION "3e")
set(WAFER_SOFTFLOAT_URL "https://www.jhauser.us/arithmetic/SoftFloat-3e.zip")
set(WAFER_SOFTFLOAT_SHA256 "21130ce885d35c1fe73fc1e1bf2244178167e05c6747cad5f450cc991714c746")

set(WAFER_TESTFLOAT_VERSION "3e")
set(WAFER_TESTFLOAT_URL "https://www.jhauser.us/arithmetic/TestFloat-3e.zip")
set(WAFER_TESTFLOAT_SHA256 "6d4bdf0096b48a653aa59fc203a9e5fe18b5a58d7a1b715107c7146776a0aad6")

set(WAFER_M4_VERSION "1.4.21")
set(WAFER_M4_URL "https://ftp.gnu.org/gnu/m4/m4-1.4.21.tar.xz")
set(WAFER_M4_SHA256 "f25c6ab51548a73a75558742fb031e0625d6485fe5f9155949d6486a2408ab66")

set(WAFER_GMP_VERSION "6.3.0")
set(WAFER_GMP_URL "https://gmplib.org/download/gmp/gmp-6.3.0.tar.xz")
set(WAFER_GMP_SHA256 "a3c2b80201b89e68616f4ad30bc66aee4927c3ce50e33929ca819d5c43538898")

set(WAFER_MPFR_VERSION "4.2.2")
set(WAFER_MPFR_URL "https://www.mpfr.org/mpfr-current/mpfr-4.2.2.tar.xz")
set(WAFER_MPFR_SHA256 "b67ba0383ef7e8a8563734e2e889ef5ec3c3b898a01d00fa0a6869ad81c6ce01")

# Bulk functional-model dependency. This is a full commit archive pin rather
# than a host package/SONAME requirement. CMake remains offline; only the
# dependency bootstrap may fetch and build it.
set(WAFER_ONEDNN_VERSION "3.12")
set(WAFER_ONEDNN_COMMIT "80afa71049cd69a3df32adcccb623b12cd7baa22")
set(WAFER_ONEDNN_URL "https://codeload.github.com/uxlfoundation/oneDNN/tar.gz/80afa71049cd69a3df32adcccb623b12cd7baa22")
set(WAFER_ONEDNN_SHA256 "f13ce92168cae7bd25f5efc43f00b37ff14bcc62ffdea3d85220b1c50c104f17")

# Functional-event model dependency. This is the official Accellera release
# archive. CMake remains offline; tools/bootstrap_deps.py is its sole producer.
set(WAFER_SYSTEMC_VERSION "3.0.2")
set(WAFER_SYSTEMC_COMMIT "70b0fc8e4a74acc677b0fc73cea08f940c2115d5")
set(WAFER_SYSTEMC_URL "https://github.com/accellera-official/systemc/archive/refs/tags/3.0.2.tar.gz")
set(WAFER_SYSTEMC_SHA256 "9b3693ed286aab958b9e5d79bb0ad3bc523bbc46931100553275352038f4a0c4")
