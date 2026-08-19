import os

import lit.formats

config.name = "Wafer"
config.test_format = lit.formats.ShTest()
config.suffixes = [".mlir", ".test"]
config.test_source_root = os.path.join(config.wafer_src_root, "test")
config.test_exec_root = os.path.join(config.wafer_obj_root, "test")

path = os.pathsep.join(
    [
        config.wafer_tools_dir,
        config.filecheck_dir,
        config.llvm_tools_dir,
        os.environ.get("PATH", ""),
    ]
)
config.environment["PATH"] = path
config.environment["PYTHONPATH"] = os.pathsep.join(
    value
    for value in [config.wafer_python_dir, config.environment.get("PYTHONPATH", "")]
    if value
)
if os.path.isdir(config.tx8_deps_root):
    config.environment["TX8_DEPS_ROOT"] = config.tx8_deps_root
config.substitutions.append(("%python", config.python_executable))
config.substitutions.append(("%wafer_obj_root", config.wafer_obj_root))
config.substitutions.append(("%wafer_compile_test", config.wafer_compile_test))
config.substitutions.append(("%wafer_bulk_qualify", config.wafer_bulk_qualify))
config.substitutions.append(("%cmake", config.cmake_command))
config.substitutions.append(("%stablehlo_translate", config.stablehlo_translate))
config.environment["WAFER_STABLEHLO_TRANSLATE"] = config.stablehlo_translate
config.importer_python_executable = getattr(
    config, "importer_python_executable", config.python_executable
)
config.wafer_enable_pytorch_xla_importer = getattr(
    config, "wafer_enable_pytorch_xla_importer", "OFF"
)
config.xla_spmd_partitioner_helper = lit_config.params.get(
    "xla_spmd_partitioner_helper",
    getattr(config, "xla_spmd_partitioner_helper", ""),
)
config.substitutions.append(("%importer_python", config.importer_python_executable))
config.substitutions.append(("%wafer_src_root", config.wafer_src_root))
config.substitutions.append(
    ("%xla_spmd_partitioner_helper", config.xla_spmd_partitioner_helper)
)

if config.wafer_enable_importer_deps == "ON":
    config.available_features.add("stablehlo")

if config.wafer_enable_spmd_partitioner_deps == "ON":
    config.available_features.add("shardy")

if config.wafer_enable_pytorch_xla_importer == "ON":
    config.available_features.add("pytorch-xla-importer")

if config.xla_spmd_partitioner_helper:
    config.available_features.add("xla-spmd-helper")

if config.wafer_enable_numeric_model == "ON":
    config.available_features.add("numeric-model")
    existing_library_path = config.environment.get("LD_LIBRARY_PATH", "")
    config.environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        value
        for value in [config.numeric_model_library_path, existing_library_path]
        if value
    )

if config.wafer_enable_bulk_model == "ON":
    config.available_features.add("bulk-model")

if config.wafer_enable_systemc_model == "ON":
    config.available_features.add("systemc-model")

if (
    config.wafer_enable_numeric_model == "ON"
    and config.wafer_enable_bulk_model == "ON"
    and config.wafer_enable_systemc_model == "ON"
):
    config.available_features.add("target-model-bulk")
