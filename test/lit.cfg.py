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
config.substitutions.append(("%python", config.python_executable))
config.substitutions.append(("%wafer_obj_root", config.wafer_obj_root))
config.importer_python_executable = getattr(
    config, "importer_python_executable", config.python_executable
)
config.wafer_enable_pytorch_xla_importer = getattr(
    config, "wafer_enable_pytorch_xla_importer", "OFF"
)
config.substitutions.append(("%importer_python", config.importer_python_executable))
config.substitutions.append(("%wafer_src_root", config.wafer_src_root))

if config.wafer_enable_importer_deps == "ON":
    config.available_features.add("stablehlo")

if config.wafer_enable_spmd_partitioner_deps == "ON":
    config.available_features.add("shardy")

if config.wafer_enable_pytorch_xla_importer == "ON":
    config.available_features.add("pytorch-xla-importer")
