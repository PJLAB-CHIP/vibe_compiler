import os

import lit.formats

config.name = "Wafer"
config.test_format = lit.formats.ShTest()
config.suffixes = [".mlir"]
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
