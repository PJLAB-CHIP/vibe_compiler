#!/usr/bin/env python3
"""Compile retained probe entrypoints and check their current launch exports."""

import argparse
import pathlib
import subprocess
import tempfile

import wafer_runtime_launch_contract as launch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clangxx", type=pathlib.Path, required=True)
    args = parser.parse_args()
    inputs = pathlib.Path(__file__).resolve().parents[1] / "Inputs"
    fixtures = {
        "worker_placement": launch.GRID_KERNEL_LAUNCH,
        "ncc_execution": launch.GRID_KERNEL_LAUNCH,
        "spm_cross_tile_conflict": launch.CLUSTER_KERNEL_LAUNCH,
        "instruction_family": launch.GRID_KERNEL_LAUNCH,
        "dte_ncc_execution": launch.CLUSTER_KERNEL_LAUNCH,
        "ddr_tile_offset": launch.CLUSTER_KERNEL_LAUNCH,
        "complete_tile_barrier": launch.CLUSTER_KERNEL_LAUNCH,
        "ncc_pmu_readonly": launch.GRID_KERNEL_LAUNCH,
    }
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)

        def symbols(source: pathlib.Path) -> set[str]:
            obj = root / "probe.o"
            subprocess.run([str(args.clangxx), "-c", str(source), "-o", str(obj)],
                           check=True, capture_output=True)
            result = subprocess.run(
                [str(args.clangxx.with_name("llvm-nm")), "--defined-only",
                 "--extern-only", "--format=posix", str(obj)],
                check=True, capture_output=True, text=True,
            )
            return {line.split()[0] for line in result.stdout.splitlines() if line.split()}

        for name, contract in fixtures.items():
            launch.require_kernel_exports(
                contract, symbols(inputs / f"wafer_{name}_probe.ll"), context=name,
            )
        # Reproduce the actual old ELF/manifest mismatch, not a mocked nm result.
        stale = root / "stale.ll"
        stale.write_text((inputs / "wafer_dte_ncc_execution_probe.ll").read_text()
                         .replace("@entry(", "@main("))
        try:
            launch.require_kernel_exports(launch.CLUSTER_KERNEL_LAUNCH,
                                          symbols(stale), context="stale probe")
        except RuntimeError as error:
            if "entry" not in str(error):
                raise
        else:
            raise RuntimeError("old main export was accepted for current entry")
    print("kernel_probe_exports: 8 current objects and stale-symbol rejection passed")


if __name__ == "__main__":
    main()
