#!/usr/bin/env python3
"""End-to-end smoke for the no-replace onednn qualification CLI."""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile


def existing_file(value: str) -> pathlib.Path:
    path = pathlib.Path(value).resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file does not exist: {path}")
    return path


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"


def run(command: list[str], *, expect_success: bool = True) -> str:
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
    )
    if (completed.returncode == 0) != expect_success:
        expectation = "success" if expect_success else "failure"
        raise RuntimeError(
            f"expected {expectation}: {' '.join(command)}\n{completed.stdout}"
        )
    return completed.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", required=True, type=existing_file)
    parser.add_argument("--work-root", required=True, type=pathlib.Path)
    args = parser.parse_args()
    args.work_root.resolve().mkdir(parents=True, exist_ok=True)
    temporary = pathlib.Path(
        tempfile.mkdtemp(prefix="onednn-cli-", dir=args.work_root.resolve())
    )
    try:
        base = {
            "schema": "wafer-onednn-qualification-spec",
            "format": "bf16",
            "m": 4,
            "k": 8,
            "n": 5,
            "batch_count": 1,
            "lhs_layout": "cx",
            "rhs_layout": "cx",
            "destination_layout": "cx",
        }
        calibration_spec = temporary / "calibration-spec.json"
        held_out_spec = temporary / "held-out-spec.json"
        calibration_spec.write_text(canonical({**base, "seed": 101}), encoding="utf-8")
        held_out_spec.write_text(canonical({**base, "seed": 303}), encoding="utf-8")
        calibration = temporary / "calibration.json"
        policy = temporary / "policy.json"
        record = temporary / "record.json"
        budgets = [
            "--formal-scalar-budget",
            "1000000",
            "--formal-fma-budget",
            "1000000",
            "--maximum-total-bytes",
            "10000000",
            "--maximum-scratchpad-bytes",
            "10000000",
            "--maximum-reorder-bytes",
            "10000000",
        ]
        calibrate = [
            str(args.tool),
            "--mode",
            "calibrate",
            "--spec",
            str(calibration_spec),
            "--output",
            str(calibration),
            *budgets,
        ]
        run(calibrate)
        run(
            [
                str(args.tool),
                "--mode",
                "freeze",
                "--calibration",
                str(calibration),
                "--held-out-spec",
                str(held_out_spec),
                "--maximum-absolute-error",
                "0",
                "--maximum-relative-error",
                "0",
                "--output",
                str(policy),
            ]
        )
        validation = [
            str(args.tool),
            "--mode",
            "validate",
            "--policy",
            str(policy),
            "--output",
            str(record),
            *budgets,
        ]
        output = run(validation)
        if "onednn qualification validated" not in output:
            raise RuntimeError("validation did not report read-back success")
        record_data = json.loads(record.read_text(encoding="utf-8"))
        if record_data["schema"] != "wafer-onednn-qualification-record":
            raise RuntimeError("final record schema mismatch")
        if record_data["qualification_kind"] != "profile-bounded":
            raise RuntimeError("finite CLI corpus claimed a non-empirical proof")
        if record_data["matmul_invocations"] != 1 or record_data["backend_formal_fma"] != 0:
            raise RuntimeError("final record lost onednn dispatch evidence")
        if not record_data["raw_exact"]:
            raise RuntimeError("controlled BF16 CLI corpus was not raw exact")
        run(validation, expect_success=False)
    except (KeyError, OSError, ValueError, subprocess.SubprocessError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    print("onednn qualification CLI calibrate/freeze/validate/readback passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
