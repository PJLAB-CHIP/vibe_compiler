#!/usr/bin/env python3
"""Verify that the Direct-DTE board harness fails closed on an outer timeout."""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import tempfile
import time


def load_harness():
    path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "Support"
        / "wafer_board_direct_dte_collective_runner.py"
    )
    spec = importlib.util.spec_from_file_location("wafer_direct_dte_harness", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the Direct-DTE board harness")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    harness = load_harness()
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        child_pid = root / "child.pid"
        forbidden_next_iteration = root / "next-iteration"
        commands = [
            [
                sys.executable,
                "-c",
                (
                    "import os,pathlib,time; "
                    f"pathlib.Path({str(child_pid)!r}).write_text(str(os.getpid())); "
                    "time.sleep(30)"
                ),
            ],
            [
                sys.executable,
                "-c",
                f"import pathlib; pathlib.Path({str(forbidden_next_iteration)!r}).touch()",
            ],
        ]
        started = time.monotonic()
        try:
            for command in commands:
                harness.run(command, timeout_seconds=0.2)
        except RuntimeError as error:
            diagnostic = str(error)
        else:
            raise RuntimeError("outer deadline did not terminate the child process")

        if "will not retry or invoke reset/power operations" not in diagnostic:
            raise RuntimeError("outer deadline diagnostic omitted fail-stop policy")
        if forbidden_next_iteration.exists():
            raise RuntimeError("outer deadline continued to the next iteration")
        if not child_pid.is_file():
            raise RuntimeError("timeout child did not start")
        try:
            os.kill(int(child_pid.read_text()), 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError("timeout child was not killed and reaped")
        if time.monotonic() - started > 5:
            raise RuntimeError("outer deadline did not fail promptly")
    print("direct_dte_outer_deadline: fail-stop verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
