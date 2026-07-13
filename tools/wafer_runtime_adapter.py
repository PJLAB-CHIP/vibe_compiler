#!/usr/bin/env python3
"""Thin Python launcher for the C++ verified-package no-card runtime."""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wafer-run", required=True)
    parser.add_argument("--package-dir", required=True)
    parser.add_argument("--entry-id", required=True)
    parser.add_argument("--no-card", action="store_true")
    parser.add_argument("--max-resource-bytes")
    args = parser.parse_args()

    command = [
        args.wafer_run,
        "--package-dir",
        args.package_dir,
        "--entry-id",
        args.entry_id,
    ]
    if args.no_card:
        command.append("--no-card")
    if args.max_resource_bytes is not None:
        command.extend(["--max-resource-bytes", args.max_resource_bytes])
    completed = subprocess.run(command, check=False)
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
