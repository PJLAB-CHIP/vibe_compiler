#!/usr/bin/env python3
"""Write one current portable StableHLO program directory for board cases."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess


TRANSLATOR_ENVIRONMENT = "WAFER_STABLEHLO_TRANSLATE"


def write_program(
    source: pathlib.Path,
    module: str,
    metadata: dict[str, object],
) -> pathlib.Path:
    translator_text = os.environ.get(TRANSLATOR_ENVIRONMENT)
    if not translator_text:
        raise RuntimeError(
            f"{TRANSLATOR_ENVIRONMENT} is required to write board source"
        )
    translator = pathlib.Path(translator_text)
    if not translator.is_file():
        raise RuntimeError("configured StableHLO translator is not a file")

    functions = source / "functions"
    functions.mkdir(parents=True, exist_ok=True)
    bytecode = functions / "forward.stablehlo.bc"
    # Generated sources have no user file location. A stable stdin source name
    # keeps bytecode identical when a prepared case is checked in another dir.
    result = subprocess.run(
        [str(translator), "--serialize", "--target=1.7.1", "-", "-o", str(bytecode)],
        input=module,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "StableHLO serialization failed: "
            f"{result.stderr or result.stdout}"
        )
    if not bytecode.is_file() or bytecode.stat().st_size == 0:
        raise RuntimeError("StableHLO serialization omitted output bytecode")
    (functions / "forward.meta").write_text(
        json.dumps(metadata, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return source
