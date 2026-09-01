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
    text = functions / ".forward.mlir.staging"
    bytecode = functions / "forward.stablehlo.bc"
    text.write_text(module, encoding="utf-8")
    try:
        result = subprocess.run(
            [
                str(translator),
                "--serialize",
                "--target=1.7.1",
                str(text),
                "-o",
                str(bytecode),
            ],
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "StableHLO serialization failed: "
                f"{result.stderr or result.stdout}"
            )
    finally:
        text.unlink(missing_ok=True)
    if not bytecode.is_file() or bytecode.stat().st_size == 0:
        raise RuntimeError("StableHLO serialization omitted output bytecode")
    (functions / "forward.meta").write_text(
        json.dumps(metadata, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return source
