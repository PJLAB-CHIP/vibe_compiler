#!/usr/bin/env python3
"""PyTorch/XLA capture adapter for Wafer frontend artifacts."""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
from typing import Any, Callable


P2F1_SMOKE_SIZE = 4096


def _verify_bundle_layout(bundle_path: pathlib.Path) -> None:
    for relative in [
        pathlib.Path("functions") / "forward.mlir",
        pathlib.Path("functions") / "forward.meta",
        pathlib.Path("functions") / "forward.bytecode",
    ]:
        path = bundle_path / relative
        if not path.is_file():
            raise RuntimeError(f"missing StableHLO bundle file: {relative}")
    if not (bundle_path / "data").is_dir():
        raise RuntimeError("missing StableHLO bundle directory: data")


def _make_smoke_module(torch_module: Any, size: int) -> Any:
    class WaferCaptureSmoke4096(torch_module.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch_module.nn.Parameter(
                torch_module.empty(size, size, dtype=torch_module.float32)
            )
            self.bias = torch_module.nn.Parameter(
                torch_module.empty(size, dtype=torch_module.float32)
            )

        def forward(self, x):
            y = x @ self.weight
            y = y + self.bias
            z = torch_module.tanh(y)
            return z + y

    return WaferCaptureSmoke4096()


def _import_runtime_modules() -> tuple[Any, Any]:
    try:
        import torch
        from torch_xla import stablehlo
    except ImportError as error:
        raise RuntimeError(
            "PyTorch/XLA importer runtime is unavailable; install pinned torch "
            "Python packages, then build/install torch_xla from "
            "third_party/pytorch-xla with tools/build_pytorch_xla_runtime.py"
        ) from error
    return torch, stablehlo


def emit_p2f1_smoke_bundle(
    bundle_path: pathlib.Path,
    torch_module: Any | None = None,
    stablehlo_module: Any | None = None,
    smoke_module_factory: Callable[[], Any] | None = None,
    size: int = P2F1_SMOKE_SIZE,
) -> None:
    if torch_module is None or stablehlo_module is None:
        torch_module, stablehlo_module = _import_runtime_modules()

    if smoke_module_factory is None:
        smoke_module = _make_smoke_module(torch_module, size)
    else:
        smoke_module = smoke_module_factory()
    smoke_module.eval()
    input_tensor = torch_module.empty(size, size, dtype=torch_module.float32)

    options = stablehlo_module.StableHLOExportOptions()
    options.export_weights = True
    options.save_weights = True
    options.inline_all_constant = True
    options.include_human_readable_text = True

    with torch_module.no_grad():
        exported = torch_module.export.export(smoke_module, (input_tensor,))
        stablehlo_program = stablehlo_module.exported_program_to_stablehlo(
            exported, options=options
        )

    if bundle_path.exists():
        shutil.rmtree(bundle_path)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    stablehlo_program.save(str(bundle_path))
    _verify_bundle_layout(bundle_path)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emit-p2f1-smoke",
        action="store_true",
        help="emit the P2.F1 4096x4096 matmul+bias+tanh+residual artifact",
    )
    parser.add_argument("--output-bundle", type=pathlib.Path)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    if not args.emit_p2f1_smoke:
        raise RuntimeError("missing --emit-p2f1-smoke")
    if args.output_bundle is None:
        raise RuntimeError("missing --output-bundle")

    emit_p2f1_smoke_bundle(args.output_bundle)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except RuntimeError as error:
        print(f"wafer-pytorch-xla-capture: {error}", file=sys.stderr)
        raise SystemExit(1)
