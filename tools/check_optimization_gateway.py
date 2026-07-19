#!/usr/bin/env python3
"""Clang-AST closure check for optimization owners and their gateways."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import pathlib
import re
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class WrappedRule:
    owner: str
    gateway: str
    allowed_paths: frozenset[str]


@dataclass(frozen=True)
class SequencedRule:
    owner: str
    caller: str
    path: str
    begin: str
    commit: str
    owner_count: int
    begins_before_each_owner: int = 1


WRAPPED_RULES = (
    WrappedRule(
        "mlir::createCanonicalizerPass",
        "wafer::createOptimizationInvocationPass",
        frozenset(
            {
                "lib/Wafer/Pipelines/Pipelines.cpp",
                "lib/Wafer/Transforms/Scheduling/ScheduleTensorProgram.cpp",
            }
        ),
    ),
    WrappedRule(
        "wafer::createFunctionBoundaryBufferizationPass",
        "wafer::createOptimizationInvocationPass",
        frozenset({"lib/Wafer/Pipelines/Pipelines.cpp"}),
    ),
    WrappedRule(
        "wafer::createLegalizeStablehloToLinalgPass",
        "wafer::createOptimizationInvocationPass",
        frozenset({"lib/Wafer/Pipelines/Pipelines.cpp"}),
    ),
)

SEQUENCED_RULES = (
    SequencedRule(
        "llvm::sys::ExecuteAndWait",
        "wafer::compiler::detail::runSpmdHelper",
        "lib/Wafer/Compiler/SpmdCompilationBridge.cpp",
        "wafer::beginOptimizationInvocationV1",
        "wafer::commitOptimizationInvocationV1",
        1,
    ),
    # The first launch only materializes a deterministic action plan. The
    # second is the planned backend owner and must lie between begin/commit.
    SequencedRule(
        "llvm::sys::ExecuteAndWait",
        "wafer::compiler::detail::runDeviceLinkAction",
        "lib/Wafer/Compiler/TargetDeviceLink.cpp",
        "wafer::beginOptimizationInvocationV1",
        "wafer::commitOptimizationInvocationV1",
        2,
        0,
    ),
    SequencedRule(
        "mlir::scf::tileAndFuseProducerOfSlice",
        "wafer::tensor_program_to_tile_region::fuseCandidateProducerSlices",
        "lib/Wafer/Conversion/WaferTensorProgramToTileRegion/TileMaterialization.cpp",
        "wafer::tensor_program_to_tile_region::beginCandidateRewriteInvocation",
        "wafer::tensor_program_to_tile_region::commitCandidateRewriteInvocation",
        1,
    ),
    SequencedRule(
        "mlir::linalg::makeTiledShapes",
        "wafer::tensor_program_to_tile_region::materializeCandidateRootTileValue",
        "lib/Wafer/Conversion/WaferTensorProgramToTileRegion/TileMaterialization.cpp",
        "wafer::tensor_program_to_tile_region::beginCandidateRewriteInvocation",
        "wafer::tensor_program_to_tile_region::commitCandidateRewriteInvocation",
        1,
        2,
    ),
)

RAW_ADAPTER_RULES = {
    "mlir::bufferization::createOneShotBufferizePass": frozenset(
        {"lib/Wafer/Transforms/Passes.cpp"}
    ),
    "mlir::stablehlo::createStablehloLegalizeToLinalgPass": frozenset(
        {"lib/Wafer/Conversion/StableHLOToLinalg/LegalizeStablehloToLinalg.cpp"}
    ),
}

CLOSED_KEYS: frozenset[str] = frozenset()
LOCATION_RE = re.compile(r"^(.+?):([0-9]+):([0-9]+): note: \"call\" binds here$")


class AstQueries:
    def __init__(
        self,
        root: pathlib.Path,
        compile_commands: pathlib.Path,
        clang_query: pathlib.Path,
    ) -> None:
        self.root = root
        self.compile_commands = compile_commands
        self.clang_query = clang_query
        database = json.loads(compile_commands.read_text(encoding="utf-8"))
        self.compiled = {
            pathlib.Path(entry["file"]).resolve()
            for entry in database
            if pathlib.Path(entry["file"]).suffix in {".cc", ".cpp", ".cxx"}
        }
        self.cache: dict[tuple[pathlib.Path, str, str], str] = {}

    def run(self, path: pathlib.Path, matcher: str, output: str = "diag") -> str:
        absolute = path.resolve()
        cache_key = (absolute, matcher, output)
        if cache_key in self.cache:
            return self.cache[cache_key]
        if absolute not in self.compiled:
            raise RuntimeError(f"{path}: source is absent from compile_commands.json")
        command = [
            str(self.clang_query),
            "-p",
            str(self.compile_commands.parent),
            str(absolute),
            "-c",
            f"set output {output}",
            "-c",
            f"match {matcher}",
        ]
        completed = subprocess.run(
            command,
            cwd=self.root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=180,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"{path}: clang-query failed:\n{completed.stdout.strip()}"
            )
        self.cache[cache_key] = completed.stdout
        return completed.stdout

    def call_locations(
        self, path: pathlib.Path, callee: str, caller: str | None = None
    ) -> list[int]:
        ancestor = ""
        if caller is not None:
            ancestor = (
                ", hasAncestor(functionDecl(hasName(\""
                + caller
                + "\")))"
            )
        matcher = (
            "callExpr(isExpansionInMainFile(), callee(functionDecl(hasName(\""
            + callee
            + "\")))"
            + ancestor
            + ").bind(\"call\")"
        )
        output = self.run(path, matcher)
        locations: list[int] = []
        for line in output.splitlines():
            match = LOCATION_RE.match(line)
            if match:
                locations.append(int(match.group(2)))
        return locations

    def wrapped_call_locations(
        self, path: pathlib.Path, owner: str, gateway: str
    ) -> list[int]:
        matcher = (
            "callExpr(isExpansionInMainFile(), callee(functionDecl(hasName(\""
            + owner
            + "\"))), hasAncestor(lambdaExpr(hasAncestor(callExpr("
            "callee(functionDecl(hasName(\""
            + gateway
            + "\")))))))).bind(\"call\")"
        )
        output = self.run(path, matcher)
        return [
            int(match.group(2))
            for line in output.splitlines()
            if (match := LOCATION_RE.match(line))
        ]


def source_files(root: pathlib.Path) -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for directory in (root / "lib" / "Wafer", root / "tools" / "wafer-opt"):
        files.extend(directory.rglob("*.cpp"))
    return sorted(files)


def candidate_paths(root: pathlib.Path, identifier: str) -> list[pathlib.Path]:
    short_name = (
        identifier if identifier.endswith("::") else identifier.rsplit("::", 1)[-1]
    )
    return [
        path
        for path in source_files(root)
        if short_name in path.read_text(encoding="utf-8")
    ]


def relative(root: pathlib.Path, path: pathlib.Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=pathlib.Path, required=True)
    parser.add_argument("--compile-commands", type=pathlib.Path, required=True)
    parser.add_argument("--clang-query", type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    failures: list[str] = []
    try:
        ast = AstQueries(
            root, args.compile_commands.resolve(), args.clang_query.resolve()
        )

        for rule in WRAPPED_RULES:
            seen = 0
            for path in candidate_paths(root, rule.owner):
                locations = ast.call_locations(path, rule.owner)
                if not locations:
                    continue
                seen += len(locations)
                path_name = relative(root, path)
                if path_name not in rule.allowed_paths:
                    failures.extend(
                        f"{path_name}:{line}: AST-resolved owner {rule.owner} "
                        "bypasses its closed adapter"
                        for line in locations
                    )
                    continue
                wrapped = ast.wrapped_call_locations(path, rule.owner, rule.gateway)
                if sorted(wrapped) != sorted(locations):
                    failures.append(
                        f"{path_name}: not every AST-resolved {rule.owner} call "
                        f"is a lambda argument of {rule.gateway}"
                    )
            if seen == 0:
                failures.append(f"AST owner has no call site: {rule.owner}")

        for owner, allowed_paths in RAW_ADAPTER_RULES.items():
            seen = 0
            for path in candidate_paths(root, owner):
                locations = ast.call_locations(path, owner)
                seen += len(locations)
                path_name = relative(root, path)
                if locations and path_name not in allowed_paths:
                    failures.extend(
                        f"{path_name}:{line}: AST-resolved raw owner {owner} "
                        "bypasses its adapter"
                        for line in locations
                    )
            if seen == 0:
                failures.append(f"AST owner has no call site: {owner}")

        for rule in SEQUENCED_RULES:
            path = root / rule.path
            all_owner_locations: list[tuple[str, int]] = []
            for candidate in candidate_paths(root, rule.owner):
                for line in ast.call_locations(candidate, rule.owner):
                    all_owner_locations.append((relative(root, candidate), line))
            expected_owner_locations = ast.call_locations(
                path, rule.owner, rule.caller
            )
            path_owner_locations = ast.call_locations(path, rule.owner)
            allowed_owner_paths = {
                candidate.path
                for candidate in SEQUENCED_RULES
                if candidate.owner == rule.owner
            }
            unexpected_owner_locations = [
                location
                for location in all_owner_locations
                if location[0] not in allowed_owner_paths
            ]
            if (
                len(expected_owner_locations) != rule.owner_count
                or sorted(path_owner_locations)
                != sorted(expected_owner_locations)
                or unexpected_owner_locations
            ):
                failures.append(
                    f"{rule.path}: {rule.owner} AST call graph is not exactly "
                    f"the closed {rule.caller} adapter"
                )
                continue
            begin_locations = ast.call_locations(path, rule.begin, rule.caller)
            commit_locations = ast.call_locations(path, rule.commit, rule.caller)
            if not begin_locations or not commit_locations:
                failures.append(
                    f"{rule.path}: {rule.caller} lacks AST-resolved begin/commit"
                )
                continue
            if rule.caller.endswith("runDeviceLinkAction"):
                if not (
                    expected_owner_locations[0] < min(begin_locations)
                    < expected_owner_locations[1]
                    < max(commit_locations)
                ):
                    failures.append(
                        f"{rule.path}: backend plan/begin/owner/terminal order is invalid"
                    )
                continue
            for owner_line in expected_owner_locations:
                begins_before = sum(line < owner_line for line in begin_locations)
                commits_after = sum(line > owner_line for line in commit_locations)
                if (
                    begins_before < rule.begins_before_each_owner
                    or commits_after == 0
                ):
                    failures.append(
                        f"{rule.path}:{owner_line}: AST call is not bracketed by "
                        f"{rule.begin}/{rule.commit}"
                    )

        header = (
            root / "include/Wafer/Support/OptimizationMechanism.h"
        ).read_text(encoding="utf-8")
        key_names = re.findall(
            r"inline constexpr MechanismKey\s+([A-Za-z0-9_]+)\{[1-9][0-9]*\};",
            header,
        )
        key_matcher = (
            "declRefExpr(isExpansionInMainFile(), to(varDecl(anyOf("
            + ",".join(f'hasName(\"{name}\")' for name in key_names)
            + ")))).bind(\"key\")"
        )
        live_keys: set[str] = set()
        key_dump_re = re.compile(
            r"DeclRefExpr .* Var [^ ]+ '([A-Za-z0-9_]+)' "
            r"'const (?:wafer::)?MechanismKey'"
        )
        registry_sources = {
            "lib/Wafer/Support/OptimizationMechanism.cpp",
            "lib/Wafer/Support/OptimizationAdoption.cpp",
        }
        key_paths = [
            path
            for path in candidate_paths(root, "mechanism::")
            if relative(root, path) not in registry_sources
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            outputs = executor.map(
                lambda path: ast.run(path, key_matcher, output="dump"),
                key_paths,
            )
            for output in outputs:
                live_keys.update(key_dump_re.findall(output))
        for name in key_names:
            if name in CLOSED_KEYS:
                if name in live_keys:
                    failures.append(
                        f"closed adoption row mechanism::{name} has an AST reference"
                    )
            elif name not in live_keys:
                failures.append(
                    f"active adoption row mechanism::{name} has no AST reference"
                )
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        failures.append(str(error))

    if failures:
        for failure in sorted(set(failures)):
            print(f"optimization-gateway-check: {failure}", file=sys.stderr)
        return 1
    print(
        "optimization-gateway-check: Clang AST owner/caller/gateway and "
        f"{len(key_names)} linked mechanism rows are closed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
