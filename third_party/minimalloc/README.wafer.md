# Wafer curated MiniMalloc core

This directory is a source-derived curated port of Google's MiniMalloc at commit
`9f5cf810fec4494df473c23cffd0567989e81b69`:

- upstream repository: <https://github.com/google/minimalloc>
- upstream license: Apache License 2.0 (see `LICENSE`)
- retained algorithmic core: buffer model, sweep/partition construction,
  canonical depth-first placement, section inference, dominance pruning,
  dynamic decomposition, and the `WAT`/`TAW`/`TWA` preorder rotation

The snapshot is checked in because static memory packing is a default compiler
dependency and CMake configuration must remain offline. It is intentionally not
a git submodule: Wafer carries a small, audited port rather than MiniMalloc's
CLI, Python bindings, converters, test framework, and nested dependencies.

## Wafer modifications

Every modified upstream source file carries a prominent modification notice.
Relative to the pinned commit, Wafer:

- ports the retained core from C++20/Abseil to C++17 and the standard library;
- places symbols in `wafer_third_party::minimalloc` to avoid ODR collisions;
- exposes a typed `SolveStatus`/`SolveResult` API and never throws intentionally;
- supports fixed-capacity feasibility only; capacity minimization and IIS
  computation are omitted;
- requires strictly positive buffer sizes, preserving the assumptions behind
  canonical and dominance pruning; the Wafer adapter handles zero-byte demands
  before entering this core;
- accepts the strict gap subset used by the Wafer conflict-graph adapter: gaps
  are sorted, non-adjacent, non-empty, and strictly interior to a lifespan.
  Upstream endpoint-touching or adjacent-gap forms are outside this curated API;
- drops the unused upstream `Buffer::hint` field and internalizes or omits
  utility APIs that the fixed-capacity solver does not consume;
- removes wall-clock timeout, cancellation, logging, CLI, converter, the
  upstream standalone solution validator, Python, pybind11, and upstream
  GoogleTest targets;
- validates lifespan, gaps, windows, sizes, alignments, fixed offsets, options,
  and legality-relevant arithmetic before the sweep; heuristic-only area
  multiplication saturates deterministically instead of changing feasibility;
- adds one deterministic, global DFS-node budget shared across all partitions,
  preorder heuristics, and round-robin attempts; exhausting this budget returns
  `SolveStatus::kResourceExhausted`, separately from a complete infeasibility
  proof;
- keeps the library default budget unlimited. Wafer's compiler adapter owns any
  deliberately generous production budget and fallback policy.

The public header in this directory is an internal third-party API. The CMake
target is not installed or exported, uses hidden visibility, and must be linked
privately by Wafer implementation targets.

`PROVENANCE.json` records the exact upstream tree/file hashes, their mapping to
the retained port, and canonical digests for the curated algorithm and complete
distribution file sets. `tools/check_deps.py` validates that manifest offline.
