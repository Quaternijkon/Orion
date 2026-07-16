# KMeans Simple Nprobe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a distributed-system-comparable but algorithmically simple KMeans+nprobe benchmark mode.

**Architecture:** Keep the Qdrant distributed cluster, custom shard keys, coordinator batch-search path, and fixed lower HNSW settings. Add a distinct `kmeans_simple_nprobe` routing mode that uses one KMeans assignment per vector and selects the top `nprobe` centroid shards per query with fixed `hnsw_ef`, without Orion multi-assign, entry-point routing, per-shard dynamic EF, fission, topology convergence, or source-id dedup.

**Tech Stack:** Python benchmark harness, Qdrant REST custom shard search, pytest.

---

### Task 1: Matrix Preset

**Files:**
- Modify: `tools/method4_benchmark_matrix.py`
- Modify: `tests/tools/test_method4_benchmark_matrix.py`

- [ ] Write a failing test showing that `kmeans_simple_nprobe` expands to `routing_mode=kmeans_simple_nprobe`, `search_dispatch_mode=coordinator`, and no Method4 topology/fission fields.
- [ ] Implement the preset and partition-family tag.
- [ ] Run `env PYTEST_ADDOPTS= pytest -o addopts= tests/tools/test_method4_benchmark_matrix.py -q`.

### Task 2: Simple Nprobe Routing Mode

**Files:**
- Modify: `tools/qdrant_two_level_routing_experiment.py`
- Test: `tests/tools/test_qdrant_two_level_routing_experiment.py`

- [ ] Write failing tests for centroid top-nprobe shard selection and the generated request shape: fixed `hnsw_ef`, selected `shard_key`s, no entry-point map, no per-shard EF map, no source-id dedup.
- [ ] Implement centroid computation from KMeans assignments and a `kmeans_simple_nprobe` evaluation path.
- [ ] Ensure summary records `routing_mode=kmeans_simple_nprobe`, `multi_assign=False`, `fission_enabled=False`, and `avg_visited_shards` equal to the selected `nprobe` average.
- [ ] Run `env PYTEST_ADDOPTS= pytest -o addopts= tests/tools/test_qdrant_two_level_routing_experiment.py tests/tools/test_method4_benchmark_matrix.py -q`.

### Task 3: Benchmark Config

**Files:**
- Add: `tools/benchmark_configs/method4_recall_sweep_simple_kmeans_nprobe_20260623.json`

- [ ] Create a recall sweep config comparing `orion`, `kmeans_simple_nprobe`, and `naive`.
- [ ] Keep build/index/shard settings stable and tune only search-time candidates (`nprobe_candidates`, `base_ef_candidates`) for simple KMeans.
- [ ] Dry-run the matrix with `python3 tools/method4_benchmark_matrix.py --config tools/benchmark_configs/method4_recall_sweep_simple_kmeans_nprobe_20260623.json --run-id drycheck_simple_nprobe_20260623`.
