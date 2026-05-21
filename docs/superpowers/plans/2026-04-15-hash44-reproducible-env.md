# Hash44 Reproducible Environment And Stability Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Document the reproducible Qdrant experiment environment and run a default hash-based 44-shard tuning plus repeated QPS stability measurement under the same settings.

**Architecture:** Update the experiment documentation with the server and client environment constraints, then reuse the existing HNSW and batch benchmark scripts to build a default 44-shard collection, tune `hnsw_ef` to approximately `Recall@10 = 0.88`, and repeat the QPS measurement multiple times under fixed client launch settings.

**Tech Stack:** Markdown docs, Python experiment scripts, Qdrant HTTP API, Docker

---

### Task 1: Document the Environment

**Files:**
- Modify: `/home/taig/dry/qdrant/docs/HNSW_EXPERIMENTS.md`

- [ ] Add the 16-core Qdrant server pinning settings used in these experiments.
- [ ] Add the client-side `taskset` and thread environment variable recommendations.
- [ ] Add a short reproducibility checklist for later experimenters.

### Task 2: Tune Default Hash 44-Shard Search

**Files:**
- Output: `/home/taig/dry/qdrant/results/hnsw/<timestamp>/summary.csv`
- Output: `/home/taig/dry/qdrant/results/hnsw/<timestamp>/builds.csv`

- [ ] Run `tools/hnsw_experiment.py` against the default hash-based `shard_number=44` setup.
- [ ] Sweep query-time `hnsw_ef` around the target region and select a point near `Recall@10 = 0.88`.

### Task 3: Measure Stable QPS

**Files:**
- Create: `/home/taig/dry/qdrant/results/hash44_stability/<timestamp>/qps_runs.csv`
- Create: `/home/taig/dry/qdrant/results/hash44_stability/<timestamp>/summary.json`

- [ ] Warm up the tuned collection once using the same client-side environment settings.
- [ ] Record at least 3 to 5 repeated QPS runs with the tuned `hnsw_ef`.
- [ ] Save run-by-run values and a short summary.
