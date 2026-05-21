# Qdrant Two-Level Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and run a Qdrant-only two-level routing experiment with KMeans shard partitioning, a global user-side HNSW built from `1/32` shard samples, and adaptive per-shard `hnsw_ef`.

**Architecture:** Reuse the existing custom-sharding build flow for the lower layer, add a client-side global upper-layer HNSW using `hnswlib`, route each query through the upper layer, convert routed hit counts into shard-local `hnsw_ef`, then merge lower-layer shard results client-side.

**Tech Stack:** Python 3, NumPy, h5py, hnswlib, Qdrant REST API, CSV/JSON outputs

---

### Task 1: Add Focused Tests

**Files:**
- Create: `/home/taig/dry/qdrant/tests/tools/test_qdrant_two_level_routing_experiment.py`

- [ ] Add tests for per-shard `1/32` sampling helper behavior.
- [ ] Add tests for upper-hit-count to lower-layer `ef` mapping.
- [ ] Add tests for result merge helpers.

### Task 2: Implement Standalone Experiment Script

**Files:**
- Create: `/home/taig/dry/qdrant/tools/qdrant_two_level_routing_experiment.py`

- [ ] Implement KMeans custom-sharded collection creation or reuse.
- [ ] Implement per-shard random sampling and global upper-layer HNSW construction.
- [ ] Implement adaptive lower-layer shard routing with `ef = base + factor * hit_count`.
- [ ] Implement parameter tuning over `upper_k`, `base_ef`, and `factor`.
- [ ] Export build, tuning, and final metrics.

### Task 3: Verify Integrity

**Files:**
- Verify: `/home/taig/dry/qdrant/tools/qdrant_two_level_routing_experiment.py`

- [ ] Run focused tests.
- [ ] Run `py_compile`.
- [ ] Run a smoke test on a reduced query set.

### Task 4: Run Full Experiment

**Files:**
- Output: `/home/taig/dry/qdrant/results/qdrant_two_level/<timestamp>/`

- [ ] Tune parameters until `Recall@10 >= 0.88`.
- [ ] Run the final full-query evaluation.
- [ ] Summarize QPS and routing statistics.
