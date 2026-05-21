# KMeans Adaptive-EF Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone experiment script that builds a 44-way KMeans custom-sharded Qdrant collection, tunes uniform and adaptive `hnsw_ef` policies to reach `Recall@10 >= 0.88` at `nprobe=44`, and exports full `nprobe` sweep results.

**Architecture:** Reuse the proven custom-sharding workflow from earlier experiments, but add two search policies on top: one shared `hnsw_ef` baseline and one application-layer adaptive per-shard policy with client-side merge. Keep collection construction identical between the two policies so the comparison isolates only search behavior.

**Tech Stack:** Python 3, NumPy, h5py, Qdrant REST API, CSV/JSON outputs

---

### Task 1: Add Experiment Script

**Files:**
- Create: `/home/taig/dry/qdrant/tools/qdrant_custom_shard_adaptive_ef_experiment.py`

- [ ] Add a new standalone Python script derived from the current custom-shard workflow.
- [ ] Implement dataset loading, normalization, KMeans training, custom shard creation, and batch upsert.
- [ ] Implement uniform-`hnsw_ef` batch search over selected shard keys.
- [ ] Implement adaptive per-shard search with rank-based `hnsw_ef` schedules and client-side top-k merge.
- [ ] Implement tuning loops for:
  - uniform `hnsw_ef` at `nprobe=44`
  - adaptive schedules at `nprobe=44`
- [ ] Implement CSV/JSON result export.

### Task 2: Verify Script Integrity

**Files:**
- Verify: `/home/taig/dry/qdrant/tools/qdrant_custom_shard_adaptive_ef_experiment.py`

- [ ] Run `python3 -m py_compile /home/taig/dry/qdrant/tools/qdrant_custom_shard_adaptive_ef_experiment.py`
- [ ] Run `python3 /home/taig/dry/qdrant/tools/qdrant_custom_shard_adaptive_ef_experiment.py --help`

### Task 3: Run Uniform-EF Tuning

**Files:**
- Output: `/home/taig/dry/qdrant/results/qdrant_custom_adaptive_ef/<timestamp>/uniform_ef_tuning.csv`

- [ ] Build or reuse the KMeans custom-sharded collection.
- [ ] Run a small uniform-`hnsw_ef` sweep at `nprobe=44`.
- [ ] Pick the lowest-cost uniform `hnsw_ef` that reaches at least `0.88` recall.

### Task 4: Run Uniform-EF NProbe Sweep

**Files:**
- Output: `/home/taig/dry/qdrant/results/qdrant_custom_adaptive_ef/<timestamp>/uniform_ef_nprobe.csv`

- [ ] Run `nprobe = 1, 2, 4, 8, 16, 32, 44` using the tuned uniform `hnsw_ef`.
- [ ] Save full results with QPS and recall.

### Task 5: Run Adaptive-EF Tuning

**Files:**
- Output: `/home/taig/dry/qdrant/results/qdrant_custom_adaptive_ef/<timestamp>/adaptive_ef_tuning.csv`

- [ ] Evaluate several monotone rank-bucket schedules at `nprobe=44`.
- [ ] Pick at least one schedule that reaches at least `0.88` recall.
- [ ] Record the selected schedule explicitly in the output.

### Task 6: Run Adaptive-EF NProbe Sweep

**Files:**
- Output: `/home/taig/dry/qdrant/results/qdrant_custom_adaptive_ef/<timestamp>/adaptive_ef_nprobe.csv`

- [ ] Run `nprobe = 1, 2, 4, 8, 16, 32, 44` using the tuned adaptive schedule.
- [ ] Save full results with QPS and recall.

### Task 7: Summarize Findings

**Files:**
- Output: `/home/taig/dry/qdrant/results/qdrant_custom_adaptive_ef/<timestamp>/summary.json`

- [ ] Save the chosen build parameters, tuned uniform `hnsw_ef`, tuned adaptive schedule, and both result tables.
- [ ] Prepare a short analysis of whether adaptive `ef` improved the QPS/recall frontier under this application-layer implementation.
