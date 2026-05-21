# Shard Partition Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and run a controlled Qdrant experiment comparing hash-like, KMeans, and Balanced-KMeans shard partitioning under full-search and selective-search with matched recall.

**Architecture:** Build all three partition schemes in the same custom-sharding setup so partitioning is the only intentional variable, then tune `hnsw_ef` and `nprobe` to matched recall and compare stable QPS. Reuse the existing auto-hash baseline only as an architectural reference after the controlled comparison.

**Tech Stack:** Python 3, NumPy, h5py, Qdrant REST API, CSV/JSON outputs

---

### Task 1: Add a Standalone Comparison Script

**Files:**
- Create: `/home/taig/dry/qdrant/tools/qdrant_partition_scheme_comparison.py`

- [ ] Implement common dataset loading, normalization, collection creation, custom shard creation, and batch upsert.
- [ ] Implement three partition modes:
  - hash-like
  - kmeans
  - balanced-kmeans
- [ ] Implement full-search tuning and selective-search tuning.
- [ ] Implement final full-query evaluation for the chosen best points.
- [ ] Implement CSV/JSON result export.

### Task 2: Add Focused Tests For Partition Helpers

**Files:**
- Create: `/home/taig/dry/qdrant/tests/tools/test_qdrant_partition_scheme_comparison.py`

- [ ] Add tests for deterministic hash partition assignment.
- [ ] Add tests for balanced assignment respecting capacities.
- [ ] Add tests for helper logic that chooses the best matched-recall candidate.

### Task 3: Verify Script Integrity

**Files:**
- Verify: `/home/taig/dry/qdrant/tools/qdrant_partition_scheme_comparison.py`

- [ ] Run `python3 -m pytest -q -c /dev/null /home/taig/dry/qdrant/tests/tools/test_qdrant_partition_scheme_comparison.py`
- [ ] Run `python3 -m py_compile /home/taig/dry/qdrant/tools/qdrant_partition_scheme_comparison.py`
- [ ] Run `python3 /home/taig/dry/qdrant/tools/qdrant_partition_scheme_comparison.py --help`

### Task 4: Run Controlled Comparison

**Files:**
- Output: `/home/taig/dry/qdrant/results/qdrant_partition_scheme/<timestamp>/`

- [ ] Run the script under the fixed client/server environment.
- [ ] Tune full-search and selective-search for all three partition schemes.
- [ ] Record best matched-recall points and final evaluation metrics.

### Task 5: Summarize the General Conclusion

**Files:**
- Output: `/home/taig/dry/qdrant/results/qdrant_partition_scheme/<timestamp>/summary.json`

- [ ] Compare hash-like, kmeans, and balanced-kmeans within the controlled custom-sharding framework.
- [ ] Cross-reference the existing auto-hash baseline only as an architecture note.
- [ ] State a general recommendation for shard partitioning in Qdrant.
