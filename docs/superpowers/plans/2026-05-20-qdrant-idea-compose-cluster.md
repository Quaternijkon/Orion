# Qdrant Idea Compose Cluster Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the existing Qdrant two-level idea experiment on a single-machine three-Docker Qdrant cluster without changing the algorithm details.

**Architecture:** Add a dedicated compose cluster on ports `6733/6743/6753`. Extend the Python experiment script with cluster peer discovery and optional custom shard-key placement while leaving KMeans assignment, upper HNSW routing, dynamic lower-layer ef, and merge logic unchanged.

**Tech Stack:** Docker Compose, Qdrant REST API, Python 3, NumPy, h5py, hnswlib, pytest.

---

### Task 1: Add Placement Helper Tests

**Files:**
- Modify: `tests/tools/test_qdrant_two_level_routing_experiment.py`
- Modify: `tools/qdrant_two_level_routing_experiment.py`

- [ ] Add tests for:
  - `placement_for_shard_key(..., mode="none")` returns `None`.
  - `placement_for_shard_key(..., mode="round_robin")` returns one peer id in shard order.
  - `placement_for_shard_key(..., mode="auto")` returns `None` for one peer and round-robin for multiple peers.
  - `shard_create_body(...)` omits `placement` when `None` and includes it when provided.

- [ ] Run:

```bash
env PYTEST_ADDOPTS= pytest -o addopts= tests/tools/test_qdrant_two_level_routing_experiment.py -q
```

Expected before implementation: tests fail because the helper functions do not exist.

- [ ] Implement minimal helpers in `tools/qdrant_two_level_routing_experiment.py`.

- [ ] Re-run the same test command and expect all tests in that file to pass.

### Task 2: Add Train-Limit Test

**Files:**
- Modify: `tests/tools/test_qdrant_two_level_routing_experiment.py`
- Modify: `tools/qdrant_two_level_routing_experiment.py`

- [ ] Add a unit test for a pure helper `slice_train_rows(train, train_limit)`:
  - `None` returns all rows.
  - a positive integer returns the prefix.
  - `0` or negative values raise `ValueError`.

- [ ] Run the test file and expect failure because `slice_train_rows` does not exist.

- [ ] Implement the helper and use it immediately after loading the HDF5 `train` dataset.

- [ ] Re-run the test file and expect pass.

### Task 3: Wire Placement Into Collection Build

**Files:**
- Modify: `tools/qdrant_two_level_routing_experiment.py`

- [ ] Add CLI arguments:
  - `--shard-placement` with choices `auto`, `round_robin`, `none`, default `auto`.
  - `--train-limit` as optional positive integer.

- [ ] Discover cluster peers via `GET /cluster`.

- [ ] In `ensure_collection`, compute each shard key placement from the discovered peers and selected mode.

- [ ] Pass placement into shard-key creation.

- [ ] After indexing, fetch `GET /collections/{collection}/cluster` and return a compact distribution summary in `builds.csv` and `summary.json`.

### Task 4: Add Dedicated Idea Compose Cluster

**Files:**
- Create: `tools/compose/docker-compose.idea-cluster.yaml`

- [ ] Copy the native compose structure and adjust:
  - service names to `qdrant_idea_1`, `qdrant_idea_2`, `qdrant_idea_3`.
  - host ports to `6733/6734`, `6743/6744`, `6753/6754`.
  - volume names to `qdrant_idea_*_storage`.
  - CPU/thread environment variables to `QDRANT_IDEA_*`.

- [ ] Run:

```bash
docker compose -f tools/compose/docker-compose.idea-cluster.yaml -p qdrant-idea config
```

Expected: exit code 0 and three Qdrant services.

### Task 5: Document Commands

**Files:**
- Modify: `docs/HNSW_EXPERIMENTS.md`

- [ ] Add a section for the idea compose cluster:
  - startup command,
  - cluster check,
  - smoke command using `--train-limit`,
  - full benchmark command,
  - real multi-machine migration notes.

### Task 6: Verify

**Files:**
- No new files.

- [ ] Run unit and syntax checks:

```bash
env PYTEST_ADDOPTS= pytest -o addopts= tests/tools/test_qdrant_two_level_routing_experiment.py -q
python3 -m py_compile tools/qdrant_two_level_routing_experiment.py tests/tools/test_qdrant_two_level_routing_experiment.py
```

- [ ] Start the idea compose cluster:

```bash
docker compose -f tools/compose/docker-compose.idea-cluster.yaml -p qdrant-idea up -d
```

- [ ] Check the cluster:

```bash
curl -sS http://localhost:6733/cluster
```

- [ ] Run a small smoke command with `--train-limit`, low query counts, and `--target-recall 0.0`.

- [ ] Check the smoke collection cluster endpoint and confirm custom shards are active across multiple peers.
