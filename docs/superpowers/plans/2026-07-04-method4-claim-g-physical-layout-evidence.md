# Method4 Claim G Physical Layout Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a trustworthy evidence package for Claim G: Method4-aware physical layout reduces hot-worker load and provides tail-load evidence.

**Architecture:** Add a standalone offline analyzer that reuses the existing Method4 routing helpers, reconstructs routing traces from a deployed Orion collection, and evaluates physical placements without changing search semantics. Generate CSV/JSON artifacts plus a short experiment report that separates proven offline hot-worker evidence from missing or weak online tail-latency evidence.

**Tech Stack:** Python 3, h5py, numpy, hnswlib through `tools/qdrant_two_level_routing_experiment.py`, Qdrant REST scroll API.

---

### Task 1: Offline Placement Matrix Analyzer

**Files:**
- Create: `tools/method4_claim_g_placement_analysis.py`

- [ ] **Step 1: Add the script**

Create `tools/method4_claim_g_placement_analysis.py` with these responsibilities:

- Load upper vectors and eval queries from `/home/taig/dry/faiss/datasets/glove-200-angular.hdf5`.
- Recover upper-point shard membership and shard copy counts from `qdrant_controller_idea_full_20260521`.
- Build upper HNSW with the same global upper sample as the Orion path.
- Build query plans for `upper_k=160`, `base_ef=80`, `factor=8`, `compact_multi_ep`, `materialized`.
- Evaluate `workers in {2,3,4}` and `placement in {round_robin,size_balanced,method4_aware}`.
- Write:
  - `placement_offline_summary.csv`
  - `placement_offline_deltas.csv`
  - `placement_maps.json`
  - `run_metadata.json`

- [ ] **Step 2: Compile check**

Run:

```bash
python3 -m py_compile tools/method4_claim_g_placement_analysis.py
```

Expected: exit 0.

- [ ] **Step 3: Run the analyzer**

Run:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
python3 tools/method4_claim_g_placement_analysis.py
```

Expected: creates a timestamped directory under `results/method4_claim_g_physical_layout_20260704/`.

### Task 2: Evidence Package

**Files:**
- Create: `docs/experiments/2026-07-04-method4-claim-g-physical-layout-evidence.md`
- Read: `results/method4_claim_g_physical_layout_20260704/<run>/placement_offline_summary.csv`
- Read: `results/qdrant_goal_recall_idea_095_method4map_confirm/20260601_095702/summary.json`
- Read: `results/qdrant_goal_recall_idea_095_paired_repeat_1_strict_b200/20260527_094610/summary.json`
- Read: `results/qdrant_goal_recall_idea_095_paired_repeat_2_strict_b200/20260527_094643/summary.json`
- Read: `results/qdrant_goal_recall_idea_095_paired_repeat_3_strict_b200/20260527_094745/summary.json`

- [ ] **Step 1: Summarize offline evidence**

Document the 9 offline simulations with emphasis on:

- `peer_load_cv_pct`
- `p95_query_max_peer_load`
- `avg_query_max_peer_load`
- `avg_query_max_peer_shard_count`
- `p95_query_max_peer_shard_count`
- `co_routed_pair_cross_worker_pct`

- [ ] **Step 2: Summarize existing online evidence**

Document that existing online evidence has QPS/recall for method4-aware deployment, but no per-query P95/P99 latency metrics. Do not claim online tail-latency improvement unless a fresh latency-producing run exists.

- [ ] **Step 3: Write supported statement**

Use this scope:

```text
Offline routing-trace simulation supports that Method4-aware physical layout reduces hot-worker load proxies. Existing online results show the placement map is deployable and QPS does not drop, but current artifacts do not prove P95/P99 latency improvement.
```

### Task 3: Verification

**Files:**
- Read: generated CSV/JSON artifacts
- Read: generated experiment doc

- [ ] **Step 1: Source-path check**

Run a Python verifier that loads `run_metadata.json`, checks every source path exists, and verifies the summary contains all 9 `(worker_count, placement)` rows.

- [ ] **Step 2: Sanity check against legacy 3-worker result**

Compare the new 3-worker `round_robin` and `method4_aware` metrics against `results/qdrant_goal_recall_idea_095_placement_simulation/20260601_075041/placement_simulation.json`. Expected: same order of magnitude and same direction of improvement.

- [ ] **Step 3: No stale benchmark processes**

Run:

```bash
pgrep -af 'qdrant_two_level_routing_experiment\.py|method4_claim_g_placement_analysis\.py|method4_benchmark_matrix\.py' || true
```

Expected: no active long-running benchmark process from this task.

- [ ] **Step 4: Diff hygiene**

Run:

```bash
git diff --check
```

Expected: exit 0.
