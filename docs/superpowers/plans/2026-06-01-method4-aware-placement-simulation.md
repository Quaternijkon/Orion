# Method4-Aware Placement Simulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and run a method4-aware placement simulator that uses real method4 routing traces to estimate peer load under round-robin placement versus route-aware shard placement.

**Architecture:** The simulator will live in `tools/qdrant_two_level_routing_experiment.py` as pure functions first, so it can reuse existing method4 `query_plans` without changing search behavior. It computes per-query per-shard dynamic-EF cost, evaluates peer load for a placement, and constructs a greedy placement that balances total shard load while spreading co-routed hot shards across peers.

**Tech Stack:** Python 3, existing experiment script, pytest.

---

### Task 1: Add Placement Simulation Unit Tests

**Files:**
- Modify: `tests/tools/test_qdrant_two_level_routing_experiment.py`

- [ ] **Step 1: Write failing tests**

Add tests that define compact method4-style query plans and verify:

```python
def test_extract_shard_costs_from_compact_multi_ep_query_plans():
    module = load_module()
    plans = [
        {
            "searches": [
                {
                    "shard_key": ["centroid_00", "centroid_02"],
                    "hnsw_ef_by_shard": {"centroid_00": 40, "centroid_02": 80},
                }
            ],
            "visited_shards": 2,
            "upper_hits": 3,
            "assigned_ef_sum": 120,
            "assigned_ef_count": 2,
        },
        {
            "searches": [
                {
                    "shard_key": ["centroid_01"],
                    "hnsw_ef_by_shard": {"centroid_01": 60},
                }
            ],
            "visited_shards": 1,
            "upper_hits": 1,
            "assigned_ef_sum": 60,
            "assigned_ef_count": 1,
        },
    ]

    traces = module.query_shard_cost_traces(plans)

    assert traces == [
        {"centroid_00": 40.0, "centroid_02": 80.0},
        {"centroid_01": 60.0},
    ]
```

Add another test that verifies greedy placement improves the max-peer-load proxy on a deliberately skewed trace:

```python
def test_method4_aware_placement_reduces_estimated_tail_load():
    module = load_module()
    traces = [
        {"centroid_00": 100.0, "centroid_01": 90.0, "centroid_02": 10.0},
        {"centroid_00": 100.0, "centroid_01": 90.0, "centroid_03": 10.0},
    ]
    round_robin = {
        "centroid_00": 0,
        "centroid_01": 0,
        "centroid_02": 1,
        "centroid_03": 1,
    }

    optimized = module.greedy_method4_aware_placement(
        traces,
        peer_count=2,
        initial_placement=round_robin,
    )
    before = module.evaluate_query_peer_loads(traces, round_robin)
    after = module.evaluate_query_peer_loads(traces, optimized)

    assert after["avg_query_max_peer_load"] < before["avg_query_max_peer_load"]
    assert optimized["centroid_00"] != optimized["centroid_01"]
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
env PYTEST_ADDOPTS= pytest -o addopts= tests/tools/test_qdrant_two_level_routing_experiment.py::test_extract_shard_costs_from_compact_multi_ep_query_plans tests/tools/test_qdrant_two_level_routing_experiment.py::test_method4_aware_placement_reduces_estimated_tail_load -q
```

Expected: fail because `query_shard_cost_traces`, `greedy_method4_aware_placement`, and `evaluate_query_peer_loads` do not exist.

### Task 2: Implement Pure Placement Simulation Functions

**Files:**
- Modify: `tools/qdrant_two_level_routing_experiment.py`

- [ ] **Step 1: Add pure helpers**

Implement:

```python
def query_shard_cost_traces(query_plans: list[dict[str, Any]]) -> list[dict[str, float]]:
    ...

def evaluate_query_peer_loads(
    traces: list[dict[str, float]],
    placement: dict[str, int],
) -> dict[str, float]:
    ...

def greedy_method4_aware_placement(
    traces: list[dict[str, float]],
    peer_count: int,
    initial_placement: dict[str, int] | None = None,
) -> dict[str, int]:
    ...
```

The greedy placement should sort shards by total route cost descending and place each shard onto the peer that minimizes:

```text
projected_total_peer_load + average projected co-routed query max load
```

Keep it deterministic by breaking ties by peer id.

- [ ] **Step 2: Run targeted tests**

Run the two tests from Task 1. Expected: pass.

### Task 3: Add CLI Simulation Mode

**Files:**
- Modify: `tools/qdrant_two_level_routing_experiment.py`
- Test: `tests/tools/test_qdrant_two_level_routing_experiment.py`

- [ ] **Step 1: Add CLI flag test**

Add a parse-args test for:

```text
--placement-simulation
--placement-simulation-peer-count 3
```

- [ ] **Step 2: Implement CLI flags**

Add:

```python
parser.add_argument("--placement-simulation", action="store_true", ...)
parser.add_argument("--placement-simulation-peer-count", type=int, default=3, ...)
```

- [ ] **Step 3: Wire simulation into final summary**

After materialized routed plans exist, compute and include:

```json
"placement_simulation": {
  "peer_count": 3,
  "round_robin": {...},
  "method4_aware": {...},
  "improvement_pct": ...
}
```

For non-routed or non-materialized modes, reject the flag with a clear `ValueError`.

### Task 4: Run High-Recall Simulation Experiment

**Files:**
- Creates: `results/qdrant_goal_recall_idea_095_placement_simulation/...`

- [ ] **Step 1: Run experiment**

Run idea strict with:

```bash
env PYTEST_ADDOPTS= LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python3 tools/qdrant_two_level_routing_experiment.py \
  --base-url http://localhost:6833 \
  --collection qdrant_controller_idea_full_20260521 \
  --hdf5-path /home/taig/dry/faiss/datasets/glove-200-angular.hdf5 \
  --routing-mode faithful_original_rest \
  --num-shards 31 \
  --upper-k-candidates 160 \
  --base-ef-candidates 80 \
  --factor-candidates 8 \
  --target-recall 0.95 \
  --eval-query-count 3000 \
  --tuning-query-count 500 \
  --stability-repeats 0 \
  --batch-size 200 \
  --reuse-existing \
  --recover-routing-from-collection \
  --routed-execution-mode compact_multi_ep \
  --routed-planning-mode materialized \
  --routed-result-limit-mode top_k \
  --search-dispatch-mode coordinator \
  --shard-placement round_robin \
  --placement-peer-uri-contains qdrant_shard_ \
  --placement-simulation \
  --placement-simulation-peer-count 3 \
  --output-dir results/qdrant_goal_recall_idea_095_placement_simulation
```

- [ ] **Step 2: Record results**

Add a concise experiment note in `docs/experiments/2026-06-01-method4-aware-placement-simulation.md` with mean QPS, recall, and placement-simulation improvement metrics.
