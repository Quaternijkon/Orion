# Attraction-weighted search-induced graph labeling

Status: experimental owner-generation path; not a canonical Orion default.

## Objective

Keep the production upper HNSW completely natural and immutable, then decide
only which logical shard owns each upper node.  The owner is computed from:

1. the exact frozen level-0 upper graph;
2. top-1 attraction measured by the production upper navigator; and
3. search-induced support on existing upper-graph edges.

The partitioner never changes HNSW insertion, neighbour selection, graph bytes,
or online routing.  It never reads live shard loads and never builds a lower
HNSW before the owner checksum is frozen.

```text
natural immutable upper HNSW
        -> production-navigator calibration hits
        -> attraction vertex weights
        -> search-induced weights on existing level-0 edges
        -> weighted METIS/KaHIP labeling into P logical shards
        -> frozen owner checksum
        -> normal L0 attachment and BMR_10 materialization
```

`P` is the logical-shard count.  It is deliberately separate from the number
of physical workers; the resulting logical shards are placed on workers in a
later physical-layout stage.

## Attraction estimators

### Exact full-corpus mode (preferred)

Run every corpus vector through the frozen upper navigator once and cache its
top-k hits.  The first hit contributes one unit to that upper node:

```text
w(u) = count(point whose first upper hit is u)
```

The weights sum exactly to the logical point count.  This is not an extra
full-data navigation pass when the cached hit matrix is reused by the normal
L0 layout stage.  It does not build or inspect any lower HNSW.

### Deterministic sample mode

For a fixed, checksum-bound calibration sample, raw top-1 counts are smoothed
with a fixed Dirichlet(1) prior.  One point is reserved for every upper node;
the remaining corpus mass is assigned by deterministic largest-remainder
apportionment:

```text
p(u) = (count(u) + 1) / (sample_count + upper_node_count)
w(u) = 1 + apportioned((N - upper_node_count) * p(u))
```

This prevents zero-weight nodes when `K` is large relative to the calibration
sample.  Sample mode requires a separate selection manifest containing at
least:

```json
{
  "format_version": 1,
  "selection_method": "stable_hash_point_id_v1",
  "seed": 42,
  "row_count": 59176,
  "vectors_sha256": "..."
}
```

The selection vector checksum must equal the calibration hit export's vector
checksum.  Exact mode forbids a sample-selection manifest.

## Search-induced edge weights

The first implementation supports two frozen edge modes:

- `unit`: every undirected level-0 edge has weight one; this is an ablation;
- `hit-cooccurrence`: an existing level-0 edge receives one additional unit
  whenever both endpoints occur in the same calibration top-k row.
- `top1-star`: keep every level-0 edge, and explicitly add/weight a
  search-induced overlay from the placement terminal (rank zero) to every
  other node in the same top-k row.  This mode is intended to co-locate the
  node that owns a point with the entry-point evidence that would route to it.

`unit` and `hit-cooccurrence` introduce no non-HNSW edge.  `top1-star` records
the number of raw HNSW edges and added search-induced overlay edges separately.
The final METIS edge weight is:

```text
1 + calibration co-occurrence count
```

This is available from the existing production hit exporter without modifying
the search algorithm.  A later traversal-instrumentation experiment may replace
co-occurrence with actual traversed-edge frequency, but it must remain a read-only
measurement and a separately identified estimator version.

## Feasibility gates

Before invoking a partitioner:

```text
max attraction weight
--------------------- <= heavy_atom_fraction (default 0.20)
 mean shard target
```

Failure means that the natural upper graph is too coarse for the requested
logical-shard count.  The required response is to increase the number of upper
representatives and rebuild the natural upper graph, not to read current worker
loads or assign one upper node to multiple owners.

After partitioning, every logical shard must be non-empty and its predicted
attraction load must lie in the configured band (default `0.98..1.02` of the
mean).  Both unit edge cut and weighted search-induced cut are recorded.

## Build and export calibration hits

Build the existing production navigator exporter:

```bash
cargo build --release -p collection --example orion_export_upper_hits
```

For preferred exact mode, use the canonical full-corpus vector rows already
needed by native Orion import.  `top-k=10` provides top-1 attraction plus edge
co-occurrence evidence; the first result is the placement terminal:

```bash
target/release/examples/orion_export_upper_hits \
  /abs/layout/source-artifact.json \
  /abs/layout/orion_numeric_import.f32le \
  1183514 200 10 100 \
  /abs/run/calibration-hits.u64le \
  /abs/run/calibration-hits.manifest.json
```

The artifact, metric, preprocessing, graph bytes, and navigation implementation
are therefore exactly the same as the future Orion layout path.  Search EF is
explicitly checked by the owner generator.

## Generate METIS input or an owner

Generate only the immutable METIS graph for an external partitioner:

```bash
python3 experiments/l1_balance/prepare_attraction_weighted_metis.py \
  --artifact /abs/layout/source-artifact.json \
  --calibration-hits /abs/run/calibration-hits.u64le \
  --calibration-manifest /abs/run/calibration-hits.manifest.json \
  --placement-search-ef 100 \
  --num-partitions 32 \
  --estimator exact-top1-count-v1 \
  --edge-mode hit-cooccurrence \
  --imbalance-tolerance 0.02 \
  --heavy-atom-fraction 0.20 \
  --emit-only \
  --output-dir /abs/run/attraction-metis-input
```

If `gpmetis` is installed, omit `--emit-only`; the tool invokes the fixed
`kway/cut/seed=0` backend and validates its output.  It can also use the same
METIS library through optional PyMetis:

```text
--partitioner-backend pymetis
```

The default `auto` backend prefers `gpmetis` and then tries PyMetis.  Neither is
a required Qdrant runtime dependency: this is an offline experiment tool.  A
KaHIP or remote METIS result can instead be imported as one zero-based
partition ID per upper node:

```bash
python3 experiments/l1_balance/prepare_attraction_weighted_metis.py \
  ...same frozen inputs... \
  --partition-file /abs/run/upper-attraction.graph.part.32 \
  --output-dir /abs/run/attraction-owner
```

When a global METIS owner improves balance but fails Orion's routing-topology
gates, the same attraction weights can drive a constrained refinement from a
frozen accepted owner:

```text
--partitioner-backend attraction-refine
--initial-owner-binary /abs/frozen/C_CNBR.owner.i32le
```

This mode permits only moves toward an owner exposed by a raw level-0 HNSW
neighbour, moves each upper node at most once, reduces exact attraction-load
potential, caps the target at `1.15x` mean during construction, and caps raw
upper edge cut at `1.03x` the initial owner.  It remains a separate research
candidate rather than silently changing C_CNBR.

The topology-preserving experimental variant additionally requires the frozen
upper self-navigation export:

```text
--partitioner-backend attraction-nav-refine
--initial-owner-binary /abs/frozen/C_CNBR.owner.i32le
--upper-self-navigation-hits /abs/frozen/upper-self-top10.u64le
--upper-self-navigation-manifest /abs/frozen/upper-self-top10.manifest.json
```

It permits at most one self-navigation vote of loss and caps cumulative moved
attraction at exactly `2/25 = 8%`.  These values are explicitly
post-exploratory and must be validated on another dataset before any general
claim.

Sample mode additionally requires:

```text
--estimator sample-top1-dirichlet-v1
--calibration-selection-manifest /abs/run/calibration-selection.json
```

## Output contract

The owner directory contains:

```text
calibration-first-hit-counts.u64le
attraction-weights.u64le
upper-attraction.graph
preflight.json
ATTRACTION_WEIGHTED_METIS.owner.i32le   # omitted for --emit-only
attraction-weighted-manifest.json
checksums.sha256
```

The generated owner can enter the existing owner-by-policy offline screen
without first constructing a routed artifact:

```bash
python3 experiments/multi_assignment/screen_owner_policy_matrix.py \
  --dataset glove-200-angular \
  --phase-b-screen /abs/frozen-cnbr-screen-manifest.json \
  --owner-binary \
    ATTRACTION_METIS=/abs/run/attraction-owner/ATTRACTION_WEIGHTED_METIS.owner.i32le \
  --query-count 1000 \
  --output-dir /abs/run/attraction-owner-policy-screen
```

This evaluates `current_all_max`, `single_rank`, and `BMR_10` on the same
full-dataset attachment/query inputs used by the frozen CNBR screen.  The
screen records topology, routed work, ground-truth coverage, expansion, and
physical-copy load before any online collection is built.

The manifest binds the source artifact, upper graph checksum, calibration hit
and selection manifests, estimator, edge mode, METIS parameters, source files,
owner bytes, predicted partition loads, heavy-atom result, and graph cuts.

## Adoption boundary

This owner is only an experimental candidate.  It must not replace `C_CNBR`
based on predicted balance alone.  Before materialization it must pass the same
offline topology and routing gates used by current Orion:

- edge cut, retained degree, isolated nodes, and connected components;
- ground-truth routing coverage;
- routed logical shards and route EF sum;
- physical-copy expansion and hottest shard under unchanged BMR_10.

Only one frozen offline finalist should be materialized.  Adoption requires a
four-host, held-out, counterbalanced interleaved A/B against
`C_CNBR+BMR_10` at matched Recall@10.

## Full GloVe implementation preflight

On 2026-08-31 the emit-only path was exercised against the existing immutable
GloVe-200 production upper graph and all `1,183,514` frozen top-10 attachment
rows at placement EF `100`:

| Field | Observed |
|---|---:|
| Upper nodes | 36,984 |
| Undirected level-0 edges | 757,945 |
| Edges with non-zero calibration support | 479,315 |
| Total final edge weight | 7,572,857 |
| Maximum exact top-1 attraction | 3,537 points |
| Mean target load for P=32 | 36,984.8125 points |
| Heavy-atom ratio | 0.09563 |
| Heavy-atom limit | 0.20 |
| METIS input size | about 11.8 MB |
| METIS input SHA-256 | `9c5c183346aaf250a2663d3fc1845cb85df7fd95a0dc60e70d0a71353f14b511` |
| Total time before manifest | 41.24 s |

The heavy-atom preflight passed, so the current upper representative count is
fine enough for this necessary P=32 feasibility condition.  This run generated
partition input only with the system environment.  A temporary PyMetis 2025.2.2
environment was then used to exercise the optional backend without adding a
Qdrant runtime dependency.

## First GloVe owner screen

The first implementations were evaluated on the frozen 1,000-query GloVe
tuning prefix with the same full attachment matrix, ground truth, runtime
budget, and BMR_10 policy as C_CNBR.  These are offline owner screens, not
online QPS results.

| BMR_10 owner | GT coverage | Full coverage | Routed shards | EF sum | Physical max/mean | Physical CV | Raw edge-cut ratio | Canonical gates |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| C_CNBR reference | 0.9687 | 0.790 | 8.242 | 1127.486 | 3.113 | 0.626 | 0.650 | reference |
| Attraction + existing-edge METIS | 0.9640 | 0.764 | 9.758 | 1201.998 | 1.455 | 0.136 | 0.692 | FAIL |
| Attraction + top1-star METIS | 0.9609 | 0.763 | 9.789 | 1203.968 | 1.415 | 0.144 | 0.708 | FAIL |
| Attraction refinement from C_CNBR | 0.9622 | 0.757 | 11.042 | 1298.286 | 1.493 | 0.262 | 0.669 | FAIL |

The direct existing-edge METIS owner reached predicted attraction load
`0.9813..1.0190x` mean.  The constrained refinement reached
`0.9992..1.0014x` while keeping raw edge cut below the frozen `1.03x` cap.
Nevertheless, both failed search-topology gates.  This establishes two useful
negative results:

1. balancing exact top-1 attraction is highly effective for physical-copy
   balance, but top-1 attraction alone is not sufficient to preserve Orion's
   multi-entry routing coverage;
2. raw level-0 edge cut, even with a strict `+3%` cap, is not a sufficient
   proxy for routed-shard fan-out.

The top1-star overlay did not repair the issue and is retained as a negative
ablation.  None of these three owners is eligible for materialization or online
QPS claims.

## Navigation-constrained 8% candidate

A subsequent fixed-family screen added the upper self-navigation constraint
suggested by the failed global owners:

- start from frozen C_CNBR;
- use exact top-1 attraction weights;
- permit at most one self-navigation vote of loss;
- keep raw edge cut within `1.03x` C_CNBR;
- move each upper node at most once;
- cap cumulative moved attraction at one of
  `{8.00%, 8.25%, 8.50%, 8.75%, 9.00%}`.

On the frozen 1,000-query tuning prefix, `8.00%`, `8.25%`, and `8.50%` passed
all canonical cross-owner gates.  `8.75%` and `9.00%` failed the hottest-shard
gate.  The lowest-budget passing point, `8.00%`, was then replayed over all
10,000 bound GloVe queries and remained eligible:

| Full-query BMR_10 metric | C_CNBR | Attraction NAV8 | Change |
|---|---:|---:|---:|
| GT routing coverage mean | 0.96721 | 0.96626 | -0.00095 absolute |
| Full-coverage fraction | 0.7882 | 0.7828 | -0.0054 absolute |
| Routed shards mean | 8.0824 | 8.3117 | +2.84% |
| Route EF sum mean | 1117.3234 | 1124.6038 | +0.65% |
| Hottest physical-copy shard | 126,661 | 126,313 | -0.27% |
| Physical-copy load CV | 0.62597 | 0.57185 | -8.65% |
| Raw upper edge-cut ratio | 0.64980 | 0.65839 | +1.32% |

The deterministic implementation reproduced the selected prototype owner
byte-for-byte.  Its owner SHA-256 is:

```text
4490906ad619f683f1d5c9fa2f61c8152d5c436be3d89fd823c2a4691da96d1f
```

This was an offline-eligible, explicitly post-exploratory GloVe candidate, not
an adopted Orion default.  The hottest-shard improvement is small, although
the CV reduction is material.

## Fixed-rule SIFT replay

The exact same 8% rule was then replayed on SIFT1M without retuning.  Its
coverage and route-work gates passed, and physical-copy CV improved, but the
index expansion and hottest-shard gates failed:

| Full-query SIFT BMR_10 metric | C_CNBR | Attraction NAV8 | Result |
|---|---:|---:|---|
| GT routing coverage mean | 0.99654 | 0.99678 | pass |
| Routed shards mean | 4.7755 | 4.8957 | pass |
| Route EF sum mean | 299.0332 | 299.5432 | pass |
| Expansion ratio | 1.079695 | 1.080815 | fail |
| Hottest physical-copy shard | 88,635 | 89,520 | fail |
| Physical-copy load CV | 0.55689 | 0.51467 | improves 7.58% |

Consequently `ATTRACTION_NAV8+BMR_10` is not a dual-dataset winner and must
not be materialized as the general Orion replacement.  The result narrows the
design lesson:

- attraction-aware navigation-constrained relabeling can reduce global load
  variance without breaking routing topology;
- reducing variance does not guarantee that the single hottest shard or copy
  expansion improves on every dataset;
- any next general candidate needs a frozen multi-objective gate for
  attraction load, replication eligibility, and hotspot load rather than an
  attraction-only migration budget.

No online QPS A/B is authorized by this evidence.  Current canonical
C_CNBR+BMR_10 remains unchanged.

## Successful lower-layer follow-up

The owner-level negative results showed that a bottom-data lower bound should
not be enforced by globally relabeling the navigation graph.  The successful
follow-up therefore keeps C_CNBR unchanged and applies the capacity guarantee
to frozen BMR_10 physical copies.  See
[`../multi_assignment/CAPACITY_BOUNDED_BMR10.md`](../multi_assignment/CAPACITY_BOUNDED_BMR10.md).

`BMR_10_CAP35` guarantees a final physical-copy floor of `0.35x` mean, preserves
every point's copy count, uses only attachment-supported targets, and passes the
full GloVe and SIFT offline gates.  It is the current replacement candidate for
the post-owner placement rule, pending online QPS confirmation.
