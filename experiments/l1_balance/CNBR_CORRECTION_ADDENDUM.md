# CNBR replacement correction/addendum

Status: frozen after the fixed SIFT/GloVe family screen and before the new
formal `N_native`/`C_CNBR`/historical-`H` online confirmation.

Date: 2026-08-26.

Machine-readable contract:
[`CNBR_CORRECTION_ADDENDUM.json`](CNBR_CORRECTION_ADDENDUM.json).

## What this document corrects

The user requirement in [`design/balanced.md`](../../design/balanced.md) is to
remove Orion's current L0-informed load-balancing design and replace it, if the
evidence supports doing so, with a lightweight L1-prior patch.  The balance
patch must remain orthogonal to Orion's unchanged L0 multi-assignment rule.

This addendum therefore corrects the experimental roles and fallback semantics
in the frozen [`PROTOCOL.md`](PROTOCOL.md), without editing or erasing that
protocol or its completed evidence:

- the old P24-to-P32 L0-informed/fission build is historical role `H`, not the
  compliant baseline and not an eligible fallback;
- the causal baseline is `N_native`, the Orion core with no load-balancing
  operation;
- the proposed replacement is `C_CNBR`, an upper-only boundary repair starting
  from an exact byte-copy of the `N_native` owner array;
- if `C_CNBR` fails, the compliant fallback is `N_native`, never `H`.

This is a limited interpretation correction: it supersedes the old role
mapping, comparison interpretation, and historical-`H` fallback statement
only.  It does not supersede, rewrite, or discard any completed measurement.
For avoidance of doubt, `H` is `candidate=false`, `adoption_veto=false`, and
`fallback=false`; it is a performance anchor only.

The old protocol, its raw-regularized-v2 amendment, and the completed final
report remain immutable evidence of the earlier experiment.  This addendum
supersedes only their role mapping, comparison interpretation, and the statement
that failure should retain the old balancing scheme.  It does not relabel or
overwrite any old result.

## Frozen experimental roles

| Role | Definition | Eligibility and interpretation |
|---|---|---|
| `H` | Historical generation `3248141`: initial P=24, L0-informed weighting/refinement, then fission to 32, with multi-assignment enabled | Deployment-performance anchor only. It violates the corrected L1-prior boundary, is outside the candidate set, and cannot be a fallback. |
| `N_native` | P=32 plain native upper-vector KMeans, seed 1 and 10 centroid-update iterations; final plain squared-L2 argmin | Primary causal baseline. No balance weights, capacity, quota, refinement, fission, L0 inspection, or repair. |
| `C_CNBR` | `N_native` owner followed by cap-triggered native boundary repair using only frozen upper artifacts; selected trigger ratio `9/4 = 2.25` | Only replacement candidate. The causal effect of the balance patch is `C_CNBR` versus `N_native`. |
| `R_v2` | The completed `raw-regularized-v2 + mass-balanced-kmeans` candidate | Frozen failed historical candidate. It is not `N_native`, is not CNBR, and must not be retuned or revived. |

The completed `R_v2/H` online run is retained as negative evidence: held-out
Recall@10 was 0.92364 for `H` and 0.92571 for `R_v2`, while mean QPS was
4478.78 and 3725.32 respectively; the paired `R_v2/H` QPS-ratio 95% confidence
interval was `[0.831597, 0.831946]`.  This result proves that `R_v2` cannot
replace `H`; it does **not** answer whether a compliant small patch improves
over `N_native`.

## Construction boundary

The corrected build has one causal freeze point:

```text
frozen production upper graph and ordered L1 vectors
    -> neutral upper-only projection removes historical layout/membership
    -> N_native plain KMeans owner
    -> optional C_CNBR upper-only repair
    -> owner bytes + manifest checksum frozen; generator exits
    -> normal L0 attachment/materialization
    -> unchanged multi-assignment rule
```

Before the owner checksum is frozen, neither `N_native` nor `C_CNBR` may open,
derive, or receive full L0 attachments, observed shard sizes, L0 vote streams,
query vectors, query hits, ground truth, or a callback capable of revealing any
of them.  There is no full-data simulation, weight recalibration, L0 flow,
fission, post-assignment repair, or load-guided retry.

The neutral upper-only projection is the **only** artifact accepted by the
Phase-A owner generator.  The historical production artifact may be opened
only by the independent projection step.  Projection preserves the production
upper graph, ordered L1 labels, vector float32 bits, schema, and search
parameters exactly, while replacing historical `shard_membership` and layout
metadata with information-free sentinels.  Its source-build binding may expose
only the upper graph construction parameters `upper_sample_seed`, `upper_m`,
`upper_ef_construction`, and `upper_graph_seed`; old balance-mode or refinement
fields must not cross the Phase-A boundary.

After the owner checksum is frozen and the generator process has exited, an
independent materialization/evaluation process may perform the one normal L0
attachment pass required to build Orion.  Both `N_native` and `C_CNBR` use the
same multi-assignment configuration:

```text
enabled=true, min_max_vote=2, vote_delta=0, max_shards=0
```

Changing, disabling, or candidate-specifically tuning multi-assignment
invalidates the comparison.  Copy expansion remains a measured consequence;
CNBR does not remove multi-assignment.

## `N_native`: balance-free Orion core

`N_native` uses the frozen ordered `U=N/32` upper vectors and exactly 32
partitions.  It runs `cpp_style_kmeans_train(seed=1, max_iter=10)`, where the
iteration count denotes ten centroid-update iterations, then assigns every L1
vector by plain squared-L2 nearest-centroid argmin.  Exact-distance ties use the
lowest centroid/partition id.

There is deliberately no unit quota or exact-size requirement, no vector or
navigation mass, no capacity constraint, no topology refinement, no owner
repair, and no fission.  The resulting owner array is frozen even if its
cluster sizes are unequal.  It is the topology reference for every CNBR gate.

## `C_CNBR`: upper-only boundary repair

### Frozen proxy mass and overload trigger

For each of the `U` upper nodes, replay that node's own vector against the
immutable production upper navigator with top-k 10 and EF 100.  Every row has
exactly ten labels and contains its own label exactly once.  For upper node
`v`, define:

```text
mass(v) = number of occurrences of v across all U frozen top-10 rows
sum_v mass(v) = 10U
mean_proxy_mass = 10U / P, where P=32
trigger(alpha) = alpha * mean_proxy_mass
```

The new manifest freezes `proxy_mass_contract_id` as
`cnbr-upper-self-navigation-raw-occurrence-v1`, source as
`frozen_production_upper_self_navigation_top10`, transform as
`raw_occurrence_count`, and estimator version 1.  Numerically it is one unit of
native L1 self prior plus non-self upper-navigation popularity.  Sharing that
raw occurrence statistic does not make CNBR the retired `R_v2` algorithm:
`R_v2` rebuilt a capacity-constrained mass-balanced KMeans owner, whereas CNBR
starts from `N_native` and performs only local boundary moves.  The statistic
is an upper-only proxy, not an observed L0 load.

Every trigger ratio is stored as a rational number and compared by integer
cross-multiplication.  For selected `alpha=9/4`, for example,
`load > trigger` means `load * 4 * P > 9 * 10U`; no floating-point rounding is
allowed.  The trigger is not an exact-size quota.  It only identifies overloaded
sources and bounds the post-move target load.

### Graph and navigation quantities

The upper level-0 graph used by CNBR is the deterministic undirected union of
the frozen production upper graph's directed level-0 neighbor lists.  For a
prospective move of node `v` from source `s` to target `t`:

```text
edge_delta = deg_s(v) - deg_t(v)
nav_loss   = votes_s(v_row) - votes_t(v_row)
```

`deg_x(v)` is the number of undirected upper neighbors of `v` currently owned
by `x`.  Therefore `edge_delta` is the exact integer change in global undirected
upper edge-cut count.  `votes_x(v_row)` counts labels owned by `x` only in
`v`'s own frozen top-10 self-navigation row; `nav_loss` deliberately does not
use incoming rows, queries, or ground truth.

There is no per-move hard limit on `edge_delta` or `nav_loss`; they are ordered
topology-first by the deterministic keys below.  The cumulative edge-cut cap
and the six end-of-round graph gates remain fail-closed hard constraints.

### Deterministic repair

CNBR starts from a byte-for-byte copy of the frozen `N_native` owner.  It runs
at most eight rounds, and each node may move at most once across the whole run.

At the start of each round, snapshot owners and proxy loads.  Consider only
not-yet-moved nodes whose snapshot source load is strictly above the trigger.
All proposal-time edge counts, self-row votes, and loads use this same round
snapshot; the self-row vote count includes the row's self label.
For node `v`, the target-owner set is the union of:

- owners of `v`'s undirected upper level-0 neighbors; and
- owners of labels in `v`'s frozen top-10 self-navigation row.

Discard the current source.  A cross-owner graph neighbor is not required: a
distinct owner exposed only by the self-navigation row is a valid target.  A
target is eligible only when all four exact load predicates hold against the
round snapshot:

```text
source_load > trigger
target_load < trigger
source_load - target_load > mass(v)
target_load + mass(v) <= trigger
```

The third predicate makes the sum-of-squared proxy-load potential strictly
decrease.  The fourth is a post-move bound, not a global exact-quota promise.

Choose the target with lexicographically minimum key:

```text
(
  max(edge_delta, 0),
  max(nav_loss, 0),
  edge_delta,
  nav_loss,
  target_load,
  -mass(v),
  target_id,
)
```

Create at most one proposal per node, then sort all round proposals by:

```text
(
  -snapshot_source_load,
  target_key,
  node_id,
  source_id,
  target_id,
)
```

Commit proposals sequentially in that order.  Before each commit, recheck the
node's current owner and the four current-load predicates using exact integer
comparisons, recompute `edge_delta` against the current owner state, and require:

```text
(current_cut + edge_delta) * 100 <= N_native_cut * 103
```

If a recheck fails, skip that proposal; do not select a new target or re-propose
it in the same round.  A successful commit immediately updates owners, proxy
loads, cut count, and the global moved-node set.

At round end, recompute all six graph gate groups relative to `N_native`.  If
any gate fails, roll back the entire round and stop.  Stop successfully after a
round with zero commits or after eight accepted rounds.  For each round, the
manifest records the pre-owner checksum and loads; every formed per-node best
proposal with its snapshot fields and ordering keys; every sequential commit
decision as accepted or with an explicit skip reason; accepted moves' four
integer load-predicate proofs, recomputed edge delta, and cumulative integer cut
proof; and the post-owner checksum, loads, topology metrics, six gates,
commit/rollback status, and accepted-move count.  It does not claim to record
nodes for which no proposal was formed or every target rejected during target
filtering.

## Frozen family screen and honest selection status

Before inspecting GloVe candidate outcomes, the SIFT development screen fixed
this trigger grid:

```text
{2.25, 2.0, 1.75, 1.6, 1.5, 1.4}
= {9/4, 2/1, 7/4, 8/5, 3/2, 7/5}
```

The already-fixed family was subsequently evaluated on GloVe.  `9/4` was the
only member that passed every gate on both datasets, so it became `C_CNBR`.
This is transparently a post-exploratory, two-dataset family selection: GloVe
is **not** claimed to be a strictly untouched held-out dataset for choosing the
CNBR trigger.  All grid outcomes must be retained, no new ratio may be added,
and no threshold, ordering key, round limit, or topology gate may be changed
after this selection.  The new formal QPS run is a confirmation of the now-
frozen `9/4` candidate on the disjoint 9,000-query measurement split.

`R_v2` is not a grid member or fallback.  Its earlier failed online result is
retained rather than used for asymmetric CNBR retuning.

## Offline eligibility gates

Every identity, topology, balance, and stage-boundary result is reported for
both SIFT1M and GloVe-200-angular.  All ratios and deltas below use
`N_native` on the same immutable graph and corpus as reference.

### Exact identity and stage-boundary gates

- production upper graph, navigator, ordered labels, and vector bytes match by
  checksum;
- self-navigation rows replay exactly, have width ten, and contain self exactly
  once;
- `C_CNBR` input owner is byte-identical to `N_native`;
- candidate generation is deterministic and has no L0/query/ground-truth input;
- the owner and candidate manifest are checksum-frozen before the independent
  L0 materializer starts;
- multi-assignment code/configuration and its golden replay contract are
  unchanged between `N_native` and `C_CNBR`.

### Six upper-graph gate groups

These are checked after every accepted CNBR round and again on the final owner:

- edge-cut count is at most `1.03x` the `N_native` count;
- mean retained degree is at least `0.95x` the `N_native` value;
- retained-degree p10 drops by at most `0.025` absolute;
- isolated-node fraction is at most `N_native + 0.01` and at most `0.03`;
- mean largest-component fraction drops by at most `0.02` absolute;
- every shard's largest-component fraction is at least `0.25`.

### Post-freeze routing and L0 balance gates

Only the independent evaluator may compute these after owner freeze:

- mean query owner transitions, routed shards, route entry points, and summed
  local EF may each increase by at most 5%;
- mean ground-truth routing coverage may fall by at most 0.002 absolute;
- full-ground-truth-coverage query fraction may fall by at most 0.02 absolute;
- zero-coverage query count may exceed `N_native` by at most two and must remain
  at most 0.1% of the corpus;
- actual L0 primary-load and physical-copy-load distributions, max/mean,
  max/min, CV, expansion ratio, and empty-shard count are all mandatory;
- to claim that CNBR balances Orion, `C_CNBR` physical-copy max/mean must be
  strictly lower than `N_native` on both datasets.  Balance never overrides a
  topology failure.

## Construction-cost gate

CNBR may not add any `O(N)` L0 pass.  Its only extra work is the fixed
`U=N/32`, top-k-10 upper self-navigation replay, the generator's mandatory
input validation and proxy-mass replay, plus upper-only CNBR.  Excluding the
shared `N_native` KMeans, all three required components together must take no
more than 5% of the wall time of the same build's normal full-attachment pass:

```text
(external_self_navigation_wall
 + required_input_validation_and_mass_replay_wall
 + formal_cnbr_wall)
/ external_full_attachment_wall <= 0.05
```

The validation/mass-replay term is mandatory even when it runs in the Phase-A
generator rather than in the exporter.  Omitting it produces only a lower bound
and is not admissible evidence for the 5% gate.

The timing boundary is the integrated-construction boundary, not the wall time
of an isolated evidence-packaging process.  It starts after the upper graph,
ordered upper labels, and upper vectors have been materialized by the normal
upper-graph build and are available to both `N_native` and `C_CNBR`.  It then
includes the projected-artifact checksum, neutral-projection provenance and
redaction checks, exact upper-label/vector byte parity, self-navigation
manifest and row validation, duplicate-tie proof, and proxy-mass replay.
Shared upper-state deserialization/normalization, `N_native` KMeans, offline
topology acceptance checks, source-stability attestation, and evidence
serialization must be timed and reported separately, but are not incremental
operations of the deployed balance patch.  End-to-end construction timing is
still mandatory and prevents this separation from hiding integration overhead.

The neutral projection is an experiment-only adapter used to remove historical
membership/layout information from the old artifact.  The production
replacement must hand the newly built in-memory upper state directly to the
partitioner.  If a deployed implementation instead retains a projection pass,
that pass becomes incremental construction work and must be charged to the
gate.

The build manifest must record wall time, process CPU time, peak RSS, rows,
edges examined, proposals, commits, rounds, and timing boundaries.  End-to-end
build wall/CPU/RSS for `C_CNBR` and HashAll must also be measured and reported.
There is intentionally no invented C/HashAll numerical threshold before those
measurements exist; the 5% incremental upper-only gate and prohibition on an
extra L0 pass are the preregistered hard requirements.

## Mandatory online confirmation at 0.9 recall

The online experiment uses 32 logical shards on the same four physical hosts,
the same immutable upper navigator, data/query split, HNSW settings, 64 server
logical CPUs, round-robin physical placement, and runtime-profile procedure.
Use 1,000 tuning queries and the disjoint 9,000-query measurement split.  Use
the frozen historical `u48 / upper-EF48 / base50 / factor14` budget when every
arm lands in Recall@10 `[0.90, 0.93)`; otherwise run only a checksum-bound,
symmetric finite profile sweep.  Select saturation concurrency from tuning,
then run 5--7 paired, counterbalanced, interleaved repeats with QPS CV at most
5% and a two-sided 95% paired confidence interval on the QPS ratio.

Both comparisons are mandatory:

1. `N_native <-> C_CNBR` is the primary causal test.  Both arms must achieve
   Recall@10 at least 0.90.  Adoption of the balance patch requires every
   offline/build gate plus a paired `C_CNBR/N_native` QPS-ratio confidence
   interval whose lower bound is strictly above 1.0.
2. `H <-> C_CNBR` is the historical deployment-performance bridge.  It must be
   fully measured and reported, but it cannot establish CNBR's causal effect
   and cannot make the non-compliant `H` eligible again.

If CNBR fails its primary gates, the compliant outcome is `N_native`.  If CNBR
wins against `N_native` but remains significantly below `H`, report: "the best
compliant design is the `N_native`/`C_CNBR` winner, but it has not recovered the
historical `H` performance."  Do not call that result a lossless replacement.
Conversely, no `H/C_CNBR` result may substitute for the primary
`N_native/C_CNBR` causal comparison.

## Required evidence

The correction is complete only when the evidence bundle contains:

- frozen `N_native` and all six grid owner arrays, manifests, source hashes,
  deterministic replay hashes, the emitted per-round CNBR records, and the
  selection table;
- SIFT and GloVe identity, graph, routing, L0 primary/copy balance, expansion,
  and build-cost tables for `N_native` and every grid member;
- a proof that `9/4` is the only dual-dataset all-gate pass and that no
  post-GloVe retuning occurred;
- independent materialization manifests proving owner freeze precedes L0
  access and multi-assignment is unchanged;
- paired online raw measurements, tuning choices, recall, QPS, CV, confidence
  intervals, resource/placement proof, and completion audits for both mandatory
  comparisons;
- an addendum report that preserves the old raw-v2 failure and gives separate
  verdicts for architecture eligibility, balance improvement, `C/N` causal
  QPS, `C/H` historical recovery, and construction cost.
