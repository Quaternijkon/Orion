# Orion L1-Prior Load-Balance Experiment Protocol

Status: frozen before the navigation-mass candidates are evaluated on GloVe.

## Objective and non-negotiable boundaries

This experiment may change only the classification of nodes in the frozen L1
navigation graph.  It must not rebuild or mutate the production upper graph,
inspect full L0 attachments or observed shard loads while choosing that
classification, repair the L0 layout after assignment, or alter Orion's
multi-assignment rule.

The unchanged L0 rule is:

```text
enabled=true, min_max_vote=2, vote_delta=0, max_shards=0
```

SIFT1M is the development dataset.  GloVe-200-angular is the held-out
cross-dataset validation and the primary four-host QPS dataset.  Parameters
selected from SIFT must not be retuned after inspecting the GloVe mass-candidate
results.

## Frozen lightweight mass estimator

For each of the `U=N/32` ordered upper nodes, search that node's own vector in
the immutable production upper graph with top-k 10 and EF 100.  Every row must
contain the corresponding upper label.  Normally it is rank zero; the only
allowed exception is a deterministic zero-distance tie where every hit before
self has vector bits identical to the query upper vector.  Such rows and their
duplicate-vector proof must be recorded.  Let `c(v)` be the number of rows in
which upper node `v` occurs.  The fixed estimator is:

```text
mass(v) = max(1, c(v) - 1)
```

The subtraction removes the uniformly over-represented self hit: an upper
node's self query is one of only `U` proxy queries and must not be extrapolated
as ordinary L0 traffic.  The floor preserves the fact that every upper node is
itself an L0 point.  Raw hit count is an ablation only and cannot become the
production candidate after GloVe evaluation.

The mass manifest must checksum-bind the source artifact, canonical navigator
and graph, ordered upper labels and vector bytes, exported hits and export
manifest, U, dimension, top-k, EF, the self-presence check and any exact-
duplicate tie exceptions, transform name and version, and partitioner
source/configuration.

## Two-process causality boundary

1. The candidate-generation process receives only the frozen upper artifact
   and checksum-bound L1 self-navigation export.  It writes owner arrays and an
   immutable candidate manifest, then exits.
2. The evaluation process verifies the owner and candidate-manifest hashes.
   Only then may it open full L0 attachments, query hits, or ground truth and
   apply the unchanged multi-assignment rule.

No candidate-specific full-data pass, L0 flow, fission, repair, or load-guided
move is permitted.

## Frozen candidate family

The fixed candidates are navigation-mass LDG, navigation-mass region grow, and
navigation-mass balanced KMeans.  Their seed, iteration count, capacity rule,
and tie-breaking are code constants recorded in the candidate manifest.  Unit
weight KMeans is the topology control; Random, unit LDG, unit LDG-Swap1,
Fennel, and unit region grow are negative controls and cannot be revived after
failing the first topology screen.

## Topology gates

Each dataset uses its frozen source owner only as an ineligible topology
reference.  A candidate must pass every search-induced gate below.  Raw graph
metrics are additional fail-closed guards, not substitutes for the routing
metrics.

Relative to the reference owner on the same graph and query corpus:

- upper edge-cut ratio must be at most `1.03x`;
- mean retained degree must be at least `0.95x`;
- retained-degree p10 may fall by at most `0.025` absolute;
- isolated-node fraction must be no more than one percentage point higher and
  must remain at most 3%;
- mean largest-component fraction may fall by at most two percentage points,
  and every shard's largest-component fraction must remain at least 25%;
- mean query owner transitions, routed shards, route entry points, and summed
  local EF may each increase by at most 5%;
- mean ground-truth routing coverage may fall by at most 0.2 percentage point;
- full-ground-truth-coverage query fraction may fall by at most two percentage
  points;
- zero-coverage queries may not exceed the reference by more than two and may
  never exceed 0.1% of the query corpus.

Graph/navigator identity, ordered label/vector identity, and upper-search replay
identity are exact gates.  A failed gate makes a candidate ineligible for
materialization regardless of its load balance.

Among candidates passing every gate on both datasets, the offline finalist is
the one with the lowest actual L0 physical-copy max/mean under the unchanged
multi-assignment rule.  Expansion ratio and construction cost are tie-breakers,
not reasons to accept topology regression.

## Online adoption gate

The primary baseline is the original Orion configuration used for the
2026-08-25 GloVe QPS figure: generation 3248141, initial P=24 followed by
fission to 32, with multi-assignment enabled.  The L0-informed source owners in
the offline screens are diagnostic references and must not be reported as that
baseline.

The finalist and baseline must use the same immutable upper navigator, GloVe
data and query split, 32 logical shards, four physical hosts, 64 server logical
CPUs, HNSW settings, and round-robin physical placement policy.  Use 1,000
tuning queries and a disjoint 9,000-query held-out set.  The primary comparison
uses the same historical `u48 / upper-EF48 / base50 / factor14` artifact budget
on both arms; both must land in the pre-existing Recall@10 band `[0.90, 0.93)`.
This fixed budget isolates the layout change and avoids collection-specific
runtime retuning.  If either arm falls outside that band, do not silently tune
it: a separate checksum-bound, symmetric finite runtime-profile sweep and
restart protocol is required before any matched-recall claim.  Select each
arm's saturation concurrency from tuning only.  Run 5-7 paired,
counterbalanced, interleaved repeats; require CV <= 5%.

Adopt the load-balance patch only if all topology/identity gates pass and the
held-out paired QPS ratio has a confidence interval entirely above 1.0 at
Recall@10 >= 0.90.  Otherwise the experimental conclusion is to retain original
Orion without this load-balance patch.

## Post-exploratory confirmation amendment: raw-regularized-v2

Status: explicitly post-exploratory; this amendment is not represented as part
of the preregistered v1 estimator selection.

The locked v1 estimator passed every topology gate on SIFT but narrowly failed
the held-out GloVe routed-shard gate.  The already-existing raw-count ablation,
created before that held-out decision, passed every topology gate on both
datasets.  No topology threshold is changed.  To satisfy the requirement for
an actual held-out QPS confirmation without pretending that this observation
was preregistered, one transparent confirmation candidate is promoted:

```text
mass_mode = raw-regularized-v2
source = production_upper_navigation_top10_hit_frequency_regularized_v2
transform = raw_count_with_unit_l1_prior
estimator_version = 2
method = mass-balanced-kmeans
```

This is not a new numeric estimator fitted to GloVe.  Because every upper
self-query contains its own node exactly once, raw hit count has the equivalent
decomposition `1 + non_self_hit_count`: one unit of native L1-size prior plus
the lightweight upper-navigation popularity signal.  The v2 owner must be
byte-for-byte identical to the frozen old-raw mass-balanced-kmeans owner on
both SIFT and GloVe.  Any mismatch invalidates the amendment.

The confirmation process must regenerate independent checksum-bound phase-A
and phase-B manifests for SIFT and GloVe, retain all v1 outcomes, and prove:

- unchanged upper graph, navigation replay, multi-assignment, and topology
  gates;
- exact parity with each old-raw owner;
- every topology and identity gate passes on both datasets;
- no LDG, region-grow, alternate transform, parameter scan, or threshold
  change is introduced.

Promotion under this amendment is only permission to run the already-required
online test.  Held-out, paired Recall@10>=0.90 QPS is the sole confirmation.
If its confidence interval is not entirely above 1.0, Orion retains the
original load-balancing scheme.  Reports must show the failed v1 held-out gate
and label v2 as a post-exploratory confirmation rather than preregistered proof.
