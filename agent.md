# Orion Agent Guidelines

Orion is a distributed graph-based ANN system whose core principle is:

> Offline data organization and online query routing must follow the same
> graph-navigation semantics.

When modifying Orion, preserve the following design constraints.

## Core Constraints

### 1. Offline-online symmetry

Data partitioning and query routing must be derived from the same immutable
production upper graph, distance metric, vector preprocessing, and navigation
implementation.

Canonical Orion must build and freeze the production upper graph before computing
offline L0-to-L1 attachments. Offline attachment search and online query routing
must then use the same upper navigator and the exact same graph bytes. They may use
different explicit search budgets because attachment construction and query routing
serve different purposes.

Do not build one upper graph with `hnswlib` for offline placement and independently
rebuild another Qdrant graph for online routing. Such a dual-graph path is legacy,
an ablation, or an architectural experiment, not canonical Orion.

The preferred implementation boundary is:

- an immutable `UpperNavigator` containing the vector schema, upper vectors,
  portable HNSW graph, metric, preprocessing, and upper-search implementation;
- an `OrionRouter` that composes that navigator with shard memberships, MultiEP,
  shard selection, and contribution-aware local-search budgets.

The canonical build order is:

```text
upper sample
    -> build and checksum the production upper graph once
    -> use that graph to generate full-dataset navigation attachments
    -> topology refinement / fission / multi-assignment
    -> bind shard memberships and layout metadata to the unchanged graph
    -> load the same graph in the online Orion router
```

Finalization and runtime-profile derivation must not rebuild, replace, or reorder
the upper graph. Bind the stages with checksums such as `upper_graph_sha256`,
`attachments_sha256`, and `layout_sha256`, and fail closed on any mismatch.

### 2. Search-induced topology

Orion's topology is defined primarily by actual graph-navigation behavior, not
merely vector-space distance or raw graph adjacency.

Topology-aware partitioning and refinement should therefore be based on
search-induced relationships produced by the canonical upper navigator.

### 3. Navigation-guided placement

The full dataset must be assigned to shards according to shard evidence obtained
through navigation on the global navigation graph.

Geometric clustering may be used as initialization, for capacity management, or
for an explicitly identified experimental mechanism, but it must not replace
navigation-guided placement as the final canonical Orion partitioning principle.

### 4. Navigation-guided multi-assignment

When vectors are replicated across shards, replication must be justified by
navigation/search reachability rather than only geometric proximity.

The exact replication policy and expansion budget remain open design choices.

### 5. Navigation-to-local handoff

Online navigation results must determine both:

- which shards should be searched;
- the corresponding entry point or ordered entry points for shard-local graph
  search.

Do not discard navigation results and restart local search from unrelated default
entry points.

### 6. Contribution-aware search

Different shards may have different expected contributions to a query.

Navigation evidence, currently represented primarily by ordered shard entry-point
evidence, must be used to estimate this contribution and adapt local search effort
accordingly.

The exact mapping from evidence to search budget or EF is not fixed.

### 7. Selective routing

Canonical Orion should selectively search shards supported by navigation evidence
rather than unconditionally performing full scatter-gather.

Full-shard search may be retained only as a baseline, ablation, explicit fallback,
or debug mode. A configured canonical Orion route must fail closed rather than
silently changing algorithms when its required graph or routing artifact is
missing, corrupt, or inconsistent.

## Upper Graph Lifecycle

The production upper graph is an immutable, checksum-addressed input to layout
construction and online routing.

- Build it once with the production Qdrant/Rust graph implementation.
- Generate offline attachments through the same metric, preprocessing, and HNSW
  traversal code used online.
- Make attachment output deterministic in point-ID order even if queries are
  evaluated concurrently.
- Preserve the graph exactly when adding memberships or deriving runtime profiles.
- Rebuild attachments and the full layout whenever the upper sample, vector schema,
  metric, preprocessing, HNSW construction parameters, seed, insertion order, or
  graph bytes change.
- Runtime-only changes such as query upper EF/K or the contribution-to-local-EF
  function may reuse the graph and layout only when their contracts explicitly
  permit it.
- Keep old dual-built artifacts and accepted experiment evidence intact, but label
  them as legacy rather than retroactively claiming strict offline-online graph
  identity.

Canonical validation should include:

- exact upper-graph checksum identity across attachment, layout finalization, and
  serving;
- replay parity for upper hit IDs, order, and score bits between offline tooling and
  the online router;
- rejection of changed graph edges, upper vectors, schema, attachments, memberships,
  or layout bindings;
- a guard proving that finalization and runtime-profile derivation do not rebuild
  the graph;
- explicit separation of legacy/ablation builders from the canonical build path.

## System Objectives

The following are required objectives, but their concrete mechanisms are
intentionally unspecified:

- maintain reasonable load balance across logical shards and physical workers;
- preserve search-topological locality;
- reduce unnecessary shard fanout;
- reduce unnecessary shard-local search work;
- control index expansion caused by replication;
- improve scalability while maintaining the required recall level.

These are objectives, not assumptions about the current implementation or claims
that existing experiments have already established them. Preserve contradictory
or negative experimental findings.

In particular, do not assume a specific load-balancing algorithm. Balanced K-means,
graph partitioning, capacity constraints, shard splitting, physical placement, or
other mechanisms are experimental choices unless explicitly established later.

## Open Implementation Space

The following are not architectural constraints and may be changed or explored:

- sampling strategy and navigation-graph parameters;
- initial partitioning method;
- topology-refinement algorithm;
- load-balancing mechanism;
- multi-assignment policy and thresholds;
- shard-selection policy;
- contribution estimator;
- Dynamic-EF or other search-budget function;
- HNSW parameters;
- storage, RPC, concurrency, and deployment implementation.

Do not promote parameters or heuristics from earlier prototypes into Orion design
requirements. If an experiment changes the upper graph or its navigation semantics,
rebuild and rebind every dependent attachment and layout artifact rather than
mixing generations.

## Rule of Thumb

A change is compatible with canonical Orion if it preserves this chain:

```text
data/query
    -> one immutable production global graph navigator
    -> shard-level navigation evidence
    -> data placement or query routing
    -> shard-local entry points
    -> contribution-aware local search
```

If a change breaks this chain, uses independently rebuilt offline and online upper
graphs, or bypasses checksum-bound graph identity, treat it as an ablation,
baseline, legacy path, or architectural redesign rather than silently changing
canonical Orion.
