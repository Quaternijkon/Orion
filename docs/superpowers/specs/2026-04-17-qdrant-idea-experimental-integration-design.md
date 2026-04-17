# Qdrant Experimental Integration Of `method0andmethod3-try1` Design

## Goal

Integrate the algorithm implemented in:

- `/home/taig/dry/dHNSW/hnswlib/examples/cpp/method0andmethod3-try1.cpp`

into Qdrant in the smallest possible scope needed for fair same-platform benchmarking.

The purpose is not to productize the algorithm yet. The purpose is to make the algorithm run:

- inside Qdrant's internal vector-index/search call stack
- using Qdrant's own HNSW implementation as the graph/index primitive
- without modifying the original C++ source file

so that later benchmark comparisons between "Qdrant native HNSW" and "the user's idea" are performed on the same platform and at the same implementation level.

## Non-Goals

This first stage does **not** aim to provide:

- REST or gRPC configuration exposure
- collection-level public configuration flags
- production-ready persistence compatibility
- distributed-cluster feature support
- multi-vector generality
- payload-filter completeness
- end-to-end public API support

This stage is an **internal experimental integration** only.

## Principle

The algorithm must be reproduced **logically**, not by embedding `hnswlib` as a runtime dependency.

That means:

- preserve the algorithm structure, default control flow, and default parameters from `method0andmethod3-try1.cpp`
- replace `hnswlib` graph/index operations with equivalent Qdrant HNSW primitives

This is required to keep the final benchmark on the same platform:

- same vector scorer path
- same segment abstraction
- same Rust codebase
- same benchmark harness

## Scope Of "One-To-One" Reproduction

For this stage, "one-to-one" means reproducing the default algorithm path currently used by the C++ program.

The integration should preserve the following default behavior from the source:

- two-tier structure: `up_tier` + `down_tier`
- default `PARTITION_METHOD = 4`
- default `SEARCH_ALL_SHARDS = false`
- default `USE_MULTIPLE_ENTRY_POINTS = true`
- default `DYNAMIC_EF_SEARCH = true`
- default `BUILD_L0_HIERARCHICAL = true`
- default constants:
  - `P = 31`
  - `M_GLOBAL = 32`
  - `EF_CONSTRUCTION = 100`
  - `EF_SEARCH_UP = 100`
  - `EF_SEARCH_DOWN = 60`
  - `K_OVERLAP = 10`
  - `k_nearest = 10`

The following environment override branches are explicitly out of scope for stage 1:

- `PARTITION_METHOD_OVERRIDE`
- `SEARCH_ALL_SHARDS_OVERRIDE`
- `NUM_QUERIES_OVERRIDE`
- later experimental branches not part of the default path

## Why Qdrant HNSW Must Replace `hnswlib`

The original code relies on `hnswlib` APIs such as:

- graph construction
- nearest-neighbor search
- entry point traversal
- graph neighborhood access at level 0

Inside Qdrant, these must map to Qdrant-native HNSW primitives instead of carrying `hnswlib` along.

The relevant Qdrant code paths are:

- `/home/taig/dry/qdrant/lib/segment/src/index/vector_index_base.rs`
- `/home/taig/dry/qdrant/lib/segment/src/index/hnsw_index/hnsw.rs`
- `/home/taig/dry/qdrant/lib/segment/src/index/hnsw_index/graph_layers.rs`
- `/home/taig/dry/qdrant/lib/segment/src/entry/entry_point.rs`
- `/home/taig/dry/qdrant/lib/collection/src/collection_manager/segments_searcher.rs`

Important equivalence observations:

- `searchKnn` maps naturally to Qdrant HNSW search entry points
- search-time `ef` maps to `SearchParams.hnsw_ef`
- multiple/custom entry points are supported by Qdrant graph search
- graph link access can be reconstructed using Qdrant's graph-layer link iterators

Therefore, the correct design is:

- keep the algorithm semantics
- rewrite the underlying graph/index calls against Qdrant's internal HNSW implementation

## Integration Level

The integration point should be the **segment/vector-index layer**, not the REST layer.

The algorithm should become a new internal vector-index implementation that can be invoked by the same search path used by segment search.

This ensures benchmark fairness while keeping public API work out of scope.

## Proposed Architecture

### 1. New Experimental Vector Index

Add a new internal experimental index implementation under the segment crate, for example:

- `/home/taig/dry/qdrant/lib/segment/src/index/idea_index/mod.rs`
- `/home/taig/dry/qdrant/lib/segment/src/index/idea_index/config.rs`
- `/home/taig/dry/qdrant/lib/segment/src/index/idea_index/up_tier.rs`
- `/home/taig/dry/qdrant/lib/segment/src/index/idea_index/partition.rs`
- `/home/taig/dry/qdrant/lib/segment/src/index/idea_index/down_tier.rs`
- `/home/taig/dry/qdrant/lib/segment/src/index/idea_index/search.rs`
- `/home/taig/dry/qdrant/lib/segment/src/index/idea_index/idea_index.rs`

This implementation will satisfy Qdrant's existing `VectorIndex` trait.

### 2. Extend `VectorIndexEnum`

Add a new experimental enum variant in:

- `/home/taig/dry/qdrant/lib/segment/src/index/vector_index_base.rs`

Example intent:

- `VectorIndexEnum::IdeaExperimental(...)`

This allows the new implementation to sit beside:

- `Plain`
- `Hnsw`

and participate in the same internal search dispatch.

### 3. Rebuild Your Algorithm On Top Of Qdrant HNSW Primitives

The new experimental index will internally manage:

- one Qdrant-native HNSW graph for `up_tier`
- a set of Qdrant-native HNSW graphs for `down_tier` shards

The algorithm-specific logic that remains custom:

- selecting the `up_tier` subset
- computing L0-to-L1 attachment and overlap candidates
- L1 partitioning via balanced KMeans plus topology refinement
- shard expansion/fission logic
- multi-assignment / routing state
- query-time visited-shard logic
- dynamic shard-level search policy
- top-k merge behavior

The graph search/build primitive that becomes Qdrant-native:

- HNSW build
- HNSW search
- entry-point traversal
- graph-neighborhood access

### 4. Benchmark Entry, Not Public API

Instead of exposing the index through collection config first, add a benchmark or internal executable path that can:

- load the dataset
- build the native `HNSWIndex`
- build the new `IdeaExperimentalIndex`
- run the same query workload for both
- emit comparable metrics

This can live as a segment-level benchmark or internal experimental binary.

That is sufficient for the user's current requirement.

## Data Flow

### Build Path

1. Load base vectors into Qdrant-compatible vector storage.
2. Select the `up_tier` subset according to the algorithm's default path.
3. Build Qdrant-native `up_tier` HNSW.
4. Compute L0-to-L1 attachment and weights.
5. Partition L1 nodes according to the default algorithm path.
6. Expand shards via the algorithm's default fission path.
7. Assign all points to shard-local memberships.
8. Build one Qdrant-native `down_tier` HNSW per shard.

### Search Path

1. Run `up_tier` search with `EF_SEARCH_UP`.
2. Derive candidate entry points / candidate shards using the same logic as the C++ default path.
3. Determine visited shards and dynamic local search policy.
4. Search each selected `down_tier` shard with Qdrant-native HNSW using the same default logic and `EF_SEARCH_DOWN` policy.
5. Merge candidates and return global top-k.

## Minimal Stage-1 Capability

Stage 1 should support exactly what is needed for benchmark equivalence:

- dense vectors only
- one vector field
- cosine/angular path first
- unfiltered search first
- batch benchmarking not required initially
- one internal benchmark harness is enough

This is sufficient because the user's current request is benchmark-oriented, not productization-oriented.

## Error Handling Expectations

Stage 1 must fail clearly when used outside its supported scope.

Examples:

- unsupported distance metric -> explicit runtime error
- payload filter used -> explicit runtime error
- multi-vector segment -> explicit runtime error
- unsupported config path -> explicit runtime error

This keeps the experiment honest and prevents accidental misuse.

## Testing Strategy

### Unit-Level

Test the logic that does not require full benchmark execution:

- `up_tier` subset selection
- partition assignment
- fission decisions
- routing decisions
- candidate merge logic

### Equivalence-Level

Create small deterministic datasets and verify:

- the integrated Qdrant implementation follows the expected default-path behavior
- relative control-flow outputs match the design intent

Exact byte-for-byte graph equivalence with `hnswlib` is not required and should not be promised.

### Benchmark-Level

Provide one benchmark harness that compares:

- Qdrant native `HNSWIndex`
- Qdrant internal `IdeaExperimentalIndex`

on the same dataset and in the same process framework.

## Risks

### 1. Hidden mismatch between `hnswlib` and Qdrant HNSW behavior

This is unavoidable at the implementation-detail level.
The mitigation is to preserve algorithm semantics, not low-level graph identity.

### 2. Scope explosion

The algorithm includes many experimental branches and overrides.
Mitigation:

- implement only the default path in stage 1

### 3. Benchmark contamination by unrelated Qdrant features

Mitigation:

- keep stage 1 benchmark-oriented
- avoid public API integration for now

## Success Criteria

Stage 1 is successful if:

- the original C++ file remains untouched
- Qdrant gains an internal experimental index implementing the default path of the user's algorithm
- the implementation uses Qdrant HNSW primitives instead of `hnswlib`
- there is a Qdrant-internal benchmark path that can compare:
  - native HNSW
  - the integrated idea implementation

## Implementation Recommendation

Proceed with:

- experimental internal vector-index integration
- Qdrant-native HNSW as the only low-level graph implementation
- default algorithm path only
- benchmark harness first

Do not begin with:

- REST/gRPC exposure
- full persistence guarantees
- complete override/config coverage
- distributed product behavior

That should be postponed until the benchmark case demonstrates value.
