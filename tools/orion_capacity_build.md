# Orion fixed base

Adopted as the current implementation by user instruction on 2026-11-17.
Access-probability refinement is always part of a new canonical build; there is
no enable/disable flag. This document is versioned with the implementation.

The only active build/search pipeline is:

1. Build and freeze the native production upper graph once.
2. Search that same graph for the full dataset and save ordered attachments.
3. Create the initial capacity-feasible partition from attachment co-occurrence.
4. Search up to 65,536 deterministic input-vector proxy queries, excluding upper
   sample rows, with the online upper budgets. Save ordered IDs and distances.
5. Replay the fixed whitebox rule under candidate owners. A shard contributes
   exactly one request per query, regardless of its number of upper hits.
6. Refine owners to reduce peak predicted request probability, without increasing
   average fan-out or the attachment co-occurrence cut. A conservative affected-row
   bound preserves the native physical-capacity limit; final memberships are
   independently recounted and checked by Rust.
7. Install the finalized bundle and build one local HNSW per physical shard.
8. Online search always follows native upper navigation, whitebox selection,
   navigation-derived MultiEP entries, contribution-aware local EF, and global
   deduplicated Top10 merge.

Self-vote construction/repartition and decision-tree/oracle routing selectors are
retired. Their source may remain for historical tests, but production entry points
cannot activate them. There is no `--partition-method` or strategy switch.

Search-budget and threshold numbers remain explicit so recall can be calibrated;
these are numeric configurations of the one algorithm. M and ef_construct remain
ordinary construction parameters and must match controls in comparisons.

## Build

For the eight-host SIFT100M workflow, use the accelerated launcher and reference
manual: [orion_fast_build.md](orion_fast_build.md). It uses all available physical
cores on each of eight hosts for attachment generation and lower indexing, and
sizes controller materialization to its CPU and memory budget. Affinity and cgroup
limits are respected; the former 8-thread / 16-process / 48-core caps are removed.
This accelerates experiment preparation. Search resources, algorithm and budgets
retain their experimental settings. Actual per-host HNSW build thread counts are
recorded because insertion scheduling can change graph bytes.

Large Python builds (more than 1,000,000 rows) now require `--reuse-prepared` or
an explicit `--allow-serial-attachments`. The fast launcher uses the latter only
to obtain a short native parity prefix before automatically distributing the
remaining attachment work. Missing or failed parallel execution stops the run;
it does not silently become a full serial pass. Direct native `prepare-capacity`
remains a low-level serial phase, not the recommended full-scale entry point.

Build the native layout tool:

```sh
cargo build --release --locked -p collection --bin orion_build_layout
```

The Python planner requires NumPy and PyMetis; the HDF5 adapter additionally
requires h5py.

```sh
python tools/orion_capacity_partition.py \
  --request /absolute/path/request.json \
  --whitebox-config /absolute/path/whitebox.json \
  --output /absolute/path/new-layout \
  --rust-layout-binary /absolute/path/orion_build_layout
```

This direct command is for small inputs. For a large input, use the fast entry
above or supply an already validated `--reuse-prepared` directory; explicitly
opting into a full serial pass is intended only for deliberate diagnostics.

The optional `--reuse-prepared` input can reuse the exact checksummed frozen graph,
normalized vectors and attachments. The algorithm is identical. Existing outputs
are never overwritten. The optional maximum replication factor is a resource
constraint, not a strategy selector. Native strict maximum-vote multi-assignment
and owner-entry completion are preserved.

The HDF5 adapter `orion_native_layout.py` invokes this same pipeline and also
requires `--whitebox-config`. With `--uri` and `--collection`, it installs only the
completed bundle. Legacy self-vote `build`, `build-install` and `repartition` CLI
commands are disabled. Low-level `prepare-capacity`, `access-probes`, and
`capacity-partition` commands are phases of the same build, not alternate builders.

Configure every serving process with `ORION_WHITEBOX_PRUNER` pointing to the
numeric whitebox configuration. Missing configuration fails closed on online
routing. `ORION_LEARNED_PRUNER` and oracle shard-mask configuration are rejected.

## Guarantees and limits

- New bundles are tagged `orion-layout-v5-access-probability-capacity`.
- Historical v4 capacity bundles remain readable; no alternate legacy constructor
  is enabled by that compatibility. Historical self-vote verification uses its
  retained historical binary.
- The production upper graph, vector preprocessing and attachment checksums remain
  identical between preparation, refinement and serving.
- Access balancing assumes equal cost per shard request and uses input vectors as
  a proxy for the future query distribution. It does not consume evaluation
  queries, ground-truth neighbors or measured CPU costs, and cannot guarantee
  balance when the real query distribution differs.
- Mean fan-out is the sum of per-shard request probabilities; they are not
  normalized to sum to one. Multiple upper hits in one shard are deduplicated.
- Refinement is fixed and bounded. If no legal improving move exists, it retains
  the feasible initial owners and reports that outcome without fabricating gain.
- `access-probes/`, `access-balance.json`, `capacity-plan.json`, and native manifests
  record the input identities, exact counts, constraints and accepted changes.
- `verify --replay` independently replays all saved attachments and memberships;
  proxy refinement additionally checks its incremental route statistics by full
  replay of the cached proxy hit lists.

## Validation

The focused tests cover per-query shard deduplication, whitebox replay, capacity
and fan-out constraints, rejected legacy selectors, and a real native build with
independent full replay:

```sh
cargo test --locked -p collection --lib orion::
python tests/tools/test_orion_access_balance.py
python tests/tools/test_orion_capacity_partition.py
python tests/tools/test_orion_native_layout_args.py
ORION_LAYOUT_BINARY="$PWD/target/release/orion_build_layout" \
  python tests/tools/test_orion_capacity_native.py
```

The retained SIFT100M/Recall@10=0.95 experiment measured median QPS 69,427.20
for this base and 67,482.67 for the historical control, at actual recall 0.95030
and 0.95085 respectively. This is a small observed increase (2.88%), not a
guaranteed performance claim: real-query fan-out barely changed, and independently
built lower indexes, shard placement and measurement-time conditions differ.
Formal measurements, parameters and limits are retained in
`/home/wangyiduo/results/2026-11-16_sift100m_access_base/` in the experiment workspace.
