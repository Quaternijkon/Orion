# Fixed-CNBR multi-assignment results

Status: `BMR_10` confirmation complete; online adoption gate `PASS`.

Date: 2026-08-27 protocol date; evidence completed at
`2026-08-28T03:32:22Z` in UTC.

## Conclusion

Under the frozen current `C_CNBR` load-balancing owner, a better post-owner
multi-assignment strategy exists for the tested four-host GloVe/Cosine P=32
contract.  `BMR_10` reduces index copies and passes the preregistered
matched-recall QPS gate against `current_all_max`.

This is not evidence that load balancing changed or improved: the owner bytes,
upper navigator, upper graph, source upper artifact, attachments, HNSW config,
resource limits, and physical placement were identical.  The only accepted
treatment was the L0 multi-assignment rule after owner freeze.

## Retained negative results

The original preregistered family did not produce a dual-dataset winner.
SIFT selected `cap2_rank_prefix4`, but that fixed candidate failed the GloVe
1,000-query tuning coverage gate.  The later exploratory rank-pruning family
also had no dual-dataset pass.  These results remain negative evidence and are
not overwritten by the post-exploratory `BMR_10` addendum.

## Offline confirmation

| Dataset | Current expansion | `BMR_10` expansion | Current -> `BMR_10` mean GT coverage | Current -> `BMR_10` hottest shard | Result |
|---|---:|---:|---:|---:|---|
| SIFT1M | 1.091263 | 1.079695 | 0.99670 -> 0.99654 | 89,291 -> 88,635 | PASS |
| GloVe tuning 1k | 1.138085 | 1.100000 | 0.97060 -> 0.96870 | 131,385 -> 126,661 | PASS |

The GloVe 9,000-query measurement split was not used by the selection or
offline confirmation.  The frozen selection manifest is:

- path:
  `/proj/intelisys-PG0/exp/orion-distributed/l1-balance-20260826/multi-assignment/bmr10-selection-v1/selection-manifest.json`;
- SHA-256:
  `4a89f57094bf251fa982972fdaf7f9d2799b05a286ff45bc73205cb17b78d79c`.

## Materialization and native preparation

The GloVe `BMR_10` production bundle and live collection passed the strict
native validator.

| Field | Value |
|---|---|
| Frozen owner SHA-256 | `0e42469f1fb30b76788f556834109562eddc1ba7cac12996b02d28938c5ea11d` |
| Generation | `3253273` |
| Artifact SHA-256 | `70bf09ab677c515d7c4d776de3daa702ecd0f97c715c1d790dff2744c8433bf7` |
| Assignment/layout SHA-256 | `7409c9d5735a2dcb2ae25feb568c698c09efb32cecad87917c34d515f9afb815` |
| Membership semantic SHA-256 | `8774a8a9328ea65c20ceb9c0be4a2d1b72edc0cff85c04666f9e92c0ede991ee` |
| Logical points | 1,183,514 |
| Physical copies | 1,301,865 |
| Exact extra-copy budget | 118,351 = `floor(N/10)` |
| Maximum copies per point | 2 |
| Upper replay | PASS, 480,000 ordered hit/score-bit comparisons |
| Live collection | `orion_l1_bmr10_p32_20260827_v1` |
| Physical topology | 4 hosts, 32 logical shards, 8 shards/host, round-robin |
| Server CPU contract | 16 logical CPUs/host, 64 total |

Bundle:
`/proj/intelisys-PG0/exp/orion-distributed/l1-balance-20260826/glove-p32/materialized-bmr10-g3253273-v1`.

Preparation evidence:
`/proj/intelisys-PG0/exp/orion-distributed/l1-balance-20260826/formal-ab/prepare-bmr10-v1/preparation_manifest.json`.

## Online interleaved A/B

Both arms used the frozen `u48 / upper-EF48 / base50 / factor14` search budget.
The 1,000-query tuning Recall@10 values were 0.9272 for current and 0.9235 for
`BMR_10`, so both remained inside the frozen `[0.90, 0.93)` budget band.
Saturation concurrency was selected independently from that tuning split:
current=32 and `BMR_10`=8.

The disjoint 9,000-query split was then used for recall and five counterbalanced
20-second measurement pairs.

| Metric | Current `C_CNBR + current_all_max` | `C_CNBR + BMR_10` |
|---|---:|---:|
| Held-out Recall@10 | 0.924744 | 0.922133 |
| Mean QPS | 3,765.98 | 3,821.11 |
| QPS CV | 0.511% | 0.283% |
| Physical copies | 1,346,939 | 1,301,865 |
| Expansion ratio | 1.138085 | 1.100000 |

`BMR_10/current` paired QPS ratio:

- arithmetic mean: `1.014654`;
- geometric mean: `1.014646`;
- two-sided 95% paired log-ratio CI: `[1.009204, 1.020118]`;
- faster pairs: `5/5`;
- gate decision: `BMR10_QPS_GATE_PASS`.

Thus the measured mean QPS improvement is about 1.46%, while physical copies
fall by 45,074 (3.35% of the current physical-copy count).  Recall is not
numerically identical: `BMR_10` is lower by about 0.00261 absolute, but both
arms satisfy the frozen-budget matched-recall contract and Recall@10 >= 0.90.

Formal evidence directory:
`/proj/intelisys-PG0/exp/orion-distributed/l1-balance-20260826/formal-ab/run-cnbr-current-vs-bmr10-v2`.

Primary evidence SHA-256 values:

- `summary.json`:
  `7285b5f876c85fc22a8739f847420661a35a64e4f040c6d798170044406d252e`;
- `completion-audit.json`:
  `fe5a0c97321346c82e7f45b5c415cc3e66f6934bf2b6c5e6d4d66fcb68b475ea`;
- `run_manifest.json`:
  `797288e4eec131b0df36985cc648dd150539c7e81e3daac07e97001a32781180`;
- `measurements.json`:
  `7122cdee5d2f65758a9debddae5053c3bf20c10527537ab39cc06934813bafa3`;
- `paired_rounds.csv`:
  `fabe8e3700cae4b200bd7896eee2c3416133a02939a7ee1fd29c5ec66dc15e26`;
- `concurrency_sweep.csv`:
  `887b89663ae680286255ca0c5b349cd707748fece1ca5c66f6e5f51e4c121ef3`.

The completion audit passed all 25/25 checks, including four-host/64-CPU
resources, P=32 round-robin placement, identical owner/upper graph/HNSW,
unchanged collections and repository source bytes during measurement, Recall,
stability, and paired-CI gates.

The invalid pre-query attempt without the required client CPU affinity is
retained under `run-cnbr-current-vs-bmr10-v1`; it did not reach tuning or
measurement queries and is not experimental evidence.

## Applicability boundary

The positive conclusion applies to GloVe-200-angular, cosine distance, four
physical AMD hosts, P=32 logical shards, the recorded 64-server-CPU contract,
the frozen C_CNBR owner and upper graph, and the recorded HNSW/runtime budget.
It does not establish the same QPS improvement for SIFT online serving, other
datasets, other P values, other physical placements, lifecycle operations, or
different runtime budgets.  `BMR_10` remains an explicit post-exploratory
candidate rather than a preregistered v1-family winner.
