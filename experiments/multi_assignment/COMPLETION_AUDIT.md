# Multi-assignment goal completion audit

Status: `COMPLETE` under the frozen matched-recall definition.

Objective audited: determine whether a multi-assignment policy better than the
current policy can retain low index expansion and deliver higher QPS at the
same recall level.

## Requirement-by-requirement verdict

| Requirement | Authoritative completion criterion | Evidence | Verdict |
|---|---|---|---|
| Better multi-assignment policy exists | One fixed candidate must pass every offline, identity, materialization, recall, stability, and paired-QPS gate | `BMR_10`; online decision `BMR10_QPS_GATE_PASS`; completion audit 25/25 | PROVED |
| Low index expansion | Fixed addendum gate `expansion_ratio <= 1.10` on SIFT and GloVe, with no fallback retuning | SIFT 1.079695; GloVe 1.099999662; exact GloVe extra-copy budget 118,351 = `floor(N/10)` | PROVED |
| Same recall level | Before the held-out split was used: same frozen `u48/upper-EF48/base50/factor14` budget; both 1,000-query tuning recalls in `[0.90,0.93)`; both disjoint held-out recalls >=0.90 | Tuning current/BMR: 0.9272/0.9235; held-out current/BMR: 0.924744/0.922133 | PROVED under the frozen matched-recall contract |
| Higher QPS | Stable 5--7 paired run and two-sided 95% paired BMR/current QPS-ratio CI lower bound >1 | Mean QPS 3,765.98 -> 3,821.11; CV 0.511%/0.283%; ratio geometric mean 1.014646; 95% CI [1.009204, 1.020118]; BMR faster 5/5 | PROVED |
| Load balance and multi-assignment are orthogonal | Owner, upper navigator/graph, source artifact, attachments, logical P, HNSW, resources, and physical placement identical; only post-owner L0 assignment may differ | Every cross-arm fairness bit PASS; identical owner SHA-256 `0e4246...a11d`; P=32 round-robin on four hosts; 64 server logical CPUs | PROVED |
| Candidate selection did not use online measurement queries | Frozen selection manifest must bind `glove_online_measurement_queries_seen=false` before online confirmation | Selection SHA-256 `4a89f5...d79c`; run identity check PASS | PROVED |
| Negative searches are retained | Failed original fixed family and rank-pruning exploration must not be overwritten | `cap2_rank_prefix4` GloVe failure and rank-pruning no-winner result retained in `RESULTS.md` and their original evidence directories | PROVED |
| Applicability is bounded | Do not generalize beyond the measured dataset/topology/runtime contract | Positive claim limited to GloVe/Cosine, four physical hosts, P=32, frozen C_CNBR owner/upper graph and recorded runtime budget | PROVED |

## Recall interpretation

The two held-out Recall@10 values are not numerically identical: the absolute
gap is `0.0026111111`, with `BMR_10` lower.  Exact equality was never the frozen
experimental definition because finite-query recall estimates are discrete.
The protocol operationalized “same recall level” before measurement as the
same checksum-bound runtime budget, the same target band on tuning, and both
held-out values clearing Recall@10 0.90.  Those conditions all pass.  The
positive conclusion must therefore be stated as matched-recall-at-the-frozen
level, not equal-recall-to-all-decimal-places.

## Evidence identity

Formal run:
`/proj/intelisys-PG0/exp/orion-distributed/l1-balance-20260826/formal-ab/run-cnbr-current-vs-bmr10-v2`.

- `summary.json`: `7285b5f876c85fc22a8739f847420661a35a64e4f040c6d798170044406d252e`
- `completion-audit.json`: `fe5a0c97321346c82e7f45b5c415cc3e66f6934bf2b6c5e6d4d66fcb68b475ea`
- `run_manifest.json`: `797288e4eec131b0df36985cc648dd150539c7e81e3daac07e97001a32781180`
- `measurements.json`: `7122cdee5d2f65758a9debddae5053c3bf20c10527537ab39cc06934813bafa3`
- `paired_rounds.csv`: `fabe8e3700cae4b200bd7896eee2c3416133a02939a7ee1fd29c5ec66dc15e26`
- `concurrency_sweep.csv`: `887b89663ae680286255ca0c5b349cd707748fece1ca5c66f6e5f51e4c121ef3`

Frozen selection:
`/proj/intelisys-PG0/exp/orion-distributed/l1-balance-20260826/multi-assignment/bmr10-selection-v1/selection-manifest.json`,
SHA-256 `4a89f57094bf251fa982972fdaf7f9d2799b05a286ff45bc73205cb17b78d79c`.

The invalid v1 launch failed on benchmark-client CPU affinity before tuning or
held-out queries.  It is retained as an invalid attempt and is not evidence.

## Final scoped answer

Yes.  For the recorded GloVe-200-angular/Cosine, four-physical-host, P=32
contract, `BMR_10` is a better post-owner multi-assignment policy than
`current_all_max`: it lowers expansion from 1.138085 to 1.100000 and raises
mean QPS by about 1.46% while passing the frozen matched-recall gate.  This
does not establish the same improvement on SIFT online serving, other datasets,
other shard counts, placements, or runtime budgets.
