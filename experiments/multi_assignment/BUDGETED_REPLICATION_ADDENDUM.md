# Budgeted multi-assignment confirmation addendum

Status: frozen after SIFT development and the 1,000-query GloVe tuning screen,
before materialization and before any use of the disjoint 9,000-query online
measurement split.

Date: 2026-08-27.

## Why this is an addendum

The preregistered v1 protocol required a 20% reduction of each dataset's own
excess expansion.  That relative gate selected `cap2_rank_prefix4` on SIFT,
but the selected arm failed the unchanged GloVe tuning coverage gate.  A
second exploratory rank-pruning family also had no dual-dataset pass.  These
negative outcomes remain authoritative and are not overwritten.

The failure exposed a cross-dataset problem in the relative expansion gate:
the same fixed cap-2 rule naturally yields about 7.97% extra copies on SIFT and
10.56% on GloVe, so requiring the same relative reduction from two different
starting expansions does not define one portable index-size budget.

This addendum therefore freezes one explicitly post-exploratory confirmation
candidate with a single absolute extra-copy budget.  It does not claim that
the candidate was preregistered or that GloVe was untouched during selection.
The disjoint 9,000-query online split remains the sole QPS confirmation set.

## Frozen candidate: `BMR_10`

`BMR_10` means budgeted max-popularity replication with a 10% extra-copy
budget.  It freezes the accepted `C_CNBR` owner and applies the following rule
after owner checksum freeze:

1. map each point's ordered top-10 attachments to their `C_CNBR` owners;
2. if no shard has at least two votes, assign only the first attachment owner;
3. otherwise rank max-vote tied shards by earliest attachment position and
   shard id; the first is primary and only the second is replica-eligible;
4. score that secondary shard by the maximum frozen upper proxy mass among the
   point's supporting attachments owned by the secondary shard;
5. globally retain at most `floor(0.10 * N)` eligible secondary copies, ordered
   by descending score and then ascending point id.

Every point has one copy, no point has more than two, every copy has direct
navigation evidence, and the load-balancing owner is byte-identical across
arms.  The policy may use only the already-frozen upper proxy mass and the
normal L0 attachment stream; it may not inspect query vectors, ground truth,
observed QPS, physical placement, or candidate physical-copy loads while
choosing replicas.

## Confirmation gates

The fixed absolute low-inflation gate is `expansion_ratio <= 1.10` on both
datasets.  The original topology/coverage gates remain unchanged:

- mean ground-truth routing coverage drop at most 0.002 absolute;
- full-coverage fraction drop at most 0.02 absolute;
- zero-coverage queries at most baseline + 2 and at most 0.1%;
- mean routed shards and route EF at most 1.05x baseline;
- hottest physical-copy shard no larger than baseline;
- exact `C_CNBR` owner, graph, attachments, query, ground-truth, and proxy-mass
  checksum identity.

SIFT uses all 10,000 development queries.  GloVe selection and failure checks
use only the first 1,000 tuning queries.  There is no fallback to another score
or budget if `BMR_10` fails materialization or online confirmation.

The online test compares:

```text
C_CNBR + current_all_max  <->  C_CNBR + BMR_10
```

on the same four physical hosts, 32 logical shards, 64 server logical CPUs,
round-robin placement, upper graph, HNSW settings, resource limits, and query
split.  Select saturation concurrency from the 1,000-query tuning set, then
run 5--7 paired counterbalanced repeats on the disjoint 9,000-query split.

`BMR_10` is better only if both held-out Recall@10 values are at least 0.90,
the matched-recall contract is satisfied, QPS CV is at most 5%, and the
two-sided 95% paired `BMR_10/current` QPS-ratio confidence interval has a lower
bound strictly above 1.0.  Otherwise retain `current_all_max`.
