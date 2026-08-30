# Owner x multi-assignment QPS tournament

Status: frozen before any tournament collection is prepared or any tournament
query is issued.

Date: 2026-08-28.

## Objective

Find the highest-QPS combination of the currently available L1 owner/load
balancing strategies and post-owner L0 multi-assignment strategies under the
same GloVe-200-angular, cosine, P=32, four-host, 64-server-CPU and
Recall@10-about-0.90 contract.

Offline balance, index expansion, routing coverage, or a prior pairwise result
cannot by itself establish the winner.  Every structurally valid combination
is eligible for the online tuning screen, including combinations that failed a
stricter relative offline coverage gate, because the objective here is QPS at
the absolute online Recall@10 target rather than adoption under the earlier
fixed-owner protocol.

## Frozen candidate matrix

The owner axis is:

- `C_CNBR`;
- `HISTORICAL_PRIMARY`, defined as the first membership of each upper node in
  the checksum-bound historical Orion artifact; this is not a claim that the
  historical full layout is reproduced;
- `CCNB`, defined as the first membership of each upper node in the
  checksum-bound capacity-constrained artifact; the tournament recomputes the
  post-owner assignment and does not reuse CCNB's historical L0 layout.

The multi-assignment axis is:

- `current_all_max`: copy to every maximum-vote shard when the maximum vote is
  at least two, otherwise use the first attachment owner;
- `single_rank`: keep exactly one maximum-vote shard, ordered by earliest
  attachment rank and then shard id;
- `BMR_10`: keep the primary and globally retain at most `floor(0.10*N)`
  evidence-supported secondary copies by the frozen proxy-mass score.

The Cartesian product contains exactly nine candidates.  Owner bytes are
frozen before full L0 attachments are opened.  Every bundle must reproduce the
corresponding row of the frozen owner-policy matrix byte-for-byte.

## Common online contract

- dataset: GloVe-200-angular;
- distance: cosine;
- top-k: 10;
- physical hosts: four;
- logical shards: P=32, round-robin eight shards per host;
- Qdrant CPU: 16 logical CPUs per host, 64 total;
- lower HNSW: M=32, efConstruction=200;
- search budget: upper-k=48, upper-ef=48, dynamic-ef base=50, factor=14;
- query batch size: 200;
- tuning split: queries `[0,1000)` only;
- held-out final split: queries `[1000,10000)`, unseen until finalists are
  frozen;
- accepted Recall@10 band: `[0.90,0.93)`.

If a candidate misses the tuning recall band, it is not silently discarded or
given an unbounded EF.  A finite checksum-bound runtime-profile sweep may be
run using the same tuning split, with the smallest reasonable budget entering
the band.  The selected profile then remains frozen for the held-out test.

## Tuning-only online screen

For each candidate:

1. validate the bundle, upper replay, native preparation, placement, HNSW and
   resource identities;
2. measure tuning Recall@10;
3. select saturation concurrency from the common finite grid
   `1,2,4,8,16,32,64`;
4. run three counter-positioned 10-second QPS measurements on tuning queries;
5. record QPS mean/CV, latency, routed-shard evidence, physical-copy balance,
   index expansion and resource telemetry.

The tuning ranking uses mean QPS only among candidates in the accepted recall
band with QPS CV at most 5%.  The top two are frozen as finalists.  Tuning QPS
does not constitute the final conclusion.

## Held-out final gate

The two frozen finalists are compared with five counterbalanced paired
20-second rounds on the disjoint 9,000-query split, extending directly to seven
rounds if either QPS CV exceeds 5%.

A combination is the winner only if:

1. both finalists have held-out Recall@10 at least 0.90 and satisfy their
   frozen tuning-selected profile contract;
2. both QPS CV values are at most 5%;
3. the two-sided 95% paired winner/runner-up QPS-ratio confidence interval has
   lower bound strictly above 1.0; and
4. collection identity, resources, repository source, placement and artifact
   checksums remain unchanged during measurement.

If the confidence interval crosses 1.0, report no statistically demonstrated
single winner.  Retain all negative and failed candidates as evidence.

## Applicability boundary

The result is specific to this GloVe/Cosine/P=32/four-host/64-CPU/runtime
contract.  It does not establish the best combination for other M values,
datasets, placements or budgets.  The existing M=1..32 scalability curve must
remain labeled by the exact combination used at every point until a separate
full-range tournament is completed.
