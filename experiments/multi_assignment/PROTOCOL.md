# Orion fixed-CNBR multi-assignment protocol

Status: frozen before the candidate screen is executed.

## Causal boundary

This experiment freezes the currently accepted `C_CNBR` load-balancing result
and changes only the L0 point-to-shard replication rule that runs after the L1
owner checksum is frozen.  Every arm must use the same production upper graph,
ordered L1 labels, `C_CNBR` owner bytes, top-10 L0 attachments, 32 logical
shards, four physical hosts, round-robin physical placement, lower HNSW
settings, query split, and runtime-profile procedure.

The load balancer must not be rebuilt, repaired, retuned, or informed by a
candidate's resulting physical-copy loads.  No candidate may invoke KMeans,
CNBR, topology refinement, fission, L0 capacity flow, or physical placement
optimization.  This makes load balancing and multi-assignment separate causal
axes.

The frozen owner checksums are:

- SIFT1M `C_CNBR`: `f00ccb35f6dba918c214a64e62449776ccd6a261362a7c6cbb07e69a0b19099b`;
- GloVe-200-angular `C_CNBR`: `0e42469f1fb30b76788f556834109562eddc1ba7cac12996b02d28938c5ea11d`.

## Current rule and fixed candidate family

For every L0 point, map its ordered top-10 upper attachments to their frozen
L1 owners and count owner votes.  The current rule is:

```text
if max_vote >= 2:
    copy to every shard tied at max_vote
else:
    assign only to the owner of the first attachment
```

The candidate family is fixed before screening:

- `current_all_max`: unchanged baseline;
- `single_rank`: no replication; among max-vote ties choose the shard whose
  first attachment occurs earliest;
- `cap2_rank`: keep at most the first two max-vote shards by earliest
  attachment rank;
- `cap3_rank`: keep at most the first three max-vote shards by earliest
  attachment rank;
- `cap2_rank_prefix4`, `cap2_rank_prefix6`, `cap2_rank_prefix8`: as
  `cap2_rank`, but keep the second shard only when its first supporting
  attachment occurs in the first 4, 6, or 8 positions respectively;
- `cap2_shard_id`: deterministic control that caps at two by shard id rather
  than navigation rank.

All ties after first attachment rank use the numerical shard id.  Every policy
keeps at least one navigation-supported shard and is deterministic.  Candidate
selection cannot introduce another prefix, cap, score, or dataset-specific
parameter.

## Offline screen and low-expansion gate

SIFT1M is the development dataset.  All 10,000 SIFT queries may be used to
select one candidate.  Selection uses the following fail-closed gates relative
to `current_all_max` on the same frozen owner:

- physical point count and expansion ratio are reported exactly;
- the excess expansion `(expansion - 1)` must fall by at least 20%;
- mean ground-truth routing coverage may fall by at most 0.002 absolute;
- full-ground-truth-coverage query fraction may fall by at most 0.02 absolute;
- zero-coverage queries may increase by at most two and remain at most 0.1%;
- mean routed shards and summed route EF may not exceed 1.05x baseline;
- the hottest physical shard's copy count may not exceed baseline;
- owner, graph, attachment, query-hit, and ground-truth checksums must remain
  unchanged.

Among SIFT candidates passing every gate, select the one with the lowest
expansion ratio.  Exact ties use the candidate order listed above.  GloVe then
evaluates only the first 1,000 tuning queries as a no-retuning cross-dataset
gate, while expansion and physical-copy loads still use the full GloVe corpus.
If the already-selected SIFT winner fails a GloVe gate, no fallback candidate
is selected.

The disjoint 9,000-query GloVe measurement split must remain unseen until the
online A/B.

## Online matched-recall adoption gate

Materialize the unchanged `C_CNBR + current_all_max` baseline and the selected
`C_CNBR + candidate` arm with checksum-bound assignment bytes.  Use the same
four physical hosts, 32 logical shards, 64 server logical CPUs, round-robin
placement, upper graph, HNSW parameters, and resource limits.

Use the 1,000-query tuning split to select saturation concurrency.  Retain the
frozen `u48 / upper-EF48 / base50 / factor14` budget only if both arms land in
the preregistered recall band; otherwise use a symmetric finite runtime-profile
sweep.  Run 5--7 paired, counterbalanced, interleaved repeats on the disjoint
9,000-query split with QPS CV at most 5%.

A better multi-assignment strategy exists only if:

1. every identity and offline gate passes;
2. both held-out Recall@10 values are at least 0.90 and are matched by the
   frozen-budget or symmetric-sweep contract;
3. the candidate keeps the required low-expansion reduction; and
4. the two-sided 95% paired confidence interval for candidate/baseline QPS has
   a lower bound strictly above 1.0.

Offline routing coverage, smaller indexes, lower route EF, or a point QPS
estimate cannot substitute for the online gate.
