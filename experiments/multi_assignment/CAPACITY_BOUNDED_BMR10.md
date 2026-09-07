# C_CNBR with capacity-bounded BMR_10 placement

Status: dual-dataset offline gates pass; online QPS confirmation pending.

## Design

This candidate keeps the user's clustering plus upper self-navigation owner
unchanged.  Load guarantees are added only after that owner and the ordinary
BMR_10 copy budget are frozen:

```text
immutable natural upper HNSW
        -> C_CNBR clustering + self-navigation repair
        -> freeze owner checksum
        -> ordinary BMR_10 decides one or two copies per point
        -> capacity repair moves those copies only among attachment-supported shards
        -> freeze final L0 membership
```

The fixed experimental policy is `BMR_10_CAP35`:

```text
physical-copy lower bound = 0.35 * mean
construction upper bound = 3.20 * mean
maximum target vote loss = 2
maximum direct-repair passes = 8
```

The loose construction upper bound prevents the lower-bound repair from
creating a worse hotspot.  The independent adoption gate is stricter: the
candidate's hottest shard and load CV must not exceed ordinary BMR_10.

## Hard guarantees

For an accepted layout:

- the C_CNBR owner bytes and online shard-selection topology are unchanged;
- every point has exactly the same copy count as ordinary BMR_10;
- total physical point count and expansion ratio are unchanged;
- every copy is assigned only to a shard occurring in that point's frozen
  top-10 upper attachments;
- target vote loss is at most two;
- every shard contains at least `floor(0.35 * physical_points / P)` copies;
- an infeasible attachment candidate graph fails closed and records all
  residual under/over shards.

The implementation first attempts deterministic direct moves.  If those moves
cannot satisfy the band, it invokes the existing grouped integer capacity
circulation.  No current worker load, query latency, QPS, or ground truth is an
input to placement.

## Full-query offline results

Both datasets use P=32, their frozen C_CNBR owner, all normal L0 attachments,
ordinary BMR_10 copy selection, and all 10,000 bound queries.

### GloVe-200-angular

| Metric | C_CNBR+BMR_10 | BMR_10_CAP35 | Change |
|---|---:|---:|---:|
| Physical points | 1,301,865 | 1,301,865 | identical |
| Minimum shard copies | 8,828 | 14,239 | +61.3% |
| Maximum shard copies | 126,661 | 125,597 | -0.84% |
| Load CV | 0.62597 | 0.61189 | -2.25% |
| Minimum/mean | 0.21699 | 0.35000 | hard floor satisfied |
| GT routing coverage | 0.96721 | 0.96609 | -0.00112 |
| Full-coverage fraction | 0.7882 | 0.7839 | -0.0043 |
| Routed shards mean | 8.0824 | 8.1460 | +0.79% |
| Route EF sum mean | 1117.3234 | 1120.5034 | +0.28% |

All frozen offline gates pass.

### SIFT1M

| Metric | C_CNBR+BMR_10 | BMR_10_CAP35 | Change |
|---|---:|---:|---:|
| Physical points | 1,079,695 | 1,079,695 | identical |
| Minimum shard copies | 11,458 | 11,809 | +3.1% |
| Maximum shard copies | 88,635 | 88,606 | -0.03% |
| Load CV | 0.55689 | 0.55635 | -0.10% |
| Minimum/mean | 0.33959 | 0.35000 | hard floor satisfied |
| GT routing coverage | 0.99654 | 0.99652 | -0.00002 |
| Full-coverage fraction | 0.9701 | 0.9699 | -0.0002 |
| Routed shards mean | 4.7755 | 4.7752 | -0.01% |
| Route EF sum mean | 299.0332 | 299.0272 | unchanged |

All frozen offline gates pass.

## Interpretation

This is the first candidate in this experiment line that both:

1. provides a real lower-bound guarantee on final physical-copy distribution;
2. preserves the clustering plus self-voting navigation owner and passes the
   routing-topology gates on both datasets.

It is therefore a credible replacement candidate for the current post-owner
BMR_10 placement rule.  It is not yet proven better online: materialization and
a held-out, counterbalanced four-host QPS A/B against `C_CNBR+BMR_10` are still
required before adoption.
