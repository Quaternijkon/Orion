# Stage 15: one non-empty HNSW segment per logical shard

Stage 14 removed worker/client SMT-core overlap, but its physical QPS curve is
not a clean measurement of HNSW graph partitioning.  Qdrant's default
`max_segment_size` was derived from the single indexing thread and capped an
indexed segment at roughly 256 MB of vector storage.  A logical shard could
therefore contain several independent HNSW segments.

Observed Stage 14 total segment counts demonstrate the confound:

| Dataset / method | M=1 | M=2 | M=3 | M=4 |
|---|---:|---:|---:|---:|
| SIFT Random | 3 | 4 | 6 | 8 |
| GloVe Random | 5 | 6 | 9 | 8 |

With `default_segment_number=1`, one segment per local shard remains empty and
appendable.  Consequently, GloVe Random searched approximately 4/4/6/4
non-empty HNSW graphs per query at M=1/2/3/4, while SIFT searched 2/2/3/4.
This explains both the GloVe M=3 discontinuity and most of the apparent GloVe
M=1-to-M=4 speedup.

## Stage 15 control

- Preserve Stage 14 physical-core isolation and all validity gates.
- Preserve fixed `efSearch=24` for SIFT1M. A single GloVe HNSW graph at the
  former `efSearch=192` reaches only 0.8722 Recall@10, because the Stage 13/14
  value was calibrated while one logical shard contained several independently
  searched HNSW segments. Recalibrate GloVe only over the bounded grid
  `192, 224, 256, 320, 384`, select the minimum tuning value reaching 0.90,
  and require a disjoint 9,000-query holdout to pass. This selects
  `efSearch=320` (holdout Recall@10 = 0.9027), with a hard cap of 384.
- Preserve the Stage 13 held-out K-Means fan-out selections for SIFT1M. For
  GloVe, retune fan-out on each Stage 15 single-graph collection at fixed
  `efSearch=320`, then validate the minimum candidate on a disjoint 9,000-query
  holdout. Increase fan-out, never ef, if the held-out recall misses 0.90.
- Set optimizer `max_segment_size=2,000,000 KB`, larger than the complete
  vector storage of either dataset.
- Require exactly two total segments per logical shard after optimization:
  one non-empty indexed HNSW segment and one empty appendable segment.
- Reject a point before QPS measurement if the segment-count gate is not met.

## Interpretation boundary

This control removes a graph-count discontinuity; it does not force QPS to fit
a logarithmic curve.  For Random broadcast, every query still visits all M
workers, so even if local graph work follows `log(N/M)`, the ideal speedup is
only approximately `log(N) / log(N/M)`.  For K-Means, any stronger gain also
depends on fan-out, route balance, and scatter-gather overhead.

The bounded GloVe ef recalibration is part of the single-graph control, not an
unbounded attempt to rescue a curve. All GloVe physical points use the same
fixed `efSearch=320`; no QPS point may increase ef, and the measured fan-out is
accepted only after held-out validation.

Stage 14 evidence remains intact and is reclassified as production-default
multi-segment behavior rather than a one-HNSW-graph-per-shard complexity test.
