# QPS comparison: Orion vs Simple KMeans vs Naive Hash

Run ID: `native-20260818-scale4-32-r080-095-v1`

Protocol: GloVe-200-angular, cosine, top-k=10, 500 warmup queries, 3000 measured queries, batch=200, 3 repeats, Search API. Every selected point passed the recall window `[target, target+0.003]`. Four physical Qdrant peers were used; scales 8/16/32 are logical-shard simulations on those four peers.

Important boundary: HashAll and Simple KMeans use exact 8 and 16 logical shards. Orion fission produced real 7/10-shard boundaries for simulated scale 8 and real 15/17-shard boundaries for simulated scale 16. These are reported separately and are not labeled exact 8- or 16-worker Orion results.

## Simulated scale 4

| Target | Method | Real shards | Recall | QPS | QPS stdev | Parameters |
|---:|---|---:|---:|---:|---:|---|
| 0.80 | Naive Hash (HashAll) | 4 | 0.80083 | 2286.03 | 45.99 | `hnsw_ef=52` |
| 0.80 | Simple KMeans | 4 | 0.80030 | 2159.90 | 27.72 | `nprobe=3,hnsw_ef=84` |
| 0.80 | Orion | 4 | 0.80057 | 2262.88 | 9.83 | `upper_k=8,base=50,factor=15` |
| 0.85 | Naive Hash (HashAll) | 4 | 0.85103 | 2151.00 | 38.80 | `hnsw_ef=90` |
| 0.85 | Simple KMeans | 4 | 0.85040 | 2012.31 | 15.56 | `nprobe=3,hnsw_ef=156` |
| 0.85 | Orion | 4 | 0.85277 | 2105.32 | 7.34 | `upper_k=16,base=48,factor=14` |
| 0.90 | Naive Hash (HashAll) | 4 | 0.90180 | 1834.84 | 14.23 | `hnsw_ef=184` |
| 0.90 | Simple KMeans | 4 | 0.90010 | 1589.35 | 3.21 | `nprobe=4,hnsw_ef=272` |
| 0.90 | Orion | 4 | 0.90180 | 1778.02 | 4.11 | `upper_k=38,base=48,factor=14` |
| 0.95 | Naive Hash (HashAll) | 4 | 0.95050 | 1328.10 | 10.57 | `hnsw_ef=488` |
| 0.95 | Simple KMeans | 4 | 0.95100 | 1071.83 | 8.42 | `nprobe=4,hnsw_ef=760` |
| 0.95 | Orion | 4 | 0.95060 | 1164.58 | 7.79 | `upper_k=104,base=48,factor=16` |

## Simulated scale 8

| Target | Method | Real shards | Recall | QPS | QPS stdev | Parameters |
|---:|---|---:|---:|---:|---:|---|
| 0.80 | Naive Hash (HashAll) | 8 | 0.80157 | 2200.62 | 15.93 | `hnsw_ef=38` |
| 0.80 | Simple KMeans | 8 | 0.80130 | 2147.61 | 34.13 | `nprobe=4,hnsw_ef=80` |
| 0.80 | Orion | 7 | 0.80190 | 2335.56 | 8.45 | `upper_k=10,base=50,factor=15` |
| 0.80 | Orion | 10 | 0.80020 | 2239.94 | 1.69 | `upper_k=10,base=50,factor=21` |
| 0.85 | Naive Hash (HashAll) | 8 | 0.85097 | 2068.97 | 10.61 | `hnsw_ef=62` |
| 0.85 | Simple KMeans | 8 | 0.85270 | 1802.48 | 13.98 | `nprobe=5,hnsw_ef=120` |
| 0.85 | Orion | 7 | 0.85167 | 2216.50 | 7.15 | `upper_k=18,base=50,factor=14` |
| 0.85 | Orion | 10 | 0.85000 | 2123.25 | 10.54 | `upper_k=18,base=50,factor=15` |
| 0.90 | Naive Hash (HashAll) | 8 | 0.90050 | 1778.78 | 13.30 | `hnsw_ef=114` |
| 0.90 | Simple KMeans | 8 | 0.90123 | 1374.64 | 12.00 | `nprobe=6,hnsw_ef=216` |
| 0.90 | Orion | 7 | 0.90193 | 1896.49 | 7.54 | `upper_k=40,base=50,factor=14` |
| 0.90 | Orion | 10 | 0.90207 | 1731.07 | 5.84 | `upper_k=40,base=50,factor=16` |
| 0.95 | Naive Hash (HashAll) | 8 | 0.95037 | 1327.32 | 11.84 | `hnsw_ef=270` |
| 0.95 | Simple KMeans | 8 | 0.95120 | 1007.77 | 7.93 | `nprobe=8,hnsw_ef=450` |
| 0.95 | Orion | 7 | 0.95080 | 1284.38 | 5.60 | `upper_k=106,base=50,factor=18` |
| 0.95 | Orion | 10 | 0.95033 | 1255.47 | 4.37 | `upper_k=106,base=50,factor=18` |

## Simulated scale 16

| Target | Method | Real shards | Recall | QPS | QPS stdev | Parameters |
|---:|---|---:|---:|---:|---:|---|
| 0.80 | Naive Hash (HashAll) | 16 | 0.80257 | 2083.62 | 21.67 | `hnsw_ef=26` |
| 0.80 | Simple KMeans | 16 | 0.80113 | 2030.74 | 18.58 | `nprobe=6,hnsw_ef=70` |
| 0.80 | Orion | 15 | 0.80257 | 2292.36 | 16.57 | `upper_k=11,base=50,factor=17` |
| 0.80 | Orion | 17 | 0.80207 | 2334.41 | 11.57 | `upper_k=12,base=50,factor=15` |
| 0.85 | Naive Hash (HashAll) | 16 | 0.85277 | 1916.62 | 29.85 | `hnsw_ef=41` |
| 0.85 | Simple KMeans | 16 | 0.85220 | 1700.06 | 8.48 | `nprobe=7,hnsw_ef=110` |
| 0.85 | Orion | 15 | 0.85283 | 2119.92 | 5.91 | `upper_k=20,base=50,factor=14` |
| 0.85 | Orion | 17 | 0.85007 | 2221.60 | 7.27 | `upper_k=20,base=51,factor=14` |
| 0.90 | Naive Hash (HashAll) | 16 | 0.90197 | 1666.60 | 8.41 | `hnsw_ef=72` |
| 0.90 | Simple KMeans | 16 | 0.90233 | 1369.73 | 7.99 | `nprobe=10,hnsw_ef=158` |
| 0.90 | Orion | 15 | 0.90133 | 1821.99 | 2.37 | `upper_k=42,base=50,factor=14` |
| 0.90 | Orion | 17 | 0.90123 | 1869.48 | 0.98 | `upper_k=43,base=50,factor=14` |
| 0.95 | Naive Hash (HashAll) | 16 | 0.95040 | 1332.64 | 14.20 | `hnsw_ef=152` |
| 0.95 | Simple KMeans | 16 | 0.95247 | 903.65 | 4.00 | `nprobe=16,hnsw_ef=285` |
| 0.95 | Orion | 15 | 0.95080 | 1269.94 | 2.84 | `upper_k=108,base=50,factor=18` |
| 0.95 | Orion | 17 | 0.95023 | 1265.43 | 5.07 | `upper_k=108,base=50,factor=18` |

## Simulated scale 32

| Target | Method | Real shards | Recall | QPS | QPS stdev | Parameters |
|---:|---|---:|---:|---:|---:|---|
| 0.80 | Naive Hash (HashAll) | 32 | 0.80180 | 1855.63 | 28.99 | `hnsw_ef=18` |
| 0.80 | Simple KMeans | 32 | 0.80107 | 2147.99 | 18.22 | `nprobe=8,hnsw_ef=58` |
| 0.80 | Orion | 32 | 0.80063 | 2332.88 | 3.26 | `upper_k=13,base=50,factor=17` |
| 0.85 | Naive Hash (HashAll) | 32 | 0.85267 | 1759.56 | 17.06 | `hnsw_ef=28` |
| 0.85 | Simple KMeans | 32 | 0.85280 | 1868.54 | 20.46 | `nprobe=10,hnsw_ef=91` |
| 0.85 | Orion | 32 | 0.85213 | 2183.91 | 8.77 | `upper_k=23,base=50,factor=14` |
| 0.90 | Naive Hash (HashAll) | 32 | 0.90070 | 1579.50 | 14.51 | `hnsw_ef=46` |
| 0.90 | Simple KMeans | 32 | 0.90083 | 1505.85 | 26.65 | `nprobe=16,hnsw_ef=118` |
| 0.90 | Orion | 32 | 0.90273 | 1870.69 | 26.23 | `upper_k=46,base=50,factor=14` |
| 0.95 | Naive Hash (HashAll) | 32 | 0.95023 | 1343.52 | 10.96 | `hnsw_ef=90` |
| 0.95 | Simple KMeans | 32 | 0.95247 | 970.77 | 10.21 | `nprobe=22,hnsw_ef=230` |
| 0.95 | Orion | 32 | 0.95050 | 1360.41 | 14.00 | `upper_k=112,base=50,factor=16` |

## Main observations

- Exact scale 4: Naive Hash has the highest QPS at all four recall levels; Orion remains ahead of Simple KMeans but trails HashAll, especially at 0.95 recall.
- Simulated scale 8: the Orion 7-shard boundary leads through 0.90 recall; HashAll leads at 0.95. The Orion 10-shard boundary leads HashAll at 0.80 and 0.85, but trails it at 0.90 and 0.95.
- Simulated scale 16: both Orion 15- and 17-shard boundaries lead the exact-16 HashAll and Simple KMeans controls through 0.90 recall. HashAll leads at 0.95.
- Exact scale 32: Orion has the highest QPS at every recall target. Relative to HashAll, Orion QPS is about +25.7%, +24.1%, +18.4%, and +1.3% at targets 0.80, 0.85, 0.90, and 0.95.
- Simple KMeans is generally the slowest control at medium/high recall; its gap widens as nprobe and lower-level HNSW effort increase.

## Evidence and cleanup

- Formal points audited: 56; audit issues: 0.
- All formal points include summary, run manifest, final metrics, stability runs, and per-query metrics; all 24 Orion points also include a production-router trace.
- Scale-8 shared evidence was checksum-synchronized into local `results/`, excluding rebuildable route-screen `trace-only` files.
- Qdrant collections were removed after measurement. Non-canonical runtime payload copies were removed; canonical payloads, routing artifacts, manifests, gates, benchmarks, and original results were preserved.

