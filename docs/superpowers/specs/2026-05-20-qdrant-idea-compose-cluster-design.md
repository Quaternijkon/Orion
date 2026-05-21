# Qdrant Idea Compose Cluster Migration Design

## Goal

Move the existing Qdrant two-level idea experiment onto a single-machine multi-Docker Qdrant cluster so it matches the deployment shape used by the native naive baseline and is ready to translate to a real multi-machine cluster.

## Non-Goals

- Do not change the idea algorithm details:
  - KMeans shard assignment stays unchanged.
  - Upper-layer user-side HNSW stays unchanged.
  - Query routing from upper-layer sample hits stays unchanged.
  - Per-shard lower-layer `hnsw_ef = base_ef + factor * routed_count` stays unchanged.
  - Client-side candidate merge stays unchanged.
- Do not integrate the idea into Qdrant Rust internals in this step.
- Do not migrate every related custom-shard experiment script.

## Architecture

Add a dedicated single-host Docker Compose cluster for the idea experiment. It mirrors the native baseline cluster but uses independent ports and storage volumes so the two environments can run side-by-side:

- node 1 HTTP/gRPC: `localhost:6733` / `localhost:6734`
- node 2 HTTP/gRPC: `localhost:6743` / `localhost:6744`
- node 3 HTTP/gRPC: `localhost:6753` / `localhost:6754`
- all nodes use Qdrant cluster P2P port `6335` internally

The existing `tools/qdrant_two_level_routing_experiment.py` remains the algorithm entry point. It gains deployment-aware shard-key placement:

- `--shard-placement none`: preserve previous behavior, no placement in shard-key creation.
- `--shard-placement round_robin`: discover cluster peers and place custom shard keys round-robin across peers.
- `--shard-placement auto`: use round-robin when the target Qdrant endpoint reports more than one peer, otherwise preserve previous behavior.

The script records collection cluster distribution in the output directory after the build step so smoke and full runs can prove the custom shards are physically distributed across Docker nodes.

## Smoke Mode

Add `--train-limit` to load only the first N training vectors for quick cluster smoke runs. This is a deployment smoke mode, not a comparable recall benchmark, because the HDF5 neighbor ground truth still refers to the full dataset. Smoke runs should use a low or zero target recall and small query counts.

## Real Multi-Machine Readiness

The compose file should be easy to translate into per-node Docker commands by replacing Docker hostnames with real node IPs or DNS names and replacing local volumes with real host paths. The experiment script should already work unchanged against a real cluster by changing `--base-url` to any reachable node.

## Verification

- Unit tests cover peer discovery, placement planning, shard-key creation request bodies, and `--train-limit` loading.
- `docker compose ... config` validates the new compose file.
- A smoke run against the three-container idea cluster proves:
  - cluster forms successfully,
  - the two-level script can create a custom-sharded collection,
  - shard keys are placed across multiple peers,
  - the query path completes and writes result files.
