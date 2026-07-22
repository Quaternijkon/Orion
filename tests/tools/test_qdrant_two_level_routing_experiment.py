from __future__ import annotations

import importlib.util
import sys
import threading
from types import SimpleNamespace
from pathlib import Path

import pytest


def load_module():
    script_path = Path(__file__).resolve().parents[2] / "tools/qdrant_two_level_routing_experiment.py"
    spec = importlib.util.spec_from_file_location("two_level_exp", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_compute_sample_sizes_uses_one_over_32_per_shard_with_floor_and_min_one():
    module = load_module()

    sizes = module.compute_sample_sizes([10, 31, 32, 33, 96], denominator=32)

    assert sizes == [1, 1, 1, 1, 3]


def test_cpp_baseline_kmeans_train_initializes_centroids_from_first_sample_rows():
    module = load_module()
    data = module.np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [10.0, 10.0],
        ],
        dtype=module.np.float32,
    )

    centroids = module.cpp_baseline_kmeans_train(
        data,
        indices=[2, 3, 1],
        k=2,
        max_iter=0,
    )

    assert centroids.tolist() == [[0.0, 1.0], [10.0, 10.0]]


@pytest.mark.parametrize("routing_mode", ["cpp_kmeans_baseline", "kmeans_simple_nprobe"])
def test_main_passes_kmeans_rand_seed_to_cpp_kmeans_assignment_build(
    monkeypatch,
    tmp_path,
    routing_mode,
):
    module = load_module()
    hdf5_path = tmp_path / "seed_roles.hdf5"
    with module.h5py.File(hdf5_path, "w") as handle:
        handle.create_dataset(
            "train",
            data=module.np.array(
                [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7], [-1.0, 0.0]],
                dtype=module.np.float32,
            ),
        )
        handle.create_dataset(
            "test",
            data=module.np.array([[1.0, 0.0]], dtype=module.np.float32),
        )
        handle.create_dataset(
            "neighbors",
            data=module.np.array([[0]], dtype=module.np.int32),
        )

    args = SimpleNamespace(
        routing_mode=routing_mode,
        vector_distance="cosine",
        recover_routing_from_collection=False,
        reuse_existing=False,
        direct_peer_local_premerge=False,
        search_dispatch_mode="controller",
        output_dir=str(tmp_path / "results"),
        cluster_topology=None,
        deployment_manifest=None,
        base_url="http://example.invalid",
        collection="seed-role-test",
        hdf5_path=str(hdf5_path),
        train_limit=None,
        tuning_query_count=1,
        eval_query_count=1,
        warmup_query_count=0,
        top_k=1,
        placement_peer_uri_contains=None,
        shard_placement="none",
        shard_placement_map=None,
        num_shards=2,
        cpp_kmeans_train_size=4,
        kmeans_iters=3,
        upper_sample_seed=100,
        kmeans_rand_seed=7,
    )

    class SeedCaptured(Exception):
        pass

    captured = {}

    def capture_build(_train, num_shards, train_size, kmeans_iters, seed):
        captured.update(
            num_shards=num_shards,
            train_size=train_size,
            kmeans_iters=kmeans_iters,
            seed=seed,
        )
        raise SeedCaptured

    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "validate_args", lambda _args: None)
    monkeypatch.setattr(module, "effective_upper_search_ef", lambda _args: 100)
    monkeypatch.setattr(module, "cluster_peer_ids", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(module, "build_cpp_kmeans_baseline_assignments", capture_build)

    with pytest.raises(SeedCaptured):
        module.main()

    assert captured == {
        "num_shards": 2,
        "train_size": 4,
        "kmeans_iters": 3,
        "seed": 7,
    }


def test_cpp_kmeans_baseline_keeps_upper_sample_seed_for_upper_point_sampling(
    monkeypatch,
    tmp_path,
):
    module = load_module()
    hdf5_path = tmp_path / "upper_seed_role.hdf5"
    with module.h5py.File(hdf5_path, "w") as handle:
        handle.create_dataset(
            "train",
            data=module.np.array(
                [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7], [-1.0, 0.0]],
                dtype=module.np.float32,
            ),
        )
        handle.create_dataset(
            "test",
            data=module.np.array([[1.0, 0.0]], dtype=module.np.float32),
        )
        handle.create_dataset(
            "neighbors",
            data=module.np.array([[0]], dtype=module.np.int32),
        )

    args = SimpleNamespace(
        routing_mode="cpp_kmeans_baseline",
        vector_distance="cosine",
        recover_routing_from_collection=False,
        reuse_existing=False,
        direct_peer_local_premerge=False,
        search_dispatch_mode="controller",
        output_dir=str(tmp_path / "results"),
        cluster_topology=None,
        deployment_manifest=None,
        base_url="http://example.invalid",
        collection="upper-seed-role-test",
        hdf5_path=str(hdf5_path),
        train_limit=None,
        tuning_query_count=1,
        eval_query_count=1,
        warmup_query_count=0,
        top_k=1,
        placement_peer_uri_contains=None,
        shard_placement="none",
        shard_placement_map=None,
        num_shards=2,
        cpp_kmeans_train_size=4,
        kmeans_iters=3,
        upper_sample_seed=100,
        kmeans_rand_seed=7,
        hnsw_m=32,
        ef_construct=100,
        upload_batch_size=4,
        sample_denominator=32,
    )

    class UpperSeedCaptured(Exception):
        pass

    captured = {}

    def fake_build(train, _num_shards, _train_size, _kmeans_iters, seed):
        captured["kmeans_seed"] = seed
        return module.np.zeros(len(train), dtype=module.np.int32), module.np.zeros(
            (2, train.shape[1]), dtype=module.np.float32
        )

    def capture_upper_sample(_train, _assignments, _num_shards, _denominator, seed):
        captured["upper_sample_seed"] = seed
        raise UpperSeedCaptured

    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "validate_args", lambda _args: None)
    monkeypatch.setattr(module, "effective_upper_search_ef", lambda _args: 100)
    monkeypatch.setattr(module, "cluster_peer_ids", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(module, "build_cpp_kmeans_baseline_assignments", fake_build)
    monkeypatch.setattr(module, "ensure_collection", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(module, "sample_cpp_kmeans_upper_points", capture_upper_sample)

    with pytest.raises(UpperSeedCaptured):
        module.main()

    assert captured == {"kmeans_seed": 7, "upper_sample_seed": 100}


def test_sample_cpp_kmeans_upper_points_uses_cpp_floor_without_min_one():
    module = load_module()
    train = module.np.arange(16, dtype=module.np.float32).reshape(8, 2)
    assignments = module.np.array([0, 0, 0, 1, 1, 1, 1, 1], dtype=module.np.int32)

    upper_vectors, upper_ids, label_to_shard, rows = module.sample_cpp_kmeans_upper_points(
        train,
        assignments,
        num_shards=2,
        denominator=4,
        seed=100,
    )

    assert rows == [
        {
            "shard_id": 0,
            "shard_key": "centroid_00",
            "points_count": 3,
            "sample_count": 0,
        },
        {
            "shard_id": 1,
            "shard_key": "centroid_01",
            "points_count": 5,
            "sample_count": 1,
        },
    ]
    assert upper_vectors.shape == (1, 2)
    assert upper_ids.shape == (1,)
    assert label_to_shard == {int(upper_ids[0]): "centroid_01"}
    assert 4 <= int(upper_ids[0]) <= 8


def test_shard_efs_from_upper_hits_skips_zero_hit_shards():
    module = load_module()

    shard_keys, ef_values = module.shard_efs_from_upper_hits(
        shard_hit_counts={"s0": 0, "s1": 1, "s2": 3, "s3": 2},
        base_ef=20,
        factor=4,
    )

    assert shard_keys == ["s2", "s3", "s1"]
    assert ef_values == [32, 28, 24]


def test_merge_topk_candidates_keeps_best_score_per_id():
    module = load_module()

    merged = module.merge_topk_candidates(
        [
            [(0.95, 10), (0.80, 20), (0.70, 30)],
            [(0.92, 20), (0.85, 40), (0.60, 50)],
            [(0.91, 10), (0.89, 60)],
        ],
        top_k=4,
    )

    assert merged == [10, 20, 60, 40]


def test_merge_topk_candidates_can_use_lower_scores_for_euclid():
    module = load_module()

    merged = module.merge_topk_candidates(
        [
            [(0.10, 10), (0.30, 20), (0.70, 30)],
            [(0.20, 20), (0.40, 40), (0.05, 50)],
            [(0.08, 10), (0.15, 60)],
        ],
        top_k=4,
        score_higher_is_better=False,
    )

    assert merged == [50, 10, 60, 20]


def test_per_query_recall_rows_record_hits_and_recall():
    module = load_module()

    rows = module.per_query_recall_rows(
        top_ids_by_query=[
            [10, 20, 30],
            [40, 50, 60],
        ],
        neighbors=[
            [20, 70, 80],
            [90, 91, 92],
        ],
        top_k=3,
        query_index_offset=100,
    )

    assert rows == [
        {
            "query_index": 100,
            "hits_at_k": 1,
            "recall_at_k": 1 / 3,
            "retrieved_ids": "10 20 30",
            "ground_truth_ids": "20 70 80",
        },
        {
            "query_index": 101,
            "hits_at_k": 0,
            "recall_at_k": 0.0,
            "retrieved_ids": "40 50 60",
            "ground_truth_ids": "90 91 92",
        },
    ]


def test_peer_local_premerge_preserves_global_topk_with_cross_peer_duplicates():
    module = load_module()

    shard_results_by_peer = {
        101: [
            [(0.99, 10), (0.80, 20)],
            [(0.97, 30), (0.96, 40)],
        ],
        202: [
            [(0.98, 10), (0.95, 50)],
            [(0.94, 60), (0.93, 70)],
        ],
    }

    baseline = module.merge_topk_candidates(
        [
            group
            for peer_groups in shard_results_by_peer.values()
            for group in peer_groups
        ],
        top_k=4,
    )
    premerged = module.peer_local_premerge_candidates(
        shard_results_by_peer,
        top_k=4,
    )

    assert [group for _peer_id, group in premerged] == [
        [(0.99, 10), (0.97, 30), (0.96, 40), (0.80, 20)],
        [(0.98, 10), (0.95, 50), (0.94, 60), (0.93, 70)],
    ]
    assert module.merge_topk_candidates(
        [group for _peer_id, group in premerged],
        top_k=4,
    ) == baseline


def test_placement_for_shard_key_modes_do_not_change_routing_algorithm_inputs():
    module = load_module()

    peers = [101, 202, 303]

    assert module.placement_for_shard_key(0, peers, "none") is None
    assert module.placement_for_shard_key(0, [101], "auto") is None
    assert module.placement_for_shard_key(0, peers, "auto") == [101]
    assert module.placement_for_shard_key(1, peers, "round_robin") == [202]
    assert module.placement_for_shard_key(5, peers, "round_robin") == [303]
    assert module.placement_for_shard_key(
        2,
        peers,
        "map",
        {"centroid_02": 1},
    ) == [202]
    assert module.placement_for_shard_key(
        3,
        peers,
        "map",
        {"centroid_03": 303},
    ) == [303]


def test_load_shard_placement_map_reads_named_simulation_map(tmp_path):
    module = load_module()
    path = tmp_path / "placement_simulation.json"
    path.write_text(
        module.json.dumps(
            {
                "placements": {
                    "method4_aware": {
                        "centroid_00": 2,
                        "centroid_01": 0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    placement = module.load_shard_placement_map(path, "method4_aware")

    assert placement == {"centroid_00": 2, "centroid_01": 0}


def test_parse_args_supports_shard_placement_map(monkeypatch):
    module = load_module()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qdrant_two_level_routing_experiment.py",
            "--shard-placement",
            "map",
            "--shard-placement-map",
            "placement_simulation.json",
            "--shard-placement-map-name",
            "method4_aware",
        ],
    )

    args = module.parse_args()

    assert args.shard_placement == "map"
    assert args.shard_placement_map == "placement_simulation.json"
    assert args.shard_placement_map_name == "method4_aware"


def test_parse_args_supports_vector_distance(monkeypatch):
    module = load_module()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qdrant_two_level_routing_experiment.py",
            "--vector-distance",
            "euclid",
        ],
    )

    args = module.parse_args()

    assert args.vector_distance == "euclid"


def test_vector_distance_config_maps_cosine_and_euclid():
    module = load_module()

    cosine = module.vector_distance_config("cosine")
    euclid = module.vector_distance_config("l2")

    assert cosine == {
        "name": "cosine",
        "hnsw_space": "cosine",
        "qdrant_distance": "Cosine",
        "normalize_vectors": True,
        "score_higher_is_better": True,
    }
    assert euclid == {
        "name": "euclid",
        "hnsw_space": "l2",
        "qdrant_distance": "Euclid",
        "normalize_vectors": False,
        "score_higher_is_better": False,
    }


def test_prepare_vectors_for_distance_only_normalizes_cosine():
    module = load_module()
    vectors = module.np.array([[3.0, 4.0], [0.0, 0.0]], dtype=module.np.float32)

    cosine_vectors = module.prepare_vectors_for_distance(vectors.copy(), "cosine")
    euclid_vectors = module.prepare_vectors_for_distance(vectors.copy(), "euclid")

    assert module.np.allclose(
        cosine_vectors,
        module.np.array([[0.6, 0.8], [0.0, 0.0]], dtype=module.np.float32),
    )
    assert euclid_vectors.tolist() == vectors.tolist()


def test_create_collection_uses_configured_qdrant_distance(monkeypatch):
    module = load_module()
    calls = []

    def fake_request_json(base_url, method, path, body=None, timeout=300.0):
        calls.append(
            {
                "base_url": base_url,
                "method": method,
                "path": path,
                "body": body,
                "timeout": timeout,
            }
        )
        return {"result": {}}

    monkeypatch.setattr(module, "request_json", fake_request_json)

    module.create_collection(
        "http://qdrant",
        "collection",
        dim=128,
        m=32,
        ef_construct=100,
        vector_distance="Euclid",
    )

    assert calls[0]["body"]["vectors"]["distance"] == "Euclid"


def test_create_collection_stores_canonical_routing_build_metadata(monkeypatch):
    module = load_module()
    calls = []
    routing_build = {
        "routing_mode": "kmeans_simple_nprobe",
        "kmeans_rand_seed": 7,
        "effective_num_shards": 46,
    }

    def fake_request_json(base_url, method, path, body=None, timeout=300.0):
        calls.append({"base_url": base_url, "method": method, "path": path, "body": body})
        return {"result": {}}

    monkeypatch.setattr(module, "request_json", fake_request_json)

    module.create_collection(
        "http://qdrant",
        "collection",
        dim=200,
        m=32,
        ef_construct=100,
        vector_distance="Cosine",
        routing_build_metadata=routing_build,
    )

    envelope = calls[0]["body"]["metadata"][module.ORION_COLLECTION_METADATA_KEY]
    assert envelope["schema_version"] == module.ORION_ROUTING_BUILD_METADATA_SCHEMA_VERSION
    assert envelope["routing_build"] == routing_build
    assert envelope["routing_build_sha256"] == module.canonical_json_sha256(routing_build)
    assert module.canonical_json_sha256({"b": 2, "a": 1}) == module.canonical_json_sha256(
        {"a": 1, "b": 2}
    )


def test_create_numeric_auto_shard_collection_supports_all_native_policies(monkeypatch):
    module = load_module()
    calls = []

    def fake_request_json(base_url, method, path, body=None, timeout=300.0):
        calls.append({"base_url": base_url, "method": method, "path": path, "body": body})
        return {"result": True}

    monkeypatch.setattr(module, "request_json", fake_request_json)
    module.create_numeric_auto_shard_collection(
        "http://controller:6333",
        "hash all",
        dim=200,
        num_shards=46,
        m=32,
        ef_construct=100,
    )
    module.create_numeric_auto_shard_collection(
        "http://controller:6333",
        "orion",
        dim=200,
        num_shards=46,
        m=32,
        ef_construct=100,
        auto_shard_policy={
            "type": "ORION",
            "generation": 7,
            "artifact_sha256": "A" * 64,
        },
    )
    module.create_numeric_auto_shard_collection(
        "http://controller:6333",
        "simple",
        dim=200,
        num_shards=46,
        m=32,
        ef_construct=100,
        auto_shard_policy={
            "type": "SIMPLE_KMEANS",
            "generation": 8,
            "artifact_sha256": "B" * 64,
        },
        metadata={"native_auto_shard_prepare": {"schema_version": 1}},
    )

    default_body = calls[0]["body"]
    assert calls[0]["path"] == "/collections/hash%20all"
    assert default_body["sharding_method"] == "auto"
    assert default_body["shard_number"] == 46
    assert default_body["replication_factor"] == 1
    assert "auto_shard_policy" not in default_body
    assert calls[1]["body"]["auto_shard_policy"] == {
        "type": "orion",
        "generation": 7,
        "artifact_sha256": "a" * 64,
    }
    assert calls[2]["body"]["auto_shard_policy"] == {
        "type": "simple_kmeans",
        "generation": 8,
        "artifact_sha256": "b" * 64,
    }
    assert calls[2]["body"]["metadata"] == {
        "native_auto_shard_prepare": {"schema_version": 1}
    }


@pytest.mark.parametrize(
    "policy",
    [
        {"type": "hash_all"},
        {"type": "orion", "generation": 0, "artifact_sha256": "a" * 64},
        {"type": "orion", "generation": 1, "artifact_sha256": "short"},
        {"type": "simple_kmeans", "generation": 0, "artifact_sha256": "a" * 64},
        {"type": "simple_kmeans", "generation": 1, "artifact_sha256": "short"},
    ],
)
def test_create_numeric_auto_shard_collection_rejects_invalid_orion_policy(policy):
    module = load_module()

    with pytest.raises(ValueError):
        module.create_numeric_auto_shard_collection(
            "http://controller:6333",
            "orion",
            dim=200,
            num_shards=46,
            m=32,
            ef_construct=100,
            auto_shard_policy=policy,
        )


def test_upsert_numeric_auto_points_uses_public_ids_without_shard_key(monkeypatch):
    module = load_module()
    calls = []

    def fake_request_json(base_url, method, path, body=None, timeout=300.0):
        calls.append(
            {
                "base_url": base_url,
                "method": method,
                "path": path,
                "body": body,
                "timeout": timeout,
            }
        )
        return {"result": {"status": "completed"}}

    monkeypatch.setattr(module, "request_json", fake_request_json)
    vectors = module.np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
        dtype=module.np.float32,
    )

    summary = module.upsert_numeric_auto_points(
        "http://controller:6333",
        "hash all",
        vectors,
        vector_name="embedding",
        batch_size=2,
        timeout=45.0,
    )

    assert summary == {
        "point_count": 3,
        "batch_count": 2,
        "first_id": 0,
        "last_id": 2,
        "vector_name": "embedding",
        "uses_shard_key": False,
    }
    assert [call["method"] for call in calls] == ["PUT", "PUT"]
    assert all(
        call["path"] == "/collections/hash%20all/points?wait=true" for call in calls
    )
    assert all("shard_key" not in call["body"] for call in calls)
    assert calls[0]["body"] == {
        "points": [
            {"id": 0, "vector": {"embedding": [1.0, 0.0]}},
            {"id": 1, "vector": {"embedding": [0.0, 1.0]}},
        ]
    }
    assert calls[1]["body"] == {
        "points": [{"id": 2, "vector": {"embedding": [0.5, 0.5]}}]
    }
    assert all(call["timeout"] == 45.0 for call in calls)


def test_effective_upper_search_ef_covers_widest_upper_k_candidate():
    module = load_module()

    args = SimpleNamespace(upper_search_ef=100, upper_k_candidates=[80, 160, 280])

    assert module.effective_upper_search_ef(args) == 280


def test_parse_args_supports_multi_assignment_controls(monkeypatch):
    module = load_module()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qdrant_two_level_routing_experiment.py",
            "--orion-multi-assign-min-max-vote",
            "3",
            "--orion-multi-assign-vote-delta",
            "1",
            "--orion-multi-assign-max-shards",
            "2",
            "--simple-kmeans-multi-assign-alpha",
            "1.12",
            "--simple-kmeans-multi-assign-chunk-size",
            "1024",
        ],
    )

    args = module.parse_args()

    assert args.orion_multi_assign_min_max_vote == 3
    assert args.orion_multi_assign_vote_delta == 1
    assert args.orion_multi_assign_max_shards == 2
    assert args.simple_kmeans_multi_assign_alpha == 1.12
    assert args.simple_kmeans_multi_assign_chunk_size == 1024


def test_parse_args_supports_claim_a_partition_family(monkeypatch):
    module = load_module()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qdrant_two_level_routing_experiment.py",
            "--claim-a-partition-family",
            "random_balanced_46",
            "--claim-a-random-seed",
            "12345",
        ],
    )

    args = module.parse_args()

    assert args.claim_a_partition_family == "random_balanced_46"
    assert args.claim_a_random_seed == 12345


def test_weighted_random_l1_shards_is_reproducible_and_balanced():
    module = load_module()
    upper_indices = module.np.array([10, 20, 30, 40, 50], dtype=module.np.int64)
    weights = module.np.array([9, 1, 6, 4, 5], dtype=module.np.int64)

    first = module.weighted_random_l1_shards(
        num_points=64,
        upper_indices=upper_indices,
        up_tier_weights=weights,
        num_shards=3,
        seed=12345,
    )
    second = module.weighted_random_l1_shards(
        num_points=64,
        upper_indices=upper_indices,
        up_tier_weights=weights,
        num_shards=3,
        seed=12345,
    )

    assert first == second
    assigned = [first[int(point_id)] for point_id in upper_indices.tolist()]
    assert sorted(set(assigned)) == [0, 1, 2]
    shard_weights = module.np.zeros(3, dtype=module.np.int64)
    for point_id, weight in zip(upper_indices.tolist(), weights.tolist()):
        shard_weights[first[int(point_id)]] += int(weight)
    assert int(shard_weights.sum()) == int(weights.sum())
    assert int(shard_weights.max()) <= int(weights.sum())


def test_claim_a_load_recalibrated_partition_matches_topology_without_fission():
    module = load_module()
    train = module.np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
            [-1.0, 0.0],
            [-0.9, -0.1],
        ],
        dtype=module.np.float32,
    )
    upper_indices = module.np.array([0, 2, 4], dtype=module.np.int64)
    point_to_l1s = [[0, 2], [0, 2], [2, 0], [2, 4], [4, 2], [4, 0]]

    topology = module.build_claim_a_partition_routing_state(
        "kmeans_topology_46",
        train,
        upper_indices,
        point_to_l1s,
        num_shards=3,
        kmeans_iters=2,
        kmeans_seed=1,
        topology_iters=5,
        use_multi_assign=True,
        multi_assign_min_max_vote=2,
        multi_assign_vote_delta=0,
        multi_assign_max_shards=0,
        random_seed=12345,
    )
    load_recalibrated = module.build_claim_a_partition_routing_state(
        "kmeans_topology_load_recalibrated_46",
        train,
        upper_indices,
        point_to_l1s,
        num_shards=3,
        kmeans_iters=2,
        kmeans_seed=1,
        topology_iters=5,
        use_multi_assign=True,
        multi_assign_min_max_vote=2,
        multi_assign_vote_delta=0,
        multi_assign_max_shards=0,
        random_seed=12345,
    )

    assert load_recalibrated.point_to_shards == topology.point_to_shards
    assert load_recalibrated.l1_to_shard == topology.l1_to_shard
    assert load_recalibrated.num_shards == topology.num_shards
    assert load_recalibrated.claim_a_partition_note == (
        "load_recalibration_matches_kmeans_topology_without_fission"
    )


def test_validate_args_rejects_invalid_multi_assignment_controls():
    module = load_module()

    with pytest.raises(ValueError, match="simple-kmeans-multi-assign-alpha"):
        module.validate_args(
            SimpleNamespace(
                orion_multi_assign_min_max_vote=2,
                orion_multi_assign_vote_delta=0,
                orion_multi_assign_max_shards=0,
                simple_kmeans_multi_assign_alpha=0.99,
                simple_kmeans_multi_assign_chunk_size=1024,
            )
        )


def test_validate_args_rejects_negative_fixed_ef_shard_chunk_size():
    module = load_module()

    with pytest.raises(ValueError, match="fixed-ef-shard-chunk-size"):
        module.validate_args(
            SimpleNamespace(
                orion_multi_assign_min_max_vote=2,
                orion_multi_assign_vote_delta=0,
                orion_multi_assign_max_shards=0,
                simple_kmeans_multi_assign_alpha=1.0,
                simple_kmeans_multi_assign_chunk_size=1024,
                warmup_query_count=0,
                fixed_ef_shard_chunk_size=-1,
            )
        )


def test_shard_create_body_includes_placement_only_when_requested():
    module = load_module()

    assert module.shard_create_body("centroid_00", placement=None) == {
        "shard_key": "centroid_00",
        "shards_number": 1,
        "replication_factor": 1,
    }
    assert module.shard_create_body("centroid_01", placement=[202]) == {
        "shard_key": "centroid_01",
        "shards_number": 1,
        "replication_factor": 1,
        "placement": [202],
    }


def test_cluster_peer_ids_can_filter_controller_out_by_uri_substring(monkeypatch):
    module = load_module()

    def fake_request_json(_base_url, _method, _path):
        return {
            "result": {
                "peers": {
                    "101": {"uri": "http://qdrant_controller:6335/"},
                    "202": {"uri": "http://qdrant_shard_1:6335/"},
                    "303": {"uri": "http://qdrant_shard_2:6335/"},
                }
            }
        }

    monkeypatch.setattr(module, "request_json", fake_request_json)

    assert module.cluster_peer_ids("http://controller:6333") == [101, 202, 303]
    assert module.cluster_peer_ids("http://controller:6333", ["qdrant_shard_"]) == [202, 303]


def test_cluster_preflight_requires_exact_four_private_uris(monkeypatch):
    module = load_module()
    topology = module.load_cluster_topology(
        Path(__file__).resolve().parents[2] / "tools/distributed/cloudlab_orion_4node.json"
    )

    monkeypatch.setattr(
        module,
        "request_json",
        lambda *_args, **_kwargs: {
            "result": {
                "peer_id": 101,
                "peers": {
                    "101": {"uri": "http://10.10.1.1:6335/"},
                    "202": {"uri": "http://10.10.1.2:6335"},
                    "303": {"uri": "http://10.10.1.3:6335"},
                    "404": {"uri": "http://10.10.1.4:6335"},
                },
                "raft_info": {"pending_operations": 0},
                "consensus_thread_status": {
                    "consensus_thread_status": "working"
                },
                "message_send_failures": {},
            }
        },
    )

    result = module.validate_cluster_preflight("http://10.10.1.1:6333", topology)

    assert result["peer_count"] == 4
    assert result["controller_peer_id"] == 101
    assert result["worker_peer_ids"] == [202, 303, 404]
    assert result["pending_operations"] == 0
    assert result["consensus_thread_status"] == "working"
    assert result["message_send_failures"] == {}
    assert module.cluster_peer_ids(
        "http://10.10.1.1:6333", exact_uris=topology["worker_uris"]
    ) == [202, 303, 404]


def test_cluster_preflight_rejects_pending_or_failed_peer_transport(monkeypatch):
    module = load_module()
    topology = module.load_cluster_topology(
        Path(__file__).resolve().parents[2]
        / "tools/distributed/cloudlab_orion_4node.json"
    )
    monkeypatch.setattr(
        module,
        "request_json",
        lambda *_args, **_kwargs: {
            "result": {
                "peer_id": 101,
                "peers": {
                    "101": {"uri": "http://10.10.1.1:6335/"},
                    "202": {"uri": "http://10.10.1.2:6335"},
                    "303": {"uri": "http://10.10.1.3:6335"},
                    "404": {"uri": "http://10.10.1.4:6335"},
                },
                "raft_info": {"pending_operations": 2},
                "consensus_thread_status": {
                    "consensus_thread_status": "working"
                },
                "message_send_failures": {
                    "http://10.10.1.4:6335/": {"count": 1}
                },
            }
        },
    )

    with pytest.raises(RuntimeError, match="pending consensus operations"):
        module.validate_cluster_preflight("http://10.10.1.1:6333", topology)


def test_numeric_shard_placement_discovers_local_and_remote_rf1_replicas():
    module = load_module()
    cluster = {
        "peer_id": 101,
        "shard_count": 3,
        "local_shards": [{"shard_id": 0, "state": "Active"}],
        "remote_shards": [
            {"shard_id": 1, "peer_id": 202, "state": "Active"},
            {"shard_id": 2, "peer_id": 303, "state": "Active"},
        ],
        "shard_transfers": [],
    }

    assert module.numeric_shard_placement_from_cluster(
        cluster, expected_shard_count=3
    ) == {0: 101, 1: 202, 2: 303}
    assert module.round_robin_numeric_shard_targets(
        [0, 1, 2, 3], [202, 303, 404]
    ) == {0: 202, 1: 303, 2: 404, 3: 202}


def test_numeric_shard_placement_rejects_custom_keys_duplicate_replicas_and_transfers():
    module = load_module()
    base = {
        "peer_id": 101,
        "shard_count": 1,
        "local_shards": [{"shard_id": 0, "state": "Active"}],
        "remote_shards": [],
        "shard_transfers": [],
    }

    custom = module.json.loads(module.json.dumps(base))
    custom["local_shards"][0]["shard_key"] = "centroid_00"
    with pytest.raises(ValueError, match="custom shard key"):
        module.numeric_shard_placement_from_cluster(custom, expected_shard_count=1)

    duplicate = module.json.loads(module.json.dumps(base))
    duplicate["remote_shards"].append(
        {"shard_id": 0, "peer_id": 202, "state": "Active"}
    )
    with pytest.raises(RuntimeError, match="2 replicas"):
        module.numeric_shard_placement_from_cluster(duplicate, expected_shard_count=1)

    transferring = module.json.loads(module.json.dumps(base))
    transferring["shard_transfers"] = [{"shard_id": 0, "from": 101, "to": 202}]
    with pytest.raises(RuntimeError, match="transfers are active"):
        module.numeric_shard_placement_from_cluster(transferring, expected_shard_count=1)


def test_move_numeric_shards_round_robin_is_sequential_strict_and_idempotent(monkeypatch):
    module = load_module()
    owners = {0: 101, 1: 202, 2: 202, 3: 404}
    move_calls = []

    def collection_result():
        return {
            "config": {
                "params": {
                    "sharding_method": "auto",
                    "shard_number": 4,
                    "replication_factor": 1,
                }
            }
        }

    def collection_cluster_result():
        return {
            "peer_id": 101,
            "shard_count": 4,
            "local_shards": [
                {"shard_id": shard_id, "state": "Active"}
                for shard_id, peer_id in owners.items()
                if peer_id == 101
            ],
            "remote_shards": [
                {"shard_id": shard_id, "peer_id": peer_id, "state": "Active"}
                for shard_id, peer_id in owners.items()
                if peer_id != 101
            ],
            "shard_transfers": [],
        }

    def fake_request_json(_base_url, method, path, body=None, timeout=300.0):
        if method == "GET" and path == "/cluster":
            return {
                "result": {
                    "peer_id": 101,
                    "peers": {
                        "101": {"uri": "http://10.10.1.1:6335"},
                        "202": {"uri": "http://10.10.1.2:6335"},
                        "303": {"uri": "http://10.10.1.3:6335"},
                        "404": {"uri": "http://10.10.1.4:6335"},
                    },
                }
            }
        if method == "GET" and path == "/collections/native":
            return {"result": collection_result()}
        if method == "GET" and path == "/collections/native/cluster":
            return {"result": collection_cluster_result()}
        if method == "POST" and path == "/collections/native/cluster":
            operation = dict(body["move_shard"])
            assert owners[operation["shard_id"]] == operation["from_peer_id"]
            owners[operation["shard_id"]] = operation["to_peer_id"]
            move_calls.append(operation)
            return {"result": True}
        raise AssertionError((method, path, body, timeout))

    monkeypatch.setattr(module, "request_json", fake_request_json)
    result = module.move_numeric_shards_round_robin(
        "http://10.10.1.1:6333",
        "native",
        [202, 303, 404],
        expected_shard_count=4,
        timeout_sec=5.0,
        poll_interval_sec=0.0,
    )

    assert owners == {0: 202, 1: 303, 2: 404, 3: 202}
    assert result["valid"] is True
    assert result["shards_per_worker"] == {202: 2, 303: 1, 404: 1}
    assert [move["shard_id"] for move in move_calls] == [0, 1, 2, 3]
    assert all(move["method"] == "stream_records" for move in move_calls)

    move_calls.clear()
    repeated = module.move_numeric_shards_round_robin(
        "http://10.10.1.1:6333",
        "native",
        [202, 303, 404],
        expected_shard_count=4,
        timeout_sec=5.0,
        poll_interval_sec=0.0,
    )
    assert repeated["moves"] == []
    assert move_calls == []


def test_validate_numeric_shard_round_robin_rejects_controller_or_wrong_peer():
    module = load_module()
    info = {
        "config": {
            "params": {
                "sharding_method": "auto",
                "shard_number": 3,
                "replication_factor": 1,
            }
        }
    }
    cluster = {
        "peer_id": 101,
        "shard_count": 3,
        "local_shards": [{"shard_id": 0, "state": "Active"}],
        "remote_shards": [
            {"shard_id": 1, "peer_id": 303, "state": "Active"},
            {"shard_id": 2, "peer_id": 404, "state": "Active"},
        ],
        "shard_transfers": [],
    }

    with pytest.raises(RuntimeError, match="round-robin placement mismatch"):
        module.validate_numeric_shard_round_robin_placement(
            info, cluster, [202, 303, 404], 3
        )


def test_validate_numeric_shard_explicit_placement_requires_exact_balanced_workers():
    module = load_module()
    info = {
        "config": {
            "params": {
                "sharding_method": "auto",
                "shard_number": 3,
                "replication_factor": 1,
            }
        }
    }
    cluster = {
        "peer_id": 101,
        "shard_count": 3,
        "local_shards": [],
        "remote_shards": [
            {"shard_id": 0, "peer_id": 303, "state": "Active"},
            {"shard_id": 1, "peer_id": 202, "state": "Active"},
            {"shard_id": 2, "peer_id": 404, "state": "Active"},
        ],
        "shard_transfers": [],
    }
    expected = {0: 303, 1: 202, 2: 404}

    proof = module.validate_numeric_shard_explicit_placement(
        info, cluster, [202, 303, 404], 3, expected
    )

    assert proof["valid"] is True
    assert proof["placement_mode"] == "explicit"
    assert proof["placement"] == expected
    assert proof["expected_placement"] == expected
    assert proof["shards_per_worker"] == {202: 1, 303: 1, 404: 1}

    with pytest.raises(RuntimeError, match="explicit placement mismatch"):
        module.validate_numeric_shard_explicit_placement(
            info, cluster, [202, 303, 404], 3, {0: 202, 1: 303, 2: 404}
        )

    with pytest.raises(ValueError, match="not a worker peer"):
        module.validate_numeric_shard_explicit_placement(
            info, cluster, [202, 303, 404], 3, {0: 303, 1: 202, 2: 999}
        )


def test_runtime_health_audit_rejects_fd_and_peer_transport_failures():
    module = load_module()
    logs = {
        "controller": {
            "returncode": 0,
            "stdout": (
                "Too many open files (os error 24)\n"
                "Failed to send message to http://10.10.1.2:6335/\n"
            ),
            "stderr": "",
        }
    }
    cluster = {
        "raft_info": {"pending_operations": 0},
        "consensus_thread_status": {"consensus_thread_status": "working"},
        "message_send_failures": {},
    }
    collection = {
        "status": "green",
        "optimizer_status": "ok",
        "update_queue": {"length": 0},
    }
    placement = {
        "local_shards": [],
        "remote_shards": [{"state": "Active"}],
        "shard_transfers": [],
    }

    audit = module.audit_runtime_health(logs, cluster, collection, placement)

    assert audit["valid"] is False
    assert set(audit["log_audit"]["error_matches"]) == {
        "too_many_open_files",
        "peer_transport_failure",
    }


def test_capture_controller_transport_resources_uses_manifest_container(monkeypatch):
    module = load_module()
    calls = []
    manifest = {
        "nodes": [
            {
                "role": "controller",
                "ssh_host": "localhost",
                "container_name": "orion-controller",
            },
            {"role": "qdrant_shard_1"},
            {"role": "qdrant_shard_2"},
            {"role": "qdrant_shard_3"},
        ]
    }

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="64 6\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    snapshot = module.capture_controller_transport_resources(manifest)

    assert snapshot["available"] is True
    assert snapshot["fd_count"] == 64
    assert snapshot["p2p_established"] == 6
    assert snapshot["expected_steady_state_p2p_connections"] == 6
    assert calls[0][0][:6] == [
        "sudo",
        "-n",
        "docker",
        "exec",
        "orion-controller",
        "sh",
    ]
    assert ":18BF$" in calls[0][0][-1]


def test_transport_resource_audit_rejects_connection_fanout_and_gates_runtime():
    module = load_module()
    start = {
        "available": True,
        "fd_count": 63,
        "p2p_established": 6,
        "expected_steady_state_p2p_connections": 6,
    }
    end = {
        "available": True,
        "fd_count": 274,
        "p2p_established": 217,
        "expected_steady_state_p2p_connections": 6,
    }
    transport = module.audit_transport_resources(start, end)

    assert transport["valid"] is False
    assert transport["clean_limit"] == 12
    assert transport["delta"] == {"fd_count": 211, "p2p_established": 211}

    runtime = module.audit_runtime_health(
        logs={"controller": {"returncode": 0, "stdout": "", "stderr": ""}},
        cluster_result={
            "raft_info": {"pending_operations": 0},
            "consensus_thread_status": {"consensus_thread_status": "working"},
            "message_send_failures": {},
        },
        collection={
            "status": "green",
            "optimizer_status": "ok",
            "update_queue": {"length": 0},
        },
        collection_cluster={
            "local_shards": [],
            "remote_shards": [{"state": "Active"}],
            "shard_transfers": [],
        },
        transport_resources=transport,
    )

    assert runtime["valid"] is False
    assert runtime["checks"]["transport_resources_clean"] is False


def test_reuse_validation_rejects_schema_counts_and_controller_placement():
    module = load_module()
    info = {
        "points_count": 99,
        "config": {
            "params": {
                "vectors": {"size": 128, "distance": "Euclid"},
                "replication_factor": 2,
            },
            "hnsw_config": {"m": 16, "ef_construct": 64},
        },
    }
    cluster = {
        "peer_id": 101,
        "shard_count": 2,
        "local_shards": [
            {"shard_key": "centroid_00", "peer_id": 101, "state": "Active"}
        ],
        "remote_shards": [
            {"shard_key": "centroid_01", "peer_id": 202, "state": "Dead"}
        ],
    }

    mismatches = module.collection_reuse_mismatches(
        info,
        cluster,
        expected_dimension=200,
        expected_distance="Cosine",
        expected_hnsw_m=32,
        expected_ef_construct=100,
        expected_points_count=100,
        expected_shard_count=3,
        expected_replication_factor=1,
        allowed_peer_ids=[202, 303, 404],
    )

    assert any("dimension" in item for item in mismatches)
    assert any("distance" in item for item in mismatches)
    assert any("hnsw.m" in item for item in mismatches)
    assert any("points_count" in item for item in mismatches)
    assert any("shard_count" in item for item in mismatches)
    assert any("controller owns" in item for item in mismatches)
    assert any("inactive" in item for item in mismatches)


def test_reuse_validation_accepts_balanced_remote_active_placement():
    module = load_module()
    info = {
        "points_count": 46,
        "config": {
            "params": {
                "vectors": {"size": 200, "distance": "Cosine"},
                "replication_factor": 1,
            },
            "hnsw_config": {"m": 32, "ef_construct": 100},
        },
    }
    remote = [
        {
            "shard_key": f"centroid_{index:02d}",
            "peer_id": [202, 303, 404][index % 3],
            "state": "Active",
        }
        for index in range(46)
    ]
    cluster = {"peer_id": 101, "shard_count": 46, "local_shards": [], "remote_shards": remote}

    assert module.collection_reuse_mismatches(
        info,
        cluster,
        expected_dimension=200,
        expected_distance="Cosine",
        expected_hnsw_m=32,
        expected_ef_construct=100,
        expected_points_count=46,
        expected_shard_count=46,
        expected_replication_factor=1,
        allowed_peer_ids=[202, 303, 404],
    ) == []


def test_reuse_validation_rejects_routing_build_seed_mismatch():
    module = load_module()
    args = SimpleNamespace(
        routing_mode="kmeans_simple_nprobe",
        num_shards=46,
        hnsw_m=32,
        ef_construct=100,
        upper_sample_seed=100,
        kmeans_rand_seed=7,
        kmeans_iters=10,
        cpp_kmeans_train_size=10000,
        sample_denominator=32,
        k_overlap=10,
        topology_iters=50,
        disable_multi_assign=False,
        simple_kmeans_multi_assign_alpha=1.0,
        disable_fission=False,
        claim_a_partition_family="none",
        claim_a_random_seed=12345,
        seed=42,
        sample_size=50000,
    )
    expected = module.build_routing_build_metadata(
        args,
        train_count=100,
        effective_num_shards=46,
        vector_distance="Cosine",
    )
    actual = module.json.loads(module.json.dumps(expected))
    actual["kmeans_rand_seed"] = 1
    info = {
        "points_count": 100,
        "config": {
            "params": {
                "vectors": {"size": 200, "distance": "Cosine"},
                "replication_factor": 1,
            },
            "hnsw_config": {"m": 32, "ef_construct": 100},
            "metadata": module.collection_metadata_for_routing_build(actual),
        },
    }
    remote = [
        {
            "shard_key": f"centroid_{index:02d}",
            "peer_id": [202, 303, 404][index % 3],
            "state": "Active",
        }
        for index in range(46)
    ]
    cluster = {"peer_id": 101, "shard_count": 46, "local_shards": [], "remote_shards": remote}

    mismatches = module.collection_reuse_mismatches(
        info,
        cluster,
        expected_dimension=200,
        expected_distance="Cosine",
        expected_hnsw_m=32,
        expected_ef_construct=100,
        expected_points_count=100,
        expected_shard_count=46,
        expected_replication_factor=1,
        allowed_peer_ids=[202, 303, 404],
        expected_routing_build_metadata=expected,
    )

    assert any("routing build metadata" in item for item in mismatches)
    validation = module.collection_routing_build_metadata_validation(info, expected)
    assert validation["status"] == "mismatch"
    assert validation["verified"] is False


def test_missing_routing_build_metadata_is_backward_compatible_but_unverified():
    module = load_module()
    expected = {"routing_mode": "naive_hash_all_shards", "kmeans_rand_seed": 1}
    info = {
        "points_count": 3,
        "config": {
            "params": {
                "vectors": {"size": 200, "distance": "Cosine"},
                "replication_factor": 1,
            },
            "hnsw_config": {"m": 32, "ef_construct": 100},
        },
    }
    cluster = {
        "peer_id": 101,
        "shard_count": 3,
        "local_shards": [],
        "remote_shards": [
            {"shard_key": "centroid_00", "peer_id": 202, "state": "Active"},
            {"shard_key": "centroid_01", "peer_id": 303, "state": "Active"},
            {"shard_key": "centroid_02", "peer_id": 404, "state": "Active"},
        ],
    }

    validation = module.collection_routing_build_metadata_validation(info, expected)
    assert validation["status"] == "missing_unverified"
    assert validation["verified"] is False
    assert validation["expected_fingerprint"] == module.canonical_json_sha256(expected)
    assert module.collection_reuse_mismatches(
        info,
        cluster,
        expected_dimension=200,
        expected_distance="Cosine",
        expected_hnsw_m=32,
        expected_ef_construct=100,
        expected_points_count=3,
        expected_shard_count=3,
        expected_replication_factor=1,
        allowed_peer_ids=[202, 303, 404],
        expected_routing_build_metadata=expected,
    ) == []


def test_parse_peer_http_urls_requires_peer_id_url_pairs():
    module = load_module()

    assert module.parse_peer_http_urls(["101=http://localhost:6843", "202=http://localhost:6853"]) == {
        101: "http://localhost:6843",
        202: "http://localhost:6853",
    }

    try:
        module.parse_peer_http_urls(["not-a-pair"])
    except ValueError as exc:
        assert "expected PEER_ID=URL" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_collection_shard_key_to_peer_uses_local_and_remote_shards(monkeypatch):
    module = load_module()

    def fake_collection_cluster_info(_base_url, _collection):
        return {
            "local_shards": [{"shard_key": "centroid_00", "peer_id": 101}],
            "remote_shards": [{"shard_key": "centroid_01", "peer_id": 202}],
        }

    monkeypatch.setattr(module, "collection_cluster_info", fake_collection_cluster_info)

    assert module.collection_shard_key_to_peer("http://controller:6333", "collection") == {
        "centroid_00": 101,
        "centroid_01": 202,
    }


def test_slice_train_rows_supports_smoke_prefix_without_changing_default():
    module = load_module()

    rows = [[0], [1], [2], [3]]

    assert module.slice_train_rows(rows, None) == rows
    assert module.slice_train_rows(rows, 2) == [[0], [1]]


def test_slice_train_rows_rejects_non_positive_limits():
    module = load_module()

    try:
        module.slice_train_rows([[0]], 0)
    except ValueError as exc:
        assert "train_limit must be positive" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_parse_args_supports_end_to_end_concurrency_mode(monkeypatch):
    module = load_module()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qdrant_two_level_routing_experiment.py",
            "--concurrency-evaluation-mode",
            "end_to_end",
            "--search-dispatch-mode",
            "direct_peer",
            "--direct-peer-http-urls",
            "101=http://localhost:6843",
            "--direct-peer-local-premerge",
        ],
    )

    args = module.parse_args()

    assert args.concurrency_evaluation_mode == "end_to_end"
    assert args.search_dispatch_mode == "direct_peer"
    assert args.direct_peer_http_urls == ["101=http://localhost:6843"]
    assert args.direct_peer_local_premerge is True


def test_parse_args_supports_materialized_routed_planning(monkeypatch):
    module = load_module()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qdrant_two_level_routing_experiment.py",
            "--routed-planning-mode",
            "materialized",
        ],
    )

    args = module.parse_args()

    assert args.routed_planning_mode == "materialized"


def test_parse_args_supports_compact_materialized_routed_planning(monkeypatch):
    module = load_module()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qdrant_two_level_routing_experiment.py",
            "--routed-planning-mode",
            "compact_materialized",
        ],
    )

    args = module.parse_args()

    assert args.routed_planning_mode == "compact_materialized"


def test_parse_args_supports_pipelined_routed_planning(monkeypatch):
    module = load_module()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qdrant_two_level_routing_experiment.py",
            "--routed-planning-mode",
            "pipelined",
        ],
    )

    args = module.parse_args()

    assert args.routed_planning_mode == "pipelined"


def test_parse_args_supports_shard_major_lower_execution(monkeypatch):
    module = load_module()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qdrant_two_level_routing_experiment.py",
            "--lower-execution-order",
            "shard_major",
        ],
    )

    args = module.parse_args()

    assert args.lower_execution_order == "shard_major"


def test_parse_args_fixed_ef_shard_chunking_is_explicit_opt_in(monkeypatch):
    module = load_module()

    monkeypatch.setattr(sys, "argv", ["qdrant_two_level_routing_experiment.py"])
    assert module.parse_args().fixed_ef_shard_chunk_size == 0

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qdrant_two_level_routing_experiment.py",
            "--fixed-ef-shard-chunk-size",
            "32",
        ],
    )
    assert module.parse_args().fixed_ef_shard_chunk_size == 32


def test_parse_args_supports_placement_simulation(monkeypatch):
    module = load_module()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qdrant_two_level_routing_experiment.py",
            "--placement-simulation",
            "--placement-simulation-peer-count",
            "3",
        ],
    )

    args = module.parse_args()

    assert args.placement_simulation is True
    assert args.placement_simulation_peer_count == 3


def test_materialized_routed_evaluation_builds_all_plans_inside_timed_path(monkeypatch):
    module = load_module()

    calls = {"upper": 0, "execute": []}

    def fake_compute_upper_labels(_upper_index, queries, upper_k):
        calls["upper"] += 1
        assert len(queries) == 3
        assert upper_k == 2
        return module.np.array([[1, 2], [2, 3], [1, 3]], dtype=module.np.int64)

    def fake_execute_query_plans_once(
        _base_url,
        _collection,
        query_plans,
        neighbors,
        top_k,
        direct_peer_urls=None,
        shard_key_to_peer=None,
        lower_execution_order="query_major",
        direct_peer_local_premerge=False,
        score_higher_is_better=True,
        preencoded_search_stage_bodies=None,
    ):
        calls["execute"].append((query_plans, neighbors.tolist()))
        assert lower_execution_order == "query_major"
        assert direct_peer_local_premerge is False
        return {
            "hits": len(query_plans) * top_k,
            "query_count": len(query_plans),
            "visited_shards": sum(plan["visited_shards"] for plan in query_plans),
            "upper_hits": sum(plan["upper_hits"] for plan in query_plans),
            "assigned_ef_sum": sum(plan["assigned_ef_sum"] for plan in query_plans),
            "assigned_ef_count": sum(plan["assigned_ef_count"] for plan in query_plans),
            "search_batch_calls": 1,
            "search_request_count": sum(len(plan["searches"]) for plan in query_plans),
        }

    monkeypatch.setattr(module, "compute_upper_labels", fake_compute_upper_labels)
    monkeypatch.setattr(module, "execute_query_plans_once", fake_execute_query_plans_once)

    queries = module.np.array([[0.1], [0.2], [0.3]], dtype=module.np.float32)
    neighbors = module.np.array([[1], [2], [3]], dtype=module.np.int32)
    result = module.evaluate_config_materialized_routing(
        "http://qdrant",
        "collection",
        queries,
        neighbors,
        top_k=1,
        upper_index=object(),
        upper_k=2,
        base_ef=20,
        factor=4,
        batch_size=2,
        point_to_shards=[[], [0, 2], [1], [2]],
        num_shards=3,
        use_payload_source_id=False,
        routed_execution_mode="compact_multi_ep",
    )

    assert calls["upper"] == 1
    assert [len(batch[0]) for batch in calls["execute"]] == [2, 1]
    assert result["recall_at_k"] == 1.0
    assert result["avg_visited_shards"] == 7 / 3
    assert result["avg_upper_hits"] == 8 / 3
    assert result["search_batch_calls"] == 2
    assert result["avg_search_requests_per_query"] == 1.0


def test_pipelined_routed_evaluation_overlaps_next_plan_with_lower_search(monkeypatch):
    module = load_module()
    second_batch_planning_started = threading.Event()
    execute_calls = []

    def fake_compute_upper_labels(_upper_index, queries, upper_k):
        assert upper_k == 1
        if float(queries[0][0]) == pytest.approx(0.2):
            second_batch_planning_started.set()
        return module.np.ones((len(queries), 1), dtype=module.np.int64)

    def fake_execute_query_plans_once(
        _base_url,
        _collection,
        query_plans,
        neighbors,
        top_k,
        direct_peer_urls=None,
        shard_key_to_peer=None,
        lower_execution_order="query_major",
        direct_peer_local_premerge=False,
        score_higher_is_better=True,
        preencoded_search_stage_bodies=None,
    ):
        if not execute_calls:
            assert second_batch_planning_started.wait(timeout=1.0)
        assert preencoded_search_stage_bodies is not None
        assert len(preencoded_search_stage_bodies) == 1
        execute_calls.append((query_plans, neighbors.tolist()))
        return {
            "hits": len(query_plans) * top_k,
            "query_count": len(query_plans),
            "visited_shards": sum(plan["visited_shards"] for plan in query_plans),
            "upper_hits": sum(plan["upper_hits"] for plan in query_plans),
            "assigned_ef_sum": sum(plan["assigned_ef_sum"] for plan in query_plans),
            "assigned_ef_count": sum(plan["assigned_ef_count"] for plan in query_plans),
            "search_batch_calls": 1,
            "search_request_count": sum(len(plan["searches"]) for plan in query_plans),
        }

    monkeypatch.setattr(module, "compute_upper_labels", fake_compute_upper_labels)
    monkeypatch.setattr(module, "execute_query_plans_once", fake_execute_query_plans_once)

    result = module.evaluate_config_pipelined_routing(
        "http://qdrant",
        "collection",
        module.np.array([[0.1], [0.2]], dtype=module.np.float32),
        module.np.array([[1], [1]], dtype=module.np.int32),
        top_k=1,
        upper_index=object(),
        upper_k=1,
        base_ef=20,
        factor=4,
        batch_size=1,
        point_to_shards=[[], [0]],
        num_shards=1,
        use_payload_source_id=False,
        routed_execution_mode="compact_multi_ep",
    )

    assert len(execute_calls) == 2
    assert result["recall_at_k"] == 1.0
    assert result["avg_visited_shards"] == 1.0
    assert result["avg_search_requests_per_query"] == 1.0


def test_pipelined_and_materialized_build_identical_same_index_query_plans(monkeypatch):
    module = load_module()
    upper_labels_by_query = module.np.array(
        [
            [1, 2, 3, 7],
            [4, 5, 6, 1],
            [7, 3, 2, 5],
            [1, 4, 6, 7],
            [2, 3, 5, 7],
        ],
        dtype=module.np.int64,
    )

    class DeterministicUpperIndex:
        def get_current_count(self):
            return 8

        def knn_query(self, queries, k):
            query_ids = queries[:, 0].astype(module.np.int64)
            labels = upper_labels_by_query[query_ids, :k]
            distances = module.np.zeros(labels.shape, dtype=module.np.float32)
            return labels, distances

    queries = module.np.array(
        [[0.0, 0.1], [1.0, 0.2], [2.0, 0.3], [3.0, 0.4], [4.0, 0.5]],
        dtype=module.np.float32,
    )
    neighbors = module.np.ones((len(queries), 2), dtype=module.np.int64)
    point_to_shards = [
        [],
        [0, 2],
        [1],
        [2, 3],
        [0],
        [1, 3],
        [3],
        [0, 1, 2],
    ]
    shard_key_to_peer = {
        "centroid_00": 101,
        "centroid_01": 101,
        "centroid_02": 202,
        "centroid_03": 303,
    }
    captured_plans = []

    def fake_execute_query_plans_once(
        _base_url,
        _collection,
        query_plans,
        _neighbors,
        top_k,
        **_kwargs,
    ):
        captured_plans.extend(query_plans)
        query_count = len(query_plans)
        request_count = sum(len(plan["searches"]) for plan in query_plans)
        return {
            "hits": query_count * top_k,
            "query_count": query_count,
            "visited_shards": sum(plan["visited_shards"] for plan in query_plans),
            "upper_hits": sum(plan["upper_hits"] for plan in query_plans),
            "assigned_ef_sum": sum(plan["assigned_ef_sum"] for plan in query_plans),
            "assigned_ef_count": sum(plan["assigned_ef_count"] for plan in query_plans),
            "search_batch_calls": 1,
            "search_request_count": request_count,
            "candidate_group_count": request_count,
            "returned_candidate_count": query_count * top_k,
            "batch_latency_ms": 0.0,
        }

    monkeypatch.setattr(module, "execute_query_plans_once", fake_execute_query_plans_once)

    common_kwargs = {
        "base_url": "http://qdrant",
        "collection": "collection",
        "queries": queries,
        "neighbors": neighbors,
        "top_k": 2,
        "upper_index": DeterministicUpperIndex(),
        "upper_k": 4,
        "base_ef": 20,
        "factor": 4,
        "batch_size": 2,
        "point_to_shards": point_to_shards,
        "num_shards": 4,
        "use_payload_source_id": True,
        "routed_execution_mode": "compact_multi_ep",
        "source_id_dedup_block_size": 1001,
        "shard_key_to_peer": shard_key_to_peer,
    }

    materialized_result = module.evaluate_config_materialized_routing(**common_kwargs)
    materialized_plans = list(captured_plans)
    captured_plans.clear()
    pipelined_result = module.evaluate_config_pipelined_routing(**common_kwargs)
    pipelined_plans = list(captured_plans)

    assert len(materialized_plans) == len(pipelined_plans) == len(queries)
    assert materialized_plans == pipelined_plans
    assert all(len(plan["searches"]) == 1 for plan in pipelined_plans)
    assert all(
        "hnsw_entry_points_by_shard" in plan["searches"][0]
        and "hnsw_ef_by_shard" in plan["searches"][0]
        for plan in pipelined_plans
    )

    exact_metric_keys = {
        "recall_at_k",
        "avg_visited_shards",
        "avg_upper_hits",
        "avg_assigned_ef_per_visited_shard",
        "search_batch_calls",
        "avg_search_requests_per_query",
        "avg_candidate_groups_per_query",
        "avg_returned_candidates_per_query",
        "avg_physical_peers_per_query",
    }
    assert {
        key: materialized_result[key] for key in exact_metric_keys
    } == {
        key: pipelined_result[key] for key in exact_metric_keys
    }


def test_global_upper_indices_uses_one_global_sample_before_partitioning():
    module = load_module()

    first = module.global_upper_indices(num_points=20, denominator=4, seed=100)
    second = module.global_upper_indices(num_points=20, denominator=4, seed=100)

    assert len(first) == 5
    assert sorted(first.tolist()) == sorted(set(first.tolist()))
    assert first.tolist() == second.tolist()


def test_assign_points_by_l1_vote_matches_original_multi_assign_rules():
    module = load_module()

    point_to_l1s = [
        [10, 20, 30, 40],
        [10, 20, 50, 60],
        [70, 80],
        [99],
    ]
    reference_l1_shard = [-1] * 100
    for label, shard in {
        10: 0,
        20: 1,
        30: 1,
        40: 0,
        50: 2,
        60: 2,
        70: 3,
        80: 4,
    }.items():
        reference_l1_shard[label] = shard

    primary, point_to_shards = module.assign_points_by_l1_vote(
        point_to_l1s,
        reference_l1_shard,
        num_shards=5,
        use_multi_assign=True,
    )

    assert primary.tolist() == [0, 2, 3, 3]
    assert point_to_shards == [[0, 1], [2], [3], [3]]


def test_assign_points_by_l1_vote_can_include_shards_within_vote_delta():
    module = load_module()

    point_to_l1s = [[10, 11, 12, 20, 21, 30]]
    reference_l1_shard = [-1] * 31
    for label, shard in {
        10: 0,
        11: 0,
        12: 0,
        20: 1,
        21: 1,
        30: 2,
    }.items():
        reference_l1_shard[label] = shard

    primary, point_to_shards = module.assign_points_by_l1_vote(
        point_to_l1s,
        reference_l1_shard,
        num_shards=4,
        use_multi_assign=True,
        multi_assign_vote_delta=1,
    )

    assert primary.tolist() == [0]
    assert point_to_shards == [[0, 1]]


def test_assign_points_by_l1_vote_keeps_single_assignment_when_max_vote_is_one():
    module = load_module()

    point_to_l1s = [[10, 20, 30]]
    reference_l1_shard = [-1] * 31
    reference_l1_shard[10] = 0
    reference_l1_shard[20] = 1
    reference_l1_shard[30] = 2

    _primary, point_to_shards = module.assign_points_by_l1_vote(
        point_to_l1s,
        reference_l1_shard,
        num_shards=4,
        use_multi_assign=True,
        multi_assign_vote_delta=1,
    )

    assert point_to_shards == [[0]]


def test_assign_points_by_l1_vote_can_cap_multi_assign_shards():
    module = load_module()

    point_to_l1s = [[10, 11, 20, 21, 30, 31]]
    reference_l1_shard = [-1] * 32
    for label, shard in {
        10: 0,
        11: 0,
        20: 1,
        21: 1,
        30: 2,
        31: 2,
    }.items():
        reference_l1_shard[label] = shard

    _primary, point_to_shards = module.assign_points_by_l1_vote(
        point_to_l1s,
        reference_l1_shard,
        num_shards=4,
        use_multi_assign=True,
        multi_assign_max_shards=2,
    )

    assert point_to_shards == [[0, 1]]


def test_point_indices_by_shard_preserves_multi_assignment_expansion():
    module = load_module()

    by_shard = module.point_indices_by_shard([[0, 2], [2], [1], [0, 1]], num_shards=3)

    assert [indices.tolist() for indices in by_shard] == [[0, 3], [2, 3], [0, 1]]
    assert module.total_assigned_points([[0, 2], [2], [1], [0, 1]]) == 6
    assert module.expansion_ratio_from_assigned_points(6, 4) == 1.5


def test_expansion_ratio_rejects_empty_logical_points():
    module = load_module()

    with pytest.raises(ValueError, match="logical_points"):
        module.expansion_ratio_from_assigned_points(6, 0)


def test_route_upper_labels_uses_all_shards_of_each_upper_candidate_as_eps():
    module = load_module()

    point_to_shards = [[], [0, 2], [], [2], [], []]

    routed = module.route_upper_labels_to_shard_eps([1, 3, 5], point_to_shards)

    assert routed == {0: [1], 2: [1, 3]}
    shard_keys, ef_values = module.shard_efs_from_routed_eps(
        routed,
        num_shards=4,
        base_ef=20,
        factor=4,
    )
    assert shard_keys == ["centroid_00", "centroid_02"]
    assert ef_values == [24, 28]


def test_hash_point_to_shards_assigns_each_point_to_one_modulo_shard():
    module = load_module()

    point_to_shards = module.hash_point_to_shards(num_points=8, num_shards=3)

    assert point_to_shards == [[0], [1], [2], [0], [1], [2], [0], [1]]


def test_write_orion_numeric_shard_import_bundle_preserves_external_ids_and_membership(
    tmp_path,
):
    module = load_module()
    train = module.np.array(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        dtype=module.np.float64,
    )
    point_to_shards = [[0, 2], [1], [2]]
    graphless_path = module.write_orion_graphless_artifact(
        train,
        module.np.array([0, 2], dtype=module.np.int64),
        point_to_shards,
        num_shards=3,
        output_path=tmp_path / "graphless.json",
        generation=7,
        vector_distance="cosine",
        upper_k=2,
        upper_ef_search=4,
        dynamic_ef_base=20,
        dynamic_ef_factor=4,
        vector_name="embedding",
    )
    artifact = module.json.loads(graphless_path.read_text(encoding="utf-8"))
    assert artifact["layout_sha256"] == module.orion_layout_sha256(point_to_shards, 3)
    assert artifact["logical_point_count"] == 3
    assert artifact["physical_point_count"] == 4
    assert artifact["upper_nodes"] == [
        {
            "label": 0,
            "vector": [1.0, 2.0],
            "shard_membership": [0, 2],
        },
        {
            "label": 2,
            "vector": [5.0, 6.0],
            "shard_membership": [2],
        },
    ]
    artifact["upper_graph"] = {"entry_point": 0, "max_level": 0, "nodes": []}
    production_artifact = tmp_path / "generation-7.json"
    production_artifact.write_text(
        module.json.dumps(artifact, separators=(",", ":")),
        encoding="utf-8",
    )

    manifest_path = module.write_orion_numeric_shard_import_bundle(
        train,
        point_to_shards,
        num_shards=3,
        output_dir=tmp_path,
        orion_artifact_path=production_artifact,
        vector_name="embedding",
        row_chunk_size=2,
    )

    manifest = module.json.loads(manifest_path.read_text(encoding="utf-8"))
    vectors_path = tmp_path / manifest["vectors_file"]
    assignments_path = tmp_path / manifest["assignments_file"]
    assert manifest == {
        "format_version": 1,
        "dimension": 2,
        "point_count": 3,
        "shard_count": 3,
        "total_point_copies": 4,
        "vector_name": "embedding",
        "orion_generation": 7,
        "orion_artifact_file": "generation-7.json",
        "orion_artifact_sha256": module.sha256_path(production_artifact),
        "vectors_file": "orion_numeric_import.f32le",
        "vectors_sha256": module.sha256_path(vectors_path),
        "assignments_file": "orion_numeric_import.assignments.jsonl",
        "assignments_sha256": module.sha256_path(assignments_path),
    }
    loaded_vectors = module.np.fromfile(vectors_path, dtype="<f4").reshape(3, 2)
    module.np.testing.assert_array_equal(loaded_vectors, train.astype(module.np.float32))

    records = [
        module.json.loads(line)
        for line in assignments_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records == [
        {"id": 0, "shards": [0, 2]},
        {"id": 1, "shards": [1]},
        {"id": 2, "shards": [2]},
    ]
    assert records[0]["id"] == 0
    assert records[0]["shards"] == [0, 2]
    assert all("source_id" not in record and "payload" not in record for record in records)


def test_write_simple_kmeans_graphless_artifact_and_generic_v2_bundle(tmp_path):
    module = load_module()
    train = module.np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]],
        dtype=module.np.float32,
    )
    centroids = module.np.array(
        [[0.9, 0.1], [0.0, 1.0]],
        dtype=module.np.float32,
    )
    point_to_shards = [[0], [1], [0]]
    production_artifact = module.write_simple_kmeans_graphless_artifact(
        train,
        centroids,
        point_to_shards,
        num_shards=2,
        output_path=tmp_path / "generation-5.json",
        generation=5,
        vector_distance="cosine",
        nprobe=1,
        lower_hnsw_ef=48,
        vector_name="embedding",
    )

    artifact = module.json.loads(production_artifact.read_text(encoding="utf-8"))
    assert artifact == {
        "format_version": 1,
        "generation": 5,
        "vector_schema": {
            "vector_name": "embedding",
            "dimension": 2,
            "distance": "Cosine",
            "datatype": "float32",
        },
        "shard_count": 2,
        "layout_sha256": module.orion_layout_sha256(point_to_shards, 2),
        "logical_point_count": 3,
        "physical_point_count": 3,
        "routing_distance": "squared_l2",
        "nprobe": 1,
        "lower_hnsw_ef": 48,
        "centroids": [
            {"shard_id": 0, "vector": pytest.approx([0.9, 0.1])},
            {"shard_id": 1, "vector": pytest.approx([0.0, 1.0])},
        ],
    }

    manifest_path = module.write_numeric_shard_import_bundle_v2(
        train,
        point_to_shards,
        num_shards=2,
        output_dir=tmp_path,
        routing_policy="simple_kmeans",
        routing_generation=5,
        routing_artifact_path=production_artifact,
        vector_name="embedding",
        prefix="simple_kmeans_numeric_import",
        row_chunk_size=2,
    )

    manifest = module.json.loads(manifest_path.read_text(encoding="utf-8"))
    vectors_path = tmp_path / manifest["vectors_file"]
    assignments_path = tmp_path / manifest["assignments_file"]
    assert manifest == {
        "format_version": 2,
        "routing_policy": "simple_kmeans",
        "routing_generation": 5,
        "routing_artifact_file": "generation-5.json",
        "routing_artifact_sha256": module.sha256_path(production_artifact),
        "dimension": 2,
        "point_count": 3,
        "shard_count": 2,
        "total_point_copies": 3,
        "vector_name": "embedding",
        "vectors_file": "simple_kmeans_numeric_import.f32le",
        "vectors_sha256": module.sha256_path(vectors_path),
        "assignments_file": "simple_kmeans_numeric_import.assignments.jsonl",
        "assignments_sha256": module.sha256_path(assignments_path),
    }
    assert [
        module.json.loads(line)
        for line in assignments_path.read_text(encoding="utf-8").splitlines()
    ] == [
        {"id": 0, "shards": [0]},
        {"id": 1, "shards": [1]},
        {"id": 2, "shards": [0]},
    ]
    module.np.testing.assert_array_equal(
        module.np.fromfile(vectors_path, dtype="<f4").reshape(3, 2),
        train,
    )


def test_simple_kmeans_graphless_artifact_rejects_multi_assignment(tmp_path):
    module = load_module()
    train = module.np.array([[1.0, 0.0], [0.0, 1.0]], dtype=module.np.float32)
    centroids = train.copy()

    with pytest.raises(ValueError, match="exactly one shard per point"):
        module.write_simple_kmeans_graphless_artifact(
            train,
            centroids,
            [[0, 1], [1]],
            num_shards=2,
            output_path=tmp_path / "graphless.json",
            generation=1,
            vector_distance="cosine",
            nprobe=1,
            lower_hnsw_ef=32,
        )


def test_generic_v2_bundle_rejects_policy_artifact_mismatch(tmp_path):
    module = load_module()
    train = module.np.array([[1.0, 0.0], [0.0, 1.0]], dtype=module.np.float32)
    artifact_path = module.write_simple_kmeans_graphless_artifact(
        train,
        train.copy(),
        [[0], [1]],
        num_shards=2,
        output_path=tmp_path / "generation-1.json",
        generation=1,
        vector_distance="cosine",
        nprobe=1,
        lower_hnsw_ef=32,
    )

    with pytest.raises(ValueError, match="Orion routing artifact must contain upper_graph"):
        module.write_numeric_shard_import_bundle_v2(
            train,
            [[0], [1]],
            num_shards=2,
            output_dir=tmp_path,
            routing_policy="orion",
            routing_generation=1,
            routing_artifact_path=artifact_path,
        )

    with pytest.raises(ValueError, match="routing_policy must be one of"):
        module.write_numeric_shard_import_bundle_v2(
            train,
            [[0], [1]],
            num_shards=2,
            output_dir=tmp_path,
            routing_policy="unknown_policy",
            routing_generation=1,
            routing_artifact_path=artifact_path,
        )


def test_generic_v2_bundle_rejects_multi_copy_simple_kmeans_layout(tmp_path):
    module = load_module()
    train = module.np.array([[1.0, 0.0], [0.0, 1.0]], dtype=module.np.float32)
    point_to_shards = [[0, 1], [1]]
    artifact = {
        "format_version": 1,
        "generation": 1,
        "vector_schema": {
            "vector_name": "",
            "dimension": 2,
            "distance": "Cosine",
            "datatype": "float32",
        },
        "shard_count": 2,
        "layout_sha256": module.orion_layout_sha256(point_to_shards, 2),
        "logical_point_count": 2,
        "physical_point_count": 3,
        "routing_distance": "squared_l2",
        "nprobe": 1,
        "lower_hnsw_ef": 32,
        "centroids": [
            {"shard_id": 0, "vector": [1.0, 0.0]},
            {"shard_id": 1, "vector": [0.0, 1.0]},
        ],
    }
    artifact_path = tmp_path / "generation-1.json"
    artifact_path.write_text(module.json.dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one point copy"):
        module.write_numeric_shard_import_bundle_v2(
            train,
            point_to_shards,
            num_shards=2,
            output_dir=tmp_path,
            routing_policy="simple_kmeans",
            routing_generation=1,
            routing_artifact_path=artifact_path,
        )


def test_all_shard_keys_and_ef_searches_every_shard_with_uniform_ef():
    module = load_module()

    shard_keys, ef_values = module.all_shard_keys_and_ef(num_shards=4, hnsw_ef=36)

    assert shard_keys == ["centroid_00", "centroid_01", "centroid_02", "centroid_03"]
    assert ef_values == [36, 36, 36, 36]


def test_all_shard_search_plan_uses_one_request_covering_every_shard():
    module = load_module()

    plan = module.all_shard_search_plan(
        query=[0.1, 0.2],
        top_k=10,
        num_shards=3,
        hnsw_ef=36,
        use_payload_source_id=True,
    )

    assert plan["visited_shards"] == 3
    assert plan["upper_hits"] == 0
    assert plan["assigned_ef_sum"] == 108
    assert plan["assigned_ef_count"] == 3
    assert plan["separate_http_search_stages"] is False
    assert plan["searches"] == [
        {
            "vector": [0.1, 0.2],
            "limit": 10,
            "params": {"hnsw_ef": 36},
            "with_payload": ["source_id"],
            "with_vector": False,
            "shard_key": ["centroid_00", "centroid_01", "centroid_02"],
            "hnsw_ef_by_shard": {
                "centroid_00": 36,
                "centroid_01": 36,
                "centroid_02": 36,
            },
        }
    ]


def test_fixed_ef_shard_chunks_partition_all_46_shards_without_overlap():
    module = load_module()
    shard_keys = [module.shard_key_for_id(shard_id) for shard_id in range(46)]

    chunks = module.fixed_ef_shard_key_chunks(shard_keys, chunk_size=32)

    assert [len(chunk) for chunk in chunks] == [32, 14]
    assert [shard_key for chunk in chunks for shard_key in chunk] == shard_keys
    assert set(chunks[0]).isdisjoint(chunks[1])
    assert set().union(*(set(chunk) for chunk in chunks)) == set(shard_keys)
    with pytest.raises(ValueError, match="unique shard keys"):
        module.fixed_ef_shard_key_chunks(["centroid_00", "centroid_00"], chunk_size=1)


def test_preencoded_search_batch_body_matches_normal_json_request(monkeypatch):
    module = load_module()
    searches = [
        {
            "vector": [0.1, 0.2],
            "limit": 2,
            "params": {"hnsw_ef": 36},
            "with_payload": ["source_id"],
            "with_vector": False,
            "shard_key": ["centroid_00", "centroid_01"],
            "hnsw_ef_by_shard": {"centroid_00": 36, "centroid_01": 36},
        }
    ]
    encoded = module.encode_search_batch_body(searches)
    captured = {}

    def fake_request_json_encoded(base_url, method, path, data, timeout=300.0):
        captured.update(
            base_url=base_url,
            method=method,
            path=path,
            data=data,
            timeout=timeout,
        )
        return {"result": [[{"id": 8, "score": 0.75}]]}

    monkeypatch.setattr(module, "request_json_encoded", fake_request_json_encoded)
    monkeypatch.setattr(
        module,
        "request_json",
        lambda *_args, **_kwargs: pytest.fail("pre-encoded path re-encoded the request"),
    )

    result = module.search_batch(
        "http://controller:6333",
        "collection",
        searches,
        encoded_body=encoded,
    )

    assert encoded == module.json.dumps({"searches": searches}).encode()
    assert captured["data"] is encoded
    assert captured["method"] == "POST"
    assert captured["path"].endswith("/collections/collection/points/search/batch")
    assert result == [[(0.75, 7)]]


@pytest.mark.parametrize("api", ["search", "query"])
def test_standard_dense_vector_batch_sends_no_client_routing_hints(monkeypatch, api):
    module = load_module()
    captured = {}

    def fake_request_json(base_url, method, path, body=None, timeout=300.0):
        captured.update(
            base_url=base_url,
            method=method,
            path=path,
            body=body,
            timeout=timeout,
        )
        points = [
            {"id": 0, "score": 0.99},
            {"id": 7, "score": 0.75},
        ]
        if api == "search":
            return {"result": [points, points]}
        return {"result": [{"points": points}, {"points": points}]}

    monkeypatch.setattr(module, "request_json", fake_request_json)
    rows = module.standard_dense_vector_batch(
        "http://controller:6333",
        "native collection",
        module.np.array([[0.1, 0.2], [0.3, 0.4]], dtype=module.np.float32),
        2,
        api=api,
        vector_name="embedding",
    )

    assert captured["path"] == (
        f"/collections/native%20collection/points/{api}/batch"
    )
    assert rows == [[(0.99, 0), (0.75, 7)], [(0.99, 0), (0.75, 7)]]
    forbidden = {
        "shard_key",
        "hnsw_entry_points",
        "hnsw_entry_points_by_shard",
        "hnsw_ef_by_shard",
        "source_id_dedup_block_size",
        "params",
    }
    for request in captured["body"]["searches"]:
        assert forbidden.isdisjoint(request)
        assert request["with_payload"] is False
        assert request["with_vector"] is False
        if api == "search":
            assert request["vector"]["name"] == "embedding"
            assert "query" not in request
        else:
            assert request["using"] == "embedding"
            assert "vector" not in request


def test_standard_dense_vector_batch_rejects_non_numeric_external_ids(monkeypatch):
    module = load_module()
    monkeypatch.setattr(
        module,
        "request_json",
        lambda *_args, **_kwargs: {
            "result": [[{"id": "uuid-id", "score": 0.9}]]
        },
    )

    with pytest.raises(RuntimeError, match="integer external point IDs"):
        module.standard_dense_vector_batch(
            "http://controller:6333",
            "native",
            module.np.array([[0.1, 0.2]], dtype=module.np.float32),
            1,
        )


@pytest.mark.parametrize("api", ["search", "query"])
def test_standard_dense_vector_batch_allows_only_one_public_fixed_hnsw_ef(monkeypatch, api):
    module = load_module()
    captured = {}

    def fake_request_json(_base_url, _method, _path, body=None, timeout=300.0):
        captured["body"] = body
        points = [{"id": 0, "score": 0.9}]
        return {
            "result": [points] if api == "search" else [{"points": points}]
        }

    monkeypatch.setattr(module, "request_json", fake_request_json)
    module.standard_dense_vector_batch(
        "http://controller:6333",
        "native",
        module.np.array([[0.1, 0.2]], dtype=module.np.float32),
        1,
        api=api,
        hnsw_ef=64,
    )

    request = captured["body"]["searches"][0]
    assert request["params"] == {"hnsw_ef": 64}
    assert "shard_key" not in request
    assert "hnsw_ef_by_shard" not in request
    assert "hnsw_entry_points" not in request
    assert "hnsw_entry_points_by_shard" not in request

    with pytest.raises(ValueError, match="hnsw_ef"):
        module.standard_dense_vector_request(
            [0.1, 0.2],
            1,
            api=api,
            hnsw_ef=0,
        )


def test_standard_dense_vector_evaluator_batches_and_computes_external_id_recall(monkeypatch):
    module = load_module()
    call_sizes = []

    def fake_standard_batch(
        _base_url,
        _collection,
        queries,
        top_k,
        *,
        api="search",
        vector_name="",
        hnsw_ef=None,
        timeout=600.0,
    ):
        call_sizes.append(len(queries))
        assert top_k == 2
        assert api == "query"
        assert vector_name == ""
        assert hnsw_ef is None
        return [
            [(1.0, int(row[0])), (0.5, 8)]
            for row in queries
        ]

    monkeypatch.setattr(module, "standard_dense_vector_batch", fake_standard_batch)
    result = module.evaluate_standard_dense_vector_batches(
        "http://controller:6333",
        "native",
        module.np.array([[0.0], [1.0], [2.0]], dtype=module.np.float32),
        module.np.array([[0, 9], [1, 9], [5, 9]], dtype=module.np.int64),
        top_k=2,
        batch_size=2,
        api="query",
        include_per_query_metrics=True,
    )

    assert call_sizes == [2, 1]
    assert result["api"] == "query"
    assert result["query_count"] == 3
    assert result["hits"] == 2
    assert result["recall_at_k"] == pytest.approx(2 / 6)
    assert result["search_batch_calls"] == 2
    assert result["search_request_count"] == 3
    assert result["avg_search_requests_per_query"] == 1.0
    assert len(result["per_query_rows"]) == 3


def test_all_shard_search_plan_chunks_transport_but_keeps_all_shards_and_fixed_ef():
    module = load_module()

    plan = module.all_shard_search_plan(
        query=[0.1, 0.2],
        top_k=10,
        num_shards=46,
        hnsw_ef=72,
        use_payload_source_id=True,
        source_id_dedup_block_size=1001,
        fixed_ef_shard_chunk_size=32,
    )

    searches = plan["searches"]
    assert [len(search["shard_key"]) for search in searches] == [32, 14]
    flattened = [shard_key for search in searches for shard_key in search["shard_key"]]
    assert flattened == [module.shard_key_for_id(shard_id) for shard_id in range(46)]
    assert len(flattened) == len(set(flattened)) == 46
    assert all(search["limit"] == 10 for search in searches)
    assert all(search["params"] == {"hnsw_ef": 72} for search in searches)
    assert all(
        set(search["hnsw_ef_by_shard"]) == set(search["shard_key"])
        and set(search["hnsw_ef_by_shard"].values()) == {72}
        for search in searches
    )
    assert all(search["source_id_dedup_block_size"] == 1001 for search in searches)
    assert plan["visited_shards"] == 46
    assert plan["assigned_ef_sum"] == 46 * 72
    assert plan["assigned_ef_count"] == 46
    assert plan["separate_http_search_stages"] is True


def test_chunked_all_shards_use_separate_http_batches_then_global_topk_dedup(monkeypatch):
    module = load_module()
    calls = []

    def fake_search_batch(_base_url, _collection, searches):
        calls.append([search["shard_key"] for search in searches])
        if len(calls) == 1:
            return [[(0.90, 7), (0.80, 3)]]
        return [[(0.95, 7), (0.85, 9)]]

    monkeypatch.setattr(module, "search_batch", fake_search_batch)
    plan = module.all_shard_search_plan(
        query=[0.1, 0.2],
        top_k=2,
        num_shards=4,
        hnsw_ef=36,
        use_payload_source_id=True,
        source_id_dedup_block_size=1001,
        fixed_ef_shard_chunk_size=2,
    )

    result = module.execute_query_plans_once(
        base_url="http://controller:6333",
        collection="collection",
        query_plans=[plan],
        neighbors=[[7, 9]],
        top_k=2,
        include_per_query_metrics=True,
    )

    assert calls == [
        [["centroid_00", "centroid_01"]],
        [["centroid_02", "centroid_03"]],
    ]
    assert result["search_batch_calls"] == 2
    assert result["search_request_count"] == 2
    assert result["candidate_group_count"] == 2
    assert result["hits"] == 2
    assert result["per_query_rows"][0]["retrieved_ids"] == "7 9"


def test_chunked_all_shards_direct_peer_stages_are_sequential_and_maps_are_peer_local(
    monkeypatch,
):
    module = load_module()
    calls = []
    completed_stage_one_peers = set()

    def fake_search_batch(base_url, _collection, searches):
        assert len(searches) == 1
        search = searches[0]
        shard_keys = search["shard_key"]
        stage = 1 if shard_keys[0] in {"centroid_00", "centroid_01"} else 2
        peer = 101 if base_url.endswith("6843") else 202
        if stage == 2:
            assert completed_stage_one_peers == {101, 202}
        calls.append((stage, peer, search))
        if stage == 1:
            completed_stage_one_peers.add(peer)
        if shard_keys[0] in {"centroid_00", "centroid_02"}:
            return [[(0.90 if stage == 1 else 0.95, 7)]]
        return [[(0.80 if stage == 1 else 0.85, 9)]]

    monkeypatch.setattr(module, "search_batch", fake_search_batch)
    plan = module.all_shard_search_plan(
        query=[0.1, 0.2],
        top_k=2,
        num_shards=4,
        hnsw_ef=36,
        use_payload_source_id=True,
        source_id_dedup_block_size=1001,
        fixed_ef_shard_chunk_size=2,
    )

    result = module.execute_query_plans_once(
        base_url="http://controller:6333",
        collection="collection",
        query_plans=[plan],
        neighbors=[[7, 9]],
        top_k=2,
        direct_peer_urls={101: "http://localhost:6843", 202: "http://localhost:6853"},
        shard_key_to_peer={
            "centroid_00": 101,
            "centroid_01": 202,
            "centroid_02": 101,
            "centroid_03": 202,
        },
        include_per_query_metrics=True,
    )

    assert sorted((stage, peer) for stage, peer, _search in calls) == [
        (1, 101),
        (1, 202),
        (2, 101),
        (2, 202),
    ]
    for _stage, _peer, search in calls:
        assert set(search["hnsw_ef_by_shard"]) == set(search["shard_key"])
        assert set(search["hnsw_ef_by_shard"].values()) == {36}
    assert result["search_batch_calls"] == 4
    assert result["search_request_count"] == 4
    assert result["hits"] == 2
    assert result["per_query_rows"][0]["retrieved_ids"] == "7 9"


def test_kmeans_simple_nprobe_plan_uses_centroid_probes_with_fixed_ef_only():
    module = load_module()
    query = module.np.array([1.0, 0.0], dtype=module.np.float32)
    centroids = module.np.array(
        [
            [0.0, 1.0],
            [0.9, 0.1],
            [-1.0, 0.0],
            [0.7, 0.7],
        ],
        dtype=module.np.float32,
    )

    plan = module.kmeans_simple_nprobe_search_plan(
        query=query,
        centroids=centroids,
        nprobe=2,
        top_k=10,
        hnsw_ef=64,
        use_payload_source_id=False,
    )

    assert plan["visited_shards"] == 2
    assert plan["upper_hits"] == 0
    assert plan["assigned_ef_sum"] == 128
    assert plan["assigned_ef_count"] == 2
    assert plan["searches"] == [
        {
            "vector": [1.0, 0.0],
            "limit": 10,
            "params": {"hnsw_ef": 64},
            "with_payload": False,
            "with_vector": False,
            "shard_key": ["centroid_01", "centroid_03"],
            "hnsw_ef_by_shard": {
                "centroid_01": 64,
                "centroid_03": 64,
            },
        }
    ]
    assert "hnsw_entry_points_by_shard" not in plan["searches"][0]
    assert set(plan["searches"][0]["hnsw_ef_by_shard"].values()) == {64}
    assert "source_id_dedup_block_size" not in plan["searches"][0]


def test_kmeans_simple_nprobe_default_is_unchanged_and_explicit_chunking_is_disjoint():
    module = load_module()
    query = module.np.array([1.0, 0.0], dtype=module.np.float32)
    centroids = module.np.array(
        [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.0, 1.0]],
        dtype=module.np.float32,
    )

    default_plan = module.kmeans_simple_nprobe_search_plan(
        query,
        centroids,
        nprobe=3,
        top_k=10,
        hnsw_ef=64,
        use_payload_source_id=False,
    )
    chunked_plan = module.kmeans_simple_nprobe_search_plan(
        query,
        centroids,
        nprobe=3,
        top_k=10,
        hnsw_ef=64,
        use_payload_source_id=False,
        fixed_ef_shard_chunk_size=2,
    )

    assert len(default_plan["searches"]) == 1
    assert [len(search["shard_key"]) for search in chunked_plan["searches"]] == [2, 1]
    default_keys = default_plan["searches"][0]["shard_key"]
    chunked_keys = [
        shard_key
        for search in chunked_plan["searches"]
        for shard_key in search["shard_key"]
    ]
    assert chunked_keys == default_keys
    assert len(chunked_keys) == len(set(chunked_keys)) == 3
    assert all(search["params"] == {"hnsw_ef": 64} for search in chunked_plan["searches"])


def test_simple_kmeans_point_to_shards_by_distance_alpha_controls_expansion():
    module = load_module()
    train = module.np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=module.np.float32,
    )
    centroids = module.np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=module.np.float32,
    )

    point_to_shards = module.simple_kmeans_point_to_shards_by_distance_alpha(
        train,
        centroids,
        alpha=1.1,
        chunk_size=2,
    )

    assert point_to_shards == [[0], [0, 1], [1]]
    assert module.total_assigned_points(point_to_shards) == 4


def test_kmeans_simple_nprobe_plan_passes_source_id_dedup_block_size():
    module = load_module()
    query = module.np.array([1.0, 0.0], dtype=module.np.float32)
    centroids = module.np.array([[1.0, 0.0], [0.0, 1.0]], dtype=module.np.float32)

    plan = module.kmeans_simple_nprobe_search_plan(
        query=query,
        centroids=centroids,
        nprobe=2,
        top_k=10,
        hnsw_ef=64,
        use_payload_source_id=True,
        source_id_dedup_block_size=1001,
    )

    assert plan["searches"][0]["with_payload"] == ["source_id"]
    assert plan["searches"][0]["source_id_dedup_block_size"] == 1001


def test_routed_search_plan_groups_selected_shards_by_dynamic_ef():
    module = load_module()

    point_to_shards = [[], [0, 2], [1], [2], [1], []]
    plan = module.routed_search_plan(
        query=[0.1, 0.2],
        upper_labels=[1, 2, 3, 4],
        point_to_shards=point_to_shards,
        num_shards=3,
        top_k=10,
        base_ef=20,
        factor=4,
        search_all_shards=False,
        use_payload_source_id=False,
    )

    assert plan["visited_shards"] == 3
    assert plan["upper_hits"] == 5
    assert plan["assigned_ef_sum"] == 80
    assert plan["assigned_ef_count"] == 3
    assert plan["searches"] == [
        {
            "vector": [0.1, 0.2],
            "limit": 10,
            "params": {"hnsw_ef": 24},
            "with_payload": False,
            "with_vector": False,
            "shard_key": ["centroid_00"],
        },
        {
            "vector": [0.1, 0.2],
            "limit": 10,
            "params": {"hnsw_ef": 28},
            "with_payload": False,
            "with_vector": False,
            "shard_key": ["centroid_01", "centroid_02"],
        },
    ]


def test_compact_ef_modes_reduce_dynamic_efs_to_one_value():
    module = load_module()

    assert module.compact_ef_value([24, 28, 28], "max") == 28
    assert module.compact_ef_value([24, 28, 29], "mean_ceil") == 27


def test_routed_result_limit_can_preserve_per_shard_candidate_width():
    module = load_module()

    assert module.routed_result_limit(top_k=10, shard_count=3, mode="top_k") == 10
    assert module.routed_result_limit(top_k=10, shard_count=3, mode="per_shard_top_k") == 30
    assert module.routed_result_limit(
        top_k=10,
        shard_count=3,
        mode="fixed_multiplier",
        multiplier=4,
    ) == 40
    assert module.routed_result_limit(
        top_k=10,
        shard_count=3,
        mode="fixed_multiplier",
        multiplier=1.5,
    ) == 15


def test_routed_search_plan_can_compact_selected_shards_into_one_request():
    module = load_module()

    point_to_shards = [[], [0, 2], [1], [2], [1], []]
    plan = module.routed_search_plan(
        query=[0.1, 0.2],
        upper_labels=[1, 2, 3, 4],
        point_to_shards=point_to_shards,
        num_shards=3,
        top_k=10,
        base_ef=20,
        factor=4,
        search_all_shards=False,
        use_payload_source_id=False,
        routed_execution_mode="compact_query_ef",
        compact_ef_mode="max",
        routed_result_limit_mode="per_shard_top_k",
    )

    assert plan["visited_shards"] == 3
    assert plan["upper_hits"] == 5
    assert plan["assigned_ef_sum"] == 84
    assert plan["assigned_ef_count"] == 3
    assert plan["searches"] == [
        {
            "vector": [0.1, 0.2],
            "limit": 30,
            "params": {"hnsw_ef": 28},
            "with_payload": False,
            "with_vector": False,
            "shard_key": ["centroid_00", "centroid_01", "centroid_02"],
        }
    ]


def test_routed_search_plan_supports_fixed_candidate_multiplier():
    module = load_module()

    point_to_shards = [[], [0, 2], [1], [2], [1], []]
    plan = module.routed_search_plan(
        query=[0.1, 0.2],
        upper_labels=[1, 2, 3, 4],
        point_to_shards=point_to_shards,
        num_shards=3,
        top_k=10,
        base_ef=20,
        factor=4,
        search_all_shards=False,
        use_payload_source_id=False,
        routed_execution_mode="compact_query_ef",
        compact_ef_mode="max",
        routed_result_limit_mode="fixed_multiplier",
        routed_result_limit_multiplier=4,
    )

    assert plan["visited_shards"] == 3
    assert plan["searches"] == [
        {
            "vector": [0.1, 0.2],
            "limit": 40,
            "params": {"hnsw_ef": 28},
            "with_payload": False,
            "with_vector": False,
            "shard_key": ["centroid_00", "centroid_01", "centroid_02"],
        }
    ]


def test_routed_search_plan_can_emit_per_shard_multi_ep_requests():
    module = load_module()

    point_to_shards = [[], [0, 2], [1], [2], [1], []]
    plan = module.routed_search_plan(
        query=[0.1, 0.2],
        upper_labels=[1, 2, 3, 4],
        point_to_shards=point_to_shards,
        num_shards=3,
        top_k=10,
        base_ef=20,
        factor=4,
        search_all_shards=False,
        use_payload_source_id=False,
        routed_execution_mode="per_shard_multi_ep",
    )

    assert plan["visited_shards"] == 3
    assert plan["upper_hits"] == 5
    assert plan["assigned_ef_sum"] == 80
    assert plan["assigned_ef_count"] == 3
    assert plan["searches"] == [
        {
            "vector": [0.1, 0.2],
            "limit": 10,
            "params": {"hnsw_ef": 24},
            "with_payload": False,
            "with_vector": False,
            "shard_key": ["centroid_00"],
            "hnsw_entry_points": [1],
        },
        {
            "vector": [0.1, 0.2],
            "limit": 10,
            "params": {"hnsw_ef": 28},
            "with_payload": False,
            "with_vector": False,
            "shard_key": ["centroid_01"],
            "hnsw_entry_points": [2, 4],
        },
        {
            "vector": [0.1, 0.2],
            "limit": 10,
            "params": {"hnsw_ef": 28},
            "with_payload": False,
            "with_vector": False,
            "shard_key": ["centroid_02"],
            "hnsw_entry_points": [1, 3],
        },
    ]


def test_routed_search_plan_can_compact_multi_ep_by_shard():
    module = load_module()

    point_to_shards = [[], [0, 2], [1], [2], [1], []]
    plan = module.routed_search_plan(
        query=[0.1, 0.2],
        upper_labels=[1, 2, 3, 4],
        point_to_shards=point_to_shards,
        num_shards=3,
        top_k=10,
        base_ef=20,
        factor=4,
        search_all_shards=False,
        use_payload_source_id=False,
        routed_execution_mode="compact_multi_ep",
        source_id_dedup_block_size=1001,
    )

    assert plan["visited_shards"] == 3
    assert plan["assigned_ef_sum"] == 80
    assert plan["assigned_ef_count"] == 3
    assert plan["searches"] == [
        {
            "vector": [0.1, 0.2],
            "limit": 10,
            "params": {"hnsw_ef": 28},
            "with_payload": False,
            "with_vector": False,
            "shard_key": ["centroid_00", "centroid_01", "centroid_02"],
            "hnsw_entry_points_by_shard": {
                "centroid_00": [2],
                "centroid_01": [1004, 1006],
                "centroid_02": [2004, 2006],
            },
            "hnsw_ef_by_shard": {
                "centroid_00": 24,
                "centroid_01": 28,
                "centroid_02": 28,
            },
            "source_id_dedup_block_size": 1001,
        }
    ]


def test_compact_routed_search_plan_matches_original_compact_multi_ep_request():
    module = load_module()

    point_to_shards = [[], [0, 2], [1], [2], [1], []]
    manifest = module.build_compact_routing_manifest(
        point_to_shards,
        num_shards=3,
        source_id_dedup_block_size=1001,
    )

    compact_plan = module.compact_routed_search_plan(
        query=[0.1, 0.2],
        upper_labels=[1, 2, 3, 4],
        manifest=manifest,
        top_k=10,
        base_ef=20,
        factor=4,
        search_all_shards=False,
        use_payload_source_id=False,
        routed_result_limit_mode="top_k",
        routed_result_limit_multiplier=1,
    )
    original_plan = module.routed_search_plan(
        query=[0.1, 0.2],
        upper_labels=[1, 2, 3, 4],
        point_to_shards=point_to_shards,
        num_shards=3,
        top_k=10,
        base_ef=20,
        factor=4,
        search_all_shards=False,
        use_payload_source_id=False,
        routed_execution_mode="compact_multi_ep",
        source_id_dedup_block_size=1001,
    )

    assert compact_plan == original_plan


def test_compact_materialized_evaluator_uses_compact_plans(monkeypatch):
    module = load_module()

    calls = {"upper": 0, "execute": []}

    def fake_compute_upper_labels(_upper_index, queries, upper_k):
        calls["upper"] += 1
        assert len(queries) == 2
        assert upper_k == 2
        return module.np.array([[1, 2], [2, 3]], dtype=module.np.int64)

    def fake_execute_query_plans_once(
        _base_url,
        _collection,
        query_plans,
        neighbors,
        top_k,
        direct_peer_urls=None,
        shard_key_to_peer=None,
        lower_execution_order="query_major",
        direct_peer_local_premerge=False,
        score_higher_is_better=True,
    ):
        calls["execute"].append(query_plans)
        assert lower_execution_order == "query_major"
        assert direct_peer_local_premerge is False
        assert query_plans[0]["searches"][0]["hnsw_entry_points_by_shard"] == {
            "centroid_00": [2],
            "centroid_01": [1004],
        }
        return {
            "hits": len(query_plans) * top_k,
            "query_count": len(query_plans),
            "visited_shards": sum(plan["visited_shards"] for plan in query_plans),
            "upper_hits": sum(plan["upper_hits"] for plan in query_plans),
            "assigned_ef_sum": sum(plan["assigned_ef_sum"] for plan in query_plans),
            "assigned_ef_count": sum(plan["assigned_ef_count"] for plan in query_plans),
            "search_batch_calls": 1,
            "search_request_count": sum(len(plan["searches"]) for plan in query_plans),
        }

    monkeypatch.setattr(module, "compute_upper_labels", fake_compute_upper_labels)
    monkeypatch.setattr(module, "execute_query_plans_once", fake_execute_query_plans_once)

    result = module.evaluate_config_compact_materialized_routing(
        "http://qdrant",
        "collection",
        module.np.array([[0.1], [0.2]], dtype=module.np.float32),
        module.np.array([[1], [2]], dtype=module.np.int32),
        top_k=1,
        upper_index=object(),
        upper_k=2,
        base_ef=20,
        factor=4,
        batch_size=2,
        point_to_shards=[[], [0], [1], [1]],
        num_shards=2,
        use_payload_source_id=False,
        source_id_dedup_block_size=1001,
    )

    assert calls["upper"] == 1
    assert len(calls["execute"]) == 1
    assert result["recall_at_k"] == 1.0
    assert result["avg_search_requests_per_query"] == 1.0


def test_shard_major_expands_compact_multi_ep_searches_to_single_shard_requests():
    module = load_module()

    search = {
        "vector": [0.1, 0.2],
        "limit": 10,
        "params": {"hnsw_ef": 28},
        "with_payload": False,
        "with_vector": False,
        "shard_key": ["centroid_00", "centroid_01"],
        "hnsw_entry_points_by_shard": {
            "centroid_00": [11, 13],
            "centroid_01": [1011],
        },
        "hnsw_ef_by_shard": {
            "centroid_00": 24,
            "centroid_01": 28,
        },
        "source_id_dedup_block_size": 1001,
    }

    expanded = module.shard_major_searches_for_query(search)

    assert expanded == [
        (
            "centroid_00",
            {
                "vector": [0.1, 0.2],
                "limit": 10,
                "params": {"hnsw_ef": 24},
                "with_payload": False,
                "with_vector": False,
                "shard_key": ["centroid_00"],
                "source_id_dedup_block_size": 1001,
                "hnsw_entry_points": [11, 13],
            },
        ),
        (
            "centroid_01",
            {
                "vector": [0.1, 0.2],
                "limit": 10,
                "params": {"hnsw_ef": 28},
                "with_payload": False,
                "with_vector": False,
                "shard_key": ["centroid_01"],
                "source_id_dedup_block_size": 1001,
                "hnsw_entry_points": [1011],
            },
        ),
    ]


def test_shard_major_execution_groups_searches_by_shard_and_merges_by_original_query(monkeypatch):
    module = load_module()
    calls = []

    def fake_search_batch(_base_url, _collection, searches):
        calls.append([search["shard_key"][0] for search in searches])
        rows = []
        for search in searches:
            shard = search["shard_key"][0]
            marker = search["marker"]
            if shard == "centroid_00" and marker == "q0":
                rows.append([(0.90, 10)])
            elif shard == "centroid_00" and marker == "q1":
                rows.append([(0.95, 30)])
            elif shard == "centroid_01" and marker == "q0":
                rows.append([(0.99, 20)])
            else:
                rows.append([])
        return rows

    monkeypatch.setattr(module, "search_batch", fake_search_batch)

    result = module.execute_query_plans_once(
        base_url="http://controller:6333",
        collection="collection",
        query_plans=[
            {
                "searches": [
                    {"marker": "q0", "shard_key": ["centroid_01"], "limit": 2},
                    {"marker": "q0", "shard_key": ["centroid_00"], "limit": 2},
                ],
                "visited_shards": 2,
                "upper_hits": 2,
                "assigned_ef_sum": 48,
                "assigned_ef_count": 2,
            },
            {
                "searches": [
                    {"marker": "q1", "shard_key": ["centroid_00"], "limit": 2},
                ],
                "visited_shards": 1,
                "upper_hits": 1,
                "assigned_ef_sum": 24,
                "assigned_ef_count": 1,
            },
        ],
        neighbors=[[20, 99], [30, 99]],
        top_k=2,
        lower_execution_order="shard_major",
    )

    assert calls == [["centroid_00", "centroid_00", "centroid_01"]]
    assert result["hits"] == 2
    assert result["search_batch_calls"] == 1
    assert result["search_request_count"] == 3


def test_preplanned_concurrent_evaluator_merges_multiple_searches_per_query(monkeypatch):
    module = load_module()
    calls = []

    def fake_search_batch(_base_url, _collection, searches):
        calls.append([search["marker"] for search in searches])
        rows = []
        for search in searches:
            marker = search["marker"]
            if marker == "q0_low":
                rows.append([(0.50, 11)])
            elif marker == "q0_high":
                rows.append([(0.95, 22)])
            elif marker == "q1":
                rows.append([(0.90, 33)])
            else:
                rows.append([])
        return rows

    monkeypatch.setattr(module, "search_batch", fake_search_batch)

    plans = [
        {
            "searches": [{"marker": "q0_low"}, {"marker": "q0_high"}],
            "visited_shards": 2,
            "upper_hits": 2,
            "assigned_ef_sum": 48,
            "assigned_ef_count": 2,
        },
        {
            "searches": [{"marker": "q1"}],
            "visited_shards": 1,
            "upper_hits": 1,
            "assigned_ef_sum": 24,
            "assigned_ef_count": 1,
        },
    ]

    result = module.evaluate_preplanned_searches_concurrent(
        base_url="http://example.invalid",
        collection="collection",
        query_plans=plans,
        neighbors=[[22, 99], [33, 99]],
        top_k=2,
        batch_size=2,
        concurrency=1,
    )

    assert calls == [["q0_low", "q0_high", "q1"]]
    assert result["recall_at_k"] == 0.5
    assert result["avg_visited_shards"] == 1.5
    assert result["avg_upper_hits"] == 1.5
    assert result["avg_assigned_ef_per_visited_shard"] == 24.0


def test_direct_peer_execution_splits_searches_by_shard_owner(monkeypatch):
    module = load_module()
    calls = []

    def fake_search_batch(base_url, _collection, searches):
        calls.append((base_url, [search["shard_key"] for search in searches]))
        rows = []
        for search in searches:
            if base_url.endswith("6843"):
                rows.append([(0.95, 11)])
            else:
                rows.append([(0.90, 22)])
        return rows

    monkeypatch.setattr(module, "search_batch", fake_search_batch)

    result = module.execute_query_plans_once(
        base_url="http://controller:6333",
        collection="collection",
        query_plans=[
            {
                "searches": [
                    {
                        "vector": [0.1, 0.2],
                        "limit": 2,
                        "params": {"hnsw_ef": 32},
                        "with_payload": False,
                        "with_vector": False,
                        "shard_key": ["centroid_00", "centroid_01"],
                    }
                ],
                "visited_shards": 2,
                "upper_hits": 2,
                "assigned_ef_sum": 64,
                "assigned_ef_count": 2,
            }
        ],
        neighbors=[[11, 22]],
        top_k=2,
        direct_peer_urls={101: "http://localhost:6843", 202: "http://localhost:6853"},
        shard_key_to_peer={"centroid_00": 101, "centroid_01": 202},
    )

    assert sorted(calls) == [
        ("http://localhost:6843", [["centroid_00"]]),
        ("http://localhost:6853", [["centroid_01"]]),
    ]
    assert result["hits"] == 2
    assert result["search_batch_calls"] == 2
    assert result["search_request_count"] == 2


def test_direct_peer_local_premerge_reduces_peer_candidate_streams(monkeypatch):
    module = load_module()

    def fake_search_batch(base_url, _collection, searches):
        rows = []
        for search in searches:
            shard_key = search["shard_key"][0]
            if base_url.endswith("6843") and shard_key == "centroid_00":
                rows.append([(0.99, 10), (0.80, 20)])
            elif base_url.endswith("6843") and shard_key == "centroid_01":
                rows.append([(0.97, 30), (0.96, 40)])
            elif base_url.endswith("6853") and shard_key == "centroid_02":
                rows.append([(0.98, 10), (0.95, 50)])
            else:
                rows.append([])
        return rows

    monkeypatch.setattr(module, "search_batch", fake_search_batch)

    result = module.execute_query_plans_once(
        base_url="http://controller:6333",
        collection="collection",
        query_plans=[
            {
                "searches": [
                    {
                        "vector": [0.1, 0.2],
                        "limit": 4,
                        "params": {"hnsw_ef": 32},
                        "with_payload": False,
                        "with_vector": False,
                        "shard_key": ["centroid_00", "centroid_01", "centroid_02"],
                    }
                ],
                "visited_shards": 3,
                "upper_hits": 3,
                "assigned_ef_sum": 96,
                "assigned_ef_count": 3,
            }
        ],
        neighbors=[[10, 30, 40, 50]],
        top_k=4,
        direct_peer_urls={101: "http://localhost:6843", 202: "http://localhost:6853"},
        shard_key_to_peer={
            "centroid_00": 101,
            "centroid_01": 101,
            "centroid_02": 202,
        },
        lower_execution_order="shard_major",
        direct_peer_local_premerge=True,
    )

    assert result["hits"] == 4
    assert result["candidate_group_count"] == 2
    assert result["avg_candidate_groups_per_query"] == 2.0
    assert result["search_request_count"] == 3


def test_extract_shard_costs_from_compact_multi_ep_query_plans():
    module = load_module()
    plans = [
        {
            "searches": [
                {
                    "shard_key": ["centroid_00", "centroid_02"],
                    "hnsw_ef_by_shard": {"centroid_00": 40, "centroid_02": 80},
                }
            ],
            "visited_shards": 2,
            "upper_hits": 3,
            "assigned_ef_sum": 120,
            "assigned_ef_count": 2,
        },
        {
            "searches": [
                {
                    "shard_key": ["centroid_01"],
                    "hnsw_ef_by_shard": {"centroid_01": 60},
                }
            ],
            "visited_shards": 1,
            "upper_hits": 1,
            "assigned_ef_sum": 60,
            "assigned_ef_count": 1,
        },
    ]

    traces = module.query_shard_cost_traces(plans)

    assert traces == [
        {"centroid_00": 40.0, "centroid_02": 80.0},
        {"centroid_01": 60.0},
    ]


def test_method4_aware_placement_reduces_estimated_tail_load():
    module = load_module()
    traces = [
        {"centroid_00": 100.0, "centroid_01": 90.0, "centroid_02": 10.0},
        {"centroid_00": 100.0, "centroid_01": 90.0, "centroid_03": 10.0},
    ]
    round_robin = {
        "centroid_00": 0,
        "centroid_01": 0,
        "centroid_02": 1,
        "centroid_03": 1,
    }

    optimized = module.greedy_method4_aware_placement(
        traces,
        peer_count=2,
        initial_placement=round_robin,
    )
    before = module.evaluate_query_peer_loads(traces, round_robin)
    after = module.evaluate_query_peer_loads(traces, optimized)

    assert after["avg_query_max_peer_load"] < before["avg_query_max_peer_load"]
    assert optimized["centroid_00"] != optimized["centroid_01"]


def test_placement_simulation_summary_reports_improvement():
    module = load_module()
    plans = [
        {
            "searches": [
                {
                    "shard_key": ["centroid_00", "centroid_01", "centroid_02"],
                    "hnsw_ef_by_shard": {
                        "centroid_00": 100,
                        "centroid_01": 90,
                        "centroid_02": 10,
                    },
                }
            ],
            "visited_shards": 3,
            "upper_hits": 3,
            "assigned_ef_sum": 200,
            "assigned_ef_count": 3,
        },
        {
            "searches": [
                {
                    "shard_key": ["centroid_00", "centroid_01", "centroid_03"],
                    "hnsw_ef_by_shard": {
                        "centroid_00": 100,
                        "centroid_01": 90,
                        "centroid_03": 10,
                    },
                }
            ],
            "visited_shards": 3,
            "upper_hits": 3,
            "assigned_ef_sum": 200,
            "assigned_ef_count": 3,
        },
    ]

    summary = module.placement_simulation_summary(plans, peer_count=2)

    assert summary["peer_count"] == 2
    assert summary["method4_aware"]["avg_query_max_peer_load"] < summary["round_robin"]["avg_query_max_peer_load"]
    assert summary["improvement_pct"]["avg_query_max_peer_load"] > 0


def test_physical_execution_summary_estimates_worker_local_premerge_reduction():
    module = load_module()
    plans = [
        {
            "searches": [
                {
                    "shard_key": ["centroid_00", "centroid_01", "centroid_02"],
                    "hnsw_ef_by_shard": {
                        "centroid_00": 100,
                        "centroid_01": 80,
                        "centroid_02": 60,
                    },
                }
            ],
            "visited_shards": 3,
            "upper_hits": 6,
            "assigned_ef_sum": 240,
            "assigned_ef_count": 3,
        },
        {
            "searches": [
                {
                    "shard_key": ["centroid_01", "centroid_03"],
                    "hnsw_ef_by_shard": {
                        "centroid_01": 70,
                        "centroid_03": 50,
                    },
                }
            ],
            "visited_shards": 2,
            "upper_hits": 4,
            "assigned_ef_sum": 120,
            "assigned_ef_count": 2,
        },
    ]

    summary = module.physical_execution_summary(
        plans,
        shard_key_to_peer={
            "centroid_00": 101,
            "centroid_01": 101,
            "centroid_02": 202,
            "centroid_03": 202,
        },
        top_k=10,
    )

    assert summary["query_count"] == 2
    assert summary["avg_logical_shards_per_query"] == 2.5
    assert summary["avg_physical_peers_per_query"] == 2.0
    assert summary["avg_controller_merge_stream_reduction_pct"] == 20.0
    assert summary["avg_controller_candidate_reduction_pct"] == 20.0
    assert summary["avg_assigned_ef_sum_per_query"] == 180.0
    assert summary["avg_max_peer_assigned_ef_per_query"] == 125.0
    assert summary["p95_max_peer_assigned_ef_per_query"] == 174.5


def test_routed_end_to_end_concurrent_evaluator_includes_upper_routing(monkeypatch):
    module = load_module()
    calls = []

    def fake_compute_upper_labels(_upper_index, chunk, upper_k):
        calls.append(("upper", len(chunk), upper_k))
        return module.np.array([[1, 2], [3, 4]])

    def fake_search_batch(_base_url, _collection, searches):
        calls.append(("search", [search["shard_key"] for search in searches]))
        return [[(0.95, 101)], [(0.90, 202)]]

    monkeypatch.setattr(module, "compute_upper_labels", fake_compute_upper_labels)
    monkeypatch.setattr(module, "search_batch", fake_search_batch)

    result = module.evaluate_config_concurrent(
        base_url="http://example.invalid",
        collection="collection",
        queries=module.np.array([[0.1, 0.2], [0.3, 0.4]], dtype=module.np.float32),
        neighbors=module.np.array([[101, 999], [202, 999]]),
        top_k=2,
        upper_index=object(),
        upper_k=4,
        base_ef=20,
        factor=4,
        batch_size=2,
        concurrency=1,
        point_to_shards=[[], [0], [1], [1], [0]],
        num_shards=2,
        routed_execution_mode="compact_query_ef",
        compact_ef_mode="mean_ceil",
        routed_result_limit_mode="fixed_multiplier",
        routed_result_limit_multiplier=2,
    )

    assert calls == [
        ("upper", 2, 4),
        ("search", [["centroid_00", "centroid_01"], ["centroid_00", "centroid_01"]]),
    ]
    assert result["recall_at_k"] == 0.5
    assert result["avg_visited_shards"] == 2.0
    assert result["avg_upper_hits"] == 2.0
    assert result["search_batch_calls"] == 1
    assert result["avg_search_requests_per_query"] == 1.0


def test_all_shards_end_to_end_concurrent_evaluator_builds_all_shard_requests(monkeypatch):
    module = load_module()
    calls = []

    def fake_search_batch(_base_url, _collection, searches):
        calls.append([search["shard_key"] for search in searches])
        return [[(0.95, 11)], [(0.90, 22)]]

    monkeypatch.setattr(module, "search_batch", fake_search_batch)

    result = module.evaluate_all_shards_config_concurrent(
        base_url="http://example.invalid",
        collection="collection",
        queries=module.np.array([[0.1, 0.2], [0.3, 0.4]], dtype=module.np.float32),
        neighbors=module.np.array([[11, 99], [22, 99]]),
        top_k=2,
        num_shards=3,
        hnsw_ef=36,
        batch_size=2,
        concurrency=1,
        use_payload_source_id=False,
    )

    assert calls == [[
        ["centroid_00", "centroid_01", "centroid_02"],
        ["centroid_00", "centroid_01", "centroid_02"],
    ]]
    assert result["recall_at_k"] == 0.5
    assert result["avg_visited_shards"] == 3.0
    assert result["avg_upper_hits"] == 0.0
    assert result["avg_assigned_ef_per_visited_shard"] == 36.0
    assert result["search_batch_calls"] == 1
    assert result["avg_search_requests_per_query"] == 1.0


def test_all_shards_coordinator_default_uses_one_batch_with_full_uniform_ef_map(
    monkeypatch,
):
    module = load_module()
    calls = []

    def fake_search_batch(_base_url, _collection, searches):
        calls.append(searches)
        return [[(0.95, 11)], [(0.90, 22)]]

    monkeypatch.setattr(module, "search_batch", fake_search_batch)

    result = module.evaluate_all_shards_config(
        base_url="http://example.invalid",
        collection="collection",
        queries=module.np.array([[0.1, 0.2], [0.3, 0.4]], dtype=module.np.float32),
        neighbors=module.np.array([[11, 99], [22, 99]]),
        top_k=2,
        num_shards=46,
        hnsw_ef=72,
        batch_size=2,
        use_payload_source_id=True,
        source_id_dedup_block_size=1001,
    )

    expected_shard_keys = [
        module.shard_key_for_id(shard_id) for shard_id in range(46)
    ]
    assert len(calls) == 1
    assert len(calls[0]) == 2
    for search in calls[0]:
        assert search["shard_key"] == expected_shard_keys
        assert search["params"] == {"hnsw_ef": 72}
        assert search["hnsw_ef_by_shard"] == {
            shard_key: 72 for shard_key in expected_shard_keys
        }
        assert search["source_id_dedup_block_size"] == 1001
    assert result["recall_at_k"] == 0.5
    assert result["avg_visited_shards"] == 46.0
    assert result["avg_assigned_ef_per_visited_shard"] == 72.0
    assert result["search_batch_calls"] == 1
    assert result["avg_search_requests_per_query"] == 1.0


def test_all_shards_coordinator_chunking_uses_two_http_stages_and_keeps_global_recall(monkeypatch):
    module = load_module()
    calls = []

    def fake_search_batch(_base_url, _collection, searches):
        calls.append(searches)
        if len(calls) == 1:
            return [[(0.90, 11), (0.80, 44)], [(0.91, 22), (0.70, 55)]]
        return [[(0.95, 11), (0.85, 33)], [(0.96, 22), (0.86, 66)]]

    monkeypatch.setattr(module, "search_batch", fake_search_batch)

    result = module.evaluate_all_shards_config(
        base_url="http://example.invalid",
        collection="collection",
        queries=module.np.array([[0.1, 0.2], [0.3, 0.4]], dtype=module.np.float32),
        neighbors=module.np.array([[11, 33], [22, 66]]),
        top_k=2,
        num_shards=46,
        hnsw_ef=72,
        batch_size=2,
        use_payload_source_id=True,
        source_id_dedup_block_size=1001,
        fixed_ef_shard_chunk_size=32,
    )

    assert len(calls) == 2
    assert [[len(search["shard_key"]) for search in stage] for stage in calls] == [
        [32, 32],
        [14, 14],
    ]
    expected_chunks = [
        [module.shard_key_for_id(shard_id) for shard_id in range(32)],
        [module.shard_key_for_id(shard_id) for shard_id in range(32, 46)],
    ]
    for stage_idx, stage in enumerate(calls):
        for search in stage:
            assert search["shard_key"] == expected_chunks[stage_idx]
            assert set(search["hnsw_ef_by_shard"]) == set(search["shard_key"])
            assert set(search["hnsw_ef_by_shard"].values()) == {72}
            assert search["params"] == {"hnsw_ef": 72}
            assert search["source_id_dedup_block_size"] == 1001
    assert result["recall_at_k"] == 1.0
    assert result["avg_visited_shards"] == 46.0
    assert result["avg_assigned_ef_per_visited_shard"] == 72.0
    assert result["search_batch_calls"] == 2
    assert result["avg_search_requests_per_query"] == 2.0


def test_kmeans_simple_nprobe_end_to_end_concurrent_evaluator_uses_fixed_ef(monkeypatch):
    module = load_module()
    calls = []

    centroids = module.np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.2],
        ],
        dtype=module.np.float32,
    )

    def fake_search_batch(_base_url, _collection, searches):
        calls.append(searches)
        return [[(0.95, 11)], [(0.90, 22)]]

    def fail_compute_upper_labels(*_args, **_kwargs):
        raise AssertionError("simple nprobe must not use the upper HNSW index")

    monkeypatch.setattr(module, "search_batch", fake_search_batch)
    monkeypatch.setattr(module, "compute_upper_labels", fail_compute_upper_labels)

    result = module.evaluate_kmeans_simple_nprobe_config_concurrent(
        base_url="http://example.invalid",
        collection="collection",
        queries=module.np.array([[1.0, 0.0], [0.0, 1.0]], dtype=module.np.float32),
        neighbors=module.np.array([[11, 99], [22, 99]]),
        top_k=2,
        centroids=centroids,
        nprobe=2,
        hnsw_ef=48,
        batch_size=2,
        concurrency=1,
        use_payload_source_id=False,
    )

    assert len(calls) == 1
    assert [search["shard_key"] for search in calls[0]] == [
        ["centroid_00", "centroid_02"],
        ["centroid_01", "centroid_02"],
    ]
    assert [search["params"] for search in calls[0]] == [{"hnsw_ef": 48}, {"hnsw_ef": 48}]
    assert all("hnsw_entry_points_by_shard" not in search for search in calls[0])
    assert all(
        set(search["hnsw_ef_by_shard"].values()) == {48}
        for search in calls[0]
    )
    assert all("source_id_dedup_block_size" not in search for search in calls[0])
    assert result["recall_at_k"] == 0.5
    assert result["avg_visited_shards"] == 2.0
    assert result["avg_upper_hits"] == 0.0
    assert result["avg_assigned_ef_per_visited_shard"] == 48.0
    assert result["search_batch_calls"] == 1
    assert result["avg_search_requests_per_query"] == 1.0


def test_decode_copy_id_recovers_source_point_and_shard():
    module = load_module()

    copy_id = module.encode_copy_id(point_index=7, shard_id=3, num_points=100)

    assert module.decode_copy_id(copy_id, num_points=100) == (7, 3)


def test_add_scrolled_points_to_upper_shards_uses_payload_then_encoded_id():
    module = load_module()
    point_to_shards = [[] for _ in range(6)]

    module.add_scrolled_points_to_upper_shards(
        point_to_shards=point_to_shards,
        upper_set={1, 3},
        shard_id=2,
        points=[
            {"id": module.encode_copy_id(1, 2, 6), "payload": {}},
            {"id": 123456, "payload": {"source_id": 3}},
            {"id": module.encode_copy_id(4, 2, 6), "payload": {}},
        ],
        num_points=6,
    )

    assert point_to_shards == [[], [2], [], [2], [], []]
