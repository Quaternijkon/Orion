from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_module():
    script_path = Path("/home/taig/dry/qdrant/tools/qdrant_two_level_routing_experiment.py")
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


def test_point_indices_by_shard_preserves_multi_assignment_expansion():
    module = load_module()

    by_shard = module.point_indices_by_shard([[0, 2], [2], [1], [0, 1]], num_shards=3)

    assert [indices.tolist() for indices in by_shard] == [[0, 3], [2, 3], [0, 1]]
    assert module.total_assigned_points([[0, 2], [2], [1], [0, 1]]) == 6


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
    assert plan["searches"] == [
        {
            "vector": [0.1, 0.2],
            "limit": 10,
            "params": {"hnsw_ef": 36},
            "with_payload": ["source_id"],
            "with_vector": False,
            "shard_key": ["centroid_00", "centroid_01", "centroid_02"],
        }
    ]


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
