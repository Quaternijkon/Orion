import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path

import h5py
import numpy as np
import pytest


def load_module():
    module_path = Path("tools/method4_claim_d_high_recall_latency.py")
    spec = importlib.util.spec_from_file_location("method4_claim_d_high_recall_latency", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_summarize_batch_rows_reports_recall_latency_and_work():
    module = load_module()

    rows = [
        {
            "hits": 8,
            "query_count": 10,
            "wall_s": 0.10,
            "batch_latency_ms": 10.0,
            "visited_shards": 100,
            "assigned_ef_sum": 1000,
            "search_batch_calls": 1,
        },
        {
            "hits": 9,
            "query_count": 10,
            "wall_s": 0.20,
            "batch_latency_ms": 20.0,
            "visited_shards": 120,
            "assigned_ef_sum": 1400,
            "search_batch_calls": 1,
        },
        {
            "hits": 10,
            "query_count": 10,
            "wall_s": 0.30,
            "batch_latency_ms": 40.0,
            "visited_shards": 140,
            "assigned_ef_sum": 1800,
            "search_batch_calls": 1,
        },
    ]

    summary = module.summarize_batch_rows("method4", "r097", rows, top_k=1)

    assert summary["method"] == "method4"
    assert summary["config_label"] == "r097"
    assert summary["query_count"] == 30
    assert summary["recall_at_10"] == 0.9
    assert summary["qps"] == pytest.approx(50.0)
    assert summary["avg_visited_shards"] == 12.0
    assert summary["avg_assigned_ef_sum"] == 140.0
    assert summary["batch_latency_p50_ms"] == 20.0
    assert summary["batch_latency_p95_ms"] == 38.0
    assert summary["batch_latency_p99_ms"] == 39.6


def test_build_naive_plans_delegates_to_all_shard_helper():
    module = load_module()

    class FakeQdrantTool:
        def __init__(self):
            self.calls = []

        def build_all_shard_search_plans(
            self,
            queries,
            top_k,
            num_shards,
            hnsw_ef,
            use_payload_source_id,
            source_id_dedup_block_size,
        ):
            self.calls.append(
                (queries.shape, top_k, num_shards, hnsw_ef, use_payload_source_id, source_id_dedup_block_size)
            )
            return [{"searches": [], "visited_shards": num_shards}]

    q2l = FakeQdrantTool()
    queries = np.zeros((2, 3), dtype=np.float32)

    plans = module.build_naive_plans(
        q2l,
        queries,
        top_k=10,
        num_shards=46,
        hnsw_ef=160,
        use_payload_source_id=True,
        source_id_dedup_block_size=1234,
    )

    assert plans == [{"searches": [], "visited_shards": 46}]
    assert q2l.calls == [((2, 3), 10, 46, 160, True, 1234)]


def test_parse_naive_efs_can_skip_naive_controls():
    module = load_module()

    assert module.parse_naive_efs(None, skip_naive=True) == []
    assert module.parse_naive_efs(["naive_ef76=76"], skip_naive=True) == []


def test_parse_method4_configs_can_skip_method4_controls():
    module = load_module()

    assert module.parse_method4_configs(None, skip_method4=True) == []
    assert module.parse_method4_configs(["m4=160,80,20"], skip_method4=True) == []


def test_parse_kmeans_configs_requires_explicit_values_and_can_skip():
    module = load_module()

    assert module.parse_kmeans_configs(None) == []
    assert module.parse_kmeans_configs(["kmeans_u160_b80_f8=160,80,8"]) == [
        ("kmeans_u160_b80_f8", 160, 80, 8)
    ]
    assert module.parse_kmeans_configs(["kmeans_u160_b80_f8=160,80,8"], skip_kmeans=True) == []


def test_parse_simple_kmeans_configs_requires_explicit_values_and_can_skip():
    module = load_module()

    assert module.parse_simple_kmeans_configs(None) == []
    assert module.parse_simple_kmeans_configs(["simple_n4_ef32=4,32"]) == [("simple_n4_ef32", 4, 32)]
    assert module.parse_simple_kmeans_configs(["simple_n4_ef32=4,32"], skip_simple_kmeans=True) == []


def test_build_cpp_kmeans_plans_uses_sampled_upper_labels_and_label_to_shard():
    module = load_module()

    class FakeQdrantTool:
        def __init__(self):
            self.calls = []

        def build_cpp_kmeans_baseline_assignments(self, train, num_shards, train_size, kmeans_iters, seed):
            self.calls.append(("assign", train.shape, num_shards, train_size, kmeans_iters, seed))
            return np.array([0, 1, 0, 1], dtype=np.int32), np.zeros((2, 3), dtype=np.float32)

        def sample_cpp_kmeans_upper_points(self, train, assignments, num_shards, denominator, seed):
            self.calls.append(("sample", train.shape, tuple(assignments.tolist()), num_shards, denominator, seed))
            return (
                np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
                np.array([101, 202], dtype=np.int64),
                {101: "shard_key_0", 202: "shard_key_1"},
                [],
            )

        def build_upper_index(
            self,
            upper_vectors,
            upper_ids,
            dim,
            upper_m,
            upper_ef_construction,
            upper_search_ef,
            hnsw_space="cosine",
        ):
            self.calls.append(
                (
                    "upper",
                    upper_vectors.tolist(),
                    upper_ids.tolist(),
                    dim,
                    upper_m,
                    upper_ef_construction,
                    upper_search_ef,
                    hnsw_space,
                )
            )
            return "upper-index"

        def compute_upper_labels(self, upper_index, queries, upper_k):
            self.calls.append(("labels", upper_index, queries.shape, upper_k))
            return np.array([[101, 202], [202, 101]], dtype=np.int64)

        def legacy_routed_search_plan(
            self,
            query,
            labels_row,
            label_to_shard,
            top_k,
            base_ef,
            factor,
            use_payload_source_id,
            routed_execution_mode,
            compact_ef_mode,
            routed_result_limit_mode,
            routed_result_limit_multiplier,
            source_id_dedup_block_size,
        ):
            self.calls.append(
                (
                    "plan",
                    query,
                    labels_row.tolist(),
                    dict(label_to_shard),
                    top_k,
                    base_ef,
                    factor,
                    use_payload_source_id,
                    routed_execution_mode,
                    compact_ef_mode,
                    routed_result_limit_mode,
                    routed_result_limit_multiplier,
                    source_id_dedup_block_size,
                )
            )
            return {"labels": labels_row.tolist(), "searches": []}

    q2l = FakeQdrantTool()
    train = np.zeros((4, 3), dtype=np.float32)
    queries = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    plans = module.build_cpp_kmeans_plans(
        q2l,
        train,
        queries,
        num_shards=2,
        top_k=10,
        upper_k=2,
        base_ef=80,
        factor=8,
        cpp_kmeans_train_size=100,
        kmeans_iters=7,
        upper_sample_seed=123,
        sample_denominator=2,
        upper_m=16,
        upper_ef_construction=64,
        upper_search_ef=32,
        hnsw_space="cosine",
        source_id_dedup_block_size=999,
    )

    assert plans == [{"labels": [101, 202], "searches": []}, {"labels": [202, 101], "searches": []}]
    assert q2l.calls[0] == ("assign", (4, 3), 2, 100, 7, 123)
    assert q2l.calls[1] == ("sample", (4, 3), (0, 1, 0, 1), 2, 2, 123)
    assert q2l.calls[2][0] == "upper"
    assert q2l.calls[2][3:] == (3, 16, 64, 32, "cosine")
    assert q2l.calls[3] == ("labels", "upper-index", (2, 3), 2)
    assert q2l.calls[4][0] == "plan"
    assert q2l.calls[4][6:13] == (8, False, "compact_multi_ep", "max", "top_k", 1, None)


def test_build_simple_kmeans_plans_uses_centroids_and_nprobe_without_copy_id_encoding():
    module = load_module()

    class FakeQdrantTool:
        def __init__(self):
            self.calls = []

        def build_cpp_kmeans_baseline_assignments(self, train, num_shards, train_size, kmeans_iters, seed):
            self.calls.append(("assign", train.shape, num_shards, train_size, kmeans_iters, seed))
            return np.array([0, 1, 0, 1], dtype=np.int32), np.ones((2, 3), dtype=np.float32)

        def build_kmeans_simple_nprobe_search_plans(
            self,
            queries,
            centroids,
            nprobe,
            top_k,
            hnsw_ef,
            use_payload_source_id,
            source_id_dedup_block_size,
        ):
            self.calls.append(
                (
                    "plans",
                    queries.shape,
                    centroids.tolist(),
                    nprobe,
                    top_k,
                    hnsw_ef,
                    use_payload_source_id,
                    source_id_dedup_block_size,
                )
            )
            return [{"searches": [], "visited_shards": nprobe}]

    q2l = FakeQdrantTool()
    train = np.zeros((4, 3), dtype=np.float32)
    queries = np.zeros((2, 3), dtype=np.float32)

    plans = module.build_simple_kmeans_plans(
        q2l,
        train,
        queries,
        num_shards=2,
        top_k=10,
        nprobe=4,
        hnsw_ef=32,
        cpp_kmeans_train_size=100,
        kmeans_iters=7,
        upper_sample_seed=123,
    )

    assert plans == [{"searches": [], "visited_shards": 4}]
    assert q2l.calls[0] == ("assign", (4, 3), 2, 100, 7, 123)
    assert q2l.calls[1][0] == "plans"
    assert q2l.calls[1][3:] == (4, 10, 32, False, None)


def test_parse_args_supports_euclid_vector_distance(monkeypatch):
    module = load_module()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "method4_claim_d_high_recall_latency.py",
            "--vector-distance",
            "euclid",
        ],
    )

    args = module.parse_args()

    assert args.vector_distance == "euclid"


def test_parse_args_supports_shuffled_query_order(monkeypatch):
    module = load_module()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "method4_claim_d_high_recall_latency.py",
            "--query-order",
            "shuffled",
            "--query-shuffle-seed",
            "12345",
        ],
    )

    args = module.parse_args()

    assert args.query_order == "shuffled"
    assert args.query_shuffle_seed == 12345


def test_parse_args_supports_query_start_offset(monkeypatch):
    module = load_module()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "method4_claim_d_high_recall_latency.py",
            "--query-start-offset",
            "100",
        ],
    )

    args = module.parse_args()

    assert args.query_start_offset == 100


def test_parse_args_supports_skip_method4(monkeypatch):
    module = load_module()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "method4_claim_d_high_recall_latency.py",
            "--skip-method4",
        ],
    )

    args = module.parse_args()

    assert args.skip_method4 is True


def test_parse_args_supports_kmeans_latency_controls(monkeypatch):
    module = load_module()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "method4_claim_d_high_recall_latency.py",
            "--kmeans-collection",
            "bench095_cpp_kmeans_s31",
            "--kmeans-config",
            "kmeans_s31_u160_b80_f8=160,80,8",
            "--skip-kmeans",
            "--cpp-kmeans-train-size",
            "1234",
            "--kmeans-iters",
            "9",
        ],
    )

    args = module.parse_args()

    assert args.kmeans_collection == "bench095_cpp_kmeans_s31"
    assert args.kmeans_config == ["kmeans_s31_u160_b80_f8=160,80,8"]
    assert args.skip_kmeans is True
    assert args.cpp_kmeans_train_size == 1234
    assert args.kmeans_iters == 9


def test_parse_args_supports_simple_kmeans_latency_controls(monkeypatch):
    module = load_module()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "method4_claim_d_high_recall_latency.py",
            "--simple-kmeans-collection",
            "l2_sift100k_simple_s16_20260705",
            "--simple-kmeans-config",
            "simple_n4_ef32=4,32",
            "--skip-simple-kmeans",
        ],
    )

    args = module.parse_args()

    assert args.simple_kmeans_collection == "l2_sift100k_simple_s16_20260705"
    assert args.simple_kmeans_config == ["simple_n4_ef32=4,32"]
    assert args.skip_simple_kmeans is True


def test_query_order_indices_keeps_original_order_by_default():
    module = load_module()

    assert module.query_order_indices(
        total_queries=7,
        warmup_query_count=2,
        query_order="original",
        query_shuffle_seed=123,
    ) == [0, 1, 2, 3, 4, 5, 6]


def test_query_order_indices_shuffles_only_measured_queries_deterministically():
    module = load_module()

    first = module.query_order_indices(
        total_queries=8,
        warmup_query_count=3,
        query_order="shuffled",
        query_shuffle_seed=20260705,
    )
    second = module.query_order_indices(
        total_queries=8,
        warmup_query_count=3,
        query_order="shuffled",
        query_shuffle_seed=20260705,
    )

    assert first == second
    assert first[:3] == [0, 1, 2]
    assert sorted(first[3:]) == [3, 4, 5, 6, 7]
    assert first[3:] != [3, 4, 5, 6, 7]


def test_query_order_indices_rejects_warmup_larger_than_total():
    module = load_module()

    with pytest.raises(ValueError, match="warmup_query_count"):
        module.query_order_indices(
            total_queries=2,
            warmup_query_count=3,
            query_order="original",
            query_shuffle_seed=123,
        )


def test_load_queries_and_neighbors_preserves_euclid_vectors(tmp_path):
    module = load_module()
    hdf5_path = tmp_path / "toy-l2.hdf5"
    with h5py.File(hdf5_path, "w") as handle:
        handle.create_dataset(
            "train",
            data=np.array(
                [
                    [3.0, 4.0],
                    [6.0, 8.0],
                    [1.0, 2.0],
                    [2.0, 1.0],
                    [9.0, 0.0],
                    [0.0, 9.0],
                    [5.0, 5.0],
                    [7.0, 1.0],
                ],
                dtype=np.float32,
            ),
        )
        handle.create_dataset("test", data=np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32))
        handle.create_dataset("neighbors", data=np.array([[0, 2], [5, 0]], dtype=np.int32))

    class FakeQdrantTool:
        def global_upper_indices(self, train_count, sample_denominator, upper_sample_seed):
            return np.array([0, 2], dtype=np.int64)

        def normalize_rows(self, arr):
            raise AssertionError("Euclid path must not normalize vectors")

    args = SimpleNamespace(
        hdf5_path=str(hdf5_path),
        warmup_query_count=1,
        eval_query_count=1,
        top_k=2,
        sample_denominator=4,
        upper_sample_seed=100,
        vector_distance="euclid",
    )

    queries, upper_vectors, neighbors, train_count, dim = module.load_queries_and_neighbors(args, FakeQdrantTool())

    assert train_count == 8
    assert dim == 2
    assert queries.tolist() == [[3.0, 4.0], [0.0, 5.0]]
    assert upper_vectors.tolist() == [[3.0, 4.0], [1.0, 2.0]]
    assert neighbors.tolist() == [[0, 2], [5, 0]]


def test_load_queries_and_neighbors_honors_query_start_offset(tmp_path):
    module = load_module()
    hdf5_path = tmp_path / "toy-offset.hdf5"
    with h5py.File(hdf5_path, "w") as handle:
        handle.create_dataset(
            "train",
            data=np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [2.0, 0.0],
                ],
                dtype=np.float32,
            ),
        )
        handle.create_dataset(
            "test",
            data=np.array(
                [
                    [10.0, 0.0],
                    [11.0, 0.0],
                    [12.0, 0.0],
                    [13.0, 0.0],
                    [14.0, 0.0],
                    [15.0, 0.0],
                ],
                dtype=np.float32,
            ),
        )
        handle.create_dataset(
            "neighbors",
            data=np.array(
                [
                    [0, 1],
                    [1, 2],
                    [2, 3],
                    [3, 0],
                    [0, 2],
                    [1, 3],
                ],
                dtype=np.int32,
            ),
        )

    class FakeQdrantTool:
        def global_upper_indices(self, train_count, sample_denominator, upper_sample_seed):
            return np.array([0, 2], dtype=np.int64)

        def normalize_rows(self, arr):
            raise AssertionError("Euclid path must not normalize vectors")

    args = SimpleNamespace(
        hdf5_path=str(hdf5_path),
        query_start_offset=2,
        warmup_query_count=1,
        eval_query_count=2,
        top_k=2,
        sample_denominator=4,
        upper_sample_seed=100,
        vector_distance="euclid",
    )

    queries, _upper_vectors, neighbors, _train_count, _dim = module.load_queries_and_neighbors(
        args,
        FakeQdrantTool(),
    )

    assert queries.tolist() == [[12.0, 0.0], [13.0, 0.0], [14.0, 0.0]]
    assert neighbors.tolist() == [[2, 3], [3, 0], [0, 2]]


def test_run_plan_batches_uses_lower_scores_for_euclid():
    module = load_module()

    class FakeQdrantTool:
        def __init__(self):
            self.score_flags = []

        def execute_query_plans_once(
            self,
            base_url,
            collection,
            query_plans,
            neighbors,
            top_k,
            score_higher_is_better=True,
        ):
            self.score_flags.append(score_higher_is_better)
            return {
                "query_count": len(query_plans),
                "hits": 0,
                "visited_shards": 0,
                "assigned_ef_sum": 0,
                "search_batch_calls": 1,
            }

    args = SimpleNamespace(
        warmup_query_count=1,
        batch_size=2,
        base_url="http://example.test",
        top_k=10,
        score_higher_is_better=False,
    )
    q2l = FakeQdrantTool()

    module.run_plan_batches(
        args,
        q2l,
        method="Method4",
        config_label="l2",
        collection="collection",
        query_plans=[{"searches": []}, {"searches": []}, {"searches": []}],
        neighbors=np.zeros((3, 10), dtype=np.int64),
        repeat=1,
    )

    assert q2l.score_flags == [False, False]
