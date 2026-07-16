import importlib.util
from pathlib import Path


def load_module():
    module_path = Path("tools/method4_claim_cost_analysis.py")
    spec = importlib.util.spec_from_file_location("method4_claim_cost_analysis", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_normalize_build_row_uses_indexed_vectors_when_assigned_missing():
    cost = load_module()
    row = {
        "collection": "bench095_cpp_kmeans_s46",
        "points_count": "1183514",
        "indexed_vectors_count": "1183514",
        "segments_count": "92",
        "cluster_shard_count": "46",
        "cluster_peer_count": "4",
    }

    normalized = cost.normalize_build_row(row)

    assert normalized["logical_points_count"] == 1183514
    assert normalized["assigned_points_count"] == 1183514
    assert normalized["index_expansion_ratio"] == 1.0
    assert normalized["segments_count"] == 92


def test_storage_summary_separates_data_shards_from_controller_metadata():
    cost = load_module()
    rows = [
        {"container": "qdrant-controller-qdrant_shard_1-1", "collection": "c", "bytes": 100},
        {"container": "qdrant-controller-qdrant_shard_2-1", "collection": "c", "bytes": 200},
        {"container": "qdrant-controller-qdrant_shard_3-1", "collection": "c", "bytes": 300},
        {"container": "qdrant-controller-qdrant_controller-1", "collection": "c", "bytes": 7},
        {"container": "qdrant-controller-qdrant_shard_1-1", "collection": "other", "bytes": 999},
    ]

    summary = cost.storage_summary_for_collection(rows, "c")

    assert summary["storage_data_shards_bytes"] == 600
    assert summary["storage_total_with_controller_bytes"] == 607
    assert summary["storage_data_shards_gib"] == 600 / (1024**3)
    assert summary["storage_source"] == "docker_exec_du"


def test_build_cost_row_combines_build_live_storage_and_performance():
    cost = load_module()
    build = cost.normalize_build_row(
        {
            "collection": "orion",
            "points_count": "1400967",
            "indexed_vectors_count": "1400967",
            "logical_points_count": "1183514",
            "assigned_points_count": "1400967",
            "segments_count": "92",
            "cluster_shard_count": "46",
            "cluster_peer_count": "4",
        }
    )
    live = {"points_count": 1400967, "indexed_vectors_count": 1400967, "status": "green"}
    storage = {
        "storage_data_shards_bytes": 1024**3,
        "storage_total_with_controller_bytes": 1024**3 + 10,
        "storage_data_shards_gib": 1.0,
        "storage_source": "docker_exec_du",
    }
    perf = {
        "recall_at_k": "0.9554",
        "qps": "390.1",
        "avg_visited_shards": "23.6",
        "avg_search_requests_per_query": "1.0",
    }

    row = cost.build_cost_row(
        label="orion_current",
        family="Orion",
        strategy="default",
        collection="orion",
        build=build,
        live_info=live,
        storage=storage,
        performance=perf,
        build_source="builds.csv",
        performance_source="final_metrics.csv",
    )

    assert row["index_expansion_ratio"] == 1400967 / 1183514
    assert row["storage_data_shards_gib"] == 1.0
    assert row["live_status"] == "green"
    assert row["recall_at_10"] == 0.9554
    assert row["qps"] == 390.1
    assert row["build_time_sec"] == ""
    assert row["missing_cost_metrics"] == "build_time;memory_rss;request_bytes;controller_cpu"


def test_build_artifact_capabilities_distinguish_method4_snapshots_from_hnsw_timing():
    cost = load_module()

    method4_caps = cost.build_artifact_capabilities(
        Path("results/method4_example/20260705/builds.csv"),
        [
            "collection",
            "points_count",
            "indexed_vectors_count",
            "logical_points_count",
            "assigned_points_count",
            "cluster_shard_count",
        ],
    )
    hnsw_caps = cost.build_artifact_capabilities(
        Path("results/hnsw/20260413_091639/builds.csv"),
        [
            "collection_name",
            "create_secs",
            "upsert_secs",
            "wait_index_secs",
            "total_build_secs",
            "points_count",
        ],
    )

    assert method4_caps["artifact_family"] == "method4_collection_snapshot"
    assert method4_caps["has_build_wall_time"] is False
    assert method4_caps["method4_cost_applicability"] == "direct_count_snapshot_only"
    assert "no build duration" in method4_caps["conclusion"]
    assert hnsw_caps["artifact_family"] == "hnsw_baseline_build_timing"
    assert hnsw_caps["has_build_wall_time"] is True
    assert hnsw_caps["method4_cost_applicability"] == "not_method4_distributed"


def test_collect_build_stage_artifact_audit_groups_schemas(tmp_path):
    cost = load_module()
    method4_dir = tmp_path / "results" / "method4_run" / "analysis"
    method4_dir.mkdir(parents=True)
    (method4_dir / "builds.csv").write_text(
        "collection,points_count,indexed_vectors_count,cluster_shard_count\n"
        "m4,10,10,46\n"
    )
    hnsw_dir = tmp_path / "results" / "hnsw" / "run"
    hnsw_dir.mkdir(parents=True)
    (hnsw_dir / "builds.csv").write_text(
        "collection_name,total_build_secs,points_count,indexed_vectors_count\n"
        "h,1.5,10,10\n"
    )

    rows = cost.collect_build_stage_artifact_audit(tmp_path / "results")

    by_family = {row["artifact_family"]: row for row in rows}
    assert by_family["method4_collection_snapshot"]["file_count"] == 1
    assert by_family["method4_collection_snapshot"]["has_build_wall_time"] is False
    assert by_family["hnsw_baseline_build_timing"]["file_count"] == 1
    assert by_family["hnsw_baseline_build_timing"]["has_build_wall_time"] is True


def test_build_source_audit_rows_mark_method4_cost_sources_unsafe_for_build_time(tmp_path):
    cost = load_module()
    build_path = tmp_path / "results" / "method4" / "builds.csv"
    build_path.parent.mkdir(parents=True)
    build_path.write_text(
        "collection,points_count,indexed_vectors_count,logical_points_count,assigned_points_count,cluster_shard_count\n"
        "m4,10,10,10,12,46\n"
    )
    cases = [
        {
            "label": "case",
            "family": "Orion",
            "strategy": "default",
            "collection": "m4",
            "build_path": str(build_path),
            "performance_path": "unused.csv",
        }
    ]

    rows = cost.build_source_audit_rows(cases)

    assert rows == [
        {
            "label": "case",
            "family": "Orion",
            "strategy": "default",
            "collection": "m4",
            "build_source": str(build_path),
            "build_source_schema": "collection;points_count;indexed_vectors_count;logical_points_count;assigned_points_count;cluster_shard_count",
            "artifact_family": "method4_collection_snapshot",
            "has_build_wall_time": False,
            "has_indexing_time": False,
            "has_build_rss": False,
            "safe_to_fill_cost_build_time": False,
            "safe_to_fill_cost_build_rss": False,
            "notes": "Method4/Qdrant collection snapshot: indexed-vector, assignment, shard, and peer counts are usable, but there is no build duration or build-stage RSS.",
        }
    ]
