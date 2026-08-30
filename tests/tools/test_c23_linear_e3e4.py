from __future__ import annotations

from experiments.c23.scripts import c23_linear_e3e4 as experiment


def collection_info(*, points: int, indexed: int) -> dict[str, object]:
    return {
        "points_count": points,
        "indexed_vectors_count": indexed,
        "segments_count": 2,
        "status": "green",
        "optimizer_status": "ok",
    }


def segment_telemetry(*, threshold: int, indexed: int, plain: int) -> dict[str, object]:
    return {
        "collection_id": experiment.COLLECTION,
        "indexing_threshold_kib": threshold,
        "segments": [
            {
                "shard_id": 0,
                "segment_type": "plain",
                "num_vectors": plain,
                "num_indexed_vectors": 0,
            },
            {
                "shard_id": 0,
                "segment_type": "indexed",
                "num_vectors": indexed,
                "num_indexed_vectors": indexed,
            },
        ],
    }


def test_segment_layout_helpers_distinguish_plain_tail() -> None:
    tail = segment_telemetry(threshold=10, indexed=90, plain=10)
    complete = segment_telemetry(threshold=10, indexed=100, plain=0)

    assert experiment.plain_tail_layout_matches(tail, 100)
    assert not experiment.single_hnsw_layout_matches(tail, 100)
    assert experiment.single_hnsw_layout_matches(complete, 100)
    assert not experiment.plain_tail_layout_matches(complete, 100)


def test_wait_indexed_compacts_plain_tail_and_restores_threshold(monkeypatch) -> None:
    state = {"threshold": 10, "compacted": False}
    updates: list[int] = []
    ticks = iter(range(1000))

    def fake_collection_info(_base_url: str) -> dict[str, object]:
        indexed = 100 if state["compacted"] else 90
        return collection_info(points=100, indexed=indexed)

    def fake_segment_telemetry(_base_url: str) -> dict[str, object]:
        indexed = 100 if state["compacted"] else 90
        plain = 0 if state["compacted"] else 10
        return segment_telemetry(
            threshold=int(state["threshold"]), indexed=indexed, plain=plain
        )

    def fake_update(_base_url: str, threshold_kib: int) -> None:
        updates.append(threshold_kib)
        state["threshold"] = threshold_kib
        if threshold_kib == experiment.TAIL_COMPACTION_THRESHOLD_KIB:
            state["compacted"] = True

    monkeypatch.setattr(experiment, "collection_info", fake_collection_info)
    monkeypatch.setattr(
        experiment, "collection_segment_telemetry", fake_segment_telemetry
    )
    monkeypatch.setattr(experiment, "update_indexing_threshold", fake_update)
    monkeypatch.setattr(experiment.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(experiment.time, "sleep", lambda _seconds: None)

    result = experiment.wait_indexed(
        [{"base_url": "http://shard-0", "shard_id": 0}],
        [100],
        timeout=100,
    )

    assert result["status"] == "PASS"
    assert result["tail_compaction"]["status"] == "PASS"
    assert result["tail_compaction"]["affected_shards"] == [0]
    assert updates == [
        experiment.TAIL_COMPACTION_THRESHOLD_KIB,
        experiment.INDEXING_THRESHOLD_KIB,
    ]
    assert result["segment_telemetry"][0]["indexing_threshold_kib"] == 10
    assert experiment.single_hnsw_layout_matches(result["segment_telemetry"][0], 100)
