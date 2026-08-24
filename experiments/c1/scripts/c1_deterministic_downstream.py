#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Sequence

from c1_protocol import sha256_path


DATASETS = ("sift1m", "glove-200-angular")
METHODS = ("random", "kmeans")
LOGICAL_SHARDS = (1, 2, 4, 8, 16, 32)
PEERS = ("10.10.1.1", "10.10.1.2", "10.10.1.3", "10.10.1.4")
E2_STEM = "stage12-e2-deterministic"
E3_STEM = "stage12-e3-deterministic"
E4_SIFT_PRELIMINARY_STEM = "stage12-e4-deterministic-sift1m-preliminary"
E4_GLOVE_STEM = "stage12-e4-deterministic-glove-200-angular"
E4_SIFT_FINAL_STEM = "stage12-e4-deterministic-sift1m-final"
FINAL_STEM = "stage12-c1-deterministic-final"


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def run_command(command: list[str], *, cwd: Path) -> None:
    print("[c1-downstream]", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def copy_preserving(source: Path, destination: Path) -> None:
    if destination.is_file():
        if sha256_path(destination) != sha256_path(source):
            raise ValueError(f"preserved artifact conflicts with source: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def preserve_stage11_outputs(root: Path) -> dict[str, str]:
    runs = root / "runs"
    figures = root / "figures"
    run_history = runs / "history" / "stage11-c1-report-repair"
    figure_history = figures / "history" / "stage11-c1-report-repair"
    sources = [
        runs / name
        for name in (
            "c1_physical_scaleout.csv",
            "c1_fanout_summary.csv",
            "c1_local_work_summary.csv",
            "c1_aggregate_work_summary.csv",
            "c1_model_accuracy.csv",
            "c1_recall_sensitivity.csv",
            "c1_physical_scale_table.tex",
            "stage11-c1-report-repair-summary.json",
            "stage11-c1-report-repair-record.json",
        )
    ]
    sources.extend(sorted(figures.glob("*.pdf")))
    outputs: dict[str, str] = {}
    final_record_exists = (runs / f"{FINAL_STEM}-record.json").is_file()
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
        destination_root = figure_history if source.suffix == ".pdf" else run_history
        destination = destination_root / source.name
        if destination.is_file():
            outputs[str(destination.resolve())] = sha256_path(destination)
            continue
        if final_record_exists:
            raise FileNotFoundError(
                f"Stage 11 preserved artifact is missing after finalization: {destination}"
            )
        copy_preserving(source, destination)
        outputs[str(destination.resolve())] = sha256_path(destination)
    return outputs


def manifest_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def ensure_manifest_record(path: Path, record: dict[str, Any]) -> None:
    matches = [
        item
        for item in manifest_records(path)
        if item.get("experiment_id") == record.get("experiment_id")
    ]
    if matches:
        if len(matches) == 1 and matches[0] == record:
            return
        raise ValueError(
            f"manifest contains conflicting records for {record.get('experiment_id')}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def validate_e3_matrix(runs: Path, graph_build_seed: int) -> None:
    matrix = load_json(runs / f"{E3_STEM}-matrix-record.json")
    if (
        matrix.get("status") != "COMPLETE"
        or int(matrix.get("configuration_count") or 0) != 24
        or int(matrix.get("graph_build_seed") or -1) != graph_build_seed
        or int(matrix.get("max_indexing_threads") or 0) != 1
        or int(matrix.get("max_optimization_threads") or 0) != 1
    ):
        raise ValueError("deterministic E3 matrix record is not complete")
    for dataset in DATASETS:
        for method in METHODS:
            for shards in LOGICAL_SHARDS:
                stem = f"{E3_STEM}-{dataset}-{method}-m{shards}"
                summary_path = runs / f"{stem}-summary.json"
                tuning_path = runs / f"{stem}-tuning-pinned.json"
                per_query_path = runs / "per_query" / f"{stem}.csv"
                summary = load_json(summary_path)
                if (
                    summary.get("result_status") != "VALID_E3"
                    or float(summary.get("measurement_recall") or 0.0) < 0.90
                    or summary.get("deterministic_graph_construction") is not True
                    or int(summary.get("graph_build_seed") or -1)
                    != graph_build_seed
                    or int(summary.get("hnsw_max_indexing_threads") or 0) != 1
                    or int(summary.get("max_optimization_threads") or 0) != 1
                    or not tuning_path.is_file()
                    or not per_query_path.is_file()
                ):
                    raise ValueError(f"invalid deterministic E3 configuration: {stem}")


def valid_e2_record(path: Path, dataset: str, graph_build_seed: int) -> bool:
    if not path.is_file():
        return False
    record = load_json(path)
    return (
        record.get("record_type") == "e2_fanout_analysis"
        and record.get("dataset") == dataset
        and record.get("status") == "VALID_E2"
        and int(record.get("configuration_count") or 0) == 12
        and record.get("deterministic_graph_construction") is True
        and int(record.get("graph_build_seed") or -1) == graph_build_seed
    )


def run_e2(
    *,
    dataset: str,
    dataset_path: Path,
    repo: Path,
    root: Path,
    partition_root: Path,
    python: str,
    graph_build_seed: int,
) -> Path:
    runs = root / "runs"
    manifest = runs / "manifest.jsonl"
    stem = f"{E2_STEM}-{dataset}"
    record_path = runs / f"{stem}-fanout-record.json"
    if valid_e2_record(record_path, dataset, graph_build_seed):
        ensure_manifest_record(manifest, load_json(record_path))
        print(f"[c1-downstream] resume skip {stem}: verified VALID_E2", flush=True)
        return record_path
    run_command(
        [
            python,
            str(root / "scripts" / "c1_e2_fanout.py"),
            "--dataset",
            dataset,
            "--hdf5-path",
            str(dataset_path),
            "--partition-root",
            str(partition_root / dataset),
            "--runs-root",
            str(runs),
            "--e3-source-stem",
            f"{E3_STEM}-{{dataset}}",
            "--tuning-small-source-stem",
            f"{E3_STEM}-{{dataset}}",
            "--tuning-large-source-stem",
            f"{E3_STEM}-{{dataset}}",
            "--per-query-output",
            str(runs / "per_query" / f"{stem}-fanout.csv"),
            "--summary-csv-output",
            str(runs / f"{stem}-c1_fanout_summary.csv"),
            "--summary-json-output",
            str(runs / f"{stem}-fanout-summary.json"),
            "--record-output",
            str(record_path),
            "--experiment-id",
            f"{stem}-fanout",
            "--manifest-jsonl",
            str(manifest),
            "--figures-dir",
            str(root / "figures" / "history" / stem),
        ],
        cwd=repo,
    )
    if not valid_e2_record(record_path, dataset, graph_build_seed):
        raise RuntimeError(f"deterministic E2 validation failed: {record_path}")
    return record_path


def valid_e4_record(
    path: Path,
    dataset: str,
    graph_build_seed: int,
    *,
    external_gate: Path | None,
    supersedes: str | None,
) -> bool:
    if not path.is_file():
        return False
    record = load_json(path)
    correction_valid = (
        not supersedes
        or record.get("supersedes_experiment_id") == supersedes
    )
    gates = list(record.get("external_projection_gate_records") or [])
    if external_gate is None:
        gate_valid = not gates
    else:
        external_record = load_json(external_gate)
        gate_valid = (
            len(gates) == 1
            and gates[0].get("experiment_id") == external_record.get("experiment_id")
            and gates[0].get("source_record_sha256") == sha256_path(external_gate)
        )
    return (
        record.get("record_type") == "e4_aggregate_work_decomposition"
        and record.get("dataset") == dataset
        and str(record.get("status") or "").startswith("VALID_E4")
        and int(record.get("configuration_count") or 0) == 12
        and record.get("deterministic_graph_construction") is True
        and int(record.get("graph_build_seed") or -1) == graph_build_seed
        and int(record.get("hnsw_max_indexing_threads") or 0) == 1
        and int(record.get("max_optimization_threads") or 0) == 1
        and record.get("e2_source_stem") == E2_STEM
        and record.get("e3_source_stem") == E3_STEM
        and gate_valid
        and correction_valid
    )


def run_e4(
    *,
    dataset: str,
    stem: str,
    e1_source_stem: str,
    physical_e1_summary: Path,
    external_gate: Path | None,
    supersedes_experiment_id: str | None,
    correction_reason: str | None,
    repo: Path,
    root: Path,
    python: str,
    graph_build_seed: int,
) -> Path:
    runs = root / "runs"
    manifest = runs / "manifest.jsonl"
    record_path = runs / f"{stem}-aggregate-work-record.json"
    if valid_e4_record(
        record_path,
        dataset,
        graph_build_seed,
        external_gate=external_gate,
        supersedes=supersedes_experiment_id,
    ):
        ensure_manifest_record(manifest, load_json(record_path))
        print(f"[c1-downstream] resume skip {stem}: verified VALID_E4", flush=True)
        return record_path
    command = [
        python,
        str(root / "scripts" / "c1_e4_decomposition.py"),
        "--dataset",
        dataset,
        "--runs-root",
        str(runs),
        "--e1-source-stem",
        e1_source_stem,
        "--e2-source-stem",
        E2_STEM,
        "--e3-source-stem",
        E3_STEM,
        "--experiment-id",
        f"{stem}-aggregate-work",
        "--e2-per-query",
        str(runs / "per_query" / f"{E2_STEM}-{dataset}-fanout.csv"),
        "--physical-e1-summary",
        str(physical_e1_summary),
        "--per-query-output",
        str(runs / "per_query" / f"{stem}-aggregate-work.csv"),
        "--summary-csv-output",
        str(runs / f"{stem}-c1_aggregate_work_summary.csv"),
        "--model-accuracy-output",
        str(runs / f"{stem}-c1_model_accuracy.csv"),
        "--summary-json-output",
        str(runs / f"{stem}-aggregate-work-summary.json"),
        "--record-output",
        str(record_path),
        "--manifest-jsonl",
        str(manifest),
        "--figures-dir",
        str(root / "figures" / "history" / stem),
    ]
    if external_gate is not None:
        command.extend(["--external-projection-gate-record", str(external_gate)])
    if supersedes_experiment_id is not None and correction_reason is not None:
        command.extend(
            [
                "--supersedes-experiment-id",
                supersedes_experiment_id,
                "--correction-reason",
                correction_reason,
            ]
        )
    run_command(command, cwd=repo)
    if not valid_e4_record(
        record_path,
        dataset,
        graph_build_seed,
        external_gate=external_gate,
        supersedes=supersedes_experiment_id,
    ):
        raise RuntimeError(f"deterministic E4 validation failed: {record_path}")
    return record_path


def http_status(peer: str, collection: str) -> int:
    encoded = urllib.parse.quote(collection, safe="")
    request = urllib.request.Request(
        f"http://{peer}:6333/collections/{encoded}", method="GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=15.0) as response:
            return int(response.status)
    except urllib.error.HTTPError as error:
        return int(error.code)


def write_final_cleanup_proof(runs: Path) -> Path:
    source_path = runs / f"{E3_STEM}-glove-200-angular-kmeans-m32-cleanup.json"
    source = load_json(source_path)
    collection = str(source["collection"])
    statuses = {peer: http_status(peer, collection) for peer in PEERS}
    storage_path = Path(source["controller_storage_path"])
    if any(status != 404 for status in statuses.values()) or storage_path.exists():
        raise RuntimeError(
            f"final collection still exists: peer_status={statuses}, "
            f"storage_exists={storage_path.exists()}"
        )
    proof = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "record_type": "c1_final_collection_cleanup_proof",
        "collection": collection,
        "peers": {peer: {"http_status": status} for peer, status in statuses.items()},
        "controller_storage_path": str(storage_path),
        "controller_storage_absent": True,
        "source_cleanup_record": str(source_path.resolve()),
        "source_cleanup_record_sha256": sha256_path(source_path),
        "status": "VERIFIED_DELETED",
    }
    output = runs / "c1_final_cleanup-proof.json"
    if output.is_file():
        existing = load_json(output)
        stable_fields = set(proof) - {"timestamp"}
        if all(existing.get(field) == proof[field] for field in stable_fields):
            return output
    write_json_atomic(output, proof)
    return output


def complete_section31_proof(
    *,
    runs: Path,
    graph_build_seed: int,
    e2_records: Sequence[Path],
    e4_records: Sequence[Path],
    metadata_path: Path,
    cleanup_path: Path,
) -> Path:
    proof_path = runs / "section31-deterministic-build-proof.json"
    proof = load_json(proof_path)
    if (
        proof.get("deterministic_construction_verified") is not True
        or int(proof.get("graph_build_seed") or -1) != graph_build_seed
        or int(proof.get("max_indexing_threads") or 0) != 1
        or int(proof.get("max_optimization_threads") or 0) != 1
        or int(proof.get("identical_build_count") or 0) < 2
    ):
        raise ValueError("base Section 31 deterministic-build proof is invalid")
    e3_hashes = {
        f"{dataset}/{method}/m{shards}": sha256_path(
            runs / f"{E3_STEM}-{dataset}-{method}-m{shards}-summary.json"
        )
        for dataset in DATASETS
        for method in METHODS
        for shards in LOGICAL_SHARDS
    }
    evidence = {
        "e2_records": {str(path.resolve()): sha256_path(path) for path in e2_records},
        "e3_matrix_record": {
            str((runs / f"{E3_STEM}-matrix-record.json").resolve()): sha256_path(
                runs / f"{E3_STEM}-matrix-record.json"
            )
        },
        "e3_configuration_summaries": e3_hashes,
        "e4_records": {str(path.resolve()): sha256_path(path) for path in e4_records},
        "normalized_metadata": {str(metadata_path.resolve()): sha256_path(metadata_path)},
        "final_cleanup": {str(cleanup_path.resolve()): sha256_path(cleanup_path)},
    }
    if proof.get("authoritative_e2_e4_rerun_complete") is True:
        if proof.get("authoritative_rerun_evidence") != evidence:
            raise ValueError("completed Section 31 proof has different evidence")
        return proof_path
    backup = runs / "section31-deterministic-build-proof-pre-rerun.json"
    if backup.is_file():
        if load_json(backup) != proof:
            raise ValueError("Section 31 pre-rerun backup conflicts with current proof")
    else:
        write_json_atomic(backup, proof)
    proof["authoritative_e2_e4_rerun_complete"] = True
    proof["authoritative_rerun_completed_at"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )
    proof["authoritative_rerun_evidence"] = evidence
    proof["status"] = "VERIFIED_AUTHORITATIVE_RERUN_COMPLETE"
    write_json_atomic(proof_path, proof)
    return proof_path


def valid_final_record(path: Path, graph_build_seed: int, section31_path: Path) -> bool:
    if not path.is_file():
        return False
    record = load_json(path)
    mechanism_checks = record.get("mechanism_checks") or {}
    slow_local_supported = mechanism_checks.get("slow_local_work_supported")
    if not isinstance(slow_local_supported, bool):
        return False
    expected_c1_c_status = (
        "SUPPORTED" if slow_local_supported else "CONTRADICTED"
    )
    return (
        record.get("c1_status") == "INSUFFICIENT"
        and record.get("subclaim_status", {}).get("c1_b_high_logical_shard_fanout")
        == "CONTRADICTED"
        and record.get("subclaim_status", {}).get("c1_c_slow_local_work_decrease")
        == expected_c1_c_status
        and record.get("observed_mechanism_status", {}).get(
            "slow_local_work_decrease"
        )
        == expected_c1_c_status
        and record.get("section31_index_construction_status")
        == "VERIFIED_DETERMINISTIC_SINGLE_BUILD_AUTHORITATIVE_RERUN"
        and int(record.get("graph_build_seed") or -1) == graph_build_seed
        and record.get("section31_proof_sha256") == sha256_path(section31_path)
        and record.get("authoritative_source_stems", {}).get("e2") == E2_STEM
        and record.get("authoritative_source_stems", {}).get("e3") == E3_STEM
    )


def execute(args: argparse.Namespace) -> int:
    repo = Path(args.repo).expanduser().resolve()
    root = Path(args.root).expanduser().resolve()
    runs = root / "runs"
    partition_root = Path(args.partition_root).expanduser().resolve()
    python = str(Path(args.python).expanduser().absolute())
    topology = load_json(Path(args.topology).expanduser().resolve())
    graph_build_seed = int(topology["hnsw_graph_build_seed"])
    datasets = {
        dataset: Path(load_json(runs / f"{dataset}.dataset.json")["path"])
        for dataset in DATASETS
    }
    validate_e3_matrix(runs, graph_build_seed)

    e2_records = [
        run_e2(
            dataset=dataset,
            dataset_path=datasets[dataset],
            repo=repo,
            root=root,
            partition_root=partition_root,
            python=python,
            graph_build_seed=graph_build_seed,
        )
        for dataset in DATASETS
    ]

    sift_preliminary = run_e4(
        dataset="sift1m",
        stem=E4_SIFT_PRELIMINARY_STEM,
        e1_source_stem="stage6-e1-corrected",
        physical_e1_summary=(
            runs / "stage6-e1-corrected-sift1m-physical-scaleout-summary.json"
        ),
        external_gate=None,
        supersedes_experiment_id=None,
        correction_reason=None,
        repo=repo,
        root=root,
        python=python,
        graph_build_seed=graph_build_seed,
    )
    glove = run_e4(
        dataset="glove-200-angular",
        stem=E4_GLOVE_STEM,
        e1_source_stem="stage8-e1-corrected",
        physical_e1_summary=(
            runs
            / "stage8-e1-corrected-glove-200-angular-physical-scaleout-summary.json"
        ),
        external_gate=sift_preliminary,
        supersedes_experiment_id=None,
        correction_reason=None,
        repo=repo,
        root=root,
        python=python,
        graph_build_seed=graph_build_seed,
    )
    preliminary_id = load_json(sift_preliminary)["experiment_id"]
    sift_final = run_e4(
        dataset="sift1m",
        stem=E4_SIFT_FINAL_STEM,
        e1_source_stem="stage6-e1-corrected",
        physical_e1_summary=(
            runs / "stage6-e1-corrected-sift1m-physical-scaleout-summary.json"
        ),
        external_gate=glove,
        supersedes_experiment_id=str(preliminary_id),
        correction_reason="cross_dataset_gate_with_deterministic_glove_evidence",
        repo=repo,
        root=root,
        python=python,
        graph_build_seed=graph_build_seed,
    )

    metadata_path = runs / "c1_run_metadata.jsonl"
    run_command(
        [
            python,
            str(root / "scripts" / "c1_normalize_metadata.py"),
            "--repo",
            str(repo),
            "--root",
            str(root),
            "--e3-source-stem",
            E3_STEM,
            "--output",
            str(metadata_path),
        ],
        cwd=repo,
    )
    cleanup_path = write_final_cleanup_proof(runs)
    section31_path = complete_section31_proof(
        runs=runs,
        graph_build_seed=graph_build_seed,
        e2_records=e2_records,
        e4_records=(sift_preliminary, glove, sift_final),
        metadata_path=metadata_path,
        cleanup_path=cleanup_path,
    )

    preserved_stage11 = preserve_stage11_outputs(root)
    final_record_path = runs / f"{FINAL_STEM}-record.json"
    if valid_final_record(final_record_path, graph_build_seed, section31_path):
        ensure_manifest_record(runs / "manifest.jsonl", load_json(final_record_path))
        print(f"[c1-downstream] resume skip {FINAL_STEM}: verified final record", flush=True)
    else:
        run_command(
            [
                python,
                str(root / "scripts" / "c1_finalize.py"),
                "--root",
                str(root),
            ],
            cwd=repo,
        )
    if not valid_final_record(final_record_path, graph_build_seed, section31_path):
        raise RuntimeError(f"Stage 12 final validation failed: {final_record_path}")
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "record_type": "c1_deterministic_downstream_completion",
        "graph_build_seed": graph_build_seed,
        "e2_records": [str(path) for path in e2_records],
        "e4_records": [str(path) for path in (sift_preliminary, glove, sift_final)],
        "section31_proof": str(section31_path),
        "metadata": str(metadata_path),
        "cleanup_proof": str(cleanup_path),
        "preserved_stage11_artifacts": preserved_stage11,
        "final_record": str(final_record_path),
        "status": "COMPLETE",
    }
    write_json_atomic(runs / "stage12-deterministic-downstream-record.json", output)
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and validate deterministic C1 E2/E4/finalization"
    )
    parser.add_argument("--repo", default="/users/dry/Orion")
    parser.add_argument("--root", default="experiments/c1")
    parser.add_argument("--topology", default="experiments/c1/topology-amd-4node.json")
    parser.add_argument(
        "--partition-root",
        default="/users/dry/orion-distributed/c1-20260820-v1/partitions",
    )
    parser.add_argument("--python", default="/users/dry/orion-distributed/venv/bin/python")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    return execute(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
