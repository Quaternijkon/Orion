"""Score Recall@k offline from persisted result ids.

Recall is deliberately absent from the measured request path: the previous
harness computed it per query inside the timer, which inflated client cost and
contributed to the client-bound artifact documented in ../STAGE0.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_run(output_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate per-worker id and offset arrays, ordered by query offset."""
    id_files = sorted(output_dir.glob("ids_worker*.npy"))
    offset_files = sorted(output_dir.glob("offsets_worker*.npy"))
    if not id_files or len(id_files) != len(offset_files):
        raise FileNotFoundError(f"no complete worker output in {output_dir}")
    ids = np.concatenate([np.load(path) for path in id_files])
    offsets = np.concatenate([np.load(path) for path in offset_files])
    order = np.argsort(offsets, kind="stable")
    return ids[order], offsets[order]


def recall_at_k(
    result_ids: np.ndarray, ground_truth: np.ndarray, top_k: int
) -> np.ndarray:
    """Per-query Recall@k against the first ``top_k`` true neighbors."""
    if result_ids.shape[0] != ground_truth.shape[0]:
        raise ValueError("result and ground-truth row counts differ")
    truth = ground_truth[:, :top_k]
    per_query = np.empty(result_ids.shape[0], dtype=np.float64)
    for row in range(result_ids.shape[0]):
        found = np.intersect1d(result_ids[row, :top_k], truth[row], assume_unique=False)
        per_query[row] = found.size / top_k
    return per_query


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="directory from measure.py")
    parser.add_argument("--ground-truth", required=True, help="npy of true neighbors")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    result_ids, offsets = load_run(output_dir)
    ground_truth = np.load(args.ground_truth, mmap_mode="r")
    per_query = recall_at_k(result_ids, np.asarray(ground_truth[offsets]), args.top_k)

    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["recall"] = {
        "mean": float(per_query.mean()),
        "p5": float(np.percentile(per_query, 5)),
        "queries_below_0_90": int((per_query < 0.90).sum()),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.save(output_dir / "recall_per_query.npy", per_query)

    print(
        f"queries={per_query.size} "
        f"mean_recall@{args.top_k}={per_query.mean():.4f} "
        f"qps={summary.get('qps', float('nan')):.1f} "
        f"gates_passed={summary.get('gates_passed')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
