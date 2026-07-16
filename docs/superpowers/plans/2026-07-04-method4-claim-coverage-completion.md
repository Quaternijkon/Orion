# Method4 Claim Coverage Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the experiments that are listed in `docs/experiments/2026-06-16-method4-distributed-submission-experiment-plan.md` but not yet backed by verified artifacts, then consolidate all claim-supporting data into one evidence document.

**Architecture:** Treat the June 16 experiment plan as the authoritative claim specification. Build a coverage table by matching each listed claim and required experiment shape against current `docs/experiments`, `results`, and `tools` artifacts; rerun only missing or unverifiable experiments, and clearly downgrade claims whose planned evidence remains unavailable or too weak.

**Tech Stack:** Python 3, existing Qdrant benchmark scripts under `tools/`, CSV/JSON result manifests under `results/`, Markdown evidence docs under `docs/experiments/`.

---

### Task 1: Extract Claim Requirements

**Files:**
- Read: `docs/experiments/2026-06-16-method4-distributed-submission-experiment-plan.md`
- Create: `results/method4_claim_coverage_20260704/claim_requirements.csv`
- Create: `results/method4_claim_coverage_20260704/source_manifest.json`

- [ ] **Step 1: Parse claim sections from the plan**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
import re
text = Path("docs/experiments/2026-06-16-method4-distributed-submission-experiment-plan.md").read_text()
for match in re.finditer(r"^### Claim ([A-Z]): (.+)$", text, flags=re.M):
    print(match.group(1), match.group(2))
PY
```

Expected: print Claim A through at least Claim G with their Chinese titles.

- [ ] **Step 2: Create a machine-readable requirements CSV**

Record one row per claim with these fields:

```text
claim_id,title,required_methods,required_parameter_shape,required_metrics,completion_standard,notes
```

Populate from the plan text, preserving exact method names such as `Orion / Full Method4`, `Simple KMeans nprobe`, `Naive all-shards`, `Fixed EF`, `Dynamic EF`, `worker-local peer pre-merge`, and `method4_aware`.

- [ ] **Step 3: Save a manifest**

Write `source_manifest.json` with:

```json
{
  "created_utc": "2026-07-04T00:00:00Z",
  "plan_document": "docs/experiments/2026-06-16-method4-distributed-submission-experiment-plan.md",
  "claim_requirements_csv": "results/method4_claim_coverage_20260704/claim_requirements.csv"
}
```

### Task 2: Audit Current Evidence Coverage

**Files:**
- Read: `docs/experiments/*.md`
- Read: `results/**/summary.json`
- Read: `results/**/*.csv`
- Create: `results/method4_claim_coverage_20260704/claim_evidence_audit.csv`

- [ ] **Step 1: List candidate evidence artifacts**

Run:

```bash
rg -n "Claim [A-Z]|Recall@10|QPS|P95|P99|avg visited|visited shards|dynamic EF|pre-merge|method4-aware|multi-assignment|KMeans|naive" docs/experiments results \
  -g '*.md' -g '*.csv' -g '*.json' \
  -g '!*.bin' -g '!*.idx' -g '!*.npy' -g '!*.npz'
```

Expected: find existing evidence docs for Claim C and Claim G, plus older Method4-vs-naive, multi-assignment, pre-merge, physical trace, and batch sweep documents.

- [ ] **Step 2: Classify each claim**

For each claim, set:

```text
status in {complete, partial, missing, invalid}
```

Use `complete` only when raw result files exist, row counts match the planned experiment shape or a documented strict substitute, and metrics meet the claim's support criteria.

- [ ] **Step 3: Record missing experiments**

For every `partial` or `missing` claim, record the exact missing comparison in `claim_evidence_audit.csv`, including the required method family, target recall, batch size, repeats, metrics, and the preferred existing harness to run.

### Task 3: Fill Gaps With Existing Raw Data

**Files:**
- Read: `results/**/*`
- Modify: `results/method4_claim_coverage_20260704/claim_evidence_audit.csv`
- Create: `results/method4_claim_coverage_20260704/derived_claim_tables/`

- [ ] **Step 1: Derive tables from existing runs**

For each missing item that can be satisfied by current raw CSV/JSON, generate a concise derived CSV in `derived_claim_tables/`. Examples:

```text
claim_d_method4_vs_naive_same_recall.csv
claim_e_premerge_ablation.csv
claim_f_latency_batch_sweep.csv
claim_b_multiassign_frontier.csv
```

- [ ] **Step 2: Preserve raw provenance**

Each derived table must include a `source_path` column pointing to the raw CSV or JSON row used.

- [ ] **Step 3: Update the audit**

Change a claim from `partial` to `complete` only if the derived table fully covers the planned comparison and the source files pass existence checks.

### Task 4: Run Only Truly Missing Experiments

**Files:**
- Read: `tools/benchmark_configs/*.json`
- Read: `tools/method4_benchmark_matrix.py`
- Read: `tools/qdrant_two_level_routing_experiment.py`
- Create: `results/method4_claim_coverage_20260704/new_runs/`

- [ ] **Step 1: Check available disk and running jobs**

Run:

```bash
df -h / /home
ps -eo pid,args | rg 'method4|qdrant_two_level_routing_experiment|method4_benchmark_matrix' | rg -v 'rg|ps -eo' || true
```

Expected: no conflicting benchmark job; enough disk for any planned run.

- [ ] **Step 2: Prefer light confirmation runs**

For missing experiments that need online Qdrant execution, first run 500-query tuning or smoke configurations using existing collections. Only run 3000-query formal evaluations after the tuning row reaches the planned recall neighborhood.

- [ ] **Step 3: Save run manifests**

Every new run directory must include a manifest or summary with collection, routing mode, query count, batch size, repeats, target recall, commit/image if available, and result CSV paths.

### Task 5: Consolidate All Claim Evidence

**Files:**
- Create: `docs/experiments/2026-07-04-method4-submission-claim-evidence-summary.md`
- Read: `results/method4_claim_coverage_20260704/claim_evidence_audit.csv`
- Read: `results/method4_claim_coverage_20260704/derived_claim_tables/*.csv`
- Read: existing claim evidence docs under `docs/experiments/`

- [ ] **Step 1: Write one section per claim**

For each claim, include:

```text
Claim statement
Experiment status
Primary evidence table
Raw artifact paths
Supported wording
Boundary / avoid-claim wording
```

- [ ] **Step 2: Include one cross-claim summary table**

The table columns must be:

```text
Claim, Status, Primary contrast, Recall level, QPS/latency result, Work/load result, Artifact
```

- [ ] **Step 3: Keep unsupported planned experiments visible**

If a listed experiment remains incomplete, include it under `Remaining gaps` with the reason it is not being used as claim support.

### Task 6: Verify Evidence Integrity

**Files:**
- Read: all artifacts referenced by `docs/experiments/2026-07-04-method4-submission-claim-evidence-summary.md`

- [ ] **Step 1: Compile Python helpers**

Run:

```bash
python3 -m py_compile tools/method4_claim_c_relevance_analysis.py tools/method4_claim_g_placement_analysis.py tools/method4_claim_g_online_latency.py tools/method4_claim_g_deploy_matched_layout.py tools/method4_claim_g_batch_latency.py
```

Expected: exit code 0.

- [ ] **Step 2: Check referenced paths**

Run a Python verifier that extracts Markdown inline paths beginning with `docs/`, `results/`, or `tools/` from the final summary and fails if any path does not exist.

- [ ] **Step 3: Check formatting and hygiene**

Run:

```bash
git diff --check
ps -eo pid,args | rg 'method4_claim|qdrant_two_level_routing_experiment|method4_benchmark_matrix' | rg -v 'rg|ps -eo' || true
df -h / /home
```

Expected: no whitespace errors, no stray experiment process, disk state recorded.
