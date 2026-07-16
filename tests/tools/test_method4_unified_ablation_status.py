import importlib.util
from pathlib import Path


def load_module():
    module_path = Path("tools/method4_unified_ablation_status.py")
    spec = importlib.util.spec_from_file_location("method4_unified_ablation_status", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_default_rows_mark_available_partial_and_missing_cells():
    module = load_module()

    rows = module.default_status_rows()
    by_group = {row["ablation_group"]: row for row in rows}

    assert by_group["dynamic_ef"]["status"] == "available"
    assert by_group["dynamic_ef"]["decision"] == "include"
    assert "claim_c_dynamic_vs_fixed_latency_deltas.csv" in by_group["dynamic_ef"]["primary_artifacts"]

    assert "request_body_bytes/query" in by_group["compact_request_execution"]["available_metrics"]
    assert "response_body_bytes/query" in by_group["compact_request_execution"]["available_metrics"]
    assert "route_planning_ms/query" in by_group["compact_request_execution"]["available_metrics"]
    assert "container_network_bytes/query" in by_group["compact_request_execution"]["available_metrics"]
    assert "controller_cpu_pct" in by_group["compact_request_execution"]["available_metrics"]
    assert "claim_e_payload_bytes_summary.csv" in by_group["compact_request_execution"]["primary_artifacts"]
    assert "claim_e_wire_bytes_summary.csv" in by_group["compact_request_execution"]["primary_artifacts"]
    assert "claim_e_planning_time_summary.csv" in by_group["compact_request_execution"]["primary_artifacts"]
    assert "claim_e_container_overhead_summary.csv" in by_group["compact_request_execution"]["primary_artifacts"]
    assert "physical NIC" in by_group["compact_request_execution"]["caveat"]

    assert by_group["multi_assignment"]["status"] == "partial_proxy_only"
    assert by_group["multi_assignment"]["decision"] == "do_not_use_as_strict_unified_latency_cell"
    assert "single-run" in by_group["multi_assignment"]["caveat"]

    assert by_group["topology_no_fission"]["status"] == "available_selected_latency_cell"
    assert by_group["topology_no_fission"]["decision"] == "include_as_selected_online_ablation_cell"
    assert "claim_a_partition_online_latency_submatrix_deltas.csv" in by_group["topology_no_fission"]["primary_artifacts"]
    assert "claim_a_topology_no_fission_config_sensitivity.csv" in by_group["topology_no_fission"]["primary_artifacts"]
    assert "config sensitivity" in by_group["topology_no_fission"]["caveat"]
    assert "multi-assignment" in by_group["topology_no_fission"]["next_step"]


def test_rows_have_stable_field_order():
    module = load_module()

    assert module.FIELDNAMES[:5] == [
        "ablation_group",
        "variant",
        "mechanism_disabled",
        "evidence_scope",
        "status",
    ]
