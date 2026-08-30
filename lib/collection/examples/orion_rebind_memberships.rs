use std::collections::HashSet;
use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use collection::orion::{OrionRouter, OrionRoutingArtifact, OrionUpperNode};
use common::fs::atomic_save;
use serde::Deserialize;
use sha2::{Digest, Sha256};

const MEMBERSHIP_SIDECAR_FORMAT_VERSION: u32 = 1;

fn usage(program: &str) -> String {
    format!(
        "Usage: {program} <source-generation.json> <memberships.json> <output-generation.json>\n\
         \n\
         Rebind only generation/layout metadata and ordered upper-node L0 shard memberships\n\
         onto an existing complete production Orion artifact. The source upper graph, upper-node\n\
         order, labels, vectors, schema, search parameters, and logical point count are preserved.\n\
         Writes canonical JSON and lowercase SHA-256 to <output-generation.json>.sha256.\n\
         Existing output or checksum files are never overwritten.\n\
         \n\
         Sidecar schema (all fields required):\n\
         {{\n\
           \"format_version\": 1,\n\
           \"source_artifact_sha256\": \"<SHA-256 of exact source file bytes>\",\n\
           \"source_generation\": <source generation>,\n\
           \"generation\": <new generation greater than source>,\n\
           \"layout_sha256\": \"<canonical full L0 assignment SHA-256>\",\n\
           \"shard_count\": <logical shard count>,\n\
           \"physical_point_count\": <full L0 physical-copy count>,\n\
           \"shard_memberships\": [[0], [0, 2], ...]\n\
         }}\n\
         shard_memberships is positional and must contain exactly one non-empty, duplicate-free\n\
         list for every source upper_nodes entry. It is the final L0 multi-assignment membership,\n\
         not the unique L1 owner."
    )
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MembershipSidecar {
    format_version: u32,
    source_artifact_sha256: String,
    source_generation: u64,
    generation: u64,
    layout_sha256: String,
    shard_count: u32,
    physical_point_count: u64,
    shard_memberships: Vec<Vec<u32>>,
}

#[derive(Debug)]
struct RebindReport {
    source_artifact_sha256: String,
    source_canonical_sha256: String,
    output_artifact_sha256: String,
    source_generation: u64,
    output_generation: u64,
    upper_node_count: usize,
    shard_count: u32,
    physical_point_count: u64,
    source_upper_graph_sha256: String,
    output_upper_graph_sha256: String,
    source_upper_nodes_identity_sha256: String,
    output_upper_nodes_identity_sha256: String,
}

#[derive(Debug, Clone)]
struct ImmutableSnapshot {
    format_version: u32,
    vector_schema: collection::orion::OrionVectorSchemaFingerprint,
    logical_point_count: u64,
    upper_k: usize,
    upper_ef_search: usize,
    dynamic_ef_base: usize,
    dynamic_ef_factor: usize,
    upper_node_count: usize,
    upper_graph_sha256: String,
    upper_nodes_identity_sha256: String,
}

struct DigestWriter<'a>(&'a mut Sha256);

impl Write for DigestWriter<'_> {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.0.update(buffer);
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn checksum_path(output: &Path) -> PathBuf {
    let mut value = output.as_os_str().to_owned();
    value.push(".sha256");
    PathBuf::from(value)
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn normalized_sha256(name: &str, value: &str) -> Result<String, Box<dyn Error>> {
    let normalized = value.trim().to_ascii_lowercase();
    if normalized.len() != 64 || !normalized.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!(
            "{name} must contain exactly 64 hexadecimal characters, got {value:?}"
        )
        .into());
    }
    Ok(normalized)
}

fn graph_semantic_sha256(artifact: &OrionRoutingArtifact) -> Result<String, Box<dyn Error>> {
    let graph = artifact
        .upper_graph
        .as_ref()
        .ok_or("source artifact is graphless; a complete production upper_graph is required")?;
    let mut digest = Sha256::new();
    serde_json::to_writer(DigestWriter(&mut digest), graph)?;
    Ok(format!("{:x}", digest.finalize()))
}

/// Bit-exact identity of the ordered upper labels and vectors, deliberately excluding memberships.
fn upper_nodes_identity_sha256(nodes: &[OrionUpperNode]) -> Result<String, Box<dyn Error>> {
    let mut digest = Sha256::new();
    digest.update((nodes.len() as u64).to_le_bytes());
    for node in nodes {
        let label = serde_json::to_vec(&node.label)?;
        digest.update((label.len() as u64).to_le_bytes());
        digest.update(label);
        digest.update((node.vector.len() as u64).to_le_bytes());
        for value in &node.vector {
            digest.update(value.to_bits().to_le_bytes());
        }
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn immutable_snapshot(
    artifact: &OrionRoutingArtifact,
) -> Result<ImmutableSnapshot, Box<dyn Error>> {
    Ok(ImmutableSnapshot {
        format_version: artifact.format_version,
        vector_schema: artifact.vector_schema.clone(),
        logical_point_count: artifact.logical_point_count,
        upper_k: artifact.upper_k,
        upper_ef_search: artifact.upper_ef_search,
        dynamic_ef_base: artifact.dynamic_ef_base,
        dynamic_ef_factor: artifact.dynamic_ef_factor,
        upper_node_count: artifact.upper_nodes.len(),
        upper_graph_sha256: graph_semantic_sha256(artifact)?,
        upper_nodes_identity_sha256: upper_nodes_identity_sha256(&artifact.upper_nodes)?,
    })
}

fn verify_immutable_snapshot(
    expected: &ImmutableSnapshot,
    actual: &OrionRoutingArtifact,
) -> Result<(), Box<dyn Error>> {
    if actual.format_version != expected.format_version {
        return Err("rebind changed format_version".into());
    }
    if actual.vector_schema != expected.vector_schema {
        return Err("rebind changed vector_schema".into());
    }
    if actual.logical_point_count != expected.logical_point_count {
        return Err("rebind changed logical_point_count".into());
    }
    if actual.upper_k != expected.upper_k
        || actual.upper_ef_search != expected.upper_ef_search
        || actual.dynamic_ef_base != expected.dynamic_ef_base
        || actual.dynamic_ef_factor != expected.dynamic_ef_factor
    {
        return Err("rebind changed upper-search or dynamic-EF parameters".into());
    }
    if actual.upper_nodes.len() != expected.upper_node_count {
        return Err("rebind changed upper_nodes length".into());
    }

    let actual_graph_sha256 = graph_semantic_sha256(actual)?;
    if actual_graph_sha256 != expected.upper_graph_sha256 {
        return Err(format!(
            "rebind changed upper_graph: expected semantic SHA-256 {}, got {}",
            expected.upper_graph_sha256, actual_graph_sha256
        )
        .into());
    }
    let actual_nodes_sha256 = upper_nodes_identity_sha256(&actual.upper_nodes)?;
    if actual_nodes_sha256 != expected.upper_nodes_identity_sha256 {
        return Err(format!(
            "rebind changed ordered upper-node labels or vector bits: expected SHA-256 {}, got {}",
            expected.upper_nodes_identity_sha256, actual_nodes_sha256
        )
        .into());
    }
    Ok(())
}

fn validate_sidecar(
    sidecar: &MembershipSidecar,
    source: &OrionRoutingArtifact,
    source_file_sha256: &str,
) -> Result<String, Box<dyn Error>> {
    if sidecar.format_version != MEMBERSHIP_SIDECAR_FORMAT_VERSION {
        return Err(format!(
            "unsupported membership sidecar format_version {}; supported version is {}",
            sidecar.format_version, MEMBERSHIP_SIDECAR_FORMAT_VERSION
        )
        .into());
    }

    let bound_source_sha256 =
        normalized_sha256("source_artifact_sha256", &sidecar.source_artifact_sha256)?;
    if bound_source_sha256 != source_file_sha256 {
        return Err(format!(
            "source artifact SHA-256 mismatch: sidecar binds {}, exact input bytes have {}",
            bound_source_sha256, source_file_sha256
        )
        .into());
    }
    if sidecar.source_generation != source.generation {
        return Err(format!(
            "source generation mismatch: sidecar binds {}, artifact contains {}",
            sidecar.source_generation, source.generation
        )
        .into());
    }
    if sidecar.generation <= source.generation {
        return Err(format!(
            "new generation {} must be greater than source generation {}",
            sidecar.generation, source.generation
        )
        .into());
    }
    if sidecar.shard_count == 0 {
        return Err("shard_count must be greater than zero".into());
    }
    if sidecar.physical_point_count < source.logical_point_count {
        return Err(format!(
            "physical_point_count {} is smaller than logical_point_count {}",
            sidecar.physical_point_count, source.logical_point_count
        )
        .into());
    }
    if sidecar.shard_memberships.len() != source.upper_nodes.len() {
        return Err(format!(
            "shard_memberships has {} rows, but source upper_nodes has {} entries",
            sidecar.shard_memberships.len(),
            source.upper_nodes.len()
        )
        .into());
    }

    let mut covered_shards = HashSet::with_capacity(sidecar.shard_count as usize);
    for (index, (node, memberships)) in source
        .upper_nodes
        .iter()
        .zip(&sidecar.shard_memberships)
        .enumerate()
    {
        if memberships.is_empty() {
            return Err(format!(
                "shard_memberships[{index}] for upper label {} is empty",
                node.label
            )
            .into());
        }
        let mut unique = HashSet::with_capacity(memberships.len());
        for &shard_id in memberships {
            if shard_id >= sidecar.shard_count {
                return Err(format!(
                    "shard_memberships[{index}] for upper label {} references shard {}, but shard_count is {}",
                    node.label, shard_id, sidecar.shard_count
                )
                .into());
            }
            if !unique.insert(shard_id) {
                return Err(format!(
                    "shard_memberships[{index}] for upper label {} repeats shard {}",
                    node.label, shard_id
                )
                .into());
            }
            covered_shards.insert(shard_id);
        }
    }
    for shard_id in 0..sidecar.shard_count {
        if !covered_shards.contains(&shard_id) {
            return Err(format!(
                "no upper-node membership covers shard {shard_id}; the production routing shard union would be incomplete"
            )
            .into());
        }
    }

    normalized_sha256("layout_sha256", &sidecar.layout_sha256)
}

fn rebind_artifact(
    source_path: &Path,
    sidecar_path: &Path,
    output_path: &Path,
) -> Result<RebindReport, Box<dyn Error>> {
    let output_checksum_path = checksum_path(output_path);
    if output_path.exists() {
        return Err(format!(
            "refusing to overwrite existing artifact {}",
            output_path.display()
        )
        .into());
    }
    if output_checksum_path.exists() {
        return Err(format!(
            "refusing to overwrite existing checksum {}",
            output_checksum_path.display()
        )
        .into());
    }
    if let Some(parent) = output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs_err::create_dir_all(parent)?;
    }

    let source_bytes = fs_err::read(source_path)?;
    let source_file_sha256 = sha256_hex(&source_bytes);
    let mut artifact = OrionRoutingArtifact::from_json_slice(&source_bytes, None)?;
    if artifact.upper_graph.is_none() {
        return Err(
            "source artifact is graphless; a complete production upper_graph is required".into(),
        );
    }
    let source_canonical_sha256 = artifact.canonical_sha256()?;
    let source_generation = artifact.generation;
    let immutable = immutable_snapshot(&artifact)?;

    let sidecar_bytes = fs_err::read(sidecar_path)?;
    let sidecar: MembershipSidecar = serde_json::from_slice(&sidecar_bytes)?;
    let layout_sha256 = validate_sidecar(&sidecar, &artifact, &source_file_sha256)?;

    artifact.generation = sidecar.generation;
    artifact.layout_sha256 = layout_sha256;
    artifact.shard_count = sidecar.shard_count;
    artifact.physical_point_count = sidecar.physical_point_count;
    for (node, memberships) in artifact
        .upper_nodes
        .iter_mut()
        .zip(sidecar.shard_memberships)
    {
        node.shard_membership = memberships;
    }

    artifact.validate()?;
    verify_immutable_snapshot(&immutable, &artifact)?;
    let output_graph_sha256 = graph_semantic_sha256(&artifact)?;
    let output_nodes_sha256 = upper_nodes_identity_sha256(&artifact.upper_nodes)?;
    let output_generation = artifact.generation;
    let upper_node_count = artifact.upper_nodes.len();
    let shard_count = artifact.shard_count;
    let physical_point_count = artifact.physical_point_count;
    let canonical_json = artifact.canonical_json_bytes()?;
    let output_artifact_sha256 = sha256_hex(&canonical_json);

    // Exercise the production loader before publishing any bytes. Parsing from the exact output
    // bytes also proves the canonical representation retains every immutable upper-graph field.
    drop(artifact);
    let load_probe =
        OrionRoutingArtifact::from_json_slice(&canonical_json, Some(&output_artifact_sha256))?;
    verify_immutable_snapshot(&immutable, &load_probe)?;
    OrionRouter::new(load_probe)?;

    atomic_save(output_path, |writer| writer.write_all(&canonical_json))?;

    // Refuse to publish a checksum unless the exact on-disk bytes, typed artifact, immutable
    // upper graph, labels, and vector bits all survived publication unchanged.
    let written = fs_err::read(output_path)?;
    if written != canonical_json {
        return Err("published artifact bytes differ from canonical output bytes".into());
    }
    if sha256_hex(&written) != output_artifact_sha256 {
        return Err("published artifact raw SHA-256 differs from canonical SHA-256".into());
    }
    let written_artifact =
        OrionRoutingArtifact::from_json_slice(&written, Some(&output_artifact_sha256))?;
    verify_immutable_snapshot(&immutable, &written_artifact)?;
    atomic_save(&output_checksum_path, |writer| {
        writer.write_all(format!("{output_artifact_sha256}\n").as_bytes())
    })?;

    Ok(RebindReport {
        source_artifact_sha256: source_file_sha256,
        source_canonical_sha256,
        output_artifact_sha256,
        source_generation,
        output_generation,
        upper_node_count,
        shard_count,
        physical_point_count,
        source_upper_graph_sha256: immutable.upper_graph_sha256,
        output_upper_graph_sha256: output_graph_sha256,
        source_upper_nodes_identity_sha256: immutable.upper_nodes_identity_sha256,
        output_upper_nodes_identity_sha256: output_nodes_sha256,
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args_os();
    let program = args
        .next()
        .unwrap_or_else(|| OsString::from("orion_rebind_memberships"))
        .to_string_lossy()
        .into_owned();
    let Some(source) = args.next() else {
        return Err(usage(&program).into());
    };
    if source == "--help" || source == "-h" {
        println!("{}", usage(&program));
        return Ok(());
    }
    let sidecar = args.next().ok_or_else(|| usage(&program))?;
    let output = args.next().ok_or_else(|| usage(&program))?;
    if let Some(extra) = args.next() {
        return Err(format!("unexpected argument {extra:?}\n{}", usage(&program)).into());
    }

    let source = PathBuf::from(source);
    let sidecar = PathBuf::from(sidecar);
    let output = PathBuf::from(output);
    let report = rebind_artifact(&source, &sidecar, &output)?;
    println!("artifact={}", output.display());
    println!("checksum_file={}", checksum_path(&output).display());
    println!("source_artifact_sha256={}", report.source_artifact_sha256);
    println!("source_canonical_sha256={}", report.source_canonical_sha256);
    println!("sha256={}", report.output_artifact_sha256);
    println!("source_generation={}", report.source_generation);
    println!("generation={}", report.output_generation);
    println!("upper_node_count={}", report.upper_node_count);
    println!("shard_count={}", report.shard_count);
    println!("physical_point_count={}", report.physical_point_count);
    println!(
        "source_upper_graph_sha256={}",
        report.source_upper_graph_sha256
    );
    println!(
        "output_upper_graph_sha256={}",
        report.output_upper_graph_sha256
    );
    println!(
        "source_upper_nodes_identity_sha256={}",
        report.source_upper_nodes_identity_sha256
    );
    println!(
        "output_upper_nodes_identity_sha256={}",
        report.output_upper_nodes_identity_sha256
    );
    println!(
        "upper_graph_semantic_identity={}",
        report.source_upper_graph_sha256 == report.output_upper_graph_sha256
    );
    println!(
        "upper_nodes_label_vector_bit_identity={}",
        report.source_upper_nodes_identity_sha256 == report.output_upper_nodes_identity_sha256
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use collection::orion::{
        ORION_ROUTING_ARTIFACT_FORMAT_VERSION, OrionUpperGraphNode, OrionUpperHnswGraph,
        OrionUpperNode, OrionVectorDatatype, OrionVectorSchemaFingerprint,
    };
    use segment::types::Distance;
    use serde_json::json;

    use super::*;

    fn artifact(with_graph: bool) -> OrionRoutingArtifact {
        OrionRoutingArtifact {
            format_version: ORION_ROUTING_ARTIFACT_FORMAT_VERSION,
            generation: 9,
            vector_schema: OrionVectorSchemaFingerprint {
                vector_name: String::new(),
                dimension: 2,
                distance: Distance::Dot,
                datatype: OrionVectorDatatype::Float32,
            },
            shard_count: 2,
            layout_sha256: "a".repeat(64),
            logical_point_count: 10,
            physical_point_count: 12,
            upper_k: 1,
            upper_ef_search: 2,
            dynamic_ef_base: 20,
            dynamic_ef_factor: 4,
            upper_nodes: vec![
                OrionUpperNode {
                    label: 10_u64.into(),
                    vector: vec![1.0, -0.0],
                    shard_membership: vec![0, 1],
                },
                OrionUpperNode {
                    label: 20_u64.into(),
                    vector: vec![0.0, 1.0],
                    shard_membership: vec![1],
                },
            ],
            upper_graph: with_graph.then(|| OrionUpperHnswGraph {
                entry_point: 10_u64.into(),
                max_level: 0,
                nodes: vec![
                    OrionUpperGraphNode {
                        label: 10_u64.into(),
                        neighbors_by_level: vec![vec![20_u64.into()]],
                    },
                    OrionUpperGraphNode {
                        label: 20_u64.into(),
                        neighbors_by_level: vec![vec![10_u64.into()]],
                    },
                ],
            }),
        }
    }

    fn write_source(directory: &Path, with_graph: bool) -> (PathBuf, String) {
        let path = directory.join("generation-9.json");
        let bytes = artifact(with_graph).canonical_json_bytes().unwrap();
        let checksum = sha256_hex(&bytes);
        fs_err::write(&path, bytes).unwrap();
        (path, checksum)
    }

    fn sidecar(source_sha256: &str) -> serde_json::Value {
        json!({
            "format_version": 1,
            "source_artifact_sha256": source_sha256,
            "source_generation": 9,
            "generation": 10,
            "layout_sha256": "b".repeat(64),
            "shard_count": 3,
            "physical_point_count": 14,
            "shard_memberships": [[2, 0], [1, 2]],
        })
    }

    fn write_sidecar(directory: &Path, value: &serde_json::Value) -> PathBuf {
        let path = directory.join("memberships.json");
        fs_err::write(&path, serde_json::to_vec_pretty(value).unwrap()).unwrap();
        path
    }

    #[test]
    fn rebinds_only_metadata_and_memberships_and_refuses_overwrite() {
        let directory = tempfile::tempdir().unwrap();
        let (source_path, source_sha256) = write_source(directory.path(), true);
        let sidecar_path = write_sidecar(directory.path(), &sidecar(&source_sha256));
        let output_path = directory.path().join("generation-10.json");

        let report = rebind_artifact(&source_path, &sidecar_path, &output_path).unwrap();
        assert_eq!(report.source_artifact_sha256, source_sha256);
        assert_eq!(
            report.source_upper_graph_sha256,
            report.output_upper_graph_sha256
        );
        assert_eq!(
            report.source_upper_nodes_identity_sha256,
            report.output_upper_nodes_identity_sha256
        );

        let source = OrionRoutingArtifact::read_json(&source_path, None).unwrap();
        let rebound =
            OrionRoutingArtifact::read_json(&output_path, Some(&report.output_artifact_sha256))
                .unwrap();
        assert_eq!(rebound.generation, 10);
        assert_eq!(rebound.layout_sha256, "b".repeat(64));
        assert_eq!(rebound.shard_count, 3);
        assert_eq!(rebound.physical_point_count, 14);
        assert_eq!(rebound.upper_nodes[0].shard_membership, vec![2, 0]);
        assert_eq!(rebound.upper_nodes[1].shard_membership, vec![1, 2]);
        assert_eq!(rebound.upper_graph, source.upper_graph);
        assert_eq!(
            upper_nodes_identity_sha256(&rebound.upper_nodes).unwrap(),
            upper_nodes_identity_sha256(&source.upper_nodes).unwrap()
        );
        assert_eq!(
            fs_err::read(&output_path).unwrap(),
            rebound.canonical_json_bytes().unwrap()
        );
        assert_eq!(
            fs_err::read_to_string(checksum_path(&output_path)).unwrap(),
            format!("{}\n", report.output_artifact_sha256)
        );
        assert!(
            rebind_artifact(&source_path, &sidecar_path, &output_path)
                .unwrap_err()
                .to_string()
                .contains("overwrite")
        );
    }

    #[test]
    fn rejects_wrong_source_binding_and_graphless_source() {
        let directory = tempfile::tempdir().unwrap();
        let (source_path, source_sha256) = write_source(directory.path(), true);
        let mut value = sidecar(&source_sha256);
        value["source_artifact_sha256"] = json!("0".repeat(64));
        let sidecar_path = write_sidecar(directory.path(), &value);
        let output_path = directory.path().join("generation-10.json");
        assert!(
            rebind_artifact(&source_path, &sidecar_path, &output_path)
                .unwrap_err()
                .to_string()
                .contains("SHA-256 mismatch")
        );

        fs_err::remove_file(&sidecar_path).unwrap();
        let (graphless_path, graphless_sha256) = write_source(directory.path(), false);
        let graphless_sidecar = write_sidecar(directory.path(), &sidecar(&graphless_sha256));
        assert!(
            rebind_artifact(&graphless_path, &graphless_sidecar, &output_path)
                .unwrap_err()
                .to_string()
                .contains("graphless")
        );
    }

    #[test]
    fn rejects_invalid_membership_rows() {
        let cases = [
            (json!([[0]]), "has 1 rows"),
            (json!([[], [1, 2]]), "is empty"),
            (json!([[0, 0], [1, 2]]), "repeats shard 0"),
            (json!([[0, 3], [1, 2]]), "references shard 3"),
            (json!([[0], [0]]), "covers shard 1"),
        ];

        for (memberships, expected_error) in cases {
            let directory = tempfile::tempdir().unwrap();
            let (source_path, source_sha256) = write_source(directory.path(), true);
            let mut value = sidecar(&source_sha256);
            value["shard_memberships"] = memberships;
            let sidecar_path = write_sidecar(directory.path(), &value);
            let output_path = directory.path().join("generation-10.json");
            let error = rebind_artifact(&source_path, &sidecar_path, &output_path)
                .unwrap_err()
                .to_string();
            assert!(
                error.contains(expected_error),
                "expected {error:?} to contain {expected_error:?}"
            );
            assert!(!output_path.exists());
            assert!(!checksum_path(&output_path).exists());
        }
    }

    #[test]
    fn rejects_non_new_generation_and_unknown_sidecar_fields() {
        let directory = tempfile::tempdir().unwrap();
        let (source_path, source_sha256) = write_source(directory.path(), true);
        let mut value = sidecar(&source_sha256);
        value["generation"] = json!(9);
        let sidecar_path = write_sidecar(directory.path(), &value);
        let output_path = directory.path().join("generation-10.json");
        assert!(
            rebind_artifact(&source_path, &sidecar_path, &output_path)
                .unwrap_err()
                .to_string()
                .contains("must be greater")
        );

        fs_err::remove_file(&sidecar_path).unwrap();
        value["generation"] = json!(10);
        value["l1_owner"] = json!([0, 1]);
        let sidecar_path = write_sidecar(directory.path(), &value);
        assert!(
            rebind_artifact(&source_path, &sidecar_path, &output_path)
                .unwrap_err()
                .to_string()
                .contains("unknown field")
        );
    }
}
