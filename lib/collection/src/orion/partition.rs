//! Explicit capacity-partition variant. Owner planning is an external build
//! input; Rust reconstructs all memberships under the unchanged strict vote,
//! checks physical capacity/replication, and publishes a bound native bundle.
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use super::{OrionRoutingArtifact, bundle, capacity, layout};
use layout::Result;
use segment::types::ExtendedPointId;

/// Intermediate build material. This is deliberately NOT an installable layout:
/// no owners, data assignments, K-Means or self-vote result has been produced.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedCapacity {
    pub format_version: u32,
    pub options: layout::LayoutOptions,
    pub point_count: usize,
    pub dimension: usize,
    pub vector_encoding: layout::VectorEncoding,
    pub raw_vectors_sha256: String,
    pub vectors_file: String,
    pub vectors_sha256: String,
    pub upper_indices: Vec<usize>,
    pub source_artifact_file: String,
    pub source_artifact_sha256: String,
    pub upper_graph_sha256: String,
    pub upper_identity_sha256: String,
    pub attachments_file: String,
    pub attachments_sha256: String,
}

pub fn prepare(request: &bundle::BuildRequest, output: &Path) -> Result<PreparedCapacity> {
    request.options.validate_capacity(request.input.rows)?;
    if output.symlink_metadata().is_ok() {
        return Err("capacity preparation output already exists".into());
    }
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let stage = tempfile::Builder::new()
        .prefix(".orion-capacity-input-")
        .tempdir_in(parent)?;
    let upper = bundle::prepare_upper(request, stage.path())?;
    let router = super::OrionRouter::from_artifact_with_ef(&upper.source, request.options.efs)?;
    let index = upper
        .source
        .upper_nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.label, i))
        .collect();
    let mut scratch = router.new_route_scratch();
    let attachments_file = "upper-attachments.counted.bin".to_string();
    let mut writer = BufWriter::new(File::create_new(stage.path().join(&attachments_file))?);
    eprintln!(
        "Orion capacity: collecting attachments for {} rows (no self-vote or assignments)",
        request.input.rows
    );
    upper.normalized.visit(|_, vector| {
        let hits = bundle::search_indices(&router, &vector, &mut scratch, &index)?;
        writer.write_all(&u32::try_from(hits.len())?.to_le_bytes())?;
        for hit in hits {
            writer.write_all(&(upper.upper_indices[hit] as u64).to_le_bytes())?;
        }
        Ok(())
    })?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    drop(writer);
    let prepared = PreparedCapacity {
        format_version: 1,
        options: request.options.clone(),
        point_count: request.input.rows,
        dimension: request.input.dimension,
        vector_encoding: upper.normalized.encoding,
        raw_vectors_sha256: request.input.sha256.clone(),
        vectors_file: upper.vectors_file,
        vectors_sha256: upper.vectors_sha256,
        upper_indices: upper.upper_indices,
        source_artifact_file: upper.source_artifact_file,
        source_artifact_sha256: upper.source_artifact_sha256,
        upper_graph_sha256: upper.upper_graph_sha256,
        upper_identity_sha256: upper.upper_identity_sha256,
        attachments_sha256: bundle::file_sha256(&stage.path().join(&attachments_file))?,
        attachments_file,
    };
    bundle::write_json(&stage.path().join("capacity-input.json"), &prepared)?;
    read_prepared(stage.path())?;
    File::open(stage.path())?.sync_all()?;
    super::durable::publish_directory(stage.path(), output, "capacity-input.json")?;
    File::open(parent)?.sync_all()?;
    Ok(prepared)
}

pub fn read_prepared(dir: &Path) -> Result<PreparedCapacity> {
    let p: PreparedCapacity =
        serde_json::from_reader(File::open(dir.join("capacity-input.json"))?)?;
    if p.format_version != 1
        || p.dimension == 0
        || p.vector_encoding == layout::VectorEncoding::U8bin
    {
        return Err("unsupported capacity preparation".into());
    }
    p.options.validate_capacity(p.point_count)?;
    let vectors = bundle::checked_file(dir, &p.vectors_file, &p.vectors_sha256)?;
    let bytes = p
        .point_count
        .checked_mul(p.dimension)
        .and_then(|n| n.checked_mul(p.vector_encoding.element_bytes()))
        .ok_or("vector size overflow")?;
    if fs::metadata(vectors)?.len() != bytes as u64 {
        return Err("capacity vector shape mismatch".into());
    }
    let source = OrionRoutingArtifact::read_json(
        &bundle::checked_file(dir, &p.source_artifact_file, &p.source_artifact_sha256)?,
        None,
    )?;
    if p.upper_indices
        != layout::sample_indices(
            p.point_count,
            p.options.sample_denominator,
            p.options.sample_seed,
        )?
        || p.upper_indices.len() != source.upper_nodes.len()
        || p.upper_indices
            .iter()
            .zip(&source.upper_nodes)
            .any(|(&row, n)| n.label != ExtendedPointId::NumId(row as u64) || n.owner_shard != 0)
        || source.shard_count != 1
        || source.logical_point_count != p.point_count as u64
        || source.physical_point_count != p.point_count as u64
        || source.layout_sha256 != "0".repeat(64)
        || source.generation != p.options.generation
        || source.vector_schema.dimension != p.dimension
        || source.vector_schema.datatype != p.vector_encoding.datatype()
        || source.vector_schema.distance != p.options.distance
        || source.vector_schema.vector_name != p.options.vector_name
        || source.upper_k != p.options.upper_k
        || source.upper_ef_search != p.options.upper_ef_search
        || source.dynamic_ef_base != p.options.dynamic_ef_base
        || source.dynamic_ef_factor != p.options.dynamic_ef_factor
        || bundle::graph_sha256(&source)? != p.upper_graph_sha256
        || bundle::upper_identity_sha256(&source)? != p.upper_identity_sha256
    {
        return Err("capacity preparation graph/input binding mismatch".into());
    }
    let hits = p.options.efs.min(p.upper_indices.len());
    let bytes = hits
        .checked_mul(8)
        .and_then(|n| n.checked_add(4))
        .and_then(|n| n.checked_mul(p.point_count))
        .ok_or("attachment size overflow")?;
    let attachments = bundle::checked_file(dir, &p.attachments_file, &p.attachments_sha256)?;
    if fs::metadata(attachments)?.len() != bytes as u64 {
        return Err("capacity attachment shape mismatch".into());
    }
    Ok(p)
}

fn source_manifests(source_dir: &Path) -> Result<(bundle::ImportManifest, bundle::BuildManifest)> {
    if !source_dir.join("capacity-input.json").exists() {
        let import = bundle::verify_integrity(source_dir)?;
        let build = serde_json::from_reader(File::open(source_dir.join("build-manifest.json"))?)?;
        return Ok((import, build));
    }
    let p = read_prepared(source_dir)?;
    let import = bundle::ImportManifest {
        format_version: 1,
        vector_encoding: p.vector_encoding,
        dimension: p.dimension,
        point_count: p.point_count,
        shard_count: p.options.shard_count,
        vector_name: p.options.vector_name.clone(),
        orion_generation: p.options.generation,
        orion_artifact_file: p.source_artifact_file.clone(),
        orion_artifact_sha256: p.source_artifact_sha256.clone(),
        vectors_file: p.vectors_file,
        vectors_sha256: p.vectors_sha256,
        assignments_file: format!("{}.assignments.jsonl", p.options.bundle_prefix),
        assignments_sha256: String::new(),
        total_point_copies: p.point_count as u64,
    };
    let build = bundle::BuildManifest {
        format_version: 2,
        algorithm_version: capacity::ALGORITHM.into(),
        import_manifest_file: format!("{}.manifest.json", p.options.bundle_prefix),
        options: p.options,
        raw_vectors_sha256: p.raw_vectors_sha256,
        upper_indices: p.upper_indices,
        initial_owners: Vec::new(),
        topology_iterations: 0,
        owner_entry_copies_added: 0,
        shard_counts: Vec::new(),
        source_artifact_file: p.source_artifact_file,
        source_artifact_sha256: p.source_artifact_sha256,
        upper_graph_sha256: p.upper_graph_sha256,
        upper_identity_sha256: p.upper_identity_sha256,
        attachments_file: p.attachments_file,
        attachments_sha256: p.attachments_sha256,
        import_manifest_sha256: String::new(),
        state: "OfflineValidated".into(),
        runtime_source_artifact_sha256: None,
        capacity_max_replication_factor: None,
    };
    Ok((import, build))
}

pub fn derive(
    source_dir: &Path,
    output: &Path,
    owners: Vec<u32>,
    generation: u64,
) -> Result<bundle::BuildManifest> {
    derive_with_limit(source_dir, output, owners, generation, None)
}

pub fn derive_with_limit(
    source_dir: &Path,
    output: &Path,
    owners: Vec<u32>,
    generation: u64,
    max_replication_factor: Option<f64>,
) -> Result<bundle::BuildManifest> {
    if generation == 0 || output.symlink_metadata().is_ok() {
        return Err("partition requires positive generation and a new output directory".into());
    }
    let (mut import, mut build) = source_manifests(source_dir)?;
    capacity::validate_replication(
        import.point_count as u64,
        import.point_count,
        max_replication_factor,
    )?;
    let mut artifact = OrionRoutingArtifact::read_json(
        &source_dir.join(&import.orion_artifact_file),
        Some(&import.orion_artifact_sha256),
    )?;
    if owners.len() != artifact.upper_nodes.len() || owners.iter().any(|&s| s >= import.shard_count)
    {
        return Err("partition owners do not match upper nodes or shard range".into());
    }
    let mut row_to_upper = HashMap::new();
    let mut label_to_upper = HashMap::new();
    for (i, (&row, node)) in build
        .upper_indices
        .iter()
        .zip(&artifact.upper_nodes)
        .enumerate()
    {
        if node.label != ExtendedPointId::NumId(row as u64) {
            return Err("partition source sample identity mismatch".into());
        }
        row_to_upper.insert(row, i);
        label_to_upper.insert(row as u64, i);
    }
    if label_to_upper.len() != owners.len() {
        return Err("partition source sample count mismatch".into());
    }
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let stage = tempfile::Builder::new()
        .prefix(".orion-capacity-")
        .tempdir_in(parent)?;
    for name in [&import.vectors_file, &build.attachments_file] {
        fs::copy(source_dir.join(name), stage.path().join(name))?;
        File::open(stage.path().join(name))?.sync_all()?;
    }
    let mut hits_reader = BufReader::new(File::open(stage.path().join(&build.attachments_file))?);
    let mut assignments = BufWriter::new(File::create_new(
        stage.path().join(&import.assignments_file),
    )?);
    let mut counts = vec![0u64; import.shard_count as usize];
    let mut repairs = 0u64;
    let expected_hits = build.options.efs.min(owners.len());
    for row in 0..import.point_count {
        let mut bytes = [0; 4];
        hits_reader.read_exact(&mut bytes)?;
        let count = u32::from_le_bytes(bytes) as usize;
        if count != expected_hits {
            return Err("partition attachment count mismatch".into());
        }
        let mut hits = Vec::with_capacity(count);
        for _ in 0..count {
            let mut bytes = [0; 8];
            hits_reader.read_exact(&mut bytes)?;
            hits.push(
                *label_to_upper
                    .get(&u64::from_le_bytes(bytes))
                    .ok_or("unknown partition attachment label")?,
            );
        }
        let mut shards = layout::strict_targets(&hits, &owners)?;
        if let Some(&i) = row_to_upper.get(&row) {
            if !shards.contains(&owners[i]) {
                shards.push(owners[i]);
                repairs += 1;
            }
        }
        for &s in &shards {
            counts[s as usize] += 1;
        }
        let assignment = bundle::Assignment {
            id: row as u64,
            shards,
        };
        assignments.write_all(&assignment.canonical_bytes(import.shard_count)?)?;
    }
    if hits_reader.read(&mut [0])? != 0 {
        return Err("trailing partition attachments".into());
    }
    assignments.flush()?;
    assignments.get_ref().sync_all()?;
    drop(assignments);
    capacity::validate_counts(&counts)?;
    let copies = counts.iter().sum();
    capacity::validate_replication(copies, import.point_count, max_replication_factor)?;
    let layout_sha = bundle::file_sha256(&stage.path().join(&import.assignments_file))?;
    artifact.generation = generation;
    artifact.shard_count = import.shard_count;
    artifact.layout_sha256 = layout_sha.clone();
    artifact.physical_point_count = copies;
    for (node, &owner) in artifact.upper_nodes.iter_mut().zip(&owners) {
        node.owner_shard = owner;
    }
    if bundle::graph_sha256(&artifact)? != build.upper_graph_sha256
        || bundle::upper_identity_sha256(&artifact)? != build.upper_identity_sha256
    {
        return Err("capacity partition changed upper graph or vectors".into());
    }
    let mut neutral = OrionRoutingArtifact::read_json(
        &source_dir.join(&build.source_artifact_file),
        Some(&build.source_artifact_sha256),
    )?;
    neutral.generation = generation;
    build.source_artifact_file = format!("upper-source-generation-{generation}.json");
    build.source_artifact_sha256 =
        bundle::write_artifact(&stage.path().join(&build.source_artifact_file), &neutral)?;
    import.orion_generation = generation;
    import.orion_artifact_file = format!("generation-{generation}.json");
    import.orion_artifact_sha256 =
        bundle::write_artifact(&stage.path().join(&import.orion_artifact_file), &artifact)?;
    import.assignments_sha256 = layout_sha;
    import.total_point_copies = copies;
    bundle::write_json(&stage.path().join(&build.import_manifest_file), &import)?;
    build.import_manifest_sha256 =
        bundle::file_sha256(&stage.path().join(&build.import_manifest_file))?;
    build.algorithm_version = capacity::ALGORITHM.into();
    build.initial_owners = owners;
    build.topology_iterations = 0;
    build.options.generation = generation;
    build.shard_counts = counts;
    build.owner_entry_copies_added = repairs;
    build.runtime_source_artifact_sha256 = None;
    build.capacity_max_replication_factor = max_replication_factor;
    bundle::write_json(&stage.path().join("build-manifest.json"), &build)?;
    bundle::verify_integrity(stage.path())?;
    File::open(stage.path())?.sync_all()?;
    super::durable::publish_directory(stage.path(), output, "build-manifest.json")?;
    File::open(parent)?.sync_all()?;
    Ok(build)
}
