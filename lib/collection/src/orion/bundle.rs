//! Typed, durable offline bundle production and independent replay validation.
//! A bundle is published only by renaming its fully synced staging directory.
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Component, Path};

use segment::types::ExtendedPointId;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::layout::{self, LayoutOptions, Result, VectorEncoding, VectorInput};
use super::{
    ORION_ROUTING_ARTIFACT_FORMAT_VERSION, OrionRouter, OrionRoutingArtifact,
    OrionUpperGraphBuildOptions, OrionUpperNode, OrionVectorSchemaFingerprint,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BuildRequest {
    pub input: VectorInput,
    pub options: LayoutOptions,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Assignment {
    pub id: u64,
    pub shards: Vec<u32>,
}

impl Assignment {
    pub fn canonical_bytes(&self, shard_count: u32) -> Result<Vec<u8>> {
        let unique: HashSet<_> = self.shards.iter().collect();
        if self.shards.is_empty()
            || unique.len() != self.shards.len()
            || self.shards.iter().any(|&s| s >= shard_count)
        {
            return Err(format!("invalid membership for point {}", self.id).into());
        }
        let mut bytes = serde_json::to_vec(self)?;
        bytes.push(b'\n');
        Ok(bytes)
    }
}

/// Importer v1 wire format remains stable, including canonical assignment JSONL.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImportManifest {
    pub format_version: u32,
    #[serde(default, skip_serializing_if = "VectorEncoding::is_default")]
    pub vector_encoding: VectorEncoding,
    pub dimension: usize,
    pub point_count: usize,
    pub shard_count: u32,
    pub vector_name: String,
    pub orion_generation: u64,
    pub orion_artifact_file: String,
    pub orion_artifact_sha256: String,
    pub vectors_file: String,
    pub vectors_sha256: String,
    pub assignments_file: String,
    pub assignments_sha256: String,
    pub total_point_copies: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BuildManifest {
    pub format_version: u32,
    pub algorithm_version: String,
    pub options: LayoutOptions,
    pub raw_vectors_sha256: String,
    pub upper_indices: Vec<usize>,
    pub initial_owners: Vec<u32>,
    pub topology_iterations: usize,
    pub owner_entry_copies_added: u64,
    pub shard_counts: Vec<u64>,
    pub source_artifact_file: String,
    pub source_artifact_sha256: String,
    pub upper_graph_sha256: String,
    pub upper_identity_sha256: String,
    pub attachments_file: String,
    pub attachments_sha256: String,
    pub import_manifest_file: String,
    pub import_manifest_sha256: String,
    /// Offline validation is NOT collection readiness or cluster activation.
    pub state: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_source_artifact_sha256: Option<String>,
}

pub fn validate_filename(name: &str) -> Result<()> {
    let mut components = Path::new(name).components();
    if !matches!(components.next(), Some(Component::Normal(_)))
        || components.next().is_some()
        || name.contains(['/', '\\'])
    {
        return Err(format!("expected one safe filename, got {name:?}").into());
    }
    Ok(())
}

pub fn file_sha256(path: &Path) -> Result<String> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut hash = Sha256::new();
    let mut buf = [0_u8; 65536];
    loop {
        let count = reader.read(&mut buf)?;
        if count == 0 {
            break;
        }
        hash.update(&buf[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

pub(crate) fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut writer = BufWriter::new(File::create_new(path)?);
    serde_json::to_writer(&mut writer, value)?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    Ok(())
}

pub fn graph_sha256(artifact: &OrionRoutingArtifact) -> Result<String> {
    let graph = artifact.upper_graph.as_ref().ok_or("missing upper graph")?;
    // Graph and node fields are declared in canonical alphabetical order. Serialize
    // directly into the digest instead of allocating a full JSON Value and byte copy.
    struct HashWriter(Sha256);
    impl Write for HashWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.update(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut writer = HashWriter(Sha256::new());
    serde_json::to_writer(&mut writer, graph)?;
    Ok(format!("{:x}", writer.0.finalize()))
}

pub fn upper_identity_sha256(artifact: &OrionRoutingArtifact) -> Result<String> {
    let mut hash = Sha256::new();
    hash.update((artifact.upper_nodes.len() as u64).to_le_bytes());
    for node in &artifact.upper_nodes {
        let label = serde_json::to_vec(&node.label)?;
        hash.update((label.len() as u64).to_le_bytes());
        hash.update(label);
        hash.update((node.vector.len() as u64).to_le_bytes());
        for &x in &node.vector {
            hash.update(x.to_bits().to_le_bytes());
        }
    }
    Ok(format!("{:x}", hash.finalize()))
}

pub(crate) fn write_artifact(path: &Path, artifact: &OrionRoutingArtifact) -> Result<String> {
    artifact.validate()?;
    OrionRouter::from_artifact(artifact)?;
    write_json(path, artifact)?;
    let checksum = file_sha256(path)?;
    let mut sidecar = File::create_new(path.with_extension("json.sha256"))?;
    writeln!(sidecar, "{checksum}")?;
    sidecar.sync_all()?;
    Ok(checksum)
}

fn search_indices(
    router: &OrionRouter,
    query: &[f32],
    scratch: &mut super::router::OrionRouteScratch,
    index: &HashMap<ExtendedPointId, usize>,
) -> Result<Vec<usize>> {
    router
        .search_upper_with_scratch(query, scratch)?
        .into_iter()
        .map(|hit| {
            index
                .get(&hit.label)
                .copied()
                .ok_or_else(|| "unknown upper search label".into())
        })
        .collect()
}

/// Builds one production graph, refines its owners, then streams full assignment.
/// Memory is O(N IDs + upper vectors + upper self hits + one input row).
pub fn build(request: &BuildRequest, output: &Path) -> Result<BuildManifest> {
    build_with_sink(request, output, &mut |_, _| Ok(()))
}

/// Emit each finalized assignment immediately. A blocking sink provides backpressure without
/// retaining the full layout in memory. Publication requires file integrity, schema and count validation.
pub fn build_with_sink(
    request: &BuildRequest,
    output: &Path,
    sink: &mut dyn FnMut(&Assignment, &[f32]) -> Result<()>,
) -> Result<BuildManifest> {
    request.options.validate(request.input.rows)?;
    if output.symlink_metadata().is_ok() {
        return Err("output already exists".into());
    }
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let stage = tempfile::Builder::new()
        .prefix(".orion-build-")
        .tempdir_in(parent)?;
    // Keep failed attempts for inspection; never publish their manifests as usable bundles.
    let stage_path = stage.keep();
    let result = build_staged(request, &stage_path, sink);
    let manifest = match result {
        Ok(manifest) => manifest,
        Err(error) => {
            let _ = write_json(&stage_path.join("FAILED.json"), &error.to_string());
            return Err(format!("{error}; failed build evidence: {}", stage_path.display()).into());
        }
    };
    verify_integrity(&stage_path)?;
    File::open(&stage_path)?.sync_all()?;
    super::durable::publish_directory(&stage_path, output, "build-manifest.json")?;
    File::open(parent)?.sync_all()?;
    Ok(manifest)
}

fn build_staged(
    request: &BuildRequest,
    dir: &Path,
    sink: &mut dyn FnMut(&Assignment, &[f32]) -> Result<()>,
) -> Result<BuildManifest> {
    let input = &request.input;
    let o = &request.options;
    eprintln!(
        "Orion build: sampling {} input rows at 1/{}",
        input.rows, o.sample_denominator
    );
    let upper_indices = layout::sample_indices(input.rows, o.sample_denominator, o.sample_seed)?;
    let row_to_upper: HashMap<usize, usize> = upper_indices
        .iter()
        .enumerate()
        .map(|(i, &row)| (row, i))
        .collect();
    let encoding = input.encoding.normalized();
    let vectors_file = format!("{}.{}", o.bundle_prefix, encoding.extension());
    let mut writer = BufWriter::new(File::create_new(dir.join(&vectors_file))?);
    let mut vectors = vec![Vec::new(); upper_indices.len()];
    input.visit(|row, raw| {
        let vector = encoding.preprocess(o.distance, raw)?;
        writer.write_all(&encoding.encode(&vector)?)?;
        if let Some(&i) = row_to_upper.get(&row) {
            vectors[i] = vector;
        }
        if (row + 1) % 10_000_000 == 0 {
            eprintln!("Orion build: encoded {}/{} rows", row + 1, input.rows);
        }
        Ok(())
    })?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    drop(writer);
    let vectors_sha256 = file_sha256(&dir.join(&vectors_file))?;
    let normalized = VectorInput {
        format_version: 1,
        encoding,
        path: dir.join(&vectors_file),
        rows: input.rows,
        dimension: input.dimension,
        sha256: vectors_sha256.clone(),
    };
    eprintln!("Orion build: upper HNSW with {} nodes", upper_indices.len());
    let mut source = OrionRoutingArtifact {
        format_version: ORION_ROUTING_ARTIFACT_FORMAT_VERSION,
        generation: o.generation,
        vector_schema: OrionVectorSchemaFingerprint {
            vector_name: o.vector_name.clone(),
            dimension: input.dimension,
            distance: o.distance,
            datatype: encoding.datatype(),
        },
        shard_count: 1,
        layout_sha256: "0".repeat(64),
        logical_point_count: input.rows as u64,
        physical_point_count: input.rows as u64,
        upper_k: o.upper_k,
        upper_ef_search: o.upper_ef_search,
        dynamic_ef_base: o.dynamic_ef_base,
        dynamic_ef_factor: o.dynamic_ef_factor,
        upper_nodes: upper_indices
            .iter()
            .zip(vectors)
            .map(|(&row, vector)| OrionUpperNode {
                label: ExtendedPointId::NumId(row as u64),
                vector,
                owner_shard: 0,
            })
            .collect(),
        upper_graph: None,
    }
    .build_upper_hnsw(OrionUpperGraphBuildOptions {
        m: o.upper_m,
        ef_construct: o.upper_ef_construct,
        seed: o.upper_seed,
    })?;
    let source_artifact_file = format!("upper-source-generation-{}.json", o.generation);
    let source_artifact_sha256 = write_artifact(&dir.join(&source_artifact_file), &source)?;
    let upper_graph_sha256 = graph_sha256(&source)?;
    let upper_identity_sha256 = upper_identity_sha256(&source)?;
    source.upper_k = o.efs.min(source.upper_nodes.len());
    source.upper_ef_search = o.efs;
    let router = OrionRouter::from_artifact(&source)?;
    source.upper_k = o.upper_k;
    source.upper_ef_search = o.upper_ef_search;
    let index: HashMap<_, _> = source
        .upper_nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.label, i))
        .collect();
    let vectors: Vec<_> = source
        .upper_nodes
        .iter()
        .map(|n| n.vector.as_slice())
        .collect();
    let mut scratch = router.new_route_scratch();
    let self_hits: Vec<_> = vectors
        .iter()
        .map(|v| search_indices(&router, v, &mut scratch, &index))
        .collect::<Result<_>>()?;
    eprintln!("Orion build: K-Means owners");
    let initial_owners = layout::cluster(
        &vectors,
        o.shard_count as usize,
        o.kmeans_iterations,
        o.kmeans_seed,
    )?;
    let (owners, topology_iterations) =
        layout::refine_owners(&self_hits, &initial_owners, o.topology_iterations)?;
    drop(vectors);
    drop(self_hits);
    let assignments_file = format!("{}.assignments.jsonl", o.bundle_prefix);
    let mut assignments = BufWriter::new(File::create_new(dir.join(&assignments_file))?);
    let attachments_file = "upper-attachments.counted.bin".to_string();
    let mut attachments = BufWriter::new(File::create_new(dir.join(&attachments_file))?);
    let mut shard_counts = vec![0_u64; o.shard_count as usize];
    let mut owner_entry_copies_added = 0_u64;
    eprintln!("Orion build: assigning all {} input rows", input.rows);
    normalized.visit(|row, vector| {
        let hits = search_indices(&router, &vector, &mut scratch, &index)?;
        attachments.write_all(&u32::try_from(hits.len())?.to_le_bytes())?;
        for &hit in &hits {
            attachments.write_all(&(upper_indices[hit] as u64).to_le_bytes())?;
        }
        let mut shards = layout::strict_targets(&hits, &owners)?;
        if let Some(&i) = row_to_upper.get(&row) {
            if !shards.contains(&owners[i]) {
                shards.push(owners[i]);
                owner_entry_copies_added += 1;
            }
        }
        for &shard in &shards {
            shard_counts[shard as usize] += 1;
        }
        let assignment = Assignment {
            id: row as u64,
            shards,
        };
        assignments.write_all(&assignment.canonical_bytes(o.shard_count)?)?;
        sink(&assignment, &vector)?;
        if (row + 1) % 1_000_000 == 0 {
            eprintln!("Orion build: assigned {}/{} rows", row + 1, input.rows);
        }
        Ok(())
    })?;
    for w in [&mut assignments, &mut attachments] {
        w.flush()?;
        w.get_ref().sync_all()?;
    }
    drop(assignments);
    drop(attachments);
    let assignments_sha256 = file_sha256(&dir.join(&assignments_file))?;
    let mut final_artifact = source;
    final_artifact.shard_count = o.shard_count;
    final_artifact.layout_sha256.clone_from(&assignments_sha256);
    final_artifact.physical_point_count = shard_counts.iter().sum();
    for (node, owner) in final_artifact.upper_nodes.iter_mut().zip(&owners) {
        node.owner_shard = *owner;
    }
    if graph_sha256(&final_artifact)? != upper_graph_sha256
        || self::upper_identity_sha256(&final_artifact)? != upper_identity_sha256
    {
        return Err("upper graph identity changed during finalization".into());
    }
    let artifact_file = format!("generation-{}.json", o.generation);
    let artifact_sha256 = write_artifact(&dir.join(&artifact_file), &final_artifact)?;
    let import = ImportManifest {
        format_version: 1,
        vector_encoding: encoding,
        dimension: input.dimension,
        point_count: input.rows,
        shard_count: o.shard_count,
        vector_name: o.vector_name.clone(),
        orion_generation: o.generation,
        orion_artifact_file: artifact_file,
        orion_artifact_sha256: artifact_sha256,
        vectors_file,
        vectors_sha256,
        assignments_file,
        assignments_sha256,
        total_point_copies: final_artifact.physical_point_count,
    };
    let import_manifest_file = format!("{}.manifest.json", o.bundle_prefix);
    write_json(&dir.join(&import_manifest_file), &import)?;
    let manifest = BuildManifest {
        format_version: 2,
        algorithm_version: layout::LAYOUT_ALGORITHM_VERSION.into(),
        options: o.clone(),
        raw_vectors_sha256: input.sha256.clone(),
        upper_indices,
        initial_owners,
        topology_iterations,
        owner_entry_copies_added,
        shard_counts,
        source_artifact_file,
        source_artifact_sha256,
        upper_graph_sha256,
        upper_identity_sha256,
        attachments_sha256: file_sha256(&dir.join(&attachments_file))?,
        attachments_file,
        import_manifest_sha256: file_sha256(&dir.join(&import_manifest_file))?,
        import_manifest_file,
        state: "OfflineValidated".into(),
        runtime_source_artifact_sha256: None,
    };
    write_json(&dir.join("build-manifest.json"), &manifest)?;
    Ok(manifest)
}

fn checked_file(dir: &Path, name: &str, expected: &str) -> Result<std::path::PathBuf> {
    validate_filename(name)?;
    let path = dir.join(name);
    if path.symlink_metadata()?.file_type().is_symlink() || file_sha256(&path)? != expected {
        return Err(format!("bundle checksum mismatch or symlink: {name}").into());
    }
    Ok(path)
}

/// Independent replay of every navigation vote, owner, assignment and upper vector bit.
/// No Python audit or manifest assertion can substitute for these checks.
pub fn verify(dir: &Path) -> Result<ImportManifest> {
    verify_impl(dir, true)
}

/// Validate immutable files, schema and counts without rerunning layout algorithms.
/// Full deterministic replay remains available as the explicit `verify` diagnostic.
pub fn verify_integrity(dir: &Path) -> Result<ImportManifest> {
    verify_impl(dir, false)
}

fn verify_impl(dir: &Path, replay: bool) -> Result<ImportManifest> {
    if !replay {
        eprintln!("Orion integrity: validating manifests and graph bindings (no algorithm replay)");
    }
    let build: BuildManifest = serde_json::from_reader(BufReader::with_capacity(
        1024 * 1024,
        File::open(dir.join("build-manifest.json"))?,
    ))?;
    if build.format_version != 2
        || build.algorithm_version != layout::LAYOUT_ALGORITHM_VERSION
        || build.state != "OfflineValidated"
    {
        return Err("unsupported build manifest".into());
    }
    let import: ImportManifest = serde_json::from_reader(File::open(checked_file(
        dir,
        &build.import_manifest_file,
        &build.import_manifest_sha256,
    )?)?)?;
    if import.format_version != 1 || import.vector_encoding == VectorEncoding::U8bin {
        return Err("unsupported import manifest".into());
    }
    build.options.validate(import.point_count)?;
    let source = OrionRoutingArtifact::read_json(
        &checked_file(
            dir,
            &build.source_artifact_file,
            &build.source_artifact_sha256,
        )?,
        None,
    )?;
    if source.shard_count != 1
        || source.physical_point_count != source.logical_point_count
        || source.layout_sha256 != "0".repeat(64)
        || source.upper_nodes.iter().any(|n| n.owner_shard != 0)
    {
        return Err("upper source is not a neutral build artifact".into());
    }
    if graph_sha256(&source)? != build.upper_graph_sha256
        || upper_identity_sha256(&source)? != build.upper_identity_sha256
    {
        return Err("source graph or ordered vector identity mismatch".into());
    }
    let source_schema = source.vector_schema.clone();
    let source_generation = source.generation;
    let source_count = source.logical_point_count;
    drop(source);
    let artifact = OrionRoutingArtifact::read_json(
        &checked_file(
            dir,
            &import.orion_artifact_file,
            &import.orion_artifact_sha256,
        )?,
        None,
    )?;
    if graph_sha256(&artifact)? != build.upper_graph_sha256
        || upper_identity_sha256(&artifact)? != build.upper_identity_sha256
    {
        return Err("final graph or ordered vector identity mismatch".into());
    }
    if artifact.generation != import.orion_generation
        || artifact.shard_count != import.shard_count
        || artifact.logical_point_count != import.point_count as u64
        || artifact.physical_point_count != import.total_point_copies
        || artifact.layout_sha256 != import.assignments_sha256
        || artifact.vector_schema.dimension != import.dimension
        || artifact.vector_schema.vector_name != import.vector_name
        || artifact.vector_schema.datatype != import.vector_encoding.datatype()
        || artifact.vector_schema != source_schema
        || source_generation != artifact.generation
        || source_count != artifact.logical_point_count
    {
        return Err("artifact/import/source binding mismatch".into());
    }
    let o = &build.options;
    if o.generation != artifact.generation
        || o.shard_count != artifact.shard_count
        || o.distance != artifact.vector_schema.distance
        || o.vector_name != import.vector_name
        || o.upper_k != artifact.upper_k
        || o.upper_ef_search != artifact.upper_ef_search
        || o.dynamic_ef_base != artifact.dynamic_ef_base
        || o.dynamic_ef_factor != artifact.dynamic_ef_factor
    {
        return Err("build options do not bind final artifact".into());
    }
    if !replay {
        eprintln!("Orion integrity: checking vector/attachment checksums and byte lengths");
        let vector_path = checked_file(dir, &import.vectors_file, &import.vectors_sha256)?;
        let expected_bytes = import
            .point_count
            .checked_mul(import.dimension)
            .and_then(|n| n.checked_mul(import.vector_encoding.element_bytes()))
            .ok_or("vector byte length overflow")?;
        if fs::metadata(vector_path)?.len() != expected_bytes as u64 {
            return Err("vector byte length differs from manifest".into());
        }
        let attachments = checked_file(dir, &build.attachments_file, &build.attachments_sha256)?;
        let hits = o.efs.min(artifact.upper_nodes.len());
        let attachment_bytes = hits
            .checked_mul(8)
            .and_then(|n| n.checked_add(4))
            .and_then(|n| n.checked_mul(import.point_count))
            .ok_or("attachment size overflow")?;
        if fs::metadata(attachments)?.len() != attachment_bytes as u64 {
            return Err("attachment byte length differs from manifest".into());
        }
        let assignments = checked_file(dir, &import.assignments_file, &import.assignments_sha256)?;
        eprintln!(
            "Orion integrity: checking assignment IDs, shard ranges and counts (no searches)"
        );
        let mut reader = BufReader::with_capacity(1024 * 1024, File::open(assignments)?);
        let mut line = String::new();
        let mut rows = 0_usize;
        let mut counts = vec![0_u64; import.shard_count as usize];
        while reader.read_line(&mut line)? != 0 {
            let assignment: Assignment = serde_json::from_str(&line)?;
            if assignment.id != rows as u64 || rows >= import.point_count {
                return Err("assignment IDs or row count differ from manifest".into());
            }
            assignment.canonical_bytes(import.shard_count)?;
            for shard in assignment.shards {
                counts[shard as usize] += 1;
            }
            rows += 1;
            if rows % 10_000_000 == 0 {
                eprintln!(
                    "Orion integrity: checked {rows}/{} assignment records",
                    import.point_count
                );
            }
            line.clear();
        }
        if rows != import.point_count
            || counts != build.shard_counts
            || counts.iter().sum::<u64>() != import.total_point_copies
        {
            return Err("assignment counts differ from manifest".into());
        }
        eprintln!("Orion integrity: complete; full layout replay not requested");
        return Ok(import);
    }
    let indices = layout::sample_indices(import.point_count, o.sample_denominator, o.sample_seed)?;
    if indices != build.upper_indices || indices.len() != artifact.upper_nodes.len() {
        return Err("sample replay mismatch".into());
    }
    let mut row_to_upper = HashMap::new();
    let mut label_to_upper = HashMap::new();
    for (i, (&row, node)) in indices.iter().zip(&artifact.upper_nodes).enumerate() {
        if node.label != ExtendedPointId::NumId(row as u64) {
            return Err("sample label order mismatch".into());
        }
        row_to_upper.insert(row, i);
        label_to_upper.insert(node.label, i);
    }
    let vectors: Vec<_> = artifact
        .upper_nodes
        .iter()
        .map(|n| n.vector.as_slice())
        .collect();
    let initial = layout::cluster(
        &vectors,
        import.shard_count as usize,
        o.kmeans_iterations,
        o.kmeans_seed,
    )?;
    if initial != build.initial_owners {
        return Err("K-Means replay mismatch".into());
    }
    let router = OrionRouter::from_artifact_with_ef(&artifact, o.efs)?;
    let mut scratch = router.new_route_scratch();
    let self_hits: Vec<_> = vectors
        .iter()
        .map(|v| search_indices(&router, v, &mut scratch, &label_to_upper))
        .collect::<Result<_>>()?;
    let (owners, iterations) = layout::refine_owners(&self_hits, &initial, o.topology_iterations)?;
    if iterations != build.topology_iterations
        || !owners
            .iter()
            .zip(&artifact.upper_nodes)
            .all(|(&s, n)| s == n.owner_shard)
    {
        return Err("self-vote replay mismatch".into());
    }
    let mut assignments = BufReader::new(File::open(checked_file(
        dir,
        &import.assignments_file,
        &import.assignments_sha256,
    )?)?);
    let mut hits_reader = BufReader::new(File::open(checked_file(
        dir,
        &build.attachments_file,
        &build.attachments_sha256,
    )?)?);
    let input = VectorInput {
        format_version: 1,
        encoding: import.vector_encoding,
        path: checked_file(dir, &import.vectors_file, &import.vectors_sha256)?,
        rows: import.point_count,
        dimension: import.dimension,
        sha256: import.vectors_sha256.clone(),
    };
    let mut counts = vec![0_u64; import.shard_count as usize];
    let mut repairs = 0_u64;
    input.visit(|row, vector| {
        if let Some(&i) = row_to_upper.get(&row) {
            if vector
                .iter()
                .zip(vectors[i])
                .any(|(a, b)| a.to_bits() != b.to_bits())
            {
                return Err("upper vector differs from import bytes".into());
            }
        }
        let hits = search_indices(&router, &vector, &mut scratch, &label_to_upper)?;
        let mut count_bytes = [0; 4];
        hits_reader.read_exact(&mut count_bytes)?;
        if u32::from_le_bytes(count_bytes) as usize != hits.len() {
            return Err("attachment hit count replay mismatch".into());
        }
        for &hit in &hits {
            let mut bytes = [0; 8];
            hits_reader.read_exact(&mut bytes)?;
            if u64::from_le_bytes(bytes) != indices[hit] as u64 {
                return Err("attachment replay mismatch".into());
            }
        }
        let mut shards = layout::strict_targets(&hits, &owners)?;
        if let Some(&i) = row_to_upper.get(&row) {
            if !shards.contains(&owners[i]) {
                shards.push(owners[i]);
                repairs += 1;
            }
        }
        let mut line = Vec::new();
        assignments.read_until(b'\n', &mut line)?;
        if line
            != (Assignment {
                id: row as u64,
                shards: shards.clone(),
            })
            .canonical_bytes(import.shard_count)?
        {
            return Err(format!("assignment replay mismatch at point {row}").into());
        }
        for s in shards {
            counts[s as usize] += 1;
        }
        Ok(())
    })?;
    if assignments.read(&mut [0])? != 0
        || hits_reader.read(&mut [0])? != 0
        || counts != build.shard_counts
        || counts.iter().sum::<u64>() != import.total_point_copies
        || repairs != build.owner_entry_copies_added
    {
        return Err("bundle counts or trailing data mismatch".into());
    }
    Ok(import)
}
