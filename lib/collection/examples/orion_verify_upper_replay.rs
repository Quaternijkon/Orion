use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use collection::orion::{OrionRouter, OrionRoutingArtifact, OrionUpperHit, OrionUpperNode};
use fs_err::File;
use segment::types::ExtendedPointId;
use serde::Serialize;
use sha2::{Digest, Sha256};

const REPLAY_GATE_FORMAT_VERSION: u32 = 1;
const REPLAY_MAGIC: &[u8] = b"ORION_UPPER_REPLAY_V1\0";

fn usage(program: &str) -> String {
    format!(
        "Usage: {program} <source-generation.json> <rebound-generation.json> \
         <queries.f32le> <row-count> <dimension> <replay.bin> <manifest.json>\n\
         \n\
         Fail-closed replay gate for a membership-only Orion artifact rebind. Both artifacts are\n\
         loaded by the production OrionRouter and evaluated on the same fixed little-endian f32\n\
         query corpus. Every ordered upper-hit label and every lower-is-better distance f32 bit\n\
         pattern must match. Canonical typed upper_graph bytes and ordered upper label/vector bits\n\
         must also be identical.\n\
         \n\
         On PASS, replay.bin contains one canonical transcript shared by source and rebound:\n\
         magic bytes, row-count u64le, dimension u64le, upper-k u64le, then row-major hits. Each\n\
         hit is label tag 0 + numeric u64le, or tag 1 + 16 RFC-4122 UUID bytes, followed by the\n\
         exact distance f32::to_bits() as u32le. A PASS manifest is published only after every\n\
         gate succeeds. Existing outputs are never overwritten."
    )
}

#[derive(Debug)]
struct Options {
    source_artifact_path: PathBuf,
    rebound_artifact_path: PathBuf,
    queries_path: PathBuf,
    row_count: usize,
    dimension: usize,
    replay_path: PathBuf,
    manifest_path: PathBuf,
}

#[derive(Debug, Serialize)]
struct ArtifactEvidence {
    path: String,
    file_sha256: String,
    canonical_artifact_sha256: String,
    generation: u64,
    layout_sha256: String,
    shard_count: u32,
    physical_point_count: u64,
    canonical_upper_graph_sha256: String,
    canonical_upper_graph_size_bytes: u64,
    ordered_upper_nodes_identity_sha256: String,
}

#[derive(Debug, Serialize)]
struct QueryCorpusEvidence {
    path: String,
    sha256: String,
    row_count: usize,
    dimension: usize,
    size_bytes: u64,
}

#[derive(Debug, Serialize)]
struct ReplayEvidence {
    path: String,
    encoding: &'static str,
    sha256: String,
    size_bytes: u64,
    upper_k: usize,
    compared_hit_count: u64,
    ordered_label_and_distance_bits_sha256: String,
}

#[derive(Debug, Serialize)]
struct ReplayGates {
    generation_advanced: bool,
    immutable_metadata_equal: bool,
    vector_schema_equal: bool,
    upper_search_contract_equal: bool,
    canonical_upper_graph_bytes_equal: bool,
    ordered_upper_labels_and_vector_bits_equal: bool,
    ordered_hit_labels_equal: bool,
    ordered_distance_bits_equal: bool,
    production_router_replay_complete: bool,
}

#[derive(Debug, Serialize)]
struct ReplayManifest {
    format_version: u32,
    verdict: &'static str,
    source: ArtifactEvidence,
    rebound: ArtifactEvidence,
    query_corpus: QueryCorpusEvidence,
    replay: ReplayEvidence,
    gates: ReplayGates,
    elapsed_seconds: f64,
}

#[derive(Debug)]
struct LoadedArtifact {
    artifact: OrionRoutingArtifact,
    file_sha256: String,
    canonical_artifact_sha256: String,
    canonical_upper_graph_bytes: Vec<u8>,
    ordered_upper_nodes_identity_sha256: String,
}

fn parse_positive_usize(name: &str, value: OsString) -> Result<usize, Box<dyn Error>> {
    let value = value
        .into_string()
        .map_err(|_| format!("{name} is not valid UTF-8"))?;
    let parsed = value.parse::<usize>()?;
    if parsed == 0 {
        return Err(format!("{name} must be greater than zero").into());
    }
    Ok(parsed)
}

fn parse_args() -> Result<Options, Box<dyn Error>> {
    let mut args = env::args_os();
    let program = args
        .next()
        .unwrap_or_else(|| OsString::from("orion_verify_upper_replay"))
        .to_string_lossy()
        .into_owned();
    let Some(source_artifact_path) = args.next() else {
        return Err(usage(&program).into());
    };
    if source_artifact_path == "--help" || source_artifact_path == "-h" {
        println!("{}", usage(&program));
        std::process::exit(0);
    }
    let rebound_artifact_path = args.next().ok_or_else(|| usage(&program))?;
    let queries_path = args.next().ok_or_else(|| usage(&program))?;
    let row_count = parse_positive_usize("row-count", args.next().ok_or_else(|| usage(&program))?)?;
    let dimension = parse_positive_usize("dimension", args.next().ok_or_else(|| usage(&program))?)?;
    let replay_path = args.next().ok_or_else(|| usage(&program))?;
    let manifest_path = args.next().ok_or_else(|| usage(&program))?;
    if let Some(extra) = args.next() {
        return Err(format!("unexpected argument {extra:?}\n{}", usage(&program)).into());
    }

    Ok(Options {
        source_artifact_path: source_artifact_path.into(),
        rebound_artifact_path: rebound_artifact_path.into(),
        queries_path: queries_path.into(),
        row_count,
        dimension,
        replay_path: replay_path.into(),
        manifest_path: manifest_path.into(),
    })
}

fn checked_size(
    row_count: usize,
    width: usize,
    element_size: usize,
) -> Result<usize, Box<dyn Error>> {
    row_count
        .checked_mul(width)
        .and_then(|elements| elements.checked_mul(element_size))
        .ok_or_else(|| "requested matrix size overflows usize".into())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn temporary_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(format!(".tmp.{}", std::process::id()));
    path.with_file_name(name)
}

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

fn load_artifact(path: &Path) -> Result<LoadedArtifact, Box<dyn Error>> {
    let bytes = fs_err::read(path)?;
    let file_sha256 = sha256_hex(&bytes);
    let artifact = OrionRoutingArtifact::from_json_slice(&bytes, None)?;
    let graph = artifact
        .upper_graph
        .as_ref()
        .ok_or_else(|| format!("artifact {} is graphless", path.display()))?;
    let canonical_upper_graph_bytes = serde_json::to_vec(graph)?;
    let ordered_upper_nodes_identity_sha256 = upper_nodes_identity_sha256(&artifact.upper_nodes)?;
    let canonical_artifact_sha256 = artifact.canonical_sha256()?;
    Ok(LoadedArtifact {
        artifact,
        file_sha256,
        canonical_artifact_sha256,
        canonical_upper_graph_bytes,
        ordered_upper_nodes_identity_sha256,
    })
}

fn verify_immutable_contract(
    source: &LoadedArtifact,
    rebound: &LoadedArtifact,
) -> Result<(), Box<dyn Error>> {
    let source_artifact = &source.artifact;
    let rebound_artifact = &rebound.artifact;
    if rebound_artifact.generation <= source_artifact.generation {
        return Err(format!(
            "rebound generation {} must be greater than source generation {}",
            rebound_artifact.generation, source_artifact.generation
        )
        .into());
    }
    if source_artifact.format_version != rebound_artifact.format_version
        || source_artifact.logical_point_count != rebound_artifact.logical_point_count
        || source_artifact.dynamic_ef_base != rebound_artifact.dynamic_ef_base
        || source_artifact.dynamic_ef_factor != rebound_artifact.dynamic_ef_factor
    {
        return Err(
            "membership rebind changed immutable format/logical-count/dynamic-EF metadata".into(),
        );
    }
    if source_artifact.vector_schema != rebound_artifact.vector_schema {
        return Err("membership rebind changed vector_schema".into());
    }
    if source_artifact.upper_k != rebound_artifact.upper_k
        || source_artifact.upper_ef_search != rebound_artifact.upper_ef_search
    {
        return Err(format!(
            "membership rebind changed upper search contract: source upper_k/ef={}/{}, rebound={}/{}",
            source_artifact.upper_k,
            source_artifact.upper_ef_search,
            rebound_artifact.upper_k,
            rebound_artifact.upper_ef_search
        )
        .into());
    }
    if source.canonical_upper_graph_bytes != rebound.canonical_upper_graph_bytes {
        return Err(format!(
            "canonical upper_graph bytes differ: source SHA-256 {}, rebound SHA-256 {}",
            sha256_hex(&source.canonical_upper_graph_bytes),
            sha256_hex(&rebound.canonical_upper_graph_bytes)
        )
        .into());
    }
    if source.ordered_upper_nodes_identity_sha256 != rebound.ordered_upper_nodes_identity_sha256 {
        return Err(format!(
            "ordered upper labels or vector bits differ: source SHA-256 {}, rebound SHA-256 {}",
            source.ordered_upper_nodes_identity_sha256, rebound.ordered_upper_nodes_identity_sha256
        )
        .into());
    }
    Ok(())
}

fn compare_hit(
    row_index: usize,
    rank: usize,
    source: OrionUpperHit,
    rebound: OrionUpperHit,
) -> Result<OrionUpperHit, Box<dyn Error>> {
    if source.label != rebound.label {
        return Err(format!(
            "ordered upper-hit label mismatch at query {row_index}, rank {rank}: source {}, rebound {}",
            source.label, rebound.label
        )
        .into());
    }
    if source.distance.to_bits() != rebound.distance.to_bits() {
        return Err(format!(
            "upper-hit distance-bit mismatch at query {row_index}, rank {rank}, label {}: source 0x{:08x}, rebound 0x{:08x}",
            source.label,
            source.distance.to_bits(),
            rebound.distance.to_bits()
        )
        .into());
    }
    Ok(source)
}

fn write_hashed(
    writer: &mut BufWriter<File>,
    digest: &mut Sha256,
    byte_count: &mut u64,
    bytes: &[u8],
) -> Result<(), Box<dyn Error>> {
    writer.write_all(bytes)?;
    digest.update(bytes);
    *byte_count = byte_count
        .checked_add(u64::try_from(bytes.len())?)
        .ok_or("replay byte count overflow")?;
    Ok(())
}

fn write_label(
    writer: &mut BufWriter<File>,
    digest: &mut Sha256,
    byte_count: &mut u64,
    label: ExtendedPointId,
) -> Result<(), Box<dyn Error>> {
    match label {
        ExtendedPointId::NumId(value) => {
            write_hashed(writer, digest, byte_count, &[0])?;
            write_hashed(writer, digest, byte_count, &value.to_le_bytes())?;
        }
        ExtendedPointId::Uuid(value) => {
            write_hashed(writer, digest, byte_count, &[1])?;
            write_hashed(writer, digest, byte_count, value.as_bytes())?;
        }
    }
    Ok(())
}

fn publish_no_replace(temp: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
    fs_err::hard_link(temp, output).map_err(|error| {
        format!(
            "refusing to overwrite or failing to publish {}: {error}",
            output.display()
        )
    })?;
    fs_err::remove_file(temp)?;
    Ok(())
}

fn verify_replay(options: &Options) -> Result<ReplayManifest, Box<dyn Error>> {
    if options.replay_path == options.manifest_path {
        return Err("replay and manifest output paths must be different".into());
    }
    for output in [&options.replay_path, &options.manifest_path] {
        if output.exists() {
            return Err(format!("refusing to overwrite {}", output.display()).into());
        }
        if let Some(parent) = output
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs_err::create_dir_all(parent)?;
        }
    }
    let replay_temp = temporary_path(&options.replay_path);
    let manifest_temp = temporary_path(&options.manifest_path);
    for temp in [&replay_temp, &manifest_temp] {
        if temp.exists() {
            return Err(format!("temporary output already exists: {}", temp.display()).into());
        }
    }

    let started = Instant::now();
    let source = load_artifact(&options.source_artifact_path)?;
    let rebound = load_artifact(&options.rebound_artifact_path)?;
    verify_immutable_contract(&source, &rebound)?;
    if source.artifact.vector_schema.dimension != options.dimension {
        return Err(format!(
            "query dimension {} does not match artifact dimension {}",
            options.dimension, source.artifact.vector_schema.dimension
        )
        .into());
    }

    let expected_query_bytes =
        checked_size(options.row_count, options.dimension, size_of::<f32>())?;
    let actual_query_bytes = usize::try_from(fs_err::metadata(&options.queries_path)?.len())?;
    if actual_query_bytes != expected_query_bytes {
        return Err(format!(
            "{} has {actual_query_bytes} bytes; expected {expected_query_bytes}",
            options.queries_path.display()
        )
        .into());
    }
    let upper_k = source.artifact.upper_k;
    let compared_hit_count = options
        .row_count
        .checked_mul(upper_k)
        .and_then(|value| u64::try_from(value).ok())
        .ok_or("compared hit count overflow")?;

    let source_evidence = ArtifactEvidence {
        path: options.source_artifact_path.display().to_string(),
        file_sha256: source.file_sha256.clone(),
        canonical_artifact_sha256: source.canonical_artifact_sha256.clone(),
        generation: source.artifact.generation,
        layout_sha256: source.artifact.layout_sha256.clone(),
        shard_count: source.artifact.shard_count,
        physical_point_count: source.artifact.physical_point_count,
        canonical_upper_graph_sha256: sha256_hex(&source.canonical_upper_graph_bytes),
        canonical_upper_graph_size_bytes: u64::try_from(source.canonical_upper_graph_bytes.len())?,
        ordered_upper_nodes_identity_sha256: source.ordered_upper_nodes_identity_sha256.clone(),
    };
    let rebound_evidence = ArtifactEvidence {
        path: options.rebound_artifact_path.display().to_string(),
        file_sha256: rebound.file_sha256.clone(),
        canonical_artifact_sha256: rebound.canonical_artifact_sha256.clone(),
        generation: rebound.artifact.generation,
        layout_sha256: rebound.artifact.layout_sha256.clone(),
        shard_count: rebound.artifact.shard_count,
        physical_point_count: rebound.artifact.physical_point_count,
        canonical_upper_graph_sha256: sha256_hex(&rebound.canonical_upper_graph_bytes),
        canonical_upper_graph_size_bytes: u64::try_from(rebound.canonical_upper_graph_bytes.len())?,
        ordered_upper_nodes_identity_sha256: rebound.ordered_upper_nodes_identity_sha256.clone(),
    };

    let source_router = OrionRouter::new(source.artifact)?;
    let rebound_router = OrionRouter::new(rebound.artifact)?;

    let result = (|| -> Result<ReplayManifest, Box<dyn Error>> {
        let mut reader = BufReader::new(File::open(&options.queries_path)?);
        let mut writer = BufWriter::new(File::create_new(&replay_temp)?);
        let mut query_digest = Sha256::new();
        let mut replay_digest = Sha256::new();
        let mut replay_size_bytes = 0_u64;
        write_hashed(
            &mut writer,
            &mut replay_digest,
            &mut replay_size_bytes,
            REPLAY_MAGIC,
        )?;
        write_hashed(
            &mut writer,
            &mut replay_digest,
            &mut replay_size_bytes,
            &u64::try_from(options.row_count)?.to_le_bytes(),
        )?;
        write_hashed(
            &mut writer,
            &mut replay_digest,
            &mut replay_size_bytes,
            &u64::try_from(options.dimension)?.to_le_bytes(),
        )?;
        write_hashed(
            &mut writer,
            &mut replay_digest,
            &mut replay_size_bytes,
            &u64::try_from(upper_k)?.to_le_bytes(),
        )?;

        let mut row_bytes = vec![0_u8; options.dimension * size_of::<f32>()];
        for row_index in 0..options.row_count {
            reader.read_exact(&mut row_bytes)?;
            query_digest.update(&row_bytes);
            let query = row_bytes
                .chunks_exact(size_of::<f32>())
                .enumerate()
                .map(|(column, bytes)| {
                    let value = f32::from_le_bytes(bytes.try_into().unwrap());
                    if value.is_finite() {
                        Ok(value)
                    } else {
                        Err(format!(
                            "query {row_index} has a non-finite component at column {column}"
                        ))
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            let source_hits = source_router.search_upper(&query)?;
            let rebound_hits = rebound_router.search_upper(&query)?;
            if source_hits.len() != rebound_hits.len() {
                return Err(format!(
                    "upper-hit count mismatch at query {row_index}: source {}, rebound {}",
                    source_hits.len(),
                    rebound_hits.len()
                )
                .into());
            }
            for (rank, (source_hit, rebound_hit)) in
                source_hits.into_iter().zip(rebound_hits).enumerate()
            {
                let hit = compare_hit(row_index, rank, source_hit, rebound_hit)?;
                write_label(
                    &mut writer,
                    &mut replay_digest,
                    &mut replay_size_bytes,
                    hit.label,
                )?;
                write_hashed(
                    &mut writer,
                    &mut replay_digest,
                    &mut replay_size_bytes,
                    &hit.distance.to_bits().to_le_bytes(),
                )?;
            }
            if (row_index + 1) % 100_000 == 0 || row_index + 1 == options.row_count {
                eprintln!(
                    "replay_rows={}/{} elapsed_seconds={:.3}",
                    row_index + 1,
                    options.row_count,
                    started.elapsed().as_secs_f64()
                );
            }
        }
        writer.flush()?;

        let query_sha256 = format!("{:x}", query_digest.finalize());
        let replay_sha256 = format!("{:x}", replay_digest.finalize());
        let actual_replay_size = fs_err::metadata(&replay_temp)?.len();
        if actual_replay_size != replay_size_bytes {
            return Err(format!(
                "replay size mismatch: wrote {replay_size_bytes}, filesystem reports {actual_replay_size}"
            )
            .into());
        }

        Ok(ReplayManifest {
            format_version: REPLAY_GATE_FORMAT_VERSION,
            verdict: "PASS",
            source: source_evidence,
            rebound: rebound_evidence,
            query_corpus: QueryCorpusEvidence {
                path: options.queries_path.display().to_string(),
                sha256: query_sha256,
                row_count: options.row_count,
                dimension: options.dimension,
                size_bytes: u64::try_from(actual_query_bytes)?,
            },
            replay: ReplayEvidence {
                path: options.replay_path.display().to_string(),
                encoding: "ORION_UPPER_REPLAY_V1: header magic,row_count_u64le,dimension_u64le,upper_k_u64le; row-major label(tag+u64le_or_uuid16)+distance_f32_bits_u32le",
                sha256: replay_sha256.clone(),
                size_bytes: replay_size_bytes,
                upper_k,
                compared_hit_count,
                ordered_label_and_distance_bits_sha256: replay_sha256,
            },
            gates: ReplayGates {
                generation_advanced: true,
                immutable_metadata_equal: true,
                vector_schema_equal: true,
                upper_search_contract_equal: true,
                canonical_upper_graph_bytes_equal: true,
                ordered_upper_labels_and_vector_bits_equal: true,
                ordered_hit_labels_equal: true,
                ordered_distance_bits_equal: true,
                production_router_replay_complete: true,
            },
            elapsed_seconds: started.elapsed().as_secs_f64(),
        })
    })();

    let manifest = match result {
        Ok(manifest) => manifest,
        Err(error) => {
            fs_err::remove_file(&replay_temp).ok();
            fs_err::remove_file(&manifest_temp).ok();
            return Err(error);
        }
    };

    let mut manifest_bytes = serde_json::to_vec_pretty(&manifest)?;
    manifest_bytes.push(b'\n');
    let manifest_write_result = (|| -> Result<(), Box<dyn Error>> {
        let mut manifest_writer = File::create_new(&manifest_temp)?;
        manifest_writer.write_all(&manifest_bytes)?;
        manifest_writer.sync_all()?;
        Ok(())
    })();
    if let Err(error) = manifest_write_result {
        fs_err::remove_file(&replay_temp).ok();
        fs_err::remove_file(&manifest_temp).ok();
        return Err(error);
    }

    if let Err(error) = publish_no_replace(&replay_temp, &options.replay_path) {
        fs_err::remove_file(&manifest_temp).ok();
        return Err(error);
    }
    if let Err(error) = publish_no_replace(&manifest_temp, &options.manifest_path) {
        fs_err::remove_file(&options.replay_path).ok();
        fs_err::remove_file(&manifest_temp).ok();
        return Err(error);
    }
    Ok(manifest)
}

fn main() -> Result<(), Box<dyn Error>> {
    let options = parse_args()?;
    let manifest = verify_replay(&options)?;
    println!("verdict={}", manifest.verdict);
    println!("replay={}", options.replay_path.display());
    println!("manifest={}", options.manifest_path.display());
    println!(
        "source_upper_graph_sha256={}",
        manifest.source.canonical_upper_graph_sha256
    );
    println!(
        "rebound_upper_graph_sha256={}",
        manifest.rebound.canonical_upper_graph_sha256
    );
    println!(
        "ordered_hit_label_and_distance_bits_sha256={}",
        manifest.replay.ordered_label_and_distance_bits_sha256
    );
    println!("compared_hit_count={}", manifest.replay.compared_hit_count);
    println!("elapsed_seconds={:.6}", manifest.elapsed_seconds);
    Ok(())
}

#[cfg(test)]
mod tests {
    use collection::orion::{
        ORION_ROUTING_ARTIFACT_FORMAT_VERSION, OrionUpperGraphNode, OrionUpperHnswGraph,
        OrionUpperNode, OrionVectorDatatype, OrionVectorSchemaFingerprint,
    };
    use segment::types::Distance;

    use super::*;

    fn artifact(rebound: bool) -> OrionRoutingArtifact {
        OrionRoutingArtifact {
            format_version: ORION_ROUTING_ARTIFACT_FORMAT_VERSION,
            generation: if rebound { 10 } else { 9 },
            vector_schema: OrionVectorSchemaFingerprint {
                vector_name: String::new(),
                dimension: 2,
                distance: Distance::Dot,
                datatype: OrionVectorDatatype::Float32,
            },
            shard_count: if rebound { 3 } else { 2 },
            layout_sha256: if rebound {
                "b".repeat(64)
            } else {
                "a".repeat(64)
            },
            logical_point_count: 10,
            physical_point_count: if rebound { 14 } else { 12 },
            upper_k: 2,
            upper_ef_search: 3,
            dynamic_ef_base: 20,
            dynamic_ef_factor: 4,
            upper_nodes: vec![
                OrionUpperNode {
                    label: 10_u64.into(),
                    vector: vec![1.0, -0.0],
                    shard_membership: if rebound { vec![2, 0] } else { vec![0, 1] },
                },
                OrionUpperNode {
                    label: 20_u64.into(),
                    vector: vec![0.0, 1.0],
                    shard_membership: if rebound { vec![1, 2] } else { vec![1] },
                },
                OrionUpperNode {
                    label: 30_u64.into(),
                    vector: vec![-1.0, 0.0],
                    shard_membership: if rebound { vec![0] } else { vec![0] },
                },
            ],
            upper_graph: Some(OrionUpperHnswGraph {
                entry_point: 10_u64.into(),
                max_level: 0,
                nodes: vec![
                    OrionUpperGraphNode {
                        label: 10_u64.into(),
                        neighbors_by_level: vec![vec![20_u64.into(), 30_u64.into()]],
                    },
                    OrionUpperGraphNode {
                        label: 20_u64.into(),
                        neighbors_by_level: vec![vec![10_u64.into(), 30_u64.into()]],
                    },
                    OrionUpperGraphNode {
                        label: 30_u64.into(),
                        neighbors_by_level: vec![vec![10_u64.into(), 20_u64.into()]],
                    },
                ],
            }),
        }
    }

    fn write_artifact(path: &Path, artifact: &OrionRoutingArtifact) {
        fs_err::write(path, artifact.canonical_json_bytes().unwrap()).unwrap();
    }

    fn write_queries(path: &Path, queries: &[[f32; 2]]) {
        let bytes = queries
            .iter()
            .flat_map(|query| query.iter())
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        fs_err::write(path, bytes).unwrap();
    }

    fn options(directory: &Path) -> Options {
        Options {
            source_artifact_path: directory.join("generation-9.json"),
            rebound_artifact_path: directory.join("generation-10.json"),
            queries_path: directory.join("queries.f32le"),
            row_count: 3,
            dimension: 2,
            replay_path: directory.join("upper-replay.bin"),
            manifest_path: directory.join("upper-replay-gate.json"),
        }
    }

    fn write_valid_inputs(options: &Options) {
        write_artifact(&options.source_artifact_path, &artifact(false));
        write_artifact(&options.rebound_artifact_path, &artifact(true));
        write_queries(
            &options.queries_path,
            &[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
        );
    }

    #[test]
    fn passes_bit_exact_replay_and_refuses_overwrite() {
        let directory = tempfile::tempdir().unwrap();
        let options = options(directory.path());
        write_valid_inputs(&options);

        let manifest = verify_replay(&options).unwrap();
        assert_eq!(manifest.verdict, "PASS");
        assert_eq!(manifest.replay.compared_hit_count, 6);
        assert_eq!(
            manifest.source.canonical_upper_graph_sha256,
            manifest.rebound.canonical_upper_graph_sha256
        );
        assert_eq!(
            manifest.source.ordered_upper_nodes_identity_sha256,
            manifest.rebound.ordered_upper_nodes_identity_sha256
        );
        assert!(manifest.gates.canonical_upper_graph_bytes_equal);
        assert!(manifest.gates.ordered_hit_labels_equal);
        assert!(manifest.gates.ordered_distance_bits_equal);
        assert_eq!(
            sha256_hex(&fs_err::read(&options.replay_path).unwrap()),
            manifest.replay.sha256
        );
        let stored: serde_json::Value =
            serde_json::from_slice(&fs_err::read(&options.manifest_path).unwrap()).unwrap();
        assert_eq!(stored["verdict"], "PASS");
        assert!(
            verify_replay(&options)
                .unwrap_err()
                .to_string()
                .contains("overwrite")
        );
    }

    #[test]
    fn fails_closed_when_graph_or_vector_bits_change() {
        for mutate_graph in [true, false] {
            let directory = tempfile::tempdir().unwrap();
            let options = options(directory.path());
            write_valid_inputs(&options);
            let mut changed = artifact(true);
            if mutate_graph {
                changed.upper_graph.as_mut().unwrap().nodes[0].neighbors_by_level[0].swap(0, 1);
            } else {
                changed.upper_nodes[0].vector[1] = 0.0;
            }
            write_artifact(&options.rebound_artifact_path, &changed);

            let error = verify_replay(&options).unwrap_err().to_string();
            assert!(
                error.contains(if mutate_graph {
                    "upper_graph bytes differ"
                } else {
                    "vector bits differ"
                }),
                "unexpected error: {error}"
            );
            assert!(!options.replay_path.exists());
            assert!(!options.manifest_path.exists());
        }
    }

    #[test]
    fn compares_ordered_labels_and_distance_bits_directly() {
        let source = OrionUpperHit {
            label: 10_u64.into(),
            distance: -0.0,
        };
        let wrong_label = OrionUpperHit {
            label: 20_u64.into(),
            distance: -0.0,
        };
        assert!(
            compare_hit(3, 1, source, wrong_label)
                .unwrap_err()
                .to_string()
                .contains("label mismatch")
        );

        let wrong_bits = OrionUpperHit {
            label: 10_u64.into(),
            distance: 0.0,
        };
        assert!(
            compare_hit(3, 1, source, wrong_bits)
                .unwrap_err()
                .to_string()
                .contains("distance-bit mismatch")
        );
    }

    #[test]
    fn fails_closed_when_generation_does_not_advance() {
        let directory = tempfile::tempdir().unwrap();
        let options = options(directory.path());
        write_valid_inputs(&options);
        let mut rebound = artifact(true);
        rebound.generation = 9;
        write_artifact(&options.rebound_artifact_path, &rebound);
        assert!(
            verify_replay(&options)
                .unwrap_err()
                .to_string()
                .contains("must be greater")
        );
        assert!(!options.replay_path.exists());
        assert!(!options.manifest_path.exists());
    }

    #[test]
    fn fails_closed_on_non_finite_query() {
        let directory = tempfile::tempdir().unwrap();
        let options = options(directory.path());
        write_valid_inputs(&options);
        write_queries(
            &options.queries_path,
            &[[1.0, 0.0], [f32::NAN, 1.0], [-1.0, 0.0]],
        );
        assert!(
            verify_replay(&options)
                .unwrap_err()
                .to_string()
                .contains("non-finite")
        );
        assert!(!options.replay_path.exists());
        assert!(!options.manifest_path.exists());
    }
}
