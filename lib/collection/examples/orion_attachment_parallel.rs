//! Parallel attachment generation over one immutable production graph.
//! Every record retains the canonical u32 count + ordered u64 labels encoding.
use anyhow::{Result, ensure};
use collection::orion::{
    OrionRouter, OrionRoutingArtifact, bundle, layout, partition, whitebox_pruning,
};
use segment::types::ExtendedPointId;
use serde::Deserialize;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::PathBuf;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Export {
    artifact: PathBuf,
    vectors: PathBuf,
    rows: usize,
    dimension: usize,
    efs: usize,
    threads: usize,
    output: PathBuf,
}

fn export(r: Export) -> Result<()> {
    ensure!(
        r.rows > 0 && r.dimension > 0 && r.efs > 0 && r.threads > 0,
        "empty export"
    );
    ensure!(!r.output.exists(), "output exists");
    ensure!(
        fs::metadata(&r.vectors)?.len() == r.rows as u64 * r.dimension as u64 * 4,
        "vector size mismatch"
    );
    let artifact_sha =
        bundle::file_sha256(&r.artifact).map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let mut artifact = OrionRoutingArtifact::read_json(&r.artifact, None)?;
    ensure!(
        artifact.vector_schema.dimension == r.dimension
            && artifact.shard_count == 1
            && artifact.upper_nodes.iter().all(|n| n.owner_shard == 0),
        "expected frozen neutral graph"
    );
    artifact.upper_k = r.efs;
    artifact.upper_ef_search = r.efs;
    let distance = artifact.vector_schema.distance;
    // All hits have owner zero; this mandatory online rule cannot remove an
    // entry. route_query uses the identical production scratch search, retaining
    // hit order, while giving each thread an exclusive reusable workspace.
    let router = OrionRouter::new(artifact)?.with_whitebox(whitebox_pruning::Config {
        prefix_k: 1,
        max_prefix_entries: 0,
        max_distance_gap: 0.,
        distance,
        enabled_generations: None,
    })?;
    fs::create_dir_all(&r.output)?;
    std::thread::scope(|scope| -> Result<()> {
        let mut jobs = Vec::new();
        for t in 0..r.threads.min(r.rows) {
            let router = &router;
            let r = &r;
            jobs.push(scope.spawn(move || -> Result<()> {
                let threads = r.threads.min(r.rows);
                let start = r.rows * t / threads;
                let end = r.rows * (t + 1) / threads;
                let mut source = BufReader::new(File::open(&r.vectors)?);
                source.seek(SeekFrom::Start(start as u64 * r.dimension as u64 * 4))?;
                let mut output =
                    BufWriter::new(File::create_new(r.output.join(format!("part-{t:03}.bin")))?);
                let mut bytes = vec![0; r.dimension * 4];
                for row in start..end {
                    source.read_exact(&mut bytes)?;
                    let query: Vec<f32> = bytes
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                        .collect();
                    let targets = router.route_query(&query)?;
                    ensure!(
                        targets.len() == 1
                            && targets[0].shard_id == 0
                            && targets[0].entry_points.len() == r.efs,
                        "attachment route mismatch"
                    );
                    output.write_all(&(r.efs as u32).to_le_bytes())?;
                    for label in &targets[0].entry_points {
                        let ExtendedPointId::NumId(id) = label else {
                            anyhow::bail!("numeric IDs required")
                        };
                        output.write_all(&id.to_le_bytes())?;
                    }
                    if (row - start + 1) % 100_000 == 0 {
                        eprintln!("thread={t} rows={}/{}", row - start + 1, end - start);
                    }
                }
                output.flush()?;
                output.get_ref().sync_all()?;
                Ok(())
            }));
        }
        for job in jobs {
            job.join().map_err(|_| anyhow::anyhow!("worker panic"))??;
        }
        Ok(())
    })?;
    let hits = r.output.join("hits.bin");
    let mut output = File::create_new(&hits)?;
    for t in 0..r.threads.min(r.rows) {
        let path = r.output.join(format!("part-{t:03}.bin"));
        std::io::copy(&mut File::open(&path)?, &mut output)?;
        fs::remove_file(path)?;
    }
    output.sync_all()?;
    ensure!(
        output.metadata()?.len() == r.rows as u64 * (4 + 8 * r.efs) as u64,
        "record size mismatch"
    );
    let report = serde_json::json!({"complete":true,"rows":r.rows,"dimension":r.dimension,"efs":r.efs,
        "threads":r.threads,"artifact_sha256":artifact_sha,
        "vectors_sha256":bundle::file_sha256(&r.vectors).map_err(|e|anyhow::anyhow!(e.to_string()))?,
        "hits_sha256":bundle::file_sha256(&hits).map_err(|e|anyhow::anyhow!(e.to_string()))?});
    serde_json::to_writer_pretty(File::create_new(r.output.join("manifest.json"))?, &report)?;
    Ok(())
}

#[derive(Deserialize)]
struct Finalize {
    request: PathBuf,
    directory: PathBuf,
}
fn finalize(r: Finalize) -> Result<()> {
    let request: bundle::BuildRequest = serde_json::from_reader(File::open(r.request)?)?;
    let options = request.options;
    let source_name = format!("upper-source-generation-{}.json", options.generation);
    let source = OrionRoutingArtifact::read_json(r.directory.join(&source_name), None)?;
    let file_sha = |name: &str| {
        bundle::file_sha256(&r.directory.join(name)).map_err(|e| anyhow::anyhow!(e.to_string()))
    };
    let prepared = partition::PreparedCapacity {
        format_version: 1,
        point_count: request.input.rows,
        dimension: request.input.dimension,
        vector_encoding: layout::VectorEncoding::F32le,
        raw_vectors_sha256: request.input.sha256,
        vectors_file: format!("{}.f32le", options.bundle_prefix),
        vectors_sha256: file_sha(&format!("{}.f32le", options.bundle_prefix))?,
        upper_indices: layout::sample_indices(
            request.input.rows,
            options.sample_denominator,
            options.sample_seed,
        )
        .map_err(|e| anyhow::anyhow!(e.to_string()))?,
        source_artifact_sha256: file_sha(&source_name)?,
        source_artifact_file: source_name,
        upper_graph_sha256: bundle::graph_sha256(&source)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?,
        upper_identity_sha256: bundle::upper_identity_sha256(&source)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?,
        attachments_file: "upper-attachments.counted.bin".into(),
        attachments_sha256: file_sha("upper-attachments.counted.bin")?,
        options,
    };
    let mut manifest = BufWriter::new(File::create_new(r.directory.join("capacity-input.json"))?);
    serde_json::to_writer(&mut manifest, &prepared)?;
    manifest.flush()?;
    manifest.get_ref().sync_all()?;
    drop(manifest);
    drop(source);
    partition::read_prepared(&r.directory).map_err(|e| anyhow::anyhow!(e.to_string()))?;
    fs::write(
        r.directory.join("validated.json"),
        b"{\"validated\":true}\n",
    )?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        args.len() == 3,
        "Usage: orion_attachment_parallel export|finalize request.json"
    );
    let reader = BufReader::new(File::open(&args[2])?);
    match args[1].as_str() {
        "export" => export(serde_json::from_reader(reader)?),
        "finalize" => finalize(serde_json::from_reader(reader)?),
        _ => anyhow::bail!("unknown command"),
    }
}
