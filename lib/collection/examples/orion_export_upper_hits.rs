use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use collection::orion::{OrionRouter, OrionRoutingArtifact};
use fs_err::File;
use segment::types::ExtendedPointId;
use serde::Serialize;
use sha2::{Digest, Sha256};

fn usage(program: &str) -> String {
    format!(
        "Usage: {program} <artifact.json> <vectors.f32le> <row-count> <dimension> <top-k> <search-ef> \
         <hits.bin> <manifest.json>\n\
         \n\
         Export up to top-k production upper-HNSW hits for every input row.\n\
         Each row is encoded as a little-endian u32 count followed by that many little-endian\n\
         u64 point IDs. Existing outputs are never overwritten."
    )
}

#[derive(Debug)]
struct Options {
    artifact_path: PathBuf,
    vectors_path: PathBuf,
    row_count: usize,
    dimension: usize,
    top_k: usize,
    search_ef: usize,
    hits_path: PathBuf,
    manifest_path: PathBuf,
}

#[derive(Debug, Serialize)]
struct ExportManifest {
    format_version: u32,
    artifact_path: String,
    artifact_sha256: String,
    generation: u64,
    upper_graph_present: bool,
    source_upper_k: usize,
    source_upper_ef_search: usize,
    vectors_path: String,
    vectors_sha256: String,
    row_count: usize,
    dimension: usize,
    top_k: usize,
    search_ef: usize,
    hits_path: String,
    hits_format: &'static str,
    hits_sha256: String,
    hits_size_bytes: u64,
    total_hits: u64,
    min_hits_per_row: usize,
    max_hits_per_row: usize,
    elapsed_seconds: f64,
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
        .unwrap_or_else(|| OsString::from("orion_export_upper_hits"))
        .to_string_lossy()
        .into_owned();
    let Some(artifact_path) = args.next() else {
        return Err(usage(&program).into());
    };
    if artifact_path == "--help" || artifact_path == "-h" {
        println!("{}", usage(&program));
        std::process::exit(0);
    }
    let vectors_path = args.next().ok_or_else(|| usage(&program))?;
    let row_count = parse_positive_usize("row-count", args.next().ok_or_else(|| usage(&program))?)?;
    let dimension = parse_positive_usize("dimension", args.next().ok_or_else(|| usage(&program))?)?;
    let top_k = parse_positive_usize("top-k", args.next().ok_or_else(|| usage(&program))?)?;
    let search_ef = parse_positive_usize("search-ef", args.next().ok_or_else(|| usage(&program))?)?;
    let hits_path = args.next().ok_or_else(|| usage(&program))?;
    let manifest_path = args.next().ok_or_else(|| usage(&program))?;
    if args.next().is_some() {
        return Err(usage(&program).into());
    }
    Ok(Options {
        artifact_path: artifact_path.into(),
        vectors_path: vectors_path.into(),
        row_count,
        dimension,
        top_k,
        search_ef,
        hits_path: hits_path.into(),
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

fn numeric_id(id: ExtendedPointId) -> Result<u64, Box<dyn Error>> {
    match id {
        ExtendedPointId::NumId(value) => Ok(value),
        ExtendedPointId::Uuid(value) => {
            Err(format!("upper label {value} is a UUID; u64 export requires numeric labels").into())
        }
    }
}

fn export(options: &Options) -> Result<ExportManifest, Box<dyn Error>> {
    for output in [&options.hits_path, &options.manifest_path] {
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

    let artifact_bytes = fs_err::read(&options.artifact_path)?;
    let artifact_sha256 = sha256_hex(&artifact_bytes);
    let mut artifact = OrionRoutingArtifact::from_json_slice(&artifact_bytes, None)?;
    if artifact.vector_schema.dimension != options.dimension {
        return Err(format!(
            "dimension {} does not match artifact dimension {}",
            options.dimension, artifact.vector_schema.dimension
        )
        .into());
    }
    if options.top_k > artifact.upper_nodes.len() {
        return Err(format!(
            "top-k {} exceeds upper-node count {}",
            options.top_k,
            artifact.upper_nodes.len()
        )
        .into());
    }
    if options.search_ef != options.top_k {
        return Err(format!(
            "canonical Orion requires search-ef {} to equal top-k {}",
            options.search_ef, options.top_k
        )
        .into());
    }
    let generation = artifact.generation;
    let upper_graph_present = artifact.upper_graph.is_some();
    let source_upper_k = artifact.upper_k;
    let source_upper_ef_search = artifact.upper_ef_search;
    // Partition/assignment construction deliberately uses one budget: efs is both
    // the HNSW search EF and the number of returned voting results. Runtime upper
    // routing may use a different explicitly bound budget without changing graph bytes.
    artifact.upper_k = options.top_k;
    artifact.upper_ef_search = options.search_ef;
    let router = OrionRouter::new(artifact)?;

    let expected_vector_bytes =
        checked_size(options.row_count, options.dimension, size_of::<f32>())?;
    let actual_vector_bytes = usize::try_from(fs_err::metadata(&options.vectors_path)?.len())?;
    if actual_vector_bytes != expected_vector_bytes {
        return Err(format!(
            "{} has {actual_vector_bytes} bytes; expected {expected_vector_bytes}",
            options.vectors_path.display()
        )
        .into());
    }

    let hits_tmp = temporary_path(&options.hits_path);
    if hits_tmp.exists() {
        return Err(format!("temporary output already exists: {}", hits_tmp.display()).into());
    }
    let started = Instant::now();
    let result = (|| -> Result<(String, String, u64, usize, usize), Box<dyn Error>> {
        let mut reader = BufReader::new(File::open(&options.vectors_path)?);
        let mut writer = BufWriter::new(File::create_new(&hits_tmp)?);
        let mut row_bytes = vec![0_u8; options.dimension * size_of::<f32>()];
        let mut vectors_digest = Sha256::new();
        let mut hits_digest = Sha256::new();
        let mut total_hits = 0_u64;
        let mut min_hits_per_row = usize::MAX;
        let mut max_hits_per_row = 0_usize;

        for row_index in 0..options.row_count {
            reader.read_exact(&mut row_bytes)?;
            vectors_digest.update(&row_bytes);
            let query = row_bytes
                .chunks_exact(size_of::<f32>())
                .enumerate()
                .map(|(column, bytes)| {
                    let value = f32::from_le_bytes(bytes.try_into().unwrap());
                    if value.is_finite() {
                        Ok(value)
                    } else {
                        Err(format!(
                            "row {row_index} has a non-finite component at column {column}"
                        ))
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            let hits = router.search_upper(&query)?;
            let hit_count = hits.len().min(options.top_k);
            let hit_count_u32 = u32::try_from(hit_count)?;
            let count_bytes = hit_count_u32.to_le_bytes();
            writer.write_all(&count_bytes)?;
            hits_digest.update(count_bytes);
            total_hits = total_hits
                .checked_add(u64::from(hit_count_u32))
                .ok_or("total exported upper-hit count overflowed u64")?;
            min_hits_per_row = min_hits_per_row.min(hit_count);
            max_hits_per_row = max_hits_per_row.max(hit_count);
            for hit in hits.into_iter().take(hit_count) {
                let bytes = numeric_id(hit.label)?.to_le_bytes();
                writer.write_all(&bytes)?;
                hits_digest.update(bytes);
            }
            if (row_index + 1) % 100_000 == 0 || row_index + 1 == options.row_count {
                eprintln!(
                    "rows={}/{} elapsed_seconds={:.3}",
                    row_index + 1,
                    options.row_count,
                    started.elapsed().as_secs_f64()
                );
            }
        }
        writer.flush()?;
        Ok((
            format!("{:x}", vectors_digest.finalize()),
            format!("{:x}", hits_digest.finalize()),
            total_hits,
            min_hits_per_row,
            max_hits_per_row,
        ))
    })();

    let (vectors_sha256, hits_sha256, total_hits, min_hits_per_row, max_hits_per_row) = match result
    {
        Ok(digests) => digests,
        Err(error) => {
            fs_err::remove_file(&hits_tmp).ok();
            return Err(error);
        }
    };
    fs_err::rename(&hits_tmp, &options.hits_path)?;
    let hits_size_bytes = fs_err::metadata(&options.hits_path)?.len();
    let count_bytes = u64::try_from(checked_size(options.row_count, 1, size_of::<u32>())?)?;
    let id_bytes = total_hits
        .checked_mul(u64::try_from(size_of::<u64>())?)
        .ok_or("exported upper-hit byte size overflowed u64")?;
    let expected_hits_bytes = count_bytes
        .checked_add(id_bytes)
        .ok_or("exported upper-hit byte size overflowed u64")?;
    if hits_size_bytes != expected_hits_bytes {
        return Err(format!(
            "hit output has {hits_size_bytes} bytes; expected {expected_hits_bytes}"
        )
        .into());
    }

    let manifest = ExportManifest {
        format_version: 2,
        artifact_path: options.artifact_path.display().to_string(),
        artifact_sha256,
        generation,
        upper_graph_present,
        source_upper_k,
        source_upper_ef_search,
        vectors_path: options.vectors_path.display().to_string(),
        vectors_sha256,
        row_count: options.row_count,
        dimension: options.dimension,
        top_k: options.top_k,
        search_ef: options.search_ef,
        hits_path: options.hits_path.display().to_string(),
        hits_format: "counted_rows_u32le_then_u64le_v1",
        hits_sha256,
        hits_size_bytes,
        total_hits,
        min_hits_per_row,
        max_hits_per_row,
        elapsed_seconds: started.elapsed().as_secs_f64(),
    };
    let manifest_tmp = temporary_path(&options.manifest_path);
    let mut bytes = serde_json::to_vec_pretty(&manifest)?;
    bytes.push(b'\n');
    fs_err::write(&manifest_tmp, bytes)?;
    fs_err::rename(&manifest_tmp, &options.manifest_path)?;
    Ok(manifest)
}

fn main() -> Result<(), Box<dyn Error>> {
    let options = parse_args()?;
    let manifest = export(&options)?;
    println!("hits={}", options.hits_path.display());
    println!("manifest={}", options.manifest_path.display());
    println!("rows={}", manifest.row_count);
    println!("top_k={}", manifest.top_k);
    println!("elapsed_seconds={:.6}", manifest.elapsed_seconds);
    Ok(())
}
