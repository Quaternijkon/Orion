use std::collections::BTreeMap;
use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::io::{BufReader, Read, Write};
use std::path::PathBuf;

use collection::orion::{OrionRouter, OrionRoutingArtifact, OrionVectorSchemaFingerprint};
use common::fs::atomic_save;
use fs_err::File;
use segment::types::ExtendedPointId;
use serde::Serialize;
use sha2::{Digest, Sha256};

fn usage(program: &str) -> String {
    format!(
        "Usage: {program} <artifact.json> <queries.f32le> <query-count> <dimension> <output.json> [--per-query]\n\
         \n\
         Replays row-major little-endian f32 queries through the production Orion router.\n\
         The output is an exact offline routing trace; it never contacts Qdrant or changes server state.\n\
         Existing output files are never overwritten."
    )
}

#[derive(Debug)]
struct TraceOptions {
    artifact_path: PathBuf,
    queries_path: PathBuf,
    query_count: usize,
    dimension: usize,
    output_path: PathBuf,
    include_per_query: bool,
}

#[derive(Debug, Serialize)]
struct TraceOutput {
    format_version: u32,
    artifact: ArtifactTraceMetadata,
    queries: QueryTraceMetadata,
    aggregate: AggregateTrace,
    #[serde(skip_serializing_if = "Option::is_none")]
    per_query: Option<Vec<PerQueryTrace>>,
}

#[derive(Debug, Serialize)]
struct ArtifactTraceMetadata {
    path: String,
    generation: u64,
    layout_sha256: String,
    /// SHA-256 used by the typed production artifact checksum contract.
    sha256: String,
    /// SHA-256 of the literal input file bytes. Canonical builder output makes this equal `sha256`.
    file_sha256: String,
    vector_schema: OrionVectorSchemaFingerprint,
    shard_count: u32,
    upper_k: usize,
    upper_ef_search: usize,
    dynamic_ef_base: usize,
    dynamic_ef_factor: usize,
}

#[derive(Debug, Serialize)]
struct QueryTraceMetadata {
    path: String,
    sha256: String,
    query_count: usize,
    dimension: usize,
}

#[derive(Debug, Serialize)]
struct AggregateTrace {
    query_count: usize,
    visited_shards: DistributionStats,
    entry_point_count: DistributionStats,
    ef_sum_per_query: DistributionStats,
    per_shard: Vec<PerShardTrace>,
}

#[derive(Debug, Serialize)]
struct DistributionStats {
    average: f64,
    min: u64,
    max: u64,
    p50: u64,
    p95: u64,
    p99: u64,
}

#[derive(Debug, Serialize)]
struct PerShardTrace {
    shard_id: u32,
    query_visits: u64,
    average_ef_when_visited: f64,
    average_entry_points_when_visited: f64,
}

#[derive(Debug, Serialize)]
struct PerQueryTrace {
    query_index: usize,
    visited_shards: u64,
    entry_point_count: u64,
    ef_sum: u64,
    targets: Vec<PerQueryShardTarget>,
}

#[derive(Debug, Serialize)]
struct PerQueryShardTarget {
    shard_id: u32,
    ef: usize,
    entry_points: Vec<ExtendedPointId>,
}

#[derive(Default)]
struct ShardAccumulator {
    query_visits: u64,
    ef_sum: u64,
    entry_point_sum: u64,
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

fn parse_args() -> Result<TraceOptions, Box<dyn Error>> {
    let mut args = env::args_os();
    let program = args
        .next()
        .unwrap_or_else(|| OsString::from("orion_route_trace"))
        .to_string_lossy()
        .into_owned();
    let Some(artifact_path) = args.next() else {
        return Err(usage(&program).into());
    };
    if artifact_path == "--help" || artifact_path == "-h" {
        println!("{}", usage(&program));
        std::process::exit(0);
    }
    let queries_path = args.next().ok_or_else(|| usage(&program))?;
    let query_count =
        parse_positive_usize("query-count", args.next().ok_or_else(|| usage(&program))?)?;
    let dimension = parse_positive_usize("dimension", args.next().ok_or_else(|| usage(&program))?)?;
    let output_path = args.next().ok_or_else(|| usage(&program))?;
    let mut include_per_query = false;
    for option in args {
        match option.to_string_lossy().as_ref() {
            "--per-query" => include_per_query = true,
            "--help" | "-h" => {
                println!("{}", usage(&program));
                std::process::exit(0);
            }
            unknown => {
                return Err(format!("unknown option {unknown:?}\n{}", usage(&program)).into());
            }
        }
    }
    Ok(TraceOptions {
        artifact_path: artifact_path.into(),
        queries_path: queries_path.into(),
        query_count,
        dimension,
        output_path: output_path.into(),
        include_per_query,
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn distribution(values: &[u64]) -> DistributionStats {
    debug_assert!(!values.is_empty());
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let total = sorted.iter().map(|&value| value as f64).sum::<f64>();
    DistributionStats {
        average: total / sorted.len() as f64,
        min: sorted[0],
        max: *sorted.last().unwrap(),
        p50: nearest_rank(&sorted, 50),
        p95: nearest_rank(&sorted, 95),
        p99: nearest_rank(&sorted, 99),
    }
}

fn nearest_rank(sorted: &[u64], percentile: usize) -> u64 {
    debug_assert!(!sorted.is_empty());
    debug_assert!((1..=100).contains(&percentile));
    let rank = percentile.saturating_mul(sorted.len()).div_ceil(100);
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

fn checked_query_bytes(query_count: usize, dimension: usize) -> Result<usize, Box<dyn Error>> {
    query_count
        .checked_mul(dimension)
        .and_then(|elements| elements.checked_mul(size_of::<f32>()))
        .ok_or_else(|| "query_count and dimension overflow the expected query file size".into())
}

fn generate_trace(options: &TraceOptions) -> Result<TraceOutput, Box<dyn Error>> {
    if options.output_path.exists() {
        return Err(format!(
            "refusing to overwrite existing trace {}",
            options.output_path.display()
        )
        .into());
    }

    let artifact_bytes = fs_err::read(&options.artifact_path)?;
    let artifact_file_sha256 = sha256_hex(&artifact_bytes);
    let artifact = OrionRoutingArtifact::from_json_slice(&artifact_bytes, None)?;
    if artifact.vector_schema.dimension != options.dimension {
        return Err(format!(
            "CLI dimension {} does not match artifact dimension {}",
            options.dimension, artifact.vector_schema.dimension
        )
        .into());
    }
    let artifact_metadata = ArtifactTraceMetadata {
        path: options.artifact_path.display().to_string(),
        generation: artifact.generation,
        layout_sha256: artifact.layout_sha256.clone(),
        sha256: artifact.canonical_sha256()?,
        file_sha256: artifact_file_sha256,
        vector_schema: artifact.vector_schema.clone(),
        shard_count: artifact.shard_count,
        upper_k: artifact.upper_k,
        upper_ef_search: artifact.upper_ef_search,
        dynamic_ef_base: artifact.dynamic_ef_base,
        dynamic_ef_factor: artifact.dynamic_ef_factor,
    };
    // This production constructor rejects graphless or structurally incomplete artifacts.
    let router = OrionRouter::new(artifact)?;

    let expected_query_bytes = checked_query_bytes(options.query_count, options.dimension)?;
    let actual_query_bytes = usize::try_from(fs_err::metadata(&options.queries_path)?.len())
        .map_err(|_| "query file is too large for this platform")?;
    if actual_query_bytes != expected_query_bytes {
        return Err(format!(
            "query file {} has {actual_query_bytes} bytes; expected exactly {expected_query_bytes} ({} queries x {} dimensions x 4)",
            options.queries_path.display(),
            options.query_count,
            options.dimension
        )
        .into());
    }

    let mut reader = BufReader::new(File::open(&options.queries_path)?);
    let mut row_bytes = vec![0_u8; options.dimension * size_of::<f32>()];
    let mut query_digest = Sha256::new();
    let mut visited_shards = Vec::with_capacity(options.query_count);
    let mut entry_point_counts = Vec::with_capacity(options.query_count);
    let mut ef_sums = Vec::with_capacity(options.query_count);
    let mut shard_accumulators: BTreeMap<u32, ShardAccumulator> = BTreeMap::new();
    let mut per_query = options
        .include_per_query
        .then(|| Vec::with_capacity(options.query_count));

    for query_index in 0..options.query_count {
        reader.read_exact(&mut row_bytes)?;
        query_digest.update(&row_bytes);
        let query = row_bytes
            .chunks_exact(size_of::<f32>())
            .enumerate()
            .map(|(dimension, bytes)| {
                let value = f32::from_le_bytes(bytes.try_into().unwrap());
                if value.is_finite() {
                    Ok(value)
                } else {
                    Err(format!(
                        "query {query_index} has a non-finite component at dimension {dimension}"
                    ))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let targets = router
            .route_query(&query)
            .map_err(|error| format!("failed to route query {query_index}: {error}"))?;

        let query_visited = u64::try_from(targets.len())?;
        let query_entry_points = targets.iter().try_fold(0_u64, |total, target| {
            total.checked_add(u64::try_from(target.entry_points.len()).ok()?)
        });
        let query_entry_points = query_entry_points.ok_or("entry-point count overflowed u64")?;
        let query_ef_sum = targets.iter().try_fold(0_u64, |total, target| {
            total.checked_add(u64::try_from(target.ef).ok()?)
        });
        let query_ef_sum = query_ef_sum.ok_or("EF sum overflowed u64")?;

        for target in &targets {
            let accumulator = shard_accumulators.entry(target.shard_id).or_default();
            accumulator.query_visits = accumulator
                .query_visits
                .checked_add(1)
                .ok_or("per-shard visit count overflowed u64")?;
            accumulator.ef_sum = accumulator
                .ef_sum
                .checked_add(u64::try_from(target.ef)?)
                .ok_or("per-shard EF sum overflowed u64")?;
            accumulator.entry_point_sum = accumulator
                .entry_point_sum
                .checked_add(u64::try_from(target.entry_points.len())?)
                .ok_or("per-shard entry-point sum overflowed u64")?;
        }

        visited_shards.push(query_visited);
        entry_point_counts.push(query_entry_points);
        ef_sums.push(query_ef_sum);
        if let Some(per_query) = &mut per_query {
            per_query.push(PerQueryTrace {
                query_index,
                visited_shards: query_visited,
                entry_point_count: query_entry_points,
                ef_sum: query_ef_sum,
                targets: targets
                    .into_iter()
                    .map(|target| PerQueryShardTarget {
                        shard_id: target.shard_id,
                        ef: target.ef,
                        entry_points: target.entry_points,
                    })
                    .collect(),
            });
        }
    }

    let per_shard = shard_accumulators
        .into_iter()
        .map(|(shard_id, accumulator)| PerShardTrace {
            shard_id,
            query_visits: accumulator.query_visits,
            average_ef_when_visited: accumulator.ef_sum as f64 / accumulator.query_visits as f64,
            average_entry_points_when_visited: accumulator.entry_point_sum as f64
                / accumulator.query_visits as f64,
        })
        .collect();
    Ok(TraceOutput {
        format_version: 1,
        artifact: artifact_metadata,
        queries: QueryTraceMetadata {
            path: options.queries_path.display().to_string(),
            sha256: format!("{:x}", query_digest.finalize()),
            query_count: options.query_count,
            dimension: options.dimension,
        },
        aggregate: AggregateTrace {
            query_count: options.query_count,
            visited_shards: distribution(&visited_shards),
            entry_point_count: distribution(&entry_point_counts),
            ef_sum_per_query: distribution(&ef_sums),
            per_shard,
        },
        per_query,
    })
}

fn write_trace(options: &TraceOptions) -> Result<TraceOutput, Box<dyn Error>> {
    let trace = generate_trace(options)?;
    if let Some(parent) = options
        .output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs_err::create_dir_all(parent)?;
    }
    let json = serde_json::to_vec_pretty(&trace)?;
    atomic_save(&options.output_path, |writer| writer.write_all(&json))?;
    Ok(trace)
}

fn main() -> Result<(), Box<dyn Error>> {
    let options = parse_args()?;
    let trace = write_trace(&options)?;
    println!("trace={}", options.output_path.display());
    println!("queries={}", trace.aggregate.query_count);
    println!(
        "visited_shards_avg={:.6}",
        trace.aggregate.visited_shards.average
    );
    println!(
        "entry_point_count_avg={:.6}",
        trace.aggregate.entry_point_count.average
    );
    println!(
        "ef_sum_per_query_avg={:.6}",
        trace.aggregate.ef_sum_per_query.average
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use collection::orion::{
        ORION_ROUTING_ARTIFACT_FORMAT_VERSION, OrionRoutingArtifact, OrionUpperGraphNode,
        OrionUpperHnswGraph, OrionUpperNode, OrionVectorDatatype, OrionVectorSchemaFingerprint,
    };
    use segment::types::Distance;

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
            logical_point_count: 2,
            physical_point_count: 3,
            upper_k: 1,
            upper_ef_search: 2,
            dynamic_ef_base: 20,
            dynamic_ef_factor: 4,
            upper_nodes: vec![
                OrionUpperNode {
                    label: 10_u64.into(),
                    vector: vec![1.0, 0.0],
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

    fn write_queries(path: &Path, queries: &[[f32; 2]]) {
        let bytes = queries
            .iter()
            .flat_map(|query| query.iter())
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        fs_err::write(path, bytes).unwrap();
    }

    #[test]
    fn production_route_trace_reports_exact_route_metrics_and_refuses_overwrite() {
        let directory = tempfile::tempdir().unwrap();
        let artifact_path = directory.path().join("generation-9.json");
        let queries_path = directory.path().join("queries.f32le");
        let output_path = directory.path().join("trace.json");
        fs_err::write(
            &artifact_path,
            artifact(true).canonical_json_bytes().unwrap(),
        )
        .unwrap();
        write_queries(&queries_path, &[[1.0, 0.0], [0.0, 1.0]]);
        let options = TraceOptions {
            artifact_path,
            queries_path,
            query_count: 2,
            dimension: 2,
            output_path,
            include_per_query: true,
        };

        let trace = write_trace(&options).unwrap();
        assert_eq!(trace.artifact.generation, 9);
        assert_eq!(trace.artifact.layout_sha256, "a".repeat(64));
        assert_eq!(trace.artifact.sha256, trace.artifact.file_sha256);
        assert_eq!(trace.aggregate.visited_shards.average, 1.5);
        assert_eq!(trace.aggregate.visited_shards.min, 1);
        assert_eq!(trace.aggregate.visited_shards.max, 2);
        assert_eq!(trace.aggregate.visited_shards.p50, 1);
        assert_eq!(trace.aggregate.visited_shards.p95, 2);
        assert_eq!(trace.aggregate.entry_point_count.average, 1.5);
        assert_eq!(trace.aggregate.ef_sum_per_query.average, 36.0);
        assert_eq!(trace.aggregate.per_shard.len(), 2);
        assert_eq!(trace.aggregate.per_shard[0].query_visits, 1);
        assert_eq!(trace.aggregate.per_shard[0].average_ef_when_visited, 24.0);
        assert_eq!(trace.aggregate.per_shard[1].query_visits, 2);
        assert_eq!(trace.per_query.as_ref().unwrap().len(), 2);
        assert!(options.output_path.exists());
        assert!(
            write_trace(&options)
                .unwrap_err()
                .to_string()
                .contains("refusing to overwrite")
        );
    }

    #[test]
    fn route_trace_rejects_graphless_bad_length_dimension_and_non_finite_queries() {
        let directory = tempfile::tempdir().unwrap();
        let artifact_path = directory.path().join("artifact.json");
        let queries_path = directory.path().join("queries.f32le");
        fs_err::write(
            &artifact_path,
            artifact(false).canonical_json_bytes().unwrap(),
        )
        .unwrap();
        write_queries(&queries_path, &[[1.0, 0.0]]);
        let mut options = TraceOptions {
            artifact_path: artifact_path.clone(),
            queries_path: queries_path.clone(),
            query_count: 1,
            dimension: 2,
            output_path: directory.path().join("graphless.json"),
            include_per_query: false,
        };
        assert!(
            generate_trace(&options)
                .unwrap_err()
                .to_string()
                .contains("upper HNSW graph is required")
        );

        fs_err::write(
            &artifact_path,
            artifact(true).canonical_json_bytes().unwrap(),
        )
        .unwrap();
        options.dimension = 3;
        assert!(
            generate_trace(&options)
                .unwrap_err()
                .to_string()
                .contains("does not match artifact dimension")
        );

        options.dimension = 2;
        options.query_count = 2;
        assert!(
            generate_trace(&options)
                .unwrap_err()
                .to_string()
                .contains("expected exactly")
        );

        options.query_count = 1;
        write_queries(&queries_path, &[[f32::NAN, 0.0]]);
        assert!(
            generate_trace(&options)
                .unwrap_err()
                .to_string()
                .contains("non-finite component")
        );
    }
}
