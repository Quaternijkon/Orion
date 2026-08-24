use std::collections::HashSet;
use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use common::bitvec::BitVec;
use common::counter::hardware_counter::HardwareCounterCell;
use common::types::PointOffsetType;
use rand::SeedableRng;
use rand::rngs::StdRng;
use segment::data_types::vectors::{VectorElementType, VectorRef};
use segment::index::hnsw_index::HnswM;
use segment::index::hnsw_index::graph_layers::{GraphLayers, SearchAlgorithm};
use segment::index::hnsw_index::graph_layers_builder::GraphLayersBuilder;
use segment::index::hnsw_index::graph_links::GraphLinksFormatParam;
use segment::index::hnsw_index::point_scorer::FilteredScorer;
use segment::types::Distance;
use segment::vector_storage::dense::volatile_dense_vector_storage::new_volatile_dense_vector_storage;
use segment::vector_storage::{VectorStorage, VectorStorageEnum};
use serde::Serialize;
use sha2::{Digest, Sha256};

const FORMAT_VERSION: u32 = 1;

#[derive(Debug)]
struct Args {
    vectors: PathBuf,
    queries: PathBuf,
    ground_truth: PathBuf,
    point_ids: Option<PathBuf>,
    output_dir: PathBuf,
    dimension: usize,
    distance: Distance,
    top: usize,
    ground_truth_width: usize,
    tuning_count: usize,
    measurement_start: usize,
    measurement_count: Option<usize>,
    ef_values: Vec<usize>,
    target_recall: f64,
    hnsw_m: usize,
    ef_construction: usize,
    full_scan_threshold_kb: usize,
    graph_seed: u64,
}

#[derive(Debug)]
struct GraphBuildStats {
    construction_seconds: f64,
    average_vector_bytes: usize,
    full_scan_threshold_points: usize,
    entry_points_num: usize,
}

#[derive(Debug, Serialize)]
struct TuningRow {
    ef_search: usize,
    query_count: usize,
    recall_at_k: f64,
    distance_computations: u64,
    graph_nodes_visited: u64,
}

fn usage(program: &str) -> String {
    format!(
        "Usage: {program} --vectors FILE --queries FILE --ground-truth FILE \\\n+         --dimension N --distance euclid|cosine --output-dir DIR [options]\n\
         \n\
         Required raw formats:\n\
           vectors/queries: contiguous little-endian float32 rows\n\
           ground-truth: contiguous little-endian uint32 rows\n\
           point IDs: optional contiguous little-endian uint32 values\n\
         \n\
         Options:\n\
           --point-ids FILE            external IDs; default 0..N-1\n\
           --top N                     result/recall K (default 10)\n\
           --ground-truth-width N      truth row width (default 10)\n\
           --tuning-count N            leading tuning queries (default 1000)\n\
           --measurement-start N       first measurement query (default 1000)\n\
           --measurement-count N       measurement rows; default remaining rows\n\
           --ef-values LIST            comma-separated tuning grid\n\
           --target-recall R           selection gate (default 0.90)\n\
           --m N                       HNSW M (default 32)\n\
           --ef-construction N         HNSW construction EF (default 200)\n\
           --full-scan-threshold-kb N  production HNSW threshold (default 10)\n\
           --graph-seed N              deterministic level seed (default 20260821)"
    )
}

fn parse_value<T>(flag: &str, value: Option<OsString>) -> Result<T, Box<dyn Error>>
where
    T: std::str::FromStr,
    T::Err: Error + 'static,
{
    let value = value.ok_or_else(|| format!("{flag} requires a value"))?;
    Ok(value
        .into_string()
        .map_err(|_| format!("{flag} value is not valid UTF-8"))?
        .parse::<T>()?)
}

fn parse_ef_values(value: Option<OsString>) -> Result<Vec<usize>, Box<dyn Error>> {
    let value = value
        .ok_or("--ef-values requires a value")?
        .into_string()
        .map_err(|_| "--ef-values is not valid UTF-8")?;
    let mut values = value
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(str::parse::<usize>)
        .collect::<Result<Vec<_>, _>>()?;
    values.sort_unstable();
    values.dedup();
    if values.is_empty() || values[0] == 0 {
        return Err("--ef-values must contain positive integers".into());
    }
    Ok(values)
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut raw = env::args_os();
    let program = raw
        .next()
        .unwrap_or_else(|| OsString::from("c23_hnsw_reference"))
        .to_string_lossy()
        .into_owned();
    let mut args = Args {
        vectors: PathBuf::new(),
        queries: PathBuf::new(),
        ground_truth: PathBuf::new(),
        point_ids: None,
        output_dir: PathBuf::new(),
        dimension: 0,
        distance: Distance::Euclid,
        top: 10,
        ground_truth_width: 10,
        tuning_count: 1000,
        measurement_start: 1000,
        measurement_count: None,
        ef_values: vec![10, 20, 40, 80, 160, 320],
        target_recall: 0.90,
        hnsw_m: 32,
        ef_construction: 200,
        full_scan_threshold_kb: 10,
        graph_seed: 20260821,
    };
    let mut have_distance = false;
    while let Some(flag) = raw.next() {
        match flag.to_string_lossy().as_ref() {
            "--vectors" => {
                args.vectors = PathBuf::from(raw.next().ok_or("--vectors requires a value")?)
            }
            "--queries" => {
                args.queries = PathBuf::from(raw.next().ok_or("--queries requires a value")?)
            }
            "--ground-truth" => {
                args.ground_truth =
                    PathBuf::from(raw.next().ok_or("--ground-truth requires a value")?)
            }
            "--point-ids" => {
                args.point_ids = Some(PathBuf::from(
                    raw.next().ok_or("--point-ids requires a value")?,
                ))
            }
            "--output-dir" => {
                args.output_dir = PathBuf::from(raw.next().ok_or("--output-dir requires a value")?)
            }
            "--dimension" => args.dimension = parse_value("--dimension", raw.next())?,
            "--distance" => {
                let value: String = parse_value("--distance", raw.next())?;
                args.distance = match value.as_str() {
                    "euclid" | "l2" => Distance::Euclid,
                    "cosine" | "angular" => Distance::Cosine,
                    _ => return Err(format!("unsupported --distance {value:?}").into()),
                };
                have_distance = true;
            }
            "--top" => args.top = parse_value("--top", raw.next())?,
            "--ground-truth-width" => {
                args.ground_truth_width = parse_value("--ground-truth-width", raw.next())?
            }
            "--tuning-count" => args.tuning_count = parse_value("--tuning-count", raw.next())?,
            "--measurement-start" => {
                args.measurement_start = parse_value("--measurement-start", raw.next())?
            }
            "--measurement-count" => {
                args.measurement_count = Some(parse_value("--measurement-count", raw.next())?)
            }
            "--ef-values" => args.ef_values = parse_ef_values(raw.next())?,
            "--target-recall" => args.target_recall = parse_value("--target-recall", raw.next())?,
            "--m" => args.hnsw_m = parse_value("--m", raw.next())?,
            "--ef-construction" => {
                args.ef_construction = parse_value("--ef-construction", raw.next())?
            }
            "--full-scan-threshold-kb" => {
                args.full_scan_threshold_kb = parse_value("--full-scan-threshold-kb", raw.next())?
            }
            "--graph-seed" => args.graph_seed = parse_value("--graph-seed", raw.next())?,
            "--help" | "-h" => {
                println!("{}", usage(&program));
                std::process::exit(0);
            }
            unknown => {
                return Err(format!("unknown option {unknown:?}\n{}", usage(&program)).into());
            }
        }
    }
    if args.vectors.as_os_str().is_empty()
        || args.queries.as_os_str().is_empty()
        || args.ground_truth.as_os_str().is_empty()
        || args.output_dir.as_os_str().is_empty()
        || args.dimension == 0
        || !have_distance
    {
        return Err(usage(&program).into());
    }
    if args.top == 0
        || args.ground_truth_width < args.top
        || args.hnsw_m == 0
        || args.ef_construction == 0
        || !(0.0..=1.0).contains(&args.target_recall)
    {
        return Err("invalid non-positive size or recall parameter".into());
    }
    Ok(args)
}

fn read_f32(path: &Path) -> Result<Vec<f32>, Box<dyn Error>> {
    let bytes = fs_err::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err(format!("{} length is not divisible by 4", path.display()).into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn read_u32(path: &Path) -> Result<Vec<u32>, Box<dyn Error>> {
    let bytes = fs_err::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err(format!("{} length is not divisible by 4", path.display()).into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn sha256_path(path: &Path) -> Result<String, Box<dyn Error>> {
    let mut digest = Sha256::new();
    let mut file = File::open(path)?;
    std::io::copy(&mut file, &mut digest)?;
    Ok(format!("{:x}", digest.finalize()))
}

fn production_entry_point_parameters(
    total_vector_count: usize,
    average_vector_bytes: usize,
    full_scan_threshold_kb: usize,
) -> (usize, usize) {
    // Keep this calculation equivalent to HNSW construction in hnsw.rs. The threshold is
    // expressed as a count of average-sized vectors, and Qdrant retains ten entry slots per
    // threshold-sized group (with at least one slot).
    let full_scan_threshold_points = full_scan_threshold_kb
        .saturating_mul(1024)
        .checked_div(average_vector_bytes)
        .unwrap_or(1);
    let entry_points_num = std::cmp::max(
        1,
        total_vector_count
            .checked_div(full_scan_threshold_points)
            .unwrap_or(0)
            * 10,
    );
    (full_scan_threshold_points, entry_points_num)
}

fn build_graph(
    vectors: &[f32],
    point_count: usize,
    args: &Args,
    graph_dir: &Path,
) -> Result<(VectorStorageEnum, GraphLayers, GraphBuildStats), Box<dyn Error>> {
    let mut storage = new_volatile_dense_vector_storage(args.dimension, args.distance);
    let build_counter = HardwareCounterCell::disposable();
    for point_index in 0..point_count {
        let start = point_index * args.dimension;
        let vector = args.distance.preprocess_vector::<VectorElementType>(
            vectors[start..start + args.dimension].to_vec(),
        );
        storage.insert_vector(
            PointOffsetType::try_from(point_index)?,
            VectorRef::from(&vector),
            &build_counter,
        )?;
    }

    let average_vector_bytes = storage
        .size_of_available_vectors_in_bytes()
        .checked_div(point_count)
        .ok_or("cannot calculate average vector size")?;
    let (full_scan_threshold_points, entry_points_num) = production_entry_point_parameters(
        point_count,
        average_vector_bytes,
        args.full_scan_threshold_kb,
    );

    let started = Instant::now();
    let mut builder = GraphLayersBuilder::new(
        point_count,
        HnswM::new2(args.hnsw_m),
        args.ef_construction,
        entry_points_num,
        true,
    );
    let mut rng = StdRng::seed_from_u64(args.graph_seed);
    for point_index in 0..point_count {
        let point_id = PointOffsetType::try_from(point_index)?;
        let level = builder.get_random_layer(&mut rng);
        builder.set_levels(point_id, level);
    }
    let deleted = BitVec::repeat(false, point_count);
    for point_index in 0..point_count {
        let point_id = PointOffsetType::try_from(point_index)?;
        let scorer = FilteredScorer::new_internal(
            point_id,
            &storage,
            None,
            None,
            &deleted,
            HardwareCounterCell::disposable(),
        )?;
        builder.link_new_point(point_id, scorer);
        if point_count >= 100_000 && (point_index + 1) % 10_000 == 0 {
            eprintln!("indexed {}/{} points", point_index + 1, point_count);
        }
    }
    fs_err::create_dir_all(graph_dir)?;
    let graph = builder.into_graph_layers(graph_dir, GraphLinksFormatParam::Plain, false)?;
    Ok((
        storage,
        graph,
        GraphBuildStats {
            construction_seconds: started.elapsed().as_secs_f64(),
            average_vector_bytes,
            full_scan_threshold_points,
            entry_points_num,
        },
    ))
}

fn scorer<'a>(
    query: &[f32],
    storage: &'a VectorStorageEnum,
    deleted: &'a BitVec,
    counter: HardwareCounterCell,
) -> Result<FilteredScorer<'a>, Box<dyn Error>> {
    Ok(FilteredScorer::new(
        query.into(),
        storage,
        None,
        None,
        deleted,
        counter,
    )?)
}

fn recall_at_k(result_ids: &[u32], truth: &[u32], k: usize) -> f64 {
    let truth = truth.iter().take(k).copied().collect::<HashSet<_>>();
    let recovered = result_ids
        .iter()
        .take(k)
        .filter(|point_id| truth.contains(point_id))
        .count();
    recovered as f64 / k as f64
}

fn tune_ef(
    args: &Args,
    queries: &[f32],
    truth: &[u32],
    point_ids: &[u32],
    storage: &VectorStorageEnum,
    graph: &GraphLayers,
    deleted: &BitVec,
) -> Result<(usize, Vec<TuningRow>), Box<dyn Error>> {
    let query_count = queries.len() / args.dimension;
    let tuning_count = args.tuning_count.min(query_count);
    if tuning_count == 0 {
        return Err("tuning query count is zero".into());
    }
    let mut rows = Vec::new();
    for &ef in &args.ef_values {
        let counter = HardwareCounterCell::new();
        let accumulator = counter.new_accumulator();
        let mut recall_sum = 0.0;
        for query_index in 0..tuning_count {
            let query_start = query_index * args.dimension;
            let results = graph
                .search(
                    args.top,
                    ef,
                    SearchAlgorithm::Hnsw,
                    scorer(
                        &queries[query_start..query_start + args.dimension],
                        storage,
                        deleted,
                        counter.fork(),
                    )?,
                    None,
                    &AtomicBool::new(false),
                )
                .map_err(|error| format!("HNSW tuning search cancelled: {error:?}"))?;
            let result_ids = results
                .iter()
                .map(|result| point_ids[result.idx as usize])
                .collect::<Vec<_>>();
            let truth_start = query_index * args.ground_truth_width;
            recall_sum += recall_at_k(
                &result_ids,
                &truth[truth_start..truth_start + args.ground_truth_width],
                args.top,
            );
        }
        rows.push(TuningRow {
            ef_search: ef,
            query_count: tuning_count,
            recall_at_k: recall_sum / tuning_count as f64,
            distance_computations: accumulator.get_cpu() as u64 / (args.dimension as u64 * 4),
            graph_nodes_visited: accumulator.get_graph_nodes_visited() as u64,
        });
    }
    let selected = rows
        .iter()
        .find(|row| row.recall_at_k >= args.target_recall)
        .ok_or_else(|| {
            format!(
                "no common EF reaches target recall {}; tuning={:?}",
                args.target_recall, rows
            )
        })?
        .ef_search;
    Ok((selected, rows))
}

fn write_graph_exports(
    output_dir: &Path,
    graph: &GraphLayers,
    point_ids: &[u32],
) -> Result<(PathBuf, PathBuf, u64), Box<dyn Error>> {
    let node_path = output_dir.join("node_ids.u32le");
    let mut node_writer = BufWriter::new(File::create(&node_path)?);
    for point_id in point_ids {
        node_writer.write_all(&point_id.to_le_bytes())?;
    }
    node_writer.flush()?;

    let edge_path = output_dir.join("L0_edges.u32le");
    let mut edge_writer = BufWriter::new(File::create(&edge_path)?);
    let mut edge_count = 0u64;
    for point_index in 0..point_ids.len() {
        let source = point_ids[point_index];
        for neighbor in graph.neighbors_at_level(PointOffsetType::try_from(point_index)?, 0) {
            edge_writer.write_all(&source.to_le_bytes())?;
            edge_writer.write_all(&point_ids[neighbor as usize].to_le_bytes())?;
            edge_count += 1;
        }
    }
    edge_writer.flush()?;
    Ok((node_path, edge_path, edge_count))
}

fn write_tuning_csv(path: &Path, rows: &[TuningRow]) -> Result<(), Box<dyn Error>> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "ef_search,query_count,recall_at_k,distance_computations,graph_nodes_visited"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{},{:.9},{},{}",
            row.ef_search,
            row.query_count,
            row.recall_at_k,
            row.distance_computations,
            row.graph_nodes_visited
        )
        .map_err(|error| format!("HNSW measurement search cancelled: {error:?}"))?;
    }
    writer.flush()?;
    Ok(())
}

#[expect(clippy::too_many_arguments)]
fn write_measurement_traces(
    args: &Args,
    selected_ef: usize,
    queries: &[f32],
    truth: &[u32],
    point_ids: &[u32],
    storage: &VectorStorageEnum,
    graph: &GraphLayers,
    deleted: &BitVec,
    path: &Path,
) -> Result<(usize, f64), Box<dyn Error>> {
    let query_count = queries.len() / args.dimension;
    let start = args.measurement_start.min(query_count);
    let end = args
        .measurement_count
        .map(|count| start.saturating_add(count).min(query_count))
        .unwrap_or(query_count);
    if start >= end {
        return Err("measurement query range is empty".into());
    }
    let mut writer = BufWriter::new(File::create(path)?);
    let mut recall_sum = 0.0;
    for query_index in start..end {
        let query_start = query_index * args.dimension;
        let started = Instant::now();
        let (results, trace) = graph
            .search_hnsw_with_trace(
                args.top,
                selected_ef,
                scorer(
                    &queries[query_start..query_start + args.dimension],
                    storage,
                    deleted,
                    HardwareCounterCell::new(),
                )?,
                &AtomicBool::new(false),
            )
            .map_err(|error| format!("HNSW measurement search cancelled: {error:?}"))?;
        let latency_us = started.elapsed().as_micros();
        let result_ids = results
            .iter()
            .map(|result| point_ids[result.idx as usize])
            .collect::<Vec<_>>();
        let result_scores = results
            .iter()
            .map(|result| result.score)
            .collect::<Vec<_>>();
        let truth_start = query_index * args.ground_truth_width;
        let ground_truth_ids = truth[truth_start..truth_start + args.top].to_vec();
        let recall = recall_at_k(&result_ids, &ground_truth_ids, args.top);
        recall_sum += recall;
        let visited_node_ids = trace
            .visited_node_ids
            .iter()
            .map(|point_id| point_ids[*point_id as usize])
            .collect::<Vec<_>>();
        let visited_edges = trace
            .visited_edges
            .iter()
            .map(|edge| {
                serde_json::json!({
                    "source_node": point_ids[edge.source_node as usize],
                    "destination_node": point_ids[edge.destination_node as usize],
                    "traversal_order": edge.traversal_order,
                    "layer": edge.layer,
                })
            })
            .collect::<Vec<_>>();
        let row = serde_json::json!({
            "query_id": query_index,
            "visited_node_ids": visited_node_ids,
            "visited_edges": visited_edges,
            "distance_computations": trace.distance_computations,
            "result_ids": result_ids,
            "result_scores": result_scores,
            "ground_truth_ids": ground_truth_ids,
            "recall_at_k": recall,
            "ef_search": selected_ef,
            "local_latency_us": latency_us,
        });
        serde_json::to_writer(&mut writer, &row)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok((end - start, recall_sum / (end - start) as f64))
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    if args.output_dir.exists() {
        return Err(format!(
            "refusing to overwrite existing output directory: {}",
            args.output_dir.display()
        )
        .into());
    }
    fs_err::create_dir_all(&args.output_dir)?;
    let vectors = read_f32(&args.vectors)?;
    let queries = read_f32(&args.queries)?;
    let truth = read_u32(&args.ground_truth)?;
    if vectors.len() % args.dimension != 0 || queries.len() % args.dimension != 0 {
        return Err("vector/query raw length is not divisible by dimension".into());
    }
    let point_count = vectors.len() / args.dimension;
    let query_count = queries.len() / args.dimension;
    if point_count == 0 || query_count == 0 {
        return Err("vectors and queries must be non-empty".into());
    }
    if truth.len() != query_count * args.ground_truth_width {
        return Err("ground-truth row count or width mismatch".into());
    }
    let point_ids = match &args.point_ids {
        Some(path) => read_u32(path)?,
        None => (0..u32::try_from(point_count)?).collect(),
    };
    if point_ids.len() != point_count {
        return Err("point ID count differs from vector count".into());
    }
    if point_ids.iter().copied().collect::<HashSet<_>>().len() != point_count {
        return Err("point IDs must be unique".into());
    }

    let graph_dir = args.output_dir.join("global_graph");
    let (storage, graph, build_stats) = build_graph(&vectors, point_count, &args, &graph_dir)?;
    let deleted = BitVec::repeat(false, point_count);
    let (selected_ef, tuning_rows) = tune_ef(
        &args, &queries, &truth, &point_ids, &storage, &graph, &deleted,
    )?;
    let tuning_path = args.output_dir.join("tuning.csv");
    write_tuning_csv(&tuning_path, &tuning_rows)?;
    let (node_path, edge_path, l0_edge_count) =
        write_graph_exports(&args.output_dir, &graph, &point_ids)?;
    let trace_path = args.output_dir.join("measurement_traces.jsonl");
    let (measurement_query_count, measurement_recall) = write_measurement_traces(
        &args,
        selected_ef,
        &queries,
        &truth,
        &point_ids,
        &storage,
        &graph,
        &deleted,
        &trace_path,
    )?;
    let (entry_offset, entry_level) = graph.entry_point().ok_or("graph has no entry point")?;
    let manifest = serde_json::json!({
        "format_version": FORMAT_VERSION,
        "point_count": point_count,
        "query_count": query_count,
        "dimension": args.dimension,
        "distance": format!("{:?}", args.distance).to_lowercase(),
        "top": args.top,
        "hnsw_m": args.hnsw_m,
        "hnsw_ef_construction": args.ef_construction,
        "hnsw_full_scan_threshold_kb": args.full_scan_threshold_kb,
        "hnsw_average_vector_bytes": build_stats.average_vector_bytes,
        "hnsw_full_scan_threshold_points": build_stats.full_scan_threshold_points,
        "hnsw_entry_points_num": build_stats.entry_points_num,
        "hnsw_graph_seed": args.graph_seed,
        "construction_seconds": build_stats.construction_seconds,
        "entry_point": point_ids[entry_offset as usize],
        "entry_level": entry_level,
        "l0_edge_count": l0_edge_count,
        "tuning_query_count": args.tuning_count.min(query_count),
        "measurement_start": args.measurement_start,
        "measurement_query_count": measurement_query_count,
        "target_recall": args.target_recall,
        "selected_ef_search": selected_ef,
        "measurement_recall_at_k": measurement_recall,
        "tuning": tuning_rows,
        "files": {
            "graph": graph.files(&graph_dir).iter().map(|path| path.display().to_string()).collect::<Vec<_>>(),
            "node_ids": node_path.display().to_string(),
            "node_ids_sha256": sha256_path(&node_path)?,
            "l0_edges": edge_path.display().to_string(),
            "l0_edges_sha256": sha256_path(&edge_path)?,
            "tuning": tuning_path.display().to_string(),
            "tuning_sha256": sha256_path(&tuning_path)?,
            "measurement_traces": trace_path.display().to_string(),
            "measurement_traces_sha256": sha256_path(&trace_path)?,
        },
    });
    let manifest_path = args.output_dir.join("manifest.json");
    let mut manifest_writer = BufWriter::new(File::create(&manifest_path)?);
    serde_json::to_writer_pretty(&mut manifest_writer, &manifest)?;
    manifest_writer.write_all(b"\n")?;
    manifest_writer.flush()?;
    println!("manifest={}", manifest_path.display());
    println!("selected_ef_search={selected_ef}");
    println!("measurement_recall_at_k={measurement_recall:.9}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::production_entry_point_parameters;

    #[test]
    fn production_entry_points_match_sift1m_configuration() {
        assert_eq!(
            production_entry_point_parameters(1_000_000, 128 * 4, 10),
            (20, 500_000)
        );
    }

    #[test]
    fn production_entry_points_match_smoke_configuration() {
        assert_eq!(
            production_entry_point_parameters(2_000, 16 * 4, 10),
            (160, 120)
        );
    }

    #[test]
    fn production_entry_points_preserve_qdrant_zero_threshold_semantics() {
        assert_eq!(production_entry_point_parameters(2_000, 16 * 4, 0), (0, 1));
    }
}
