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
    assignments: PathBuf,
    output_dir: PathBuf,
    partition_method: String,
    partition_seed: u64,
    partition_sha256: String,
    dataset: String,
    dataset_sha256: String,
    dimension: usize,
    distance: Distance,
    logical_shards: usize,
    top: usize,
    ground_truth_width: usize,
    measurement_start: usize,
    measurement_count: Option<usize>,
    e3_ef_values: Vec<usize>,
    e4_ef: usize,
    hnsw_m: usize,
    ef_construction: usize,
    full_scan_threshold_kb: usize,
    graph_seed: u64,
}

#[derive(Debug, Serialize)]
struct GraphBuildStats {
    construction_seconds: f64,
    average_vector_bytes: usize,
    full_scan_threshold_points: usize,
    entry_points_num: usize,
}

#[derive(Debug)]
struct SearchResult {
    result_ids: Vec<u32>,
    distance_computations: u64,
    graph_nodes_visited: u64,
    latency_us: u128,
}

fn usage(program: &str) -> String {
    format!(
        "Usage: {program} --vectors FILE --queries FILE --ground-truth FILE \\\n           --assignments FILE --output-dir DIR --dataset NAME --dataset-sha256 HEX \\\n           --partition-method NAME --partition-seed N --partition-sha256 HEX \\\n           --dimension N --distance euclid|cosine --logical-shards N [options]\n\
         \n\
         Raw formats are contiguous little-endian float32 vectors/queries and uint32 truth/\n\
         assignments. Assignment values must be in 0..logical-shards.\n\
         \n\
         Options:\n\
           --top N                     result K (default 10)\n\
           --ground-truth-width N      truth row width (default 10)\n\
           --measurement-start N       first measurement query (default 1000)\n\
           --measurement-count N       measurement rows; default remaining rows\n\
           --e3-ef-values LIST         fixed E3 grid (default 10,20,40,80,160,320)\n\
           --e4-ef N                   common unpartitioned EF for E4 (required)\n\
           --m N                       HNSW M (default 32)\n\
           --ef-construction N         HNSW construction EF (default 200)\n\
           --full-scan-threshold-kb N  production HNSW threshold (default 10)\n\
           --graph-seed N              deterministic per-shard level seed (default 20260821)"
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

fn parse_list(flag: &str, value: Option<OsString>) -> Result<Vec<usize>, Box<dyn Error>> {
    let value = value
        .ok_or_else(|| format!("{flag} requires a value"))?
        .into_string()
        .map_err(|_| format!("{flag} is not valid UTF-8"))?;
    let mut values = value
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(str::parse::<usize>)
        .collect::<Result<Vec<_>, _>>()?;
    values.sort_unstable();
    values.dedup();
    if values.is_empty() || values[0] == 0 {
        return Err(format!("{flag} must contain positive integers").into());
    }
    Ok(values)
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut raw = env::args_os();
    let program = raw
        .next()
        .unwrap_or_else(|| OsString::from("c23_local_eval"))
        .to_string_lossy()
        .into_owned();
    let mut args = Args {
        vectors: PathBuf::new(),
        queries: PathBuf::new(),
        ground_truth: PathBuf::new(),
        assignments: PathBuf::new(),
        output_dir: PathBuf::new(),
        partition_method: String::new(),
        partition_seed: 0,
        partition_sha256: String::new(),
        dataset: String::new(),
        dataset_sha256: String::new(),
        dimension: 0,
        distance: Distance::Euclid,
        logical_shards: 0,
        top: 10,
        ground_truth_width: 10,
        measurement_start: 1000,
        measurement_count: None,
        e3_ef_values: vec![10, 20, 40, 80, 160, 320],
        e4_ef: 0,
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
            "--assignments" => {
                args.assignments =
                    PathBuf::from(raw.next().ok_or("--assignments requires a value")?)
            }
            "--output-dir" => {
                args.output_dir = PathBuf::from(raw.next().ok_or("--output-dir requires a value")?)
            }
            "--partition-method" => {
                args.partition_method = parse_value("--partition-method", raw.next())?
            }
            "--partition-seed" => {
                args.partition_seed = parse_value("--partition-seed", raw.next())?
            }
            "--partition-sha256" => {
                args.partition_sha256 = parse_value("--partition-sha256", raw.next())?
            }
            "--dataset" => args.dataset = parse_value("--dataset", raw.next())?,
            "--dataset-sha256" => {
                args.dataset_sha256 = parse_value("--dataset-sha256", raw.next())?
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
            "--logical-shards" => {
                args.logical_shards = parse_value("--logical-shards", raw.next())?
            }
            "--top" => args.top = parse_value("--top", raw.next())?,
            "--ground-truth-width" => {
                args.ground_truth_width = parse_value("--ground-truth-width", raw.next())?
            }
            "--measurement-start" => {
                args.measurement_start = parse_value("--measurement-start", raw.next())?
            }
            "--measurement-count" => {
                args.measurement_count = Some(parse_value("--measurement-count", raw.next())?)
            }
            "--e3-ef-values" => args.e3_ef_values = parse_list("--e3-ef-values", raw.next())?,
            "--e4-ef" => args.e4_ef = parse_value("--e4-ef", raw.next())?,
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
        || args.assignments.as_os_str().is_empty()
        || args.output_dir.as_os_str().is_empty()
        || args.partition_method.is_empty()
        || args.partition_sha256.is_empty()
        || args.dataset.is_empty()
        || args.dataset_sha256.is_empty()
        || args.dimension == 0
        || args.logical_shards == 0
        || args.e4_ef == 0
        || !have_distance
    {
        return Err(usage(&program).into());
    }
    if args.top == 0
        || args.ground_truth_width < args.top
        || args.hnsw_m == 0
        || args.ef_construction == 0
    {
        return Err("invalid non-positive size parameter".into());
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

fn build_local_graph(
    vectors: &[f32],
    global_ids: &[u32],
    args: &Args,
    shard_id: usize,
    graph_dir: &Path,
) -> Result<(VectorStorageEnum, GraphLayers, GraphBuildStats), Box<dyn Error>> {
    let mut storage = new_volatile_dense_vector_storage(args.dimension, args.distance);
    let build_counter = HardwareCounterCell::disposable();
    for &global_id in global_ids {
        let start = global_id as usize * args.dimension;
        let vector = args.distance.preprocess_vector::<VectorElementType>(
            vectors[start..start + args.dimension].to_vec(),
        );
        storage.insert_vector(
            PointOffsetType::try_from(storage.total_vector_count())?,
            VectorRef::from(&vector),
            &build_counter,
        )?;
    }
    let average_vector_bytes = storage
        .size_of_available_vectors_in_bytes()
        .checked_div(global_ids.len())
        .ok_or("cannot calculate average local vector size")?;
    let (full_scan_threshold_points, entry_points_num) = production_entry_point_parameters(
        global_ids.len(),
        average_vector_bytes,
        args.full_scan_threshold_kb,
    );
    let started = Instant::now();
    let mut builder = GraphLayersBuilder::new(
        global_ids.len(),
        HnswM::new2(args.hnsw_m),
        args.ef_construction,
        entry_points_num,
        true,
    );
    let mut rng = StdRng::seed_from_u64(args.graph_seed);
    for local_index in 0..global_ids.len() {
        let local_id = PointOffsetType::try_from(local_index)?;
        let level = builder.get_random_layer(&mut rng);
        builder.set_levels(local_id, level);
    }
    let deleted = BitVec::repeat(false, global_ids.len());
    for local_index in 0..global_ids.len() {
        let local_id = PointOffsetType::try_from(local_index)?;
        let scorer = FilteredScorer::new_internal(
            local_id,
            &storage,
            None,
            None,
            &deleted,
            HardwareCounterCell::disposable(),
        )?;
        builder.link_new_point(local_id, scorer);
        if global_ids.len() >= 100_000 && (local_index + 1) % 10_000 == 0 {
            eprintln!(
                "shard {shard_id}/{}, indexed {}/{} points",
                args.logical_shards,
                local_index + 1,
                global_ids.len()
            );
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

fn search_local(
    args: &Args,
    query: &[f32],
    ef: usize,
    global_ids: &[u32],
    storage: &VectorStorageEnum,
    graph: &GraphLayers,
    deleted: &BitVec,
) -> Result<SearchResult, Box<dyn Error>> {
    let counter = HardwareCounterCell::new();
    let accumulator = counter.new_accumulator();
    let started = Instant::now();
    let results = graph
        .search(
            args.top.min(global_ids.len()),
            ef,
            SearchAlgorithm::Hnsw,
            scorer(query, storage, deleted, counter)?,
            None,
            &AtomicBool::new(false),
        )
        .map_err(|error| format!("local HNSW search cancelled: {error:?}"))?;
    Ok(SearchResult {
        result_ids: results
            .iter()
            .map(|result| global_ids[result.idx as usize])
            .collect(),
        distance_computations: accumulator.get_cpu() as u64 / (args.dimension as u64 * 4),
        graph_nodes_visited: accumulator.get_graph_nodes_visited() as u64,
        latency_us: started.elapsed().as_micros(),
    })
}

fn recovered_truth(result_ids: &[u32], target_ids: &[u32]) -> Vec<u32> {
    let targets = target_ids.iter().copied().collect::<HashSet<_>>();
    result_ids
        .iter()
        .copied()
        .filter(|point_id| targets.contains(point_id))
        .collect()
}

fn target_ranks(result_ids: &[u32], target_ids: &[u32]) -> Vec<usize> {
    target_ids
        .iter()
        .map(|target| {
            result_ids
                .iter()
                .position(|point_id| point_id == target)
                .map(|rank| rank + 1)
                .unwrap_or(0)
        })
        .collect()
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
    let assignments = read_u32(&args.assignments)?;
    if vectors.len() % args.dimension != 0 || queries.len() % args.dimension != 0 {
        return Err("vector/query raw length is not divisible by dimension".into());
    }
    let point_count = vectors.len() / args.dimension;
    let query_count = queries.len() / args.dimension;
    if assignments.len() != point_count {
        return Err("assignment count differs from vector count".into());
    }
    if truth.len() != query_count * args.ground_truth_width {
        return Err("ground-truth row count or width mismatch".into());
    }
    if assignments
        .iter()
        .any(|&shard_id| shard_id as usize >= args.logical_shards)
    {
        return Err("assignment contains an invalid shard ID".into());
    }
    let start = args.measurement_start.min(query_count);
    let end = args
        .measurement_count
        .map(|count| start.saturating_add(count).min(query_count))
        .unwrap_or(query_count);
    if start >= end {
        return Err("measurement query range is empty".into());
    }

    let total_started = Instant::now();
    let mut shard_manifests = Vec::new();
    for shard_id in 0..args.logical_shards {
        let shard_started = Instant::now();
        let global_ids = assignments
            .iter()
            .enumerate()
            .filter(|entry| *entry.1 as usize == shard_id)
            .map(|(point_id, _)| u32::try_from(point_id))
            .collect::<Result<Vec<_>, _>>()?;
        if global_ids.is_empty() {
            return Err(format!("shard {shard_id} is empty").into());
        }
        let shard_dir = args.output_dir.join(format!("shard-{shard_id:03}"));
        fs_err::create_dir_all(&shard_dir)?;
        let graph_dir = shard_dir.join("graph");
        let (storage, graph, build_stats) =
            build_local_graph(&vectors, &global_ids, &args, shard_id, &graph_dir)?;
        let deleted = BitVec::repeat(false, global_ids.len());
        let e3_path = shard_dir.join("e3_query_shard.jsonl");
        let e4_path = shard_dir.join("e4_query_shard.jsonl");
        let mut e3_writer = BufWriter::new(File::create(&e3_path)?);
        let mut e4_writer = BufWriter::new(File::create(&e4_path)?);
        let mut e3_row_count = 0usize;
        let mut e4_row_count = 0usize;
        for query_index in start..end {
            let query_start = query_index * args.dimension;
            let query = &queries[query_start..query_start + args.dimension];
            let truth_start = query_index * args.ground_truth_width;
            let global_truth = &truth[truth_start..truth_start + args.top];
            let local_truth = global_truth
                .iter()
                .copied()
                .filter(|point_id| assignments[*point_id as usize] as usize == shard_id)
                .collect::<Vec<_>>();

            let e4 = search_local(
                &args,
                query,
                args.e4_ef,
                &global_ids,
                &storage,
                &graph,
                &deleted,
            )?;
            let recovered = recovered_truth(&e4.result_ids, global_truth);
            let e4_row = serde_json::json!({
                "query_id": query_index,
                "shard_id": shard_id,
                "ef_search": args.e4_ef,
                "result_ids": e4.result_ids,
                "recovered_ground_truth_ids": recovered,
                "distance_computations": e4.distance_computations,
                "graph_nodes_visited": e4.graph_nodes_visited,
                "local_latency_us": e4.latency_us,
            });
            serde_json::to_writer(&mut e4_writer, &e4_row)?;
            e4_writer.write_all(b"\n")?;
            e4_row_count += 1;

            if !local_truth.is_empty() {
                for &ef in &args.e3_ef_values {
                    let result =
                        search_local(&args, query, ef, &global_ids, &storage, &graph, &deleted)?;
                    let recovered = recovered_truth(&result.result_ids, &local_truth);
                    let row = serde_json::json!({
                        "query_id": query_index,
                        "shard_id": shard_id,
                        "ground_truth_ids_in_shard": local_truth,
                        "ground_truth_count_in_shard": local_truth.len(),
                        "ef_search": ef,
                        "local_recovered_ground_truth": recovered.len(),
                        "local_target_recall": recovered.len() as f64 / local_truth.len() as f64,
                        "local_target_ranks": target_ranks(&result.result_ids, &local_truth),
                        "result_ids": result.result_ids,
                        "distance_computations": result.distance_computations,
                        "graph_nodes_visited": result.graph_nodes_visited,
                        "local_latency_us": result.latency_us,
                    });
                    serde_json::to_writer(&mut e3_writer, &row)?;
                    e3_writer.write_all(b"\n")?;
                    e3_row_count += 1;
                }
            }
        }
        e3_writer.flush()?;
        e4_writer.flush()?;
        let (entry_offset, entry_level) = graph
            .entry_point()
            .ok_or("local graph has no entry point")?;
        let shard_manifest = serde_json::json!({
            "format_version": FORMAT_VERSION,
            "dataset": args.dataset,
            "dataset_sha256": args.dataset_sha256,
            "partition_method": args.partition_method,
            "partition_seed": args.partition_seed,
            "partition_sha256": args.partition_sha256,
            "logical_shards": args.logical_shards,
            "shard_id": shard_id,
            "local_point_count": global_ids.len(),
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
            "total_shard_seconds": shard_started.elapsed().as_secs_f64(),
            "entry_point_global_id": global_ids[entry_offset as usize],
            "entry_level": entry_level,
            "measurement_start": start,
            "measurement_query_count": end - start,
            "e3_ef_values": args.e3_ef_values,
            "e4_ef_search": args.e4_ef,
            "e3_row_count": e3_row_count,
            "e4_row_count": e4_row_count,
            "files": {
                "graph": graph.files(&graph_dir).iter().map(|path| path.display().to_string()).collect::<Vec<_>>(),
                "e3_query_shard": e3_path.display().to_string(),
                "e3_query_shard_sha256": sha256_path(&e3_path)?,
                "e4_query_shard": e4_path.display().to_string(),
                "e4_query_shard_sha256": sha256_path(&e4_path)?,
            },
        });
        let shard_manifest_path = shard_dir.join("manifest.json");
        let mut shard_manifest_writer = BufWriter::new(File::create(&shard_manifest_path)?);
        serde_json::to_writer_pretty(&mut shard_manifest_writer, &shard_manifest)?;
        shard_manifest_writer.write_all(b"\n")?;
        shard_manifest_writer.flush()?;
        shard_manifests.push(serde_json::json!({
            "shard_id": shard_id,
            "manifest": shard_manifest_path.display().to_string(),
            "manifest_sha256": sha256_path(&shard_manifest_path)?,
            "local_point_count": global_ids.len(),
        }));
        println!(
            "completed shard {shard_id}/{} points={} e3_rows={} e4_rows={}",
            args.logical_shards,
            global_ids.len(),
            e3_row_count,
            e4_row_count
        );
    }
    let manifest = serde_json::json!({
        "format_version": FORMAT_VERSION,
        "dataset": args.dataset,
        "dataset_sha256": args.dataset_sha256,
        "partition_method": args.partition_method,
        "partition_seed": args.partition_seed,
        "partition_path": args.assignments.display().to_string(),
        "partition_sha256": args.partition_sha256,
        "logical_shards": args.logical_shards,
        "point_count": point_count,
        "query_count": query_count,
        "dimension": args.dimension,
        "distance": format!("{:?}", args.distance).to_lowercase(),
        "top": args.top,
        "hnsw_m": args.hnsw_m,
        "hnsw_ef_construction": args.ef_construction,
        "hnsw_full_scan_threshold_kb": args.full_scan_threshold_kb,
        "hnsw_graph_seed": args.graph_seed,
        "measurement_start": start,
        "measurement_query_count": end - start,
        "e3_ef_values": args.e3_ef_values,
        "e4_ef_search": args.e4_ef,
        "total_seconds": total_started.elapsed().as_secs_f64(),
        "shards": shard_manifests,
    });
    let manifest_path = args.output_dir.join("manifest.json");
    let mut manifest_writer = BufWriter::new(File::create(&manifest_path)?);
    serde_json::to_writer_pretty(&mut manifest_writer, &manifest)?;
    manifest_writer.write_all(b"\n")?;
    manifest_writer.flush()?;
    println!("manifest={}", manifest_path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{production_entry_point_parameters, recovered_truth, target_ranks};

    #[test]
    fn target_recovery_and_ranks_are_reproducible() {
        let results = [20, 10, 30, 40];
        let targets = [10, 40, 50];
        assert_eq!(recovered_truth(&results, &targets), vec![10, 40]);
        assert_eq!(target_ranks(&results, &targets), vec![2, 4, 0]);
    }

    #[test]
    fn local_entry_points_match_production_calculation() {
        assert_eq!(
            production_entry_point_parameters(31_250, 128 * 4, 10),
            (20, 15_620)
        );
    }
}
