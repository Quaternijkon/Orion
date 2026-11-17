//! Offline locality experiment worker. No server, network, QPS, or production writes.
//! Python prepares immutable requests; this executable calls the production routers
//! and the same GraphLayers::search used by unquantized local HNSW search.
use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

use anyhow::{Result, bail, ensure};
use collection::orion::{OrionRouter, OrionRoutingArtifact, whitebox_pruning};
use collection::simple_kmeans::{SimpleKmeansRouter, SimpleKmeansRoutingArtifact};
use common::bitvec::BitVec;
use common::counter::hardware_accumulator::HwMeasurementAcc;
use common::counter::hardware_counter::HardwareCounterCell;
use rand::{SeedableRng, rngs::StdRng};
use segment::data_types::query_context::QueryContext;
use segment::data_types::vectors::VectorRef;
use segment::entry::entry_point::ReadSegmentEntry;
use segment::index::hnsw_index::HnswM;
use segment::index::hnsw_index::graph_layers::SearchAlgorithm;
use segment::index::hnsw_index::graph_layers_builder::GraphLayersBuilder;
use segment::index::hnsw_index::graph_links::GraphLinksFormatParam;
use segment::index::hnsw_index::point_scorer::FilteredScorer;
use segment::segment_constructor::load_segment;
use segment::types::SearchParams;
use segment::types::{Distance, ExtendedPointId};
use segment::vector_storage::dense::volatile_dense_vector_storage::{
    VolatileDenseVectorStorage, new_volatile_dense_vector_storage,
};
use segment::vector_storage::{VectorStorage, VectorStorageEnum};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BuildSegmentRequest {
    vectors: PathBuf,
    ids: PathBuf,
    output: PathBuf,
    dimension: usize,
    distance: Distance,
    m: usize,
    ef_construct: usize,
    threads: usize,
    seed: u64,
}

fn build_local_segment(request: BuildSegmentRequest, writer: &mut impl Write) -> Result<()> {
    use common::budget::ResourceBudget;
    use common::progress_tracker::new_progress_tracker;
    use segment::data_types::vectors::only_default_vector;
    use segment::entry::entry_point::{SegmentEntry, StorageSegmentEntry};
    use segment::segment_constructor::{build_segment, segment_builder::SegmentBuilder};
    use segment::types::{HnswConfig, HnswGlobalConfig, Indexes, SegmentConfig};
    ensure!(
        request.m == 32 && request.ef_construct == 200 && request.threads > 0,
        "formal experiment requires M32/ef_construct200 and a positive indexing thread count"
    );
    ensure!(
        !request.output.exists(),
        "refuse to overwrite a lower index"
    );
    let bytes = fs::read(&request.ids)?;
    ensure!(bytes.len() % 8 == 0, "invalid shard IDs");
    let ids: Vec<u64> = bytes
        .chunks_exact(8)
        .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    ensure!(
        !ids.is_empty() && ids.windows(2).all(|w| w[0] < w[1]),
        "shard IDs must be ordered and unique"
    );
    ensure!(
        fs::metadata(&request.vectors)?.len() == ids.len() as u64 * request.dimension as u64 * 4,
        "shard vector size mismatch"
    );
    let config: SegmentConfig = serde_json::from_value(serde_json::json!({
        "vector_data":{"":{"size":request.dimension,"distance":request.distance,
            "storage_type":"InRamChunkedMmap","index":{"type":"plain","options":{}},"quantization_config":null}},
        "payload_storage_type":{"type":"mmap"}}))?;
    fs::create_dir_all(&request.output)?;
    let staging = tempfile::tempdir_in(&request.output)?;
    let mut source = build_segment(staging.path(), &config, None, true)?;
    let counter = HardwareCounterCell::disposable();
    let mut vectors = BufReader::new(File::open(&request.vectors)?);
    for (offset, id) in ids.iter().enumerate() {
        let vector = read_vector(&mut vectors, request.dimension, false)?;
        source.upsert_point(
            offset as u64 + 1,
            (*id).into(),
            only_default_vector(&vector),
            &counter,
        )?;
        if (offset + 1) % 100_000 == 0 {
            eprintln!("lower ingest: {}/{}", offset + 1, ids.len());
        }
    }
    let mut indexed = config.clone();
    indexed.vector_data.get_mut("").unwrap().index = Indexes::Hnsw(HnswConfig {
        m: request.m,
        ef_construct: request.ef_construct,
        full_scan_threshold: 10,
        max_indexing_threads: request.threads,
        on_disk: Some(false),
        ..Default::default()
    });
    let stopped = AtomicBool::new(false);
    let mut builder = SegmentBuilder::new(staging.path(), &indexed, &HnswGlobalConfig::default())?;
    builder.update(&[&source], &stopped)?;
    drop(source);
    let budget = ResourceBudget::new(request.threads, request.threads);
    let permit = budget
        .try_acquire(request.threads, request.threads)
        .unwrap();
    let (view, progress) = new_progress_tracker();
    let mut rng = StdRng::seed_from_u64(request.seed);
    let segment = std::thread::scope(|scope| {
        let (done, receive) = std::sync::mpsc::channel::<()>();
        scope.spawn(move || {
            while let Err(std::sync::mpsc::RecvTimeoutError::Timeout) =
                receive.recv_timeout(std::time::Duration::from_secs(15))
            {
                eprintln!(
                    "lower build progress: {}",
                    serde_json::to_string(&view.snapshot("lower_index")).unwrap()
                );
            }
        });
        let result = builder.build(
            &request.output,
            uuid::Uuid::new_v4(),
            None,
            permit,
            &stopped,
            &mut rng,
            &counter,
            progress,
        );
        let _ = done.send(());
        result
    })?;
    segment.flush(true)?;
    ensure!(
        segment.available_point_count() == ids.len(),
        "built shard point count mismatch"
    );
    json_line(
        writer,
        &serde_json::json!({"complete":true,"points":ids.len(),"m":request.m,
        "ef_construct":request.ef_construct,"threads":request.threads,"config":segment.segment_config}),
    )?;
    Ok(())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RouteRequest {
    method: String,
    artifact: PathBuf,
    queries: PathBuf,
    query_count: usize,
    dimension: usize,
    shard_count: u32,
    point_count: u64,
    physical_point_count: u64,
    layout_sha256: String,
    distance: Distance,
    datatype: String,
    upper_k: Option<usize>,
    upper_ef_search: Option<usize>,
    dynamic_ef_base: Option<usize>,
    dynamic_ef_factor: Option<usize>,
    whitebox: Option<whitebox_pruning::Config>,
    nprobe: Option<usize>,
    lower_hnsw_ef: Option<usize>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Target {
    shard_id: u32,
    ef: usize,
    // None is conventional HNSW; Some is explicit Orion MultiEP, never fallback.
    entry_points: Option<Vec<u64>>,
}

#[derive(Serialize)]
struct RouteRow {
    query_index: usize,
    targets: Vec<Target>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LowerRequest {
    vectors: PathBuf,
    vector_encoding: String,
    datatype: String,
    point_count: usize,
    dimension: usize,
    distance: Distance,
    ids: PathBuf,
    queries: PathBuf,
    query_count: usize,
    tasks: PathBuf,
    work_dir: PathBuf,
    top_k: usize,
    m: usize,
    ef_construct: usize,
    seed: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Task {
    profile: String,
    query_index: usize,
    target: Target,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SegmentRequest {
    segments: Vec<PathBuf>,
    queries: PathBuf,
    query_count: usize,
    dimension: usize,
    distance: Distance,
    vector_name: String,
    tasks: PathBuf,
    truth_ids: PathBuf,
    top_k: usize,
}

// Only invoke on independent extracted working copies. The native loader may
// migrate metadata; the original archives must remain immutable.
fn search_segments(request: SegmentRequest, writer: &mut impl Write) -> Result<()> {
    ensure!(
        !request.segments.is_empty() && request.top_k > 0,
        "empty segment request"
    );
    let queries = queries(&request.queries, request.query_count, request.dimension)?;
    let raw = fs::read(&request.truth_ids)?;
    ensure!(raw.len() % 8 == 0, "invalid truth ID file");
    let truth: Vec<u64> = raw
        .chunks_exact(8)
        .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let tasks = BufReader::new(File::open(&request.tasks)?)
        .lines()
        .map(|l| Ok(serde_json::from_str::<Task>(&l?)?))
        .collect::<Result<Vec<_>>>()?;
    let needed_entries: HashSet<u64> = tasks
        .iter()
        .flat_map(|t| t.target.entry_points.iter().flatten().copied())
        .collect();
    let mut seen_entries = HashSet::new();
    let stopped = AtomicBool::new(false);
    for (segment_index, path) in request.segments.iter().enumerate() {
        let uuid = path
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or_else(|| anyhow::anyhow!("invalid segment path"))?
            .parse()?;
        let segment = load_segment(path, uuid, None, &stopped)?;
        let vector_config = segment
            .segment_config
            .vector_data
            .get(&request.vector_name)
            .ok_or_else(|| anyhow::anyhow!("missing vector name"))?;
        ensure!(
            vector_config.size == request.dimension && vector_config.distance == request.distance,
            "archived vector schema mismatch"
        );
        ensure!(
            vector_config.quantization_config.is_none(),
            "quantized archives are not supported"
        );
        let found: Vec<_> = truth
            .iter()
            .copied()
            .filter(|&id| segment.has_point(id.into()))
            .collect();
        seen_entries.extend(
            needed_entries
                .iter()
                .copied()
                .filter(|&id| segment.has_point(id.into())),
        );
        let count = segment.available_point_count();
        json_line(
            writer,
            &serde_json::json!({"kind":"membership", "segment_index":segment_index,
            "point_count": count, "gt_ids":found, "config":segment.segment_config}),
        )?;
        if count == 0 {
            continue;
        }
        let mut context = QueryContext::new(0, HwMeasurementAcc::new());
        context.add_available_point_count(count);
        let context = context.get_segment_query_context();
        for task in &tasks {
            ensure!(
                task.query_index < queries.len() && task.target.ef > 0,
                "invalid search task"
            );
            let entries = task.target.entry_points.as_ref().map(|v| {
                v.iter()
                    .copied()
                    .map(ExtendedPointId::NumId)
                    .collect::<Vec<_>>()
            });
            let query = queries[task.query_index].clone().into();
            let params = SearchParams {
                hnsw_ef: Some(task.target.ef),
                ..Default::default()
            };
            let mut result = segment.search_batch(
                &request.vector_name,
                &[&query],
                &false.into(),
                &false.into(),
                None,
                request.top_k,
                entries.as_deref(),
                Some(&params),
                &context,
            )?;
            ensure!(result.len() == 1, "invalid segment response");
            let points = result
                .pop()
                .unwrap()
                .into_iter()
                .map(|p| {
                    let score = request.distance.postprocess_score(p.score);
                    Ok(Point {
                        id: numeric(p.id)?,
                        score: if request.distance == Distance::Euclid {
                            -score
                        } else {
                            score
                        },
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            json_line(
                writer,
                &serde_json::json!({"kind":"search", "segment_index":segment_index,
                "profile":task.profile,"query_index":task.query_index,"points":points}),
            )?;
        }
    }
    ensure!(
        needed_entries == seen_entries,
        "some Orion owner entries are absent from the archived shard"
    );
    Ok(())
}

#[derive(Serialize)]
struct Point {
    id: u64,
    // Postprocess exactly as LocalShard does, then negate Euclid so all exported
    // scores are larger-is-better. This also preserves f32 postprocessing ties.
    score: f32,
}

fn numeric(id: ExtendedPointId) -> Result<u64> {
    match id {
        ExtendedPointId::NumId(id) => Ok(id),
        _ => bail!("locality experiments require numeric source IDs"),
    }
}

fn queries(path: &PathBuf, count: usize, dim: usize) -> Result<Vec<Vec<f32>>> {
    ensure!(count > 0 && dim > 0, "empty queries");
    ensure!(
        fs::metadata(path)?.len() == (count as u64) * (dim as u64) * 4,
        "query file size mismatch"
    );
    let mut reader = BufReader::new(File::open(path)?);
    (0..count)
        .map(|_| read_vector(&mut reader, dim, false))
        .collect()
}

fn read_vector(reader: &mut impl Read, dim: usize, bytes: bool) -> Result<Vec<f32>> {
    let mut raw = vec![0; dim * if bytes { 1 } else { 4 }];
    reader.read_exact(&mut raw)?;
    let vector: Vec<f32> = if bytes {
        raw.iter().map(|&x| f32::from(x)).collect()
    } else {
        raw.chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    };
    ensure!(vector.iter().all(|x| x.is_finite()), "non-finite vector");
    Ok(vector)
}

fn json_line(writer: &mut impl Write, value: &impl Serialize) -> Result<()> {
    serde_json::to_writer(&mut *writer, value)?;
    writer.write_all(b"\n")?;
    Ok(())
}

fn route(request: RouteRequest, writer: &mut impl Write) -> Result<()> {
    let queries = queries(&request.queries, request.query_count, request.dimension)?;
    match request.method.as_str() {
        "orion" => {
            let mut artifact = OrionRoutingArtifact::read_json(&request.artifact, None)?;
            ensure!(
                artifact.vector_schema.dimension == request.dimension,
                "dimension mismatch"
            );
            ensure!(
                artifact.shard_count == request.shard_count
                    && artifact.logical_point_count == request.point_count
                    && artifact.physical_point_count == request.physical_point_count
                    && artifact.layout_sha256 == request.layout_sha256
                    && artifact.vector_schema.distance == request.distance
                    && serde_json::to_value(artifact.vector_schema.datatype)? == request.datatype,
                "Orion artifact does not match the experiment dataset/layout/schema"
            );
            if let Some(v) = request.upper_k {
                artifact.upper_k = v;
            }
            if let Some(v) = request.upper_ef_search {
                artifact.upper_ef_search = v;
            }
            if let Some(v) = request.dynamic_ef_base {
                artifact.dynamic_ef_base = v;
            }
            if let Some(v) = request.dynamic_ef_factor {
                artifact.dynamic_ef_factor = v;
            }
            let config = request
                .whitebox
                .ok_or_else(|| anyhow::anyhow!("whitebox required"))?;
            let router = OrionRouter::new(artifact)?.with_whitebox(config)?;
            for (query_index, query) in queries.iter().enumerate() {
                let targets = router
                    .route_query(query)?
                    .into_iter()
                    .map(|t| {
                        Ok(Target {
                            shard_id: t.shard_id,
                            ef: t.ef,
                            entry_points: Some(
                                t.entry_points
                                    .into_iter()
                                    .map(numeric)
                                    .collect::<Result<_>>()?,
                            ),
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                json_line(
                    writer,
                    &RouteRow {
                        query_index,
                        targets,
                    },
                )?;
            }
        }
        "kmeans" => {
            let mut artifact = SimpleKmeansRoutingArtifact::read_json(&request.artifact, None)?;
            ensure!(
                artifact.vector_schema.dimension == request.dimension,
                "dimension mismatch"
            );
            ensure!(
                artifact.shard_count == request.shard_count
                    && artifact.logical_point_count == request.point_count
                    && artifact.physical_point_count == request.physical_point_count
                    && artifact.layout_sha256 == request.layout_sha256
                    && artifact.vector_schema.distance == request.distance
                    && serde_json::to_value(artifact.vector_schema.datatype)? == request.datatype,
                "KMeans artifact does not match the experiment dataset/layout/schema"
            );
            artifact.nprobe = request
                .nprobe
                .ok_or_else(|| anyhow::anyhow!("nprobe required"))?;
            artifact.lower_hnsw_ef = request
                .lower_hnsw_ef
                .ok_or_else(|| anyhow::anyhow!("EF required"))?;
            let router = SimpleKmeansRouter::new(artifact)?;
            for (query_index, query) in queries.iter().enumerate() {
                let targets = router
                    .route_query(query)?
                    .into_iter()
                    .map(|t| Target {
                        shard_id: t.shard_id,
                        ef: t.ef,
                        entry_points: None,
                    })
                    .collect();
                json_line(
                    writer,
                    &RouteRow {
                        query_index,
                        targets,
                    },
                )?;
            }
        }
        _ => bail!("unknown method"),
    }
    Ok(())
}

fn lower(request: LowerRequest, writer: &mut impl Write) -> Result<()> {
    ensure!(
        request.m > 0 && request.ef_construct > 0 && request.top_k > 0,
        "invalid HNSW parameters"
    );
    let bytes = match request.vector_encoding.as_str() {
        "f32le" => false,
        "u8" => true,
        _ => bail!("unsupported vector encoding"),
    };
    let byte_storage = match request.datatype.as_str() {
        "float32" => false,
        "uint8" => true,
        _ => bail!("unsupported vector datatype"),
    };
    ensure!(
        fs::metadata(&request.vectors)?.len()
            == (request.point_count as u64)
                * (request.dimension as u64)
                * if bytes { 1 } else { 4 },
        "base-vector file size mismatch"
    );
    let raw_ids = fs::read(&request.ids)?;
    ensure!(raw_ids.len() % 8 == 0, "invalid ID file length");
    let ids: Vec<u64> = raw_ids
        .chunks_exact(8)
        .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    drop(raw_ids);
    ensure!(
        ids.windows(2).all(|w| w[0] < w[1]),
        "IDs must be sorted and unique"
    );
    ensure!(
        ids.iter().all(|&x| x < request.point_count as u64),
        "source ID out of range"
    );
    let count = u32::try_from(ids.len())?;
    let queries = queries(&request.queries, request.query_count, request.dimension)?;
    let mut storage = if byte_storage {
        VectorStorageEnum::DenseVolatileByte(VolatileDenseVectorStorage::new(
            request.dimension,
            request.distance,
        ))
    } else {
        new_volatile_dense_vector_storage(request.dimension, request.distance)
    };
    let mut source = BufReader::new(File::open(&request.vectors)?);
    let stride = (request.dimension * if bytes { 1 } else { 4 }) as u64;
    for (offset, &id) in ids.iter().enumerate() {
        source.seek(SeekFrom::Start(id * stride))?;
        let vector = read_vector(&mut source, request.dimension, bytes)?;
        let vector = if byte_storage {
            request.distance.preprocess_vector::<u8>(vector)
        } else {
            request.distance.preprocess_vector::<f32>(vector)
        };
        storage.insert_vector(
            offset as u32,
            VectorRef::from(&vector),
            &HardwareCounterCell::disposable(),
        )?;
    }
    // Same parameters and deterministic insertion order for both methods.
    let deleted = BitVec::repeat(false, ids.len());
    let mut builder = GraphLayersBuilder::new(
        ids.len(),
        HnswM::new2(request.m),
        request.ef_construct,
        1,
        true,
    );
    let mut rng = StdRng::seed_from_u64(request.seed);
    for id in 0..count {
        let level = builder.get_random_layer(&mut rng);
        builder.set_levels(id, level);
    }
    for id in 0..count {
        let scorer = FilteredScorer::new_internal(
            id,
            &storage,
            None,
            None,
            &deleted,
            HardwareCounterCell::disposable(),
        )?;
        builder.link_new_point(id, scorer);
        if (id + 1) % 100_000 == 0 {
            eprintln!("lower HNSW: {}/{count}", id + 1);
        }
    }
    fs::create_dir_all(&request.work_dir)?;
    let temporary = tempfile::tempdir_in(&request.work_dir)?;
    let graph = builder.into_graph_layers(temporary.path(), GraphLinksFormatParam::Plain, false)?;
    let stopped = AtomicBool::new(false);
    for line in BufReader::new(File::open(&request.tasks)?).lines() {
        let task: Task = serde_json::from_str(&line?)?;
        ensure!(
            task.query_index < queries.len() && task.target.ef > 0,
            "invalid search task"
        );
        let entries = task
            .target
            .entry_points
            .map(|points| {
                points
                    .into_iter()
                    .map(|id| {
                        ids.binary_search(&id).map(|x| x as u32).map_err(|_| {
                            anyhow::anyhow!("Orion owner entry {id} is missing from its shard")
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;
        ensure!(
            entries.as_ref().is_none_or(|v| !v.is_empty()),
            "empty explicit entries"
        );
        let scorer = FilteredScorer::new(
            queries[task.query_index].clone().into(),
            &storage,
            None,
            None,
            &deleted,
            HardwareCounterCell::disposable(),
        )?;
        let points = graph
            .search(
                request.top_k,
                task.target.ef,
                SearchAlgorithm::Hnsw,
                scorer,
                entries.as_deref(),
                &stopped,
            )
            .map_err(|_| anyhow::anyhow!("local HNSW search cancelled"))?
            .into_iter()
            .map(|p| Point {
                id: ids[p.idx as usize],
                score: {
                    let score = request.distance.postprocess_score(p.score);
                    if request.distance == Distance::Euclid {
                        -score
                    } else {
                        score
                    }
                },
            })
            .collect::<Vec<_>>();
        ensure!(
            points.iter().all(|p| p.score.is_finite()),
            "non-finite score"
        );
        json_line(
            writer,
            &serde_json::json!({"profile": task.profile,
            "query_index": task.query_index, "points": points}),
        )?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() == 2 && args[1] == "--build-capabilities" {
        println!("{}", serde_json::json!({
            "schema_version": 1,
            "unrestricted_positive_indexing_threads": true
        }));
        return Ok(());
    }
    if args.len() != 4 {
        bail!("Usage: shard_locality <route|lower|segments|build-segment> <request.json> <new-output.jsonl> | --build-capabilities");
    }
    // Python publishes output only after this worker exits successfully.
    let mut writer = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&args[3])?,
    );
    let reader = BufReader::new(File::open(&args[2])?);
    match args[1].to_str() {
        Some("route") => route(serde_json::from_reader(reader)?, &mut writer)?,
        Some("lower") => lower(serde_json::from_reader(reader)?, &mut writer)?,
        Some("segments") => search_segments(serde_json::from_reader(reader)?, &mut writer)?,
        Some("build-segment") => {
            build_local_segment(serde_json::from_reader(reader)?, &mut writer)?
        }
        _ => bail!("unknown command"),
    }
    writer.flush()?;
    Ok(())
}
