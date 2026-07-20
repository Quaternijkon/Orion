//! Controlled offline importer for static Orion and Simple KMeans collections.
//!
//! The importer deliberately uses Qdrant's internal `PointsInternal/Upsert`
//! endpoint with an explicit numeric shard ID. This lets an offline routing
//! build place external point IDs in numeric logical shards without custom
//! shard keys, synthetic IDs, or routing payloads. Medium/strong write ordering
//! is required so the selected `ShardReplicaSet` still applies the collection's
//! normal replication and write-consistency rules.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{BufRead, BufReader, Read};
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use api::grpc::qdrant::auto_shard_policy::Policy as AutoShardPolicyVariant;
use api::grpc::qdrant::collections_internal_client::CollectionsInternalClient;
use api::grpc::qdrant::point_id::PointIdOptions;
use api::grpc::qdrant::points_internal_client::PointsInternalClient;
use api::grpc::qdrant::vector::Vector as VectorVariant;
use api::grpc::qdrant::vectors::VectorsOptions;
use api::grpc::qdrant::vectors_config::Config as VectorsConfigVariant;
use api::grpc::qdrant::{
    CollectionInfo, CountPoints, CountPointsInternal, Datatype, DenseVector, Distance,
    GetCollectionInfoRequest, GetCollectionInfoRequestInternal, NamedVectors, PointId, PointStruct,
    ShardingMethod, UpdateMode, UpdateStatus, UpsertPoints, UpsertPointsInternal, Vector, Vectors,
    WaitUntil, WriteOrdering, WriteOrderingType,
};
use clap::{Parser, ValueEnum};
use fs_err::File;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tonic::metadata::MetadataValue;
use tonic::transport::{Channel, Endpoint};
use tonic::{Request, codec::CompressionEncoding};

const LEGACY_ORION_MANIFEST_FORMAT_VERSION: u32 = 1;
const GENERIC_MANIFEST_FORMAT_VERSION: u32 = 2;
const CHECKPOINT_FORMAT_VERSION: u32 = 2;

#[derive(Debug, Parser)]
#[command(
    name = "orion-numeric-shard-import",
    about = "Import an offline static-routing layout into numeric Qdrant shards",
    long_about = None
)]
struct Args {
    /// Manifest describing the f32 vector file and JSONL shard assignments.
    #[arg(long)]
    manifest: PathBuf,

    /// Qdrant internal gRPC endpoint (normally the P2P port, e.g. 6335).
    #[arg(long)]
    uri: String,

    /// Coordinator HTTP endpoint used to discover peer URIs and shard placement.
    /// Defaults to the same host as --uri with port reduced by two (6335 -> 6333).
    #[arg(long)]
    http_url: Option<String>,

    /// Target collection name.
    #[arg(long)]
    collection: String,

    /// Number of logical points read before flushing per-shard requests.
    #[arg(long, default_value = "512")]
    batch_size: NonZeroUsize,

    /// Server and gRPC request timeout.
    #[arg(long, default_value = "120")]
    request_timeout_secs: NonZeroU64,

    /// Wait until each upsert reaches this durability/visibility stage.
    #[arg(long, value_enum, default_value_t = ImportWait::Visible)]
    wait: ImportWait,

    /// Medium is normally preferred; weak is intentionally unsupported because
    /// an explicit internal shard request with weak ordering only updates the
    /// receiving local replica.
    #[arg(long, value_enum, default_value_t = ImportOrdering::Medium)]
    ordering: ImportOrdering,

    /// Optional Qdrant API key for secured internal endpoints.
    #[arg(long, env = "QDRANT_API_KEY", hide_env_values = true)]
    api_key: Option<String>,

    /// Resume an interrupted import using the matching local checkpoint. Without
    /// this flag every numeric shard must be empty before the first RPC.
    #[arg(long)]
    resume: bool,

    /// Validate all local files without opening a network connection.
    #[arg(long)]
    dry_run: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ImportWait {
    Wal,
    Segment,
    Visible,
}

impl ImportWait {
    fn grpc(self) -> WaitUntil {
        match self {
            Self::Wal => WaitUntil::Wal,
            Self::Segment => WaitUntil::Segment,
            Self::Visible => WaitUntil::Visible,
        }
    }

    fn public_wait(self) -> bool {
        matches!(self, Self::Visible)
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ImportOrdering {
    Medium,
    Strong,
}

impl ImportOrdering {
    fn grpc(self) -> WriteOrderingType {
        match self {
            Self::Medium => WriteOrderingType::Medium,
            Self::Strong => WriteOrderingType::Strong,
        }
    }
}

/// Normalized import format shared by legacy Orion v1 and generic v2 manifests.
///
/// `vectors_file` is exactly `point_count * dimension` little-endian f32
/// values in row-major order. `assignments_file` is UTF-8 JSONL with exactly
/// one record per vector row, for example:
///
/// `{"id": 42, "shards": [3, 7]}`
#[derive(Debug)]
struct ImportManifest {
    format_version: u32,
    dimension: usize,
    point_count: usize,
    shard_count: u32,
    total_point_copies: u64,
    routing: StaticRoutingBinding,
    vector_name: String,
    vectors_file: PathBuf,
    vectors_sha256: String,
    assignments_file: PathBuf,
    assignments_sha256: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum StaticRoutingPolicy {
    Orion,
    SimpleKmeans,
}

impl StaticRoutingPolicy {
    fn as_str(self) -> &'static str {
        match self {
            Self::Orion => "orion",
            Self::SimpleKmeans => "simple_kmeans",
        }
    }

    fn display_name(self) -> &'static str {
        match self {
            Self::Orion => "Orion",
            Self::SimpleKmeans => "Simple KMeans",
        }
    }
}

#[derive(Debug)]
struct StaticRoutingBinding {
    policy: StaticRoutingPolicy,
    generation: u64,
    artifact_file: PathBuf,
    artifact_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyOrionImportManifest {
    format_version: u32,
    dimension: usize,
    point_count: usize,
    shard_count: u32,
    total_point_copies: u64,
    orion_generation: u64,
    orion_artifact_file: PathBuf,
    orion_artifact_sha256: String,
    #[serde(default)]
    vector_name: String,
    vectors_file: PathBuf,
    vectors_sha256: String,
    assignments_file: PathBuf,
    assignments_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericImportManifest {
    format_version: u32,
    routing_policy: StaticRoutingPolicy,
    routing_generation: u64,
    routing_artifact_file: PathBuf,
    routing_artifact_sha256: String,
    dimension: usize,
    point_count: usize,
    shard_count: u32,
    total_point_copies: u64,
    #[serde(default)]
    vector_name: String,
    vectors_file: PathBuf,
    vectors_sha256: String,
    assignments_file: PathBuf,
    assignments_sha256: String,
}

#[derive(Debug, Deserialize)]
struct OrionArtifactBinding {
    format_version: u32,
    generation: u64,
    vector_schema: ArtifactVectorSchema,
    shard_count: u32,
    layout_sha256: String,
    logical_point_count: u64,
    physical_point_count: u64,
    upper_graph: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct ArtifactVectorSchema {
    vector_name: String,
    dimension: usize,
    distance: String,
    datatype: String,
}

#[derive(Debug, Deserialize)]
struct SimpleKmeansArtifactBinding {
    format_version: u32,
    generation: u64,
    vector_schema: ArtifactVectorSchema,
    shard_count: u32,
    layout_sha256: String,
    logical_point_count: u64,
    physical_point_count: u64,
    routing_distance: String,
    nprobe: usize,
    lower_hnsw_ef: usize,
    centroids: Vec<SimpleKmeansCentroidBinding>,
}

#[derive(Debug, Deserialize)]
struct SimpleKmeansCentroidBinding {
    shard_id: u32,
    vector: Vec<f32>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, Deserialize)]
#[serde(untagged)]
enum ManifestPointId {
    Number(u64),
    Uuid(String),
}

impl ManifestPointId {
    fn validate(&self) -> Result<()> {
        if let Self::Uuid(value) = self {
            uuid::Uuid::parse_str(value)
                .with_context(|| format!("point ID {value:?} is not a valid UUID"))?;
        }
        Ok(())
    }

    fn to_grpc(&self) -> PointId {
        let point_id_options = match self {
            Self::Number(value) => PointIdOptions::Num(*value),
            Self::Uuid(value) => PointIdOptions::Uuid(value.clone()),
        };
        PointId {
            point_id_options: Some(point_id_options),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AssignmentRecord {
    id: ManifestPointId,
    shards: Vec<u32>,
}

#[derive(Debug)]
struct ValidatedInput {
    manifest_path: PathBuf,
    manifest: ImportManifest,
    vectors_path: PathBuf,
    assignments_path: PathBuf,
    vector_distance: Distance,
    vector_bytes_per_point: usize,
    total_copies: u64,
    copies_per_shard: Vec<u64>,
}

#[derive(Debug, Eq, PartialEq)]
struct ImportCheckpoint {
    collection: String,
    manifest_sha256: String,
    vectors_sha256: String,
    assignments_sha256: String,
    routing_policy: StaticRoutingPolicy,
    routing_generation: u64,
    routing_artifact_sha256: String,
    status: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyOrionImportCheckpoint {
    format_version: u32,
    collection: String,
    manifest_sha256: String,
    vectors_sha256: String,
    assignments_sha256: String,
    orion_generation: u64,
    orion_artifact_sha256: String,
    status: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct GenericImportCheckpoint {
    format_version: u32,
    collection: String,
    manifest_sha256: String,
    vectors_sha256: String,
    assignments_sha256: String,
    routing_policy: StaticRoutingPolicy,
    routing_generation: u64,
    routing_artifact_sha256: String,
    status: String,
}

#[derive(Debug)]
struct ShardPlacement {
    owner_uri_by_shard: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct RestEnvelope<T> {
    result: T,
}

#[derive(Debug, Deserialize)]
struct ClusterPeer {
    uri: String,
}

#[derive(Debug, Deserialize)]
struct ClusterState {
    peer_id: u64,
    peers: HashMap<String, ClusterPeer>,
}

#[derive(Debug, Deserialize)]
struct CollectionClusterState {
    peer_id: u64,
    shard_count: u64,
    #[serde(default)]
    local_shards: Vec<LocalShardPlacement>,
    #[serde(default)]
    remote_shards: Vec<RemoteShardPlacement>,
    #[serde(default)]
    shard_transfers: Vec<serde_json::Value>,
    #[serde(default)]
    resharding_operations: Vec<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct LocalShardPlacement {
    shard_id: u32,
    state: String,
}

#[derive(Debug, Deserialize)]
struct RemoteShardPlacement {
    shard_id: u32,
    peer_id: u64,
    state: String,
}

#[derive(Default)]
struct ImportStats {
    logical_points: u64,
    point_copies: u64,
    grpc_requests: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let input = validate_input(&args.manifest)?;

    println!(
        "validated format=v{} policy={} generation={} points={} copies={} dimension={} shards={} vector_name={:?}",
        input.manifest.format_version,
        input.manifest.routing.policy.as_str(),
        input.manifest.routing.generation,
        input.manifest.point_count,
        input.total_copies,
        input.manifest.dimension,
        input.manifest.shard_count,
        input.manifest.vector_name,
    );

    if args.dry_run {
        println!("dry-run complete; no gRPC requests were sent");
        return Ok(());
    }
    validate_checkpoint_mode(&args, &input)?;

    let timeout = Duration::from_secs(args.request_timeout_secs.get());
    let endpoint = Endpoint::from_shared(args.uri.clone())?
        .connect_timeout(timeout)
        .timeout(timeout);
    let channel = endpoint
        .connect()
        .await
        .with_context(|| format!("failed to connect to internal gRPC endpoint {}", args.uri))?;
    let mut points_client = PointsInternalClient::new(channel)
        .send_compressed(CompressionEncoding::Gzip)
        .accept_compressed(CompressionEncoding::Gzip);
    let placement = discover_shard_owners(&args, &input).await?;
    preflight_collection(&placement, &args, &input).await?;
    write_checkpoint(&args, &input, "in_progress")?;
    let stats = import(&input, &args, &mut points_client).await?;
    verify_imported_counts(&placement, &args, &input).await?;
    write_checkpoint(&args, &input, "complete")?;
    println!(
        "import complete: logical_points={} point_copies={} grpc_requests={}",
        stats.logical_points, stats.point_copies, stats.grpc_requests,
    );
    Ok(())
}

fn read_import_manifest(manifest_path: &Path) -> Result<ImportManifest> {
    let manifest_file = File::open(manifest_path)
        .with_context(|| format!("failed to open manifest {}", manifest_path.display()))?;
    let value: serde_json::Value = serde_json::from_reader(BufReader::new(manifest_file))
        .with_context(|| format!("failed to parse manifest {}", manifest_path.display()))?;
    let format_version = value
        .get("format_version")
        .and_then(serde_json::Value::as_u64)
        .context("manifest format_version must be an unsigned integer")?;
    match u32::try_from(format_version).context("manifest format_version does not fit in u32")? {
        LEGACY_ORION_MANIFEST_FORMAT_VERSION => {
            let legacy: LegacyOrionImportManifest =
                serde_json::from_value(value).with_context(|| {
                    format!(
                        "failed to parse legacy Orion manifest {}",
                        manifest_path.display()
                    )
                })?;
            Ok(ImportManifest {
                format_version: legacy.format_version,
                dimension: legacy.dimension,
                point_count: legacy.point_count,
                shard_count: legacy.shard_count,
                total_point_copies: legacy.total_point_copies,
                routing: StaticRoutingBinding {
                    policy: StaticRoutingPolicy::Orion,
                    generation: legacy.orion_generation,
                    artifact_file: legacy.orion_artifact_file,
                    artifact_sha256: legacy.orion_artifact_sha256,
                },
                vector_name: legacy.vector_name,
                vectors_file: legacy.vectors_file,
                vectors_sha256: legacy.vectors_sha256,
                assignments_file: legacy.assignments_file,
                assignments_sha256: legacy.assignments_sha256,
            })
        }
        GENERIC_MANIFEST_FORMAT_VERSION => {
            let generic: GenericImportManifest =
                serde_json::from_value(value).with_context(|| {
                    format!(
                        "failed to parse generic routing manifest {}",
                        manifest_path.display()
                    )
                })?;
            Ok(ImportManifest {
                format_version: generic.format_version,
                dimension: generic.dimension,
                point_count: generic.point_count,
                shard_count: generic.shard_count,
                total_point_copies: generic.total_point_copies,
                routing: StaticRoutingBinding {
                    policy: generic.routing_policy,
                    generation: generic.routing_generation,
                    artifact_file: generic.routing_artifact_file,
                    artifact_sha256: generic.routing_artifact_sha256,
                },
                vector_name: generic.vector_name,
                vectors_file: generic.vectors_file,
                vectors_sha256: generic.vectors_sha256,
                assignments_file: generic.assignments_file,
                assignments_sha256: generic.assignments_sha256,
            })
        }
        unsupported => bail!(
            "unsupported format_version {unsupported}; expected legacy Orion version {LEGACY_ORION_MANIFEST_FORMAT_VERSION} or generic version {GENERIC_MANIFEST_FORMAT_VERSION}"
        ),
    }
}

fn validate_input(manifest_path: &Path) -> Result<ValidatedInput> {
    let manifest = read_import_manifest(manifest_path)?;
    ensure!(
        manifest.dimension > 0,
        "dimension must be greater than zero"
    );
    ensure!(
        manifest.point_count > 0,
        "point_count must be greater than zero"
    );
    ensure!(
        manifest.shard_count > 0,
        "shard_count must be greater than zero"
    );
    ensure!(
        manifest.routing.generation > 0,
        "routing_generation must be greater than zero"
    );

    let manifest_dir = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let vectors_path = manifest_dir.join(&manifest.vectors_file);
    let assignments_path = manifest_dir.join(&manifest.assignments_file);
    let artifact_path = manifest_dir.join(&manifest.routing.artifact_file);
    let vector_bytes_per_point = manifest
        .dimension
        .checked_mul(size_of::<f32>())
        .context("dimension overflows vector row byte length")?;
    let expected_vector_bytes = manifest
        .point_count
        .checked_mul(vector_bytes_per_point)
        .context("point_count and dimension overflow vector file byte length")?;
    let actual_vector_bytes = usize::try_from(
        fs_err::metadata(&vectors_path)
            .with_context(|| format!("failed to stat {}", vectors_path.display()))?
            .len(),
    )
    .context("vector file is too large for this platform")?;
    ensure!(
        actual_vector_bytes == expected_vector_bytes,
        "vector file {} has {} bytes; expected exactly {} ({} points x {} dimensions x 4)",
        vectors_path.display(),
        actual_vector_bytes,
        expected_vector_bytes,
        manifest.point_count,
        manifest.dimension,
    );
    validate_sha256("vectors_sha256", &manifest.vectors_sha256)?;
    validate_sha256("assignments_sha256", &manifest.assignments_sha256)?;
    validate_sha256("routing_artifact_sha256", &manifest.routing.artifact_sha256)?;
    verify_file_sha256(&vectors_path, &manifest.vectors_sha256, "vector file")?;
    verify_file_sha256(
        &assignments_path,
        &manifest.assignments_sha256,
        "assignment file",
    )?;
    verify_file_sha256(
        &artifact_path,
        &manifest.routing.artifact_sha256,
        "static routing artifact",
    )?;
    let vector_distance = validate_routing_artifact(&artifact_path, &manifest)?;

    let assignments_file = File::open(&assignments_path)
        .with_context(|| format!("failed to open {}", assignments_path.display()))?;
    let mut seen_ids = HashSet::with_capacity(manifest.point_count.min(1_000_000));
    let mut record_count = 0usize;
    let mut total_copies = 0u64;
    let shard_count = usize::try_from(manifest.shard_count).unwrap();
    let mut copies_per_shard = vec![0_u64; shard_count];
    for (line_index, line) in BufReader::new(assignments_file).lines().enumerate() {
        let line_number = line_index + 1;
        let line = line.with_context(|| {
            format!(
                "failed reading assignment line {line_number} from {}",
                assignments_path.display()
            )
        })?;
        ensure!(
            !line.trim().is_empty(),
            "assignment line {line_number} is empty"
        );
        let record: AssignmentRecord = serde_json::from_str(&line)
            .with_context(|| format!("invalid assignment JSON at line {line_number}"))?;
        validate_assignment(&record, manifest.shard_count, line_number)?;
        ensure!(
            seen_ids.insert(record.id),
            "duplicate point ID at assignment line {line_number}; each vector row must have one unique external ID"
        );
        record_count += 1;
        total_copies = total_copies
            .checked_add(u64::try_from(record.shards.len()).unwrap())
            .context("total point-copy count overflowed u64")?;
        for shard_id in record.shards {
            let shard_copies = &mut copies_per_shard[usize::try_from(shard_id).unwrap()];
            *shard_copies = (*shard_copies)
                .checked_add(1)
                .context("per-shard point-copy count overflowed u64")?;
        }
    }
    ensure!(
        record_count == manifest.point_count,
        "assignment file has {record_count} records; manifest declares {} points",
        manifest.point_count
    );
    ensure!(
        total_copies == manifest.total_point_copies,
        "assignment file contains {total_copies} point copies; manifest declares {}",
        manifest.total_point_copies
    );

    Ok(ValidatedInput {
        manifest_path: manifest_path.to_path_buf(),
        manifest,
        vectors_path,
        assignments_path,
        vector_distance,
        vector_bytes_per_point,
        total_copies,
        copies_per_shard,
    })
}

#[allow(clippy::too_many_arguments)]
fn validate_artifact_common(
    policy: StaticRoutingPolicy,
    format_version: u32,
    generation: u64,
    vector_schema: &ArtifactVectorSchema,
    shard_count: u32,
    layout_sha256: &str,
    logical_point_count: u64,
    physical_point_count: u64,
    manifest: &ImportManifest,
) -> Result<()> {
    let name = policy.display_name();
    ensure!(
        format_version == 1,
        "{name} artifact format_version is {format_version}; expected 1"
    );
    ensure!(
        generation == manifest.routing.generation,
        "{name} artifact generation is {generation}; manifest requires {}",
        manifest.routing.generation
    );
    ensure!(
        shard_count == manifest.shard_count,
        "{name} artifact shard_count is {shard_count}; manifest requires {}",
        manifest.shard_count
    );
    ensure!(
        layout_sha256.eq_ignore_ascii_case(&manifest.assignments_sha256),
        "{name} artifact layout_sha256 does not match the canonical assignment file"
    );
    ensure!(
        logical_point_count == u64::try_from(manifest.point_count).unwrap(),
        "{name} artifact logical_point_count is {logical_point_count}; manifest requires {}",
        manifest.point_count
    );
    ensure!(
        physical_point_count == manifest.total_point_copies,
        "{name} artifact physical_point_count is {physical_point_count}; manifest requires {}",
        manifest.total_point_copies
    );
    ensure!(
        vector_schema.vector_name == manifest.vector_name,
        "{name} artifact vector name does not match the import manifest"
    );
    ensure!(
        vector_schema.dimension == manifest.dimension,
        "{name} artifact vector dimension is {}; manifest requires {}",
        vector_schema.dimension,
        manifest.dimension
    );
    ensure!(
        vector_schema.datatype.eq_ignore_ascii_case("float32"),
        "numeric importer requires a float32 {name} artifact"
    );
    Ok(())
}

fn parse_artifact_distance(distance: &str) -> Result<Distance> {
    match distance.to_ascii_lowercase().as_str() {
        "cosine" => Ok(Distance::Cosine),
        "dot" => Ok(Distance::Dot),
        "euclid" => Ok(Distance::Euclid),
        "manhattan" => Ok(Distance::Manhattan),
        _ => bail!("artifact contains unsupported vector distance {distance:?}"),
    }
}

fn validate_routing_artifact(artifact_path: &Path, manifest: &ImportManifest) -> Result<Distance> {
    let bytes = fs_err::read(artifact_path)
        .with_context(|| format!("failed to read {}", artifact_path.display()))?;
    let distance = match manifest.routing.policy {
        StaticRoutingPolicy::Orion => {
            let artifact: OrionArtifactBinding =
                serde_json::from_slice(&bytes).with_context(|| {
                    format!("failed to parse Orion artifact {}", artifact_path.display())
                })?;
            validate_artifact_common(
                StaticRoutingPolicy::Orion,
                artifact.format_version,
                artifact.generation,
                &artifact.vector_schema,
                artifact.shard_count,
                &artifact.layout_sha256,
                artifact.logical_point_count,
                artifact.physical_point_count,
                manifest,
            )?;
            ensure!(
                artifact.upper_graph.is_some(),
                "numeric importer requires an Orion production artifact with upper_graph"
            );
            parse_artifact_distance(&artifact.vector_schema.distance)?
        }
        StaticRoutingPolicy::SimpleKmeans => {
            let value: serde_json::Value = serde_json::from_slice(&bytes).with_context(|| {
                format!(
                    "failed to parse Simple KMeans artifact {}",
                    artifact_path.display()
                )
            })?;
            ensure!(
                value.get("upper_graph").is_none(),
                "Simple KMeans artifact must not contain upper_graph"
            );
            let artifact: SimpleKmeansArtifactBinding = serde_json::from_value(value)
                .with_context(|| {
                    format!(
                        "failed to parse Simple KMeans artifact {}",
                        artifact_path.display()
                    )
                })?;
            validate_artifact_common(
                StaticRoutingPolicy::SimpleKmeans,
                artifact.format_version,
                artifact.generation,
                &artifact.vector_schema,
                artifact.shard_count,
                &artifact.layout_sha256,
                artifact.logical_point_count,
                artifact.physical_point_count,
                manifest,
            )?;
            ensure!(
                artifact.logical_point_count == artifact.physical_point_count,
                "single-assignment Simple KMeans requires physical_point_count to equal logical_point_count"
            );
            ensure!(
                manifest.total_point_copies == u64::try_from(manifest.point_count).unwrap(),
                "single-assignment Simple KMeans requires exactly one physical point copy per logical point"
            );
            ensure!(
                artifact.routing_distance == "squared_l2",
                "Simple KMeans routing_distance must be squared_l2"
            );
            ensure!(
                artifact.nprobe > 0 && artifact.nprobe <= artifact.shard_count as usize,
                "Simple KMeans nprobe {} must be between 1 and shard_count {}",
                artifact.nprobe,
                artifact.shard_count
            );
            ensure!(
                artifact.lower_hnsw_ef > 0,
                "Simple KMeans lower_hnsw_ef must be greater than zero"
            );
            ensure!(
                artifact.centroids.len() == artifact.shard_count as usize,
                "Simple KMeans artifact has {} centroids; expected exactly {}",
                artifact.centroids.len(),
                artifact.shard_count
            );
            let mut seen_shards = HashSet::with_capacity(artifact.centroids.len());
            for centroid in artifact.centroids {
                ensure!(
                    centroid.shard_id < artifact.shard_count,
                    "Simple KMeans centroid references out-of-range shard {}",
                    centroid.shard_id
                );
                ensure!(
                    seen_shards.insert(centroid.shard_id),
                    "Simple KMeans artifact repeats centroid for shard {}",
                    centroid.shard_id
                );
                ensure!(
                    centroid.vector.len() == manifest.dimension,
                    "Simple KMeans centroid for shard {} has dimension {}; expected {}",
                    centroid.shard_id,
                    centroid.vector.len(),
                    manifest.dimension
                );
                ensure!(
                    centroid.vector.iter().all(|value| value.is_finite()),
                    "Simple KMeans centroid for shard {} contains a non-finite value",
                    centroid.shard_id
                );
            }
            parse_artifact_distance(&artifact.vector_schema.distance)?
        }
    };
    Ok(distance)
}

fn validate_sha256(field: &str, value: &str) -> Result<()> {
    ensure!(
        value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "{field} must contain exactly 64 hexadecimal characters"
    );
    Ok(())
}

fn verify_file_sha256(path: &Path, expected: &str, description: &str) -> Result<()> {
    let actual = file_sha256(path)?;
    ensure!(
        actual.eq_ignore_ascii_case(expected),
        "{description} {} has SHA-256 {actual}; manifest declares {expected}",
        path.display()
    );
    Ok(())
}

fn file_sha256(path: &Path) -> Result<String> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 8 * 1024 * 1024];
    loop {
        let read = reader
            .read(&mut buffer)
            .with_context(|| format!("failed to hash {}", path.display()))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn checkpoint_path(manifest_path: &Path) -> PathBuf {
    let mut path = manifest_path.as_os_str().to_owned();
    path.push(".import-state.json");
    PathBuf::from(path)
}

fn expected_checkpoint(
    args: &Args,
    input: &ValidatedInput,
    status: &str,
) -> Result<ImportCheckpoint> {
    Ok(ImportCheckpoint {
        collection: args.collection.clone(),
        manifest_sha256: file_sha256(&input.manifest_path)?,
        vectors_sha256: input.manifest.vectors_sha256.clone(),
        assignments_sha256: input.manifest.assignments_sha256.clone(),
        routing_policy: input.manifest.routing.policy,
        routing_generation: input.manifest.routing.generation,
        routing_artifact_sha256: input.manifest.routing.artifact_sha256.clone(),
        status: status.to_string(),
    })
}

fn read_checkpoint(path: &Path) -> Result<ImportCheckpoint> {
    let value: serde_json::Value = serde_json::from_reader(BufReader::new(
        File::open(path)
            .with_context(|| format!("failed to open checkpoint {}", path.display()))?,
    ))
    .with_context(|| format!("failed to parse checkpoint {}", path.display()))?;
    let format_version = value
        .get("format_version")
        .and_then(serde_json::Value::as_u64)
        .context("checkpoint format_version must be an unsigned integer")?;
    match u32::try_from(format_version).context("checkpoint format_version does not fit in u32")? {
        1 => {
            let legacy: LegacyOrionImportCheckpoint = serde_json::from_value(value)?;
            ensure!(
                legacy.format_version == 1,
                "invalid legacy checkpoint version"
            );
            Ok(ImportCheckpoint {
                collection: legacy.collection,
                manifest_sha256: legacy.manifest_sha256,
                vectors_sha256: legacy.vectors_sha256,
                assignments_sha256: legacy.assignments_sha256,
                routing_policy: StaticRoutingPolicy::Orion,
                routing_generation: legacy.orion_generation,
                routing_artifact_sha256: legacy.orion_artifact_sha256,
                status: legacy.status,
            })
        }
        CHECKPOINT_FORMAT_VERSION => {
            let generic: GenericImportCheckpoint = serde_json::from_value(value)?;
            Ok(ImportCheckpoint {
                collection: generic.collection,
                manifest_sha256: generic.manifest_sha256,
                vectors_sha256: generic.vectors_sha256,
                assignments_sha256: generic.assignments_sha256,
                routing_policy: generic.routing_policy,
                routing_generation: generic.routing_generation,
                routing_artifact_sha256: generic.routing_artifact_sha256,
                status: generic.status,
            })
        }
        unsupported => bail!("unsupported checkpoint format_version {unsupported}"),
    }
}

fn validate_checkpoint_mode(args: &Args, input: &ValidatedInput) -> Result<()> {
    let path = checkpoint_path(&input.manifest_path);
    if !args.resume {
        ensure!(
            !path.exists(),
            "import checkpoint {} already exists; pass --resume only if this is the same interrupted import, or recreate the empty collection and remove the checkpoint",
            path.display()
        );
        return Ok(());
    }
    ensure!(
        path.exists(),
        "--resume requires checkpoint {}",
        path.display()
    );
    let stored = read_checkpoint(&path)?;
    let expected = expected_checkpoint(args, input, &stored.status)?;
    ensure!(
        stored.collection == expected.collection
            && stored
                .manifest_sha256
                .eq_ignore_ascii_case(&expected.manifest_sha256)
            && stored
                .vectors_sha256
                .eq_ignore_ascii_case(&expected.vectors_sha256)
            && stored
                .assignments_sha256
                .eq_ignore_ascii_case(&expected.assignments_sha256)
            && stored.routing_policy == expected.routing_policy
            && stored.routing_generation == expected.routing_generation
            && stored
                .routing_artifact_sha256
                .eq_ignore_ascii_case(&expected.routing_artifact_sha256),
        "checkpoint {} does not describe this exact collection, manifest, vectors, assignments, routing policy, and artifact",
        path.display()
    );
    ensure!(
        matches!(stored.status.as_str(), "in_progress" | "complete"),
        "checkpoint {} has unsupported status {:?}",
        path.display(),
        stored.status
    );
    Ok(())
}

fn write_checkpoint(args: &Args, input: &ValidatedInput, status: &str) -> Result<()> {
    let path = checkpoint_path(&input.manifest_path);
    let checkpoint = expected_checkpoint(args, input, status)?;
    let checkpoint = GenericImportCheckpoint {
        format_version: CHECKPOINT_FORMAT_VERSION,
        collection: checkpoint.collection,
        manifest_sha256: checkpoint.manifest_sha256,
        vectors_sha256: checkpoint.vectors_sha256,
        assignments_sha256: checkpoint.assignments_sha256,
        routing_policy: checkpoint.routing_policy,
        routing_generation: checkpoint.routing_generation,
        routing_artifact_sha256: checkpoint.routing_artifact_sha256,
        status: checkpoint.status,
    };
    let mut tmp_os = path.as_os_str().to_owned();
    tmp_os.push(".tmp");
    let tmp = PathBuf::from(tmp_os);
    fs_err::write(&tmp, serde_json::to_vec_pretty(&checkpoint)?)
        .with_context(|| format!("failed to write checkpoint {}", tmp.display()))?;
    fs_err::rename(&tmp, &path)
        .with_context(|| format!("failed to publish checkpoint {}", path.display()))?;
    Ok(())
}

fn validate_assignment(
    record: &AssignmentRecord,
    shard_count: u32,
    line_number: usize,
) -> Result<()> {
    record
        .id
        .validate()
        .with_context(|| format!("invalid point ID at assignment line {line_number}"))?;
    ensure!(
        !record.shards.is_empty(),
        "assignment line {line_number} has no target shards"
    );
    let mut unique = HashSet::with_capacity(record.shards.len());
    for &shard_id in &record.shards {
        ensure!(
            shard_id < shard_count,
            "assignment line {line_number} targets shard {shard_id}, but shard_count is {shard_count}"
        );
        ensure!(
            unique.insert(shard_id),
            "assignment line {line_number} repeats shard {shard_id}"
        );
    }
    Ok(())
}

async fn discover_shard_owners(args: &Args, input: &ValidatedInput) -> Result<ShardPlacement> {
    let base_url = coordinator_http_url(args)?;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(args.request_timeout_secs.get()))
        .build()?;
    let cluster: RestEnvelope<ClusterState> =
        rest_get(&client, args, &format!("{base_url}/cluster"))
            .await
            .context("failed to discover Qdrant peer URIs")?;
    let collection_name = urlencoding::encode(&args.collection);
    let collection: RestEnvelope<CollectionClusterState> = rest_get(
        &client,
        args,
        &format!("{base_url}/collections/{collection_name}/cluster"),
    )
    .await
    .context("failed to discover collection shard placement")?;

    ensure!(
        collection.result.peer_id == cluster.result.peer_id,
        "cluster endpoints disagree about the current peer ID"
    );
    ensure!(
        collection.result.shard_count == u64::from(input.manifest.shard_count),
        "collection cluster reports {} logical shards; manifest requires {}",
        collection.result.shard_count,
        input.manifest.shard_count
    );
    ensure!(
        collection.result.shard_transfers.is_empty(),
        "collection has active shard transfers; refusing offline import"
    );
    ensure!(
        collection.result.resharding_operations.is_empty(),
        "collection has active resharding operations; refusing offline import"
    );

    let peer_uris = cluster
        .result
        .peers
        .into_iter()
        .map(|(peer_id, peer)| {
            let peer_id = peer_id
                .parse::<u64>()
                .with_context(|| format!("cluster returned invalid peer ID {peer_id:?}"))?;
            ensure!(
                !peer.uri.is_empty(),
                "cluster peer {peer_id} has an empty URI"
            );
            Ok((peer_id, peer.uri.trim_end_matches('/').to_string()))
        })
        .collect::<Result<HashMap<_, _>>>()?;
    let mut owner_peer_by_shard = vec![None; usize::try_from(input.manifest.shard_count).unwrap()];
    for local in collection.result.local_shards {
        ensure_active_replica(local.shard_id, cluster.result.peer_id, &local.state)?;
        let slot = owner_peer_by_shard
            .get_mut(usize::try_from(local.shard_id).unwrap())
            .with_context(|| format!("collection reports out-of-range shard {}", local.shard_id))?;
        slot.get_or_insert(cluster.result.peer_id);
    }
    for remote in collection.result.remote_shards {
        ensure_active_replica(remote.shard_id, remote.peer_id, &remote.state)?;
        let slot = owner_peer_by_shard
            .get_mut(usize::try_from(remote.shard_id).unwrap())
            .with_context(|| {
                format!("collection reports out-of-range shard {}", remote.shard_id)
            })?;
        slot.get_or_insert(remote.peer_id);
    }

    let owner_uri_by_shard = owner_peer_by_shard
        .into_iter()
        .enumerate()
        .map(|(shard_id, peer_id)| {
            let peer_id = peer_id
                .with_context(|| format!("numeric shard {shard_id} has no Active replica"))?;
            peer_uris
                .get(&peer_id)
                .cloned()
                .with_context(|| format!("cluster did not advertise a URI for peer {peer_id}"))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(ShardPlacement { owner_uri_by_shard })
}

fn ensure_active_replica(shard_id: u32, peer_id: u64, state: &str) -> Result<()> {
    ensure!(
        state.eq_ignore_ascii_case("active"),
        "numeric shard {shard_id} replica on peer {peer_id} is {state:?}, not Active"
    );
    Ok(())
}

fn coordinator_http_url(args: &Args) -> Result<String> {
    if let Some(http_url) = &args.http_url {
        return Ok(http_url.trim_end_matches('/').to_string());
    }
    let mut url = url::Url::parse(&args.uri)
        .with_context(|| format!("--uri {:?} is not a valid URL", args.uri))?;
    let internal_port = url
        .port()
        .context("--uri must include the internal gRPC port, normally 6335")?;
    let http_port = internal_port
        .checked_sub(2)
        .context("cannot derive HTTP port from --uri; pass --http-url explicitly")?;
    url.set_port(Some(http_port))
        .map_err(|_| anyhow::anyhow!("failed to set derived HTTP port"))?;
    url.set_path("");
    url.set_query(None);
    url.set_fragment(None);
    Ok(url.as_str().trim_end_matches('/').to_string())
}

async fn rest_get<T: DeserializeOwned>(
    client: &reqwest::Client,
    args: &Args,
    url: &str,
) -> Result<T> {
    let mut request = client.get(url);
    if let Some(api_key) = &args.api_key {
        request = request.header("api-key", api_key);
    }
    let response = request.send().await?;
    let status = response.status();
    let body = response.text().await?;
    ensure!(
        status.is_success(),
        "GET {url} returned HTTP {status}: {body}"
    );
    serde_json::from_str(&body).with_context(|| format!("GET {url} returned invalid JSON"))
}

async fn preflight_collection(
    placement: &ShardPlacement,
    args: &Args,
    input: &ValidatedInput,
) -> Result<()> {
    let shard_zero_uri = placement
        .owner_uri_by_shard
        .first()
        .context("collection has no shard zero owner")?;
    let mut collections_client = connect_collections_internal(shard_zero_uri, args).await?;
    let request = authenticated_request(
        args,
        GetCollectionInfoRequestInternal {
            get_collection_info_request: Some(GetCollectionInfoRequest {
                collection_name: args.collection.clone(),
            }),
            shard_id: 0,
        },
    )?;
    let info = collections_client
        .get(request)
        .await
        .with_context(|| {
            format!(
                "failed to inspect collection {:?} on shard zero owner {shard_zero_uri}",
                args.collection,
            )
        })?
        .into_inner()
        .result
        .context("collection info response did not contain a result")?;
    validate_collection_info(&info, input)?;

    let counts = exact_counts_by_owner(placement, args).await?;
    for (shard_id, (count, &expected)) in
        counts.into_iter().zip(&input.copies_per_shard).enumerate()
    {
        if args.resume {
            ensure!(
                count <= expected,
                "numeric shard {shard_id} contains {count} points while the checkpointed manifest requires only {expected}; refusing resume because stale copies may exist"
            );
        } else {
            ensure!(
                count == 0,
                "numeric shard {shard_id} already contains {count} points; start with a fresh collection or use --resume with the exact matching local checkpoint from an interrupted import"
            );
        }
    }
    println!(
        "collection preflight passed: auto-sharded {} generation={} artifact={} across {} numeric shards (resume={})",
        input.manifest.routing.policy.display_name(),
        input.manifest.routing.generation,
        input.manifest.routing.artifact_sha256,
        input.manifest.shard_count,
        args.resume,
    );
    Ok(())
}

fn validate_collection_info(info: &CollectionInfo, input: &ValidatedInput) -> Result<()> {
    let config = info
        .config
        .as_ref()
        .context("collection info is missing its config")?;
    let params = config
        .params
        .as_ref()
        .context("collection config is missing params")?;
    ensure!(
        params.shard_number == input.manifest.shard_count,
        "collection has {} numeric shards; import manifest requires {}",
        params.shard_number,
        input.manifest.shard_count
    );
    let sharding_method = params
        .sharding_method
        .map(ShardingMethod::try_from)
        .transpose()
        .context("collection has an unknown sharding_method value")?
        .unwrap_or(ShardingMethod::Auto);
    ensure!(
        sharding_method == ShardingMethod::Auto,
        "static numeric import requires sharding_method=auto, got {}",
        sharding_method.as_str_name()
    );

    let policy = config
        .auto_shard_policy
        .as_ref()
        .and_then(|policy| policy.policy.as_ref())
        .context("collection is not configured with an explicit static auto-shard policy")?;
    validate_collection_policy(policy, &input.manifest.routing)?;

    let vectors_config = params
        .vectors_config
        .as_ref()
        .and_then(|vectors| vectors.config.as_ref())
        .context("collection config is missing dense vector parameters")?;
    let vector_params = match (vectors_config, input.manifest.vector_name.as_str()) {
        (VectorsConfigVariant::Params(params), "") => params,
        (VectorsConfigVariant::ParamsMap(map), name) if !name.is_empty() => {
            ensure!(
                map.map.len() == 1,
                "static numeric import requires a single dense vector schema, but collection has {} named vectors",
                map.map.len()
            );
            map.map
                .get(name)
                .with_context(|| format!("collection has no named vector {name:?}"))?
        }
        (VectorsConfigVariant::Params(_), name) => {
            bail!("collection uses the default vector, but manifest requires named vector {name:?}")
        }
        (VectorsConfigVariant::ParamsMap(_), "") => {
            bail!("collection uses named vectors, but manifest requires the default vector")
        }
        (VectorsConfigVariant::ParamsMap(_), _) => unreachable!(),
    };
    let manifest_dimension = u64::try_from(input.manifest.dimension)
        .context("manifest dimension does not fit in u64")?;
    ensure!(
        vector_params.size == manifest_dimension,
        "collection vector dimension is {}; import manifest requires {}",
        vector_params.size,
        input.manifest.dimension
    );
    let collection_distance = Distance::try_from(vector_params.distance)
        .context("collection has an unknown vector distance")?;
    ensure!(
        collection_distance == input.vector_distance,
        "collection vector distance is {}; routing artifact requires {}",
        collection_distance.as_str_name(),
        input.vector_distance.as_str_name()
    );
    let datatype = vector_params
        .datatype
        .map(Datatype::try_from)
        .transpose()
        .context("collection has an unknown vector datatype")?
        .unwrap_or(Datatype::Float32);
    ensure!(
        matches!(datatype, Datatype::Default | Datatype::Float32),
        "numeric importer writes f32 vectors, but collection datatype is {}",
        datatype.as_str_name()
    );
    ensure!(
        vector_params.multivector_config.is_none(),
        "numeric importer only supports single dense vectors"
    );
    Ok(())
}

fn validate_collection_policy(
    policy: &AutoShardPolicyVariant,
    expected: &StaticRoutingBinding,
) -> Result<()> {
    let (actual_policy, generation, artifact_sha256) = match policy {
        AutoShardPolicyVariant::Orion(orion) => (
            StaticRoutingPolicy::Orion,
            orion.generation,
            orion.artifact_sha256.as_str(),
        ),
        AutoShardPolicyVariant::SimpleKmeans(simple_kmeans) => (
            StaticRoutingPolicy::SimpleKmeans,
            simple_kmeans.generation,
            simple_kmeans.artifact_sha256.as_str(),
        ),
        AutoShardPolicyVariant::HashAll(_) => {
            bail!(
                "collection auto-shard policy is HashAll, not {}",
                expected.policy.display_name()
            )
        }
    };
    ensure!(
        actual_policy == expected.policy,
        "collection auto-shard policy is {}, but import manifest requires {}",
        actual_policy.display_name(),
        expected.policy.display_name()
    );
    ensure!(
        generation == expected.generation,
        "collection {} generation is {}; import manifest requires {}",
        actual_policy.display_name(),
        generation,
        expected.generation
    );
    ensure!(
        artifact_sha256.eq_ignore_ascii_case(&expected.artifact_sha256),
        "collection {} artifact digest is {}; import manifest requires {}",
        actual_policy.display_name(),
        artifact_sha256,
        expected.artifact_sha256
    );
    Ok(())
}

async fn exact_shard_count(
    client: &mut PointsInternalClient<Channel>,
    args: &Args,
    shard_id: u32,
) -> Result<u64> {
    let request = authenticated_request(
        args,
        CountPointsInternal {
            count_points: Some(CountPoints {
                collection_name: args.collection.clone(),
                filter: None,
                exact: Some(true),
                read_consistency: None,
                shard_key_selector: None,
                timeout: Some(args.request_timeout_secs.get()),
            }),
            shard_id: Some(shard_id),
        },
    )?;
    let response = client
        .count(request)
        .await
        .with_context(|| format!("failed to count numeric shard {shard_id}"))?
        .into_inner();
    Ok(response
        .result
        .with_context(|| format!("numeric shard {shard_id} count response has no result"))?
        .count)
}

async fn connect_points_internal(uri: &str, args: &Args) -> Result<PointsInternalClient<Channel>> {
    let timeout = Duration::from_secs(args.request_timeout_secs.get());
    let channel = Endpoint::from_shared(uri.to_string())?
        .connect_timeout(timeout)
        .timeout(timeout)
        .connect()
        .await
        .with_context(|| format!("failed to connect to shard-owner internal gRPC URI {uri}"))?;
    Ok(PointsInternalClient::new(channel)
        .send_compressed(CompressionEncoding::Gzip)
        .accept_compressed(CompressionEncoding::Gzip))
}

async fn connect_collections_internal(
    uri: &str,
    args: &Args,
) -> Result<CollectionsInternalClient<Channel>> {
    let timeout = Duration::from_secs(args.request_timeout_secs.get());
    let channel = Endpoint::from_shared(uri.to_string())?
        .connect_timeout(timeout)
        .timeout(timeout)
        .connect()
        .await
        .with_context(|| format!("failed to connect to shard-owner internal gRPC URI {uri}"))?;
    Ok(CollectionsInternalClient::new(channel)
        .send_compressed(CompressionEncoding::Gzip)
        .accept_compressed(CompressionEncoding::Gzip))
}

async fn exact_counts_by_owner(placement: &ShardPlacement, args: &Args) -> Result<Vec<u64>> {
    let mut shards_by_uri: BTreeMap<&str, Vec<u32>> = BTreeMap::new();
    for (shard_id, uri) in placement.owner_uri_by_shard.iter().enumerate() {
        shards_by_uri
            .entry(uri.as_str())
            .or_default()
            .push(u32::try_from(shard_id).unwrap());
    }
    let mut counts = vec![0_u64; placement.owner_uri_by_shard.len()];
    for (uri, shard_ids) in shards_by_uri {
        let mut client = connect_points_internal(uri, args).await?;
        for shard_id in shard_ids {
            counts[usize::try_from(shard_id).unwrap()] =
                exact_shard_count(&mut client, args, shard_id).await?;
        }
    }
    Ok(counts)
}

async fn verify_imported_counts(
    placement: &ShardPlacement,
    args: &Args,
    input: &ValidatedInput,
) -> Result<()> {
    let counts = exact_counts_by_owner(placement, args).await?;
    let mut actual_total = 0_u64;
    for (shard_id, (&expected, actual)) in input.copies_per_shard.iter().zip(counts).enumerate() {
        ensure!(
            actual == expected,
            "numeric shard {shard_id} contains {actual} points after import; expected {expected}"
        );
        actual_total = actual_total
            .checked_add(actual)
            .context("post-import point-copy count overflowed u64")?;
    }
    ensure!(
        actual_total == input.total_copies,
        "post-import shards contain {actual_total} point copies; manifest requires {}",
        input.total_copies
    );
    println!(
        "post-import exact count verification passed for {} point copies across {} shards",
        actual_total, input.manifest.shard_count
    );
    Ok(())
}

async fn import(
    input: &ValidatedInput,
    args: &Args,
    client: &mut PointsInternalClient<Channel>,
) -> Result<ImportStats> {
    let assignments_file = File::open(&input.assignments_path)?;
    let mut assignments = BufReader::new(assignments_file);
    let vectors_file = File::open(&input.vectors_path)?;
    let mut vectors = BufReader::new(vectors_file);
    let mut row_bytes = vec![0_u8; input.vector_bytes_per_point];
    let mut assignment_line = String::new();
    let mut vectors_digest = Sha256::new();
    let mut assignments_digest = Sha256::new();
    let mut batches: BTreeMap<u32, Vec<PointStruct>> = BTreeMap::new();
    let mut stats = ImportStats::default();

    for row_index in 0..input.manifest.point_count {
        let line_number = row_index + 1;
        assignment_line.clear();
        let read = assignments.read_line(&mut assignment_line)?;
        ensure!(read > 0, "validated assignment file ended unexpectedly");
        assignments_digest.update(assignment_line.as_bytes());
        let record: AssignmentRecord = serde_json::from_str(assignment_line.trim_end())
            .with_context(|| {
                format!("assignment changed after validation at line {line_number}")
            })?;
        validate_assignment(&record, input.manifest.shard_count, line_number).with_context(
            || format!("assignment changed after validation at line {line_number}"),
        )?;
        vectors.read_exact(&mut row_bytes).with_context(|| {
            format!("vector file changed after validation while reading row {row_index}")
        })?;
        vectors_digest.update(&row_bytes);
        let vector = decode_vector(&row_bytes, line_number)?;
        queue_point_copies(&mut batches, &record, vector, &input.manifest.vector_name);
        stats.logical_points += 1;
        stats.point_copies += u64::try_from(record.shards.len()).unwrap();

        if (row_index + 1) % args.batch_size.get() == 0 {
            flush_batches(client, args, &mut batches, &mut stats).await?;
        }
    }
    flush_batches(client, args, &mut batches, &mut stats).await?;

    assignment_line.clear();
    ensure!(
        assignments.read_line(&mut assignment_line)? == 0,
        "assignment file gained records after validation"
    );
    let mut trailing_vector_byte = [0_u8; 1];
    ensure!(
        vectors.read(&mut trailing_vector_byte)? == 0,
        "vector file gained bytes after validation"
    );
    let imported_vectors_sha256 = format!("{:x}", vectors_digest.finalize());
    let imported_assignments_sha256 = format!("{:x}", assignments_digest.finalize());
    ensure!(
        imported_vectors_sha256.eq_ignore_ascii_case(&input.manifest.vectors_sha256),
        "vector file changed after validation: imported SHA-256 {imported_vectors_sha256}, expected {}",
        input.manifest.vectors_sha256
    );
    ensure!(
        imported_assignments_sha256.eq_ignore_ascii_case(&input.manifest.assignments_sha256),
        "assignment file changed after validation: imported SHA-256 {imported_assignments_sha256}, expected {}",
        input.manifest.assignments_sha256
    );

    ensure!(
        stats.point_copies == input.total_copies,
        "input changed after validation: imported {} copies, expected {}",
        stats.point_copies,
        input.total_copies
    );
    Ok(stats)
}

fn decode_vector(bytes: &[u8], line_number: usize) -> Result<Vec<f32>> {
    debug_assert_eq!(bytes.len() % size_of::<f32>(), 0);
    let mut vector = Vec::with_capacity(bytes.len() / size_of::<f32>());
    for chunk in bytes.chunks_exact(size_of::<f32>()) {
        let value = f32::from_le_bytes(chunk.try_into().unwrap());
        ensure!(
            value.is_finite(),
            "vector row {line_number} contains a non-finite value"
        );
        vector.push(value);
    }
    Ok(vector)
}

fn queue_point_copies(
    batches: &mut BTreeMap<u32, Vec<PointStruct>>,
    record: &AssignmentRecord,
    vector: Vec<f32>,
    vector_name: &str,
) {
    for &shard_id in &record.shards {
        batches.entry(shard_id).or_default().push(PointStruct {
            id: Some(record.id.to_grpc()),
            payload: HashMap::new(),
            vectors: Some(grpc_vectors(vector.clone(), vector_name)),
        });
    }
}

fn grpc_vectors(vector: Vec<f32>, vector_name: &str) -> Vectors {
    let dense = Vector {
        #[allow(deprecated)]
        data: Vec::new(),
        #[allow(deprecated)]
        indices: None,
        #[allow(deprecated)]
        vectors_count: None,
        vector: Some(VectorVariant::Dense(DenseVector { data: vector })),
    };
    let vectors_options = if vector_name.is_empty() {
        VectorsOptions::Vector(dense)
    } else {
        VectorsOptions::Vectors(NamedVectors {
            vectors: HashMap::from([(vector_name.to_string(), dense)]),
        })
    };
    Vectors {
        vectors_options: Some(vectors_options),
    }
}

async fn flush_batches(
    client: &mut PointsInternalClient<Channel>,
    args: &Args,
    batches: &mut BTreeMap<u32, Vec<PointStruct>>,
    stats: &mut ImportStats,
) -> Result<()> {
    let queued = std::mem::take(batches);
    for (shard_id, points) in queued {
        let point_count = points.len();
        let request = build_request(args, shard_id, points)?;
        let response = client
            .upsert(request)
            .await
            .with_context(|| {
                format!(
                    "PointsInternal/Upsert failed for numeric shard {shard_id} after {} successful requests",
                    stats.grpc_requests
                )
            })?
            .into_inner();
        let result = response
            .result
            .with_context(|| format!("shard {shard_id} returned no update result"))?;
        let status =
            UpdateStatus::try_from(result.status).unwrap_or(UpdateStatus::UnknownUpdateStatus);
        if !matches!(status, UpdateStatus::Acknowledged | UpdateStatus::Completed) {
            bail!(
                "shard {shard_id} rejected batch with status {}",
                status.as_str_name()
            );
        }
        stats.grpc_requests += 1;
        println!(
            "upserted shard={} points={} status={} requests={}",
            shard_id,
            point_count,
            status.as_str_name(),
            stats.grpc_requests,
        );
    }
    Ok(())
}

fn build_request(
    args: &Args,
    shard_id: u32,
    points: Vec<PointStruct>,
) -> Result<Request<UpsertPointsInternal>> {
    let timeout_secs = args.request_timeout_secs.get();
    let wait = args.wait.grpc();
    let request = UpsertPointsInternal {
        upsert_points: Some(UpsertPoints {
            collection_name: args.collection.clone(),
            wait: Some(args.wait.public_wait()),
            points,
            ordering: Some(WriteOrdering {
                r#type: args.ordering.grpc() as i32,
            }),
            shard_key_selector: None,
            update_filter: None,
            timeout: Some(timeout_secs),
            update_mode: Some(UpdateMode::Upsert as i32),
        }),
        shard_id: Some(shard_id),
        clock_tag: None,
        wait_override: Some(wait as i32),
    };
    authenticated_request(args, request)
}

fn authenticated_request<T>(args: &Args, message: T) -> Result<Request<T>> {
    let mut request = Request::new(message);
    request.set_timeout(Duration::from_secs(args.request_timeout_secs.get()));
    if let Some(api_key) = &args.api_key {
        let value: MetadataValue<_> = api_key
            .parse()
            .context("Qdrant API key is not valid gRPC metadata")?;
        request.metadata_mut().insert("api-key", value);
    }
    Ok(request)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(manifest: PathBuf) -> Args {
        Args {
            manifest,
            uri: "http://127.0.0.1:6335".to_string(),
            http_url: None,
            collection: "orion-test".to_string(),
            batch_size: NonZeroUsize::new(2).unwrap(),
            request_timeout_secs: NonZeroU64::new(30).unwrap(),
            wait: ImportWait::Visible,
            ordering: ImportOrdering::Medium,
            api_key: None,
            resume: false,
            dry_run: false,
        }
    }

    fn sha256(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn simple_kmeans_fixture() -> (tempfile::TempDir, PathBuf) {
        let directory = tempfile::tempdir().unwrap();
        let vectors = [1.0_f32, 0.0, 0.0, 1.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let assignments = b"{\"id\":0,\"shards\":[0]}\n{\"id\":1,\"shards\":[1]}\n";
        fs_err::write(directory.path().join("vectors.f32le"), &vectors).unwrap();
        fs_err::write(directory.path().join("assignments.jsonl"), assignments).unwrap();
        let artifact = serde_json::to_vec(&serde_json::json!({
            "format_version": 1,
            "generation": 8,
            "vector_schema": {
                "vector_name": "",
                "dimension": 2,
                "distance": "Cosine",
                "datatype": "float32"
            },
            "shard_count": 2,
            "layout_sha256": sha256(assignments),
            "logical_point_count": 2,
            "physical_point_count": 2,
            "routing_distance": "squared_l2",
            "nprobe": 1,
            "lower_hnsw_ef": 48,
            "centroids": [
                {"shard_id": 0, "vector": [1.0, 0.0]},
                {"shard_id": 1, "vector": [0.0, 1.0]}
            ]
        }))
        .unwrap();
        fs_err::write(directory.path().join("generation-8.json"), &artifact).unwrap();
        let manifest_path = directory.path().join("manifest.json");
        fs_err::write(
            &manifest_path,
            serde_json::to_vec(&serde_json::json!({
                "format_version": 2,
                "routing_policy": "simple_kmeans",
                "routing_generation": 8,
                "routing_artifact_file": "generation-8.json",
                "routing_artifact_sha256": sha256(&artifact),
                "dimension": 2,
                "point_count": 2,
                "shard_count": 2,
                "total_point_copies": 2,
                "vector_name": "",
                "vectors_file": "vectors.f32le",
                "vectors_sha256": sha256(&vectors),
                "assignments_file": "assignments.jsonl",
                "assignments_sha256": sha256(assignments),
            }))
            .unwrap(),
        )
        .unwrap();
        (directory, manifest_path)
    }

    #[test]
    fn one_external_id_is_queued_for_each_numeric_shard() {
        let record = AssignmentRecord {
            id: ManifestPointId::Number(42),
            shards: vec![1, 3],
        };
        let mut batches = BTreeMap::new();
        queue_point_copies(&mut batches, &record, vec![0.25, 0.75], "");

        assert_eq!(batches.keys().copied().collect::<Vec<_>>(), vec![1, 3]);
        let first = batches[&1][0].id.as_ref().unwrap();
        let second = batches[&3][0].id.as_ref().unwrap();
        assert_eq!(first, second);
        assert_eq!(
            first.point_id_options,
            Some(PointIdOptions::Num(42)),
            "copies must retain the external point ID"
        );

        let request = build_request(&args(PathBuf::new()), 3, batches.remove(&3).unwrap())
            .unwrap()
            .into_inner();
        assert_eq!(request.shard_id, Some(3));
        let upsert = request.upsert_points.unwrap();
        assert!(upsert.shard_key_selector.is_none());
        assert_eq!(
            upsert.ordering.unwrap().r#type,
            WriteOrderingType::Medium as i32
        );
    }

    #[test]
    fn input_validation_rejects_duplicate_or_out_of_range_shards() {
        let duplicate = AssignmentRecord {
            id: ManifestPointId::Number(1),
            shards: vec![0, 0],
        };
        assert!(validate_assignment(&duplicate, 2, 1).is_err());

        let out_of_range = AssignmentRecord {
            id: ManifestPointId::Number(1),
            shards: vec![2],
        };
        assert!(validate_assignment(&out_of_range, 2, 1).is_err());
    }

    #[test]
    fn little_endian_vector_decode_is_exact() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1.5_f32.to_le_bytes());
        bytes.extend_from_slice(&(-2.25_f32).to_le_bytes());
        assert_eq!(decode_vector(&bytes, 1).unwrap(), vec![1.5, -2.25]);
    }

    #[test]
    fn manifest_checksums_are_required_and_verified_before_import() {
        let directory = tempfile::tempdir().unwrap();
        let vectors = [1.0_f32, 2.0, 3.0, 4.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let assignments = b"{\"id\":0,\"shards\":[0,1]}\n{\"id\":1,\"shards\":[1]}\n";
        fs_err::write(directory.path().join("vectors.f32le"), &vectors).unwrap();
        fs_err::write(directory.path().join("assignments.jsonl"), assignments).unwrap();
        let artifact = serde_json::to_vec(&serde_json::json!({
            "format_version": 1,
            "generation": 7,
            "vector_schema": {
                "vector_name": "",
                "dimension": 2,
                "distance": "Cosine",
                "datatype": "float32"
            },
            "shard_count": 2,
            "layout_sha256": sha256(assignments),
            "logical_point_count": 2,
            "physical_point_count": 3,
            "upper_graph": {"entry_point": 0, "max_level": 0, "nodes": []}
        }))
        .unwrap();
        fs_err::write(directory.path().join("generation-7.json"), &artifact).unwrap();
        let manifest_path = directory.path().join("manifest.json");
        fs_err::write(
            &manifest_path,
            serde_json::to_vec(&serde_json::json!({
                "format_version": 1,
                "dimension": 2,
                "point_count": 2,
                "shard_count": 2,
                "total_point_copies": 3,
                "orion_generation": 7,
                "orion_artifact_file": "generation-7.json",
                "orion_artifact_sha256": sha256(&artifact),
                "vector_name": "",
                "vectors_file": "vectors.f32le",
                "vectors_sha256": sha256(&vectors),
                "assignments_file": "assignments.jsonl",
                "assignments_sha256": sha256(assignments),
            }))
            .unwrap(),
        )
        .unwrap();

        let validated = validate_input(&manifest_path).unwrap();
        assert_eq!(validated.total_copies, 3);
        assert_eq!(
            validated.manifest.routing.policy,
            StaticRoutingPolicy::Orion
        );
        assert_eq!(validated.manifest.routing.generation, 7);

        let mut changed = vectors;
        changed[0] ^= 1;
        fs_err::write(directory.path().join("vectors.f32le"), changed).unwrap();
        let error = validate_input(&manifest_path).unwrap_err();
        assert!(error.to_string().contains("manifest declares"));
    }

    #[test]
    fn generic_v2_simple_kmeans_manifest_and_artifact_are_validated() {
        let (directory, manifest_path) = simple_kmeans_fixture();
        let validated = validate_input(&manifest_path).unwrap();
        assert_eq!(validated.manifest.format_version, 2);
        assert_eq!(
            validated.manifest.routing.policy,
            StaticRoutingPolicy::SimpleKmeans
        );
        assert_eq!(validated.total_copies, 2);
        assert_eq!(validated.copies_per_shard, vec![1, 1]);

        let artifact_path = directory.path().join("generation-8.json");
        let mut artifact: serde_json::Value =
            serde_json::from_slice(&fs_err::read(&artifact_path).unwrap()).unwrap();
        artifact
            .as_object_mut()
            .unwrap()
            .insert("upper_graph".to_string(), serde_json::Value::Null);
        fs_err::write(&artifact_path, serde_json::to_vec(&artifact).unwrap()).unwrap();
        let error = validate_routing_artifact(&artifact_path, &validated.manifest).unwrap_err();
        assert!(error.to_string().contains("must not contain upper_graph"));

        artifact.as_object_mut().unwrap().remove("upper_graph");
        artifact["centroids"][1]["shard_id"] = serde_json::json!(0);
        fs_err::write(&artifact_path, serde_json::to_vec(&artifact).unwrap()).unwrap();
        let error = validate_routing_artifact(&artifact_path, &validated.manifest).unwrap_err();
        assert!(error.to_string().contains("repeats centroid for shard 0"));
    }

    #[test]
    fn collection_policy_preflight_matches_manifest_policy_generation_and_digest() {
        let expected = StaticRoutingBinding {
            policy: StaticRoutingPolicy::SimpleKmeans,
            generation: 8,
            artifact_file: PathBuf::from("generation-8.json"),
            artifact_sha256: "a".repeat(64),
        };
        let simple =
            AutoShardPolicyVariant::SimpleKmeans(api::grpc::qdrant::SimpleKmeansAutoShardPolicy {
                generation: 8,
                artifact_sha256: "a".repeat(64),
            });
        validate_collection_policy(&simple, &expected).unwrap();

        let orion = AutoShardPolicyVariant::Orion(api::grpc::qdrant::OrionAutoShardPolicy {
            generation: 8,
            artifact_sha256: "a".repeat(64),
        });
        let error = validate_collection_policy(&orion, &expected).unwrap_err();
        assert!(error.to_string().contains("requires Simple KMeans"));
    }

    #[test]
    fn checkpoint_rejects_cross_policy_resume() {
        let (_directory, manifest_path) = simple_kmeans_fixture();
        let input = validate_input(&manifest_path).unwrap();
        let mut args = args(manifest_path.clone());
        args.resume = true;
        let expected = expected_checkpoint(&args, &input, "in_progress").unwrap();
        let wrong_policy = GenericImportCheckpoint {
            format_version: CHECKPOINT_FORMAT_VERSION,
            collection: expected.collection,
            manifest_sha256: expected.manifest_sha256,
            vectors_sha256: expected.vectors_sha256,
            assignments_sha256: expected.assignments_sha256,
            routing_policy: StaticRoutingPolicy::Orion,
            routing_generation: expected.routing_generation,
            routing_artifact_sha256: expected.routing_artifact_sha256,
            status: expected.status,
        };
        fs_err::write(
            checkpoint_path(&manifest_path),
            serde_json::to_vec_pretty(&wrong_policy).unwrap(),
        )
        .unwrap();
        let error = validate_checkpoint_mode(&args, &input).unwrap_err();
        assert!(error.to_string().contains("routing policy"));
    }

    #[test]
    fn legacy_v1_orion_checkpoint_is_normalized() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("checkpoint.json");
        fs_err::write(
            &path,
            serde_json::to_vec(&serde_json::json!({
                "format_version": 1,
                "collection": "legacy-orion",
                "manifest_sha256": "a".repeat(64),
                "vectors_sha256": "b".repeat(64),
                "assignments_sha256": "c".repeat(64),
                "orion_generation": 7,
                "orion_artifact_sha256": "d".repeat(64),
                "status": "in_progress"
            }))
            .unwrap(),
        )
        .unwrap();

        let checkpoint = read_checkpoint(&path).unwrap();
        assert_eq!(checkpoint.routing_policy, StaticRoutingPolicy::Orion);
        assert_eq!(checkpoint.routing_generation, 7);
        assert_eq!(checkpoint.routing_artifact_sha256, "d".repeat(64));
    }
}
