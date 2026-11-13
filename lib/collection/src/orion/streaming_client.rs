//! Producer/consumer build-install client. The bounded row channel overlaps offline navigation
//! with acknowledged remote ingestion; failed consumers close the channel and join the builder.
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::time::Duration;

use super::bundle::{self, BuildManifest, BuildRequest};
use super::layout::Result;
use super::lifecycle::ControlCommand;
use super::streaming::{MAX_BATCH_BYTES, MAX_BATCH_ROWS, StreamRow, StreamSpec};

struct Client {
    http: reqwest::Client,
    collection_url: url::Url,
    api_key: Option<String>,
    timeout: Duration,
}

/// Offline capacity builds must finish before any collection or data is created.
pub(crate) async fn ensure_collection_for_bundle(
    dir: &Path,
    uri: &str,
    collection: &str,
    timeout: Duration,
    api_key: Option<&str>,
) -> Result<()> {
    let import = bundle::verify_integrity(dir)?;
    let build: BuildManifest =
        serde_json::from_reader(File::open(dir.join("build-manifest.json"))?)?;
    let spec = StreamSpec {
        vector_encoding: import.vector_encoding,
        options: build.options,
        point_count: import.point_count,
        dimension: import.dimension,
        raw_vectors_sha256: build.raw_vectors_sha256,
    };
    Client::new(uri, collection, timeout, api_key)?
        .ensure_collection(&spec)
        .await
}

impl Client {
    fn new(uri: &str, collection: &str, timeout: Duration, api_key: Option<&str>) -> Result<Self> {
        let mut collection_url = url::Url::parse(uri)?;
        collection_url
            .path_segments_mut()
            .map_err(|_| "invalid deployment URI")?
            .pop_if_empty()
            .extend(["collections", collection]);
        collection_url.set_query(None);
        Ok(Self {
            http: reqwest::Client::builder().timeout(timeout).build()?,
            collection_url,
            api_key: api_key.map(str::to_owned),
            timeout,
        })
    }

    fn authenticate(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.api_key {
            Some(key) => request.header("api-key", key),
            None => request,
        }
    }

    async fn ensure_collection(&self, spec: &StreamSpec) -> Result<()> {
        let response = self
            .authenticate(self.http.get(self.collection_url.clone()))
            .send()
            .await?;
        if response.status().is_success() {
            return Ok(());
        }
        if response.status() != reqwest::StatusCode::NOT_FOUND {
            return Err(format!("collection lookup failed: {}", response.status()).into());
        }
        let vector = serde_json::json!({"size": spec.dimension, "distance": spec.options.distance, "datatype": spec.vector_encoding.datatype()});
        let vectors = if spec.options.vector_name.is_empty() {
            vector
        } else {
            serde_json::json!({spec.options.vector_name.clone(): vector})
        };
        let mut url = self.collection_url.clone();
        url.query_pairs_mut()
            .append_pair("timeout", &self.timeout.as_secs().max(1).to_string());
        let response = self.authenticate(self.http.put(url)).json(&serde_json::json!({
            "vectors": vectors, "shard_number": spec.options.shard_count,
            "auto_shard_policy": {"type": "orion", "generation": spec.options.generation, "artifact_sha256": "0".repeat(64)},
            "optimizers_config": {"default_segment_number": 1, "indexing_threshold": 1},
        })).send().await?;
        if !response.status().is_success() {
            return Err(format!("collection creation failed: {}", response.text().await?).into());
        }
        Ok(())
    }

    async fn send(&self, command: &ControlCommand) -> Result<()> {
        let mut url = self.collection_url.clone();
        url.path_segments_mut()
            .map_err(|_| "invalid URI")?
            .push("orion");
        url.query_pairs_mut()
            .append_pair("timeout", &self.timeout.as_secs().max(1).to_string());
        for attempt in 0..3 {
            let result = async {
                let response = self
                    .authenticate(self.http.post(url.clone()))
                    .json(command)
                    .send()
                    .await?;
                let status = response.status();
                let body = response.text().await?;
                if !status.is_success() {
                    return Err(format!("Orion stream operation failed ({status}): {body}").into());
                }
                let value: serde_json::Value = serde_json::from_str(&body)?;
                if value.get("result") != Some(&serde_json::Value::Bool(true)) {
                    return Err("Orion stream operation was not acknowledged".into());
                }
                Result::Ok(())
            }
            .await;
            match result {
                Ok(()) => return Ok(()),
                Err(error) if attempt == 2 => return Err(error),
                Err(error) => {
                    eprintln!("retrying stream operation: {error}");
                    tokio::time::sleep(Duration::from_millis(100 * (attempt + 1))).await;
                }
            }
        }
        unreachable!()
    }
}

fn produce(
    request: &BuildRequest,
    output: &Path,
    sender: tokio::sync::mpsc::Sender<StreamRow>,
) -> Result<BuildManifest> {
    let mut sink = |assignment: &bundle::Assignment, vector: &[f32]| -> Result<()> {
        sender
            .blocking_send(StreamRow {
                assignment: assignment.clone(),
                vector: vector.to_vec(),
            })
            .map_err(|_| "stream consumer stopped")?;
        Ok(())
    };
    if !output.exists() {
        return bundle::build_with_sink(request, output, &mut sink);
    }
    // A prior attempt can have published the bundle just before its final RPC failed.
    // Replay its exact row stream, with the same deterministic batch boundaries.
    let import = bundle::verify_integrity(output)?;
    let build: BuildManifest =
        serde_json::from_reader(File::open(output.join("build-manifest.json"))?)?;
    check_existing(request, &build, &import)?;
    let mut vectors = BufReader::new(File::open(output.join(import.vectors_file))?);
    let mut bytes = vec![
        0;
        import
            .dimension
            .checked_mul(import.vector_encoding.element_bytes())
            .ok_or("dimension overflow")?
    ];
    for line in BufReader::new(File::open(output.join(import.assignments_file))?).lines() {
        let assignment = serde_json::from_str(&line?)?;
        vectors.read_exact(&mut bytes)?;
        let vector = import.vector_encoding.decode(&bytes)?;
        sink(&assignment, &vector)?;
    }
    Ok(build)
}

fn check_existing(
    request: &BuildRequest,
    build: &BuildManifest,
    import: &bundle::ImportManifest,
) -> Result<()> {
    if build.raw_vectors_sha256 != request.input.sha256
        || import.point_count != request.input.rows
        || import.dimension != request.input.dimension
        || import.vector_encoding != request.input.encoding.normalized()
        || serde_json::to_vec(&build.options)? != serde_json::to_vec(&request.options)?
    {
        return Err("existing output belongs to a different build".into());
    }
    Ok(())
}

async fn consume(
    client: &Client,
    generation: u64,
    receiver: &mut tokio::sync::mpsc::Receiver<StreamRow>,
) -> Result<()> {
    let mut rows = Vec::new();
    let mut bytes = 2; // JSON array brackets
    let mut sequence = 1;
    while let Some(row) = receiver.recv().await {
        let size = serde_json::to_vec(&row)?.len();
        if size + 2 > MAX_BATCH_BYTES {
            return Err("a vector row exceeds the stream batch byte limit".into());
        }
        if !rows.is_empty() && (rows.len() == MAX_BATCH_ROWS || bytes + size + 1 > MAX_BATCH_BYTES)
        {
            client
                .send(&ControlCommand::StreamBatch {
                    generation,
                    sequence,
                    rows: std::mem::take(&mut rows),
                })
                .await?;
            sequence += 1;
            bytes = 2;
        }
        bytes += size + usize::from(!rows.is_empty());
        rows.push(row);
    }
    if !rows.is_empty() {
        client
            .send(&ControlCommand::StreamBatch {
                generation,
                sequence,
                rows,
            })
            .await?;
    }
    Ok(())
}

pub async fn build_install(
    request: BuildRequest,
    output: &Path,
    uri: &str,
    collection: &str,
    timeout: Duration,
    api_key: Option<&str>,
) -> Result<BuildManifest> {
    request.options.validate(request.input.rows)?;
    if output.exists() {
        let import = bundle::verify_integrity(output)?;
        let build = serde_json::from_reader(File::open(output.join("build-manifest.json"))?)?;
        check_existing(&request, &build, &import)?;
    }
    let spec = StreamSpec::from(&request);
    let generation = spec.options.generation;
    let client = Client::new(uri, collection, timeout, api_key)?;
    client.ensure_collection(&spec).await?;
    client.send(&ControlCommand::BeginStream { spec }).await?;
    let (sender, mut receiver) = tokio::sync::mpsc::channel(MAX_BATCH_ROWS);
    let output_path = output.to_owned();
    let producer = tokio::task::spawn_blocking(move || produce(&request, &output_path, sender));
    let consumed = consume(&client, generation, &mut receiver).await;
    drop(receiver); // Unblock a failed producer even when the queue is full.
    let produced = producer.await?;
    consumed?;
    let manifest = produced?;
    // Prepare checks the acknowledged stream against the final replayed bundle, skips the
    // ordinary import pass, and waits for HNSW readiness before activating this generation.
    super::deploy::install_streamed(output, uri, collection, timeout, api_key).await?;
    Ok(manifest)
}
