//! Rust API client for bounded, resumable generation upload and activation.
use super::{
    bundle,
    layout::Result,
    lifecycle::{ControlCommand, FileBinding, MAX_CHUNK_BYTES},
};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::time::{Duration, Instant};

pub async fn admin(
    uri: &str,
    collection: &str,
    suffix: &str,
    command: serde_json::Value,
    timeout: Duration,
    api_key: Option<&str>,
) -> Result<()> {
    let mut url = url::Url::parse(uri)?;
    url.path_segments_mut()
        .map_err(|_| "invalid deployment URI")?
        .pop_if_empty()
        .extend(["collections", collection, "orion"]);
    if !suffix.is_empty() {
        url.path_segments_mut()
            .map_err(|_| "invalid URI")?
            .push(suffix);
    }
    url.query_pairs_mut()
        .append_pair("timeout", &timeout.as_secs().max(1).to_string());
    let client = reqwest::Client::builder().timeout(timeout).build()?;
    let mut request = client.post(url).json(&command);
    if let Some(key) = api_key {
        request = request.header("api-key", key);
    }
    let response = request.send().await?;
    let status = response.status();
    let body = response.text().await?;
    if !status.is_success() {
        return Err(format!("Orion operation failed ({status}): {body}").into());
    }
    let value: serde_json::Value = serde_json::from_str(&body)?;
    if value.get("result") != Some(&serde_json::Value::Bool(true)) {
        return Err("Orion operation did not acknowledge success".into());
    }
    Ok(())
}

pub async fn install(
    dir: &Path,
    uri: &str,
    collection: &str,
    timeout: Duration,
    api_key: Option<&str>,
) -> Result<()> {
    let import = bundle::verify(dir)?;
    let generation = import.orion_generation;
    let client = reqwest::Client::builder().timeout(timeout).build()?;
    let mut url = url::Url::parse(uri)?;
    url.path_segments_mut()
        .map_err(|_| "invalid deployment URI")?
        .pop_if_empty()
        .extend(["collections", collection, "orion"]);
    url.query_pairs_mut()
        .append_pair("timeout", &timeout.as_secs().max(1).to_string());
    let send = |command: ControlCommand| {
        let client = &client;
        let url = &url;
        async move {
            let mut request = client.post(url.clone()).json(&command);
            if let Some(key) = api_key {
                request = request.header("api-key", key);
            }
            let response = request.send().await?;
            let status = response.status();
            let body = response.text().await?;
            if !status.is_success() {
                return Err(format!("Orion collection operation failed ({status}): {body}").into());
            }
            let response: serde_json::Value = serde_json::from_str(&body)?;
            if response.get("result") != Some(&serde_json::Value::Bool(true)) {
                return Err("Orion operation did not acknowledge success".into());
            }
            Result::Ok(())
        }
    };
    let mut files = BTreeMap::new();
    let build: bundle::BuildManifest =
        serde_json::from_reader(File::open(dir.join("build-manifest.json"))?)?;
    for name in [
        "build-manifest.json".to_string(),
        build.source_artifact_file,
        build.attachments_file,
        build.import_manifest_file,
        import.orion_artifact_file,
        import.vectors_file,
        import.assignments_file,
    ] {
        bundle::validate_filename(&name)?;
        let path = dir.join(&name);
        files.insert(
            name,
            FileBinding {
                size: std::fs::metadata(&path)?.len(),
                sha256: bundle::file_sha256(&path)?,
            },
        );
    }
    send(ControlCommand::Begin {
        generation,
        files: files.clone(),
    })
    .await?;
    for (name, binding) in files {
        let mut input = File::open(dir.join(&name))?;
        let mut offset = 0_u64;
        while offset < binding.size {
            let mut bytes = vec![0; ((binding.size - offset) as usize).min(MAX_CHUNK_BYTES)];
            input.read_exact(&mut bytes)?;
            let length = bytes.len() as u64;
            send(ControlCommand::Chunk {
                generation,
                file: name.clone(),
                offset,
                bytes,
            })
            .await?;
            offset += length;
        }
    }
    send(ControlCommand::Prepare { generation }).await?;
    send(ControlCommand::Import { generation }).await?;
    let start = Instant::now();
    loop {
        match send(ControlCommand::Validate { generation }).await {
            Ok(()) => break,
            Err(error) if start.elapsed() >= timeout => return Err(error),
            Err(error) => {
                eprintln!("waiting for Orion readiness: {error}");
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
        }
    }
    send(ControlCommand::Activate { generation }).await?;
    send(ControlCommand::Cleanup { generation }).await?;
    Ok(())
}
