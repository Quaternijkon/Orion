//! Runtime-only derivation: Python provides scalar parameters, Rust enforces the boundary.
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::path::Path;

use super::layout::Result;
use super::{OrionRoutingArtifact, bundle};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeProfile {
    pub generation: u64,
    pub upper_k: usize,
    pub upper_ef_search: usize,
    pub dynamic_ef_base: usize,
    pub dynamic_ef_factor: usize,
}

impl RuntimeProfile {
    pub fn apply(&self, source: &OrionRoutingArtifact) -> Result<OrionRoutingArtifact> {
        if self.generation <= source.generation || self.dynamic_ef_factor == 0 {
            return Err(
                "runtime generation must increase and dynamic EF factor must be positive".into(),
            );
        }
        let mut artifact = source.clone();
        artifact.generation = self.generation;
        artifact.upper_k = self.upper_k;
        artifact.upper_ef_search = self.upper_ef_search;
        artifact.dynamic_ef_base = self.dynamic_ef_base;
        artifact.dynamic_ef_factor = self.dynamic_ef_factor;
        artifact.validate()?;
        super::OrionRouter::new(artifact.clone())?;
        Ok(artifact)
    }
}

/// Source and target are standalone bundles, copied and synced without mutable hardlinks.
pub fn derive(
    source_dir: &Path,
    output: &Path,
    profile: &RuntimeProfile,
) -> Result<bundle::BuildManifest> {
    let mut import = bundle::verify(source_dir)?;
    let mut build: bundle::BuildManifest =
        serde_json::from_reader(File::open(source_dir.join("build-manifest.json"))?)?;
    let source_artifact = OrionRoutingArtifact::read_json(
        &source_dir.join(&import.orion_artifact_file),
        Some(&import.orion_artifact_sha256),
    )?;
    let artifact = profile.apply(&source_artifact)?;
    if output.symlink_metadata().is_ok() {
        return Err("output already exists".into());
    }
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let stage = tempfile::Builder::new()
        .prefix(".orion-profile-")
        .tempdir_in(parent)?;
    for name in [
        &import.vectors_file,
        &import.assignments_file,
        &build.attachments_file,
    ] {
        fs::copy(source_dir.join(name), stage.path().join(name))?;
        File::open(stage.path().join(name))?.sync_all()?;
    }
    let upper_source = OrionRoutingArtifact::read_json(
        &source_dir.join(&build.source_artifact_file),
        Some(&build.source_artifact_sha256),
    )?;
    let upper_source = profile.apply(&upper_source)?;
    build.runtime_source_artifact_sha256 = Some(import.orion_artifact_sha256.clone());
    build.source_artifact_file = format!("upper-source-generation-{}.json", profile.generation);
    build.source_artifact_sha256 = bundle::write_artifact(
        &stage.path().join(&build.source_artifact_file),
        &upper_source,
    )?;
    import.orion_generation = profile.generation;
    import.orion_artifact_file = format!("generation-{}.json", profile.generation);
    import.orion_artifact_sha256 =
        bundle::write_artifact(&stage.path().join(&import.orion_artifact_file), &artifact)?;
    bundle::write_json(&stage.path().join(&build.import_manifest_file), &import)?;
    build.import_manifest_sha256 =
        bundle::file_sha256(&stage.path().join(&build.import_manifest_file))?;
    build.options.generation = profile.generation;
    build.options.upper_k = profile.upper_k;
    build.options.upper_ef_search = profile.upper_ef_search;
    build.options.dynamic_ef_base = profile.dynamic_ef_base;
    build.options.dynamic_ef_factor = profile.dynamic_ef_factor;
    bundle::write_json(&stage.path().join("build-manifest.json"), &build)?;
    bundle::verify(stage.path())?;
    File::open(stage.path())?.sync_all()?;
    super::durable::publish_directory(stage.path(), output, "build-manifest.json")?;
    File::open(parent)?.sync_all()?;
    Ok(build)
}
