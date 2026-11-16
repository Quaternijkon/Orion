//! Construction-time proxy queries on the frozen production upper graph.
//! No evaluation queries, ground truth, local indexes or CPU costs are inputs.
use std::collections::{BTreeSet, HashSet};
use std::fs::{self, File};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use segment::types::ExtendedPointId;
use serde::{Deserialize, Serialize};

use super::{
    OrionRouter, OrionRoutingArtifact, bundle, layout::Result, partition, whitebox_pruning,
};

pub const PROXY_COUNT: usize = 65_536;
pub const PROXY_SEED: u64 = 0x4f52_494f_4e41_4343;

#[derive(Debug, Serialize, Deserialize)]
pub struct ProbeManifest {
    pub format_version: u32,
    pub source_point_count: usize,
    pub source_vectors_sha256: String,
    pub upper_graph_sha256: String,
    pub upper_identity_sha256: String,
    pub query_count: usize,
    pub top_k: usize,
    pub search_ef: usize,
    pub seed: u64,
    pub query_ids_file: String,
    pub query_ids_sha256: String,
    pub hit_ids_file: String,
    pub hit_ids_sha256: String,
    pub distances_file: String,
    pub distances_sha256: String,
    pub whitebox: whitebox_pruning::Config,
    pub excludes_upper_sample: bool,
    pub uses_external_queries: bool,
    pub uses_ground_truth: bool,
    pub equal_cost_per_shard_request: bool,
}

pub(crate) fn sample_rows(rows: usize, upper: &[usize]) -> Result<Vec<usize>> {
    let excluded: HashSet<_> = upper.iter().copied().collect();
    if excluded.len() != upper.len() || upper.iter().any(|&i| i >= rows) || rows <= upper.len() {
        return Err(
            "proxy sampling requires distinct valid upper IDs and non-upper input rows".into(),
        );
    }
    let count = PROXY_COUNT.min(rows - upper.len());
    let mut rng = StdRng::seed_from_u64(PROXY_SEED);
    let mut selected = BTreeSet::new();
    while selected.len() < count {
        let row = rng.random_range(0..rows);
        if !excluded.contains(&row) {
            selected.insert(row);
        }
    }
    Ok(selected.into_iter().collect())
}

/// Output rows are fixed-width: top_k u64 IDs and top_k f32 distances.
/// Queries are sampled input vectors outside the upper sample, preventing exact
/// upper self-hits. Changing owners only requires replaying these cached hits.
pub fn generate(source: &Path, output: &Path, whitebox: &Path) -> Result<ProbeManifest> {
    if output.symlink_metadata().is_ok() {
        return Err("access-probe output already exists".into());
    }
    let p = partition::read_prepared(source)?;
    let config = whitebox_pruning::Config::from_json(&fs::read(whitebox)?)?;
    let artifact = OrionRoutingArtifact::read_json(
        &source.join(&p.source_artifact_file),
        Some(&p.source_artifact_sha256),
    )?;
    config
        .for_artifact(&artifact)?
        .ok_or("missing canonical whitebox rule")?;
    let router = OrionRouter::from_artifact(&artifact)?;
    let mut scratch = router.new_route_scratch();
    let selected = sample_rows(p.point_count, &p.upper_indices)?;
    let parent = output.parent().unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let stage = tempfile::Builder::new()
        .prefix(".orion-access-")
        .tempdir_in(parent)?;
    let ids_name = "proxy-rows.u64le";
    let hits_name = "upper-hits.u64le";
    let distances_name = "upper-distances.f32le";
    let mut ids = BufWriter::new(File::create_new(stage.path().join(ids_name))?);
    let mut hits = BufWriter::new(File::create_new(stage.path().join(hits_name))?);
    let mut distances = BufWriter::new(File::create_new(stage.path().join(distances_name))?);
    let mut vectors = File::open(source.join(&p.vectors_file))?;
    let width = p
        .dimension
        .checked_mul(p.vector_encoding.element_bytes())
        .ok_or("proxy vector width overflow")?;
    let mut bytes = vec![0; width];
    for (number, &row) in selected.iter().enumerate() {
        vectors.seek(SeekFrom::Start(
            (row as u64)
                .checked_mul(width as u64)
                .ok_or("proxy offset overflow")?,
        ))?;
        vectors.read_exact(&mut bytes)?;
        let query = p.vector_encoding.decode(&bytes)?;
        let result = router.search_upper_with_scratch(&query, &mut scratch)?;
        if result.len() != p.options.upper_k {
            return Err("upper search returned an incomplete proxy hit list".into());
        }
        ids.write_all(&(row as u64).to_le_bytes())?;
        for hit in result {
            let ExtendedPointId::NumId(id) = hit.label else {
                return Err("numeric proxy labels required".into());
            };
            hits.write_all(&id.to_le_bytes())?;
            distances.write_all(&hit.distance.to_le_bytes())?;
        }
        if (number + 1) % 4096 == 0 {
            eprintln!("Orion access probes: {}/{}", number + 1, selected.len());
        }
    }
    for writer in [&mut ids, &mut hits, &mut distances] {
        writer.flush()?;
        writer.get_ref().sync_all()?;
    }
    let result = ProbeManifest {
        format_version: 1,
        source_point_count: p.point_count,
        source_vectors_sha256: p.vectors_sha256,
        upper_graph_sha256: p.upper_graph_sha256,
        upper_identity_sha256: p.upper_identity_sha256,
        query_count: selected.len(),
        top_k: p.options.upper_k,
        search_ef: p.options.upper_ef_search,
        seed: PROXY_SEED,
        query_ids_file: ids_name.into(),
        query_ids_sha256: bundle::file_sha256(&stage.path().join(ids_name))?,
        hit_ids_file: hits_name.into(),
        hit_ids_sha256: bundle::file_sha256(&stage.path().join(hits_name))?,
        distances_file: distances_name.into(),
        distances_sha256: bundle::file_sha256(&stage.path().join(distances_name))?,
        whitebox: config,
        excludes_upper_sample: true,
        uses_external_queries: false,
        uses_ground_truth: false,
        equal_cost_per_shard_request: true,
    };
    bundle::write_json(&stage.path().join("access-probes.json"), &result)?;
    File::open(stage.path())?.sync_all()?;
    super::durable::publish_directory(stage.path(), output, "access-probes.json")?;
    File::open(parent)?.sync_all()?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn probes_exclude_upper_self_hits_and_are_deterministic() {
        let ids = sample_rows(100_000, &[0, 1, 17, 50_000]).unwrap();
        assert_eq!(ids.len(), PROXY_COUNT);
        assert!(ids.windows(2).all(|v| v[0] < v[1]));
        assert!(
            ![0, 1, 17, 50_000]
                .iter()
                .any(|x| ids.binary_search(x).is_ok())
        );
        assert_eq!(ids, sample_rows(100_000, &[0, 1, 17, 50_000]).unwrap());
        assert_eq!(sample_rows(4, &[1, 3]).unwrap(), vec![0, 2]);
        assert!(sample_rows(2, &[0, 1]).is_err());
        assert!(sample_rows(4, &[1, 1]).is_err());
    }
}
