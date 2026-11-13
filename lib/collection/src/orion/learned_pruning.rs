//! Optional, data-bound learned shard selection. Runtime inputs are upper-search
//! statistics only; the model contains no query IDs, query vectors, or answers.
use super::learned_core::{FEATURES, FeatureContext, Node, Tree};
use super::{OrionRoutingArtifact, OrionShardTarget};
use segment::types::Distance;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTree {
    left: Vec<i32>,
    right: Vec<i32>,
    feature: Vec<i32>,
    threshold: Vec<f64>,
    value: Vec<f64>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawModel {
    format_version: u32,
    feature_names: Vec<String>,
    layout_sha256: String,
    upper_graph_sha256: String,
    routing_owners_sha256: String,
    dimension: usize,
    shard_count: u32,
    upper_k: usize,
    upper_ef_search: usize,
    base: usize,
    factor: usize,
    data_counts: [u64; 8],
    initial_logit: f64,
    learning_rate: f64,
    threshold: f64,
    trees: Vec<RawTree>,
    training_uses_query_truth: bool,
    runtime_uses_query_truth: bool,
    default_cutoffs: [f64; 8],
    #[serde(default)]
    experiment_modes: Option<BTreeMap<u64, Option<[f64; 8]>>>,
}

#[derive(Debug)]
pub struct Model {
    layout_sha256: String,
    upper_graph_sha256: String,
    routing_owners_sha256: String,
    upper_k: usize,
    base: usize,
    factor: usize,
    data_counts: [u64; 8],
    initial: f64,
    rate: f64,
    trees: Vec<Tree>,
    default_cutoffs: [f64; 8],
    modes: Option<BTreeMap<u64, Option<[f64; 8]>>>,
}
impl Model {
    pub fn from_json(bytes: &[u8]) -> Result<Self, String> {
        let raw: RawModel = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        if raw.format_version != 1
            || raw.dimension != 200
            || raw.shard_count != 8
            || raw.upper_ef_search != 256
            || !(1..=256).contains(&raw.upper_k)
            || raw.base == 0
            || raw.factor == 0
            || raw.data_counts.contains(&0)
            || raw.runtime_uses_query_truth
            || !raw.training_uses_query_truth
            || raw.feature_names.iter().map(String::as_str).ne(FEATURES)
            || !raw.initial_logit.is_finite()
            || !raw.learning_rate.is_finite()
            || !(0.0..=1.0).contains(&raw.learning_rate)
            || raw.learning_rate == 0.0
            || !(0.0..=1.0).contains(&raw.threshold)
            || raw.trees.is_empty()
            || raw.trees.len() > 1024
        {
            return Err("invalid learned-selector schema or parameters".into());
        }
        for hash in [
            &raw.layout_sha256,
            &raw.upper_graph_sha256,
            &raw.routing_owners_sha256,
        ] {
            if hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err("invalid selector binding hash".into());
            }
        }
        if !raw.default_cutoffs.iter().all(|x| x.is_finite()) {
            return Err("non-finite selector cutoff".into());
        }
        if let Some(modes) = &raw.experiment_modes {
            if modes.is_empty()
                || modes.iter().any(|(g, c)| {
                    *g == 0 || c.as_ref().is_some_and(|c| !c.iter().all(|x| x.is_finite()))
                })
            {
                return Err("invalid selector experiment modes".into());
            }
        }
        let mut trees = Vec::with_capacity(raw.trees.len());
        let mut bound = raw.initial_logit.abs();
        for t in raw.trees {
            let n = t.left.len();
            if [
                t.right.len(),
                t.feature.len(),
                t.threshold.len(),
                t.value.len(),
            ]
            .iter()
            .any(|&m| m != n)
            {
                return Err("selector tree arrays differ in length".into());
            }
            let mut nodes = Vec::with_capacity(n);
            for i in 0..n {
                if t.left[i] >= 0 && t.feature[i] < 0 {
                    return Err("negative branch feature".into());
                }
                nodes.push(Node {
                    left: t.left[i],
                    right: t.right[i],
                    feature: t.feature[i].max(0) as usize,
                    threshold: t.threshold[i],
                    value: t.value[i],
                });
            }
            let tree = Tree { nodes };
            tree.validate()?;
            bound +=
                raw.learning_rate * tree.nodes.iter().map(|n| n.value.abs()).fold(0.0, f64::max);
            trees.push(tree);
        }
        if !bound.is_finite() || bound > 1e6 {
            return Err("selector score range is invalid".into());
        }
        Ok(Self {
            layout_sha256: raw.layout_sha256,
            upper_graph_sha256: raw.upper_graph_sha256,
            routing_owners_sha256: raw.routing_owners_sha256,
            upper_k: raw.upper_k,
            base: raw.base,
            factor: raw.factor,
            data_counts: raw.data_counts,
            initial: raw.initial_logit,
            rate: raw.learning_rate,
            trees,
            default_cutoffs: raw.default_cutoffs,
            modes: raw.experiment_modes,
        })
    }
    pub fn score(&self, features: &[f32; 29]) -> f64 {
        let mut score = self.initial;
        for tree in &self.trees {
            score += self.rate * tree.predict(features);
        }
        score
    }
    fn validate_scope(
        &self,
        artifact: &OrionRoutingArtifact,
    ) -> Result<([f64; 8], [f64; 8]), String> {
        if artifact.vector_schema.dimension != 200
            || artifact.vector_schema.distance != Distance::Cosine
            || artifact.shard_count != 8
            || artifact.layout_sha256 != self.layout_sha256
            || artifact.upper_k != self.upper_k
            || artifact.upper_ef_search != 256
            || artifact.dynamic_ef_base != self.base
            || artifact.dynamic_ef_factor != self.factor
            || self.data_counts.iter().map(|&n| n as u128).sum::<u128>()
                != artifact.physical_point_count as u128
            || super::bundle::graph_sha256(artifact).map_err(|e| e.to_string())?
                != self.upper_graph_sha256
        {
            return Err("learned selector does not match frozen layout/search profile".into());
        }
        let mut hash = Sha256::new();
        let mut samples = [0usize; 8];
        for node in &artifact.upper_nodes {
            hash.update(node.owner_shard.to_le_bytes());
            samples[node.owner_shard as usize] += 1;
        }
        if format!("{:x}", hash.finalize()) != self.routing_owners_sha256 {
            return Err("learned selector owner binding mismatch".into());
        }
        let samples = samples.map(|n| n as f64 / artifact.upper_nodes.len() as f64);
        let total = artifact.physical_point_count as f64;
        let data = self.data_counts.map(|n| n as f64 / total);
        Ok((samples, data))
    }
}

#[derive(Debug)]
pub struct Policy {
    model: Arc<Model>,
    cutoffs: [f64; 8],
    samples: [f64; 8],
    data: [f64; 8],
}
impl Policy {
    pub fn from_model(
        model: Arc<Model>,
        artifact: &OrionRoutingArtifact,
    ) -> Result<Option<Self>, String> {
        let (samples, data) = model.validate_scope(artifact)?;
        let cutoffs = if let Some(modes) = &model.modes {
            match modes.get(&artifact.generation).ok_or_else(|| {
                format!(
                    "unregistered selector experiment generation {}",
                    artifact.generation
                )
            })? {
                Some(c) => *c,
                None => return Ok(None),
            }
        } else {
            model.default_cutoffs
        };
        Ok(Some(Self {
            model,
            cutoffs,
            samples,
            data,
        }))
    }
    pub fn apply(
        &self,
        hits: impl IntoIterator<Item = (u32, f32)>,
        mut targets: Vec<OrionShardTarget>,
    ) -> Result<Vec<OrionShardTarget>, String> {
        if targets.is_empty() {
            return Err("learned selector received no original targets".into());
        }
        let context = FeatureContext::new(hits, self.samples, self.data)?;
        let mut keep = [false; 8];
        let mut best = (f64::NEG_INFINITY, 0u32);
        for target in &targets {
            let score = self.model.score(&context.features(target.shard_id)?);
            keep[target.shard_id as usize] = score >= self.cutoffs[target.shard_id as usize];
            if score > best.0 {
                best = (score, target.shard_id);
            }
        }
        if !keep.iter().any(|&x| x) {
            keep[best.1 as usize] = true;
        }
        targets.retain(|t| keep[t.shard_id as usize]);
        Ok(targets)
    }
}

static MODEL: OnceLock<Result<Option<Arc<Model>>, String>> = OnceLock::new();
pub(super) fn configured(artifact: &OrionRoutingArtifact) -> Result<Option<Policy>, String> {
    let model = MODEL.get_or_init(|| {
        let path = match std::env::var("ORION_LEARNED_PRUNER") {
            Ok(path) => path,
            Err(std::env::VarError::NotPresent) => return Ok(None),
            Err(e) => return Err(e.to_string()),
        };
        let bytes =
            std::fs::read(&path).map_err(|e| format!("read learned selector {path}: {e}"))?;
        Model::from_json(&bytes).map(|m| Some(Arc::new(m)))
    });
    match model {
        Ok(Some(model)) => Policy::from_model(Arc::clone(model), artifact),
        Ok(None) => Ok(None),
        Err(e) => Err(e.clone()),
    }
}
