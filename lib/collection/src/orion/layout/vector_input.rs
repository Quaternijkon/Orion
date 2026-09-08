use std::fs::File;
use std::io::{BufReader, Read};
use std::path::PathBuf;

use segment::types::Distance;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::Result;

/// External adapters may supply ONLY raw, unnormalized f32 data and this envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VectorInput {
    pub format_version: u32,
    pub path: PathBuf,
    pub rows: usize,
    pub dimension: usize,
    pub sha256: String,
}

impl VectorInput {
    pub fn visit(&self, mut visit: impl FnMut(usize, Vec<f32>) -> Result<()>) -> Result<()> {
        if self.format_version != 1 || self.rows == 0 || self.dimension == 0 {
            return Err("invalid raw f32 input format or shape".into());
        }
        let row_bytes = self
            .dimension
            .checked_mul(4)
            .ok_or("vector row size overflow")?;
        let expected = self
            .rows
            .checked_mul(row_bytes)
            .ok_or("vector input size overflow")?;
        let file = File::open(&self.path)?;
        if file.metadata()?.len() != u64::try_from(expected)? {
            return Err("raw vector byte length does not match shape".into());
        }
        let mut reader = BufReader::new(file);
        let mut bytes = vec![0; row_bytes];
        let mut digest = Sha256::new();
        for row in 0..self.rows {
            reader.read_exact(&mut bytes)?;
            digest.update(&bytes);
            let vector: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            if vector.iter().any(|x| !x.is_finite()) {
                return Err(format!("non-finite raw vector at row {row}").into());
            }
            visit(row, vector)?;
        }
        if reader.read(&mut [0])? != 0 || format!("{:x}", digest.finalize()) != self.sha256 {
            return Err("raw vector input checksum mismatch or trailing data".into());
        }
        Ok(())
    }
}

pub(crate) fn preprocess(distance: Distance, vector: Vec<f32>) -> Result<Vec<f32>> {
    let processed = super::super::router::preprocess_vector(distance, vector);
    if processed.iter().any(|x| !x.is_finite()) {
        return Err("non-finite vector after segment metric preprocessing".into());
    }
    Ok(processed)
}
