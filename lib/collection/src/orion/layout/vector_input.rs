use std::fs::File;
use std::io::{BufReader, Read};
use std::path::PathBuf;

use segment::types::Distance;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::Result;
use crate::orion::OrionVectorDatatype;

/// Physical input encoding, independent of the float-valued API transport.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VectorEncoding {
    #[default]
    F32le,
    U8,
    /// BIGANN header: little-endian u32 row count, u32 dimension, then bytes.
    U8bin,
}

impl VectorEncoding {
    pub fn is_default(&self) -> bool {
        *self == Self::F32le
    }
    pub fn normalized(self) -> Self {
        match self {
            Self::U8bin => Self::U8,
            other => other,
        }
    }
    pub fn element_bytes(self) -> usize {
        if self == Self::F32le { 4 } else { 1 }
    }
    pub fn datatype(self) -> OrionVectorDatatype {
        if self == Self::F32le {
            OrionVectorDatatype::Float32
        } else {
            OrionVectorDatatype::Uint8
        }
    }
    pub fn extension(self) -> &'static str {
        match self {
            Self::F32le => "f32le",
            Self::U8 => "u8",
            Self::U8bin => "u8bin",
        }
    }
    pub fn decode(self, bytes: &[u8]) -> Result<Vec<f32>> {
        let vector: Vec<_> = if self == Self::F32le {
            if bytes.len() % 4 != 0 {
                return Err("unaligned f32 row".into());
            }
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect()
        } else {
            bytes.iter().map(|&b| f32::from(b)).collect()
        };
        if vector.iter().any(|x| !x.is_finite()) {
            return Err("non-finite input vector".into());
        }
        Ok(vector)
    }
    pub fn encode(self, vector: &[f32]) -> Result<Vec<u8>> {
        if vector.iter().any(|x| !x.is_finite()) {
            return Err("non-finite input vector".into());
        }
        if self == Self::F32le {
            Ok(vector.iter().flat_map(|x| x.to_le_bytes()).collect())
        } else {
            if vector
                .iter()
                .any(|&x| !(0.0..=255.0).contains(&x) || x.fract() != 0.0)
            {
                return Err("uint8 input must contain exact integers in [0,255]".into());
            }
            Ok(vector.iter().map(|&x| x as u8).collect())
        }
    }
    pub fn preprocess(self, distance: Distance, vector: Vec<f32>) -> Result<Vec<f32>> {
        if self == Self::F32le {
            preprocess(distance, vector)
        } else {
            self.encode(&vector)?;
            Ok(distance.preprocess_vector::<u8>(vector))
        }
    }
}

/// External adapters supply raw vectors; Rust validates encoding, shape and checksum.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VectorInput {
    pub format_version: u32,
    #[serde(default, skip_serializing_if = "VectorEncoding::is_default")]
    pub encoding: VectorEncoding,
    pub path: PathBuf,
    pub rows: usize,
    pub dimension: usize,
    pub sha256: String,
}

impl VectorInput {
    pub fn visit(&self, mut visit: impl FnMut(usize, Vec<f32>) -> Result<()>) -> Result<()> {
        if self.format_version != 1 || self.rows == 0 || self.dimension == 0 {
            return Err("invalid raw vector input format or shape".into());
        }
        let row_bytes = self
            .dimension
            .checked_mul(self.encoding.element_bytes())
            .ok_or("vector row size overflow")?;
        let expected = self
            .rows
            .checked_mul(row_bytes)
            .ok_or("vector input size overflow")?;
        let header_bytes = if self.encoding == VectorEncoding::U8bin {
            8
        } else {
            0
        };
        let expected = expected
            .checked_add(header_bytes)
            .ok_or("vector input size overflow")?;
        let file = File::open(&self.path)?;
        if file.metadata()?.len() != u64::try_from(expected)? {
            return Err("raw vector byte length does not match shape".into());
        }
        let mut reader = BufReader::new(file);
        let mut bytes = vec![0; row_bytes];
        let mut digest = Sha256::new();
        if header_bytes != 0 {
            let mut header = [0; 8];
            reader.read_exact(&mut header)?;
            if u32::from_le_bytes(header[..4].try_into().unwrap()) as usize != self.rows
                || u32::from_le_bytes(header[4..].try_into().unwrap()) as usize != self.dimension
            {
                return Err("u8bin header does not match declared shape".into());
            }
            digest.update(header);
        }
        for row in 0..self.rows {
            reader.read_exact(&mut bytes)?;
            digest.update(&bytes);
            let vector = self.encoding.decode(&bytes)?;
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
