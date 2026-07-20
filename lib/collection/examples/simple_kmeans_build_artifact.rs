use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::io::Write;
use std::path::{Path, PathBuf};

use collection::simple_kmeans::{SimpleKmeansRouter, SimpleKmeansRoutingArtifact};
use common::fs::atomic_save;

fn usage(program: &str) -> String {
    format!(
        "Usage: {program} <input.json> <output.json>\n\
         \n\
         Validates a typed static Simple KMeans nprobe artifact and rewrites it as Rust\n\
         canonical JSON. Writes the lowercase canonical SHA-256 to <output.json>.sha256.\n\
         Existing output or checksum files are never overwritten."
    )
}

fn checksum_path(output: &Path) -> PathBuf {
    let mut value = output.as_os_str().to_owned();
    value.push(".sha256");
    PathBuf::from(value)
}

fn build_artifact(input: &Path, output: &Path) -> Result<String, Box<dyn Error>> {
    let checksum_path = checksum_path(output);
    if output.exists() {
        return Err(format!(
            "refusing to overwrite existing artifact {}",
            output.display()
        )
        .into());
    }
    if checksum_path.exists() {
        return Err(format!(
            "refusing to overwrite existing checksum {}",
            checksum_path.display()
        )
        .into());
    }
    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs_err::create_dir_all(parent)?;
    }

    let artifact = SimpleKmeansRoutingArtifact::read_json(input, None)?;
    let canonical_json = artifact.canonical_json_bytes()?;
    let checksum = artifact.canonical_sha256()?;

    // Validate the exact typed artifact with the production router before publishing any bytes.
    SimpleKmeansRouter::new(artifact)?;
    atomic_save(output, |writer| writer.write_all(&canonical_json))?;

    // Refuse to publish a checksum unless the bytes on disk load with that checksum and can still
    // construct the production router.
    let loaded = SimpleKmeansRoutingArtifact::read_json(output, Some(&checksum))?;
    SimpleKmeansRouter::new(loaded)?;
    atomic_save(&checksum_path, |writer| {
        writer.write_all(format!("{checksum}\n").as_bytes())
    })?;
    Ok(checksum)
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args_os();
    let program = args
        .next()
        .unwrap_or_else(|| OsString::from("simple_kmeans_build_artifact"))
        .to_string_lossy()
        .into_owned();
    let Some(input) = args.next() else {
        return Err(usage(&program).into());
    };
    if input == "--help" || input == "-h" {
        println!("{}", usage(&program));
        return Ok(());
    }
    let output = args.next().ok_or_else(|| usage(&program))?;
    if let Some(extra) = args.next() {
        return Err(format!("unexpected argument {extra:?}\n{}", usage(&program)).into());
    }

    let input = PathBuf::from(input);
    let output = PathBuf::from(output);
    let checksum = build_artifact(&input, &output)?;
    println!("artifact={}", output.display());
    println!("checksum_file={}", checksum_path(&output).display());
    println!("sha256={checksum}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use collection::simple_kmeans::{
        SIMPLE_KMEANS_ROUTING_ARTIFACT_FORMAT_VERSION, SimpleKmeansCentroid,
        SimpleKmeansRoutingArtifact, SimpleKmeansRoutingDistance, SimpleKmeansVectorDatatype,
        SimpleKmeansVectorSchemaFingerprint,
    };
    use segment::types::Distance;

    use super::*;

    fn artifact() -> SimpleKmeansRoutingArtifact {
        SimpleKmeansRoutingArtifact {
            format_version: SIMPLE_KMEANS_ROUTING_ARTIFACT_FORMAT_VERSION,
            generation: 1,
            vector_schema: SimpleKmeansVectorSchemaFingerprint {
                vector_name: String::new(),
                dimension: 2,
                distance: Distance::Cosine,
                datatype: SimpleKmeansVectorDatatype::Float32,
            },
            shard_count: 2,
            layout_sha256: "c".repeat(64),
            logical_point_count: 10,
            physical_point_count: 10,
            routing_distance: SimpleKmeansRoutingDistance::SquaredL2,
            nprobe: 1,
            lower_hnsw_ef: 48,
            centroids: vec![
                SimpleKmeansCentroid {
                    shard_id: 0,
                    vector: vec![1.0, 0.0],
                },
                SimpleKmeansCentroid {
                    shard_id: 1,
                    vector: vec![0.0, 1.0],
                },
            ],
        }
    }

    #[test]
    fn writes_canonical_artifact_and_refuses_overwrite() {
        let temp = tempfile::tempdir().unwrap();
        let input = temp.path().join("input.json");
        let output = temp.path().join("generation-1.json");
        fs_err::write(&input, serde_json::to_vec_pretty(&artifact()).unwrap()).unwrap();

        let checksum = build_artifact(&input, &output).unwrap();
        assert_eq!(
            fs_err::read(&output).unwrap(),
            artifact().canonical_json_bytes().unwrap()
        );
        assert_eq!(
            fs_err::read_to_string(checksum_path(&output)).unwrap(),
            format!("{checksum}\n")
        );
        assert!(
            build_artifact(&input, &output)
                .unwrap_err()
                .to_string()
                .contains("overwrite")
        );
    }
}
