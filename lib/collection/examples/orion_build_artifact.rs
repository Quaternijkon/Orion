use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::io::Write;
use std::path::{Path, PathBuf};

use collection::orion::{
    ORION_UPPER_HNSW_DEFAULT_EF_CONSTRUCT, ORION_UPPER_HNSW_DEFAULT_M,
    ORION_UPPER_HNSW_DEFAULT_SEED, OrionRouter, OrionRoutingArtifact, OrionUpperGraphBuildOptions,
};
use common::fs::atomic_save;

fn usage(program: &str) -> String {
    format!(
        "Usage: {program} <graphless-input.json> <output.json> [--seed N] [--m N] [--ef N]\n\
         \n\
         Builds a deterministic Orion upper HNSW with Qdrant GraphLayersBuilder.\n\
         Defaults: --seed {ORION_UPPER_HNSW_DEFAULT_SEED} --m {ORION_UPPER_HNSW_DEFAULT_M} \
         --ef {ORION_UPPER_HNSW_DEFAULT_EF_CONSTRUCT}.\n\
         Writes canonical JSON to <output.json> and its lowercase SHA-256 to\n\
         <output.json>.sha256. The input must not already contain upper_graph."
    )
}

fn parse_value<T>(flag: &str, value: Option<OsString>) -> Result<T, Box<dyn Error>>
where
    T: std::str::FromStr,
    T::Err: Error + 'static,
{
    let value = value.ok_or_else(|| format!("{flag} requires a value"))?;
    Ok(value
        .into_string()
        .map_err(|_| format!("{flag} value is not valid UTF-8"))?
        .parse::<T>()?)
}

fn checksum_path(output: &Path) -> PathBuf {
    let mut value = output.as_os_str().to_owned();
    value.push(".sha256");
    PathBuf::from(value)
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args_os();
    let program = args
        .next()
        .unwrap_or_else(|| OsString::from("orion_build_artifact"))
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

    let mut options = OrionUpperGraphBuildOptions::default();
    while let Some(flag) = args.next() {
        match flag.to_string_lossy().as_ref() {
            "--seed" => options.seed = parse_value("--seed", args.next())?,
            "--m" => options.m = parse_value("--m", args.next())?,
            "--ef" | "--ef-construct" => {
                options.ef_construct = parse_value("--ef", args.next())?;
            }
            "--help" | "-h" => {
                println!("{}", usage(&program));
                return Ok(());
            }
            unknown => {
                return Err(format!("unknown option {unknown:?}\n{}", usage(&program)).into());
            }
        }
    }

    let input = PathBuf::from(input);
    let output = PathBuf::from(output);
    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs_err::create_dir_all(parent)?;
    }

    let graphless = OrionRoutingArtifact::read_json(&input, None)?;
    let built = graphless.build_upper_hnsw(options)?;
    let canonical_json = built.canonical_json_bytes()?;
    let checksum = built.canonical_sha256()?;

    atomic_save(&output, |writer| writer.write_all(&canonical_json))?;

    // Refuse to publish a checksum for bytes that the production router cannot load.
    let loaded = OrionRoutingArtifact::read_json(&output, Some(&checksum))?;
    OrionRouter::new(loaded)?;

    let checksum_path = checksum_path(&output);
    atomic_save(&checksum_path, |writer| {
        writer.write_all(format!("{checksum}\n").as_bytes())
    })?;

    println!("artifact={}", output.display());
    println!("checksum_file={}", checksum_path.display());
    println!("sha256={checksum}");
    println!("upper_m={}", options.m);
    println!("upper_ef_construct={}", options.ef_construct);
    println!("upper_seed={}", options.seed);
    Ok(())
}
