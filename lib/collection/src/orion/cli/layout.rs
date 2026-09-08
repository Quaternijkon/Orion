use crate::orion::{bundle, layout::Result};
use clap::{Parser, Subcommand};
use std::fs::File;
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "Rust authority for Orion offline layout and bundle validation")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Repartition current records through a new generation of the same graph.
    Repartition {
        #[arg(long)]
        uri: String,
        #[arg(long)]
        collection: String,
        #[arg(long)]
        generation: u64,
        #[arg(long)]
        shard_count: u32,
        #[arg(long, default_value = "7200")]
        timeout_secs: u64,
        #[arg(long, env = "QDRANT_API_KEY", hide_env_values = true)]
        api_key: Option<String>,
    },
    /// Complete an interrupted two-phase write from its durable intent.
    RecoverWrite {
        #[arg(long)]
        uri: String,
        #[arg(long)]
        collection: String,
        #[arg(long, default_value = "7200")]
        timeout_secs: u64,
        #[arg(long, env = "QDRANT_API_KEY", hide_env_values = true)]
        api_key: Option<String>,
    },
    /// Roll back an uncommitted generation, preserving the active generation.
    Rollback {
        #[arg(long)]
        uri: String,
        #[arg(long)]
        collection: String,
        #[arg(long)]
        generation: u64,
        #[arg(long, default_value = "7200")]
        timeout_secs: u64,
        #[arg(long, env = "QDRANT_API_KEY", hide_env_values = true)]
        api_key: Option<String>,
    },
    /// Upload, import and activate through the Rust collection consensus operation.
    Install {
        #[arg(long)]
        bundle: PathBuf,
        #[arg(long)]
        uri: String,
        #[arg(long)]
        collection: String,
        #[arg(long, default_value = "7200")]
        timeout_secs: u64,
        #[arg(long, env = "QDRANT_API_KEY", hide_env_values = true)]
        api_key: Option<String>,
    },
    /// Build from checksummed raw f32 input. All output must be new.
    Build {
        #[arg(long)]
        request: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Replay every owner and point assignment through the frozen production graph.
    Verify {
        #[arg(long)]
        bundle: PathBuf,
    },
    /// Derive a verified runtime profile without rebuilding the graph or assignments.
    Profile {
        #[arg(long)]
        source: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        generation: u64,
        #[arg(long)]
        upper_k: usize,
        #[arg(long)]
        upper_ef_search: usize,
        #[arg(long)]
        dynamic_ef_base: usize,
        #[arg(long)]
        dynamic_ef_factor: usize,
    },
}

pub fn run() -> Result<()> {
    match Args::parse().command {
        Command::Repartition {
            uri,
            collection,
            generation,
            shard_count,
            timeout_secs,
            api_key,
        } => {
            tokio::runtime::Runtime::new()?.block_on(crate::orion::deploy::admin(
                &uri,
                &collection,
                "repartition",
                serde_json::json!({"generation":generation,"shard_count":shard_count}),
                std::time::Duration::from_secs(timeout_secs),
                api_key.as_deref(),
            ))?;
            println!("{{\"repartitioned\":true}}");
        }
        Command::RecoverWrite {
            uri,
            collection,
            timeout_secs,
            api_key,
        } => {
            tokio::runtime::Runtime::new()?.block_on(crate::orion::deploy::admin(
                &uri,
                &collection,
                "",
                serde_json::to_value(crate::orion::lifecycle::ControlCommand::RecoverWrite)?,
                std::time::Duration::from_secs(timeout_secs),
                api_key.as_deref(),
            ))?;
            println!("{{\"recovered\":true}}");
        }
        Command::Rollback {
            uri,
            collection,
            generation,
            timeout_secs,
            api_key,
        } => {
            tokio::runtime::Runtime::new()?.block_on(crate::orion::deploy::admin(
                &uri,
                &collection,
                "",
                serde_json::to_value(crate::orion::lifecycle::ControlCommand::Rollback {
                    generation,
                })?,
                std::time::Duration::from_secs(timeout_secs),
                api_key.as_deref(),
            ))?;
            println!("{{\"rolled_back\":true}}");
        }
        Command::Install {
            bundle,
            uri,
            collection,
            timeout_secs,
            api_key,
        } => {
            tokio::runtime::Runtime::new()?.block_on(crate::orion::deploy::install(
                &bundle,
                &uri,
                &collection,
                std::time::Duration::from_secs(timeout_secs),
                api_key.as_deref(),
            ))?;
            println!("{{\"activated\":true}}");
        }
        Command::Build { request, output } => {
            let mut request: bundle::BuildRequest = serde_json::from_reader(File::open(&request)?)?;
            if request.input.path.is_relative() {
                return Err("raw input path must be absolute".into());
            }
            request.input.path = request.input.path.canonicalize()?;
            println!(
                "{}",
                serde_json::to_string(&bundle::build(&request, &output)?)?
            );
        }
        Command::Verify { bundle: path } => {
            println!("{}", serde_json::to_string(&bundle::verify(&path)?)?);
        }
        Command::Profile {
            source,
            output,
            generation,
            upper_k,
            upper_ef_search,
            dynamic_ef_base,
            dynamic_ef_factor,
        } => {
            let profile = crate::orion::profile::RuntimeProfile {
                generation,
                upper_k,
                upper_ef_search,
                dynamic_ef_base,
                dynamic_ef_factor,
            };
            println!(
                "{}",
                serde_json::to_string(&crate::orion::profile::derive(&source, &output, &profile)?)?
            );
        }
    }
    Ok(())
}
