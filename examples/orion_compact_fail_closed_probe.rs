//! Live fail-closed probe for Orion's internal compact peer-search RPC.
//!
//! This deliberately sends an unsupported wire version to one Qdrant peer and succeeds only when
//! the peer rejects it with `INVALID_ARGUMENT`. It never sends a valid search, writes points, or
//! changes collection state.

use std::time::Duration;

use anyhow::{Context, Result, bail};
use api::grpc::qdrant::CoreSearchBatchByShardCompactInternal;
use api::grpc::qdrant::points_internal_client::PointsInternalClient;
use clap::Parser;
use tonic::Code;
use tonic::Request;
use tonic::transport::Endpoint;

#[derive(Debug, Parser)]
#[command(
    name = "orion-compact-fail-closed-probe",
    about = "Verify that a Qdrant peer rejects an unsupported Orion compact wire version"
)]
struct Args {
    /// Internal Qdrant gRPC endpoint, normally a peer's advertised P2P URI.
    #[arg(long)]
    uri: String,

    /// Existing collection name. The unsupported wire version is rejected before collection use.
    #[arg(long)]
    collection: String,

    /// Deliberately unsupported compact protocol version.
    #[arg(long, default_value_t = 3)]
    wire_version: u32,

    /// Connection and request timeout in seconds.
    #[arg(long, default_value_t = 15)]
    timeout_secs: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    if args.collection.is_empty() {
        bail!("--collection must not be empty");
    }
    if matches!(args.wire_version, 1 | 2) {
        bail!("wire versions 1 and 2 are supported; choose an unsupported version for this probe");
    }
    if args.timeout_secs == 0 {
        bail!("--timeout-secs must be positive");
    }

    let timeout = Duration::from_secs(args.timeout_secs);
    let channel = Endpoint::from_shared(args.uri.clone())
        .with_context(|| format!("invalid peer URI {}", args.uri))?
        .connect_timeout(timeout)
        .timeout(timeout)
        .connect()
        .await
        .with_context(|| format!("failed to connect to peer {}", args.uri))?;
    let mut client = PointsInternalClient::new(channel);
    let request = CoreSearchBatchByShardCompactInternal {
        collection_name: args.collection,
        query_templates: Vec::new(),
        searches: Vec::new(),
        timeout: Some(args.timeout_secs),
        wire_version: args.wire_version,
    };

    match client
        .core_search_batch_by_shard_compact(Request::new(request))
        .await
    {
        Err(status)
            if status.code() == Code::InvalidArgument
                && status
                    .message()
                    .to_ascii_lowercase()
                    .contains("wire_version") =>
        {
            println!(
                "fail-closed: peer rejected unsupported compact wire version {}: {}",
                args.wire_version,
                status.message()
            );
            Ok(())
        }
        Err(status) => bail!(
            "peer rejected the malformed request with unexpected status {}: {}",
            status.code(),
            status.message()
        ),
        Ok(_) => bail!(
            "peer accepted unsupported compact wire version {}; fail-closed contract violated",
            args.wire_version
        ),
    }
}
