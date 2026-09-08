//! Compatibility entry point; use the official collection binary for new builds.
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    collection::orion::import::run().await
}
