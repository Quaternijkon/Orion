#[tokio::main]
async fn main() -> anyhow::Result<()> {
    collection::orion::import::run().await
}
