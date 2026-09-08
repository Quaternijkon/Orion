//! Compatibility entry point. Implementation lives in collection::orion.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    collection::orion::cli::build_artifact::run()
}
