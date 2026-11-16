//! Analysis-only caller of unmodified native APIs. Never used by the timed client or server.
use collection::orion::{OrionRouter, OrionRoutingArtifact};
use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 5 {
        return Err("artifact queries.f32le dimension output.jsonl".into());
    }
    let artifact = OrionRoutingArtifact::read_json(Path::new(&a[1]), None)?;
    let dim: usize = a[3].parse()?;
    assert_eq!(dim, artifact.vector_schema.dimension);
    let owners: HashMap<_, _> = artifact
        .upper_nodes
        .iter()
        .map(|n| (n.label, n.owner_shard))
        .collect();
    let router = OrionRouter::new(artifact)?;
    let input = File::open(&a[2])?;
    assert_eq!(input.metadata()?.len() as usize, 10000 * dim * 4);
    let mut input = BufReader::new(input);
    let mut output = BufWriter::new(File::create_new(&a[4])?);
    let mut bytes = vec![0; dim * 4];
    for query_index in 0..10000 {
        input.read_exact(&mut bytes)?;
        let q: Vec<_> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let upper = router.search_upper(&q)?;
        let before = router.route_upper_hits(&upper)?;
        let after = router.route_query(&q)?;
        for t in &after {
            let original = before
                .iter()
                .find(|x| x.shard_id == t.shard_id)
                .ok_or("new target introduced")?;
            assert_eq!(t, original);
        }
        let value = serde_json::json!({"query_index":query_index,"upper":upper.iter().map(|h|serde_json::json!({"id":h.label,"owner":owners[&h.label],"distance":h.distance})).collect::<Vec<_>>(),
   "before":before.iter().map(|t|serde_json::json!({"shard":t.shard_id,"entries":t.entry_points,"ef":t.ef})).collect::<Vec<_>>(),
   "after":after.iter().map(|t|serde_json::json!({"shard":t.shard_id,"entries":t.entry_points,"ef":t.ef})).collect::<Vec<_>>()});
        serde_json::to_writer(&mut output, &value)?;
        output.write_all(b"\n")?;
    }
    output.flush()?;
    Ok(())
}
