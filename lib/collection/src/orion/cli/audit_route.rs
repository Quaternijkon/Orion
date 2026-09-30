//! Read-only diagnostic using the exact production router and existing artifacts.
use crate::orion::layout::Result;
use crate::orion::{OrionRouter, OrionRoutingArtifact};
use clap::Parser;
use serde_json::json;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use std::sync::{Arc, Barrier};
use std::time::Instant;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    artifact: PathBuf,
    #[arg(long)]
    queries: PathBuf,
    #[arg(long)]
    truth: PathBuf,
    #[arg(long)]
    assignments: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 128)]
    dimension: usize,
}
fn faults() -> u64 {
    std::fs::read_to_string("/proc/self/stat")
        .unwrap()
        .split_whitespace()
        .nth(9)
        .unwrap()
        .parse()
        .unwrap()
}
fn save(path: &std::path::Path, value: &serde_json::Value) -> Result<()> {
    let mut out = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(&mut out, value)?;
    out.flush()?;
    Ok(())
}
pub fn run() -> Result<()> {
    let args = Args::parse();
    std::fs::create_dir_all(&args.output)?;
    let mut raw = Vec::new();
    File::open(&args.queries)?.read_to_end(&mut raw)?;
    if raw.len() % (4 * args.dimension) != 0 {
        return Err("query shape mismatch".into());
    }
    let queries: Vec<Vec<f32>> = raw
        .chunks_exact(4 * args.dimension)
        .map(|row| {
            row.chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect()
        })
        .collect();
    drop(raw);
    eprintln!("audit: loading production artifact");
    let artifact = OrionRoutingArtifact::read_json(&args.artifact, None)?;
    if artifact.vector_schema.dimension != args.dimension {
        return Err("dimension mismatch".into());
    }
    let meta = json!({"upper_nodes":artifact.upper_nodes.len(),"upper_k":artifact.upper_k,"upper_ef":artifact.upper_ef_search,"generation":artifact.generation,"dimension":args.dimension,"datatype":artifact.vector_schema.datatype});
    eprintln!("audit: compiling production router");
    let router = Arc::new(OrionRouter::new(artifact)?);
    let mut scratch = router.new_route_scratch();
    let mut masks = Vec::new();
    let mut fanouts = Vec::new();
    let mut efs = Vec::new();
    let mut golden = Vec::new();
    eprintln!(
        "audit: routing {} queries with unchanged parameters",
        queries.len()
    );
    for query in &queries {
        let targets = router.route_query_with_scratch(query, &mut scratch)?;
        let mut mask = 0_u64;
        let mut sum = 0;
        for t in &targets {
            mask |= 1_u64 << t.shard_id;
            sum += t.ef;
        }
        masks.push(mask);
        fanouts.push(targets.len());
        efs.push(sum);
        golden.push(targets);
    }
    save(
        &args.output.join("routing.json"),
        &json!({"metadata":meta,"shard_masks":masks,"fanouts":fanouts,"ef_sums":efs}),
    )?;
    let mut measurements = Vec::new();
    // ABBA controls order while holding vectors, graph and search results fixed.
    for (round, reuse) in [false, true, true, false].into_iter().enumerate() {
        let barrier = Arc::new(Barrier::new(17));
        let before = faults();
        let rows = std::thread::scope(|scope| -> Result<Vec<(u128, u128, u128)>> {
            let origin = Instant::now();
            let mut handles = Vec::new();
            for worker in 0..16 {
                let barrier = barrier.clone();
                let router = router.clone();
                let queries = &queries;
                let golden = &golden;
                handles.push(scope.spawn(move || -> Result<Vec<(u128, u128, u128)>> {
                    let mut scratch = if reuse {
                        Some(router.new_route_scratch())
                    } else {
                        None
                    };
                    barrier.wait();
                    let mut out = Vec::new();
                    for i in (worker..queries.len().min(4000)).step_by(16) {
                        let begin = origin.elapsed().as_nanos();
                        let targets = if let Some(s) = &mut scratch {
                            router.route_query_with_scratch(&queries[i], s)?
                        } else {
                            {
                                let mut fresh = router.new_route_scratch();
                                router.route_query_with_scratch(&queries[i], &mut fresh)?
                            }
                        };
                        let end = origin.elapsed().as_nanos();
                        if targets != golden[i] {
                            return Err("scratch mode changed production routes".into());
                        }
                        std::hint::black_box(&targets);
                        out.push((begin, end, end - begin));
                    }
                    Ok(out)
                }));
            }
            barrier.wait();
            let mut out = Vec::new();
            for handle in handles {
                out.extend(handle.join().map_err(|_| "worker panic")??);
            }
            Ok(out)
        })?;
        let elapsed = (rows.iter().map(|r| r.1).max().unwrap()
            - rows.iter().map(|r| r.0).min().unwrap()) as f64
            / 1e9;
        let value = json!({"round":round,"reuse":reuse,"queries":rows.len(),"seconds":elapsed,"qps":rows.len() as f64/elapsed,"mean_ms":rows.iter().map(|r|r.2 as f64/1e6).sum::<f64>()/rows.len() as f64,"minor_faults":faults()-before,"identical_routes":true});
        eprintln!("audit scratch: {value}");
        measurements.push(value);
        save(
            &args.output.join("scratch-measurements.json"),
            &json!(measurements),
        )?;
    }
    drop(scratch);
    drop(golden);
    drop(router);
    let truth_bytes = std::fs::read(&args.truth)?;
    let truth: Vec<u64> = truth_bytes
        .chunks_exact(8)
        .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    if truth.len() != queries.len() * 10 {
        return Err("truth shape mismatch".into());
    }
    let mut wanted = truth.clone();
    wanted.sort_unstable();
    wanted.dedup();
    let mut memberships = std::collections::HashMap::with_capacity(wanted.len());
    let mut reader = BufReader::with_capacity(8 * 1024 * 1024, File::open(&args.assignments)?);
    let mut line = Vec::new();
    let mut row = 0_u64;
    let mut next = 0;
    eprintln!("audit: extracting memberships only for ground-truth IDs");
    while next < wanted.len() && reader.read_until(b'\n', &mut line)? != 0 {
        if row == wanted[next] {
            let assignment: crate::orion::bundle::Assignment = serde_json::from_slice(&line)?;
            if assignment.id != row {
                return Err("non-contiguous assignment IDs".into());
            }
            let mask = assignment
                .shards
                .iter()
                .fold(0_u64, |m, &s| m | (1_u64 << s));
            memberships.insert(row, mask);
            next += 1;
        }
        row += 1;
        line.clear();
        if row % 100_000_000 == 0 {
            eprintln!("audit: read {row} assignment records");
        }
    }
    if next != wanted.len() {
        return Err("ground truth IDs absent from assignments".into());
    }
    let accessible: Vec<_> = truth
        .chunks_exact(10)
        .zip(&masks)
        .map(|(ids, &route)| ids.iter().filter(|id| memberships[id] & route != 0).count())
        .collect();
    save(
        &args.output.join("routing-recall-ceiling.json"),
        &json!({"queries":queries.len(),"accessible_gt_counts":accessible,"recall_ceiling":accessible.iter().sum::<usize>() as f64/(queries.len()*10) as f64,"extracted_gt_ids":memberships.len(),"metadata":meta}),
    )?;
    eprintln!("audit: complete");
    Ok(())
}
