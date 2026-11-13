use super::*;
use crate::orion::{bundle, profile};
use segment::types::Distance;
use std::fs::{self, File};
use std::io::Write;

#[test]
fn frozen_python_votes_and_layout_checksum_match_exactly() {
    use sha2::{Digest, Sha256};
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/python-v1-votes.json")).unwrap();
    let owners: Vec<u32> = serde_json::from_value(fixture["owners"].clone()).unwrap();
    let mut hash = Sha256::new();
    let mut row = 0;
    for case in fixture["cases"].as_array().unwrap() {
        let hits: Vec<usize> = serde_json::from_value(case["hits"].clone()).unwrap();
        let result = strict_targets(&hits, &owners);
        if case.get("error").is_some() {
            assert!(result.is_err());
            continue;
        }
        let expected: Vec<u32> = serde_json::from_value(case["shards"].clone()).unwrap();
        assert_eq!(result.unwrap(), expected);
        assert_eq!(u64::from(expected[0]), case["self_owner"].as_u64().unwrap());
        hash.update(
            bundle::Assignment {
                id: row,
                shards: expected,
            }
            .canonical_bytes(3)
            .unwrap(),
        );
        row += 1;
    }
    assert_eq!(
        format!("{:x}", hash.finalize()),
        fixture["layout_sha256"].as_str().unwrap()
    );
    let numpy_sample: Vec<usize> =
        serde_json::from_value(fixture["sample_n65_m4_seed100"].clone()).unwrap();
    // Acknowledge the versioned RNG difference, never relabel v2 as Python byte parity.
    assert_ne!(sample_indices(65, 4, 100).unwrap(), numpy_sample);
}

fn request(dir: &std::path::Path, distance: Distance) -> bundle::BuildRequest {
    let path = dir.join("raw.f32le");
    let mut file = File::create(&path).unwrap();
    for row in 0..64 {
        for value in [((row % 16) as f32) / 8.0 - 1.0, (row / 16) as f32 - 1.5] {
            file.write_all(&value.to_le_bytes()).unwrap();
        }
    }
    bundle::BuildRequest {
        input: VectorInput {
            format_version: 1,
            encoding: Default::default(),
            sha256: bundle::file_sha256(&path).unwrap(),
            path,
            rows: 64,
            dimension: 2,
        },
        options: LayoutOptions {
            generation: 7,
            shard_count: 2,
            distance,
            vector_name: String::new(),
            sample_denominator: 4,
            sample_seed: 100,
            upper_m: 4,
            upper_ef_construct: 32,
            upper_seed: 100,
            efs: 3,
            upper_k: 4,
            upper_ef_search: 4,
            dynamic_ef_base: 20,
            dynamic_ef_factor: 4,
            kmeans_iterations: 10,
            kmeans_seed: 1,
            topology_iterations: 50,
            bundle_prefix: "import".into(),
        },
    }
}

#[test]
fn capacity_build_starts_without_a_reference_layout_and_counts_all_copies() {
    use crate::orion::partition;
    let root = tempfile::tempdir().unwrap();
    let mut req = request(root.path(), Distance::Euclid);
    req.options.efs = 16; // All16 upper entries, evenly split across two owners.
    req.options.kmeans_iterations = 0;
    req.options.topology_iterations = 0;
    let prepared = root.path().join("prepared");
    let input = partition::prepare(&req, &prepared).unwrap();
    assert!(!prepared.join("build-manifest.json").exists());
    assert!(!prepared.join("import.assignments.jsonl").exists());
    assert!(bundle::verify_integrity(&prepared).is_err());
    let owners: Vec<_> = (0..input.upper_indices.len())
        .map(|i| (i % 2) as u32)
        .collect();
    let rejected = root.path().join("rejected");
    assert!(
        partition::derive_with_limit(&prepared, &rejected, owners.clone(), 8, Some(1.1)).is_err()
    );
    assert!(!rejected.exists());
    let output = root.path().join("final");
    let built = partition::derive(&prepared, &output, owners, 8).unwrap();
    assert_eq!(built.topology_iterations, 0);
    assert_eq!(built.shard_counts, vec![64, 64]);
    assert_eq!(built.capacity_max_replication_factor, None);
    assert_eq!(built.upper_graph_sha256, input.upper_graph_sha256);
    assert_eq!(built.attachments_sha256, input.attachments_sha256);
    assert_eq!(bundle::verify(&output).unwrap().total_point_copies, 128);
    // A configured absolute bound must also be enforced when reading a bundle.
    let mut tampered = built;
    tampered.capacity_max_replication_factor = Some(1.1);
    fs::write(
        output.join("build-manifest.json"),
        serde_json::to_vec(&tampered).unwrap(),
    )
    .unwrap();
    assert!(bundle::verify_integrity(&output).is_err());
}

#[test]
fn direct_capacity_preparation_preserves_the_canonical_graph_and_attachments() {
    use crate::orion::partition;
    let root = tempfile::tempdir().unwrap();
    let req = request(root.path(), Distance::Cosine);
    let canonical = bundle::build(&req, &root.path().join("canonical")).unwrap();
    let prepared = root.path().join("prepared");
    let direct = partition::prepare(&req, &prepared).unwrap();
    assert_eq!(direct.upper_graph_sha256, canonical.upper_graph_sha256);
    assert_eq!(direct.attachments_sha256, canonical.attachments_sha256);
    assert_eq!(direct.upper_indices, canonical.upper_indices);
    fs::write(prepared.join(&direct.attachments_file), b"corrupt").unwrap();
    assert!(partition::read_prepared(&prepared).is_err());
}

#[test]
fn uint8_bundle_is_compact_and_replays_the_same_euclidean_layout() {
    let root = tempfile::tempdir().unwrap();
    let mut float = request(root.path(), Distance::Euclid);
    let values: Vec<u8> = (0..64)
        .flat_map(|i| [((i % 16) * 16) as u8, ((i / 16) * 64) as u8])
        .collect();
    fs::write(
        &float.input.path,
        values
            .iter()
            .flat_map(|&v| f32::from(v).to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    float.input.sha256 = bundle::file_sha256(&float.input.path).unwrap();
    let mut byte = float.clone();
    byte.input.path = root.path().join("raw.u8bin");
    byte.input.encoding = VectorEncoding::U8bin;
    let mut raw = [64_u32.to_le_bytes(), 2_u32.to_le_bytes()].concat();
    raw.extend(&values);
    fs::write(&byte.input.path, &raw).unwrap();
    byte.input.sha256 = bundle::file_sha256(&byte.input.path).unwrap();
    let float_dir = root.path().join("float");
    let byte_dir = root.path().join("byte");
    bundle::build(&float, &float_dir).unwrap();
    bundle::build(&byte, &byte_dir).unwrap();
    let imported = bundle::verify(&byte_dir).unwrap();
    assert_eq!(
        bundle::verify_integrity(&byte_dir).unwrap().vectors_sha256,
        imported.vectors_sha256
    );
    assert_eq!(imported.vector_encoding, VectorEncoding::U8);
    assert_eq!(
        fs::read(byte_dir.join(&imported.vectors_file)).unwrap(),
        values
    );
    assert_eq!(
        fs::read(float_dir.join("import.assignments.jsonl")).unwrap(),
        fs::read(byte_dir.join("import.assignments.jsonl")).unwrap()
    );
    let artifact = crate::orion::OrionRoutingArtifact::read_json(
        byte_dir.join(imported.orion_artifact_file),
        None,
    )
    .unwrap();
    assert_eq!(
        artifact.vector_schema.datatype,
        crate::orion::OrionVectorDatatype::Uint8
    );
    let float_artifact =
        crate::orion::OrionRoutingArtifact::read_json(float_dir.join("generation-7.json"), None)
            .unwrap();
    assert_eq!(artifact.upper_graph, float_artifact.upper_graph);
    use sha2::{Digest, Sha256};
    let legacy_graph_hash = format!(
        "{:x}",
        Sha256::digest(
            serde_json::to_vec(
                &serde_json::to_value(artifact.upper_graph.as_ref().unwrap()).unwrap()
            )
            .unwrap()
        )
    );
    assert_eq!(bundle::graph_sha256(&artifact).unwrap(), legacy_graph_hash);

    // Checksums protect byte input and compact published output independently.
    raw[8] ^= 1;
    fs::write(&byte.input.path, &raw).unwrap();
    assert!(byte.input.visit(|_, _| Ok(())).is_err());
    fs::write(byte_dir.join(imported.vectors_file), &raw[8..]).unwrap();
    assert!(bundle::verify(&byte_dir).is_err());
    assert!(bundle::verify_integrity(&byte_dir).is_err());
}

#[test]
fn uint8_input_rejects_header_shape_truncation_and_lossy_values() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("input.u8bin");
    let mut bytes = [2_u32.to_le_bytes(), 2_u32.to_le_bytes()].concat();
    bytes.extend([0, 127, 128, 255]);
    fs::write(&path, &bytes).unwrap();
    let mut input = VectorInput {
        format_version: 1,
        encoding: VectorEncoding::U8bin,
        path: path.clone(),
        rows: 2,
        dimension: 2,
        sha256: bundle::file_sha256(&path).unwrap(),
    };
    let mut rows = Vec::new();
    input
        .visit(|_, row| {
            rows.push(row);
            Ok(())
        })
        .unwrap();
    assert_eq!(rows, vec![vec![0.0, 127.0], vec![128.0, 255.0]]);
    bytes[..4].copy_from_slice(&1_u32.to_le_bytes());
    fs::write(&path, &bytes).unwrap();
    input.sha256 = bundle::file_sha256(&path).unwrap();
    assert!(input.visit(|_, _| Ok(())).is_err());
    fs::write(&path, &bytes[..bytes.len() - 1]).unwrap();
    assert!(input.visit(|_, _| Ok(())).is_err());
    for invalid in [-1.0, 256.0, 0.5, f32::NAN, f32::INFINITY] {
        assert!(VectorEncoding::U8.encode(&[invalid]).is_err());
    }
    let legacy = serde_json::json!({"format_version":1,"path":"legacy.f32le","rows":1,"dimension":1,"sha256":"0".repeat(64)});
    let legacy: VectorInput = serde_json::from_value(legacy).unwrap();
    assert_eq!(legacy.encoding, VectorEncoding::F32le);
    assert!(
        serde_json::to_value(legacy)
            .unwrap()
            .get("encoding")
            .is_none()
    );
}

#[test]
fn strict_votes_preserve_primary_ties_all_one_and_reject_bad_hits() {
    let owners = [0, 1, 0, 1, 2];
    assert_eq!(strict_targets(&[3, 0, 2, 1, 4], &owners).unwrap(), [1, 0]);
    assert_eq!(strict_targets(&[4, 0, 1], &owners).unwrap(), [2]);
    assert_eq!(strict_targets(&[0], &owners).unwrap(), [0]);
    assert!(strict_targets(&[], &owners).is_err());
    assert!(strict_targets(&[9], &owners).is_err());
    assert!(strict_targets(&[1, 1], &owners).is_err());
}

#[test]
fn synchronous_self_vote_preserves_last_round_and_fixed_point() {
    assert_eq!(
        refine_owners(&[vec![1], vec![0]], &[0, 1], 8).unwrap(),
        (vec![0, 1], 8)
    );
    assert_eq!(
        refine_owners(&[vec![1], vec![1]], &[0, 1], 1).unwrap(),
        (vec![1, 1], 1)
    );
    assert!(refine_owners(&[vec![]], &[0], 3).is_err());
}

#[test]
fn sampling_is_ordered_unique_repeatable_and_floors_n_over_m() {
    let sample = sample_indices(65, 4, 100).unwrap();
    assert_eq!(sample.len(), 16);
    assert_eq!(sample, sample_indices(65, 4, 100).unwrap());
    assert_ne!(sample, sample_indices(65, 4, 101).unwrap());
    assert_eq!(
        sample
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len(),
        16
    );
    assert!(sample_indices(3, 4, 1).is_err());
    assert!(sample_indices(4, 0, 1).is_err());
}

#[test]
fn kmeans_retains_empty_centers_and_is_deterministic() {
    assert_eq!(
        cluster(&vec![vec![1.0, 1.0]; 4], 3, 2, 1).unwrap(),
        [0, 0, 0, 0]
    );
    assert!(cluster(&[vec![f32::NAN]], 1, 2, 1).is_err());
    assert!(cluster(&Vec::<Vec<f32>>::new(), 1, 2, 1).is_err());
}

#[test]
fn assignment_wire_bytes_remain_python_compatible() {
    let record = bundle::Assignment {
        id: 42,
        shards: vec![3, 1],
    };
    assert_eq!(
        record.canonical_bytes(4).unwrap(),
        b"{\"id\":42,\"shards\":[3,1]}\n"
    );
    assert!(
        bundle::Assignment {
            id: 0,
            shards: vec![1, 1]
        }
        .canonical_bytes(2)
        .is_err()
    );
    assert!(
        bundle::Assignment {
            id: 0,
            shards: vec![2]
        }
        .canonical_bytes(2)
        .is_err()
    );
}

#[test]
fn raw_input_fails_closed_for_shape_checksum_nonfinite_and_overflow() {
    let dir = tempfile::tempdir().unwrap();
    let mut r = request(dir.path(), Distance::Euclid);
    r.input.sha256 = "0".repeat(64);
    assert!(r.input.visit(|_, _| Ok(())).is_err());
    r.input.dimension = usize::MAX;
    assert!(r.input.visit(|_, _| Ok(())).is_err());
    r.input.dimension = 2;
    fs::write(&r.input.path, f32::NAN.to_le_bytes()).unwrap();
    r.input.rows = 1;
    r.input.dimension = 1;
    r.input.sha256 = bundle::file_sha256(&r.input.path).unwrap();
    assert!(r.input.visit(|_, _| Ok(())).is_err());
}

#[test]
fn standalone_bundle_replays_and_is_bit_exact_across_repeated_builds() {
    for distance in [Distance::Euclid, Distance::Cosine] {
        let dir = tempfile::tempdir().unwrap();
        let r = request(dir.path(), distance);
        let first = dir.path().join("first");
        let second = dir.path().join("second");
        bundle::build(&r, &first).unwrap();
        bundle::build(&r, &second).unwrap();
        let manifest = bundle::verify(&first).unwrap();
        assert_eq!(manifest.point_count, 64);
        for entry in fs::read_dir(&first).unwrap() {
            let entry = entry.unwrap();
            assert_eq!(
                fs::read(entry.path()).unwrap(),
                fs::read(second.join(entry.file_name())).unwrap()
            );
        }
        assert!(bundle::build(&r, &first).is_err());
    }
}

#[test]
fn streaming_emits_before_publication_and_preserves_the_entire_bundle() {
    for distance in [Distance::Euclid, Distance::Cosine] {
        let dir = tempfile::tempdir().unwrap();
        let request = request(dir.path(), distance);
        let reference = dir.path().join("reference");
        let streamed = dir.path().join("streamed");
        bundle::build(&request, &reference).unwrap();
        let mut ids = Vec::new();
        let mut assignments = Vec::new();
        let mut vectors = Vec::new();
        bundle::build_with_sink(&request, &streamed, &mut |row, vector| {
            assert!(
                !streamed.exists(),
                "ingestion must start before bundle publication"
            );
            ids.push(row.id);
            assignments.extend(row.canonical_bytes(request.options.shard_count)?);
            for value in vector {
                vectors.extend(value.to_le_bytes());
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(ids, (0..64).collect::<Vec<_>>());
        let import = bundle::verify(&streamed).unwrap();
        assert_eq!(
            assignments,
            fs::read(streamed.join(import.assignments_file)).unwrap()
        );
        assert_eq!(
            vectors,
            fs::read(streamed.join(import.vectors_file)).unwrap()
        );
        for entry in fs::read_dir(reference).unwrap() {
            let entry = entry.unwrap();
            assert_eq!(
                fs::read(entry.path()).unwrap(),
                fs::read(streamed.join(entry.file_name())).unwrap()
            );
        }
        let failed = dir.path().join("failed");
        let mut calls = 0;
        assert!(
            bundle::build_with_sink(&request, &failed, &mut |_, _| {
                calls += 1;
                if calls == 7 {
                    return Err("ingestion failed".into());
                }
                Ok(())
            })
            .is_err()
        );
        assert_eq!(
            calls, 7,
            "a failed consumer must stop assignment production"
        );
        assert!(!failed.exists());
    }
}

#[test]
fn replay_rejects_assignment_corruption_even_with_rebound_checksums() {
    let dir = tempfile::tempdir().unwrap();
    let r = request(dir.path(), Distance::Euclid);
    let output = dir.path().join("bundle");
    let mut build = bundle::build(&r, &output).unwrap();
    let mut import = bundle::verify(&output).unwrap();
    let path = output.join(&import.assignments_file);
    let text = fs::read_to_string(&path).unwrap();
    let mut lines: Vec<bundle::Assignment> = text
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    lines[63].shards = vec![(lines[63].shards[0] + 1) % 2];
    let bytes: Vec<u8> = lines
        .iter()
        .flat_map(|a| a.canonical_bytes(2).unwrap())
        .collect();
    fs::write(&path, bytes).unwrap();
    import.assignments_sha256 = bundle::file_sha256(&path).unwrap();
    let artifact_path = output.join(&import.orion_artifact_file);
    let mut artifact = crate::orion::OrionRoutingArtifact::read_json(&artifact_path, None).unwrap();
    artifact.layout_sha256 = import.assignments_sha256.clone();
    fs::write(&artifact_path, artifact.canonical_json_bytes().unwrap()).unwrap();
    import.orion_artifact_sha256 = bundle::file_sha256(&artifact_path).unwrap();
    fs::write(
        output.join(&build.import_manifest_file),
        serde_json::to_vec(&import).unwrap(),
    )
    .unwrap();
    build.import_manifest_sha256 =
        bundle::file_sha256(&output.join(&build.import_manifest_file)).unwrap();
    fs::write(
        output.join("build-manifest.json"),
        serde_json::to_vec(&build).unwrap(),
    )
    .unwrap();
    assert!(
        bundle::verify(&output)
            .unwrap_err()
            .to_string()
            .contains("assignment replay mismatch")
    );
}

#[test]
fn profile_keeps_graph_memberships_and_vectors_and_rejects_generation_rollback() {
    let dir = tempfile::tempdir().unwrap();
    let r = request(dir.path(), Distance::Euclid);
    let source = dir.path().join("source");
    let target = dir.path().join("profile");
    let original = bundle::build(&r, &source).unwrap();
    let profile = profile::RuntimeProfile {
        generation: 8,
        upper_k: 5,
        upper_ef_search: 16,
        dynamic_ef_base: 30,
        dynamic_ef_factor: 2,
    };
    let derived = profile::derive(&source, &target, &profile).unwrap();
    let before = bundle::verify(&source).unwrap();
    let after = bundle::verify(&target).unwrap();
    assert_eq!(after.orion_generation, 8);
    assert_eq!(derived.options.upper_k, 5);
    assert_eq!(derived.options.upper_ef_search, 16);
    assert_eq!(derived.options.dynamic_ef_base, 30);
    assert_eq!(derived.options.dynamic_ef_factor, 2);
    assert_eq!(before.assignments_sha256, after.assignments_sha256);
    assert_eq!(before.vectors_sha256, after.vectors_sha256);
    assert_eq!(original.upper_graph_sha256, derived.upper_graph_sha256);
    assert_eq!(
        original.upper_identity_sha256,
        derived.upper_identity_sha256
    );
    assert!(profile::derive(&target, &dir.path().join("rollback"), &profile).is_err());
}

#[test]
fn unsafe_bundle_names_and_decoupled_sampling_are_rejected() {
    for name in ["", "..", ".", "../x", "/tmp/x", "a/b", "a\\b"] {
        assert!(bundle::validate_filename(name).is_err());
    }
    let dir = tempfile::tempdir().unwrap();
    let mut r = request(dir.path(), Distance::Euclid);
    r.options.sample_denominator = 2;
    assert!(bundle::build(&r, &dir.path().join("bad")).is_err());
}

#[test]
fn build_votes_on_actual_hits_when_efs_exceeds_sample_size() {
    let dir = tempfile::tempdir().unwrap();
    let mut r = request(dir.path(), Distance::Euclid);
    r.options.efs = 32;
    let output = dir.path().join("partial");
    let manifest = bundle::build(&r, &output).unwrap();
    assert_eq!(manifest.upper_indices.len(), 16);
    assert_eq!(bundle::verify(&output).unwrap().point_count, 64);
}

#[test]
fn integrity_checks_counts_without_reexecuting_layout() {
    let root = tempfile::tempdir().unwrap();
    let request = request(root.path(), Distance::Euclid);
    let dir = root.path().join("integrity");
    bundle::build(&request, &dir).unwrap();
    let path = dir.join("build-manifest.json");
    let mut build: bundle::BuildManifest =
        serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    // Audit-only K-Means intermediate data is deliberately not recomputed by integrity mode.
    build.initial_owners[0] = (build.initial_owners[0] + 1) % 2;
    fs::write(&path, serde_json::to_vec(&build).unwrap()).unwrap();
    assert!(bundle::verify_integrity(&dir).is_ok());
    assert!(bundle::verify(&dir).is_err());
    // Count validation is still mandatory even when file checksums are intact.
    build.shard_counts[0] += 1;
    fs::write(&path, serde_json::to_vec(&build).unwrap()).unwrap();
    assert!(bundle::verify_integrity(&dir).is_err());
}
