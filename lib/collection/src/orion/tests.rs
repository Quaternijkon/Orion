use std::collections::{BTreeMap, HashSet};

use segment::types::{Distance, ExtendedPointId};

use super::*;

fn id(value: u64) -> ExtendedPointId {
    ExtendedPointId::NumId(value)
}

fn valid_artifact() -> OrionRoutingArtifact {
    OrionRoutingArtifact {
        format_version: ORION_ROUTING_ARTIFACT_FORMAT_VERSION,
        generation: 7,
        vector_schema: OrionVectorSchemaFingerprint {
            vector_name: "".to_string(),
            dimension: 2,
            distance: Distance::Euclid,
            datatype: OrionVectorDatatype::Float32,
        },
        shard_count: 4,
        layout_sha256: "a".repeat(64),
        logical_point_count: 4,
        physical_point_count: 6,
        upper_k: 3,
        upper_ef_search: 4,
        dynamic_ef_base: 20,
        dynamic_ef_factor: 4,
        upper_nodes: vec![
            OrionUpperNode {
                label: id(10),
                vector: vec![0.0, 0.0],
                shard_membership: vec![2, 0],
            },
            OrionUpperNode {
                label: id(20),
                vector: vec![1.0, 0.0],
                shard_membership: vec![1, 2],
            },
            OrionUpperNode {
                label: id(30),
                vector: vec![2.0, 0.0],
                shard_membership: vec![1],
            },
            OrionUpperNode {
                label: id(40),
                vector: vec![3.0, 0.0],
                shard_membership: vec![3],
            },
        ],
        upper_graph: Some(OrionUpperHnswGraph {
            entry_point: id(40),
            max_level: 1,
            nodes: vec![
                OrionUpperGraphNode {
                    label: id(10),
                    neighbors_by_level: vec![vec![id(20)]],
                },
                OrionUpperGraphNode {
                    label: id(20),
                    neighbors_by_level: vec![vec![id(10), id(30)]],
                },
                OrionUpperGraphNode {
                    label: id(30),
                    neighbors_by_level: vec![vec![id(20), id(40)], vec![id(40)]],
                },
                OrionUpperGraphNode {
                    label: id(40),
                    neighbors_by_level: vec![vec![id(30)], vec![id(30)]],
                },
            ],
        }),
    }
}

fn route_upper_labels_per_shard_dedup_reference(
    artifact: &OrionRoutingArtifact,
    ordered_labels: &[ExtendedPointId],
) -> Vec<OrionShardTarget> {
    #[derive(Default)]
    struct TargetBuilder {
        entry_points: Vec<ExtendedPointId>,
        seen: HashSet<ExtendedPointId>,
    }

    let mut targets: BTreeMap<_, TargetBuilder> = BTreeMap::new();
    for &label in ordered_labels.iter().take(artifact.upper_k) {
        let node = artifact
            .upper_nodes
            .iter()
            .find(|node| node.label == label)
            .expect("reference labels must exist in the artifact");
        for &shard_id in &node.shard_membership {
            let target = targets.entry(shard_id).or_default();
            if target.seen.insert(label) {
                target.entry_points.push(label);
            }
        }
    }

    targets
        .into_iter()
        .map(|(shard_id, target)| OrionShardTarget {
            shard_id,
            ef: artifact.dynamic_ef_base + artifact.dynamic_ef_factor * target.entry_points.len(),
            entry_points: target.entry_points,
        })
        .collect()
}

fn deterministic_component(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bucket = ((*state >> 32) % 2_001) as i32 - 1_000;
    bucket as f32 / 137.0
}

fn assert_scratch_route_matches_reference(
    router: &OrionRouter,
    scratch: &mut OrionRouteScratch,
    query: &[f32],
) {
    let reference_hits = router.search_upper(query).unwrap();
    let scratch_hits = router.search_upper_with_scratch(query, scratch).unwrap();
    assert_eq!(
        scratch_hits
            .iter()
            .map(|hit| (hit.label, hit.distance.to_bits()))
            .collect::<Vec<_>>(),
        reference_hits
            .iter()
            .map(|hit| (hit.label, hit.distance.to_bits()))
            .collect::<Vec<_>>(),
        "upper IDs/order/score bits differ for query={query:?}",
    );

    let reference_targets = router.route_upper_hits(&reference_hits).unwrap();
    let scratch_targets = router.route_query_with_scratch(query, scratch).unwrap();
    assert_eq!(
        scratch_targets, reference_targets,
        "target shard IDs, ordered entry points, or per-shard EF differ for query={query:?}",
    );
    assert_eq!(
        scratch_targets.len(),
        reference_targets.len(),
        "visited-shard count differs for query={query:?}",
    );
    assert_eq!(
        scratch_targets
            .iter()
            .map(|target| target.ef)
            .sum::<usize>(),
        reference_targets
            .iter()
            .map(|target| target.ef)
            .sum::<usize>(),
        "EF sum differs for query={query:?}",
    );
}

#[test]
fn artifact_round_trip_and_checksum_ignore_json_whitespace() {
    let artifact = valid_artifact();
    let checksum = artifact.canonical_sha256().unwrap();
    assert_eq!(checksum.len(), 64);

    let pretty_json = serde_json::to_vec_pretty(&artifact).unwrap();
    let loaded =
        OrionRoutingArtifact::from_json_slice(&pretty_json, Some(&checksum.to_uppercase()))
            .unwrap();
    assert_eq!(loaded, artifact);
    assert_eq!(loaded.canonical_sha256().unwrap(), checksum);
}

#[test]
fn artifact_rejects_checksum_mismatch_and_malformed_checksum() {
    let json = serde_json::to_vec(&valid_artifact()).unwrap();
    let mismatch = "0".repeat(64);
    assert!(matches!(
        OrionRoutingArtifact::from_json_slice(&json, Some(&mismatch)),
        Err(OrionRoutingError::ChecksumMismatch { .. })
    ));
    assert!(matches!(
        OrionRoutingArtifact::from_json_slice(&json, Some("abc")),
        Err(OrionRoutingError::InvalidChecksum { .. })
    ));
}

#[test]
fn artifact_rejects_dimension_membership_and_shard_errors() {
    let mut artifact = valid_artifact();
    artifact.layout_sha256 = "not-a-digest".to_string();
    assert!(matches!(
        artifact.validate(),
        Err(OrionRoutingError::InvalidLayoutChecksum { .. })
    ));

    let mut artifact = valid_artifact();
    artifact.logical_point_count = 0;
    assert!(matches!(
        artifact.validate(),
        Err(OrionRoutingError::EmptyLogicalPointCount)
    ));

    let mut artifact = valid_artifact();
    artifact.physical_point_count = artifact.logical_point_count - 1;
    assert!(matches!(
        artifact.validate(),
        Err(OrionRoutingError::InvalidPhysicalPointCount { .. })
    ));

    let mut artifact = valid_artifact();
    artifact.upper_nodes[0].vector.pop();
    assert!(matches!(
        artifact.validate(),
        Err(OrionRoutingError::UpperVectorDimensionMismatch { .. })
    ));

    let mut artifact = valid_artifact();
    artifact.upper_nodes[0].shard_membership.clear();
    assert!(matches!(
        artifact.validate(),
        Err(OrionRoutingError::EmptyShardMembership { .. })
    ));

    let mut artifact = valid_artifact();
    artifact.upper_nodes[0].shard_membership = vec![4];
    assert!(matches!(
        artifact.validate(),
        Err(OrionRoutingError::ShardOutOfRange { .. })
    ));

    let mut artifact = valid_artifact();
    artifact.upper_nodes[0].shard_membership = vec![2, 2];
    assert!(matches!(
        artifact.validate(),
        Err(OrionRoutingError::DuplicateShardMembership { .. })
    ));
}

#[test]
fn artifact_rejects_incomplete_or_invalid_graph() {
    let mut artifact = valid_artifact();
    artifact.upper_graph.as_mut().unwrap().nodes.pop();
    assert!(matches!(
        artifact.validate(),
        Err(OrionRoutingError::MissingGraphNode { label }) if label == id(40)
    ));

    let mut artifact = valid_artifact();
    artifact.upper_graph.as_mut().unwrap().nodes[0].neighbors_by_level[0] = vec![id(999)];
    assert!(matches!(
        artifact.validate(),
        Err(OrionRoutingError::GraphNeighborNotFound { neighbor, .. }) if neighbor == id(999)
    ));

    let mut artifact = valid_artifact();
    artifact.upper_graph.as_mut().unwrap().max_level = 2;
    assert!(matches!(
        artifact.validate(),
        Err(OrionRoutingError::GraphMaxLevelMismatch { .. })
    ));
}

#[test]
fn production_requires_graph_and_testing_fallback_is_explicit() {
    let mut artifact = valid_artifact();
    artifact.upper_graph = None;
    assert!(matches!(
        OrionRouter::new(artifact.clone()),
        Err(OrionRoutingError::MissingSerializedUpperGraph)
    ));

    let router = OrionRouter::new_brute_force_testing(artifact).unwrap();
    let hits = router.search_upper(&[0.9, 0.0]).unwrap();
    assert_eq!(
        hits.iter().map(|hit| hit.label).collect::<Vec<_>>(),
        vec![id(20), id(10), id(30)]
    );
    let mut scratch = router.new_route_scratch();
    assert_scratch_route_matches_reference(&router, &mut scratch, &[0.9, 0.0]);
}

#[test]
fn hnsw_search_runs_greedy_upper_levels_then_level_zero_ef_search() {
    let router = OrionRouter::new(valid_artifact()).unwrap();
    let hits = router.search_upper(&[0.1, 0.0]).unwrap();
    assert_eq!(
        hits.iter().map(|hit| hit.label).collect::<Vec<_>>(),
        vec![id(10), id(20), id(30)]
    );
    assert!(
        hits.windows(2)
            .all(|pair| pair[0].distance <= pair[1].distance)
    );
}

#[test]
fn routes_all_memberships_with_sorted_shards_ordered_unique_eps_and_dynamic_ef() {
    let router = OrionRouter::new(valid_artifact()).unwrap();
    let targets = router
        .route_upper_labels([id(20), id(10), id(20), id(40)])
        .unwrap();

    // upper_k=3 fixes the routing budget; the fourth label is not adaptively considered.
    assert_eq!(
        targets,
        vec![
            OrionShardTarget {
                shard_id: 0,
                entry_points: vec![id(10)],
                ef: 24,
            },
            OrionShardTarget {
                shard_id: 1,
                entry_points: vec![id(20)],
                ef: 24,
            },
            OrionShardTarget {
                shard_id: 2,
                entry_points: vec![id(20), id(10)],
                ef: 28,
            },
        ]
    );
}

#[test]
fn optimized_route_planner_matches_per_shard_dedup_reference() {
    let artifact = valid_artifact();
    let router = OrionRouter::new_brute_force_testing(artifact.clone()).unwrap();
    let labels = [id(10), id(20), id(30), id(40)];

    // Exhaustively cover duplicates, all membership overlaps, and labels beyond upper_k. The
    // reference is the previous BTreeMap plus one HashSet per target-shard implementation.
    for sequence_len in 0usize..=5 {
        let sequence_count = labels.len().pow(sequence_len as u32);
        for mut encoded_sequence in 0..sequence_count {
            let mut sequence = Vec::with_capacity(sequence_len);
            for _ in 0..sequence_len {
                sequence.push(labels[encoded_sequence % labels.len()]);
                encoded_sequence /= labels.len();
            }

            assert_eq!(
                router.route_upper_labels(sequence.iter().copied()).unwrap(),
                route_upper_labels_per_shard_dedup_reference(&artifact, &sequence),
                "sequence={sequence:?}",
            );
        }
    }
}

#[test]
fn route_query_combines_server_side_upper_search_and_route_plan() {
    let router = OrionRouter::new(valid_artifact()).unwrap();
    let hits = router.search_upper(&[0.1, 0.0]).unwrap();
    let targets = router.route_query(&[0.1, 0.0]).unwrap();
    assert_eq!(targets, router.route_upper_hits(&hits).unwrap());
    assert_eq!(
        targets
            .iter()
            .map(|target| target.shard_id)
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert_eq!(targets[2].entry_points, vec![id(10), id(20)]);
    assert_eq!(targets[2].ef, 28);
}

#[test]
fn reusable_route_scratch_has_randomized_exact_parity_for_cosine_and_euclid() {
    for distance in [Distance::Cosine, Distance::Euclid] {
        let mut artifact = valid_artifact();
        artifact.vector_schema.distance = distance;
        if distance == Distance::Cosine {
            artifact.upper_nodes[0].vector = vec![2.0, 1.0];
            artifact.upper_nodes[1].vector = vec![-1.0, 3.0];
            artifact.upper_nodes[2].vector = vec![-4.0, -2.0];
            artifact.upper_nodes[3].vector = vec![3.0, -5.0];
        }

        let router = OrionRouter::new(artifact).unwrap();
        let mut scratch = router.new_route_scratch();

        // Include exact-distance ties, a zero cosine vector, scale changes, and alternating signs
        // before reusing the same task-local scratch for a deterministic randomized sequence.
        for query in [
            [0.5, 0.0],
            [1.5, 0.0],
            [0.0, 0.0],
            [1.0, -1.0],
            [100.0, -100.0],
        ] {
            assert_scratch_route_matches_reference(&router, &mut scratch, &query);
        }

        let mut state = 0x4f52_494f_4e5f_5254;
        for _ in 0..512 {
            let query = [
                deterministic_component(&mut state),
                deterministic_component(&mut state),
            ];
            assert_scratch_route_matches_reference(&router, &mut scratch, &query);
        }
    }
}

#[test]
fn reusable_route_scratch_preserves_glove_sized_cosine_score_bits() {
    let mut artifact = valid_artifact();
    artifact.vector_schema.dimension = 200;
    artifact.vector_schema.distance = Distance::Cosine;

    let mut state = 0x474c_4f56_4532_3030;
    for node in &mut artifact.upper_nodes {
        node.vector = (0..artifact.vector_schema.dimension)
            .map(|_| deterministic_component(&mut state))
            .collect();
    }

    let router = OrionRouter::new(artifact).unwrap();
    let mut scratch = router.new_route_scratch();
    for _ in 0..64 {
        let query = (0..200)
            .map(|_| deterministic_component(&mut state))
            .collect::<Vec<_>>();
        assert_scratch_route_matches_reference(&router, &mut scratch, &query);
    }
}

#[test]
fn reusable_route_scratch_preserves_query_error_semantics_after_success() {
    let router = OrionRouter::new(valid_artifact()).unwrap();
    let mut scratch = router.new_route_scratch();
    assert_scratch_route_matches_reference(&router, &mut scratch, &[0.25, -0.5]);

    assert!(matches!(
        router.route_query_with_scratch(&[1.0], &mut scratch),
        Err(OrionRoutingError::QueryDimensionMismatch {
            expected: 2,
            actual: 1,
        })
    ));
    assert!(matches!(
        router.route_query_with_scratch(&[1.0, f32::INFINITY], &mut scratch),
        Err(OrionRoutingError::NonFiniteQueryVector { dimension: 1 })
    ));
    assert!(matches!(
        router.search_upper(&[f32::MAX, f32::MAX]),
        Err(OrionRoutingError::NonFiniteDistance { label }) if label == id(40)
    ));
    assert!(matches!(
        router.route_query_with_scratch(&[f32::MAX, f32::MAX], &mut scratch),
        Err(OrionRoutingError::NonFiniteDistance { label }) if label == id(40)
    ));

    // An error must not leave reusable heap/set/query state observable by the next route.
    assert_scratch_route_matches_reference(&router, &mut scratch, &[2.75, 0.125]);
}

#[test]
fn cosine_upper_search_uses_qdrant_preprocessing_and_is_scale_invariant() {
    let mut artifact = valid_artifact();
    artifact.vector_schema.distance = Distance::Cosine;
    artifact.upper_nodes[0].vector = vec![2.0, 0.0];
    artifact.upper_nodes[1].vector = vec![0.0, 3.0];
    artifact.upper_nodes[2].vector = vec![-2.0, 0.0];
    artifact.upper_nodes[3].vector = vec![0.0, -4.0];

    let router = OrionRouter::new(artifact).unwrap();
    let unit = router.search_upper(&[1.0, 0.0]).unwrap();
    let scaled = router.search_upper(&[100.0, 0.0]).unwrap();

    assert_eq!(
        unit.iter().map(|hit| hit.label).collect::<Vec<_>>(),
        scaled.iter().map(|hit| hit.label).collect::<Vec<_>>(),
    );
    assert_eq!(unit[0].label, id(10));
    assert!((unit[0].distance + 1.0).abs() < 1e-6);
}

#[test]
fn incomplete_upper_hnsw_search_is_rejected_instead_of_routing_a_partial_union() {
    let mut artifact = valid_artifact();
    artifact.upper_graph = Some(OrionUpperHnswGraph {
        entry_point: id(10),
        max_level: 0,
        nodes: vec![
            OrionUpperGraphNode {
                label: id(10),
                neighbors_by_level: vec![vec![]],
            },
            OrionUpperGraphNode {
                label: id(20),
                neighbors_by_level: vec![vec![id(30)]],
            },
            OrionUpperGraphNode {
                label: id(30),
                neighbors_by_level: vec![vec![id(20)]],
            },
            OrionUpperGraphNode {
                label: id(40),
                neighbors_by_level: vec![vec![]],
            },
        ],
    });

    let router = OrionRouter::new(artifact).unwrap();
    assert!(matches!(
        router.search_upper(&[0.0, 0.0]),
        Err(OrionRoutingError::IncompleteUpperSearch {
            expected: 3,
            actual: 1,
        })
    ));
    let mut scratch = router.new_route_scratch();
    assert!(matches!(
        router.route_query_with_scratch(&[0.0, 0.0], &mut scratch),
        Err(OrionRoutingError::IncompleteUpperSearch {
            expected: 3,
            actual: 1,
        })
    ));
}

#[test]
fn query_and_schema_mismatches_are_rejected() {
    let artifact = valid_artifact();
    let mut other_schema = artifact.vector_schema.clone();
    other_schema.distance = Distance::Cosine;
    assert!(matches!(
        artifact.validate_schema(&other_schema),
        Err(OrionRoutingError::VectorSchemaMismatch { .. })
    ));

    let router = OrionRouter::new(artifact).unwrap();
    assert!(matches!(
        router.search_upper(&[1.0]),
        Err(OrionRoutingError::QueryDimensionMismatch { .. })
    ));
    assert!(matches!(
        router.search_upper(&[f32::NAN, 0.0]),
        Err(OrionRoutingError::NonFiniteQueryVector { dimension: 0 })
    ));
}

#[test]
fn unknown_upper_hit_is_not_silently_ignored() {
    let router = OrionRouter::new(valid_artifact()).unwrap();
    assert!(matches!(
        router.route_upper_labels([id(999)]),
        Err(OrionRoutingError::UnknownUpperLabel { label }) if label == id(999)
    ));
}
