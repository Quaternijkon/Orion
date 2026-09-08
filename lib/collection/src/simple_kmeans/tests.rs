use segment::types::Distance;

use super::*;

fn artifact() -> SimpleKmeansRoutingArtifact {
    SimpleKmeansRoutingArtifact {
        format_version: SIMPLE_KMEANS_ROUTING_ARTIFACT_FORMAT_VERSION,
        generation: 7,
        vector_schema: SimpleKmeansVectorSchemaFingerprint {
            vector_name: String::new(),
            dimension: 2,
            distance: Distance::Dot,
            datatype: SimpleKmeansVectorDatatype::Float32,
        },
        shard_count: 3,
        layout_sha256: "a".repeat(64),
        logical_point_count: 10,
        physical_point_count: 10,
        routing_distance: SimpleKmeansRoutingDistance::SquaredL2,
        nprobe: 2,
        lower_hnsw_ef: 64,
        centroids: vec![
            SimpleKmeansCentroid {
                shard_id: 0,
                vector: vec![1.0, 0.0],
            },
            SimpleKmeansCentroid {
                shard_id: 1,
                vector: vec![0.0, 1.0],
            },
            SimpleKmeansCentroid {
                shard_id: 2,
                vector: vec![-1.0, 0.0],
            },
        ],
    }
}

#[test]
fn artifact_roundtrip_and_checksum_are_stable() {
    let artifact = artifact();
    let bytes = artifact.canonical_json_bytes().unwrap();
    let checksum = artifact.canonical_sha256().unwrap();
    assert_eq!(
        SimpleKmeansRoutingArtifact::from_json_slice(&bytes, Some(&checksum)).unwrap(),
        artifact
    );
}

#[test]
fn artifact_rejects_invalid_nprobe_and_centroid_coverage() {
    let mut invalid = artifact();
    invalid.nprobe = 4;
    assert!(matches!(
        invalid.validate(),
        Err(SimpleKmeansRoutingError::NprobeExceedsShardCount { .. })
    ));

    let mut invalid = artifact();
    invalid.centroids[2].shard_id = 1;
    assert!(matches!(
        invalid.validate(),
        Err(SimpleKmeansRoutingError::DuplicateShardCentroid { .. })
    ));

    let mut invalid = artifact();
    invalid.physical_point_count += 1;
    assert!(matches!(
        invalid.validate(),
        Err(SimpleKmeansRoutingError::PhysicalPointCountMismatch { .. })
    ));
}

#[test]
fn router_returns_nearest_nprobe_numeric_shards_with_fixed_ef() {
    let router = SimpleKmeansRouter::new(artifact()).unwrap();
    let targets = router.route_query(&[0.9, 0.1]).unwrap();
    assert_eq!(
        targets,
        vec![
            SimpleKmeansShardTarget {
                shard_id: 0,
                ef: 64,
            },
            SimpleKmeansShardTarget {
                shard_id: 1,
                ef: 64,
            },
        ]
    );
}

#[test]
fn router_rejects_bad_query_shape_and_non_finite_values() {
    let router = SimpleKmeansRouter::new(artifact()).unwrap();
    assert!(matches!(
        router.route_query(&[1.0]),
        Err(SimpleKmeansRoutingError::QueryDimensionMismatch { .. })
    ));
    assert!(matches!(
        router.route_query(&[f32::NAN, 0.0]),
        Err(SimpleKmeansRoutingError::NonFiniteQueryVector { .. })
    ));
}

#[test]
fn cosine_schema_normalizes_query_but_preserves_raw_centroid_l2_ranking() {
    let mut artifact = artifact();
    artifact.vector_schema.distance = Distance::Cosine;
    artifact.nprobe = 1;
    artifact.centroids = vec![
        SimpleKmeansCentroid {
            // Cosine similarity would rank this centroid first after re-normalization.
            shard_id: 0,
            vector: vec![10.0, 0.0],
        },
        SimpleKmeansCentroid {
            // Squared L2 against normalized [1, 0] correctly ranks this raw mean first.
            shard_id: 1,
            vector: vec![0.9, 0.1],
        },
        SimpleKmeansCentroid {
            shard_id: 2,
            vector: vec![-1.0, 0.0],
        },
    ];
    let router = SimpleKmeansRouter::new(artifact).unwrap();
    assert_eq!(router.route_query(&[2.0, 0.0]).unwrap()[0].shard_id, 1);
}

#[test]
fn cosine_query_normalization_matches_python_tiny_norm_threshold() {
    let mut artifact = artifact();
    artifact.vector_schema.distance = Distance::Cosine;
    artifact.nprobe = 1;
    artifact.centroids = vec![
        SimpleKmeansCentroid {
            shard_id: 0,
            vector: vec![1.0, 0.0],
        },
        SimpleKmeansCentroid {
            shard_id: 1,
            vector: vec![0.0, 0.0],
        },
        SimpleKmeansCentroid {
            shard_id: 2,
            vector: vec![-1.0, 0.0],
        },
    ];
    let router = SimpleKmeansRouter::new(artifact).unwrap();
    // 1e-8 is far below the old 1e-6 cutoff but above Python normalize_rows' 1e-12 cutoff.
    assert_eq!(router.route_query(&[1.0e-8, 0.0]).unwrap()[0].shard_id, 0);
}
