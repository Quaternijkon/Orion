use api::grpc::qdrant as grpc;
use collection::config::AutoShardPolicy;
use storage::content_manager::collection_meta_ops::CollectionMetaOperations;

const VALID_SHA256: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn convert_create_policy(
    auto_shard_policy: Option<grpc::AutoShardPolicy>,
) -> Option<AutoShardPolicy> {
    let operation = CollectionMetaOperations::try_from(grpc::CreateCollection {
        collection_name: "test_collection".to_string(),
        auto_shard_policy,
        ..Default::default()
    })
    .unwrap();

    let operation = match operation {
        CollectionMetaOperations::CreateCollection(operation) => operation,
        _ => panic!("expected create collection operation"),
    };
    operation.create_collection.auto_shard_policy
}

#[test]
fn grpc_create_collection_policy_is_backward_compatible_and_canonical() {
    assert_eq!(convert_create_policy(None), None);
    assert_eq!(
        convert_create_policy(Some(grpc::AutoShardPolicy {
            policy: Some(grpc::auto_shard_policy::Policy::HashAll(
                grpc::HashAllAutoShardPolicy {},
            )),
        })),
        None,
    );
}

#[test]
fn grpc_create_collection_preserves_orion_policy() {
    let expected = AutoShardPolicy::Orion {
        generation: 9,
        artifact_sha256: VALID_SHA256.to_string(),
    };
    let actual = convert_create_policy(Some(grpc::AutoShardPolicy {
        policy: Some(grpc::auto_shard_policy::Policy::Orion(
            grpc::OrionAutoShardPolicy {
                generation: 9,
                artifact_sha256: VALID_SHA256.to_string(),
            },
        )),
    }));

    assert_eq!(actual, Some(expected));
}

#[test]
fn grpc_create_collection_preserves_simple_kmeans_policy() {
    let expected = AutoShardPolicy::SimpleKmeans {
        generation: 10,
        artifact_sha256: VALID_SHA256.to_string(),
    };
    let actual = convert_create_policy(Some(grpc::AutoShardPolicy {
        policy: Some(grpc::auto_shard_policy::Policy::SimpleKmeans(
            grpc::SimpleKmeansAutoShardPolicy {
                generation: 10,
                artifact_sha256: VALID_SHA256.to_string(),
            },
        )),
    }));

    assert_eq!(actual, Some(expected));
}
