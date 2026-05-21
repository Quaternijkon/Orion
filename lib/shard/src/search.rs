use std::collections::HashMap;

use common::types::ScoreType;
#[cfg(feature = "api")]
use itertools::Itertools as _;
#[cfg(feature = "api")]
use segment::data_types::vectors::NamedQuery;
use segment::types::{
    Filter, PointIdType, SearchParams, ShardKey, WithPayloadInterface, WithVector,
};
#[cfg(feature = "api")]
use segment::{data_types::vectors::VectorInternal, vector_storage::query::ContextPair};

use crate::query::query_enum::QueryEnum;

/// DEPRECATED: Search method should be removed and replaced with `ShardQueryRequest`
#[derive(Clone, Debug, PartialEq)]
pub struct CoreSearchRequest {
    /// Every kind of query that can be performed on segment level
    pub query: QueryEnum,
    /// Look only for points which satisfies this conditions
    pub filter: Option<Filter>,
    /// Additional search params
    pub params: Option<SearchParams>,
    /// Optional HNSW entry points used to start graph search from specific point IDs.
    pub hnsw_entry_points: Option<Vec<PointIdType>>,
    /// Optional per-shard HNSW entry points. This is consumed by collection fanout
    /// and converted to `hnsw_entry_points` before a shard-local search is sent.
    pub hnsw_entry_points_by_shard: Option<HashMap<ShardKey, Vec<PointIdType>>>,
    /// Optional per-shard HNSW EF values. This is consumed by collection fanout
    /// and converted to `params.hnsw_ef` before a shard-local search is sent.
    pub hnsw_ef_by_shard: Option<HashMap<ShardKey, usize>>,
    /// Optional block size for experiment collections that encode duplicate copies
    /// as `shard_id * block_size + source_id + 1`.
    pub source_id_dedup_block_size: Option<u64>,
    /// Max number of result to return
    pub limit: usize,
    /// Offset of the first result to return.
    /// May be used to paginate results.
    /// Note: large offset values may cause performance issues.
    pub offset: usize,
    /// Select which payload to return with the response. Default is false.
    pub with_payload: Option<WithPayloadInterface>,
    /// Options for specifying which vectors to include into response. Default is false.
    pub with_vector: Option<WithVector>,
    pub score_threshold: Option<ScoreType>,
}

impl CoreSearchRequest {
    pub fn search_rate_cost(&self) -> usize {
        let mut cost = self.query.search_cost();

        if let Some(filter) = &self.filter {
            cost += filter.total_conditions_count();
        }

        cost
    }
}

#[cfg(feature = "api")]
impl From<api::rest::SearchRequestInternal> for CoreSearchRequest {
    fn from(request: api::rest::SearchRequestInternal) -> Self {
        #[cfg(feature = "api")]
        use segment::data_types::vectors::NamedVectorStruct;

        let api::rest::SearchRequestInternal {
            vector,
            filter,
            score_threshold,
            limit,
            offset,
            params,
            hnsw_entry_points,
            hnsw_entry_points_by_shard,
            hnsw_ef_by_shard,
            source_id_dedup_block_size,
            with_vector,
            with_payload,
        } = request;
        Self {
            query: QueryEnum::Nearest(NamedQuery::from(NamedVectorStruct::from(vector))),
            filter,
            params,
            hnsw_entry_points,
            hnsw_entry_points_by_shard,
            hnsw_ef_by_shard,
            source_id_dedup_block_size,
            limit,
            offset: offset.unwrap_or_default(),
            with_payload,
            with_vector,
            score_threshold,
        }
    }
}

#[cfg(feature = "api")]
impl TryFrom<api::grpc::qdrant::CoreSearchPoints> for CoreSearchRequest {
    type Error = tonic::Status;

    fn try_from(value: api::grpc::qdrant::CoreSearchPoints) -> Result<Self, Self::Error> {
        use segment::data_types::vectors::VectorInternal;
        use segment::vector_storage::query::{ContextQuery, DiscoverQuery, RecoQuery};

        let query = value
            .query
            .and_then(|query| query.query)
            .map(|query| {
                Ok(match query {
                    api::grpc::qdrant::query_enum::Query::NearestNeighbors(vector) => {
                        let vector_internal = VectorInternal::try_from(vector)?;
                        QueryEnum::Nearest(NamedQuery::from(
                            api::grpc::conversions::into_named_vector_struct(
                                value.vector_name,
                                vector_internal,
                            )?,
                        ))
                    }
                    api::grpc::qdrant::query_enum::Query::RecommendBestScore(query) => {
                        QueryEnum::RecommendBestScore(NamedQuery {
                            query: RecoQuery::try_from(query)?,
                            using: value.vector_name,
                        })
                    }
                    api::grpc::qdrant::query_enum::Query::RecommendSumScores(query) => {
                        QueryEnum::RecommendSumScores(NamedQuery {
                            query: RecoQuery::try_from(query)?,
                            using: value.vector_name,
                        })
                    }
                    api::grpc::qdrant::query_enum::Query::Discover(query) => {
                        let Some(target) = query.target else {
                            return Err(tonic::Status::invalid_argument("Target is not specified"));
                        };

                        let pairs = query
                            .context
                            .into_iter()
                            .map(try_context_pair_from_grpc)
                            .try_collect()?;

                        QueryEnum::Discover(NamedQuery {
                            query: DiscoverQuery::new(target.try_into()?, pairs),
                            using: value.vector_name,
                        })
                    }
                    api::grpc::qdrant::query_enum::Query::Context(query) => {
                        let pairs = query
                            .context
                            .into_iter()
                            .map(try_context_pair_from_grpc)
                            .try_collect()?;

                        QueryEnum::Context(NamedQuery {
                            query: ContextQuery::new(pairs),
                            using: value.vector_name,
                        })
                    }
                })
            })
            .transpose()?
            .ok_or_else(|| tonic::Status::invalid_argument("Query is not specified"))?;

        Ok(Self {
            query,
            filter: value.filter.map(|f| f.try_into()).transpose()?,
            params: value.params.map(|p| p.into()),
            hnsw_entry_points: if value.hnsw_entry_points.is_empty() {
                None
            } else {
                Some(
                    value
                        .hnsw_entry_points
                        .into_iter()
                        .map(PointIdType::try_from)
                    .collect::<Result<Vec<_>, _>>()?,
                )
            },
            hnsw_entry_points_by_shard: None,
            hnsw_ef_by_shard: None,

            source_id_dedup_block_size: None,
            limit: value.limit as usize,
            offset: value.offset.unwrap_or_default() as usize,
            with_payload: value.with_payload.map(|wp| wp.try_into()).transpose()?,
            with_vector: Some(
                value
                    .with_vectors
                    .map(|with_vectors| with_vectors.into())
                    .unwrap_or_default(),
            ),
            score_threshold: value.score_threshold,
        })
    }
}

#[cfg(feature = "api")]
fn try_context_pair_from_grpc(
    pair: api::grpc::qdrant::ContextPair,
) -> Result<ContextPair<VectorInternal>, tonic::Status> {
    let api::grpc::qdrant::ContextPair { positive, negative } = pair;
    match (positive, negative) {
        (Some(positive), Some(negative)) => Ok(ContextPair {
            positive: positive.try_into()?,
            negative: negative.try_into()?,
        }),
        _ => Err(tonic::Status::invalid_argument(
            "All context pairs must have both positive and negative parts",
        )),
    }
}

#[cfg(feature = "api")]
impl TryFrom<api::grpc::qdrant::SearchPoints> for CoreSearchRequest {
    type Error = tonic::Status;

    fn try_from(value: api::grpc::qdrant::SearchPoints) -> Result<Self, Self::Error> {
        use sparse::common::sparse_vector::validate_sparse_vector_impl;

        let api::grpc::qdrant::SearchPoints {
            collection_name: _,
            vector,
            filter,
            limit,
            with_payload,
            params,
            score_threshold,
            offset,
            vector_name,
            with_vectors,
            read_consistency: _,
            timeout: _,
            shard_key_selector: _,
            sparse_indices,
        } = value;

        if let Some(sparse_indices) = &sparse_indices {
            let api::grpc::qdrant::SparseIndices { data } = sparse_indices;
            validate_sparse_vector_impl(data, &vector).map_err(|e| {
                tonic::Status::invalid_argument(format!(
                    "Sparse indices does not match sparse vector conditions: {e}"
                ))
            })?;
        }

        let vector_internal =
            VectorInternal::from_vector_and_indices(vector, sparse_indices.map(|v| v.data));

        let vector_struct =
            api::grpc::conversions::into_named_vector_struct(vector_name, vector_internal)?;

        Ok(Self {
            query: QueryEnum::Nearest(NamedQuery::from(vector_struct)),
            filter: filter.map(Filter::try_from).transpose()?,
            params: params.map(SearchParams::from),
            hnsw_entry_points: None,
            hnsw_entry_points_by_shard: None,
            hnsw_ef_by_shard: None,

            source_id_dedup_block_size: None,
            limit: limit as usize,
            offset: offset.map(|v| v as usize).unwrap_or_default(),
            with_payload: with_payload
                .map(WithPayloadInterface::try_from)
                .transpose()?,
            with_vector: with_vectors.map(WithVector::from),
            score_threshold: score_threshold.map(|s| s as ScoreType),
        })
    }
}

#[derive(Debug, Clone)]
pub struct CoreSearchRequestBatch {
    pub searches: Vec<CoreSearchRequest>,
}

#[cfg(all(test, feature = "api"))]
mod tests {
    use std::collections::HashMap;

    use api::grpc::qdrant;
    use serde_json::json;
    use segment::data_types::vectors::VectorInternal;
    use segment::types::{PointIdType, ShardKey};

    use super::*;

    #[test]
    fn rest_search_request_preserves_hnsw_entry_points() {
        let request: api::rest::SearchRequestInternal = serde_json::from_value(json!({
            "vector": [1.0, 2.0, 3.0, 4.0],
            "limit": 10,
            "hnsw_entry_points": [11, 29]
        }))
        .unwrap();

        let core_request = CoreSearchRequest::from(request);

        assert_eq!(
            core_request.hnsw_entry_points,
            Some(vec![PointIdType::from(11), PointIdType::from(29)])
        );
    }

    #[test]
    fn rest_search_request_preserves_hnsw_entry_points_and_ef_by_shard() {
        let request: api::rest::SearchRequestInternal = serde_json::from_value(json!({
            "vector": [1.0, 2.0, 3.0, 4.0],
            "limit": 10,
            "params": {"hnsw_ef": 28},
            "source_id_dedup_block_size": 1001,
            "hnsw_entry_points_by_shard": {
                "centroid_00": [11],
                "centroid_01": [29, 31]
            },
            "hnsw_ef_by_shard": {
                "centroid_00": 24,
                "centroid_01": 28
            }
        }))
        .unwrap();

        let core_request = CoreSearchRequest::from(request);

        assert_eq!(
            core_request.hnsw_entry_points_by_shard,
            Some(HashMap::from([
                (ShardKey::from("centroid_00"), vec![PointIdType::from(11)]),
                (
                    ShardKey::from("centroid_01"),
                    vec![PointIdType::from(29), PointIdType::from(31)],
                ),
            ]))
        );
        assert_eq!(
            core_request.hnsw_ef_by_shard,
            Some(HashMap::from([
                (ShardKey::from("centroid_00"), 24),
                (ShardKey::from("centroid_01"), 28),
            ]))
        );
        assert_eq!(core_request.source_id_dedup_block_size, Some(1001));
    }

    #[test]
    fn grpc_core_search_request_preserves_hnsw_entry_points() {
        let grpc_request = qdrant::CoreSearchPoints {
            collection_name: "test".to_string(),
            query: Some(qdrant::QueryEnum {
                query: Some(qdrant::query_enum::Query::NearestNeighbors(
                    VectorInternal::Dense(vec![1.0, 2.0, 3.0, 4.0]).into(),
                )),
            }),
            filter: None,
            limit: 10,
            with_payload: None,
            params: None,
            score_threshold: None,
            offset: None,
            vector_name: None,
            with_vectors: None,
            read_consistency: None,
            hnsw_entry_points: vec![PointIdType::from(11).into(), PointIdType::from(29).into()],
        };

        let core_request = CoreSearchRequest::try_from(grpc_request).unwrap();

        assert_eq!(
            core_request.hnsw_entry_points,
            Some(vec![PointIdType::from(11), PointIdType::from(29)])
        );
    }
}
