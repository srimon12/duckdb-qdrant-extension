//! Live checks against a Qdrant 1.20 server.
//! Set `QDRANT_TEST_URL` (gRPC, usually port 6334) to run them.

use std::time::Duration;

use qdrant_client::qdrant::{
    CreateCollectionBuilder, CreateFieldIndexCollectionBuilder, Distance, FieldType, Fusion,
    NamedVectors, PointStruct, RecommendStrategy, SparseVectorParamsBuilder,
    SparseVectorsConfigBuilder, TextIndexParamsBuilder, TokenizerType, VectorParamsBuilder,
    VectorsConfigBuilder,
};
use qdrant_client::{Payload, Qdrant};

use crate::batch::run_batch;
use crate::client::{Endpoint, block_on};
use crate::collections::list_collections;
use crate::discover::run_discover;
use crate::hybrid::{run_hybrid, run_hybrid_text};
use crate::query::run_nearest;
use crate::recommend::run_recommend;
use crate::scroll::{ScrollRequest, run_scroll};
use crate::text::run_text;
use crate::types::{
    BatchMode, BatchRequest, DiscoverRequest, HybridRequest, HybridTextRequest, QueryRequest,
    RecommendRequest, Scope, SparseVector, TextRequest, batch_limit, parse_batch_queries,
    parse_context_pairs, parse_filter,
};
use crate::upsert::{UpsertBind, upsert_batch};

fn endpoint() -> Option<Endpoint> {
    let url = std::env::var("QDRANT_TEST_URL").ok()?;
    if url.is_empty() {
        return None;
    }
    Some(Endpoint {
        url,
        api_key: std::env::var("QDRANT_TEST_API_KEY").ok(),
        timeout_secs: 30,
    })
}

fn scope(endpoint: &Endpoint, collection: &str, limit: u64) -> Scope {
    Scope {
        collection: collection.to_string(),
        limit,
        filter: None,
        using: None,
        score_threshold: None,
        with_payload: true,
        payload_fields: None,
        endpoint: endpoint.clone(),
    }
}

struct Guard {
    client: Qdrant,
    name: String,
}

impl Drop for Guard {
    fn drop(&mut self) {
        let client = self.client.clone();
        let name = self.name.clone();
        let _ = block_on(async move { client.delete_collection(name).await });
    }
}

#[test]
fn live_qdrant_120() {
    let Some(endpoint) = endpoint() else {
        return;
    };
    live(&endpoint).unwrap();
}

fn live(endpoint: &Endpoint) -> Result<(), String> {
    let client = Qdrant::from_url(&endpoint.url)
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|err| err.to_string())?;
    let name = format!("ddb_q_{}", std::process::id());
    let _guard = Guard {
        client: client.clone(),
        name: name.clone(),
    };
    block_on(client.delete_collection(name.clone())).ok();

    let mut vectors = VectorsConfigBuilder::default();
    vectors.add_named_vector_params("dense", VectorParamsBuilder::new(4, Distance::Dot));
    let mut sparse = SparseVectorsConfigBuilder::default();
    sparse.add_named_vector_params("sparse", SparseVectorParamsBuilder::default());
    block_on(async {
        client
            .create_collection(
                CreateCollectionBuilder::new(name.clone())
                    .vectors_config(vectors)
                    .sparse_vectors_config(sparse),
            )
            .await
    })
    .map_err(|err| err.to_string())?;

    let points = [
        (0u64, [1.0, 0.0, 0.0, 0.0], vec![1u32], vec![1.0], 2020),
        (1, [0.0, 1.0, 0.0, 0.0], vec![2], vec![1.0], 2024),
        (2, [0.0, 0.0, 1.0, 0.0], vec![1, 2], vec![0.2, 0.8], 2024),
        (3, [0.9, 0.1, 0.0, 0.0], vec![1], vec![0.4], 2024),
    ];
    let upserted = points
        .into_iter()
        .map(|(id, dense, indices, values, year)| {
            let payload = Payload::try_from(serde_json::json!({
                "year": year,
                "color": "red",
                "body": "cashew granola bars"
            }))
            .map_err(|err| err.to_string())?;
            Ok(PointStruct::new(
                id,
                NamedVectors::default()
                    .add_vector("dense", dense.to_vec())
                    .add_vector(
                        "sparse",
                        qdrant_client::qdrant::Vector::new_sparse(indices, values),
                    ),
                payload,
            ))
        })
        .collect::<Result<Vec<_>, String>>()?;
    block_on(async {
        client
            .upsert_points(
                qdrant_client::qdrant::UpsertPointsBuilder::new(name.clone(), upserted).wait(true),
            )
            .await
    })
    .map_err(|err| err.to_string())?;

    block_on(async {
        client
            .create_field_index(
                CreateFieldIndexCollectionBuilder::new(name.clone(), "body", FieldType::Text)
                    .field_index_params(
                        TextIndexParamsBuilder::new(TokenizerType::Word)
                            .bm25_scoring()
                            .build(),
                    )
                    .wait(true),
            )
            .await
    })
    .map_err(|err| err.to_string())?;

    let mut text_scope = scope(endpoint, &name, 2);
    text_scope.using = Some("body".to_string());
    let text_rows = run_text(&TextRequest {
        scope: text_scope.clone(),
        text: "cashew granola".to_string(),
        k: Some(1.5),
        b: Some(0.5),
    })?;
    assert!(
        !text_rows.is_empty(),
        "BM25 over the body index should match every point"
    );

    let mut nearest_scope = scope(endpoint, &name, 2);
    nearest_scope.using = Some("dense".to_string());
    let nearest = run_nearest(&QueryRequest {
        scope: nearest_scope.clone(),
        vector: vec![1.0, 0.0, 0.0, 0.0],
    })?;
    assert_eq!(nearest[0].id, "0", "closest dense point should be id 0");
    assert!(
        nearest[0].payload.contains("2020"),
        "{}",
        nearest[0].payload
    );

    let dense_queries = parse_batch_queries(
        "[[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0]]",
        BatchMode::Dense,
        None,
        None,
    )?;
    let dense_vectors = [vec![1.0_f32, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
    let dense_batch = run_batch(&BatchRequest {
        scope: nearest_scope.clone(),
        queries: dense_queries.clone(),
    })?;
    for (query_index, vector) in dense_vectors.iter().enumerate() {
        let ids: Vec<&str> = dense_batch
            .iter()
            .filter(|row| row.query_index == query_index as u64)
            .map(|row| row.id.as_str())
            .collect();
        let single = run_nearest(&QueryRequest {
            scope: nearest_scope.clone(),
            vector: vector.clone(),
        })?;
        let single_ids: Vec<&str> = single.iter().map(|row| row.id.as_str()).collect();
        assert_eq!(ids, single_ids, "batch must match one query per call");
    }
    assert_eq!(
        dense_batch
            .iter()
            .find(|row| row.query_index == 0)
            .map(|row| row.id.as_str()),
        Some("0")
    );

    let mut text_batch_scope = scope(endpoint, &name, 2);
    text_batch_scope.using = Some("body".to_string());
    let text_batch = run_batch(&BatchRequest {
        scope: text_batch_scope,
        queries: parse_batch_queries(
            r#"["cashew granola","cashew"]"#,
            BatchMode::Text,
            Some(1.2),
            Some(0.75),
        )?,
    })?;
    let batch_text_ids: Vec<&str> = text_batch
        .iter()
        .filter(|row| row.query_index == 0)
        .map(|row| row.id.as_str())
        .collect();
    let single_text_ids: Vec<&str> = text_rows.iter().map(|row| row.id.as_str()).collect();
    assert_eq!(
        batch_text_ids, single_text_ids,
        "text batch must match qdrant_text"
    );

    let hybrid_text = run_hybrid_text(&HybridTextRequest {
        scope: scope(endpoint, &name, 2),
        dense: vec![1.0, 0.0, 0.0, 0.0],
        dense_using: Some("dense".to_string()),
        text: "cashew granola".to_string(),
        text_field: "body".to_string(),
        fusion: Fusion::Rrf,
        prefetch_limit: 10,
    })?;
    assert!(
        !hybrid_text.is_empty(),
        "dense + text hybrid must return rows"
    );

    let too_many = run_batch(&BatchRequest {
        scope: scope(endpoint, &name, 130),
        queries: dense_queries,
    });
    let too_many = too_many.expect_err("a batch limit above 128 must fail");
    assert!(too_many.contains("128"), "{too_many}");
    assert!(batch_limit(130).unwrap_err().contains("130"));

    let mut filtered = scope(endpoint, &name, 4);
    filtered.using = Some("dense".to_string());
    filtered.filter = parse_filter(r#"{"must":[{"key":"year","range":{"gte":2024}}]}"#)?;
    let filtered_rows = run_nearest(&QueryRequest {
        scope: filtered,
        vector: vec![1.0, 0.0, 0.0, 0.0],
    })?;
    assert!(filtered_rows.iter().all(|row| row.id != "0"));
    assert_eq!(filtered_rows[0].id, "3");

    let mut selected = scope(endpoint, &name, 1);
    selected.using = Some("dense".to_string());
    selected.payload_fields = Some(vec!["year".to_string()]);
    let selected_rows = run_nearest(&QueryRequest {
        scope: selected,
        vector: vec![0.0, 1.0, 0.0, 0.0],
    })?;
    assert!(
        selected_rows[0].payload.contains("year"),
        "{}",
        selected_rows[0].payload
    );
    assert!(
        !selected_rows[0].payload.contains("color"),
        "payload selector should drop color: {}",
        selected_rows[0].payload
    );

    let mut hybrid_scope = scope(endpoint, &name, 3);
    let hybrid = run_hybrid(&HybridRequest {
        scope: hybrid_scope.clone(),
        dense: vec![1.0, 0.0, 0.0, 0.0],
        dense_using: Some("dense".to_string()),
        sparse: SparseVector {
            indices: vec![2],
            values: vec![1.0],
        },
        sparse_using: "sparse".to_string(),
        fusion: Fusion::Rrf,
        prefetch_limit: 10,
    })?;
    assert!(!hybrid.is_empty());
    hybrid_scope.limit = 3;
    let dbsf = run_hybrid(&HybridRequest {
        scope: hybrid_scope,
        dense: vec![0.0, 0.0, 1.0, 0.0],
        dense_using: Some("dense".to_string()),
        sparse: SparseVector {
            indices: vec![1],
            values: vec![1.0],
        },
        sparse_using: "sparse".to_string(),
        fusion: Fusion::Dbsf,
        prefetch_limit: 10,
    })?;
    assert!(!dbsf.is_empty());

    let mut recommend_scope = scope(endpoint, &name, 2);
    recommend_scope.using = Some("dense".to_string());
    let recommended = run_recommend(&RecommendRequest {
        scope: recommend_scope,
        positive_ids: vec![0.into()],
        negative_ids: vec![2.into()],
        positive_vector: None,
        negative_vector: None,
        strategy: RecommendStrategy::AverageVector,
    })?;
    assert!(!recommended.is_empty());
    assert!(recommended.iter().all(|row| row.id != "0"));

    let mut discover_scope = scope(endpoint, &name, 2);
    discover_scope.using = Some("dense".to_string());
    let discovered = run_discover(
        "qdrant_discover",
        &DiscoverRequest {
            scope: discover_scope.clone(),
            target: Some(vec![1.0, 0.0, 0.0, 0.0]),
            pairs: parse_context_pairs(r#"[{"positive":"1","negative":"2"}]"#)?,
        },
    )?;
    assert!(!discovered.is_empty());
    let context_rows = run_discover(
        "qdrant_context",
        &DiscoverRequest {
            scope: discover_scope,
            target: None,
            pairs: parse_context_pairs(
                r#"[{"positive":[0.0,1.0,0.0,0.0],"negative":[0.0,0.0,1.0,0.0]}]"#,
            )?,
        },
    )?;
    assert!(!context_rows.is_empty());

    let mut scroll_scope = scope(endpoint, &name, 10);
    let scrolled = run_scroll(&ScrollRequest {
        scope: scroll_scope.clone(),
        offset: None,
    })?;
    assert_eq!(scrolled.len(), 4);
    scroll_scope.limit = 2;
    let start = scrolled[1].id.clone();
    let page = run_scroll(&ScrollRequest {
        scope: scroll_scope,
        offset: Some(start.clone()),
    })?;
    assert!(
        page.iter().all(|row| row.id.as_str() >= start.as_str()),
        "scroll offset should not return ids before {start}: {page:?}"
    );

    let names = list_collections(endpoint)?;
    assert!(names.iter().any(|item| item == &name), "{names:?}");

    let info_client = client.clone();
    let info_name = name.clone();
    let info = block_on(async move { info_client.collection_info(info_name).await })
        .map_err(|err| err.to_string())?;
    let points = info.result.and_then(|info| info.points_count).unwrap_or(0);
    assert_eq!(points, 4);

    let mut many_ids = Vec::new();
    let mut many_vectors = Vec::new();
    let mut many_payloads = Vec::new();
    for index in 0..130 {
        many_ids.push(format!("{}", 100 + index));
        many_vectors.push(vec![0.01, 0.0, 0.0, index as f32 / 130.0]);
        many_payloads.push(r#"{"year":2024}"#.to_string());
    }
    upsert_batch(&UpsertBind {
        collection: name.clone(),
        ids: many_ids,
        vectors: many_vectors,
        payloads: many_payloads,
        using: Some("dense".to_string()),
        endpoint: endpoint.clone(),
    })?;
    let mut paged = scope(endpoint, &name, 130);
    paged.using = Some("dense".to_string());
    let paged_rows = run_nearest(&QueryRequest {
        scope: paged,
        vector: vec![1.0, 0.0, 0.0, 0.0],
    })?;
    assert_eq!(
        paged_rows.len(),
        130,
        "large limits must cross the 128-row page"
    );

    let status = upsert_batch(&UpsertBind {
        collection: name.clone(),
        ids: vec!["9".to_string()],
        vectors: vec![vec![0.2, 0.2, 0.2, 0.2]],
        payloads: vec![r#"{"year":2025}"#.to_string()],
        using: Some("dense".to_string()),
        endpoint: endpoint.clone(),
    })?;
    assert!(
        status.eq_ignore_ascii_case("completed") || status.eq_ignore_ascii_case("acknowledged"),
        "{status}"
    );

    let mut mismatch_scope = scope(endpoint, &name, 1);
    mismatch_scope.using = Some("dense".to_string());
    let mismatch = run_nearest(&QueryRequest {
        scope: mismatch_scope,
        vector: vec![1.0, 0.0],
    });
    let mismatch = mismatch.expect_err("short vector must fail");
    assert!(
        mismatch.to_ascii_lowercase().contains("dimension")
            || mismatch.to_ascii_lowercase().contains("vector"),
        "{mismatch}"
    );

    let missing = run_nearest(&QueryRequest {
        scope: scope(endpoint, "missing_collection_ddb", 1),
        vector: vec![1.0, 0.0, 0.0, 0.0],
    });
    assert!(missing.is_err(), "missing collection must fail");
    Ok(())
}
