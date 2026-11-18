//! DuckDB values to Qdrant query types.
//!
//! Filters accept the Qdrant REST JSON shape for must, should, must_not,
//! min_should, field match, field range, has_id, is_empty, is_null, and nested.
//! Other condition kinds return an error instead of being dropped.

use qdrant_client::qdrant::with_payload_selector::SelectorOptions;
use qdrant_client::qdrant::{
    Condition, ContextInputBuilder, DiscoverInputBuilder, Filter, Fusion, MinShould,
    PayloadIncludeSelector, PointId, PrefetchQueryBuilder, Query, QueryPoints, QueryPointsBuilder,
    Range, RecommendInputBuilder, RecommendStrategy, TextQueryBuilder, VectorInput, query,
};
use serde_json::Value;

use crate::client::Endpoint;

pub const DEFAULT_URL: &str = "http://localhost:6334";
pub const DEFAULT_LIMIT: u64 = 10;
pub const DEFAULT_TIMEOUT_SECS: u64 = 30;
pub const PAGE_SIZE: u64 = 128;

#[derive(Clone)]
pub struct Scope {
    pub collection: String,
    pub limit: u64,
    pub filter: Option<Filter>,
    pub using: Option<String>,
    pub score_threshold: Option<f32>,
    pub with_payload: bool,
    pub payload_fields: Option<Vec<String>>,
    pub endpoint: Endpoint,
}

#[derive(Clone)]
pub struct QueryRequest {
    pub scope: Scope,
    pub vector: Vec<f32>,
}

#[derive(Clone)]
pub struct HybridRequest {
    pub scope: Scope,
    pub dense: Vec<f32>,
    pub dense_using: Option<String>,
    pub sparse: SparseVector,
    pub sparse_using: String,
    pub fusion: Fusion,
    pub prefetch_limit: u64,
}

#[derive(Clone)]
pub struct TextRequest {
    pub scope: Scope,
    pub text: String,
    pub k: Option<f32>,
    pub b: Option<f32>,
}

#[derive(Clone)]
pub struct HybridTextRequest {
    pub scope: Scope,
    pub dense: Vec<f32>,
    pub dense_using: Option<String>,
    pub text: String,
    pub text_field: String,
    pub fusion: Fusion,
    pub prefetch_limit: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BatchMode {
    Dense,
    Text,
}

/// One `query_batch` call: every query carries the same scope and filter.
#[derive(Clone)]
pub struct BatchRequest {
    pub scope: Scope,
    pub queries: Vec<Query>,
}

#[derive(Clone)]
pub struct SparseVector {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

#[derive(Clone)]
pub struct RecommendRequest {
    pub scope: Scope,
    pub positive_ids: Vec<PointId>,
    pub negative_ids: Vec<PointId>,
    pub positive_vector: Option<Vec<f32>>,
    pub negative_vector: Option<Vec<f32>>,
    pub strategy: RecommendStrategy,
}

#[derive(Clone)]
pub struct DiscoverRequest {
    pub scope: Scope,
    pub target: Option<Vec<f32>>,
    pub pairs: Vec<(VectorInput, VectorInput)>,
}

pub fn dense_vector(values: Vec<Option<f64>>) -> Result<Vec<f32>, String> {
    if values.is_empty() {
        return Err("query vector is empty".to_string());
    }
    let mut vector = Vec::with_capacity(values.len());
    for (index, value) in values.into_iter().enumerate() {
        let value = value.ok_or_else(|| format!("query vector contains NULL at index {index}"))?;
        if !value.is_finite() {
            return Err(format!(
                "query vector contains a non-finite value at index {index}"
            ));
        }
        vector.push(value as f32);
    }
    Ok(vector)
}

pub fn parse_filter(text: &str) -> Result<Option<Filter>, String> {
    let text = text.trim();
    if text.is_empty() {
        return Ok(None);
    }
    let value: Value =
        serde_json::from_str(text).map_err(|err| format!("invalid filter JSON: {err}"))?;
    Ok(Some(filter_from_value(&value)?))
}

pub fn payload_selector(with_payload: bool, fields: Option<&[String]>) -> SelectorOptions {
    match fields {
        Some([]) => false.into(),
        Some(fields) => SelectorOptions::Include(PayloadIncludeSelector::new(fields.to_vec())),
        None if with_payload => true.into(),
        None => false.into(),
    }
}

pub fn decorate(scope: &Scope, mut query: QueryPointsBuilder) -> QueryPointsBuilder {
    query = query.limit(scope.limit).with_payload(payload_selector(
        scope.with_payload,
        scope.payload_fields.as_deref(),
    ));
    if let Some(filter) = &scope.filter {
        query = query.filter(filter.clone());
    }
    if let Some(using) = &scope.using {
        query = query.using(using.clone());
    }
    if let Some(threshold) = scope.score_threshold {
        query = query.score_threshold(threshold);
    }
    query
}

pub fn nearest_page(request: &QueryRequest, offset: u64, page: u64) -> QueryPointsBuilder {
    decorate(
        &request.scope,
        QueryPointsBuilder::new(request.scope.collection.clone()).query(request.vector.clone()),
    )
    .limit(page)
    .offset(offset)
}

pub fn text_page(request: &TextRequest, offset: u64, page: u64) -> QueryPointsBuilder {
    decorate(
        &request.scope,
        QueryPointsBuilder::new(request.scope.collection.clone()).query(text_query(
            &request.text,
            request.k,
            request.b,
        )),
    )
    .limit(page)
    .offset(offset)
}

pub fn text_query(text: &str, k: Option<f32>, b: Option<f32>) -> Query {
    let mut builder = TextQueryBuilder::new(text.to_string());
    if let Some(k) = k {
        builder = builder.k(k);
    }
    if let Some(b) = b {
        builder = builder.b(b);
    }
    Query {
        variant: Some(query::Variant::Text(builder.build())),
    }
}

pub fn hybrid_text_query(request: &HybridTextRequest, offset: u64) -> QueryPointsBuilder {
    let mut dense = PrefetchQueryBuilder::default()
        .query(Query::new_nearest(request.dense.clone()))
        .limit(request.prefetch_limit);
    if let Some(using) = &request.dense_using {
        dense = dense.using(using.clone());
    }
    let text = PrefetchQueryBuilder::default()
        .using(request.text_field.clone())
        .query(text_query(&request.text, None, None))
        .limit(request.prefetch_limit);
    decorate(
        &request.scope,
        QueryPointsBuilder::new(request.scope.collection.clone())
            .query(Query::new_fusion(request.fusion))
            .add_prefetch(dense)
            .add_prefetch(text),
    )
    .offset(offset)
}

pub fn parse_batch_mode(text: &str) -> Result<BatchMode, String> {
    match text.trim().to_ascii_lowercase().as_str() {
        "" | "dense" => Ok(BatchMode::Dense),
        "text" => Ok(BatchMode::Text),
        other => Err(format!("mode must be dense or text, got {other}")),
    }
}

/// A batch call returns one page per query in v1, so a limit must fit in one page.
pub fn batch_limit(limit: u64) -> Result<u64, String> {
    if limit > PAGE_SIZE {
        return Err(format!(
            "limit {limit} is above {PAGE_SIZE}; qdrant_batch returns one page per query in v1"
        ));
    }
    Ok(limit)
}

pub fn parse_batch_queries(
    text: &str,
    mode: BatchMode,
    k: Option<f32>,
    b: Option<f32>,
) -> Result<Vec<Query>, String> {
    let value: Value =
        serde_json::from_str(text).map_err(|err| format!("invalid queries JSON: {err}"))?;
    let array = value
        .as_array()
        .ok_or_else(|| "queries must be a JSON array".to_string())?;
    if array.is_empty() {
        return Err("queries is empty".to_string());
    }
    match mode {
        BatchMode::Dense => array
            .iter()
            .enumerate()
            .map(|(index, item)| {
                let vector = f32_array(item).map_err(|err| format!("query {index}: {err}"))?;
                if vector.is_empty() {
                    return Err(format!("query {index}: vector is empty"));
                }
                Ok(Query::new_nearest(vector))
            })
            .collect(),
        BatchMode::Text => array
            .iter()
            .enumerate()
            .map(|(index, item)| {
                let text = item
                    .as_str()
                    .ok_or_else(|| format!("query {index}: text queries must be JSON strings"))?;
                if text.trim().is_empty() {
                    return Err(format!("query {index}: text is empty"));
                }
                Ok(text_query(text, k, b))
            })
            .collect(),
    }
}

/// One request per query. `scope.using` is the vector name, or the text field.
pub fn batch_queries(scope: &Scope, queries: &[Query]) -> Vec<QueryPoints> {
    queries
        .iter()
        .map(|query| {
            decorate(
                scope,
                QueryPointsBuilder::new(scope.collection.clone()).query(query.clone()),
            )
            .build()
        })
        .collect()
}

pub fn hybrid_query(request: &HybridRequest, offset: u64) -> QueryPointsBuilder {
    let mut dense = PrefetchQueryBuilder::default()
        .query(Query::new_nearest(request.dense.clone()))
        .limit(request.prefetch_limit);
    if let Some(using) = &request.dense_using {
        dense = dense.using(using.clone());
    }
    let sparse = PrefetchQueryBuilder::default()
        .using(request.sparse_using.clone())
        .query(VectorInput::new_sparse(
            request.sparse.indices.clone(),
            request.sparse.values.clone(),
        ))
        .limit(request.prefetch_limit);
    decorate(
        &request.scope,
        QueryPointsBuilder::new(request.scope.collection.clone())
            .query(Query::new_fusion(request.fusion))
            .add_prefetch(dense)
            .add_prefetch(sparse),
    )
    .offset(offset)
}

pub fn recommend_query(request: &RecommendRequest) -> QueryPointsBuilder {
    let mut recommend = RecommendInputBuilder::default().strategy(request.strategy as i32);
    for id in &request.positive_ids {
        recommend = recommend.add_positive(id.clone());
    }
    for id in &request.negative_ids {
        recommend = recommend.add_negative(id.clone());
    }
    if let Some(vector) = &request.positive_vector {
        recommend = recommend.add_positive(vector.clone());
    }
    if let Some(vector) = &request.negative_vector {
        recommend = recommend.add_negative(vector.clone());
    }
    decorate(
        &request.scope,
        QueryPointsBuilder::new(request.scope.collection.clone())
            .query(Query::new_recommend(recommend)),
    )
}

pub fn discover_query(request: &DiscoverRequest) -> QueryPointsBuilder {
    let mut context = ContextInputBuilder::default();
    for (positive, negative) in &request.pairs {
        context = context.add_pair(positive.clone(), negative.clone());
    }
    let query = match &request.target {
        Some(target) => Query::new_discover(DiscoverInputBuilder::new(target.clone(), context)),
        None => Query::new_context(context),
    };
    decorate(
        &request.scope,
        QueryPointsBuilder::new(request.scope.collection.clone()).query(query),
    )
}

pub fn parse_sparse(text: &str) -> Result<SparseVector, String> {
    let value: Value =
        serde_json::from_str(text).map_err(|err| format!("invalid sparse query JSON: {err}"))?;
    let object = value
        .as_object()
        .ok_or_else(|| "sparse query must be a JSON object with indices and values".to_string())?;
    let indices = u32_array(object.get("indices").ok_or("sparse query needs indices")?)?;
    let values = f32_array(object.get("values").ok_or("sparse query needs values")?)?;
    if indices.len() != values.len() {
        return Err(format!(
            "sparse query indices ({}) and values ({}) differ in length",
            indices.len(),
            values.len()
        ));
    }
    if indices.is_empty() {
        return Err("sparse query is empty".to_string());
    }
    Ok(SparseVector { indices, values })
}

pub fn parse_fusion(text: &str) -> Result<Fusion, String> {
    match text.trim().to_ascii_lowercase().as_str() {
        "" | "rrf" => Ok(Fusion::Rrf),
        "dbsf" => Ok(Fusion::Dbsf),
        other => Err(format!("fusion must be rrf or dbsf, got {other}")),
    }
}

pub fn parse_strategy(text: &str) -> Result<RecommendStrategy, String> {
    match text.trim().to_ascii_lowercase().as_str() {
        "" | "average" | "average_vector" => Ok(RecommendStrategy::AverageVector),
        "best" | "best_score" => Ok(RecommendStrategy::BestScore),
        "sum" | "sum_scores" => Ok(RecommendStrategy::SumScores),
        other => Err(format!(
            "strategy must be average, best, or sum, got {other}"
        )),
    }
}

pub fn parse_payload_fields(text: &str) -> Result<Option<Vec<String>>, String> {
    let text = text.trim();
    if text.is_empty() {
        return Ok(None);
    }
    if text.starts_with('[') {
        let value: Value = serde_json::from_str(text)
            .map_err(|err| format!("invalid payload_fields JSON: {err}"))?;
        let array = value
            .as_array()
            .ok_or_else(|| "payload_fields must be a JSON array of strings".to_string())?;
        let mut fields = Vec::with_capacity(array.len());
        for item in array {
            let field = item
                .as_str()
                .ok_or_else(|| "payload_fields entries must be strings".to_string())?;
            fields.push(field.to_string());
        }
        return Ok(Some(fields));
    }
    Ok(Some(
        text.split(',')
            .map(str::trim)
            .filter(|field| !field.is_empty())
            .map(str::to_string)
            .collect(),
    ))
}

pub fn parse_id_list(text: &str) -> Result<Vec<PointId>, String> {
    let text = text.trim();
    if text.is_empty() {
        return Ok(Vec::new());
    }
    let value: Value =
        serde_json::from_str(text).map_err(|err| format!("invalid id list JSON: {err}"))?;
    let array = value
        .as_array()
        .ok_or_else(|| "id list must be a JSON array".to_string())?;
    array.iter().map(point_id_value).collect()
}

pub fn point_id_text(text: &str) -> Result<PointId, String> {
    if text.is_empty() {
        return Err("point id is empty".to_string());
    }
    if text.chars().all(|ch| ch.is_ascii_digit())
        && let Ok(number) = text.parse::<u64>()
    {
        return Ok(PointId::from(number));
    }
    Ok(PointId::from(text.to_string()))
}

pub fn parse_context_pairs(text: &str) -> Result<Vec<(VectorInput, VectorInput)>, String> {
    let value: Value =
        serde_json::from_str(text).map_err(|err| format!("invalid context JSON: {err}"))?;
    let array = value
        .as_array()
        .ok_or_else(|| "context must be a JSON array of positive and negative pairs".to_string())?;
    if array.is_empty() {
        return Err("context has no pairs".to_string());
    }
    array
        .iter()
        .map(|item| {
            let object = item
                .as_object()
                .ok_or_else(|| "context pair must be a JSON object".to_string())?;
            let positive = example_input(
                object
                    .get("positive")
                    .ok_or("context pair needs positive")?,
            )?;
            let negative = example_input(
                object
                    .get("negative")
                    .ok_or("context pair needs negative")?,
            )?;
            Ok((positive, negative))
        })
        .collect()
}

fn point_id_value(value: &Value) -> Result<PointId, String> {
    match value {
        Value::Number(number) => {
            let id = number
                .as_u64()
                .ok_or_else(|| "point ids must be non-negative integers or strings".to_string())?;
            Ok(PointId::from(id))
        }
        Value::String(text) => point_id_text(text),
        _ => Err("point ids must be non-negative integers or strings".to_string()),
    }
}

fn example_input(value: &Value) -> Result<VectorInput, String> {
    match value {
        Value::Array(items) => {
            let vector = f32_array(value)?;
            if vector.is_empty() {
                return Err("context vector is empty".to_string());
            }
            let _ = items;
            Ok(vector.into())
        }
        other => Ok(point_id_value(other)?.into()),
    }
}

fn u32_array(value: &Value) -> Result<Vec<u32>, String> {
    let array = value
        .as_array()
        .ok_or_else(|| "indices must be an array".to_string())?;
    array
        .iter()
        .map(|item| {
            let number = item
                .as_u64()
                .ok_or_else(|| "sparse indices must be non-negative integers".to_string())?;
            u32::try_from(number).map_err(|_| format!("sparse index {number} does not fit in u32"))
        })
        .collect()
}

fn f32_array(value: &Value) -> Result<Vec<f32>, String> {
    let array = value
        .as_array()
        .ok_or_else(|| "values must be an array of numbers".to_string())?;
    array
        .iter()
        .enumerate()
        .map(|(index, item)| {
            let number = item
                .as_f64()
                .ok_or_else(|| format!("value at index {index} is not a number"))?;
            if !number.is_finite() {
                return Err(format!("value at index {index} is not finite"));
            }
            let value = number as f32;
            if !value.is_finite() {
                return Err(format!("value at index {index} is out of range for f32"));
            }
            Ok(value)
        })
        .collect()
}

fn filter_from_value(value: &Value) -> Result<Filter, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "filter must be a JSON object".to_string())?;
    let mut filter = Filter::default();
    for (key, child) in object {
        match key.as_str() {
            "must" => filter.must = condition_list(child)?,
            "should" => filter.should = condition_list(child)?,
            "must_not" => filter.must_not = condition_list(child)?,
            "min_should" => filter.min_should = Some(min_should(child)?),
            other => return Err(format!("unsupported filter field {other}")),
        }
    }
    Ok(filter)
}

fn min_should(value: &Value) -> Result<MinShould, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "min_should must be a JSON object".to_string())?;
    let mut conditions = Vec::new();
    let mut min_count = None;
    for (key, child) in object {
        match key.as_str() {
            "conditions" => conditions = condition_list(child)?,
            "min_count" => {
                min_count = Some(
                    child
                        .as_u64()
                        .ok_or_else(|| "min_should.min_count must be an integer".to_string())?,
                );
            }
            other => return Err(format!("unsupported min_should field {other}")),
        }
    }
    let min_count = min_count.ok_or_else(|| "min_should.min_count is required".to_string())?;
    Ok(MinShould {
        conditions,
        min_count,
    })
}

fn condition_list(value: &Value) -> Result<Vec<Condition>, String> {
    let array = value
        .as_array()
        .ok_or_else(|| "filter conditions must be a JSON array".to_string())?;
    array.iter().map(condition_from_value).collect()
}

fn condition_from_value(value: &Value) -> Result<Condition, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "filter condition must be a JSON object".to_string())?;
    if object.contains_key("must")
        || object.contains_key("should")
        || object.contains_key("must_not")
        || object.contains_key("min_should")
    {
        return Ok(Condition::from(filter_from_value(value)?));
    }
    if let Some(body) = object.get("is_empty") {
        return Ok(Condition::is_empty(object_key(body, "is_empty")?));
    }
    if let Some(body) = object.get("is_null") {
        return Ok(Condition::is_null(object_key(body, "is_null")?));
    }
    if let Some(body) = object.get("has_id") {
        return Ok(Condition::has_id(point_ids(body)?));
    }
    if let Some(body) = object.get("has_vector") {
        let name = body
            .as_str()
            .ok_or_else(|| "has_vector must be a string".to_string())?;
        return Ok(Condition::has_vector(name));
    }
    if let Some(body) = object.get("nested") {
        return nested_condition(body);
    }
    if object.contains_key("key") {
        return field_condition(object);
    }
    Err("unsupported filter condition".to_string())
}

fn object_key(value: &Value, name: &str) -> Result<String, String> {
    value
        .as_object()
        .and_then(|object| object.get("key"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| format!("{name} needs a string key"))
}

fn nested_condition(value: &Value) -> Result<Condition, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "nested condition must be a JSON object".to_string())?;
    let key = object
        .get("key")
        .and_then(Value::as_str)
        .ok_or_else(|| "nested condition needs a string key".to_string())?;
    let filter = object
        .get("filter")
        .ok_or_else(|| "nested condition needs a filter".to_string())?;
    Ok(Condition::nested(key, filter_from_value(filter)?))
}

fn field_condition(object: &serde_json::Map<String, Value>) -> Result<Condition, String> {
    let key = object
        .get("key")
        .and_then(Value::as_str)
        .ok_or_else(|| "field condition needs a string key".to_string())?;
    let operators: Vec<&String> = object
        .keys()
        .filter(|name| name.as_str() != "key")
        .collect();
    let [operator] = operators.as_slice() else {
        return Err(format!("field {key} needs exactly one condition"));
    };
    match operator.as_str() {
        "match" => match_condition(key, &object["match"]),
        "range" => range_condition(key, &object["range"]),
        other => Err(format!("unsupported field condition {other} on {key}")),
    }
}

fn match_condition(key: &str, value: &Value) -> Result<Condition, String> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("match on {key} must be a JSON object"))?;
    let operators: Vec<&String> = object.keys().collect();
    let [operator] = operators.as_slice() else {
        return Err(format!("match on {key} needs exactly one matcher"));
    };
    let body = &object[*operator];
    match operator.as_str() {
        "value" => match_value(key, body),
        "any" => match_any(key, body),
        "text" => Ok(Condition::matches_text(
            key,
            json_string(body, "match.text")?,
        )),
        "phrase" => Ok(Condition::matches_phrase(
            key,
            json_string(body, "match.phrase")?,
        )),
        "text_any" => Ok(Condition::matches_text_any(
            key,
            json_string(body, "match.text_any")?,
        )),
        "prefix" => Ok(Condition::matches_prefix(
            key,
            json_string(body, "match.prefix")?,
        )),
        "substring" => Ok(Condition::matches_substring(
            key,
            json_string(body, "match.substring")?,
        )),
        other => Err(format!("unsupported match {other} on {key}")),
    }
}

fn match_value(key: &str, value: &Value) -> Result<Condition, String> {
    match value {
        Value::String(text) => Ok(Condition::matches(key, text.clone())),
        Value::Bool(flag) => Ok(Condition::matches(key, *flag)),
        Value::Number(number) => {
            let integer = number.as_i64().ok_or_else(|| {
                format!("match value for {key} must be an integer, string, or bool")
            })?;
            Ok(Condition::matches(key, integer))
        }
        _ => Err(format!(
            "match value for {key} must be an integer, string, or bool"
        )),
    }
}

fn match_any(key: &str, value: &Value) -> Result<Condition, String> {
    let array = value
        .as_array()
        .ok_or_else(|| format!("match.any on {key} must be an array"))?;
    if array.iter().all(Value::is_string) {
        let values = array
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect::<Vec<_>>();
        return Ok(Condition::matches(key, values));
    }
    if array.iter().all(|item| item.as_i64().is_some()) {
        let values = array.iter().filter_map(Value::as_i64).collect::<Vec<_>>();
        return Ok(Condition::matches(key, values));
    }
    Err(format!(
        "match.any on {key} must be all strings or all integers"
    ))
}

fn range_condition(key: &str, value: &Value) -> Result<Condition, String> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("range on {key} must be a JSON object"))?;
    let mut range = Range::default();
    for (name, child) in object {
        let number = child
            .as_f64()
            .ok_or_else(|| format!("range {name} on {key} must be a number"))?;
        match name.as_str() {
            "gt" => range.gt = Some(number),
            "gte" => range.gte = Some(number),
            "lt" => range.lt = Some(number),
            "lte" => range.lte = Some(number),
            other => return Err(format!("unsupported range field {other} on {key}")),
        }
    }
    Ok(Condition::range(key, range))
}

fn json_string(value: &Value, name: &str) -> Result<String, String> {
    value
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| format!("{name} must be a string"))
}

fn point_ids(value: &Value) -> Result<Vec<PointId>, String> {
    let array = value
        .as_array()
        .ok_or_else(|| "has_id must be an array".to_string())?;
    array
        .iter()
        .map(|item| match item {
            Value::Number(number) => {
                let id = number.as_u64().ok_or_else(|| {
                    "has_id values must be non-negative integers or UUID strings".to_string()
                })?;
                Ok(PointId::from(id))
            }
            Value::String(text) => Ok(PointId::from(text.clone())),
            _ => Err("has_id values must be non-negative integers or UUID strings".to_string()),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use qdrant_client::qdrant::Bm25Params;
    use qdrant_client::qdrant::FieldCondition;
    use qdrant_client::qdrant::condition::ConditionOneOf;

    fn field(condition: &Condition) -> &FieldCondition {
        match &condition.condition_one_of {
            Some(ConditionOneOf::Field(field)) => field,
            other => panic!("expected a field condition, got {other:?}"),
        }
    }

    #[test]
    fn parses_year_range_from_the_plan() {
        let filter = parse_filter(r#"{"must":[{"key":"year","range":{"gte":2024}}]}"#)
            .unwrap()
            .unwrap();
        assert_eq!(filter.must.len(), 1);
        let field = field(&filter.must[0]);
        assert_eq!(field.key, "year");
        assert_eq!(
            field.range.as_ref().and_then(|range| range.gte),
            Some(2024.0)
        );
    }

    #[test]
    fn rejects_unknown_filter_field() {
        let error = parse_filter(r#"{"must_all":[]}"#).unwrap_err();
        assert!(error.contains("must_all"), "{error}");
    }

    #[test]
    fn rejects_invalid_json() {
        let error = parse_filter("{").unwrap_err();
        assert!(error.contains("invalid filter JSON"), "{error}");
    }

    #[test]
    fn empty_filter_text_is_absent() {
        assert!(parse_filter("  ").unwrap().is_none());
    }

    #[test]
    fn vector_rejects_null_and_keeps_order() {
        assert!(dense_vector(vec![]).is_err());
        assert!(
            dense_vector(vec![Some(1.0), None])
                .unwrap_err()
                .contains("index 1")
        );
        assert_eq!(
            dense_vector(vec![Some(0.5), Some(-1.0)]).unwrap(),
            vec![0.5, -1.0]
        );
    }

    #[test]
    fn text_page_names_the_field_and_sends_text() {
        let request = TextRequest {
            scope: Scope {
                collection: "products".to_string(),
                limit: 10,
                filter: None,
                using: Some("text".to_string()),
                score_threshold: None,
                with_payload: true,
                payload_fields: None,
                endpoint: crate::client::Endpoint {
                    url: DEFAULT_URL.to_string(),
                    api_key: None,
                    timeout_secs: DEFAULT_TIMEOUT_SECS,
                },
            },
            text: "cashew granola bars".to_string(),
            k: Some(1.5),
            b: Some(0.5),
        };
        let query = text_page(&request, 0, 10).build();
        assert_eq!(query.collection_name, "products");
        assert_eq!(query.using.as_deref(), Some("text"));
        assert_eq!(query.limit, Some(10));
        let Some(query::Variant::Text(text)) = query.query.and_then(|query| query.variant) else {
            panic!("expected a text query");
        };
        assert_eq!(text.query, "cashew granola bars");
        assert_eq!(
            text.scoring,
            Some(qdrant_client::qdrant::text_query::Scoring::Bm25(
                Bm25Params {
                    k: Some(1.5),
                    b: Some(0.5),
                }
            ))
        );
    }

    fn scope(limit: u64) -> Scope {
        Scope {
            collection: "products".to_string(),
            limit,
            filter: None,
            using: None,
            score_threshold: None,
            with_payload: true,
            payload_fields: None,
            endpoint: crate::client::Endpoint {
                url: DEFAULT_URL.to_string(),
                api_key: None,
                timeout_secs: DEFAULT_TIMEOUT_SECS,
            },
        }
    }

    #[test]
    fn batch_dense_queries_parse_in_order() {
        let queries =
            parse_batch_queries("[[0.1,0.2],[0.3,0.4]]", BatchMode::Dense, None, None).unwrap();
        assert_eq!(queries.len(), 2);
        assert!(matches!(
            queries[1].variant.as_ref(),
            Some(query::Variant::Nearest(_))
        ));

        let mut scope = scope(3);
        scope.using = Some("dense".to_string());
        scope.filter = parse_filter(r#"{"must":[{"key":"year","range":{"gte":2024}}]}"#).unwrap();
        let points = batch_queries(&scope, &queries);
        assert_eq!(points.len(), 2);
        assert!(points.iter().all(|point| {
            point.filter.is_some()
                && point.limit == Some(3)
                && point.using.as_deref() == Some("dense")
        }));
    }

    #[test]
    fn batch_text_queries_carry_bm25_weights() {
        let queries = parse_batch_queries(
            r#"["granola","cashew"]"#,
            BatchMode::Text,
            Some(1.5),
            Some(0.5),
        )
        .unwrap();
        let Some(query::Variant::Text(text)) = queries[0].variant.as_ref() else {
            panic!("expected a text query");
        };
        assert_eq!(text.query, "granola");
        assert_eq!(
            text.scoring,
            Some(qdrant_client::qdrant::text_query::Scoring::Bm25(
                Bm25Params {
                    k: Some(1.5),
                    b: Some(0.5),
                }
            ))
        );
    }

    #[test]
    fn batch_rejects_bad_queries() {
        assert!(
            parse_batch_queries("[]", BatchMode::Dense, None, None)
                .unwrap_err()
                .contains("empty")
        );
        assert!(
            parse_batch_queries("{", BatchMode::Dense, None, None)
                .unwrap_err()
                .contains("invalid queries JSON")
        );
        assert!(parse_batch_queries(r#"[[1.0]]"#, BatchMode::Text, None, None).is_err());
        assert!(
            parse_batch_queries(r#"["a"," "]"#, BatchMode::Text, None, None)
                .unwrap_err()
                .contains("query 1")
        );
        assert!(
            parse_batch_queries("[[1e400]]", BatchMode::Dense, None, None)
                .unwrap_err()
                .contains("invalid queries JSON")
        );
        assert!(
            parse_batch_queries("[[1e300]]", BatchMode::Dense, None, None)
                .unwrap_err()
                .contains("out of range")
        );
        assert!(parse_batch_mode("sparse").unwrap_err().contains("sparse"));
    }

    #[test]
    fn batch_limit_rejects_more_than_one_page() {
        assert_eq!(batch_limit(PAGE_SIZE).unwrap(), PAGE_SIZE);
        assert!(batch_limit(130).unwrap_err().contains("130"));
    }

    #[test]
    fn hybrid_text_sends_dense_and_text_prefetches() {
        let request = HybridTextRequest {
            scope: scope(10),
            dense: vec![0.1, 0.2],
            dense_using: Some("dense".to_string()),
            text: "granola".to_string(),
            text_field: "text".to_string(),
            fusion: Fusion::Rrf,
            prefetch_limit: 100,
        };
        let query = hybrid_text_query(&request, 128).build();
        assert_eq!(query.prefetch.len(), 2);
        assert_eq!(query.prefetch[0].using.as_deref(), Some("dense"));
        assert_eq!(query.prefetch[1].using.as_deref(), Some("text"));
        assert_eq!(query.prefetch[1].limit, Some(100));
        assert_eq!(query.offset, Some(128));
        assert!(matches!(
            query.prefetch[1]
                .query
                .as_ref()
                .and_then(|query| query.variant.as_ref()),
            Some(query::Variant::Text(_))
        ));
        assert!(matches!(
            query.query.and_then(|query| query.variant),
            Some(query::Variant::Fusion(_))
        ));
    }

    #[test]
    fn nearest_query_keeps_limit_filter_and_named_vector() {
        let request = QueryRequest {
            scope: Scope {
                collection: "articles".to_string(),
                limit: 50,
                filter: parse_filter(r#"{"must":[{"key":"year","range":{"gte":2024}}]}"#).unwrap(),
                using: Some("dense".to_string()),
                score_threshold: Some(0.75),
                with_payload: true,
                payload_fields: None,
                endpoint: crate::client::Endpoint {
                    url: DEFAULT_URL.to_string(),
                    api_key: None,
                    timeout_secs: DEFAULT_TIMEOUT_SECS,
                },
            },
            vector: vec![0.1, 0.2],
        };
        let query = nearest_page(&request, 0, request.scope.limit).build();
        assert_eq!(query.collection_name, "articles");
        assert_eq!(query.limit, Some(50));
        assert_eq!(query.using.as_deref(), Some("dense"));
        assert_eq!(query.score_threshold, Some(0.75));
        assert!(query.filter.is_some());
    }
}
