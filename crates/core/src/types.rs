//! DuckDB values to Qdrant query types.
//!
//! Filters accept the Qdrant REST JSON shape for must, should, must_not,
//! min_should, field match, field range, has_id, is_empty, is_null, and nested.
//! Other condition kinds return an error instead of being dropped.

use qdrant_client::qdrant::{Condition, Filter, MinShould, PointId, QueryPointsBuilder, Range};
use serde_json::Value;

pub const DEFAULT_URL: &str = "http://localhost:6334";
pub const DEFAULT_LIMIT: u64 = 10;
pub const DEFAULT_TIMEOUT_SECS: u64 = 30;

#[derive(Clone)]
pub struct QueryRequest {
    pub collection: String,
    pub vector: Vec<f32>,
    pub limit: u64,
    pub filter: Option<Filter>,
    pub using: Option<String>,
    pub score_threshold: Option<f32>,
    pub url: String,
    pub api_key: Option<String>,
    pub timeout_secs: u64,
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

pub fn nearest_query(request: &QueryRequest) -> QueryPointsBuilder {
    let mut query = QueryPointsBuilder::new(request.collection.clone())
        .query(request.vector.clone())
        .limit(request.limit)
        .with_payload(true);
    if let Some(filter) = &request.filter {
        query = query.filter(filter.clone());
    }
    if let Some(using) = &request.using {
        query = query.using(using.clone());
    }
    if let Some(threshold) = request.score_threshold {
        query = query.score_threshold(threshold);
    }
    query
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
    fn nearest_query_keeps_limit_filter_and_named_vector() {
        let request = QueryRequest {
            collection: "articles".to_string(),
            vector: vec![0.1, 0.2],
            limit: 50,
            filter: parse_filter(r#"{"must":[{"key":"year","range":{"gte":2024}}]}"#).unwrap(),
            using: Some("dense".to_string()),
            score_threshold: Some(0.75),
            url: DEFAULT_URL.to_string(),
            api_key: None,
            timeout_secs: DEFAULT_TIMEOUT_SECS,
        };
        let query = nearest_query(&request).build();
        assert_eq!(query.collection_name, "articles");
        assert_eq!(query.limit, Some(50));
        assert_eq!(query.using.as_deref(), Some("dense"));
        assert_eq!(query.score_threshold, Some(0.75));
        assert!(query.filter.is_some());
    }
}
