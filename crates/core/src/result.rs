//! `ScoredPoint` to the stable result row: id, score, payload JSON.

use std::collections::HashMap;

use qdrant_client::qdrant::value::Kind;
use qdrant_client::qdrant::{PointId, ScoredPoint, Value};
use serde_json::json;

#[derive(Clone, Debug, PartialEq)]
pub struct QueryRow {
    pub id: String,
    pub score: f64,
    pub payload: String,
}

pub fn scored_point_row(point: ScoredPoint) -> Result<QueryRow, String> {
    Ok(QueryRow {
        id: point_id(&point.id)?,
        score: f64::from(point.score),
        payload: payload_json(&point.payload)?,
    })
}

fn point_id(id: &Option<PointId>) -> Result<String, String> {
    let Some(id) = id else {
        return Err("qdrant point is missing an id".to_string());
    };
    match id.point_id_options {
        Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(number)) => {
            Ok(number.to_string())
        }
        Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(ref text)) => Ok(text.clone()),
        None => Err("qdrant point is missing an id".to_string()),
    }
}

fn payload_json(payload: &HashMap<String, Value>) -> Result<String, String> {
    let mut object = serde_json::Map::new();
    for (key, value) in payload {
        object.insert(key.clone(), json_value(value)?);
    }
    serde_json::to_string(&object).map_err(|err| format!("payload JSON: {err}"))
}

fn json_value(value: &Value) -> Result<serde_json::Value, String> {
    let Some(kind) = &value.kind else {
        return Ok(serde_json::Value::Null);
    };
    Ok(match kind {
        Kind::NullValue(_) => serde_json::Value::Null,
        Kind::DoubleValue(number) => json!(number),
        Kind::IntegerValue(number) => json!(number),
        Kind::StringValue(text) => json!(text),
        Kind::BoolValue(flag) => json!(flag),
        Kind::StructValue(fields) => {
            let mut object = serde_json::Map::new();
            for (key, child) in &fields.fields {
                object.insert(key.clone(), json_value(child)?);
            }
            serde_json::Value::Object(object)
        }
        Kind::ListValue(list) => {
            let items = list
                .values
                .iter()
                .map(json_value)
                .collect::<Result<Vec<_>, _>>()?;
            serde_json::Value::Array(items)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use qdrant_client::qdrant::point_id::PointIdOptions;

    fn string_value(text: &str) -> Value {
        Value {
            kind: Some(Kind::StringValue(text.to_string())),
        }
    }

    #[test]
    fn maps_numeric_id_score_and_payload() {
        let point = ScoredPoint {
            id: Some(PointId {
                point_id_options: Some(PointIdOptions::Num(7)),
            }),
            payload: HashMap::from([("color".to_string(), string_value("red"))]),
            score: 0.5,
            ..ScoredPoint::default()
        };
        let row = scored_point_row(point).unwrap();
        assert_eq!(row.id, "7");
        assert_eq!(row.score, 0.5);
        assert_eq!(row.payload, r#"{"color":"red"}"#);
    }

    #[test]
    fn maps_uuid_id() {
        let point = ScoredPoint {
            id: Some(PointId {
                point_id_options: Some(PointIdOptions::Uuid("abc".to_string())),
            }),
            score: 1.0,
            ..ScoredPoint::default()
        };
        assert_eq!(scored_point_row(point).unwrap().id, "abc");
    }
}
