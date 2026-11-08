//! `qdrant_upsert`: insert or replace points. One gRPC call for the whole batch.

use std::sync::Mutex;

use duckdb_neo::{
    DuckDBType, FromValue, Result,
    bind_arguments::BindArgument,
    connection::Context,
    data_chunk::DataChunkRef,
    data_chunk::VectorCollection,
    error::Error,
    extension::Extension,
    scalar::{ScalarCallbacks, ScalarFunctionBuilder},
    signature::{FunctionDocumentationBuilder, FunctionParameter, SignatureBuilder},
    table_function::{
        BindFunctionHandle, ExecColumnInfo, InitColumnData, TableFunctionBuilder,
        TableFunctionCallbacks,
    },
    types::List,
};
use qdrant_client::Payload;
use qdrant_client::qdrant::{PointStruct, UpdateStatus, UpsertPointsBuilder};

use crate::CATEGORY;
use crate::args::{failure, invalid, optional_string, required_string, string_list};
use crate::client::Endpoint;
use crate::client::{client, retry};
use crate::emit::Fetched;
use crate::error::context;
use crate::settings::endpoint;
use crate::types::{dense_vector, point_id_text};

#[derive(Clone)]
pub(crate) struct UpsertBind {
    pub(crate) collection: String,
    pub(crate) ids: Vec<String>,
    pub(crate) vectors: Vec<Vec<f32>>,
    pub(crate) payloads: Vec<String>,
    pub(crate) using: Option<String>,
    pub(crate) endpoint: Endpoint,
}

#[derive(Clone)]
struct UpsertRow {
    id: String,
    status: String,
}

struct UpsertTable;
struct UpsertScalar;

impl TableFunctionCallbacks for UpsertTable {
    type GlobalState = Mutex<Fetched<UpsertRow>>;
    type LocalState = ();
    type BindData = UpsertBind;

    fn init_global_state(
        &self,
        _bind_data: &Self::BindData,
        _context: &Context,
        _column_data: InitColumnData<'_>,
    ) -> Result<(Option<Self::GlobalState>, Option<usize>)> {
        Ok((Some(Mutex::new(Fetched::new())), Some(1)))
    }

    fn init_local_state(
        &self,
        _bind_data: &Self::BindData,
        _global_state: Option<&Self::GlobalState>,
        _context: &Context,
        _column_data: InitColumnData<'_>,
    ) -> Result<Option<Self::LocalState>> {
        Ok(None)
    }

    fn bind(
        &self,
        context: &Context,
        arguments: Vec<BindArgument>,
        bind_handle: BindFunctionHandle<'_>,
    ) -> Result<Self::BindData> {
        bind_handle.add_result_column("id", &String::logical_type(context)?)?;
        bind_handle.add_result_column("status", &String::logical_type(context)?)?;
        let collection = required_string(&arguments, 0, "collection")?;
        let ids = string_list(&arguments, 1, "ids")?;
        let vectors = vector_matrix(&arguments, 2)?;
        let mut payloads = string_list(&arguments, 3, "payloads")?;
        if ids.len() != vectors.len() {
            return Err(Error::invalid_input(format!(
                "ids ({}) and vectors ({}) differ in length",
                ids.len(),
                vectors.len()
            )));
        }
        if payloads.is_empty() {
            payloads = vec!["{}".to_string(); ids.len()];
        } else if payloads.len() != ids.len() {
            return Err(Error::invalid_input(format!(
                "ids ({}) and payloads ({}) differ in length",
                ids.len(),
                payloads.len()
            )));
        }
        let using_text = optional_string(&arguments, 4)?;
        let using = if using_text.is_empty() {
            None
        } else {
            Some(using_text)
        };
        let endpoint = endpoint(
            context,
            &optional_string(&arguments, 5)?,
            &optional_string(&arguments, 6)?,
            match crate::args::optional_i64(&arguments, 7)? {
                Some(0) => None,
                other => other,
            },
        )?;
        Ok(UpsertBind {
            collection,
            ids,
            vectors,
            payloads,
            using,
            endpoint,
        })
    }

    fn exec(
        &self,
        bind_data: &Self::BindData,
        global_state: Option<&Self::GlobalState>,
        _local_state: Option<&mut Self::LocalState>,
        _context: &Context,
        mut output: DataChunkRef<'_>,
        _column_info: ExecColumnInfo<'_>,
    ) -> Result<()> {
        let state =
            global_state.ok_or_else(|| Error::internal("qdrant_upsert: missing global state"))?;
        let mut scan = state
            .lock()
            .map_err(|_| Error::internal("qdrant_upsert: scan lock poisoned"))?;
        scan.load(|| {
            let status = upsert_batch(bind_data).map_err(failure)?;
            Ok(bind_data
                .ids
                .iter()
                .map(|id| UpsertRow {
                    id: id.clone(),
                    status: status.clone(),
                })
                .collect())
        })?;
        let capacity = output.capacity()?;
        let (start, end) = scan.window(capacity);
        let count = end - start;
        {
            let mut id = output.get_vector_at_mut::<String>(0)?;
            id.set_size(count)?;
            let mut writer = id.writer()?;
            for index in 0..count {
                writer.set(index, Some(&scan.rows[start + index].id))?;
            }
        }
        {
            let mut status = output.get_vector_at_mut::<String>(1)?;
            status.set_size(count)?;
            let mut writer = status.writer()?;
            for index in 0..count {
                writer.set(index, Some(&scan.rows[start + index].status))?;
            }
        }
        scan.next = end;
        Ok(())
    }
}

pub fn upsert_batch(bind: &UpsertBind) -> std::result::Result<String, String> {
    let points = bind
        .ids
        .iter()
        .zip(bind.vectors.iter())
        .zip(bind.payloads.iter())
        .map(|((id, vector), payload)| point(id, vector, payload, bind.using.as_deref()))
        .collect::<std::result::Result<Vec<_>, String>>()?;
    let qdrant = client(&bind.endpoint)?;
    let request = UpsertPointsBuilder::new(bind.collection.clone(), points)
        .wait(true)
        .build();
    let response = retry(|| {
        let qdrant = qdrant.clone();
        let request = request.clone();
        async move { qdrant.upsert_points(request).await }
    })
    .map_err(|err| context("qdrant_upsert", &bind.collection, err))?;
    Ok(response
        .result
        .and_then(|result| UpdateStatus::try_from(result.status).ok())
        .map(|status| status.as_str_name().to_string())
        .unwrap_or_else(|| "unknown".to_string()))
}

fn point(
    id: &str,
    vector: &[f32],
    payload: &str,
    using: Option<&str>,
) -> std::result::Result<PointStruct, String> {
    let payload_json: serde_json::Value = serde_json::from_str(payload)
        .map_err(|err| format!("invalid payload JSON for {id}: {err}"))?;
    let payload =
        Payload::try_from(payload_json).map_err(|err| format!("payload for {id}: {err}"))?;
    let id = point_id_text(id)?;
    let vectors: qdrant_client::qdrant::Vectors = match using {
        Some(name) => qdrant_client::qdrant::NamedVectors::default()
            .add_vector(name, vector.to_vec())
            .into(),
        None => vector.to_vec().into(),
    };
    Ok(PointStruct::new(id, vectors, payload))
}

fn vector_matrix(arguments: &[BindArgument], index: usize) -> Result<Vec<Vec<f32>>> {
    let arg = arguments
        .get(index)
        .ok_or_else(|| Error::invalid_input("missing argument vectors"))?;
    let Some(value) = &arg.value else {
        return Err(Error::invalid_input("vectors is NULL"));
    };
    let rows = <List<List<f64>>>::from_value(value)?.unwrap_or_default().0;
    let mut vectors = Vec::with_capacity(rows.len());
    for (offset, row) in rows.into_iter().enumerate() {
        let row = row.ok_or_else(|| {
            Error::invalid_input(format!("vectors contains NULL at index {offset}"))
        })?;
        vectors.push(dense_vector(row.0).map_err(invalid)?);
    }
    Ok(vectors)
}

impl ScalarCallbacks for UpsertScalar {
    type BindData = ();
    type InitData = ();

    fn exec(
        &self,
        _bind_data: Option<&Self::BindData>,
        _init_data: Option<&mut Self::InitData>,
        context: &Context,
        input: &mut VectorCollection,
        output: duckdb_neo::vector::Vector<'_, duckdb_neo::vector::Unknown>,
    ) -> Result<()> {
        let rows = input.row_count()?;
        let mut output = output.cast::<String>()?;
        output.set_size(rows)?;
        let mut writer = output.writer()?;
        let collections = input.get_vector_at::<String>(0)?;
        let ids = input.get_vector_at::<String>(1)?;
        let vectors = input.get_vector_at::<List<f64>>(2)?;
        let payloads = input.get_vector_at::<String>(3)?;
        let names = input.get_vector_at::<String>(4)?;
        let collection_reader = collections.reader()?;
        let id_reader = ids.reader()?;
        let vector_reader = vectors.reader()?;
        let payload_reader = payloads.reader()?;
        let name_reader = names.reader()?;
        let endpoint = endpoint(context, "", "", None)?;
        for index in 0..rows {
            let collection = collection_reader
                .get(index)?
                .ok_or_else(|| Error::invalid_input("collection is NULL"))?
                .to_string();
            let id = id_reader
                .get(index)?
                .ok_or_else(|| Error::invalid_input("id is NULL"))?
                .to_string();
            let vector = vector_reader
                .get(index)?
                .ok_or_else(|| Error::invalid_input("vector is NULL"))?;
            let mut values = Vec::with_capacity(vector.len());
            for item in vector.iter() {
                let number = item.ok_or_else(|| Error::invalid_input("vector contains NULL"))?;
                values.push(*number);
            }
            let payload = payload_reader
                .get(index)?
                .map(|text| text.to_string())
                .unwrap_or_else(|| "{}".to_string());
            let using = name_reader
                .get(index)?
                .map(|text| text.to_string())
                .filter(|text| !text.is_empty());
            let vector = dense_vector(values.into_iter().map(Some).collect()).map_err(invalid)?;
            let status = upsert_batch(&UpsertBind {
                collection,
                ids: vec![id],
                vectors: vec![vector],
                payloads: vec![payload],
                using,
                endpoint: endpoint.clone(),
            })
            .map_err(failure)?;
            writer.set(index, Some(&status))?;
        }
        Ok(())
    }
}

pub fn register(extension: &Extension) -> Result<()> {
    TableFunctionBuilder::new(
        "qdrant_upsert",
        SignatureBuilder::new()
            .param(FunctionParameter::new::<String>("collection"))
            .param(FunctionParameter::new::<List<String>>("ids"))
            .param(FunctionParameter::new::<List<List<f64>>>("vectors"))
            .param(FunctionParameter::with_default(
                "payloads",
                List(Vec::<Option<String>>::new()),
            ))
            .param(FunctionParameter::with_default("using", String::new()))
            .param(FunctionParameter::with_default("url", String::new()))
            .param(FunctionParameter::with_default("api_key", String::new()))
            .param(FunctionParameter::with_default("timeout_secs", 0_i64))
            .documentation(
                FunctionDocumentationBuilder::new()
                    .description("Upsert a batch of points. Returns id and Qdrant update status.")
                    .add_example(
                        "Two points",
                        "SELECT * FROM qdrant_upsert(collection := 'articles', ids := ['1','2'], vectors := [[0.1, 0.2], [0.3, 0.4]]::DOUBLE[][])",
                    )
                    .add_category(CATEGORY),
            ),
        UpsertTable,
    )
    .register(extension)?;

    ScalarFunctionBuilder::new(
        "qdrant_upsert_one",
        SignatureBuilder::new()
            .param(FunctionParameter::new::<String>("collection"))
            .param(FunctionParameter::new::<String>("id"))
            .param(FunctionParameter::new::<List<f64>>("vector"))
            .param(FunctionParameter::with_default("payload", "{}".to_string()))
            .param(FunctionParameter::with_default("using", String::new()))
            .return_type(String::logical_type(extension)?)
            .documentation(
                FunctionDocumentationBuilder::new()
                    .description("Upsert one point and return the Qdrant update status. Uses the session settings.")
                    .add_example(
                        "From a table",
                        "SELECT qdrant_upsert_one('articles', id, embedding, payload) FROM articles",
                    )
                    .add_category(CATEGORY),
            ),
        UpsertScalar,
    )
    .register(extension)
}
