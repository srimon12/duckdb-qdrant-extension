//! `qdrant_query`: nearest neighbours from one collection.

use duckdb_neo::{
    DuckDBType, FromValue, Result,
    bind_arguments::BindArgument,
    connection::Context,
    data_chunk::DataChunkRef,
    error::Error,
    extension::Extension,
    signature::{FunctionDocumentationBuilder, FunctionParameter, SignatureBuilder},
    table_function::{
        BindFunctionHandle, ExecColumnInfo, InitColumnData, TableFunctionBuilder,
        TableFunctionCallbacks,
    },
    types::List,
};
use std::sync::Mutex;

use crate::CATEGORY;
use crate::client::{block_on, client};
use crate::result::{QueryRow, scored_point_row};
use crate::types::{
    DEFAULT_LIMIT, DEFAULT_TIMEOUT_SECS, DEFAULT_URL, QueryRequest, dense_vector, nearest_query,
    parse_filter,
};

struct QueryTable;

struct Scan {
    rows: Vec<QueryRow>,
    next: usize,
    fetched: bool,
}

impl TableFunctionCallbacks for QueryTable {
    type GlobalState = Mutex<Scan>;
    type LocalState = ();
    type BindData = QueryRequest;

    fn init_global_state(
        &self,
        _bind_data: &Self::BindData,
        _context: &Context,
        _column_data: InitColumnData<'_>,
    ) -> Result<(Option<Self::GlobalState>, Option<usize>)> {
        Ok((
            Some(Mutex::new(Scan {
                rows: Vec::new(),
                next: 0,
                fetched: false,
            })),
            Some(1),
        ))
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
        bind_handle.add_result_column("score", &f64::logical_type(context)?)?;
        bind_handle.add_result_column("payload", &String::logical_type(context)?)?;

        let collection = required_string(&arguments, 0, "collection")?;
        if collection.is_empty() {
            return Err(Error::invalid_input("collection is empty"));
        }
        let vector = dense_vector(required_list(&arguments, 1)?).map_err(invalid)?;
        let limit = optional_i64(&arguments, 2)?.unwrap_or(DEFAULT_LIMIT as i64);
        if limit <= 0 {
            return Err(Error::invalid_input("limit must be greater than 0"));
        }
        let filter = parse_filter(&optional_string(&arguments, 3)?).map_err(invalid)?;
        let using = {
            let name = optional_string(&arguments, 4)?;
            if name.is_empty() { None } else { Some(name) }
        };
        let score_threshold = optional_f64(&arguments, 5)?
            .filter(|value| value.is_finite())
            .map(|value| value as f32);
        let url_arg = optional_string(&arguments, 6)?;
        let url = if url_arg.is_empty() {
            std::env::var("QDRANT_URL").unwrap_or_else(|_| DEFAULT_URL.to_string())
        } else {
            url_arg
        };
        let api_key_arg = optional_string(&arguments, 7)?;
        let api_key = if api_key_arg.is_empty() {
            std::env::var("QDRANT_API_KEY").ok()
        } else {
            Some(api_key_arg)
        };
        let timeout_secs = optional_i64(&arguments, 8)?.unwrap_or(DEFAULT_TIMEOUT_SECS as i64);
        if timeout_secs <= 0 {
            return Err(Error::invalid_input("timeout_secs must be greater than 0"));
        }

        Ok(QueryRequest {
            collection,
            vector,
            limit: limit as u64,
            filter,
            using,
            score_threshold,
            url,
            api_key,
            timeout_secs: timeout_secs as u64,
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
            global_state.ok_or_else(|| Error::internal("qdrant_query: missing global state"))?;
        let mut scan = state
            .lock()
            .map_err(|_| Error::internal("qdrant_query: scan lock poisoned"))?;
        if !scan.fetched {
            scan.rows = fetch(bind_data)?;
            scan.fetched = true;
        }

        let capacity = output.capacity()?;
        let start = scan.next;
        let end = (start + capacity).min(scan.rows.len());
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
            let mut score = output.get_vector_at_mut::<f64>(1)?;
            score.set_size(count)?;
            let mut writer = score.writer()?;
            for index in 0..count {
                writer.set(index, Some(scan.rows[start + index].score))?;
            }
        }
        {
            let mut payload = output.get_vector_at_mut::<String>(2)?;
            payload.set_size(count)?;
            let mut writer = payload.writer()?;
            for index in 0..count {
                writer.set(index, Some(&scan.rows[start + index].payload))?;
            }
        }
        scan.next = end;
        Ok(())
    }
}

fn fetch(request: &QueryRequest) -> Result<Vec<QueryRow>> {
    let qdrant = client(
        &request.url,
        request.api_key.as_deref(),
        request.timeout_secs,
    )
    .map_err(failure)?;
    let query = nearest_query(request);
    let response = block_on(qdrant.query(query)).map_err(|err| {
        failure(format!(
            "qdrant_query collection={} vector_len={} failed: {err}",
            request.collection,
            request.vector.len()
        ))
    })?;
    response
        .result
        .into_iter()
        .map(|point| scored_point_row(point).map_err(failure))
        .collect()
}

fn invalid(message: String) -> Error {
    Error::invalid_input(message)
}

fn failure(message: String) -> Error {
    Error::internal(message)
}

fn argument<'a>(
    arguments: &'a [BindArgument],
    index: usize,
    name: &str,
) -> Result<&'a BindArgument> {
    arguments
        .get(index)
        .ok_or_else(|| Error::invalid_input(format!("missing argument {name}")))
}

fn required_string(arguments: &[BindArgument], index: usize, name: &str) -> Result<String> {
    let arg = argument(arguments, index, name)?;
    let Some(value) = &arg.value else {
        return Err(Error::invalid_input(format!("{name} is NULL")));
    };
    String::from_value(value)?.ok_or_else(|| Error::invalid_input(format!("{name} is NULL")))
}

fn optional_string(arguments: &[BindArgument], index: usize) -> Result<String> {
    let Some(arg) = arguments.get(index) else {
        return Ok(String::new());
    };
    let Some(value) = &arg.value else {
        return Ok(String::new());
    };
    Ok(String::from_value(value)?.unwrap_or_default())
}

fn optional_i64(arguments: &[BindArgument], index: usize) -> Result<Option<i64>> {
    let Some(arg) = arguments.get(index) else {
        return Ok(None);
    };
    let Some(value) = &arg.value else {
        return Ok(None);
    };
    i64::from_value(value)
}

fn optional_f64(arguments: &[BindArgument], index: usize) -> Result<Option<f64>> {
    let Some(arg) = arguments.get(index) else {
        return Ok(None);
    };
    let Some(value) = &arg.value else {
        return Ok(None);
    };
    f64::from_value(value)
}

fn required_list(arguments: &[BindArgument], index: usize) -> Result<Vec<Option<f64>>> {
    let arg = argument(arguments, index, "query")?;
    let Some(value) = &arg.value else {
        return Err(Error::invalid_input("query vector is NULL"));
    };
    let list = <List<f64>>::from_value(value)?.unwrap_or_default();
    Ok(list.0)
}

pub fn register(extension: &Extension) -> Result<()> {
    TableFunctionBuilder::new(
        "qdrant_query",
        SignatureBuilder::new()
            .param(FunctionParameter::new::<String>("collection"))
            .param(FunctionParameter::new::<List<f64>>("query"))
            .param(FunctionParameter::with_default("limit", DEFAULT_LIMIT as i64))
            .param(FunctionParameter::with_default("filter", String::new()))
            .param(FunctionParameter::with_default("using", String::new()))
            .param(FunctionParameter::with_default("score_threshold", f64::NAN))
            .param(FunctionParameter::with_default("url", String::new()))
            .param(FunctionParameter::with_default("api_key", String::new()))
            .param(FunctionParameter::with_default("timeout_secs", DEFAULT_TIMEOUT_SECS as i64))
            .documentation(
                FunctionDocumentationBuilder::new()
                    .description(
                        "Nearest neighbours from a Qdrant collection. Returns id, score, and payload JSON.",
                    )
                    .add_example(
                        "Nearest 50",
                        "SELECT * FROM qdrant_query(collection := 'articles', query := [0.1, 0.2, 0.3]::DOUBLE[], limit := 50)",
                    )
                    .add_category(CATEGORY),
            ),
        QueryTable,
    )
    .register(extension)
}
