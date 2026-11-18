//! `qdrant_query`: nearest neighbours from one collection.

use std::sync::Mutex;

use duckdb_neo::{
    Result,
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

use crate::CATEGORY;
use crate::args::{
    failure, invalid, optional_string, positive_limit, required_string, required_vector,
    score_threshold,
};
use crate::emit::{Fetched, declare_scored, write_scored};
use crate::run::run_pages;
use crate::settings::{endpoint, with_payload};
use crate::types::{
    DEFAULT_LIMIT, QueryRequest, Scope, nearest_page, parse_filter, parse_payload_fields,
};

struct QueryTable;

impl TableFunctionCallbacks for QueryTable {
    type GlobalState = Mutex<Fetched<crate::result::QueryRow>>;
    type LocalState = ();
    type BindData = QueryRequest;

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
        declare_scored(context, bind_handle)?;
        let collection = required_string(&arguments, 0, "collection")?;
        let vector = required_vector(&arguments, 1, "query")?;
        Ok(QueryRequest {
            scope: read_scope(context, &arguments, collection, 2)?,
            vector,
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
        scan.load(|| run_nearest(bind_data).map_err(failure))?;
        let capacity = output.capacity()?;
        let (start, end) = scan.window(capacity);
        write_scored(&mut output, &scan.rows, start, end - start)?;
        scan.next = end;
        Ok(())
    }
}

pub fn run_nearest(
    request: &QueryRequest,
) -> std::result::Result<Vec<crate::result::QueryRow>, String> {
    let request = request.clone();
    run_pages(&request.scope, "qdrant_query", |offset, page| {
        nearest_page(&request, offset, page)
    })
}

pub fn read_scope(
    context: &Context,
    arguments: &[BindArgument],
    collection: String,
    limit_index: usize,
) -> Result<Scope> {
    let limit = positive_limit(arguments, limit_index, DEFAULT_LIMIT as i64)?;
    let filter = parse_filter(&optional_string(arguments, limit_index + 1)?).map_err(invalid)?;
    let using_text = optional_string(arguments, limit_index + 2)?;
    let using = if using_text.is_empty() {
        None
    } else {
        Some(using_text)
    };
    let score_threshold = score_threshold(arguments, limit_index + 3)?;
    let payload_fields =
        parse_payload_fields(&optional_string(arguments, limit_index + 4)?).map_err(invalid)?;
    let endpoint = endpoint(
        context,
        &optional_string(arguments, limit_index + 5)?,
        &optional_string(arguments, limit_index + 6)?,
        optional_i64_non_zero(arguments, limit_index + 7)?,
    )?;
    Ok(Scope {
        collection,
        limit,
        filter,
        using,
        score_threshold,
        with_payload: with_payload(context)?,
        payload_fields,
        endpoint,
    })
}

fn optional_i64_non_zero(arguments: &[BindArgument], index: usize) -> Result<Option<i64>> {
    match crate::args::optional_i64(arguments, index)? {
        Some(0) => Ok(None),
        other => Ok(other),
    }
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
            .param(FunctionParameter::with_default("payload_fields", String::new()))
            .param(FunctionParameter::with_default("url", String::new()))
            .param(FunctionParameter::with_default("api_key", String::new()))
            .param(FunctionParameter::with_default("timeout_secs", 0_i64))
            .documentation(
                FunctionDocumentationBuilder::new()
                    .description(
                        "Nearest neighbours from a Qdrant collection. Returns id, score, and payload JSON.",
                    )
                    .add_example(
                        "Nearest 50",
                        "SELECT * FROM qdrant_query(collection := 'articles', query := [0.1, 0.2, 0.3]::DOUBLE[], \"limit\" := 50)",
                    )
                    .add_category(CATEGORY),
            ),
        QueryTable,
    )
    .register(extension)
}
