//! `qdrant_batch`: many queries, one filter, one `query_batch` RPC.
//!
//! Dense mode sends query vectors; text mode sends BM25 text queries against one
//! payload field. Qdrant 1.20.2 optimizes exactly this shape. The rows keep the
//! server order per query, and `query_index` says which request they came from.

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
};
use qdrant_client::qdrant::QueryBatchPointsBuilder;

use crate::CATEGORY;
use crate::args::{
    bm25_weight, failure, invalid, optional_string, positive_limit, required_string,
    score_threshold,
};
use crate::client::{client, retry};
use crate::emit::{Fetched, declare_batch, write_batch};
use crate::error::context;
use crate::result::{BatchRow, scored_rows};
use crate::settings::{endpoint, with_payload};
use crate::types::{
    BatchMode, BatchRequest, DEFAULT_LIMIT, Scope, batch_limit, batch_queries, parse_batch_mode,
    parse_batch_queries, parse_filter, parse_payload_fields,
};

struct BatchTable;

impl TableFunctionCallbacks for BatchTable {
    type GlobalState = Mutex<Fetched<BatchRow>>;
    type LocalState = ();
    type BindData = BatchRequest;

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
        declare_batch(context, bind_handle)?;
        let collection = required_string(&arguments, 0, "collection")?;
        let queries_json = required_string(&arguments, 1, "queries")?;
        let mode = parse_batch_mode(&optional_string(&arguments, 2)?).map_err(invalid)?;
        let using_text = optional_string(&arguments, 3)?;
        let field_text = optional_string(&arguments, 4)?;
        let k = bm25_weight(&arguments, 5)?;
        let b = bm25_weight(&arguments, 6)?;
        let limit =
            batch_limit(positive_limit(&arguments, 7, DEFAULT_LIMIT as i64)?).map_err(invalid)?;
        if mode == BatchMode::Text && !using_text.is_empty() {
            return Err(invalid(
                "using names a dense vector; text mode uses field".to_string(),
            ));
        }
        if mode == BatchMode::Dense && !field_text.is_empty() && field_text != "text" {
            return Err(invalid(
                "field names a text payload field; dense mode uses using".to_string(),
            ));
        }
        let field = if field_text.is_empty() {
            "text".to_string()
        } else {
            field_text
        };
        let using = match mode {
            BatchMode::Text => Some(field),
            BatchMode::Dense if using_text.is_empty() => None,
            BatchMode::Dense => Some(using_text),
        };
        let queries = parse_batch_queries(&queries_json, mode, k, b).map_err(invalid)?;
        Ok(BatchRequest {
            scope: Scope {
                collection,
                limit,
                filter: parse_filter(&optional_string(&arguments, 8)?).map_err(invalid)?,
                using,
                score_threshold: score_threshold(&arguments, 9)?,
                with_payload: with_payload(context)?,
                payload_fields: parse_payload_fields(&optional_string(&arguments, 10)?)
                    .map_err(invalid)?,
                endpoint: endpoint(
                    context,
                    &optional_string(&arguments, 11)?,
                    &optional_string(&arguments, 12)?,
                    match crate::args::optional_i64(&arguments, 13)? {
                        Some(0) => None,
                        other => other,
                    },
                )?,
            },
            queries,
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
            global_state.ok_or_else(|| Error::internal("qdrant_batch: missing global state"))?;
        let mut scan = state
            .lock()
            .map_err(|_| Error::internal("qdrant_batch: scan lock poisoned"))?;
        scan.load(|| run_batch(bind_data).map_err(failure))?;
        let capacity = output.capacity()?;
        let (start, end) = scan.window(capacity);
        write_batch(&mut output, &scan.rows, start, end - start)?;
        scan.next = end;
        Ok(())
    }
}

pub fn run_batch(request: &BatchRequest) -> std::result::Result<Vec<BatchRow>, String> {
    let scope = &request.scope;
    batch_limit(scope.limit)?;
    let qdrant = client(&scope.endpoint)?;
    let batch = QueryBatchPointsBuilder::new(
        scope.collection.clone(),
        batch_queries(scope, &request.queries),
    )
    .build();
    let response = retry(|| {
        let qdrant = qdrant.clone();
        let batch = batch.clone();
        async move { qdrant.query_batch(batch).await }
    })
    .map_err(|err| context("qdrant_batch", &scope.collection, err))?;
    let mut rows = Vec::new();
    for (query_index, result) in response.result.into_iter().enumerate() {
        for row in scored_rows(result.result)
            .map_err(|err| context("qdrant_batch", &scope.collection, err))?
        {
            rows.push(BatchRow {
                query_index: query_index as u64,
                id: row.id,
                score: row.score,
                payload: row.payload,
            });
        }
    }
    Ok(rows)
}

pub fn register(extension: &Extension) -> Result<()> {
    TableFunctionBuilder::new(
        "qdrant_batch",
        SignatureBuilder::new()
            .param(FunctionParameter::new::<String>("collection"))
            .param(FunctionParameter::new::<String>("queries"))
            .param(FunctionParameter::with_default("mode", "dense".to_string()))
            .param(FunctionParameter::with_default("using", String::new()))
            .param(FunctionParameter::with_default("field", "text".to_string()))
            .param(FunctionParameter::with_default("k", 1.2_f64))
            .param(FunctionParameter::with_default("b", 0.75_f64))
            .param(FunctionParameter::with_default("limit", DEFAULT_LIMIT as i64))
            .param(FunctionParameter::with_default("filter", String::new()))
            .param(FunctionParameter::with_default("score_threshold", f64::NAN))
            .param(FunctionParameter::with_default("payload_fields", String::new()))
            .param(FunctionParameter::with_default("url", String::new()))
            .param(FunctionParameter::with_default("api_key", String::new()))
            .param(FunctionParameter::with_default("timeout_secs", 0_i64))
            .documentation(
                FunctionDocumentationBuilder::new()
                    .description(
                        "Many queries in one Qdrant query_batch RPC with one shared filter. mode is dense or text. Returns query_index, id, score, and payload JSON. limit is at most 128. A query with no hits adds no rows.",
                    )
                    .add_example(
                        "Two dense queries",
                        "SELECT * FROM qdrant_batch(collection := 'articles', queries := '[[0.1,0.2],[0.3,0.4]]', mode := 'dense', \"using\" := 'dense', \"limit\" := 10)",
                    )
                    .add_example(
                        "Two text queries",
                        "SELECT * FROM qdrant_batch(collection := 'articles', queries := '[\"granola\",\"cashew\"]', mode := 'text', field := 'text', \"limit\" := 10)",
                    )
                    .add_category(CATEGORY),
            ),
        BatchTable,
    )
    .register(extension)
}
