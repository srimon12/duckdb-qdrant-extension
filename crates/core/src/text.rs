//! `qdrant_text`: BM25 over a payload field's full-text index, from Qdrant 1.20.
//!
//! The field must have a text index with `scoring` enabled. The query is plain
//! text; Qdrant tokenizes it with the index settings. No vector is involved.

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

use crate::CATEGORY;
use crate::args::{
    bm25_weight, failure, invalid, optional_string, positive_limit, required_string,
    score_threshold,
};
use crate::emit::{Fetched, declare_scored, write_scored};
use crate::run::run_pages;
use crate::settings::{endpoint, with_payload};
use crate::types::{
    DEFAULT_LIMIT, Scope, TextRequest, parse_filter, parse_payload_fields, text_page,
};

struct TextTable;

impl TableFunctionCallbacks for TextTable {
    type GlobalState = Mutex<Fetched<crate::result::QueryRow>>;
    type LocalState = ();
    type BindData = TextRequest;

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
        let text = required_string(&arguments, 1, "query")?;
        let field_text = optional_string(&arguments, 2)?;
        let field = if field_text.is_empty() {
            "text".to_string()
        } else {
            field_text
        };
        Ok(TextRequest {
            scope: Scope {
                collection,
                limit: positive_limit(&arguments, 3, DEFAULT_LIMIT as i64)?,
                filter: parse_filter(&optional_string(&arguments, 4)?).map_err(invalid)?,
                // A text query names the payload field in `using`.
                using: Some(field),
                score_threshold: score_threshold(&arguments, 5)?,
                with_payload: with_payload(context)?,
                payload_fields: parse_payload_fields(&optional_string(&arguments, 6)?)
                    .map_err(invalid)?,
                endpoint: endpoint(
                    context,
                    &optional_string(&arguments, 7)?,
                    &optional_string(&arguments, 8)?,
                    match crate::args::optional_i64(&arguments, 9)? {
                        Some(0) => None,
                        other => other,
                    },
                )?,
            },
            text,
            k: bm25_weight(&arguments, 10)?,
            b: bm25_weight(&arguments, 11)?,
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
            global_state.ok_or_else(|| Error::internal("qdrant_text: missing global state"))?;
        let mut scan = state
            .lock()
            .map_err(|_| Error::internal("qdrant_text: scan lock poisoned"))?;
        scan.load(|| run_text(bind_data).map_err(failure))?;
        let capacity = output.capacity()?;
        let (start, end) = scan.window(capacity);
        write_scored(&mut output, &scan.rows, start, end - start)?;
        scan.next = end;
        Ok(())
    }
}

pub fn run_text(
    request: &TextRequest,
) -> std::result::Result<Vec<crate::result::QueryRow>, String> {
    let request = request.clone();
    run_pages(&request.scope, "qdrant_text", |offset, page| {
        text_page(&request, offset, page)
    })
}

pub fn register(extension: &Extension) -> Result<()> {
    TableFunctionBuilder::new(
        "qdrant_text",
        SignatureBuilder::new()
            .param(FunctionParameter::new::<String>("collection"))
            .param(FunctionParameter::new::<String>("query"))
            .param(FunctionParameter::with_default("field", "text".to_string()))
            .param(FunctionParameter::with_default("limit", DEFAULT_LIMIT as i64))
            .param(FunctionParameter::with_default("filter", String::new()))
            .param(FunctionParameter::with_default("score_threshold", f64::NAN))
            .param(FunctionParameter::with_default("payload_fields", String::new()))
            .param(FunctionParameter::with_default("url", String::new()))
            .param(FunctionParameter::with_default("api_key", String::new()))
            .param(FunctionParameter::with_default("timeout_secs", 0_i64))
            .param(FunctionParameter::with_default("k", 1.2_f64))
            .param(FunctionParameter::with_default("b", 0.75_f64))
            .documentation(
                FunctionDocumentationBuilder::new()
                    .description(
                        "BM25 text search over a payload field indexed with scoring. Qdrant 1.20 and later. Returns id, score, and payload JSON.",
                    )
                    .add_example(
                        "BM25 products",
                        "SELECT * FROM qdrant_text(collection := 'esci-products', query := 'cashew granola bars', field := 'text', \"limit\" := 10)",
                    )
                    .add_example(
                        "Tuned BM25",
                        "SELECT * FROM qdrant_text(collection := 'esci-products', query := 'granola', k := 1.5, b := 0.5)",
                    )
                    .add_category(CATEGORY),
            ),
        TextTable,
    )
    .register(extension)
}
