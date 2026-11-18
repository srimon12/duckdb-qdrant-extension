//! `qdrant_hybrid`: dense plus sparse prefetch, fused with RRF or DBSF.

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
use crate::args::{failure, invalid, optional_string, required_string, required_vector};
use crate::emit::{Fetched, declare_scored, write_scored};
use crate::run::run_pages;
use crate::types::{
    HybridRequest, HybridTextRequest, hybrid_query, hybrid_text_query, parse_fusion, parse_sparse,
};

struct HybridTable;
struct HybridTextTable;

impl TableFunctionCallbacks for HybridTable {
    type GlobalState = Mutex<Fetched<crate::result::QueryRow>>;
    type LocalState = ();
    type BindData = HybridRequest;

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
        let dense = required_vector(&arguments, 1, "dense_query")?;
        let sparse =
            parse_sparse(&required_string(&arguments, 2, "sparse_query")?).map_err(invalid)?;
        let fusion = parse_fusion(&optional_string(&arguments, 3)?).map_err(invalid)?;
        let prefetch_raw = crate::args::optional_i64(&arguments, 5)?.unwrap_or(100);
        if prefetch_raw <= 0 {
            return Err(Error::invalid_input(
                "prefetch_limit must be greater than 0",
            ));
        }
        let scope = crate::types::Scope {
            collection,
            limit: crate::args::positive_limit(&arguments, 4, 10)?,
            filter: crate::types::parse_filter(&optional_string(&arguments, 6)?)
                .map_err(invalid)?,
            using: None,
            score_threshold: crate::args::score_threshold(&arguments, 9)?,
            with_payload: crate::settings::with_payload(context)?,
            payload_fields: crate::types::parse_payload_fields(&optional_string(&arguments, 10)?)
                .map_err(invalid)?,
            endpoint: crate::settings::endpoint(
                context,
                &optional_string(&arguments, 11)?,
                &optional_string(&arguments, 12)?,
                match crate::args::optional_i64(&arguments, 13)? {
                    Some(0) => None,
                    other => other,
                },
            )?,
        };
        let dense_using_text = optional_string(&arguments, 7)?;
        let dense_using = if dense_using_text.is_empty() {
            None
        } else {
            Some(dense_using_text)
        };
        let sparse_using_text = optional_string(&arguments, 8)?;
        let sparse_using = if sparse_using_text.is_empty() {
            "sparse".to_string()
        } else {
            sparse_using_text
        };
        let prefetch_limit = prefetch_raw as u64;
        Ok(HybridRequest {
            scope,
            dense,
            dense_using,
            sparse,
            sparse_using,
            fusion,
            prefetch_limit,
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
            global_state.ok_or_else(|| Error::internal("qdrant_hybrid: missing global state"))?;
        let mut scan = state
            .lock()
            .map_err(|_| Error::internal("qdrant_hybrid: scan lock poisoned"))?;
        scan.load(|| run_hybrid(bind_data).map_err(failure))?;
        let capacity = output.capacity()?;
        let (start, end) = scan.window(capacity);
        write_scored(&mut output, &scan.rows, start, end - start)?;
        scan.next = end;
        Ok(())
    }
}

pub fn run_hybrid(
    request: &HybridRequest,
) -> std::result::Result<Vec<crate::result::QueryRow>, String> {
    let request = request.clone();
    run_pages(&request.scope, "qdrant_hybrid", |_offset, page| {
        let mut scoped = request.clone();
        scoped.scope.limit = page;
        hybrid_query(&scoped)
    })
}

impl TableFunctionCallbacks for HybridTextTable {
    type GlobalState = Mutex<Fetched<crate::result::QueryRow>>;
    type LocalState = ();
    type BindData = HybridTextRequest;

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
        let dense = required_vector(&arguments, 1, "dense_query")?;
        let text = required_string(&arguments, 2, "text_query")?;
        let fusion = parse_fusion(&optional_string(&arguments, 3)?).map_err(invalid)?;
        let dense_using_text = optional_string(&arguments, 4)?;
        let dense_using = if dense_using_text.is_empty() {
            None
        } else {
            Some(dense_using_text)
        };
        let text_field_text = optional_string(&arguments, 5)?;
        let text_field = if text_field_text.is_empty() {
            "text".to_string()
        } else {
            text_field_text
        };
        let prefetch_raw = crate::args::optional_i64(&arguments, 6)?.unwrap_or(100);
        if prefetch_raw <= 0 {
            return Err(Error::invalid_input(
                "prefetch_limit must be greater than 0",
            ));
        }
        let scope = crate::types::Scope {
            collection,
            limit: crate::args::positive_limit(&arguments, 7, 10)?,
            filter: crate::types::parse_filter(&optional_string(&arguments, 8)?)
                .map_err(invalid)?,
            using: None,
            score_threshold: crate::args::score_threshold(&arguments, 9)?,
            with_payload: crate::settings::with_payload(context)?,
            payload_fields: crate::types::parse_payload_fields(&optional_string(&arguments, 10)?)
                .map_err(invalid)?,
            endpoint: crate::settings::endpoint(
                context,
                &optional_string(&arguments, 11)?,
                &optional_string(&arguments, 12)?,
                match crate::args::optional_i64(&arguments, 13)? {
                    Some(0) => None,
                    other => other,
                },
            )?,
        };
        Ok(HybridTextRequest {
            scope,
            dense,
            dense_using,
            text,
            text_field,
            fusion,
            prefetch_limit: prefetch_raw as u64,
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
        let state = global_state
            .ok_or_else(|| Error::internal("qdrant_hybrid_text: missing global state"))?;
        let mut scan = state
            .lock()
            .map_err(|_| Error::internal("qdrant_hybrid_text: scan lock poisoned"))?;
        scan.load(|| run_hybrid_text(bind_data).map_err(failure))?;
        let capacity = output.capacity()?;
        let (start, end) = scan.window(capacity);
        write_scored(&mut output, &scan.rows, start, end - start)?;
        scan.next = end;
        Ok(())
    }
}

pub fn run_hybrid_text(
    request: &HybridTextRequest,
) -> std::result::Result<Vec<crate::result::QueryRow>, String> {
    let request = request.clone();
    run_pages(&request.scope, "qdrant_hybrid_text", |_offset, page| {
        let mut scoped = request.clone();
        scoped.scope.limit = page;
        hybrid_text_query(&scoped)
    })
}

pub fn register(extension: &Extension) -> Result<()> {
    TableFunctionBuilder::new(
        "qdrant_hybrid",
        SignatureBuilder::new()
            .param(FunctionParameter::new::<String>("collection"))
            .param(FunctionParameter::new::<List<f64>>("dense_query"))
            .param(FunctionParameter::new::<String>("sparse_query"))
            .param(FunctionParameter::with_default("fusion", "rrf".to_string()))
            .param(FunctionParameter::with_default("limit", 10_i64))
            .param(FunctionParameter::with_default("prefetch_limit", 100_i64))
            .param(FunctionParameter::with_default("filter", String::new()))
            .param(FunctionParameter::with_default("dense_using", String::new()))
            .param(FunctionParameter::with_default("sparse_using", "sparse".to_string()))
            .param(FunctionParameter::with_default("score_threshold", f64::NAN))
            .param(FunctionParameter::with_default("payload_fields", String::new()))
            .param(FunctionParameter::with_default("url", String::new()))
            .param(FunctionParameter::with_default("api_key", String::new()))
            .param(FunctionParameter::with_default("timeout_secs", 0_i64))
            .documentation(
                FunctionDocumentationBuilder::new()
                    .description("Hybrid dense and sparse search fused with RRF or DBSF.")
                    .add_example(
                        "RRF",
                        "SELECT * FROM qdrant_hybrid(collection := 'docs', dense_query := [0.1, 0.2]::DOUBLE[], sparse_query := '{\"indices\":[1],\"values\":[0.5]}', fusion := 'rrf')",
                    )
                    .add_category(CATEGORY),
            ),
        HybridTable,
    )
    .register(extension)?;

    TableFunctionBuilder::new(
        "qdrant_hybrid_text",
        SignatureBuilder::new()
            .param(FunctionParameter::new::<String>("collection"))
            .param(FunctionParameter::new::<List<f64>>("dense_query"))
            .param(FunctionParameter::new::<String>("text_query"))
            .param(FunctionParameter::with_default("fusion", "rrf".to_string()))
            .param(FunctionParameter::with_default("dense_using", String::new()))
            .param(FunctionParameter::with_default("text_field", "text".to_string()))
            .param(FunctionParameter::with_default("prefetch_limit", 100_i64))
            .param(FunctionParameter::with_default("limit", 10_i64))
            .param(FunctionParameter::with_default("filter", String::new()))
            .param(FunctionParameter::with_default("score_threshold", f64::NAN))
            .param(FunctionParameter::with_default("payload_fields", String::new()))
            .param(FunctionParameter::with_default("url", String::new()))
            .param(FunctionParameter::with_default("api_key", String::new()))
            .param(FunctionParameter::with_default("timeout_secs", 0_i64))
            .documentation(
                FunctionDocumentationBuilder::new()
                    .description(
                        "Dense and BM25 text prefetch fused on the server with RRF or DBSF. No sparse vector. Returns id, score, and payload JSON.",
                    )
                    .add_example(
                        "RRF over dense and text",
                        "SELECT * FROM qdrant_hybrid_text(collection := 'esci-products', dense_query := [0.1, 0.2]::DOUBLE[], text_query := 'cashew granola bars', fusion := 'rrf', dense_using := 'dense', text_field := 'text')",
                    )
                    .add_category(CATEGORY),
            ),
        HybridTextTable,
    )
    .register(extension)
}
