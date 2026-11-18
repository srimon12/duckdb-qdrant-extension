//! `qdrant_scroll`: read points in id order, paged by Qdrant's next offset.

use std::sync::Mutex;

use duckdb_neo::{
    DuckDBType, Result,
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
use qdrant_client::qdrant::ScrollPointsBuilder;

use crate::CATEGORY;
use crate::args::{failure, invalid, optional_string, positive_limit, required_string};
use crate::client::{client, retry};
use crate::emit::{Fetched, write_scroll};
use crate::error::context;
use crate::result::{ScrollRow, retrieved_row};
use crate::settings::{endpoint, with_payload};
use crate::types::{
    DEFAULT_LIMIT, PAGE_SIZE, Scope, parse_filter, parse_payload_fields, payload_selector,
    point_id_text,
};

struct ScrollTable;

/// A scroll is not a search. `offset` is a point id, not a vector name.
pub(crate) struct ScrollRequest {
    pub(crate) scope: Scope,
    pub(crate) offset: Option<String>,
}

impl TableFunctionCallbacks for ScrollTable {
    type GlobalState = Mutex<Fetched<ScrollRow>>;
    type LocalState = ();
    type BindData = ScrollRequest;

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
        bind_handle.add_result_column("payload", &String::logical_type(context)?)?;
        let collection = required_string(&arguments, 0, "collection")?;
        let limit = positive_limit(&arguments, 1, DEFAULT_LIMIT as i64)?;
        let filter = parse_filter(&optional_string(&arguments, 2)?).map_err(invalid)?;
        let offset_text = optional_string(&arguments, 3)?;
        let offset = if offset_text.is_empty() {
            None
        } else {
            Some(offset_text)
        };
        let payload_fields =
            parse_payload_fields(&optional_string(&arguments, 4)?).map_err(invalid)?;
        let endpoint = endpoint(
            context,
            &optional_string(&arguments, 5)?,
            &optional_string(&arguments, 6)?,
            match crate::args::optional_i64(&arguments, 7)? {
                Some(0) => None,
                other => other,
            },
        )?;
        Ok(ScrollRequest {
            scope: Scope {
                collection,
                limit,
                filter,
                using: None,
                score_threshold: None,
                with_payload: with_payload(context)?,
                payload_fields,
                endpoint,
            },
            offset,
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
            global_state.ok_or_else(|| Error::internal("qdrant_scroll: missing global state"))?;
        let mut scan = state
            .lock()
            .map_err(|_| Error::internal("qdrant_scroll: scan lock poisoned"))?;
        scan.load(|| run_scroll(bind_data).map_err(failure))?;
        let capacity = output.capacity()?;
        let (start, end) = scan.window(capacity);
        write_scroll(&mut output, &scan.rows, start, end - start)?;
        scan.next = end;
        Ok(())
    }
}

pub fn run_scroll(request: &ScrollRequest) -> std::result::Result<Vec<ScrollRow>, String> {
    let scope = &request.scope;
    let qdrant = client(&scope.endpoint)?;
    let mut rows = Vec::new();
    let mut offset = request
        .offset
        .clone()
        .map(|text| point_id_text(&text))
        .transpose()?;
    while (rows.len() as u64) < scope.limit {
        let size = (scope.limit - rows.len() as u64).min(PAGE_SIZE);
        let page_limit = u32::try_from(size).expect("scroll page size fits in u32");
        let mut request = ScrollPointsBuilder::new(scope.collection.clone())
            .limit(page_limit)
            .with_payload(payload_selector(
                scope.with_payload,
                scope.payload_fields.as_deref(),
            ));
        if let Some(filter) = &scope.filter {
            request = request.filter(filter.clone());
        }
        if let Some(id) = offset.clone() {
            request = request.offset(id);
        }
        let built = request.build();
        let response = retry(|| {
            let qdrant = qdrant.clone();
            let built = built.clone();
            async move { qdrant.scroll(built).await }
        })
        .map_err(|err| context("qdrant_scroll", &scope.collection, err))?;
        let count = response.result.len() as u64;
        for point in response.result {
            rows.push(
                retrieved_row(point)
                    .map_err(|err| context("qdrant_scroll", &scope.collection, err))?,
            );
        }
        offset = response.next_page_offset;
        if count == 0 || offset.is_none() || count < size {
            break;
        }
    }
    Ok(rows)
}

pub fn register(extension: &Extension) -> Result<()> {
    TableFunctionBuilder::new(
        "qdrant_scroll",
        SignatureBuilder::new()
            .param(FunctionParameter::new::<String>("collection"))
            .param(FunctionParameter::with_default(
                "limit",
                DEFAULT_LIMIT as i64,
            ))
            .param(FunctionParameter::with_default("filter", String::new()))
            .param(FunctionParameter::with_default("offset", String::new()))
            .param(FunctionParameter::with_default(
                "payload_fields",
                String::new(),
            ))
            .param(FunctionParameter::with_default("url", String::new()))
            .param(FunctionParameter::with_default("api_key", String::new()))
            .param(FunctionParameter::with_default("timeout_secs", 0_i64))
            .documentation(
                FunctionDocumentationBuilder::new()
                    .description(
                        "Phase 3. Walk points by id. Returns id and payload. offset is a point id.",
                    )
                    .add_example(
                        "First 100",
                        "SELECT * FROM qdrant_scroll(collection := 'articles', \"limit\" := 100)",
                    )
                    .add_category(CATEGORY),
            ),
        ScrollTable,
    )
    .register(extension)
}
