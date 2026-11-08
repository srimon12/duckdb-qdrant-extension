//! `qdrant_discover` and `qdrant_context`.

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
use crate::args::{failure, invalid, optional_vector, required_string};
use crate::emit::{Fetched, declare_scored, write_scored};
use crate::query::read_scope;
use crate::run::run_pages;
use crate::types::{DiscoverRequest, discover_query, parse_context_pairs};

struct DiscoverTable;
struct ContextTable;

impl TableFunctionCallbacks for DiscoverTable {
    type GlobalState = Mutex<Fetched<crate::result::QueryRow>>;
    type LocalState = ();
    type BindData = DiscoverRequest;

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
        bind_discover(context, &arguments, true)
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
        scan_discover("qdrant_discover", bind_data, global_state, &mut output)
    }
}

impl TableFunctionCallbacks for ContextTable {
    type GlobalState = Mutex<Fetched<crate::result::QueryRow>>;
    type LocalState = ();
    type BindData = DiscoverRequest;

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
        let pairs =
            parse_context_pairs(&crate::args::optional_string(&arguments, 1)?).map_err(invalid)?;
        let scope = read_scope(context, &arguments, collection, 2)?;
        Ok(DiscoverRequest {
            scope,
            target: None,
            pairs,
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
        scan_discover("qdrant_context", bind_data, global_state, &mut output)
    }
}

fn bind_discover(
    context: &Context,
    arguments: &[BindArgument],
    require_target: bool,
) -> Result<DiscoverRequest> {
    let collection = required_string(arguments, 0, "collection")?;
    let target = optional_vector(arguments, 1)?;
    if require_target && target.is_none() {
        return Err(Error::invalid_input(
            "qdrant_discover needs a target vector",
        ));
    }
    let pairs =
        parse_context_pairs(&crate::args::optional_string(arguments, 2)?).map_err(invalid)?;
    let scope = read_scope(context, arguments, collection, 3)?;
    Ok(DiscoverRequest {
        scope,
        target,
        pairs,
    })
}

fn scan_discover(
    operation: &'static str,
    bind_data: &DiscoverRequest,
    global_state: Option<&Mutex<Fetched<crate::result::QueryRow>>>,
    output: &mut DataChunkRef<'_>,
) -> Result<()> {
    let state = global_state
        .ok_or_else(|| Error::internal(format!("{operation}: missing global state")))?;
    let mut scan = state
        .lock()
        .map_err(|_| Error::internal(format!("{operation}: scan lock poisoned")))?;
    scan.load(|| run_discover(operation, bind_data).map_err(failure))?;
    let capacity = output.capacity()?;
    let (start, end) = scan.window(capacity);
    write_scored(output, &scan.rows, start, end - start)?;
    scan.next = end;
    Ok(())
}

pub fn run_discover(
    operation: &str,
    request: &DiscoverRequest,
) -> std::result::Result<Vec<crate::result::QueryRow>, String> {
    let request = request.clone();
    let operation = operation.to_string();
    run_pages(&request.scope, &operation, |offset, page| {
        let mut scoped = request.clone();
        scoped.scope.limit = page;
        discover_query(&scoped).offset(offset)
    })
}

pub fn register(extension: &Extension) -> Result<()> {
    let tail = |signature: SignatureBuilder| {
        signature
            .param(FunctionParameter::with_default("limit", 10_i64))
            .param(FunctionParameter::with_default("filter", String::new()))
            .param(FunctionParameter::with_default("using", String::new()))
            .param(FunctionParameter::with_default("score_threshold", f64::NAN))
            .param(FunctionParameter::with_default(
                "payload_fields",
                String::new(),
            ))
            .param(FunctionParameter::with_default("url", String::new()))
            .param(FunctionParameter::with_default("api_key", String::new()))
            .param(FunctionParameter::with_default("timeout_secs", 0_i64))
    };
    TableFunctionBuilder::new(
        "qdrant_discover",
        tail(SignatureBuilder::new()
            .param(FunctionParameter::new::<String>("collection"))
            .param(FunctionParameter::new::<List<f64>>("target"))
            .param(FunctionParameter::new::<String>("context")))
        .documentation(
            FunctionDocumentationBuilder::new()
                .description("Discover points from a target vector and positive/negative context pairs.")
                .add_example(
                    "One pair",
                    "SELECT * FROM qdrant_discover(collection := 'articles', target := [0.1, 0.2]::DOUBLE[], context := '[{\"positive\":\"1\",\"negative\":\"2\"}]')",
                )
                .add_category(CATEGORY),
        ),
        DiscoverTable,
    )
    .register(extension)?;
    TableFunctionBuilder::new(
        "qdrant_context",
        tail(SignatureBuilder::new()
            .param(FunctionParameter::new::<String>("collection"))
            .param(FunctionParameter::new::<String>("context")))
        .documentation(
            FunctionDocumentationBuilder::new()
                .description("Search by context pairs without a target vector.")
                .add_example(
                    "One pair",
                    "SELECT * FROM qdrant_context(collection := 'articles', context := '[{\"positive\":\"1\",\"negative\":\"2\"}]')",
                )
                .add_category(CATEGORY),
        ),
        ContextTable,
    )
    .register(extension)
}
