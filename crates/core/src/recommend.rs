//! `qdrant_recommend`: positive and negative examples through the universal query API.

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
use crate::args::{failure, invalid, optional_string, optional_vector, required_string};
use crate::emit::{Fetched, declare_scored, write_scored};
use crate::query::read_scope;
use crate::run::run_pages;
use crate::types::{RecommendRequest, parse_id_list, parse_strategy, recommend_query};

struct RecommendTable;

impl TableFunctionCallbacks for RecommendTable {
    type GlobalState = Mutex<Fetched<crate::result::QueryRow>>;
    type LocalState = ();
    type BindData = RecommendRequest;

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
        let positive_ids = parse_id_list(&optional_string(&arguments, 1)?).map_err(invalid)?;
        let negative_ids = parse_id_list(&optional_string(&arguments, 2)?).map_err(invalid)?;
        let positive_vector = optional_vector(&arguments, 3)?;
        let negative_vector = optional_vector(&arguments, 4)?;
        if positive_ids.is_empty() && positive_vector.is_none() {
            return Err(Error::invalid_input(
                "qdrant_recommend needs a positive id or a positive vector",
            ));
        }
        let strategy = parse_strategy(&optional_string(&arguments, 5)?).map_err(invalid)?;
        let scope = read_scope(context, &arguments, collection, 6)?;
        Ok(RecommendRequest {
            scope,
            positive_ids,
            negative_ids,
            positive_vector,
            negative_vector,
            strategy,
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
            .ok_or_else(|| Error::internal("qdrant_recommend: missing global state"))?;
        let mut scan = state
            .lock()
            .map_err(|_| Error::internal("qdrant_recommend: scan lock poisoned"))?;
        scan.load(|| run_recommend(bind_data).map_err(failure))?;
        let capacity = output.capacity()?;
        let (start, end) = scan.window(capacity);
        write_scored(&mut output, &scan.rows, start, end - start)?;
        scan.next = end;
        Ok(())
    }
}

pub fn run_recommend(
    request: &RecommendRequest,
) -> std::result::Result<Vec<crate::result::QueryRow>, String> {
    let request = request.clone();
    run_pages(&request.scope, "qdrant_recommend", |offset, page| {
        let mut scoped = request.clone();
        scoped.scope.limit = page;
        recommend_query(&scoped).offset(offset)
    })
}

pub fn register(extension: &Extension) -> Result<()> {
    TableFunctionBuilder::new(
        "qdrant_recommend",
        SignatureBuilder::new()
            .param(FunctionParameter::new::<String>("collection"))
            .param(FunctionParameter::with_default("positive", String::new()))
            .param(FunctionParameter::with_default("negative", String::new()))
            .param(FunctionParameter::with_default("positive_vector", List(Vec::<Option<f64>>::new())))
            .param(FunctionParameter::with_default("negative_vector", List(Vec::<Option<f64>>::new())))
            .param(FunctionParameter::with_default("strategy", "average".to_string()))
            .param(FunctionParameter::with_default("limit", 10_i64))
            .param(FunctionParameter::with_default("filter", String::new()))
            .param(FunctionParameter::with_default("using", String::new()))
            .param(FunctionParameter::with_default("score_threshold", f64::NAN))
            .param(FunctionParameter::with_default("payload_fields", String::new()))
            .param(FunctionParameter::with_default("url", String::new()))
            .param(FunctionParameter::with_default("api_key", String::new()))
            .param(FunctionParameter::with_default("timeout_secs", 0_i64))
            .documentation(
                FunctionDocumentationBuilder::new()
                    .description("Recommend points from positive and negative ids or vectors.")
                    .add_example(
                        "Positive id",
                        "SELECT * FROM qdrant_recommend(collection := 'articles', positive := '[1]')",
                    )
                    .add_category(CATEGORY),
            ),
        RecommendTable,
    )
    .register(extension)
}
