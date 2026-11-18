//! `qdrant_collections` and `qdrant_collection_info`.

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
use qdrant_client::qdrant::CollectionStatus;

use crate::CATEGORY;
use crate::args::{failure, optional_string, required_string};
use crate::client::Endpoint;
use crate::client::{client, retry};
use crate::emit::Fetched;
use crate::error::context;
use crate::settings::endpoint;

#[derive(Clone)]
struct ListBind {
    endpoint: Endpoint,
}

#[derive(Clone)]
struct InfoBind {
    name: String,
    endpoint: Endpoint,
}

struct InfoRow {
    status: String,
    points_count: i64,
    indexed_vectors_count: i64,
    segments_count: i64,
    created_at: Option<String>,
}

struct ListTable;
struct InfoTable;

fn connection_args(
    context: &Context,
    arguments: &[BindArgument],
    start: usize,
) -> Result<Endpoint> {
    endpoint(
        context,
        &optional_string(arguments, start)?,
        &optional_string(arguments, start + 1)?,
        match crate::args::optional_i64(arguments, start + 2)? {
            Some(0) => None,
            other => other,
        },
    )
}

impl TableFunctionCallbacks for ListTable {
    type GlobalState = Mutex<Fetched<String>>;
    type LocalState = ();
    type BindData = ListBind;

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
        bind_handle.add_result_column("name", &String::logical_type(context)?)?;
        Ok(ListBind {
            endpoint: connection_args(context, &arguments, 0)?,
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
            .ok_or_else(|| Error::internal("qdrant_collections: missing global state"))?;
        let mut scan = state
            .lock()
            .map_err(|_| Error::internal("qdrant_collections: scan lock poisoned"))?;
        scan.load(|| list_collections(&bind_data.endpoint).map_err(failure))?;
        let capacity = output.capacity()?;
        let (start, end) = scan.window(capacity);
        let count = end - start;
        let mut name = output.get_vector_at_mut::<String>(0)?;
        name.set_size(count)?;
        let mut writer = name.writer()?;
        for index in 0..count {
            writer.set(index, Some(&scan.rows[start + index]))?;
        }
        scan.next = end;
        Ok(())
    }
}

impl TableFunctionCallbacks for InfoTable {
    type GlobalState = Mutex<Fetched<InfoRow>>;
    type LocalState = ();
    type BindData = InfoBind;

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
        bind_handle.add_result_column("status", &String::logical_type(context)?)?;
        bind_handle.add_result_column("points_count", &i64::logical_type(context)?)?;
        bind_handle.add_result_column("indexed_vectors_count", &i64::logical_type(context)?)?;
        bind_handle.add_result_column("segments_count", &i64::logical_type(context)?)?;
        bind_handle.add_result_column("created_at", &String::logical_type(context)?)?;
        Ok(InfoBind {
            name: required_string(&arguments, 0, "collection")?,
            endpoint: connection_args(context, &arguments, 1)?,
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
            .ok_or_else(|| Error::internal("qdrant_collection_info: missing global state"))?;
        let mut scan = state
            .lock()
            .map_err(|_| Error::internal("qdrant_collection_info: scan lock poisoned"))?;
        scan.load(|| collection_info(bind_data).map_err(failure))?;
        let capacity = output.capacity()?;
        let (start, end) = scan.window(capacity);
        let count = end - start;
        {
            let mut column = output.get_vector_at_mut::<String>(0)?;
            column.set_size(count)?;
            let mut writer = column.writer()?;
            for index in 0..count {
                writer.set(index, Some(&scan.rows[start + index].status))?;
            }
        }
        write_i64(&mut output, 1, count, start, &scan.rows, |row| {
            row.points_count
        })?;
        write_i64(&mut output, 2, count, start, &scan.rows, |row| {
            row.indexed_vectors_count
        })?;
        write_i64(&mut output, 3, count, start, &scan.rows, |row| {
            row.segments_count
        })?;
        {
            let mut column = output.get_vector_at_mut::<String>(4)?;
            column.set_size(count)?;
            let mut writer = column.writer()?;
            for index in 0..count {
                writer.set(index, scan.rows[start + index].created_at.as_deref())?;
            }
        }
        scan.next = end;
        Ok(())
    }
}

fn write_i64(
    output: &mut DataChunkRef<'_>,
    column_index: usize,
    count: usize,
    start: usize,
    rows: &[InfoRow],
    value: impl Fn(&InfoRow) -> i64,
) -> Result<()> {
    let mut column = output.get_vector_at_mut::<i64>(column_index)?;
    column.set_size(count)?;
    let mut writer = column.writer()?;
    for index in 0..count {
        writer.set(index, Some(value(&rows[start + index])))?;
    }
    Ok(())
}

pub fn list_collections(endpoint: &Endpoint) -> std::result::Result<Vec<String>, String> {
    let qdrant = client(endpoint)?;
    let response = retry(|| {
        let qdrant = qdrant.clone();
        async move { qdrant.list_collections().await }
    })
    .map_err(|err| format!("qdrant_collections failed: {err}"))?;
    Ok(response
        .collections
        .into_iter()
        .map(|collection| collection.name)
        .collect())
}

fn collection_info(bind: &InfoBind) -> std::result::Result<Vec<InfoRow>, String> {
    let qdrant = client(&bind.endpoint)?;
    let name = bind.name.clone();
    let response = retry(|| {
        let qdrant = qdrant.clone();
        let name = name.clone();
        async move { qdrant.collection_info(name).await }
    })
    .map_err(|err| context("qdrant_collection_info", &bind.name, err))?;
    let info = response.result.ok_or_else(|| {
        format!(
            "qdrant_collection_info collection={} returned no info",
            bind.name
        )
    })?;
    let status = CollectionStatus::try_from(info.status)
        .map(|status| status.as_str_name().to_string())
        .unwrap_or_else(|_| format!("status {}", info.status));
    Ok(vec![InfoRow {
        status,
        points_count: info.points_count.unwrap_or(0) as i64,
        indexed_vectors_count: info.indexed_vectors_count.unwrap_or(0) as i64,
        segments_count: info.segments_count as i64,
        created_at: info
            .created_at
            .as_ref()
            .and_then(|ts| rfc3339_utc(ts.seconds, ts.nanos)),
    }])
}

/// Format a UTC timestamp. Qdrant 1.20 records `created_at` on the collection.
/// A timestamp before the Unix epoch is not a valid record, so it becomes NULL.
fn rfc3339_utc(seconds: i64, _nanos: i32) -> Option<String> {
    if seconds < 0 {
        return None;
    }
    let days = seconds.div_euclid(86_400);
    let tod = seconds.rem_euclid(86_400) as u32;
    let (year, month, day) = civil_from_days(days);
    let hour = tod / 3600;
    let minute = (tod % 3600) / 60;
    let second = tod % 60;
    Some(format!(
        "{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z"
    ))
}

fn civil_from_days(days: i64) -> (i32, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let year = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let month_part = (5 * doy + 2) / 153;
    let day = doy - (153 * month_part + 2) / 5 + 1;
    let month = if month_part < 10 {
        month_part + 3
    } else {
        month_part - 9
    };
    let year = if month <= 2 { year + 1 } else { year };
    (year as i32, month as u32, day as u32)
}

fn connection_signature(signature: SignatureBuilder) -> SignatureBuilder {
    signature
        .param(FunctionParameter::with_default("url", String::new()))
        .param(FunctionParameter::with_default("api_key", String::new()))
        .param(FunctionParameter::with_default("timeout_secs", 0_i64))
}

pub fn register(extension: &Extension) -> Result<()> {
    TableFunctionBuilder::new(
        "qdrant_collections",
        connection_signature(SignatureBuilder::new()).documentation(
            FunctionDocumentationBuilder::new()
                .description("List Qdrant collection names.")
                .add_example("Names", "SELECT * FROM qdrant_collections()")
                .add_category(CATEGORY),
        ),
        ListTable,
    )
    .register(extension)?;
    TableFunctionBuilder::new(
        "qdrant_collection_info",
        connection_signature(
            SignatureBuilder::new().param(FunctionParameter::new::<String>("collection")),
        )
        .documentation(
            FunctionDocumentationBuilder::new()
                .description(
                    "Phase 3. Status, point counts, and created_at for one collection. Not a search.",
                )
                .add_example(
                    "Articles",
                    "SELECT * FROM qdrant_collection_info(collection := 'articles')",
                )
                .add_category(CATEGORY),
        ),
        InfoTable,
    )
    .register(extension)
}

#[cfg(test)]
mod tests {
    use super::rfc3339_utc;

    #[test]
    fn unix_epoch_is_rfc3339() {
        assert_eq!(rfc3339_utc(0, 0).as_deref(), Some("1970-01-01T00:00:00Z"));
        assert_eq!(
            rfc3339_utc(1_700_000_000, 0).as_deref(),
            Some("2023-11-14T22:13:20Z")
        );
    }

    #[test]
    fn negative_timestamp_is_null() {
        assert_eq!(rfc3339_utc(-5, 0), None);
    }
}
