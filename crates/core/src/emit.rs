//! Write fetched rows into a DuckDB chunk, one vector at a time.

use duckdb_neo::{Result, data_chunk::DataChunkRef};

use crate::result::{QueryRow, ScrollRow};

pub struct Fetched<T> {
    pub rows: Vec<T>,
    pub next: usize,
    fetched: bool,
}

impl<T> Fetched<T> {
    pub fn new() -> Self {
        Self {
            rows: Vec::new(),
            next: 0,
            fetched: false,
        }
    }

    pub fn load(&mut self, fetch: impl FnOnce() -> Result<Vec<T>>) -> Result<()> {
        if !self.fetched {
            self.rows = fetch()?;
            self.fetched = true;
        }
        Ok(())
    }

    pub fn window(&self, capacity: usize) -> (usize, usize) {
        let end = (self.next + capacity).min(self.rows.len());
        (self.next, end)
    }
}

pub fn write_scored(
    output: &mut DataChunkRef<'_>,
    rows: &[QueryRow],
    start: usize,
    count: usize,
) -> Result<()> {
    {
        let mut id = output.get_vector_at_mut::<String>(0)?;
        id.set_size(count)?;
        let mut writer = id.writer()?;
        for index in 0..count {
            writer.set(index, Some(&rows[start + index].id))?;
        }
    }
    {
        let mut score = output.get_vector_at_mut::<f64>(1)?;
        score.set_size(count)?;
        let mut writer = score.writer()?;
        for index in 0..count {
            writer.set(index, Some(rows[start + index].score))?;
        }
    }
    {
        let mut payload = output.get_vector_at_mut::<String>(2)?;
        payload.set_size(count)?;
        let mut writer = payload.writer()?;
        for index in 0..count {
            writer.set(index, Some(&rows[start + index].payload))?;
        }
    }
    Ok(())
}

pub fn write_scroll(
    output: &mut DataChunkRef<'_>,
    rows: &[ScrollRow],
    start: usize,
    count: usize,
) -> Result<()> {
    {
        let mut id = output.get_vector_at_mut::<String>(0)?;
        id.set_size(count)?;
        let mut writer = id.writer()?;
        for index in 0..count {
            writer.set(index, Some(&rows[start + index].id))?;
        }
    }
    {
        let mut payload = output.get_vector_at_mut::<String>(1)?;
        payload.set_size(count)?;
        let mut writer = payload.writer()?;
        for index in 0..count {
            writer.set(index, Some(&rows[start + index].payload))?;
        }
    }
    Ok(())
}

pub fn declare_scored(
    context: &duckdb_neo::connection::Context,
    bind_handle: duckdb_neo::table_function::BindFunctionHandle<'_>,
) -> Result<()> {
    use duckdb_neo::DuckDBType;
    bind_handle.add_result_column("id", &String::logical_type(context)?)?;
    bind_handle.add_result_column("score", &f64::logical_type(context)?)?;
    bind_handle.add_result_column("payload", &String::logical_type(context)?)?;
    Ok(())
}
