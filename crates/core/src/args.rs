//! Named-argument readers shared by the table functions.

use duckdb_neo::{FromValue, Result, bind_arguments::BindArgument, error::Error, types::List};

use crate::types::dense_vector;

pub fn invalid(message: String) -> Error {
    Error::invalid_input(message)
}

pub fn failure(message: String) -> Error {
    Error::internal(message)
}

pub fn required_string(arguments: &[BindArgument], index: usize, name: &str) -> Result<String> {
    let arg = argument(arguments, index, name)?;
    let Some(value) = &arg.value else {
        return Err(Error::invalid_input(format!("{name} is NULL")));
    };
    let text = String::from_value(value)?
        .ok_or_else(|| Error::invalid_input(format!("{name} is NULL")))?;
    if text.is_empty() {
        return Err(Error::invalid_input(format!("{name} is empty")));
    }
    Ok(text)
}

pub fn optional_string(arguments: &[BindArgument], index: usize) -> Result<String> {
    let Some(arg) = arguments.get(index) else {
        return Ok(String::new());
    };
    let Some(value) = &arg.value else {
        return Ok(String::new());
    };
    Ok(String::from_value(value)?.unwrap_or_default())
}

pub fn optional_i64(arguments: &[BindArgument], index: usize) -> Result<Option<i64>> {
    let Some(arg) = arguments.get(index) else {
        return Ok(None);
    };
    let Some(value) = &arg.value else {
        return Ok(None);
    };
    i64::from_value(value)
}

pub fn optional_f64(arguments: &[BindArgument], index: usize) -> Result<Option<f64>> {
    let Some(arg) = arguments.get(index) else {
        return Ok(None);
    };
    let Some(value) = &arg.value else {
        return Ok(None);
    };
    f64::from_value(value)
}

pub fn positive_limit(arguments: &[BindArgument], index: usize, default: i64) -> Result<u64> {
    let limit = optional_i64(arguments, index)?.unwrap_or(default);
    if limit <= 0 {
        return Err(Error::invalid_input("limit must be greater than 0"));
    }
    Ok(limit as u64)
}

pub fn score_threshold(arguments: &[BindArgument], index: usize) -> Result<Option<f32>> {
    Ok(optional_f64(arguments, index)?
        .filter(|value| value.is_finite())
        .map(|value| value as f32))
}

pub fn required_vector(arguments: &[BindArgument], index: usize, name: &str) -> Result<Vec<f32>> {
    dense_vector(number_list(arguments, index, name)?).map_err(invalid)
}

pub fn optional_vector(arguments: &[BindArgument], index: usize) -> Result<Option<Vec<f32>>> {
    let Some(arg) = arguments.get(index) else {
        return Ok(None);
    };
    let Some(value) = &arg.value else {
        return Ok(None);
    };
    let list = <List<f64>>::from_value(value)?.unwrap_or_default().0;
    if list.is_empty() {
        return Ok(None);
    }
    dense_vector(list).map(Some).map_err(invalid)
}

pub fn string_list(arguments: &[BindArgument], index: usize, name: &str) -> Result<Vec<String>> {
    let Some(arg) = arguments.get(index) else {
        return Ok(Vec::new());
    };
    let Some(value) = &arg.value else {
        return Ok(Vec::new());
    };
    let list = <List<String>>::from_value(value)?.unwrap_or_default().0;
    let mut texts = Vec::with_capacity(list.len());
    for (offset, item) in list.into_iter().enumerate() {
        let text = item.ok_or_else(|| {
            Error::invalid_input(format!("{name} contains NULL at index {offset}"))
        })?;
        texts.push(text);
    }
    Ok(texts)
}

fn number_list(arguments: &[BindArgument], index: usize, name: &str) -> Result<Vec<Option<f64>>> {
    let arg = argument(arguments, index, name)?;
    let Some(value) = &arg.value else {
        return Err(Error::invalid_input(format!("{name} is NULL")));
    };
    Ok(<List<f64>>::from_value(value)?.unwrap_or_default().0)
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
