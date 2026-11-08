//! `SET qdrant_url`, `qdrant_api_key`, `qdrant_timeout_secs`, `qdrant_with_payload`.
//!
//! A function argument wins over the setting. The setting's default is the
//! environment value captured when the extension loads.

use duckdb_neo::{
    DuckDBType, FromValue, Result, ToValue,
    connection::{Context, SettingScope},
    custom_option::{CustomOptionBuilder, CustomOptionCallbacks},
    error::Error,
    extension::Extension,
    value::Value,
};

use crate::client::Endpoint;
use crate::types::{DEFAULT_TIMEOUT_SECS, DEFAULT_URL};

struct AcceptUrl;
struct AcceptKey;
struct AcceptTimeout;
struct AcceptPayload;

impl CustomOptionCallbacks for AcceptUrl {
    fn update(
        &self,
        _context: &Context,
        value: Value,
        _scope: SettingScope,
        is_reset: bool,
    ) -> Result<Value> {
        if is_reset {
            return Ok(value);
        }
        let text = String::from_value(&value)?.unwrap_or_default();
        if text.trim().is_empty() {
            return Err(Error::invalid_input("qdrant_url must not be empty"));
        }
        Ok(value)
    }
}

impl CustomOptionCallbacks for AcceptKey {
    fn update(
        &self,
        _context: &Context,
        value: Value,
        _scope: SettingScope,
        _is_reset: bool,
    ) -> Result<Value> {
        Ok(value)
    }
}

impl CustomOptionCallbacks for AcceptTimeout {
    fn update(
        &self,
        _context: &Context,
        value: Value,
        _scope: SettingScope,
        is_reset: bool,
    ) -> Result<Value> {
        if is_reset {
            return Ok(value);
        }
        match i64::from_value(&value)? {
            Some(seconds) if seconds > 0 => Ok(value),
            _ => Err(Error::invalid_input(
                "qdrant_timeout_secs must be greater than 0",
            )),
        }
    }
}

impl CustomOptionCallbacks for AcceptPayload {
    fn update(
        &self,
        _context: &Context,
        value: Value,
        _scope: SettingScope,
        _is_reset: bool,
    ) -> Result<Value> {
        Ok(value)
    }
}

pub fn register(extension: &Extension) -> Result<()> {
    let url_default = std::env::var("QDRANT_URL").unwrap_or_else(|_| DEFAULT_URL.to_string());
    let key_default = std::env::var("QDRANT_API_KEY").unwrap_or_default();

    CustomOptionBuilder::new(
        "qdrant_url",
        "gRPC URL of the Qdrant server",
        String::logical_type(extension)?,
        AcceptUrl,
    )
    .default_value(url_default.value(extension)?)
    .default_scope(SettingScope::Session)
    .register(extension)?;

    CustomOptionBuilder::new(
        "qdrant_api_key",
        "Qdrant API key. Empty means no key. The value is not logged.",
        String::logical_type(extension)?,
        AcceptKey,
    )
    .default_value(key_default.value(extension)?)
    .default_scope(SettingScope::Session)
    .register(extension)?;

    CustomOptionBuilder::new(
        "qdrant_timeout_secs",
        "Qdrant client timeout in seconds",
        i64::logical_type(extension)?,
        AcceptTimeout,
    )
    .default_value((DEFAULT_TIMEOUT_SECS as i64).value(extension)?)
    .default_scope(SettingScope::Session)
    .register(extension)?;

    CustomOptionBuilder::new(
        "qdrant_with_payload",
        "Return point payloads as JSON",
        bool::logical_type(extension)?,
        AcceptPayload,
    )
    .default_value(true.value(extension)?)
    .default_scope(SettingScope::Session)
    .register(extension)?;
    Ok(())
}

pub fn endpoint(
    context: &Context,
    url_arg: &str,
    api_key_arg: &str,
    timeout_arg: Option<i64>,
) -> Result<Endpoint> {
    let url = if url_arg.is_empty() {
        setting_string(context, "qdrant_url")?
    } else {
        url_arg.to_string()
    };
    if url.trim().is_empty() {
        return Err(Error::invalid_input("qdrant_url is empty"));
    }
    let api_key = if api_key_arg.is_empty() {
        let configured = setting_string(context, "qdrant_api_key")?;
        if configured.is_empty() {
            None
        } else {
            Some(configured)
        }
    } else {
        Some(api_key_arg.to_string())
    };
    let timeout_secs = match timeout_arg {
        Some(seconds) if seconds > 0 => seconds as u64,
        Some(_) => return Err(Error::invalid_input("timeout_secs must be greater than 0")),
        None => {
            let seconds =
                setting_i64(context, "qdrant_timeout_secs")?.unwrap_or(DEFAULT_TIMEOUT_SECS as i64);
            if seconds <= 0 {
                return Err(Error::invalid_input(
                    "qdrant_timeout_secs must be greater than 0",
                ));
            }
            seconds as u64
        }
    };
    Ok(Endpoint {
        url,
        api_key,
        timeout_secs,
    })
}

pub fn with_payload(context: &Context) -> Result<bool> {
    let (value, _) = context.config()?.get_value("qdrant_with_payload")?;
    Ok(bool::from_value(&value)?.unwrap_or(true))
}

fn setting_string(context: &Context, name: &str) -> Result<String> {
    let (value, _) = context.config()?.get_value(name)?;
    Ok(String::from_value(&value)?.unwrap_or_default())
}

fn setting_i64(context: &Context, name: &str) -> Result<Option<i64>> {
    let (value, _) = context.config()?.get_value(name)?;
    i64::from_value(&value)
}
