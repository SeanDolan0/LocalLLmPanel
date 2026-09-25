use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::io::{Read, Seek};

const MAX_METADATA_ENTRIES: usize = 100_000;
const MAX_STRING_BYTES: u64 = 64 * 1024 * 1024;
const MAX_ARRAY_ITEMS: u64 = 10_000_000;

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum GgufValue {
    Uint(u64),
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Array { element_type: u32, values: Vec<u64> },
}

impl GgufValue {
    fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Uint(value) => Some(*value),
            Self::Int(value) if *value >= 0 => Some(*value as u64),
            Self::Bool(value) => Some(u64::from(*value)),
            _ => None,
        }
    }

    fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct GgufMetadata {
    pub context_length: Option<usize>,
    pub block_count: Option<usize>,
    pub head_count: Option<usize>,
    pub head_count_kv: Option<usize>,
    pub key_head_dim: Option<usize>,
    pub value_head_dim: Option<usize>,
}

struct GgufReader<R> {
    inner: R,
}

impl<R: Read + Seek> GgufReader<R> {
    fn new(mut inner: R) -> Result<Self> {
        let mut magic = [0u8; 4];
        inner.read_exact(&mut magic).context("read GGUF magic")?;
        if &magic != b"GGUF" {
            bail!("not a GGUF file");
        }
        let version = read_u32(&mut inner).context("read GGUF version")?;
        if !matches!(version, 2 | 3) {
            bail!("unsupported GGUF version {version}; expected 2 or 3");
        }
        Ok(Self { inner })
    }

    fn read_u16(&mut self) -> Result<u16> {
        let mut bytes = [0u8; 2];
        self.inner.read_exact(&mut bytes)?;
        Ok(u16::from_le_bytes(bytes))
    }

    fn read_i16(&mut self) -> Result<i16> {
        Ok(self.read_u16()? as i16)
    }

    fn read_u32(&mut self) -> Result<u32> {
        let mut bytes = [0u8; 4];
        self.inner.read_exact(&mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_i32(&mut self) -> Result<i32> {
        Ok(self.read_u32()? as i32)
    }

    fn read_u64(&mut self) -> Result<u64> {
        let mut bytes = [0u8; 8];
        self.inner.read_exact(&mut bytes)?;
        Ok(u64::from_le_bytes(bytes))
    }

    fn read_i64(&mut self) -> Result<i64> {
        Ok(self.read_u64()? as i64)
    }

    fn read_f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.read_u32()?))
    }

    fn read_f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.read_u64()?))
    }

    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()?;
        if len > MAX_STRING_BYTES {
            bail!("GGUF string is too large: {len} bytes");
        }
        let mut bytes = vec![0u8; len as usize];
        self.inner.read_exact(&mut bytes)?;
        String::from_utf8(bytes).context("GGUF string is not UTF-8")
    }

    fn read_value(&mut self) -> Result<GgufValue> {
        let value_type = self.read_u32()?;
        if value_type != 9 {
            return self.read_typed_value(value_type);
        }
        let element_type = self.read_u32()?;
        let len = self.read_u64()?;
        if len > MAX_ARRAY_ITEMS {
            bail!("GGUF metadata array is too large: {len} items");
        }
        let mut values = Vec::new();
        let retain = len <= 1_024 && matches!(element_type, 0 | 1 | 2 | 3 | 4 | 5 | 10 | 11);
        for _ in 0..len {
            let value = self.read_typed_value(element_type)?;
            if retain {
                if let Some(value) = value.as_u64() {
                    values.push(value);
                }
            }
        }
        Ok(GgufValue::Array {
            element_type,
            values,
        })
    }

    fn read_typed_value(&mut self, value_type: u32) -> Result<GgufValue> {
        match value_type {
            0 => Ok(GgufValue::Uint(u64::from(self.read_u8()?))),
            1 => Ok(GgufValue::Int(i64::from(self.read_i8()?))),
            2 => Ok(GgufValue::Uint(u64::from(self.read_u16()?))),
            3 => Ok(GgufValue::Int(i64::from(self.read_i16()?))),
            4 => Ok(GgufValue::Uint(u64::from(self.read_u32()?))),
            5 => Ok(GgufValue::Int(i64::from(self.read_i32()?))),
            6 => Ok(GgufValue::Float(f64::from(self.read_f32()?))),
            7 => Ok(GgufValue::Bool(self.read_u8()? != 0)),
            8 => Ok(GgufValue::String(self.read_string()?)),
            9 => self.read_value(),
            10 => Ok(GgufValue::Uint(self.read_u64()?)),
            11 => Ok(GgufValue::Int(self.read_i64()?)),
            12 => Ok(GgufValue::Float(self.read_f64()?)),
            other => bail!("unsupported GGUF metadata value type {other}"),
        }
    }

    fn read_u8(&mut self) -> Result<u8> {
        let mut byte = [0u8; 1];
        self.inner.read_exact(&mut byte)?;
        Ok(byte[0])
    }

    fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_u8()? as i8)
    }
}

fn read_u32<R: Read>(reader: &mut R) -> Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

pub(crate) fn read_metadata<R: Read + Seek>(reader: R) -> Result<GgufMetadata> {
    let mut reader = GgufReader::new(reader)?;
    let _tensor_count = reader.read_u64()?;
    let metadata_count = reader.read_u64()?;
    if metadata_count > MAX_METADATA_ENTRIES as u64 {
        bail!("GGUF metadata has too many entries: {metadata_count}");
    }

    let mut values = HashMap::new();
    let mut previous_key = "<none>".to_string();
    for _ in 0..metadata_count {
        let key = reader.read_string()?;
        let value = reader
            .read_value()
            .with_context(|| format!("read GGUF metadata value for {key} after {previous_key}"))?;
        previous_key = key.clone();
        values.insert(key, value);
    }

    let architecture = values
        .get("general.architecture")
        .and_then(GgufValue::as_str)
        .map(str::to_string);
    let prefix = architecture.as_deref().map(|arch| format!("{arch}."));
    let get_arch = |suffix: &str| {
        prefix
            .as_ref()
            .and_then(|prefix| values.get(&format!("{prefix}{suffix}")))
            .and_then(GgufValue::as_u64)
            .and_then(|value| usize::try_from(value).ok())
    };
    let embedding_length = get_arch("embedding_length");
    let head_count = get_arch("attention.head_count");
    let fallback_head_dim = match (embedding_length, head_count.filter(|count| *count > 0)) {
        (Some(embedding), Some(count)) => Some(embedding / count),
        _ => None,
    };
    let key_head_dim = get_arch("attention.key_length").or(fallback_head_dim);
    let value_head_dim = get_arch("attention.value_length").or(fallback_head_dim);

    Ok(GgufMetadata {
        context_length: get_arch("context_length"),
        block_count: get_arch("block_count"),
        head_count,
        head_count_kv: get_arch("attention.head_count_kv").or(head_count),
        key_head_dim,
        value_head_dim,
    })
}

pub(crate) fn metadata_from_path(path: &std::path::Path) -> Result<GgufMetadata> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("open GGUF metadata for {}", path.display()))?;
    let reader = std::io::BufReader::new(file);
    read_metadata(reader)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn push_u32(buf: &mut Vec<u8>, value: u32) {
        buf.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u64(buf: &mut Vec<u8>, value: u64) {
        buf.extend_from_slice(&value.to_le_bytes());
    }

    fn push_string(buf: &mut Vec<u8>, value: &str) {
        push_u64(buf, value.len() as u64);
        buf.extend_from_slice(value.as_bytes());
    }

    fn push_meta(buf: &mut Vec<u8>, key: &str, value: GgufValue) {
        push_string(buf, key);
        match value {
            GgufValue::Uint(value) => {
                push_u32(buf, 4);
                buf.extend_from_slice(&(value as u32).to_le_bytes());
            }
            GgufValue::String(value) => {
                push_u32(buf, 8);
                push_string(buf, &value);
            }
            GgufValue::Array { values, .. } => {
                push_u32(buf, 9);
                push_u32(buf, 4);
                push_u64(buf, values.len() as u64);
                for value in values {
                    buf.extend_from_slice(&(value as u32).to_le_bytes());
                }
            }
            _ => unreachable!(),
        }
    }

    fn sample_gguf() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        push_u32(&mut buf, 3);
        push_u64(&mut buf, 1);
        push_u64(&mut buf, 7);
        push_meta(
            &mut buf,
            "general.architecture",
            GgufValue::String("qwen2".into()),
        );
        push_meta(&mut buf, "qwen2.context_length", GgufValue::Uint(32_768));
        push_meta(&mut buf, "qwen2.block_count", GgufValue::Uint(28));
        push_meta(&mut buf, "qwen2.attention.head_count", GgufValue::Uint(28));
        push_meta(
            &mut buf,
            "qwen2.attention.head_count_kv",
            GgufValue::Uint(4),
        );
        push_meta(&mut buf, "qwen2.embedding_length", GgufValue::Uint(3_584));
        push_meta(
            &mut buf,
            "tokenizer.ggml.tokens",
            GgufValue::Array {
                element_type: 8,
                values: Vec::new(),
            },
        );
        buf
    }

    #[test]
    fn parses_qwen_gguf_metadata_and_skips_arrays() {
        let metadata = read_metadata(Cursor::new(sample_gguf())).unwrap();
        assert_eq!(metadata.context_length, Some(32_768));
        assert_eq!(metadata.block_count, Some(28));
        assert_eq!(metadata.head_count, Some(28));
        assert_eq!(metadata.head_count_kv, Some(4));
        assert_eq!(metadata.key_head_dim, Some(128));
        assert_eq!(metadata.value_head_dim, Some(128));
    }

    #[test]
    fn rejects_non_gguf_input() {
        assert!(read_metadata(Cursor::new(b"not a gguf".to_vec())).is_err());
    }

    #[test]
    #[ignore]
    fn live_gguf_metadata_probe() {
        let path = std::env::var("LLM_TEST_GGUF_PATH")
            .expect("set LLM_TEST_GGUF_PATH to a local GGUF file");
        let metadata = metadata_from_path(std::path::Path::new(&path)).unwrap();
        eprintln!("{metadata:#?}");
        assert!(metadata.context_length.unwrap_or_default() > 0);
        assert!(metadata.block_count.unwrap_or_default() > 0);
    }
}
