//! Bounded first-party reader for the public safetensors container layout.
//!
//! This reads an already-open file handle and never resolves model paths,
//! downloads weights, or maps a third-party inference API. The caller remains
//! responsible for authenticating a signed model package before admission.

use std::{
    collections::BTreeMap,
    fmt,
    io::{Read, Seek, SeekFrom},
};

use serde::{
    Deserialize,
    de::{self, MapAccess, Visitor},
};

use crate::{PackageError, PackageResult};

const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;
const MAX_FILE_BYTES: u64 = 32 * 1024 * 1024 * 1024;
const MAX_TENSOR_RANK: usize = 8;
const MAX_READ_ELEMENTS: usize = 16 * 1024 * 1024;
const MAX_TENSOR_NAME_BYTES: usize = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDType {
    F32,
    F16,
    BF16,
}

impl TensorDType {
    fn bytes_per_element(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafeTensorMetadata {
    pub dtype: TensorDType,
    pub shape: Vec<usize>,
    /// Half-open byte range relative to the data section after the JSON header.
    pub data_offsets: [u64; 2],
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTensorMetadata {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

struct UniqueStringMap;

impl<'de> Deserialize<'de> for UniqueStringMap {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct StringMapVisitor;
        impl<'de> Visitor<'de> for StringMapVisitor {
            type Value = UniqueStringMap;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a unique map of bounded safetensors metadata strings")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut values = BTreeMap::new();
                while let Some((key, value)) = map.next_entry::<String, String>()? {
                    if key.is_empty()
                        || key.len() > MAX_TENSOR_NAME_BYTES
                        || key.chars().any(char::is_control)
                        || value.len() > 4096
                        || value.chars().any(char::is_control)
                        || values.insert(key, value).is_some()
                    {
                        return Err(de::Error::custom(
                            "invalid or repeated safetensors metadata key",
                        ));
                    }
                }
                let _ = values;
                Ok(UniqueStringMap)
            }
        }
        deserializer.deserialize_map(StringMapVisitor)
    }
}

struct Header(BTreeMap<String, RawTensorMetadata>);

impl<'de> Deserialize<'de> for Header {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct HeaderVisitor;
        impl<'de> Visitor<'de> for HeaderVisitor {
            type Value = Header;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a safetensors header with unique tensor names")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut tensors = BTreeMap::new();
                let mut saw_metadata = false;
                while let Some(name) = map.next_key::<String>()? {
                    if name == "__metadata__" {
                        if saw_metadata {
                            return Err(de::Error::custom("repeated safetensors metadata"));
                        }
                        let _metadata = map.next_value::<UniqueStringMap>()?;
                        saw_metadata = true;
                        continue;
                    }
                    if name.is_empty()
                        || name.len() > MAX_TENSOR_NAME_BYTES
                        || name.chars().any(char::is_control)
                    {
                        return Err(de::Error::custom("invalid safetensors tensor name"));
                    }
                    let metadata = map.next_value::<RawTensorMetadata>()?;
                    if tensors.insert(name, metadata).is_some() {
                        return Err(de::Error::custom("repeated safetensors tensor name"));
                    }
                }
                if tensors.is_empty() {
                    return Err(de::Error::custom("safetensors header contains no tensors"));
                }
                Ok(Header(tensors))
            }
        }
        deserializer.deserialize_map(HeaderVisitor)
    }
}

/// Reader for bounded tensor metadata and on-demand f32 conversion. Reusing a
/// single reader avoids copying entire multi-gigabyte model shards into RAM.
pub struct SafeTensorReader<R> {
    reader: R,
    data_start: u64,
    data_bytes: u64,
    tensors: BTreeMap<String, SafeTensorMetadata>,
}

impl<R: Read + Seek> SafeTensorReader<R> {
    pub fn new(mut reader: R) -> PackageResult<Self> {
        let file_bytes = reader.seek(SeekFrom::End(0))?;
        if !(8..=MAX_FILE_BYTES).contains(&file_bytes) {
            return Err(PackageError::Invalid(
                "Safetensors file length is outside Sage importer bounds".into(),
            ));
        }
        reader.seek(SeekFrom::Start(0))?;
        let mut prefix = [0u8; 8];
        reader.read_exact(&mut prefix)?;
        let header_bytes = u64::from_le_bytes(prefix);
        if header_bytes == 0 || header_bytes > MAX_HEADER_BYTES {
            return Err(PackageError::Invalid(
                "Safetensors header exceeds Sage importer bounds".into(),
            ));
        }
        let data_start = 8u64
            .checked_add(header_bytes)
            .filter(|start| *start <= file_bytes)
            .ok_or_else(|| PackageError::Invalid("Safetensors header length overflow".into()))?;
        let mut header_bytes_buffer = vec![0u8; header_bytes as usize];
        reader.read_exact(&mut header_bytes_buffer)?;
        let Header(raw_tensors) = serde_json::from_slice(&header_bytes_buffer).map_err(|_| {
            PackageError::Invalid("Safetensors header is malformed or ambiguous".into())
        })?;
        let data_bytes = file_bytes - data_start;

        let mut tensors = BTreeMap::new();
        let mut ranges = Vec::with_capacity(raw_tensors.len());
        for (name, raw) in raw_tensors {
            let dtype = match raw.dtype.as_str() {
                "F32" => TensorDType::F32,
                "F16" => TensorDType::F16,
                "BF16" => TensorDType::BF16,
                _ => {
                    return Err(PackageError::Invalid(format!(
                        "Unsupported safetensors dtype for tensor {name}"
                    )));
                }
            };
            if raw.shape.is_empty()
                || raw.shape.len() > MAX_TENSOR_RANK
                || raw.shape.contains(&0)
                || raw.data_offsets[0] > raw.data_offsets[1]
                || raw.data_offsets[1] > data_bytes
            {
                return Err(PackageError::Invalid(format!(
                    "Invalid shape or data offsets for tensor {name}"
                )));
            }
            let elements = raw
                .shape
                .iter()
                .try_fold(1u64, |count, dimension| count.checked_mul(*dimension))
                .ok_or_else(|| PackageError::Invalid("Safetensors tensor shape overflow".into()))?;
            let expected_bytes =
                elements
                    .checked_mul(dtype.bytes_per_element())
                    .ok_or_else(|| {
                        PackageError::Invalid("Safetensors tensor byte count overflow".into())
                    })?;
            if raw.data_offsets[1] - raw.data_offsets[0] != expected_bytes {
                return Err(PackageError::Invalid(format!(
                    "Tensor byte length does not match shape for {name}"
                )));
            }
            let shape = raw
                .shape
                .into_iter()
                .map(|dimension| {
                    usize::try_from(dimension).map_err(|_| {
                        PackageError::Invalid("Tensor dimension exceeds platform bounds".into())
                    })
                })
                .collect::<PackageResult<Vec<_>>>()?;
            ranges.push((raw.data_offsets[0], raw.data_offsets[1], name.clone()));
            tensors.insert(
                name,
                SafeTensorMetadata {
                    dtype,
                    shape,
                    data_offsets: raw.data_offsets,
                },
            );
        }
        ranges.sort_by_key(|range| range.0);
        let mut expected_start = 0u64;
        for (start, end, name) in ranges {
            if start != expected_start || end < start {
                return Err(PackageError::Invalid(format!(
                    "Safetensors data section has overlap or a gap before {name}"
                )));
            }
            expected_start = end;
        }
        if expected_start != data_bytes {
            return Err(PackageError::Invalid(
                "Safetensors data section contains unreferenced trailing bytes".into(),
            ));
        }
        Ok(Self {
            reader,
            data_start,
            data_bytes,
            tensors,
        })
    }

    pub fn tensors(&self) -> &BTreeMap<String, SafeTensorMetadata> {
        &self.tensors
    }

    pub fn data_bytes(&self) -> u64 {
        self.data_bytes
    }

    pub fn source_mut(&mut self) -> &mut R {
        &mut self.reader
    }

    /// Reads one bounded contiguous range and decodes it using Sage-owned
    /// binary16/bfloat16 conversion. Tensor traversal and quantization can
    /// stream through a shard a small range at a time.
    pub fn read_f32_range(
        &mut self,
        name: &str,
        start_element: u64,
        count: usize,
    ) -> PackageResult<Vec<f32>> {
        if count == 0 || count > MAX_READ_ELEMENTS {
            return Err(PackageError::Invalid(
                "Safetensors read range is empty or exceeds Sage importer bounds".into(),
            ));
        }
        let metadata = self
            .tensors
            .get(name)
            .ok_or_else(|| PackageError::Invalid("Requested model tensor is unavailable".into()))?;
        let elements = metadata
            .shape
            .iter()
            .try_fold(1u64, |total, dimension| {
                total.checked_mul(u64::try_from(*dimension).ok()?)
            })
            .ok_or_else(|| PackageError::Invalid("Tensor shape overflow during read".into()))?;
        let count_u64 = u64::try_from(count).map_err(|_| {
            PackageError::Invalid("Tensor read range exceeds platform bounds".into())
        })?;
        let end_element = start_element
            .checked_add(count_u64)
            .filter(|end| *end <= elements)
            .ok_or_else(|| PackageError::Invalid("Tensor read range exceeds its shape".into()))?;
        let bytes_per_element = metadata.dtype.bytes_per_element();
        let relative_start = start_element
            .checked_mul(bytes_per_element)
            .and_then(|offset| metadata.data_offsets[0].checked_add(offset))
            .ok_or_else(|| PackageError::Invalid("Tensor byte offset overflow".into()))?;
        let byte_count = end_element
            .checked_sub(start_element)
            .and_then(|length| length.checked_mul(bytes_per_element))
            .ok_or_else(|| PackageError::Invalid("Tensor read size overflow".into()))?;
        let byte_count = usize::try_from(byte_count).map_err(|_| {
            PackageError::Invalid("Tensor read size exceeds platform bounds".into())
        })?;
        let file_offset = self
            .data_start
            .checked_add(relative_start)
            .ok_or_else(|| PackageError::Invalid("Tensor file offset overflow".into()))?;
        let mut raw = vec![0u8; byte_count];
        self.reader.seek(SeekFrom::Start(file_offset))?;
        self.reader.read_exact(&mut raw)?;
        let values = match metadata.dtype {
            TensorDType::F32 => raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|bytes| f32::from_le_bytes(*bytes))
                .collect::<Vec<_>>(),
            TensorDType::F16 => raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|bytes| crate::f16_to_f32(u16::from_le_bytes(*bytes)))
                .collect::<Vec<_>>(),
            TensorDType::BF16 => raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|bytes| crate::bf16_to_f32(u16::from_le_bytes(*bytes)))
                .collect::<Vec<_>>(),
        };
        if values.len() != count || values.iter().any(|value| !value.is_finite()) {
            return Err(PackageError::Invalid(
                "Model tensor contains non-finite weights or invalid encoded data".into(),
            ));
        }
        Ok(values)
    }
}
