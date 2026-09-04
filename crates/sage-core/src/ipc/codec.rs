use prost::Message;
use sage_protocol::{MAX_FRAME_BYTES, sage::ipc::v1::Frame};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

use crate::error::{CoreError, CoreResult};

pub async fn read_frame<R: AsyncRead + Unpin>(reader: &mut R) -> CoreResult<Frame> {
    let length = reader.read_u32().await? as usize;
    if length == 0 || length > MAX_FRAME_BYTES {
        return Err(CoreError::Protocol(format!(
            "frame length {length} is outside the accepted range"
        )));
    }
    let mut bytes = vec![0_u8; length];
    reader.read_exact(&mut bytes).await?;
    Frame::decode(bytes.as_slice()).map_err(|error| CoreError::Protocol(error.to_string()))
}

pub async fn write_frame<W: AsyncWrite + Unpin>(writer: &mut W, frame: &Frame) -> CoreResult<()> {
    let bytes = frame.encode_to_vec();
    if bytes.is_empty() || bytes.len() > MAX_FRAME_BYTES {
        return Err(CoreError::Protocol(
            "encoded frame is outside the accepted size range".into(),
        ));
    }
    writer.write_u32(bytes.len() as u32).await?;
    writer.write_all(&bytes).await?;
    writer.flush().await?;
    Ok(())
}
