//! Closed, headless peer jobs that operate only on explicitly leased bytes.
//!
//! These executors have no filesystem, process, account, sensor, or network
//! interfaces. They are deterministic CPU functions; running them in this
//! crate does not provide OS process isolation for a donor worker.

use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

use super::{
    AuthorizedPartitionRequest, DelegatedJobKind, PartitionResult, PeerError, PeerResult,
    UnitOutput,
};

#[cfg(test)]
use super::{
    ComputePlan, ComputeUnitSpec, DeviceIdentity, DisclosureScope, LeasePermit, LeaseScope,
    PeerDispatchJournal, PeerId, PeerLease, PeerLeaseBook, PeerLeaseQuotas,
};

const HISTOGRAM_RESOURCE_DOMAIN: &[u8] = b"sage:peer:worker:byte-histogram:v1\0";
const MANDELBROT_RESOURCE_DOMAIN: &[u8] = b"sage:peer:worker:mandelbrot-rgb-tile:v1\0";
const HISTOGRAM_MAGIC: &[u8; 4] = b"HST1";
const HISTOGRAM_HEADER_BYTES: usize = 12;
const HISTOGRAM_BINS: usize = 256;
const HISTOGRAM_OUTPUT_BYTES: usize = HISTOGRAM_HEADER_BYTES + HISTOGRAM_BINS * 8;
const MAX_HISTOGRAM_INPUT_BYTES: usize = 16 * 1024 * 1024;
const MANDELBROT_TILE_VERSION: u8 = 1;
const MANDELBROT_TILE_INPUT_BYTES: usize = 47;
const Q32_SCALE: f64 = 4_294_967_296.0;
const Q32_LIMIT: i64 = 4_i64 << 32;
const MAX_IMAGE_DIMENSION: u16 = 16_384;
const MAX_TILE_PIXELS: u64 = 65_536;
const MAX_TILE_ITERATION_WORK: u64 = 16_777_216;
const MAX_MANDELBROT_ITERATIONS: u16 = 4_096;

type CancellableExecutor = fn(&[u8], &mut dyn FnMut() -> bool) -> PeerResult<Zeroizing<Vec<u8>>>;

pub fn byte_histogram_v1_resource_id() -> [u8; 32] {
    Sha256::digest(HISTOGRAM_RESOURCE_DOMAIN).into()
}

pub fn mandelbrot_rgb_tile_v1_resource_id() -> [u8; 32] {
    Sha256::digest(MANDELBROT_RESOURCE_DOMAIN).into()
}

/// Exact integer/fixed-point inputs for one independently rendered tile.
/// Viewport bounds use signed Q32.32, avoiding locale and decimal parsing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MandelbrotTileSpec {
    pub max_iterations: u16,
    pub image_width: u16,
    pub image_height: u16,
    pub tile_x: u16,
    pub tile_y: u16,
    pub tile_width: u16,
    pub tile_height: u16,
    pub minimum_x_q32: i64,
    pub maximum_x_q32: i64,
    pub minimum_y_q32: i64,
    pub maximum_y_q32: i64,
}

impl MandelbrotTileSpec {
    /// Encode and validate the closed 47-byte `MandelbrotTileV1` input.
    pub fn encode(self) -> PeerResult<Zeroizing<Vec<u8>>> {
        let mut bytes = Zeroizing::new(Vec::new());
        bytes
            .try_reserve_exact(MANDELBROT_TILE_INPUT_BYTES)
            .map_err(|_| PeerError::InvalidComputeJob)?;
        bytes.push(MANDELBROT_TILE_VERSION);
        bytes.extend_from_slice(&self.max_iterations.to_be_bytes());
        bytes.extend_from_slice(&self.image_width.to_be_bytes());
        bytes.extend_from_slice(&self.image_height.to_be_bytes());
        bytes.extend_from_slice(&self.tile_x.to_be_bytes());
        bytes.extend_from_slice(&self.tile_y.to_be_bytes());
        bytes.extend_from_slice(&self.tile_width.to_be_bytes());
        bytes.extend_from_slice(&self.tile_height.to_be_bytes());
        bytes.extend_from_slice(&self.minimum_x_q32.to_be_bytes());
        bytes.extend_from_slice(&self.maximum_x_q32.to_be_bytes());
        bytes.extend_from_slice(&self.minimum_y_q32.to_be_bytes());
        bytes.extend_from_slice(&self.maximum_y_q32.to_be_bytes());
        parse_tile(&bytes)?;
        Ok(bytes)
    }
}

#[derive(Debug, Clone, Copy)]
struct ParsedTile {
    max_iterations: u16,
    image_width: u16,
    image_height: u16,
    tile_x: u16,
    tile_y: u16,
    tile_width: u16,
    tile_height: u16,
    minimum_x: f64,
    maximum_x: f64,
    minimum_y: f64,
    maximum_y: f64,
}

/// Execute one non-cloneable request only after the caller has consumed the
/// owner's live, one-use lease proof. Dispatch is restricted to two versioned
/// input schemas and their exact resource identifiers.
pub fn execute_authorized_partition(
    request: AuthorizedPartitionRequest,
) -> PeerResult<PartitionResult> {
    execute_authorized_partition_with_control(request, || false)
}

/// Execute an authorized partition with bounded cooperative cancellation.
/// Histogram work checks every 16 KiB; Mandelbrot work checks every 64 pixels
/// and at each row boundary. Cancellation discards and zeroizes partial output.
pub fn execute_authorized_partition_with_control(
    request: AuthorizedPartitionRequest,
    mut should_cancel: impl FnMut() -> bool,
) -> PeerResult<PartitionResult> {
    cancellation_checkpoint(&mut should_cancel)?;
    let executor: CancellableExecutor = match request.job() {
        DelegatedJobKind::BatchAnalysis
            if request.resource_id() == &byte_histogram_v1_resource_id() =>
        {
            byte_histogram
        }
        DelegatedJobKind::FrameRender
            if request.resource_id() == &mandelbrot_rgb_tile_v1_resource_id() =>
        {
            mandelbrot_rgb_tile
        }
        _ => return Err(PeerError::InvalidComputeJob),
    };

    let mut outputs = Vec::new();
    outputs
        .try_reserve_exact(request.inputs().len())
        .map_err(|_| PeerError::InvalidComputeResult)?;
    let mut output_bytes = 0_u64;
    if request.maximum_output_bytes_per_unit().len() != request.inputs().len() {
        return Err(PeerError::InvalidComputeResult);
    }
    for (input, unit_output_limit) in request
        .inputs()
        .iter()
        .zip(request.maximum_output_bytes_per_unit())
    {
        cancellation_checkpoint(&mut should_cancel)?;
        let bytes = executor(input.bytes(), &mut should_cancel)?;
        if bytes.len() > *unit_output_limit as usize {
            return Err(PeerError::InvalidComputeResult);
        }
        output_bytes = output_bytes
            .checked_add(bytes.len() as u64)
            .filter(|total| *total <= request.maximum_output_bytes())
            .ok_or(PeerError::InvalidComputeResult)?;
        outputs.push(UnitOutput::from_zeroizing(
            input.index(),
            Sha256::digest(input.bytes()).into(),
            bytes,
        )?);
    }
    PartitionResult::new(
        *request.job_id(),
        *request.plan_digest(),
        request.partition_index(),
        outputs,
    )
}

/// Independently recompute a built-in result from the caller's original input.
/// The caller should also bind the result to its plan and authenticated peer.
pub fn verify_builtin_output(
    job: DelegatedJobKind,
    resource_id: &[u8; 32],
    input: &[u8],
    output: &[u8],
) -> bool {
    verify_builtin_output_with_control(job, resource_id, input, output, || false).unwrap_or(false)
}

/// Independently verify one closed built-in result while allowing Stop to
/// interrupt long verification. `Ok(false)` means the result is invalid;
/// `ComputeCancelled` means verification did not complete.
pub fn verify_builtin_output_with_control(
    job: DelegatedJobKind,
    resource_id: &[u8; 32],
    input: &[u8],
    output: &[u8],
    mut should_cancel: impl FnMut() -> bool,
) -> PeerResult<bool> {
    cancellation_checkpoint(&mut should_cancel)?;
    match job {
        DelegatedJobKind::BatchAnalysis if resource_id == &byte_histogram_v1_resource_id() => {
            verify_byte_histogram_with_control(input, output, &mut should_cancel)
        }
        DelegatedJobKind::FrameRender if resource_id == &mandelbrot_rgb_tile_v1_resource_id() => {
            verify_mandelbrot_rgb_tile_with_control(input, output, &mut should_cancel)
        }
        _ => Ok(false),
    }
}

fn cancellation_checkpoint(should_cancel: &mut dyn FnMut() -> bool) -> PeerResult<()> {
    if should_cancel() {
        Err(PeerError::ComputeCancelled)
    } else {
        Ok(())
    }
}

fn byte_histogram(
    input: &[u8],
    should_cancel: &mut dyn FnMut() -> bool,
) -> PeerResult<Zeroizing<Vec<u8>>> {
    let counts = histogram_counts(input, should_cancel)?;
    let mut output = Zeroizing::new(Vec::new());
    output
        .try_reserve_exact(HISTOGRAM_OUTPUT_BYTES)
        .map_err(|_| PeerError::InvalidComputeResult)?;
    output.extend_from_slice(HISTOGRAM_MAGIC);
    output.extend_from_slice(&(input.len() as u64).to_be_bytes());
    for count in counts {
        output.extend_from_slice(&count.to_be_bytes());
    }
    Ok(output)
}

/// Count four independent byte streams so repeated symbols do not serialize
/// every increment through one hot counter. The fixed 16 KiB blocks preserve
/// bounded cancellation; the input cap keeps each u32 lane far from overflow.
fn histogram_counts(
    input: &[u8],
    should_cancel: &mut dyn FnMut() -> bool,
) -> PeerResult<[u64; HISTOGRAM_BINS]> {
    if input.is_empty() || input.len() > MAX_HISTOGRAM_INPUT_BYTES {
        return Err(PeerError::InvalidComputeJob);
    }
    let mut lanes = [[0_u32; HISTOGRAM_BINS]; 4];
    for block in input.chunks(16_384) {
        cancellation_checkpoint(should_cancel)?;
        let (quads, remainder) = block.as_chunks::<4>();
        for quad in quads {
            lanes[0][usize::from(quad[0])] += 1;
            lanes[1][usize::from(quad[1])] += 1;
            lanes[2][usize::from(quad[2])] += 1;
            lanes[3][usize::from(quad[3])] += 1;
        }
        for byte in remainder {
            lanes[0][usize::from(*byte)] += 1;
        }
    }
    cancellation_checkpoint(should_cancel)?;

    let mut counts = [0_u64; HISTOGRAM_BINS];
    for (index, count) in counts.iter_mut().enumerate() {
        *count = lanes[0][index] as u64
            + lanes[1][index] as u64
            + lanes[2][index] as u64
            + lanes[3][index] as u64;
    }
    Ok(counts)
}

fn verify_byte_histogram_with_control(
    input: &[u8],
    output: &[u8],
    should_cancel: &mut dyn FnMut() -> bool,
) -> PeerResult<bool> {
    if input.is_empty()
        || input.len() > MAX_HISTOGRAM_INPUT_BYTES
        || output.len() != HISTOGRAM_OUTPUT_BYTES
        || !output.starts_with(HISTOGRAM_MAGIC)
        || output.get(4..12) != Some((input.len() as u64).to_be_bytes().as_slice())
    {
        return Ok(false);
    }
    let expected = histogram_counts(input, should_cancel)?;
    Ok(expected.iter().enumerate().all(|(index, count)| {
        let start = HISTOGRAM_HEADER_BYTES + index * 8;
        output.get(start..start + 8) == Some(count.to_be_bytes().as_slice())
    }))
}

fn parse_tile(input: &[u8]) -> PeerResult<ParsedTile> {
    if input.len() != MANDELBROT_TILE_INPUT_BYTES || input[0] != MANDELBROT_TILE_VERSION {
        return Err(PeerError::InvalidComputeJob);
    }
    let spec = ParsedTile {
        max_iterations: read_u16(input, 1)?,
        image_width: read_u16(input, 3)?,
        image_height: read_u16(input, 5)?,
        tile_x: read_u16(input, 7)?,
        tile_y: read_u16(input, 9)?,
        tile_width: read_u16(input, 11)?,
        tile_height: read_u16(input, 13)?,
        minimum_x: read_i64(input, 15)? as f64 / Q32_SCALE,
        maximum_x: read_i64(input, 23)? as f64 / Q32_SCALE,
        minimum_y: read_i64(input, 31)? as f64 / Q32_SCALE,
        maximum_y: read_i64(input, 39)? as f64 / Q32_SCALE,
    };
    let pixels = u64::from(spec.tile_width)
        .checked_mul(u64::from(spec.tile_height))
        .ok_or(PeerError::InvalidComputeJob)?;
    let work = pixels
        .checked_mul(u64::from(spec.max_iterations))
        .ok_or(PeerError::InvalidComputeJob)?;
    if spec.max_iterations == 0
        || spec.max_iterations > MAX_MANDELBROT_ITERATIONS
        || spec.image_width == 0
        || spec.image_height == 0
        || spec.image_width > MAX_IMAGE_DIMENSION
        || spec.image_height > MAX_IMAGE_DIMENSION
        || spec.tile_width == 0
        || spec.tile_height == 0
        || u32::from(spec.tile_x) + u32::from(spec.tile_width) > u32::from(spec.image_width)
        || u32::from(spec.tile_y) + u32::from(spec.tile_height) > u32::from(spec.image_height)
        || pixels > MAX_TILE_PIXELS
        || work > MAX_TILE_ITERATION_WORK
        || read_i64(input, 15)? < -Q32_LIMIT
        || read_i64(input, 15)? > Q32_LIMIT
        || read_i64(input, 23)? < -Q32_LIMIT
        || read_i64(input, 23)? > Q32_LIMIT
        || read_i64(input, 31)? < -Q32_LIMIT
        || read_i64(input, 31)? > Q32_LIMIT
        || read_i64(input, 39)? < -Q32_LIMIT
        || read_i64(input, 39)? > Q32_LIMIT
        || spec.minimum_x >= spec.maximum_x
        || spec.minimum_y >= spec.maximum_y
    {
        return Err(PeerError::InvalidComputeJob);
    }
    Ok(spec)
}

fn mandelbrot_rgb_tile(
    input: &[u8],
    should_cancel: &mut dyn FnMut() -> bool,
) -> PeerResult<Zeroizing<Vec<u8>>> {
    let tile = parse_tile(input)?;
    let pixel_count = usize::from(tile.tile_width)
        .checked_mul(usize::from(tile.tile_height))
        .ok_or(PeerError::InvalidComputeJob)?;
    let output_len = pixel_count
        .checked_mul(3)
        .ok_or(PeerError::InvalidComputeJob)?;
    let mut output = Zeroizing::new(Vec::new());
    output
        .try_reserve_exact(output_len)
        .map_err(|_| PeerError::InvalidComputeResult)?;
    let x_step = (tile.maximum_x - tile.minimum_x) / f64::from(tile.image_width);
    let mut x_coordinates = Vec::new();
    x_coordinates
        .try_reserve_exact(usize::from(tile.tile_width))
        .map_err(|_| PeerError::InvalidComputeResult)?;
    for tile_x in 0..tile.tile_width {
        let image_x = u32::from(tile.tile_x) + u32::from(tile_x);
        x_coordinates.push(tile.minimum_x + (f64::from(image_x) + 0.5) * x_step);
    }
    let y_step = (tile.maximum_y - tile.minimum_y) / f64::from(tile.image_height);
    for tile_y in 0..tile.tile_height {
        cancellation_checkpoint(should_cancel)?;
        let image_y = u32::from(tile.tile_y) + u32::from(tile_y);
        let cy = tile.minimum_y + (f64::from(image_y) + 0.5) * y_step;
        for x_chunk in x_coordinates.chunks(64) {
            cancellation_checkpoint(should_cancel)?;
            for cx in x_chunk {
                let shade = mandelbrot_shade(*cx, cy, tile.max_iterations);
                output.extend_from_slice(&[shade, shade, shade]);
            }
        }
    }
    cancellation_checkpoint(should_cancel)?;
    debug_assert_eq!(output.len(), output_len);
    Ok(output)
}

fn verify_mandelbrot_rgb_tile_with_control(
    input: &[u8],
    output: &[u8],
    should_cancel: &mut dyn FnMut() -> bool,
) -> PeerResult<bool> {
    let Ok(tile) = parse_tile(input) else {
        return Ok(false);
    };
    let expected_len = usize::from(tile.tile_width)
        .checked_mul(usize::from(tile.tile_height))
        .and_then(|pixels| pixels.checked_mul(3));
    if expected_len != Some(output.len()) {
        return Ok(false);
    }
    let x_step = (tile.maximum_x - tile.minimum_x) / f64::from(tile.image_width);
    let y_step = (tile.maximum_y - tile.minimum_y) / f64::from(tile.image_height);
    for tile_y in 0..tile.tile_height {
        cancellation_checkpoint(should_cancel)?;
        let image_y = u32::from(tile.tile_y) + u32::from(tile_y);
        let cy = tile.minimum_y + (f64::from(image_y) + 0.5) * y_step;
        for chunk_start in (0..usize::from(tile.tile_width)).step_by(64) {
            cancellation_checkpoint(should_cancel)?;
            let chunk_end = (chunk_start + 64).min(usize::from(tile.tile_width));
            for tile_x in chunk_start..chunk_end {
                let image_x = u32::from(tile.tile_x) + tile_x as u32;
                let cx = tile.minimum_x + (f64::from(image_x) + 0.5) * x_step;
                let expected = reference_mandelbrot_shade(cx, cy, tile.max_iterations);
                let offset = (usize::from(tile_y) * usize::from(tile.tile_width) + tile_x) * 3;
                if output.get(offset..offset + 3) != Some([expected; 3].as_slice()) {
                    return Ok(false);
                }
            }
        }
    }
    cancellation_checkpoint(should_cancel)?;
    Ok(true)
}

fn mandelbrot_shade(cx: f64, cy: f64, max_iterations: u16) -> u8 {
    let (mut x, mut y) = (0.0_f64, 0.0_f64);
    let mut iteration = 0_u16;
    while iteration < max_iterations && x * x + y * y <= 4.0 {
        let next_x = x * x - y * y + cx;
        let next_y = 2.0 * x * y + cy;
        x = next_x;
        y = next_y;
        iteration += 1;
    }
    if iteration == max_iterations {
        0
    } else {
        ((u32::from(iteration) * 255) / u32::from(max_iterations)) as u8
    }
}

fn reference_mandelbrot_shade(cx: f64, cy: f64, max_iterations: u16) -> u8 {
    let mut real = 0.0_f64;
    let mut imaginary = 0.0_f64;
    for index in 0..max_iterations {
        if real * real + imaginary * imaginary > 4.0 {
            return ((u32::from(index) * 255) / u32::from(max_iterations)) as u8;
        }
        let real_squared = real * real;
        let imaginary_squared = imaginary * imaginary;
        let next_imaginary = (real + real) * imaginary + cy;
        let next_real = real_squared - imaginary_squared + cx;
        real = next_real;
        imaginary = next_imaginary;
    }
    0
}

fn read_u16(bytes: &[u8], offset: usize) -> PeerResult<u16> {
    let value = bytes
        .get(offset..offset + 2)
        .ok_or(PeerError::InvalidComputeJob)?;
    Ok(u16::from_be_bytes([value[0], value[1]]))
}

fn read_i64(bytes: &[u8], offset: usize) -> PeerResult<i64> {
    let value = bytes
        .get(offset..offset + 8)
        .ok_or(PeerError::InvalidComputeJob)?;
    Ok(i64::from_be_bytes(
        value.try_into().map_err(|_| PeerError::InvalidComputeJob)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn authorize_one(
        job: DelegatedJobKind,
        resource_id: [u8; 32],
        input: &[u8],
        maximum_output_bytes: u32,
    ) -> (
        ComputePlan,
        Vec<u8>,
        PeerId,
        AuthorizedPartitionRequest,
        PeerLeaseBook,
        LeasePermit,
    ) {
        let owner = DeviceIdentity::from_seed([81; 32]);
        let requester = DeviceIdentity::from_seed([93; 32]);
        let input_digest: [u8; 32] = Sha256::digest(input).into();
        let plan = ComputePlan::new(
            job,
            resource_id,
            vec![
                ComputeUnitSpec::new(0, input_digest, input.len() as u32, maximum_output_bytes)
                    .unwrap(),
            ],
            1,
        )
        .unwrap();
        let lease = PeerLease::issue(
            &owner,
            requester.peer_id(),
            100,
            120,
            [LeaseScope {
                job,
                resource_id,
                disclosure: DisclosureScope::ExplicitJobInputAndResult,
            }],
            PeerLeaseQuotas {
                maximum_concurrent_jobs: 1,
                maximum_jobs: 1,
                maximum_input_bytes_per_job: input.len() as u64,
                maximum_output_bytes_per_job: u64::from(maximum_output_bytes),
                maximum_total_input_bytes: input.len() as u64,
                maximum_total_output_bytes: u64::from(maximum_output_bytes),
            },
        )
        .unwrap();
        let grant = lease.grant();
        let request = plan
            .make_partition_request(
                &grant,
                &owner.public_identity(),
                requester.peer_id(),
                0,
                vec![super::super::UnitInput::new(0, input.to_vec()).unwrap()],
                101,
            )
            .unwrap();
        let received = super::super::ReceivedPartitionRequest::decode(
            &request.encode().unwrap(),
            requester.peer_id(),
        )
        .unwrap();
        let prepared = plan.prepare_partition_request(received).unwrap();
        let mut lease_book = PeerLeaseBook::new(owner.peer_id());
        let lease_id = lease_book.insert(lease).unwrap();
        let permit = lease_book
            .reserve(lease_id, requester.peer_id(), prepared.lease_request(), 101)
            .unwrap();
        let proof = lease_book.authorize_dispatch(&permit, 101).unwrap();
        let mut nonce = [0; 16];
        getrandom::fill(&mut nonce).unwrap();
        let nonce = u128::from_be_bytes(nonce);
        let journal_path = std::env::temp_dir().join(format!(
            "sage-peer-worker-journal-{}-{nonce:032x}.bin",
            std::process::id()
        ));
        let mut journal =
            PeerDispatchJournal::open(&journal_path, Zeroizing::new([0x5a; 32])).unwrap();
        let authorized = prepared.authorize_durable(proof, &mut journal).unwrap();
        assert!(journal.contains(authorized.job_id(), authorized.partition_index()));
        drop(journal);
        std::fs::remove_file(journal_path).unwrap();
        (
            plan,
            input.to_vec(),
            owner.peer_id(),
            authorized,
            lease_book,
            permit,
        )
    }

    fn tile_input(cx_min: i64, cx_max: i64) -> Zeroizing<Vec<u8>> {
        MandelbrotTileSpec {
            max_iterations: 8,
            image_width: 1,
            image_height: 1,
            tile_x: 0,
            tile_y: 0,
            tile_width: 1,
            tile_height: 1,
            minimum_x_q32: cx_min,
            maximum_x_q32: cx_max,
            minimum_y_q32: -(1_i64 << 32),
            maximum_y_q32: 1_i64 << 32,
        }
        .encode()
        .unwrap()
    }

    fn branch_per_element_histogram_baseline(
        input: &[u8],
        should_cancel: &mut dyn FnMut() -> bool,
    ) -> PeerResult<Zeroizing<Vec<u8>>> {
        if input.is_empty() || input.len() > MAX_HISTOGRAM_INPUT_BYTES {
            return Err(PeerError::InvalidComputeJob);
        }
        let mut counts = [0_u64; HISTOGRAM_BINS];
        for (index, byte) in input.iter().enumerate() {
            if index % 16_384 == 0 {
                cancellation_checkpoint(should_cancel)?;
            }
            counts[usize::from(*byte)] += 1;
        }
        cancellation_checkpoint(should_cancel)?;
        let mut output = Zeroizing::new(Vec::new());
        output
            .try_reserve_exact(HISTOGRAM_OUTPUT_BYTES)
            .map_err(|_| PeerError::InvalidComputeResult)?;
        output.extend_from_slice(HISTOGRAM_MAGIC);
        output.extend_from_slice(&(input.len() as u64).to_be_bytes());
        for count in counts {
            output.extend_from_slice(&count.to_be_bytes());
        }
        Ok(output)
    }

    #[test]
    #[ignore = "release-mode comparison of scalar and four-lane histogram counting"]
    fn four_lane_histogram_benchmark() {
        use std::time::Instant;

        fn median(mut values: Vec<u128>) -> u128 {
            values.sort_unstable();
            values[values.len() / 2]
        }

        fn time_baseline(input: &[u8]) -> u128 {
            let start = Instant::now();
            for _ in 0..4 {
                let mut never_cancel = || false;
                std::hint::black_box(
                    branch_per_element_histogram_baseline(
                        std::hint::black_box(input),
                        &mut never_cancel,
                    )
                    .expect("baseline histogram"),
                );
            }
            start.elapsed().as_nanos() / 4
        }

        fn time_four_lane(input: &[u8]) -> u128 {
            let start = Instant::now();
            for _ in 0..4 {
                let mut never_cancel = || false;
                std::hint::black_box(
                    byte_histogram(std::hint::black_box(input), &mut never_cancel)
                        .expect("four-lane histogram"),
                );
            }
            start.elapsed().as_nanos() / 4
        }

        let mut random = vec![0_u8; 8 * 1024 * 1024];
        let mut state = 0x9e37_79b9_u32;
        for byte in &mut random {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            *byte = state as u8;
        }
        let workloads = [("skewed", vec![0xa5; random.len()]), ("mixed", random)];

        for (name, input) in workloads {
            let mut never_cancel = || false;
            let baseline = branch_per_element_histogram_baseline(&input, &mut never_cancel)
                .expect("baseline histogram");
            let mut never_cancel = || false;
            let four_lane = byte_histogram(&input, &mut never_cancel).expect("four-lane histogram");
            assert_eq!(baseline.as_slice(), four_lane.as_slice());

            let mut baseline_ns = Vec::with_capacity(7);
            let mut four_lane_ns = Vec::with_capacity(7);
            for round in 0..7 {
                if round % 2 == 0 {
                    baseline_ns.push(time_baseline(&input));
                    four_lane_ns.push(time_four_lane(&input));
                } else {
                    four_lane_ns.push(time_four_lane(&input));
                    baseline_ns.push(time_baseline(&input));
                }
            }
            let baseline_p50 = median(baseline_ns);
            let four_lane_p50 = median(four_lane_ns);
            eprintln!(
                "{name}: scalar p50={baseline_p50} ns, four-lane p50={four_lane_p50} ns, speedup={:.3}x",
                baseline_p50 as f64 / four_lane_p50 as f64
            );
        }
    }

    #[test]
    fn histogram_worker_requires_live_lease_and_verifies_typed_output() {
        let input = [0, 0xff, b'a', b'b', 0xff, 0, b'a'];
        let resource = byte_histogram_v1_resource_id();
        let (plan, source, peer, authorized, mut lease_book, permit) = authorize_one(
            DelegatedJobKind::BatchAnalysis,
            resource,
            &input,
            HISTOGRAM_OUTPUT_BYTES as u32,
        );
        let result = execute_authorized_partition(authorized).unwrap();
        let verified = plan
            .verify_partition_result(0, peer, result, |_, output| {
                verify_builtin_output(DelegatedJobKind::BatchAnalysis, &resource, &source, output)
            })
            .unwrap();
        let checkpoint = verified.checkpoint();
        let output = checkpoint.outputs()[0].bytes();
        assert_eq!(&output[..4], HISTOGRAM_MAGIC);
        assert_eq!(
            u64::from_be_bytes(output[4..12].try_into().unwrap()),
            input.len() as u64
        );
        assert_eq!(
            u64::from_be_bytes(
                output[HISTOGRAM_HEADER_BYTES..HISTOGRAM_HEADER_BYTES + 8]
                    .try_into()
                    .unwrap()
            ),
            2
        );
        let settled = lease_book.settle(permit, output.len() as u64).unwrap();
        assert!(settled.dispatched);
        assert_eq!(settled.output_bytes, output.len() as u64);

        let mut corrupted = output.to_vec();
        corrupted[HISTOGRAM_HEADER_BYTES] ^= 1;
        assert!(!verify_builtin_output(
            DelegatedJobKind::BatchAnalysis,
            &resource,
            &source,
            &corrupted,
        ));
    }

    #[test]
    fn mandelbrot_worker_checks_work_bounds_and_matches_reference_verifier() {
        let inside = tile_input(-(1_i64 << 32), 1_i64 << 32);
        let resource = mandelbrot_rgb_tile_v1_resource_id();
        let (plan, source, peer, authorized, mut lease_book, permit) =
            authorize_one(DelegatedJobKind::FrameRender, resource, &inside, 3);
        let result = execute_authorized_partition(authorized).unwrap();
        let verified = plan
            .verify_partition_result(0, peer, result, |_, output| {
                verify_builtin_output(DelegatedJobKind::FrameRender, &resource, &source, output)
            })
            .unwrap();
        assert_eq!(verified.checkpoint().outputs()[0].bytes(), &[0, 0, 0]);
        assert!(lease_book.settle(permit, 3).unwrap().dispatched);

        let outside = tile_input(2_i64 << 32, 4_i64 << 32);
        let rendered = mandelbrot_rgb_tile(&outside, &mut || false).unwrap();
        assert_eq!(rendered.as_slice(), &[31, 31, 31]);
        let resource = mandelbrot_rgb_tile_v1_resource_id();
        assert!(verify_builtin_output(
            DelegatedJobKind::FrameRender,
            &resource,
            &outside,
            &rendered,
        ));
        assert!(!verify_builtin_output(
            DelegatedJobKind::FrameRender,
            &resource,
            &outside,
            &[0, 0, 0],
        ));

        let excessive = MandelbrotTileSpec {
            max_iterations: MAX_MANDELBROT_ITERATIONS,
            image_width: 1,
            image_height: 1,
            tile_x: 0,
            tile_y: 0,
            tile_width: 1,
            tile_height: 1,
            minimum_x_q32: -(1_i64 << 32),
            maximum_x_q32: 1_i64 << 32,
            minimum_y_q32: -(1_i64 << 32),
            maximum_y_q32: 1_i64 << 32,
        }
        .encode();
        assert!(excessive.is_ok());
        let too_much_work = MandelbrotTileSpec {
            image_width: 256,
            image_height: 256,
            tile_width: 256,
            tile_height: 256,
            max_iterations: MAX_MANDELBROT_ITERATIONS,
            tile_x: 0,
            tile_y: 0,
            minimum_x_q32: -(1_i64 << 32),
            maximum_x_q32: 1_i64 << 32,
            minimum_y_q32: -(1_i64 << 32),
            maximum_y_q32: 1_i64 << 32,
        }
        .encode();
        assert_eq!(too_much_work.err(), Some(PeerError::InvalidComputeJob));
    }

    #[test]
    fn worker_rejects_unknown_resource_even_after_valid_lease_admission() {
        let (_, _, _, authorized, _, _) = authorize_one(
            DelegatedJobKind::BatchAnalysis,
            [0x55; 32],
            b"explicit bytes",
            HISTOGRAM_OUTPUT_BYTES as u32,
        );
        assert_eq!(
            execute_authorized_partition(authorized).err(),
            Some(PeerError::InvalidComputeJob)
        );
    }

    #[test]
    fn worker_rejects_single_unit_output_overrun_before_returning_results() {
        let (_, _, _, authorized, _, _) = authorize_one(
            DelegatedJobKind::BatchAnalysis,
            byte_histogram_v1_resource_id(),
            b"one byte",
            1,
        );
        assert_eq!(
            execute_authorized_partition(authorized).err(),
            Some(PeerError::InvalidComputeResult)
        );
    }

    #[test]
    fn worker_cancellation_discards_partial_histogram_and_settles_zero_bytes() {
        let input = vec![0x5a; 64 * 1024];
        let (_, _, _, authorized, mut lease_book, permit) = authorize_one(
            DelegatedJobKind::BatchAnalysis,
            byte_histogram_v1_resource_id(),
            &input,
            HISTOGRAM_OUTPUT_BYTES as u32,
        );
        let mut checkpoints = 0;
        let result = execute_authorized_partition_with_control(authorized, || {
            checkpoints += 1;
            checkpoints >= 4
        });
        assert_eq!(result.err(), Some(PeerError::ComputeCancelled));
        assert!(checkpoints >= 4);

        let settlement = lease_book.settle(permit, 0).unwrap();
        assert!(settlement.dispatched);
        assert_eq!(settlement.output_bytes, 0);
    }

    #[test]
    fn worker_cancellation_interrupts_mandelbrot_at_bounded_pixel_check() {
        let input = MandelbrotTileSpec {
            max_iterations: 1_024,
            image_width: 128,
            image_height: 128,
            tile_x: 0,
            tile_y: 0,
            tile_width: 128,
            tile_height: 128,
            minimum_x_q32: -(1_i64 << 30),
            maximum_x_q32: 1_i64 << 30,
            minimum_y_q32: -(1_i64 << 30),
            maximum_y_q32: 1_i64 << 30,
        }
        .encode()
        .unwrap();
        let (_, _, _, authorized, _, _) = authorize_one(
            DelegatedJobKind::FrameRender,
            mandelbrot_rgb_tile_v1_resource_id(),
            &input,
            (128 * 128 * 3) as u32,
        );
        let mut checkpoints = 0;

        let result = execute_authorized_partition_with_control(authorized, || {
            checkpoints += 1;
            checkpoints >= 5
        });

        assert_eq!(result.err(), Some(PeerError::ComputeCancelled));
        assert!(checkpoints >= 5);
    }

    #[test]
    fn verifier_cancellation_is_distinct_from_invalid_output() {
        let input = vec![0xa5; 64 * 1024];
        let mut never_cancel = || false;
        let output = byte_histogram(&input, &mut never_cancel).unwrap();
        let resource = byte_histogram_v1_resource_id();
        let mut checkpoints = 0;

        let verification = verify_builtin_output_with_control(
            DelegatedJobKind::BatchAnalysis,
            &resource,
            &input,
            &output,
            || {
                checkpoints += 1;
                checkpoints >= 3
            },
        );

        assert_eq!(verification, Err(PeerError::ComputeCancelled));
        assert!(checkpoints >= 3);
        assert!(!verify_builtin_output(
            DelegatedJobKind::BatchAnalysis,
            &resource,
            &input,
            &output[..output.len() - 1],
        ));
    }
}
