//! Variant that does not have a specialized version for different bit-widths
//!

use super::{BitPacker, UnsafeBitPacker};

const BLOCK_LEN: usize = 32 * 4;

mod scalar {
    use super::BLOCK_LEN;
    use std::ptr;

    type DataType = [u64; 2];

    fn set1(el: u64) -> DataType {
        [el; 2]
    }

    fn right_shift_64(el: DataType, shift: i32) -> DataType {
        [el[0] >> shift, el[1] >> shift]
    }

    fn left_shift_64(el: DataType, shift: i32) -> DataType {
        [el[0] << shift, el[1] << shift | el[0] >> (64 - shift)]
    }

    fn op_or(left: DataType, right: DataType) -> DataType {
        [left[0] | right[0], left[1] | right[1]]
    }

    fn op_and(left: DataType, right: DataType) -> DataType {
        [left[0] & right[0], left[1] & right[1]]
    }

    unsafe fn load_unaligned(addr: *const DataType) -> DataType {
        ptr::read_unaligned(addr)
    }

    unsafe fn store_unaligned(addr: *mut DataType, data: DataType) {
        ptr::write_unaligned(addr, data);
    }

    fn or_collapse_to_u64(accumulator: DataType) -> u64 {
        accumulator[0] | accumulator[1]
    }

    fn compute_delta(curr: DataType, prev: DataType) -> DataType {
        [curr[0].wrapping_sub(prev[1]), curr[1].wrapping_sub(curr[0])]
    }

    fn integrate_delta(offset: DataType, delta: DataType) -> DataType {
        let el0 = offset[1].wrapping_add(delta[0]);
        let el1 = el0.wrapping_add(delta[1]);
        [el0, el1]
    }

    fn add(left: DataType, right: DataType) -> DataType {
        [
            left[0].wrapping_add(right[0]),
            left[1].wrapping_add(right[1]),
        ]
    }

    fn sub(left: DataType, right: DataType) -> DataType {
        [
            left[0].wrapping_sub(right[0]),
            left[1].wrapping_sub(right[1]),
        ]
    }

    declare_bitpacker_simple!(cfg(any(debug, not(debug))));
}

/// `BitPacker4x` packs integers in groups of 4. This gives an opportunity
/// to leverage `SSE3` instructions to encode and decode the stream.
///
/// One block must contain `128 integers`.
#[derive(Clone, Copy)]
pub struct BitPacker4x;

impl BitPacker for BitPacker4x {
    const BLOCK_LEN: usize = BLOCK_LEN;

    /// Returns the best available implementation for the current CPU.
    fn new() -> Self {
        BitPacker4x
    }

    fn compress(&self, decompressed: &[u64], compressed: &mut [u8], num_bits: u8) -> usize {
        unsafe { scalar::UnsafeBitPackerImpl::compress(decompressed, compressed, num_bits) }
    }

    fn compress_sorted(
        &self,
        initial: u64,
        decompressed: &[u64],
        compressed: &mut [u8],
        num_bits: u8,
    ) -> usize {
        unsafe {
            scalar::UnsafeBitPackerImpl::compress_sorted(
                initial,
                decompressed,
                compressed,
                num_bits,
            )
        }
    }

    fn compress_strictly_sorted(
        &self,
        initial: Option<u64>,
        decompressed: &[u64],
        compressed: &mut [u8],
        num_bits: u8,
    ) -> usize {
        unsafe {
            scalar::UnsafeBitPackerImpl::compress_strictly_sorted(
                initial,
                decompressed,
                compressed,
                num_bits,
            )
        }
    }

    fn decompress(&self, compressed: &[u8], decompressed: &mut [u64], num_bits: u8) -> usize {
        unsafe { scalar::UnsafeBitPackerImpl::decompress(compressed, decompressed, num_bits) }
    }

    fn decompress_sorted(
        &self,
        initial: u64,
        compressed: &[u8],
        decompressed: &mut [u64],
        num_bits: u8,
    ) -> usize {
        unsafe {
            scalar::UnsafeBitPackerImpl::decompress_sorted(
                initial,
                compressed,
                decompressed,
                num_bits,
            )
        }
    }

    fn decompress_strictly_sorted(
        &self,
        initial: Option<u64>,
        compressed: &[u8],
        decompressed: &mut [u64],
        num_bits: u8,
    ) -> usize {
        unsafe {
            scalar::UnsafeBitPackerImpl::decompress_strictly_sorted(
                initial,
                compressed,
                decompressed,
                num_bits,
            )
        }
    }

    fn num_bits(&self, decompressed: &[u64]) -> u8 {
        unsafe { scalar::UnsafeBitPackerImpl::num_bits(decompressed) }
    }

    fn num_bits_sorted(&self, initial: u64, decompressed: &[u64]) -> u8 {
        unsafe { scalar::UnsafeBitPackerImpl::num_bits_sorted(initial, decompressed) }
    }

    fn num_bits_strictly_sorted(&self, initial: Option<u64>, decompressed: &[u64]) -> u8 {
        unsafe { scalar::UnsafeBitPackerImpl::num_bits_strictly_sorted(initial, decompressed) }
    }
}
