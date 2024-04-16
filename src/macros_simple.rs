macro_rules! declare_bitpacker_simple {
    ($cpufeature:meta) => {
        fn compute_num_bytes_per_block(num_bits: usize) -> usize {
            (num_bits * BLOCK_LEN) / 8
        }
        use super::UnsafeBitPacker;
        use crate::most_significant_bit;

        pub unsafe fn pack<TDeltaComputer: Transformer>(
            input_arr: &[u64],
            output_arr: &mut [u8],
            num_bits: usize,
            mut delta_computer: TDeltaComputer,
        ) -> usize {
            let num_bytes_per_block = compute_num_bytes_per_block(num_bits);
            assert_eq!(
                input_arr.len(),
                BLOCK_LEN,
                "Input block too small {}, (expected {})",
                input_arr.len(),
                BLOCK_LEN
            );
            assert!(
                output_arr.len() >= num_bytes_per_block,
                "Output array too small (numbits {}). {} <= {}",
                num_bits,
                output_arr.len(),
                num_bytes_per_block
            );

            let input_ptr = input_arr.as_ptr();
            let mut output_ptr = output_arr.as_mut_ptr() as *mut DataType;
            let mut out_register: DataType = delta_computer.transform(set1(*input_ptr));

            for i in 1..BLOCK_LEN {
                let bits_filled: usize = i * num_bits;
                let inner_cursor: usize = bits_filled % 64;
                let remaining: usize = 64 - inner_cursor;

                let offset_ptr = input_ptr.add(i);
                let in_register: DataType = delta_computer.transform(set1(*offset_ptr));

                out_register = if inner_cursor > 0 {
                    let shifted = left_shift_64(in_register, inner_cursor as i32);
                    op_or(out_register, shifted)
                } else {
                    in_register
                };

                if remaining <= num_bits {
                    store_unaligned(output_ptr, out_register);
                    output_ptr = output_ptr.add(1);
                    if remaining < num_bits {
                        out_register = right_shift_64(in_register, remaining as i32);
                    }
                }
            }

            let in_register: DataType =
                delta_computer.transform(set1(*input_ptr.add(BLOCK_LEN - 1)));
            let shifted = left_shift_64(in_register, (64 - num_bits) as i32);
            out_register = op_or(out_register, shifted);
            store_unaligned(output_ptr, out_register);
            num_bytes_per_block
        }

        pub unsafe fn pack_64<TDeltaComputer: Transformer>(
            input_arr: &[u64],
            output_arr: &mut [u8],
            mut delta_computer: TDeltaComputer,
        ) -> usize {
            assert_eq!(
                input_arr.len(),
                BLOCK_LEN,
                "Input block too small {}, (expected {})",
                input_arr.len(),
                BLOCK_LEN
            );
            let num_bytes_per_block = compute_num_bytes_per_block(64);
            assert!(
                output_arr.len() >= num_bytes_per_block,
                "Output array too small (numbits {}). {} <= {}",
                64,
                output_arr.len(),
                num_bytes_per_block
            );

            let input_ptr: *const DataType = input_arr.as_ptr() as *const DataType;
            let output_ptr = output_arr.as_mut_ptr() as *mut DataType;
            for i in 0..BLOCK_LEN {
                let input_offset_ptr = input_ptr.add(i);
                let output_offset_ptr = output_ptr.add(i);
                let input_register = load_unaligned(input_offset_ptr);
                let output_register = delta_computer.transform(input_register);
                store_unaligned(output_offset_ptr, output_register);
            }
            num_bytes_per_block
        }

        pub unsafe fn unpack<Output: Sink>(
            compressed: &[u8],
            mut output: Output,
            num_bits: usize,
        ) -> usize {
            let num_bytes_per_block = compute_num_bytes_per_block(num_bits);
            assert!(
                compressed.len() >= num_bytes_per_block,
                "Compressed array seems too small. ({} < {}) ",
                compressed.len(),
                num_bytes_per_block
            );

            let mut input_ptr = compressed.as_ptr() as *const DataType;

            let mask_scalar: u64 = ((1u128 << num_bits) - 1u128) as u64;
            let mask = set1(mask_scalar);

            let mut in_register: DataType = load_unaligned(input_ptr);

            let out_register = op_and(in_register, mask);
            output.process(out_register);

            for i in 1..BLOCK_LEN {
                let inner_cursor: usize = (i * num_bits) % 64;
                let inner_capacity: usize = 64 - inner_cursor;

                let shifted_in_register = right_shift_64(in_register, inner_cursor as i32);
                let mut out_register: DataType = op_and(shifted_in_register, mask);

                if inner_capacity <= num_bits && i != BLOCK_LEN - 1 {
                    input_ptr = input_ptr.add(1);
                    in_register = load_unaligned(input_ptr);

                    if inner_capacity < num_bits {
                        let shifted =
                            op_and(left_shift_64(in_register, inner_capacity as i32), mask);
                        out_register = op_or(out_register, shifted);
                    }
                }

                output.process(out_register);
            }

            // Handle the last block
            input_ptr = input_ptr.add(1);
            in_register = load_unaligned(input_ptr);
            let shifted = op_and(left_shift_64(in_register, (64 - num_bits) as i32), mask);
            output.process(shifted);

            num_bytes_per_block
        }

        pub unsafe fn unpack_64<Output: Sink>(compressed: &[u8], mut output: Output) -> usize {
            let num_bytes_per_block = compute_num_bytes_per_block(64);
            assert!(
                compressed.len() >= num_bytes_per_block,
                "Compressed array seems too small. ({} < {}) ",
                compressed.len(),
                num_bytes_per_block
            );
            let input_ptr = compressed.as_ptr() as *const DataType;
            for i in 0..BLOCK_LEN {
                let input_offset_ptr = input_ptr.add(i);
                let in_register: DataType = load_unaligned(input_offset_ptr);
                output.process(in_register);
            }
            num_bytes_per_block
        }

        pub trait Transformer {
            unsafe fn transform(&mut self, data: DataType) -> DataType;
        }

        struct NoDelta;

        impl Transformer for NoDelta {
            unsafe fn transform(&mut self, current: DataType) -> DataType {
                current
            }
        }

        struct DeltaComputer {
            pub previous: DataType,
        }

        impl Transformer for DeltaComputer {
            unsafe fn transform(&mut self, current: DataType) -> DataType {
                let result = compute_delta(current, self.previous);
                self.previous = current;
                result
            }
        }

        struct StrictDeltaComputer {
            pub previous: DataType,
        }

        impl Transformer for StrictDeltaComputer {
            #[inline]
            unsafe fn transform(&mut self, current: DataType) -> DataType {
                let result = compute_delta(current, self.previous);
                self.previous = current;
                sub(result, set1(1))
            }
        }

        pub trait Sink {
            unsafe fn process(&mut self, data_type: DataType);
        }

        struct Store {
            output_ptr: *mut DataType,
        }

        impl Store {
            fn new(output_ptr: *mut DataType) -> Store {
                Store { output_ptr }
            }
        }

        struct DeltaIntegrate {
            current: DataType,
            output_ptr: *mut DataType,
        }

        impl DeltaIntegrate {
            unsafe fn new(initial: u64, output_ptr: *mut DataType) -> DeltaIntegrate {
                DeltaIntegrate {
                    current: set1(initial),
                    output_ptr,
                }
            }
        }

        impl Sink for DeltaIntegrate {
            #[inline]
            unsafe fn process(&mut self, delta: DataType) {
                self.current = integrate_delta(self.current, delta);
                store_unaligned(self.output_ptr, self.current);
                self.output_ptr = self.output_ptr.add(1);
            }
        }

        struct StrictDeltaIntegrate {
            current: DataType,
            output_ptr: *mut DataType,
        }

        impl StrictDeltaIntegrate {
            unsafe fn new(initial: u64, output_ptr: *mut DataType) -> StrictDeltaIntegrate {
                StrictDeltaIntegrate {
                    current: set1(initial),
                    output_ptr,
                }
            }
        }

        impl Sink for StrictDeltaIntegrate {
            #[inline]
            unsafe fn process(&mut self, delta: DataType) {
                self.current = integrate_delta(self.current, add(delta, set1(1)));
                store_unaligned(self.output_ptr, self.current);
                self.output_ptr = self.output_ptr.add(1);
            }
        }

        impl Sink for Store {
            #[inline]
            unsafe fn process(&mut self, out_register: DataType) {
                store_unaligned(self.output_ptr, out_register);
                self.output_ptr = self.output_ptr.add(1);
            }
        }

        pub struct UnsafeBitPackerImpl;

        impl UnsafeBitPacker for UnsafeBitPackerImpl {
            const BLOCK_LEN: usize = BLOCK_LEN;

            unsafe fn compress(decompressed: &[u64], compressed: &mut [u8], num_bits: u8) -> usize {
                if num_bits == 0u8 {
                    return 0;
                }
                if num_bits == 64u8 {
                    return pack_64(decompressed, compressed, NoDelta);
                }
                pack(decompressed, compressed, num_bits as usize, NoDelta)
            }

            unsafe fn compress_sorted(
                initial: u64,
                decompressed: &[u64],
                compressed: &mut [u8],
                num_bits: u8,
            ) -> usize {
                if num_bits == 0u8 {
                    return 0;
                }
                let delta_computer = DeltaComputer {
                    previous: set1(initial),
                };
                if num_bits == 64u8 {
                    return pack_64(decompressed, compressed, delta_computer);
                }
                pack(decompressed, compressed, num_bits as usize, delta_computer)
            }

            unsafe fn compress_strictly_sorted(
                initial: Option<u64>,
                decompressed: &[u64],
                compressed: &mut [u8],
                num_bits: u8,
            ) -> usize {
                let initial = initial.unwrap_or(u64::MAX);
                if num_bits == 0u8 {
                    return 0;
                }
                let delta_computer = StrictDeltaComputer {
                    previous: set1(initial),
                };
                if num_bits == 64u8 {
                    return pack_64(decompressed, compressed, delta_computer);
                }
                pack(decompressed, compressed, num_bits as usize, delta_computer)
            }

            unsafe fn decompress(
                compressed: &[u8],
                decompressed: &mut [u64],
                num_bits: u8,
            ) -> usize {
                assert!(
                    decompressed.len() >= BLOCK_LEN,
                    "The output array is not large enough : ({} >= {})",
                    decompressed.len(),
                    BLOCK_LEN
                );
                let output_ptr = decompressed.as_mut_ptr() as *mut DataType;
                let mut output = Store::new(output_ptr);
                if num_bits == 0u8 {
                    let zero = set1(0u64);
                    for _ in 0..BLOCK_LEN {
                        output.process(zero);
                    }
                    return 0;
                }
                if num_bits == 64u8 {
                    return unpack_64(compressed, output);
                }
                unpack(compressed, output, num_bits as usize)
            }

            unsafe fn decompress_sorted(
                initial: u64,
                compressed: &[u8],
                decompressed: &mut [u64],
                num_bits: u8,
            ) -> usize {
                assert!(
                    decompressed.len() >= BLOCK_LEN,
                    "The output array is not large enough : ({} >= {})",
                    decompressed.len(),
                    BLOCK_LEN
                );
                let output_ptr = decompressed.as_mut_ptr() as *mut DataType;
                let mut output = DeltaIntegrate::new(initial, output_ptr);
                if num_bits == 0u8 {
                    let zero = set1(0u64);
                    for _ in 0..BLOCK_LEN {
                        output.process(zero);
                    }
                    return 0;
                }
                if num_bits == 64u8 {
                    return unpack_64(compressed, output);
                }
                unpack(compressed, output, num_bits as usize)
            }

            unsafe fn decompress_strictly_sorted(
                initial: Option<u64>,
                compressed: &[u8],
                decompressed: &mut [u64],
                num_bits: u8,
            ) -> usize {
                assert!(
                    decompressed.len() >= BLOCK_LEN,
                    "The output array is not large enough : ({} >= {})",
                    decompressed.len(),
                    BLOCK_LEN
                );
                let initial = initial.unwrap_or(u64::MAX);
                let output_ptr = decompressed.as_mut_ptr() as *mut DataType;
                let mut output = StrictDeltaIntegrate::new(initial, output_ptr);
                if num_bits == 0u8 {
                    let zero = set1(0u64);
                    for _ in 0..BLOCK_LEN {
                        output.process(zero);
                    }
                    return 0;
                }
                if num_bits == 64u8 {
                    return unpack_64(compressed, output);
                }
                unpack(compressed, output, num_bits as usize)
            }

            unsafe fn num_bits(decompressed: &[u64]) -> u8 {
                assert_eq!(
                    decompressed.len(),
                    BLOCK_LEN,
                    "`decompressed`'s len is not `BLOCK_LEN={}`",
                    BLOCK_LEN
                );

                let data: *const DataType = decompressed.as_ptr() as *const DataType;
                let mut accumulator = load_unaligned(data);

                for i in 1..BLOCK_LEN {
                    let newvec = load_unaligned(data.add(i));
                    accumulator = op_or(accumulator, newvec);
                }

                most_significant_bit(or_collapse_to_u64(accumulator))
            }

            unsafe fn num_bits_sorted(initial: u64, decompressed: &[u64]) -> u8 {
                let initial_vec = set1(initial);
                let data: *const DataType = decompressed.as_ptr() as *const DataType;
                let first = load_unaligned(data);
                let mut accumulator = compute_delta(load_unaligned(data), initial_vec);
                let mut previous = first;

                for i in 1..BLOCK_LEN - 1 {
                    let current = load_unaligned(data.add(i));
                    let delta = compute_delta(current, previous);
                    accumulator = op_or(accumulator, delta);
                    previous = current;
                }

                let current = load_unaligned(data.add(BLOCK_LEN - 1));
                let delta = compute_delta(current, previous);
                accumulator = op_or(accumulator, delta);
                most_significant_bit(or_collapse_to_u64(accumulator))
            }

            unsafe fn num_bits_strictly_sorted(initial: Option<u64>, decompressed: &[u64]) -> u8 {
                let initial = initial.unwrap_or(u64::MAX);
                let initial_vec = set1(initial);
                let data: *const DataType = decompressed.as_ptr() as *const DataType;
                let first = load_unaligned(data);
                let one = set1(1);
                let mut accumulator = sub(compute_delta(load_unaligned(data), initial_vec), one);
                let mut previous = first;

                for i in 1..BLOCK_LEN - 1 {
                    let current = load_unaligned(data.add(i));
                    let delta = sub(compute_delta(current, previous), one);
                    accumulator = op_or(accumulator, delta);
                    previous = current;
                }

                let current = load_unaligned(data.add(BLOCK_LEN - 1));
                let delta = sub(compute_delta(current, previous), one);
                accumulator = op_or(accumulator, delta);
                most_significant_bit(or_collapse_to_u64(accumulator))
            }
        }

        #[cfg(test)]
        mod tests {
            use super::UnsafeBitPackerImpl;
            use crate::tests::{test_suite_compress_decompress, DeltaKind};
            use crate::UnsafeBitPacker;
            const BLOCK_LEN: usize = 32 * 4;

            #[test]
            fn test_num_bits() {
                for num_bits in 0..64 {
                    for pos in 0..BLOCK_LEN {
                        let mut vals = [0u64; UnsafeBitPackerImpl::BLOCK_LEN];
                        if num_bits > 0 {
                            vals[pos] = 1 << (num_bits - 1);
                        }
                        assert_eq!(
                            unsafe { UnsafeBitPackerImpl::num_bits(&vals[..]) },
                            num_bits
                        );
                    }
                }
            }

            #[test]
            fn test_num_bits_sorted() {
                for initial in &[0u64, 1, 100, u64::MAX] {
                    for num_bits in 0..64 {
                        for pos in 0..BLOCK_LEN {
                            let mut vals = [0u64; UnsafeBitPackerImpl::BLOCK_LEN];
                            if num_bits > 0 {
                                vals[pos] = *initial + (1 << (num_bits - 1));
                            }
                            assert_eq!(
                                unsafe {
                                    UnsafeBitPackerImpl::num_bits_sorted(*initial, &vals[..])
                                },
                                num_bits
                            );
                        }
                    }
                }
            }

            #[test]
            fn test_num_bits_strictly_sorted() {
                for initial in &[Some(0u64), Some(1), Some(100), None] {
                    for num_bits in 0..64 {
                        for pos in 0..BLOCK_LEN {
                            let mut vals = [0u64; UnsafeBitPackerImpl::BLOCK_LEN];
                            if num_bits > 0 {
                                let base = initial.unwrap_or(0);
                                vals[pos] = base + (1 << (num_bits - 1));
                            }
                            assert_eq!(
                                unsafe {
                                    UnsafeBitPackerImpl::num_bits_strictly_sorted(
                                        *initial,
                                        &vals[..],
                                    )
                                },
                                num_bits
                            );
                        }
                    }
                }
            }

            #[test]
            fn test_bitpacker_nodelta() {
                test_suite_compress_decompress::<UnsafeBitPackerImpl>(DeltaKind::NoDelta);
            }

            #[test]
            fn test_bitpacker_delta() {
                test_suite_compress_decompress::<UnsafeBitPackerImpl>(DeltaKind::Delta);
            }

            #[test]
            fn test_bitpacker_strict_delta() {
                test_suite_compress_decompress::<UnsafeBitPackerImpl>(DeltaKind::StrictDelta);
            }
        }
    };
}
