use itertools::FoldWhile::{Continue, Done};
use itertools::{concat, Itertools};
use num::traits::ToBytes;

use crate::curves::PrimeOrderCurve;
use crate::field::HasByteRepresentation;

fn c_bit_scalar_mult<C: PrimeOrderCurve>(c: usize, base_vec: &[C], scalar_vec: &[u64]) -> C {
    // We create buckets for when the scalar is 1, up to 2^c - 1.
    let mut buckets: Vec<C> = vec![C::zero(); (1 << c) - 1];
    scalar_vec.iter().zip(base_vec).for_each(|(scalar, base)| {
        // Since this is a c-bit scalar, the scalar must fall in one of these buckets.
        debug_assert!(
            scalar.to_le_bytes().as_ref().len() * 8 - (scalar.leading_zeros() as usize) <= c
        );
        // We skip when the scalar is 0 because it won't contribute to the MSM.
        if *scalar != 0 {
            buckets[*scalar as usize - 1] += *base;
        }
    });
    // We perform an optimized addition of the \sum_i{i * value}
    // by doing a reverse sum and keeping track of running sum
    // of all the values along with the accumulation.
    // I.e., 3S_3 + 2S_2 + S_1 =
    // S_3 + (S_3 + S_2) + (S_3 + S_2 + S_1).
    let (sum, _) = buckets
        .iter()
        .rev()
        .fold((C::zero(), C::zero()), |(acc, prev), elem| {
            let t = prev + *elem;
            (acc + t, t)
        });
    sum
}

fn combine_c_bit_scalar_mults<C: PrimeOrderCurve>(c: usize, bucket_sums: &[C]) -> C {
    // We combine the c-bit scalar multiplications by going through
    // the bucket sums, which is in order of most-significant
    // contribution to the scalar multiplication, and keep adding it
    // to the accumulator while multiplying by 2^c. So we should get
    // \sum_i{2^{i*c} * bucket_val[n - i]}.
    if !bucket_sums.is_empty() {
        let all_but_last =
            bucket_sums
                .iter()
                .take(bucket_sums.len() - 1)
                .fold(C::zero(), |mut acc, elem| {
                    acc += *elem;
                    (0..c).for_each(|_idx| {
                        acc = acc.double();
                    });
                    acc
                });
        all_but_last + *bucket_sums.last().unwrap()
    } else {
        C::zero()
    }
}

/// Helper function to compute the number of bits in a scalar.
fn num_bits<C: PrimeOrderCurve>(n: C::Scalar) -> usize {
    let u64_chunks = n.to_u64s_le();
    debug_assert_eq!(u64_chunks.len(), 4);
    u64_chunks
        .iter()
        .rev()
        .fold_while(192_usize, |acc, chunk| {
            if *chunk == 0 {
                if acc == 0 {
                    Done(0)
                } else {
                    Continue(acc - 64)
                }
            } else {
                Done(acc + chunk.ilog2() as usize + 1)
            }
        })
        .into_inner()
}

/// Overall function to compute the MSM of a vector of
/// group elements to a vector of scalar field elements
/// using Pippenger's Algorithm.
///
/// `c_bucket_size` is the parameter into how we want to
/// split the larger MSM into smaller MSMs.
pub fn scalar_mult_pippenger<C: PrimeOrderCurve>(
    c_bucket_size: usize,
    base_vec: &[C],
    scalar_vec: &[C::Scalar],
) -> C {
    assert_eq!(scalar_vec.len(), base_vec.len());
    let n = scalar_vec.len();
    if n != 0 {
        let max_input_mle_value = scalar_vec.iter().max().unwrap();
        let max_num_bits_needed = num_bits::<C>(*max_input_mle_value);
        // We take the ceiling division to compute the window size.
        let num_buckets = max_num_bits_needed.div_ceil(c_bucket_size);
        let mut bucket_groups = vec![vec![0_u64; n]; num_buckets];
        scalar_vec.iter().enumerate().for_each(|(idx, elem)| {
            // We compute all the bits of the field elements.
            let elem_bits_full = elem
                .to_bytes_le()
                .iter()
                .rev()
                .flat_map(|byte| (0..8).rev().map(move |i| ((byte >> i) & 1u8) != 0))
                .collect_vec();
            let elem_bits_full_len = elem_bits_full.len();
            // We only use the most significant field elements rounded to include
            // the minimum number of full windows we can use.
            let elem_bits = if (num_buckets * c_bucket_size) < elem_bits_full_len {
                elem_bits_full
                    .into_iter()
                    .skip(elem_bits_full_len - (num_buckets * c_bucket_size))
                    .collect_vec()
            } else {
                let padding_len = (num_buckets * c_bucket_size) - elem_bits_full_len;
                let padding = vec![false; padding_len];
                concat(vec![padding, elem_bits_full])
            };
            // We create the buckets by iterating through each of the scalar
            // field elements and splitting them by their most significant
            // to least significant "c-bit chunk."
            (0..num_buckets).for_each(|bucket_idx| {
                let bits_in_bucket =
                    &elem_bits[(c_bucket_size * bucket_idx)..(c_bucket_size * (bucket_idx + 1))];
                let bucket_value = bits_in_bucket
                    .iter()
                    .rev()
                    .enumerate()
                    .map(|(i, &b)| (b as usize) << i)
                    .sum::<usize>();
                bucket_groups[bucket_idx][idx] = bucket_value as u64;
            });
        });
        // Perform the smaller MSMs.
        let c_bit_scalar_mults = bucket_groups
            .iter()
            .map(|bucket_group| c_bit_scalar_mult(c_bucket_size, base_vec, bucket_group))
            .collect_vec();
        // Combine all the smaller MSMs.
        combine_c_bit_scalar_mults(c_bucket_size, &c_bit_scalar_mults)
    } else {
        C::zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::halo2curves::bn256::G1 as Bn256;
    use halo2curves::ff::Field;
    type Scalar = <Bn256 as PrimeOrderCurve>::Scalar;

    fn naive_msm<C: PrimeOrderCurve>(scalar_vec: &[C::Scalar], base_vec: &[C]) -> C {
        scalar_vec
            .iter()
            .zip(base_vec)
            .fold(C::zero(), |acc, (scalar, base)| acc + *base * *scalar)
    }

    #[test]
    fn test_pippenger_1() {
        let mut rng = rand::thread_rng();
        let scalar_vec = vec![6_u64, 15_u64, 13_u64, 12_u64]
            .into_iter()
            .map(Scalar::from)
            .collect_vec();
        let base_vec = (0..4).map(|_idx| Bn256::random(&mut rng)).collect_vec();
        let naive_msm = naive_msm(&scalar_vec, &base_vec);
        let pip_msm = scalar_mult_pippenger(2, &base_vec, &scalar_vec);
        assert_eq!(naive_msm, pip_msm);
    }

    #[test]
    fn test_pippenger_2() {
        let mut rng = rand::thread_rng();
        let scalar_vec = vec![289041_u64, 114202_u64, 124023_u64, 858222_u64]
            .into_iter()
            .map(Scalar::from)
            .collect_vec();
        let base_vec = (0..4).map(|_idx| Bn256::random(&mut rng)).collect_vec();
        let naive_msm = naive_msm(&scalar_vec, &base_vec);
        let pip_msm = scalar_mult_pippenger(6, &base_vec, &scalar_vec);
        assert_eq!(naive_msm, pip_msm);
    }

    #[test]
    fn test_pippenger_3() {
        const NUM_ELEMS: usize = 20;
        let mut rng = rand::thread_rng();
        let scalar_vec = (0..NUM_ELEMS)
            .into_iter()
            .map(|_idx| Scalar::random(&mut rng))
            .collect_vec();
        let base_vec = (0..NUM_ELEMS)
            .map(|_idx| Bn256::random(&mut rng))
            .collect_vec();
        let naive_msm = naive_msm(&scalar_vec, &base_vec);
        let pip_msm = scalar_mult_pippenger(6, &base_vec, &scalar_vec);
        assert_eq!(naive_msm, pip_msm);
    }
}
