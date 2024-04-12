/// Version of Digest (as found in standard Rust library)
/// but working with Poseidon hash function and FieldExt
pub mod poseidon_digest;

use self::poseidon_digest::FieldHashFnDigest;
use remainder_shared_types::FieldExt;
use remainder_shared_types::Poseidon;
use std::convert::TryInto;
use std::marker::PhantomData;

/// Poseidon Sponge Hasher struct with implementation for FieldHashFnDigest
#[derive(Clone)]
pub struct PoseidonSpongeHasher<F: FieldExt> {
    halo2_sponge: Poseidon<F, 3, 2>,
    phantom_data: PhantomData<F>,
}

// TODO!(ryancao): Do the fix where you just make two different configs

// ------------------------ FOR HALO2 POSEIDON ------------------------
// NOTE: These are SUPER slow. Don't use them except in tiny tests
fn get_new_halo2_sponge<F: FieldExt>() -> Poseidon<F, 3, 2> {
    Poseidon::<F, 3, 2>::new(8, 57)
}

fn get_new_halo2_sponge_with_params<F: FieldExt>(
    poseidon_params: PoseidonParams,
) -> Poseidon<F, 3, 2> {
    Poseidon::<F, 3, 2>::new(poseidon_params.full_rounds, poseidon_params.partial_rounds)
}

fn get_new_column_poseidon_sponge<F: FieldExt>() -> Poseidon<F, 3, 2> {
    Poseidon::<F, 3, 2>::new(8, 63)
}

fn get_new_merkle_poseidon_sponge<F: FieldExt>() -> Poseidon<F, 3, 2> {
    Poseidon::<F, 3, 2>::new(8, 57)
}

/// Parameters to pass into a new Poseidon hasher construct
pub struct PoseidonParams {
    full_rounds: usize,
    partial_rounds: usize,
    rate: usize,
    width: usize,
}

impl PoseidonParams {
    /// Constructs a new PoseidonParams
    pub fn new(full_rounds: usize, partial_rounds: usize, rate: usize, width: usize) -> Self {
        Self {
            full_rounds,
            partial_rounds,
            rate,
            width,
        }
    }
}

impl<F: FieldExt> FieldHashFnDigest<F> for PoseidonSpongeHasher<F> {
    type HashFnParams = PoseidonParams;

    // --- DON'T USE THESE ---
    fn new() -> Self {
        Self {
            // sponge: get_new_sponge(),
            halo2_sponge: get_new_halo2_sponge(),
            phantom_data: PhantomData,
        }
    }

    fn new_with_params(params: Self::HashFnParams) -> Self {
        Self {
            // sponge: get_new_sponge_with_params(params.full_rounds, params.partial_rounds, params.rate)
            halo2_sponge: get_new_halo2_sponge_with_params(params),
            phantom_data: PhantomData,
        }
    }

    // --- USE THESE INSTEAD ---
    fn new_merkle_hasher(static_merkle_poseidon_sponge: &Poseidon<F, 3, 2>) -> Self {
        Self {
            // halo2_sponge: MERKLE_POSEIDON_SPONGE.clone(),
            halo2_sponge: static_merkle_poseidon_sponge.clone(),
            phantom_data: PhantomData,
        }
    }

    fn new_column_hasher(static_column_poseidon_sponge: &Poseidon<F, 3, 2>) -> Self {
        Self {
            // halo2_sponge: COLUMN_POSEIDON_SPONGE.clone(),
            halo2_sponge: static_column_poseidon_sponge.clone(),
            phantom_data: PhantomData,
        }
    }

    // --- TODO!(ryancao): Add error/Result stuff to this? ---
    fn update(&mut self, data: &[F]) {
        // self.sponge.absorb(&data);
        self.halo2_sponge.update(data);
    }

    // --- TODO!(ryancao): What is the point of this again? ---
    fn chain(mut self, data: &[F]) -> Self
    where
        Self: Sized,
    {
        self.update(data);
        self
    }

    // --- Returns the single element squeezed from the sponge after absorbing ---
    fn finalize(mut self) -> F {
        let result: F = self.halo2_sponge.squeeze().try_into().unwrap();
        result
    }

    // --- For now, just calls finalize() and then reset() ---
    fn finalize_reset(&mut self) -> F {
        let output = self.halo2_sponge.squeeze();
        self.reset();
        output
    }

    fn reset(&mut self) {
        // --- Basically just resets the sponge ---
        self.halo2_sponge = get_new_halo2_sponge();
    }

    // --- Output size is always 1 ---
    fn output_size() -> usize {
        1
    }

    // --- Make a new sponge, absorb everything, then squeeze a single field element ---
    fn digest(data: &[F]) -> F {
        // let mut sponge = get_new_sponge::<F>();
        // sponge.absorb(&data);
        // sponge.squeeze_native_field_elements(1)[0]
        let mut sponge = get_new_halo2_sponge();
        sponge.update(data);
        sponge.squeeze()
    }
}

/// For Merkle tree hashing stuff
pub trait MerkleCRHHasher<F: FieldExt> {
    /// TODO!(ryancao): Trait bounds on this?
    type Output;

    /// Creates a new instance of the Merkle hasher
    fn new(label: &'static str) -> Self;

    /// TODO!(ryancao): Should we propagate the error?
    fn evaluate(&self, left_input: F, right_input: F) -> Self::Output;
}

// /// Wrapper for Digest trait, so that we can Merkle-ize using Riad's code.
// #[derive(Clone, Default)]
// pub struct PoseidonMerkleHasher<F: FieldExt> {
//     // poseidon_crh: PoseidonTwoToOneCRH<F>,
//     // poseidon_params: PoseidonConfig<F>,
//     _phantom_data: PhantomData<F>
// }

// impl<F: FieldExt> MerkleCRHHasher<F> for PoseidonMerkleHasher<F> {

//     type Output = F;

//     fn new(label: &'static str) -> Self {
//         // TODO!(ryancao): Fix this as Nick would
//         let (ark, mds) = find_poseidon_ark_and_mds::<F>(F::MODULUS_BIT_SIZE as u64, 2, 8, 60, 0);

//         let params = PoseidonConfig::new(8, 60, 5, mds, ark, 2, 1);
//         Self {
//             // poseidon_crh: PoseidonTwoToOneCRH {
//             //     field_phantom: std::marker::PhantomData
//             // },
//             // poseidon_params: params,
//             _phantom_data: PhantomData,
//         }
//     }

//     fn evaluate(&self, left_input: F, right_input: F) -> Self::Output {
//         <PoseidonTwoToOneCRH<F> as TwoToOneCRHScheme>::evaluate(&self.poseidon_params, left_input, right_input).unwrap()
//     }
// }

// #[cfg(test)]
// mod tests {

//     use ark_bn254::Fr;
//     use ark_std::{Zero, One};
//     // use ark_crypto_primitives::crh::poseidon::CRH as PoseidonCRH;
//     use ark_crypto_primitives::crh::{CRHScheme, TwoToOneCRHScheme};

//     use super::{PoseidonMerkleHasher, MerkleCRHHasher};

//     #[test]
//     fn test_poseidon() {

//         let zero = Fr::zero();
//         let one = Fr::one();
//         let vec = vec![zero, one];
//         let poseidon_hasher: PoseidonMerkleHasher<Fr> = PoseidonMerkleHasher::new("test");
//         // TODO!(ryancao): How can we cast this to a `TwoToOneCRHScheme`?
//         let hi = poseidon_hasher.evaluate(zero, one);
//         dbg!(hi);
//     }
// }
