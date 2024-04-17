/// Version of Digest (as found in standard Rust library)
/// but working with Poseidon hash function and FieldExt
pub mod poseidon_digest;

use self::poseidon_digest::FieldHashFnDigest;
use remainder_shared_types::FieldExt;
use remainder_shared_types::Poseidon;
use std::convert::TryInto;
use std::marker::PhantomData;

/// Wrapper around [Poseidon] sponge which provides an implementation
/// for [FieldHashFnDigest] (the trait whose implementation is necessary for
/// usage within the column hashing/Merkle hashing within Ligero).
#[derive(Clone)]
pub struct PoseidonSpongeHasher<F: FieldExt> {
    halo2_sponge: Poseidon<F, 3, 2>,
    phantom_data: PhantomData<F>,
}

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

/// Parameters to pass into a new Poseidon hasher construct.
pub struct PoseidonParams {
    /// Number of full s-box rounds.
    full_rounds: usize,
    /// Number of partial rounds, i.e. the s-box (e.g. x^5) is applied to only
    /// a single element within the state rather than the whole state.
    partial_rounds: usize,
    /// Number of elements which can be absorbed at the same time.
    _rate: usize,
    /// Total vector length of the Poseidon state (rate + capacity).
    _width: usize,
}

impl PoseidonParams {
    /// Generic constructor from given parameters
    pub fn new(full_rounds: usize, partial_rounds: usize, rate: usize, width: usize) -> Self {
        Self {
            full_rounds,
            partial_rounds,
            _rate: rate,
            _width: width,
        }
    }
}

impl<F: FieldExt> FieldHashFnDigest<F> for PoseidonSpongeHasher<F> {
    type HashFnParams = PoseidonParams;

    /// Initializes with generic Halo2 [Poseidon] sponge.
    ///
    /// USED FOR TESTING ONLY! INEFFICIENT!
    fn new() -> Self {
        Self {
            halo2_sponge: get_new_halo2_sponge(),
            phantom_data: PhantomData,
        }
    }

    /// Initializes with Halo2 [Poseidon] sponge with specified parameters.
    ///
    /// USED FOR TESTING ONLY! INEFFICIENT!
    fn new_with_params(params: Self::HashFnParams) -> Self {
        Self {
            halo2_sponge: get_new_halo2_sponge_with_params(params),
            phantom_data: PhantomData,
        }
    }

    /// Initializes with a global/static Halo2 [Poseidon] sponge for efficiency.
    ///
    /// Note that this is the Merkle hasher, i.e. the hasher for the Merkle tree
    /// component of the Ligero commitment.
    fn new_merkle_hasher(static_merkle_poseidon_sponge: &Poseidon<F, 3, 2>) -> Self {
        Self {
            halo2_sponge: static_merkle_poseidon_sponge.clone(),
            phantom_data: PhantomData,
        }
    }

    /// Initializes with a global/static Halo2 [Poseidon] sponge for efficiency.
    ///
    /// Note that this is the column hasher, i.e. the hasher for the matrix
    /// columns component of the Ligero commitment.
    fn new_column_hasher(static_column_poseidon_sponge: &Poseidon<F, 3, 2>) -> Self {
        Self {
            halo2_sponge: static_column_poseidon_sponge.clone(),
            phantom_data: PhantomData,
        }
    }

    /// Absorbs the given data into the current sponge.
    fn update(&mut self, data: &[F]) {
        self.halo2_sponge.update(data);
    }

    /// Returns the single element squeezed from the sponge after absorbing.
    fn finalize(mut self) -> F {
        let result: F = self.halo2_sponge.squeeze().try_into().unwrap();
        result
    }

    /// Returns a final value squeezed from the current sponge and resets the state.
    fn finalize_reset(&mut self) -> F {
        let output = self.halo2_sponge.squeeze();
        self.reset();
        output
    }

    /// Resets the sponge state.
    fn reset(&mut self) {
        self.halo2_sponge = get_new_halo2_sponge();
    }

    /// Sponge output is a single field element at a time.
    fn output_size() -> usize {
        1
    }

    /// Make a new sponge, absorb everything, then squeeze a single field element
    fn digest(data: &[F]) -> F {
        let mut sponge = get_new_halo2_sponge();
        sponge.update(data);
        sponge.squeeze()
    }
}
