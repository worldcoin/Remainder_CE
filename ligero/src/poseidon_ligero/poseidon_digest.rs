use shared_types::Field;
use shared_types::Poseidon;

/// FieldHashFnDigest gives you a trait which is basically `Digest` but compatible
/// with Field-based hash functions.
pub trait FieldHashFnDigest<F: Field> {
    /// Parameters to pass into `new_with_params` such that a digest instance
    /// with specific hash function parameters can be created.
    type HashFnParams;

    /// (FieldHashFnDigest) Create new hasher instance with default params
    fn new() -> Self;

    /// (FieldHashFnDigest) Create new hasher instance but with given params
    fn new_with_params(params: Self::HashFnParams) -> Self;

    /// (FieldHashFnDigest) Create new hasher instance with default params for Merkle hashing
    fn new_merkle_hasher(static_merkle_poseidon_hasher: &Poseidon<F, 3, 2>) -> Self;

    /// (FieldHashFnDigest) Create new hasher instance with default params for column hashing
    fn new_column_hasher(static_column_poseidon_hasher: &Poseidon<F, 3, 2>) -> Self;

    /// (FieldHashFnDigest) Digest data, updating the internal state.
    ///
    /// This method can be called repeatedly for use with streaming messages.
    fn update(&mut self, data: &[F]);

    /// (FieldHashFnDigest) Retrieve result and consume hasher instance.
    fn finalize(self) -> F;

    /// (FieldHashFnDigest) Retrieve result and reset hasher instance.
    ///
    /// This method sometimes can be more efficient compared to hasher
    /// re-creation.
    fn finalize_reset(&mut self) -> F;

    /// (FieldHashFnDigest) Reset hasher instance to its initial state.
    fn reset(&mut self);

    /// (FieldHashFnDigest) Get output size of the hasher
    fn output_size() -> usize;

    /// (FieldHashFnDigest) Convenience function to compute hash of the `data`. It will handle
    /// hasher creation, data feeding and finalization.
    fn digest(data: &[F]) -> F;
}
