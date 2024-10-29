//! This is a module for constructing an [InstantiatedCircuit] given a
//! [LayouterCircuit], which is a closure provided by the circuit builder that
//! takes in one parameter, the context, along with all of the "circuit
//! builders" necessary in order to represent the data-dependency relationships
//! within the circuit. Additionally, the circuit builder in this closure also
//! provides the necessary input data to populate the circuits with.
//!
//! The important distinction in this compilation process is that first, using
//! the data-dependency relationships in the circuit, a [GKRCircuitDescription]
//! is generated, which represents the data-dependencies in the circuit along
//! with the shape of the circuit itself (the number of variables in the
//! different MLEs that make up a layer).
//!
//! Then, using this circuit description along with the data inputs, the
//! `instantiate()` function will create an [InstantiatedCircuit], where now
//! the circuit description is "filled in" with its associated data.

#[cfg(test)]
mod tests;

/// Defines the type of hash used when adding a circuit description's hash
/// into the transcript.
pub enum CircuitHashType {
    /// This uses Rust's [DefaultHasher] implementation and uses the
    /// #[derive(Hash)] implementation. The hash function implemented
    /// underneath is not cryptographically secure, and thus this option
    /// is generally not recommended.
    DefaultRustHash,
    /// This converts the circuit description into a JSON string via
    /// [serde::Serialize] and hashes the bytes using [Sha3_256].
    Sha3_256,
    /// This converts the circuit description into a JSON string via
    /// [serde::Serialize] and hashes the bytes in chunks of 16, converting
    /// them first to field elements.
    Poseidon,
}
