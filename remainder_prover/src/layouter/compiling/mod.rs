#[cfg(test)]
mod tests;

/// Defines the type of hash used when adding a circuit description's hash
/// into the transcript.
pub enum CircuitHashType {
    /// This uses Rust's [std::collections::hash_map::DefaultHasher] implementation and uses the
    /// #[derive(Hash)] implementation. The hash function implemented underneath is not
    /// cryptographically secure, and thus this option is generally not recommended.
    DefaultRustHash,
    /// This converts the circuit description into a JSON string via
    /// [serde::Serialize] and hashes the bytes using [Sha3_256].
    Sha3_256,
    /// This converts the circuit description into a JSON string via
    /// [serde::Serialize] and hashes the bytes in chunks of 16, converting
    /// them first to field elements.
    Poseidon,
}
