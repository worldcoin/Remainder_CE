//! Module that deals with the specific circuit to prove the result of a decision tree

/// Example structs for zkDT circuit
pub mod structs;
/// Concrete circuit implementation
pub mod zkdt_helpers;
pub mod dt2zkdt;
pub mod trees;
pub mod helpers;
pub mod zkdt_layer;
pub mod zkdt_circuit_parts;
pub mod zkdt_circuit;
/// For cache-ing and other auxiliary things
pub mod constants;
/// Binary recomp circuit component
pub mod binary_recomp_circuit;
/// For test circuits, mostly for debugging
pub mod test_circuits;