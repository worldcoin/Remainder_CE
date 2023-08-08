//! Module that deals with the specific circuit to prove the result of a decision tree

/// Example structs for zkDT circuit
pub mod structs;
/// Concrete circuit implementation
pub mod zkdt_circuit;
/// Conversion from decision tree model to circuit-ready form
pub mod dt2zkdt;
pub mod zkdt_layer;
