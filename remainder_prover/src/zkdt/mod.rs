//! Module that deals with the specific circuit to prove the result of a decision tree

/// Example structs for zkDT circuit
pub mod structs;
/// Concrete circuit implementation
pub mod data_pipeline;
pub mod helpers;
pub mod builders;
pub mod zkdt_circuit;
pub mod zkdt_multiple_tree_circuit;
/// For cache-ing and other auxiliary things
pub mod constants;
/// Binary recomp circuit component
pub mod binary_recomp_circuit;
/// For test circuits, mostly for debugging
pub mod test_circuits;

pub mod attribute_consistency_circuit;
pub mod permutation_circuit;
pub mod multiset_circuit;
/// Path consistency circuit component
pub mod path_consistency_circuit;
/// For the two "bits are binary" checkers
pub mod bits_are_binary_circuit;
pub mod input_multiset_circuit;
pub mod legacy_circuits;
pub mod input_data_to_circuit_adapter;
pub mod cache_upshot_catboost_inputs_for_testing;