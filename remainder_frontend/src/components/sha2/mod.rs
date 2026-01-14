pub mod brent_kung_adder;
pub mod nonlinear_gates;
/// Utilities related to Ripple Carry Adder
pub mod ripple_carry_adder;
pub mod sha256_bit_decomp;
use crate::layouter::builder::{Circuit, CircuitBuilder, InputLayerNodeRef, NodeRef};
use remainder_shared_types::Field;

/// Trait that's implemented by all SHA256 adders
pub trait AdderGateTrait<F: Field> {
    /// Data type of actual elements such as u32 (in sha256) or u64 (in
    /// sha512)
    type IntegralType: std::ops::Add<Output = Self::IntegralType>;

    /// Build the layout of the adder circuit. The `carry_layer`
    /// Auxiliary data is meant for passing in any additional layer or
    /// circuit or wire information that can help the prover.
    fn layout_adder_circuit(
        circuit_builder: &mut CircuitBuilder<F>, // Circuit builder
        x_node: &NodeRef,                        // reference to x in x + y
        y_node: &NodeRef,                        // reference to y in x + y
        carry_layer: Option<InputLayerNodeRef>,  // Carry Layer information
    ) -> Self;

    /// Returns a reference to the note that corresponds to the adder
    /// gate
    fn get_output(&self) -> NodeRef;

    /// Optional trait that compute the actual sum and create any possible commitments
    /// needed.
    fn perform_addition(
        &self,
        _circuit: &mut Circuit<F>,
        x: Self::IntegralType,
        y: Self::IntegralType,
    ) -> Self::IntegralType;
}
