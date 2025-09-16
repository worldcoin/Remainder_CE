#![allow(clippy::type_complexity)]
use ark_std::log2;
use remainder_shared_types::Field;
use serde::{Deserialize, Serialize};

use crate::{
    layouter::builder::{Circuit, CircuitBuilder, LayerVisibility},
    worldcoin_mpc::{
        components::WorldcoinMpcComponents,
        parameters::{GR4_ELEM_BIT_LENGTH, GR4_MULTIPLICATION_WIRINGS},
    },
};
use remainder::{
    circuit_layout::ProvableCircuit, input_layer::InputLayerDescription,
    layer::gate::BinaryOperation, prover::GKRCircuitDescription,
};

use super::{
    data::SecretShareCircuitInputs,
    parameters::{ENCODING_MATRIX_NUM_VARS_COLS, ENCODING_MATRIX_NUM_VARS_ROWS},
};

use anyhow::Result;

/// Description of a `HyraxProof` for the mpc circuit.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "F: Field")]
pub struct MPCCircuitDescription<F: Field> {
    /// The circuit description.
    pub circuit_description: GKRCircuitDescription<F>,

    /// The private input layer for the slope (private and generated randomly).
    pub slope_input_layer: InputLayerDescription,

    /// The private input layer for the iris code (private), obtained through
    /// running the iriscode circuit
    pub iris_code_input_layer: InputLayerDescription,

    /// The private input layer for the mask code (private),  obtained through
    /// running the iriscode circuit
    pub mask_code_input_layer: InputLayerDescription,

    /// The input layer for quotients and multiplicities
    pub auxilary_input_layer: InputLayerDescription,

    /// The input layer for shares_reduced_modulo_gr4_modulus, public
    pub shares_input_layer: InputLayerDescription,

    /// all other public inputs, such as the evaluation points, the encoding matrix,
    /// the lookup_table_values. These should stay invariant for the same party, different
    /// iris/mask codes (i.e. left/right eye)
    pub auxiliary_invariant_public_input_layer: InputLayerDescription,
}

/// Builds the mpc circuit.
/// The full circuit spec can be referenced here:
/// <https://www.notion.so/MPC-Circuit-Builders-8fe15c8d8f4b4db18cbf85344ccbb9df?pvs=4>
///
/// To summerize, this circuit takes in slope, iris code and mask code as private inputs,
/// computes the masked iris code, encodes the masked iris code to a GR4 element.
/// Then, the circuit does a GR4 multiplication between the evaluation points and the slopes.
/// The result is then summed with the masked iris code to get the computed shares.
/// The quotients, shares_reduced_modulo_gr4_modulus are to count for congruency (modulo 2^16).
pub fn build_circuit<F: Field, const NUM_IRIS_4_CHUNKS: usize>(
    layer_visibility: LayerVisibility,
) -> Circuit<F> {
    let mut builder = CircuitBuilder::<F>::new();

    let num_vars_dataparallel = log2(NUM_IRIS_4_CHUNKS) as usize;
    // the 2 extra num_vars are for the 4 chunks of the iris code (every 4 iris code values
    // are combined into a single GR4 elements)
    let num_vars = num_vars_dataparallel + 2;

    let auxilary_input_layer_node = builder.add_input_layer("Aux", LayerVisibility::Public);
    let shares_input_layer_node = builder.add_input_layer("Shares", LayerVisibility::Public);
    let auxiliary_invariant_public_input_layer_node =
        builder.add_input_layer("Auxiliary Invariant", LayerVisibility::Public);
    let slope_input_layer_node = builder.add_input_layer("Slope", layer_visibility);
    let iris_code_input_layer_node = builder.add_input_layer("Irir Code Input", layer_visibility);
    let mask_code_input_layer_node = builder.add_input_layer("Mask Code Input", layer_visibility);

    let iris_code = builder.add_input_shred("Iriscode", num_vars, &iris_code_input_layer_node);
    let mask_code = builder.add_input_shred("Maskcode", num_vars, &mask_code_input_layer_node);
    let slopes = builder.add_input_shred("Slopes", num_vars, &slope_input_layer_node);
    let quotients = builder.add_input_shred("Quotients", num_vars, &auxilary_input_layer_node);
    let shares_reduced_modulo_gr4_modulus = builder.add_input_shred(
        "Shares Reduced Modulo GR4 Modulus",
        num_vars,
        &shares_input_layer_node,
    );
    let multiplicities_shares = builder.add_input_shred(
        "Multiplicities Shares",
        GR4_ELEM_BIT_LENGTH as usize,
        &auxilary_input_layer_node,
    );
    let multiplicities_slopes = builder.add_input_shred(
        "Multiplicities Slopes",
        GR4_ELEM_BIT_LENGTH as usize,
        &auxilary_input_layer_node,
    );

    let encoding_matrix = builder.add_input_shred(
        "Encoding Matrix",
        ENCODING_MATRIX_NUM_VARS_ROWS + ENCODING_MATRIX_NUM_VARS_COLS,
        &auxiliary_invariant_public_input_layer_node,
    );
    let evaluation_points = builder.add_input_shred(
        "Evaluation Points",
        num_vars,
        &auxiliary_invariant_public_input_layer_node,
    );

    let lookup_table_values = builder.add_input_shred(
        "Lookup Table Values",
        GR4_ELEM_BIT_LENGTH as usize,
        &auxiliary_invariant_public_input_layer_node,
    );
    let fiat_shamir_challenge_node = builder.add_fiat_shamir_challenge_node(1);
    let lookup_table = builder.add_lookup_table(&lookup_table_values, &fiat_shamir_challenge_node);

    let masked_iris_code =
        WorldcoinMpcComponents::masked_iris_code(&mut builder, &iris_code, &mask_code);
    // The matrix multiplication node encodes the masked iris code to a GR4 element
    let encoded_masked_iris_code = builder.add_matmult_node(
        &masked_iris_code,
        (num_vars_dataparallel, ENCODING_MATRIX_NUM_VARS_ROWS),
        &encoding_matrix,
        (ENCODING_MATRIX_NUM_VARS_ROWS, ENCODING_MATRIX_NUM_VARS_COLS),
    );

    // The gate node performs a GR4 multiplication between the evaluation points and the slopes
    let evaluation_points_times_slopes = builder.add_gate_node(
        &evaluation_points,
        &slopes,
        GR4_MULTIPLICATION_WIRINGS.to_vec(),
        BinaryOperation::Mul,
        Some(num_vars_dataparallel),
    );

    let computed_shares = WorldcoinMpcComponents::sum(
        &mut builder,
        &encoded_masked_iris_code,
        &evaluation_points_times_slopes,
    );

    let _congruence = WorldcoinMpcComponents::congruence(
        &mut builder,
        &quotients,
        &computed_shares,
        &shares_reduced_modulo_gr4_modulus,
    );

    let _lookup_constraint_shares = builder.add_lookup_constraint(
        &lookup_table,
        &shares_reduced_modulo_gr4_modulus,
        &multiplicities_shares,
    );

    let _lookup_constraint_slopes =
        builder.add_lookup_constraint(&lookup_table, &slopes, &multiplicities_slopes);

    builder.build().unwrap()
}

/// Generates a mapping from Layer IDs to their respective MLEs,
/// by attaching the `MPCInputBuilderMetadata` onto a circuit that is
/// described through the `input_builder_metadata` of an MPC secret share circuit.
pub fn mpc_attach_data<F: Field>(
    circuit: &mut Circuit<F>,
    secret_share_circuit_data: SecretShareCircuitInputs<F>,
) {
    circuit.set_input("Iriscode", secret_share_circuit_data.iris_codes);
    circuit.set_input("Maskcode", secret_share_circuit_data.masks);
    circuit.set_input("Slopes", secret_share_circuit_data.slopes);
    circuit.set_input("Quotients", secret_share_circuit_data.quotients);
    circuit.set_input(
        "Shares Reduced Modulo GR4 Modulus",
        secret_share_circuit_data.shares_reduced_modulo_gr4_modulus,
    );
    circuit.set_input(
        "Multiplicities Shares",
        secret_share_circuit_data.multiplicities_shares,
    );
    circuit.set_input(
        "Multiplicities Slopes",
        secret_share_circuit_data.multiplicities_slopes,
    );
    circuit.set_input("Encoding Matrix", secret_share_circuit_data.encoding_matrix);
    circuit.set_input(
        "Evaluation Points",
        secret_share_circuit_data.evaluation_points,
    );
    circuit.set_input(
        "Lookup Table Values",
        secret_share_circuit_data.lookup_table_values,
    );
}
