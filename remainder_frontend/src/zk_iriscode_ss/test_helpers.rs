#![allow(clippy::type_complexity)]
use crate::layouter::builder::{Circuit, LayerVisibility};
use crate::zk_iriscode_ss::circuits::build_iriscode_circuit_description;
use crate::zk_iriscode_ss::data::{
    build_iriscode_circuit_auxiliary_data, build_iriscode_circuit_data, wirings_to_reroutings,
    IriscodeCircuitAuxData, IriscodeCircuitInputData,
};
use ndarray::Array2;
use remainder::circuit_layout::ProvableCircuit;
use remainder::input_layer::{
    ligero_input_layer::LigeroInputLayerDescription, InputLayerDescription,
};
use remainder_hyrax::circuit_layout::HyraxProvableCircuit;
use remainder_ligero::ligero_structs::LigeroAuxInfo;
use remainder_shared_types::{Bn256Point, Field, Fr};

use super::circuits::iriscode_ss_attach_input_data;

use anyhow::Result;

/// Helper function for `test_small_circuit_with_hyrax_layer`.
pub fn build_ligero_layer_spec(
    input_layer_desc: &InputLayerDescription,
) -> LigeroInputLayerDescription<Fr> {
    let aux = LigeroAuxInfo::<Fr>::new(
        1 << input_layer_desc.num_vars,
        4,       // rho_inv
        1.0,     // ratio
        Some(1), // maybe_num_col_opens
    );
    LigeroInputLayerDescription {
        layer_id: input_layer_desc.layer_id,
        num_vars: input_layer_desc.num_vars,
        aux,
    }
}

pub fn build_small_circuit_and_data<F: Field, const BASE: u64>(
    use_private_layers: bool,
) -> Result<(
    Circuit<F>,
    IriscodeCircuitAuxData<F>,
    IriscodeCircuitInputData<F>,
)> {
    // rewirings for the 2x2 identity matrix
    let wirings = &[(0, 0, 0, 0), (0, 1, 0, 1), (1, 0, 1, 0), (1, 1, 1, 1)];
    let reroutings = wirings_to_reroutings(wirings, 2, 2);
    let circuit = build_iriscode_circuit_description::<F, 2, 2, 2, 1, 1, 1, BASE, 2>(
        if use_private_layers {
            LayerVisibility::Private
        } else {
            LayerVisibility::Public
        },
        vec![reroutings.clone().to_vec()],
        reroutings,
    )?;

    let rh_multiplicand = [1, 0, 6, -1];
    let thresholds_matrix = [1, 0, 1, 0];

    let aux_data = build_iriscode_circuit_auxiliary_data::<F, 1, 1, 1, 2>(
        &rh_multiplicand,
        &thresholds_matrix,
    );
    let input_data = build_iriscode_circuit_data::<F, 2, 2, 1, 1, 1, BASE, 2>(
        Array2::from_shape_vec((2, 2), vec![1, 2, 3, 4]).unwrap(),
        &rh_multiplicand,
        &thresholds_matrix,
        vec![wirings.clone().to_vec()],
        wirings,
    );

    Ok((circuit, aux_data, input_data))
}

fn small_circuit_with_inputs<F: Field>(use_private_layers: bool) -> Result<Circuit<F>> {
    const BASE: u64 = 16;

    let (circuit, aux_data, input_data) =
        build_small_circuit_and_data::<_, BASE>(use_private_layers)?;

    iriscode_ss_attach_input_data::<_, BASE>(circuit, input_data, aux_data)
}

/// Return a provable circuit description with public inputs for a trivial 2x2 identity matrix
/// circuit.
pub fn small_circuit_with_public_inputs() -> Result<ProvableCircuit<Fr>> {
    let circuit = small_circuit_with_inputs(false)?;
    circuit.gen_provable_circuit()
}

/// Return a provable circuit description with public inputs for a trivial 2x2 identity matrix
/// circuit.
pub fn small_circuit_with_private_inputs() -> Result<ProvableCircuit<Fr>> {
    let circuit = small_circuit_with_inputs(true)?;
    circuit.gen_provable_circuit()
}

/// Return a hyrax provable circuit description with public inputs for a trivial 2x2 identity matrix
/// circuit.
pub fn small_hyrax_circuit_with_public_inputs() -> Result<HyraxProvableCircuit<Bn256Point>> {
    let circuit = small_circuit_with_inputs(false)?;
    circuit.gen_hyrax_provable_circuit()
}

/// Return a hyrax provable circuit description with hyrax private inputs for a trivial 2x2 identity
/// matrix circuit.
pub fn small_hyrax_circuit_with_private_inputs() -> Result<HyraxProvableCircuit<Bn256Point>> {
    let circuit = small_circuit_with_inputs(true)?;
    circuit.gen_hyrax_provable_circuit()
}
