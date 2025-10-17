#![allow(clippy::type_complexity)]

use crate::components::digits::DigitComponents;
use crate::layouter::builder::{Circuit, CircuitBuilder, LayerVisibility};
use crate::zk_iriscode_ss::components::ZkIriscodeComponent;
use crate::zk_iriscode_ss::data::IriscodeCircuitAuxData;
use remainder::mle::evals::MultilinearExtension;
use remainder::utils::arithmetic::log2_ceil;

use itertools::Itertools;
use remainder_shared_types::Field;

use super::data::IriscodeCircuitInputData;

use anyhow::Result;

/// The input layer for the image (typically private).
pub const V3_INPUT_IMAGE_LAYER: &str = "Input image (to reroute)";
/// The input layer for the digit values and the digit multiplicities (typically private).
pub const V3_DIGITS_LAYER: &str = "Digit values and multiplicities";
/// The input layer for the iris/mask code.
pub const V3_SIGN_BITS_LAYER: &str = "Sign Bits";
/// All the other public inputs (lookup table values, to sub from matmult & RH multiplicand of matmult).
pub const V3_AUXILIARY_LAYER: &str = "Auxiliary Data";

pub const V3_INPUT_IMAGE_SHRED: &str = "Image to reroute";
pub const V3_DIGITS_SHRED_TEMPLATE: &str = "Digits Input Shred";
pub const V3_DIGITS_MULTIPLICITIES_SHRED: &str = "Digits multiplicities";
pub const V3_TO_SUB_MATMULT_SHRED: &str = "Input to subtract from MatMult";
pub const V3_RH_MATMULT_SHRED: &str = "RH Multiplicand of MatMult";
pub const V3_LOOKUP_SHRED: &str = "Lookup table values for digit range check";
pub const V3_SIGN_BITS_SHRED: &str = "Sign Bits";

/// Build the [IriscodeCircuitDescription], return the circuit description and
/// the input builder metadata, for the v3 (RLC) iriscode circuit.
pub fn build_iriscode_circuit_description<
    F: Field,
    const IM_STRIP_ROWS: usize,
    const IM_STRIP_COLS: usize,
    const IM_NUM_VARS: usize,
    const MATMULT_ROWS_NUM_VARS: usize,
    const MATMULT_COLS_NUM_VARS: usize,
    const MATMULT_INTERNAL_DIM_NUM_VARS: usize,
    const BASE: u64,
    const NUM_DIGITS: usize,
>(
    layer_visibility: LayerVisibility,
    image_strip_reroutings: Vec<Vec<(u32, u32)>>,
    lh_matrix_reroutings: Vec<(u32, u32)>,
) -> Result<Circuit<F>> {
    let mut builder = CircuitBuilder::<F>::new();

    assert!(BASE.is_power_of_two());
    let log_base = log2_ceil(BASE) as usize;
    let num_strips = image_strip_reroutings.len();
    assert!(num_strips.is_power_of_two());
    let log_num_strips = log2_ceil(num_strips) as usize;

    // Image input layer
    let to_reroute_input_layer = builder.add_input_layer(V3_INPUT_IMAGE_LAYER, layer_visibility);
    let to_reroute =
        builder.add_input_shred(V3_INPUT_IMAGE_SHRED, IM_NUM_VARS, &to_reroute_input_layer);

    // Digits and multiplicities input layer
    let digits_input_layer = builder.add_input_layer(V3_DIGITS_LAYER, layer_visibility);
    let digits_input_shreds: Vec<_> = (0..NUM_DIGITS)
        .map(|i| {
            builder.add_input_shred(
                &format!("{V3_DIGITS_SHRED_TEMPLATE} {i}"),
                log_num_strips + MATMULT_ROWS_NUM_VARS + MATMULT_COLS_NUM_VARS,
                &digits_input_layer,
            )
        })
        .collect();

    let digit_multiplicities = builder.add_input_shred(
        V3_DIGITS_MULTIPLICITIES_SHRED,
        log_base,
        &digits_input_layer,
    );

    // Auxiliary inputs
    let auxiliary_input_layer =
        builder.add_input_layer(V3_AUXILIARY_LAYER, LayerVisibility::Public);

    let to_sub_from_matmult = builder.add_input_shred(
        V3_TO_SUB_MATMULT_SHRED,
        log_num_strips + MATMULT_ROWS_NUM_VARS + MATMULT_COLS_NUM_VARS,
        &auxiliary_input_layer,
    );

    let rh_matmult_multiplicand = builder.add_input_shred(
        V3_RH_MATMULT_SHRED,
        MATMULT_INTERNAL_DIM_NUM_VARS + MATMULT_COLS_NUM_VARS,
        &auxiliary_input_layer,
    );

    let lookup_table_values =
        builder.add_input_shred(V3_LOOKUP_SHRED, log_base, &auxiliary_input_layer);

    // Sign bits (iris/mask code)
    let sign_bits_input_layer = builder.add_input_layer(V3_SIGN_BITS_LAYER, layer_visibility);
    let sign_bits = builder.add_input_shred(
        V3_SIGN_BITS_SHRED,
        log_num_strips + MATMULT_ROWS_NUM_VARS + MATMULT_COLS_NUM_VARS,
        &sign_bits_input_layer,
    );

    // Verifier challenges for RLC
    let rlc_challenges = (0..num_strips)
        .map(|_| builder.add_fiat_shamir_challenge_node(1))
        .collect_vec();
    let rlc_challenges_generic = rlc_challenges
        .clone()
        .into_iter()
        .map(|node| node.into())
        .collect_vec();

    // Verifier challenge for lookup
    let lookup_challenge = builder.add_fiat_shamir_challenge_node(1);

    // Intermediate layers

    // Image decomposition layers
    let image_strip_nodes = image_strip_reroutings
        .into_iter()
        .map(|reroutings| {
            builder.add_identity_gate_node(
                &to_reroute,
                reroutings,
                log2_ceil(IM_STRIP_ROWS * IM_STRIP_COLS) as usize,
                None,
            )
        })
        .collect_vec();

    // Image RLC layer
    let image_rlc = ZkIriscodeComponent::sum_of_products(
        &mut builder,
        rlc_challenges_generic.iter().collect(),
        image_strip_nodes.iter().collect(),
    );

    // Reroute the image to the LH matrix multiplicand
    let rerouted_image = builder.add_identity_gate_node(
        &image_rlc,
        lh_matrix_reroutings,
        MATMULT_ROWS_NUM_VARS + MATMULT_INTERNAL_DIM_NUM_VARS,
        None,
    );

    // Matmult layer
    let matmult = builder.add_matmult_node(
        &rerouted_image,
        (MATMULT_ROWS_NUM_VARS, MATMULT_INTERNAL_DIM_NUM_VARS),
        &rh_matmult_multiplicand,
        (MATMULT_INTERNAL_DIM_NUM_VARS, MATMULT_COLS_NUM_VARS),
    );

    // Thresholds RLC layer
    let to_sub_from_matmult_splits = builder.add_split_node(&to_sub_from_matmult, log_num_strips);

    let to_sub_from_matmult_rlc = ZkIriscodeComponent::sum_of_products(
        &mut builder,
        rlc_challenges_generic.iter().collect(),
        to_sub_from_matmult_splits.iter().collect(),
    );

    // Subtract the thresholds from the result of matmult
    let subtractor = builder.add_sector(matmult - to_sub_from_matmult_rlc);

    // Create an RLC node for each of the NUM_DIGITS digital places
    let digits_split_nodes = digits_input_shreds
        .iter()
        .map(|shred| builder.add_split_node(shred, log_num_strips))
        .collect_vec();
    let digits_rlc = digits_split_nodes
        .iter()
        .map(|splits| {
            let digit_rlc = ZkIriscodeComponent::sum_of_products(
                &mut builder,
                rlc_challenges_generic.iter().collect(),
                splits.iter().collect(),
            );
            digit_rlc
        })
        .collect_vec();

    // Concatenate the digits (which are stored for each digital place separately) into a single
    // MLE for the lookup
    let digits_concatenator = DigitComponents::digits_concatenator(
        &mut builder,
        &digits_input_shreds.iter().collect_vec(),
    );

    // Lookup table and constraint
    let lookup_table = builder.add_lookup_table(&lookup_table_values, &lookup_challenge);
    // println!("{:?} = Lookup table", builder.get_id(&lookup_table));
    let _lookup_constraint =
        builder.add_lookup_constraint(&lookup_table, &digits_concatenator, &digit_multiplicities);
    // println!("{:?} = Lookup constraint", lookup_constraint.id());

    // Form the unsigned recomposition of the RLC'd digits
    let unsigned_recomp = DigitComponents::unsigned_recomposition(
        &mut builder,
        &digits_rlc.iter().collect_vec(),
        BASE,
    );

    // Iriscode RLC layer
    let sign_bits_splits = builder.add_split_node(&sign_bits, log_num_strips);

    let sign_bits_rlc = ZkIriscodeComponent::sum_of_products(
        &mut builder,
        rlc_challenges_generic.iter().collect(),
        sign_bits_splits.iter().collect(),
    );

    // Complementary recomp check using the unsigned recomp of the RLC'd digits and the RLC'd sign bits
    let complementary_checker = DigitComponents::complementary_recomp_check(
        &mut builder,
        &subtractor,
        &sign_bits_rlc,
        &unsigned_recomp,
        BASE,
        NUM_DIGITS,
    );
    builder.set_output(&complementary_checker);

    let bits_are_binary = DigitComponents::bits_are_binary(&mut builder, &sign_bits);
    builder.set_output(&bits_are_binary);

    // Generate the circuit description and input builder
    builder.build_without_layer_combination()
}

pub fn iriscode_ss_attach_aux_data<F: Field, const BASE: u64>(
    mut circuit: Circuit<F>,
    iriscode_aux_data: IriscodeCircuitAuxData<F>,
) -> Result<Circuit<F>> {
    circuit.set_input(
        V3_RH_MATMULT_SHRED,
        iriscode_aux_data.rh_matmult_multiplicand,
    );

    circuit.set_input(
        V3_TO_SUB_MATMULT_SHRED,
        iriscode_aux_data.to_sub_from_matmult,
    );

    circuit.set_input(
        V3_LOOKUP_SHRED,
        MultilinearExtension::new((0..BASE).map(F::from).collect()),
    );

    Ok(circuit)
}

/// Generates a mapping from Layer IDs to their respective MLEs,
/// by attaching the `iriscode_data` onto a circuit that is
/// described through the `input_builder_metadata`.
pub fn iriscode_ss_attach_input_data<F: Field, const BASE: u64>(
    mut circuit: Circuit<F>,
    iriscode_input_data: IriscodeCircuitInputData<F>,
    iriscode_aux_data: IriscodeCircuitAuxData<F>,
) -> Result<Circuit<F>> {
    circuit.set_input(V3_INPUT_IMAGE_SHRED, iriscode_input_data.to_reroute);
    circuit.set_input(
        V3_RH_MATMULT_SHRED,
        iriscode_aux_data.rh_matmult_multiplicand,
    );

    iriscode_input_data
        .digits
        .into_iter()
        .enumerate()
        .for_each(|(i, mle)| {
            circuit.set_input(&format!("{V3_DIGITS_SHRED_TEMPLATE} {i}"), mle);
        });

    circuit.set_input(V3_SIGN_BITS_SHRED, iriscode_input_data.sign_bits);
    circuit.set_input(
        V3_TO_SUB_MATMULT_SHRED,
        iriscode_aux_data.to_sub_from_matmult,
    );
    circuit.set_input(
        V3_DIGITS_MULTIPLICITIES_SHRED,
        iriscode_input_data.digit_multiplicities,
    );
    circuit.set_input(
        V3_LOOKUP_SHRED,
        MultilinearExtension::new((0..BASE).map(F::from).collect()),
    );

    Ok(circuit)
}
