use itertools::Itertools;
use ndarray::Array2;
use remainder_shared_types::{Field, Fr};

use crate::{
    layouter::builder::{Circuit, LayerVisibility},
    zk_iriscode_ss::{
        data::{
            build_iriscode_circuit_auxiliary_data, build_iriscode_circuit_data,
            IriscodeCircuitAuxData,
        },
        parameters::LOG_NUM_STRIPS,
    },
};

use super::{
    circuits::{build_iriscode_circuit_description, iriscode_ss_attach_input_data},
    data::{wirings_to_reroutings, IriscodeCircuitInputData},
    decode::{decode_i32_array, decode_i64_array, decode_wirings},
    io::read_bytes_from_file,
    parameters::{
        BASE, IMAGE_STRIP_WIRINGS, IM_NUM_COLS, IM_NUM_ROWS, IM_NUM_VARS, IM_STRIP_NUM_COLS,
        IM_STRIP_NUM_ROWS, IRIS_RH_MULTIPLICAND, IRIS_THRESHOLDS, LH_MATRIX_WIRINGS,
        MASK_RH_MULTIPLICAND, MASK_THRESHOLDS, MATMULT_COLS_NUM_VARS,
        MATMULT_INTERNAL_DIM_NUM_VARS, MATMULT_ROWS_NUM_VARS, NUM_DIGITS,
    },
};

use anyhow::Result;

/// Loads circuit structure data and witnesses for a run of the iris code circuit from disk for
/// either the iris or mask case.
///
/// # Arguments:
/// * `image_bytes` gives the bytes of an image (could be the iris or the mask).
/// * `is_mask` indicates whether to load the files for the mask or the iris.
pub fn load_worldcoin_data<F: Field>(
    image_bytes: Vec<u8>,
    is_mask: bool,
) -> IriscodeCircuitInputData<F> {
    let image: Array2<u8> =
        Array2::from_shape_vec((IM_NUM_ROWS, IM_NUM_COLS), image_bytes).unwrap();
    if is_mask {
        build_iriscode_circuit_data::<
            F,
            IM_STRIP_NUM_ROWS,
            IM_STRIP_NUM_COLS,
            MATMULT_ROWS_NUM_VARS,
            MATMULT_COLS_NUM_VARS,
            MATMULT_INTERNAL_DIM_NUM_VARS,
            BASE,
            NUM_DIGITS,
        >(
            image,
            &decode_i32_array(MASK_RH_MULTIPLICAND),
            &decode_i64_array(MASK_THRESHOLDS),
            IMAGE_STRIP_WIRINGS
                .iter()
                .map(|wirings| decode_wirings(wirings))
                .collect_vec(),
            &decode_wirings(LH_MATRIX_WIRINGS),
        )
    } else {
        build_iriscode_circuit_data::<
            F,
            IM_STRIP_NUM_ROWS,
            IM_STRIP_NUM_COLS,
            MATMULT_ROWS_NUM_VARS,
            MATMULT_COLS_NUM_VARS,
            MATMULT_INTERNAL_DIM_NUM_VARS,
            BASE,
            NUM_DIGITS,
        >(
            image,
            &decode_i32_array(IRIS_RH_MULTIPLICAND),
            &decode_i64_array(IRIS_THRESHOLDS),
            IMAGE_STRIP_WIRINGS
                .iter()
                .map(|wirings| decode_wirings(wirings))
                .collect_vec(),
            &decode_wirings(LH_MATRIX_WIRINGS),
        )
    }
}

pub fn build_worldcoin_aux_data<F: Field>(is_mask: bool) -> IriscodeCircuitAuxData<F> {
    if is_mask {
        build_iriscode_circuit_auxiliary_data::<
            F,
            MATMULT_COLS_NUM_VARS,
            MATMULT_INTERNAL_DIM_NUM_VARS,
            { 1 << (LOG_NUM_STRIPS) },
            { 1 << MATMULT_ROWS_NUM_VARS },
        >(
            &decode_i32_array(MASK_RH_MULTIPLICAND),
            &decode_i64_array(MASK_THRESHOLDS),
        )
    } else {
        build_iriscode_circuit_auxiliary_data::<
            F,
            MATMULT_COLS_NUM_VARS,
            MATMULT_INTERNAL_DIM_NUM_VARS,
            { 1 << (LOG_NUM_STRIPS) },
            { 1 << MATMULT_ROWS_NUM_VARS },
        >(
            &decode_i32_array(IRIS_RH_MULTIPLICAND),
            &decode_i64_array(IRIS_THRESHOLDS),
        )
    }
}

/// Return the [Circuit] for the v3 iris code circuit.
pub fn circuit_description() -> Result<Circuit<Fr>> {
    let image_strip_reroutings = IMAGE_STRIP_WIRINGS
        .iter()
        .map(|wirings| {
            wirings_to_reroutings(
                &decode_wirings(wirings),
                IM_STRIP_NUM_COLS,
                IM_STRIP_NUM_COLS,
            )
        })
        .collect::<Vec<_>>();
    let lh_matrix_reroutings = wirings_to_reroutings(
        &decode_wirings(LH_MATRIX_WIRINGS),
        IM_STRIP_NUM_COLS,
        1 << MATMULT_INTERNAL_DIM_NUM_VARS,
    );
    build_iriscode_circuit_description::<
        Fr,
        IM_STRIP_NUM_ROWS,
        IM_STRIP_NUM_COLS,
        IM_NUM_VARS,
        MATMULT_ROWS_NUM_VARS,
        MATMULT_COLS_NUM_VARS,
        MATMULT_INTERNAL_DIM_NUM_VARS,
        BASE,
        NUM_DIGITS,
    >(
        LayerVisibility::Private,
        image_strip_reroutings,
        lh_matrix_reroutings,
    )
}

/// Return the circuit description, and inputs for the iris code circuit, in either the mask (true)
/// or iris (false) case.
/// If `image_bytes` is `None`, the default test image is used.
/// # Example:
/// ```
/// use remainder_frontend::zk_iriscode_ss::v3::circuit_description_and_inputs;
/// let circuit_result = circuit_description_and_inputs(false, None);
/// ```
pub fn circuit_description_and_inputs(
    is_mask: bool,
    image_bytes: Option<Vec<u8>>,
) -> Result<Circuit<Fr>> {
    let image_bytes = if let Some(image_bytes) = image_bytes {
        image_bytes
    } else {
        let image_type = if is_mask { "mask" } else { "iris" };
        let remainder_prover_root_dir = env!("CARGO_MANIFEST_DIR");
        let image_path =
            format!("{remainder_prover_root_dir}/src/zk_iriscode_ss/constants/v3-split-images/{image_type}/test_image.bin");
        dbg!(&image_path);
        read_bytes_from_file(&image_path)
    };
    let circuit = circuit_description().unwrap();
    let aux_data = build_worldcoin_aux_data::<Fr>(is_mask);
    let input_data = load_worldcoin_data::<Fr>(image_bytes, is_mask);
    iriscode_ss_attach_input_data::<Fr, BASE>(circuit, input_data, aux_data)
}
