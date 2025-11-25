use std::collections::HashMap;

use ark_std::log2;
use itertools::Itertools;
use rand::Rng;
use shared_types::Field;

use remainder::{
    layer::{
        gate::{compute_gate_data_outputs, BinaryOperation},
        matmult::product_two_matrices_from_flattened_vectors,
    },
    mle::evals::MultilinearExtension,
};

use crate::{
    hyrax_worldcoin_mpc::mpc_prover::MPCCircuitConstData,
    worldcoin_mpc::parameters::{
        EVALUATION_POINTS_U64, GR4_MULTIPLICATION_WIRINGS, TEST_MASKED_IRIS_CODES,
        TEST_RANDOMNESSES, TEST_SHARES,
    },
};

use super::parameters::{ENCODING_MATRIX_U64_TRANSPOSE, GR4_MODULUS};

/// Used for instantiating the mpc circuit.
#[derive(Debug, Clone)]
pub struct MPCCircuitInputData<F: Field> {
    /// The iris codes, they are {0, 1} valued.
    /// Needed to calculate the masked code which we secret share
    /// between the three parties.
    pub iris_codes: MultilinearExtension<F>,

    /// The masks, they are {0, 1} valued.
    /// Needed to calculate the masked code which we secret share
    /// between the three parties.
    pub masks: MultilinearExtension<F>,

    /// The slopes, they are random elements in GR4
    /// These are generated randomly beforehand and supplied into the circuit
    pub slopes: MultilinearExtension<F>,

    /// The quotients, they are elements in GR4: GR(size_of(F), 4) is a Galois
    /// extension of Z/size_of(F)Z over the monic polynomial x^4 - x - 1
    /// The naively calculated secret shares might be outside of the range of
    /// [0..2^16], which are the range of the coefficients of the particular GR4
    /// we choose GR(2^16, 4)
    /// Therefore, we supply `quotients` and `shares_reduced_modulo_gr4_modulus`,
    /// so that `shares_reduced_modulo_gr4_modulus` + `quotients` * `GR4_MODULUS`
    /// equals to the naively calculated secret shares
    pub quotients: MultilinearExtension<F>,

    /// The shares_reduced_modulo_gr4_modulus, they are elemnts in GR4: GR(2^16, 4)
    /// We supply `quotients` and `shares_reduced_modulo_gr4_modulus`,
    /// so that `shares_reduced_modulo_gr4_modulus` + `quotients` * `GR4_MODULUS`
    /// equals to the naively calculated secret shares
    pub shares_reduced_modulo_gr4_modulus: MultilinearExtension<F>,

    /// The multiplicies is used for lookup (expected_shares), we calculate the occurances
    /// of different numbers between [0..2^16]
    pub multiplicities_shares: MultilinearExtension<F>,

    /// The multiplicies is used for lookup (slope), we calculate the occurances of different
    /// numbers between [0..2^16]
    pub multiplicities_slopes: MultilinearExtension<F>,
}

pub fn gen_mpc_evaluation_points<
    F: Field,
    const NUM_IRIS_4_CHUNKS: usize,
    const PARTY_IDX: usize,
>() -> MultilinearExtension<F> {
    MultilinearExtension::new(
        EVALUATION_POINTS_U64[PARTY_IDX]
            .into_iter()
            .map(|x| F::from(x))
            .cycle()
            .take(NUM_IRIS_4_CHUNKS * 4)
            .collect(),
    )
}

pub fn gen_mpc_encoding_matrix<F: Field, const NUM_IRIS_4_CHUNKS: usize>() -> MultilinearExtension<F>
{
    MultilinearExtension::new(
        ENCODING_MATRIX_U64_TRANSPOSE
            .into_iter()
            .map(|x| F::from(x))
            .collect_vec(),
    )
}

pub fn gen_mpc_common_aux_data<F: Field, const NUM_IRIS_4_CHUNKS: usize, const PARTY_IDX: usize>(
) -> MPCCircuitConstData<F> {
    let evaluation_points = gen_mpc_evaluation_points::<F, NUM_IRIS_4_CHUNKS, PARTY_IDX>();
    let encoding_matrix = gen_mpc_encoding_matrix::<F, NUM_IRIS_4_CHUNKS>();
    let lookup_table_values = MultilinearExtension::new((0..GR4_MODULUS).map(F::from).collect());

    MPCCircuitConstData {
        evaluation_points,
        encoding_matrix,
        lookup_table_values,
    }
}

/// Selects the encoding matrix, and
/// Calculates the quotients, expected_shares (modulo gr4), and the multiplicities
/// returns as a tuple, in the folllowing order:
/// (encoding_matrix, quotients, expected_shares, multiplicities)
#[allow(clippy::type_complexity)]
pub fn gen_mpc_input_data<F: Field, const NUM_IRIS_4_CHUNKS: usize>(
    iris_codes: &MultilinearExtension<F>,
    masks: &MultilinearExtension<F>,
    slopes: &MultilinearExtension<F>,
    encoding_matrix: &MultilinearExtension<F>,
    evaluation_points: &MultilinearExtension<F>,
) -> MPCCircuitInputData<F> {
    let num_copies = NUM_IRIS_4_CHUNKS;

    // let lookup_table_values = MultilinearExtension::new((0..GR4_MODULUS).map(F::from).collect());

    // masked_iris_codes dimension is: (NUM_IRIS_4_CHUNKS, 4)
    let mut masked_iris_codes = iris_codes
        .iter()
        .zip(masks.iter())
        .map(|(iris_code, mask)| F::from(2).neg() * iris_code - mask + F::from(GR4_MODULUS))
        .collect_vec();

    // encoding_matrix dimension is: (4, 4)
    let encoded_masked_iris_code = product_two_matrices_from_flattened_vectors(
        &masked_iris_codes,
        &encoding_matrix.to_vec(),
        num_copies,
        4,
        4,
        4,
    );

    let evaluation_points_times_slopes = compute_gate_data_outputs(
        GR4_MULTIPLICATION_WIRINGS.to_vec(),
        log2(num_copies.next_power_of_two()) as usize,
        evaluation_points,
        slopes,
        BinaryOperation::Mul,
    );

    let mut shares_before_modulo_gr4 = encoded_masked_iris_code
        .into_iter()
        .zip(evaluation_points_times_slopes.iter())
        .map(|(a, b)| a + b)
        .collect_vec();

    // because the modulo is 2^16, we can just take the smallest 16 bits as the
    // reduced modulo gr4 shares, and the rest bits as the quotients
    let (quotients, expected_shares): (Vec<F>, Vec<F>) = shares_before_modulo_gr4
        .clone()
        .into_iter()
        .map(|x| {
            let mut bytes = x.to_bytes_le();
            let mut without_first_two_bytes = bytes.split_off(2);

            // for quotient: pads the rest two zero bytes at the end
            without_first_two_bytes.append(&mut [0u8, 0u8].to_vec());

            // for remainder(modulus): pads the rest 30 zero bytes
            bytes.append(&mut [0u8; 30].to_vec());

            (
                F::from_bytes_le(&without_first_two_bytes),
                F::from_bytes_le(&bytes),
            )
        })
        .unzip();

    let f_gr4_modulus = F::from(GR4_MODULUS);

    // calculates the multiplicities of shares
    let mut counts_shares: HashMap<F, u64> = HashMap::new();
    expected_shares.iter().for_each(|x| {
        // check that indeed the shares are less than the modulus
        assert!(x < &f_gr4_modulus);

        *counts_shares.entry(*x).or_insert(0) += 1;
    });

    let mut multiplicities_shares = vec![F::ZERO; GR4_MODULUS as usize];
    counts_shares.iter().for_each(|(k, v)| {
        multiplicities_shares[k.to_u64s_le()[0] as usize] = F::from(*v);
    });
    // number of 0s as implicit paddings
    let num_elements = num_copies * 4;
    let num_zeros = num_elements.next_power_of_two() - num_elements;
    multiplicities_shares[0] += F::from(num_zeros as u64);

    // the same process for slopes
    let mut counts_slopes: HashMap<F, u64> = HashMap::new();
    slopes.iter().for_each(|x| {
        // check that indeed the shares are less than the modulus
        assert!(x < f_gr4_modulus);

        *counts_slopes.entry(x).or_insert(0) += 1;
    });

    let mut multiplicities_slopes = vec![F::ZERO; GR4_MODULUS as usize];
    counts_slopes.iter().for_each(|(k, v)| {
        multiplicities_slopes[k.to_u64s_le()[0] as usize] = F::from(*v);
    });
    // number of 0s as implicit paddings
    let num_elements = num_copies * 4;
    let num_zeros = num_elements.next_power_of_two() - num_elements;
    multiplicities_slopes[0] += F::from(num_zeros as u64);

    quotients
        .iter()
        .zip(shares_before_modulo_gr4.iter())
        .zip(expected_shares.iter())
        .for_each(|((quotient, share_before_modulo_gr4), expected_share)| {
            assert_eq!(
                *quotient * F::from(GR4_MODULUS) + expected_share,
                *share_before_modulo_gr4
            );
        });

    // zeroize shares_before_modulo_gr4
    for f in shares_before_modulo_gr4.iter_mut() {
        f.zeroize();
    }
    // zeroize masked_iris_codes
    for f in masked_iris_codes.iter_mut() {
        f.zeroize();
    }

    let quotients = MultilinearExtension::new(quotients);
    let shares_reduced_modulo_gr4_modulus = MultilinearExtension::new(expected_shares);
    let multiplicities_shares = MultilinearExtension::new(multiplicities_shares);
    let multiplicities_slopes = MultilinearExtension::new(multiplicities_slopes);

    MPCCircuitInputData::<F> {
        iris_codes: iris_codes.clone(),
        masks: masks.clone(),
        slopes: slopes.clone(),
        quotients,
        shares_reduced_modulo_gr4_modulus,
        multiplicities_shares,
        multiplicities_slopes,
        // lookup_table_values,
    }
}

/// create test data for mpc circuits, control the size of such
pub fn generate_trivial_test_data<
    F: Field,
    const NUM_IRIS_4_CHUNKS: usize,
    const PARTY_IDX: usize,
>() -> (MPCCircuitConstData<F>, MPCCircuitInputData<F>) {
    let num_copies = NUM_IRIS_4_CHUNKS;
    let mut rng = rand::thread_rng();

    let iris_codes = MultilinearExtension::new(
        (0..4 * num_copies)
            .map(|_| F::from(rng.gen_range(0..=1)))
            .collect(),
    );
    let masks = MultilinearExtension::new(
        (0..4 * num_copies)
            .map(|_| F::from(rng.gen_range(0..=1)))
            .collect(),
    );
    let slopes = MultilinearExtension::new(
        (0..4 * num_copies)
            .map(|_| F::from(rng.gen_range(0..=(GR4_MODULUS - 1))))
            .collect(),
    );

    let mpc_aux_data = gen_mpc_common_aux_data::<F, NUM_IRIS_4_CHUNKS, PARTY_IDX>();

    let mpc_input_data = gen_mpc_input_data::<F, NUM_IRIS_4_CHUNKS>(
        &iris_codes,
        &masks,
        &slopes,
        &mpc_aux_data.encoding_matrix,
        &mpc_aux_data.evaluation_points,
    );

    assert_eq!(mpc_input_data.quotients.len(), num_copies * 4);
    assert_eq!(
        mpc_input_data.shares_reduced_modulo_gr4_modulus.len(),
        num_copies * 4
    );
    assert_eq!(slopes.len(), num_copies * 4);
    assert_eq!(mpc_aux_data.evaluation_points.len(), num_copies * 4);
    assert_eq!(
        mpc_input_data.multiplicities_shares.len(),
        GR4_MODULUS as usize
    );
    assert_eq!(
        mpc_input_data.multiplicities_slopes.len(),
        GR4_MODULUS as usize
    );
    assert_eq!(mpc_aux_data.lookup_table_values.len(), GR4_MODULUS as usize);

    (mpc_aux_data, mpc_input_data)
}

/// Fetch one quadruplets from the test data given by Inversed,
/// `test_idx` specifies which copy
pub fn fetch_inversed_test_data<
    F: Field,
    const NUM_IRIS_4_CHUNKS: usize,
    const PARTY_IDX: usize,
>(
    test_idx: usize,
) -> (MPCCircuitConstData<F>, MPCCircuitInputData<F>) {
    let num_copies = NUM_IRIS_4_CHUNKS;
    if test_idx + NUM_IRIS_4_CHUNKS >= TEST_MASKED_IRIS_CODES.len() {
        panic!("test_idx out of range");
    }
    let mut rng = rand::thread_rng();

    let masked_iris_codes = MultilinearExtension::new(
        (0..num_copies)
            .flat_map(|batch_idx| {
                TEST_MASKED_IRIS_CODES[batch_idx + test_idx]
                    .into_iter()
                    .map(F::from)
                    .collect::<Vec<F>>()
            })
            .collect_vec(),
    );
    let iris_codes = MultilinearExtension::new(
        (0..num_copies * 4)
            .map(|_| F::from(rng.gen_range(0..=1)))
            .collect(),
    );
    assert_eq!(masked_iris_codes.len(), iris_codes.len());
    let masks = MultilinearExtension::new(
        masked_iris_codes
            .iter()
            .zip(iris_codes.iter())
            .map(|(masked_iris_code, iris_code)| F::from(2).neg() * iris_code - masked_iris_code)
            .collect(),
    );
    let slopes = MultilinearExtension::new(
        (0..num_copies)
            .flat_map(|batch_idx| {
                TEST_RANDOMNESSES[batch_idx + test_idx]
                    .into_iter()
                    .map(F::from)
                    .collect::<Vec<F>>()
            })
            .collect_vec(),
    );

    let mpc_aux_data = gen_mpc_common_aux_data::<F, NUM_IRIS_4_CHUNKS, PARTY_IDX>();
    let mpc_input_data = gen_mpc_input_data::<F, NUM_IRIS_4_CHUNKS>(
        &iris_codes,
        &masks,
        &slopes,
        &mpc_aux_data.encoding_matrix,
        &mpc_aux_data.evaluation_points,
    );

    assert_eq!(mpc_input_data.quotients.len(), num_copies * 4);
    assert_eq!(
        mpc_input_data.shares_reduced_modulo_gr4_modulus.len(),
        num_copies * 4
    );
    assert_eq!(slopes.len(), num_copies * 4);
    assert_eq!(mpc_aux_data.evaluation_points.len(), num_copies * 4);
    assert_eq!(
        mpc_input_data.multiplicities_shares.len(),
        GR4_MODULUS as usize
    );
    assert_eq!(
        mpc_input_data.multiplicities_slopes.len(),
        GR4_MODULUS as usize
    );
    assert_eq!(mpc_aux_data.lookup_table_values.len(), GR4_MODULUS as usize);

    mpc_input_data
        .shares_reduced_modulo_gr4_modulus
        .iter()
        .zip(
            (0..num_copies)
                .flat_map(|batch_idx| TEST_SHARES[PARTY_IDX][batch_idx + test_idx].into_iter())
                .collect_vec()
                .iter(),
        )
        .for_each(|(a, b)| {
            assert_eq!(a, F::from(*b));
        });

    (mpc_aux_data, mpc_input_data)
}
