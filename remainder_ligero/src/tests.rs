// Copyright 2021 Riad S. Wahby <rsw@cs.stanford.edu>
//
// This file is part of lcpc-2d, which is part of lcpc.
//
// Licensed under the Apache License, Version 2.0 (see
// LICENSE or https://www.apache.org/licenses/LICENSE-2.0).
// This file may not be copied, modified, or distributed
// except according to those terms.

use super::LcRoot;
use crate::{
    ligero_commit::remainder_ligero_commit,
    ligero_structs::LigeroAuxInfo,
    log2,
    utils::{get_random_coeffs_for_multilinear_poly, halo2_ifft},
    verify_column_path, verify_column_value, LcColumn,
};

// For serialization/deserialization of the various structs
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use halo2_proofs::poly::EvaluationDomain;
// For BN-254
use itertools::{iterate, Itertools};
use rand::Rng;
use remainder_shared_types::transcript::{
    poseidon_transcript::PoseidonSponge, ProverTranscript, VerifierTranscript,
};
use remainder_shared_types::{
    transcript::{TranscriptReader, TranscriptWriter},
    Poseidon,
};
use std::iter::repeat_with;

#[test]
fn log2_test() {
    use super::log2;

    for idx in 0..31 {
        assert_eq!(log2(1usize << idx), idx);
    }
}

/// The purpose of this test is to check that the parallel version of the
/// Merkle hashing step, i.e. [merkleize()], yields the same result as the
/// serial version, i.e. [merkleize_ser()], using Poseidon as the hashing
/// function.
#[test]
fn test_merkleize() {
    use ark_std::test_rng;
    use remainder_shared_types::Fr;

    let master_default_poseidon_merkle_hasher = Poseidon::<Fr, 3, 2>::new(8, 57);
    let master_default_poseidon_column_hasher = Poseidon::<Fr, 3, 2>::new(8, 57);

    use super::{merkleize, merkleize_ser};

    const MLE_NUM_VARS: usize = 4;
    const RHO_INV: u8 = 4;
    const RATIO: f64 = 4.0;
    let mut rng = test_rng();
    let random_ml_coeffs = get_random_coeffs_for_multilinear_poly(MLE_NUM_VARS, &mut rng);
    let aux = LigeroAuxInfo::new(
        random_ml_coeffs.len().next_power_of_two(),
        RHO_INV,
        RATIO,
        None,
    );
    let (mut test_comm, _root) = remainder_ligero_commit(&random_ml_coeffs, &aux);
    let mut test_comm_2 = test_comm.clone();

    merkleize(&mut test_comm);
    merkleize_ser(
        &mut test_comm_2,
        &master_default_poseidon_merkle_hasher,
        &master_default_poseidon_column_hasher,
    );

    assert_eq!(&test_comm.comm, &test_comm_2.comm);
    assert_eq!(&test_comm.coeffs, &test_comm_2.coeffs);
    assert_eq!(&test_comm.hashes, &test_comm_2.hashes);
}

/// The purpose of this test is to check that the parallel version of the matrix
/// vector product, i.e. [eval_outer()], yields the same result as that of the
/// serial version, i.e. [eval_outer_ser()].
#[test]
fn test_eval_outer() {
    use ark_std::test_rng;
    use remainder_shared_types::Fr;

    use super::{eval_outer, eval_outer_ser};

    const MLE_NUM_VARS: usize = 4;
    const RHO_INV: u8 = 4;
    const RATIO: f64 = 4.0;
    let mut rng = test_rng();
    let random_ml_coeffs = get_random_coeffs_for_multilinear_poly(MLE_NUM_VARS, &mut rng);
    let aux = LigeroAuxInfo::new(
        random_ml_coeffs.len().next_power_of_two(),
        RHO_INV,
        RATIO,
        None,
    );
    let (test_comm, _root) = remainder_ligero_commit(&random_ml_coeffs, &aux);

    let mut rng = rand::thread_rng();
    let tensor: Vec<Fr> = repeat_with(|| Fr::from(rng.gen::<u64>()))
        .take(test_comm.n_rows)
        .collect();

    let res1 = eval_outer(&test_comm, &tensor[..]).unwrap();
    let res2 = eval_outer_ser(&test_comm, &tensor[..]).unwrap();

    assert_eq!(&res1[..], &res2[..]);
}

/// The purpose of this test is to check that the column opening step within
/// the Ligero verification process, i.e. the [open_column()] and
/// [verify_column_value()] and [verify_column_path()]
/// steps, are working as intended given a valid commitment generated via
/// [remainder_ligero_commit()].
#[test]
fn test_open_column() {
    use super::{merkleize, open_column};
    use remainder_shared_types::Fr;

    let master_default_poseidon_merkle_hasher = Poseidon::<Fr, 3, 2>::new(8, 57);
    let master_default_poseidon_column_hasher = Poseidon::<Fr, 3, 2>::new(8, 57);

    let mut rng = rand::thread_rng();

    let test_comm = {
        const MLE_NUM_VARS: usize = 4;
        const RHO_INV: u8 = 4;
        const RATIO: f64 = 4.0;
        let random_ml_coeffs = get_random_coeffs_for_multilinear_poly(MLE_NUM_VARS, &mut rng);
        let aux = LigeroAuxInfo::new(
            random_ml_coeffs.len().next_power_of_two(),
            RHO_INV,
            RATIO,
            None,
        );
        let (mut tmp, _root) = remainder_ligero_commit(&random_ml_coeffs, &aux);
        merkleize(&mut tmp);
        tmp
    };

    let root = test_comm.get_root();
    for _ in 0..1 {
        let col_num = rng.gen::<usize>() % test_comm.encoded_num_cols;

        let mut transcript_writer: TranscriptWriter<Fr, PoseidonSponge<Fr>> =
            TranscriptWriter::new("Testing column opens");

        let _column = open_column(&mut transcript_writer, &test_comm, col_num).unwrap();

        let mut transcript_reader: TranscriptReader<Fr, PoseidonSponge<Fr>> =
            TranscriptReader::new(transcript_writer.get_transcript());

        let path_len = log2(test_comm.encoded_num_cols);
        let col_vals = transcript_reader
            .consume_elements("Column elements", test_comm.n_rows)
            .unwrap();
        let merkle_path_vals = transcript_reader
            .consume_elements("Merkle path", path_len)
            .unwrap();

        let column_from_transcript = LcColumn {
            col_idx: col_num,
            col: col_vals,
            path: merkle_path_vals,
            phantom_data: std::marker::PhantomData,
        };

        let column_hash_check = verify_column_path::<LigeroAuxInfo<Fr>, Fr>(
            &column_from_transcript,
            col_num,
            root.as_ref(),
            &master_default_poseidon_merkle_hasher,
            &master_default_poseidon_column_hasher,
        );
        let column_value_check = verify_column_value::<LigeroAuxInfo<Fr>, Fr>(
            &column_from_transcript,
            &[],
            &Fr::from(0_u64),
        );

        assert!(column_hash_check && column_value_check);
    }
}

/// This test checks that the [CanonicalDeserialize] and [CanonicalSerialize]
/// derivations for an arbitrary struct work as intended. More of an example
/// snippet than a true test!
#[test]
fn arkworks_serialize_test() {
    // Example from https://docs.rs/ark-serialize/latest/ark_serialize/
    use ark_bn254::Fr;
    use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
    let one = Fr::from(1_u32);
    let two = Fr::from(2_u32);
    let mut one_compressed_bytes: Vec<u8> = Vec::new();
    let mut two_uncompressed_bytes: Vec<_> = Vec::new();
    one.serialize_compressed(&mut one_compressed_bytes).unwrap();
    two.serialize_uncompressed(&mut two_uncompressed_bytes)
        .unwrap();
    let one_deserialized = Fr::deserialize_compressed(&*one_compressed_bytes).unwrap();
    let two_deserialized = Fr::deserialize_uncompressed(&*two_uncompressed_bytes).unwrap();
    assert_eq!(one_deserialized, one);
    assert_eq!(two_deserialized, two);

    // With derive for a struct
    #[derive(CanonicalSerialize, CanonicalDeserialize, PartialEq, Debug)]
    struct TestStruct {
        one: Fr,
        two: Fr,
    }

    let test_struct = TestStruct { one, two };

    let mut test_struct_bytes = Vec::new();
    let _ = test_struct.serialize_compressed(&mut test_struct_bytes);
    let test_struct_deserialized = TestStruct::deserialize_compressed(&*test_struct_bytes).unwrap();
    assert_eq!(test_struct, test_struct_deserialized);
}

/// This test serves as an example of using Arkworks' FFT and IFFT functionality.
#[test]
fn arkworks_bn_fft_test() {
    // Example from: https://github.com/arkworks-rs/algebra/blob/master/poly/src/domain/general.rs
    use ark_bn254::Fr;
    use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, Polynomial};
    use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};

    // IFFT a polynomial, then FFT the evaluations, and ensure the result is the same as the original thing
    let orig_coeffs = vec![Fr::from(1u8), Fr::from(2u8), Fr::from(3u8), Fr::from(4u8)];
    let small_domain = GeneralEvaluationDomain::<Fr>::new(8).unwrap();
    let fft_evals: Vec<Fr> = small_domain.ifft(&orig_coeffs);
    let ifft_coeffs: Vec<Fr> = small_domain.fft(&fft_evals);
    let orig_poly = DensePolynomial::from_coefficients_vec(orig_coeffs);
    let ifft_poly = DensePolynomial::from_coefficients_vec(ifft_coeffs);
    assert_eq!(orig_poly.degree(), 3);
    assert_eq!(ifft_poly.degree(), 3);
    assert_eq!(orig_poly, ifft_poly);
}

/// This test serves as an example of using Halo2's FFT and IFFT functionality.
#[test]
fn halo2_bn_fft_test() {
    use ark_std::log2;
    use ark_std::test_rng;
    use remainder_shared_types::Fr;

    let mut rng = test_rng();

    let log_num_coeffs = 10;
    let rho_inv = 4;
    let num_coeffs = 2_usize.pow(log_num_coeffs);
    let num_evals = num_coeffs * rho_inv;
    assert!(num_evals.is_power_of_two());
    let log_num_evals = log2(num_evals);

    let coeffs = repeat_with(|| Fr::from(rng.gen::<u64>()))
        .take(num_coeffs)
        .collect_vec();

    // Note that `2^{j + 1}` is the total number of evaluations you actually want, and `2^k` is the number of coeffs
    let evaluation_domain: EvaluationDomain<Fr> =
        EvaluationDomain::new(rho_inv as u32, log_num_coeffs);

    // Creates the polynomial in coeff form and performs the FFT from 2^3 coeffs --> 2^3 evals
    let polynomial_coeff = evaluation_domain.coeff_from_vec(coeffs);
    let polynomial_eval_form = evaluation_domain.coeff_to_extended(polynomial_coeff.clone());
    assert_eq!(polynomial_eval_form.len(), 2_usize.pow(log_num_evals));

    // Perform the IFFT and assert that the resulting polynomial has degree 7
    let ifft_coeffs = evaluation_domain.extended_to_coeff(polynomial_eval_form);
    let orig_raw_coeffs = polynomial_coeff.iter().collect_vec();
    let ifft_raw_coeffs = ifft_coeffs.into_iter().collect_vec();

    // All coefficients past the original should be zero
    ifft_raw_coeffs
        .clone()
        .into_iter()
        .skip(2_usize.pow(log_num_coeffs))
        .for_each(|coeff| {
            assert_eq!(coeff, Fr::zero());
        });

    // IFFT'd coefficients should match the original
    orig_raw_coeffs
        .into_iter()
        .zip(ifft_raw_coeffs)
        .for_each(|(x, y)| {
            assert_eq!(*x, y);
        });
}

/// This test confirms the functionality of the initial Ligero commitment step by
/// * Computing a Ligero matrix M from its corresponding MLE coefficients.
/// * Computing its encoding, M', row-wise.
/// * Computing the matrix-vector product b^T M'
/// * Computing the reverse encoding IFFT(b^T M')
/// * Computing the matrix-vector product IFFT(b^T M') a, and checking it against
///     b^T M a.
#[test]
fn poseidon_commit_test() {
    use super::poseidon_ligero::PoseidonSpongeHasher;
    use super::{commit, eval_outer, eval_outer_fft};
    use ark_std::test_rng;
    use remainder_shared_types::Fr;

    // Grabs random (univariate poly!) coefficients and the rho value
    const MLE_NUM_VARS: usize = 4;
    const RHO_INV: u8 = 4;
    const RATIO: f64 = 4.0;
    let mut rng = test_rng();
    let random_mle_coeffs = get_random_coeffs_for_multilinear_poly(MLE_NUM_VARS, &mut rng);

    // Preps the FFT encoding and grabs the matrix size, then computes the commitment
    let enc = LigeroAuxInfo::<Fr>::new(random_mle_coeffs.len(), RHO_INV, RATIO, None);
    let comm =
        commit::<PoseidonSpongeHasher<Fr>, LigeroAuxInfo<_>, Fr>(&random_mle_coeffs, &enc).unwrap();

    // For a univariate commitment, `x` is the eval point
    let x = Fr::from(rng.gen::<u64>());

    // Zipping the coefficients against 1, x, x^2, ...
    let eval = comm
        .coeffs
        .iter()
        // Just computing 1, x, x^2, ...
        .zip(iterate(Fr::from(1), |&v| v * x).take(random_mle_coeffs.len()))
        .fold(Fr::from(0), |acc, (c, r)| acc + *c * r);

    // The "a" vector in b^T M a (the one which increments by ones)
    let roots_lo: Vec<Fr> = iterate(Fr::from(1), |&v| v * x)
        .take(comm.orig_num_cols)
        .collect();

    // The "b" vector in b^T M a (the one which increments by sqrt(N))
    let roots_hi: Vec<Fr> = {
        let xr = x * roots_lo.last().unwrap(); // x * x^{sqrt(N) - 1} --> x^{sqrt(N)}
        iterate(Fr::from(1), |&v| v * xr)
            .take(comm.n_rows)
            .collect()
    };

    // Literally does b^T M (I'm pretty sure)
    let coeffs_flattened = eval_outer(&comm, &roots_hi[..]).unwrap();

    // Then does (b^T M) a (I'm pretty sure)
    let eval2 = coeffs_flattened
        .iter()
        .zip(roots_lo.iter())
        .fold(Fr::from(0), |acc, (c, r)| acc + *c * r);

    // Basically the big tensor product and the actual polynomial evaluation should be the same
    assert_eq!(eval, eval2);

    // Compute b^T M' (RLC of the columns in encoded M'), which should be a codeword as well
    let poly_fft = eval_outer_fft(&comm, &roots_hi[..]).unwrap();
    let coeffs = halo2_ifft(poly_fft, RHO_INV);

    // So after the IFFT, we should receive a univariate polynomial of degree (num cols in M)
    assert!(coeffs
        .iter()
        .skip(comm.orig_num_cols)
        .all(|&v| v == Fr::from(0)));

    // And if we "evaluate" this polynomial (b^T M, in theory) against `a`, we should still
    // get the same evaluation
    let eval3 = coeffs
        .iter()
        .zip(roots_lo.iter())
        .fold(Fr::from(0), |acc, (c, r)| acc + *c * r);
    assert_eq!(eval2, eval3);
}

/// This test confirms the functionality of the [prove()] and [verify()] functions
/// within Ligero, by
/// * Computing a Ligero commitment from a random MLE.
/// * Computing its evaluation at a random challenge point.
/// * Computing an evaluation proof.
/// * Verifying the (commitment, evaluation proof) against the corresponding
///     challenge point + claimed evaluation.
#[test]
fn poseidon_end_to_end_test() {
    use super::poseidon_ligero::PoseidonSpongeHasher;
    use super::{commit, prove, verify};
    use ark_std::test_rng;
    use remainder_shared_types::Fr;

    // RNG for testing
    let mut rng = test_rng();
    let ml_num_vars = 8;

    // commit to a random polynomial at a random rate
    let coeffs = get_random_coeffs_for_multilinear_poly(ml_num_vars, &mut rng);
    let rho_inv = 4;
    let ratio = 1_f64;
    let enc = LigeroAuxInfo::<Fr>::new(coeffs.len(), rho_inv, ratio, None);
    let comm = commit::<PoseidonSpongeHasher<Fr>, LigeroAuxInfo<Fr>, Fr>(&coeffs, &enc).unwrap();

    // this is the polynomial commitment
    let root: LcRoot<LigeroAuxInfo<Fr>, Fr> = comm.get_root();

    // For a univariate commitment, `x` is the eval point
    let x = Fr::from(rng.gen::<u64>());

    // Zipping the coefficients against 1, x, x^2, ... to compute the evaluation.
    let eval = comm
        .coeffs
        .iter()
        // Just computing 1, x, x^2, ...
        .zip(iterate(Fr::from(1), |&v| v * x).take(coeffs.len()))
        .fold(Fr::from(0), |acc, (c, r)| acc + *c * r);

    // The "a" vector in b^T M a (the one which increments by ones)
    let inner_tensor: Vec<Fr> = iterate(Fr::from(1), |&v| v * x)
        .take(comm.orig_num_cols)
        .collect();

    // The "b" vector in b^T M a (the one which increments by sqrt(N))
    let outer_tensor: Vec<Fr> = {
        let xr = x * inner_tensor.last().unwrap(); // x * x^{sqrt(N) - 1} --> x^{sqrt(N)}
        iterate(Fr::from(1), |&v| v * xr)
            .take(comm.n_rows)
            .collect()
    };

    // Replacing the old transcript with the Remainder one
    let mut transcript_writer = TranscriptWriter::<Fr, PoseidonSponge<Fr>>::new("test transcript");

    // Transcript includes the Merkle root, the code rate, and the number of columns to be sampled
    transcript_writer.append("polycommit", root.root);

    prove(&comm, &outer_tensor[..], &enc, &mut transcript_writer).unwrap();

    let transcript = transcript_writer.get_transcript();
    let mut transcript_reader = TranscriptReader::<Fr, PoseidonSponge<Fr>>::new(transcript);
    let prover_root = transcript_reader.consume_element("polycommit").unwrap();
    assert_eq!(prover_root, root.root);

    // Verify the proof and return the prover-claimed result
    let res = verify(
        root.as_ref(),
        &outer_tensor[..],
        &inner_tensor[..],
        &enc,
        &mut transcript_reader,
    )
    .unwrap();

    // Checks that both evaluations are correct
    assert_eq!(res, eval);
}
