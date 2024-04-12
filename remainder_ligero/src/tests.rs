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
    collapse_columns,
    ligero_ml_helper::{get_ml_inner_outer_tensors, naive_eval_mle_at_challenge_point},
    ligero_structs::{LigeroCommit, LigeroEncoding, LigeroEvalProof},
    poseidon_ligero::PoseidonSpongeHasher,
    utils::{get_random_coeffs_for_multilinear_poly, halo2_ifft},
};

// --- For serialization/deserialization of the various structs ---
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use halo2_proofs::poly::EvaluationDomain;
// --- For BN-254 ---
use itertools::{iterate, Itertools};
use rand::Rng;
use remainder_shared_types::{transcript::poseidon_transcript::PoseidonSponge, FieldExt};
use remainder_shared_types::{
    transcript::{TranscriptReader, TranscriptWriter},
    Poseidon,
};
use std::{iter::repeat_with, marker::PhantomData};

const LIGERO_AUX_INFO_FILENAME: &str = "ligero_aux_info.txt";
const LIGERO_PROOF_FILENAME: &str = "ligero_proof.txt";
const LIGERO_ROOT_FILENAME: &str = "ligero_root.txt";

#[test]
fn log2() {
    use super::log2;

    for idx in 0..31 {
        assert_eq!(log2(1usize << idx), idx);
    }
}

#[test]
fn merkleize() {
    use remainder_shared_types::Fr;

    // --- This is SUPER ugly, but for the sake of efficiency... ---
    // TODO!(ryancao): Riperoni
    let master_default_poseidon_merkle_hasher = Poseidon::<Fr, 3, 2>::new(8, 57);
    let master_default_poseidon_column_hasher = Poseidon::<Fr, 3, 2>::new(8, 57);

    use super::{merkleize, merkleize_ser};

    let mut test_comm = random_comm::<Fr>();
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

#[test]
fn eval_outer() {
    use remainder_shared_types::Fr;

    use super::{eval_outer, eval_outer_ser};

    let test_comm = random_comm();
    let mut rng = rand::thread_rng();
    let tensor: Vec<Fr> = repeat_with(|| Fr::from(rng.gen::<u64>()))
        .take(test_comm.n_rows)
        .collect();

    let res1 = eval_outer(&test_comm, &tensor[..]).unwrap();
    let res2 = eval_outer_ser(&test_comm, &tensor[..]).unwrap();

    assert_eq!(&res1[..], &res2[..]);
}

#[test]
fn open_column() {
    use super::{merkleize, open_column, verify_column};
    use remainder_shared_types::Fr;

    // --- This is SUPER ugly, but for the sake of efficiency... ---
    // TODO!(ryancao): Riperoni
    let master_default_poseidon_merkle_hasher = Poseidon::<Fr, 3, 2>::new(8, 57);
    let master_default_poseidon_column_hasher = Poseidon::<Fr, 3, 2>::new(8, 57);

    let mut rng = rand::thread_rng();

    let test_comm = {
        let mut tmp = random_comm::<Fr>();
        merkleize(&mut tmp);
        tmp
    };

    let root = test_comm.get_root();
    for _ in 0..64 {
        let col_num = rng.gen::<usize>() % test_comm.encoded_num_cols;
        let column = open_column(&test_comm, col_num).unwrap();
        assert!(verify_column::<
            PoseidonSpongeHasher<Fr>,
            LigeroEncoding<Fr>,
            Fr,
        >(
            &column,
            col_num,
            root.as_ref(),
            &[],
            &Fr::from(0_u64),
            &master_default_poseidon_column_hasher,
            &master_default_poseidon_merkle_hasher
        ));
    }
}

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

    // --- With derive for a struct ---
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

#[test]
fn arkworks_bn_fft_test() {
    // Example from: https://github.com/arkworks-rs/algebra/blob/master/poly/src/domain/general.rs
    use ark_bn254::Fr;
    // use ark_std::test_rng;
    use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, Polynomial};
    use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
    // use ark_ff::FftField;

    // --- Let's try IFFT-ing a polynomial, then FFT-ing it, and seeing if we get back the same thing ---
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

    // let coeffs = vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(4), Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(4)];
    let coeffs = repeat_with(|| Fr::from(rng.gen::<u64>()))
        .take(num_coeffs)
        .collect_vec();

    // --- Note that `2^{j + 1}` is the total number of evaluations you actually want, and `2^k` is the number of coeffs ---
    let evaluation_domain: EvaluationDomain<Fr> =
        EvaluationDomain::new(rho_inv as u32, log_num_coeffs);

    // --- Creates the polynomial in coeff form and performs the FFT from 2^3 coeffs --> 2^3 evals ---
    let polynomial_coeff = evaluation_domain.coeff_from_vec(coeffs);
    let polynomial_eval_form = evaluation_domain.coeff_to_extended(&polynomial_coeff);
    assert_eq!(polynomial_eval_form.len(), 2_usize.pow(log_num_evals));

    // --- Perform the IFFT and assert that the resulting polynomial has degree 7 ---
    let ifft_coeffs = evaluation_domain.extended_to_coeff(polynomial_eval_form);
    let orig_raw_coeffs = polynomial_coeff.iter().collect_vec();
    let ifft_raw_coeffs = ifft_coeffs.into_iter().collect_vec();

    // --- All coefficients past the original should be zero ---
    ifft_raw_coeffs
        .clone()
        .into_iter()
        .skip(2_usize.pow(log_num_coeffs))
        .for_each(|coeff| {
            assert_eq!(coeff, Fr::zero());
        });

    // --- IFFT'd coefficients should match the original ---
    orig_raw_coeffs
        .into_iter()
        .zip(ifft_raw_coeffs.into_iter())
        .for_each(|(x, y)| {
            assert_eq!(*x, y);
        });
}

#[test]
fn poseidon_commit_test() {
    use super::poseidon_ligero::PoseidonSpongeHasher;
    use super::{commit, eval_outer, eval_outer_fft};
    use ark_std::test_rng;
    use remainder_shared_types::Fr;

    // --- RNG for testing ---
    let mut rng = test_rng();

    // --- Grabs random (univariate poly!) coefficients and the rho value ---
    let (coeffs, rho) = random_coeffs_rho_bn254::<Fr>();
    let rho_inv: u8 = (1.0 / rho) as u8;
    let ratio: f64 = 1_f64;

    // --- Preps the FFT encoding and grabs the matrix size ---
    let enc = LigeroEncoding::<Fr>::new(coeffs.len(), rho, ratio);

    // --- Commitment needs to use PoseidonHasher instead of Blake3 ---
    let comm = commit::<PoseidonSpongeHasher<Fr>, LigeroEncoding<_>, Fr>(&coeffs, &enc).unwrap();

    // --- For a univariate commitment, `x` is the eval point ---
    let x = Fr::from(rng.gen::<u64>());

    // --- Zipping the coefficients against 1, x, x^2, ... ---
    // Literally computing the evaluation. I'm dumb - Ryan
    let eval = comm
        .coeffs
        .iter()
        // --- Just computing 1, x, x^2, ... ---
        .zip(iterate(Fr::from(1), |&v| v * x).take(coeffs.len()))
        .fold(Fr::from(0), |acc, (c, r)| acc + *c * r);

    // --- The "a" vector in b^T M a (the one which increments by ones) ---
    let roots_lo: Vec<Fr> = iterate(Fr::from(1), |&v| v * x)
        .take(comm.orig_num_cols)
        .collect();

    // --- The "b" vector in b^T M a (the one which increments by sqrt(N)) ---
    let roots_hi: Vec<Fr> = {
        let xr = x * roots_lo.last().unwrap(); // x * x^{sqrt(N) - 1} --> x^{sqrt(N)}
        iterate(Fr::from(1), |&v| v * xr)
            .take(comm.n_rows)
            .collect()
    };

    // --- Literally does b^T M (I'm pretty sure) ---
    let coeffs_flattened = eval_outer(&comm, &roots_hi[..]).unwrap();

    // --- Then does (b^T M) a (I'm pretty sure) ---
    let eval2 = coeffs_flattened
        .iter()
        .zip(roots_lo.iter())
        .fold(Fr::from(0), |acc, (c, r)| acc + *c * r);

    // --- Basically the big tensor product and the actual polynomial evaluation should be the same ---
    assert_eq!(eval, eval2);

    // --- Need to confirm this, but sounds like it does b^T M'...? (RLC of the columns in encoded M') ---
    let poly_fft = eval_outer_fft(&comm, &roots_hi[..]).unwrap();

    // Okay so `poly_fft` gives us
    // (b_1 * f_1(1) + ... + b_{\sqrt{N}} * f_{\sqrt{N}}(1),
    // (b_1 * f_1(2) + ... + b_{\sqrt{N}} * f_{\sqrt{N}}(2),
    // (b_1 * f_1(3) + ... + b_{\sqrt{N}} * f_{\sqrt{N}}(3),
    // ...,
    // b_1 * f_1(\rho^{-1} * \sqrt{N}) + ... + b_{\sqrt{N}} * f_{\sqrt{N}}(\rho^{-1} * \sqrt{N})
    // Note that coordinates all have the same evaluation point,
    // and we sum across linear combination of functions f_1, ..., f_{\sqrt{N}}
    // with coefficients b_1, ..., b_{\sqrt{N}}
    // ---
    // Okay so after the IFFT, we should basically get the coefficients of
    // (b_1 * f_1 + ... + b_{\sqrt{N}} f_{\sqrt{N}})(x)
    let coeffs = halo2_ifft(poly_fft, rho_inv);

    // --- So after the IFFT, we should receive a univariate polynomial of degree (num cols in M) ---
    assert!(coeffs
        .iter()
        .skip(comm.orig_num_cols)
        .all(|&v| v == Fr::from(0)));

    // --- And if we "evaluate" this polynomial (b^T M, in theory) against `a`, we should still ---
    // --- get the same evaluation ---
    let eval3 = coeffs
        .iter()
        .zip(roots_lo.iter())
        .fold(Fr::from(0), |acc, (c, r)| acc + *c * r);
    assert_eq!(eval2, eval3);
}

#[test]
fn poseidon_end_to_end_test() {
    use super::poseidon_ligero::PoseidonSpongeHasher;
    use super::{commit, prove, verify};
    use ark_std::test_rng;
    use remainder_shared_types::Fr;

    // --- RNG for testing ---
    let mut rng = test_rng();

    // commit to a random polynomial at a random rate
    let (coeffs, rho) = random_coeffs_rho_bn254();
    let ratio = 1_f64;
    let enc = LigeroEncoding::<Fr>::new(coeffs.len(), rho, ratio);
    let comm = commit::<PoseidonSpongeHasher<Fr>, LigeroEncoding<Fr>, Fr>(&coeffs, &enc).unwrap();

    // this is the polynomial commitment
    let root: LcRoot<LigeroEncoding<Fr>, Fr> = comm.get_root();

    // --- For a univariate commitment, `x` is the eval point ---
    let x = Fr::from(rng.gen::<u64>());

    // --- Zipping the coefficients against 1, x, x^2, ... ---
    // Literally computing the evaluation. I'm dumb - Ryan
    let eval = comm
        .coeffs
        .iter()
        // --- Just computing 1, x, x^2, ... ---
        .zip(iterate(Fr::from(1), |&v| v * x).take(coeffs.len()))
        .fold(Fr::from(0), |acc, (c, r)| acc + *c * r);

    // compute the outer and inner tensors for powers of x
    // NOTE: we treat coeffs as a univariate polynomial, but it doesn't
    // really matter --- the only difference from a multilinear is the
    // way we compute outer_tensor and inner_tensor from the eval point

    // --- The "a" vector in b^T M a (the one which increments by ones) ---
    let inner_tensor: Vec<Fr> = iterate(Fr::from(1), |&v| v * x)
        .take(comm.orig_num_cols)
        .collect();

    // --- The "b" vector in b^T M a (the one which increments by sqrt(N)) ---
    let outer_tensor: Vec<Fr> = {
        let xr = x * inner_tensor.last().unwrap(); // x * x^{sqrt(N) - 1} --> x^{sqrt(N)}
        iterate(Fr::from(1), |&v| v * xr)
            .take(comm.n_rows)
            .collect()
    };

    // --- The above is all the same as the `commit` test functionality ---

    // compute an evaluation proof
    // --- Replacing the old transcript with the Remainder one ---
    let mut transcript_writer = TranscriptWriter::<Fr, PoseidonSponge<Fr>>::new("test transcript");

    // --- Transcript includes the Merkle root, the code rate, and the number of columns to be sampled ---
    transcript_writer.append("polycommit", root.root);

    // TODO!(ryancao): Uhhhhhhh figure out how to add the rate to the transcript...
    // tr1.append_field_element("rate", rho);
    // let _ = tr1.append_field_element("ncols", Fr::from(N_COL_OPENS as u64));

    // --- Okay this function is new; let's check it out ---
    // Tl;dr this gives us the random vectors to check well-formedness from
    // As well as the actual columns we're opening at, plus proofs that those
    // columns are consistent against the Merkle root
    let pf: LigeroEvalProof<PoseidonSpongeHasher<Fr>, LigeroEncoding<Fr>, Fr> =
        prove(&comm, &outer_tensor[..], &enc, &mut transcript_writer).unwrap();

    // ------------------- SERIALIZATION -------------------
    // --- Serializing the auxiliaries ---
    // let ligero_proof_aux_info = LcProofAuxiliaryInfo {
    //     // TODO!(ryancao): Is this rounding correctly? I think only for values of `rho` which are powers of two...?
    //     rho_inv: (1.0 / rho) as u8,
    //     encoded_num_cols: comm.encoded_num_cols,
    //     orig_num_cols: comm.orig_num_cols,
    //     num_rows: comm.n_rows,
    // };
    // let mut encoded_proof_aux_info_bytes: Vec<u8> = Vec::new();
    // let _ = ligero_proof_aux_info.serialize_compressed(&mut encoded_proof_aux_info_bytes);

    // --- First things first, we want to serialize the LigeroEvalProof ---
    // let mut encoded_proof_bytes: Vec<u8> = Vec::new();
    // pf.serialize_compressed(&mut encoded_proof_bytes).unwrap();

    // --- Next, we want to serialize the commitment root ---
    // let mut encoded_root_bytes: Vec<u8> = Vec::new();
    // let lc_root_to_be_encoded = LcRoot::<LigeroEncoding<Fr>, Fr> {
    //     root: root.root,
    //     _p: PhantomData
    // };
    // let _ = lc_root_to_be_encoded.serialize_compressed(&mut encoded_root_bytes).unwrap();

    // --- Write both things (root and LigeroProof) to file ---
    // fs::write(LIGERO_AUX_INFO_FILENAME, encoded_proof_aux_info_bytes).expect("Unable to write proof auxiliaries to file");
    // fs::write(LIGERO_PROOF_FILENAME, encoded_proof_bytes.clone()).expect("Unable to write proof to file");
    // fs::write(LIGERO_ROOT_FILENAME, encoded_root_bytes.clone()).expect("Unable to write root to file");

    // verify it and finish evaluation
    // Q: Why do we have a second transcript???
    // Answer: I guess it's because we're simulating the verifier receiving the transcript
    // ...Perhaps there's also some state reset stuff that needs to be done?
    let transcript = transcript_writer.get_transcript();
    let mut transcript_reader = TranscriptReader::<Fr, PoseidonSponge<Fr>>::new(transcript);
    let prover_root = transcript_reader.consume_element("polycommit").unwrap();
    assert_eq!(prover_root, root.root);
    // TODO!(ryancao): Uhhhhhhh figure out how to add the rate to the transcript...
    // tr2.append_field_element("rate", rho);
    // let _ = tr2.append_field_element("ncols", Fr::from(N_COL_OPENS as u64));

    // (ryancao): Oh hmmm is this where they test the randomly generated vector...?
    // Answer: No, it's just them setting up another FFT to use in `verify()`
    // TODO!(ryancao): Get rid of this entirely, or have some other way of just passing along the `orig_num_cols` and `encoded_num_cols`
    let enc2 =
        LigeroEncoding::<Fr>::new_from_dims(pf.get_orig_num_cols(), pf.get_encoded_num_cols());

    // --- We need to check out this `verify()` function ---
    // --- Okay basically checks that
    // a) All the `r^T M'` s (i.e. column-wise) are consistent with the verifier-derived enc(r^T M)
    // b) All the `b^T M'` s (i.e. column-wise) are consistent with the verifier-derived enc(b^T M)
    // c) All the columns are consistent with the merkle commitment
    // d) Evaluates (b^T M) * a on its own (where b^T M is given by the prover) and returns the result
    //      as the evaluation
    let res = verify(
        root.as_ref(),
        &outer_tensor[..],
        &inner_tensor[..],
        &pf,
        &enc2,
        &mut transcript_reader,
    )
    .unwrap();

    // --- Does the same thing (i.e. verification), but after deserializing ---
    // let encoded_proof_bytes_2 = fs::read(LIGERO_PROOF_FILENAME).expect("Unable to read proof from file");
    // let encoded_root_bytes_2 = fs::read(LIGERO_ROOT_FILENAME).expect("Unable to read root from file");

    // let root2 = LcRoot::<LigeroEncoding<Fr>, Fr>::deserialize_compressed(&*encoded_root_bytes_2).unwrap();
    // let pf2 = LigeroEvalProof::<PoseidonSpongeHasher<Fr>, LigeroEncoding<Fr>, Fr>::deserialize_compressed(&*encoded_proof_bytes_2).unwrap();

    // let mut tr3 = PoseidonTranscript::new("test transcript 3");
    // let _ = tr3.append_field_element("polycommit", root.root);
    // TODO!(ryancao): Uhhhhhhh figure out how to add the rate to the transcript...
    // tr3.append_field_element("rate", rho);
    // let _ = tr3.append_field_element("ncols", Fr::from(N_COL_OPENS as u64));

    // let enc3 = LigeroEncoding::<Fr>::new_from_dims(pf2.get_orig_num_cols(), pf2.get_encoded_num_cols());
    // let res2 = verify(
    //     root2.as_ref(),
    //     &outer_tensor[..],
    //     &inner_tensor[..],
    //     &pf2,
    //     &enc3,
    //     &mut tr3,
    // ).unwrap();

    // --- Checks that both evaluations are correct ---
    assert_eq!(res, eval);
    // assert_eq!(res, res2);
}

/// Poseidon multilinear commitment test (checking that the matrix generated is correct
/// when one computes the inner + outer tensor products vs. the actual evaluation)
#[test]
fn poseidon_ml_commit_test() {
    use super::poseidon_ligero::PoseidonSpongeHasher;
    use super::{commit, eval_outer, eval_outer_fft};
    use ark_std::test_rng;
    use remainder_shared_types::Fr;

    // --- RNG for testing ---
    let mut rng = test_rng();

    // --- Generate a random polynomial and set rho ---
    let rho_inv = 4;
    let rho = 1. / (rho_inv as f64);
    let num_ml_vars = 16;
    let log_num_rows = 8;
    let num_rows = 2_usize.pow(log_num_rows);
    let log_orig_num_cols = 8;
    let orig_num_cols = 2_usize.pow(log_orig_num_cols);
    let coeffs = get_random_coeffs_for_multilinear_poly::<Fr>(num_ml_vars);
    let ratio = 1_f64;

    // --- Create commitment ---
    let enc = LigeroEncoding::<Fr>::new(coeffs.len(), rho, ratio);
    let comm = commit::<PoseidonSpongeHasher<Fr>, LigeroEncoding<Fr>, Fr>(&coeffs, &enc).unwrap();

    // --- Generating the random multilinear point ---
    let challenge_coord: Vec<Fr> = repeat_with(|| Fr::from(rng.gen::<u64>()))
        .take(num_ml_vars)
        .collect_vec();

    // --- Computing the raw evaluation of the MLE at the coordinate generated above ---
    let eval = naive_eval_mle_at_challenge_point(&comm.coeffs, &challenge_coord);

    // --- Compute "a" and "b" from `challenge_coord` ---
    let (inner_tensor, outer_tensor) =
        get_ml_inner_outer_tensors(&challenge_coord, num_rows, orig_num_cols);

    // --- Literally does b^T M (I'm pretty sure) ---
    let coeffs_flattened = eval_outer(&comm, &outer_tensor[..]).unwrap();

    // --- Then does (b^T M) a (I'm pretty sure) ---
    let eval2 = coeffs_flattened
        .iter()
        .zip(inner_tensor.iter())
        .fold(Fr::from(0), |acc, (c, r)| acc + *c * r);

    // --- Basically the big tensor product and the actual polynomial evaluation should be the same ---
    assert_eq!(eval, eval2);

    // --- Need to confirm this, but sounds like it does b^T M'...? (RLC of the columns in encoded M') ---
    let poly_fft = eval_outer_fft(&comm, &outer_tensor[..]).unwrap();

    // Okay so `poly_fft` gives us
    // (b_1 * f_1(1) + ... + b_{\sqrt{N}} * f_{\sqrt{N}}(1),
    // (b_1 * f_1(2) + ... + b_{\sqrt{N}} * f_{\sqrt{N}}(2),
    // (b_1 * f_1(3) + ... + b_{\sqrt{N}} * f_{\sqrt{N}}(3),
    // ...,
    // b_1 * f_1(\rho^{-1} * \sqrt{N}) + ... + b_{\sqrt{N}} * f_{\sqrt{N}}(\rho^{-1} * \sqrt{N})
    // Note that coordinates all have the same evaluation point,
    // and we sum across linear combination of functions f_1, ..., f_{\sqrt{N}}
    // with coefficients b_1, ..., b_{\sqrt{N}}
    // ---
    // Okay so after the IFFT, we should basically get the coefficients of
    // (b_1 * f_1 + ... + b_{\sqrt{N}} f_{\sqrt{N}})(x)
    // --- TODO!(ryancao): Does this truncate AND round to the next largest power of 2? ---
    let coeffs = halo2_ifft(poly_fft, rho_inv);

    // --- So after the IFFT, we should receive a univariate polynomial of degree (num cols in M) ---
    assert!(coeffs
        .iter()
        .skip(comm.orig_num_cols)
        .all(|&v| v == Fr::from(0)));

    // --- And if we "evaluate" this polynomial (b^T M, in theory) against `a`, we should still ---
    // --- get the same evaluation ---
    let eval3 = coeffs
        .iter()
        .zip(inner_tensor.iter())
        .fold(Fr::from(0), |acc, (c, r)| acc + *c * r);
    assert_eq!(eval2, eval3);
}

/// Poseidon multilinear end-to-end test
#[test]
fn poseidon_ml_end_to_end_test() {
    use super::poseidon_ligero::PoseidonSpongeHasher;
    use super::{commit, prove, verify};
    use ark_std::test_rng;
    use remainder_shared_types::Fr;
    // use std::fs;

    // --- RNG for testing ---
    let mut rng = test_rng();

    // --- Generate a random polynomial and set rho ---
    let rho_inv = 4;
    let rho = 1. / (rho_inv as f64);
    let num_ml_vars = 8;
    let log_num_rows = 4;
    let num_rows = 2_usize.pow(log_num_rows);
    let log_orig_num_cols = 4;
    let orig_num_cols = 2_usize.pow(log_orig_num_cols);
    let coeffs = get_random_coeffs_for_multilinear_poly::<Fr>(num_ml_vars);

    // --- Create commitment ---
    let ratio = 1_f64;
    let enc = LigeroEncoding::<Fr>::new(coeffs.len(), rho, ratio);
    let comm = commit::<PoseidonSpongeHasher<Fr>, LigeroEncoding<Fr>, Fr>(&coeffs, &enc).unwrap();

    // --- Only component of commitment which needs to be sent to the verifier is the commitment root ---
    let root: LcRoot<LigeroEncoding<Fr>, Fr> = comm.get_root();

    // --- Generating the random multilinear point ---
    let challenge_coord: Vec<Fr> = repeat_with(|| Fr::from(rng.gen::<u64>()))
        .take(num_ml_vars)
        .collect_vec();

    // --- Computing the raw evaluation of the MLE at the coordinate generated above ---
    let eval = naive_eval_mle_at_challenge_point(&comm.coeffs, &challenge_coord);

    // --- Compute "a" and "b" from `challenge_coord` ---
    let mut poly = vec![Fr::zero(); comm.orig_num_cols];
    let (inner_tensor, outer_tensor) =
        get_ml_inner_outer_tensors(&challenge_coord, num_rows, orig_num_cols);

    // --- Sanitycheck that b^T M a is equivalent to the naive evaluation ---
    collapse_columns::<LigeroEncoding<Fr>, Fr>(
        &coeffs,
        &outer_tensor,
        &mut poly[..],
        num_rows,
        orig_num_cols,
        0,
    );
    let b_transpose_m_times_a: Fr = poly
        .into_iter()
        .zip(inner_tensor.clone().into_iter())
        .fold(Fr::zero(), |acc, (poly_coeff, inner_tensor_coeff)| {
            acc + (poly_coeff * inner_tensor_coeff)
        });
    assert_eq!(b_transpose_m_times_a, eval);

    // ------------ The above is all the same as the `commit` test functionality ------------

    // --- Replacing the old transcript with the Remainder one ---
    let mut transcript_writer = TranscriptWriter::<Fr, PoseidonSponge<Fr>>::new("test transcript");

    // --- Transcript includes the Merkle root, the code rate, and the number of columns to be sampled ---
    transcript_writer.append("polycommit", root.root);
    // let _ = tr1.append_field_element("rate", Fr::from(rho_inv));
    // let _ = tr1.append_field_element("ncols", Fr::from(N_COL_OPENS as u64));

    // --- Okay this function is new; let's check it out ---
    // Tl;dr this gives us the random vectors to check well-formedness from
    // As well as the actual columns we're opening at, plus proofs that those
    // columns are consistent against the Merkle root
    let pf: LigeroEvalProof<PoseidonSpongeHasher<Fr>, LigeroEncoding<Fr>, Fr> =
        prove(&comm, &outer_tensor[..], &enc, &mut transcript_writer).unwrap();

    // ------------------- SERIALIZATION -------------------
    // --- Serializing the auxiliaries ---
    // let ligero_proof_aux_info = LcProofAuxiliaryInfo {
    //     // TODO!(ryancao): Is this rounding correctly? I think only for values of `rho` which are powers of two...?
    //     rho_inv: (1.0 / rho) as u8,
    //     encoded_num_cols: comm.encoded_num_cols,
    //     orig_num_cols: comm.orig_num_cols,
    //     num_rows: comm.n_rows,
    // };
    // let mut encoded_proof_aux_info_bytes: Vec<u8> = Vec::new();
    // let _ = ligero_proof_aux_info.serialize_compressed(&mut encoded_proof_aux_info_bytes);

    // --- First things first, we want to serialize the LigeroEvalProof ---
    // let mut encoded_proof_bytes: Vec<u8> = Vec::new();
    // pf.serialize_compressed(&mut encoded_proof_bytes).unwrap();

    // --- Next, we want to serialize the commitment root ---
    // let mut encoded_root_bytes: Vec<u8> = Vec::new();
    // let lc_root_to_be_encoded = LcRoot::<LigeroEncoding<Fr>, Fr> {
    //     root: root.root,
    //     _p: PhantomData
    // };
    // let _ = lc_root_to_be_encoded.serialize_compressed(&mut encoded_root_bytes).unwrap();

    // --- Write both things (root and LigeroProof) to file ---
    // fs::write(LIGERO_AUX_INFO_FILENAME, encoded_proof_aux_info_bytes).expect("Unable to write proof auxiliaries to file");
    // fs::write(LIGERO_PROOF_FILENAME, encoded_proof_bytes.clone()).expect("Unable to write proof to file");
    // fs::write(LIGERO_ROOT_FILENAME, encoded_root_bytes.clone()).expect("Unable to write root to file");

    // verify it and finish evaluation
    // Q: Why do we have a second transcript???
    // Answer: I guess it's because we're simulating the verifier receiving the transcript
    // ...Perhaps there's also some state reset stuff that needs to be done?
    let transcript = transcript_writer.get_transcript();
    let mut transcript_reader = TranscriptReader::<Fr, PoseidonSponge<Fr>>::new(transcript);
    let prover_root = transcript_reader.consume_element("polycommit").unwrap();
    assert_eq!(prover_root, root.root);
    // let _ = tr2.append_field_element("rate", Fr::from(rho_inv));
    // let _ = tr2.append_field_element("ncols", Fr::from(N_COL_OPENS as u64));

    // (ryancao): Oh hmmm is this where they test the randomly generated vector...?
    // Answer: No, it's just them setting up another FFT to use in `verify()`
    // TODO!(ryancao): Get rid of this entirely, or have some other way of just passing along the `orig_num_cols` and `encoded_num_cols`
    let enc2 =
        LigeroEncoding::<Fr>::new_from_dims(pf.get_orig_num_cols(), pf.get_encoded_num_cols());

    // --- We need to check out this `verify()` function ---
    // --- Okay basically checks that
    // a) All the `r^T M'` s (i.e. column-wise) are consistent with the verifier-derived enc(r^T M)
    // b) All the `b^T M'` s (i.e. column-wise) are consistent with the verifier-derived enc(b^T M)
    // c) All the columns are consistent with the merkle commitment
    // d) Evaluates (b^T M) * a on its own (where b^T M is given by the prover) and returns the result
    //      as the evaluation
    let res = verify(
        root.as_ref(),
        &outer_tensor[..],
        &inner_tensor[..],
        &pf,
        &enc2,
        &mut transcript_reader,
    )
    .unwrap();

    // --- Does the same thing (i.e. verification), but after deserializing ---
    // let encoded_proof_bytes_2 = fs::read(LIGERO_PROOF_FILENAME).expect("Unable to read proof from file");
    // let encoded_root_bytes_2 = fs::read(LIGERO_ROOT_FILENAME).expect("Unable to read root from file");

    // let root2 = LcRoot::<LigeroEncoding<Fr>, Fr>::deserialize_compressed(&*encoded_root_bytes_2).unwrap();
    // let pf2 = LigeroEvalProof::<PoseidonSpongeHasher<Fr>, LigeroEncoding<Fr>, Fr>::deserialize_compressed(&*encoded_proof_bytes_2).unwrap();

    // let mut tr3 = PoseidonTranscript::new("test transcript 3");
    // let _ = tr3.append_field_element("polycommit", root.root);
    // let _ = tr3.append_field_element("rate", Fr::from(rho_inv));
    // let _ = tr3.append_field_element("ncols", Fr::from(N_COL_OPENS as u64));

    // let enc3 = LigeroEncoding::<Fr>::new_from_dims(pf2.get_orig_num_cols(), pf2.get_encoded_num_cols());
    // let res2 = verify(
    //     root2.as_ref(),
    //     &outer_tensor[..],
    //     &inner_tensor[..],
    //     &pf2,
    //     &enc3,
    //     &mut tr3,
    // ).unwrap();

    // --- Checks that both evaluations are correct ---
    assert_eq!(res, eval);
    // assert_eq!(res, res2);
}

/// Ryan's note -- this basically replaces the other `random_coeffs_rho()` but
/// returns ark_bn254::Fr instead of Ft63
/// TODO!(ryancao): Pass in the RNG if possible!
fn random_coeffs_rho_bn254<F: FieldExt>() -> (Vec<F>, f64) {
    // --- This is Riad's RNG ---
    let mut rng = rand::thread_rng();

    // --- LGL is the "log length" ---
    let lgl = 8 + rng.gen::<usize>() % 8;
    let len_base = 1 << (lgl - 1);
    let len = len_base + (rng.gen::<usize>() % len_base);

    (
        // --- Uniform random sampling over Fr (well, sort of...) ---
        repeat_with(|| F::from(rng.gen::<u64>()))
            .take(len)
            .collect(),
        // Rho is something between 0.1 and 0.9
        // rng.gen_range(0.1f64..0.9f64),
        0.25f64,
    )
}

// Honestly not sure why this function exists lol -- Ryan
fn random_comm<F: FieldExt>() -> LigeroCommit<PoseidonSpongeHasher<F>, F> {
    let mut rng = rand::thread_rng();

    let lgl = 8 + rng.gen::<usize>() % 8;
    let len_base = 1 << (lgl - 1);
    let len = len_base + (rng.gen::<usize>() % len_base);
    let rho = rng.gen_range(0.1f64..0.9f64);
    let ratio = 1_f64;
    let (n_rows, orig_num_cols, encoded_num_cols) =
        LigeroEncoding::<F>::get_dims(len, rho, ratio).unwrap();

    let coeffs_len = (orig_num_cols - 1) * n_rows + 1 + (rng.gen::<usize>() % n_rows);
    let coeffs = {
        let mut tmp = repeat_with(|| F::from(rng.gen::<u64>()))
            .take(coeffs_len)
            .collect::<Vec<F>>();
        tmp.resize(orig_num_cols * n_rows, F::zero());
        tmp
    };

    let comm_len = n_rows * encoded_num_cols;
    let comm: Vec<F> = repeat_with(|| F::from(rng.gen::<u64>()))
        .take(comm_len)
        .collect();

    LigeroCommit::<PoseidonSpongeHasher<F>, F> {
        comm,
        coeffs,
        n_rows,
        encoded_num_cols,
        orig_num_cols,
        hashes: vec![F::default(); 2 * encoded_num_cols - 1],
        phantom_data: PhantomData,
        phantom_data_2: PhantomData,
    }
}
