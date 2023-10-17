// Copyright 2021 Riad S. Wahby <rsw@cs.stanford.edu>
//
// This file is part of lcpc-2d, which is part of lcpc.
//
// Licensed under the Apache License, Version 2.0 (see
// LICENSE or https://www.apache.org/licenses/LICENSE-2.0).
// This file may not be copied, modified, or distributed
// except according to those terms.
#![deny(missing_docs)]

/*!
lcpc2d is a polynomial commitment scheme based on linear codes

The Remainder version of Ligero creates a non-interactive prover
transcript and uses Poseidon as the column, Merkle, and Fiat-Shamir
hashes. Additionally, it adds (explicit) multilinear functionality
to the codebase.
*/

use crate::utils::get_least_significant_bits_to_usize_little_endian;
use ark_ff::biginteger::BigInteger;
use ark_std::{start_timer, end_timer};
use err_derive::Error;
use itertools::Itertools;
use poseidon::Poseidon;
use poseidon_ligero::poseidon_digest::FieldHashFnDigest;
use poseidon_ligero::PoseidonSpongeHasher;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

// --- Actual field trait + transcript stuff ---
use remainder_shared_types::{
    transcript::{poseidon_transcript::PoseidonTranscript, Transcript as RemainderTranscript},
    FieldExt,
};

mod macros;

/// For converting between this codebase's types and the types the page would like to have
pub mod adapter;
/// Public functions for univariate and multilinear Ligero commitment (with Poseidon)
pub mod ligero_commit;
/// For multilinear commitment stuff
pub mod ligero_ml_helper;
/// For actual Ligero proof structs
pub mod ligero_structs;
/// For Poseidon hashing (implementation with respect to Digest and Transcript)
pub mod poseidon_ligero;
#[cfg(test)]
pub mod tests;
/// Helper functions
pub mod utils;

/// TODO!(ryancao): Perhaps we should rename this? After everything is working.
/// We are distinguishing it from `FieldHash` for now.
pub trait PoseidonFieldHash: FieldExt {
    /// Update the digest `d` with the `self` (since `self` should already be a field element)
    fn digest_update<D: FieldHashFnDigest<Self>>(&self, d: &mut D) {
        d.update(&[*self])
    }

    /// Update the [remainder::transcript::Transcript] with label `l` and element `self`
    fn transcript_update(&self, t: &mut impl RemainderTranscript<Self>, l: &'static str) {
        let _ = t.append_field_element(l, *self);
    }
}

// --- Ryan's addendum ---
impl<F: FieldExt> PoseidonFieldHash for F {
    fn digest_update<D: FieldHashFnDigest<F>>(&self, d: &mut D) {
        d.update(&[*self])
    }

    fn transcript_update(&self, t: &mut impl RemainderTranscript<F>, l: &'static str) {
        let _ = t.append_field_element(l, *self);
    }
}

/// Trait for a linear encoding used by the polycommit
pub trait LcEncoding<F: FieldExt>: Clone + std::fmt::Debug + Sync {
    /// Field over which coefficients are defined
    // type F: Field + FieldHash + std::fmt::Debug + Clone;
    // type F: FieldExt + std::fmt::Debug + Clone;

    /// Domain separation label - degree test (see def_labels!())
    const LABEL_DT: &'static [u8];
    /// Domain separation label - random lin combs (see def_labels!())
    const LABEL_PR: &'static [u8];
    /// Domain separation label - eval comb (see def_labels!())
    const LABEL_PE: &'static [u8];
    /// Domain separation label - column openings (see def_labels!())
    const LABEL_CO: &'static [u8];

    /// Error type for encoding
    type Err: std::fmt::Debug + std::error::Error + Send;

    /// Encoding function
    fn encode(&self, inp: &mut [F]) -> Result<(), Self::Err>;

    /// Get dimensions for this encoding instance on an input vector of length `len`
    fn get_dims(&self, len: usize) -> (usize, usize, usize);

    /// Check that supplied dimensions are compatible with this encoding
    fn dims_ok(&self, orig_num_cols: usize, encoded_num_cols: usize) -> bool;

    /// Get the number of column openings required for this encoding
    fn get_n_col_opens(&self) -> usize;

    /// Get the number of degree tests required for this encoding
    fn get_n_degree_tests(&self) -> usize;
}

// local accessors for enclosed types
type ErrT<E, F> = <E as LcEncoding<F>>::Err;

/// Err variant for prover operations
#[derive(Debug, Error)]
pub enum ProverError<ErrT>
where
    ErrT: std::fmt::Debug + std::error::Error + 'static,
{
    /// size too big
    #[error(display = "encoded_num_cols is too large for this encoding")]
    TooBig,
    /// error encoding a vector
    #[error(display = "encoding error: {:?}", _0)]
    Encode(#[source] ErrT),
    /// inconsistent LcCommit fields
    #[error(display = "inconsistent commitment fields")]
    Commit,
    /// bad column number
    #[error(display = "bad column number")]
    ColumnNumber,
    /// bad outer tensor
    #[error(display = "outer tensor: wrong size")]
    OuterTensor,
}

/// result of a prover operation
pub type ProverResult<T, ErrT> = Result<T, ProverError<ErrT>>;

/// Err variant for verifier operations
#[derive(Debug, Error)]
pub enum VerifierError<ErrT>
where
    ErrT: std::fmt::Debug + std::error::Error + 'static,
{
    /// wrong number of column openings in proof
    #[error(display = "wrong number of column openings in proof")]
    NumColOpens,
    /// failed to verify column merkle path
    #[error(display = "column verification: merkle path failed")]
    ColumnPath,
    /// failed to verify column dot product for poly eval
    #[error(display = "column verification: eval dot product failed")]
    ColumnEval,
    /// failed to verify column dot product for degree test
    #[error(display = "column verification: degree test dot product failed")]
    ColumnDegree,
    /// bad outer tensor
    #[error(display = "outer tensor: wrong size")]
    OuterTensor,
    /// bad inner tensor
    #[error(display = "inner tensor: wrong size")]
    InnerTensor,
    /// encoding dimensions do not match proof
    #[error(display = "encoding dimension mismatch")]
    EncodingDims,
    /// error encoding a vector
    #[error(display = "encoding error: {:?}", _0)]
    Encode(#[source] ErrT),
}

/// --- For encoding the matrix size and other useful info ---
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LcProofAuxiliaryInfo {
    /// Inverse of the encoding rate rho
    pub rho_inv: u8,
    /// Number of columns of the encoded matrix
    pub encoded_num_cols: usize,
    /// Number of columns of the original matrix
    pub orig_num_cols: usize,
    /// Number of rows of the matrix
    pub num_rows: usize,
}

/// result of a verifier operation
pub type VerifierResult<T, ErrT> = Result<T, VerifierError<ErrT>>;

/// a commitment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LcCommit<D, E, F>
where
{
    // --- Flattened version of M' (encoded) matrix ---
    comm: Vec<F>,
    // --- Flattened version of M (non-encoded) matrix ---
    coeffs: Vec<F>,
    // --- Matrix dims ---
    n_rows: usize,           // Height of M (and M')
    encoded_num_cols: usize, // Width of M'
    orig_num_cols: usize,    // Width of M
    // --- TODO!(ryancao): What this? ---
    hashes: Vec<F>,
    phantom_data: PhantomData<D>,
    phantom_data_2: PhantomData<E>,
}

impl<D, E, F> LcCommit<D, E, F>
where
    D: FieldHashFnDigest<F> + Send + Sync,
    F: FieldExt,
    E: LcEncoding<F> + Send + Sync,
{
    /// returns the Merkle root of this polynomial commitment (which is the commitment itself)
    pub fn get_root(&self) -> LcRoot<E, F> {
        LcRoot {
            root: (self.hashes.last().cloned().unwrap() as F),
            _p: Default::default(),
        }
    }

    /// return the number of coefficients encoded in each matrix row
    pub fn get_orig_num_cols(&self) -> usize {
        self.orig_num_cols
    }

    /// return the number of columns in the encoded matrix
    pub fn get_encoded_num_cols(&self) -> usize {
        self.encoded_num_cols
    }

    /// return the number of rows in the encoded matrix
    pub fn get_n_rows(&self) -> usize {
        self.n_rows
    }

    /// generate a commitment to a polynomial
    pub fn commit(coeffs: &[F], enc: &E) -> ProverResult<Self, ErrT<E, F>> {
        commit::<D, E, F>(coeffs, enc)
    }

    /// Generate an evaluation of a committed polynomial
    pub fn prove(
        &self,
        outer_tensor: &[F],
        enc: &E,
        tr: &mut PoseidonTranscript<F>,
    ) -> ProverResult<LcEvalProof<D, E, F>, ErrT<E, F>> {
        prove(self, outer_tensor, enc, tr)
    }
}

/// A Merkle root corresponding to a committed polynomial
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "F: FieldExt")]
pub struct LcRoot<E, F>
where
    F: FieldExt,
    E: LcEncoding<F> + Send + Sync,
{
    /// Root of the Merkle Tree
    pub root: F,
    _p: PhantomData<E>,
    // phantom_data: PhantomData<D>
}

impl<E, F> LcRoot<E, F>
where
    F: FieldExt,
    E: LcEncoding<F> + Send + Sync,
{
    /// Convert this value into a raw F
    pub fn into_raw(self) -> F {
        self.root
    }
}

impl<E, F> AsRef<F> for LcRoot<E, F>
where
    F: FieldExt,
    E: LcEncoding<F> + Send + Sync,
{
    fn as_ref(&self) -> &F {
        &self.root
    }
}

/// A column opening and the corresponding Merkle path.
#[derive(Debug, Clone)]
pub struct LcColumn<E, F>
where
    F: FieldExt,
    E: Send + Sync,
{
    col_idx: usize,
    col: Vec<F>,  // The actual column from M
    path: Vec<F>, // TODO!(ryancao)
    phantom_data: PhantomData<E>,
}

/// An evaluation and proof of its correctness and of the low-degreeness of the commitment.
#[derive(Debug, Clone)]
pub struct LcEvalProof<D, E, F>
where
    D: FieldHashFnDigest<F> + Send + Sync,
    F: FieldExt,
    E: LcEncoding<F> + Send + Sync,
{
    encoded_num_cols: usize,
    p_eval: Vec<F>,
    // --- No longer doing the well-formedness check ---
    // p_random_vec: Vec<Vec<F>>,
    columns: Vec<LcColumn<E, F>>,
    phantom_data: PhantomData<D>,
}

impl<D, E, F> LcEvalProof<D, E, F>
where
    D: FieldHashFnDigest<F> + Send + Sync,
    F: FieldExt,
    E: LcEncoding<F> + Send + Sync,
{
    /// Get the number of elements in an encoded vector
    pub fn get_encoded_num_cols(&self) -> usize {
        self.encoded_num_cols
    }

    /// Get the number of elements in an unencoded vector
    pub fn get_orig_num_cols(&self) -> usize {
        self.p_eval.len()
    }

    /// Verify an evaluation proof and return the resulting evaluation
    pub fn verify(
        &self,
        root: &F,
        outer_tensor: &[F],
        inner_tensor: &[F],
        enc: &E,
        tr: &mut PoseidonTranscript<F>,
    ) -> VerifierResult<F, ErrT<E, F>> {
        verify::<D, E, F>(root, outer_tensor, inner_tensor, self, enc, tr)
    }
}

/// Compute number of degree tests required for `lambda`-bit security
/// for a code with `len`-length codewords over `flog2`-bit field
/// -- This is used in Verify and Prove
pub fn n_degree_tests(lambda: usize, len: usize, flog2: usize) -> usize {
    // -- den = log2(|F|) - log2(|codeword|) = how many bits of security are left in the field?
    // -- |codeword| = encoded_num_cols
    let den = flog2 - log2(len);

    // -- The expression below simplifies to: (λ+den-1)/den = (λ-1)/den + 1
    // -- This implies that (λ-1)/den will always be >= 1
    (lambda + den - 1) / den
}

// parallelization limit when working on columns
const LOG_MIN_NCOLS: usize = 5;

/// Commit to a univariate polynomial whose coefficients are `coeffs` using encoding `enc`
/// --- Note that our hash function needs to implement `Digest` to be used here ---
/// --- In the test cases, `coeffs_in` is literally a Vec<F> ---
fn commit<D, E, F>(coeffs_in: &[F], enc: &E) -> ProverResult<LcCommit<D, E, F>, ErrT<E, F>>
where
    F: FieldExt,
    D: FieldHashFnDigest<F> + Send + Sync,
    E: LcEncoding<F> + Send + Sync,
{
    // --- Matrix size params ---
    // n_rows: Total number of matrix rows (i.e. height)
    // orig_num_cols: Total number of UNENCODED matrix cols (i.e. width)
    // encoded_num_cols: Total number of ENCODED matrix cols (i.e. orig_num_cols * \rho^{-1})
    let (n_rows, orig_num_cols, encoded_num_cols) = enc.get_dims(coeffs_in.len());

    // check that parameters are ok
    assert!(n_rows * orig_num_cols >= coeffs_in.len());
    assert!((n_rows - 1) * orig_num_cols < coeffs_in.len());
    assert!(enc.dims_ok(orig_num_cols, encoded_num_cols));

    // matrix (encoded as a vector)
    // XXX(zk) pad coeffs
    // --- `coeffs` should be the original coefficients ---
    let mut coeffs = vec![F::zero(); n_rows * orig_num_cols];
    // --- `comm` should be the matrix of FFT-encoded rows ---
    let mut comm = vec![F::zero(); n_rows * encoded_num_cols];

    // local copy of coeffs with padding
    coeffs
        .par_chunks_mut(orig_num_cols)
        .zip(coeffs_in.par_chunks(orig_num_cols))
        .for_each(|(c, c_in)| {
            c[..c_in.len()].copy_from_slice(c_in);
        });

    // now compute FFTs
    // --- Go through each row of M' (the encoded matrix), as well as each row of M (the unencoded matrix) ---
    // --- and make a copy, then perform the encoding (i.e. FFT) ---

    let fft_timer = start_timer!(|| format!("starting fft"));
    comm.par_chunks_mut(encoded_num_cols)
        .zip(coeffs.par_chunks(orig_num_cols))
        .try_for_each(|(r, c)| {
            r[..c.len()].copy_from_slice(c);
            enc.encode(r)
        })?;
    end_timer!(fft_timer);

    // compute Merkle tree
    let encoded_num_cols_np2 = encoded_num_cols
        .checked_next_power_of_two()
        .ok_or(ProverError::TooBig)?;

    let mut ret = LcCommit {
        comm,
        coeffs,
        n_rows,
        encoded_num_cols,
        orig_num_cols,
        // --- There are 2^{k + 1} - 1 total hash things ---
        // TODO!(ryancao): Why...?
        hashes: vec![F::default(); 2 * encoded_num_cols_np2 - 1],
        phantom_data: PhantomData,
        phantom_data_2: PhantomData,
    };

    // --- A sanitycheck of some sort, I assume? ---
    check_comm(&ret, enc)?;

    // --- Computes Merkle commitments for each column using the Digest ---
    // --- then hashes all the col commitments together using the Digest again ---
    let merkel_timer = start_timer!(|| format!("merkelize root"));
    merkleize(&mut ret);
    end_timer!(merkel_timer);

    Ok(ret)
}

// -- This seems to be checking the size of various commitment parameters
fn check_comm<D, E, F>(comm: &LcCommit<D, E, F>, enc: &E) -> ProverResult<(), ErrT<E, F>>
where
    D: FieldHashFnDigest<F> + Send + Sync,
    F: FieldExt,
    E: LcEncoding<F> + Send + Sync,
{
    // -- |commitment| = |rows| * |cols|, where cols are the ENCODED cols
    let comm_sz = comm.comm.len() != comm.n_rows * comm.encoded_num_cols;
    // -- |commitment_coeffs|  = |rows| * |orig_num_cols| where `orig_num_cols` are the UNENCODED cols
    // -- that is, this is the |coeffs| of the actual polynomial
    let coeff_sz = comm.coeffs.len() != comm.n_rows * comm.orig_num_cols;
    // -- hmmm...does the prover keep the hashes of all the merkle tree nodes,
    // -- so that it does not have to recompute all this during the opening phase?
    let hashlen = comm.hashes.len() != 2 * comm.encoded_num_cols.next_power_of_two() - 1;
    let dims = !enc.dims_ok(comm.orig_num_cols, comm.encoded_num_cols);

    if comm_sz || coeff_sz || hashlen || dims {
        Err(ProverError::Commit)
    } else {
        Ok(())
    }
}

fn merkleize<D, E, F>(comm: &mut LcCommit<D, E, F>)
where
    F: FieldExt,
    D: FieldHashFnDigest<F> + Send + Sync,
    E: LcEncoding<F> + Send + Sync,
{
    // --- This is SUPER ugly, but for the sake of efficiency... ---
    // TODO!(ryancao): Riperoni
    let merkle_hash = start_timer!(|| format!("first hash"));
    let master_default_poseidon_merkle_hasher = Poseidon::<F, 3, 2>::new(8, 57);
    end_timer!(merkle_hash);
    let column_hash = start_timer!(|| format!("second hash"));
    let master_default_poseidon_column_hasher = Poseidon::<F, 3, 2>::new(8, 57);
    end_timer!(column_hash);

    // --- Basically `hashes` is of length 2^h - 1, where h is the height of the Merkle tree ---
    // The idea is that the first 2^{h - 1} items are the leaf nodes (i.e. the column hashes)
    // and the remainder comes from the Merkle tree. Actually the order is EXACTLY as you'd expect,
    // with the layers of the tree being flattened and literally appended from bottom to top

    // step 1: hash each column of the commitment (we always reveal a full column)

    let hash_column_timer = start_timer!(|| format!("hashing the columns"));
    let hashes = &mut comm.hashes[..comm.encoded_num_cols];
    hash_columns::<D, E, F>(
        &comm.comm,
        hashes,
        comm.n_rows,
        comm.encoded_num_cols,
        0,
        &master_default_poseidon_column_hasher,
    );
    end_timer!(hash_column_timer);

    // step 2: compute rest of Merkle tree
    let len_plus_one = comm.hashes.len() + 1;
    assert!(len_plus_one.is_power_of_two());
    let (hin, hout) = comm.hashes.split_at_mut(len_plus_one / 2);

    let merkelize_tree = start_timer!(|| format!("merkelize tree"));
    merkle_tree::<D, F>(hin, hout, &master_default_poseidon_merkle_hasher);
    end_timer!(merkelize_tree);
}

fn hash_columns<D, E, F>(
    comm: &[F],              // The flattened version of M'
    hashes: &mut [F],        // This is the thing we are populating
    n_rows: usize,           // Height of M and M'
    encoded_num_cols: usize, // Width of M'
    offset: usize,           // Gets set to zero above
    master_default_poseidon_column_hasher: &Poseidon<F, 3, 2>,
) where
    F: FieldExt,
    D: FieldHashFnDigest<F> + Send + Sync,
    E: LcEncoding<F> + Send + Sync,
{
    if hashes.len() <= (1 << LOG_MIN_NCOLS) {
        // base case: run the computation

        // 1. prepare the digests for each column
        let mut digests = Vec::with_capacity(hashes.len());
        for _ in 0..hashes.len() {
            // let column_hash_poseidon_params = PoseidonParams::new(8, 63, 8, 9);
            // let dig = PoseidonSpongeHasher::new_with_params(column_hash_poseidon_params);
            // let halo2_default_sponge = dig.get_()
            // dig = PoseidonSpongeHasher::new_column_hasher(halo2_default_sponge);
            let dig =
                PoseidonSpongeHasher::new_column_hasher(master_default_poseidon_column_hasher);
            digests.push(dig);
        }

        // 2. for each row, update the digests for each column
        for row in 0..n_rows {
            for (col, digest) in digests.iter_mut().enumerate() {
                // --- Updates the digest with the value at `comm[row * encoded_num_cols + offset + col]` ---
                // TODO!(ryancao): We can simply replace this with a sponge absorb
                let com_val: F = comm[row * encoded_num_cols + offset + col];
                com_val.digest_update(digest);
            }
        }

        // 3. finalize each digest and write the results back
        for (col, digest) in digests.into_iter().enumerate() {
            hashes[col] = digest.finalize();
        }
    } else {
        // recursive case: split and execute in parallel
        let half_cols = hashes.len() / 2;
        let (lo, hi) = hashes.split_at_mut(half_cols);
        rayon::join(
            || {
                hash_columns::<D, E, F>(
                    comm,
                    lo,
                    n_rows,
                    encoded_num_cols,
                    offset,
                    master_default_poseidon_column_hasher,
                )
            },
            || {
                hash_columns::<D, E, F>(
                    comm,
                    hi,
                    n_rows,
                    encoded_num_cols,
                    offset + half_cols,
                    master_default_poseidon_column_hasher,
                )
            },
        );
    }
}

/// @param ins: The leaves to the Merkle tree
/// @param outs: The hashes for the internal nodes up to the root
fn merkle_tree<D, F>(
    ins: &[F],
    outs: &mut [F],
    master_default_poseidon_merkle_hasher: &Poseidon<F, 3, 2>,
) where
    F: FieldExt,
    D: FieldHashFnDigest<F> + Send + Sync,
{
    // --- The outs (i.e. rest of the tree) should be 2^{h - 1} - 1 while the ins should be 2^{h - 1} ---
    assert_eq!(ins.len(), outs.len() + 1);

    // --- Merkle-ize just the next layer ---
    let (outs, rems) = outs.split_at_mut((outs.len() + 1) / 2);
    merkle_layer::<D, F>(ins, outs, master_default_poseidon_merkle_hasher);

    if !rems.is_empty() {
        // --- Recursively merkleize until we have nothing remaining (i.e. a single element left) ---
        merkle_tree::<D, F>(outs, rems, master_default_poseidon_merkle_hasher)
    }
}

/// --- Computes a single Merkle tree layer by hashing adjacent pairs of "leaves" ---
fn merkle_layer<D, F>(
    ins: &[F],
    outs: &mut [F],
    master_default_poseidon_merkle_hasher: &Poseidon<F, 3, 2>,
) where
    F: FieldExt,
    D: FieldHashFnDigest<F> + Send + Sync,
{
    assert_eq!(ins.len(), 2 * outs.len());

    if ins.len() <= (1 << LOG_MIN_NCOLS) {
        // base case: just compute all of the hashes

        // let mut digest = D::new();
        // let hash_init = start_timer!(|| format!("initialize hash"));
        // let mut digest = D::new_merkle_hasher(master_default_poseidon_merkle_hasher);
        // end_timer!(hash_init);
        let ok = start_timer!(|| format!("bro idk {:?}", outs.len()));
        for idx in 0..outs.len() {
            let mut digest = D::new_merkle_hasher(master_default_poseidon_merkle_hasher);
            // --- I see. We update the digest with the things we want to "hash" ---
            // --- Then call `finalize()` or something like that to get the hash ---
            digest.update(&[ins[2 * idx]]);
            digest.update(&[ins[2 * idx + 1]]);
            outs[idx] = digest.finalize();
        }
        end_timer!(ok);
    } else {
        // recursive case: split and compute
        let (inl, inr) = ins.split_at(ins.len() / 2);
        let (outl, outr) = outs.split_at_mut(outs.len() / 2);
        rayon::join(
            || merkle_layer::<D, F>(inl, outl, master_default_poseidon_merkle_hasher),
            || merkle_layer::<D, F>(inr, outr, master_default_poseidon_merkle_hasher),
        );
    }
}

/// Open the commitment to one column
/// @param comm -- actual Ligero commitment
/// @param column -- the index of the column to open
/// @return TODO!(ryancao)
fn open_column<D, E, F>(
    comm: &LcCommit<D, E, F>,
    mut column: usize,
) -> ProverResult<LcColumn<E, F>, ErrT<E, F>>
where
    F: FieldExt,
    D: FieldHashFnDigest<F> + Send + Sync,
    E: LcEncoding<F> + Send + Sync,
{
    // make sure arguments are well formed
    if column >= comm.encoded_num_cols {
        return Err(ProverError::ColumnNumber);
    }

    // column of values
    let col = comm
        .comm
        .iter()
        // --- Start collecting at the `column`th coordinate ---
        .skip(column)
        // --- Skip num_cols (i.e. row length) number of elements to grab each column value ---
        .step_by(comm.encoded_num_cols)
        .cloned()
        .collect();

    // Merkle path
    let mut hashes = &comm.hashes[..];
    let path_len = log2(comm.encoded_num_cols);
    let mut path = Vec::with_capacity(path_len);
    for _ in 0..path_len {
        // --- Ahh I see this is the clever way of getting the "other" child ---
        // Either n - 1 or n + 1: nice work, Riad
        let other = (column & !1) | (!column & 1);
        assert_eq!(other ^ column, 1);
        // --- Mmmmmm okay so `hashes` contains all of the Merkle hashes. I see I see ---
        path.push(hashes[other]);
        let (_, hashes_new) = hashes.split_at((hashes.len() + 1) / 2);
        hashes = hashes_new;
        column >>= 1;
    }
    assert_eq!(column, 0);

    // --- Returns the actual column of values, plus the Merkle path ---
    // --- To verify this, we should hash the column of values and check that ---
    // --- the final hash of such, when used as the target leaf node within ---
    // --- a Merkle proof, given the Merkle path `path` below, yields the ---
    // --- original commitment ---
    Ok(LcColumn {
        col,
        path,
        phantom_data: PhantomData,
        col_idx: column,
    })
}

const fn log2(v: usize) -> usize {
    (63 - (v.next_power_of_two() as u64).leading_zeros()) as usize
}

/// Verify the evaluation of a committed polynomial and return the result
fn verify<D, E, F>(
    root: &F,
    outer_tensor: &[F], // b^T
    inner_tensor: &[F], // a
    proof: &LcEvalProof<D, E, F>,
    // This is not real. Well, really it just gives you the setup for being able to compute an FFT
    enc: &E,
    tr: &mut impl RemainderTranscript<F>,
) -> VerifierResult<F, ErrT<E, F>>
where
    F: FieldExt,
    D: FieldHashFnDigest<F> + Send + Sync,
    E: LcEncoding<F> + Send + Sync,
{
    // --- Grab ONE global copy of Merkle + column hashing Poseidon ---
    // --- This is SUPER ugly, but for the sake of efficiency... ---
    // TODO!(ryancao): Riperoni
    // TODO!(ryancao): Put in the new rate for the column hash... maybe?
    let master_default_poseidon_merkle_hasher = Poseidon::<F, 3, 2>::new(8, 57);
    let master_default_poseidon_column_hasher = Poseidon::<F, 3, 2>::new(8, 57);

    // make sure arguments are well formed
    let n_col_opens = enc.get_n_col_opens();
    if n_col_opens != proof.columns.len() || n_col_opens == 0 {
        return Err(VerifierError::NumColOpens);
    }
    let n_rows = proof.columns[0].col.len();
    let encoded_num_cols = proof.get_encoded_num_cols();
    let orig_num_cols = proof.get_orig_num_cols();
    if inner_tensor.len() != orig_num_cols {
        return Err(VerifierError::InnerTensor);
    }
    if outer_tensor.len() != n_rows {
        return Err(VerifierError::OuterTensor);
    }
    if !enc.dims_ok(orig_num_cols, encoded_num_cols) {
        return Err(VerifierError::EncodingDims);
    }

    // step 1: random tensor for degree test and random columns to test
    // step 1a: extract random tensor from transcript
    // we run multiple instances of this to boost soundness

    // --- This is for the verifier-generated versions of `r` ---
    let _rand_tensor_vec: Vec<Vec<F>> = Vec::new();

    // --- This is for the verifier evaluations of r^T M' ---
    // --- but computed where r^T comes from the prover (???) ---
    let _p_random_fft: Vec<Vec<F>> = Vec::new();

    // --- No longer doing the well-formedness check ---
    // let n_degree_tests = enc.get_n_degree_tests();

    // -- This for loop is for soundness amplification by repetition
    // for i in 0..n_degree_tests {

    //     // --- TODO!(ryancao): Propagate the error up! ---
    //     let rand_tensor = tr.get_challenges("random_vec_well_formedness_check", n_rows).unwrap();
    //     rand_tensor_vec.push(rand_tensor);

    //     // step 1b: eval encoding of p_random
    //     {
    //         let mut tmp = Vec::with_capacity(encoded_num_cols);
    //         tmp.extend_from_slice(&proof.p_random_vec[i][..]); // Copies over the random vec from the prover
    //         tmp.resize(encoded_num_cols, F::from(0));
    //         // Ohhhhh I see. This gives the RLC \rho^{-1} * \sqrt{N} -coordinate thingy
    //         // of r^T M' (where `r` was prover-generated). Note that we got this by basically
    //         // calling `enc(r^T M)`
    //         enc.encode(&mut tmp)?;
    //         p_random_fft.push(tmp);
    //     };

    //     // step 1c: push p_random...
    //     proof.p_random_vec[i]
    //         .iter()
    //         .for_each(|coeff| coeff.transcript_update(tr, "LABEL_PR"));
    // }

    // ...and p_eval into the transcript
    proof
        .p_eval
        .iter()
        .for_each(|coeff| coeff.transcript_update(tr, "LABEL_PE"));

    // step 1d: extract columns to open
    // --- The verifier does this independently as well ---
    let cols_to_open: Vec<usize> = {
        tr.get_challenges("column_indices", n_col_opens)
            .unwrap()
            .into_iter()
            .map(|challenge| compute_col_idx_from_transcript_challenge(challenge, encoded_num_cols))
            .collect()
    };

    // step 2: p_eval fft for column checks
    // --- Takes the prover claimed value for b^T M and computes enc(b^T M) = b^T M' ---
    let p_eval_fft = {
        let mut tmp = Vec::with_capacity(encoded_num_cols);
        tmp.extend_from_slice(&proof.p_eval[..]);
        tmp.resize(encoded_num_cols, F::from(0));
        enc.encode(&mut tmp)?;
        tmp
    };

    // step 3: check p_random, p_eval, and col paths
    cols_to_open
        .par_iter()
        .zip(&proof.columns[..])
        .try_for_each(|(&col_num, column)| {
            // --- No longer doing the well-formedness check ---
            // --- Okay so we zip the indices with the actual columns ---
            // let rand = {
            //     let mut rand = true;
            //     // --- This is just 1 for us; we don't need boosting ---
            //     for i in 0..n_degree_tests {
            //         rand &=
            //             // --- This literally does r^T M'_j and checks against (r^T M')[j]
            //             // (the latter is computed by the verifier) ---
            //             verify_column_value::<D, E, F>(column, &rand_tensor_vec[i], &p_random_fft[i][col_num]);
            //     }
            //     rand
            // };

            // --- Does the RLC evaluation check for b^T as well ---
            let eval = verify_column_value::<D, E, F>(column, outer_tensor, &p_eval_fft[col_num]);

            // --- Merkle path verification: Does hashing for each column, then Merkle tree hashes ---
            // TODO!(ryancao): Make this use Poseidon
            let path = verify_column_path::<D, E, F>(
                column,
                col_num,
                root,
                &master_default_poseidon_merkle_hasher,
                &master_default_poseidon_column_hasher,
            );

            // --- "Very elegant, Riad" - Ryan ---
            match (eval, path) {
                // --- No longer doing the well-formedness check ---
                // (false, _, _) => Err(VerifierError::ColumnDegree),
                (false, _) => Err(VerifierError::ColumnEval),
                (_, false) => Err(VerifierError::ColumnPath),
                _ => Ok(()),
            }
        })?;

    // step 4: evaluate and return
    // --- Computes dot product between inner_tensor (i.e. a) and proof.p_eval (i.e. b^T M) ---
    Ok(inner_tensor
        .par_iter()
        .zip(&proof.p_eval[..])
        .fold(|| F::zero(), |a, (t, e)| a + *t * e)
        .reduce(|| F::zero(), |a, v| a + v))
}

// Check a column opening
fn verify_column_path<D, E, F>(
    column: &LcColumn<E, F>,
    col_num: usize,
    root: &F,
    master_default_poseidon_merkle_hasher: &Poseidon<F, 3, 2>,
    master_default_poseidon_column_hasher: &Poseidon<F, 3, 2>,
) -> bool
where
    F: FieldExt,
    D: FieldHashFnDigest<F> + Send + Sync,
    E: LcEncoding<F> + Send + Sync,
{
    // --- New Poseidon params + Poseidon hasher ---
    // let poseidon_column_hash_params = PoseidonParams::new(8, 63, 8, 9);
    // let mut digest = PoseidonSpongeHasher::new_with_params(poseidon_column_hash_params);
    let mut digest = PoseidonSpongeHasher::new_column_hasher(master_default_poseidon_column_hasher);

    // TODO!(ryancao): What's this about?
    // digest.update(&[F::default()]);

    // --- Just eat up the column elements themselves ---
    for e in &column.col[..] {
        e.digest_update(&mut digest);
    }

    // check Merkle path
    // let mut hash = digest.finalize_reset();
    let mut hash = digest.finalize();
    digest = PoseidonSpongeHasher::new_merkle_hasher(master_default_poseidon_merkle_hasher);
    let mut col = col_num;
    // TODO!(ryancao): Understand this...?
    for p in &column.path[..] {
        if col % 2 == 0 {
            digest.update(&[hash]);
            digest.update(&[*p]);
        } else {
            digest.update(&[*p]);
            digest.update(&[hash]);
        }
        // hash = digest.finalize_reset();
        hash = digest.finalize();
        digest = PoseidonSpongeHasher::new_merkle_hasher(master_default_poseidon_merkle_hasher);
        col >>= 1;
    }

    &hash == root
}

// check column value
fn verify_column_value<D, E, F>(
    column: &LcColumn<E, F>, // The actual Ligero matrix col M_j
    tensor: &[F],            // The random r^T we are evaluating at
    poly_eval: &F,           // The RLC'd, evaluated version r^T M'[j]
) -> bool
where
    F: FieldExt,
    D: FieldHashFnDigest<F> + Send + Sync,
    E: LcEncoding<F> + Send + Sync,
{
    let tensor_eval = tensor
        .iter()
        .zip(&column.col[..])
        .fold(F::zero(), |a, (t, e)| a + *t * e);

    poly_eval == &tensor_eval
}

/// Computes the column index from a challenge over F by taking the
/// least significant bits from the bit-wise representation of `challenge`
/// and directly using those as the `col_idx`
fn compute_col_idx_from_transcript_challenge<F: FieldExt>(
    challenge: F,
    encoded_num_cols: usize,
) -> usize {
    // --- Get the number of necessary bits ---
    let log_col_len = log2(encoded_num_cols);
    debug_assert!(log_col_len < 32);

    let challenge_le_bytes = challenge.to_bytes_le();
    let col_idx =
        get_least_significant_bits_to_usize_little_endian(challenge_le_bytes, log_col_len);

    // --- Sanitycheck ---
    assert!(col_idx < encoded_num_cols);
    col_idx
}

/// Evaluate the committed polynomial using the supplied "outer" tensor
/// and generate a proof of (1) low-degreeness and (2) correct evaluation.
fn prove<D, E, F, T: RemainderTranscript<F>>(
    comm: &LcCommit<D, E, F>,
    outer_tensor: &[F],
    enc: &E,
    tr: &mut T,
) -> ProverResult<LcEvalProof<D, E, F>, ErrT<E, F>>
where
    F: FieldExt,
    D: FieldHashFnDigest<F> + Send + Sync,
    E: LcEncoding<F> + Send + Sync,
{
    // make sure arguments are well formed
    check_comm(comm, enc)?;
    if outer_tensor.len() != comm.n_rows {
        return Err(ProverError::OuterTensor);
    }

    // --- No longer doing the well-formedness check (START) ---
    // let mut p_random_vec: Vec<Vec<F>> = Vec::new();
    // first, evaluate the polynomial on a random tensor (low-degree test)
    // we repeat this to boost soundness
    // let n_degree_tests = enc.get_n_degree_tests()
    // for _i in 0..n_degree_tests {
    //     let p_random = {
    //         // --- Instead, we sample random challenges from the transcript ---
    //         // TODO!(ryancao): Make this return an actual error!
    //         let rand_tensor = tr.get_challenges("rand_Ligero_well_formedness_vec", comm.n_rows).unwrap();

    //         let mut tmp = vec![F::zero(); comm.orig_num_cols];

    //         // --- This takes the dot product against `comm.coeffs` and stores the result in `tmp` ---
    //         // --- Basically this is the prover's claimed value for r^T M ---
    //         collapse_columns::<E, F>(
    //             &comm.coeffs,
    //             &rand_tensor,
    //             &mut tmp,
    //             comm.n_rows,
    //             comm.orig_num_cols,
    //             0,
    //         );
    //         tmp
    //     };
    //     // add p_random to the transcript
    //     p_random
    //         .iter()
    //         .for_each(|coeff| coeff.transcript_update(tr, "LABEL_PR"));

    //     p_random_vec.push(p_random);
    // }
    // --- No longer doing the well-formedness check (END) ---

    // next, evaluate the polynomial using the supplied tensor
    let p_eval = {
        let mut tmp = vec![F::zero(); comm.orig_num_cols];
        // --- Take the vector-matrix product b^T M ---
        collapse_columns::<E, F>(
            &comm.coeffs,
            outer_tensor,
            &mut tmp,
            comm.n_rows,
            comm.orig_num_cols,
            0,
        );
        tmp
    };
    // add p_eval to the transcript
    p_eval
        .iter()
        .for_each(|coeff| coeff.transcript_update(tr, "LABEL_PE"));

    // now extract the column numbers to open
    let n_col_opens = enc.get_n_col_opens();
    let columns: Vec<LcColumn<E, F>> = {
        // --- I think we need to do a mod operation here... ---
        let cols_to_open: Vec<usize> = tr
            .get_challenges("column_indices", n_col_opens)
            .unwrap()
            .into_iter()
            .enumerate()
            .map(|(_i, challenge)| {
                compute_col_idx_from_transcript_challenge(challenge, comm.encoded_num_cols)
            })
            .collect();

        // --- Let's check out this `open_column` function ---
        // Yeah so `open_column()` basically gives you the column itself, along with a Merkle path
        // Pretty straightforward. "I LIKE it! - Creed" - Ryan
        // TODO!(ryancao): Put the parallelism back
        cols_to_open
            .par_iter()
            .map(|&col| open_column(comm, col))
            .collect::<ProverResult<Vec<LcColumn<E, F>>, ErrT<E, F>>>()?
    };

    Ok(LcEvalProof {
        encoded_num_cols: comm.encoded_num_cols, // Number of columns
        p_eval,                                  // Actual b^T M value
        // --- No longer doing the well-formedness check ---
        // p_random_vec, // Random vectors to check well-formedness
        columns, // Columns plus necessary opening proof content
        phantom_data: PhantomData,
    })
}

/// This takes the product b^T M
///
/// ## Arguments
/// * `coeffs` - M, but flattened
/// * `tensor` - b^T
/// * `poly` - The component of M we are currently looking at (for parallelism)
/// * `n_rows` - Height of M
/// * `orig_num_cols` - Width of M
/// * `offset` - Something we don't use...?
fn collapse_columns<E, F>(
    coeffs: &[F],
    tensor: &[F],
    poly: &mut [F],
    n_rows: usize,
    orig_num_cols: usize,
    offset: usize,
) where
    F: FieldExt,
    E: LcEncoding<F> + Send + Sync,
{
    if poly.len() <= (1 << LOG_MIN_NCOLS) {
        // base case: run the computation
        // row-by-row, compute elements of dot product
        for (row, tensor_val) in tensor.iter().enumerate() {
            for (col, val) in poly.iter_mut().enumerate() {
                let entry = row * orig_num_cols + offset + col;
                *val += coeffs[entry] * tensor_val;
            }
        }
    } else {
        // recursive case: split and execute in parallel
        let half_cols = poly.len() / 2;
        let (lo, hi) = poly.split_at_mut(half_cols);
        rayon::join(
            || collapse_columns::<E, F>(coeffs, tensor, lo, n_rows, orig_num_cols, offset),
            || {
                collapse_columns::<E, F>(
                    coeffs,
                    tensor,
                    hi,
                    n_rows,
                    orig_num_cols,
                    offset + half_cols,
                )
            },
        );
    }
}

// TESTING ONLY //

#[cfg(test)]
fn merkleize_ser<D, E, F>(
    comm: &mut LcCommit<D, E, F>,
    master_default_poseidon_merkle_hasher: &Poseidon<F, 3, 2>,
    master_default_poseidon_column_hasher: &Poseidon<F, 3, 2>,
) where
    F: FieldExt,
    D: FieldHashFnDigest<F> + Send + Sync,
    E: LcEncoding<F> + Send + Sync,
{
    let hashes = &mut comm.hashes;

    // hash each column
    for (col, hash) in hashes.iter_mut().enumerate().take(comm.encoded_num_cols) {
        // let poseidon_column_hash_params = PoseidonParams::new(8, 63, 8, 9);
        // let mut digest = PoseidonSpongeHasher::new_with_params(poseidon_column_hash_params);
        let mut digest =
            PoseidonSpongeHasher::new_column_hasher(master_default_poseidon_column_hasher);
        // digest.update(&[F::default()]);
        for row in 0..comm.n_rows {
            comm.comm[row * comm.encoded_num_cols + col].digest_update(&mut digest);
        }
        *hash = digest.finalize();
    }

    // compute rest of Merkle tree
    let (mut ins, mut outs) = hashes.split_at_mut(comm.encoded_num_cols);
    while !outs.is_empty() {
        for idx in 0..ins.len() / 2 {
            // let mut digest = D::new();
            let mut digest = D::new_merkle_hasher(master_default_poseidon_merkle_hasher);
            digest.update(&[ins[2 * idx]]);
            digest.update(&[ins[2 * idx + 1]]);
            outs[idx] = digest.finalize();
        }
        let (new_ins, new_outs) = outs.split_at_mut((outs.len() + 1) / 2);
        ins = new_ins;
        outs = new_outs;
    }
}

#[cfg(test)]
// Check a column opening
fn verify_column<D, E, F>(
    column: &LcColumn<E, F>,
    col_num: usize,
    root: &F,
    tensor: &[F],
    poly_eval: &F,
    master_default_poseidon_column_hasher: &Poseidon<F, 3, 2>,
    master_default_poseidon_merkle_hasher: &Poseidon<F, 3, 2>,
) -> bool
where
    F: FieldExt,
    D: FieldHashFnDigest<F> + Send + Sync,
    E: LcEncoding<F> + Send + Sync,
{
    verify_column_path::<D, E, F>(
        column,
        col_num,
        root,
        master_default_poseidon_merkle_hasher,
        master_default_poseidon_column_hasher,
    ) && verify_column_value::<D, E, F>(column, tensor, poly_eval)
}

// Evaluate the committed polynomial using the "outer" tensor
// --- I'm not sure why Riad is calling it a "tensor" because it's ---
// --- definitely just a vector but it's the b^T M' component ---
#[cfg(test)]
fn eval_outer<D, E, F>(comm: &LcCommit<D, E, F>, tensor: &[F]) -> ProverResult<Vec<F>, ErrT<E, F>>
where
    F: FieldExt,
    D: FieldHashFnDigest<F> + Send + Sync,
    E: LcEncoding<F> + Send + Sync,
{
    if tensor.len() != comm.n_rows {
        return Err(ProverError::OuterTensor);
    }

    // allocate result and compute
    let mut poly = vec![F::zero(); comm.orig_num_cols];
    collapse_columns::<E, F>(
        &comm.coeffs,
        tensor,
        &mut poly,
        comm.n_rows,
        comm.orig_num_cols,
        0,
    );

    Ok(poly)
}

#[cfg(test)]
fn eval_outer_ser<D, E, F>(
    comm: &LcCommit<D, E, F>,
    tensor: &[F],
) -> ProverResult<Vec<F>, ErrT<E, F>>
where
    F: FieldExt,
    D: FieldHashFnDigest<F> + Send + Sync,
    E: LcEncoding<F> + Send + Sync,
{
    if tensor.len() != comm.n_rows {
        return Err(ProverError::OuterTensor);
    }

    let mut poly = vec![F::zero(); comm.orig_num_cols];
    for (row, tensor_val) in tensor.iter().enumerate() {
        for (col, val) in poly.iter_mut().enumerate() {
            let entry = row * comm.orig_num_cols + col;
            *val += comm.coeffs[entry] * tensor_val;
        }
    }

    Ok(poly)
}

/// Computes b^T M' (where `tensor` is b^T and `comm.comm` is M')
#[cfg(test)]
fn eval_outer_fft<D, E, F>(
    comm: &LcCommit<D, E, F>,
    tensor: &[F],
) -> ProverResult<Vec<F>, ErrT<E, F>>
where
    F: FieldExt,
    D: FieldHashFnDigest<F> + Send + Sync,
    E: LcEncoding<F> + Send + Sync,
{
    if tensor.len() != comm.n_rows {
        return Err(ProverError::OuterTensor);
    }

    // --- M' row length worth of 0s ---
    let mut poly_fft = vec![F::zero(); comm.encoded_num_cols];

    // --- For each coefficient within `tensor` corresponding to a row in M'... ---
    for (coeffs, tensorval) in comm.comm.chunks(comm.encoded_num_cols).zip(tensor.iter()) {
        for (coeff, polyval) in coeffs.iter().zip(poly_fft.iter_mut()) {
            *polyval += *coeff * tensorval;
        }
    }

    Ok(poly_fft)
}
