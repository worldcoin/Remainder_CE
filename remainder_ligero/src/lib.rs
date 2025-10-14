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
use ark_std::{end_timer, start_timer};
use itertools::Itertools;
use poseidon_ligero::poseidon_digest::FieldHashFnDigest;
use poseidon_ligero::PoseidonSpongeHasher;
use rayon::prelude::*;
use remainder_shared_types::{
    transcript::{ProverTranscript, TranscriptSponge, VerifierTranscript},
    Field, Poseidon,
};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use thiserror::Error;

mod macros;

/// For converting between this codebase's types and the types the
/// Halo2-GKR verifier would like to have.
pub mod adapter;
/// Public functions for univariate and multilinear Ligero commitment (with Poseidon)
pub mod ligero_commit;
/// For multilinear commitment stuff
pub mod ligero_ml_helper;
/// For actual Ligero proof structs
pub mod ligero_structs;
/// For Poseidon hashing (implementation with respect to Digest and Transcript)
pub mod poseidon_ligero;
/// Tests for Ligero PCS functionality
#[cfg(test)]
pub mod tests;
/// Helper functions
pub mod utils;

/// Trait wrapper over `Field` which gives a field element the ability to be
/// absorbed into a [FieldHashFnDigest], as well as a [TranscriptSponge].
pub trait PoseidonFieldHash: Field {
    /// Update the digest `d` with the `self` (since `self` should already be a field element)
    fn digest_update<D: FieldHashFnDigest<Self>>(&self, d: &mut D) {
        d.update(&[*self])
    }

    /// Update the [TranscriptSponge] with label `l` and element `self`
    fn transcript_update(&self, t: &mut impl TranscriptSponge<Self>, _l: &'static str) {
        t.absorb(*self);
    }
}

impl<F: Field> PoseidonFieldHash for F {
    fn digest_update<D: FieldHashFnDigest<F>>(&self, d: &mut D) {
        d.update(&[*self])
    }

    fn transcript_update(&self, t: &mut impl TranscriptSponge<F>, _l: &'static str) {
        t.absorb(*self);
    }
}

/// Trait for a linear encoding used by the Ligero PCS. In particular, the
/// encoding provides the metadata around e.g. the dimensions of the unencoded
/// and encoded matrices, as well as other information e.g. the number of
/// degree tests required (in the multilinear GKR + Ligero PCS case, none!)
///
/// Additionally, provides metadata around
pub trait LcEncoding<F: Field>: Clone + std::fmt::Debug + Sync {
    /// Domain separation label - degree test (see def_labels!())
    const LABEL_DT: &'static [u8];
    /// Domain separation label - random lin combs (see def_labels!())
    const LABEL_PR: &'static [u8];
    /// Domain separation label - eval comb (see def_labels!())
    const LABEL_PE: &'static [u8];
    /// Domain separation label - column openings (see def_labels!())
    const LABEL_CO: &'static [u8];

    /// Error type for encoding
    type Err: std::fmt::Debug + Send;

    /// Encoding function
    fn encode(&self, inp: &mut [F]) -> Result<(), Self::Err>;

    /// Get dimensions for this encoding instance on an input vector of length `len`
    fn get_dims_for_input_len(&self, len: usize) -> (usize, usize, usize);

    /// Gets encoding dimensions given the state stored within the encoding.
    fn get_dims(&self) -> (usize, usize, usize);

    /// Get the number of column openings required for this encoding
    fn get_n_col_opens(&self) -> usize;

    /// Ensures that dimensions passed in match expected dimensions for the encoding
    fn dims_ok(&self, orig_num_cols: usize, encoded_num_cols: usize) -> bool;

    /// Get the number of degree tests required for this encoding
    fn get_n_degree_tests(&self) -> usize;
}

// local accessors for enclosed types
type ErrT<E, F> = <E as LcEncoding<F>>::Err;

/// Err variant for prover operations
#[derive(Debug, Error)]
pub enum ProverError<ErrT>
where
    ErrT: std::fmt::Debug + 'static,
{
    /// size too big
    #[error("encoded_num_cols is too large for this encoding")]
    TooBig,
    /// error encoding a vector
    #[error("encoding error: {:?}", _0)]
    Encode(#[from] ErrT),
    /// inconsistent LcCommit fields
    #[error("inconsistent commitment fields")]
    Commit,
    /// bad column number
    #[error("bad column number")]
    ColumnNumber,
    /// bad outer tensor
    #[error("outer tensor: wrong size")]
    OuterTensor,
}

/// result of a prover operation
pub type ProverResult<T, ErrT> = Result<T, ProverError<ErrT>>;

/// Err variant for verifier operations
#[derive(Debug, Error)]
pub enum VerifierError<ErrT>
where
    ErrT: std::fmt::Debug + 'static,
{
    /// wrong number of column openings in proof
    #[error("wrong number of column openings in proof")]
    NumColOpens,
    /// failed to verify column merkle path
    #[error("column verification: merkle path failed")]
    ColumnPath,
    /// failed to verify column dot product for poly eval
    #[error("column verification: eval dot product failed")]
    ColumnEval,
    /// failed to verify column dot product for degree test
    #[error("column verification: degree test dot product failed")]
    ColumnDegree,
    /// bad outer tensor
    #[error("outer tensor: wrong size")]
    OuterTensor,
    /// bad inner tensor
    #[error("inner tensor: wrong size")]
    InnerTensor,
    /// encoding dimensions do not match proof
    #[error("encoding dimension mismatch")]
    EncodingDims,
    /// error encoding a vector
    #[error("encoding error: {:?}", _0)]
    Encode(#[from] ErrT),
}

/// result of a verifier operation
pub type VerifierResult<T, ErrT> = Result<T, VerifierError<ErrT>>;

/// Ligero commitment to be used by the prover. This should *not* be sent
/// to the verifier (only `self.comm`)!
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LcCommit<D, E, F> {
    // Flattened version of M' (encoded) matrix
    comm: Vec<F>,
    // Flattened version of M (non-encoded) matrix
    coeffs: Vec<F>,
    /// Height of M (and M')
    n_rows: usize,
    /// Width of M'
    encoded_num_cols: usize,
    /// Width of M
    orig_num_cols: usize,
    /// All values within the Merkle tree (where leaves are column-wise linear hashes)
    hashes: Vec<F>,
    phantom_data: PhantomData<D>,
    phantom_data_2: PhantomData<E>,
}

impl<D, E, F> LcCommit<D, E, F>
where
    D: FieldHashFnDigest<F> + Send + Sync,
    F: Field,
    E: LcEncoding<F> + Send + Sync,
{
    /// Returns the Merkle root of this polynomial commitment (which is the commitment itself)
    pub fn get_root(&self) -> LcRoot<E, F> {
        LcRoot {
            root: (self.hashes.last().cloned().unwrap() as F),
            _p: Default::default(),
        }
    }

    /// Returns the number of coefficients encoded in each matrix row
    pub fn get_orig_num_cols(&self) -> usize {
        self.orig_num_cols
    }

    /// Returns the number of columns in the encoded matrix
    pub fn get_encoded_num_cols(&self) -> usize {
        self.encoded_num_cols
    }

    /// Returns the number of rows in the encoded matrix
    pub fn get_n_rows(&self) -> usize {
        self.n_rows
    }

    /// Generates a commitment to a polynomial represented by `coeffs`
    ///
    /// ## Arguments
    /// * `coeffs` - coefficients of the polynomial to be committed to
    /// * `enc` - encoding to be used for transforming M --> M'
    pub fn commit(coeffs: &[F], enc: &E) -> ProverResult<Self, ErrT<E, F>> {
        commit::<D, E, F>(coeffs, enc)
    }

    /// Generates an evaluation proof for `self` as a commitment.
    ///
    /// ## Arguments
    /// * `outer_tensor` - the "b^T" vector over the split expanded challenge point
    /// * `enc` - encoding used for transforming M --> M'
    ///
    /// ## Returns
    /// * `proof` - Ligero evaluation proof for committed polynomial at the
    ///   challenge point represented by `outer_tensor`
    pub fn prove(
        &self,
        outer_tensor: &[F],
        enc: &E,
        tr: &mut impl ProverTranscript<F>,
    ) -> ProverResult<(), ErrT<E, F>> {
        prove(self, outer_tensor, enc, tr)
    }
}

/// A Merkle root corresponding to a committed polynomial.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "F: Field")]
pub struct LcRoot<E, F>
where
    F: Field,
    E: LcEncoding<F> + Send + Sync,
{
    /// The root of the Merkle tree is a single field element
    pub root: F,
    _p: PhantomData<E>,
}

impl<E, F> LcRoot<E, F>
where
    F: Field,
    E: LcEncoding<F> + Send + Sync,
{
    /// Convert this value into a raw F
    pub fn into_raw(self) -> F {
        self.root
    }

    /// Generate a new [LcRoot] with the give root.
    pub fn new(root: F) -> Self {
        Self {
            root,
            _p: Default::default(),
        }
    }
}

impl<E, F> AsRef<F> for LcRoot<E, F>
where
    F: Field,
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
    F: Field,
    E: Send + Sync,
{
    /// The column index within M'
    col_idx: usize,
    /// The actual column values
    col: Vec<F>,
    /// The Merkle path, i.e. the siblings within the Merkle tree
    path: Vec<F>,
    phantom_data: PhantomData<E>,
}

/// An evaluation and proof of its correctness and of the low-degreeness of the commitment.
#[derive(Debug, Clone)]
pub struct LcEvalProof<D, E, F>
where
    D: FieldHashFnDigest<F> + Send + Sync,
    F: Field,
    E: LcEncoding<F> + Send + Sync,
{
    /// Width of M'
    encoded_num_cols: usize,
    /// Challenge point to be evaluated at
    p_eval: Vec<F>,
    /// Columns randomly sampled via Fiat-Shamir to be checked against the
    /// commitment + claimed value
    columns: Vec<LcColumn<E, F>>,
    phantom_data: PhantomData<D>,
}

impl<D, E, F> LcEvalProof<D, E, F>
where
    D: FieldHashFnDigest<F> + Send + Sync,
    F: Field,
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
}

/// Compute number of degree tests required for `lambda`-bit security
/// for a code with `len`-length codewords over `flog2`-bit field.
/// This is used in Verify and Prove.
pub fn n_degree_tests(lambda: usize, len: usize, flog2: usize) -> usize {
    // -- den = log2(|F|) - log2(|codeword|) = how many bits of security are left in the field?
    // -- |codeword| = encoded_num_cols
    let den = flog2 - log2(len);

    // -- The expression below simplifies to: (λ+den-1)/den = (λ-1)/den + 1
    // -- This implies that (λ-1)/den will always be >= 1
    lambda.div_ceil(den)
}

// parallelization limit when working on columns
const LOG_MIN_NCOLS: usize = 5;

/// Commit to a polynomial whose coefficients are `coeffs_in` using encoding `enc`.
///
/// ## Arguments
/// * `coeffs_in` - coefficients of the polynomial to be committed to.
/// * `enc` - encoding to perform over rows of M --> M'
///
/// ## Returns
/// * `commitment` - Ligero commitment to be used by the prover
fn commit<D, E, F>(coeffs_in: &[F], enc: &E) -> ProverResult<LcCommit<D, E, F>, ErrT<E, F>>
where
    F: Field,
    D: FieldHashFnDigest<F> + Send + Sync,
    E: LcEncoding<F> + Send + Sync,
{
    // Matrix size params
    let (n_rows, orig_num_cols, encoded_num_cols) = enc.get_dims_for_input_len(coeffs_in.len());

    // check that parameters are ok
    assert!(n_rows * orig_num_cols >= coeffs_in.len());
    assert!((n_rows - 1) * orig_num_cols < coeffs_in.len());
    assert!(enc.dims_ok(orig_num_cols, encoded_num_cols));

    // `coeffs` should be the original coefficients
    let mut coeffs = vec![F::ZERO; n_rows * orig_num_cols];
    // `comm` should be the matrix of FFT-encoded rows
    let mut comm = vec![F::ZERO; n_rows * encoded_num_cols];

    // Copy of `coeffs` with padding
    coeffs
        .par_chunks_mut(orig_num_cols)
        .zip(coeffs_in.par_chunks(orig_num_cols))
        .for_each(|(c, c_in)| {
            c[..c_in.len()].copy_from_slice(c_in);
        });

    // Go through each row of M' (the encoded matrix), as well as each row of M (the unencoded matrix)
    // and make a copy, then perform the encoding (i.e. FFT)
    let fft_timer = start_timer!(|| "starting fft".to_string());
    comm.par_chunks_mut(encoded_num_cols)
        .zip(coeffs.par_chunks(orig_num_cols))
        .try_for_each(|(r, c)| {
            r[..c.len()].copy_from_slice(c);
            enc.encode(r)
        })?;
    end_timer!(fft_timer);

    // Compute Merkle tree
    let encoded_num_cols_np2 = encoded_num_cols
        .checked_next_power_of_two()
        .ok_or(ProverError::TooBig)?;

    let mut ret = LcCommit {
        comm,
        coeffs,
        n_rows,
        encoded_num_cols,
        orig_num_cols,
        hashes: vec![F::default(); 2 * encoded_num_cols_np2 - 1],
        phantom_data: PhantomData,
        phantom_data_2: PhantomData,
    };

    // A sanitycheck of some sort, I assume?
    check_comm(&ret, enc)?;

    // Computes Merkle commitments for each column using the Digest
    // then hashes all the col commitments together using the Digest again
    let merkel_timer = start_timer!(|| "merkelize root".to_string());
    merkleize(&mut ret);
    end_timer!(merkel_timer);

    Ok(ret)
}

/// Sanitycheck for commitment and encoding dimensionality.
///
/// ## Arguments
/// * `comm` - Ligero commitment struct
/// * `enc` - Encoding used to compute M --> M'
fn check_comm<D, E, F>(comm: &LcCommit<D, E, F>, enc: &E) -> ProverResult<(), ErrT<E, F>>
where
    D: FieldHashFnDigest<F> + Send + Sync,
    F: Field,
    E: LcEncoding<F> + Send + Sync,
{
    // M' total flattened length must be M' rows * M' cols
    let comm_sz = comm.comm.len() != comm.n_rows * comm.encoded_num_cols;
    // M total flattened length must be M rows * M cols
    let coeff_sz = comm.coeffs.len() != comm.n_rows * comm.orig_num_cols;
    // Merkle tree total length must be 2 * M' cols - 1 (since there are M' cols leaves)
    let hashlen = comm.hashes.len() != 2 * comm.encoded_num_cols.next_power_of_two() - 1;
    // Dimension check of matrix against encoding
    let dims = !enc.dims_ok(comm.orig_num_cols, comm.encoded_num_cols);

    if comm_sz || coeff_sz || hashlen || dims {
        Err(ProverError::Commit)
    } else {
        Ok(())
    }
}

/// Modifies the `comm.hashes` field within `comm` to contain all of the Merkle
/// tree nodes, starting from the leaves. In other words, if the Merkle tree
/// looks like the following:
/// ```text
///       [1]
///   [2]     [3]
/// [4] [5] [6] [7]
/// ```
/// The resulting flattened hashes within `comm.hashes` will be
/// `(4, 5, 6, 7, 2, 3, 1)`.
///
/// ## Arguments
/// * `comm` - Ligero commitment struct whose `hashes` field is to be populated.
fn merkleize<D, E, F>(comm: &mut LcCommit<D, E, F>)
where
    F: Field,
    D: FieldHashFnDigest<F> + Send + Sync,
    E: LcEncoding<F> + Send + Sync,
{
    let master_default_poseidon_merkle_hasher = Poseidon::<F, 3, 2>::new(8, 57);
    let master_default_poseidon_column_hasher = Poseidon::<F, 3, 2>::new(8, 57);

    // Basically `hashes` is of length 2^h - 1, where h is the height of the Merkle tree
    // The idea is that the first 2^{h - 1} items are the leaf nodes (i.e. the column hashes)
    // and the remainder comes from the Merkle tree. Actually the order is EXACTLY as you'd expect,
    // with the layers of the tree being flattened and literally appended from bottom to top

    // step 1: hash each column of the commitment (we always reveal a full column)
    let hash_column_timer = start_timer!(|| "hashing the columns".to_string());
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

    let merkelize_tree = start_timer!(|| "merkelize tree".to_string());
    merkle_tree::<D, F>(hin, hout, &master_default_poseidon_merkle_hasher);
    end_timer!(merkelize_tree);
}

/// Computes the column hashes from `comm` interpreted as a matrix of size
/// `n_rows` by `encoded_num_cols`, and writes the results into `hashes`.
///
/// ## Arguments
/// * `comm` - The flattened version of M'
/// * `hashes` - This is the thing we are populating with the column hashes.
/// * `n_rows` - Height of M and M'
/// * `encoded_num_cols` - Width of M'
/// * `offset` - Gets set to zero above; not used.
fn hash_columns<D, E, F>(
    comm: &[F],
    hashes: &mut [F],
    n_rows: usize,
    encoded_num_cols: usize,
    offset: usize,
    master_default_poseidon_column_hasher: &Poseidon<F, 3, 2>,
) where
    F: Field,
    D: FieldHashFnDigest<F> + Send + Sync,
    E: LcEncoding<F> + Send + Sync,
{
    if hashes.len() <= (1 << LOG_MIN_NCOLS) {
        // base case: run the computation

        // 1. prepare the digests for each column
        let mut digests = Vec::with_capacity(hashes.len());
        for _ in 0..hashes.len() {
            let dig =
                PoseidonSpongeHasher::new_column_hasher(master_default_poseidon_column_hasher);
            digests.push(dig);
        }

        // 2. for each row, update the digests for each column
        for row in 0..n_rows {
            for (col, digest) in digests.iter_mut().enumerate() {
                // Updates the digest with the value at `comm[row * encoded_num_cols + offset + col]`
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

/// Computes the remaining Merkle tree layers from the layer of leaves. In other
/// words, if `ins` is `[1, 2, 3, 4]`, then our Merkle tree appears as follows:
/// ```text
///       [hash(hash(1, 2), hash(3, 4))]
///   [hash(1, 2)]              [hash(3, 4)]
/// [1]           [2]           [3]           [4]
/// ```
/// and `outs` is `[hash(1, 2), hash(3, 4), hash(hash(1, 2), hash(3, 4))]`.
///
/// ## Arguments
/// `ins` - The leaves to the Merkle tree
/// `outs` - The hashes for the internal nodes up to the root
/// `master_default_poseidon_merkle_hasher` - Hasher instance to be used
fn merkle_tree<D, F>(
    ins: &[F],
    outs: &mut [F],
    master_default_poseidon_merkle_hasher: &Poseidon<F, 3, 2>,
) where
    F: Field,
    D: FieldHashFnDigest<F> + Send + Sync,
{
    // The outs (i.e. rest of the tree) should be 2^{h - 1} - 1 while the ins should be 2^{h - 1}
    assert_eq!(ins.len(), outs.len() + 1);

    // Merkle-ize just the next layer
    let (outs, rems) = outs.split_at_mut(outs.len().div_ceil(2));
    merkle_layer::<D, F>(ins, outs, master_default_poseidon_merkle_hasher);

    if !rems.is_empty() {
        // Recursively merkleize until we have nothing remaining (i.e. a single element left)
        merkle_tree::<D, F>(outs, rems, master_default_poseidon_merkle_hasher)
    }
}

/// Computes a single layer of Merkle tree from a previous layer. In other words,
/// if `ins` is `[1, 2, 3, 4]`, then `outs` should be `[hash(1, 2), hash(3, 4)]`.
///
/// ## Arguments
/// `ins` - The leaves to the Merkle tree
/// `outs` - The hashes for the next layer of internal nodes
/// `master_default_poseidon_merkle_hasher` - Hasher instance to be used
fn merkle_layer<D, F>(
    ins: &[F],
    outs: &mut [F],
    master_default_poseidon_merkle_hasher: &Poseidon<F, 3, 2>,
) where
    F: Field,
    D: FieldHashFnDigest<F> + Send + Sync,
{
    assert_eq!(ins.len(), 2 * outs.len());

    if ins.len() <= (1 << LOG_MIN_NCOLS) {
        // base case: just compute all of the hashes

        for idx in 0..outs.len() {
            let mut digest = D::new_merkle_hasher(master_default_poseidon_merkle_hasher);
            digest.update(&[ins[2 * idx]]);
            digest.update(&[ins[2 * idx + 1]]);
            outs[idx] = digest.finalize();
        }
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

/// Open the commitment to a single column of M' by
/// * Sending the column in the clear to the verifier
/// * Sending the Merkle path to the Merkle root from that column's corresponding
///   leaf node hash
///
/// Additionally, appends each of the column values *and* each of the Merkle
/// paths to the transcript writer, to match the transcript reader of the
/// verifier.
///
/// ## Arguments
/// * `comm` - actual Ligero commitment
/// * `column` - the index of the column to open
///
/// ## Returns
/// * `column_pf` - the column opening proof to be sent to the verifier
fn open_column<D, E, F>(
    transcript_writer: &mut impl ProverTranscript<F>,
    comm: &LcCommit<D, E, F>,
    mut column: usize,
) -> ProverResult<LcColumn<E, F>, ErrT<E, F>>
where
    F: Field,
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
        // Start collecting at the `column`th coordinate
        .skip(column)
        // Skip num_cols (i.e. row length) number of elements to grab each column value
        .step_by(comm.encoded_num_cols)
        .cloned()
        .collect_vec();

    // Append column values to transcript
    transcript_writer.append_elements("Column elements", &col);

    // Merkle path
    let mut hashes = &comm.hashes[..];
    let path_len = log2(comm.encoded_num_cols);
    let mut path = Vec::with_capacity(path_len);
    for _ in 0..path_len {
        // A clever way of getting the "other" child, i.e. either n - 1 or n + 1
        let other = (column & !1) | (!column & 1);
        assert_eq!(other ^ column, 1);
        path.push(hashes[other]);
        let (_, hashes_new) = hashes.split_at(hashes.len().div_ceil(2));
        hashes = hashes_new;
        column >>= 1;
    }
    assert_eq!(column, 0);

    // Append Merkle path to transcript
    transcript_writer.append_elements("Merkle path", &path);

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

/// Verify the evaluation of a committed polynomial and return the result. Checks that
/// * All the `r^T M'` s (i.e. column-wise) are consistent with the verifier-derived enc(r^T M)
/// * All the `b^T M'` s (i.e. column-wise) are consistent with the verifier-derived enc(b^T M)
/// * All the columns are consistent with the merkle commitment
/// * Evaluates (b^T M) * a on its own (where b^T M is given by the prover) and returns the result
///   as the evaluation
///
/// ## Arguments
/// * `root` - Merkle root, i.e. the Ligero commitment
/// * `outer_tensor` - b^T
/// * `inner_tensor` - a
/// * `proof` - Ligero evaluation proof, i.e. columns + Merkle paths
/// * `enc` - Encoding for computing M --> M'
/// * `tr` - Fiat-Shamir transcript
fn verify<E, F>(
    root: &F,
    outer_tensor: &[F],
    inner_tensor: &[F],
    aux: &E,
    transcript_reader: &mut impl VerifierTranscript<F>,
) -> VerifierResult<F, ErrT<E, F>>
where
    F: Field,
    E: LcEncoding<F> + Send + Sync,
{
    // Grab ONE global copy of Merkle + column hashing Poseidon
    let master_default_poseidon_merkle_hasher = Poseidon::<F, 3, 2>::new(8, 57);
    let master_default_poseidon_column_hasher = Poseidon::<F, 3, 2>::new(8, 57);

    // make sure arguments are well formed
    let (num_rows, orig_num_cols, encoded_num_cols) = aux.get_dims();
    if aux.get_n_col_opens() == 0 {
        return Err(VerifierError::NumColOpens);
    }
    if inner_tensor.len() != orig_num_cols {
        return Err(VerifierError::InnerTensor);
    }
    if outer_tensor.len() != num_rows {
        return Err(VerifierError::OuterTensor);
    }
    if !aux.dims_ok(orig_num_cols, encoded_num_cols) {
        return Err(VerifierError::EncodingDims);
    }

    // The prover first sends over the claimed value of b^T M, i.e. `p_eval`
    let p_eval = transcript_reader
        .consume_elements("LABEL_PE", orig_num_cols)
        .unwrap();

    // step 1d: extract columns to open
    // The verifier does this independently as well
    let cols_to_open: Vec<usize> = {
        transcript_reader
            .get_challenges("Column opening indices", aux.get_n_col_opens())
            .unwrap()
            .into_iter()
            .map(|challenge| compute_col_idx_from_transcript_challenge(challenge, encoded_num_cols))
            .collect()
    };

    // step 2: p_eval fft for column checks
    // Takes the prover claimed value for b^T M and computes enc(b^T M) = b^T M'
    let p_eval_fft = {
        let mut tmp = Vec::with_capacity(encoded_num_cols);
        tmp.extend_from_slice(&p_eval[..]);
        tmp.resize(encoded_num_cols, F::from(0));
        aux.encode(&mut tmp).unwrap(); // TOOD(ryancao): Change this back to error propagation
        tmp
    };

    // step 3: check p_random, p_eval, and col paths
    cols_to_open.iter().try_for_each(|col_idx| {
        // Read all column values + Merkle path values from transcript for given column index
        let column_vals = transcript_reader
            .consume_elements("Column elements", num_rows)
            .unwrap();
        let merkle_path_vals = transcript_reader
            .consume_elements("Merkle path", log2(encoded_num_cols))
            .unwrap();

        // Construct `LcColumn` struct for column value + path verification
        let column = LcColumn {
            col_idx: *col_idx,
            col: column_vals,
            path: merkle_path_vals,
            phantom_data: PhantomData,
        };

        // Does the RLC evaluation check for b^T as well
        let eval = verify_column_value::<E, F>(&column, outer_tensor, &p_eval_fft[*col_idx]);

        // Merkle path verification: Does hashing for each column, then Merkle tree hashes
        let path = verify_column_path::<E, F>(
            &column,
            *col_idx,
            root,
            &master_default_poseidon_merkle_hasher,
            &master_default_poseidon_column_hasher,
        );

        match (eval, path) {
            (false, _) => Err(VerifierError::ColumnEval),
            (_, false) => Err(VerifierError::ColumnPath),
            _ => Ok(()),
        }
    })?;

    // step 4: evaluate and return
    // Computes dot product between inner_tensor (i.e. a) and proof.p_eval (i.e. b^T M)
    #[cfg(not(feature = "parallel"))]
    return Ok(inner_tensor
        .iter()
        .zip(&p_eval[..])
        .map(|(t, e)| *t * e)
        .reduce(|a, v| a + v)
        .unwrap_or(F::ZERO));

    #[cfg(feature = "parallel")]
    return Ok(inner_tensor
        .par_iter()
        .zip(&p_eval[..])
        .map(|(t, e)| *t * e)
        .reduce(|| F::ZERO, |a, v| a + v));
}

/// Check a column opening by
/// * Computing a linear hash over the column elements
/// * Taking that hash as the Merkle leaf, and computing pairwise hashes against
///   the Merkle path
/// * Checking that against `root`
///
/// ## Arguments
/// * `column` - M' column to be opened
/// * `col_num` - Index of the column within M'
/// * `root` - Merkle root, i.e. the Ligero commitment
/// * `master_default_poseidon_merkle_hasher` - Hasher for Merkle path
/// * `master_default_poseidon_column_hasher` - Hasher for column --> leaf
fn verify_column_path<E, F>(
    column: &LcColumn<E, F>,
    col_num: usize,
    root: &F,
    master_default_poseidon_merkle_hasher: &Poseidon<F, 3, 2>,
    master_default_poseidon_column_hasher: &Poseidon<F, 3, 2>,
) -> bool
where
    F: Field,
    E: LcEncoding<F> + Send + Sync,
{
    // New Poseidon params + Poseidon hasher
    let mut digest = PoseidonSpongeHasher::new_column_hasher(master_default_poseidon_column_hasher);

    // Just eat up the column elements themselves
    for e in &column.col[..] {
        e.digest_update(&mut digest);
    }

    // check Merkle path
    let mut hash = digest.finalize();
    digest = PoseidonSpongeHasher::new_merkle_hasher(master_default_poseidon_merkle_hasher);
    let mut col = col_num;
    for p in &column.path[..] {
        if col.is_multiple_of(2) {
            digest.update(&[hash]);
            digest.update(&[*p]);
        } else {
            digest.update(&[*p]);
            digest.update(&[hash]);
        }
        hash = digest.finalize();
        digest = PoseidonSpongeHasher::new_merkle_hasher(master_default_poseidon_merkle_hasher);
        col >>= 1;
    }

    &hash == root
}

/// Checks that the jth index of `b_T M'` matches the value of `b_T * M'[j]`, where
/// the `b_T M'` value is computed via `enc(b_T M)`, and the second via the prover
/// sending over `M'[j]` and the verifier manually computing `b_T * M'[j]`.
///
/// ## Arguments
/// * `column` - The actual Ligero matrix col `M_j`
/// * `tensor` - The random `b^T` we are evaluating at
/// * `poly_eval` - The RLC'd, evaluated version `b^T M'[j]`
fn verify_column_value<E, F>(column: &LcColumn<E, F>, tensor: &[F], poly_eval: &F) -> bool
where
    F: Field,
    E: LcEncoding<F> + Send + Sync,
{
    let tensor_eval = tensor
        .iter()
        .zip(&column.col[..])
        .fold(F::ZERO, |a, (t, e)| a + *t * e);

    poly_eval == &tensor_eval
}

/// Computes the column index from a challenge over F by taking the
/// least significant bits from the bit-wise representation of `challenge`
/// and directly using those as the `col_idx`.
///
/// Note that this works since the number of columns in M' is always
/// a power of two!
///
/// ## Arguments
/// * `challenge` - the Fiat-Shamir challenge representing the column idx
/// * `encoded_num_cols` - the number of columns within M'
///
/// ## Returns
/// * `col_idx` - a value 0 \leq col_idx < encoded_num_cols
fn compute_col_idx_from_transcript_challenge<F: Field>(
    challenge: F,
    encoded_num_cols: usize,
) -> usize {
    // Get the number of necessary bits
    let log_col_len = log2(encoded_num_cols);
    debug_assert!(log_col_len < 32);

    let challenge_le_bytes = challenge.to_repr().as_ref().to_vec();
    let col_idx =
        get_least_significant_bits_to_usize_little_endian(challenge_le_bytes.to_vec(), log_col_len);

    // Sanitycheck
    assert!(col_idx < encoded_num_cols);
    col_idx
}

/// Evaluate the committed polynomial using the supplied "outer" tensor
/// and generate a proof of (1) low-degreeness and (2) correct evaluation.
///
/// ## Arguments
/// * `comm` - Prover-side commitment generated via the [commit()] function
/// * `outer_tensor` - b^T
/// * `enc` - Encoding auxiliary struct containing metadata from FFT
/// * `tr` - Fiat-Shamir transcript
fn prove<D, E, F>(
    comm: &LcCommit<D, E, F>,
    outer_tensor: &[F],
    enc: &E,
    tr: &mut impl ProverTranscript<F>,
) -> ProverResult<(), ErrT<E, F>>
where
    F: Field,
    D: FieldHashFnDigest<F> + Send + Sync,
    E: LcEncoding<F> + Send + Sync,
{
    // make sure arguments are well formed
    check_comm(comm, enc)?;
    if outer_tensor.len() != comm.n_rows {
        return Err(ProverError::OuterTensor);
    }

    // next, evaluate the polynomial using the supplied tensor
    let p_eval = {
        let mut tmp = vec![F::ZERO; comm.orig_num_cols];
        // Take the vector-matrix product b^T M
        collapse_columns::<E, F>(&comm.coeffs, outer_tensor, &mut tmp, comm.orig_num_cols, 0);
        tmp
    };

    // add p_eval to the transcript
    p_eval
        .iter()
        .for_each(|coeff| tr.append("LABEL_PE", *coeff));

    // Sample the appropriate number of columns to open from the transcript
    let n_col_opens = enc.get_n_col_opens();
    let _columns: Vec<LcColumn<E, F>> = {
        let cols_to_open: Vec<usize> = tr
            .get_challenges("Column opening indices", n_col_opens)
            .into_iter()
            .map(|challenge| {
                compute_col_idx_from_transcript_challenge(challenge, comm.encoded_num_cols)
            })
            .collect();

        // Send columns + Merkle paths to verifier
        cols_to_open
            .iter()
            .map(|&col| open_column(tr, comm, col))
            .collect::<ProverResult<Vec<LcColumn<E, F>>, ErrT<E, F>>>()?
    };

    Ok(())
}

/// This takes the product b^T M, and stores the result in `poly`.
///
/// ## Arguments
/// * `coeffs` - M, but flattened
/// * `tensor` - b^T
/// * `poly` - The component of M we are currently looking at (for parallelism)
/// * `n_rows` - Height of M
/// * `orig_num_cols` - Width of M
/// * `offset` - Not used; this is always set to zero.
pub fn collapse_columns<E, F>(
    coeffs: &[F],
    tensor: &[F],
    poly: &mut [F],
    orig_num_cols: usize,
    offset: usize,
) where
    F: Field,
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
            || collapse_columns::<E, F>(coeffs, tensor, lo, orig_num_cols, offset),
            || collapse_columns::<E, F>(coeffs, tensor, hi, orig_num_cols, offset + half_cols),
        );
    }
}

// ----------------------- TESTING ONLY ----------------------- //

/// Non-parallel version of [merkleize()] function above! For testing
/// correctness of the parallel function.
#[cfg(test)]
fn merkleize_ser<D, E, F>(
    comm: &mut LcCommit<D, E, F>,
    master_default_poseidon_merkle_hasher: &Poseidon<F, 3, 2>,
    master_default_poseidon_column_hasher: &Poseidon<F, 3, 2>,
) where
    F: Field,
    D: FieldHashFnDigest<F> + Send + Sync,
    E: LcEncoding<F> + Send + Sync,
{
    let hashes = &mut comm.hashes;

    // hash each column
    for (col, hash) in hashes.iter_mut().enumerate().take(comm.encoded_num_cols) {
        let mut digest =
            PoseidonSpongeHasher::new_column_hasher(master_default_poseidon_column_hasher);
        for row in 0..comm.n_rows {
            comm.comm[row * comm.encoded_num_cols + col].digest_update(&mut digest);
        }
        *hash = digest.finalize();
    }

    // compute rest of Merkle tree
    let (mut ins, mut outs) = hashes.split_at_mut(comm.encoded_num_cols);
    while !outs.is_empty() {
        for idx in 0..ins.len() / 2 {
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

/// Evaluate the committed polynomial using the "outer" tensor, i.e. b^T M.
///
/// ## Arguments
/// * `comm` - Prover Ligero commitment generated by [commit()] containing M.
/// * `tensor` - b^T
#[cfg(test)]
fn eval_outer<D, E, F>(comm: &LcCommit<D, E, F>, tensor: &[F]) -> ProverResult<Vec<F>, ErrT<E, F>>
where
    F: Field,
    D: FieldHashFnDigest<F> + Send + Sync,
    E: LcEncoding<F> + Send + Sync,
{
    if tensor.len() != comm.n_rows {
        return Err(ProverError::OuterTensor);
    }

    // allocate result and compute
    let mut poly = vec![F::ZERO; comm.orig_num_cols];
    collapse_columns::<E, F>(&comm.coeffs, tensor, &mut poly, comm.orig_num_cols, 0);

    Ok(poly)
}

/// Serial version of [eval_outer()] for testing purposes only!
#[cfg(test)]
fn eval_outer_ser<D, E, F>(
    comm: &LcCommit<D, E, F>,
    tensor: &[F],
) -> ProverResult<Vec<F>, ErrT<E, F>>
where
    F: Field,
    D: FieldHashFnDigest<F> + Send + Sync,
    E: LcEncoding<F> + Send + Sync,
{
    if tensor.len() != comm.n_rows {
        return Err(ProverError::OuterTensor);
    }

    let mut poly = vec![F::ZERO; comm.orig_num_cols];
    for (row, tensor_val) in tensor.iter().enumerate() {
        for (col, val) in poly.iter_mut().enumerate() {
            let entry = row * comm.orig_num_cols + col;
            *val += comm.coeffs[entry] * tensor_val;
        }
    }

    Ok(poly)
}

/// Computes b^T M' (where `tensor` is b^T and `comm.comm` is M').
///
/// ## Arguments
/// * `comm` - Prover Ligero commitment generated by [commit()] containing M.
/// * `tensor` - b^T
#[cfg(test)]
fn eval_outer_fft<D, E, F>(
    comm: &LcCommit<D, E, F>,
    tensor: &[F],
) -> ProverResult<Vec<F>, ErrT<E, F>>
where
    F: Field,
    D: FieldHashFnDigest<F> + Send + Sync,
    E: LcEncoding<F> + Send + Sync,
{
    if tensor.len() != comm.n_rows {
        return Err(ProverError::OuterTensor);
    }

    // Allocate resulting vector
    let mut poly_fft = vec![F::ZERO; comm.encoded_num_cols];

    // Compute dot product column-by-column in M'
    for (coeffs, tensorval) in comm.comm.chunks(comm.encoded_num_cols).zip(tensor.iter()) {
        for (coeff, polyval) in coeffs.iter().zip(poly_fft.iter_mut()) {
            *polyval += *coeff * tensorval;
        }
    }

    Ok(poly_fft)
}
