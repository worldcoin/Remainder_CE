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
*/

use digest::{Digest, Output, generic_array::typenum::UTerm};
use err_derive::Error;
use ff::{Field, PrimeField};
use merlin::Transcript;
use rand::{
    distributions::{Distribution, Uniform},
    SeedableRng, Rng,
};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use remainder::FieldExt;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::iter::repeat_with;

use ark_std::test_rng;

mod macros;

#[cfg(test)]
mod tests;
/// For Poseidon hashing (implementation with respect to Digest and Transcript)
pub mod poseidon_ligero;

/// Trait for a field element that can be hashed via [digest::Digest]
pub trait FieldHash {
    /// A representation of `Self` that can be converted to a slice of `u8`.
    type HashRepr: AsRef<[u8]>;

    /// Convert `Self` into a `HashRepr` for hashing
    fn to_hash_repr(&self) -> Self::HashRepr;

    /// Update the digest `d` with the `HashRepr` of `Self`
    fn digest_update<D: Digest>(&self, d: &mut D) {
        d.update(self.to_hash_repr())
    }

    /// Update the [merlin::Transcript] `t` with the `HashRepr` of `Self` with label `l`
    fn transcript_update(&self, t: &mut Transcript, l: &'static [u8]) {
        t.append_message(l, self.to_hash_repr().as_ref())
    }
}

// impl<T: PrimeField> FieldHash for T {
//     type HashRepr = T::Repr;

//     fn to_hash_repr(&self) -> Self::HashRepr {
//         PrimeField::to_repr(self)
//     }
// }

// --- Ryan's addendum ---
// impl<F: FieldExt> FieldHash for F {
//     type HashRepr = UTerm;

//     fn to_hash_repr(&self) -> Self::HashRepr {
//         FieldExt::to_repr(self)
//     }

//     fn digest_update<D: Digest>(&self, d: &mut D) {
//         d.update(self.to_hash_repr())
//     }

//     fn transcript_update(&self, t: &mut Transcript, l: &'static [u8]) {
//         t.append_message(l, self.to_hash_repr().as_ref())
//     }
// }

/// Trait representing bit size information for a field
pub trait SizedField {
    /// Ceil of log2(cardinality)
    const CLOG2: u32;
    /// Floor of log2(cardinality)
    const FLOG2: u32;
}

impl<T: PrimeField> SizedField for T {
    const CLOG2: u32 = <T as PrimeField>::NUM_BITS;
    const FLOG2: u32 = <T as PrimeField>::NUM_BITS - 1;
}

/// Trait for a linear encoding used by the polycommit
pub trait LcEncoding: Clone + std::fmt::Debug + Sync {
    /// Field over which coefficients are defined
    // type F: Field + FieldHash + std::fmt::Debug + Clone;
    type F: FieldExt + std::fmt::Debug + Clone;

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
    fn encode<T: AsMut<[Self::F]>>(&self, inp: T) -> Result<(), Self::Err>;

    /// Get dimensions for this encoding instance on an input vector of length `len`
    fn get_dims(&self, len: usize) -> (usize, usize, usize);

    /// Check that supplied dimensions are compatible with this encoding
    fn dims_ok(&self, n_per_row: usize, n_cols: usize) -> bool;

    /// Get the number of column openings required for this encoding
    fn get_n_col_opens(&self) -> usize;

    /// Get the number of degree tests required for this encoding
    fn get_n_degree_tests(&self) -> usize;
}

// local accessors for enclosed types
type FldT<E> = <E as LcEncoding>::F;
type ErrT<E> = <E as LcEncoding>::Err;

/// Err variant for prover operations
#[derive(Debug, Error)]
pub enum ProverError<ErrT>
where
    ErrT: std::fmt::Debug + std::error::Error + 'static,
{
    /// size too big
    #[error(display = "n_cols is too large for this encoding")]
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

/// result of a verifier operation
pub type VerifierResult<T, ErrT> = Result<T, VerifierError<ErrT>>;

/// a commitment
#[derive(Debug, Clone)]
pub struct LcCommit<E, F>
where
    E: LcEncoding,
    F: FieldExt
{
    // --- Flattened version of M' (encoded) matrix ---
    comm: Vec<FldT<E>>,
    // --- Flattened version of M (non-encoded) matrix ---
    coeffs: Vec<FldT<E>>,
    // --- Matrix dims ---
    n_rows: usize, // Height of M (and M')
    n_cols: usize, // Width of M
    n_per_row: usize, // Width of M'
    // --- TODO!(ryancao): What this? ---
    // hashes: Vec<Output<D>>,
    hashes: Vec<F>
}

#[derive(Debug, Serialize, Deserialize)]
struct WrappedLcCommit<S>
where
    S: Serialize,
{
    comm: Vec<S>,
    coeffs: Vec<S>,
    n_rows: usize,
    n_cols: usize,
    n_per_row: usize,
    hashes: Vec<WrappedOutput>,
}

impl<S> WrappedLcCommit<S>
where
    S: Serialize,
{
    /// turn a WrappedLcCommit into an LcCommit
    fn unwrap<E, F: FieldExt>(self) -> LcCommit<E, F>
    where
        E: LcEncoding<F = F>,
    {
        let hashes = self.hashes.into_iter().map(|c| c.unwrap::<E>().root).collect();

        LcCommit {
            comm: self.comm,
            coeffs: self.coeffs,
            n_rows: self.n_rows,
            n_cols: self.n_cols,
            n_per_row: self.n_per_row,
            hashes,
        }
    }
}

impl<E, F> LcCommit<E, F>
where
    E: LcEncoding,
    E::F: Serialize,
    F: FieldExt,
{
    fn wrapped(&self) -> WrappedLcCommit<FldT<E>> {
        let hashes_wrapped = self.hashes.iter().map(|h| WrappedOutput { bytes: h.to_vec() }).collect();

        WrappedLcCommit {
            comm: self.comm.clone(),
            coeffs: self.coeffs.clone(),
            n_rows: self.n_rows,
            n_cols: self.n_cols,
            n_per_row: self.n_per_row,
            hashes: hashes_wrapped,
        }
    }
}

impl<E, F> Serialize for LcCommit<E, F>
where
    E: LcEncoding,
    E::F: Serialize,
    F: FieldExt
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.wrapped().serialize(serializer)
    }
}

impl<'de, E, F> Deserialize<'de> for LcCommit<E, F>
where
    E: LcEncoding,
    E::F: Serialize + Deserialize<'de>,
    F: FieldExt
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: Deserializer<'de>,
    {
        Ok(WrappedLcCommit::<FldT<E>>::deserialize(deserializer)?.unwrap())
    }
}

impl<E, F> LcCommit<E, F>
where
    E: LcEncoding,
    F: FieldExt,
{
    /// returns the Merkle root of this polynomial commitment (which is the commitment itself)
    pub fn get_root(&self) -> LcRoot<E, F> {
        LcRoot {
            root: self.hashes.last().cloned().unwrap(),
            _p: Default::default(),
        }
    }

    /// return the number of coefficients encoded in each matrix row
    pub fn get_n_per_row(&self) -> usize {
        self.n_per_row
    }

    /// return the number of columns in the encoded matrix
    pub fn get_n_cols(&self) -> usize {
        self.n_cols
    }

    /// return the number of rows in the encoded matrix
    pub fn get_n_rows(&self) -> usize {
        self.n_rows
    }

    /// generate a commitment to a polynomial
    pub fn commit(coeffs: &[FldT<E>], enc: &E) -> ProverResult<Self, ErrT<E>> {
        commit(coeffs, enc)
    }

    /// Generate an evaluation of a committed polynomial
    pub fn prove(
        &self,
        outer_tensor: &[FldT<E>],
        enc: &E,
        tr: &mut Transcript,
    ) -> ProverResult<LcEvalProof<E, F>, ErrT<E>> {
        prove(self, outer_tensor, enc, tr)
    }
}

/// A Merkle root corresponding to a committed polynomial
#[derive(Debug, Clone)]
pub struct LcRoot<E, F>
where
    E: LcEncoding,
    F: FieldExt,
{
    root: F,
    _p: std::marker::PhantomData<E>,
}

impl<E, F> LcRoot<E, F>
where
    E: LcEncoding,
    F: FieldExt,
{
    fn wrapped(&self) -> WrappedOutput {
        WrappedOutput {
            bytes: self.root.to_vec(),
        }
    }

    /// Convert this value into a raw F
    pub fn into_raw(self) -> F {
        self.root
    }
}

impl<E, F> AsRef<F> for LcRoot<E, F>
where
    E: LcEncoding,
    F: FieldExt
{
    fn as_ref(&self) -> &F {
        &self.root
    }
}

// support impl for serializing and deserializing proofs
#[derive(Debug, Clone, Deserialize, Serialize)]
struct WrappedOutput {
    /// wrapped output
    #[serde(with = "serde_bytes")]
    pub bytes: Vec<u8>,
}

impl WrappedOutput {
    fn unwrap<E, F>(self) -> LcRoot<E, F>
    where
        E: LcEncoding,
        F: FieldExt,
    {
        LcRoot {
            root: self.bytes.into_iter().collect::<F>(),
            _p: Default::default(),
        }
    }
}

impl<E, F> Serialize for LcRoot<E, F>
where
    E: LcEncoding,
    F: FieldExt,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.wrapped().serialize(serializer)
    }
}

impl<'de, E, F> Deserialize<'de> for LcRoot<E, F>
where
    E: LcEncoding,
    F: FieldExt
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: Deserializer<'de>,
    {
        Ok(WrappedOutput::deserialize(deserializer)?.unwrap())
    }
}

/// A column opening and the corresponding Merkle path.
#[derive(Debug, Clone)]
pub struct LcColumn<E, F>
where
    E: LcEncoding,
    F: FieldExt,
{
    col: Vec<FldT<E>>,
    path: Vec<F>,
}

impl<E, F> LcColumn<E, F>
where
    E: LcEncoding,
    E::F: Serialize,
    F: FieldExt
{
    fn wrapped(&self) -> WrappedLcColumn<FldT<E>> {
        let path_wrapped = (0..self.path.len())
            .map(|i| WrappedOutput {
                bytes: self.path[i].to_vec(),
            })
            .collect();

        WrappedLcColumn {
            col: self.col.clone(),
            path: path_wrapped,
        }
    }
}

// A column opening and the corresponding Merkle path.
#[derive(Debug, Clone, Deserialize, Serialize)]
struct WrappedLcColumn<F>
where
    F: Serialize,
{
    col: Vec<F>,
    path: Vec<WrappedOutput>,
}

impl<S> WrappedLcColumn<S>
where
    S: Serialize,
{
    /// turn WrappedLcColumn into LcColumn
    fn unwrap<E, F>(self) -> LcColumn<E, F>
    where
        E: LcEncoding<F = F>,
        F: FieldExt,
    {
        let col = self.col;
        let path = self
            .path
            .into_iter()
            .map(|v| v.bytes.into_iter().collect::<F>())
            .collect();

        LcColumn { col, path }
    }
}

impl<E, F> Serialize for LcColumn<E, F>
where
    E: LcEncoding,
    E::F: Serialize,
    F: FieldExt,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.wrapped().serialize(serializer)
    }
}

impl<'de, E, F> Deserialize<'de> for LcColumn<E, F>
where
    E: LcEncoding,
    E::F: Serialize + Deserialize<'de>,
    F: FieldExt,
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: Deserializer<'de>,
    {
        Ok(WrappedLcColumn::<FldT<E>>::deserialize(deserializer)?.unwrap())
    }
}

/// An evaluation and proof of its correctness and of the low-degreeness of the commitment.
#[derive(Debug, Clone)]
pub struct LcEvalProof<E, F>
where
    E: LcEncoding,
    F: FieldExt,
{
    n_cols: usize,
    p_eval: Vec<FldT<E>>,
    p_random_vec: Vec<Vec<FldT<E>>>,
    columns: Vec<LcColumn<E, F>>,
}

impl<E, F> LcEvalProof<E, F>
where
    E: LcEncoding,
    F: FieldExt,
{
    /// Get the number of elements in an encoded vector
    pub fn get_n_cols(&self) -> usize {
        self.n_cols
    }

    /// Get the number of elements in an unencoded vector
    pub fn get_n_per_row(&self) -> usize {
        self.p_eval.len()
    }

    /// Verify an evaluation proof and return the resulting evaluation
    pub fn verify(
        &self,
        root: &Output<F>,
        outer_tensor: &[FldT<E>],
        inner_tensor: &[FldT<E>],
        enc: &E,
        tr: &mut Transcript,
    ) -> VerifierResult<FldT<E>, ErrT<E>> {
        verify(root, outer_tensor, inner_tensor, self, enc, tr)
    }
}

impl<E, F> LcEvalProof<E, F>
where
    E: LcEncoding,
    E::F: Serialize,
    F: FieldExt,
{
    fn wrapped(&self) -> WrappedLcEvalProof<FldT<E>> {
        let columns_wrapped = (0..self.columns.len())
            .map(|i| self.columns[i].wrapped())
            .collect();

        WrappedLcEvalProof {
            n_cols: self.n_cols,
            p_eval: self.p_eval.clone(),
            p_random_vec: self.p_random_vec.clone(),
            columns: columns_wrapped,
        }
    }
}

/// An evaluation and proof of its correctness and of the low-degreeness of the commitment.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WrappedLcEvalProof<F>
where
    F: Serialize,
{
    n_cols: usize,
    p_eval: Vec<F>,
    p_random_vec: Vec<Vec<F>>,
    columns: Vec<WrappedLcColumn<F>>,
}

impl<F> WrappedLcEvalProof<F>
where
    F: Serialize,
{
    /// turn a WrappedLcEvalProof into an LcEvalProof
    fn unwrap<E, F>(self) -> LcEvalProof<E, F>
    where
        E: LcEncoding<F = F>,
        F: FieldExt,
    {
        let columns = self.columns.into_iter().map(|c| c.unwrap()).collect();

        LcEvalProof {
            n_cols: self.n_cols,
            p_eval: self.p_eval,
            p_random_vec: self.p_random_vec,
            columns,
        }
    }
}

impl<E, F> Serialize for LcEvalProof<E, F>
where
    E: LcEncoding,
    E::F: Serialize,
    F: FieldExt,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.wrapped().serialize(serializer)
    }
}

impl<'de, E, F> Deserialize<'de> for LcEvalProof<E, F>
where
    E: LcEncoding,
    E::F: Serialize + Deserialize<'de>,
    F: FieldExt,
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: Deserializer<'de>,
    {
        Ok(WrappedLcEvalProof::<FldT<E>>::deserialize(deserializer)?.unwrap())
    }
}

/// Compute number of degree tests required for `lambda`-bit security
/// for a code with `len`-length codewords over `flog2`-bit field
/// -- This is used in Verify and Prove
pub fn n_degree_tests(lambda: usize, len: usize, flog2: usize) -> usize {
    // -- den = log2(|F|) - log2(|codeword|) = how many bits of security are left in the field?
    // -- |codeword| = n_cols
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
fn commit<E, F>(coeffs_in: &[FldT<E>], enc: &E) -> ProverResult<LcCommit<E, F>, ErrT<E>>
where
    E: LcEncoding,
    F: FieldExt
{
    // --- Matrix size params ---
    // n_rows: Total number of matrix rows (i.e. height)
    // n_per_row: Total number of UNENCODED matrix cols (i.e. width)
    // n_cols: Total number of ENCODED matrix cols (i.e. n_per_row * \rho^{-1})
    let (n_rows, n_per_row, n_cols) = enc.get_dims(coeffs_in.len());

    // check that parameters are ok
    assert!(n_rows * n_per_row >= coeffs_in.len());
    assert!((n_rows - 1) * n_per_row < coeffs_in.len());
    assert!(enc.dims_ok(n_per_row, n_cols));

    // matrix (encoded as a vector)
    // XXX(zk) pad coeffs
    // let mut coeffs = vec![FldT::<E>::zero(); n_rows * n_per_row];
    // let mut comm = vec![FldT::<E>::zero(); n_rows * n_cols];
    let mut coeffs = vec![FldT::<E>::from(0_u32); n_rows * n_per_row];
    let mut comm = vec![FldT::<E>::from(0_u32); n_rows * n_cols];

    // local copy of coeffs with padding
    coeffs
        .par_chunks_mut(n_per_row)
        .zip(coeffs_in.par_chunks(n_per_row))
        .for_each(|(c, c_in)| {
            c[..c_in.len()].copy_from_slice(c_in);
        });

    // now compute FFTs
    // --- Go through each row of M' (the encoded matrix), as well as each row of M (the unencoded matrix) ---
    // --- and make a copy, then perform the encoding (i.e. FFT) ---
    comm.par_chunks_mut(n_cols)
        .zip(coeffs.par_chunks(n_per_row))
        .try_for_each(|(r, c)| {
            r[..c.len()].copy_from_slice(c);
            enc.encode(r)
        })?;

    // compute Merkle tree
    let n_cols_np2 = n_cols
        .checked_next_power_of_two()
        .ok_or(ProverError::TooBig)?;

    let mut ret = LcCommit {
        comm,
        coeffs,
        n_rows,
        n_cols,
        n_per_row,
        // --- There are 2^{k + 1} - 1 total hash things ---
        // TODO!(ryancao): Why...?
        hashes: vec![<Output<D> as Default>::default(); 2 * n_cols_np2 - 1],
    };

    // --- A sanitycheck of some sort, I assume? ---
    check_comm(&ret, enc)?;

    // --- Computes Merkle commitments for each column using the Digest ---
    // --- then hashes all the col commitments together using the Digest again ---
    merkleize(&mut ret);

    Ok(ret)
}

// -- This seems to be checking the size of various commitment parameters
fn check_comm<E, F>(comm: &LcCommit<E, F>, enc: &E) -> ProverResult<(), ErrT<E>>
where
    E: LcEncoding,
    F: FieldExt,
{
    // -- |commitment| = |rows| * |cols|, where cols are the ENCODED cols 
    let comm_sz = comm.comm.len() != comm.n_rows * comm.n_cols;
    // -- |commitment_coeffs|  = |rows| * |n_per_row| where `n_per_row` are the UNENCODED cols 
    // -- that is, this is the |coeffs| of the actual polynomial
    let coeff_sz = comm.coeffs.len() != comm.n_rows * comm.n_per_row;
    // -- hmmm...does the prover keep the hashes of all the merkle tree nodes, 
    // -- so that it does not have to recompute all this during the opening phase?
    let hashlen = comm.hashes.len() != 2 * comm.n_cols.next_power_of_two() - 1;
    let dims = !enc.dims_ok(comm.n_per_row, comm.n_cols);

    if comm_sz || coeff_sz || hashlen || dims {
        Err(ProverError::Commit)
    } else {
        Ok(())
    }
}

fn merkleize<E, F>(comm: &mut LcCommit<E, F>)
where
    E: LcEncoding,
    F: FieldExt,
{
    // step 1: hash each column of the commitment (we always reveal a full column)
    let hashes = &mut comm.hashes[..comm.n_cols];
    hash_columns::<E, F>(&comm.comm, hashes, comm.n_rows, comm.n_cols, 0);

    // step 2: compute rest of Merkle tree
    let len_plus_one = comm.hashes.len() + 1;
    assert!(len_plus_one.is_power_of_two());
    let (hin, hout) = comm.hashes.split_at_mut(len_plus_one / 2);
    merkle_tree::<F>(hin, hout);
}

fn hash_columns<D, E>(
    comm: &[FldT<E>],
    // --- This is the thing we are populating ---
    hashes: &mut [Output<D>],
    n_rows: usize,
    n_cols: usize,
    offset: usize, // Gets set to zero above
) where
    D: Digest,
    E: LcEncoding,
{
    if hashes.len() <= (1 << LOG_MIN_NCOLS) {
        // base case: run the computation
        // 1. prepare the digests for each column
        let mut digests = Vec::with_capacity(hashes.len());
        for _ in 0..hashes.len() {
            // column hashes start with a block of 0's
            let mut dig = D::new();
            dig.update(<Output<D> as Default>::default());
            digests.push(dig);
        }
        // 2. for each row, update the digests for each column
        for row in 0..n_rows {
            for (col, digest) in digests.iter_mut().enumerate() {
                // --- Updates the digest with the value at `comm[row * n_cols + offset + col]` ---
                // TODO!(ryancao): We can simply replace this with a sponge absorb
                comm[row * n_cols + offset + col].digest_update(digest);
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
            || hash_columns::<D, E>(comm, lo, n_rows, n_cols, offset),
            || hash_columns::<D, E>(comm, hi, n_rows, n_cols, offset + half_cols),
        );
    }
}

fn merkle_tree<D>(ins: &[Output<D>], outs: &mut [Output<D>])
where
    D: Digest,
{
    // array should always be of length 2^k - 1
    assert_eq!(ins.len(), outs.len() + 1);

    let (outs, rems) = outs.split_at_mut((outs.len() + 1) / 2);
    merkle_layer::<D>(ins, outs);

    if !rems.is_empty() {
        // --- Recursively merkleize until we have nothing remaining (i.e. a single element left) ---
        merkle_tree::<D>(outs, rems)
    }
}

/// --- Computes a single Merkle tree layer by hashing adjacent pairs of "leaves" ---
fn merkle_layer<D>(ins: &[Output<D>], outs: &mut [Output<D>])
where
    D: Digest,
{
    assert_eq!(ins.len(), 2 * outs.len());

    if ins.len() <= (1 << LOG_MIN_NCOLS) {
        // base case: just compute all of the hashes

        // TODO!(ryancao): Replace this with the Poseidon hasher
        let mut digest = D::new();
        for idx in 0..outs.len() {
            // --- I see. We update the digest with the things we want to "hash" ---
            // --- Then call `finalize()` or something like that to get the hash ---
            digest.update(ins[2 * idx].as_ref());
            digest.update(ins[2 * idx + 1].as_ref());
            outs[idx] = digest.finalize_reset();
        }

    } else {
        // recursive case: split and compute
        let (inl, inr) = ins.split_at(ins.len() / 2);
        let (outl, outr) = outs.split_at_mut(outs.len() / 2);
        rayon::join(
            || merkle_layer::<D>(inl, outl),
            || merkle_layer::<D>(inr, outr),
        );
    }
}

/// Open the commitment to one column
/// @param comm -- actual Ligero commitment
/// @param column -- the index of the column to open
/// @return TODO!(ryancao)
fn open_column<E, F>(
    comm: &LcCommit<E, F>,
    mut column: usize,
) -> ProverResult<LcColumn<E, F>, ErrT<E>>
where
    E: LcEncoding,
    F: FieldExt,
{
    // make sure arguments are well formed
    if column >= comm.n_cols {
        return Err(ProverError::ColumnNumber);
    }

    // column of values
    let col = comm
        .comm
        .iter()
        // --- Start collecting at the `column`th coordinate ---
        .skip(column)
        // --- Skip num_cols (i.e. row length) number of elements to grab each column value ---
        .step_by(comm.n_cols)
        .cloned()
        .collect();

    // Merkle path
    let mut hashes = &comm.hashes[..];
    let path_len = log2(comm.n_cols);
    let mut path = Vec::with_capacity(path_len);
    for _ in 0..path_len {
        // --- Ahh I see this is the clever way of getting the "other" child ---
        // Either n - 1 or n + 1: nice work, Riad
        let other = (column & !1) | (!column & 1);
        assert_eq!(other ^ column, 1);
        // --- Mmmmmm okay so `hashes` contains all of the Merkle hashes. I see I see ---
        path.push(hashes[other].clone());
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
    Ok(LcColumn { col, path })
}

const fn log2(v: usize) -> usize {
    (63 - (v.next_power_of_two() as u64).leading_zeros()) as usize
}

/// Verify the evaluation of a committed polynomial and return the result
fn verify<E, F: FieldExt>(
    root: &Output<F>,
    outer_tensor: &[FldT<E>], // b^T
    inner_tensor: &[FldT<E>], // a
    proof: &LcEvalProof<E, F>,
    // This is not real. Well, really it just gives you the setup for being able to compute an FFT
    enc: &E,
    tr: &mut Transcript,
) -> VerifierResult<FldT<E>, ErrT<E>>
where
    D: Digest,
    E: LcEncoding,
{
    // make sure arguments are well formed
    let n_col_opens = enc.get_n_col_opens();
    if n_col_opens != proof.columns.len() || n_col_opens == 0 {
        return Err(VerifierError::NumColOpens);
    }
    let n_rows = proof.columns[0].col.len();
    let n_cols = proof.get_n_cols();
    let n_per_row = proof.get_n_per_row();
    if inner_tensor.len() != n_per_row {
        return Err(VerifierError::InnerTensor);
    }
    if outer_tensor.len() != n_rows {
        return Err(VerifierError::OuterTensor);
    }
    if !enc.dims_ok(n_per_row, n_cols) {
        return Err(VerifierError::EncodingDims);
    }

    // step 1: random tensor for degree test and random columns to test
    // step 1a: extract random tensor from transcript
    // we run multiple instances of this to boost soundness

    // --- This is for the verifier-generated versions of `r` ---
    let mut rand_tensor_vec: Vec<Vec<FldT<E>>> = Vec::new();

    // --- This is for the verifier evaluations of r^T M' ---
    // --- but computed where r^T comes from the prover (???) ---
    let mut p_random_fft: Vec<Vec<FldT<E>>> = Vec::new();

    let n_degree_tests = enc.get_n_degree_tests();
    // -- This for loop is for soundness amplification by repetition

    // --- TODO!(ryancao): We replaced ChaCha20 with this. Replace this with FS! ---
    let mut rng = test_rng();

    for i in 0..n_degree_tests {
        let rand_tensor: Vec<FldT<E>> = {
            // --- Hmm well luckily they're re-using the RNG for consistency ---
            // --- TODO!(ryancao): Need to Poseidon this in the same way ---
            // let mut key: <ChaCha20Rng as SeedableRng>::Seed = Default::default();
            // tr.challenge_bytes(E::LABEL_DT, &mut key);
            // let mut deg_test_rng = ChaCha20Rng::from_seed(key);
            // // XXX(optimization) could expand seed in parallel instead of in series
            // repeat_with(|| FldT::<E>::random(&mut deg_test_rng))
            //     .take(n_rows)
            //     .collect()

            // --- Instead of the above, we'll just use a test_rng ---
            repeat_with(|| FldT::<E>::from(rng.gen::<u64>()))
                .take(n_rows)
                .collect()
        };

        rand_tensor_vec.push(rand_tensor);

        // step 1b: eval encoding of p_random
        {
            let mut tmp = Vec::with_capacity(n_cols);
            tmp.extend_from_slice(&proof.p_random_vec[i][..]); // Copies over the random vec from the prover
            tmp.resize(n_cols, FldT::<E>::from(0_u32));
            // Ohhhhh I see. This gives the RLC \rho^{-1} * \sqrt{N} -coordinate thingy
            // of r^T M' (where `r` was prover-generated). Note that we got this by basically
            // calling `enc(r^T M)`
            enc.encode(&mut tmp)?;
            p_random_fft.push(tmp);
        };

        // step 1c: push p_random...
        proof.p_random_vec[i]
            .iter()
            .for_each(|coeff| coeff.transcript_update(tr, E::LABEL_PR));
    }

    // ...and p_eval into the transcript
    proof
        .p_eval
        .iter()
        .for_each(|coeff| coeff.transcript_update(tr, E::LABEL_PE));

    // step 1d: extract columns to open
    // --- The verifier does this independently as well ---
    let cols_to_open: Vec<usize> = {
        let mut key: <ChaCha20Rng as SeedableRng>::Seed = Default::default();
        tr.challenge_bytes(E::LABEL_CO, &mut key);
        let mut cols_rng = ChaCha20Rng::from_seed(key);
        // XXX(optimization) could expand seed in parallel instead of in series
        let col_range = Uniform::new(0usize, n_cols);
        repeat_with(|| col_range.sample(&mut cols_rng))
            .take(n_col_opens)
            .collect()
    };

    // step 2: p_eval fft for column checks
    // --- Takes the prover claimed value for b^T M and computes enc(b^T M) = b^T M' ---
    let p_eval_fft = {
        let mut tmp = Vec::with_capacity(n_cols);
        tmp.extend_from_slice(&proof.p_eval[..]);
        tmp.resize(n_cols, FldT::<E>::from(0_u32));
        enc.encode(&mut tmp)?;
        tmp
    };

    // step 3: check p_random, p_eval, and col paths
    cols_to_open
        .par_iter()
        .zip(&proof.columns[..])
        .try_for_each(|(&col_num, column)| {

            // --- Okay so we zip the indices with the actual columns ---
            let rand = {
                let mut rand = true;
                // --- This is just 1 for us; we don't need boosting ---
                for i in 0..n_degree_tests {
                    rand &=
                        // --- This literally does r^T M'_j and checks against (r^T M')[j] 
                        // (the latter is computed by the verifier) ---
                        verify_column_value(column, &rand_tensor_vec[i], &p_random_fft[i][col_num]);
                }
                rand
            };

            // --- Does the RLC evaluation check for b^T as well ---
            let eval = verify_column_value(column, outer_tensor, &p_eval_fft[col_num]);

            // --- Merkle path verification: Does hashing for each column, then Merkle tree hashes ---
            // TODO!(ryancao): Make this use Poseidon
            let path = verify_column_path(column, col_num, root);

            // --- "Very elegant, Riad" - Ryan ---
            match (rand, eval, path) {
                (false, _, _) => Err(VerifierError::ColumnDegree),
                (_, false, _) => Err(VerifierError::ColumnEval),
                (_, _, false) => Err(VerifierError::ColumnPath),
                _ => Ok(()),
            }
        })?;

    // step 4: evaluate and return
    // --- Computes dot product between inner_tensor (i.e. a) and proof.p_eval (i.e. b^T M) ---
    Ok(inner_tensor
        .par_iter()
        .zip(&proof.p_eval[..])
        .fold(|| FldT::<E>::from(0_u32), |a, (t, e)| a + *t * e)
        .reduce(|| FldT::<E>::from(0_u32), |a, v| a + v))
}

// Check a column opening
fn verify_column_path<D, E>(column: &LcColumn<D, E>, col_num: usize, root: &Output<D>) -> bool
where
    D: Digest,
    E: LcEncoding,
{
    let mut digest = D::new();
    digest.update(<Output<D> as Default>::default());

    // --- Just eat up the column elements themselves ---
    for e in &column.col[..] {
        e.digest_update(&mut digest);
    }

    // check Merkle path
    let mut hash = digest.finalize_reset();
    let mut col = col_num;
    // TODO!(ryancao): Understand this...?
    for p in &column.path[..] {
        if col % 2 == 0 {
            digest.update(&hash);
            digest.update(p);
        } else {
            digest.update(p);
            digest.update(&hash);
        }
        hash = digest.finalize_reset();
        col >>= 1;
    }

    &hash == root
}

// check column value
fn verify_column_value<D, E>(
    column: &LcColumn<D, E>, // The actual Ligero matrix col M_j
    tensor: &[FldT<E>], // The random r^T we are evaluating at
    poly_eval: &FldT<E>, // The RLC'd, evaluated version r^T M'[j]
) -> bool
where
    D: Digest,
    E: LcEncoding,
{
    let tensor_eval = tensor
        .iter()
        .zip(&column.col[..])
        .fold(FldT::<E>::from(0_u32), |a, (t, e)| a + *t * e);

    poly_eval == &tensor_eval
}

/// Evaluate the committed polynomial using the supplied "outer" tensor
/// and generate a proof of (1) low-degreeness and (2) correct evaluation.
fn prove<D, E>(
    comm: &LcCommit<D, E>,
    outer_tensor: &[FldT<E>],
    enc: &E,
    tr: &mut Transcript,
) -> ProverResult<LcEvalProof<D, E>, ErrT<E>>
where
    D: Digest,
    E: LcEncoding,
{
    // make sure arguments are well formed
    check_comm(comm, enc)?;
    if outer_tensor.len() != comm.n_rows {
        return Err(ProverError::OuterTensor);
    }

    // first, evaluate the polynomial on a random tensor (low-degree test)
    // we repeat this to boost soundness
    let mut p_random_vec: Vec<Vec<FldT<E>>> = Vec::new();
    let n_degree_tests = enc.get_n_degree_tests();

    // --- Similarly to the verifier above, we'll use the test_rng() from ark-std ---
    // TODO!(ryancao): Replace this with FS!
    let mut rng = test_rng();

    for _i in 0..n_degree_tests {
        let p_random = {
            // let mut key: <ChaCha20Rng as SeedableRng>::Seed = Default::default();
            // tr.challenge_bytes(E::LABEL_DT, &mut key);
            // let mut deg_test_rng = ChaCha20Rng::from_seed(key);
            // // XXX(optimization) could expand seed in parallel instead of in series\
            // // --- This is truly the random tensor ---
            // // TODO!(ryancao): Sample ALL of these points using Poseidon
            // let rand_tensor: Vec<FldT<E>> = repeat_with(|| FldT::<E>::random(&mut deg_test_rng))
            //     .take(comm.n_rows)
            //     .collect();

            // --- Replacing the above ChaCha20 stuff with dummy RNG ---
            let rand_tensor: Vec<FldT<E>> = repeat_with(|| FldT::<E>::from(rng.gen::<u64>()))
                .take(comm.n_rows)
                .collect();


            let mut tmp = vec![FldT::<E>::from(0_u32); comm.n_per_row];

            // --- This takes the dot product against `comm.coeffs` and stores the result in `tmp` ---
            // --- Basically this is the prover's claimed value for r^T M ---
            collapse_columns::<E>(
                &comm.coeffs,
                &rand_tensor,
                &mut tmp,
                comm.n_rows,
                comm.n_per_row,
                0,
            );
            tmp
        };
        // add p_random to the transcript
        p_random
            .iter()
            .for_each(|coeff| coeff.transcript_update(tr, E::LABEL_PR));

        p_random_vec.push(p_random);
    }

    // next, evaluate the polynomial using the supplied tensor
    let p_eval = {
        let mut tmp = vec![FldT::<E>::from(0_u32); comm.n_per_row];
        // --- Take the vector-matrix product b^T M ---
        collapse_columns::<E>(
            &comm.coeffs,
            outer_tensor,
            &mut tmp,
            comm.n_rows,
            comm.n_per_row,
            0,
        );
        tmp
    };
    // add p_eval to the transcript
    p_eval
        .iter()
        .for_each(|coeff| coeff.transcript_update(tr, E::LABEL_PE));

    // now extract the column numbers to open
    let n_col_opens = enc.get_n_col_opens();
    let columns: Vec<LcColumn<D, E>> = {
        let mut key: <ChaCha20Rng as SeedableRng>::Seed = Default::default();
        tr.challenge_bytes(E::LABEL_CO, &mut key);
        // --- Same here: replace this RNG with Poseidon sample ---
        let mut cols_rng = ChaCha20Rng::from_seed(key);
        // XXX(optimization) could expand seed in parallel instead of in series
        let col_range = Uniform::new(0usize, comm.n_cols);
        let cols_to_open: Vec<usize> = repeat_with(|| col_range.sample(&mut cols_rng))
            .take(n_col_opens)
            .collect();

        // --- Let's check out this `open_column` function ---
        // Yeah so `open_column()` basically gives you the column itself, along with a Merkle path
        // Pretty straightforward. "I LIKE it! - Creed" - Ryan
        cols_to_open
        .par_iter()
        .map(|&col| open_column(comm, col))
            .collect::<ProverResult<Vec<LcColumn<D, E>>, ErrT<E>>>()?
    };

    Ok(LcEvalProof {
        n_cols: comm.n_cols, // Number of columns
        p_eval, // Actual b^T M value
        p_random_vec, // Random vectors to check well-formedness
        columns, // Columns plus necessary opening proof content
    })
}

fn collapse_columns<E>(
    coeffs: &[FldT<E>],
    tensor: &[FldT<E>],
    poly: &mut [FldT<E>],
    n_rows: usize,
    n_per_row: usize,
    offset: usize,
) where
    E: LcEncoding,
{
    if poly.len() <= (1 << LOG_MIN_NCOLS) {
        // base case: run the computation
        // row-by-row, compute elements of dot product
        for (row, tensor_val) in tensor.iter().enumerate() {
            for (col, val) in poly.iter_mut().enumerate() {
                let entry = row * n_per_row + offset + col;
                *val += coeffs[entry] * tensor_val;
            }
        }
    } else {
        // recursive case: split and execute in parallel
        let half_cols = poly.len() / 2;
        let (lo, hi) = poly.split_at_mut(half_cols);
        rayon::join(
            || collapse_columns::<E>(coeffs, tensor, lo, n_rows, n_per_row, offset),
            || collapse_columns::<E>(coeffs, tensor, hi, n_rows, n_per_row, offset + half_cols),
        );
    }
}

// TESTING ONLY //

#[cfg(test)]
fn merkleize_ser<D, E>(comm: &mut LcCommit<D, E>)
where
    D: Digest,
    E: LcEncoding,
{
    let hashes = &mut comm.hashes;

    // hash each column
    for (col, hash) in hashes.iter_mut().enumerate().take(comm.n_cols) {
        let mut digest = D::new();
        digest.update(<Output<D> as Default>::default());
        for row in 0..comm.n_rows {
            comm.comm[row * comm.n_cols + col].digest_update(&mut digest);
        }
        *hash = digest.finalize();
    }

    // compute rest of Merkle tree
    let (mut ins, mut outs) = hashes.split_at_mut(comm.n_cols);
    while !outs.is_empty() {
        for idx in 0..ins.len() / 2 {
            let mut digest = D::new();
            digest.update(ins[2 * idx].as_ref());
            digest.update(ins[2 * idx + 1].as_ref());
            outs[idx] = digest.finalize();
        }
        let (new_ins, new_outs) = outs.split_at_mut((outs.len() + 1) / 2);
        ins = new_ins;
        outs = new_outs;
    }
}

#[cfg(test)]
// Check a column opening
fn verify_column<D, E>(
    column: &LcColumn<D, E>,
    col_num: usize,
    root: &Output<D>,
    tensor: &[FldT<E>],
    poly_eval: &FldT<E>,
) -> bool
where
    D: Digest,
    E: LcEncoding,
{
    verify_column_path(column, col_num, root) && verify_column_value(column, tensor, poly_eval)
}

// Evaluate the committed polynomial using the "outer" tensor
// --- I'm not sure why Riad is calling it a "tensor" because it's ---
// --- definitely just a vector but it's the b^T M' component ---
#[cfg(test)]
fn eval_outer<D, E>(
    comm: &LcCommit<D, E>,
    tensor: &[FldT<E>],
) -> ProverResult<Vec<FldT<E>>, ErrT<E>>
where
    D: Digest,
    E: LcEncoding,
{
    if tensor.len() != comm.n_rows {
        return Err(ProverError::OuterTensor);
    }

    // allocate result and compute
    let mut poly = vec![FldT::<E>::from(0_u32); comm.n_per_row];
    collapse_columns::<E>(
        &comm.coeffs,
        tensor,
        &mut poly,
        comm.n_rows,
        comm.n_per_row,
        0,
    );

    Ok(poly)
}

#[cfg(test)]
fn eval_outer_ser<D, E>(
    comm: &LcCommit<D, E>,
    tensor: &[FldT<E>],
) -> ProverResult<Vec<FldT<E>>, ErrT<E>>
where
    D: Digest,
    E: LcEncoding,
{
    if tensor.len() != comm.n_rows {
        return Err(ProverError::OuterTensor);
    }

    let mut poly = vec![FldT::<E>::from(0_u32); comm.n_per_row];
    for (row, tensor_val) in tensor.iter().enumerate() {
        for (col, val) in poly.iter_mut().enumerate() {
            let entry = row * comm.n_per_row + col;
            *val += comm.coeffs[entry] * tensor_val;
        }
    }

    Ok(poly)
}

/// TODO!(ryancao): Actually understand this
#[cfg(test)]
fn eval_outer_fft<D, E>(
    comm: &LcCommit<D, E>,
    tensor: &[FldT<E>],
) -> ProverResult<Vec<FldT<E>>, ErrT<E>>
where
    D: Digest,
    E: LcEncoding,
{
    if tensor.len() != comm.n_rows {
        return Err(ProverError::OuterTensor);
    }

    // --- M row length worth of 0s ---
    let mut poly_fft = vec![FldT::<E>::from(0_u32); comm.n_cols];

    // --- For each coefficient within `tensor` corresponding to a row in M ---
    for (coeffs, tensorval) in comm.comm.chunks(comm.n_cols).zip(tensor.iter()) {
        // --- Nah this is just 
        for (coeff, polyval) in coeffs.iter().zip(poly_fft.iter_mut()) {
            *polyval += *coeff * tensorval;
        }
    }

    Ok(poly_fft)
}
