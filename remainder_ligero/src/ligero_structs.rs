use itertools::Itertools;

use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use crate::utils::{get_ligero_matrix_dims, halo2_fft};
use crate::FieldExt;
use crate::{def_labels, LcCommit, LcEncoding, LcEvalProof, LcRoot};
// use fffft::FFTError;

/// Auxiliary struct which simply keeps track of Ligero hyperparameters, e.g.
/// the matrix width and code rate.
///
/// TODO!(ryancao): Deprecate this and/or merge it with Ligero aux information!
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LigeroEncoding<F: FieldExt> {
    /// Width of the M matrix representing the original polynomial's coeffs
    pub orig_num_cols: usize,
    /// Width of the M' matrix representing the encoded version of M's rows
    pub encoded_num_cols: usize,
    /// Code rate, i.e. the ratio `encoded_num_cols` / `orig_num_cols`
    pub rho_inv: u8,
    /// Required for generic
    pub phantom: PhantomData<F>,
}

/// Total number of columns to be sent over
pub const N_COL_OPENS: usize = 128usize;

impl<F> LigeroEncoding<F>
where
    F: FieldExt,
{
    /// Grabs the matrix dimensions for M and M'
    pub fn get_dims(len: usize, rho_inv: u8, ratio: f64) -> Option<(usize, usize, usize)> {
        get_ligero_matrix_dims(len, rho_inv, ratio)
    }

    fn _dims_ok(orig_num_cols: usize, encoded_num_cols: usize) -> bool {
        let sz = orig_num_cols < encoded_num_cols;
        let pow = encoded_num_cols.is_power_of_two();
        sz && pow
    }

    /// Allows creation from total number of coefficients, code rate, and
    /// matrix width-to-height ratio.
    pub fn new(num_coeffs: usize, rho_inv: u8, ratio: f64) -> Self {
        // --- Computes the matrix size for the commitment ---
        let (_, orig_num_cols, encoded_num_cols) =
            Self::get_dims(num_coeffs, rho_inv, ratio).unwrap();
        assert!(Self::_dims_ok(orig_num_cols, encoded_num_cols));
        Self {
            orig_num_cols,
            encoded_num_cols,
            rho_inv,
            phantom: PhantomData,
        }
    }

    /// Allows creation from unencoded matrix width + encoded matrix width
    pub fn new_from_dims(orig_num_cols: usize, encoded_num_cols: usize) -> Self {
        let rho_inv = (encoded_num_cols / orig_num_cols) as u8;
        assert!(Self::_dims_ok(orig_num_cols, encoded_num_cols));
        Self {
            orig_num_cols,
            encoded_num_cols,
            rho_inv,
            phantom: PhantomData,
        }
    }
}

impl<F> LcEncoding<F> for LigeroEncoding<F>
where
    F: FieldExt,
{
    type Err = &'static str;

    def_labels!(lcpc2d_test);

    /// Computes the FFT of the row `inp`.
    ///
    /// ## Arguments
    /// * `inp` - Slice of coefficients of length `self.rho_inv` *
    ///     `self.orig_num_cols`. Note that only the first `self.orig_num_cols`
    ///     values should be nonzero.
    fn encode(&self, inp: &mut [F]) -> Result<(), Self::Err> {
        // --- So we need to convert num_cols(M) coefficients into num_cols(M) * (1 / rho) evaluations ---
        // --- All the coefficients past the original number of cols should be zero-padded ---
        debug_assert!(inp.iter().skip(self.orig_num_cols).all(|&v| v == F::ZERO));

        // --- TODO!(ryancao): This is wasteful (we clone twice!!!) ---
        let evals = halo2_fft(
            inp.iter().copied().take(self.orig_num_cols).collect_vec(),
            self.rho_inv,
        );
        inp.copy_from_slice(&evals[..]);

        Ok(())
    }

    /// Returns the matrix dimensions of the matrix generated from
    /// such an encoding, given a number of coefficients.
    ///
    /// ## Arguments
    /// * `num_coeffs` - Total number of coefficients in the polynomial.
    fn get_dims(&self, num_coeffs: usize) -> (usize, usize, usize) {
        let n_rows = (num_coeffs + self.orig_num_cols - 1) / self.orig_num_cols;
        (n_rows, self.orig_num_cols, self.encoded_num_cols)
    }

    /// Checks that an externally passed-in set of dimensions agrees with the
    /// dimensions which this encoding was initialized with.
    ///
    /// ## Arguments
    /// * `orig_num_cols` - externally determined width of M
    /// * `encoded_num_cols` - externally determined width of M'
    fn dims_ok(&self, orig_num_cols: usize, encoded_num_cols: usize) -> bool {
        let ok = Self::_dims_ok(orig_num_cols, encoded_num_cols);
        let np = orig_num_cols == self.orig_num_cols;
        let nc = encoded_num_cols == self.encoded_num_cols;

        ok && np && nc
    }

    fn get_n_col_opens(&self) -> usize {
        N_COL_OPENS
    }

    fn get_n_degree_tests(&self) -> usize {
        1
    }
}

/// Ligero commitment over generic `LcCommit`
pub type LigeroCommit<D, F> = LcCommit<D, LigeroEncoding<F>, F>;

/// Ligero evaluation proof over generic `LcEvalProof`
pub type LigeroEvalProof<D, E, F> = LcEvalProof<D, E, F>;

/// Ligero root over generic `LcRoot`
pub type LigeroRoot<F> = LcRoot<LigeroEncoding<F>, F>;
