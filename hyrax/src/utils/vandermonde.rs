use itertools::repeat_n;
use shared_types::Field;

/// Performs conversion from evaluations of a polynomial of degree d at 0, 1, ..., d to the d+1
/// coefficients, using the Vandermonde matrix inverse V^-1.
/// The conversion is done in O(d^2) time.
/// Employs a decomposition of V^-1 = U^-1 D^-1 L^-1, where U^-1, D^-1, and L^-1 are upper
/// triangular, diagonal, and lower triangular, respectively. The utility of this decomposition is
/// that the U^-1 (resp. D^-1, L^-1) matrix for rank d' is found as the principal minor of U^-1
/// (resp. D^-1, L^-1) of rank d, for any d >= d'. Thus only the decomposition for the largest d
/// required needs to be stored.
/// The decomposition grows elastically as needed.
/// For details of the decomposition, see "Symmetric functions and the Vandermonde matrix", Oruç &
/// Akmaz, 2004, specifically equations 4.4 - 4.6, all evaluated at q=1.  Identities 2.6 and 2.9
/// (both evaluated at q=1) tell us how to extend the U^-1, L^-1 (D^-1 being obvious).
#[derive(Debug)]
pub struct VandermondeInverse<F: Field> {
    /// The matrix L^-1, in row major order.
    pub l_inv: Vec<Vec<F>>,
    /// The diagonal matrix D^-1 (just the diagonal entries)
    pub d_inv: Vec<F>,
    /// The matrix U^-1, in row major order.
    pub u_inv: Vec<Vec<F>>,
    /// The number of rows/columns in the matrices.
    pub rank: usize,
}

impl<F: Field> Default for VandermondeInverse<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Field> VandermondeInverse<F> {
    /// Create a new VandermondeInverse with rank 1 (rank grows elastically as needed).
    pub fn new() -> Self {
        VandermondeInverse {
            l_inv: vec![vec![F::ONE]],
            d_inv: vec![F::ONE],
            u_inv: vec![vec![F::ONE]],
            rank: 1,
        }
    }

    /// Increase the rank by one.
    pub fn extend(&mut self) {
        let n = self.rank;

        // Extend the l inverse matrix
        // extend the existing rows i.e. add a column
        self.l_inv = self
            .l_inv
            .iter()
            .map(|row| {
                let mut new_row = row.clone();
                new_row.push(F::ZERO);
                new_row
            })
            .collect();
        // add a new row
        let last_row = self.l_inv.last().unwrap();
        let mut next_row = Vec::with_capacity(n + 1);
        let mut first_val = F::ONE;
        if n % 2 == 1 {
            first_val = first_val.neg();
        }
        next_row.push(first_val);
        for j in 1..n + 1 {
            let value = last_row[j - 1] - last_row[j];
            next_row.push(value);
        }
        self.l_inv.push(next_row);

        // Extend the diagonal matrix
        self.d_inv
            .push(self.d_inv[n - 1] * F::from(n as u64).invert().unwrap());

        // Extend the u inverse matrix
        // NB it seems silly that this is row major, but then we do have to perform matrix-vector
        // multiplication, with the column vector on the right, so perhaps this is not so silly.
        // Add a row of zeros
        self.u_inv.push(repeat_n(F::ZERO, n).collect());
        // Determine the new column
        let mut next_col = Vec::with_capacity(n + 1);
        next_col.push(F::ZERO);
        for k in 1..n + 1 {
            next_col
                .push(self.u_inv[k - 1][n - 1] - F::from((n - 1) as u64) * self.u_inv[k][n - 1]);
        }
        self.u_inv = self
            .u_inv
            .iter()
            .zip(next_col.iter())
            .map(|(row, val)| {
                let mut new_row = row.clone();
                new_row.push(*val);
                new_row
            })
            .collect();

        self.rank += 1;
    }

    /// Convert evaluations of a polynomial of degree d at 0, 1, ..., d to the d+1 coefficients.
    /// Runs in O(d^2) time.
    pub fn convert_to_coefficients(&mut self, evaluations: Vec<F>) -> Vec<F> {
        let rank = evaluations.len();
        // Extend the decomposition as needed
        while self.rank < rank {
            self.extend();
        }
        // NB in the following, we extract the principal minors using .take(rank)
        let l_inv_dot_eval: Vec<_> = self
            .l_inv
            .iter()
            .take(rank)
            .map(|row| {
                row.iter()
                    .zip(evaluations.iter())
                    .fold(F::ZERO, |acc, (l_inv, eval)| acc + *l_inv * eval)
            })
            .collect();
        let d_inv_dot_l_inv_dot_eval = self
            .d_inv
            .iter()
            .take(rank)
            .zip(l_inv_dot_eval.iter())
            .map(|(d_inv, l_inv_dot)| *d_inv * l_inv_dot)
            .collect::<Vec<F>>();
        let coefficients = self
            .u_inv
            .iter()
            .take(rank)
            .map(|row| {
                row.iter()
                    .zip(d_inv_dot_l_inv_dot_eval.iter())
                    .fold(F::ZERO, |acc, (u_inv, val)| acc + *u_inv * val)
            })
            .collect();
        coefficients
    }
}
