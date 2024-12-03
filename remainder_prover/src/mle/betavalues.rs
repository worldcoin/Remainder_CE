//! Module for dealing with the Beta equality function.

use itertools::Itertools;
use remainder_shared_types::{utils::bookkeeping_table::initialize_tensor, Field};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fmt::Debug};

use super::{evals::MultilinearExtension, MleIndex};

/// A struct that holds the claim and the relevant bound values for the beta
/// equality MLE. Rather than storing the entire beta table, we simply store the
/// points in the claim that are still "unbound", and the points that have been
/// bound using the \[Thaler13\] definition of a beta table.
///
/// Beta tables are used to "linearize" an expression that we wish to evaluate
/// over a claimed point `(g_0, ..., g_n)`. Therefore we create an MLE that
/// evaluates to `1` at this point and `0` at every other point, which is a
/// beta table. This would be a table of size `2^n`.
///
/// Instead, we choose to store just the individual values in a hash map as we
/// don't need the entire representation in order to perform the computations
/// with beta tables.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "F: Field")]
pub struct BetaValues<F: Field> {
    /// The challenges in the claim that have not yet been "bound" in the
    /// sumcheck protocol. Keys are the "round" of sumcheck and values are
    /// which challenge in the claim that corresponds to. Every key in
    /// unbound values should be >= the current round index in sumcheck.
    pub unbound_values: HashMap<usize, F>,

    /// The challenges that have already been bound in the sumcheck protocol.
    /// Keys are the round it corresponds to, and values are
    /// `(1 - r_i)(1 - g_i) + r_i*g_i` where `i` is the key, `g_i` is the
    /// claim challenge point,
    /// and `r_i` is the current challenge point. Every key here should be
    /// < the current round index in sumcheck.
    pub updated_values: HashMap<usize, F>,
}

impl<F: Field> BetaValues<F> {
    /// Constructs a new beta table using a vector of the challenge points in a
    /// claim along with it's corresponding round index as a tuple.
    pub fn new(layer_claim_vars_and_index: Vec<(usize, F)>) -> Self {
        let mut beta_elems_map = HashMap::<usize, F>::new();
        layer_claim_vars_and_index.iter().for_each(|(idx, elem)| {
            beta_elems_map.insert(*idx, *elem);
        });
        BetaValues {
            unbound_values: beta_elems_map,
            updated_values: HashMap::<usize, F>::new(),
        }
    }

    /// Updates the given value of beta using a new challenge point. Simply
    /// `(1 - r_i)*(1 - g_i) + (r_i * g_i)` for an index `i`, previous claim
    /// challenge point `g_i` and current challenge `r_i`.
    ///
    /// We remove it from the unbound hashmap and add it to the bound hashmap.
    pub fn beta_update(&mut self, round_index: usize, challenge: F) {
        let val_to_update = self.unbound_values.remove(&round_index).unwrap();
        let updated_val =
            ((F::ONE - val_to_update) * (F::ONE - challenge)) + (val_to_update * challenge);
        self.updated_values.insert(round_index, updated_val);
    }

    /// Given a vector of mle indices, returns the relevant beta bound and
    /// unbound values we need. if the index is `Indexed(usize)`, then we grab
    /// the *unbound* value and if it is `Bound(usize, chal)` we grab the
    /// *bound* value.
    pub fn get_relevant_beta_unbound_and_bound(
        &self,
        mle_indices: &[MleIndex<F>],
    ) -> (Vec<F>, Vec<F>) {
        let bound_betas = mle_indices
            .iter()
            .filter_map(|index| match index {
                MleIndex::Bound(_, round_idx) => self.updated_values.get(round_idx).copied(),
                _ => None,
            })
            .collect_vec();

        let unbound_betas = mle_indices
            .iter()
            .filter_map(|index| match index {
                MleIndex::Indexed(round_idx) => self.unbound_values.get(round_idx).copied(),
                _ => None,
            })
            .collect_vec();

        (unbound_betas, bound_betas)
    }

    /// Takes two challenge points and computes the fully bound beta equality
    /// value.
    pub fn compute_beta_over_two_challenges(challenge_one: &[F], challenge_two: &[F]) -> F {
        assert_eq!(challenge_one.len(), challenge_two.len());

        // Formula is just: \prod_i (x_i * y_i) + (1 - x_i) * (1 - y_i)
        let one = F::ONE;
        challenge_one
            .iter()
            .zip(challenge_two.iter())
            .fold(F::ONE, |acc, (x_i, y_i)| {
                acc * ((*x_i * y_i) + (one - x_i) * (one - y_i))
            })
    }

    /// Computes the value of `\beta(challenge; b)`, where `b` is the binary
    /// representation of `idx`.
    pub fn compute_beta_over_challenge_and_index(challenge: &[F], idx: usize) -> F {
        let n = challenge.len();
        challenge.iter().enumerate().fold(F::ONE, |acc, (i, x_i)| {
            let mask = 1_usize << (n - 1 - i);
            if idx & mask != 0 {
                // i-th bit is on, multiply by `x_i`
                acc * x_i
            } else {
                // i-th bit is off, multiply by `(1 - x_i)`
                acc * (F::ONE - x_i)
            }
        })
    }

    /// Returns the full beta equality table as defined in \[Thaler13\], so over
    /// `n` challenge points it returns a table of size `2^n`. This is when we
    /// do still need the entire beta table. Essentially this is an MLE whose coefficients represent
    /// whether the index is equal to the random challenge point.
    pub fn new_beta_equality_mle(layer_claim_vars: Vec<F>) -> MultilinearExtension<F> {
        MultilinearExtension::new(initialize_tensor(&layer_claim_vars))
    }
}
