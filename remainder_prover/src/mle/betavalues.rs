//! Module for dealing with the Beta equality function.

use itertools::Itertools;
use remainder_shared_types::{utils::bookkeeping_table::initialize_tensor, Field};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use crate::mle::mle_bookkeeping_table::MleBookkeepingTables;

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
/// Instead, we represent the beta MLE `\beta(g_0, ..., g_{n-1}; x_0, ..., x_{n-1})` by mapping each
/// index `i` that participates in this MLE to either `g_i`, if `x_i` has not yet been bound, or to
/// `(1 - r_i)*(1 - g_i) + r_i*g_i`, if `x_i` has already been bound to `r_i`. Indices in the range
/// `{0, 1, ..., n-1}` which are not part of this MLE are called "unassigned" and they don't map to
/// any value.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "F: Field")]
pub struct BetaValues<F: Field> {
    /// Map from each index to a beta value type
    pub values: Vec<BetaValueType<F>>,
}

/// Type of beta values
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "F: Field")]
pub enum BetaValueType<F: Field> {
    /// In a linear round, there is no corresponding beta value
    Unassigned,
    /// Challenges that have not yet been "bound" in the
    /// sumcheck protocol, as `g_i`
    Unbound(F),
    /// Challenges that have already been bound in the sumcheck protocol,
    /// as `(1 - r_i)(1 - g_i) + r_i*g_i` where `i` is the key, `g_i` is the
    /// claim challenge point, and `r_i` is the current challenge point.
    Updated(F),
}

impl<F: Field> BetaValues<F> {
    /// Constructs a new beta table using a vector of the challenge points in a
    /// claim along with it's corresponding round index as a tuple.
    pub fn new(layer_claim_vars_and_index: Vec<(usize, F)>) -> Self {
        let mut values = Vec::new();
        layer_claim_vars_and_index.iter().for_each(|(idx, elem)| {
            if values.len() <= *idx {
                values.extend(vec![BetaValueType::Unassigned; *idx + 1 - values.len()]);
            }
            values[*idx] = BetaValueType::Unbound(*elem);
        });
        BetaValues { values }
    }

    /// Updates the given value of beta using a new challenge point. Simply
    /// `(1 - r_i)*(1 - g_i) + (r_i * g_i)` for an index `i`, previous claim
    /// challenge point `g_i` and current challenge `r_i`.
    ///
    /// We remove it from the unbound hashmap and add it to the bound hashmap.
    pub fn beta_update(&mut self, round_index: usize, challenge: F) {
        assert!(self.values.len() > round_index);
        if let BetaValueType::Unbound(val_to_update) = self.values[round_index] {
            let updated_val =
                ((F::ONE - val_to_update) * (F::ONE - challenge)) + (val_to_update * challenge);
            self.values[round_index] = BetaValueType::Updated(updated_val);
        } else {
            unreachable!()
        }
    }

    /// obtain the value at an index if it is updated
    /// do not check for out of bound
    pub fn get_updated_value(&self, index: usize) -> Option<F> {
        if self.values.len() <= index {
            None
        } else if let BetaValueType::Updated(v) = self.values[index] {
            Some(v)
        } else {
            None
        }
    }

    /// obtain the value at an index if it is unbounded
    pub fn get_unbound_value(&self, index: usize) -> Option<F> {
        match self.values.get(index) {
            Some(BetaValueType::Unbound(v)) => Some(*v),
            _ => None,
        }
    }

    /// determine if no entry is unbound
    pub fn is_fully_bounded(&self) -> bool {
        (0..self.values.len())
            .filter(|i| self.get_unbound_value(*i).is_some())
            .count()
            == 0
    }

    /// obtain product of all updated values
    pub fn fold_updated_values(&self) -> F {
        (0..self.values.len())
            .filter_map(|i| self.get_updated_value(i))
            .product()
    }

    /// Given a vector of mle indices, returns the relevant beta bound and
    /// unbound values we need. if the index is `Indexed(usize)`, then we grab
    /// the *unbound* value and if it is `Bound(usize, chal)` we grab the
    /// *bound* value.
    pub fn get_relevant_beta_unbound_and_bound(
        &self,
        mle_indices: &[MleIndex<F>],
        round_index: usize,
        computes_evals: bool,
    ) -> (Vec<F>, Vec<F>) {
        // We always want every bound value so far.
        // If we are computing evaluations for this node, then we want
        // all of the updated beta values so far.
        let bound_betas = if computes_evals {
            (0..self.values.len())
                .filter_map(|i| self.get_updated_value(i))
                .collect()
        }
        // Otherwise, we are just computing the sum of the variables
        // multiplied by the beta this round. This means that there is
        // a beta MLE outside of this expression that factors in the
        // updated values already, either in the form of the verifier
        // computing the full sumcheck evaluation multiplied by the fully
        // bound beta, or in the form of a selector where the updated
        // values are factored directly into the evaluation of the selector
        // variable itself.
        else {
            Vec::new()
        };

        let mut unbound_betas = mle_indices
            .iter()
            .filter_map(|index| match index {
                MleIndex::Indexed(i) => self.get_unbound_value(*i),
                _ => None,
            })
            .collect_vec();

        // If the MLE indices does not contain the current "independent variable",
        // then we want to manually include it in our unbound betas.
        if !mle_indices.contains(&MleIndex::Indexed(round_index)) && computes_evals {
            unbound_betas.push(self.get_unbound_value(round_index).unwrap())
        }
        (unbound_betas, bound_betas)
    }

    /// Given a bookkeeping table, returns the relevante beta bound and unbound values
    pub fn get_relevant_beta_unbound_and_bound_from_bookkeeping_table(
        &self,
        bookkeeping_table: &MleBookkeepingTables<F>,
    ) -> (Vec<F>, Vec<F>) {
        // We always want every bound value so far.
        let bound_betas = (0..self.values.len())
            .filter_map(|i| self.get_updated_value(i))
            .collect();

        let unbound_betas = bookkeeping_table
            .indices
            .iter()
            .map(|i| self.get_unbound_value(*i).unwrap())
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
