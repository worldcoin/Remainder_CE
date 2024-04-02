//!Module for dealing with the Beta equality function

use std::{collections::HashMap, fmt::Debug};

use remainder_shared_types::FieldExt;
use serde::{Deserialize, Serialize};

use thiserror::Error;

#[derive(Error, Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "F: FieldExt")]
pub struct NewBeta<F: FieldExt> {
    pub unbound_values: HashMap<usize, F>,
    pub updated_values: HashMap<usize, F>,
}

pub fn compute_beta_over_two_challenges<F: FieldExt>(
    challenge_one: &Vec<F>,
    challenge_two: &Vec<F>,
) -> F {
    assert_eq!(challenge_one.len(), challenge_two.len());

    // --- Formula is just \prod_i (x_i * y_i) + (1 - x_i) * (1 - y_i) ---
    let one = F::one();
    challenge_one
        .iter()
        .zip(challenge_two.iter())
        .fold(F::one(), |acc, (x_i, y_i)| {
            acc * ((*x_i * y_i) + (one - x_i) * (one - y_i))
        })
}

impl<F: FieldExt> NewBeta<F> {
    /// Construct a new beta table using a single claim
    pub fn new(layer_claim_vars_and_index: Vec<(usize, F)>) -> Self {
        let mut beta_elems_map = HashMap::<usize, F>::new();
        layer_claim_vars_and_index.iter().for_each(|(idx, elem)| {
            beta_elems_map.insert(*idx, *elem);
        });
        NewBeta {
            unbound_values: beta_elems_map,
            updated_values: HashMap::<usize, F>::new(),
        }
    }

    pub(crate) fn beta_update(&mut self, round_index: usize, challenge: F) {
        let val_to_update = self.unbound_values.remove(&round_index).unwrap();
        let updated_val =
            ((F::one() - val_to_update) * (F::one() - challenge)) + (val_to_update * challenge);
        self.updated_values.insert(round_index, updated_val);
    }
}
