//! Utilities involving the claims a layer makes.

/// A struct for managing groups of claims originating from the same layer.
pub mod claim_group;

/// Tests for claim-related things.
#[cfg(test)]
pub mod tests;

pub mod claim_aggregation;

use std::{collections::HashMap, fmt};

use remainder_shared_types::{field::ExtensionField, Field};
use thiserror::Error;

use serde::{Deserialize, Serialize};

use crate::{layer::LayerId, mle::mle_enum::LiftTo};

/// Errors to do with aggregating and collecting claims.
#[derive(Error, Debug, Clone)]
pub enum ClaimError {
    /// MLE indices must all be fixed.
    #[error("MLE indices must all be fixed")]
    ClaimMleIndexError,

    /// MLE within MleRef has multiple values within it.
    #[error("MLE within MleRef has multiple values within it")]
    MleRefMleError,

    /// Error aggregating claims.
    #[error("Error aggregating claims")]
    ClaimAggroError,

    /// All claims in a group should agree on the number of variables.
    #[error("All claims in a group should agree on the number of variables")]
    NumVarsMismatch,

    /// All claims in a group should agree the destination layer field.
    #[error("All claims in a group should agree the destination layer field")]
    LayerIdMismatch,

    /// Zero MLE refs cannot be used as intermediate values within a circuit!
    #[error("Zero MLE refs cannot be used as intermediate values within a circuit")]
    IntermediateZeroMLERefError,
}

/// A claim without any source/destination layer information.  See related
/// [Claim] wrapper.
/// It contains a `point \in F^n` along with the `evaluation \in F` that an
/// associated layer MLE is expected to evaluate to. In other words, if
/// `\tilde{V} : F^n -> F` is the MLE of a layer, then this claim asserts:
/// `\tilde{V}(point) == result`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(bound = "F: Field")]
pub struct RawClaim<F: Field> {
    /// The point in `F^n` at which to evaluate the layer MLE.
    point: Vec<F>,

    /// The expected result of evaluating a layer's MLE on `point`.
    evaluation: F,
}

impl<F: Field> RawClaim<F> {
    /// Constructs a new [RawClaim] from a given `point` and `evaluation`.
    pub fn new(point: Vec<F>, evaluation: F) -> Self {
        Self { point, evaluation }
    }

    /// Returns the length of the `point` vector.
    pub fn get_num_vars(&self) -> usize {
        self.point.len()
    }

    /// Returns a reference to `point \in F^n`.
    pub fn get_point(&self) -> &[F] {
        &self.point
    }

    /// Returns the expected evaluation.
    pub fn get_eval(&self) -> F {
        self.evaluation
    }
}

/// Lifts a [RawClaim<F>] to a [RawClaim<E>] in the trivial way.
impl<F: Field, E: ExtensionField<F>> LiftTo<RawClaim<E>> for RawClaim<F> {
    fn lift(&self) -> RawClaim<E> {
        RawClaim {
            point: self.point.lift(),
            evaluation: self.evaluation.into(),
        }
    }
}

/// A claim with source/destination layer information.
/// This is a wrapper around [RawClaim] which holds a `point \in F^n` and an
/// `evaluation \in F`. The claim asserts that the layer MLE `\tilde{V} : F^n ->
/// F` of the layer with ID `to_layer_id` evaluates to `evaluation` when
/// computed on `point`: `tilde{V}(point) == result`.
#[derive(Clone, Serialize, Deserialize, PartialEq)]
#[serde(bound = "F: Field")]
pub struct Claim<F: Field> {
    /// The underlying claim.
    claim: RawClaim<F>,

    /// The layer ID of the layer that produced this claim; origin layer.
    from_layer_id: LayerId,

    /// The layer ID of the layer containing the MLE this claim applies to;
    /// destination layer.
    to_layer_id: LayerId,
}

impl<F: Field> Claim<F> {
    /// Generate a new claim, given a point, an expected evaluation and
    /// origin/destination information.
    pub fn new(point: Vec<F>, evaluation: F, from_layer_id: LayerId, to_layer_id: LayerId) -> Self {
        Self {
            claim: RawClaim::new(point, evaluation),
            from_layer_id,
            to_layer_id,
        }
    }

    /// Returns the length of the `point` vector.
    pub fn get_num_vars(&self) -> usize {
        self.claim.get_num_vars()
    }

    /// Returns a reference to the `point \in F^n``.
    pub fn get_point(&self) -> &[F] {
        self.claim.get_point()
    }

    /// Returns the expected evaluation.
    pub fn get_eval(&self) -> F {
        self.claim.get_eval()
    }

    /// Returns the source Layer ID.
    pub fn get_from_layer_id(&self) -> LayerId {
        self.from_layer_id
    }

    /// Returns the destination Layer ID.
    pub fn get_to_layer_id(&self) -> LayerId {
        self.to_layer_id
    }

    /// Returns a reference to the underlying [RawClaim].
    pub fn get_raw_claim(&self) -> &RawClaim<F> {
        &self.claim
    }
}

impl<F: Field> From<Claim<F>> for RawClaim<F> {
    fn from(value: Claim<F>) -> Self {
        Self {
            point: value.claim.point,
            evaluation: value.claim.evaluation,
        }
    }
}

impl<F: fmt::Debug + Field> fmt::Debug for Claim<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Claim")
            .field("point", &self.get_point().to_vec())
            .field("result", &self.claim.get_eval())
            .field("from_layer_id", &self.from_layer_id)
            .field("to_layer_id", &self.to_layer_id)
            .finish()
    }
}

/// Keeps track of claims made on layers by mapping [LayerId]s to a collection
/// of [Claim]s made on that layer.
/// This is a wrapper type around a [HashMap] with a more convenient
/// interface for inserting and querying claims.
pub struct ClaimTracker<F: Field> {
    /// Maps layer IDs to vectors of claims.
    claim_map: HashMap<LayerId, Vec<Claim<F>>>,
}

impl<F: Field> ClaimTracker<F> {
    /// Generate an empty [ClaimTracker].
    pub fn new() -> Self {
        Self {
            claim_map: HashMap::<LayerId, Vec<Claim<F>>>::new(),
        }
    }

    /// Generate an empty [ClaimTracker] with a given initial capacity for
    /// better performance.
    pub fn new_with_capacity(capacity: usize) -> Self {
        Self {
            claim_map: HashMap::<LayerId, Vec<Claim<F>>>::with_capacity(capacity),
        }
    }

    /// Inserts `claim` into the list of claims made *on* the layer with ID
    /// `layer_id`.
    pub fn insert(&mut self, layer_id: LayerId, claim: Claim<F>) {
        debug_assert_eq!(claim.get_to_layer_id(), layer_id);

        if let Some(claims) = self.claim_map.get_mut(&layer_id) {
            claims.push(claim);
        } else {
            self.claim_map.insert(layer_id, vec![claim]);
        }
    }

    /// Returns a reference to a vector of all the claims made so far on layer
    /// with ID `layer_id`, if any. If no claims were ever made to `layer_id`,
    /// it returns `None`.
    pub fn get(&self, layer_id: LayerId) -> Option<&Vec<Claim<F>>> {
        self.claim_map.get(&layer_id)
    }

    /// Removes all claims made on layer with ID `layer_id` and returns them.
    pub fn remove(&mut self, layer_id: LayerId) -> Option<Vec<Claim<F>>> {
        self.claim_map.remove(&layer_id)
    }

    /// Returns whether the claim tracker is empty.
    pub fn is_empty(&self) -> bool {
        self.claim_map.is_empty()
    }
}

/// Clippy suggested this.
/// See: <https://rust-lang.github.io/rust-clippy/master/index.html#new_without_default>.
impl<F: Field> Default for ClaimTracker<F> {
    fn default() -> Self {
        ClaimTracker::new()
    }
}
