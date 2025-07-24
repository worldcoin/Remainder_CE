//! Standardized expression representation for oracle query (GKR verifier) +
//! determining necessary proof-of-products (Hyrax prover + verifier)
use remainder_shared_types::Field;

use super::LayerId;
use crate::mle::dense::DenseMle;
use crate::mle::mle_description::MleDescription;
use crate::mle::Mle;

/// A struct that closely resembles the expression tree, but stores the commitment
/// to MLE evaluations and their products
pub enum PostSumcheckLayerTree<F: Field, T> {
    /// fully bounded Mle
    Mle {
        /// the id of the layer upon which this is a claim
        layer_id: LayerId,
        /// the evaluation point
        point: Vec<F>,
        /// the value (C::Scalar), commitment to the value (C), or CommittedScalar
        value: T,
    },
    /// constant
    Constant {
        /// a circuit constant
        coefficient: F,
    },
    /// addition
    Add {
        /// left child
        left: Box<PostSumcheckLayerTree<F, T>>,
        /// right child
        right: Box<PostSumcheckLayerTree<F, T>>,
        /// depends on the circuit structure, the prover may or may not commit to an add value
        value: Option<T>,
    },
    /// multiplication
    Mult {
        /// left child
        left: Box<PostSumcheckLayerTree<F, T>>,
        /// right child
        right: Box<PostSumcheckLayerTree<F, T>>,
        /// mult values are only committed when the children do not involve a constant
        value: Option<T>,
    }
}

// CONSTRUCTORS
impl<F: Field, T> PostSumcheckLayerTree<F, T> {
    /// Creates a new node from a constant
    pub fn constant(constant: F) -> Self {
        Self::Constant { coefficient: constant }
    }
}

impl<F: Field> PostSumcheckLayerTree<F, Option<F>> {
    /// Creates a new node from an [MleDescription]
    pub fn mle(mle: &MleDescription<F>, bindings: &[F]) -> Self {
        Self::Mle {
            layer_id: mle.layer_id(),
            point: mle.get_claim_point(bindings),
            value: None,
        }
    }

    /// Creates a multiplication node from two nodes
    pub fn mult(lhs: Self, rhs: Self) -> Self {
        Self::Mult { 
            left: Box::new(lhs), 
            right: Box::new(rhs), 
            value: None, 
        }
    }

    /// Creates an addition node from two nodes
    pub fn add(lhs: Self, rhs: Self) -> Self {
        Self::Add { 
            left: Box::new(lhs), 
            right: Box::new(rhs), 
            value: None, 
        }
    }
}

impl<F: Field> PostSumcheckLayerTree<F, F> {
    // obtain the evaluation or coefficient
    fn get_value(&self) -> F {
        match self {
            PostSumcheckLayerTree::Mle{layer_id: _, point: _, value } => *value,
            PostSumcheckLayerTree::Constant{coefficient} => *coefficient,
            PostSumcheckLayerTree::Add{left, right, value} => {
                if value.is_some() {
                    value.unwrap()
                } else {
                    left.get_value() + right.get_value()
                }
            }
            PostSumcheckLayerTree::Mult{left, right, value} => {
                if value.is_some() {
                    value.unwrap()
                } else {
                    left.get_value() * right.get_value()
                }
            },
        }
    }

    /// Creates a new node from a fully bound MleRef.
    /// Panics if it is not fully bound.
    pub fn mle(mle: &DenseMle<F>) -> Self {
        // ensure all MLEs are fully bound
        assert!(mle.is_fully_bounded());
        Self::Mle {
            layer_id: mle.layer_id(),
            point: mle.get_bound_point(),
            value: mle.value(),
        }
    }

    /// Creates a multiplication node from two nodes
    pub fn mult(lhs: Self, rhs: Self) -> Self {
        let value = lhs.get_value() * rhs.get_value();
        Self::Mult { 
            left: Box::new(lhs), 
            right: Box::new(rhs),
            // some intermediate multiplication values do not need to be recorded
            // but we write down everything and remove later in [remove_add_values]
            value: Some(value), 
        }
    }

    /// Creates an addition node from two nodes
    pub fn add(lhs: Self, rhs: Self) -> Self {
        let value = lhs.get_value() + rhs.get_value();
        Self::Add { 
            left: Box::new(lhs), 
            right: Box::new(rhs),
            // some intermediate addition values do not need to be recorded
            // but we write down everything and remove later in [remove_add_values]
            value: Some(value),
        }
    }
}

// OPTIMIZATIONS
impl<F: Field, T: Clone> PostSumcheckLayerTree<F, T> {
    /// iterate through the entire tree, remove values of the nodes if
    /// 1. it is an addition node, AND
    /// 2. it is not a child of a multiplication node
    /// `parent_requires_value` is FALSE at root
    pub fn remove_add_values(&mut self, parent_requires_value: bool) {
        match self {
            Self::Mle { layer_id: _, point: _, value: _ } => {},
            Self::Constant { coefficient: _ } => {},
            Self::Mult { left, right, value } => {
                let left_child_is_constant = if let Self::Constant {..} = **left { true } else { false };
                let right_child_is_constant = if let Self::Constant {..} = **right { true } else { false };
                // Only requires value if neither child is constant
                let requires_value = !left_child_is_constant && !right_child_is_constant;
                left.remove_add_values(requires_value);
                right.remove_add_values(requires_value);
                if !parent_requires_value && !requires_value {
                    *value = None;
                }
            }
            Self::Add { left, right, value } => {
                left.remove_add_values(false);
                right.remove_add_values(false);
                // if the parent do not require the value of an addition node, remove the value
                if !parent_requires_value {
                    *value = None;
                }
            }
        }
    }
}

// EVALUATIONS
impl<F: Field, T: Clone> PostSumcheckLayerTree<F, T> {
    /// Return the resulting value of the tree.
    /// The value is stored at the root
    pub fn get_result(&self) -> Option<T> {
        match self {
            Self::Constant { coefficient: _ } => None,
            Self::Mle { layer_id: _, point: _, value } => Some(value.clone()),
            Self::Add { left: _, right: _, value } => value.clone(),
            Self::Mult { left: _, right: _, value } => value.clone(),
        }
    }

    /// Returns a vector of the values in post-order
    pub fn get_values(&self) -> Vec<T> {
        match self {
            Self::Constant { coefficient: _ } => Vec::new(),
            Self::Mle { layer_id: _, point: _, value } => vec![value.clone()],
            Self::Add { left, right, value } => {
                let left_values = left.get_values();
                let right_values = right.get_values();
                // include value if necessary
                left_values.into_iter().chain(right_values).chain(
                    if let Some(val) = value { vec![val.clone()] }
                    else { Vec::new() }
                ).collect()
            }
            Self::Mult { left, right, value } => {
                let left_values = left.get_values();
                let right_values = right.get_values();
                // include value if necessary
                left_values.into_iter().chain(right_values).chain(
                    if let Some(val) = value { vec![val.clone()] }
                    else { Vec::new() }
                ).collect()
            }
        }
    }

    /// Return a vector of triples (x, y, z) where z=x*y in post-order
    pub fn get_product_triples(&self) -> Vec<(T, T, T)> {
        match self {
            Self::Constant { coefficient: _ } => Vec::new(),
            Self::Mle { layer_id: _, point: _, value: _ } => Vec::new(),
            Self::Add { left, right, value: _ } => {
                let left_triples = left.get_product_triples();
                let right_triples = right.get_product_triples();
                left_triples.into_iter().chain(right_triples).collect()
            }
            Self::Mult { left, right, value } => {
                let left_triples = left.get_product_triples();
                let right_triples = right.get_product_triples();
                // only include the product if neither child is constant
                left_triples.into_iter().chain(right_triples).chain(
                    match (left.get_result(), right.get_result(), value) {
                        (Some(x), Some(y), Some(z)) => vec![(x, y, z.clone())],
                        _ => Vec::new()
                    }
                ).collect()
            }
        }
    }
}