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
        /// mult values are always committed
        value: T,
    }
}

impl<F: Field> PostSumcheckLayerTree<F, F> {
    /// Evaluate the PostSumcheckLayer to a single scalar
    pub fn evaluate_scalar(&self) -> F {
        match self {
            PostSumcheckLayerTree::Mle{layer_id: _, point: _, value } => value.clone(),
            PostSumcheckLayerTree::Constant{coefficient} => coefficient.clone(),
            PostSumcheckLayerTree::Add{left, right, value} => {
                if value.is_some() {
                    value.unwrap()
                } else {
                    left.evaluate_scalar() + right.evaluate_scalar()
                }
            }
            PostSumcheckLayerTree::Mult{left: _, right: _, value} => value.clone(),
        }
    }
}

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
            Self::Constant { coefficient } => *coefficient,
            Self::Mle { layer_id: _, point: _, value } => *value,
            Self::Add { left: _, right: _, value } => value.unwrap(),
            Self::Mult { left: _, right: _, value } => *value,
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
            value, 
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

    /// iterate through the entire tree, remove values of the nodes if
    /// 1. it is an addition node, AND
    /// 2. it is not a child of a multiplication node
    /// `mult_child` is FALSE at root
    pub fn remove_add_values(&mut self, mult_child: bool) {
        match self {
            Self::Mle { layer_id: _, point: _, value: _ } => {},
            Self::Constant { coefficient: _ } => {},
            Self::Mult { left, right, value: _ } => {
                left.remove_add_values(true);
                right.remove_add_values(true);
            }
            Self::Add { left, right, value } => {
                left.remove_add_values(false);
                right.remove_add_values(false);
                // The only time we need to change something
                if !mult_child {
                    *value = None;
                }
            }
        }
    }
}
