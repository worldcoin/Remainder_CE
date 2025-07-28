//! Standardized expression representation for oracle query (GKR verifier) +
//! determining necessary proof-of-products (Hyrax prover + verifier)
use std::fmt::{self, Debug};
use std::ops::{Add, Mul};

use remainder_shared_types::Field;

use super::LayerId;
use crate::mle::dense::DenseMle;
use crate::mle::mle_description::MleDescription;
use crate::mle::Mle;

/// A struct that closely resembles the expression tree, but stores the commitment
/// to MLE evaluations and their products
#[derive(Debug)]
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
        // the prover never committs to add values
    },
    /// multiplication
    Mult {
        /// left child
        left: Box<PostSumcheckLayerTree<F, T>>,
        /// right child
        right: Box<PostSumcheckLayerTree<F, T>>,
        /// mult values are only committed when the children do not involve a constant
        value: Option<T>,
    },
}

// pretty printing
impl<F: Field, T> fmt::Display for PostSumcheckLayerTree<F, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_with_prec(f, 0)
    }
}
impl<F: Field, T> PostSumcheckLayerTree<F, T> {
    // Define precedence levels: Add = 1, Mul = 2, Mle = 3 (highest so it never gets wrapped)
    fn fmt_with_prec(&self, f: &mut fmt::Formatter<'_>, prec: u8) -> fmt::Result {
        match self {
            PostSumcheckLayerTree::Mle { .. } => write!(f, "MLE"),
            PostSumcheckLayerTree::Constant { .. } => write!(f, "C"),
            PostSumcheckLayerTree::Add { left, right, .. } => {
                let my_prec = 1;
                let need_parens = my_prec < prec;
                if need_parens {
                    write!(f, "(")?;
                }
                left.fmt_with_prec(f, my_prec)?;
                write!(f, " + ")?;
                right.fmt_with_prec(f, my_prec)?;
                if need_parens {
                    write!(f, ")")?;
                }
                Ok(())
            }
            PostSumcheckLayerTree::Mult { left, right, .. } => {
                let my_prec = 2;
                let need_parens = my_prec < prec;
                if need_parens {
                    write!(f, "(")?;
                }
                left.fmt_with_prec(f, my_prec)?;
                write!(f, " * ")?;
                right.fmt_with_prec(f, my_prec)?;
                if need_parens {
                    write!(f, ")")?;
                }
                Ok(())
            }
        }
    }
}

// CONSTRUCTORS
impl<F: Field, T> PostSumcheckLayerTree<F, T> {
    /// Creates a new node from a constant
    pub fn constant(constant: F) -> Self {
        Self::Constant {
            coefficient: constant,
        }
    }

    // Helper function to decide whether a multiplication node requires commitment
    fn requires_mult_value(lhs: &Self, rhs: &Self) -> bool {
        let left_child_is_constant = matches!(lhs, Self::Constant { .. });
        let right_child_is_constant = matches!(rhs, Self::Constant { .. });
        // Only requires value if neither child is constant
        !left_child_is_constant && !right_child_is_constant
    }
}

impl<F: Field, T> Add for PostSumcheckLayerTree<F, T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::Add {
            left: Box::new(self),
            right: Box::new(rhs),
        }
    }
}
impl<F: Field, T> Add<F> for PostSumcheckLayerTree<F, T> {
    type Output = Self;
    fn add(self, rhs: F) -> Self {
        self + Self::Constant { coefficient: rhs }
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
}

impl<F: Field> Mul for PostSumcheckLayerTree<F, Option<F>> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let requires_value = Self::requires_mult_value(&self, &rhs);
        Self::Mult {
            left: Box::new(self),
            right: Box::new(rhs),
            value: if requires_value { Some(None) } else { None },
        }
    }
}
impl<F: Field> Mul<F> for PostSumcheckLayerTree<F, Option<F>> {
    type Output = Self;
    fn mul(self, rhs: F) -> Self {
        self * Self::Constant { coefficient: rhs }
    }
}

impl<F: Field> PostSumcheckLayerTree<F, F> {
    // obtain the evaluation or coefficient
    fn get_field_value(&self) -> F {
        match self {
            PostSumcheckLayerTree::Mle { value, .. } => *value,
            PostSumcheckLayerTree::Constant { coefficient } => *coefficient,
            PostSumcheckLayerTree::Add { left, right } => {
                left.get_field_value() + right.get_field_value()
            }
            PostSumcheckLayerTree::Mult { left, right, value } => {
                if value.is_some() {
                    value.unwrap()
                } else {
                    left.get_field_value() * right.get_field_value()
                }
            }
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
}

impl<F: Field> Mul for PostSumcheckLayerTree<F, F> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let value = self.get_field_value() * rhs.get_field_value();
        let requires_value = Self::requires_mult_value(&self, &rhs);
        Self::Mult {
            left: Box::new(self),
            right: Box::new(rhs),
            value: if requires_value { Some(value) } else { None },
        }
    }
}
impl<F: Field> Mul<F> for PostSumcheckLayerTree<F, F> {
    type Output = Self;
    fn mul(self, rhs: F) -> Self {
        self * Self::Constant { coefficient: rhs }
    }
}

/// Set the values of the PostSumcheckLayer to the given values in post-order,
/// panicking if the lengths do not match, returning a new instance.
/// Counterpart to [PostSumcheckLayerTree::get_values].
pub fn new_with_values<F: Field, S, T: Clone>(
    post_sumcheck_layer: &PostSumcheckLayerTree<F, S>,
    values: &[T],
    next_index: &mut usize, // index to the next value
) -> PostSumcheckLayerTree<F, T> {
    match post_sumcheck_layer {
        PostSumcheckLayerTree::Constant { coefficient } => PostSumcheckLayerTree::Constant {
            coefficient: *coefficient,
        },
        PostSumcheckLayerTree::Mle {
            layer_id, point, ..
        } => PostSumcheckLayerTree::Mle {
            layer_id: *layer_id,
            point: point.clone(),
            value: {
                let new_value = values[*next_index].clone();
                *next_index += 1;
                new_value
            },
        },
        PostSumcheckLayerTree::Add { left, right } => PostSumcheckLayerTree::Add {
            left: Box::new(new_with_values(left, values, next_index)),
            right: Box::new(new_with_values(right, values, next_index)),
        },
        PostSumcheckLayerTree::Mult { left, right, value } => PostSumcheckLayerTree::Mult {
            left: Box::new(new_with_values(left, values, next_index)),
            right: Box::new(new_with_values(right, values, next_index)),
            value: if value.is_some() {
                let new_value = values[*next_index].clone();
                *next_index += 1;
                Some(new_value)
            } else {
                None
            },
        },
    }
}

// EVALUATIONS
impl<F: Field, T: Clone> PostSumcheckLayerTree<F, T> {
    /// Returns a vector of the values in post-order
    /// Counterpart to [new_with_values].
    pub fn get_values(&self) -> Vec<T> {
        match self {
            Self::Constant { .. } => Vec::new(),
            Self::Mle { value, .. } => vec![value.clone()],
            Self::Add { left, right } => {
                let left_values = left.get_values();
                let right_values = right.get_values();
                // include value if necessary
                left_values.into_iter().chain(right_values).collect()
            }
            Self::Mult { left, right, value } => {
                let left_values = left.get_values();
                let right_values = right.get_values();
                // include value if necessary
                left_values
                    .into_iter()
                    .chain(right_values)
                    .chain(if let Some(val) = value {
                        vec![val.clone()]
                    } else {
                        Vec::new()
                    })
                    .collect()
            }
        }
    }
}

/// evaluation of a node, can be F (constants) or T (polynomials)
#[derive(Debug, PartialEq)]
pub enum EvalResult<F: PartialEq, T: PartialEq> {
    /// a public field value
    F(F),
    /// a private value
    T(T),
}
impl<F: Field + PartialEq, T> PostSumcheckLayerTree<F, T>
where
    T: Debug + Clone + PartialEq + Add<Output = T> + Mul<F, Output = T>,
{
    /// Commit the values
    /// Note that we cannot simply commit to every scalar evaluation,
    /// instead, depending on whether the entry is public or private,
    /// there are different types of operations
    /// Corresponds to [PostSumcheckLayerTree::get_result]
    pub fn commit<Comm>(
        tree: &PostSumcheckLayerTree<F, F>,
        comm: &mut Comm,
        one: T,
    ) -> (Self, EvalResult<F, T>)
    where
        Comm: FnMut(F) -> T,
    {
        match tree {
            PostSumcheckLayerTree::Constant { coefficient } => (
                Self::Constant {
                    coefficient: *coefficient,
                },
                EvalResult::F(*coefficient),
            ),
            PostSumcheckLayerTree::Mle {
                layer_id,
                point,
                value,
            } => {
                let t = comm(*value);
                (
                    Self::Mle {
                        layer_id: *layer_id,
                        point: point.clone(),
                        value: t.clone(),
                    },
                    EvalResult::T(t),
                )
            }
            PostSumcheckLayerTree::Add { left, right, .. } => {
                let (left_tree, left_eval) = Self::commit(left, comm, one.clone());
                let (right_tree, right_eval) = Self::commit(right, comm, one.clone());
                let computed_result = match (left_eval, right_eval) {
                    (EvalResult::T(t1), EvalResult::T(t2)) => EvalResult::T(t1 + t2),
                    (EvalResult::T(t1), EvalResult::F(f2)) => EvalResult::T(t1 + one.clone() * f2),
                    (EvalResult::F(f1), EvalResult::T(t2)) => EvalResult::T(t2 + one.clone() * f1),
                    (EvalResult::F(f1), EvalResult::F(f2)) => EvalResult::F(f1 + f2),
                };
                (
                    Self::Add {
                        left: Box::new(left_tree),
                        right: Box::new(right_tree),
                    },
                    computed_result,
                )
            }
            PostSumcheckLayerTree::Mult { left, right, value } => {
                let (left_tree, left_eval) = Self::commit(left, comm, one.clone());
                let (right_tree, right_eval) = Self::commit(right, comm, one.clone());
                let computed_result = match (left_eval, right_eval) {
                    (EvalResult::T(_), EvalResult::T(_)) => EvalResult::T(comm(value.unwrap())),
                    (EvalResult::T(t1), EvalResult::F(f2)) => EvalResult::T(t1 * f2),
                    (EvalResult::F(f1), EvalResult::T(t2)) => EvalResult::T(t2 * f1),
                    (EvalResult::F(f1), EvalResult::F(f2)) => EvalResult::F(f1 * f2),
                };
                let node_value = if let EvalResult::T(t) = &computed_result {
                    Some(t.clone())
                } else {
                    None
                };
                (
                    Self::Mult {
                        left: Box::new(left_tree),
                        right: Box::new(right_tree),
                        value: if value.is_some() { node_value } else { None },
                    },
                    computed_result,
                )
            }
        }
    }

    /// evaluate and verify any redundant values
    /// supply the committed value for ONE for constant evaluation
    pub fn get_result(&self, one: T) -> T {
        match self.get_result_helper(one.clone()) {
            EvalResult::T(t) => t,
            EvalResult::F(f) => one * f,
        }
    }

    // recursive helper
    fn get_result_helper(&self, one: T) -> EvalResult<F, T> {
        match self {
            PostSumcheckLayerTree::Mle { value, .. } => EvalResult::T(value.clone()),
            PostSumcheckLayerTree::Constant { coefficient } => EvalResult::F(*coefficient),
            PostSumcheckLayerTree::Add { left, right } => {
                match (
                    left.get_result_helper(one.clone()),
                    right.get_result_helper(one.clone()),
                ) {
                    (EvalResult::T(t1), EvalResult::T(t2)) => EvalResult::T(t1 + t2),
                    (EvalResult::T(t1), EvalResult::F(f2)) => EvalResult::T(t1 + one.clone() * f2),
                    (EvalResult::F(f1), EvalResult::T(t2)) => EvalResult::T(t2 + one.clone() * f1),
                    (EvalResult::F(f1), EvalResult::F(f2)) => EvalResult::F(f1 + f2),
                }
            }
            PostSumcheckLayerTree::Mult { left, right, value } => {
                let computed_result = match (
                    left.get_result_helper(one.clone()),
                    right.get_result_helper(one.clone()),
                ) {
                    (EvalResult::T(_), EvalResult::T(_)) => {
                        // cannot perform multiplication on T, instead query `value`
                        if let Some(val) = value {
                            EvalResult::T(val.clone())
                        } else {
                            panic!("Cannot obtain the hyrax commitment of a multiplication!")
                        }
                    }
                    (EvalResult::T(t1), EvalResult::F(f2)) => EvalResult::T(t1 * f2),
                    (EvalResult::F(f1), EvalResult::T(t2)) => EvalResult::T(t2 * f1),
                    (EvalResult::F(f1), EvalResult::F(f2)) => EvalResult::F(f1 * f2),
                };
                if let Some(val) = value {
                    assert!(computed_result == EvalResult::T(val.clone()));
                }
                computed_result
            }
        }
    }

    /// Return a vector of triples (x, y, z) where z=x*y in post-order
    pub fn get_product_triples(&self, one: T) -> Vec<(T, T, T)> {
        match self {
            Self::Constant { .. } => Vec::new(),
            Self::Mle { .. } => Vec::new(),
            Self::Add { left, right } => {
                let left_triples = left.get_product_triples(one.clone());
                let right_triples = right.get_product_triples(one.clone());
                left_triples.into_iter().chain(right_triples).collect()
            }
            Self::Mult { left, right, value } => {
                let left_triples = left.get_product_triples(one.clone());
                let right_triples = right.get_product_triples(one.clone());
                // only include the product if neither child is constant
                left_triples
                    .into_iter()
                    .chain(right_triples)
                    .chain(if let Some(z) = value {
                        let x = left.get_result(one.clone());
                        let y = right.get_result(one.clone());
                        vec![(x, y, z.clone())]
                    } else {
                        Vec::new()
                    })
                    .collect()
            }
        }
    }
}
