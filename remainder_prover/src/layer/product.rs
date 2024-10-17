//! Standardized expression representation for oracle query (GKR verifier) +
//! determining necessary proof-of-products (Hyrax prover + verifier)

use remainder_shared_types::Field;

use super::LayerId;
use crate::expression::circuit_expr::MleDescription;
use crate::expression::verifier_expr::VerifierMle;
use crate::mle::dense::DenseMle;
use crate::mle::Mle;

/// Represents a normal form for a layer expression in which the layer is represented as a linear
/// combination of products of other layer MLEs, the coefficients of which are public.
#[derive(Debug)]
pub struct PostSumcheckLayer<F: Field, T>(pub Vec<Product<F, T>>);

// FIXME can we implement all of these evaluate functions with a single function using trait bounds?
// needs to have a zero() method (Default?).  need mulassign?
impl<F: Field> PostSumcheckLayer<F, F> {
    /// Evaluate the PostSumcheckLayer to a single scalar
    pub fn evaluate_scalar(&self) -> F {
        self.0.iter().fold(F::ZERO, |acc, product| {
            acc + product.get_result() * product.coefficient
        })
    }
}

/// Represents a fully bound product of MLEs, or a single MLE (which we consider a simple product).
/// Data structure for extracting the values to be commited to, their "mxi" values (i.e. their
/// public coefficients), and also the claims made on other layers.
#[derive(Debug)]
pub struct Product<F: Field, T> {
    /// the evaluated MLEs that are being multiplied and their intermediates
    /// in the order a, b, ab, c, abc, d, abcd, ...
    pub intermediates: Vec<Intermediate<F, T>>,
    /// the (public) coefficient i.e. the "mxi"
    pub coefficient: F,
}

impl<F: Field> Product<F, Option<F>> {
    /// Creates a new Product from a vector of [MleDescriptions].
    pub fn new(mles: &[MleDescription<F>], coefficient: F, bindings: &[F]) -> Self {
        if mles.is_empty() {
            return Product {
                intermediates: vec![Intermediate::Composite {
                    value: Some(F::ONE),
                }],
                coefficient,
            };
        }
        let mut intermediates = vec![Self::build_atom(&mles[0], bindings)];
        mles.iter().skip(1).for_each(|mle_ref| {
            intermediates.push(Self::build_atom(mle_ref, bindings));
            intermediates.push(Intermediate::Composite { value: None });
        });
        Product {
            intermediates,
            coefficient,
        }
    }

    // Helper function for new
    fn build_atom(mle: &MleDescription<F>, bindings: &[F]) -> Intermediate<F, Option<F>> {
        Intermediate::Atom {
            layer_id: mle.layer_id(),
            point: mle.get_claim_point(bindings),
            value: None,
        }
    }
}

impl<F: Field> Product<F, F> {
    /// Creates a new Product from a vector of fully bound MleRefs.
    /// Panics if any are not fully bound.
    pub fn new(mle_refs: &[DenseMle<F>], coefficient: F) -> Self {
        // ensure all MLEs are fully bound
        assert!(mle_refs.iter().all(|mle_ref| mle_ref.len() == 1));
        if mle_refs.is_empty() {
            return Product {
                intermediates: vec![Intermediate::Composite { value: F::ONE }],
                coefficient,
            };
        }
        let mut intermediates = vec![Self::build_atom(&mle_refs[0])];
        let _ = mle_refs
            .iter()
            .skip(1)
            .fold(mle_refs[0].first(), |acc, mle_ref| {
                let prod_val = acc * mle_ref.first();
                intermediates.push(Self::build_atom(mle_ref));
                intermediates.push(Intermediate::Composite { value: prod_val });
                prod_val
            });
        Product {
            intermediates,
            coefficient,
        }
    }

    // Helper function for new
    fn build_atom(mle_ref: &DenseMle<F>) -> Intermediate<F, F> {
        Intermediate::Atom {
            layer_id: mle_ref.layer_id,
            point: mle_ref.get_bound_point(),
            value: mle_ref.first(),
        }
    }

    /// Creates a new Product from a vector of fully bound Mles, which are represented as a [VerifierMle]
    pub fn new_from_verifier_mle(verifier_mles: &[VerifierMle<F>], coefficient: F) -> Self {
        if verifier_mles.is_empty() {
            return Product {
                intermediates: vec![Intermediate::Composite { value: F::ONE }],
                coefficient,
            };
        }
        let mut intermediates = vec![Self::build_atom_from_verifier_mle(&verifier_mles[0])];
        let _ = verifier_mles
            .iter()
            .skip(1)
            .fold(verifier_mles[0].value(), |acc, verifier_mle| {
                let prod_val = acc * verifier_mle.value();
                intermediates.push(Self::build_atom_from_verifier_mle(verifier_mle));
                intermediates.push(Intermediate::Composite { value: prod_val });
                prod_val
            });
        Product {
            intermediates,
            coefficient,
        }
    }

    // Helper function for `new_from_verifier_mle`
    fn build_atom_from_verifier_mle(verifier_mle: &VerifierMle<F>) -> Intermediate<F, F> {
        Intermediate::Atom {
            layer_id: verifier_mle.layer_id(),
            point: verifier_mle.get_bound_point(),
            value: verifier_mle.value(),
        }
    }
}

#[derive(Clone, Debug)]
/// Represents either an atomic factor of a product (i.e. an evaluation of an MLE), or the result of
/// an intermediate product of atoms.
pub enum Intermediate<F: Field, T> {
    /// A struct representing a single MLE and a commitment to its evaluation.
    Atom {
        /// the id of the layer upon which this is a claim
        layer_id: LayerId,
        /// the evaluation point
        point: Vec<F>,
        /// the value (C::Scalar), commitment to the value (C), or CommittedScalar
        value: T,
    },
    /// A struct representing a commitment to the product of two MLE evaluations.
    Composite {
        /// the value, commitment to the value, or CommittedScalar
        value: T,
    },
}

impl<F: Field, T: Copy> PostSumcheckLayer<F, T> {
    /// Returns a vector of the values of the intermediates (in an order compatible with [set_values]).
    pub fn get_values(&self) -> Vec<T> {
        self.0
            .iter()
            .flat_map(|product| {
                product
                    .intermediates
                    .iter()
                    .map(|pp| match pp {
                        Intermediate::Atom { value, .. } => *value,
                        Intermediate::Composite { value } => *value,
                    })
                    .collect::<Vec<T>>()
            })
            .collect()
    }

    /// Return a vector of all the coefficients of the products.
    pub fn get_coefficients(&self) -> Vec<F> {
        self.0.iter().map(|product| product.coefficient).collect()
    }
}

/// Set the values of the PostSumcheckLayer to the given values, panicking if the lengths do not match,
/// returning a new instance. Counterpart to [get_values].
pub fn new_with_values<F: Field, S, T: Clone>(
    post_sumcheck_layer: &PostSumcheckLayer<F, S>,
    values: &[T],
) -> PostSumcheckLayer<F, T> {
    let total_len: usize = post_sumcheck_layer
        .0
        .iter()
        .map(|product| product.intermediates.len())
        .sum();
    assert_eq!(total_len, values.len());
    let mut start = 0;
    PostSumcheckLayer(
        post_sumcheck_layer
            .0
            .iter()
            .map(|product| {
                let end = start + product.intermediates.len();
                let product_values = values[start..end].to_vec();
                start = end;
                new_with_values_single(product, product_values)
            })
            .collect(),
    )
}

/// Helper for [new_with_values].
/// Set the values of the Product to the given values, panicking if the lengths do not match.
fn new_with_values_single<F: Field, S, T>(
    product: &Product<F, S>,
    values: Vec<T>,
) -> Product<F, T> {
    assert_eq!(product.intermediates.len(), values.len());
    Product {
        coefficient: product.coefficient,
        intermediates: product
            .intermediates
            .iter()
            .zip(values)
            .map(|(pp, value)| match pp {
                Intermediate::Atom {
                    layer_id, point, ..
                } => Intermediate::Atom {
                    layer_id: *layer_id,
                    point: point.clone(),
                    value,
                },
                Intermediate::Composite { .. } => Intermediate::Composite { value },
            })
            .collect(),
    }
}

impl<F: Field, T: Clone> Product<F, T> {
    /// Return the resulting value of the product.
    /// Useful to build the commitment to the oracle evaluation.
    pub fn get_result(&self) -> T {
        let last = &self.intermediates[self.intermediates.len() - 1];
        match last {
            Intermediate::Atom { value, .. } => {
                // this product had better consist of just one MLE!
                assert_eq!(self.intermediates.len(), 1);
                value.clone()
            }
            Intermediate::Composite { value, .. } => value.clone(),
        }
    }

    /// Return a vector of triples (x, y, z) where z=x*y, or None.
    pub fn get_product_triples(&self) -> Option<Vec<(T, T, T)>> {
        if self.intermediates.len() > 1 {
            assert!(self.intermediates.len() >= 3);
            let values = self
                .intermediates
                .iter()
                .map(|pp| match pp {
                    Intermediate::Atom { value, .. } => value.clone(),
                    Intermediate::Composite { value, .. } => value.clone(),
                })
                .collect::<Vec<_>>();
            Some(
                values
                    .windows(3)
                    .map(|window| {
                        if let [x, y, z] = window {
                            (x.clone(), y.clone(), z.clone())
                        } else {
                            unreachable!()
                        }
                    })
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        }
    }
}
