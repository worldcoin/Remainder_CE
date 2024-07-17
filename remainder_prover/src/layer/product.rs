use crate::{
    claims::wlx_eval::ClaimMle,
    mle::{dense::DenseMle, mle_enum::MleEnum, Mle},
};
use rand::Rng;
use remainder_shared_types::halo2curves::ff::Field;
use remainder_shared_types::{
    curves::PrimeOrderCurve,
    ec::CurveExt,
    layer::LayerId,
    pedersen::{CommittedScalar, PedersenCommitter},
    FieldExt,
};

/// Represents a normal form for a layer expression in which the layer is represented as a linear
/// combination of products of other layer MLEs, the coefficients of which are public.
#[derive(Debug)]
pub struct PostSumcheckLayer<F: FieldExt, T>(pub Vec<Product<F, T>>);

// FIXME can we implement all of these evaluate functions with a single function using trait bounds?
// needs to have a zero() method (Default?).  need mulassign?
impl<F: FieldExt> PostSumcheckLayer<F, F> {
    /// Evaluate the PostSumcheckLayer to a single scalar
    pub fn evaluate_scalar(&self) -> F {
        self.0.iter().fold(F::ZERO, |acc, product| {
            acc + product.get_result() * product.coefficient
        })
    }
}

impl<C: PrimeOrderCurve> PostSumcheckLayer<C::Scalar, CommittedScalar<C>> {
    /// Evaluate the PostSumcheckLayer to a single CommittedScalar.
    pub fn evaluate_committed_scalar(&self) -> CommittedScalar<C> {
        self.0
            .iter()
            .fold(CommittedScalar::zero(), |acc, (product)| {
                acc + product.get_result() * product.coefficient
            })
    }

    /// Turn all the CommittedScalars into commitments i.e. Cs.
    pub fn as_commitments(&self) -> PostSumcheckLayer<C::Scalar, C> {
        PostSumcheckLayer(
            self.0
                .iter()
                .map(|product| {
                    let commitments = product
                        .intermediates
                        .iter()
                        .map(|pp| match pp {
                            Intermediate::Atom {
                                layer_id,
                                point,
                                value,
                                mle_enum: _,
                            } => Intermediate::Atom {
                                layer_id: *layer_id,
                                point: point.clone(),
                                value: value.commitment.clone(),
                                mle_enum: None,
                            },
                            Intermediate::Composite { value } => Intermediate::Composite {
                                value: value.commitment.clone(),
                            },
                        })
                        .collect();
                    Product {
                        coefficient: product.coefficient,
                        intermediates: commitments,
                    }
                })
                .collect(),
        )
    }
}

impl<C: PrimeOrderCurve> PostSumcheckLayer<C::Scalar, C> {
    /// Evaluate the PostSumcheckLayer to a single scalar
    pub fn evaluate(&self) -> C {
        self.0.iter().fold(C::zero(), |acc, product| {
            acc + product.get_result() * product.coefficient
        })
    }
}

/// Represents a fully bound product of MLEs, or a single MLE (which we consider a simple product).
/// Data structure for extracting the values to be commited to, their "mxi" values (i.e. their
/// public coefficients), and also the claims made on other layers.
#[derive(Debug)]
pub struct Product<F: FieldExt, T> {
    /// the evaluated MLEs that are being multiplied and their intermediates
    /// in the order a, b, ab, c, abc, d, abcd, ...
    intermediates: Vec<Intermediate<F, T>>,
    /// the (public) coefficient i.e. the "mxi"
    coefficient: F,
}

impl<F: FieldExt> Product<F, F> {
    /// Creates a new Product from a vector of fully bound MleRefs.
    /// Panics if any are not fully bound.
    pub fn new(mle_refs: &[DenseMle<F>], coefficient: F) -> Self {
        // ensure all MLEs are fully bound
        assert!(mle_refs
            .iter()
            .all(|mle_ref| mle_ref.bookkeeping_table().len() == 1));
        if mle_refs.len() == 0 {
            return Product {
                intermediates: vec![Intermediate::Composite { value: F::ONE }],
                coefficient,
            };
        }
        let mut intermediates = vec![Self::build_atom(&mle_refs[0])];
        let _ = mle_refs
            .iter()
            .skip(1)
            .fold(mle_refs[0].bookkeeping_table()[0], |acc, mle_ref| {
                let prod_val = acc * mle_ref.bookkeeping_table()[0];
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
            point: mle_ref.get_claim_point(),
            mle_enum: Some(MleEnum::Dense(mle_ref.clone())),
            value: mle_ref.bookkeeping_table()[0],
        }
    }
}

#[derive(Clone, Debug)]
/// Represents either an atomic factor of a product (i.e. an evaluation of an MLE), or the result of
/// an intermediate product of atoms.
enum Intermediate<F: FieldExt, T> {
    Atom {
        /// the id of the layer upon which this is a claim
        layer_id: LayerId,
        /// the evaluation point
        point: Vec<F>,
        /// the value (C::Scalar), commitment to the value (C), or CommittedScalar
        value: T,
        /// the mle enum associated with the point and the value above,
        /// populated for the prover but None for the verifier.
        mle_enum: Option<MleEnum<F>>,
    },
    Composite {
        /// the value, commitment to the value, or CommittedScalar
        value: T,
    },
}

/// Returns a CommittedScalar version of the PostSumcheckLayer.
pub fn commit_to_post_sumcheck_layer<C: PrimeOrderCurve>(
    post_sumcheck_layer: &PostSumcheckLayer<C::Scalar, C::Scalar>,
    committer: &PedersenCommitter<C>,
    mut blinding_rng: &mut impl Rng,
) -> PostSumcheckLayer<C::Scalar, CommittedScalar<C>> {
    PostSumcheckLayer(
        post_sumcheck_layer
            .0
            .iter()
            .map(|product| commit_to_product(product, committer, &mut blinding_rng))
            .collect(),
    )
}

// Helper for commit_to_post_sumcheck_layer.
// Returns a CommittedScalar version of the Product.
fn commit_to_product<C: PrimeOrderCurve>(
    product: &Product<C::Scalar, C::Scalar>,
    committer: &PedersenCommitter<C>,
    mut blinding_rng: &mut impl Rng,
) -> Product<C::Scalar, CommittedScalar<C>> {
    let committed_scalars = product
        .intermediates
        .iter()
        .map(|pp| match pp {
            Intermediate::Atom {
                layer_id,
                point,
                value,
                mle_enum,
            } => Intermediate::Atom {
                layer_id: *layer_id,
                point: point.clone(),
                value: committer.committed_scalar(value, &C::Scalar::random(&mut blinding_rng)),
                mle_enum: mle_enum.clone(),
            },
            Intermediate::Composite { value } => Intermediate::Composite {
                value: committer.committed_scalar(value, &C::Scalar::random(&mut blinding_rng)),
            },
        })
        .collect();
    Product {
        intermediates: committed_scalars,
        coefficient: product.coefficient,
    }
}

impl<F: FieldExt, T: Copy> PostSumcheckLayer<F, T> {
    /// Returns a vector of the values of the intermediates (in an order compatible with [set_values]).
    pub fn get_values(&self) -> Vec<T> {
        self.0
            .iter()
            .map(|product| {
                product
                    .intermediates
                    .iter()
                    .map(|pp| match pp {
                        Intermediate::Atom { value, .. } => value.clone(),
                        Intermediate::Composite { value } => value.clone(),
                    })
                    .collect::<Vec<T>>()
            })
            .flatten()
            .collect()
    }

    /// Return a vector of all the coefficients of the products.
    pub fn get_coefficients(&self) -> Vec<F> {
        self.0.iter().map(|product| product.coefficient).collect()
    }
}

/// Set the values of the PostSumcheckLayer to the given values, panicking if the lengths do not match,
/// returning a new instance. Counterpart to [get_values].
pub fn new_with_values<F: FieldExt, S, T: Clone>(
    post_sumcheck_layer: &PostSumcheckLayer<F, S>,
    values: &Vec<T>,
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
fn new_with_values_single<F: FieldExt, S, T>(
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
                    layer_id,
                    point,
                    mle_enum,
                    ..
                } => Intermediate::Atom {
                    layer_id: *layer_id,
                    point: point.clone(),
                    value,
                    mle_enum: mle_enum.clone(),
                },
                Intermediate::Composite { .. } => Intermediate::Composite { value },
            })
            .collect(),
    }
}

impl<F: FieldExt, T: Clone> Product<F, T> {
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

    /// Return the claims made on other layers by the atomic factors of this product.
    pub fn get_claims(&self) -> Vec<HyraxClaim<F, T>> {
        self.intermediates
            .iter()
            .filter_map(|pp| match pp {
                Intermediate::Atom {
                    layer_id,
                    point,
                    value,
                    mle_enum,
                } => Some(HyraxClaim {
                    to_layer_id: *layer_id,
                    mle_enum: mle_enum.clone(),
                    point: point.clone(),
                    evaluation: value.clone(),
                }),
                Intermediate::Composite { .. } => None,
            })
            .collect()
    }
}

/// Implementation of HyraxClaim as used by the prover
impl<C: PrimeOrderCurve> HyraxClaim<C::Scalar, CommittedScalar<C>> {
    /// Convert to a raw [Claim] for claim aggregation
    pub fn to_claim(&self) -> ClaimMle<C::Scalar> {
        let mut claim = ClaimMle::new_raw(self.point.clone(), self.evaluation.value);
        claim.to_layer_id = Some(self.to_layer_id.clone());
        claim.mle_ref = self.mle_enum.clone();
        claim
    }

    /// Convert to a HyraxClaim<C::Scalar, C>
    pub fn to_claim_commitment(&self) -> HyraxClaim<C::Scalar, C> {
        HyraxClaim {
            point: self.point.clone(),
            to_layer_id: self.to_layer_id,
            mle_enum: self.mle_enum.clone(),
            evaluation: self.evaluation.commitment,
        }
    }
}

/// Represents a claim made on a layer by an atomic factor of a product.
/// T could be:
///     C::Scalar (if used by the prover), or
///     CommittedScalar<C> (if used by the prover)
///     to interface with claim aggregation code in remainder
///     C (this is the verifier's view, i.e. just the commitment)
#[derive(Clone, Debug)]
pub struct HyraxClaim<F: FieldExt, T> {
    /// Id of the layer upon which the claim is made
    pub to_layer_id: LayerId,
    /// The evaluation point
    pub point: Vec<F>,
    /// The original mle_enum (or None)
    pub mle_enum: Option<MleEnum<F>>,
    /// The value of the claim
    pub evaluation: T,
}
