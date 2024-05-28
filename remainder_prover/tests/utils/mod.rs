use itertools::Itertools;
use rand::Rng;
use remainder::mle::Mle;
use remainder::{
    builders::layer_builder::LayerBuilder,
    expression::{generic_expr::Expression, prover_expr::ProverExpr},
    layer::LayerId,
    mle::{dense::DenseMle, MleIndex},
};
use remainder_shared_types::{FieldExt, Fr};

/// Returns an MLE with all Fr::one() for testing according to the number of variables.
pub fn get_dummy_one_mle(num_vars: usize) -> DenseMle<Fr> {
    let mle_vec = (0..(1 << num_vars)).map(|_| Fr::one()).collect_vec();
    DenseMle::new_from_raw(mle_vec, LayerId::Input(0))
}

/// Returns an MLE with random elements generated from u64 for testing according to the number of variables.
pub fn get_dummy_random_mle(num_vars: usize, rng: &mut impl Rng) -> DenseMle<Fr> {
    let mle_vec = (0..(1 << num_vars))
        .map(|_| Fr::from(rng.gen::<u64>()))
        .collect_vec();
    DenseMle::new_from_raw(mle_vec, LayerId::Input(0))
}

/// Returns a vector of MLEs for dataparallel testing according to the number of variables and
/// number of dataparallel bits.
pub fn get_dummy_random_mle_vec(
    num_vars: usize,
    num_dataparallel_bits: usize,
    rng: &mut impl Rng,
) -> Vec<DenseMle<Fr>> {
    (0..(1 << num_dataparallel_bits))
        .map(|_| {
            let mle_vec = (0..(1 << num_vars))
                .map(|_| Fr::from(rng.gen::<u64>()))
                .collect_vec();
            DenseMle::new_from_raw(mle_vec, LayerId::Input(0))
        })
        .collect_vec()
}

/// A builder which returns an expression with three nested selectors:
/// - innermost_selector: sel(`inner_inner_sel_mle`, `inner_inner_sel_mle * inner_inner_sel_mle`)
/// - inner_selector: sel(`innermost_selector`, `inner_sel_mle`)
/// - overall_expression: sel(`inner_selector`, `outer_sel_mle`).
///
/// ## Arguments
/// * `inner_inner_sel_mle` - An MLE with arbitrary bookkeeping table values.
/// * `inner_sel_mle` - An MLE with arbitrary bookkeeping table values, but double
/// the size of `inner_inner_sel_mle`
/// * `outer_sel_mle` - An MLE with arbitrary bookkeeping table values, but double
/// the size of `inner_sel_mle`
pub struct TripleNestedSelectorBuilder<F: FieldExt> {
    inner_inner_sel_mle: DenseMle<F>,
    inner_sel_mle: DenseMle<F>,
    outer_sel_mle: DenseMle<F>,
}
impl<F: FieldExt> LayerBuilder<F> for TripleNestedSelectorBuilder<F> {
    type Successor = DenseMle<F>;

    fn build_expression(&self) -> Expression<F, ProverExpr> {
        let inner_inner_sel = Expression::<F, ProverExpr>::products(vec![
            self.inner_inner_sel_mle.clone().clone(),
            self.inner_inner_sel_mle.clone().clone(),
        ])
        .concat_expr(Expression::<F, ProverExpr>::mle(
            self.inner_inner_sel_mle.clone().clone(),
        ));
        let inner_sel = Expression::<F, ProverExpr>::mle(self.inner_sel_mle.clone().clone())
            .concat_expr(inner_inner_sel);
        let outer_sel = Expression::<F, ProverExpr>::mle(self.outer_sel_mle.clone().clone())
            .concat_expr(inner_sel);
        outer_sel
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        let inner_inner_sel_bt = self
            .inner_inner_sel_mle
            .current_mle
            .get_evals_vector()
            .iter()
            .flat_map(|elem| vec![*elem, *elem * elem]);

        let inner_sel_bt = inner_inner_sel_bt
            .zip(self.inner_sel_mle.current_mle.get_evals_vector().iter())
            .flat_map(|(elem_1, elem_2)| vec![elem_1, *elem_2]);

        let final_bt = inner_sel_bt
            .zip(self.outer_sel_mle.current_mle.get_evals_vector().iter())
            .flat_map(|(elem_1, elem_2)| vec![elem_1, *elem_2])
            .collect_vec();

        let mut out = DenseMle::new_from_raw(final_bt, id);
        if let Some(prefix_bits) = prefix_bits {
            out.add_prefix_bits(prefix_bits);
        }
        out
    }
}
impl<F: FieldExt> TripleNestedSelectorBuilder<F> {
    pub fn new(
        inner_inner_sel_mle: DenseMle<F>,
        inner_sel_mle: DenseMle<F>,
        outer_sel_mle: DenseMle<F>,
    ) -> Self {
        Self {
            inner_inner_sel_mle,
            inner_sel_mle,
            outer_sel_mle,
        }
    }
}

/// A builder which returns the following expression:
/// - `mle_1` * `mle_2` + (10 * `mle_1`)
///
/// ## Arguments
/// * `mle_1` - An MLE with arbitrary bookkeeping table values.
/// * `mle_2` - An MLE with arbitrary bookkeeping table values; same size as `mle_1`.
pub struct ProductScaledBuilder<F: FieldExt> {
    mle_1: DenseMle<F>,
    mle_2: DenseMle<F>,
}
impl<F: FieldExt> LayerBuilder<F> for ProductScaledBuilder<F> {
    type Successor = DenseMle<F>;

    fn build_expression(&self) -> Expression<F, ProverExpr> {
        let prod_expr =
            Expression::<F, ProverExpr>::products(vec![self.mle_1.clone(), self.mle_2.clone()]);
        let scaled_expr = self.mle_1.clone().expression() * F::from(10_u64);
        prod_expr + scaled_expr
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        let prod_bt = self
            .mle_1
            .current_mle
            .get_evals_vector()
            .iter()
            .zip(self.mle_2.current_mle.get_evals_vector().iter())
            .map(|(elem_1, elem_2)| *elem_1 * elem_2);

        let scaled_bt = self
            .mle_1
            .current_mle
            .get_evals_vector()
            .iter()
            .map(|elem| F::from(10_u64) * elem);

        let final_bt = prod_bt
            .zip(scaled_bt)
            .map(|(elem_1, elem_2)| elem_1 + elem_2)
            .collect_vec();

        let mut out = DenseMle::new_from_raw(final_bt, id);
        if let Some(prefix_bits) = prefix_bits {
            out.add_prefix_bits(prefix_bits);
        }
        out
    }
}
impl<F: FieldExt> ProductScaledBuilder<F> {
    pub fn new(mle_1: DenseMle<F>, mle_2: DenseMle<F>) -> Self {
        Self { mle_1, mle_2 }
    }
}

/// A builder which returns the following expression:
/// - `mle_1` * `mle_2` + (`mle_1` + `mle_2`)
///
/// ## Arguments
/// * `mle_1` - An MLE with arbitrary bookkeeping table values.
/// * `mle_2` - An MLE with arbitrary bookkeeping table values; same size as `mle_1`.
pub struct ProductSumBuilder<F: FieldExt> {
    mle_1: DenseMle<F>,
    mle_2: DenseMle<F>,
}
impl<F: FieldExt> LayerBuilder<F> for ProductSumBuilder<F> {
    type Successor = DenseMle<F>;

    fn build_expression(&self) -> Expression<F, ProverExpr> {
        let prod_expr =
            Expression::<F, ProverExpr>::products(vec![self.mle_1.clone(), self.mle_2.clone()]);
        let sum_expr = self.mle_1.clone().expression() + self.mle_2.clone().expression();
        prod_expr + sum_expr
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        let prod_bt = self
            .mle_1
            .current_mle
            .get_evals_vector()
            .iter()
            .zip(self.mle_2.current_mle.get_evals_vector().iter())
            .map(|(elem_1, elem_2)| *elem_1 * elem_2);

        let sum_bt = self
            .mle_1
            .current_mle
            .get_evals_vector()
            .iter()
            .zip(self.mle_2.current_mle.get_evals_vector().iter())
            .map(|(elem_1, elem_2)| *elem_1 + elem_2);

        let final_bt = prod_bt
            .zip(sum_bt)
            .map(|(elem_1, elem_2)| elem_1 + elem_2)
            .collect_vec();

        let mut out = DenseMle::new_from_raw(final_bt, id);
        if let Some(prefix_bits) = prefix_bits {
            out.add_prefix_bits(prefix_bits);
        }
        out
    }
}
impl<F: FieldExt> ProductSumBuilder<F> {
    pub fn new(mle_1: DenseMle<F>, mle_2: DenseMle<F>) -> Self {
        Self { mle_1, mle_2 }
    }
}

/// A builder which returns the following expression:
/// - `mle_1` + 10 + (`mle_2` * 10)
///
/// ## Arguments
/// * `mle_1` - An MLE with arbitrary bookkeeping table values.
/// * `mle_2` - An MLE with arbitrary bookkeeping table values; same size as `mle_1`.
pub struct ConstantScaledSumBuilder<F: FieldExt> {
    mle_1: DenseMle<F>,
    mle_2: DenseMle<F>,
}
impl<F: FieldExt> LayerBuilder<F> for ConstantScaledSumBuilder<F> {
    type Successor = DenseMle<F>;

    fn build_expression(&self) -> Expression<F, ProverExpr> {
        let constant_expr = Expression::<F, ProverExpr>::mle(self.mle_1.clone())
            + Expression::<F, ProverExpr>::constant(F::from(10_u64));
        let scaled_expr = self.mle_2.clone().expression() * F::from(10_u64);
        constant_expr + scaled_expr
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        let constant_bt = self
            .mle_1
            .current_mle
            .get_evals_vector()
            .iter()
            .map(|elem| *elem + F::from(10_u64));

        let scaled_bt = self
            .mle_2
            .current_mle
            .get_evals_vector()
            .iter()
            .map(|elem| F::from(10_u64) * elem);

        let final_bt = constant_bt
            .zip(scaled_bt)
            .map(|(elem_1, elem_2)| elem_1 + elem_2)
            .collect_vec();

        let mut out = DenseMle::new_from_raw(final_bt, id);
        if let Some(prefix_bits) = prefix_bits {
            out.add_prefix_bits(prefix_bits);
        }
        out
    }
}
impl<F: FieldExt> ConstantScaledSumBuilder<F> {
    pub fn new(mle_1: DenseMle<F>, mle_2: DenseMle<F>) -> Self {
        Self { mle_1, mle_2 }
    }
}
