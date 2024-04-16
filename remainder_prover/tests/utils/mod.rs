use ark_std::test_rng;
use itertools::Itertools;
use rand::Rng;
use remainder::{
    expression::{generic_expr::Expression, prover_expr::ProverExpr},
    layer::{LayerBuilder, LayerId},
    mle::{dense::DenseMle, MleIndex},
};
use remainder_shared_types::{FieldExt, Fr};

pub(crate) fn get_dummy_one_mle(num_vars: usize) -> DenseMle<Fr, Fr> {
    let mut rng = test_rng();
    let mle_vec = (0..(1 << num_vars)).map(|_| Fr::one()).collect_vec();
    DenseMle::new_from_raw(mle_vec, LayerId::Input(0), None)
}

pub(crate) fn get_dummy_random_mle(num_vars: usize, rng: &mut impl Rng) -> DenseMle<Fr, Fr> {
    let mle_vec = (0..(1 << num_vars))
        .map(|_| Fr::from(rng.gen::<u64>()))
        .collect_vec();
    DenseMle::new_from_raw(mle_vec, LayerId::Input(0), None)
}

pub(crate) fn get_dummy_random_mle_vec(
    num_vars: usize,
    num_dataparallel_bits: usize,
    rng: &mut impl Rng,
) -> Vec<DenseMle<Fr, Fr>> {
    (0..(1 << num_dataparallel_bits))
        .map(|_| {
            let mle_vec = (0..(1 << num_vars))
                .map(|_| Fr::from(rng.gen::<u64>()))
                .collect_vec();
            DenseMle::new_from_raw(mle_vec, LayerId::Input(0), None)
        })
        .collect_vec()
}

pub(crate) struct TripleNestedSelectorBuilder<F: FieldExt> {
    inner_inner_sel_mle: DenseMle<F, F>,
    inner_sel_mle: DenseMle<F, F>,
    outer_sel_mle: DenseMle<F, F>,
}
impl<F: FieldExt> LayerBuilder<F> for TripleNestedSelectorBuilder<F> {
    type Successor = DenseMle<F, F>;

    fn build_expression(&self) -> Expression<F, ProverExpr> {
        let inner_inner_sel = Expression::products(vec![
            self.inner_inner_sel_mle.mle_ref(),
            self.inner_inner_sel_mle.mle_ref(),
        ])
        .concat_expr(Expression::mle(self.inner_inner_sel_mle.mle_ref()));
        let inner_sel = Expression::mle(self.inner_sel_mle.mle_ref()).concat_expr(inner_inner_sel);
        let outer_sel = Expression::mle(self.outer_sel_mle.mle_ref()).concat_expr(inner_sel);
        outer_sel
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        let inner_inner_sel_bt = self
            .inner_inner_sel_mle
            .mle
            .iter()
            .flat_map(|elem| vec![*elem, *elem * elem]);

        let inner_sel_bt = inner_inner_sel_bt
            .zip(self.inner_sel_mle.mle.iter())
            .flat_map(|(elem_1, elem_2)| vec![elem_1, *elem_2]);

        let final_bt = inner_sel_bt
            .zip(self.outer_sel_mle.mle.iter())
            .flat_map(|(elem_1, elem_2)| vec![elem_1, *elem_2])
            .collect_vec();

        DenseMle::new_from_raw(final_bt, id, prefix_bits)
    }
}
impl<F: FieldExt> TripleNestedSelectorBuilder<F> {
    pub(crate) fn new(
        inner_inner_sel_mle: DenseMle<F, F>,
        inner_sel_mle: DenseMle<F, F>,
        outer_sel_mle: DenseMle<F, F>,
    ) -> Self {
        Self {
            inner_inner_sel_mle,
            inner_sel_mle,
            outer_sel_mle,
        }
    }
}

pub(crate) struct ProductScaledBuilder<F: FieldExt> {
    mle_1: DenseMle<F, F>,
    mle_2: DenseMle<F, F>,
}
impl<F: FieldExt> LayerBuilder<F> for ProductScaledBuilder<F> {
    type Successor = DenseMle<F, F>;

    fn build_expression(&self) -> Expression<F, ProverExpr> {
        let prod_expr = Expression::products(vec![self.mle_1.mle_ref(), self.mle_2.mle_ref()]);
        let scaled_expr = Expression::scaled(
            Box::new(Expression::mle(self.mle_1.mle_ref())),
            F::from(10_u64),
        );
        prod_expr + scaled_expr
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        let prod_bt = self
            .mle_1
            .mle
            .iter()
            .zip(self.mle_2.mle.iter())
            .map(|(elem_1, elem_2)| *elem_1 * elem_2);

        let scaled_bt = self.mle_1.mle.iter().map(|elem| F::from(10_u64) * elem);

        let final_bt = prod_bt
            .zip(scaled_bt)
            .map(|(elem_1, elem_2)| elem_1 + elem_2)
            .collect_vec();

        DenseMle::new_from_raw(final_bt, id, prefix_bits)
    }
}
impl<F: FieldExt> ProductScaledBuilder<F> {
    pub(crate) fn new(mle_1: DenseMle<F, F>, mle_2: DenseMle<F, F>) -> Self {
        Self { mle_1, mle_2 }
    }
}

pub(crate) struct ProductSumBuilder<F: FieldExt> {
    mle_1: DenseMle<F, F>,
    mle_2: DenseMle<F, F>,
}
impl<F: FieldExt> LayerBuilder<F> for ProductSumBuilder<F> {
    type Successor = DenseMle<F, F>;

    fn build_expression(&self) -> Expression<F, ProverExpr> {
        let prod_expr = Expression::products(vec![self.mle_1.mle_ref(), self.mle_2.mle_ref()]);
        let sum_expr = Expression::sum(
            Box::new(Expression::mle(self.mle_1.mle_ref())),
            Box::new(Expression::mle(self.mle_2.mle_ref())),
        );
        prod_expr + sum_expr
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        let prod_bt = self
            .mle_1
            .mle
            .iter()
            .zip(self.mle_2.mle.iter())
            .map(|(elem_1, elem_2)| *elem_1 * elem_2);

        let sum_bt = self
            .mle_1
            .mle
            .iter()
            .zip(self.mle_2.mle.iter())
            .map(|(elem_1, elem_2)| *elem_1 + elem_2);

        let final_bt = prod_bt
            .zip(sum_bt)
            .map(|(elem_1, elem_2)| elem_1 + elem_2)
            .collect_vec();

        DenseMle::new_from_raw(final_bt, id, prefix_bits)
    }
}
impl<F: FieldExt> ProductSumBuilder<F> {
    pub(crate) fn new(mle_1: DenseMle<F, F>, mle_2: DenseMle<F, F>) -> Self {
        Self { mle_1, mle_2 }
    }
}

pub(crate) struct ConstantScaledSumBuilder<F: FieldExt> {
    mle_1: DenseMle<F, F>,
    mle_2: DenseMle<F, F>,
}
impl<F: FieldExt> LayerBuilder<F> for ConstantScaledSumBuilder<F> {
    type Successor = DenseMle<F, F>;

    fn build_expression(&self) -> Expression<F, ProverExpr> {
        let constant_expr =
            Expression::mle(self.mle_1.mle_ref()) + Expression::constant(F::from(10_u64));
        let scaled_expr = Expression::scaled(
            Box::new(Expression::mle(self.mle_2.mle_ref())),
            F::from(10_u64),
        );
        constant_expr + scaled_expr
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        let constant_bt = self.mle_1.mle.iter().map(|elem| *elem + F::from(10_u64));

        let scaled_bt = self.mle_2.mle.iter().map(|elem| F::from(10_u64) * elem);

        let final_bt = constant_bt
            .zip(scaled_bt)
            .map(|(elem_1, elem_2)| elem_1 + elem_2)
            .collect_vec();

        DenseMle::new_from_raw(final_bt, id, prefix_bits)
    }
}
impl<F: FieldExt> ConstantScaledSumBuilder<F> {
    pub(crate) fn new(mle_1: DenseMle<F, F>, mle_2: DenseMle<F, F>) -> Self {
        Self { mle_1, mle_2 }
    }
}
