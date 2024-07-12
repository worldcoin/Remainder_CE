use ark_std::log2;
use itertools::Itertools;
use rand::Rng;
use remainder::expression::abstract_expr::ExprBuilder;
use remainder::layouter::component::Component;
use remainder::layouter::nodes::circuit_inputs::InputShred;
use remainder::layouter::nodes::circuit_outputs::OutputNode;
use remainder::layouter::nodes::sector::Sector;
use remainder::layouter::nodes::{CircuitNode, ClaimableNode, Context};
use remainder::mle::evals::{Evaluations, MultilinearExtension};

use remainder::mle::dense::DenseMle;
use remainder_shared_types::layer::LayerId;
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

/// Returns a vector with random elements generated from u64 for testing according to the number of variables.
pub fn get_dummy_random_vec(num_vars: usize, rng: &mut impl Rng) -> Vec<Fr> {
    (0..(1 << num_vars))
        .map(|_| Fr::from(rng.gen::<u64>()))
        .collect_vec()
}

/// Returns an [InputShred] with the appropriate [MultilinearExtension] as the data generated from random u64
pub fn get_dummy_input_shred(num_vars: usize, rng: &mut impl Rng, ctx: &Context) -> InputShred<Fr> {
    // let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);
    let mle_vec = (0..(1 << num_vars))
        .map(|_| Fr::from(rng.gen::<u64>()))
        .collect_vec();
    InputShred::new(
        ctx,
        MultilinearExtension::new_from_evals(Evaluations::new(num_vars, mle_vec)),
        None,
    )
}

/// Returns an [InputShred] with the appropriate [MultilinearExtension], but given as input an mle_vec
pub fn get_input_shred_from_vec(mle_vec: Vec<Fr>, ctx: &Context) -> InputShred<Fr> {
    assert!(mle_vec.len().is_power_of_two());
    InputShred::new(
        ctx,
        MultilinearExtension::new_from_evals(Evaluations::new(
            log2(mle_vec.len()) as usize,
            mle_vec,
        )),
        None,
    )
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

pub struct TripleNestedBuilderComponent<F: FieldExt> {
    pub first_layer_sector: Sector<F>,
}

impl<F: FieldExt> TripleNestedBuilderComponent<F> {
    pub fn new(
        ctx: &Context,
        inner_inner_sel: &dyn ClaimableNode<F = F>,
        inner_sel: &dyn ClaimableNode<F = F>,
        outer_sel: &dyn ClaimableNode<F = F>,
    ) -> Self {
        let triple_nested_selector_sector = Sector::new(
            ctx,
            &[inner_inner_sel, inner_sel, outer_sel],
            |triple_sel_nodes| {
                assert_eq!(triple_sel_nodes.len(), 3);
                let inner_inner_sel_mle = triple_sel_nodes[0];
                let inner_sel_mle = triple_sel_nodes[1];
                let outer_sel_mle = triple_sel_nodes[2];

                let inner_inner_sel =
                    ExprBuilder::<F>::products(vec![inner_inner_sel_mle, inner_inner_sel_mle])
                        .concat_expr(inner_inner_sel_mle.expr());
                let inner_sel = inner_sel_mle.expr().concat_expr(inner_inner_sel);

                outer_sel_mle.expr().concat_expr(inner_sel)
            },
            |data| {
                let inner_inner_sel_data = data[0];
                let inner_sel_data = data[1];
                let outer_sel_data = data[2];
                let inner_inner_sel_bt = inner_inner_sel_data
                    .get_evals_vector()
                    .iter()
                    .flat_map(|elem| vec![*elem, *elem * elem]);

                let inner_sel_bt = inner_inner_sel_bt
                    .zip(inner_sel_data.get_evals_vector().iter())
                    .flat_map(|(elem_1, elem_2)| vec![elem_1, *elem_2]);

                let final_bt = inner_sel_bt
                    .zip(outer_sel_data.get_evals_vector().iter())
                    .flat_map(|(elem_1, elem_2)| vec![elem_1, *elem_2])
                    .collect_vec();

                MultilinearExtension::new(final_bt)
            },
        );

        Self {
            first_layer_sector: triple_nested_selector_sector,
        }
    }

    pub fn get_output_sector(&self) -> &Sector<F> {
        &self.first_layer_sector
    }
}

impl<F: FieldExt, N> Component<N> for TripleNestedBuilderComponent<F>
where
    N: CircuitNode + From<Sector<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.first_layer_sector.into()]
    }
}

/// A builder which takes the difference of an MLE from itself to return a zero layer.

pub struct DifferenceBuilderComponent<F: FieldExt> {
    pub output_sector: Sector<F>,
    pub output_node: OutputNode<F>,
}

impl<F: FieldExt> DifferenceBuilderComponent<F> {
    pub fn new(ctx: &Context, input: &dyn ClaimableNode<F = F>) -> Self {
        let zero_output_sector = Sector::new(
            ctx,
            &[input],
            |input_vec| {
                assert_eq!(input_vec.len(), 1);
                let input_data = input_vec[0];
                input_data.expr() - input_data.expr()
            },
            |data| MultilinearExtension::new_sized_zero(data[0].num_vars()),
        );

        let output = OutputNode::new_zero(ctx, &zero_output_sector);

        Self {
            output_sector: zero_output_sector,
            output_node: output,
        }
    }
}

impl<F: FieldExt, N> Component<N> for DifferenceBuilderComponent<F>
where
    N: CircuitNode + From<Sector<F>> + From<OutputNode<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.output_sector.into(), self.output_node.into()]
    }
}

/// A builder which returns the following expression:
/// - `mle_1` * `mle_2` + (10 * `mle_1`)
///
/// ## Arguments
/// * `mle_1` - An MLE with arbitrary bookkeeping table values.
/// * `mle_2` - An MLE with arbitrary bookkeeping table values; same size as `mle_1`.
pub struct ProductScaledBuilderComponent<F: FieldExt> {
    pub first_layer_sector: Sector<F>,
}

impl<F: FieldExt> ProductScaledBuilderComponent<F> {
    pub fn new(
        ctx: &Context,
        mle_1: &dyn ClaimableNode<F = F>,
        mle_2: &dyn ClaimableNode<F = F>,
    ) -> Self {
        let product_scaled_sector = Sector::new(
            ctx,
            &[mle_1, mle_2],
            |product_scaled_nodes| {
                assert_eq!(product_scaled_nodes.len(), 2);
                let mle_1 = product_scaled_nodes[0];
                let mle_2 = product_scaled_nodes[1];

                ExprBuilder::<F>::products(vec![mle_1, mle_2])
                    + ExprBuilder::<F>::scaled(mle_1.expr(), F::from(10_u64))
            },
            |data| {
                let mle_1 = data[0];
                let mle_2 = data[1];
                let prod_bt = mle_1
                    .get_evals_vector()
                    .iter()
                    .zip(mle_2.get_evals_vector().iter())
                    .map(|(elem_1, elem_2)| *elem_1 * elem_2);

                let scaled_bt = mle_1
                    .get_evals_vector()
                    .iter()
                    .map(|elem| F::from(10_u64) * elem);

                let final_bt = prod_bt
                    .zip(scaled_bt)
                    .map(|(elem_1, elem_2)| elem_1 + elem_2)
                    .collect_vec();

                MultilinearExtension::new(final_bt)
            },
        );

        Self {
            first_layer_sector: product_scaled_sector,
        }
    }

    pub fn get_output_sector(&self) -> &Sector<F> {
        &self.first_layer_sector
    }
}

impl<F: FieldExt, N> Component<N> for ProductScaledBuilderComponent<F>
where
    N: CircuitNode + From<Sector<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.first_layer_sector.into()]
    }
}

/// A builder which returns the following expression:
/// - `mle_1` * `mle_2` + (`mle_1` + `mle_2`)
///
/// ## Arguments
/// * `mle_1` - An MLE with arbitrary bookkeeping table values.
/// * `mle_2` - An MLE with arbitrary bookkeeping table values; same size as `mle_1`.

pub struct ProductSumBuilderComponent<F: FieldExt> {
    pub first_layer_sector: Sector<F>,
}

impl<F: FieldExt> ProductSumBuilderComponent<F> {
    pub fn new(
        ctx: &Context,
        mle_1: &dyn ClaimableNode<F = F>,
        mle_2: &dyn ClaimableNode<F = F>,
    ) -> Self {
        let product_sum_sector = Sector::new(
            ctx,
            &[mle_1, mle_2],
            |product_sum_nodes| {
                assert_eq!(product_sum_nodes.len(), 2);
                let mle_1 = product_sum_nodes[0];
                let mle_2 = product_sum_nodes[1];

                ExprBuilder::<F>::products(vec![mle_1, mle_2]) + (mle_1.expr() + mle_2.expr())
            },
            |data| {
                let mle_1 = data[0];
                let mle_2 = data[1];
                let prod_bt = mle_1
                    .get_evals_vector()
                    .iter()
                    .zip(mle_2.get_evals_vector().iter())
                    .map(|(elem_1, elem_2)| *elem_1 * elem_2);

                let sum_bt = mle_1
                    .get_evals_vector()
                    .iter()
                    .zip(mle_2.get_evals_vector().iter())
                    .map(|(elem_1, elem_2)| *elem_1 + elem_2);

                let final_bt = prod_bt
                    .zip(sum_bt)
                    .map(|(elem_1, elem_2)| elem_1 + elem_2)
                    .collect_vec();

                MultilinearExtension::new(final_bt)
            },
        );

        Self {
            first_layer_sector: product_sum_sector,
        }
    }

    pub fn get_output_sector(&self) -> &Sector<F> {
        &self.first_layer_sector
    }
}

impl<F: FieldExt, N> Component<N> for ProductSumBuilderComponent<F>
where
    N: CircuitNode + From<Sector<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.first_layer_sector.into()]
    }
}

/// A builder which returns the following expression:
/// - `mle_1` + 10 + (`mle_2` * 10)
///
/// ## Arguments
/// * `mle_1` - An MLE with arbitrary bookkeeping table values.
/// * `mle_2` - An MLE with arbitrary bookkeeping table values; same size as `mle_1`.

pub struct ConstantScaledSumBuilderComponent<F: FieldExt> {
    pub first_layer_sector: Sector<F>,
}

impl<F: FieldExt> ConstantScaledSumBuilderComponent<F> {
    pub fn new(
        ctx: &Context,
        mle_1: &dyn ClaimableNode<F = F>,
        mle_2: &dyn ClaimableNode<F = F>,
    ) -> Self {
        let constant_scaled_sector = Sector::new(
            ctx,
            &[mle_1, mle_2],
            |constant_scaled_nodes| {
                assert_eq!(constant_scaled_nodes.len(), 2);
                let mle_1 = constant_scaled_nodes[0];
                let mle_2 = constant_scaled_nodes[1];

                ExprBuilder::<F>::scaled(mle_2.expr(), F::from(10_u64))
                    + (mle_1.expr() + ExprBuilder::constant(F::from(10_u64)))
            },
            |data| {
                let mle_1 = data[0];
                let mle_2 = data[1];
                let constant_bt = mle_1
                    .get_evals_vector()
                    .iter()
                    .map(|elem| *elem + F::from(10_u64));

                let scaled_bt = mle_2
                    .get_evals_vector()
                    .iter()
                    .map(|elem| F::from(10_u64) * elem);

                let final_bt = constant_bt
                    .zip(scaled_bt)
                    .map(|(elem_1, elem_2)| elem_1 + elem_2)
                    .collect_vec();
                MultilinearExtension::new(final_bt)
            },
        );

        Self {
            first_layer_sector: constant_scaled_sector,
        }
    }

    pub fn get_output_sector(&self) -> &Sector<F> {
        &self.first_layer_sector
    }
}

impl<F: FieldExt, N> Component<N> for ConstantScaledSumBuilderComponent<F>
where
    N: CircuitNode + From<Sector<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.first_layer_sector.into()]
    }
}
