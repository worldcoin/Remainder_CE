use ark_std::log2;
use itertools::{repeat_n, Itertools};
use rand::Rng;
use remainder::expression::abstract_expr::ExprBuilder;
use remainder::layer::LayerId;
use remainder::layouter::component::Component;
use remainder::layouter::nodes::circuit_inputs::{InputLayerNode, InputShred, InputShredData};
use remainder::layouter::nodes::circuit_outputs::OutputNode;
use remainder::layouter::nodes::sector::Sector;
use remainder::layouter::nodes::CircuitNode;
use remainder::mle::evals::{Evaluations, MultilinearExtension};

use remainder::mle::dense::DenseMle;
use remainder::mle::MleIndex;
use remainder_shared_types::{Field, Fr};

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

/// Returns an [InputShred] with the appropriate [MultilinearExtension] as the data generated from random u64
pub fn get_dummy_input_shred_and_data(
    num_vars: usize,
    rng: &mut impl Rng,
    input_node: &InputLayerNode,
) -> (InputShred, InputShredData<Fr>) {
    let mle_vec = (0..(1 << num_vars))
        .map(|_| Fr::from(rng.gen::<u64>()))
        .collect_vec();
    let data = MultilinearExtension::new_from_evals(Evaluations::new(num_vars, mle_vec));
    let input_shred = InputShred::new(data.num_vars(), input_node);
    let input_shred_data = InputShredData::new(input_shred.id(), data);
    (input_shred, input_shred_data)
}

/// Returns an [InputShred] with the appropriate [MultilinearExtension], but given as input an mle_vec
pub fn get_input_shred_and_data_from_vec(
    mle_vec: Vec<Fr>,
    input_node: &InputLayerNode,
) -> (InputShred, InputShredData<Fr>) {
    assert!(mle_vec.len().is_power_of_two());
    let data = MultilinearExtension::new(mle_vec);
    let input_shred = InputShred::new(data.num_vars(), input_node);
    let input_shred_data = InputShredData::new(input_shred.id(), data);
    (input_shred, input_shred_data)
}

/// Returns the total MLE indices given a Vec<bool>
/// for the prefix bits and then the number of free
/// bits after.
pub fn get_total_mle_indices<F: Field>(
    prefix_bits: &[bool],
    num_free_bits: usize,
) -> Vec<MleIndex<F>> {
    prefix_bits
        .iter()
        .map(|bit| MleIndex::Fixed(*bit))
        .chain(repeat_n(MleIndex::Free, num_free_bits))
        .collect()
}

/// Returns the total number of variables which would be present in an MLE
/// which combines the bookkeeping tables of the given MLEs (given in terms
/// of the number of variables representing each one)
pub fn get_total_combined_mle_num_vars(all_num_vars: &[usize]) -> usize {
    log2(all_num_vars.iter().fold(0, |acc, elem| acc + (1 << *elem))) as usize
}

/// A builder which returns an expression with three nested selectors:
/// * innermost_selector: sel(`inner_inner_sel_mle`, `inner_inner_sel_mle * inner_inner_sel_mle`)
/// * inner_selector: sel(`innermost_selector`, `inner_sel_mle`)
/// * overall_expression: sel(`inner_selector`, `outer_sel_mle`).
///
/// ## Arguments
/// * `inner_inner_sel_mle` - An MLE with arbitrary bookkeeping table values.
/// * `inner_sel_mle` - An MLE with arbitrary bookkeeping table values, but double
/// the size of `inner_inner_sel_mle`
/// * `outer_sel_mle` - An MLE with arbitrary bookkeeping table values, but double
/// the size of `inner_sel_mle`

pub struct TripleNestedBuilderComponent<F: Field> {
    pub first_layer_sector: Sector<F>,
}

impl<F: Field> TripleNestedBuilderComponent<F> {
    pub fn new(
        inner_inner_sel: &dyn CircuitNode,
        inner_sel: &dyn CircuitNode,
        outer_sel: &dyn CircuitNode,
    ) -> Self {
        let triple_nested_selector_sector = Sector::new(
            &[inner_inner_sel, inner_sel, outer_sel],
            |triple_sel_nodes| {
                assert_eq!(triple_sel_nodes.len(), 3);
                let inner_inner_sel_mle = triple_sel_nodes[0];
                let inner_sel_mle = triple_sel_nodes[1];
                let outer_sel_mle = triple_sel_nodes[2];

                let inner_inner_sel =
                    inner_inner_sel_mle
                        .expr()
                        .select(ExprBuilder::<F>::products(vec![
                            inner_inner_sel_mle,
                            inner_inner_sel_mle,
                        ]));
                let inner_sel = inner_inner_sel.select(inner_sel_mle.expr());

                inner_sel.select(outer_sel_mle.expr())
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

impl<F: Field, N> Component<N> for TripleNestedBuilderComponent<F>
where
    N: CircuitNode + From<Sector<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.first_layer_sector.into()]
    }
}

/// A builder which takes the difference of an MLE from itself to return a zero layer.

pub struct DifferenceBuilderComponent<F: Field> {
    pub output_sector: Sector<F>,
    pub output_node: OutputNode,
}

impl<F: Field> DifferenceBuilderComponent<F> {
    pub fn new(input: &dyn CircuitNode) -> Self {
        let zero_output_sector = Sector::new(&[input], |input_vec| {
            assert_eq!(input_vec.len(), 1);
            let input_data = input_vec[0];
            input_data.expr() - input_data.expr()
        });

        let output = OutputNode::new_zero(&zero_output_sector);

        Self {
            output_sector: zero_output_sector,
            output_node: output,
        }
    }
}

impl<F: Field, N> Component<N> for DifferenceBuilderComponent<F>
where
    N: CircuitNode + From<Sector<F>> + From<OutputNode>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.output_sector.into(), self.output_node.into()]
    }
}

/// A builder which returns the following expression:
/// `mle_1 * mle_2 + (10 * mle_1)`
///
/// ## Arguments
/// * `mle_1` - An MLE with arbitrary bookkeeping table values.
/// * `mle_2` - An MLE with arbitrary bookkeeping table values; same size as `mle_1`.
pub struct ProductScaledBuilderComponent<F: Field> {
    pub first_layer_sector: Sector<F>,
}

impl<F: Field> ProductScaledBuilderComponent<F> {
    pub fn new(mle_1: &dyn CircuitNode, mle_2: &dyn CircuitNode) -> Self {
        let product_scaled_sector = Sector::new(&[mle_1, mle_2], |product_scaled_nodes| {
            assert_eq!(product_scaled_nodes.len(), 2);
            let mle_1 = product_scaled_nodes[0];
            let mle_2 = product_scaled_nodes[1];

            ExprBuilder::<F>::products(vec![mle_1, mle_2])
                + ExprBuilder::<F>::scaled(mle_1.expr(), F::from(10_u64))
        });

        Self {
            first_layer_sector: product_scaled_sector,
        }
    }

    pub fn get_output_sector(&self) -> &Sector<F> {
        &self.first_layer_sector
    }
}

impl<F: Field, N> Component<N> for ProductScaledBuilderComponent<F>
where
    N: CircuitNode + From<Sector<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.first_layer_sector.into()]
    }
}

/// A builder which returns the following expression:
/// `mle_1 * mle_2 + (mle_1 + mle_2)`.
///
/// ## Arguments
/// * `mle_1` - An MLE with arbitrary bookkeeping table values.
/// * `mle_2` - An MLE with arbitrary bookkeeping table values; same size as `mle_1`.

pub struct ProductSumBuilderComponent<F: Field> {
    pub first_layer_sector: Sector<F>,
}

impl<F: Field> ProductSumBuilderComponent<F> {
    pub fn new(mle_1: &dyn CircuitNode, mle_2: &dyn CircuitNode) -> Self {
        let product_sum_sector = Sector::new(&[mle_1, mle_2], |product_sum_nodes| {
            assert_eq!(product_sum_nodes.len(), 2);
            let mle_1 = product_sum_nodes[0];
            let mle_2 = product_sum_nodes[1];

            ExprBuilder::<F>::products(vec![mle_1, mle_2]) + (mle_1.expr() + mle_2.expr())
        });

        Self {
            first_layer_sector: product_sum_sector,
        }
    }

    pub fn get_output_sector(&self) -> &Sector<F> {
        &self.first_layer_sector
    }
}

impl<F: Field, N> Component<N> for ProductSumBuilderComponent<F>
where
    N: CircuitNode + From<Sector<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.first_layer_sector.into()]
    }
}

/// A builder which returns the following expression:
/// `mle_1 + 10 + (mle_2 * 10)`.
///
/// ## Arguments
/// * `mle_1` - An MLE with arbitrary bookkeeping table values.
/// * `mle_2` - An MLE with arbitrary bookkeeping table values; same size as `mle_1`.

pub struct ConstantScaledSumBuilderComponent<F: Field> {
    pub first_layer_sector: Sector<F>,
}

impl<F: Field> ConstantScaledSumBuilderComponent<F> {
    pub fn new(mle_1: &dyn CircuitNode, mle_2: &dyn CircuitNode) -> Self {
        let constant_scaled_sector = Sector::new(&[mle_1, mle_2], |constant_scaled_nodes| {
            assert_eq!(constant_scaled_nodes.len(), 2);
            let mle_1 = constant_scaled_nodes[0];
            let mle_2 = constant_scaled_nodes[1];

            ExprBuilder::<F>::scaled(mle_2.expr(), F::from(10_u64))
                + (mle_1.expr() + ExprBuilder::constant(F::from(10_u64)))
        });

        Self {
            first_layer_sector: constant_scaled_sector,
        }
    }

    pub fn get_output_sector(&self) -> &Sector<F> {
        &self.first_layer_sector
    }
}

impl<F: Field, N> Component<N> for ConstantScaledSumBuilderComponent<F>
where
    N: CircuitNode + From<Sector<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.first_layer_sector.into()]
    }
}
