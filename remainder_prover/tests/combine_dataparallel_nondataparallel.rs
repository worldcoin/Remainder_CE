use ark_std::test_rng;
use itertools::Itertools;

use remainder::{
    builders::{
        combine_input_layers::InputLayerBuilder, combine_layers::combine_layers,
        layer_builder::simple_builders::ZeroBuilder,
    },
    expression::abstract_expr::ExprBuilder,
    input_layer::public_input_layer::PublicInputLayer,
    layer::LayerId,
    layouter::{
        component::Component,
        nodes::{circuit_outputs::OutputNode, sector::Sector, CircuitNode, ClaimableNode, Context},
    },
    mle::{dense::DenseMle, evals::MultilinearExtension, Mle},
    prover::{
        helpers::test_circuit, layers::Layers, proof_system::DefaultProofSystem, GKRCircuit,
        Witness,
    },
};
use remainder_shared_types::FieldExt;

pub mod utils;

use utils::{ProductScaledBuilder, TripleNestedSelectorBuilder};

use crate::utils::get_dummy_random_mle;

/// A circuit which takes in two vectors of MLEs of the same size:
/// * Layer 0: [ProductScaledBuilder] with the two inputs
/// * Layer 1: [ProductScaledBuilder] with the output of Layer 0 and output of Layer 0.
///
/// The expected output of this circuit is the zero MLE.
///
/// ## Arguments
/// * `mle_1_vec` - An MLE vec with arbitrary bookkeeping table values.
/// * `mle_2_vec` - An MLE vec with arbitrary bookkeeping table values, same size as `mle_1_vec`.
struct DataParallelCircuit<F: FieldExt> {
    mle_1: DenseMle<F>,
    mle_2: DenseMle<F>,
}

impl<F: FieldExt> GKRCircuit<F> for DataParallelCircuit<F> {
    type ProofSystem = DefaultProofSystem;

    fn synthesize(&mut self) -> Witness<F, Self::ProofSystem> {
        let mut layers = Layers::new();

        let first_layer_builder = ProductScaledBuilder::new(self.mle_1.clone(), self.mle_2.clone());
        let first_layer_output = layers.add_gkr(first_layer_builder);

        let second_layer_builder =
            ProductScaledBuilder::new(first_layer_output.clone(), first_layer_output);

        let second_layer_output = layers.add_gkr(second_layer_builder);

        let zero_builders = ZeroBuilder::new(second_layer_output);
        let output = layers.add_gkr(zero_builders);

        Witness {
            layers,
            output_layers: vec![output.get_enum()],
            input_layers: vec![],
        }
    }
}

struct DataParallelComponent<F: FieldExt> {
    first_layer_sector: Sector<F>,
    second_layer_sector: Sector<F>,
    output_sector: Sector<F>,
}

impl<F: FieldExt> DataParallelComponent<F> {
    pub fn new(
        ctx: &Context,
        mle_1_input: &dyn ClaimableNode<F = F>,
        mle_2_input: &dyn ClaimableNode<F = F>,
    ) -> Self {
        let product_scaled_nodes = &[mle_1_input, mle_2_input];
        let product_scaled_sector = Sector::new(
            ctx,
            product_scaled_nodes,
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

        let product_scaled_meta_sector = Sector::new(
            ctx,
            &[&product_scaled_sector],
            |prod_scaled_result_vec| {
                assert_eq!(prod_scaled_result_vec.len(), 1);
                let prod_scaled_result = prod_scaled_result_vec[0];

                ExprBuilder::<F>::products(vec![prod_scaled_result, prod_scaled_result])
                    + ExprBuilder::<F>::scaled(prod_scaled_result.expr(), F::from(10_u64))
            },
            |data| {
                let mle_1 = data[0];
                let final_bt = mle_1
                    .get_evals_vector()
                    .iter()
                    .map(|elem| *elem * elem + F::from(10_u64) * elem)
                    .collect_vec();

                MultilinearExtension::new(final_bt)
            },
        );

        let zero_output_sector = Sector::new(
            ctx,
            &[&product_scaled_meta_sector],
            |product_meta_vec| {
                assert_eq!(product_meta_vec.len(), 1);
                let prod_meta = product_meta_vec[0];

                prod_meta.expr() - prod_meta.expr()
            },
            |_data| MultilinearExtension::new_zero(),
        );

        Self {
            first_layer_sector: product_scaled_sector,
            second_layer_sector: product_scaled_meta_sector,
            output_sector: zero_output_sector,
        }
    }
}

impl<F: FieldExt, N> Component<N> for DataParallelComponent<F>
where
    N: CircuitNode + From<Sector<F>> + From<OutputNode<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![
            self.first_layer_sector.into(),
            self.second_layer_sector.into(),
            self.output_sector.into(),
        ]
    }
}

/// A circuit in which:
/// * Layer 0: [TripleNestedSelectorBuilder] with the three inputs
/// * Layer 1: [ZeroBuilder] with output of Layer 0 and itself.
///
/// The expected output of this circuit is the zero MLE.
///
/// ## Arguments
/// * `inner_inner_sel_mle` - An MLE with arbitrary bookkeeping table values.
/// * `inner_sel_mle` - An MLE with arbitrary bookkeeping table values, but double
/// the size of `inner_inner_sel_mle`
/// * `outer_sel_mle` - An MLE with arbitrary bookkeeping table values, but double
/// the size of `inner_sel_mle`
struct TripleNestedSelectorCircuit<F: FieldExt> {
    inner_inner_sel_mle: DenseMle<F>,
    inner_sel_mle: DenseMle<F>,
    outer_sel_mle: DenseMle<F>,
}

impl<F: FieldExt> GKRCircuit<F> for TripleNestedSelectorCircuit<F> {
    type ProofSystem = DefaultProofSystem;

    fn synthesize(&mut self) -> Witness<F, Self::ProofSystem> {
        let mut layers = Layers::new();

        let first_layer_builder = TripleNestedSelectorBuilder::new(
            self.inner_inner_sel_mle.clone(),
            self.inner_sel_mle.clone(),
            self.outer_sel_mle.clone(),
        );
        let first_layer_output = layers.add_gkr(first_layer_builder);

        let zero_builder = ZeroBuilder::new(first_layer_output);
        let output = layers.add_gkr(zero_builder);

        Witness {
            layers,
            output_layers: vec![output.get_enum()],
            input_layers: vec![],
        }
    }
}

struct TripleNestedSelectorComponent<F: FieldExt> {
    first_layer_sector: Sector<F>,
    output_sector: Sector<F>,
}

impl<F: FieldExt> TripleNestedSelectorComponent<F> {
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

        let zero_output_sector = Sector::new(
            ctx,
            &[&triple_nested_selector_sector],
            |triple_sel_vec| {
                assert_eq!(triple_sel_vec.len(), 1);
                let triple_sel_data = triple_sel_vec[0];

                triple_sel_data.expr() - triple_sel_data.expr()
            },
            |_data| MultilinearExtension::new_zero(),
        );

        Self {
            first_layer_sector: triple_nested_selector_sector,
            output_sector: zero_output_sector,
        }
    }
}

impl<F: FieldExt, N> Component<N> for TripleNestedSelectorComponent<F>
where
    N: CircuitNode + From<Sector<F>> + From<OutputNode<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.first_layer_sector.into(), self.output_sector.into()]
    }
}

/// A circuit in which:
/// * Layer 0: [ProductScaledBuilder] with the two inputs
/// * Layer 1: [ZeroBuilder] with output of Layer 0 and itself.
///
/// The expected output of this circuit is the zero MLE.
///
/// ## Arguments
/// * `mle_1` - An MLE with arbitrary bookkeeping table values.
/// * `mle_2` - An MLE with arbitrary bookkeeping table values, same size as `mle_1`.
struct ScaledProductCircuit<F: FieldExt> {
    mle_1: DenseMle<F>,
    mle_2: DenseMle<F>,
}

impl<F: FieldExt> GKRCircuit<F> for ScaledProductCircuit<F> {
    type ProofSystem = DefaultProofSystem;

    fn synthesize(&mut self) -> Witness<F, Self::ProofSystem> {
        let mut layers = Layers::new();

        let first_layer_builder = ProductScaledBuilder::new(self.mle_1.clone(), self.mle_2.clone());
        let first_layer_output = layers.add_gkr(first_layer_builder);

        let zero_builder = ZeroBuilder::new(first_layer_output);
        let output = layers.add_gkr(zero_builder);

        Witness {
            layers,
            output_layers: vec![output.get_enum()],
            input_layers: vec![],
        }
    }
}

struct ScaledProductComponent<F: FieldExt> {
    first_layer_sector: Sector<F>,
    output_sector: Sector<F>,
}

impl<F: FieldExt> ScaledProductComponent<F> {
    pub fn new(
        ctx: &Context,
        mle_1_input: &dyn ClaimableNode<F = F>,
        mle_2_input: &dyn ClaimableNode<F = F>,
    ) -> Self {
        let product_scaled_nodes = &[mle_1_input, mle_2_input];
        let product_scaled_sector = Sector::new(
            ctx,
            product_scaled_nodes,
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

        let zero_output_sector = Sector::new(
            ctx,
            &[&product_scaled_sector],
            |product_scaled_vec| {
                assert_eq!(product_scaled_vec.len(), 1);
                let prod_scaled_data = product_scaled_vec[0];

                prod_scaled_data.expr() - prod_scaled_data.expr()
            },
            |_data| MultilinearExtension::new_zero(),
        );

        Self {
            first_layer_sector: product_scaled_sector,
            output_sector: zero_output_sector,
        }
    }
}

impl<F: FieldExt, N> Component<N> for ScaledProductComponent<F>
where
    N: CircuitNode + From<Sector<F>> + From<OutputNode<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.first_layer_sector.into(), self.output_sector.into()]
    }
}

#[test]
fn test_combined_components_circuit() {}

/// A circuit which combines the [DataParallelCircuit], [TripleNestedSelectorCircuit],
/// and [ScaledProductCircuit].
/// The expected output of this circuit is the zero MLE.
///
/// ## Arguments
/// * `mle_1_vec`, `mle_2_vec` - inputs to [DataParallelCircuit] both arbitrary bookkeeping
/// table values, same size.
/// * `mle_4`, `mle_5`, `mle_6` - inputs to [TripleNestedSelectorCircuit], `mle_4` has the same
/// size as the mles in `mle_1_vec`, arbitrary bookkeeping table values. `mle_5` has one more
/// variable than `mle_4`, `mle_6` has one more variable than `mle_5`, both arbitrary bookkeeping
/// table values.
/// * `mle_3`, `mle_4` - inputs to [ScaledProductCircuit], both arbitrary bookkeeping table values,
/// same size.
/// * `num_dataparallel_bits` - The number of bits that represent which copy index the circuit is.
struct CombinedCircuitAlt<F: FieldExt> {
    mle_1: DenseMle<F>,
    mle_2: DenseMle<F>,
    mle_3: DenseMle<F>,
    mle_4: DenseMle<F>,
    mle_5: DenseMle<F>,
    mle_6: DenseMle<F>,
}

impl<F: FieldExt> GKRCircuit<F> for CombinedCircuitAlt<F> {
    type ProofSystem = DefaultProofSystem;

    fn synthesize(&mut self) -> Witness<F, Self::ProofSystem> {
        self.mle_1.layer_id = LayerId::Input(0);
        self.mle_2.layer_id = LayerId::Input(0);
        self.mle_3.layer_id = LayerId::Input(0);
        self.mle_4.layer_id = LayerId::Input(0);
        self.mle_5.layer_id = LayerId::Input(0);
        self.mle_6.layer_id = LayerId::Input(0);

        let input_commit: Vec<&mut dyn Mle<F>> = vec![
            &mut self.mle_1,
            &mut self.mle_2,
            &mut self.mle_3,
            &mut self.mle_4,
            &mut self.mle_5,
            &mut self.mle_6,
        ];

        let input_commit_builder =
            InputLayerBuilder::<F>::new(input_commit, None, LayerId::Input(0));

        let input_layer: PublicInputLayer<F> =
            input_commit_builder.to_input_layer::<PublicInputLayer<F>>();

        let input_layer_enum = input_layer.into();

        let mut batched_circuit = DataParallelCircuit {
            mle_1: self.mle_1.clone(),
            mle_2: self.mle_2.clone(),
        };
        let mut triple_nested_sel_circuit = TripleNestedSelectorCircuit {
            inner_inner_sel_mle: self.mle_4.clone(),
            inner_sel_mle: self.mle_5.clone(),
            outer_sel_mle: self.mle_6.clone(),
        };
        let mut product_scaled_circuit = ScaledProductCircuit {
            mle_1: self.mle_3.clone(),
            mle_2: self.mle_4.clone(),
        };

        let batched_circuit_witness = batched_circuit.synthesize();
        let triple_nested_sel_witness = triple_nested_sel_circuit.synthesize();
        let product_scaled_witness = product_scaled_circuit.synthesize();

        let Witness {
            layers: batched_layers,
            output_layers: batched_output_layers,
            input_layers: _,
        } = batched_circuit_witness;

        let Witness {
            layers: triple_nested_layers,
            output_layers: triple_output_layers,
            input_layers: _,
        } = triple_nested_sel_witness;

        let Witness {
            layers: product_scaled_layers,
            output_layers: product_scaled_output_layers,
            input_layers: _,
        } = product_scaled_witness;

        let (layers, output_layers) = combine_layers(
            vec![batched_layers, triple_nested_layers, product_scaled_layers],
            vec![
                batched_output_layers,
                triple_output_layers,
                product_scaled_output_layers,
            ],
        )
        .unwrap();

        Witness {
            layers,
            output_layers,
            input_layers: vec![input_layer_enum],
        }
    }
}

impl<F: FieldExt> CombinedCircuitAlt<F> {
    fn new(
        mle_1: DenseMle<F>,
        mle_2: DenseMle<F>,
        mle_3: DenseMle<F>,
        mle_4: DenseMle<F>,
        mle_5: DenseMle<F>,
        mle_6: DenseMle<F>,
    ) -> Self {
        Self {
            mle_1,
            mle_2,
            mle_3,
            mle_4,
            mle_5,
            mle_6,
        }
    }
}

// TODO(vishady): this test fails based off of our current implementation of remainder!! The current problem is the way
// selector bits are treated when combining expressions.

// (N_1, ..., N_n, C_1, ..., C_n, H_1, ..., H_n, W_1, ..., W_n)
// MLE(b_1, b_2, c_1, c_2, c_3) --> 5 iterated bits total, 2 dataparallel bits
// MLE(c_1) --> 4 iterated bits total, 1 dataparallel bit

#[test]
fn test_combined_dataparallel_nondataparallel_circuit_alt() {
    const VARS_MLE_1_2: usize = 2;
    const VARS_MLE_3: usize = VARS_MLE_1_2 + 1;
    const VARS_MLE_4: usize = VARS_MLE_3 + 1;
    const NUM_DATA_PARALLEL_BITS: usize = 1;
    let mut rng = test_rng();

    let mle_1_vec = (0..1 << NUM_DATA_PARALLEL_BITS)
        .map(|_| get_dummy_random_mle(VARS_MLE_1_2, &mut rng))
        .collect_vec();
    let mle_2_vec = (0..1 << NUM_DATA_PARALLEL_BITS)
        .map(|_| get_dummy_random_mle(VARS_MLE_1_2, &mut rng))
        .collect_vec();
    let mle_3 = get_dummy_random_mle(VARS_MLE_1_2, &mut rng);
    let mle_4 = get_dummy_random_mle(VARS_MLE_1_2, &mut rng);
    let mle_5 = get_dummy_random_mle(VARS_MLE_3, &mut rng);
    let mle_6 = get_dummy_random_mle(VARS_MLE_4, &mut rng);

    // These checks can possibly be done with the newly designed batching bits/system
    assert_eq!(mle_3.num_iterated_vars(), mle_4.num_iterated_vars());
    assert_eq!(mle_5.num_iterated_vars(), mle_4.num_iterated_vars() + 1);
    assert_eq!(mle_6.num_iterated_vars(), mle_5.num_iterated_vars() + 1);
    let all_num_vars: Vec<usize> = mle_1_vec
        .iter()
        .chain(mle_2_vec.iter())
        .map(|mle| mle.num_iterated_vars())
        .collect();
    let all_vars_same = all_num_vars
        .iter()
        .fold(true, |acc, elem| (*elem == mle_3.num_iterated_vars()) & acc);
    assert!(all_vars_same);
    assert_eq!(mle_1_vec.len(), mle_2_vec.len());
    assert_eq!(mle_1_vec.len(), 1 << NUM_DATA_PARALLEL_BITS);
    // These checks can possibly be done with the newly designed batching bits/system

    // the batched mle should be able to demonstrate that there's NUM_DATA_PARALLEL_BITS of batch bits
    let mle_1_vec_batched = DenseMle::batch_mles(mle_1_vec);
    let mle_2_vec_batched = DenseMle::batch_mles(mle_2_vec);
    // the batched mle should be able to demonstrate that there's NUM_DATA_PARALLEL_BITS of batch bits

    let combined_circuit = CombinedCircuitAlt::new(
        mle_1_vec_batched,
        mle_2_vec_batched,
        mle_3,
        mle_4,
        mle_5,
        mle_6,
    );
    test_circuit(combined_circuit, None)
}
