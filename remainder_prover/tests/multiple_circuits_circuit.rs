use ark_std::test_rng;
use itertools::Itertools;
use rand::Rng;
use remainder::{
    expression::{
        generic_expr::{Expression, ExpressionNode, ExpressionType},
        prover_expr::ProverExpr,
    },
    layer::{
        batched::{self, combine_zero_mle_ref, BatchedLayer},
        layer_enum::LayerEnum,
        simple_builders::ZeroBuilder,
        LayerBuilder, LayerId,
    },
    mle::{
        dense::{DenseMle, Tuple2},
        Mle, MleIndex, MleRef,
    },
    prover::{
        combine_layers::combine_layers,
        helpers::test_circuit,
        input_layer::{
            combine_input_layers::InputLayerBuilder,
            enum_input_layer::{CommitmentEnum, InputLayerEnum},
            public_input_layer::PublicInputLayer,
            InputLayer,
        },
        GKRCircuit, GKRError, Layers, Witness,
    },
};
use remainder_shared_types::{
    transcript::{self, poseidon_transcript::PoseidonSponge, TranscriptWriter},
    FieldExt, Fr,
};
use tracing::instrument;

use crate::utils::get_dummy_random_mle;
mod utils;

struct ProductScaledBuilder<F: FieldExt> {
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
    fn new(mle_1: DenseMle<F, F>, mle_2: DenseMle<F, F>) -> Self {
        Self { mle_1, mle_2 }
    }
}

struct TripleNestedSelectorBuilder<F: FieldExt> {
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
            .zip(self.inner_inner_sel_mle.mle.iter())
            .flat_map(|(elem_1, elem_2)| vec![*elem_1, *elem_2 * elem_2]);

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
    fn new(
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

struct BatchedCircuit<F: FieldExt> {
    mle_1_vec: Vec<DenseMle<F, F>>,
    mle_2_vec: Vec<DenseMle<F, F>>,
    mle_3_vec: Vec<DenseMle<F, F>>,
    mle_4_vec: Vec<DenseMle<F, F>>,
}

impl<F: FieldExt> GKRCircuit<F> for BatchedCircuit<F> {
    type Sponge = PoseidonSponge<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Sponge> {
        let mut layers = Layers::new();

        let first_layer_builders = (self
            .mle_1_vec
            .clone()
            .into_iter()
            .zip(self.mle_2_vec.clone().into_iter()))
        .map(|(mle_1, mle_2)| ProductScaledBuilder::new(mle_1, mle_2))
        .collect_vec();
        let batched_first_layer_builder = BatchedLayer::new(first_layer_builders);
        let first_layer_output = layers.add_gkr(batched_first_layer_builder);

        let second_layer_builders = (first_layer_output.iter().zip(
            self.mle_3_vec
                .clone()
                .into_iter()
                .zip(self.mle_4_vec.clone().into_iter()),
        ))
        .map(|(mle, (mle_3, mle_4))| TripleNestedSelectorBuilder::new(mle.clone(), mle_3, mle_4))
        .collect_vec();

        let batched_second_layer_builder = BatchedLayer::new(second_layer_builders);
        // let second_layer_output = layers.add_gkr(batched_second_layer_builder);

        let zero_builders = first_layer_output
            .iter()
            .map(|mle| ZeroBuilder::new(mle.clone()))
            .collect_vec();
        let batched_output_builder = BatchedLayer::new(zero_builders);
        let output_mles = layers.add_gkr(batched_output_builder);
        let output = combine_zero_mle_ref(output_mles);

        Witness {
            layers,
            output_layers: vec![output.get_enum()],
            input_layers: vec![],
        }
    }
}

struct TripleNestedSelectorCircuit<F: FieldExt> {
    inner_inner_sel_mle: DenseMle<F, F>,
    inner_sel_mle: DenseMle<F, F>,
    outer_sel_mle: DenseMle<F, F>,
}

impl<F: FieldExt> GKRCircuit<F> for TripleNestedSelectorCircuit<F> {
    type Sponge = PoseidonSponge<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Sponge> {
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

struct ScaledProductCircuit<F: FieldExt> {
    mle_1: DenseMle<F, F>,
    mle_2: DenseMle<F, F>,
}

impl<F: FieldExt> GKRCircuit<F> for ScaledProductCircuit<F> {
    type Sponge = PoseidonSponge<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Sponge> {
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

struct CombinedCircuit<F: FieldExt> {
    mle_1_vec: Vec<DenseMle<F, F>>,
    mle_2_vec: Vec<DenseMle<F, F>>,
    mle_3_vec: Vec<DenseMle<F, F>>,
    mle_4_vec: Vec<DenseMle<F, F>>,
    mle_5: DenseMle<F, F>,
    mle_6: DenseMle<F, F>,
    mle_7: DenseMle<F, F>,
    mle_8: DenseMle<F, F>,
    num_data_parallel_bits: usize,
}

impl<F: FieldExt> GKRCircuit<F> for CombinedCircuit<F> {
    type Sponge = PoseidonSponge<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Sponge> {
        unimplemented!()
    }

    fn synthesize_and_commit(
        &mut self,
        transcript_writer: &mut TranscriptWriter<F, Self::Sponge>,
    ) -> Result<(Witness<F, Self::Sponge>, Vec<CommitmentEnum<F>>), GKRError> {
        let mut mle_1_combined = DenseMle::<F, F>::combine_mle_batch(self.mle_1_vec.clone());
        let mut mle_2_combined = DenseMle::<F, F>::combine_mle_batch(self.mle_2_vec.clone());
        let mut mle_3_combined = DenseMle::<F, F>::combine_mle_batch(self.mle_3_vec.clone());
        let mut mle_4_combined = DenseMle::<F, F>::combine_mle_batch(self.mle_4_vec.clone());
        mle_1_combined.layer_id = LayerId::Input(0);
        mle_2_combined.layer_id = LayerId::Input(0);
        mle_3_combined.layer_id = LayerId::Input(0);
        mle_4_combined.layer_id = LayerId::Input(0);

        let input_commit: Vec<Box<&mut dyn Mle<F>>> = vec![
            Box::new(&mut mle_1_combined),
            Box::new(&mut mle_2_combined),
            Box::new(&mut mle_3_combined),
            Box::new(&mut mle_4_combined),
            // Box::new(&mut self.mle_5),
            // Box::new(&mut self.mle_6),
            // Box::new(&mut self.mle_7),
            // Box::new(&mut self.mle_8),
        ];

        self.mle_1_vec
            .iter_mut()
            .zip(
                self.mle_2_vec
                    .iter_mut()
                    .zip(self.mle_3_vec.iter_mut().zip(self.mle_4_vec.iter_mut())),
            )
            .for_each(|(mle_1, (mle_2, (mle_3, mle_4)))| {
                mle_1.layer_id = LayerId::Input(0);
                mle_2.layer_id = LayerId::Input(0);
                mle_3.layer_id = LayerId::Input(0);
                mle_4.layer_id = LayerId::Input(0);
                mle_1.set_prefix_bits(Some(vec![MleIndex::Iterated; self.num_data_parallel_bits]));
                mle_2.set_prefix_bits(Some(vec![MleIndex::Iterated; self.num_data_parallel_bits]));
                mle_3.set_prefix_bits(Some(vec![MleIndex::Iterated; self.num_data_parallel_bits]));
                mle_4.set_prefix_bits(Some(vec![MleIndex::Iterated; self.num_data_parallel_bits]));
            });

        self.mle_5.layer_id = LayerId::Input(0);
        self.mle_6.layer_id = LayerId::Input(0);
        self.mle_7.layer_id = LayerId::Input(0);
        self.mle_8.layer_id = LayerId::Input(0);

        let input_commit_builder =
            InputLayerBuilder::<F>::new(input_commit, None, LayerId::Input(0));

        let input_layer: PublicInputLayer<F, Self::Sponge> =
            input_commit_builder.to_input_layer::<PublicInputLayer<F, Self::Sponge>>();

        let mut input_layer_enum = input_layer.to_enum();

        let input_layer_commit = input_layer_enum
            .commit()
            .map_err(|err| GKRError::InputLayerError(err))
            .unwrap();

        InputLayerEnum::prover_append_commitment_to_transcript(
            &input_layer_commit,
            transcript_writer,
        );
        let mut batched_circuit = BatchedCircuit {
            mle_1_vec: self.mle_1_vec.clone(),
            mle_2_vec: self.mle_2_vec.clone(),
            mle_3_vec: self.mle_3_vec.clone(),
            mle_4_vec: self.mle_4_vec.clone(),
        };
        let mut triple_nested_sel_circuit = TripleNestedSelectorCircuit {
            inner_inner_sel_mle: self.mle_5.clone(),
            inner_sel_mle: self.mle_6.clone(),
            outer_sel_mle: self.mle_7.clone(),
        };
        let mut product_scaled_circuit = ScaledProductCircuit {
            mle_1: self.mle_7.clone(),
            mle_2: self.mle_8.clone(),
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
            vec![
                batched_layers,
                // triple_nested_layers,
                // product_scaled_layers
            ],
            vec![
                batched_output_layers,
                // triple_output_layers,
                // product_scaled_output_layers,
            ],
        )
        .unwrap();

        Ok((
            Witness {
                layers,
                output_layers,
                input_layers: vec![input_layer_enum],
            },
            vec![input_layer_commit],
        ))
    }
}

#[test]
fn test_combined_circuit() {
    const VARS_MLE_1_2: usize = 2;
    const VARS_MLE_3: usize = VARS_MLE_1_2 + 1;
    const VARS_MLE_4: usize = VARS_MLE_3 + 1;
    const NUM_DATA_PARALLEL_BITS: usize = 3;

    let mle_1_vec = (0..1 << NUM_DATA_PARALLEL_BITS)
        .map(|_| get_dummy_random_mle(VARS_MLE_1_2))
        .collect_vec();
    let mle_2_vec = (0..1 << NUM_DATA_PARALLEL_BITS)
        .map(|_| get_dummy_random_mle(VARS_MLE_1_2))
        .collect_vec();
    let mle_3_vec = (0..1 << NUM_DATA_PARALLEL_BITS)
        .map(|_| get_dummy_random_mle(VARS_MLE_3))
        .collect_vec();
    let mle_4_vec = (0..1 << NUM_DATA_PARALLEL_BITS)
        .map(|_| get_dummy_random_mle(VARS_MLE_4))
        .collect_vec();
    let mle_5 = get_dummy_random_mle(VARS_MLE_1_2);
    let mle_6 = get_dummy_random_mle(VARS_MLE_1_2);
    let mle_7 = get_dummy_random_mle(VARS_MLE_3);
    let mle_8 = get_dummy_random_mle(VARS_MLE_4);

    let combined_circuit: CombinedCircuit<Fr> = CombinedCircuit {
        mle_1_vec,
        mle_2_vec,
        mle_3_vec,
        mle_4_vec,
        mle_5,
        mle_6,
        mle_7,
        mle_8,
        num_data_parallel_bits: NUM_DATA_PARALLEL_BITS,
    };
    test_circuit(combined_circuit, None)
}
