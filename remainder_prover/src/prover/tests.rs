use crate::expression::{
    generic_expr::{Expression, ExpressionNode, ExpressionType},
    prover_expr::ProverExpr,
};
use crate::{gate::gate::BinaryOperation, prover::helpers::test_circuit};
use ark_std::{end_timer, log2, start_timer, test_rng, One};
use itertools::{repeat_n, Itertools};
use rand::Rng;
use remainder_ligero::ligero_commit::remainder_ligero_commit_prove;
use serde_json::{from_reader, to_writer};

use std::{cmp::max, fs, iter::repeat_with, path::Path, time::Instant};

use crate::{
    layer::{
        batched::{combine_mles, combine_zero_mle_ref, BatchedLayer},
        empty_layer::EmptyLayer,
        from_mle,
        layer_enum::LayerEnum,
        simple_builders::{EqualityCheck, ZeroBuilder},
        LayerBuilder, LayerId,
    },
    mle::{
        dense::{DenseMle, Tuple2},
        zero::ZeroMleRef,
        Mle, MleIndex, MleRef,
    },
    prover::input_layer::enum_input_layer::CommitmentEnum,
    utils::get_random_mle,
};
use remainder_shared_types::{
    transcript::{poseidon_transcript::PoseidonTranscript, Transcript},
    FieldExt, Fr,
};

use super::{
    combine_layers::combine_layers,
    input_layer::{
        self, combine_input_layers::InputLayerBuilder, enum_input_layer::InputLayerEnum,
        ligero_input_layer::LigeroInputLayer, public_input_layer::PublicInputLayer,
        random_input_layer::RandomInputLayer, InputLayer,
    },
    test_helper_circuits::{EmptyLayerAddBuilder, EmptyLayerBuilder, EmptyLayerSubBuilder},
    GKRCircuit, GKRError, Layers, Witness,
};

/// This circuit is a 4 --> 2 circuit, such that
/// [x_1, x_2, x_3, x_4, (y_1, y_2)] --> [x_1 * x_3, x_2 * x_4] --> [x_1 * x_3 - y_1, x_2 * x_4 - y_2]
/// Note that we also have the difference thingy (of size 2)
struct SimpleCircuit<F: FieldExt> {
    mle: DenseMle<F, Tuple2<F>>,
    size: usize,
}
impl<F: FieldExt> GKRCircuit<F> for SimpleCircuit<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        // --- The input layer should just be the concatenation of `mle` and `output_input` ---
        let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut self.mle)];
        let mut input_layer =
            InputLayerBuilder::new(input_mles, Some(vec![self.size]), LayerId::Input(0));
        let mle_clone = &self.mle;

        // --- Create Layers to be added to ---
        let mut layers = Layers::new();

        // --- Create a SimpleLayer from the first `mle` within the circuit ---
        let mult_builder = from_mle(
            mle_clone,
            // --- The expression is a simple product between the first and second halves ---
            |mle| Expression::products(vec![mle.first(), mle.second()]),
            // --- The witness generation simply zips the two halves and multiplies them ---
            |mle, layer_id, prefix_bits| {
                DenseMle::new_from_iter(
                    mle.into_iter()
                        .map(|Tuple2((first, second))| first * second),
                    layer_id,
                    prefix_bits,
                )
            },
        );

        // --- Stacks the two aforementioned layers together into a single layer ---
        // --- Then adds them to the overall circuit ---
        let first_layer_output = layers.add_gkr(mult_builder);

        // --- Ahh. So we're doing the thing where we add the "real" circuit output as a circuit input, ---
        // --- then check if the difference between the two is zero ---
        let mut output_input =
            DenseMle::new_from_iter(first_layer_output.into_iter(), LayerId::Input(0), None);

        // --- Index the input-output layer ONLY for the input ---
        let _ = input_layer.add_extra_mle(Box::new(&mut output_input));

        // --- The input layer should just be the concatenation of `mle` and `output_input` ---
        // The input layer is ready at this point!
        let input_layer: LigeroInputLayer<F, Self::Transcript> =
            input_layer.to_input_layer_with_rho_inv(4, 1.);
        let input_layers = vec![input_layer.to_enum()];

        // --- Subtract the computed circuit output from the advice circuit output ---
        let output_diff_builder = from_mle(
            (first_layer_output, output_input.clone()),
            |(mle1, mle2)| mle1.mle_ref().expression() - mle2.mle_ref().expression(),
            |(mle1, mle2), layer_id, prefix_bits| {
                let num_vars = max(mle1.num_iterated_vars(), mle2.num_iterated_vars());
                ZeroMleRef::new(num_vars, prefix_bits, layer_id)
            },
        );

        // --- Add this final layer to the circuit ---
        let circuit_output = layers.add_gkr(output_diff_builder);

        Witness {
            layers,
            output_layers: vec![circuit_output.get_enum()],
            input_layers,
        }
    }
}

struct CircuitNoLinearIndex<F: FieldExt> {
    mle_1: DenseMle<F, F>,
    mle_2: DenseMle<F, F>,
    neg_mle_2: DenseMle<F, F>,
    size: usize,
}
impl<F: FieldExt> GKRCircuit<F> for CircuitNoLinearIndex<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        // --- The input layer should just be the concatenation of `mle` and `output_input` ---
        let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![
            Box::new(&mut self.mle_1),
            Box::new(&mut self.mle_2),
            Box::new(&mut self.neg_mle_2),
        ];
        let mut input_layer =
            InputLayerBuilder::new(input_mles, Some(vec![self.size]), LayerId::Input(0));

        // --- Create Layers to be added to ---
        let mut layers = Layers::new();

        // --- Create a SimpleLayer from the first `mle` within the circuit ---
        let mult_builder = from_mle(
            (self.mle_1.clone(), self.mle_2.clone()),
            // --- The expression is a simple product between the first and second halves ---
            |(mle_1, mle_2)| Expression::products(vec![mle_1.mle_ref(), mle_2.mle_ref()]),
            // --- The witness generation simply zips the two halves and multiplies them ---
            |(mle_1, mle_2), layer_id, prefix_bits| {
                DenseMle::new_from_iter(
                    mle_1
                        .mle_ref()
                        .bookkeeping_table()
                        .into_iter()
                        .zip(mle_2.mle_ref().bookkeeping_table())
                        .map(|(first, second)| *first * second),
                    layer_id,
                    prefix_bits,
                )
            },
        );

        // --- Create a SimpleLayer from the first `mle` within the circuit ---
        let mult_builder_neg = from_mle(
            (self.mle_1.clone(), self.neg_mle_2.clone()),
            // --- The expression is a simple product between the first and second halves ---
            |(mle_1, mle_2)| Expression::products(vec![mle_1.mle_ref(), mle_2.mle_ref()]),
            // --- The witness generation simply zips the two halves and multiplies them ---
            |(mle_1, mle_2), layer_id, prefix_bits| {
                DenseMle::new_from_iter(
                    mle_1
                        .mle_ref()
                        .bookkeeping_table()
                        .into_iter()
                        .zip(mle_2.mle_ref().bookkeeping_table())
                        .map(|(first, second)| *first * second),
                    layer_id,
                    prefix_bits,
                )
            },
        );

        // --- Stacks the two aforementioned layers together into a single layer ---
        // --- Then adds them to the overall circuit ---
        let first_layer_output = layers.add_gkr(mult_builder);
        let second_layer_output = layers.add_gkr(mult_builder_neg);

        let input_layer: PublicInputLayer<F, Self::Transcript> = input_layer.to_input_layer();
        let input_layers = vec![input_layer.to_enum()];

        // --- Subtract the computed circuit output from the advice circuit output ---
        let output_diff_builder = from_mle(
            (first_layer_output, second_layer_output.clone()),
            |(mle1, mle2)| mle1.mle_ref().expression() - mle2.mle_ref().expression(),
            |(mle1, mle2), layer_id, prefix_bits| {
                let num_vars = max(mle1.num_iterated_vars(), mle2.num_iterated_vars());
                ZeroMleRef::new(num_vars, prefix_bits, layer_id)
            },
        );

        // --- Add this final layer to the circuit ---
        let circuit_output = layers.add_gkr(output_diff_builder);

        Witness {
            layers,
            output_layers: vec![circuit_output.get_enum()],
            input_layers,
        }
    }
}

/// Circuit which just subtracts its two halves! No input-output layer needed.
struct SimplestCircuit<F: FieldExt> {
    mle: DenseMle<F, Tuple2<F>>,
}
impl<F: FieldExt> GKRCircuit<F> for SimplestCircuit<F> {
    type Transcript = PoseidonTranscript<F>;

    // const CIRCUIT_HASH: Option<[u8; 32]> = Some([
    //     201,181,0,14,124,41,18,30,207,198,237,142,57,140,114,224,28,140,62,0,109,36,200,27,208,218,32,166,8,35,115,46,
    // ]);
    const CIRCUIT_HASH: Option<[u8; 32]> = None;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        // --- The input layer should just be the concatenation of `mle` and `output_input` ---
        let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut self.mle)];
        let input_layer = InputLayerBuilder::new(input_mles, None, LayerId::Input(0));
        let mle_clone = self.mle.clone();

        // --- Create Layers to be added to ---
        let mut layers: Layers<F, Self::Transcript> = Layers::new();

        // --- Create a SimpleLayer from the first `mle` within the circuit ---
        let diff_builder = from_mle(
            mle_clone,
            // --- The expression is a simple diff between the first and second halves ---
            |mle| {
                let first_half = Box::new(Expression::mle(mle.first()));
                let second_half = Box::new(Expression::mle(mle.second()));
                let negated_second_half = Box::new(Expression::negated(second_half));
                Expression::sum(first_half, negated_second_half)
            },
            // --- The witness generation simply zips the two halves and subtracts them ---
            |mle, layer_id, prefix_bits| {
                // DenseMle::new_from_iter(
                //     mle.into_iter()
                //         .map(|Tuple2((first, second))| first - second),
                //     layer_id,
                //     prefix_bits,
                // )
                // --- The output SHOULD be all zeros ---
                let num_vars = max(mle.first().num_vars(), mle.second().num_vars());
                ZeroMleRef::new(num_vars, prefix_bits, layer_id)
            },
        );

        // --- Stacks the two aforementioned layers together into a single layer ---
        // --- Then adds them to the overall circuit ---
        let first_layer_output = layers.add_gkr(diff_builder);

        // --- The input layer should just be the concatenation of `mle` and `output_input` ---
        let input_layer: LigeroInputLayer<F, Self::Transcript> =
            input_layer.to_input_layer_with_rho_inv(4, 1.);

        Witness {
            layers,
            output_layers: vec![first_layer_output.get_enum()],
            input_layers: vec![input_layer.to_enum()],
        }
    }
}

/// Circuit which just subtracts its two halves! No input-output layer needed.
struct SimplestBatchedCircuit<F: FieldExt> {
    batched_first_second_mle: Vec<DenseMle<F, Tuple2<F>>>,
    batch_bits: usize,
}
impl<F: FieldExt> GKRCircuit<F> for SimplestBatchedCircuit<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        // --- Grab combined
        let mut combined_batched_first_second_mle =
            DenseMle::<F, Tuple2<F>>::combine_mle_batch(self.batched_first_second_mle.clone());
        let input_mles: Vec<Box<&mut dyn Mle<F>>> =
            vec![Box::new(&mut combined_batched_first_second_mle)];
        let input_layer_builder = InputLayerBuilder::new(input_mles, None, LayerId::Input(0));

        let num_dataparallel_circuit_copies = self.batched_first_second_mle.len();
        let num_dataparallel_bits = log2(num_dataparallel_circuit_copies) as usize;

        // --- Create Layers to be added to ---
        let mut layers: Layers<F, Self::Transcript> = Layers::new();

        // --- Create a SimpleLayer from the first `mle` within the circuit ---
        let diff_builders = self
            .batched_first_second_mle
            .iter_mut()
            .map(|mle| {
                // --- First add batching bits to the MLE (this is a hacky fix and will be removed) ---
                mle.set_prefix_bits(Some(
                    combined_batched_first_second_mle
                        .get_prefix_bits()
                        .iter()
                        .flatten()
                        .cloned()
                        .chain(repeat_n(MleIndex::Iterated, num_dataparallel_bits))
                        .collect_vec(),
                ));

                from_mle(
                    mle,
                    // --- The expression is a simple diff between the first and second halves ---
                    |mle| {
                        let first_half = Expression::mle(mle.first());
                        let second_half = Expression::mle(mle.second());
                        first_half - second_half
                    },
                    // --- The witness generation simply zips the two halves and subtracts them ---
                    |mle, layer_id, prefix_bits| {
                        // let hi = DenseMle::new_from_iter(
                        //     mle.into_iter()
                        //         .map(|Tuple2((first, second))| first - second),
                        //     layer_id,
                        //     prefix_bits.clone(),
                        // );
                        // dbg!(&hi.mle_ref().bookkeeping_table);
                        // --- The output SHOULD be all zeros ---
                        let num_vars = max(mle.first().num_vars(), mle.second().num_vars());
                        ZeroMleRef::new(num_vars, prefix_bits, layer_id)
                    },
                )
            })
            .collect_vec();

        // --- Convert the vector of builders into a batched builder which can be added to `layers` ---
        let batched_builder = BatchedLayer::new(diff_builders);
        let batched_result = layers.add_gkr(batched_builder);
        let batched_zero = combine_zero_mle_ref(batched_result);

        // --- The input layer should just be the concatenation of `mle` and `output_input` ---
        let input_layer: PublicInputLayer<F, Self::Transcript> =
            input_layer_builder.to_input_layer();

        Witness {
            layers,
            output_layers: vec![batched_zero.get_enum()],
            input_layers: vec![input_layer.to_enum()],
        }
    }
}

/// This circuit checks how RandomLayer works by multiplying the MLE by a constant,
/// taking in that result as advice in a publiclayer and doing an equality check
/// on the result of the mult and the advice
struct RandomCircuit<F: FieldExt> {
    mle: DenseMle<F, F>,
}

impl<F: FieldExt> GKRCircuit<F> for RandomCircuit<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        unimplemented!()
    }

    fn synthesize_and_commit(
        &mut self,
        transcript: &mut Self::Transcript,
    ) -> Result<(Witness<F, Self::Transcript>, Vec<CommitmentEnum<F>>), GKRError> {
        let input_layer: LigeroInputLayer<F, _> =
            InputLayerBuilder::new(vec![Box::new(&mut self.mle)], None, LayerId::Input(0))
                .to_input_layer_with_rho_inv(4, 1.);
        let mut input = input_layer.to_enum();

        let input_commit = input.commit().map_err(GKRError::InputLayerError)?;
        InputLayerEnum::append_commitment_to_transcript(&input_commit, transcript).unwrap();

        let random = RandomInputLayer::new(transcript, 1, LayerId::Input(1));
        let random_mle = random.get_mle();
        let mut random = random.to_enum();
        let random_commit = random.commit().map_err(GKRError::InputLayerError)?;

        let mut layers = Layers::new();

        let layer_1 = from_mle(
            (self.mle.clone(), random_mle),
            |(mle, random)| Expression::products(vec![mle.mle_ref(), random.mle_ref()]),
            |(mle, random), layer_id, prefix_bits| {
                DenseMle::new_from_iter(
                    mle.into_iter()
                        .zip(random.into_iter().cycle())
                        .map(|(item, random)| item * random),
                    layer_id,
                    prefix_bits,
                )
            },
        );

        let output = layers.add_gkr(layer_1);

        let mut output_input = output.clone();
        output_input.layer_id = LayerId::Input(2);
        let mut input_layer_2 =
            InputLayerBuilder::new(vec![Box::new(&mut output_input)], None, LayerId::Input(2))
                .to_input_layer::<PublicInputLayer<F, _>>()
                .to_enum();
        let input_layer_2_commit = input_layer_2.commit().map_err(GKRError::InputLayerError)?;
        InputLayerEnum::append_commitment_to_transcript(&input_layer_2_commit, transcript).unwrap();

        let layer_2 = EqualityCheck::new(output, output_input);
        let output = layers.add_gkr(layer_2);

        Ok((
            Witness {
                layers,
                output_layers: vec![output.get_enum()],
                input_layers: vec![input, random, input_layer_2],
            },
            vec![input_commit, random_commit, input_layer_2_commit],
        ))
    }
}

/// This circuit has two separate input layers, each with two MLEs inside, where
/// the MLEs within the input layer are the same size but the input layers themselves
/// are different sizes.
///
/// The MLEs within each input layer are first added together, then their results
/// are added. The final layer is just a ZeroLayerBuilder (i.e. subtracts the final
/// layer from itself for convenience).
///
/// TODO!(ryancao): If this still doesn't fail, change the MLEs within each input layer
///     to be different sizes and see if it does
/// TODO!(ryancao): If this still doesn't fail, make it batched and see if it fails then
struct MultiInputLayerCircuit<F: FieldExt> {
    input_layer_1_mle_1: DenseMle<F, F>,
    input_layer_1_mle_2: DenseMle<F, F>,

    input_layer_2_mle_1: DenseMle<F, F>,
    input_layer_2_mle_2: DenseMle<F, F>,
}
impl<F: FieldExt> GKRCircuit<F> for MultiInputLayerCircuit<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        unimplemented!()
    }

    fn synthesize_and_commit(
        &mut self,
        transcript: &mut Self::Transcript,
    ) -> Result<(Witness<F, Self::Transcript>, Vec<CommitmentEnum<F>>), GKRError> {
        // --- Publicly commit to each input layer ---
        let mut input_layer_1 = InputLayerBuilder::new(
            vec![
                Box::new(&mut self.input_layer_1_mle_1),
                Box::new(&mut self.input_layer_1_mle_2),
            ],
            None,
            LayerId::Input(0),
        )
        .to_input_layer::<PublicInputLayer<F, _>>()
        .to_enum();
        let input_layer_1_commitment = input_layer_1.commit().map_err(GKRError::InputLayerError)?;
        InputLayerEnum::append_commitment_to_transcript(&input_layer_1_commitment, transcript)
            .unwrap();

        // --- Second input layer (public) commitment ---
        let mut input_layer_2 = InputLayerBuilder::new(
            vec![
                Box::new(&mut self.input_layer_2_mle_1),
                Box::new(&mut self.input_layer_2_mle_2),
            ],
            None,
            LayerId::Input(1),
        )
        .to_input_layer::<PublicInputLayer<F, _>>()
        .to_enum();
        let input_layer_2_commitment = input_layer_2.commit().map_err(GKRError::InputLayerError)?;
        InputLayerEnum::append_commitment_to_transcript(&input_layer_2_commitment, transcript)
            .unwrap();

        let mut layers = Layers::new();

        // --- Add the first input layer MLEs to one another ---
        let layer_1 = from_mle(
            // Lol this hack though
            (
                self.input_layer_1_mle_1.clone(),
                self.input_layer_1_mle_2.clone(),
            ),
            |(input_layer_1_mle_1, input_layer_1_mle_2)| {
                let input_layer_1_mle_1_expr_ptr =
                    Box::new(Expression::mle(input_layer_1_mle_1.mle_ref()));
                let input_layer_1_mle_2_expr_ptr =
                    Box::new(Expression::mle(input_layer_1_mle_2.mle_ref()));
                Expression::sum(input_layer_1_mle_1_expr_ptr, input_layer_1_mle_2_expr_ptr)
            },
            |(input_layer_1_mle_1, input_layer_1_mle_2), layer_id, prefix_bits| {
                DenseMle::new_from_iter(
                    input_layer_1_mle_1
                        .into_iter()
                        .zip(input_layer_1_mle_2.into_iter().cycle())
                        .map(|(input_layer_1_mle_1_elem, input_layer_1_mle_2_elem)| {
                            input_layer_1_mle_1_elem + input_layer_1_mle_2_elem
                        }),
                    layer_id,
                    prefix_bits,
                )
            },
        );

        // --- Add the second input layer MLEs to one another ---
        let layer_2 = from_mle(
            // Lol this hack though
            (
                self.input_layer_2_mle_1.clone(),
                self.input_layer_2_mle_2.clone(),
            ),
            |(input_layer_2_mle_1, input_layer_2_mle_2)| {
                let input_layer_2_mle_1_expr_ptr =
                    Box::new(Expression::mle(input_layer_2_mle_1.mle_ref()));
                let input_layer_2_mle_2_expr_ptr =
                    Box::new(Expression::mle(input_layer_2_mle_2.mle_ref()));
                dbg!(input_layer_2_mle_1.layer_id);
                dbg!(input_layer_2_mle_2.layer_id);
                Expression::sum(input_layer_2_mle_1_expr_ptr, input_layer_2_mle_2_expr_ptr)
            },
            |(input_layer_2_mle_1, input_layer_2_mle_2), layer_id, prefix_bits| {
                DenseMle::new_from_iter(
                    input_layer_2_mle_1
                        .into_iter()
                        .zip(input_layer_2_mle_2.into_iter().cycle())
                        .map(|(input_layer_2_mle_1_elem, input_layer_2_mle_2_elem)| {
                            input_layer_2_mle_1_elem + input_layer_2_mle_2_elem
                        }),
                    layer_id,
                    prefix_bits,
                )
            },
        );

        let first_layer_output = layers.add_gkr(layer_1);
        let second_layer_output = layers.add_gkr(layer_2);

        // --- Next layer should take the two and add them ---
        let layer_3 = from_mle(
            // Lol this hack though
            (first_layer_output, second_layer_output),
            |(first_layer_output_mle_param, second_layer_output_mle_param)| {
                let first_layer_output_mle_param_expr_ptr =
                    Box::new(Expression::mle(first_layer_output_mle_param.mle_ref()));
                let second_layer_output_mle_param_expr_ptr =
                    Box::new(Expression::mle(second_layer_output_mle_param.mle_ref()));
                Expression::sum(
                    first_layer_output_mle_param_expr_ptr,
                    second_layer_output_mle_param_expr_ptr,
                )
            },
            |(first_layer_output_mle_param, second_layer_output_mle_param),
             layer_id,
             prefix_bits| {
                DenseMle::new_from_iter(
                    first_layer_output_mle_param
                        .into_iter()
                        .zip(second_layer_output_mle_param.into_iter().cycle())
                        .map(
                            |(
                                first_layer_output_mle_param_elem,
                                second_layer_output_mle_param_elem,
                            )| {
                                first_layer_output_mle_param_elem
                                    + second_layer_output_mle_param_elem
                            },
                        ),
                    layer_id,
                    prefix_bits,
                )
            },
        );
        let third_layer_output = layers.add_gkr(layer_3);

        // --- Subtract the last layer from itself so we get all zeros ---
        let layer_4 = EqualityCheck::new(third_layer_output.clone(), third_layer_output);
        let fourth_layer_output = layers.add_gkr(layer_4);

        Ok((
            Witness {
                layers,
                output_layers: vec![fourth_layer_output.get_enum()],
                input_layers: vec![input_layer_1, input_layer_2],
            },
            vec![input_layer_1_commitment, input_layer_2_commitment],
        ))
    }
}
impl<F: FieldExt> MultiInputLayerCircuit<F> {
    /// Constructor
    pub fn new(
        input_layer_1_mle_1: DenseMle<F, F>,
        input_layer_1_mle_2: DenseMle<F, F>,
        input_layer_2_mle_1: DenseMle<F, F>,
        input_layer_2_mle_2: DenseMle<F, F>,
    ) -> Self {
        Self {
            input_layer_1_mle_1,
            input_layer_1_mle_2,
            input_layer_2_mle_1,
            input_layer_2_mle_2,
        }
    }
}

/// This circuit is a 4k --> k circuit, such that
/// [x_1, x_2, x_3, x_4] --> [x_1 * x_3, x_2 + x_4] --> [(x_1 * x_3) - (x_2 + x_4)]
struct TestCircuit<F: FieldExt> {
    mle: DenseMle<F, Tuple2<F>>,
    mle_2: DenseMle<F, Tuple2<F>>,
    size: usize,
}

impl<F: FieldExt> GKRCircuit<F> for TestCircuit<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        // --- The input layer should just be the concatenation of `mle`, `mle_2`, and `output_input` ---
        // let mut self_mle_clone = self.mle.clone();
        // let mut self_mle_2_clone = self.mle_2.clone();
        let input_mles: Vec<Box<&mut dyn Mle<F>>> =
            vec![Box::new(&mut self.mle), Box::new(&mut self.mle_2)];
        let mut input_layer =
            InputLayerBuilder::new(input_mles, Some(vec![self.size]), LayerId::Input(0));
        let mle_clone = self.mle.clone();
        let mle_2_clone = self.mle_2.clone();

        // --- Create Layers to be added to ---

        // --- Create Layers to be added to ---
        let mut layers: Layers<F, Self::Transcript> = Layers::new();

        // --- Create a SimpleLayer from the first `mle` within the circuit ---
        let builder = from_mle(
            mle_clone,
            // --- The expression is a simple product between the first and second halves ---
            |mle| Expression::products(vec![mle.first(), mle.second()]),
            // --- The witness generation simply zips the two halves and multiplies them ---
            |mle, layer_id, prefix_bits| {
                DenseMle::new_from_iter(
                    mle.into_iter()
                        .map(|Tuple2((first, second))| first * second),
                    layer_id,
                    prefix_bits,
                )
            },
        );

        // --- Similarly here, but with addition between the two halves ---
        // --- Note that EACH of `mle` and `mle_2` are parts of the input layer ---
        let builder2 = from_mle(
            mle_2_clone,
            |mle| mle.first().expression() + mle.second().expression(),
            |mle, layer_id, prefix_bits| {
                DenseMle::new_from_iter(
                    mle.into_iter()
                        .map(|Tuple2((first, second))| first + second),
                    layer_id,
                    prefix_bits,
                )
            },
        );

        // --- Stacks the two aforementioned layers together into a single layer ---
        // --- Then adds them to the overall circuit ---
        let builder3 = builder.concat(builder2);
        let output = layers.add_gkr(builder3);

        // --- Creates a single layer which takes [x_1, ..., x_n, y_1, ..., y_n] and returns [x_1 - y_1, ..., x_n - y_n] ---
        let builder4 = from_mle(
            output,
            |(mle1, mle2)| mle1.mle_ref().expression() - mle2.mle_ref().expression(),
            |(mle1, mle2), layer_id, prefix_bits| {
                DenseMle::new_from_iter(
                    mle1.clone()
                        .into_iter()
                        .zip(mle2.clone().into_iter())
                        .map(|(first, second)| first - second),
                    layer_id,
                    prefix_bits,
                )
            },
        );

        // --- Appends this to the circuit ---
        let computed_output = layers.add_gkr(builder4);

        // --- Ahh. So we're doing the thing where we add the "real" circuit output as a circuit input, ---
        // --- then check if the difference between the two is zero ---
        let mut output_input =
            DenseMle::new_from_iter(computed_output.into_iter(), LayerId::Input(0), None);

        // --- Input layer should be finalized at this point ---
        let _ = input_layer.add_extra_mle(Box::new(&mut output_input));
        let input_layer: PublicInputLayer<F, Self::Transcript> = input_layer.to_input_layer();

        // --- Subtract the computed circuit output from the advice circuit output ---
        let builder5 = from_mle(
            (computed_output, output_input.clone()),
            |(mle1, mle2)| mle1.mle_ref().expression() - mle2.mle_ref().expression(),
            |(mle1, mle2), layer_id, prefix_bits| {
                let num_vars = max(mle1.num_iterated_vars(), mle2.num_iterated_vars());
                ZeroMleRef::new(num_vars, prefix_bits, layer_id)
            },
        );

        // --- Add this final layer to the circuit ---
        let circuit_circuit_output = layers.add_gkr(builder5);

        Witness {
            layers,
            output_layers: vec![circuit_circuit_output.get_enum()],
            input_layers: vec![input_layer.to_enum()],
        }
    }
}

/// Circuit which just subtracts its two halves with gate mle
struct SimplestGateCircuit<F: FieldExt> {
    mle: DenseMle<F, F>,
    negmle: DenseMle<F, F>,
}
impl<F: FieldExt> GKRCircuit<F> for SimplestGateCircuit<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        // --- The input layer should just be the concatenation of `mle` and `output_input` ---
        let input_mles: Vec<Box<&mut dyn Mle<F>>> =
            vec![Box::new(&mut self.mle), Box::new(&mut self.negmle)];
        let input_layer = InputLayerBuilder::new(input_mles, None, LayerId::Input(0))
            .to_input_layer::<PublicInputLayer<F, _>>()
            .to_enum();

        // --- Create Layers to be added to ---
        let mut layers = Layers::new();

        let mut nonzero_gates = vec![];
        let num_vars = 1 << self.mle.mle_ref().num_vars();

        (0..num_vars).for_each(|idx| {
            nonzero_gates.push((idx, idx, idx));
        });

        let first_layer_output = layers.add_gate(
            nonzero_gates,
            self.mle.mle_ref(),
            self.negmle.mle_ref(),
            None,
            BinaryOperation::Add,
        );

        let output_layer_builder = ZeroBuilder::new(first_layer_output);

        let output_layer_mle = layers.add_gkr(output_layer_builder);

        Witness {
            layers,
            output_layers: vec![output_layer_mle.get_enum()],
            input_layers: vec![input_layer],
        }
    }
}

/// Circuit which just subtracts its two halves with gate mle
struct SimplestGateCircuitUneven<F: FieldExt> {
    mle: DenseMle<F, F>,
    negmle: DenseMle<F, F>,
}
impl<F: FieldExt> GKRCircuit<F> for SimplestGateCircuitUneven<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        // --- The input layer should just be the concatenation of `mle` and `output_input` ---
        let input_mles: Vec<Box<&mut dyn Mle<F>>> =
            vec![Box::new(&mut self.mle), Box::new(&mut self.negmle)];
        let input_layer = InputLayerBuilder::new(input_mles, None, LayerId::Input(0))
            .to_input_layer::<PublicInputLayer<F, _>>()
            .to_enum();

        // --- Create Layers to be added to ---
        let mut layers = Layers::new();

        let mut nonzero_gates = vec![(0, 0, 0)];

        let first_layer_output = layers.add_gate(
            nonzero_gates,
            self.mle.mle_ref(),
            self.negmle.mle_ref(),
            None,
            BinaryOperation::Add,
        );

        let output_layer_builder = ZeroBuilder::new(first_layer_output);

        let output_layer_mle =
            layers.add::<_, EmptyLayer<F, Self::Transcript>>(output_layer_builder);

        Witness {
            layers,
            output_layers: vec![output_layer_mle.get_enum()],
            input_layers: vec![input_layer],
        }
    }
}

/// Circuit which just subtracts its two halves with gate mle
struct MulAddSimplestGateCircuit<F: FieldExt> {
    mle_1: DenseMle<F, F>,
    mle_2: DenseMle<F, F>,
    neg_mle_2: DenseMle<F, F>,
}
impl<F: FieldExt> GKRCircuit<F> for MulAddSimplestGateCircuit<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        // --- The input layer should just be the concatenation of `mle` and `output_input` ---
        let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![
            Box::new(&mut self.mle_1),
            Box::new(&mut self.mle_2),
            Box::new(&mut self.neg_mle_2),
        ];
        let input_layer = InputLayerBuilder::new(input_mles, None, LayerId::Input(0))
            .to_input_layer::<PublicInputLayer<F, _>>()
            .to_enum();

        // --- Create Layers to be added to ---
        let mut layers = Layers::new();

        let mut nonzero_gates = vec![];
        let num_vars = 1 << self.mle_1.mle_ref().num_vars();

        (0..num_vars).for_each(|idx| {
            nonzero_gates.push((idx, idx, idx));
        });

        let pos_mul_output = layers.add_gate(
            nonzero_gates.clone(),
            self.mle_1.mle_ref(),
            self.mle_2.mle_ref(),
            None,
            BinaryOperation::Mul,
        );

        let neg_mul_output = layers.add_gate(
            nonzero_gates.clone(),
            self.mle_1.mle_ref(),
            self.neg_mle_2.mle_ref(),
            None,
            BinaryOperation::Mul,
        );

        let add_gate_layer_output = layers.add_gate(
            nonzero_gates,
            pos_mul_output.mle_ref(),
            neg_mul_output.mle_ref(),
            None,
            BinaryOperation::Add,
        );

        let output_layer_builder = ZeroBuilder::new(add_gate_layer_output);

        let output_layer_mle = layers.add_gkr(output_layer_builder);

        // (layers, vec![first_layer_output.mle_ref().get_enum()], input_layer)
        Witness {
            layers,
            output_layers: vec![output_layer_mle.get_enum()],
            input_layers: vec![input_layer],
        }
    }
}

struct SimplestAddMulBatchedGateCircuit<F: FieldExt> {
    mle_1: DenseMle<F, F>,
    mle_2: DenseMle<F, F>,
    neg_mle_2: DenseMle<F, F>,
    batch_bits: usize,
}
impl<F: FieldExt> GKRCircuit<F> for SimplestAddMulBatchedGateCircuit<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        // --- The input layer should just be the concatenation of `mle` and `output_input` ---
        let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![
            Box::new(&mut self.mle_1),
            Box::new(&mut self.mle_2),
            Box::new(&mut self.neg_mle_2),
        ];
        let input_layer = InputLayerBuilder::new(input_mles, None, LayerId::Input(0))
            .to_input_layer::<PublicInputLayer<F, _>>()
            .to_enum();

        // --- Create Layers to be added to ---
        let mut layers = Layers::new();

        let mut nonzero_gates = vec![];
        let table_size = 1 << (self.neg_mle_2.mle_ref().num_vars() - self.batch_bits);

        (0..table_size).for_each(|idx| {
            nonzero_gates.push((idx, idx, idx));
        });

        let neg_mul_output = layers.add_gate(
            nonzero_gates.clone(),
            self.mle_1.mle_ref(),
            self.neg_mle_2.mle_ref(),
            Some(self.batch_bits),
            BinaryOperation::Mul,
        );

        let pos_mul_output = layers.add_gate(
            nonzero_gates.clone(),
            self.mle_1.mle_ref(),
            self.mle_2.mle_ref(),
            Some(self.batch_bits),
            BinaryOperation::Mul,
        );

        let add_gate_layer_output = layers.add_gate(
            nonzero_gates,
            pos_mul_output.mle_ref(),
            neg_mul_output.mle_ref(),
            Some(self.batch_bits),
            BinaryOperation::Add,
        );

        let output_layer_builder = ZeroBuilder::new(add_gate_layer_output);

        let output_layer_mle = layers.add_gkr(output_layer_builder);

        // (layers, vec![first_layer_output.mle_ref().get_enum()], input_layer)
        Witness {
            layers,
            output_layers: vec![output_layer_mle.get_enum()],
            input_layers: vec![input_layer],
        }
    }
}

/// Circuit which just subtracts its two halves with gate mle
struct SimplestGateCircuitCombined<F: FieldExt> {
    mle: DenseMle<F, F>,
    negmle: DenseMle<F, F>,
}
impl<F: FieldExt> GKRCircuit<F> for SimplestGateCircuitCombined<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        // --- The input layer should just be the concatenation of `mle` and `output_input` ---
        let input_mles: Vec<Box<&mut dyn Mle<F>>> =
            vec![Box::new(&mut self.mle), Box::new(&mut self.negmle)];
        let input_layer = InputLayerBuilder::new(input_mles, None, LayerId::Input(0))
            .to_input_layer::<PublicInputLayer<F, _>>()
            .to_enum();

        // --- Create Layers to be added to ---
        let mut layers = Layers::new();

        let mut nonzero_gates = vec![];
        let num_vars = 1 << self.mle.mle_ref().num_vars();

        (0..num_vars).for_each(|idx| {
            nonzero_gates.push((idx, idx, idx));
        });

        let first_layer_output = layers.add_gate(
            nonzero_gates,
            self.mle.mle_ref(),
            self.negmle.mle_ref(),
            None,
            BinaryOperation::Add,
        );

        let output_layer_builder = ZeroBuilder::new(first_layer_output);

        let output_layer_mle = layers.add_gkr(output_layer_builder);

        Witness {
            layers,
            output_layers: vec![output_layer_mle.get_enum()],
            input_layers: vec![input_layer],
        }
    }
}

/// Circuit which just subtracts its two halves with batched gate mle
struct SimplestBatchedGateCircuit<F: FieldExt> {
    mle: DenseMle<F, F>,
    negmle: DenseMle<F, F>,
    batch_bits: usize,
}
impl<F: FieldExt> GKRCircuit<F> for SimplestBatchedGateCircuit<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        // --- The input layer should just be the concatenation of `mle` and `output_input` ---
        let input_mles: Vec<Box<&mut dyn Mle<F>>> =
            vec![Box::new(&mut self.mle), Box::new(&mut self.negmle)];
        let input_layer = InputLayerBuilder::new(input_mles, None, LayerId::Input(0))
            .to_input_layer::<PublicInputLayer<F, _>>()
            .to_enum();

        // --- Create Layers to be added to ---
        let mut layers = Layers::new();

        let mut nonzero_gates = vec![];
        let table_size = 1 << (self.negmle.mle_ref().num_vars() - self.batch_bits);

        (0..table_size).for_each(|idx| {
            nonzero_gates.push((idx, idx, idx));
        });

        let first_layer_output = layers.add_gate(
            nonzero_gates,
            self.mle.mle_ref(),
            self.negmle.mle_ref(),
            Some(self.batch_bits),
            BinaryOperation::Add,
        );

        let output_layer_builder = ZeroBuilder::new(first_layer_output);

        let output_layer_mle = layers.add_gkr(output_layer_builder);

        // (layers, vec![first_layer_output.mle_ref().get_enum()], input_layer)
        Witness {
            layers,
            output_layers: vec![output_layer_mle.get_enum()],
            input_layers: vec![input_layer],
        }
    }
}

/// Circuit which subtracts its two halves, except for the part where one half is
/// comprised of a pre-committed Ligero input layer and the other half is comprised
/// of a Ligero input layer which is committed to on the spot.
///
/// The circuit itself produces independent claims on its two input MLEs, and is basically
/// two indpendent circuits via the fact that it basically subtracts each input MLE
/// from itself and calls that the output layer. In particular, this allows us to test
/// whether Halo2 generates the same VK given that we have the same pre-committed Ligero layer
/// but a DIFFERENT live-committed Ligero layer
struct SimplePrecommitCircuit<F: FieldExt> {
    mle: DenseMle<F, F>,
    mle2: DenseMle<F, F>,
}
impl<F: FieldExt> GKRCircuit<F> for SimplePrecommitCircuit<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        // --- The precommitted input layer MLE is just the first MLE ---
        let precommitted_input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut self.mle)];
        let precommitted_input_layer_builder =
            InputLayerBuilder::new(precommitted_input_mles, None, LayerId::Input(0));

        // --- The non-precommitted input layer MLE is just the second ---
        let live_committed_input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut self.mle2)];
        let live_committed_input_layer_builder =
            InputLayerBuilder::new(live_committed_input_mles, None, LayerId::Input(1));

        let mle_clone = self.mle.clone();
        let mle2_clone = self.mle2.clone();

        // --- Create Layers to be added to ---
        let mut layers: Layers<F, Self::Transcript> = Layers::new();

        // --- Create a SimpleLayer from the first `mle` within the circuit ---
        let diff_builder = from_mle(
            mle_clone.clone(),
            // --- The expression is a simple diff between the first and second halves ---
            |_mle| {
                let first_half = Expression::mle(_mle.mle_ref());
                let second_half = Expression::mle(_mle.mle_ref());
                first_half - second_half
            },
            // --- The output SHOULD be all zeros ---
            |_mle, layer_id, prefix_bits| {
                ZeroMleRef::new(mle_clone.mle_ref().num_vars(), prefix_bits, layer_id)
            },
        );

        // --- Similarly as the above, but with the circuit's second MLE ---
        let diff_builder_2 = from_mle(
            mle2_clone.clone(),
            // --- The expression is a simple diff between the first and second halves ---
            |_mle| {
                let first_half = Expression::mle(_mle.mle_ref());
                let second_half = Expression::mle(_mle.mle_ref());
                first_half - second_half
            },
            // --- The output SHOULD be all zeros ---
            |_mle, layer_id, prefix_bits| {
                ZeroMleRef::new(mle2_clone.mle_ref().num_vars(), prefix_bits, layer_id)
            },
        );

        // --- Stacks the two aforementioned layers together into a single layer ---
        // --- Then adds them to the overall circuit ---
        let first_layer_output_1 = layers.add_gkr(diff_builder);
        let first_layer_output_2 = layers.add_gkr(diff_builder_2);

        // --- We should have two input layers: a single pre-committed and a single regular Ligero layer ---
        let rho_inv = 4;
        let ratio = 1_f64;
        let (_, ligero_comm, ligero_root, ligero_aux) =
            remainder_ligero_commit_prove(&self.mle.mle, rho_inv, ratio);
        let precommitted_input_layer: LigeroInputLayer<F, Self::Transcript> =
            precommitted_input_layer_builder.to_input_layer_with_precommit(
                ligero_comm,
                ligero_aux,
                ligero_root,
                true,
            );
        let live_committed_input_layer: LigeroInputLayer<F, Self::Transcript> =
            live_committed_input_layer_builder.to_input_layer_with_rho_inv(4, 1.);

        Witness {
            layers,
            output_layers: vec![
                first_layer_output_1.get_enum(),
                first_layer_output_2.get_enum(),
            ],
            input_layers: vec![
                precommitted_input_layer.to_enum(),
                live_committed_input_layer.to_enum(),
            ],
        }
    }
}

/// Circuit which has an empty layer as an intermediate layer with multiple claims going both
/// to and from it, to thoroughly test the expected behavior of the `EmptyLayer`.
///
/// Note that all three MLEs have size 2! The structure of the circuit is as follows:
/// * The two empty layer src MLEs' elements are each multiplied together and then added between
/// the two to yield a single empty layer.
/// * Next, that empty layer's MLE is subtracted from the first MLE
/// * Finally, that empty layer's MLE is added to the second MLE
struct EmptyLayerTestCircuit<F: FieldExt> {
    mle: DenseMle<F, F>,
    mle2: DenseMle<F, F>,
    empty_layer_src_mle: DenseMle<F, F>,
    other_empty_layer_src_mle: DenseMle<F, F>,
}
impl<F: FieldExt> GKRCircuit<F> for EmptyLayerTestCircuit<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        // --- We're not testing commitments here; just use PublicInputLayer ---
        let input_layer = InputLayerBuilder::new(
            vec![
                Box::new(&mut self.mle),
                Box::new(&mut self.mle2),
                Box::new(&mut self.empty_layer_src_mle),
                Box::new(&mut self.other_empty_layer_src_mle),
            ],
            None,
            LayerId::Input(0),
        )
        .to_input_layer::<PublicInputLayer<F, _>>();

        // --- Create Layers to be added to ---
        let mut layers: Layers<F, Self::Transcript> = Layers::new();

        // --- Creates empty layer and adds to circuit ---
        let empty_layer_builder = EmptyLayerBuilder::new(
            self.empty_layer_src_mle.clone(),
            self.other_empty_layer_src_mle.clone(),
        );
        let empty_layer_result =
            layers.add::<_, EmptyLayer<F, Self::Transcript>>(empty_layer_builder);

        // --- Subtracts from `self.mle` ---
        let sub_builder = EmptyLayerSubBuilder::new(empty_layer_result.clone(), self.mle.clone());
        let sub_result = layers.add_gkr(sub_builder);

        // --- Adds to `self.mle2` ---
        let add_builder = EmptyLayerAddBuilder::new(empty_layer_result, self.mle2.clone());
        let add_result = layers.add_gkr(add_builder);

        Witness {
            layers,
            output_layers: vec![sub_result.get_enum(), add_result.get_enum()],
            input_layers: vec![input_layer.to_enum()],
        }
    }
}

struct CombineCircuit<F: FieldExt> {
    test_circuit: TestCircuit<F>,
    simple_circuit: SimpleCircuit<F>,
}

impl<F: FieldExt> GKRCircuit<F> for CombineCircuit<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        let test_witness = self.test_circuit.synthesize();
        let simple_witness = self.simple_circuit.synthesize();

        let Witness {
            layers: test_layers,
            output_layers: test_outputs,
            input_layers: test_inputs,
        } = test_witness;

        let Witness {
            layers: mut simple_layers,
            output_layers: simple_outputs,
            input_layers: simple_inputs,
        } = simple_witness;

        // for input vv
        let input_layers: Vec<InputLayerEnum<F, PoseidonTranscript<F>>> = test_inputs
            .into_iter()
            .chain(simple_inputs.into_iter().map(|mut input| {
                let new_layer_id = match input.layer_id() {
                    LayerId::Input(id) => LayerId::Input(id + 1),
                    LayerId::RandomInput(id) => LayerId::RandomInput(id + 1),
                    LayerId::Layer(_) => panic!(),
                    LayerId::Output(_) => panic!(),
                };
                input.set_layer_id(new_layer_id);
                input
            }))
            .collect();

        for layer in simple_layers.0.iter_mut() {
            let expression = match layer {
                LayerEnum::Gkr(layer) => &mut layer.expression,
                LayerEnum::EmptyLayer(layer) => &mut layer.expr,
                _ => panic!(),
            };

            let mut closure =
                for<'a, 'b> |expr: &'a mut ExpressionNode<F, ProverExpr>,
                             mle_vec: &'b mut <ProverExpr as ExpressionType<F>>::MleVec|
                             -> Result<(), ()> {
                    match expr {
                        ExpressionNode::Mle(mle_vec_idx) => {
                            let mle_ref = mle_vec_idx.get_mle_mut(mle_vec);

                            if mle_ref.layer_id == LayerId::Input(0) {
                                mle_ref.layer_id = LayerId::Input(1)
                            }
                            Ok(())
                        }
                        ExpressionNode::Product(mle_vec_indices) => {
                            for mle_vec_index in mle_vec_indices {
                                let mle_ref = mle_vec_index.get_mle_mut(mle_vec);

                                if mle_ref.layer_id == LayerId::Input(0) {
                                    mle_ref.layer_id = LayerId::Input(1)
                                }
                            }
                            Ok(())
                        }
                        ExpressionNode::Constant(_)
                        | ExpressionNode::Scaled(_, _)
                        | ExpressionNode::Sum(_, _)
                        | ExpressionNode::Negated(_)
                        | ExpressionNode::Selector(_, _, _) => Ok(()),
                    }
                };

            expression.traverse_mut(&mut closure).unwrap();
        }

        // for input ^^
        let (layers, output_layers) = combine_layers(
            vec![test_layers, simple_layers],
            vec![test_outputs, simple_outputs],
        )
        .unwrap();

        Witness {
            layers,
            output_layers,
            input_layers,
        }
    }
}
// ------------------------------------ BASIC REGULAR CIRCUITS ------------------------------------

#[test]
fn test_gkr_simple_circuit() {
    let mut rng = test_rng();
    let size = 5;

    // --- This should be 2^2 ---
    let mle: DenseMle<Fr, Tuple2<Fr>> = DenseMle::new_from_iter(
        (0..1 << 5).map(|_| (Fr::from(rng.gen::<u64>()), Fr::from(rng.gen::<u64>())).into()),
        LayerId::Input(0),
        None,
    );

    let circuit: SimpleCircuit<Fr> = SimpleCircuit { mle, size };
    test_circuit(circuit, Some(Path::new("simple_circuit_optimized.json")));
}

#[test]
fn test_gkr_nonlinear_circuit() {
    let mut rng = test_rng();
    let size = 2;

    let mle_1: DenseMle<Fr, Fr> = DenseMle::new_from_iter(
        (0..1 << size).map(|_| Fr::from(rng.gen::<u64>())),
        LayerId::Input(0),
        None,
    );

    let mle_2: DenseMle<Fr, Fr> = DenseMle::new_from_iter(
        (0..1 << size).map(|_| Fr::from(rng.gen::<u64>())),
        LayerId::Input(0),
        None,
    );

    let neg_mle_2: DenseMle<Fr, Fr> = DenseMle::new_from_iter(
        mle_2
            .mle_ref()
            .bookkeeping_table()
            .iter()
            .map(|elem| elem.neg()),
        LayerId::Input(0),
        None,
    );

    let circuit: CircuitNoLinearIndex<Fr> = CircuitNoLinearIndex {
        mle_1,
        mle_2,
        neg_mle_2,
        size,
    };
    test_circuit(circuit, Some(Path::new("simple_circuit_optimized.json")));
}

#[test]
fn test_gkr_simplest_circuit() {
    let mut rng = test_rng();
    let size = 1 << 4;

    // --- This should be 2^2 ---
    let mle: DenseMle<Fr, Tuple2<Fr>> = DenseMle::new_from_iter(
        (0..size).map(|_| {
            let num = Fr::from(rng.gen::<u64>());
            //let second_num = Fr::from(rng.gen::<u64>());
            (num, num).into()
        }),
        LayerId::Input(0),
        None,
    );

    let mut circuit: SimplestCircuit<Fr> = SimplestCircuit { mle };
    dbg!(circuit.gen_circuit_hash().to_bytes());
    test_circuit(circuit, Some(Path::new("simplest_circuit_optimized.json")));
}

#[test]
fn test_test_circuit() {
    let mut rng = test_rng();
    let size = 4;
    let size_expanded = 1 << size;

    // --- This should be 2^2 ---
    let mle: DenseMle<Fr, Tuple2<Fr>> = DenseMle::new_from_iter(
        (0..size_expanded).map(|_| (Fr::from(rng.gen::<u64>()), Fr::from(rng.gen::<u64>())).into()),
        LayerId::Input(0),
        None,
    );
    // --- This should be 2^2 ---
    let mle_2: DenseMle<Fr, Tuple2<Fr>> = DenseMle::new_from_iter(
        (0..size_expanded).map(|_| (Fr::from(rng.gen::<u64>()), Fr::from(rng.gen::<u64>())).into()),
        LayerId::Input(0),
        None,
    );

    let circuit: TestCircuit<Fr> = TestCircuit { mle, mle_2, size };

    test_circuit(circuit, Some(Path::new("./gkr_proof_optimized.json")));
}

// ------------------------------------ REGULAR DATAPARALLEL TESTING CIRCUITS ------------------------------------

#[test]
fn test_gkr_simplest_batched_circuit() {
    let mut rng = test_rng();
    let size = 1 << 3;

    let batch_size = 1 << 2;
    // --- This should be 2^2 ---
    let batched_mle: Vec<DenseMle<Fr, Tuple2<Fr>>> = (0..batch_size)
        .map(|_idx1| {
            DenseMle::new_from_iter(
                (0..size).map(|_idx| {
                    let num = Fr::from(rng.gen::<u64>());
                    //let second_num = Fr::from(rng.gen::<u64>());
                    // let num = Fr::from(idx + idx1);
                    (num, num).into()
                }),
                LayerId::Input(0),
                None,
            )
        })
        .collect_vec();
    // let mle: DenseMle<Fr, Tuple2<Fr>> = DenseMle::new_from_iter(
    //     (0..size).map(|idx| (Fr::from(idx + 1), Fr::from(idx + 1)).into()),
    //     LayerId::Input(0),
    //     None,
    // );

    let circuit: SimplestBatchedCircuit<Fr> = SimplestBatchedCircuit {
        batched_first_second_mle: batched_mle,
        batch_bits: 2,
    };
    test_circuit(circuit, None);
}

// ------------------------------------ COMMITMENT TESTING CIRCUITS ------------------------------------

#[test]
fn test_gkr_circuit_with_precommit() {
    let mut rng = test_rng();
    let size = 1 << 5;

    // --- MLE contents ---
    let items = repeat_with(|| Fr::from(rng.gen::<u64>()))
        .take(size)
        .collect_vec();
    let items2 = repeat_with(|| Fr::from(rng.gen::<u64>()))
        .take(size)
        .collect_vec();

    let mle: DenseMle<Fr, Fr> = DenseMle::new_from_raw(items, LayerId::Input(0), None);
    let mle2: DenseMle<Fr, Fr> = DenseMle::new_from_raw(items2, LayerId::Input(1), None);

    let circuit: SimplePrecommitCircuit<Fr> = SimplePrecommitCircuit { mle, mle2 };

    test_circuit(
        circuit,
        Some(Path::new("./gkr_proof_with_precommit_optimized.json")),
    );
}

// ------------------------------------ INPUT LAYER TESTING CIRCUITS ------------------------------------

#[test]
fn test_multiple_input_layers_circuit() {
    let mut rng = test_rng();
    let input_layer_1_mle_1 = get_random_mle::<Fr>(3, &mut rng);
    let input_layer_1_mle_2 = get_random_mle::<Fr>(2, &mut rng);

    let mut input_layer_2_mle_1 = get_random_mle::<Fr>(2, &mut rng);
    let mut input_layer_2_mle_2 = get_random_mle::<Fr>(1, &mut rng);

    // --- Yikes ---
    input_layer_2_mle_1.layer_id = LayerId::Input(1);
    input_layer_2_mle_2.layer_id = LayerId::Input(1);

    let circuit = MultiInputLayerCircuit::new(
        input_layer_1_mle_1,
        input_layer_1_mle_2,
        input_layer_2_mle_1,
        input_layer_2_mle_2,
    );

    test_circuit(
        circuit,
        Some(Path::new("./multiple_input_layers_circuit_optimized.json")),
    );
}

#[test]
fn test_random_layer_circuit() {
    // let subscriber = tracing_subscriber::fmt().with_max_level(Level::TRACE).finish();
    // tracing::subscriber::set_global_default(subscriber)
    //     .map_err(|_err| eprintln!("Unable to set global default subscriber"));

    let mut rng = test_rng();

    let num_vars = 5;
    let mle = get_random_mle::<Fr>(num_vars, &mut rng);
    let circuit = RandomCircuit { mle };

    test_circuit(circuit, Some(Path::new("./random_proof_optimized.json")));
}

// ------------------------------------ GATE CIRCUITS ------------------------------------

#[test]
fn test_gkr_gate_simplest_circuit() {
    let mut rng = test_rng();
    let size = 1 << 4;

    // --- This should be 2^2 ---
    let mle: DenseMle<Fr, Fr> = DenseMle::new_from_iter(
        (0..size).map(|_| Fr::from(rng.gen::<u64>())),
        LayerId::Input(0),
        None,
    );

    let negmle = DenseMle::new_from_iter(
        mle.mle_ref()
            .current_mle
            .get_evals_vector()
            .clone()
            .into_iter()
            .map(|elem| -elem),
        LayerId::Input(0),
        None,
    );

    let circuit: SimplestGateCircuit<Fr> = SimplestGateCircuit { mle, negmle };

    test_circuit(circuit, Some(Path::new("./gate_proof_optimized.json")));

    // panic!();
}

#[test]
fn test_gkr_gate_simplest_circuit_uneven() {
    let mut rng = test_rng();
    let size = 1 << 4;

    // --- This should be 2^2 ---
    let mle: DenseMle<Fr, Fr> = DenseMle::new_from_iter(
        (0..size).map(|_| Fr::from(rng.gen::<u64>())),
        LayerId::Input(0),
        None,
    );

    let negmle = DenseMle::new_from_raw(
        vec![mle.mle_ref().bookkeeping_table()[0].neg()],
        LayerId::Input(0),
        None,
    );

    let circuit: SimplestGateCircuitUneven<Fr> = SimplestGateCircuitUneven { mle, negmle };

    test_circuit(circuit, Some(Path::new("./gate_proof_testing_uneven.json")));

    // panic!();
}

#[test]
fn test_gkr_gate_simplest_circuit_combined() {
    let mut rng = test_rng();
    let size = 1 << 4;

    // --- This should be 2^2 ---
    let mle: DenseMle<Fr, Fr> = DenseMle::new_from_iter(
        (0..size).map(|_| Fr::from(rng.gen::<u64>())),
        LayerId::Input(0),
        None,
    );

    dbg!("hello");

    let negmle = DenseMle::new_from_iter(
        mle.mle_ref()
            .bookkeeping_table()
            .into_iter()
            .map(|elem| -elem),
        LayerId::Input(0),
        None,
    );

    dbg!("hfboooo");

    let circuit: SimplestGateCircuitCombined<Fr> = SimplestGateCircuitCombined { mle, negmle };

    test_circuit(circuit, None);

    // panic!();
}

#[test]
fn test_gkr_gate_batched_simplest_circuit() {
    // let subscriber = tracing_subscriber::fmt().with_max_level(Level::TRACE).finish();
    // tracing::subscriber::set_global_default(subscriber)
    //     .map_err(|_err| eprintln!("Unable to set global default subscriber"));

    let mut rng = test_rng();
    let size = 1 << 4;

    // --- This should be 2^4 ---
    let mle: DenseMle<Fr, Fr> = DenseMle::new_from_iter(
        (0..size).map(|_| Fr::from(rng.gen::<u64>())),
        LayerId::Input(0),
        // this is the batched bits
        None,
    );

    let negmle = DenseMle::new_from_iter(
        mle.mle_ref()
            .current_mle
            .get_evals_vector()
            .clone()
            .into_iter()
            .map(|elem| -elem),
        LayerId::Input(0),
        // this is the batched bits
        None,
    );

    let circuit: SimplestBatchedGateCircuit<Fr> = SimplestBatchedGateCircuit {
        mle,
        negmle,
        batch_bits: 2,
    };

    test_circuit(
        circuit,
        Some(Path::new("./gate_batch_proof2_optimized.json")),
    );
}

#[test]
fn test_gkr_gate_batched_simplest_circuit_uneven() {
    // let subscriber = tracing_subscriber::fmt().with_max_level(Level::TRACE).finish();
    // tracing::subscriber::set_global_default(subscriber)
    //     .map_err(|_err| eprintln!("Unable to set global default subscriber"));

    let mut rng = test_rng();
    let size = 1 << 4;
    let size2 = 1 << 3;

    // --- This should be 2^4 ---
    let mle: DenseMle<Fr, Fr> = DenseMle::new_from_iter(
        (0..size).map(|_| Fr::from(rng.gen::<u64>())),
        LayerId::Input(0),
        // These are NOT the batched bits
        None,
    );

    let negmle = DenseMle::new_from_iter(
        mle.mle_ref().current_mle.get_evals_vector()[0..size2]
            .iter()
            .map(|elem| -elem),
        LayerId::Input(0),
        // These are NOT the batched bits
        None,
    );

    let circuit: SimplestBatchedGateCircuit<Fr> = SimplestBatchedGateCircuit {
        mle,
        negmle,
        batch_bits: 2,
    };

    test_circuit(
        circuit,
        Some(Path::new("./gate_batch_proof_uneven_optimized.json")),
    );
}

#[test]
fn test_gkr_add_mul_gate_batched_simplest_circuit() {
    // let subscriber = tracing_subscriber::fmt().with_max_level(Level::TRACE).finish();
    // tracing::subscriber::set_global_default(subscriber)
    //     .map_err(|_err| eprintln!("Unable to set global default subscriber"));

    let _rng = test_rng();
    let size = 1 << 2;

    let mle_1: DenseMle<Fr, Fr> = DenseMle::new_from_iter(
        (0..size).map(|_| {
            // let num = Fr::from(rng.gen::<u64>());

            Fr::one()
        }),
        LayerId::Input(0),
        None,
    );

    let mle_2: DenseMle<Fr, Fr> = DenseMle::new_from_iter(
        (0..size).map(|_| {
            // let num = Fr::from(rng.gen::<u64>());

            Fr::one()
        }),
        LayerId::Input(0),
        None,
    );

    let neg_mle_2 = DenseMle::new_from_iter(
        mle_2
            .mle_ref()
            .bookkeeping_table()
            .into_iter()
            .map(|elem| -elem),
        LayerId::Input(0),
        None,
    );
    // let mle: DenseMle<Fr, Tuple2<Fr>> = DenseMle::new_from_iter(
    //     (0..size).map(|idx| (Fr::from(idx + 1), Fr::from(idx + 1)).into()),
    //     LayerId::Input,
    //     None,
    // );

    let circuit: SimplestAddMulBatchedGateCircuit<Fr> = SimplestAddMulBatchedGateCircuit {
        mle_1,
        mle_2,
        neg_mle_2,
        batch_bits: 1,
    };

    test_circuit(
        circuit,
        Some(Path::new("./gate_batch_proof1_optimized.json")),
    );

    // panic!();
}

#[test]
fn test_gkr_add_mul_gate_simplest_circuit() {
    let mut rng = test_rng();
    let size = 1 << 4;

    let mle_1: DenseMle<Fr, Fr> = DenseMle::new_from_iter(
        (0..size).map(|_| Fr::from(rng.gen::<u64>())),
        LayerId::Input(0),
        None,
    );

    let mle_2: DenseMle<Fr, Fr> = DenseMle::new_from_iter(
        (0..size).map(|_| Fr::from(rng.gen::<u64>())),
        LayerId::Input(0),
        None,
    );

    let neg_mle_2 = DenseMle::new_from_iter(
        mle_2
            .mle_ref()
            .bookkeeping_table()
            .into_iter()
            .map(|elem| -elem),
        LayerId::Input(0),
        None,
    );

    let circuit: MulAddSimplestGateCircuit<Fr> = MulAddSimplestGateCircuit {
        mle_1,
        mle_2,
        neg_mle_2,
    };

    test_circuit(
        circuit,
        Some(Path::new("./mul_gate_simple_proof_optimized.json")),
    );

    // panic!();
}

// ------------------------------------ EMPTY LAYER CIRCUITS ------------------------------------

#[test]
fn test_empty_layer_circuit() {
    let mle: DenseMle<Fr, Fr> =
        DenseMle::new_from_raw(vec![Fr::from(14), Fr::from(14)], LayerId::Input(0), None);
    let mle2: DenseMle<Fr, Fr> = DenseMle::new_from_raw(
        vec![Fr::from(14).neg(), Fr::from(14).neg()],
        LayerId::Input(0),
        None,
    );
    let empty_layer_src_mle: DenseMle<Fr, Fr> =
        DenseMle::new_from_raw(vec![Fr::from(1), Fr::from(2)], LayerId::Input(0), None);
    let other_empty_layer_src_mle: DenseMle<Fr, Fr> =
        DenseMle::new_from_raw(vec![Fr::from(3), Fr::from(4)], LayerId::Input(0), None);

    let circuit: EmptyLayerTestCircuit<Fr> = EmptyLayerTestCircuit {
        mle,
        mle2,
        empty_layer_src_mle,
        other_empty_layer_src_mle,
    };

    test_circuit(circuit, Some(Path::new("empty_layer_proof_optimized.json")));
}

// ------------------------------------ CIRCUIT COMBINATOR CIRCUITS ------------------------------------

#[test]
fn test_combine_circuit() {
    let mut rng = test_rng();
    let size = 4;
    let size_expanded = 1 << size;
    // let subscriber = tracing_subscriber::fmt().with_max_level(Level::TRACE).finish();
    // tracing::subscriber::set_global_default(subscriber)
    //     .map_err(|_err| eprintln!("Unable to set global default subscriber"));

    // --- This should be 2^2 ---
    let mle: DenseMle<Fr, Tuple2<Fr>> = DenseMle::new_from_iter(
        (0..size_expanded).map(|_| (Fr::from(rng.gen::<u64>()), Fr::from(rng.gen::<u64>())).into()),
        LayerId::Input(0),
        None,
    );
    // --- This should be 2^2 ---
    let mle_2: DenseMle<Fr, Tuple2<Fr>> = DenseMle::new_from_iter(
        (0..size_expanded).map(|_| (Fr::from(rng.gen::<u64>()), Fr::from(rng.gen::<u64>())).into()),
        LayerId::Input(0),
        None,
    );

    let test_circuit_1: TestCircuit<Fr> = TestCircuit { mle, mle_2, size };

    let size = 4;

    // --- This should be 2^2 ---
    let mle: DenseMle<Fr, Tuple2<Fr>> = DenseMle::new_from_iter(
        (0..1 << size).map(|_| (Fr::from(rng.gen::<u64>()), Fr::from(rng.gen::<u64>())).into()),
        LayerId::Input(0),
        None,
    );

    let simple_circuit: SimpleCircuit<Fr> = SimpleCircuit { mle, size };

    let circuit = CombineCircuit {
        test_circuit: test_circuit_1,
        simple_circuit,
    };

    test_circuit(circuit, None);
}

struct Combine3Circuit<F: FieldExt> {
    test_circuit: TestCircuit<F>,
    simple_circuit: SimpleCircuit<F>,
    batch_circuit: SimplestBatchedCircuit<F>,
}

impl<F: FieldExt> GKRCircuit<F> for Combine3Circuit<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        let test_witness = self.test_circuit.synthesize();
        let simple_witness = self.simple_circuit.synthesize();
        let batch_witness = self.batch_circuit.synthesize();

        let Witness {
            layers: test_layers,
            output_layers: test_outputs,
            input_layers: test_inputs,
        } = test_witness;

        let Witness {
            layers: mut simple_layers,
            output_layers: simple_outputs,
            input_layers: simple_inputs,
        } = simple_witness;

        let Witness {
            layers: mut batch_layers,
            output_layers: batch_outputs,
            input_layers: batch_inputs,
        } = batch_witness;

        // for input vv
        let input_layers: Vec<InputLayerEnum<F, PoseidonTranscript<F>>> = test_inputs
            .into_iter()
            .chain(simple_inputs.into_iter().map(|mut input| {
                let new_layer_id = match input.layer_id() {
                    LayerId::RandomInput(id) => LayerId::RandomInput(id + 1),
                    LayerId::Input(id) => LayerId::Input(id + 1),
                    LayerId::Layer(_) => panic!(),
                    LayerId::Output(_) => panic!(),
                };
                input.set_layer_id(new_layer_id);
                input
            }))
            .chain(batch_inputs.into_iter().map(|mut input| {
                let new_layer_id = match input.layer_id() {
                    LayerId::RandomInput(id) => LayerId::RandomInput(id + 2),
                    LayerId::Input(id) => LayerId::Input(id + 2),
                    LayerId::Layer(_) => panic!(),
                    LayerId::Output(_) => panic!(),
                };
                input.set_layer_id(new_layer_id);
                input
            }))
            .collect();

        for layer in simple_layers.0.iter_mut() {
            let expression = match layer {
                LayerEnum::Gkr(layer) => &mut layer.expression,
                LayerEnum::EmptyLayer(layer) => &mut layer.expr,
                _ => panic!(),
            };

            let mut closure =
                for<'a, 'b> |expr: &'a mut ExpressionNode<F, ProverExpr>,
                             mle_vec: &'b mut <ProverExpr as ExpressionType<F>>::MleVec|
                             -> Result<(), ()> {
                    match expr {
                        ExpressionNode::Mle(mle_vec_idx) => {
                            let mle_ref = mle_vec_idx.get_mle_mut(mle_vec);

                            if mle_ref.layer_id == LayerId::Input(0) {
                                mle_ref.layer_id = LayerId::Input(1)
                            }
                            Ok(())
                        }
                        ExpressionNode::Product(mle_vec_indices) => {
                            for mle_vec_index in mle_vec_indices {
                                let mle_ref = mle_vec_index.get_mle_mut(mle_vec);

                                if mle_ref.layer_id == LayerId::Input(0) {
                                    mle_ref.layer_id = LayerId::Input(1)
                                }
                            }
                            Ok(())
                        }
                        ExpressionNode::Constant(_)
                        | ExpressionNode::Scaled(_, _)
                        | ExpressionNode::Sum(_, _)
                        | ExpressionNode::Negated(_)
                        | ExpressionNode::Selector(_, _, _) => Ok(()),
                    }
                };

            expression.traverse_mut(&mut closure).unwrap();
        }

        // for input ^^

        for layer in batch_layers.0.iter_mut() {
            let expression = match layer {
                LayerEnum::Gkr(layer) => &mut layer.expression,
                LayerEnum::EmptyLayer(layer) => &mut layer.expr,
                _ => panic!(),
            };

            let mut closure =
                for<'a, 'b> |expr: &'a mut ExpressionNode<F, ProverExpr>,
                             mle_vec: &'b mut <ProverExpr as ExpressionType<F>>::MleVec|
                             -> Result<(), ()> {
                    match expr {
                        ExpressionNode::Mle(mle_vec_idx) => {
                            let mle_ref = mle_vec_idx.get_mle_mut(mle_vec);

                            if mle_ref.layer_id == LayerId::Input(0) {
                                mle_ref.layer_id = LayerId::Input(2)
                            }
                            Ok(())
                        }
                        ExpressionNode::Product(mle_vec_indices) => {
                            for mle_vec_index in mle_vec_indices {
                                let mle_ref = mle_vec_index.get_mle_mut(mle_vec);

                                if mle_ref.layer_id == LayerId::Input(0) {
                                    mle_ref.layer_id = LayerId::Input(2)
                                }
                            }
                            Ok(())
                        }
                        ExpressionNode::Constant(_)
                        | ExpressionNode::Scaled(_, _)
                        | ExpressionNode::Sum(_, _)
                        | ExpressionNode::Negated(_)
                        | ExpressionNode::Selector(_, _, _) => Ok(()),
                    }
                };

            expression.traverse_mut(&mut closure).unwrap();
        }

        let (layers, output_layers) = combine_layers(
            vec![test_layers, simple_layers, batch_layers],
            vec![test_outputs, simple_outputs, batch_outputs],
        )
        .unwrap();

        Witness {
            layers,
            output_layers,
            input_layers,
        }
    }
}

#[test]
fn test_combine_3_circuit() {
    let mut rng = test_rng();
    let size = 4;
    let size_expanded = 1 << size;
    // let subscriber = tracing_subscriber::fmt().with_max_level(Level::TRACE).finish();
    // tracing::subscriber::set_global_default(subscriber)
    //     .map_err(|_err| eprintln!("Unable to set global default subscriber"));

    // --- This should be 2^2 ---
    let mle: DenseMle<Fr, Tuple2<Fr>> = DenseMle::new_from_iter(
        (0..size_expanded).map(|_| (Fr::from(rng.gen::<u64>()), Fr::from(rng.gen::<u64>())).into()),
        LayerId::Input(0),
        None,
    );
    // --- This should be 2^2 ---
    let mle_2: DenseMle<Fr, Tuple2<Fr>> = DenseMle::new_from_iter(
        (0..size_expanded).map(|_| (Fr::from(rng.gen::<u64>()), Fr::from(rng.gen::<u64>())).into()),
        LayerId::Input(0),
        None,
    );

    let test_circuit_1: TestCircuit<Fr> = TestCircuit { mle, mle_2, size };

    let size = 4;

    // --- This should be 2^2 ---
    let mle: DenseMle<Fr, Tuple2<Fr>> = DenseMle::new_from_iter(
        (0..1 << size).map(|_| (Fr::from(rng.gen::<u64>()), Fr::from(rng.gen::<u64>())).into()),
        LayerId::Input(0),
        None,
    );
    // let mle: DenseMle<Fr, Tuple2<Fr>> = DenseMle::new_from_iter(
    //     (0..size).map(|idx| (Fr::from(idx + 2), Fr::from(idx + 2)).into()),
    //     LayerId::Input(0),
    //     None,
    // );

    let simple_circuit: SimpleCircuit<Fr> = SimpleCircuit { mle, size };

    let size = 1 << 3;

    let batch_size = 1 << 2;
    // --- This should be 2^2 ---
    let batched_mle: Vec<DenseMle<Fr, Tuple2<Fr>>> = (0..batch_size)
        .map(|_idx1| {
            DenseMle::new_from_iter(
                (0..size).map(|_idx| {
                    let num = Fr::from(rng.gen::<u64>());
                    //let second_num = Fr::from(rng.gen::<u64>());
                    // let num = Fr::from(idx + idx1);
                    (num, num).into()
                }),
                LayerId::Input(0),
                None,
            )
        })
        .collect_vec();
    // let mle: DenseMle<Fr, Tuple2<Fr>> = DenseMle::new_from_iter(
    //     (0..size).map(|idx| (Fr::from(idx + 1), Fr::from(idx + 1)).into()),
    //     LayerId::Input(0),
    //     None,
    // );

    let batch_circuit = SimplestBatchedCircuit {
        batched_first_second_mle: batched_mle,
        batch_bits: 2,
    };

    let circuit = Combine3Circuit {
        test_circuit: test_circuit_1,
        simple_circuit,
        batch_circuit,
    };

    test_circuit(circuit, None);
}
