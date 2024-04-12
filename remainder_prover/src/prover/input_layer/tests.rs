use std::{iter::repeat_with, path::Path};

use ark_std::test_rng;
use itertools::Itertools;
use rand::Rng;
use remainder_ligero::ligero_commit::remainder_ligero_commit_prove;
use remainder_shared_types::{
    transcript::{poseidon_transcript::PoseidonSponge, TranscriptWriter},
    FieldExt, Fr,
};

use crate::{
    expression::generic_expr::Expression,
    layer::{from_mle, simple_builders::EqualityCheck, LayerId},
    mle::{dense::DenseMle, zero::ZeroMleRef, Mle, MleRef},
    prover::{helpers::test_circuit, GKRCircuit, GKRError, Layers, Witness},
    utils::get_random_mle,
};

use super::{
    combine_input_layers::InputLayerBuilder,
    enum_input_layer::{CommitmentEnum, InputLayerEnum},
    ligero_input_layer::LigeroInputLayer,
    public_input_layer::PublicInputLayer,
    random_input_layer::RandomInputLayer,
    InputLayer,
};

/// This circuit checks how RandomLayer works by multiplying the MLE by a constant,
/// taking in that result as advice in a publiclayer and doing an equality check
/// on the result of the mult and the advice
struct RandomCircuit<F: FieldExt> {
    mle: DenseMle<F, F>,
}

impl<F: FieldExt> GKRCircuit<F> for RandomCircuit<F> {
    type Sponge = PoseidonSponge<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Sponge> {
        unimplemented!()
    }

    fn synthesize_and_commit(
        &mut self,
        transcript_writer: &mut TranscriptWriter<F, Self::Sponge>,
    ) -> Result<(Witness<F, Self::Sponge>, Vec<CommitmentEnum<F>>), GKRError> {
        let mut input =
            InputLayerBuilder::new(vec![Box::new(&mut self.mle)], None, LayerId::Input(0))
                .to_input_layer_with_rho_inv(4, 1.);
        let mut input = input.to_enum();

        let input_commit = input.commit().map_err(GKRError::InputLayerError)?;
        InputLayerEnum::prover_append_commitment_to_transcript(&input_commit, transcript_writer);

        let random = RandomInputLayer::new(transcript_writer, 1, LayerId::Input(1));
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
        InputLayerEnum::prover_append_commitment_to_transcript(
            &input_layer_2_commit,
            transcript_writer,
        );

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
    type Sponge = PoseidonSponge<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Sponge> {
        unimplemented!()
    }

    fn synthesize_and_commit(
        &mut self,
        transcript_writer: &mut TranscriptWriter<F, Self::Sponge>,
    ) -> Result<(Witness<F, Self::Sponge>, Vec<CommitmentEnum<F>>), GKRError> {
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
        InputLayerEnum::prover_append_commitment_to_transcript(
            &input_layer_1_commitment,
            transcript_writer,
        );

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
        InputLayerEnum::prover_append_commitment_to_transcript(
            &input_layer_2_commitment,
            transcript_writer,
        );

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
    type Sponge = PoseidonSponge<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Sponge> {
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
        let mut layers: Layers<F, Self::Sponge> = Layers::new();

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
        let precommitted_input_layer: LigeroInputLayer<F, Self::Sponge> =
            precommitted_input_layer_builder.to_input_layer_with_precommit(
                ligero_comm,
                ligero_aux,
                ligero_root,
                true,
            );
        let live_committed_input_layer: LigeroInputLayer<F, Self::Sponge> =
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
