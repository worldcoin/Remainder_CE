use std::cmp::max;

use ark_std::test_rng;
use remainder_ligero::ligero_commit::remainder_ligero_commit;
use remainder_shared_types::{transcript::TranscriptWriter, FieldExt, Fr};

use crate::{
    builders::{
        combine_input_layers::InputLayerBuilder,
        layer_builder::{from_mle, simple_builders::EqualityCheck, LayerBuilder},
    },
    expression::{generic_expr::Expression, prover_expr::ProverExpr},
    layer::LayerId,
    mle::{dense::DenseMle, Mle},
    output_layer::mle_output_layer::MleOutputLayer,
    prover::{
        helpers::test_circuit, layers::Layers, proof_system::DefaultProofSystem, CircuitInputLayer,
        CircuitTranscript, GKRCircuit, GKRError, GKRVerifierKey, Witness,
    },
    utils::get_random_mle,
};

use super::{
    enum_input_layer::InputLayerEnum, ligero_input_layer::LigeroInputLayer,
    public_input_layer::PublicInputLayer, random_input_layer::RandomInputLayer, InputLayer,
};

/// This circuit takes in a single MLE and performs the following:
/// * Adds the Ligero commitment of `mle` to the FS transcript
/// * Samples a single random value from the FS transcript
/// * Computes the scalar-vector product of that random value against the
///     bookkeeping table of `mle`
///
/// ## Arguments
/// * `mle` - Input MLE. Can be any length.
struct RandomCircuit<F: FieldExt> {
    mle: DenseMle<F>,
}
impl<F: FieldExt> GKRCircuit<F> for RandomCircuit<F> {
    type ProofSystem = DefaultProofSystem;

    fn synthesize(&mut self) -> Witness<F, Self::ProofSystem> {
        unimplemented!()
    }

    fn synthesize_and_commit(
        &mut self,
        transcript_writer: &mut TranscriptWriter<F, CircuitTranscript<F, Self>>,
    ) -> Result<
        (
            Witness<F, Self::ProofSystem>,
            Vec<<CircuitInputLayer<F, Self> as InputLayer<F>>::Commitment>,
            GKRVerifierKey<F, Self::ProofSystem>,
        ),
        GKRError,
    > {
        let input = InputLayerBuilder::new(vec![&mut self.mle], None, LayerId::Input(0))
            .to_ligero_input_layer_with_rho_inv(4, 1.);
        let mut input: CircuitInputLayer<F, Self> = input.into();

        let input_commit = input.commit().map_err(GKRError::InputLayerError)?;
        InputLayerEnum::append_commitment_to_transcript(&input_commit, transcript_writer);

        // TODO!(ryancao): Fix the `RandomInputLayer::new()` argument to make it less confusing
        let random = RandomInputLayer::new(transcript_writer, 1, LayerId::Input(1));
        let random_mle = random.get_mle();
        let mut random: CircuitInputLayer<F, Self> = random.into();
        let random_commit = random.commit().map_err(GKRError::InputLayerError)?;

        let mut layers = Layers::new();

        let layer_1 = from_mle(
            (self.mle.clone(), random_mle),
            |(mle, random)| {
                Expression::<F, ProverExpr>::products(vec![mle.clone(), random.clone()])
            },
            |(mle, random), layer_id, prefix_bits| {
                let mut out = DenseMle::new_from_iter(
                    mle.clone()
                        .into_iter()
                        .zip(random.clone().into_iter().cycle())
                        .map(|(item, random)| item * random),
                    layer_id,
                );
                if let Some(prefix_bits) = prefix_bits {
                    out.add_prefix_bits(prefix_bits);
                }
                out
            },
        );

        let output = layers.add_gkr(layer_1);

        let mut output_input = output.clone();
        output_input.layer_id = LayerId::Input(2);
        let mut input_layer_2: CircuitInputLayer<F, Self> =
            InputLayerBuilder::new(vec![&mut output_input], None, LayerId::Input(2))
                .to_input_layer::<PublicInputLayer<F>>()
                .into();
        let input_layer_2_commit = input_layer_2.commit().map_err(GKRError::InputLayerError)?;
        InputLayerEnum::append_commitment_to_transcript(&input_layer_2_commit, transcript_writer);

        let layer_2 = EqualityCheck::new(output, output_input);
        let output = layers.add_gkr(layer_2);

        let output_layers = vec![MleOutputLayer::new_zero(output)];

        Ok((
            Witness {
                layers,
                output_layers,
                input_layers: vec![input, random, input_layer_2],
            },
            vec![input_commit, random_commit, input_layer_2_commit],
            todo!(),
        ))
    }
}

/// Simple builder which adds two MLEs together, element-wise.
/// Performs implicit wraparound.
///
/// ## Arguments
/// * `mle_1`, `mle_2` - MLEs whose bookkeeping tables are to be added together.
struct WraparoundAddBuilder<F: FieldExt> {
    mle_1: DenseMle<F>,
    mle_2: DenseMle<F>,
}
impl<F: FieldExt> LayerBuilder<F> for WraparoundAddBuilder<F> {
    type Successor = DenseMle<F>;

    fn build_expression(&self) -> Expression<F, crate::expression::prover_expr::ProverExpr> {
        self.mle_1.clone().expression() + self.mle_2.clone().expression()
    }

    fn next_layer(
        &self,
        id: LayerId,
        prefix_bits: Option<Vec<crate::mle::MleIndex<F>>>,
    ) -> Self::Successor {
        let mle_1_mle_ref = self.mle_1.clone();
        let mle_2_mle_ref = self.mle_2.clone();
        let result_num_elems = max(
            1 << self.mle_1.num_iterated_vars(),
            1 << self.mle_2.num_iterated_vars(),
        );
        let result_bookkeeping_table = mle_1_mle_ref
            .bookkeeping_table()
            .iter()
            .cycle()
            .zip(mle_2_mle_ref.bookkeeping_table().iter().cycle())
            .map(|(elem_1, elem_2)| *elem_1 + *elem_2)
            .take(result_num_elems);
        let mut out = DenseMle::new_from_iter(result_bookkeeping_table, id);
        if let Some(prefix_bits) = prefix_bits {
            out.add_prefix_bits(prefix_bits);
        }
        out
    }
}
impl<F: FieldExt> WraparoundAddBuilder<F> {
    fn new(mle_1: DenseMle<F>, mle_2: DenseMle<F>) -> Self {
        Self { mle_1, mle_2 }
    }
}

/// This circuit has two separate input layers, each with two MLEs inside, where
/// all MLEs can be different sizes.
///
/// The MLEs within each input layer are first added together, then their results
/// are added. The final layer is just a `ZeroLayerBuilder` (i.e. subtracts the final
/// layer from itself for convenience).
///
/// ## Arguments
/// * `input_layer_1_mle_1`, `input_layer_1_mle_2` - MLEs to be combined in
///     the same input layer.
/// * `input_layer_2_mle_1`, `input_layer_2_mle_2` - MLEs to be combined in
///     the same input layer.
struct MultiInputLayerCircuit<F: FieldExt> {
    input_layer_1_mle_1: DenseMle<F>,
    input_layer_1_mle_2: DenseMle<F>,
    input_layer_2_mle_1: DenseMle<F>,
    input_layer_2_mle_2: DenseMle<F>,
}
impl<F: FieldExt> GKRCircuit<F> for MultiInputLayerCircuit<F> {
    type ProofSystem = DefaultProofSystem;

    fn synthesize(&mut self) -> Witness<F, Self::ProofSystem> {
        unimplemented!()
    }

    fn synthesize_and_commit(
        &mut self,
        transcript_writer: &mut TranscriptWriter<F, CircuitTranscript<F, Self>>,
    ) -> Result<
        (
            Witness<F, Self::ProofSystem>,
            Vec<<CircuitInputLayer<F, Self> as InputLayer<F>>::Commitment>,
            GKRVerifierKey<F, Self::ProofSystem>,
        ),
        GKRError,
    > {
        // --- First input layer contains the first two MLE arguments ---
        let mut input_layer_1: CircuitInputLayer<F, Self> = InputLayerBuilder::new(
            vec![&mut self.input_layer_1_mle_1, &mut self.input_layer_1_mle_2],
            None,
            LayerId::Input(0),
        )
        .to_input_layer::<PublicInputLayer<F>>()
        .into();
        let input_layer_1_commitment = input_layer_1.commit().map_err(GKRError::InputLayerError)?;
        InputLayerEnum::append_commitment_to_transcript(
            &input_layer_1_commitment,
            transcript_writer,
        );

        // --- Second input layer contains the next two MLE arguments ---
        let mut input_layer_2: CircuitInputLayer<F, Self> = InputLayerBuilder::new(
            vec![&mut self.input_layer_2_mle_1, &mut self.input_layer_2_mle_2],
            None,
            LayerId::Input(1),
        )
        .to_input_layer::<PublicInputLayer<F>>()
        .into();
        let input_layer_2_commitment = input_layer_2.commit().map_err(GKRError::InputLayerError)?;
        InputLayerEnum::append_commitment_to_transcript(
            &input_layer_2_commitment,
            transcript_writer,
        );

        let mut layers = Layers::new();

        let layer_1_builder = WraparoundAddBuilder::new(
            self.input_layer_1_mle_1.clone(),
            self.input_layer_2_mle_2.clone(),
        );
        let layer_2_builder = WraparoundAddBuilder::new(
            self.input_layer_2_mle_1.clone(),
            self.input_layer_2_mle_2.clone(),
        );

        let first_layer_output = layers.add_gkr(layer_1_builder);
        let second_layer_output = layers.add_gkr(layer_2_builder);

        let layer_3_builder = WraparoundAddBuilder::new(first_layer_output, second_layer_output);

        let third_layer_output = layers.add_gkr(layer_3_builder);

        let layer_4 = EqualityCheck::new(third_layer_output.clone(), third_layer_output);
        let fourth_layer_output = layers.add_gkr(layer_4);

        let output_layers = vec![MleOutputLayer::new_zero(fourth_layer_output)];

        Ok((
            Witness {
                layers,
                output_layers,
                input_layers: vec![input_layer_1, input_layer_2],
            },
            vec![input_layer_1_commitment, input_layer_2_commitment],
            todo!(),
        ))
    }
}
impl<F: FieldExt> MultiInputLayerCircuit<F> {
    pub fn new(
        input_layer_1_mle_1: DenseMle<F>,
        input_layer_1_mle_2: DenseMle<F>,
        input_layer_2_mle_1: DenseMle<F>,
        input_layer_2_mle_2: DenseMle<F>,
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
/// two independent circuits via the fact that it basically subtracts each input MLE
/// from itself and calls that the output layer. In particular, this allows us to test
/// whether Halo2 generates the same VK given that we have the same pre-committed Ligero layer
/// but a DIFFERENT live-committed Ligero layer.
///
/// ## Arguments
/// * `mle`, `mle_2` - MLEs of any size.
struct SimplePrecommitCircuit<F: FieldExt> {
    mle: DenseMle<F>,
    mle_2: DenseMle<F>,
}
impl<F: FieldExt> GKRCircuit<F> for SimplePrecommitCircuit<F> {
    type ProofSystem = DefaultProofSystem;

    fn synthesize(&mut self) -> Witness<F, Self::ProofSystem> {
        // --- Set layer IDs to be correct with respect to input layers ---
        self.mle.layer_id = LayerId::Input(0);
        self.mle_2.layer_id = LayerId::Input(1);

        // --- The precommitted input layer MLE is just the first MLE ---
        let precommitted_input_mles: Vec<&mut dyn Mle<F>> = vec![&mut self.mle];
        let precommitted_input_layer_builder =
            InputLayerBuilder::new(precommitted_input_mles, None, LayerId::Input(0));

        // --- The non-precommitted input layer MLE is just the second ---
        let live_committed_input_mles: Vec<&mut dyn Mle<F>> = vec![&mut self.mle_2];
        let live_committed_input_layer_builder =
            InputLayerBuilder::new(live_committed_input_mles, None, LayerId::Input(1));

        let mut layers: Layers<F, _> = Layers::new();

        let diff_builder_1 = EqualityCheck::new(self.mle.clone(), self.mle.clone());
        let diff_builder_2 = EqualityCheck::new(self.mle_2.clone(), self.mle_2.clone());

        // --- Stacks the two aforementioned layers together into a single layer ---
        // --- Then adds them to the overall circuit ---
        let first_layer_output_1 = layers.add_gkr(diff_builder_1);
        let first_layer_output_2 = layers.add_gkr(diff_builder_2);

        // --- We should have two input layers: a single pre-committed and a single regular Ligero layer ---
        let rho_inv = 4;
        let ratio = 1_f64;
        let (ligero_aux, ligero_comm, ligero_root) = remainder_ligero_commit(
            self.mle.current_mle.get_evals_vector(),
            rho_inv,
            ratio,
            None,
        );
        let precommitted_input_layer: LigeroInputLayer<F> = precommitted_input_layer_builder
            .to_input_layer_with_precommit(ligero_comm, ligero_aux, ligero_root, true);
        let live_committed_input_layer: LigeroInputLayer<F> =
            live_committed_input_layer_builder.to_ligero_input_layer_with_rho_inv(4, 1.);

        let output_layers = vec![
            MleOutputLayer::new_zero(first_layer_output_1),
            MleOutputLayer::new_zero(first_layer_output_2),
        ];

        Witness {
            layers,
            output_layers,
            input_layers: vec![
                precommitted_input_layer.into(),
                live_committed_input_layer.into(),
            ],
        }
    }
}
impl<F: FieldExt> SimplePrecommitCircuit<F> {
    fn new(mle: DenseMle<F>, mle_2: DenseMle<F>) -> Self {
        Self { mle, mle_2 }
    }
}

#[test]
fn test_gkr_circuit_with_precommit() {
    const NUM_ITERATED_BITS: usize = 4;
    let mut rng = test_rng();

    let mle: DenseMle<Fr> = get_random_mle(NUM_ITERATED_BITS, &mut rng);
    let mle_2: DenseMle<Fr> = get_random_mle(NUM_ITERATED_BITS, &mut rng);

    let circuit: SimplePrecommitCircuit<Fr> = SimplePrecommitCircuit::new(mle, mle_2);

    test_circuit(circuit, None);
}

#[test]
fn test_multiple_input_layers_circuit() {
    let mut rng = test_rng();
    let input_layer_1_mle_1 = get_random_mle::<Fr>(3, &mut rng);
    let input_layer_1_mle_2 = get_random_mle::<Fr>(2, &mut rng);

    let mut input_layer_2_mle_1 = get_random_mle::<Fr>(2, &mut rng);
    let mut input_layer_2_mle_2 = get_random_mle::<Fr>(1, &mut rng);

    input_layer_2_mle_1.layer_id = LayerId::Input(1);
    input_layer_2_mle_2.layer_id = LayerId::Input(1);

    let circuit = MultiInputLayerCircuit::new(
        input_layer_1_mle_1,
        input_layer_1_mle_2,
        input_layer_2_mle_1,
        input_layer_2_mle_2,
    );

    test_circuit(circuit, None);
}

#[test]
fn test_random_layer_circuit() {
    const NUM_ITERATED_BITS: usize = 4;
    let mut rng = test_rng();
    let mle = get_random_mle::<Fr>(NUM_ITERATED_BITS, &mut rng);
    let circuit = RandomCircuit { mle };

    test_circuit(circuit, None);
}
