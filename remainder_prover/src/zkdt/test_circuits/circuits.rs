

use ark_std::log2;
use itertools::{repeat_n, Itertools};
use remainder_shared_types::{FieldExt, transcript::poseidon_transcript::PoseidonTranscript};

use crate::{mle::{dense::DenseMle, Mle, MleRef, MleIndex}, prover::{GKRCircuit, Witness, input_layer::{combine_input_layers::InputLayerBuilder, public_input_layer::PublicInputLayer, InputLayer, random_input_layer::RandomInputLayer, enum_input_layer::InputLayerEnum}, Layers, GKRError}, layer::{LayerId, batched::{BatchedLayer, combine_zero_mle_ref}}};

use super::circuit_builders::{FSRandomBuilder, SelfMinusSelfBuilder};

/// Testing circuit for a batched MLE against a non-batched random input layer
pub struct BatchedFSRandomCircuit<F: FieldExt> {
    batched_mle: Vec<DenseMle<F, F>>,
    val_mle_size: usize,
}
impl<F: FieldExt> GKRCircuit<F> for BatchedFSRandomCircuit<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        unimplemented!()
    }

    fn synthesize_and_commit(
            &mut self,
            transcript: &mut Self::Transcript,
        ) -> Result<(Witness<F, Self::Transcript>, Vec<crate::prover::input_layer::enum_input_layer::CommitmentEnum<F>>), crate::prover::GKRError> {

        // --- For the input layer, we need to first merge all of the input MLEs FIRST by mle_idx, then by dataparallel index ---
        // --- This assures that (going left-to-right in terms of the bits) we have [input_prefix_bits], [dataparallel_bits], [mle_idx], [iterated_bits] ---
        let mut combined_batched_mle = DenseMle::<F, F>::combine_mle_batch(self.batched_mle.clone());

        // --- Circuit has two input layers: one is the `batched_mle` input layer ---
        let public_input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut combined_batched_mle)];
        let public_input_layer_builder = InputLayerBuilder::new(public_input_mles, None, LayerId::Input(0));
        let public_input_layer: PublicInputLayer<F, Self::Transcript> = public_input_layer_builder.to_input_layer();
        let mut public_input_layer_enum = public_input_layer.to_enum();
        let input_commit = public_input_layer_enum
            .commit()
            .map_err(GKRError::InputLayerError)?;
        InputLayerEnum::append_commitment_to_transcript(&input_commit, transcript).unwrap();

        // --- The other is a `RandomInputLayer` for `val_mle` ---
        let val_mle_input_layer = RandomInputLayer::new(transcript, self.val_mle_size, LayerId::Input(1));
        let val_mle = val_mle_input_layer.get_mle();
        let mut val_mle_input_layer_enum = val_mle_input_layer.to_enum();
        let random_commit = val_mle_input_layer_enum
            .commit()
            .map_err(GKRError::InputLayerError)?;

        // --- Dataparallel/batching stuff + sanitychecks ---
        let num_subcircuit_copies = self.batched_mle.len();
        let num_dataparallel_bits = log2(num_subcircuit_copies) as usize;

        // --- Create `Layers` struct to add layers to ---
        let mut layers: Layers<F> = Layers::new();

        // --- Create single batched layer ---
        let random_sub_layer_builders = self.batched_mle.iter_mut().map(|mle| {
            // --- First index the input + batched bits ---
            mle.set_prefix_bits(
                Some(
                    combined_batched_mle.get_prefix_bits().iter().flatten().cloned().chain(
                        repeat_n(MleIndex::Iterated, num_dataparallel_bits)
                    ).collect_vec()
                )
            );
            FSRandomBuilder::new(
                mle.clone(),
                val_mle.clone()
            )
        }).collect_vec();
        let batched_random_sub_layer_builder = BatchedLayer::new(random_sub_layer_builders);

        // --- Grab output layer and flatten ---
        let batched_result_mle = layers.add_gkr(batched_random_sub_layer_builder);

        // --- Builder which subtracts the result from itself to have a zero output layer ---
        let self_minus_self_builders = batched_result_mle.into_iter().map(|result_mle| {
            SelfMinusSelfBuilder::new(
                result_mle
            )
        }).collect_vec();
        let batched_self_minus_self_builder = BatchedLayer::new(self_minus_self_builders);
        let batched_final_result = layers.add_gkr(batched_self_minus_self_builder);
        let combined_batched_final_result = combine_zero_mle_ref(batched_final_result);

        Ok((
            Witness { 
                layers, 
                output_layers: vec![combined_batched_final_result.get_enum()], 
                input_layers: vec![public_input_layer_enum, val_mle_input_layer_enum]
            },
            vec![input_commit, random_commit],
        ))
    }
}

impl<F: FieldExt> BatchedFSRandomCircuit<F> {
    /// Constructor
    pub fn new(
        batched_mle: Vec<DenseMle<F, F>>,
        val_mle_size: usize,
    ) -> Self {
        Self {
            batched_mle,
            val_mle_size
        }
    }
}