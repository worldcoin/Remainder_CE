







use ark_std::{log2};
use itertools::{Itertools, repeat_n};

use crate::{mle::{dense::DenseMle, MleRef, Mle, MleIndex}, layer::{LayerBuilder, empty_layer::EmptyLayer, batched::{BatchedLayer, combine_zero_mle_ref}, LayerId}, zkdt::builders::{FSInputPackingBuilder}, prover::{input_layer::{combine_input_layers::InputLayerBuilder, public_input_layer::PublicInputLayer, InputLayer, MleInputLayer, enum_input_layer::InputLayerEnum, random_input_layer::RandomInputLayer}}};
use crate::{prover::{GKRCircuit, Layers, Witness, GKRError}};
use remainder_shared_types::{FieldExt, transcript::{Transcript, poseidon_transcript::PoseidonTranscript}};

use super::super::{builders::{InputPackingBuilder, SplitProductBuilder, EqualityCheck}, structs::{InputAttribute}};

use crate::prover::input_layer::enum_input_layer::CommitmentEnum;

pub(crate) struct FSPermutationCircuit<F: FieldExt> {
    input_data_mle_vec: Vec<DenseMle<F, InputAttribute<F>>>,               // batched
    permuted_input_data_mle_vec: Vec<DenseMle<F, InputAttribute<F>>>,      // batched
}

impl<F: FieldExt> GKRCircuit<F> for FSPermutationCircuit<F> {
    type Transcript = PoseidonTranscript<F>;
    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        unimplemented!()
    }

    fn synthesize_and_commit(
            &mut self,
            transcript: &mut Self::Transcript,
        ) -> Result<(Witness<F, Self::Transcript>, Vec<CommitmentEnum<F>>), GKRError> {
            let mut dummy_input_data_mle_combined = DenseMle::<F, InputAttribute<F>>::combine_mle_batch(self.input_data_mle_vec.clone());
            let mut dummy_permuted_input_data_mle_combined = DenseMle::<F, InputAttribute<F>>::combine_mle_batch(self.permuted_input_data_mle_vec.clone());
    
            let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![
                Box::new(&mut dummy_input_data_mle_combined),
                Box::new(&mut dummy_permuted_input_data_mle_combined),
            ];
            let input_layer = InputLayerBuilder::new(input_mles, None, LayerId::Input(0));
            let input_layer: PublicInputLayer<F, Self::Transcript> = input_layer.to_input_layer();
            // TODO!(ende) change back to ligero
            let mut input_layer = input_layer.to_enum();

            let input_commit = input_layer
                .commit()
                .map_err(GKRError::InputLayerError)?;
                InputLayerEnum::append_commitment_to_transcript(&input_commit, transcript).unwrap();
            

            // FS 
            let random_r = RandomInputLayer::new(transcript, 1, LayerId::Input(1));
            let r_mle = random_r.get_mle();
            let mut random_r = random_r.to_enum();
            let random_r_commit = random_r
                .commit()
                .map_err(GKRError::InputLayerError)?;

            let random_r_packing = RandomInputLayer::new(transcript, 1, LayerId::Input(2));
            let r_packing_mle = random_r_packing.get_mle();
            let mut random_r_packing = random_r_packing.to_enum();
            let random_r_packing_commit = random_r_packing
                .commit()
                .map_err(GKRError::InputLayerError)?;
            // FS

            let mut layers: Layers<_, Self::Transcript> = Layers::new();
    
            let batch_bits = log2(self.input_data_mle_vec.len()) as usize;
        
        
            let input_packing_builder = BatchedLayer::new(
                self.input_data_mle_vec.iter().map(
                    |input_data_mle| {
                        let mut input_data_mle = input_data_mle.clone();
                        // TODO!(ende) fix this atrocious fixed(false)
                        input_data_mle.set_prefix_bits(Some(dummy_input_data_mle_combined.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, batch_bits)).collect_vec()));
                        FSInputPackingBuilder::new(
                            input_data_mle,
                            r_mle.clone(),
                            r_packing_mle.clone()
                        )
                    }).collect_vec());
    
            let input_permuted_packing_builder = BatchedLayer::new(
                self.permuted_input_data_mle_vec.iter().map(
                    |input_data_mle| {
                        let mut input_data_mle = input_data_mle.clone();
                        // TODO!(ende) fix this atrocious fixed(true)
                        input_data_mle.set_prefix_bits(Some(dummy_permuted_input_data_mle_combined.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, batch_bits)).collect_vec()));
                        FSInputPackingBuilder::new(
                            input_data_mle,
                            r_mle.clone(),
                            r_packing_mle.clone()
                        )
                    }).collect_vec());
    
            let packing_builders = input_packing_builder.concat(input_permuted_packing_builder);
    
            let (mut input_packed, mut input_permuted_packed) = layers.add_gkr(packing_builders);
    
            let input_len = 1 << (self.input_data_mle_vec[0].num_iterated_vars() - 1);
            for _ in 0..log2(input_len) {
                let prod_builder = BatchedLayer::new(
                    input_packed.into_iter().map(
                        |input_packed| SplitProductBuilder::new(input_packed)
                    ).collect());
                let prod_permuted_builder = BatchedLayer::new(
                    input_permuted_packed.into_iter().map(
                        |input_permuted_packed| SplitProductBuilder::new(input_permuted_packed)
                    ).collect());
                let split_product_builders = prod_builder.concat(prod_permuted_builder);
                (input_packed, input_permuted_packed) = layers.add_gkr(split_product_builders);
            }
    
            let difference_builder = EqualityCheck::new_batched(
                input_packed,
                input_permuted_packed,
            );
    
            let difference_mle = layers.add_gkr(difference_builder);
    
            let circuit_output = combine_zero_mle_ref(difference_mle);
    
            Ok((
                Witness {
                    layers,
                    output_layers: vec![circuit_output.get_enum()],
                    input_layers: vec![input_layer, random_r, random_r_packing],
                },
                vec![input_commit, random_r_commit, random_r_packing_commit],
            ))

    }
}

impl<F: FieldExt> FSPermutationCircuit<F> {
    pub fn new(
        input_data_mle_vec: Vec<DenseMle<F, InputAttribute<F>>>,               // batched
        permuted_input_data_mle_vec: Vec<DenseMle<F, InputAttribute<F>>>,      // batched
    ) -> Self {
        Self {
            input_data_mle_vec,
            permuted_input_data_mle_vec
        }
    }
}

pub struct PermutationCircuit<F: FieldExt> {
    input_data_mle_vec: Vec<DenseMle<F, InputAttribute<F>>>,               // batched
    permuted_input_data_mle_vec: Vec<DenseMle<F, InputAttribute<F>>>,      // batched
    r: F,
    r_packing: F,
}

impl<F: FieldExt> GKRCircuit<F> for PermutationCircuit<F> {
    type Transcript = PoseidonTranscript<F>;
    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {

        let mut dummy_input_data_mle_combined = DenseMle::<F, InputAttribute<F>>::combine_mle_batch(self.input_data_mle_vec.clone());
        let mut dummy_permuted_input_data_mle_combined = DenseMle::<F, InputAttribute<F>>::combine_mle_batch(self.permuted_input_data_mle_vec.clone());

        let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![
            Box::new(&mut dummy_input_data_mle_combined),
            Box::new(&mut dummy_permuted_input_data_mle_combined),
        ];
        let input_layer = InputLayerBuilder::new(input_mles, None, LayerId::Input(0));
        let input_layer: PublicInputLayer<F, Self::Transcript> = input_layer.to_input_layer();
        // TODO!(ende) change back to ligero

        let mut layers: Layers<_, Self::Transcript> = Layers::new();

        let batch_bits = log2(self.input_data_mle_vec.len()) as usize;
    
    
        let input_packing_builder = BatchedLayer::new(
            self.input_data_mle_vec.iter().map(
                |input_data_mle| {
                    let mut input_data_mle = input_data_mle.clone();
                    // TODO!(ende) fix this atrocious fixed(false)
                    input_data_mle.set_prefix_bits(Some(dummy_input_data_mle_combined.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, batch_bits)).collect_vec()));
                    InputPackingBuilder::new(
                        input_data_mle,
                        self.r,
                        self.r_packing
                    )
                }).collect_vec());

        let input_permuted_packing_builder = BatchedLayer::new(
            self.permuted_input_data_mle_vec.iter().map(
                |input_data_mle| {
                    let mut input_data_mle = input_data_mle.clone();
                    // TODO!(ende) fix this atrocious fixed(true)
                    input_data_mle.set_prefix_bits(Some(dummy_permuted_input_data_mle_combined.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, batch_bits)).collect_vec()));
                    InputPackingBuilder::new(
                        input_data_mle,
                        self.r,
                        self.r_packing
                    )
                }).collect_vec());

        let packing_builders = input_packing_builder.concat(input_permuted_packing_builder);

        let (mut input_packed, mut input_permuted_packed) = layers.add_gkr(packing_builders);

        let input_len = 1 << (self.input_data_mle_vec[0].num_iterated_vars() - 1);
        for _ in 0..log2(input_len) {
            let prod_builder = BatchedLayer::new(
                input_packed.into_iter().map(
                    |input_packed| SplitProductBuilder::new(input_packed)
                ).collect());
            let prod_permuted_builder = BatchedLayer::new(
                input_permuted_packed.into_iter().map(
                    |input_permuted_packed| SplitProductBuilder::new(input_permuted_packed)
                ).collect());
            let split_product_builders = prod_builder.concat(prod_permuted_builder);
            (input_packed, input_permuted_packed) = layers.add_gkr(split_product_builders);
        }

        let difference_builder = EqualityCheck::new_batched(
            input_packed,
            input_permuted_packed,
        );

        let difference_mle = layers.add_gkr(difference_builder);

        let circuit_output = combine_zero_mle_ref(difference_mle);

        Witness {
            layers,
            output_layers: vec![circuit_output.get_enum()],
            input_layers: vec![input_layer.to_enum()],
        }
    }
}

impl<F: FieldExt> PermutationCircuit<F> {
    pub fn new(
        input_data_mle_vec: Vec<DenseMle<F, InputAttribute<F>>>,               // batched
        permuted_input_data_mle_vec: Vec<DenseMle<F, InputAttribute<F>>>,      // batched
        r: F,
        r_packing: F,
    ) -> Self {
        Self {
            input_data_mle_vec,
            permuted_input_data_mle_vec,
            r,
            r_packing
        }
    }
}

/// permutation circuit, non batched version
pub(crate) struct NonBatchedPermutationCircuit<F: FieldExt> {
    input_data_mle: DenseMle<F, InputAttribute<F>>,
    permuted_input_data_mle: DenseMle<F, InputAttribute<F>>,
    r: F,
    r_packing: F,
    input_len: usize,
}

impl<F: FieldExt> GKRCircuit<F> for NonBatchedPermutationCircuit<F> {
    type Transcript = PoseidonTranscript<F>;
    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        let mut layers: Layers<_, Self::Transcript> = Layers::new();

        // layer 0: packing
        let input_packing_builder: InputPackingBuilder<F> = InputPackingBuilder::new(
            self.input_data_mle.clone(),
            self.r,
            self.r_packing);

        let input_permuted_packing_builder: InputPackingBuilder<F> = InputPackingBuilder::new(
            self.permuted_input_data_mle.clone(),
            self.r,
            self.r_packing);

        let packing_builders = input_packing_builder.concat(input_permuted_packing_builder);

        let (mut input_packed, mut input_permuted_packed) = layers.add_gkr(packing_builders);

        for _ in 0..log2(self.input_len) {
            let prod_builder = SplitProductBuilder::new(
                input_packed
            );
            let permuted_prod_builder = SplitProductBuilder::new(
                input_permuted_packed
            );
            let split_product_builders = prod_builder.concat(permuted_prod_builder);
            (input_packed, input_permuted_packed) = layers.add_gkr(split_product_builders);
        }

        let difference_builder = EqualityCheck::new(
            input_packed,
            input_permuted_packed,
        );

        let _difference_mle = layers.add::<_, EmptyLayer<F, Self::Transcript>>(difference_builder);

        todo!()
    }
}

impl<F: FieldExt> NonBatchedPermutationCircuit<F> {
    pub fn new(
        input_data_mle: DenseMle<F, InputAttribute<F>>,
        permuted_input_data_mle: DenseMle<F, InputAttribute<F>>,
        r: F,
        r_packing: F,
        input_len: usize,
    ) -> Self {
        Self {
            input_data_mle,
            permuted_input_data_mle,
            r,
            r_packing,
            input_len
        }
    }
}