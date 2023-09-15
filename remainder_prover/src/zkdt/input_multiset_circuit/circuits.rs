use ark_bn254::Fr;
use ark_crypto_primitives::sponge::poseidon::get_default_poseidon_parameters_internal;
use ark_ff::BigInteger;
use rand::Rng;
use rayon::{iter::Split, vec};
use tracing_subscriber::fmt::layer;
use std::{io::Empty, iter};

use ark_std::{log2, test_rng};
use itertools::{Itertools, repeat_n};

use crate::{mle::{dense::DenseMle, MleRef, beta::BetaTable, Mle, MleIndex}, layer::{LayerBuilder, empty_layer::EmptyLayer, batched::{BatchedLayer, combine_zero_mle_ref, unbatch_mles}, LayerId, Padding}, sumcheck::{compute_sumcheck_message, Evals, get_round_degree}, zkdt::{builders::{BitExponentiationBuilderInput, IdentityBuilder, FSInputPackingBuilder, FSDecisionPackingBuilder, FSLeafPackingBuilder, FSRMinusXBuilder}, structs::BinDecomp4Bit}, prover::{input_layer::{ligero_input_layer::LigeroInputLayer, combine_input_layers::InputLayerBuilder, public_input_layer::PublicInputLayer, InputLayer, MleInputLayer, enum_input_layer::InputLayerEnum, self, random_input_layer::RandomInputLayer}, combine_layers::combine_layers}};
use crate::{prover::{GKRCircuit, Layers, Witness, GKRError}, mle::{mle_enum::MleEnum}};
use remainder_shared_types::{FieldExt, transcript::{Transcript, poseidon_transcript::PoseidonTranscript}};

use super::super::{builders::{InputPackingBuilder, SplitProductBuilder, EqualityCheck, DecisionPackingBuilder, LeafPackingBuilder, ConcatBuilder, RMinusXBuilder, BitExponentiationBuilder, SquaringBuilder, ProductBuilder}, structs::{InputAttribute, DecisionNode, LeafNode, BinDecomp16Bit}, binary_recomp_circuit::circuit_builders::{BinaryRecompBuilder, NodePathDiffBuilder, BinaryRecompCheckerBuilder, PartialBitsCheckerBuilder}, data_pipeline::dummy_data_generator::{BatchedCatboostMles, generate_mles_batch_catboost_single_tree}};

use crate::prover::input_layer::enum_input_layer::CommitmentEnum;

pub(crate) struct InputMultiSetCircuit<F: FieldExt> {
    input_data_mle_vec: Vec<DenseMle<F, InputAttribute<F>>>,
    permuted_input_data_mle_vec: Vec<DenseMle<F, InputAttribute<F>>>,
    multiplicities_bin_decomp_mle_input_vec: Vec<DenseMle<F, BinDecomp4Bit<F>>>,
}

impl<F: FieldExt> InputMultiSetCircuit<F> {
    pub fn new(
        input_data_mle_vec: Vec<DenseMle<F, InputAttribute<F>>>,
        permuted_input_data_mle_vec: Vec<DenseMle<F, InputAttribute<F>>>,
        multiplicities_bin_decomp_mle_input_vec: Vec<DenseMle<F, BinDecomp4Bit<F>>>,
    ) -> Self {
        Self {
            input_data_mle_vec,
            permuted_input_data_mle_vec,
            multiplicities_bin_decomp_mle_input_vec,
        }
    }
}

impl<F: FieldExt> GKRCircuit<F> for InputMultiSetCircuit<F> {
    type Transcript = PoseidonTranscript<F>;
    
    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        unimplemented!()
    }

    fn synthesize_and_commit(
            &mut self,
            transcript: &mut Self::Transcript,
        ) -> Result<(Witness<F, Self::Transcript>, Vec<CommitmentEnum<F>>), GKRError> {
        
            let mut input_data_mle_vec_combined = DenseMle::<F, InputAttribute<F>>::combine_mle_batch(self.input_data_mle_vec.clone());
            let mut permuted_input_data_mle_vec_combined = DenseMle::<F, InputAttribute<F>>::combine_mle_batch(self.permuted_input_data_mle_vec.clone());
            let mut multiplicities_bin_decomp_mle_input_vec_combined = DenseMle::<F, BinDecomp4Bit<F>>::combine_mle_batch(self.multiplicities_bin_decomp_mle_input_vec.clone());
    
            let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![
                Box::new(&mut input_data_mle_vec_combined),
                Box::new(&mut permuted_input_data_mle_vec_combined),
                Box::new(&mut multiplicities_bin_decomp_mle_input_vec_combined),
            ];

            let input_layer = InputLayerBuilder::new(input_mles, None, LayerId::Input(0));

            let batch_bits = log2(self.input_data_mle_vec.len()) as usize;

            for multiplicities_bin_decomp_mle_input in self.multiplicities_bin_decomp_mle_input_vec.iter_mut() {
                multiplicities_bin_decomp_mle_input.add_prefix_bits(Some(multiplicities_bin_decomp_mle_input_vec_combined.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, batch_bits)).collect_vec()))
            }

            let input_layer: PublicInputLayer<F, Self::Transcript> = input_layer.to_input_layer();
            let mut input_layer = input_layer.to_enum();
    
            let input_commit = input_layer
                .commit()
                .map_err(|err| GKRError::InputLayerError(err))?;
                InputLayerEnum::append_commitment_to_transcript(&input_commit, transcript).unwrap();

            // FS 
            let random_r = RandomInputLayer::new(transcript, 1, LayerId::Input(1));
            let r_mle = random_r.get_mle();
            let mut random_r = random_r.to_enum();
            let random_r_commit = random_r
                .commit()
                .map_err(|err| GKRError::InputLayerError(err))?;

            let random_r_another = RandomInputLayer::new(transcript, 1, LayerId::Input(2));
            let r_mle_another = random_r_another.get_mle();
            let mut random_r_another = random_r_another.to_enum();
            let random_r_another_commit = random_r_another
                .commit()
                .map_err(|err| GKRError::InputLayerError(err))?;

            let random_r_packing = RandomInputLayer::new(transcript, 1, LayerId::Input(3));
            let r_packing_mle = random_r_packing.get_mle();
            let mut random_r_packing = random_r_packing.to_enum();
            let random_r_packing_commit = random_r_packing
                .commit()
                .map_err(|err| GKRError::InputLayerError(err))?;
            // FS

            let mut layers: Layers<_, Self::Transcript> = Layers::new();
    
            let batch_bits = log2(self.input_data_mle_vec.len()) as usize;
        
            let input_packing_builders = BatchedLayer::new(
                self.input_data_mle_vec.iter().map(
                    |input_data_mle| {
                        let mut input_data_mle: DenseMle<F, InputAttribute<F>> = input_data_mle.clone();
                        input_data_mle.add_prefix_bits(Some(input_data_mle_vec_combined.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, batch_bits)).collect_vec()));
                        FSInputPackingBuilder::new(
                            input_data_mle,
                            r_mle.clone(),
                            r_packing_mle.clone()
                        )
                    }).collect_vec());
    
            let input_permuted_packing_builders = BatchedLayer::new(
                self.permuted_input_data_mle_vec.iter().map(
                    |input_data_mle| {
                        let mut input_data_mle = input_data_mle.clone();
                        input_data_mle.add_prefix_bits(Some(permuted_input_data_mle_vec_combined.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, batch_bits)).collect_vec()));
                        FSInputPackingBuilder::new(
                            input_data_mle,
                            r_mle.clone(),
                            r_packing_mle.clone()
                        )
                    }).collect_vec());
    
            let packing_builders = input_packing_builders.concat(input_permuted_packing_builders);
    
            let (mut input_packed_vec, mut input_permuted_packed_vec) = layers.add_gkr(packing_builders);
    
            // layer 1: (r - x)
            let r_minus_x_builders = BatchedLayer::new(
                input_packed_vec.iter().map(
                    |input_packed| {
                        let mut input_packed = input_packed.clone();
                        FSRMinusXBuilder::new(
                            input_packed, r_mle_another.clone()
                        )
                    }
                ).collect_vec()
            );
            
            let r_minus_x_power_vec = layers.add_gkr(r_minus_x_builders);
    
            let mut multiplicities_bin_decomp_mle_input_vec = self.multiplicities_bin_decomp_mle_input_vec.clone();

            // layer 2, part 1: (r - x) * b_ij + (1 - b_ij)
            let prev_prod_builders = BatchedLayer::new(
                r_minus_x_power_vec.iter().zip(
                    multiplicities_bin_decomp_mle_input_vec.iter()
                ).map(|(r_minus_x_power, multiplicities_bin_decomp_mle_input)| {
                    BitExponentiationBuilderInput::new(
                        multiplicities_bin_decomp_mle_input.clone(),
                        0,
                        r_minus_x_power.clone()
                    )
                }).collect_vec()
            );
    
            // layer 2, part 2: (r - x)^2
            let r_minus_x_square_builders = BatchedLayer::new(
                r_minus_x_power_vec.iter().map(
                    |r_minus_x_power| {
                        let mut r_minus_x_power = r_minus_x_power.clone();
                        SquaringBuilder::new(
                            r_minus_x_power
                        )
                    }
                ).collect_vec()
            );
    
            let layer_2_builders = prev_prod_builders.concat(r_minus_x_square_builders);
            let (mut prev_prod_vec, r_minus_x_power_vec) = layers.add_gkr(layer_2_builders);
    
            // layer 3, part 1: (r - x)^2 * b_ij + (1 - b_ij)
            let prev_prod_builders= BatchedLayer::new(
                r_minus_x_power_vec.iter().zip(
                    multiplicities_bin_decomp_mle_input_vec.iter()
                ).map(
                    |(r_minus_x_power, multiplicities_bin_decomp_mle)| {
                        BitExponentiationBuilderInput::new(
                            multiplicities_bin_decomp_mle.clone(),
                            1,
                            r_minus_x_power.clone()
                        )
                    }
                ).collect_vec()
            );
    
            // layer 3, part 2: (r - x)^4
            let r_minus_x_square_builders = BatchedLayer::new(
                r_minus_x_power_vec.iter().map(
                    |r_minus_x_power| {
                        let mut r_minus_x_power = r_minus_x_power.clone();
                        SquaringBuilder::new(
                            r_minus_x_power
                        )
                    }
                ).collect_vec()
            );
    
            let layer_3_builders = prev_prod_builders.concat(r_minus_x_square_builders);
            let (mut curr_prod_vec, mut r_minus_x_power_vec) = layers.add_gkr(layer_3_builders);
    
            // need to square from (r - x)^(2^2) to (r - x)^(2^15),
            // so needs 13 more iterations
            // in each iteration, get the following:
            // (r - x)^(2^(i+1)), (r - x)^(2^i) * b_ij + (1 - b_ij), PROD ALL[(r - x)^(2^(i-1)) * b_ij + (1 - b_ij)]
    
            // layer 4, part 1
            let r_minus_x_square_builders = BatchedLayer::new(
                r_minus_x_power_vec.iter().map(
                    |r_minus_x_power| {
                        let mut r_minus_x_power = r_minus_x_power.clone();
                        SquaringBuilder::new(
                            r_minus_x_power
                        )
                    }
                ).collect_vec()
            );

            // layer 4, part 2
            let curr_prod_builders = BatchedLayer::new(
                r_minus_x_power_vec.iter().zip(
                    multiplicities_bin_decomp_mle_input_vec.iter()
                ).map(
                    |(r_minus_x_power, multiplicities_bin_decomp_mle)| {
                        BitExponentiationBuilderInput::new(
                            multiplicities_bin_decomp_mle.clone(),
                            2,
                            r_minus_x_power.clone()
                        )
                    }
                ).collect_vec()
            );

            // layer 4, part 3
            let prod_builders = BatchedLayer::new(
                curr_prod_vec.iter().zip(
                    prev_prod_vec.iter()
                ).map(|(curr_prod, prev_prod)| {
                    ProductBuilder::new(
                        curr_prod.clone(),
                        prev_prod.clone()
                    )
                }).collect_vec()
            );

            let layer_4_builders = r_minus_x_square_builders.concat(curr_prod_builders).concat_with_padding(prod_builders, Padding::Right(1));

            ((r_minus_x_power_vec, curr_prod_vec), prev_prod_vec) = layers.add_gkr(layer_4_builders);
    
            // at this point we have
            // (r - x)^(2^3), (r - x)^(2^2) * b_ij + (1 - b_ij), PROD ALL[(r - x)^(2^1) * b_ij + (1 - b_ij)]
            // need to BitExponentiate 1 time
            // and PROD w prev_prod 2 times
    
            // layer 5, part 1
            let curr_prod_builders = BatchedLayer::new(
                r_minus_x_power_vec.iter().zip(
                    multiplicities_bin_decomp_mle_input_vec.iter()
                ).map(
                    |(r_minus_x_power, multiplicities_bin_decomp_mle)| {
                        BitExponentiationBuilderInput::new(
                            multiplicities_bin_decomp_mle.clone(),
                            3,
                            r_minus_x_power.clone()
                        )
                    }
                ).collect_vec()
            );
    
            // layer 5, part 2
            let prod_builders = BatchedLayer::new(
                curr_prod_vec.iter().zip(
                    prev_prod_vec.iter()
                ).map(|(curr_prod, prev_prod)| {
                    ProductBuilder::new(
                        curr_prod.clone(),
                        prev_prod.clone()
                    )
                }).collect_vec()
            );
    
            let layer_5_builders = curr_prod_builders.concat(prod_builders);
            let (curr_prod_vec, prev_prod_vec) = layers.add_gkr(layer_5_builders);
    
            // layer 6
            let prod_builders = BatchedLayer::new(
                curr_prod_vec.iter().zip(
                    prev_prod_vec.iter()
                ).map(|(curr_prod, prev_prod)| {
                    ProductBuilder::new(
                        curr_prod.clone(),
                        prev_prod.clone()
                    )
                }).collect_vec()
            );
    
            let prev_prod_vec = layers.add_gkr(prod_builders);
    
            let mut exponentiated_input_vec = prev_prod_vec;

    
            let input_len = 1 << (self.input_data_mle_vec[0].num_iterated_vars() - 1);
            for _ in 0..log2(input_len) {
                let split_product_builders = BatchedLayer::new(
                    exponentiated_input_vec.into_iter().map(
                        |exponentiated_input| SplitProductBuilder::new(exponentiated_input)
                    ).collect());

                exponentiated_input_vec = layers.add_gkr(split_product_builders);
            }
            
            // **** above is input exponentiated ****
            // **** below is all decision nodes on the path multiplied ****
    
            // layer 13: packing
    
            let batch_bits = log2(self.permuted_input_data_mle_vec.len()) as usize;
    
            let path_packing_builders = BatchedLayer::new(
                self.permuted_input_data_mle_vec.iter().map(
                    |permuted_input_data_mle| {
                        let mut permuted_input_data_mle = permuted_input_data_mle.clone();
                        permuted_input_data_mle.add_prefix_bits(Some(permuted_input_data_mle_vec_combined.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, batch_bits)).collect_vec()));
                        FSInputPackingBuilder::new(
                            permuted_input_data_mle,
                            r_mle.clone(),
                            r_packing_mle.clone()
                        )
                    }
                ).collect_vec());
            let path_packed_vec = layers.add_gkr(path_packing_builders);
    
            // layer 14: r - x
    
            let r_minus_x_path_builders = BatchedLayer::new(
                path_packed_vec.iter().map(|path_packed| FSRMinusXBuilder::new(
                    path_packed.clone(),
                    r_mle_another.clone()
                )).collect_vec());
    
            let r_minus_x_path_vec = layers.add_gkr(r_minus_x_path_builders);
            let mut path_product_vec = r_minus_x_path_vec;

            // layer 15, 16, 17
            let permuted_input_len = 1 << (self.permuted_input_data_mle_vec[0].num_iterated_vars() - 1);
            for _ in 0..log2(permuted_input_len) {
                let split_product_builders = BatchedLayer::new(
                    path_product_vec.into_iter().map(
                        |path_product| SplitProductBuilder::new(path_product)
                    ).collect());

                    path_product_vec = layers.add_gkr(split_product_builders);
            }
    
            // layer 18
            let difference_builder = EqualityCheck::new_batched(
                exponentiated_input_vec,
                path_product_vec
            );
    
            let circuit_output = layers.add_gkr(difference_builder);

            let circuit_output = combine_zero_mle_ref(circuit_output);

            Ok((
                Witness {
                    layers,
                    output_layers: vec![circuit_output.get_enum()],
                    input_layers: vec![input_layer, random_r, random_r_another, random_r_packing],
                },
                vec![input_commit, random_r_commit, random_r_another_commit, random_r_packing_commit],
            ))
        }
}