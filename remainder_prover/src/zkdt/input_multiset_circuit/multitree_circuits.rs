use ark_std::{log2};
use itertools::{Itertools, repeat_n, multizip};

use crate::{mle::{dense::DenseMle, MleRef, Mle, MleIndex}, layer::{LayerBuilder, batched::{BatchedLayer, combine_zero_mle_ref}, LayerId, Padding}, zkdt::{builders::{BitExponentiationBuilderInput, FSInputPackingBuilder, FSRMinusXBuilder}, structs::BinDecomp4Bit}, prover::{input_layer::{combine_input_layers::InputLayerBuilder, public_input_layer::PublicInputLayer, InputLayer, MleInputLayer, enum_input_layer::InputLayerEnum, random_input_layer::RandomInputLayer}}};
use crate::{prover::{GKRCircuit, Layers, Witness, GKRError}};
use remainder_shared_types::{FieldExt, transcript::{Transcript, poseidon_transcript::PoseidonTranscript}};

use super::super::{builders::{SplitProductBuilder, EqualityCheck, SquaringBuilder, ProductBuilder}, structs::{InputAttribute}};

use crate::prover::input_layer::enum_input_layer::CommitmentEnum;

/// Computes the multiset characteristic polynomial evaluated
/// at `r_mle` for any input x and its "permutation" `\bar{x}`.
pub(crate) struct InputMultiSetCircuitMultiTree<F: FieldExt> {
    pub input_data_mle_vec: Vec<DenseMle<F, InputAttribute<F>>>,
    pub permuted_input_data_mle_vec_tree: Vec<Vec<DenseMle<F, InputAttribute<F>>>>,
    pub multiplicities_bin_decomp_mle_input_vec_tree: Vec<Vec<DenseMle<F, BinDecomp4Bit<F>>>>,
    pub r_mle: DenseMle<F, F>,
    pub r_packing_mle: DenseMle<F, F>,
}

impl<F: FieldExt> InputMultiSetCircuitMultiTree<F> {
    pub fn new(
        input_data_mle_vec: Vec<DenseMle<F, InputAttribute<F>>>,
        permuted_input_data_mle_vec_tree: Vec<Vec<DenseMle<F, InputAttribute<F>>>>,
        multiplicities_bin_decomp_mle_input_vec_tree: Vec<Vec<DenseMle<F, BinDecomp4Bit<F>>>>,
        r_mle: DenseMle<F, F>,
        r_packing_mle: DenseMle<F, F>,
    ) -> Self {
        Self {
            input_data_mle_vec,
            permuted_input_data_mle_vec_tree,
            multiplicities_bin_decomp_mle_input_vec_tree,
            r_mle,
            r_packing_mle,
        }
    }
}

impl<F: FieldExt> GKRCircuit<F> for InputMultiSetCircuitMultiTree<F> {
    type Transcript = PoseidonTranscript<F>;
    
    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        unimplemented!()
    }

    fn synthesize_and_commit(
            &mut self,
            transcript: &mut Self::Transcript,
        ) -> Result<(Witness<F, Self::Transcript>, Vec<CommitmentEnum<F>>), GKRError> {
            unimplemented!()
    }
}

impl<F: FieldExt> InputMultiSetCircuitMultiTree<F> {
    pub fn yield_sub_circuit(&mut self) -> Witness<F, PoseidonTranscript<F>> {

        let mut layers: Layers<_, PoseidonTranscript<F>> = Layers::new();

        let num_dataparallel_bits = log2(self.multiplicities_bin_decomp_mle_input_vec_tree[0].len()) as usize;
        let num_tree_bits = log2(self.multiplicities_bin_decomp_mle_input_vec_tree.len()) as usize;

        self.input_data_mle_vec.iter_mut().for_each(|mle| {
            mle.set_prefix_bits(Some(mle.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, num_dataparallel_bits)).collect_vec()));
        });


        self.permuted_input_data_mle_vec_tree.iter_mut().for_each(|mle_vec| {
            mle_vec.iter_mut().for_each(|mle| {
                mle.set_prefix_bits(Some(mle.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, num_dataparallel_bits + num_tree_bits)).collect_vec()));
            })
        });


        self.multiplicities_bin_decomp_mle_input_vec_tree.iter_mut().for_each(|mle_vec| {
            mle_vec.iter_mut().for_each(|mle| {
                mle.set_prefix_bits(Some(mle.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, num_dataparallel_bits + num_tree_bits)).collect_vec()));
            })
        });


        // --- Layer 0: Compute the "packed" version of the decision and leaf tree nodes ---
        // Note that this also "evaluates" each packed entry at the random characteristic polynomial
        // evaluation challenge point `self.r`.

        // let mut input_data_mle_vec_tree = vec![self.input_data_mle_vec.clone(); 1 << num_tree_bits];
        // let input_packing_builders_vec = input_data_mle_vec_tree.iter_mut().map(
        //     |input_data_mle_vec| {
        //         BatchedLayer::new(
        //             input_data_mle_vec.iter().map(
        //                 |input_data_mle| {
        //                     let mut input_data_mle: DenseMle<F, InputAttribute<F>> = input_data_mle.clone();
        //                     FSInputPackingBuilder::new(
        //                         input_data_mle,
        //                         self.r_mle.clone(),
        //                         self.r_packing_mle.clone()
        //                     )
        //                 }).collect_vec())
        //     }
        // ).collect_vec();

        let input_packing_builders = BatchedLayer::new(
            self.input_data_mle_vec.iter().map(
                |input_data_mle| {
                    let mut input_data_mle: DenseMle<F, InputAttribute<F>> = input_data_mle.clone();
                    FSInputPackingBuilder::new(
                        input_data_mle,
                        self.r_mle.clone(),
                        self.r_packing_mle.clone()
                    )
                }).collect_vec());

        // let input_packed_vecs = layers.add_gkr(BatchedLayer::new(input_packing_builders_vec));
        let input_packed_vec = layers.add_gkr(input_packing_builders);

        // --- Layer 2, part 1: computes (r - x) * b_ij + (1 - b_ij) ---
        // Note that this is for the actual exponentiation computation:
        // we have that (r - x)^c_i = \prod_{j = 0}^{15} (r - x)^{2^{b_ij}} * b_{ij} + (1 - b_ij)
        // where \sum_{j = 0}^{15} 2^j b_{ij} = c_i.

        
        let layer_2_builders_vec = self.multiplicities_bin_decomp_mle_input_vec_tree.iter().map(
            |multiplicities_bin_decomp_mle_input_vec| {
                let prev_prod_builder = BatchedLayer::new(
                    input_packed_vec.clone().iter().zip(
                        multiplicities_bin_decomp_mle_input_vec.iter()
                    ).map(|(r_minus_x_power, multiplicities_bin_decomp_mle_input)| {
                        BitExponentiationBuilderInput::new(
                            multiplicities_bin_decomp_mle_input.clone(),
                            0,
                            r_minus_x_power.clone()
                        )
                    }).collect_vec()
                );

                let r_minus_x_square_builders = BatchedLayer::new(
                    input_packed_vec.iter().map(
                        |r_minus_x_power| {
                            let r_minus_x_power = r_minus_x_power.clone();
                            SquaringBuilder::new(
                                r_minus_x_power
                            )
                        }
                    ).collect_vec()
                );

                (prev_prod_builder, r_minus_x_square_builders)

            }
        ).collect_vec();

        // --- Layer 2, part 2: (r - x)^2 ---
        // Note that we need to compute (r - x)^{2^0}, ..., (r - x)^{2^{3}}
        // We do this via repeated squaring of the previous power.

        let (prev_prod_builder_vec, r_minus_x_square_builders_vec): (Vec<_>, Vec<_>) = layer_2_builders_vec.into_iter().unzip();
        
        let (prev_prod_vecs, r_minus_x_power_vecs): (Vec<_>, Vec<_>) = layers.add_gkr(BatchedLayer::new(prev_prod_builder_vec).concat(BatchedLayer::new(r_minus_x_square_builders_vec)));

        // layer 3, part 1: (r - x)^2 * b_ij + (1 - b_ij)
        // layer 3, part 2: (r - x)^4
        let (prev_prod_builders_vec, r_minus_x_square_builders_vec): (Vec<_>, Vec<_>) = r_minus_x_power_vecs.into_iter().zip(self.multiplicities_bin_decomp_mle_input_vec_tree.iter()).map(
            |(r_minus_x_power_vec, multiplicities_bin_decomp_mle_input_vec)| {
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

                 
                let r_minus_x_square_builders = BatchedLayer::new(
                    r_minus_x_power_vec.iter().map(
                        |r_minus_x_power| {
                            let r_minus_x_power = r_minus_x_power.clone();
                            SquaringBuilder::new(
                                r_minus_x_power
                            )
                        }
                    ).collect_vec()
                );

                (prev_prod_builders, r_minus_x_square_builders)

            }
        ).unzip();
        

        let (curr_prod_vecs, r_minus_x_power_vecs): (Vec<_>, Vec<_>) = layers.add_gkr(BatchedLayer::new(prev_prod_builders_vec).concat(BatchedLayer::new(r_minus_x_square_builders_vec)));
        
        let (r_minus_x_square_builders_vec, curr_prod_builders_vec, prod_builders_vec) = 
            multizip((r_minus_x_power_vecs.iter(), self.multiplicities_bin_decomp_mle_input_vec_tree.iter(), curr_prod_vecs.iter(), prev_prod_vecs.iter())).map(
                |(r_minus_x_power_vec, multiplicities_bin_decomp_mle_input_vec, curr_prod_vec, prev_prod_vec)| {
                    // layer 4, part 1
                    let r_minus_x_square_builders = BatchedLayer::new(
                        r_minus_x_power_vec.iter().map(
                            |r_minus_x_power| {
                                let r_minus_x_power = r_minus_x_power.clone();
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

                    (r_minus_x_square_builders, curr_prod_builders, prod_builders)

                    // let layer_4_builders = r_minus_x_square_builders.concat(curr_prod_builders).concat_with_padding(prod_builders, Padding::Right(1));

                    // layer_4_builders
                }
            ).multiunzip();
        
        let ((r_minus_x_power_vecs, curr_prod_vecs), prev_prod_vecs) = layers.add_gkr(BatchedLayer::new(r_minus_x_square_builders_vec).concat(BatchedLayer::new(curr_prod_builders_vec)).concat_with_padding(BatchedLayer::new(prod_builders_vec), Padding::Right(1)));
        
        // at this point we have
        // (r - x)^(2^3), (r - x)^(2^2) * b_ij + (1 - b_ij), PROD ALL[(r - x)^(2^1) * b_ij + (1 - b_ij)]
        // need to BitExponentiate 1 time
        // and PROD w prev_prod 2 times
        let (curr_prod_builders_vec, prod_builders_vec) = 
            multizip((r_minus_x_power_vecs.iter(), self.multiplicities_bin_decomp_mle_input_vec_tree.iter(), curr_prod_vecs.iter(), prev_prod_vecs.iter())).map(
                |(r_minus_x_power_vec, multiplicities_bin_decomp_mle_input_vec, curr_prod_vec, prev_prod_vec)| {
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

                    (curr_prod_builders, prod_builders)

                    // let layer_5_builders = curr_prod_builders.concat(prod_builders);
                    // layer_5_builders
                }
            ).unzip();
        
        
        let (curr_prod_vecs, prev_prod_vecs) = layers.add_gkr(BatchedLayer::new(curr_prod_builders_vec).concat(BatchedLayer::new(prod_builders_vec)));

        
        let prod_builders_vec = curr_prod_vecs.iter().zip(prev_prod_vecs.iter()).map(
            |(curr_prod_vec, prev_prod_vec)| {
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
                prod_builders
            }
        ).collect_vec();
        
        let prev_prod_vecs = layers.add_gkr(BatchedLayer::new(prod_builders_vec));

        let mut exponentiated_input_vecs = prev_prod_vecs;

       let input_len = 1 << (self.input_data_mle_vec[0].num_iterated_vars() - 1);
                for _ in 0..log2(input_len) {
                    let split_product_builders_vec = exponentiated_input_vecs.clone().into_iter().map(
                        |exponentiated_input_vec| {
                            BatchedLayer::new(
                                exponentiated_input_vec.into_iter().map(
                                    |exponentiated_input| SplitProductBuilder::new(exponentiated_input)
                                ).collect())
                        }
                    ).collect_vec();
                    
                    exponentiated_input_vecs = layers.add_gkr(BatchedLayer::new(split_product_builders_vec));
                }
            

        // **** above is input exponentiated ****
        // **** below is all decision nodes on the path multiplied ****

        // layer 13: packing

        let path_packing_builders_vec = self.permuted_input_data_mle_vec_tree.iter().map(
            |permuted_input_data_mle_vec| {
                let path_packing_builders = BatchedLayer::new(
                    permuted_input_data_mle_vec.iter().map(
                        |permuted_input_data_mle| {
                            let mut permuted_input_data_mle = permuted_input_data_mle.clone();
                            FSInputPackingBuilder::new(
                                permuted_input_data_mle,
                                self.r_mle.clone(),
                                self.r_packing_mle.clone()
                            )
                        }
                    ).collect_vec());
                path_packing_builders
            }
        ).collect_vec();


        
        let path_packed_vecs = layers.add_gkr(BatchedLayer::new(path_packing_builders_vec));

        let mut path_product_vecs = path_packed_vecs;

        // layer 15, 16, 17

        
            let permuted_input_len = 1 << (self.permuted_input_data_mle_vec_tree[0][0].num_iterated_vars() - 1);
            for _ in 0..log2(permuted_input_len) {
                let split_product_builders_vec = path_product_vecs.clone().into_iter().map(
                    |path_product_vec| {
                        BatchedLayer::new(
                            path_product_vec.into_iter().map(
                                |path_product| SplitProductBuilder::new(path_product)
                            ).collect())
                    }
                ).collect_vec();
                
                path_product_vecs = layers.add_gkr(BatchedLayer::new(split_product_builders_vec));
            }


        let difference_builder_vecs = exponentiated_input_vecs.into_iter().zip(path_product_vecs.into_iter()).map(
            |(exponentiated_input_vec, path_product_vec)| {
                // layer 18
                EqualityCheck::new_batched(
                    exponentiated_input_vec,
                    path_product_vec
                )
            }
        ).collect_vec();

        let circuit_output_vecs = layers.add_gkr(BatchedLayer::new(difference_builder_vecs));

        let circuit_output = combine_zero_mle_ref(
            circuit_output_vecs.into_iter().map(
                |circuit_output_vec| {
                    combine_zero_mle_ref(circuit_output_vec)
                }
            ).collect_vec()
        );

        println!("# layers -- input multiset: {:?}", layers.next_layer_id());

        Witness {
            layers,
            output_layers: vec![circuit_output.get_enum()],
            input_layers: vec![],
        }
    }
}