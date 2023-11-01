use ark_std::{log2};
use itertools::{Itertools, repeat_n, multizip};

use crate::{mle::{dense::DenseMle, MleRef, Mle, MleIndex}, layer::{LayerBuilder, batched::{BatchedLayer, combine_zero_mle_ref, unbatch_mles, unflatten_mle}, LayerId, Padding}, zkdt::{builders::{BitExponentiationBuilderInput, FSInputPackingBuilder, FSRMinusXBuilder, DumbProduct}, structs::{BinDecomp4Bit, BinDecomp8Bit}}, prover::{input_layer::{combine_input_layers::InputLayerBuilder, public_input_layer::PublicInputLayer, InputLayer, MleInputLayer, enum_input_layer::InputLayerEnum, random_input_layer::RandomInputLayer}}};
use crate::{prover::{GKRCircuit, Layers, Witness, GKRError}};
use remainder_shared_types::{FieldExt, transcript::{Transcript, poseidon_transcript::PoseidonTranscript}};

use super::super::{builders::{SplitProductBuilder, EqualityCheck, SquaringBuilder, ProductBuilder}, structs::{InputAttribute}};

use crate::prover::input_layer::enum_input_layer::CommitmentEnum;

/// Computes the multiset characteristic polynomial evaluated
/// at `r_mle` for any input x and its "permutation" `\bar{x}`.
pub(crate) struct InputMultiSetCircuitMultiTree<F: FieldExt> {
    pub input_data_mle_vec: Vec<DenseMle<F, InputAttribute<F>>>,
    pub permuted_input_data_mle_vec_tree: Vec<Vec<DenseMle<F, InputAttribute<F>>>>,
    pub multiplicities_bin_decomp_mle_input_vec: Vec<DenseMle<F, BinDecomp8Bit<F>>>,
    pub r_mle: DenseMle<F, F>,
    pub r_packing_mle: DenseMle<F, F>,
}

impl<F: FieldExt> InputMultiSetCircuitMultiTree<F> {
    pub fn new(
        input_data_mle_vec: Vec<DenseMle<F, InputAttribute<F>>>,
        permuted_input_data_mle_vec_tree: Vec<Vec<DenseMle<F, InputAttribute<F>>>>,
        multiplicities_bin_decomp_mle_input_vec: Vec<DenseMle<F, BinDecomp8Bit<F>>>,
        r_mle: DenseMle<F, F>,
        r_packing_mle: DenseMle<F, F>,
    ) -> Self {
        Self {
            input_data_mle_vec,
            permuted_input_data_mle_vec_tree,
            multiplicities_bin_decomp_mle_input_vec,
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

        let num_dataparallel_bits = log2(self.permuted_input_data_mle_vec_tree[0].len()) as usize;
        let num_tree_bits = log2(self.permuted_input_data_mle_vec_tree.len()) as usize;

        self.input_data_mle_vec.iter_mut().for_each(|mle| {
            mle.set_prefix_bits(Some(mle.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, num_dataparallel_bits)).collect_vec()));
        });


        self.permuted_input_data_mle_vec_tree.iter_mut().for_each(|mle_vec| {
            mle_vec.iter_mut().for_each(|mle| {
                mle.set_prefix_bits(Some(mle.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, num_dataparallel_bits + num_tree_bits)).collect_vec()));
            })
        });

        self.multiplicities_bin_decomp_mle_input_vec.iter_mut().for_each(|mle| {
            mle.set_prefix_bits(Some(mle.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, num_dataparallel_bits)).collect_vec()));
        });


        // --- Layer 0: Compute the "packed" version of the decision and leaf tree nodes ---
        // Note that this also "evaluates" each packed entry at the random characteristic polynomial
        // evaluation challenge point `self.r`.
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

        // let input_permuted_packing_builders = BatchedLayer::new(
        //     self.permuted_input_data_mle_vec.iter().map(
        //         |input_data_mle| {
        //             let mut input_data_mle = input_data_mle.clone();
        //             input_data_mle.add_prefix_bits(Some(input_data_mle.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, batch_bits)).collect_vec()));
        //             FSInputPackingBuilder::new(
        //                 input_data_mle,
        //                 self.r_mle.clone(),
        //                 self.r_packing_mle.clone()
        //             )
        //         }).collect_vec());

        // let packing_builders = input_packing_builders.concat(input_permuted_packing_builders);

        let input_packed_vec = layers.add_gkr(input_packing_builders);

        // let (input_packed_vec, _input_permuted_packed_vec) = layers.add_gkr(packing_builders);

        let multiplicities_bin_decomp_mle_input_vec = self.multiplicities_bin_decomp_mle_input_vec.clone();

        // --- Layer 2, part 1: computes (r - x) * b_ij + (1 - b_ij) ---
        // Note that this is for the actual exponentiation computation:
        // we have that (r - x)^c_i = \prod_{j = 0}^{15} (r - x)^{2^{b_ij}} * b_{ij} + (1 - b_ij)
        // where \sum_{j = 0}^{15} 2^j b_{ij} = c_i.
        let prev_prod_builders = BatchedLayer::new(
            input_packed_vec.iter().zip(
                multiplicities_bin_decomp_mle_input_vec.iter()
            ).map(|(r_minus_x_power, multiplicities_bin_decomp_mle_input)| {
                BitExponentiationBuilderInput::new(
                    multiplicities_bin_decomp_mle_input.clone(),
                    0,
                    r_minus_x_power.clone()
                )
            }).collect_vec()
        );

        // --- Layer 2, part 2: (r - x)^2 ---
        // Note that we need to compute (r - x)^{2^0}, ..., (r - x)^{2^{3}}
        // We do this via repeated squaring of the previous power.
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
                    let r_minus_x_power = r_minus_x_power.clone();
                    SquaringBuilder::new(
                        r_minus_x_power
                    )
                }
            ).collect_vec()
        );

        let layer_3_builders = prev_prod_builders.concat(r_minus_x_square_builders);
        let (mut curr_prod_vec, mut r_minus_x_power_vec) = layers.add_gkr(layer_3_builders);

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

        let path_packing_builders_vec = self.permuted_input_data_mle_vec_tree.iter().map(
            |permuted_input_data_mle_vec| {
                let permuted_input_packing_builders = BatchedLayer::new(
                    permuted_input_data_mle_vec.iter().map(
                        |permuted_input_data_mle| {
                            let permuted_input_data_mle = permuted_input_data_mle.clone();
                            FSInputPackingBuilder::new(
                                permuted_input_data_mle,
                                self.r_mle.clone(),
                                self.r_packing_mle.clone()
                            )
                        }
                    ).collect_vec());
                permuted_input_packing_builders
                // layers.add_gkr(permuted_input_packing_builders)
            }
        ).collect_vec();

        // let permuted_input_data_mle_vec_tree_transposed = (0..self.permuted_input_data_mle_vec_tree[0].len()).map(
        //     |idx| {
        //         self.permuted_input_data_mle_vec_tree.iter().map(
        //             |mle_vec| mle_vec[idx].clone()
        //         ).collect_vec()
        //     }
        // ).collect_vec();

        // let path_packing_builders_vec = permuted_input_data_mle_vec_tree_transposed.iter().map(
        //     |permuted_input_data_mle_vec| {
        //         let permuted_input_packing_builders = BatchedLayer::new(
        //             permuted_input_data_mle_vec.iter().map(
        //                 |permuted_input_data_mle| {
        //                     let permuted_input_data_mle = permuted_input_data_mle.clone();
        //                     FSInputPackingBuilder::new(
        //                         permuted_input_data_mle,
        //                         self.r_mle.clone(),
        //                         self.r_packing_mle.clone()
        //                     )
        //                 }
        //             ).collect_vec());
        //         permuted_input_packing_builders
        //     }
        // ).collect_vec();


        
        let permuted_input_packed_vecs = layers.add_gkr(BatchedLayer::new(path_packing_builders_vec));

        let mut permuted_product_vecs = permuted_input_packed_vecs;

        // permuted_product_vecs.iter_mut().for_each(
        //     |permuted_product_vec| {
        //         permuted_product_vec.iter_mut().for_each(
        //             |mle| {
        //                 let prefix_bits = mle.prefix_bits.clone().unwrap();
        //                 let nonbatching_bits = prefix_bits[0..(prefix_bits.len() - (num_tree_bits + num_dataparallel_bits))].to_vec();
        //                 let dataparallel_bits = prefix_bits[(prefix_bits.len() - num_dataparallel_bits)..].to_vec();
        //                 let tree_bits = prefix_bits[(prefix_bits.len() - (num_tree_bits + num_dataparallel_bits)) .. (prefix_bits.len() - num_dataparallel_bits)].to_vec();

        //                 dbg!(&prefix_bits, &nonbatching_bits, &dataparallel_bits, &tree_bits);
        //                 mle.set_prefix_bits(Some(nonbatching_bits.into_iter().chain(dataparallel_bits.into_iter()).chain(tree_bits.into_iter()).collect_vec()));
        //             }
        //         )
        //     }
        // );


        // vec<vec

        // prefix bits = (tree_1, tree_2, data_1, )
        // prefix bits (tree_1)
        // (data_1)

        // let mut permuted_product_vec = permuted_product_vecs.into_iter().map(
        //     |mle_vec| {
        //         unbatch_mles(mle_vec)
        //     }
        // ).collect_vec();

        // let permuted_product_unbatched_by_tree_and_samples = unbatch_mles(permuted_product_unbatched_by_samples);

        // let mut permuted_packed_vec = unflatten_mle(permuted_product_unbatched_by_tree_and_samples, num_dataparallel_bits);



        // let mut permuted_product_vec = (0..permuted_product_vecs[0].len()).map(
        //     |idx| {
        //         unbatch_mles(permuted_product_vecs.iter().map(
        //             |vec| vec[idx].clone()
        //         ).collect_vec())
        //     }
        // ).collect_vec();

        // layer 15, 16, 17

        
            // for _ in 0..permuted_product_vec[0].num_iterated_vars() {
            //     let split_product_builders = BatchedLayer::new(
            //         permuted_product_vec.into_iter().map(
            //             |permuted_packed_mle| SplitProductBuilder::new(permuted_packed_mle)
            //         ).collect());
    
            //         permuted_product_vec = layers.add_gkr(split_product_builders);
            // }

            let permuted_input_len = 1 << (self.permuted_input_data_mle_vec_tree[0][0].num_iterated_vars() - 1);
            for _ in 0..log2(permuted_input_len) {
                let split_product_builders_vec = permuted_product_vecs.clone().into_iter().map(
                    |permuted_product_vec| {
                        BatchedLayer::new(
                            permuted_product_vec.into_iter().map(
                                |permuted_product| SplitProductBuilder::new(permuted_product)
                            ).collect())
                    }
                ).collect_vec();
                
                permuted_product_vecs = layers.add_gkr(BatchedLayer::new(split_product_builders_vec));
            }
        // dbg!(&permuted_product_vecs);

        let product_vec = (0..permuted_product_vecs[0].len()).map(
            |idx| {
                let products = DumbProduct::new(
                    permuted_product_vecs.iter().map(
                        |permuted_product_vec| {
                            permuted_product_vec[idx].clone()
                        }
                    ).collect_vec()
                );
                products
            }
        ).collect_vec();



        let permuted_product_vec_builder = BatchedLayer::new(vec![BatchedLayer::new(product_vec)]);

        let permuted_product_vec = layers.add_gkr(permuted_product_vec_builder);
        
        // dbg!(&permuted_product_vec);


        let difference_builder = EqualityCheck::new_batched(
                exponentiated_input_vec,
                permuted_product_vec[0].clone()
            );

        let circuit_output_vecs = layers.add_gkr(difference_builder);

        let circuit_output = combine_zero_mle_ref(circuit_output_vecs);

        println!("# layers -- input multiset: {:?}", layers.next_layer_id());

        Witness {
            layers,
            output_layers: vec![circuit_output.get_enum()],
            input_layers: vec![],
        }
    }
}