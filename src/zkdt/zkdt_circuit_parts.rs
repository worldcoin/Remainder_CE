use ark_std::log2;
use itertools::Itertools;

use crate::{prover::{GKRCircuit, Layers}, FieldExt, mle::{dense::DenseMle, MleRef}, transcript::poseidon_transcript::PoseidonTranscript, layer::{LayerBuilder, empty_layer::EmptyLayer, batched::{BatchedLayer, combine_zero_mle_ref}}};

use super::{zkdt_layer::{InputPackingBuilder, SplitProductBuilder, EqualityCheck, AttributeConsistencyBuilder, DecisionPackingBuilder, LeafPackingBuilder, ConcatBuilder, RMinusXBuilder, BitExponentiationBuilder, SquaringBuilder, ProductBuilder}, structs::{InputAttribute, DecisionNode, LeafNode, BinDecomp16Bit}};

struct PermutationCircuit<F: FieldExt> {
    dummy_input_data_mle_vec: Vec<DenseMle<F, InputAttribute<F>>>,               // batched
    dummy_permuted_input_data_mle_vec: Vec<DenseMle<F, InputAttribute<F>>>,      // batched
    r: F,
    r_packing: F,
    input_len: usize,
    num_inputs: usize
}

impl<F: FieldExt> GKRCircuit<F> for PermutationCircuit<F> {
    type Transcript = PoseidonTranscript<F>;
    fn synthesize(&mut self) -> (Layers<F, Self::Transcript>, Vec<Box<dyn MleRef<F = F>>>) {
        let mut layers = Layers::new();

        // layer 0: packing
        // let input_packing_builder: InputPackingBuilder<F> = InputPackingBuilder::new(
        //     self.dummy_input_data_mle_vec.clone(),
        //     self.r,
        //     self.r_packing);

        let input_packing_builder = BatchedLayer::new(self.dummy_input_data_mle_vec.iter().map(|input_data_mle| InputPackingBuilder::new(
            input_data_mle.clone(),
            self.r,
            self.r_packing)).collect_vec());

        let input_permuted_packing_builder = BatchedLayer::new(self.dummy_permuted_input_data_mle_vec.iter().map(|input_data_mle| InputPackingBuilder::new(
            input_data_mle.clone(),
            self.r,
            self.r_packing)).collect_vec());

        let packing_builders = input_packing_builder.concat(input_permuted_packing_builder);
        let (mut input_packed, mut input_permuted_packed) = layers.add_gkr(packing_builders);

        for _ in 0..log2(self.input_len) {
            let prod_builder = BatchedLayer::new(input_packed.into_iter().map(|input_packed| SplitProductBuilder::new(
                input_packed
            )).collect());
            let prod_permuted_builder = BatchedLayer::new(input_permuted_packed.into_iter().map(|input_permuted_packed| SplitProductBuilder::new(
                input_permuted_packed
            )).collect());
            let split_product_builders = prod_builder.concat(prod_permuted_builder);
            (input_packed, input_permuted_packed) = layers.add_gkr(split_product_builders);
        }

        let difference_builder = EqualityCheck::new_batched(
            input_packed,
            input_permuted_packed,
        );

        let difference_mle = layers.add_gkr(difference_builder);

        let difference_mle = combine_zero_mle_ref(difference_mle);

        (layers, vec![Box::new(difference_mle)])
    }
}

struct AttributeConsistencyCircuit<F: FieldExt> {
    dummy_permuted_input_data_mle_vec: DenseMle<F, InputAttribute<F>>, // batched
    dummy_decision_node_paths_mle_vec: DenseMle<F, DecisionNode<F>>,     // batched
    tree_height: usize,
}

impl<F: FieldExt> GKRCircuit<F> for AttributeConsistencyCircuit<F> {
    type Transcript = PoseidonTranscript<F>;
    fn synthesize(&mut self) -> (Layers<F, Self::Transcript>, Vec<Box<dyn MleRef<F = F>>>) {
        let mut layers = Layers::new();

        let attribute_consistency_builder = AttributeConsistencyBuilder::new(
            self.dummy_permuted_input_data_mle_vec.clone(),
            self.dummy_decision_node_paths_mle_vec.clone(),
            self.tree_height
        );

        let difference_mle = layers.add_gkr(attribute_consistency_builder);

        (layers, vec![Box::new(difference_mle.mle_ref())])
    }
}

struct MultiSetCircuit<F: FieldExt> {
    dummy_decision_nodes_mle: DenseMle<F, DecisionNode<F>>,
    dummy_leaf_nodes_mle: DenseMle<F, LeafNode<F>>,
    dummy_multiplicities_bin_decomp_mle: DenseMle<F, BinDecomp16Bit<F>>,
    dummy_decision_node_paths_mle_vec: DenseMle<F, DecisionNode<F>>, // batched
    dummy_leaf_node_paths_mle_vec: DenseMle<F, LeafNode<F>>,         // batched
    r: F,
    r_packings: (F, F),
    tree_height: usize,
    num_inputs: usize,
}

impl<F: FieldExt> GKRCircuit<F> for MultiSetCircuit<F> {
    type Transcript = PoseidonTranscript<F>;
    
    fn synthesize(&mut self) -> (Layers<F, Self::Transcript>, Vec<Box<dyn MleRef<F = F>>>) {
        let mut layers = Layers::new();

        // layer 0
        let decision_packing_builder = DecisionPackingBuilder::new(
            self.dummy_decision_nodes_mle.clone(), self.r, self.r_packings);

        let leaf_packing_builder = LeafPackingBuilder::new(
            self.dummy_leaf_nodes_mle.clone(), self.r, self.r_packings.0
        );

        let packing_builders = decision_packing_builder.concat(leaf_packing_builder);
        let (decision_packed, leaf_packed) = layers.add_gkr(packing_builders);

        // layer 1
        let decision_leaf_concat_builder = ConcatBuilder::new(
            decision_packed, leaf_packed
        );
        let x_packed = layers.add_gkr(decision_leaf_concat_builder);

        // layer 2
        let r_minus_x_builder =  RMinusXBuilder::new(
            x_packed, self.r
        );
        let mut r_minus_x = layers.add_gkr(r_minus_x_builder);

        // layer 3
        let prev_prod_builder = BitExponentiationBuilder::new(
            self.dummy_multiplicities_bin_decomp_mle.clone(),
            0,
            r_minus_x.clone()
        );
        let mut prev_prod = layers.add_gkr(prev_prod_builder);

        for i in 1..16 {

            // layer 3, or i + 2
            let r_minus_x_square_builder = SquaringBuilder::new(
                r_minus_x
            );
            let r_minus_x_square = layers.add_gkr(r_minus_x_square_builder);

            // layer 4, or i + 3
            let curr_prod_builder = BitExponentiationBuilder::new(
                self.dummy_multiplicities_bin_decomp_mle.clone(),
                i,
                r_minus_x_square.clone()
            );
            let curr_prod = layers.add_gkr(curr_prod_builder);

            // layer 5, or i + 4
            let prod_builder = ProductBuilder::new(
                curr_prod,
                prev_prod
            );
            prev_prod = layers.add_gkr(prod_builder);

            r_minus_x = r_minus_x_square;

        }

        let mut exponentiated_nodes = prev_prod;

        for _ in 0..self.tree_height {

            // layer 20, or i+20
            let prod_builder = SplitProductBuilder::new(
                exponentiated_nodes
            );
            exponentiated_nodes = layers.add_gkr(prod_builder);
        }

        // **** above is nodes exponentiated ****
        // **** below is all decision nodes on the path multiplied ****


        // layer 0: packing
        let decision_path_packing_builder = DecisionPackingBuilder::new(
            self.dummy_decision_node_paths_mle_vec.clone(),
            self.r,
            self.r_packings
        );

        let leaf_path_packing_builder = LeafPackingBuilder::new(
            self.dummy_leaf_node_paths_mle_vec.clone(),
            self.r,
            self.r_packings.0
        );

        let path_packing_builders = decision_path_packing_builder.concat(leaf_path_packing_builder);
        let (decision_path_packed, leaf_path_packed) = layers.add_gkr(path_packing_builders);

        // layer 1: concat
        let path_decision_leaf_concat_builder = ConcatBuilder::new(
            decision_path_packed, leaf_path_packed
        );
        let x_path_packed = layers.add_gkr(path_decision_leaf_concat_builder);

        // layer 2: r - x
        let r_minus_x_path_builder =  RMinusXBuilder::new(
            x_path_packed, self.r
        );
        let r_minus_x_path = layers.add_gkr(r_minus_x_path_builder);

        let mut path_product = r_minus_x_path;

        // layer remaining: product it together
        for _ in 0..log2(self.tree_height * self.num_inputs) {
            let prod_builder = SplitProductBuilder::new(
                path_product
            );
            path_product = layers.add_gkr(prod_builder);
        }

        let difference_builder = EqualityCheck::new(
            exponentiated_nodes,
            path_product
        );

        let difference = layers.add_gkr(difference_builder);

        (layers, vec![Box::new(difference)])
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use ark_std::{test_rng, UniformRand};

    use crate::{zkdt::zkdt_circuit::{DummyMles, generate_dummy_mles, NUM_DUMMY_INPUTS, DUMMY_INPUT_LEN, generate_dummy_mles_batch, BatchedDummyMles}, prover::GKRCircuit, transcript::{poseidon_transcript::PoseidonTranscript, Transcript}};

    use super::PermutationCircuit;


    #[test]
    fn test_permutation_circuit() {
        let mut rng = test_rng();

        let BatchedDummyMles {
            dummy_input_data_mle,
            dummy_permuted_input_data_mle, ..
        } = generate_dummy_mles_batch();

        let mut circuit = PermutationCircuit {
            dummy_input_data_mle_vec: dummy_input_data_mle,
            dummy_permuted_input_data_mle_vec: dummy_permuted_input_data_mle,
            r: Fr::rand(&mut rng),
            r_packing: Fr::rand(&mut rng),
            input_len: DUMMY_INPUT_LEN,
            num_inputs: 1,
        };

        let mut transcript = PoseidonTranscript::new("Permutation Circuit Prover Transcript");

        let proof = circuit.prove(&mut transcript);

        match proof {
            Ok(proof) => {
                let mut transcript = PoseidonTranscript::new("Permutation Circuit Verifier Transcript");
                let result = circuit.verify(&mut transcript, proof);
                if let Err(err) = result {
                    println!("{}", err);
                    panic!();
                }
            },
            Err(err) => {
                println!("{}", err);
                panic!();
            }
        }
    }
}