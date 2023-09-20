use ark_std::log2;
use itertools::{repeat_n, Itertools};
use remainder_shared_types::{FieldExt, transcript::poseidon_transcript::PoseidonTranscript};

use crate::{prover::{Witness, Layers}, layer::{batched::{BatchedLayer, combine_zero_mle_ref}, LayerBuilder}, mle::{Mle, MleIndex, dense::DenseMle, MleRef}};

use super::{builders::{FSInputPackingBuilder, SplitProductBuilder, EqualityCheck}, structs::InputAttribute};

/// Deprecated! No need for this particular circuit anymore.
pub struct PermutationSubCircuit<F: FieldExt> {
    pub input_data_mle_vec: Vec<DenseMle<F, InputAttribute<F>>>,               // batched
    pub permuted_input_data_mle_vec: Vec<DenseMle<F, InputAttribute<F>>>,      // batched
    pub r_mle: DenseMle<F, F>,
    pub r_packing_mle: DenseMle<F, F>,
}

impl<F: FieldExt> PermutationSubCircuit<F> {
    pub fn yield_sub_circuit(&mut self) -> Witness<F, PoseidonTranscript<F>> {
        let mut layers: Layers<_, PoseidonTranscript<F>> = Layers::new();

        let batch_bits = log2(self.input_data_mle_vec.len()) as usize;


        let input_packing_builder = BatchedLayer::new(
            self.input_data_mle_vec.iter().map(
                |input_data_mle| {
                    let mut input_data_mle = input_data_mle.clone();
                    // TODO!(ende) fix this atrocious fixed(false)
                    input_data_mle.add_prefix_bits(Some(input_data_mle.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, batch_bits)).collect_vec()));
                    FSInputPackingBuilder::new(
                        input_data_mle,
                        self.r_mle.clone(),
                        self.r_packing_mle.clone()
                    )
                }).collect_vec());

        let input_permuted_packing_builder = BatchedLayer::new(
            self.permuted_input_data_mle_vec.iter().map(
                |input_data_mle| {
                    let mut input_data_mle = input_data_mle.clone();
                    // TODO!(ende) fix this atrocious fixed(true)
                    input_data_mle.add_prefix_bits(Some(input_data_mle.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, batch_bits)).collect_vec()));
                    FSInputPackingBuilder::new(
                        input_data_mle,
                        self.r_mle.clone(),
                        self.r_packing_mle.clone()
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
            input_layers: vec![],
        }
    }
}