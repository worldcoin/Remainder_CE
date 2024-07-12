use ark_std::test_rng;

use remainder::{
    builders::{
        combine_input_layers::InputLayerBuilder, layer_builder::simple_builders::ZeroBuilder,
    },
    input_layer::public_input_layer::PublicInputLayer,
    mle::{dense::DenseMle, Mle},
    prover::{
        helpers::test_circuit, layers::Layers, proof_system::DefaultProofSystem, GKRCircuit,
        Witness,
    },
};
use remainder_shared_types::{layer::LayerId, FieldExt};
use utils::ProductScaledBuilder;

use crate::utils::get_dummy_random_mle_vec;
pub mod utils;

/// A circuit which does the following:
/// * Layer 0: [ProductScaledBuilder] with `mle_1_vec`, `mle_2_vec`
/// * Layer 1: [ZeroBuilder] with the output of Layer 0 and itself.
///
/// The expected output of this circuit is the zero MLE.
///
/// ## Arguments
/// * `mle_1_vec`, `mle_2_vec` - inputs to [ProductScaledBuilder] both arbitrary bookkeeping
/// table values, same size.
///
/// * `num_dataparallel_bits` - The number of bits that represent which copy index the circuit is.
struct NonSelectorDataparallelCircuitAlt<F: FieldExt> {
    mle_1_vec: DenseMle<F>,
    mle_2_vec: DenseMle<F>,
}

impl<F: FieldExt> GKRCircuit<F> for NonSelectorDataparallelCircuitAlt<F> {
    type ProofSystem = DefaultProofSystem;

    fn synthesize(&mut self) -> Witness<F, Self::ProofSystem> {
        let mut layers = Layers::new();

        self.mle_1_vec.layer_id = LayerId::Input(0);
        self.mle_2_vec.layer_id = LayerId::Input(0);

        let input_commit: Vec<&mut dyn Mle<F>> = vec![&mut self.mle_1_vec, &mut self.mle_2_vec];

        let input_commit_builder =
            InputLayerBuilder::<F>::new(input_commit, None, LayerId::Input(0));

        let input_layer: PublicInputLayer<F> =
            input_commit_builder.to_input_layer::<PublicInputLayer<F>>();

        let input_layer_enum = input_layer.into();

        let first_layer_builder =
            ProductScaledBuilder::new(self.mle_1_vec.clone(), self.mle_2_vec.clone());

        let first_layer_output = layers.add_gkr(first_layer_builder);

        let zero_builder = ZeroBuilder::new(first_layer_output);
        let output = layers.add_gkr(zero_builder);

        Witness {
            layers,
            output_layers: vec![output.get_enum()],
            input_layers: vec![input_layer_enum],
        }
    }
}

impl<F: FieldExt> NonSelectorDataparallelCircuitAlt<F> {
    fn new(mle_1_vec: DenseMle<F>, mle_2_vec: DenseMle<F>) -> Self {
        Self {
            mle_1_vec,
            mle_2_vec,
        }
    }
}

#[test]
fn test_simple_dataparallel_circuit_alt() {
    const NUM_DATA_PARALLEL_BITS: usize = 3;
    const NUM_VARS_MLE_1_2: usize = 2;
    let mut rng = test_rng();

    let mle_1_vec = get_dummy_random_mle_vec(NUM_VARS_MLE_1_2, NUM_DATA_PARALLEL_BITS, &mut rng);
    let mle_2_vec = get_dummy_random_mle_vec(NUM_VARS_MLE_1_2, NUM_DATA_PARALLEL_BITS, &mut rng);

    // These checks can possibly be done with the newly designed batching bits/system
    let all_num_vars: Vec<usize> = mle_1_vec
        .iter()
        .chain(mle_2_vec.iter())
        .map(|mle| mle.num_iterated_vars())
        .collect();
    let all_vars_same = all_num_vars.iter().fold(true, |acc, elem| {
        (*elem == mle_1_vec[0].num_iterated_vars()) & acc
    });
    assert!(all_vars_same);
    assert_eq!(mle_1_vec.len(), mle_2_vec.len());
    assert_eq!(mle_1_vec.len(), 1 << NUM_DATA_PARALLEL_BITS);
    // These checks can possibly be done with the newly designed batching bits/system

    // the batched mle should be able to demonstrate that there's NUM_DATA_PARALLEL_BITS of batch bits
    let mle_1_vec_batched = DenseMle::batch_mles(mle_1_vec);
    let mle_2_vec_batched = DenseMle::batch_mles(mle_2_vec);
    // the batched mle should be able to demonstrate that there's NUM_DATA_PARALLEL_BITS of batch bits

    let circuit = NonSelectorDataparallelCircuitAlt::new(mle_1_vec_batched, mle_2_vec_batched);
    test_circuit(circuit, None);
}
