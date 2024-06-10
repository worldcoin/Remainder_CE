use ark_std::test_rng;

use remainder::{
    builders::{
        combine_input_layers::InputLayerBuilder, layer_builder::simple_builders::ZeroBuilder,
    },
    input_layer::public_input_layer::PublicInputLayer,
    layer::LayerId,
    mle::{dense::DenseMle, Mle},
    prover::{
        helpers::test_circuit, layers::Layers, proof_system::DefaultProofSystem, GKRCircuit,
        Witness,
    },
};
use remainder_shared_types::FieldExt;
use utils::{ProductScaledBuilder, TripleNestedSelectorBuilder};

use crate::utils::get_dummy_random_mle_vec;

pub mod utils;

/// A circuit which does the following:
/// * Layer 0: [ProductScaledBuilder] with `mle_1_vec`, `mle_2_vec`
/// * Layer 1: [TripleNestedSelectorBuilder] with output of Layer 0, `mle_3_vec`, `mle_4_vec`
/// * Layer 2: [ZeroBuilder] with the output of Layer 1 and itself.
///
/// The expected output of this circuit is the zero MLE.
///
/// ## Arguments
/// * `mle_1_vec`, `mle_2_vec` - inputs to [ProductScaledBuilder] both arbitrary bookkeeping
/// table values, same size.
/// * `mle_3_vec`, `mle_4_vec` - inputs to [TripleNestedSelectorBuilder], both arbitrary bookkeeping table values,
/// `mle_3_vec` mles have one more variable than in `mle_1_vec`, `mle_2_vec`, and `mle_4_vec` mles
/// have one more variable than in `mle_3_vec`.
///
/// * `num_dataparallel_bits` - The number of bits that represent which copy index the circuit is.
struct DataParallelCircuitAlt<F: FieldExt> {
    mle_1: DenseMle<F>,
    mle_2: DenseMle<F>,
    mle_3: DenseMle<F>,
    mle_4: DenseMle<F>,
}

impl<F: FieldExt> GKRCircuit<F> for DataParallelCircuitAlt<F> {
    type ProofSystem = DefaultProofSystem;

    fn synthesize(&mut self) -> Witness<F, Self::ProofSystem> {
        let mut layers = Layers::new();

        self.mle_1.layer_id = LayerId::Input(0);
        self.mle_2.layer_id = LayerId::Input(0);
        self.mle_3.layer_id = LayerId::Input(0);
        self.mle_4.layer_id = LayerId::Input(0);

        let input_commit: Vec<&mut dyn Mle<F>> = vec![
            &mut self.mle_1,
            &mut self.mle_2,
            &mut self.mle_3,
            &mut self.mle_4,
        ];

        let input_commit_builder =
            InputLayerBuilder::<F>::new(input_commit, None, LayerId::Input(0));

        let input_layer: PublicInputLayer<F> =
            input_commit_builder.to_input_layer::<PublicInputLayer<F>>();

        let input_layer_enum = input_layer.into();

        let first_layer_builder = ProductScaledBuilder::new(self.mle_1.clone(), self.mle_2.clone());
        let first_layer_output = layers.add_gkr(first_layer_builder);

        let second_layer_builder = TripleNestedSelectorBuilder::new(
            first_layer_output.clone(),
            self.mle_3.clone(),
            self.mle_4.clone(),
        );

        let second_layer_output = layers.add_gkr(second_layer_builder);

        let zero_builder = ZeroBuilder::new(second_layer_output.clone());
        let output = layers.add_gkr(zero_builder);

        Witness {
            layers,
            output_layers: vec![output.get_enum()],
            input_layers: vec![input_layer_enum],
        }
    }
}

impl<F: FieldExt> DataParallelCircuitAlt<F> {
    fn new(mle_1: DenseMle<F>, mle_2: DenseMle<F>, mle_3: DenseMle<F>, mle_4: DenseMle<F>) -> Self {
        Self {
            mle_1,
            mle_2,
            mle_3,
            mle_4,
        }
    }
}

// TODO(vishady): this test fails based off of our current implementation of remainder!! The current problem is the way
// selector bits are treated when combining expressions.

#[test]
fn test_dataparallel_selector_alt() {
    const NUM_DATA_PARALLEL_BITS: usize = 3;
    const NUM_VARS_MLE_1_2: usize = 2;
    const NUM_VARS_MLE_3: usize = NUM_VARS_MLE_1_2 + 1;
    const NUM_VARS_MLE_4: usize = NUM_VARS_MLE_3 + 1;
    let mut rng = test_rng();

    let mle_1_vec = get_dummy_random_mle_vec(NUM_VARS_MLE_1_2, NUM_DATA_PARALLEL_BITS, &mut rng);
    let mle_2_vec = get_dummy_random_mle_vec(NUM_VARS_MLE_1_2, NUM_DATA_PARALLEL_BITS, &mut rng);
    let mle_3_vec = get_dummy_random_mle_vec(NUM_VARS_MLE_3, NUM_DATA_PARALLEL_BITS, &mut rng);
    let mle_4_vec = get_dummy_random_mle_vec(NUM_VARS_MLE_4, NUM_DATA_PARALLEL_BITS, &mut rng);

    // These checks can possibly be done with the newly designed batching bits/system
    assert_eq!(mle_1_vec.len(), mle_2_vec.len());
    assert_eq!(mle_3_vec.len(), mle_2_vec.len());
    assert_eq!(mle_3_vec.len(), mle_4_vec.len());
    assert_eq!(mle_1_vec.len(), 1 << NUM_DATA_PARALLEL_BITS);
    let all_num_vars_1_2: Vec<usize> = mle_1_vec
        .iter()
        .chain(mle_2_vec.iter())
        .map(|mle| mle.num_iterated_vars())
        .collect();
    let all_vars_same_1_2 = all_num_vars_1_2.iter().fold(true, |acc, elem| {
        (*elem == mle_3_vec[0].num_iterated_vars() - 1) & acc
    });
    assert!(all_vars_same_1_2);
    let all_num_vars_3: Vec<usize> = mle_3_vec
        .iter()
        .map(|mle| mle.num_iterated_vars())
        .collect();
    let all_vars_same_3 = all_num_vars_3.iter().fold(true, |acc, elem| {
        (*elem == mle_4_vec[0].num_iterated_vars() - 1) & acc
    });
    assert!(all_vars_same_3);
    let all_num_vars_4: Vec<usize> = mle_4_vec
        .iter()
        .map(|mle| mle.num_iterated_vars())
        .collect();
    let all_vars_same_4 = all_num_vars_4.iter().fold(true, |acc, elem| {
        (*elem == mle_4_vec[0].num_iterated_vars()) & acc
    });
    assert!(all_vars_same_4);
    // These checks can possibly be done with the newly designed batching bits/system

    // the batched mle should be able to demonstrate that there's NUM_DATA_PARALLEL_BITS of batch bits
    let mle_1_vec_batched = DenseMle::batch_mles(mle_1_vec);
    let mle_2_vec_batched = DenseMle::batch_mles(mle_2_vec);
    let mle_3_vec_batched = DenseMle::batch_mles(mle_3_vec);
    let mle_4_vec_batched = DenseMle::batch_mles(mle_4_vec);
    // the batched mle should be able to demonstrate that there's NUM_DATA_PARALLEL_BITS of batch bits

    let circuit = DataParallelCircuitAlt::new(
        mle_1_vec_batched,
        mle_2_vec_batched,
        mle_3_vec_batched,
        mle_4_vec_batched,
    );
    test_circuit(circuit, None);
}
