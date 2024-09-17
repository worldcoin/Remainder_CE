use ark_std::test_rng;

use itertools::Itertools;
use remainder::{
    layouter::{
        compiling::LayouterCircuit,
        component::{Component, ComponentSet},
        nodes::{
            circuit_inputs::{InputLayerData, InputLayerNode, InputLayerType},
            circuit_outputs::OutputNode,
            node_enum::NodeEnum,
            sector::Sector,
            CircuitNode, Context,
        },
    },
    mle::{dense::DenseMle, Mle},
    prover::helpers::test_circuit,
};
use remainder_shared_types::{Field, Fr};
use utils::{
    get_input_shred_and_data_from_vec, DifferenceBuilderComponent, ProductScaledBuilderComponent,
};

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

struct NonSelectorDataparallelComponent<F: Field> {
    first_layer_component: ProductScaledBuilderComponent<F>,
    output_component: DifferenceBuilderComponent<F>,
}

impl<F: Field> NonSelectorDataparallelComponent<F> {
    /// A simple wrapper around the [TripleNestedBuilderComponent] which
    /// additionally contains a [DifferenceBuilderComponent] for zero output
    pub fn new(
        ctx: &Context,
        mle_1_input: &dyn CircuitNode,
        mle_2_input: &dyn CircuitNode,
    ) -> Self {
        let first_layer_component =
            ProductScaledBuilderComponent::new(ctx, mle_1_input, mle_2_input);

        let output_component =
            DifferenceBuilderComponent::new(ctx, &first_layer_component.get_output_sector());

        Self {
            first_layer_component,
            output_component,
        }
    }
}

impl<F: Field, N> Component<N> for NonSelectorDataparallelComponent<F>
where
    N: CircuitNode + From<Sector<F>> + From<OutputNode>,
{
    fn yield_nodes(self) -> Vec<N> {
        self.first_layer_component
            .yield_nodes()
            .into_iter()
            .chain(self.output_component.yield_nodes())
            .collect_vec()
    }
}

#[test]
fn test_dataparallel_simple_newmainder() {
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

    // TODO(%): the batched mle should be able to demonstrate that there's NUM_DATA_PARALLEL_BITS of batch bits
    let dataparallel_mle_1 = DenseMle::batch_mles(mle_1_vec);
    let dataparallel_mle_2 = DenseMle::batch_mles(mle_2_vec);

    let circuit = LayouterCircuit::new(|ctx| {
        let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);
        let (dataparallel_input_mle_1, dataparallel_input_mle_1_data) =
            get_input_shred_and_data_from_vec(
                dataparallel_mle_1.bookkeeping_table().to_vec(),
                ctx,
                &input_layer,
            );
        let (dataparallel_input_mle_2, dataparallel_input_mle_2_data) =
            get_input_shred_and_data_from_vec(
                dataparallel_mle_2.bookkeeping_table().to_vec(),
                ctx,
                &input_layer,
            );
        let input_data = InputLayerData::new(
            input_layer.id(),
            vec![dataparallel_input_mle_1_data, dataparallel_input_mle_2_data],
            None,
        );

        let component_1 = NonSelectorDataparallelComponent::new(
            ctx,
            &dataparallel_input_mle_1,
            &dataparallel_input_mle_2,
        );

        let mut all_nodes: Vec<NodeEnum<Fr>> = vec![
            input_layer.into(),
            dataparallel_input_mle_1.into(),
            dataparallel_input_mle_2.into(),
        ];

        all_nodes.extend(component_1.yield_nodes());
        (
            ComponentSet::<NodeEnum<Fr>>::new_raw(all_nodes),
            vec![input_data],
        )
    });

    test_circuit(circuit, None)
}
