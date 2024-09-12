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
use remainder_shared_types::{FieldExt, Fr};
use utils::DifferenceBuilderComponent;
use utils::{ProductScaledBuilderComponent, TripleNestedBuilderComponent};

use crate::utils::{get_dummy_random_mle_vec, get_input_shred_and_data_from_vec};

pub mod utils;

struct DataparallelTripleNestedSelectorComponent<F: FieldExt> {
    first_layer_component: TripleNestedBuilderComponent<F>,
    output_component: DifferenceBuilderComponent<F>,
}

impl<F: FieldExt> DataparallelTripleNestedSelectorComponent<F> {
    /// A simple wrapper around the [TripleNestedBuilderComponent] which
    /// additionally contains a [DifferenceBuilderComponent] for zero output
    pub fn new(
        ctx: &Context,
        mle_1_input: &dyn CircuitNode,
        mle_2_input: &dyn CircuitNode,
        mle_3_input: &dyn CircuitNode,
    ) -> Self {
        let first_layer_component =
            TripleNestedBuilderComponent::new(ctx, mle_1_input, mle_2_input, mle_3_input);

        let output_component =
            DifferenceBuilderComponent::new(ctx, &first_layer_component.get_output_sector());

        Self {
            first_layer_component,
            output_component,
        }
    }
}

impl<F: FieldExt, N> Component<N> for DataparallelTripleNestedSelectorComponent<F>
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
#[test]
fn test_dataparallel_selector_alt_newmainder() {
    const NUM_DATAPARALLEL_BITS: usize = 3;
    const NUM_VARS_MLE_1_2: usize = 2;
    const NUM_VARS_MLE_3: usize = NUM_VARS_MLE_1_2 + 1;
    const NUM_VARS_MLE_4: usize = NUM_VARS_MLE_3 + 1;
    let mut rng = test_rng();

    // This is not strictly necessary; the setup of `DenseMle` -->
    // `batch_mles()` --> `bookkeeping_table` is just to emulate what
    // batching *would* look like
    let mle_1_vec = get_dummy_random_mle_vec(NUM_VARS_MLE_1_2, NUM_DATAPARALLEL_BITS, &mut rng);
    let mle_2_vec = get_dummy_random_mle_vec(NUM_VARS_MLE_1_2, NUM_DATAPARALLEL_BITS, &mut rng);
    let mle_3_vec = get_dummy_random_mle_vec(NUM_VARS_MLE_3, NUM_DATAPARALLEL_BITS, &mut rng);
    let mle_4_vec = get_dummy_random_mle_vec(NUM_VARS_MLE_4, NUM_DATAPARALLEL_BITS, &mut rng);

    // These checks can possibly be done with the newly designed batching bits/system
    assert_eq!(mle_1_vec.len(), mle_2_vec.len());
    assert_eq!(mle_3_vec.len(), mle_2_vec.len());
    assert_eq!(mle_3_vec.len(), mle_4_vec.len());
    assert_eq!(mle_1_vec.len(), 1 << NUM_DATAPARALLEL_BITS);
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

    // TODO(%): the batched mle should be able to demonstrate that there's NUM_DATA_PARALLEL_BITS of batch bits
    let dataparallel_mle_1 = DenseMle::batch_mles(mle_1_vec);
    let dataparallel_mle_2 = DenseMle::batch_mles(mle_2_vec);
    let dataparallel_mle_3 = DenseMle::batch_mles(mle_3_vec);
    let dataparallel_mle_4 = DenseMle::batch_mles(mle_4_vec);

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
        let (dataparallel_input_mle_3, dataparallel_input_mle_3_data) =
            get_input_shred_and_data_from_vec(
                dataparallel_mle_3.bookkeeping_table().to_vec(),
                ctx,
                &input_layer,
            );
        let (dataparallel_input_mle_4, dataparallel_input_mle_4_data) =
            get_input_shred_and_data_from_vec(
                dataparallel_mle_4.bookkeeping_table().to_vec(),
                ctx,
                &input_layer,
            );
        let input_data = InputLayerData::new(
            input_layer.id(),
            vec![
                dataparallel_input_mle_1_data,
                dataparallel_input_mle_2_data,
                dataparallel_input_mle_3_data,
                dataparallel_input_mle_4_data,
            ],
            None,
        );

        let component_1 = ProductScaledBuilderComponent::new(
            ctx,
            &dataparallel_input_mle_1,
            &dataparallel_input_mle_2,
        );
        let component_2 = DataparallelTripleNestedSelectorComponent::new(
            ctx,
            &component_1.get_output_sector(),
            &dataparallel_input_mle_3,
            &dataparallel_input_mle_4,
        );

        let mut all_nodes: Vec<NodeEnum<Fr>> = vec![
            input_layer.into(),
            dataparallel_input_mle_1.into(),
            dataparallel_input_mle_2.into(),
            dataparallel_input_mle_3.into(),
            dataparallel_input_mle_4.into(),
        ];

        all_nodes.extend(component_1.yield_nodes());
        all_nodes.extend(component_2.yield_nodes());
        (
            ComponentSet::<NodeEnum<Fr>>::new_raw(all_nodes),
            vec![input_data],
        )
    });

    test_circuit(circuit, None)
}
