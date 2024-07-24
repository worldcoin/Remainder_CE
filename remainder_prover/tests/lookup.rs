
use remainder::{
    layouter::{
        compiling::LayouterCircuit,
        component::{Component, ComponentSet},
        nodes::{
            circuit_inputs::{InputLayerNode, InputLayerType}, lookup::{LookupNode, LookupShred}, node_enum::NodeEnum, CircuitNode
        },
    },
    mle::{dense::DenseMle, Mle},
    prover::helpers::test_circuit,

};
use remainder_shared_types::{FieldExt, Fr};

pub mod utils;
use utils::get_input_shred_from_vec;

/// Test the case where there is only one LookupShred for the LookupNode i.e. just one constrained MLE.
#[test]
pub fn single_shred_test() {
    let circuit = LayouterCircuit::new(|ctx| {
        let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);
        let table = get_input_shred_from_vec(vec![Fr::from(0u64), Fr::from(1u64)], ctx);
        let lookup_node = LookupNode::new(ctx, table.id());
        let constrained = get_input_shred_from_vec(
            vec![Fr::from(0u64), Fr::from(1u64), Fr::from(1u64), Fr::from(1u64)],
            ctx);
        let multiplicities = get_input_shred_from_vec(
            vec![Fr::from(1u64), Fr::from(3u64)],
            ctx);
        let lookup_shred = LookupShred::new(ctx, &lookup_node, &constrained, &multiplicities);

        let nodes: Vec<NodeEnum<Fr>> = vec![input_layer.into(), table.into(), lookup_node.into(), constrained.into(), multiplicities.into(), lookup_shred.into()];
        ComponentSet::<NodeEnum<Fr>>::new_raw(nodes)
    });
    test_circuit(circuit, None)
}

#[test]
pub fn multi_shred_test() {
    // Build the LHS of the equation (defined by the constrained values)
    // Input layer for the would-be random value r has layer id: Input(1)
    // Layer that calcs r - constrained has layer id: Layer(0)
    // Layer that sets the numerators to 1 has layer id: Layer(1)
    // Iteration 0 of build_fractional_sumcheck has layer id: Layer(2)
    // Iteration 1 of build_fractional_sumcheck has layer id: Layer(3)
    // Iteration 2 of build_fractional_sumcheck has layer id: Layer(4)
    // Build the RHS of the equation (defined by the table values and multiplicities)
    // Multiplicities location: CircuitLocation { layer_id: Input(0), prefix_bits: [true, false, true] }
    // Layer that aggs the multiplicities has layer id: Layer(5)
    // Layer that calculates r - table has layer id: Layer(6)
    // Iteration 0 of build_fractional_sumcheck has layer id: Layer(7)
    // Input layer that for LHS denom prod inverse has layer id: Input(2)
    // Input layer that for RHS denom prod inverse has layer id: Input(3)
    // Layer calcs product of (product of denoms) and their inverses has layer id: Layer(8)
    // Layer that checks that fractions are equal has layer id: Layer(9)
    let circuit = LayouterCircuit::new(|ctx| {
        let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);
        let table = get_input_shred_from_vec(vec![Fr::from(3u64), Fr::from(4u64)], ctx);
        let lookup_node = LookupNode::new(ctx, table.id());

        let constrained_0 = get_input_shred_from_vec(
            vec![Fr::from(3u64), Fr::from(3u64), Fr::from(3u64), Fr::from(4u64)],
            ctx);
        let multiplicities_0 = get_input_shred_from_vec(
            vec![Fr::from(3u64), Fr::from(1u64)],
            ctx);
        let lookup_shred_0 = LookupShred::new(ctx, &lookup_node, &constrained_0, &multiplicities_0);

        let constrained_1 = get_input_shred_from_vec(
            vec![Fr::from(4u64), Fr::from(4u64), Fr::from(4u64), Fr::from(4u64)],
            ctx);
        let multiplicities_1 = get_input_shred_from_vec(
            vec![Fr::from(0u64), Fr::from(4u64)],
            ctx);
        let lookup_shred_1 = LookupShred::new(ctx, &lookup_node, &constrained_1, &multiplicities_1);

        let nodes: Vec<NodeEnum<Fr>> = vec![
            input_layer.into(), table.into(), lookup_node.into(),
            constrained_0.into(), multiplicities_0.into(), lookup_shred_0.into(),
            constrained_1.into(), multiplicities_1.into(), lookup_shred_1.into()
        ];
        ComponentSet::<NodeEnum<Fr>>::new_raw(nodes)
    });
    test_circuit(circuit, None)
}

// TODO test failure