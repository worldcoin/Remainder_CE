
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

#[test]
pub fn test() {
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