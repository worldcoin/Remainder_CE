
use remainder::{
    layouter::{
        compiling::LayouterCircuit,
        component::{Component, ComponentSet},
        nodes::{
            // circuit_inputs::{InputLayerNode, InputLayerType},
            // circuit_outputs::OutputNode,
            node_enum::NodeEnum,
            CircuitNode, //ClaimableNode, Context,
            lookup::{LookupNode, LookupShred},
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
        let table = get_input_shred_from_vec(vec![Fr::from(0u64), Fr::from(1u64)], ctx);
        let mut lookup_node = LookupNode::new(ctx, table.id());
        let constrained = get_input_shred_from_vec(
            vec![Fr::from(0u64), Fr::from(1u64), Fr::from(1u64), Fr::from(1u64)],
            ctx);
        let multiplicities = get_input_shred_from_vec(
            vec![Fr::from(1u64), Fr::from(3u64)],
            ctx);
        let lookup_shred = LookupShred::new(ctx, &lookup_node, &constrained, &multiplicities);
        // FIXME this should go in the layouter
        lookup_node.add_shred(lookup_shred);

        let nodes: Vec<NodeEnum<Fr>> = vec![constrained.into(), multiplicities.into()];//FIXME, lookup_node];
        ComponentSet::<NodeEnum<Fr>>::new_raw(nodes)
    });
    test_circuit(circuit, None)
}