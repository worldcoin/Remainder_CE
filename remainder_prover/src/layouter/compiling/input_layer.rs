use remainder_shared_types::FieldExt;

use crate::{
    input_layer::{
        ligero_input_layer::LigeroInputLayer, public_input_layer::PublicInputLayer, InputLayer,
        MleInputLayer,
    },
    layer::LayerId,
    layouter::nodes::{
        circuit_inputs::{InputLayerNode, InputLayerType},
        ClaimableNode,
    },
    prover::{proof_system::ProofSystem, Witness},
};

impl<F: FieldExt> InputLayerNode<F> {
    fn compile<Pf: ProofSystem<F, InputLayer = IL>, IL>(self, witness: &mut Witness<F, Pf>)
    where
        IL: InputLayer<F> + From<PublicInputLayer<F>> + From<LigeroInputLayer<F>>,
    {
        let input_layers = &mut witness.input_layers;
        let layer_id = LayerId::Input(input_layers.len());
        let mle = self.get_data();
        let out: IL = match self.input_layer_type {
            InputLayerType::LigeroInputLayer => LigeroInputLayer::new(mle.clone(), layer_id).into(),
            InputLayerType::PublicInputLayer => PublicInputLayer::new(mle.clone(), layer_id).into(),
            InputLayerType::Default => LigeroInputLayer::new(mle.clone(), layer_id).into(),
        };
        input_layers.push(out);
    }
}
