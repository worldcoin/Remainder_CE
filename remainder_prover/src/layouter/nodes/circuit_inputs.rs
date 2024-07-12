//! Nodes that represent inputs to a circuit in the circuit DAG\

mod compile_inputs;

use remainder_shared_types::{input_layer::InputLayer, FieldExt};

use crate::{
    expression::{abstract_expr::AbstractExpr, generic_expr::Expression},
    input_layer::{ligero_input_layer::LigeroInputLayer, public_input_layer::PublicInputLayer},
    layouter::{compiling::WitnessBuilder, layouting::CircuitMap},
    mle::evals::MultilinearExtension,
    prover::proof_system::ProofSystem,
};

use super::{CircuitNode, ClaimableNode, CompilableNode, Context, NodeId};

/// A node that represents some Data that will eventually be added to an InputLayer
#[derive(Debug, Clone)]
pub struct InputShred<F> {
    id: NodeId,
    pub(crate) parent: Option<NodeId>,
    data: MultilinearExtension<F>,
}

impl<F: FieldExt> CircuitNode for InputShred<F> {
    fn id(&self) -> NodeId {
        self.id
    }

    fn sources(&self) -> Vec<NodeId> {
        vec![]
    }
}

impl<F: FieldExt> ClaimableNode for InputShred<F> {
    type F = F;

    fn get_data(&self) -> &MultilinearExtension<Self::F> {
        &self.data
    }

    fn get_expr(&self) -> Expression<Self::F, AbstractExpr> {
        Expression::<F, AbstractExpr>::mle(self.id)
    }
}

impl<F: FieldExt> InputShred<F> {
    /// Creates a new InputShred from data
    ///
    /// Specifying a source indicates to the layouter that this
    /// InputShred should be appended to the source when laying out
    pub fn new(
        ctx: &Context,
        data: MultilinearExtension<F>,
        source: Option<&InputLayerNode<F>>,
    ) -> Self {
        let id = ctx.get_new_id();
        let parent = source.map(CircuitNode::id);

        InputShred { id, parent, data }
    }
}

/// An enum representing the different types
/// of InputLayer an InputLayerNode can be compiled into
#[derive(Debug, Clone)]
pub enum InputLayerType {
    ///An InputLayer that will be compiled into a `LigeroInputLayer`
    LigeroInputLayer,
    /// A PublicInputLayer
    PublicInputLayer,
    /// A default InputLayerType
    Default,
}

#[derive(Debug, Clone)]
/// A node that represents an InputLayer
///
/// TODO! probably split this up into more node types
/// that indicate different things to the layouter
pub struct InputLayerNode<F> {
    id: NodeId,
    children: Vec<InputShred<F>>,
    pub(in crate::layouter) input_layer_type: InputLayerType,
}

impl<F: FieldExt> CircuitNode for InputLayerNode<F> {
    fn id(&self) -> NodeId {
        self.id
    }

    fn children(&self) -> Option<Vec<NodeId>> {
        Some(self.children.iter().map(CircuitNode::id).collect())
    }

    fn sources(&self) -> Vec<NodeId> {
        vec![]
    }
}

impl<F: FieldExt> InputLayerNode<F> {
    /// A constructor for an InputLayerNode. Can either be initialized empty
    /// or with some children.
    pub fn new(
        ctx: &Context,
        children: Option<Vec<InputShred<F>>>,
        input_layer_type: InputLayerType,
    ) -> Self {
        InputLayerNode {
            id: ctx.get_new_id(),
            children: children.unwrap_or_default(),
            input_layer_type,
        }
    }

    /// A method to add an InputShred to this InputLayerNode
    pub fn add_shred(&mut self, new_shred: InputShred<F>) {
        self.children.push(new_shred);
    }
}

/// An `InputLayerNode` that can't have any InputShreds automatically added to it
#[derive(Debug, Clone)]
pub struct SealedInputNode<F>(InputLayerNode<F>);

impl<F: FieldExt> CircuitNode for SealedInputNode<F> {
    fn id(&self) -> NodeId {
        self.0.id()
    }

    fn sources(&self) -> Vec<NodeId> {
        vec![]
    }

    fn children(&self) -> Option<Vec<NodeId>> {
        self.0.children()
    }
}

impl<F: FieldExt> SealedInputNode<F> {
    /// Wraps an `InputLayerNode` to indicate that it is Sealed
    pub fn new(node: InputLayerNode<F>) -> Self {
        Self(node)
    }
}

impl<F: FieldExt, Pf: ProofSystem<F, InputLayer = IL>, IL> CompilableNode<F, Pf>
    for SealedInputNode<F>
where
    IL: InputLayer<F> + From<PublicInputLayer<F>> + From<LigeroInputLayer<F>>,
{
    fn compile<'a>(
        &'a self,
        witness_builder: &mut WitnessBuilder<F, Pf>,
        circuit_map: &mut CircuitMap<'a, F>,
    ) -> Result<(), crate::layouter::layouting::DAGError> {
        self.0.compile(witness_builder, circuit_map)
    }
}
