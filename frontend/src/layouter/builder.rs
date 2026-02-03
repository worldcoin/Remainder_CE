//! A Circuit Builder struct that owns the [super::nodes::CircuitNode]s used
//! during circuit creation.

use core::fmt;
use std::{
    collections::{HashMap, HashSet},
    marker::PhantomData,
    ops::{Add, BitXor, Mul, Sub},
    rc::{Rc, Weak},
};

use ark_std::log2;
use hyrax::{
    gkr::input_layer::HyraxInputLayerDescription, provable_circuit::HyraxProvableCircuit,
    verifiable_circuit::HyraxVerifiableCircuit,
};
use itertools::Itertools;
use ligero::ligero_structs::LigeroAuxInfo;
use serde::{Deserialize, Serialize};
use shared_types::{curves::PrimeOrderCurve, Field, Halo2FFTFriendlyField};

use crate::{
    abstract_expr::AbstractExpression,
    layouter::{
        layouting::LayoutingError,
        nodes::{
            circuit_inputs::{InputLayerNode, InputShred},
            circuit_outputs::OutputNode,
            fiat_shamir_challenge::FiatShamirChallengeNode,
            gate::GateNode,
            identity_gate::IdentityGateNode,
            lookup::{LookupConstraint, LookupTable},
            matmult::MatMultNode,
            sector::{generate_sector_circuit_description, Sector},
            split_node::SplitNode,
            CircuitNode, NodeId,
        },
    },
};
use remainder::{
    circuit_layout::CircuitLocation,
    input_layer::{ligero_input_layer::LigeroInputLayerDescription, InputLayerDescription},
    layer::{gate::BinaryOperation, layer_enum::LayerDescriptionEnum, LayerId},
    mle::evals::MultilinearExtension,
    output_layer::OutputLayerDescription,
    provable_circuit::ProvableCircuit,
    prover::{GKRCircuitDescription, GKRError},
    utils::mle::build_composite_mle,
    verifiable_circuit::VerifiableCircuit,
};

use anyhow::{anyhow, bail, Result};

use tracing::debug;

/// A dynamically-typed reference to a [CircuitNode].
/// Used only in the front-end during the circuit-building phase.
///
/// A developer building a circuit can use these references to
/// 1. indicate specific nodes when calling methods of [CircuitBuilder],
/// 2. generate [AbstractExpression]s, typically to be used in defining [Sector] nodes.
#[derive(Clone, Debug)]
pub struct NodeRef<F: Field> {
    ptr: Weak<dyn CircuitNode>,
    _phantom: PhantomData<F>,
}

impl<F: Field> NodeRef<F> {
    fn new(ptr: Weak<dyn CircuitNode>) -> Self {
        Self {
            ptr,
            _phantom: PhantomData,
        }
    }

    /// Generates an abstract expression containing a single MLE with the data in the node
    /// referenced to by [Self].
    pub fn expr(&self) -> AbstractExpression<F> {
        self.ptr.upgrade().unwrap().id().expr()
    }

    /// Returns the [NodeId] of the node references by [Self].
    pub fn id(&self) -> NodeId {
        self.ptr.upgrade().unwrap().id()
    }

    /// Returns the number of variables of the MLE of this node.
    pub fn get_num_vars(&self) -> usize {
        self.ptr.upgrade().unwrap().get_num_vars()
    }
}

impl<F: Field> From<NodeRef<F>> for AbstractExpression<F> {
    fn from(value: NodeRef<F>) -> Self {
        value.expr()
    }
}

impl<F: Field> From<&NodeRef<F>> for AbstractExpression<F> {
    fn from(value: &NodeRef<F>) -> Self {
        value.expr()
    }
}

/// A reference to a [InputLayerNode]; a specialized version of [NodeRef].
/// Used only in the front-end during the circuit-building phase.
#[derive(Clone, Debug)]
pub struct InputLayerNodeRef<F: Field> {
    ptr: Weak<InputLayerNode>,
    _phantom: PhantomData<F>,
}

impl<F: Field> InputLayerNodeRef<F> {
    fn new(ptr: Weak<InputLayerNode>) -> Self {
        Self {
            ptr,
            _phantom: PhantomData,
        }
    }

    /// Generates an abstract expression containing a single MLE with the data in the node
    /// referenced to by [Self].
    pub fn expr(&self) -> AbstractExpression<F> {
        self.ptr.upgrade().unwrap().id().expr()
    }
}

impl<F: Field> From<InputLayerNodeRef<F>> for AbstractExpression<F> {
    fn from(value: InputLayerNodeRef<F>) -> Self {
        value.expr()
    }
}

impl<F: Field> From<&InputLayerNodeRef<F>> for AbstractExpression<F> {
    fn from(value: &InputLayerNodeRef<F>) -> Self {
        value.expr()
    }
}

/// A reference to a [FiatShamirChallengeNode]; a specialized version of [NodeRef].
/// Used only in the front-end during the circuit-building phase.
#[derive(Clone, Debug)]
pub struct FSNodeRef<F: Field> {
    ptr: Weak<FiatShamirChallengeNode>,
    _phantom: PhantomData<F>,
}

impl<F: Field> FSNodeRef<F> {
    fn new(ptr: Weak<FiatShamirChallengeNode>) -> Self {
        Self {
            ptr,
            _phantom: PhantomData,
        }
    }

    /// Generates an abstract expression containing a single MLE with the data in the node
    /// referenced to by [Self].
    pub fn expr(&self) -> AbstractExpression<F> {
        self.ptr.upgrade().unwrap().id().expr()
    }
}

impl<F: Field> From<FSNodeRef<F>> for NodeRef<F> {
    fn from(value: FSNodeRef<F>) -> Self {
        NodeRef::new(value.ptr)
    }
}

impl<F: Field> From<FSNodeRef<F>> for AbstractExpression<F> {
    fn from(value: FSNodeRef<F>) -> Self {
        value.expr()
    }
}

impl<F: Field> From<&FSNodeRef<F>> for AbstractExpression<F> {
    fn from(value: &FSNodeRef<F>) -> Self {
        value.expr()
    }
}

/// A reference to a [LookupTable]; a specialized version of [NodeRef].
/// Used only in the front-end during the circuit-building phase.
#[derive(Clone, Debug)]
pub struct LookupTableNodeRef {
    ptr: Weak<LookupTable>,
}

impl LookupTableNodeRef {
    fn new(ptr: Weak<LookupTable>) -> Self {
        Self { ptr }
    }

    /// Generates an abstract expression containing a single MLE with the data in the node
    /// referenced to by [Self].
    pub fn expr<F: Field>(&self) -> AbstractExpression<F> {
        self.ptr.upgrade().unwrap().id().expr()
    }
}

/// A reference to a [LookupConstraint]; a specialized version of [NodeRef].
/// Used only in the front-end during the circuit-building phase.
#[derive(Clone, Debug)]
pub struct LookupConstraintNodeRef {
    ptr: Weak<LookupConstraint>,
}

impl LookupConstraintNodeRef {
    fn new(ptr: Weak<LookupConstraint>) -> Self {
        Self { ptr }
    }

    /// Generates an abstract expression containing a single MLE with the data in the node
    /// referenced to by [Self].
    pub fn expr<F: Field>(&self) -> AbstractExpression<F> {
        self.ptr.upgrade().unwrap().id().expr()
    }
}

/// A struct that owns and manages [super::nodes::CircuitNode]s during
/// circuit creation.
pub struct CircuitBuilder<F: Field> {
    input_layer_nodes: Vec<Rc<InputLayerNode>>,
    input_shred_nodes: Vec<Rc<InputShred>>,
    fiat_shamir_challenge_nodes: Vec<Rc<FiatShamirChallengeNode>>,
    output_nodes: Vec<Rc<OutputNode>>,
    sector_nodes: Vec<Rc<Sector<F>>>,
    gate_nodes: Vec<Rc<GateNode>>,
    identity_gate_nodes: Vec<Rc<IdentityGateNode>>,
    split_nodes: Vec<Rc<SplitNode>>,
    matmult_nodes: Vec<Rc<MatMultNode>>,
    lookup_constraint_nodes: Vec<Rc<LookupConstraint>>,
    lookup_table_nodes: Vec<Rc<LookupTable>>,
    node_to_ptr: HashMap<NodeId, NodeRef<F>>,
    circuit_map: CircuitMap,
}

impl<F: Field> CircuitBuilder<F> {
    /// Constructs an empty [CircuitBuilder].
    pub fn new() -> Self {
        Self {
            input_layer_nodes: vec![],
            input_shred_nodes: vec![],
            fiat_shamir_challenge_nodes: vec![],
            output_nodes: vec![],
            sector_nodes: vec![],
            gate_nodes: vec![],
            identity_gate_nodes: vec![],
            split_nodes: vec![],
            matmult_nodes: vec![],
            lookup_constraint_nodes: vec![],
            lookup_table_nodes: vec![],
            node_to_ptr: HashMap::new(),
            circuit_map: CircuitMap::new(),
        }
    }

    fn into_owned_helper<T: CircuitNode + fmt::Debug>(xs: Vec<Rc<T>>) -> Vec<T> {
        xs.into_iter()
            .map(|ptr| Rc::try_unwrap(ptr).unwrap())
            .collect()
    }

    /// Generates a circuit description of all the nodes added so far.
    ///
    /// Returns a [Circuit] struct containing the circuit description and all necessary metadata for
    /// attaching inputs.
    pub fn build_with_max_layer_size(
        mut self,
        maybe_maximum_log_layer_size: Option<usize>,
    ) -> Result<Circuit<F>> {
        let input_layer_nodes = Self::into_owned_helper(self.input_layer_nodes);
        let input_shred_nodes = Self::into_owned_helper(self.input_shred_nodes);
        let fiat_shamir_challenge_nodes = Self::into_owned_helper(self.fiat_shamir_challenge_nodes);
        let output_nodes = Self::into_owned_helper(self.output_nodes);
        let sector_nodes = Self::into_owned_helper(self.sector_nodes);
        let gate_nodes = Self::into_owned_helper(self.gate_nodes);
        let identity_gate_nodes = Self::into_owned_helper(self.identity_gate_nodes);
        let split_nodes = Self::into_owned_helper(self.split_nodes);
        let matmult_nodes = Self::into_owned_helper(self.matmult_nodes);
        let lookup_constraint_nodes = Self::into_owned_helper(self.lookup_constraint_nodes);
        let lookup_table_nodes = Self::into_owned_helper(self.lookup_table_nodes);
        let id_to_sector_nodes_map: HashMap<NodeId, Sector<F>> = sector_nodes
            .iter()
            .cloned()
            .map(|node| (node.id(), node))
            .collect();

        // If the specified maximum layer size is 0, then this means we do not want to combine any layers.
        let should_combine = maybe_maximum_log_layer_size != Some(0);

        let (
            input_layer_nodes,
            fiat_shamir_challenge_nodes,
            intermediate_node_layers,
            lookup_nodes,
            output_nodes,
        ) = super::layouting::layout(
            input_layer_nodes,
            input_shred_nodes,
            fiat_shamir_challenge_nodes,
            output_nodes,
            sector_nodes,
            gate_nodes,
            identity_gate_nodes,
            split_nodes,
            matmult_nodes,
            lookup_constraint_nodes,
            lookup_table_nodes,
            should_combine,
        )
        .unwrap();

        let mut intermediate_layers = Vec::<LayerDescriptionEnum<F>>::new();
        let mut output_layers = Vec::<OutputLayerDescription<F>>::new();

        let input_layers = input_layer_nodes
            .iter()
            .map(|input_layer_node| {
                let input_layer_description = input_layer_node
                    .generate_input_layer_description::<F>(&mut self.circuit_map)
                    .unwrap();
                self.circuit_map.insert_shreds_into_input_layer(
                    input_layer_description.layer_id,
                    input_layer_node
                        .input_shreds
                        .iter()
                        .map(CircuitNode::id)
                        .collect(),
                );
                input_layer_description
            })
            .collect_vec();

        let fiat_shamir_challenges = fiat_shamir_challenge_nodes
            .iter()
            .map(|fiat_shamir_challenge_node| {
                fiat_shamir_challenge_node.generate_circuit_description::<F>(&mut self.circuit_map)
            })
            .collect_vec();

        for layer in &intermediate_node_layers {
            // We have no nodes to combine in this layer. Therefore we can directly
            // compile it and add it to the layer circuit descriptions.
            if layer.len() == 1 {
                intermediate_layers.extend(
                    layer
                        .first()
                        .unwrap()
                        .generate_circuit_description(&mut self.circuit_map)?,
                );
            } else {
                // If there are nodes to combine, they must all be sectors. We first
                // check whether they are sectors and grab their associated node as
                // a Vec<&Sector<F>>.
                //
                // From this, we can generate their circuit description.
                let sectors = layer
                    .iter()
                    .map(|sector| {
                        assert!(id_to_sector_nodes_map.contains_key(&sector.id()));
                        id_to_sector_nodes_map.get(&sector.id()).unwrap()
                    })
                    .collect_vec();
                intermediate_layers.extend(generate_sector_circuit_description(
                    &sectors,
                    &mut self.circuit_map,
                    maybe_maximum_log_layer_size,
                ));
            }
        }

        // Get the contributions of each LookupTable to the circuit description.
        (intermediate_layers, output_layers) = lookup_nodes.iter().fold(
            (intermediate_layers, output_layers),
            |(mut lookup_intermediate_acc, mut lookup_output_acc), lookup_node| {
                let (intermediate_layers, output_layer) = lookup_node
                    .generate_lookup_circuit_description(&mut self.circuit_map)
                    .unwrap();
                lookup_intermediate_acc.extend(intermediate_layers);
                lookup_output_acc.push(output_layer);
                (lookup_intermediate_acc, lookup_output_acc)
            },
        );
        output_layers =
            output_nodes
                .iter()
                .fold(output_layers, |mut output_layer_acc, output_node| {
                    output_layer_acc
                        .extend(output_node.generate_circuit_description(&mut self.circuit_map));
                    output_layer_acc
                });

        let mut circuit_description = GKRCircuitDescription {
            input_layers,
            fiat_shamir_challenges,
            intermediate_layers,
            output_layers,
        };
        circuit_description.index_mle_indices(0);

        Ok(Circuit::new(circuit_description, self.circuit_map))
    }

    /// A build function that combines layers greedily such that the circuit is optimized for having
    /// the smallest number of layers possible.
    pub fn build_with_layer_combination(self) -> Result<Circuit<F>> {
        self.build_with_max_layer_size(None)
    }

    /// A build function that does not combine any layers.
    pub fn build_without_layer_combination(self) -> Result<Circuit<F>> {
        self.build_with_max_layer_size(Some(0))
    }

    /// A default build function which does _not_ combine layers.
    /// Equivalent to `build_without_layer_combination`.
    pub fn build(self) -> Result<Circuit<F>> {
        self.build_without_layer_combination()
    }
}

impl<F: Field> CircuitBuilder<F> {
    /// Adds an [InputLayerNode] labeled `layer_label` to the builder's node collection, intented to
    /// become a `layer_kind` input later during circuit instantiation.
    ///
    /// Returns a weak pointer to the newly created layer node.
    ///
    /// Note that Input Layers and Input Shred have disjoint label scopes. A label has to be unique
    /// only in its respective scope, regardless of the inclusive relation between shreds and input
    /// layers.
    ///
    /// # Panics
    /// If `layer_label` has already been used for an existing Input Layer.
    pub fn add_input_layer(
        &mut self,
        layer_label: &str,
        layer_visibility: LayerVisibility,
    ) -> InputLayerNodeRef<F> {
        let node = Rc::new(InputLayerNode::new(None));
        let node_weak_ref = Rc::downgrade(&node);

        let layer_id = node.input_layer_id();

        self.circuit_map
            .add_input_layer(layer_id, layer_label, layer_visibility);

        self.node_to_ptr
            .insert(node.id(), NodeRef::new(node_weak_ref.clone()));

        self.input_layer_nodes.push(node);

        InputLayerNodeRef::new(node_weak_ref)
    }

    // Adds an [InputShred] labeled `label` to the builder's node collection.
    /// Returns a reference to the newly created node.
    ///
    /// Note that no method in [Self] requires to differentiate between a reference to an input
    /// shred as opposed to a generic [NodeRef], so there is no need to retain the specific type
    /// information in the returned type.
    ///
    /// # Panics
    /// If `label` has already been used for an existing Input Shred.
    pub fn add_input_shred(
        &mut self,
        label: &str,
        num_vars: usize,
        source: &InputLayerNodeRef<F>,
    ) -> NodeRef<F> {
        let source = source
            .ptr
            .upgrade()
            .expect("InputShred's source data has already been dropped");
        let node = Rc::new(InputShred::new(num_vars, &source));
        let node_weak_ref = Rc::downgrade(&node);

        let node_id = node.id();

        self.node_to_ptr
            .insert(node_id, NodeRef::new(node_weak_ref.clone()));

        // Associate `label` with the `NodeId` of the newly created node.
        self.circuit_map.add_input_shred(label, node_id);

        self.input_shred_nodes.push(node);

        NodeRef::new(node_weak_ref)
    }

    /// Adds an _zero_ [OutputNode] (using `OutputNode::new_zero()`) to the builder's node
    /// collection.
    ///
    /// TODO(Makis): Add a check for ensuring each node can be set as output at most once.
    pub fn set_output(&mut self, source: &NodeRef<F>) {
        let source = source
            .ptr
            .upgrade()
            .expect("Sector source has already been dropped");
        let node = Rc::new(OutputNode::new_zero(source.as_ref()));
        self.output_nodes.push(node);
    }

    /// Adds a [FiatShamirChallengeNode] to the builder's node colllection.
    /// Returns a typed reference to the newly created node.
    pub fn add_fiat_shamir_challenge_node(&mut self, num_challenges: usize) -> FSNodeRef<F> {
        let node = Rc::new(FiatShamirChallengeNode::new(num_challenges));
        let node_weak_ref = Rc::downgrade(&node);
        self.node_to_ptr
            .insert(node.id(), NodeRef::new(node_weak_ref.clone()));
        self.fiat_shamir_challenge_nodes.push(node);
        FSNodeRef::new(node_weak_ref)
    }

    /// Adds a [Sector] to the builder's node collection.
    /// Returns a typed reference to the newly created node.
    pub fn add_sector(&mut self, expr: AbstractExpression<F>) -> NodeRef<F> {
        let node_ids_in_use: HashSet<NodeId> = expr.get_sources().into_iter().collect();

        let num_vars_map: HashMap<NodeId, usize> = node_ids_in_use
            .into_iter()
            .map(|id| (id, self.get_ptr_from_node_id(id).get_num_vars()))
            .collect();

        let num_vars = expr
            .get_num_vars(&num_vars_map)
            .expect("Internal error duing 'num_vars' computation of an AbstractExpression");

        let node = Rc::new(Sector::<F>::new(expr, num_vars));
        let node_weak_ref = Rc::downgrade(&node);

        self.node_to_ptr
            .insert(node.id(), NodeRef::new(node_weak_ref.clone()));

        self.sector_nodes.push(node);
        NodeRef::new(node_weak_ref)
    }

    /// Adds an [IdentityGateNode] to the builder's node collection.
    /// Returns a reference to the newly created node.
    ///
    /// Note that no method in [Self] requires to differentiate between a reference to an identity
    /// gate node as opposed to a generic [NodeRef], so there is no need to retain the specific type
    /// information in the returned type.
    pub fn add_identity_gate_node(
        &mut self,
        pre_routed_data: &NodeRef<F>,
        non_zero_gates: Vec<(u32, u32)>,
        num_vars: usize,
        num_dataparallel_vars: Option<usize>,
    ) -> NodeRef<F> {
        let pre_routed_data = pre_routed_data
            .ptr
            .upgrade()
            .expect("`pre_routed_data` reference given to identity gate has been dropped");
        let node = Rc::new(IdentityGateNode::new(
            pre_routed_data.as_ref(),
            non_zero_gates,
            num_vars,
            num_dataparallel_vars,
        ));
        let node_weak_ref = Rc::downgrade(&node);
        self.node_to_ptr
            .insert(node.id(), NodeRef::new(node_weak_ref.clone()));
        self.identity_gate_nodes.push(node);
        NodeRef::new(node_weak_ref)
    }

    /// Adds an [GateNode] to the builder's node collection.
    /// Returns a reference to the newly created node.
    ///
    /// Note that no method in [Self] requires to differentiate between a reference to a gate node
    /// as opposed to a generic [NodeRef], so there is no need to retain the specific type
    /// information in the returned type.
    pub fn add_gate_node(
        &mut self,
        lhs: &NodeRef<F>,
        rhs: &NodeRef<F>,
        nonzero_gates: Vec<(u32, u32, u32)>,
        gate_operation: BinaryOperation,
        num_dataparallel_bits: Option<usize>,
    ) -> NodeRef<F> {
        let lhs = lhs
            .ptr
            .upgrade()
            .expect("lhs give to GateNode has already been dropped");
        let rhs = rhs
            .ptr
            .upgrade()
            .expect("rhs give to GateNode has already been dropped");
        let node = Rc::new(GateNode::new(
            lhs.as_ref(),
            rhs.as_ref(),
            nonzero_gates,
            gate_operation,
            num_dataparallel_bits,
        ));
        let node_weak_ref = Rc::downgrade(&node);
        self.node_to_ptr
            .insert(node.id(), NodeRef::new(node_weak_ref.clone()));
        self.gate_nodes.push(node);
        NodeRef::new(node_weak_ref)
    }

    /// Adds an [MatMultNode] to the builder's node collection.
    /// Returns a reference to the newly created node.
    ///
    /// Note that no method in [Self] requires to differentiate between a reference to a matmult
    /// node as opposed to a generic [NodeRef], so there is no need to retain the specific type
    /// information in the returned type.
    pub fn add_matmult_node(
        &mut self,
        matrix_a_node: &NodeRef<F>,
        rows_cols_num_vars_a: (usize, usize),
        matrix_b_node: &NodeRef<F>,
        rows_cols_num_vars_b: (usize, usize),
    ) -> NodeRef<F> {
        let matrix_a_node = matrix_a_node
            .ptr
            .upgrade()
            .expect("Matrix A input to MatMultNode has been dropped");
        let matrix_b_node = matrix_b_node
            .ptr
            .upgrade()
            .expect("Matrix B input to MatMultNode has been dropped");
        let node = Rc::new(MatMultNode::new(
            matrix_a_node.as_ref(),
            rows_cols_num_vars_a,
            matrix_b_node.as_ref(),
            rows_cols_num_vars_b,
        ));
        let node_weak_ref = Rc::downgrade(&node);
        self.node_to_ptr
            .insert(node.id(), NodeRef::new(node_weak_ref.clone()));
        self.matmult_nodes.push(node);
        NodeRef::new(node_weak_ref)
    }

    /// Adds an [LookupTable] to the builder's node collection.
    /// Returns a typed reference to the newly created node.
    pub fn add_lookup_table(
        &mut self,
        table: &NodeRef<F>,
        fiat_shamir_challenge_node: &FSNodeRef<F>,
    ) -> LookupTableNodeRef {
        let table = table
            .ptr
            .upgrade()
            .expect("Table input to LookupTable has already been dropped");
        let fiat_shamir_challenge_node = fiat_shamir_challenge_node
            .ptr
            .upgrade()
            .expect("FiatShamirChallegeNode input to LookupTable has already been dropped");
        let node = Rc::new(LookupTable::new(
            table.as_ref(),
            fiat_shamir_challenge_node.as_ref(),
        ));
        let node_ref = Rc::downgrade(&node);
        self.node_to_ptr
            .insert(node.id(), NodeRef::new(node_ref.clone()));
        self.lookup_table_nodes.push(node);
        LookupTableNodeRef::new(node_ref)
    }

    /// Adds an [LookupConstraint] to the builder's node collection.
    /// Returns a typed reference to the newly created node.
    pub fn add_lookup_constraint(
        &mut self,
        lookup_table: &LookupTableNodeRef,
        constrained: &NodeRef<F>,
        multiplicities: &NodeRef<F>,
    ) -> LookupConstraintNodeRef {
        let lookup_table = lookup_table
            .ptr
            .upgrade()
            .expect("LookupTable input to LookupConstraint has already been dropped");
        let constrained = constrained
            .ptr
            .upgrade()
            .expect("constrained input to LookupConstrained has already been dropped");
        let multiplicities = multiplicities
            .ptr
            .upgrade()
            .expect("multiplicites input to LookupConstrained has already been dropped");
        let node = Rc::new(LookupConstraint::new(
            lookup_table.as_ref(),
            constrained.as_ref(),
            multiplicities.as_ref(),
        ));
        let node_ref = Rc::downgrade(&node);
        self.node_to_ptr
            .insert(node.id(), NodeRef::new(node_ref.clone()));
        self.lookup_constraint_nodes.push(node);
        LookupConstraintNodeRef::new(node_ref)
    }

    /// Adds an [SplitNode] to the builder's node collection.
    /// Returns a vector of reference to the `2^num_vars` newly created nodes.
    ///
    /// Note that no method in [Self] requires to differentiate between a reference to a split node
    /// as opposed to a generic [NodeRef], so there is no need to retain the specific type
    /// information in the returned type.
    pub fn add_split_node(&mut self, input_node: &NodeRef<F>, num_vars: usize) -> Vec<NodeRef<F>> {
        let input_node = input_node
            .ptr
            .upgrade()
            .expect("input_node to SplitNode has already been dropped");
        let nodes = SplitNode::new(input_node.as_ref(), num_vars)
            .into_iter()
            .map(Rc::new)
            .collect_vec();
        debug_assert_eq!(nodes.len(), 1 << num_vars);
        let node_refs = nodes
            .iter()
            .map(|node| NodeRef::new(Rc::downgrade(node) as Weak<dyn CircuitNode>))
            .collect_vec();
        nodes
            .iter()
            .zip(node_refs.iter())
            .for_each(|(node, node_ref)| {
                self.node_to_ptr.insert(node.id(), node_ref.clone());
            });
        self.split_nodes.extend(nodes);
        node_refs
    }

    fn get_ptr_from_node_id(&self, id: NodeId) -> Rc<dyn CircuitNode> {
        self.node_to_ptr[&id].ptr.upgrade().unwrap()
    }
}

impl<F: Field> Default for CircuitBuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod test {
    use shared_types::Fr;

    use super::*;

    #[test]
    #[should_panic]
    pub fn test_unique_input_layer_label() {
        let mut builder = CircuitBuilder::<Fr>::new();

        let _input_layer1 = builder.add_input_layer("Public Input Layer", LayerVisibility::Public);
        let _input_layer2 = builder.add_input_layer("Public Input Layer", LayerVisibility::Public);
    }

    #[test]
    pub fn test_scope_mixing() {
        let mut builder = CircuitBuilder::<Fr>::new();

        let input_layer = builder.add_input_layer("label", LayerVisibility::Public);

        builder.add_input_shred("label", 1, &input_layer);
    }

    #[test]
    #[should_panic]
    pub fn test_unique_input_shred_label() {
        let mut builder = CircuitBuilder::<Fr>::new();

        let input_layer = builder.add_input_layer("Public Input Layer", LayerVisibility::Public);
        builder.add_input_shred("shred1", 1, &input_layer);
        builder.add_input_shred("shred1", 1, &input_layer);
    }

    #[test]
    #[should_panic]
    pub fn test_unique_input_shred_label2() {
        let mut builder = CircuitBuilder::<Fr>::new();

        let input_layer1 = builder.add_input_layer("Input Layer 1", LayerVisibility::Public);
        let input_layer2 = builder.add_input_layer("Input Layer 2", LayerVisibility::Committed);

        builder.add_input_shred("shred1", 1, &input_layer1);
        builder.add_input_shred("shred1", 1, &input_layer2);
    }
}

/// The Layer kind defines the visibility of an input layer's data.
#[derive(Clone, Debug, PartialEq, Copy, Serialize, Deserialize)]
pub enum LayerVisibility {
    /// Input layers whose data are visible to the verifier.
    Public,

    /// Input layers whose data are only accessible through their commitments; according to some
    /// Polynomial Commitment Scheme (PCS). The specific commitment scheme is determined when the
    /// circuit is finalized.
    Committed,
}

/// Used only inside a [CircuitMap] to keep track of its state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
enum CircuitMapState {
    /// The circuit is under construction, meaning that some of the internals mappings might be in
    /// an incomplete state.
    UnderConstruction,

    /// The circuit has been built, and all internal mappings must be in a complete and consistent
    /// state.
    Ready,
}

/// Manages the relations between all different kinds of identifiers used to specify nodes during
/// circuit building and circuit instantiation.
/// Keeps track of [LayerId]s, [NodeId]s, Labels, [LayerVisibility]s, [CircuitLocation]s.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CircuitMap {
    state: CircuitMapState,
    shreds_in_layer: HashMap<LayerId, Vec<NodeId>>,
    label_to_shred_id: HashMap<String, NodeId>,
    layer_label_to_layer_id: HashMap<String, LayerId>,
    layer_visibility: HashMap<LayerId, LayerVisibility>,
    node_location: HashMap<NodeId, (CircuitLocation, usize)>,
}

impl CircuitMap {
    /// Constructs an empty [CircuitMap] in a `CircuitMapState::UnderConstruction` state.
    /// It can receive new data until the [Self::freeze] method is called which will transition it
    /// to the `CircuitMapState::Ready` state, at which point it can only be used to answer queries.
    pub fn new() -> Self {
        Self {
            state: CircuitMapState::UnderConstruction,
            shreds_in_layer: HashMap::new(),
            label_to_shred_id: HashMap::new(),
            layer_label_to_layer_id: HashMap::new(),
            layer_visibility: HashMap::new(),
            node_location: HashMap::new(),
        }
    }

    /// Associates the node with ID `node_id` to its corresponding circuit location as well as
    /// number of variables.
    ///
    /// # Panics
    /// If [self] is not in state `CircuitMapState::UnderConstruction`,
    /// or if the node with ID `node_id` has already been assigned a location.
    pub fn add_node_id_and_location_num_vars(
        &mut self,
        node_id: NodeId,
        value: (CircuitLocation, usize),
    ) {
        assert_eq!(self.state, CircuitMapState::UnderConstruction);
        assert!(!self.node_location.contains_key(&node_id));
        self.node_location.insert(node_id, value);
    }

    /// Adds a collection of `shreds` to Input Layer with ID `input_layer_id`.
    ///
    /// # Panics
    /// If [self] is not in state `CircuitMapState::UnderConstruction`,
    /// or if `input_layer_id` has already been assigned shreds.
    pub fn insert_shreds_into_input_layer(&mut self, input_layer_id: LayerId, shreds: Vec<NodeId>) {
        assert_eq!(self.state, CircuitMapState::UnderConstruction);
        assert!(!self.shreds_in_layer.contains_key(&input_layer_id));
        self.shreds_in_layer.insert(input_layer_id, shreds);
    }

    /// Using `node_id`, retrieves the number of variables and location of this
    /// node in the circuit, or returns an error if `node_id` is missing.
    /// This method is safe to use in any `CircuitMapState`.
    pub fn get_location_num_vars_from_node_id(
        &self,
        node_id: &NodeId,
    ) -> Result<&(CircuitLocation, usize)> {
        self.node_location
            .get(node_id)
            .ok_or(anyhow!(LayoutingError::DanglingNodeId(*node_id)))
    }

    /// Adds a new Input Layer with ID `layer_id`, label `layer_label`, and visibility defined by
    /// `layer_kind`.
    ///
    /// # Panics
    /// If [self] is _not_ in state `CircuitMapState::UnderConstruction`, or if a layer already
    /// exists with either an ID equal to `layer_id` or a label equal to `layer_label`.
    pub fn add_input_layer(
        &mut self,
        layer_id: LayerId,
        layer_label: &str,
        layer_kind: LayerVisibility,
    ) {
        assert_eq!(self.state, CircuitMapState::UnderConstruction);
        assert!(!self.layer_visibility.contains_key(&layer_id));
        assert!(!self.layer_label_to_layer_id.contains_key(layer_label));
        self.layer_label_to_layer_id
            .insert(String::from(layer_label), layer_id);
        self.layer_visibility.insert(layer_id, layer_kind);
    }

    /// Adds a new Input Shred labeled `label` with node ID `shred_id`.
    ///
    /// # Panics
    /// If [self] is _not_ in a `CircuitMapState::UnderConstruction`, or if `label` is already in
    /// use.
    ///
    /// # Note
    /// While this method does _not_ panic if `shred_id` has already been given a different label,
    /// it is considered a semantic error to associate two different labels with the same node, and
    /// [Self::freeze] may detect it and panic .
    pub fn add_input_shred(&mut self, label: &str, shred_id: NodeId) {
        assert_eq!(self.state, CircuitMapState::UnderConstruction);
        assert!(!self.label_to_shred_id.contains_key(label));
        self.label_to_shred_id.insert(String::from(label), shred_id);
    }

    /// Desctructively mutates [self] to transition it from a `CircuitMapState::UnderConstruction`
    /// state to a `CircuitMapState::Ready` state. As part of the transition, consistency checks are
    /// performed to ensure all internal mappings have all the expected properties.
    ///
    /// # Panics
    /// If [self] is already in a `CircuitMapState::Ready` state, or if its internal state is
    /// inconsistent.
    pub fn freeze(mut self) -> Self {
        assert_eq!(self.state, CircuitMapState::UnderConstruction);

        // Ensure consistency between `self.shreds_in_layer` and `self.layer_visibility`: their domains
        // should conincide.
        assert_eq!(
            self.shreds_in_layer.keys().sorted().collect_vec(),
            self.layer_visibility.keys().sorted().collect_vec()
        );

        // Keep the shred location of the input shreds only.
        let input_shred_location: HashMap<NodeId, (CircuitLocation, usize)> = self
            .label_to_shred_id
            .values()
            .map(|shred_id| (*shred_id, self.node_location[shred_id].clone()))
            .collect();
        self.node_location = input_shred_location;

        // Ensure consistency between `self.shreds_in_layer` and `self.shred_location`: the
        // flattened image of the former should equal the domain of the latter.
        assert_eq!(
            self.shreds_in_layer
                .values()
                .flatten()
                .sorted()
                .collect_vec(),
            self.node_location.keys().sorted().collect_vec()
        );

        // Ensure consistency between `self.label_to_shred_id` and `self.shred_location`: the image
        // of the former shoould equal the domain of the latter.
        assert_eq!(
            self.label_to_shred_id.values().sorted().collect_vec(),
            self.node_location.keys().sorted().collect_vec()
        );

        // TODO: Ensure all circuit locations are covered in `self.shred_location`.
        // TODO: Ensure mappings are bijective.

        self.state = CircuitMapState::Ready;

        self
    }

    /// Returns the [LayerVisibility] of the Input Layer that the shred with label `shred_label` is in,
    /// or an error if the `shred_label` does not correspond to any input shred.
    pub fn get_node_kind(&self, shred_label: &str) -> Result<LayerVisibility> {
        // The call to `self.get_node_id` will check ensure the state is `Ready`.

        // This lookup may fail because the caller provided an invalid label.
        // In this case, return an `Error`.
        let shred_id = self.get_node_id(shred_label)?;

        // Subsequent lookups should never fail, assuming `Self` maintains a consistent state.
        let layer_id = self.node_location[&shred_id].0.layer_id;
        let layer_visibility = self.layer_visibility[&layer_id];

        Ok(layer_visibility)
    }

    /// Returns the [NodeId] of the Input Shred labeled `shred_label`, or an error if `shred_label`
    /// does not correspond to any input shred.
    pub fn get_node_id(&self, shred_label: &str) -> Result<NodeId> {
        // This lookup may fail because the caller provided an invalid label.
        // In this case, return an error.
        let shred_id = self
            .label_to_shred_id
            .get(shred_label)
            .ok_or(anyhow!("Unrecognized Shred Label '{shred_label}'"))?;

        Ok(*shred_id)
    }

    /// Returns a vector of all [NodeId]s of Input Shreds.
    ///
    /// # Panics
    /// If [self] is _not_ in `CircuitMapState::Ready` state.
    pub fn get_all_input_shred_ids(&self) -> Vec<NodeId> {
        assert_eq!(self.state, CircuitMapState::Ready);

        self.node_location.keys().cloned().collect_vec()
    }

    /// Returns the label of the input shred with ID `shred_id`, or an error if there is no input
    /// shred with ID `shred_id`.
    ///
    /// Intended to be used only for error-reporting; current implementation is inefficient.
    ///
    /// # Panics
    /// If [self] is _not_ in state `CircuitMapState::Ready`.
    pub fn get_shred_label_from_id(&self, shred_id: NodeId) -> Result<String> {
        assert_eq!(self.state, CircuitMapState::Ready);

        // Reverse lookup `shred_id` in `self.label_to_shred_id`.
        let labels = self
            .label_to_shred_id
            .iter()
            .filter_map(|(label, node_id)| {
                if *node_id == shred_id {
                    Some(label)
                } else {
                    None
                }
            })
            .collect_vec();

        if labels.is_empty() {
            bail!("Unrecognized Input Shred ID '{shred_id}'");
        } else {
            // Panic if more than one label maps to this input shred ID as this indicates an
            // inconsistent internal state.
            assert_eq!(labels.len(), 1);

            Ok(labels[0].clone())
        }
    }

    /// Returns a vector of all input layer IDs.
    ///
    /// TODO: Consider returning an iterator instead of `Vec`.
    ///
    /// # Panics
    /// If [self] is _not_ in `CircuitMapState::Ready` state.
    pub fn get_all_input_layer_ids(&self) -> Vec<LayerId> {
        assert_eq!(self.state, CircuitMapState::Ready);

        self.layer_visibility.keys().cloned().collect_vec()
    }

    /// Returns a vector of all _public_ input layer IDs.
    ///
    /// TODO: Consider returning an iterator instead of `Vec`.
    ///
    /// # Panics
    /// If [self] is _not_ in `CircuitMapState::Ready` state.
    pub fn get_all_public_input_layer_ids(&self) -> Vec<LayerId> {
        assert_eq!(self.state, CircuitMapState::Ready);

        self.layer_visibility
            .iter()
            .filter_map(|(layer_id, visibility)| match *visibility {
                LayerVisibility::Public => Some(layer_id),
                LayerVisibility::Committed => None,
            })
            .cloned()
            .collect_vec()
    }

    /// Returns a vector of all Input Shred IDs in the Input Layer with ID `layer_id`, or an error
    /// if there is no input layer with that ID.
    ///
    /// # Panics
    /// If [self] is _not_ in `CircuitMapState::Ready`.
    pub fn get_input_shreds_from_layer_id(&self, layer_id: LayerId) -> Result<Vec<NodeId>> {
        assert_eq!(self.state, CircuitMapState::Ready);

        Ok(self
            .shreds_in_layer
            .get(&layer_id)
            .ok_or(anyhow!("Unrecognized Input Layer ID '{layer_id}'"))?
            .clone())
    }

    /// Returns the [CircuitLocation] and number of variables of the Input Shred with ID `shred_id`,
    /// or an error if no input shred with this ID exists.
    ///
    /// # Panics
    /// If [self] is _not_ in `CircuitMapState::Ready` state.
    pub fn get_shred_location(&self, shred_id: NodeId) -> Result<(CircuitLocation, usize)> {
        assert_eq!(self.state, CircuitMapState::Ready);

        self.node_location
            .get(&shred_id)
            .ok_or(anyhow!("Unrecognized Shred ID '{shred_id}'."))
            .cloned()
    }

    /// Returns a vector with all [LayerId]s of the Input Layers with [LayerVisibility::Committed]
    /// visibility.
    ///
    /// # Panics
    /// If [self] is _not_ in `CircuitMapState::Ready` state.
    pub fn get_all_committed_layers(&self) -> Vec<LayerId> {
        assert_eq!(self.state, CircuitMapState::Ready);

        self.layer_visibility
            .iter()
            .filter_map(|(layer_id, layer_visibility)| {
                if *layer_visibility == LayerVisibility::Committed {
                    Some(layer_id)
                } else {
                    None
                }
            })
            .cloned()
            .collect_vec()
    }

    /// Returns the layer ID of the input layer with label `layer_label`, or error if no such layer
    /// exists.
    pub fn get_layer_id_from_label(&self, layer_label: &str) -> Result<LayerId> {
        self.layer_label_to_layer_id
            .get(layer_label)
            .cloned()
            .ok_or(anyhow!("No Input Layer with label {layer_label}."))
    }
}

impl Default for CircuitMap {
    fn default() -> Self {
        Self::new()
    }
}

/// A circuit whose structure is fixed, but is not yet ready to be proven or verified because its
/// missing all or some of its input data. This structs provides an API for attaching inputs and
/// generating a form of the circuit that can be proven or verified respectively, for various
/// proving systems (vanilla GKR with Ligero, or Hyrax).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "F: Field")]
pub struct Circuit<F: Field> {
    circuit_description: GKRCircuitDescription<F>,
    pub circuit_map: CircuitMap,
    partial_inputs: HashMap<NodeId, MultilinearExtension<F>>,
}

impl<F: Field> Circuit<F> {
    /// Constructor to be used by [CircuitBuilder].
    fn new(circuit_description: GKRCircuitDescription<F>, circuit_map: CircuitMap) -> Self {
        assert_eq!(circuit_map.state, CircuitMapState::UnderConstruction);

        Self {
            circuit_description,
            circuit_map: circuit_map.freeze(),
            partial_inputs: HashMap::new(),
        }
    }

    /// Return the [GKRCircuitDescription] inside this [Circuit].
    pub fn get_circuit_description(&self) -> &GKRCircuitDescription<F> {
        &self.circuit_description
    }

    /// Assign `data` to the Input Shred with label `shred_label`.
    ///
    /// # Panics
    /// If `shred_label` does not correspond to any Input Shred, or if the this Input Shred has
    /// already been assigned data. Use [Self::update_input] for replacing the data of a shred.
    pub fn set_input(&mut self, shred_label: &str, data: MultilinearExtension<F>) {
        let node_id = self.circuit_map.get_node_id(shred_label).unwrap();

        if self.partial_inputs.contains_key(&node_id) {
            panic!("Input Shred with label '{shred_label}' has already been assigned data.");
        }

        self.partial_inputs.insert(node_id, data);
    }

    /// Assign `data` to the Input Shred with label `shred_label`, discarding any existing data
    /// associated with this Input Shred.
    ///
    /// # Panics
    /// If `shred_label` does not correspond to any Input Shred.
    pub fn update_input(&mut self, shred_label: &str, data: MultilinearExtension<F>) {
        let node_id = self.circuit_map.get_node_id(shred_label).unwrap();
        self.partial_inputs.insert(node_id, data);
    }

    /// Returns whether the circuit contains an Input Layer labeled `label`.
    pub fn contains_layer(&self, label: &str) -> bool {
        self.circuit_map.get_layer_id_from_label(label).is_ok()
    }

    fn input_shred_contains_data(&self, shred_id: NodeId) -> bool {
        self.partial_inputs.contains_key(&shred_id)
    }

    /// Returns whether data has already been assigned to the Input Layer labeled `label`, or an
    /// error if no such input layer exists.
    pub fn input_layer_contains_data(&self, label: &str) -> Result<bool> {
        let layer_id = self.circuit_map.get_layer_id_from_label(label)?;

        Ok(self
            .circuit_map
            .get_input_shreds_from_layer_id(layer_id)?
            .iter()
            .all(|shred_id| self.input_shred_contains_data(*shred_id)))
    }

    /// Returns the Input Layer Description of the Input Layer with label `layer_label`.
    ///
    /// # Panics
    /// If no such layer exists, or if `self` is in an inconsistent state.
    pub fn get_input_layer_description_ref(&self, layer_label: &str) -> &InputLayerDescription {
        let layer_id = self
            .circuit_map
            .get_layer_id_from_label(layer_label)
            .unwrap();

        let x: Vec<&InputLayerDescription> = self
            .circuit_description
            .input_layers
            .iter()
            .filter(|input_layer| input_layer.layer_id == layer_id)
            .collect();

        assert_eq!(x.len(), 1);

        x[0]
    }

    /// Builds the layer MLE for `layer_id` by combining the data in all the input shreds of that layer.
    ///
    /// Returns error if `layer_id` is an invalid input layer ID, or if any shred data is missing.
    fn build_input_layer_data(&self, layer_id: LayerId) -> Result<MultilinearExtension<F>> {
        let input_shred_ids = self.circuit_map.get_input_shreds_from_layer_id(layer_id)?;

        let mut shred_mles_and_prefix_bits = vec![];
        for input_shred_id in input_shred_ids {
            let mle = self.partial_inputs.get(&input_shred_id).ok_or(anyhow!(
                "Input shred {input_shred_id} does not contain any data!"
            ))?;

            let (circuit_location, num_vars) =
                self.circuit_map.get_shred_location(input_shred_id).unwrap();

            if num_vars != mle.num_vars() {
                return Err(anyhow!(GKRError::InputShredLengthMismatch(
                    input_shred_id.get_id(),
                    num_vars,
                    mle.num_vars(),
                )));
            }
            shred_mles_and_prefix_bits.push((mle, circuit_location.prefix_bits))
        }

        Ok(build_composite_mle(&shred_mles_and_prefix_bits))
    }

    fn build_public_input_layer_data(
        &self,
        verifier_optional_inputs: bool,
    ) -> Result<HashMap<LayerId, MultilinearExtension<F>>> {
        let mut public_inputs: HashMap<LayerId, MultilinearExtension<F>> = HashMap::new();

        for input_layer_id in self.circuit_map.get_all_public_input_layer_ids() {
            // Attempt to build the input layer's MLE.
            let maybe_layer_mle = self.build_input_layer_data(input_layer_id);
            match maybe_layer_mle {
                Ok(layer_mle) => {
                    public_inputs.insert(input_layer_id, layer_mle);
                }
                Err(err) => {
                    // In the verifier case, we skip adding input data to a
                    // particular input layer if any of the inputs are missing.
                    if !verifier_optional_inputs {
                        return Result::Err(err);
                    }
                }
            }
        }

        Ok(public_inputs)
    }

    fn build_all_input_layer_data(&self) -> Result<HashMap<LayerId, MultilinearExtension<F>>> {
        // Ensure all Input Shreds have been assigned input data.
        /*
        if let Some(shred_id) = self
            .circuit_map
            .get_all_input_shred_ids()
            .into_iter()
            .find(|shred_id| !self.partial_inputs.contains_key(shred_id))
        {
            // Try to return a readable error message if possible.
            if let Ok(shred_label) = self.circuit_map.get_shred_label_from_id(shred_id) {
                bail!("Circuit Instantiation Failed: Input Shred '{shred_label}' has not been assigned any data. The label of this shred is not available.");
            } else {
                bail!("Circuit Instantiation Failed: Input Shred ID '{shred_id}' has not been assigned any data.");
            }
        }
        */

        // Build Input Layer data.
        let mut inputs: HashMap<LayerId, MultilinearExtension<F>> = HashMap::new();

        for input_layer_id in self.circuit_map.get_all_input_layer_ids() {
            let layer_mle = self.build_input_layer_data(input_layer_id)?;
            inputs.insert(input_layer_id, layer_mle);
        }

        Ok(inputs)
    }

    /// Helper function for grabbing all of the committed input layers + descriptions
    /// from the circuit map.
    ///
    /// We do this by first filtering all input layers which are
    /// [LayerVisibility::Committed], getting all input "shreds" which correspond to
    /// those input layers, and aggregating those to compute the number of variables
    /// required to represent each input layer.
    ///
    /// Finally, we set a default configuration for the Ligero PCS used to commit to
    /// each of the committed input layers' MLEs. TODO(tfHARD team): add support for
    /// custom settings for the PCS configurations.
    fn get_all_committed_input_layer_descriptions_to_ligero(
        &self,
    ) -> Vec<LigeroInputLayerDescription<F>> {
        self.circuit_map
            .get_all_committed_layers()
            .into_iter()
            .map(|layer_id| {
                let raw_needed_capacity = self
                    .circuit_map
                    .get_input_shreds_from_layer_id(layer_id)
                    .unwrap()
                    .into_iter()
                    .fold(0, |acc, shred_id| {
                        let (_, num_vars) = self.circuit_map.get_shred_location(shred_id).unwrap();
                        acc + (1_usize << num_vars)
                    });
                let padded_needed_capacity = (1 << log2(raw_needed_capacity)) as usize;
                let total_num_vars = log2(padded_needed_capacity) as usize;

                LigeroInputLayerDescription {
                    layer_id,
                    num_vars: total_num_vars,
                    aux: LigeroAuxInfo::<F>::new(1 << (total_num_vars), 4, 1.0, None),
                }
            })
            .collect()
    }

    /// Returns a [HyraxVerifiableCircuit] containing the public input layer data that have been
    /// added to `self` so far.
    pub fn gen_hyrax_verifiable_circuit<C>(&self) -> Result<HyraxVerifiableCircuit<C>>
    where
        C: PrimeOrderCurve<Scalar = F>,
    {
        let public_inputs = self.build_public_input_layer_data(true)?;

        debug!("Public inputs available: {:#?}", public_inputs.keys());
        debug!(
            "Layer Labels to Layer ID map: {:#?}",
            self.circuit_map.layer_label_to_layer_id
        );

        let hyrax_private_inputs = self
            .circuit_map
            .get_all_committed_layers()
            .into_iter()
            .map(|layer_id| {
                let raw_needed_capacity = self
                    .circuit_map
                    .get_input_shreds_from_layer_id(layer_id)
                    .unwrap()
                    .into_iter()
                    .fold(0, |acc, shred_id| {
                        let (_, num_vars) = self.circuit_map.get_shred_location(shred_id).unwrap();
                        acc + (1_usize << num_vars)
                    });
                let padded_needed_capacity = (1 << log2(raw_needed_capacity)) as usize;
                let total_num_vars = log2(padded_needed_capacity) as usize;

                Ok((
                    layer_id,
                    (
                        HyraxInputLayerDescription::new(layer_id, total_num_vars),
                        None,
                    ),
                ))
            })
            .collect::<Result<HashMap<_, _>>>()?;

        Ok(HyraxVerifiableCircuit::new(
            self.circuit_description.clone(),
            public_inputs,
            hyrax_private_inputs,
            self.circuit_map.layer_label_to_layer_id.clone(),
        ))
    }

    /// Produces a provable form of this circuit for the Hyrax-GKR proving system which uses Hyrax
    /// as a commitment scheme for private input layers, and offers zero-knowledge guarantees.
    /// Requires all input data to be populated (use `Self::set_input()` on _all_ input shreds).
    ///
    /// # Returns
    /// The generated provable circuit, or an error if the [self] is missing input data.
    pub fn gen_hyrax_provable_circuit<C>(&self) -> Result<HyraxProvableCircuit<C>>
    where
        C: PrimeOrderCurve<Scalar = F>,
    {
        let inputs = self.build_all_input_layer_data()?;

        let hyrax_private_inputs = self
            .circuit_map
            .get_all_committed_layers()
            .into_iter()
            .map(|layer_id| {
                let raw_needed_capacity = self
                    .circuit_map
                    .get_input_shreds_from_layer_id(layer_id)
                    .unwrap()
                    .into_iter()
                    .fold(0, |acc, shred_id| {
                        let (_, num_vars) = self.circuit_map.get_shred_location(shred_id).unwrap();
                        acc + (1_usize << num_vars)
                    });
                let padded_needed_capacity = (1 << log2(raw_needed_capacity)) as usize;
                let total_num_vars = log2(padded_needed_capacity) as usize;

                Ok((
                    layer_id,
                    (
                        HyraxInputLayerDescription::new(layer_id, total_num_vars),
                        None,
                    ),
                ))
            })
            .collect::<Result<HashMap<_, _>>>()?;

        Ok(HyraxProvableCircuit::new(
            self.circuit_description.clone(),
            inputs,
            hyrax_private_inputs,
            self.circuit_map.layer_label_to_layer_id.clone(),
        ))
    }
}

impl<F: Halo2FFTFriendlyField> Circuit<F> {
    /// Produces a provable form of this circuit for the vanilla GKR proving system which uses
    /// Ligero as a commitment scheme for committed input layers, and does _not_ offer any
    /// zero-knowledge guarantees.
    /// Requires all input data to be populated (use `Self::set_input()` on _all_ input shreds).
    ///
    /// # Returns
    /// The generated provable circuit, or an error if the [self] is missing input data.
    pub fn gen_provable_circuit(&self) -> Result<ProvableCircuit<F>> {
        let inputs = self.build_all_input_layer_data()?;

        let ligero_committed_inputs = self
            .get_all_committed_input_layer_descriptions_to_ligero()
            .into_iter()
            .map(|ligero_input_layer_description| {
                (
                    ligero_input_layer_description.layer_id,
                    (ligero_input_layer_description, None),
                )
            })
            .collect();

        Ok(ProvableCircuit::new(
            self.circuit_description.clone(),
            inputs,
            ligero_committed_inputs,
            self.circuit_map.layer_label_to_layer_id.clone(),
        ))
    }

    /// Returns a [VerifiableCircuit] initialized with all input data which is already
    /// known to the verifier, but no commitments to the data in the committed input layers
    /// yet.
    #[allow(clippy::type_complexity)]
    pub fn gen_verifiable_circuit(&self) -> Result<VerifiableCircuit<F>> {
        // Input data which is known to the verifier ahead of time -- note that
        // this data was manually appended using the `circuit.set_input()`
        // function.
        let verifier_predetermined_public_inputs = self.build_public_input_layer_data(true)?;

        // Sets default Ligero parameters for each of the committed input layers.
        let ligero_committed_inputs = self
            .get_all_committed_input_layer_descriptions_to_ligero()
            .into_iter()
            .map(|ligero_input_layer_description| {
                (
                    ligero_input_layer_description.layer_id,
                    (ligero_input_layer_description, None),
                )
            })
            .collect();

        Ok(VerifiableCircuit::new(
            self.circuit_description.clone(),
            verifier_predetermined_public_inputs,
            ligero_committed_inputs,
            self.circuit_map.layer_label_to_layer_id.clone(),
        ))
    }
}

/// implement the Add, Sub, and Mul traits for NodeRef and FSNodeRef
macro_rules! impl_add {
    ($Lhs:ty) => {
        impl<F: Field, Rhs: Into<AbstractExpression<F>>> Add<Rhs> for $Lhs {
            type Output = AbstractExpression<F>;

            fn add(self, rhs: Rhs) -> Self::Output {
                self.expr() + rhs.into()
            }
        }
    };
}
impl_add!(NodeRef<F>);
impl_add!(&NodeRef<F>);
impl_add!(InputLayerNodeRef<F>);
impl_add!(&InputLayerNodeRef<F>);
impl_add!(FSNodeRef<F>);
impl_add!(&FSNodeRef<F>);

macro_rules! impl_sub {
    ($Lhs:ty) => {
        impl<F: Field, Rhs: Into<AbstractExpression<F>>> Sub<Rhs> for $Lhs {
            type Output = AbstractExpression<F>;

            fn sub(self, rhs: Rhs) -> Self::Output {
                self.expr() - rhs.into()
            }
        }
    };
}
impl_sub!(NodeRef<F>);
impl_sub!(&NodeRef<F>);
impl_sub!(InputLayerNodeRef<F>);
impl_sub!(&InputLayerNodeRef<F>);
impl_sub!(FSNodeRef<F>);
impl_sub!(&FSNodeRef<F>);

macro_rules! impl_mul {
    ($Lhs:ty) => {
        impl<F: Field, Rhs: Into<AbstractExpression<F>>> Mul<Rhs> for $Lhs {
            type Output = AbstractExpression<F>;

            fn mul(self, rhs: Rhs) -> Self::Output {
                self.expr() * rhs.into()
            }
        }
    };
}
impl_mul!(NodeRef<F>);
impl_mul!(&NodeRef<F>);
impl_mul!(InputLayerNodeRef<F>);
impl_mul!(&InputLayerNodeRef<F>);
impl_mul!(FSNodeRef<F>);
impl_mul!(&FSNodeRef<F>);

macro_rules! impl_xor {
    ($Lhs:ty) => {
        impl<F: Field, Rhs: Into<AbstractExpression<F>>> BitXor<Rhs> for $Lhs {
            type Output = AbstractExpression<F>;

            fn bitxor(self, rhs: Rhs) -> Self::Output {
                self.expr() ^ rhs.into()
            }
        }
    };
}
impl_xor!(NodeRef<F>);
impl_xor!(&NodeRef<F>);
impl_xor!(InputLayerNodeRef<F>);
impl_xor!(&InputLayerNodeRef<F>);
impl_xor!(FSNodeRef<F>);
impl_xor!(&FSNodeRef<F>);
