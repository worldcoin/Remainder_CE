use remainder_shared_types::FieldExt;

use crate::{mle::{dense::DenseMle, MleIndex, zero::ZeroMleRef, Mle}, zkdt::structs::{BinDecomp16Bit, DecisionNode, LeafNode}, layer::{LayerBuilder, LayerId}, expression::ExpressionStandard};

/// b_s grabbing
pub struct SignBit<F: FieldExt> {
    bit_decomp_diff_mle: DenseMle<F, BinDecomp16Bit<F>>,
}

impl<F: FieldExt> LayerBuilder<F> for SignBit<F> {
    type Successor = DenseMle<F, F>;

    fn build_expression(&self) -> ExpressionStandard<F> {
        
        self.bit_decomp_diff_mle.mle_bit_refs()[15].clone().expression()
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        
        DenseMle::new_from_iter(self
            .bit_decomp_diff_mle.into_iter()
            .map(|sign_bit|
                sign_bit.bits[15]), id, prefix_bits)
    }
}

impl<F: FieldExt> SignBit<F> {
    /// Constructor
    pub fn new(
        bit_decomp_diff_mle: DenseMle<F, BinDecomp16Bit<F>>
    ) -> Self {
        Self {
            bit_decomp_diff_mle
        }
    }
}

/// 1 - b_s
pub struct OneMinusSignBit<F: FieldExt> {
    bit_decomp_diff_mle: DenseMle<F, BinDecomp16Bit<F>>,
}

impl<F: FieldExt> LayerBuilder<F> for OneMinusSignBit<F> {
    type Successor = DenseMle<F, F>;

    fn build_expression(&self) -> ExpressionStandard<F> {
        
        ExpressionStandard::Constant(F::one()) - ExpressionStandard::Mle(self.bit_decomp_diff_mle.mle_bit_refs()[15].clone())
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        
        DenseMle::new_from_iter(self
            .bit_decomp_diff_mle.into_iter()
            .map(|sign_bit| {
                F::one() - sign_bit.bits[15]
            }
            ), id, prefix_bits)
    }
}

impl<F: FieldExt> OneMinusSignBit<F> {
    /// Constructor
    pub fn new(
        bit_decomp_diff_mle: DenseMle<F, BinDecomp16Bit<F>>
    ) -> Self {
        Self {
            bit_decomp_diff_mle
        }
    }
}

/// Testing purposes only!
pub struct SignBitProductBuilder<F: FieldExt> {
    pos_bit_mle: DenseMle<F, F>,
    neg_bit_mle: DenseMle<F, F>,
    pos_bit_sum: DenseMle<F, F>,
    neg_bit_sum: DenseMle<F, F>
}

impl<F: FieldExt> LayerBuilder<F> for SignBitProductBuilder<F> {

    type Successor = ZeroMleRef<F>;

    fn build_expression(&self) -> ExpressionStandard<F> {
        
        // dbg!(&yes);
        ExpressionStandard::Product(vec![self.pos_bit_mle.mle_ref(), self.pos_bit_sum.mle_ref()]) + 
        ExpressionStandard::Product(vec![self.neg_bit_mle.mle_ref(), self.neg_bit_sum.mle_ref()])
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        let _hello = DenseMle::new_from_iter(
            self.pos_bit_mle.into_iter().zip(self.pos_bit_sum.into_iter()).zip(
                self.neg_bit_mle.into_iter().zip(self.neg_bit_sum.into_iter())
            ).map(
                |((pos_bit, pos_val), (neg_bit, neg_val))| {
                    (pos_bit * pos_val) + (neg_bit * neg_val)
                }
            ), id, prefix_bits.clone()
        );
        ZeroMleRef::new(self.pos_bit_mle.num_iterated_vars(), prefix_bits, id)
    }
}

impl<F: FieldExt> SignBitProductBuilder<F> {
    /// Constructor
    pub fn new(
        pos_bit_mle: DenseMle<F, F>,
        neg_bit_mle: DenseMle<F, F>,
        pos_bit_sum: DenseMle<F, F>,
        neg_bit_sum: DenseMle<F, F>
    ) -> Self {
        Self {
            pos_bit_mle,
            neg_bit_mle,
            pos_bit_sum,
            neg_bit_sum
        }
    }
}

pub struct PrevNodeLeftBuilderDecision<F: FieldExt> {
    mle_path_decision: DenseMle<F, DecisionNode<F>>,
}

impl<F: FieldExt> LayerBuilder<F> for PrevNodeLeftBuilderDecision<F> {
    type Successor = DenseMle<F, F>;

    fn build_expression(&self) -> ExpressionStandard<F> {
        
        ExpressionStandard::Negated(Box::new(ExpressionStandard::Scaled(Box::new(ExpressionStandard::Mle(self.mle_path_decision.node_id())), F::from(2_u64)) + 
        ExpressionStandard::Constant(F::one())))
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        DenseMle::new_from_iter(self
            .mle_path_decision
            .into_iter()
            .map(|DecisionNode { node_id: path_node_id, ..}|
                ((F::from(2_u64) * path_node_id)+ F::one()).neg()), id, prefix_bits)
    }
}

impl<F: FieldExt> PrevNodeLeftBuilderDecision<F> {
    /// Constructor
    pub fn new(
        mle_path_decision: DenseMle<F, DecisionNode<F>>
    ) -> Self {
        Self {
            mle_path_decision
        }
    }
}

pub struct PrevNodeRightBuilderDecision<F: FieldExt> {
    mle_path_decision: DenseMle<F, DecisionNode<F>>,
}

impl<F: FieldExt> LayerBuilder<F> for PrevNodeRightBuilderDecision<F> {
    type Successor = DenseMle<F, F>;

    fn build_expression(&self) -> ExpressionStandard<F> {
        ExpressionStandard::Negated(Box::new(ExpressionStandard::Scaled(Box::new(ExpressionStandard::Mle(self.mle_path_decision.node_id())), F::from(2_u64)) + 
        ExpressionStandard::Constant(F::from(2_u64))))
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        DenseMle::new_from_iter(self
            .mle_path_decision
            .into_iter()
            .map(|DecisionNode { node_id: path_node_id, ..}|
                ((F::from(2_u64) * path_node_id) + F::from(2_u64)).neg()), id, prefix_bits)
    }
}

impl<F: FieldExt> PrevNodeRightBuilderDecision<F> {
    /// Constructor
    pub fn new(
        mle_path_decision: DenseMle<F, DecisionNode<F>>
    ) -> Self {
        Self {
            mle_path_decision
        }
    }
}

pub struct CurrNodeBuilderDecision<F: FieldExt> {
    mle_path_decision: DenseMle<F, DecisionNode<F>>,
}

impl<F: FieldExt> LayerBuilder<F> for CurrNodeBuilderDecision<F> {
    type Successor = DenseMle<F, F>;

    fn build_expression(&self) -> ExpressionStandard<F> {
        ExpressionStandard::Mle(self.mle_path_decision.node_id())
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        DenseMle::new_from_iter(self
            .mle_path_decision
            .into_iter()
            .map(|DecisionNode { node_id: path_node_id, ..}|
                path_node_id), id, prefix_bits)
    }
}

impl<F: FieldExt> CurrNodeBuilderDecision<F> {
    /// Constructor
    pub fn new(
        mle_path_decision: DenseMle<F, DecisionNode<F>>
    ) -> Self {
        Self {
            mle_path_decision
        }
    }
}

pub struct CurrNodeBuilderLeaf<F: FieldExt> {
    mle_path_leaf: DenseMle<F, LeafNode<F>>,
}

impl<F: FieldExt> LayerBuilder<F> for CurrNodeBuilderLeaf<F> {
    type Successor = DenseMle<F, F>;

    fn build_expression(&self) -> ExpressionStandard<F> {
        ExpressionStandard::Mle(self.mle_path_leaf.node_id())
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        DenseMle::new_from_iter(self
            .mle_path_leaf
            .into_iter()
            .map(|LeafNode { node_id: path_node_id, ..}|
                path_node_id), id, prefix_bits)
    }
}

impl<F: FieldExt> CurrNodeBuilderLeaf<F> {
    /// Constructor
    pub fn new(
        mle_path_leaf: DenseMle<F, LeafNode<F>>
    ) -> Self {
        Self {
            mle_path_leaf
        }
    }
}

/// FOR TESTING ONLY
pub struct DumbBuilder<F: FieldExt> {
    mle: DenseMle<F, F>,
}

impl<F: FieldExt> LayerBuilder<F> for DumbBuilder<F> {
    type Successor = ZeroMleRef<F>;

    fn build_expression(&self) -> ExpressionStandard<F> {
        
        ExpressionStandard::Mle(self.mle.mle_ref()) + ExpressionStandard::Negated(Box::new(ExpressionStandard::Mle(self.mle.mle_ref())))
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        let num_vars = self.mle.num_iterated_vars();
        let zero_mle: ZeroMleRef<F> = ZeroMleRef::new(num_vars, prefix_bits, id);
        zero_mle
    }
}

impl<F: FieldExt> DumbBuilder<F> {
    /// Constructor
    pub fn new(
        mle: DenseMle<F, F>
    ) -> Self {
        Self {
            mle
        }
    }
}