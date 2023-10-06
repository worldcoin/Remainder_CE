//!The LayerBuilders that build the ZKDT Circuit
use std::cmp::max;

use ark_std::log2;
use itertools::Itertools;

use crate::expression::{ExpressionStandard, Expression};
use crate::layer::batched::BatchedLayer;
use crate::layer::{LayerBuilder, LayerId};
use crate::mle::MleRef;
use crate::mle::dense::{DenseMle, Tuple2};
use crate::mle::{zero::ZeroMleRef, Mle, MleIndex};
use remainder_shared_types::FieldExt;
use super::structs::{BinDecomp16Bit, InputAttribute, DecisionNode, LeafNode, BinDecomp4Bit};

/// multiply the binary tree pair-wise between the tuple
struct ProductTreeBuilder<F: FieldExt> {
    mle: DenseMle<F, Tuple2<F>>,
}

impl<F: FieldExt> LayerBuilder<F> for ProductTreeBuilder<F> {
    type Successor = DenseMle<F, F>;
    //a function that multiplies the parts of the tuple pair-wise
    fn build_expression(&self) -> ExpressionStandard<F> {
        ExpressionStandard::products(vec![self.mle.first(), self.mle.second()])
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        // Create flatmle from tuple mle
        DenseMle::new_from_iter(
            self.mle
                .into_iter()
                .map(|Tuple2((first, second))| first * second),
            id,
            prefix_bits,
        )
    }
}

/// calculates the difference between two mles
/// effectively means checking they are equal
/// should spit out ZeroMleRef
pub struct EqualityCheck<F: FieldExt> {
    mle_1: DenseMle<F, F>,
    mle_2: DenseMle<F, F>,
}

impl<F: FieldExt> LayerBuilder<F> for EqualityCheck<F> {
    type Successor = ZeroMleRef<F>;
    // the difference between two mles, should be zero valued
    fn build_expression(&self) -> ExpressionStandard<F> {
        ExpressionStandard::Mle(self.mle_1.mle_ref()) - 
        ExpressionStandard::Mle(self.mle_2.mle_ref())
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        let num_vars = max(self.mle_1.num_iterated_vars(), self.mle_2.num_iterated_vars());
        ZeroMleRef::new(num_vars, prefix_bits, id)
    }
}

impl<F: FieldExt> EqualityCheck<F> {
    /// creates new difference mle
    pub fn new(
        mle_1: DenseMle<F, F>,
        mle_2: DenseMle<F, F>,
    ) -> Self {
        Self {
            mle_1, mle_2
        }
    }

    /// creates a batched layer for equality check
    pub fn new_batched(
        mle_1: Vec<DenseMle<F, F>>,
        mle_2: Vec<DenseMle<F, F>>,
    ) -> BatchedLayer<F, Self> {
        BatchedLayer::new(mle_1.into_iter().zip(mle_2.into_iter()).map(|(mle_1, mle_2)| Self {
            mle_1, mle_2
        }).collect())
    }
}

/// checks consistency (attr_id are the same) between
/// permuted input x, and the decision node path
pub struct AttributeConsistencyBuilder<F: FieldExt> {
    mle_input: DenseMle<F, InputAttribute<F>>,
    mle_path: DenseMle<F, DecisionNode<F>>,
    tree_height: usize
}

impl<F: FieldExt> LayerBuilder<F> for AttributeConsistencyBuilder<F> {
    type Successor = DenseMle<F, F>;

    fn build_expression(&self) -> ExpressionStandard<F> {
        ExpressionStandard::Mle(self.mle_input.attr_id(Some(log2(self.tree_height) as usize))) -
        ExpressionStandard::Mle(self.mle_path.attr_id())
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {

        DenseMle::new_from_iter(self
            .mle_input
            .into_iter()
            .zip(self.mle_path.into_iter())
            .map(|(InputAttribute { attr_id: input_attr_ids, .. }, DecisionNode { attr_id: path_attr_ids, ..})|
                input_attr_ids - path_attr_ids), id, prefix_bits)
    }
}

impl<F: FieldExt> AttributeConsistencyBuilder<F> {
    /// create new halfed multiplied mle
    pub(crate) fn new(
        mle_input: DenseMle<F, InputAttribute<F>>,
        mle_path: DenseMle<F, DecisionNode<F>>,
        tree_height: usize
    ) -> Self {
        Self {
            mle_input, mle_path, tree_height
        }
    }
}

/// Similar to EqualityCheck, returns ZeroMleRef
pub struct AttributeConsistencyBuilderZeroRef<F: FieldExt> {
    mle_input: DenseMle<F, InputAttribute<F>>,
    mle_path: DenseMle<F, DecisionNode<F>>,
    tree_height: usize
}

impl<F: FieldExt> LayerBuilder<F> for AttributeConsistencyBuilderZeroRef<F> {
    type Successor = ZeroMleRef<F>;

    fn build_expression(&self) -> ExpressionStandard<F> {
        ExpressionStandard::Mle(self.mle_input.attr_id(Some(log2(self.tree_height-1) as usize))) - 
        ExpressionStandard::Mle(self.mle_path.attr_id())
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        let num_vars = self.mle_path.num_iterated_vars() - 2;
        ZeroMleRef::new(num_vars, prefix_bits, id)
    }
}

impl<F: FieldExt> AttributeConsistencyBuilderZeroRef<F> {
    /// create new halfed multiplied mle
    pub(crate) fn new(
        mle_input: DenseMle<F, InputAttribute<F>>,
        mle_path: DenseMle<F, DecisionNode<F>>,
        tree_height: usize
    ) -> Self {
        Self {
            mle_input, mle_path, tree_height
        }
    }
}

/// chunks mle into two halfs, multiply them together
pub struct SplitProductBuilder<F: FieldExt> {
    mle: DenseMle<F, F>,
}

impl<F: FieldExt> LayerBuilder<F> for SplitProductBuilder<F> {
    type Successor = DenseMle<F, F>;
    //a function that multiplies the parts of the tuple pair-wise
    fn build_expression(&self) -> ExpressionStandard<F> {

        // TODO!(ende): remove the optional padding of 1!!!
        let split_mle = self.mle.split(F::one());

        ExpressionStandard::products(vec![split_mle.first(), split_mle.second()])
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {

        // Create flatmle from tuple mle
        let split_mle = self.mle.split(F::one());
        DenseMle::new_from_iter(split_mle
            .into_iter()
            .map(|Tuple2((a, b))| a * b), id, prefix_bits)
    }
}

impl<F: FieldExt> SplitProductBuilder<F> {
    /// create new halfed multiplied mle
    pub fn new(
        mle: DenseMle<F, F>,
    ) -> Self {
        Self {
            mle
        }
    }
}

/// concats two mles together
pub struct IdentityBuilder<F: FieldExt> {
    mle_1: DenseMle<F, F>,
}

impl<F: FieldExt> LayerBuilder<F> for IdentityBuilder<F> {
    type Successor = DenseMle<F, F>;
    fn build_expression(&self) -> ExpressionStandard<F> {
        ExpressionStandard::Mle(self.mle_1.mle_ref())
    }
    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        DenseMle::new_from_iter(self.mle_1
            .into_iter(), id, prefix_bits) 
    }
}

impl<F: FieldExt> IdentityBuilder<F> {
    /// create new leaf node packed
    pub fn new(
        mle_1: DenseMle<F, F>,
    ) -> Self {
        Self {
            mle_1
        }
    }
}

/// concats two mles together
pub struct ConcatBuilder<F: FieldExt> {
    mle_1: DenseMle<F, F>,
    mle_2: DenseMle<F, F>,
}

impl<F: FieldExt> LayerBuilder<F> for ConcatBuilder<F> {
    type Successor = DenseMle<F, F>;
    fn build_expression(&self) -> ExpressionStandard<F> {
        ExpressionStandard::Selector(MleIndex::Iterated,
                                     Box::new(ExpressionStandard::Mle(self.mle_1.mle_ref())),
                                     Box::new(ExpressionStandard::Mle(self.mle_2.mle_ref())))
    }
    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        let max_num_var = std::cmp::max(self.mle_1.num_iterated_vars(), self.mle_2.num_iterated_vars());
        let interleave_size = 1 << (max_num_var + 1);
        let range_size = 1 << max_num_var;
        let mut interleaved_mle_vec = vec![F::zero(); interleave_size];
        let zero = &F::zero();
        let first_mle_ref = self.mle_1.mle_ref();
        let second_mle_ref = self.mle_2.mle_ref();

        (0..range_size).for_each(
            |idx| {
                let mle_1_idx = 1 << self.mle_1.num_iterated_vars();
                let mle_2_idx = 1 << self.mle_2.num_iterated_vars();

                let index_1 = {
                    if idx >= mle_1_idx { idx % mle_1_idx } else { idx }
                };

                let index_2 = {
                    if idx >= mle_2_idx { idx % mle_2_idx } else { idx }
                };

                let first_mle_val = first_mle_ref.bookkeeping_table.get(index_1).unwrap_or(zero);
                let second_mle_val = second_mle_ref.bookkeeping_table.get(index_2).unwrap_or(zero);
                interleaved_mle_vec[idx*2] = *first_mle_val;
                interleaved_mle_vec[idx*2 + 1] = *second_mle_val;

            }
        );

        DenseMle::new_from_raw(interleaved_mle_vec, id, prefix_bits)
    }
}

impl<F: FieldExt> ConcatBuilder<F> {
    /// create new leaf node packed
    pub fn new(
        mle_1: DenseMle<F, F>,
        mle_2: DenseMle<F, F>,
    ) -> Self {
        Self {
            mle_1, mle_2
        }
    }
}

/// Takes x, outputs r-x
/// first step in exponantiation
pub struct FSRMinusXBuilder<F: FieldExt> {
    packed_x: DenseMle<F, F>,
    r_mle: DenseMle<F, F>,
}

impl<F: FieldExt> LayerBuilder<F> for FSRMinusXBuilder<F> {
    type Successor = DenseMle<F, F>;

    fn build_expression(&self) -> ExpressionStandard<F> {
        ExpressionStandard::Mle(self.r_mle.mle_ref()) - (ExpressionStandard::Mle(self.packed_x.mle_ref()))
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        let r = self.r_mle.mle_ref().bookkeeping_table[0];
        DenseMle::new_from_iter(self.packed_x.clone().into_iter().map(
            |x| r - x
        ), id, prefix_bits)
    }
}

impl<F: FieldExt> FSRMinusXBuilder<F> {
    /// create new leaf node packed
    pub fn new(
        packed_x: DenseMle<F, F>,
        r_mle: DenseMle<F, F>,
    ) -> Self {
        Self {
            packed_x, r_mle
        }
    }
}

/// Takes x, outputs r-x
/// first step in exponantiation
pub struct RMinusXBuilder<F: FieldExt> {
    packed_x: DenseMle<F, F>,
    r: F,
}

impl<F: FieldExt> LayerBuilder<F> for RMinusXBuilder<F> {
    type Successor = DenseMle<F, F>;

    fn build_expression(&self) -> ExpressionStandard<F> {
        ExpressionStandard::Constant(self.r) - (ExpressionStandard::Mle(self.packed_x.mle_ref()))
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        DenseMle::new_from_iter(self.packed_x.clone().into_iter().map(
            |x| self.r - x
        ), id, prefix_bits)
    }
}

impl<F: FieldExt> RMinusXBuilder<F> {
    /// create new leaf node packed
    pub fn new(
        packed_x: DenseMle<F, F>,
        r: F,
    ) -> Self {
        Self {
            packed_x, r
        }
    }
}

/// Takes x, outputs x^2
/// used in repeated squaring in exponantiation
pub struct SquaringBuilder<F: FieldExt> {
    mle: DenseMle<F, F>,
}

impl<F: FieldExt> LayerBuilder<F> for SquaringBuilder<F> {
    type Successor = DenseMle<F, F>;
    fn build_expression(&self) -> ExpressionStandard<F> {
        ExpressionStandard::products(vec![self.mle.mle_ref(), self.mle.mle_ref()])
    }
    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        DenseMle::new_from_iter(self.mle
            .clone()
            .into_iter()
            .map(|x| x * x), id, prefix_bits)
    }
}

impl<F: FieldExt> SquaringBuilder<F> {
    /// create new leaf node packed
    pub fn new(
        mle: DenseMle<F, F>,
    ) -> Self {
        Self {
            mle
        }
    }
}

/// takes a densemleref that is all zeros and returns a zeromleref as the successor 
pub struct ZeroBuilder<F: FieldExt> {
    mle: DenseMle<F, F>,
}

impl<F: FieldExt> LayerBuilder<F> for ZeroBuilder<F> {
    type Successor = ZeroMleRef<F>;
    fn build_expression(&self) -> ExpressionStandard<F> {
        ExpressionStandard::Mle(self.mle.mle_ref())
    }
    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {

        let mle_num_vars = self.mle.num_iterated_vars();
        ZeroMleRef::new(mle_num_vars, prefix_bits, id)
    }
}

impl<F: FieldExt> ZeroBuilder<F> {
    /// create new leaf node packed
    pub fn new(
        mle: DenseMle<F, F>,
    ) -> Self {
        Self {
            mle
        }
    }
}

/// Takes r_minus_x_power (r-x_i)^j, outputs b_ij * (r-x_i)^j + (1-b_ij)
pub struct BitExponentiationBuilder<F: FieldExt> {
    bin_decomp: DenseMle<F, BinDecomp16Bit<F>>,
    bit_index: usize,
    r_minus_x_power: DenseMle<F, F>,
}

impl<F: FieldExt> LayerBuilder<F> for BitExponentiationBuilder<F> {
    type Successor = DenseMle<F, F>;
    fn build_expression(&self) -> ExpressionStandard<F> {
        let b_ij = self.bin_decomp.mle_bit_refs()[15-self.bit_index].clone();
        ExpressionStandard::Sum(Box::new(ExpressionStandard::products(vec![self.r_minus_x_power.mle_ref(), b_ij.clone()])),
                                Box::new(ExpressionStandard::Constant(F::one()) - ExpressionStandard::Mle(b_ij)))
    }
    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        //TODO!(fix this so it uses bin_decomp.into_iter)
        let b_ij = self.bin_decomp.mle_bit_refs()[15-self.bit_index].clone();
        DenseMle::new_from_iter(self.r_minus_x_power
            .clone()
            .into_iter()
            .zip(b_ij.bookkeeping_table.into_iter())
            .map(
            |(r_minus_x_power, b_ij_iter)| (b_ij_iter * r_minus_x_power + (F::one() - b_ij_iter))), id, prefix_bits)
    }
}

impl<F: FieldExt> BitExponentiationBuilder<F> {
    /// create new leaf node packed
    pub(crate) fn new(
        bin_decomp: DenseMle<F, BinDecomp16Bit<F>>,
        bit_index: usize,
        r_minus_x_power: DenseMle<F, F>,
    ) -> Self {
        Self {
            bin_decomp, bit_index, r_minus_x_power
        }
    }
}

/// Takes r_minus_x_power (r-x_i)^j, outputs b_ij * (r-x_i)^j + (1-b_ij)
pub struct BitExponentiationBuilderCatBoost<F: FieldExt> {
    bin_decomp: DenseMle<F, BinDecomp16Bit<F>>,
    bit_index: usize,
    r_minus_x_power: DenseMle<F, F>,
}

impl<F: FieldExt> LayerBuilder<F> for BitExponentiationBuilderCatBoost<F> {
    type Successor = DenseMle<F, F>;
    fn build_expression(&self) -> ExpressionStandard<F> {
        let b_ij = self.bin_decomp.mle_bit_refs()[self.bit_index].clone();
        ExpressionStandard::Sum(Box::new(ExpressionStandard::products(vec![self.r_minus_x_power.mle_ref(), b_ij.clone()])),
                                Box::new(ExpressionStandard::Constant(F::one()) - ExpressionStandard::Mle(b_ij)))
    }
    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        //TODO!(fix this so it uses bin_decomp.into_iter)
        let b_ij = self.bin_decomp.mle_bit_refs()[self.bit_index].clone();
        DenseMle::new_from_iter(self.r_minus_x_power
            .clone()
            .into_iter()
            .zip(b_ij.bookkeeping_table.into_iter())
            .map(
            |(r_minus_x_power, b_ij_iter)| (b_ij_iter * r_minus_x_power + (F::one() - b_ij_iter))), id, prefix_bits)
    }
}

impl<F: FieldExt> BitExponentiationBuilderCatBoost<F> {
    /// create new leaf node packed
    pub(crate) fn new(
        bin_decomp: DenseMle<F, BinDecomp16Bit<F>>,
        bit_index: usize,
        r_minus_x_power: DenseMle<F, F>,
    ) -> Self {
        Self {
            bin_decomp, bit_index, r_minus_x_power
        }
    }
}

/// Takes r_minus_x_power (r-x_i)^j, outputs b_ij * (r-x_i)^j + (1-b_ij)
pub struct BitExponentiationBuilderInput<F: FieldExt> {
    bin_decomp: DenseMle<F, BinDecomp4Bit<F>>,
    bit_index: usize,
    r_minus_x_power: DenseMle<F, F>,
}

impl<F: FieldExt> LayerBuilder<F> for BitExponentiationBuilderInput<F> {
    type Successor = DenseMle<F, F>;
    fn build_expression(&self) -> ExpressionStandard<F> {
        let b_ij = self.bin_decomp.mle_bit_refs()[self.bit_index].clone();

        ExpressionStandard::Sum(Box::new(ExpressionStandard::products(vec![self.r_minus_x_power.mle_ref(), b_ij.clone()])),
                                Box::new(ExpressionStandard::Constant(F::one()) - ExpressionStandard::Mle(b_ij)))
    }
    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        //TODO!(fix this so it uses bin_decomp.into_iter)
        let b_ij = self.bin_decomp.mle_bit_refs()[self.bit_index].clone();
        DenseMle::new_from_iter(self.r_minus_x_power
            .clone()
            .into_iter()
            .zip(b_ij.bookkeeping_table.into_iter())
            .map(
            |(r_minus_x_power, b_ij_iter)| (b_ij_iter * r_minus_x_power + (F::one() - b_ij_iter))), id, prefix_bits)
    }
}

impl<F: FieldExt> BitExponentiationBuilderInput<F> {
    /// create new leaf node packed
    pub(crate) fn new(
        bin_decomp: DenseMle<F, BinDecomp4Bit<F>>,
        bit_index: usize,
        r_minus_x_power: DenseMle<F, F>,
    ) -> Self {
        Self {
            bin_decomp, bit_index, r_minus_x_power
        }
    }
}

/// Takes (1) b_ij * (r-x_i)^j + (1-b_ij), (2) prev_prods PROD(b_ij * (r-x_i)^j + (1-b_ij)) across j
/// Outputs (1) * (2). naming (1) as multiplier
pub struct ProductBuilder<F: FieldExt> {
    multiplier: DenseMle<F, F>,
    prev_prod: DenseMle<F, F>,
}

impl<F: FieldExt> LayerBuilder<F> for ProductBuilder<F> {
    type Successor = DenseMle<F, F>;

    fn build_expression(&self) -> ExpressionStandard<F> {
        ExpressionStandard::products(vec![self.multiplier.mle_ref(), self.prev_prod.mle_ref()])
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        DenseMle::new_from_iter(self.prev_prod
            .clone()
            .into_iter()
            .zip(self.multiplier.clone().into_iter())
            .map(
            |(prev_prod, multiplier)| prev_prod * multiplier
        ), id, prefix_bits)
    }
}

impl<F: FieldExt> ProductBuilder<F> {
    /// create new leaf node packed
    pub fn new(
        multiplier: DenseMle<F, F>,
        prev_prod: DenseMle<F, F>,
    ) -> Self {
        Self {
            multiplier, prev_prod
        }
    }
}

/// packs leaf node mles
pub struct LeafPackingBuilder<F: FieldExt> {
    mle: DenseMle<F, LeafNode<F>>,
    r: F,
    r_packing: F
}

impl<F: FieldExt> LayerBuilder<F> for LeafPackingBuilder<F> {
    type Successor = DenseMle<F, F>;

    // expressions = r - (x.node_id + r_packing * x.node_val)
    fn build_expression(&self) -> ExpressionStandard<F> {
        ExpressionStandard::Constant(self.r) - (ExpressionStandard::Mle(self.mle.node_id()) +
        ExpressionStandard::Scaled(Box::new(ExpressionStandard::Mle(self.mle.node_val())), self.r_packing))
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        DenseMle::new_from_iter(self.mle.into_iter().map(
            |LeafNode {node_id, node_val}|
            self.r - (node_id + self.r_packing * node_val)
        ), id, prefix_bits)
    }
}

impl<F: FieldExt> LeafPackingBuilder<F> {
    /// create new leaf node packed
    pub(crate) fn new(
        mle: DenseMle<F, LeafNode<F>>,
        r: F,
        r_packing: F
    ) -> Self {
        Self {
            mle, r, r_packing
        }
    }
}

/// packs leaf node mles, FS version
pub struct FSLeafPackingBuilder<F: FieldExt> {
    mle: DenseMle<F, LeafNode<F>>,
    r_mle: DenseMle<F, F>,
    r_packing_mle: DenseMle<F, F>
}

impl<F: FieldExt> LayerBuilder<F> for FSLeafPackingBuilder<F> {
    type Successor = DenseMle<F, F>;

    // expressions = r - (x.node_id + r_packing * x.node_val)
    fn build_expression(&self) -> ExpressionStandard<F> {
        ExpressionStandard::Mle(self.r_mle.mle_ref()) - (ExpressionStandard::Mle(self.mle.node_id()) +
        ExpressionStandard::products(vec![self.mle.node_val(), self.r_packing_mle.mle_ref()]))
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        let r = self.r_mle.mle_ref().bookkeeping_table[0];
        let r_packing = self.r_packing_mle.mle_ref().bookkeeping_table[0];

        DenseMle::new_from_iter(self.mle.into_iter().map(
            |LeafNode {node_id, node_val}|
            r - (node_id + r_packing * node_val)
        ), id, prefix_bits)
    }
}

impl<F: FieldExt> FSLeafPackingBuilder<F> {
    /// create new leaf node packed
    pub(crate) fn new(
        mle: DenseMle<F, LeafNode<F>>,
        r_mle: DenseMle<F, F>,
        r_packing_mle: DenseMle<F, F>
    ) -> Self {
        Self {
            mle, r_mle, r_packing_mle
        }
    }
}

/// Result: r - (x.node_id + r_packing[0] * x.attr_id + r_packing[1] * x.threshold)
pub struct DecisionPackingBuilder<F: FieldExt> {
    mle: DenseMle<F, DecisionNode<F>>,
    r: F,
    r_packings: (F, F)
}

impl<F: FieldExt> LayerBuilder<F> for DecisionPackingBuilder<F> {
    type Successor = DenseMle<F, F>;

    fn build_expression(&self) -> ExpressionStandard<F> {
        ExpressionStandard::Constant(self.r) - (ExpressionStandard::Mle(self.mle.node_id()) +
        ExpressionStandard::Scaled(Box::new(ExpressionStandard::Mle(self.mle.attr_id())), self.r_packings.0) +
        ExpressionStandard::Scaled(Box::new(ExpressionStandard::Mle(self.mle.threshold())), self.r_packings.1))
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        DenseMle::new_from_iter(self.mle.into_iter().map(
            |DecisionNode { node_id, attr_id, threshold }|
            self.r - (node_id + self.r_packings.0 * attr_id + self.r_packings.1 * threshold)
        ), id, prefix_bits)
    }
}

impl<F: FieldExt> DecisionPackingBuilder<F> {
    /// create new decision node packed
    pub(crate) fn new(
        mle: DenseMle<F, DecisionNode<F>>,
        r: F,
        r_packings: (F, F)
    ) -> Self {
        Self {
            mle, r, r_packings
        }
    }
}

/// packs decision node mles, FS version
pub struct FSDecisionPackingBuilder<F: FieldExt> {
    mle: DenseMle<F, DecisionNode<F>>,
    r_mle: DenseMle<F, F>,
    r_packing_mle: DenseMle<F, F>,
    r_packing_another_mle: DenseMle<F, F>
}

impl<F: FieldExt> LayerBuilder<F> for FSDecisionPackingBuilder<F> {
    type Successor = DenseMle<F, F>;

    // expressions = r - (x.node_id + r_packing[0] * x.attr_id + r_packing[1] * x.threshold)
    fn build_expression(&self) -> ExpressionStandard<F> {
        ExpressionStandard::Mle(self.r_mle.mle_ref()) - (ExpressionStandard::Mle(self.mle.node_id()) +
        ExpressionStandard::products(vec![self.mle.attr_id(), self.r_packing_mle.mle_ref()]) +
        ExpressionStandard::products(vec![self.mle.threshold(), self.r_packing_another_mle.mle_ref()]))
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        let r = self.r_mle.mle_ref().bookkeeping_table[0];
        let r_packing = self.r_packing_mle.mle_ref().bookkeeping_table[0];
        let r_packing_another = self.r_packing_another_mle.mle_ref().bookkeeping_table[0];

        DenseMle::new_from_iter(self.mle.into_iter().map(
            |DecisionNode { node_id, attr_id, threshold }|
            r - (node_id + r_packing * attr_id + r_packing_another * threshold)
        ), id, prefix_bits)
    }
}

impl<F: FieldExt> FSDecisionPackingBuilder<F> {
    /// create new decision node packed
    pub(crate) fn new(
        mle: DenseMle<F, DecisionNode<F>>,
        r_mle: DenseMle<F, F>,
        r_packing_mle: DenseMle<F, F>,
        r_packing_another_mle: DenseMle<F, F>
    ) -> Self {
        Self {
            mle, r_mle, r_packing_mle, r_packing_another_mle
        }
    }
}

/// packs input x, FS version
pub struct FSInputPackingBuilder<F: FieldExt> {
    mle: DenseMle<F, InputAttribute<F>>,
    r_mle: DenseMle<F, F>,
    r_packing_mle: DenseMle<F, F>
}

impl<F: FieldExt> LayerBuilder<F> for FSInputPackingBuilder<F> {
    type Successor = DenseMle<F, F>;

    // expressions = r - (x.attr_id + r_packing * x.attr_val)
    fn build_expression(&self) -> ExpressionStandard<F> {
        ExpressionStandard::Mle(self.r_mle.mle_ref()) - (ExpressionStandard::Mle(self.mle.attr_id(None)) +
        ExpressionStandard::products(vec![self.mle.attr_val(None), self.r_packing_mle.mle_ref()]))
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        let r = self.r_mle.mle_ref().bookkeeping_table[0];
        let r_packing = self.r_packing_mle.mle_ref().bookkeeping_table[0];
        DenseMle::new_from_iter(self.mle.into_iter().map(|InputAttribute { attr_id, attr_val }| r - (attr_id + r_packing * attr_val)), id, prefix_bits)
    }
}

impl<F: FieldExt> FSInputPackingBuilder<F> {
    /// create new decision node packed
    pub(crate) fn new(
        mle: DenseMle<F, InputAttribute<F>>,
        r_mle: DenseMle<F, F>,
        r_packing_mle: DenseMle<F, F>
    ) -> Self {
        Self {
            mle, r_mle, r_packing_mle
        }
    }
}

/// packs input x
pub struct InputPackingBuilder<F: FieldExt> {
    mle: DenseMle<F, InputAttribute<F>>,
    r: F,
    r_packing: F
}

impl<F: FieldExt> LayerBuilder<F> for InputPackingBuilder<F> {
    type Successor = DenseMle<F, F>;

    // expressions = r - (x.attr_id + r_packing * x.attr_val)
    fn build_expression(&self) -> ExpressionStandard<F> {
        ExpressionStandard::Constant(self.r) - (ExpressionStandard::Mle(self.mle.attr_id(None)) +
        ExpressionStandard::Scaled(Box::new(ExpressionStandard::Mle(self.mle.attr_val(None))), self.r_packing))
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        DenseMle::new_from_iter(self.mle.into_iter().map(|InputAttribute { attr_id, attr_val }| self.r - (attr_id + self.r_packing * attr_val)), id, prefix_bits)
    }
}

impl<F: FieldExt> InputPackingBuilder<F> {
    /// create new decision node packed
    pub(crate) fn new(
        mle: DenseMle<F, InputAttribute<F>>,
        r: F,
        r_packing: F
    ) -> Self {
        Self {
            mle, r, r_packing
        }
    }
}

struct BinaryDecompBuilder<F: FieldExt> {
    mle: DenseMle<F, BinDecomp16Bit<F>>,
}

impl<F: FieldExt> LayerBuilder<F> for BinaryDecompBuilder<F> {
    type Successor = ZeroMleRef<F>;

    // Returns an expression that checks if the bits are binary.
    fn build_expression(&self) -> ExpressionStandard<F> {
        let decomp_bit_mle = self.mle.mle_bit_refs();
        let expressions = decomp_bit_mle
            .into_iter()
            .map(|bit| {
                let b = ExpressionStandard::Mle(bit.clone());
                let b_squared = ExpressionStandard::Product(vec![bit.clone(), bit]);
                b - b_squared
            })
            .collect_vec();

        let chunk_and_concat = |expr: &[ExpressionStandard<F>]| {
            let chunks = expr.chunks(2);
            chunks
                .map(|chunk| chunk[0].clone().concat_expr(chunk[1].clone()))
                .collect_vec()
        };

        let expr1 = chunk_and_concat(&expressions);
        let expr2 = chunk_and_concat(&expr1);
        let expr3 = chunk_and_concat(&expr2);

        expr3[0].clone().concat_expr(expr3[1].clone())
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        ZeroMleRef::new(self.mle.num_iterated_vars() + 4, prefix_bits, id)
    }
}


#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::{mle::{dense::DenseMle, MleRef}, zkdt::data_pipeline::dummy_data_generator::{generate_dummy_mles, NUM_DUMMY_INPUTS, TREE_HEIGHT, generate_dummy_mles_batch, DummyMles, generate_mles_batch_catboost_single_tree, BatchedCatboostMles, BatchedDummyMles}};
    use halo2_base::halo2_proofs::halo2curves::{bn256::Fr, FieldExt};
    use ark_std::log2;

    #[test]
    fn test_product_tree_builder_next_layer() {
        let tuple_vec = vec![
            (Fr::from(0), Fr::from(1)),
            (Fr::from(2), Fr::from(3)),
            (Fr::from(4), Fr::from(5)),
            (Fr::from(6), Fr::from(7)),
        ];

        let mle = DenseMle::new_from_iter(tuple_vec
            
            .into_iter()
            .map(Tuple2::from), LayerId::Input(0), None);

        let builder = ProductTreeBuilder { mle };
        let next_layer = builder.next_layer(LayerId::Layer(0), None);

        let expected_flat_mle_vec = vec![Fr::from(0), Fr::from(6), Fr::from(20), Fr::from(42)];
        assert_eq!(
            next_layer.mle_ref().bookkeeping_table(),
            &expected_flat_mle_vec
        );
    }


    #[test]
    fn test_binary_decomp_builder() {
        let DummyMles {dummy_binary_decomp_diffs_mle, ..} = generate_dummy_mles::<Fr>();

        let first_bin_decomp_bit_mle = dummy_binary_decomp_diffs_mle.mle_bit_refs();
        let _first_bin_decomp_bit_expr =
            ExpressionStandard::Mle(first_bin_decomp_bit_mle[0].clone());

        let binary_decomp_builder = BinaryDecompBuilder {
            mle: dummy_binary_decomp_diffs_mle,
        };

        let _binary_decomp_expr: ExpressionStandard<_>= binary_decomp_builder.build_expression();
        assert_eq!(1, 1)
    }

    #[test]
    #[ignore]
    fn test_input_packing_builder() {
        // hand compute
        // for this to pass, change the parameters into the following:
        // const NUM_DUMMY_INPUTS: usize = 1 << 8;
        // const DUMMY_INPUT_LEN: usize = 1 << 1;
        // const TREE_HEIGHT: usize = 2;

        let DummyMles {dummy_input_data_mle, ..} = generate_dummy_mles::<Fr>();

        let (r, r_packing) = (Fr::from(3), Fr::from(5));
        let input_packing_builder = InputPackingBuilder{
                                                                                            mle: dummy_input_data_mle.clone(),
                                                                                            r,
                                                                                            r_packing
                                                                                        };
        let input_packed_expression = input_packing_builder.build_expression();
        println!("layer expression: {:?}", input_packed_expression);

        let next_layer = input_packing_builder.next_layer(LayerId::Layer(0), None);
        let next_layer_should_be = dummy_input_data_mle.attr_id(None).bookkeeping_table
                            .iter()
                            .zip(dummy_input_data_mle.attr_val(None).bookkeeping_table.iter())
                            .map(|(a, b)| {r - (a + &(r_packing * b))})
                            .collect_vec();

        assert_eq!(next_layer.mle_ref().bookkeeping_table, next_layer_should_be);
        println!("layer mle: {:?}", next_layer.mle_ref().bookkeeping_table);


        // the attr_id: [0, 1], its values are: [3233, 2380],  r = 3, r_packing = 5
        // characteristic poly w packing: [3 - (0 + 5 * 3233), 3 - (1 + 5 * 2380)], which is [-16162, -11898]
        assert_eq!(next_layer_should_be, vec![Fr::from(16162).neg(), Fr::from(11898).neg()]);

    }

    #[test]
    #[ignore]
    fn test_decision_node_packing_builder() {
        // hand compute
        // for this to pass, change the parameters into the following:
        // const NUM_DUMMY_INPUTS: usize = 1 << 8;
        // const DUMMY_INPUT_LEN: usize = 1 << 1;
        // const TREE_HEIGHT: usize = 2;

        let DummyMles {dummy_decision_node_paths_mle, ..} = generate_dummy_mles::<Fr>();

        let (r, r_packings) = (Fr::from(3), (Fr::from(5), Fr::from(4)));
        let input_packing_builder = DecisionPackingBuilder{
                                                                                            mle: dummy_decision_node_paths_mle.clone(),
                                                                                            r,
                                                                                            r_packings
                                                                                        };
        let input_packed_expression = input_packing_builder.build_expression();
        println!("layer expression: {:?}", input_packed_expression);

        let next_layer = input_packing_builder.next_layer(LayerId::Layer(0), None);
        let next_layer_should_be = dummy_decision_node_paths_mle.node_id().bookkeeping_table
                            .iter()
                            .zip(dummy_decision_node_paths_mle.attr_id().bookkeeping_table.iter())
                            .zip(dummy_decision_node_paths_mle.threshold().bookkeeping_table.iter())
                            .map(|((a, b), c)| {r - (a + &(r_packings.0 * b) + &(r_packings.1 * c))})
                            .collect_vec();

        assert_eq!(next_layer.mle_ref().bookkeeping_table, next_layer_should_be);
        println!("layer mle: {:?}", next_layer.mle_ref().bookkeeping_table);


        // the node_id: [0], its attr_id is: [0], its threshold is: [1206].  r = 3, r_packings = 5, 4
        // characteristic poly w packing: [3 - (0 + 5 * 0 + 4 * 1206 )], which is [-4821]
        println!("{:?}", dummy_decision_node_paths_mle);
        assert_eq!(next_layer_should_be, vec![Fr::from(4821).neg()]);

    }

    #[test]
    #[ignore]
    fn test_leaf_node_packing_builder() {
        // hand compute
        // for this to pass, change the parameters into the following:
        // const NUM_DUMMY_INPUTS: usize = 1 << 8;
        // const DUMMY_INPUT_LEN: usize = 1 << 1;
        // const TREE_HEIGHT: usize = 2;

        let DummyMles {dummy_leaf_node_paths_mle, ..} = generate_dummy_mles::<Fr>();

        let (r, r_packing) = (Fr::from(3), Fr::from(5));
        let input_packing_builder = LeafPackingBuilder{
                                                                                            mle: dummy_leaf_node_paths_mle.clone(),
                                                                                            r,
                                                                                            r_packing
                                                                                        };
        let input_packed_expression = input_packing_builder.build_expression();
        println!("layer expression: {:?}", input_packed_expression);

        let next_layer = input_packing_builder.next_layer(LayerId::Layer(0), None);
        let next_layer_should_be = dummy_leaf_node_paths_mle.node_id().bookkeeping_table
                            .iter()
                            .zip(dummy_leaf_node_paths_mle.node_val().bookkeeping_table.iter())
                            .map(|(a, b)| {r - (a + &(r_packing * b))})
                            .collect_vec();

        assert_eq!(next_layer.mle_ref().bookkeeping_table, next_layer_should_be);
        println!("layer mle: {:?}", next_layer.mle_ref().bookkeeping_table);


        // the node_id: [2], its node_val is: 17299145535799709783. r = 3, r_packing = 5
        // characteristic poly w packing: [3 - (2 + 5 * 17299145535799709783)]
        println!("{:?}", dummy_leaf_node_paths_mle);
        assert_eq!(next_layer_should_be, vec![ Fr::from(3) - (Fr::from(2) + Fr::from(5) * Fr::from(17299145535799709783_u64))]);

    }

    #[test]
    #[ignore]
    fn test_expo_builder() {
        // hand compute
        // for this to pass, change the parameters into the following:
        // const NUM_DUMMY_INPUTS: usize = 1 << 8;
        // const DUMMY_INPUT_LEN: usize = 1 << 1;
        // const TREE_HEIGHT: usize = 2;
        // RMinusXBuilder -> (SquaringBuilder -> BitExponentiationBuilder -> ProductBuilder ->)
        let DummyMles {
            dummy_multiplicities_bin_decomp_mle,
            dummy_decision_nodes_mle,
            dummy_leaf_nodes_mle, ..} = generate_dummy_mles::<Fr>();

        println!("multiplicities {:?}", dummy_multiplicities_bin_decomp_mle);
        println!("decision nodes: {:?}", dummy_decision_nodes_mle);
        println!("leaf nodes: {:?}", dummy_leaf_nodes_mle);

        let (r, r_packings) = (Fr::from(3), (Fr::from(5), Fr::from(4)));
        let another_r = Fr::from(6);

        // WHOLE TREE: decision nodes packing
        let decision_packing_builder = DecisionPackingBuilder{
            mle: dummy_decision_nodes_mle,
            r,
            r_packings
        };
        let _ = decision_packing_builder.build_expression();
        let decision_packed = decision_packing_builder.next_layer(LayerId::Layer(0), None);
        // characteristic poly w packing: [3 - (0 + 5 * 0 + 4 * 1206 )], which is [-4821]
        assert_eq!(decision_packed.mle_ref().bookkeeping_table,
            vec![ Fr::from(4821).neg()]);

        // WHOLE TREE: leaf nodes packing
        let leaf_packing_builder = LeafPackingBuilder{
            mle: dummy_leaf_nodes_mle,
            r,
            r_packing: another_r
        };
        let _ = leaf_packing_builder.build_expression();
        let leaf_packed = leaf_packing_builder.next_layer(LayerId::Layer(0), None);
        // characteristic poly w packing: [3 - (1 + 6 * 8929665060402191575)]
        // characteristic poly w packing: [3 - (2 + 6 * 17299145535799709783)]
        assert_eq!(leaf_packed.mle_ref().bookkeeping_table,
            vec![ Fr::from(3) - (Fr::from(1) + Fr::from(6) * Fr::from(8929665060402191575_u64)),
                        Fr::from(3) - (Fr::from(2) + Fr::from(6) * Fr::from(17299145535799709783_u64))]);

        println!("decision {:?}", decision_packed);
        println!("leaf {:?}", leaf_packed);

        let decision_leaf_concat_builder = ConcatBuilder{
            mle_1: decision_packed,
            mle_2: leaf_packed
        };
        let _ = decision_leaf_concat_builder.build_expression();
        let x_packed = decision_leaf_concat_builder.next_layer(LayerId::Layer(0), None);
        assert_eq!(x_packed.mle_ref().bookkeeping_table,
            vec![ Fr::from(4821).neg(),
                        Fr::from(3) - (Fr::from(1) + Fr::from(6) * Fr::from(8929665060402191575_u64)),
                        Fr::from(3) - (Fr::from(2) + Fr::from(6) * Fr::from(17299145535799709783_u64))]);

        // r-x
        let r_minus_x_builder =  RMinusXBuilder {
            packed_x: x_packed,
            r: another_r,
        };
        let _ = r_minus_x_builder.build_expression();
        let mut r_minus_x = r_minus_x_builder.next_layer(LayerId::Layer(0), None);
        assert_eq!(r_minus_x.mle_ref().bookkeeping_table,
            vec![ Fr::from(6+4821),
                        Fr::from(3) + (Fr::from(1) + Fr::from(6) * Fr::from(8929665060402191575_u64)),
                        Fr::from(3) + (Fr::from(2) + Fr::from(6) * Fr::from(17299145535799709783_u64))]);

        // b_ij * (r-x) + (1 - b_ij), j = 0
        let prev_prod_builder = BitExponentiationBuilder {
            bin_decomp: dummy_multiplicities_bin_decomp_mle.clone(),
            bit_index: 0,
            r_minus_x_power: r_minus_x.clone()
        };
        let _ = prev_prod_builder.build_expression();
        let mut prev_prod = prev_prod_builder.next_layer(LayerId::Layer(0), None);
        assert_eq!(prev_prod.mle_ref().bookkeeping_table,
            vec![ Fr::from(1),
                        Fr::from(3) + (Fr::from(1) + Fr::from(6) * Fr::from(8929665060402191575_u64)),
                        Fr::from(3) + (Fr::from(2) + Fr::from(6) * Fr::from(17299145535799709783_u64))]);

        for i in 1..16 {
            // (r-x)^2
            let r_minus_x_square_builder = SquaringBuilder {
                mle: r_minus_x
            };
            let _ = r_minus_x_square_builder.build_expression();
            let r_minus_x_square = r_minus_x_square_builder.next_layer(LayerId::Layer(0), None);

            // b_ij * (r-x)^2 + (1 - b_ij), j = 1..15
            let curr_prod_builder = BitExponentiationBuilder {
                bin_decomp: dummy_multiplicities_bin_decomp_mle.clone(),
                bit_index: i,
                r_minus_x_power: r_minus_x_square.clone()
            };
            let _ = curr_prod_builder.build_expression();
            let curr_prod = curr_prod_builder.next_layer(LayerId::Layer(0), None);

            // PROD(b_ij * (r-x) + (1 - b_ij))
            let prod_builder = ProductBuilder {
                multiplier: curr_prod,
                prev_prod
            };
            let _ = prod_builder.build_expression();
            prev_prod = prod_builder.next_layer(LayerId::Layer(0), None);

            r_minus_x = r_minus_x_square;

        }

        // Multiplicities is [256, 59, 197]
        assert_eq!(prev_prod.mle_ref().bookkeeping_table,
            vec![
                    // --- TODO!(ryancao): Is the below fix correct for `pow()`? ---
                    Fr::from(6+4821).pow(&[256_u64, 0, 0, 0]),
                    (Fr::from(3) + (Fr::from(1) + Fr::from(6) * Fr::from(8929665060402191575_u64))).pow(&[59_u64, 0, 0, 0]),
                    (Fr::from(3) + (Fr::from(2) + Fr::from(6) * Fr::from(17299145535799709783_u64))).pow(&[197_u64, 0, 0, 0])
                ]);

    }

    #[test]
    #[ignore]
    fn test_multiset_builder_dummy() {
        // hand compute
        // for this to pass, change the parameters into the following:
        // const NUM_DUMMY_INPUTS: usize = 1 << 2;
        // const DUMMY_INPUT_LEN: usize = 1 << 1;
        // const TREE_HEIGHT: usize = 2;
        // RMinusXBuilder -> (SquaringBuilder -> BitExponentiationBuilder -> ProductBuilder ->)
        let BatchedDummyMles {
            dummy_input_data_mle: _,
            dummy_permuted_input_data_mle: _,
            dummy_decision_node_paths_mle,
            dummy_leaf_node_paths_mle,
            dummy_binary_decomp_diffs_mle: _,
            dummy_multiplicities_bin_decomp_mle,
            dummy_decision_nodes_mle,
            dummy_leaf_nodes_mle,
        } = generate_dummy_mles_batch::<Fr>();

        println!("multiplicities {:?}", dummy_multiplicities_bin_decomp_mle);
        println!("decision nodes: {:?}", dummy_decision_nodes_mle);
        println!("leaf nodes: {:?}", dummy_leaf_nodes_mle);
        println!("node_paths nodes: {:?}", dummy_decision_node_paths_mle);
        println!("leaf_paths nodes: {:?}", dummy_leaf_node_paths_mle);

        let (r, r_packings) = (Fr::from(3), (Fr::from(5), Fr::from(4)));
        let another_r = Fr::from(6);


        // ********** MULTISET PART I **********
        // ------ LAYER ID: 0 ------

        // WHOLE TREE: decision nodes packing
        let decision_packing_builder = DecisionPackingBuilder{
            mle: dummy_decision_nodes_mle,
            r,
            r_packings
        };
        let _ = decision_packing_builder.build_expression();
        let decision_packed = decision_packing_builder.next_layer(LayerId::Layer(0), None);
        // characteristic poly w packing: [3 - (0 + 5 * 1 + 4 * 2085 )], which is [-8342]
        assert_eq!(decision_packed.mle_ref().bookkeeping_table,
            vec![ Fr::from(8342).neg()]);

        // WHOLE TREE: leaf nodes packing
        let leaf_packing_builder = LeafPackingBuilder{
            mle: dummy_leaf_nodes_mle,
            r,
            r_packing: another_r
        };
        let _ = leaf_packing_builder.build_expression();
        let leaf_packed = leaf_packing_builder.next_layer(LayerId::Layer(0), None);
        // characteristic poly w packing: [3 - (1 + 6 * 4845552296702772050)]
        // characteristic poly w packing: [3 - (2 + 6 * 3424474836643299239)]
        assert_eq!(leaf_packed.mle_ref().bookkeeping_table,
            vec![ Fr::from(3) - (Fr::from(1) + Fr::from(6) * Fr::from(4845552296702772050_u64)),
                        Fr::from(3) - (Fr::from(2) + Fr::from(6) * Fr::from(3424474836643299239_u64))]);

        // println!("decision {:?}", decision_packed);
        // println!("leaf {:?}", leaf_packed);

        // ------ LAYER ID: 1 ------

        let decision_leaf_concat_builder = ConcatBuilder{
            mle_1: decision_packed,
            mle_2: leaf_packed
        };
        let _ = decision_leaf_concat_builder.build_expression();
        let x_packed = decision_leaf_concat_builder.next_layer(LayerId::Layer(1), None);
        assert_eq!(x_packed.mle_ref().bookkeeping_table,
            vec![ Fr::from(8342).neg(),
                        Fr::from(3) - (Fr::from(1) + Fr::from(6) * Fr::from(4845552296702772050_u64)),
                        Fr::from(3) - (Fr::from(2) + Fr::from(6) * Fr::from(3424474836643299239_u64))]);

        // ------ LAYER ID: 2 ------

        // r-x
        let r_minus_x_builder =  RMinusXBuilder {
            packed_x: x_packed,
            r: another_r,
        };
        let _ = r_minus_x_builder.build_expression();
        let mut r_minus_x = r_minus_x_builder.next_layer(LayerId::Layer(2), None);
        assert_eq!(r_minus_x.mle_ref().bookkeeping_table,
            vec![ Fr::from(6+8342),
                        Fr::from(3) + (Fr::from(1) + Fr::from(6) * Fr::from(4845552296702772050_u64)),
                        Fr::from(3) + (Fr::from(2) + Fr::from(6) * Fr::from(3424474836643299239_u64))]);

        // ------ LAYER ID: 3 ------

        // b_ij * (r-x) + (1 - b_ij), j = 0
        let prev_prod_builder = BitExponentiationBuilder {
            bin_decomp: dummy_multiplicities_bin_decomp_mle.clone(),
            bit_index: 0,
            r_minus_x_power: r_minus_x.clone()
        };
        let _ = prev_prod_builder.build_expression();
        let mut prev_prod = prev_prod_builder.next_layer(LayerId::Layer(3), None);
        assert_eq!(prev_prod.mle_ref().bookkeeping_table,
            vec![ Fr::from(1),
                        Fr::from(3) + (Fr::from(1) + Fr::from(6) * Fr::from(4845552296702772050_u64)),
                        Fr::from(3) + (Fr::from(2) + Fr::from(6) * Fr::from(3424474836643299239_u64))]);




        for i in 1..16 {

            // ------ LAYER ID: 3 ------
            // (r-x)^2 can be put in the same layer as: b_ij * (r-x) + (1 - b_ij), j = 0
            // in general, (r-x)^(2^(j+1)) can be put in the same lyaer as : b_ij * (r-x) + (1 - b_ij), j = j

            // (r-x)^2
            let r_minus_x_square_builder = SquaringBuilder {
                mle: r_minus_x
            };
            let _ = r_minus_x_square_builder.build_expression();
            let r_minus_x_square = r_minus_x_square_builder.next_layer(LayerId::Layer(i+2), None);

            // b_ij * (r-x)^2 + (1 - b_ij), j = 1..15
            let curr_prod_builder = BitExponentiationBuilder {
                bin_decomp: dummy_multiplicities_bin_decomp_mle.clone(),
                bit_index: i,
                r_minus_x_power: r_minus_x_square.clone()
            };
            let _ = curr_prod_builder.build_expression();
            let curr_prod = curr_prod_builder.next_layer(LayerId::Layer(i+3), None);

            // PROD(b_ij * (r-x) + (1 - b_ij))
            let prod_builder = ProductBuilder {
                multiplier: curr_prod,
                prev_prod
            };
            let _ = prod_builder.build_expression();
            prev_prod = prod_builder.next_layer(LayerId::Layer(i+4), None);

            r_minus_x = r_minus_x_square;

        }

        // ------ NEXT LAYER ID: 20 ------
        // the last layer of prev_prod is LayerId::Layer(15+4) = LayerId::Layer(19)


        // Multiplicities is [4, 1, 3]
        assert_eq!(prev_prod.mle_ref().bookkeeping_table,
            vec![
                    Fr::from(6+8342).pow(&[4_u64, 0, 0, 0]),
                    (Fr::from(3) + (Fr::from(1) + Fr::from(6) * Fr::from(4845552296702772050_u64))).pow(&[1_u64, 0, 0, 0]),
                    (Fr::from(3) + (Fr::from(2) + Fr::from(6) * Fr::from(3424474836643299239_u64))).pow(&[3_u64, 0, 0, 0])
                ]);


        // ------ NEXT LAYER ID: 20 ------

        let mut exponentiated_nodes = prev_prod;
        for i in 0..TREE_HEIGHT {
            let prod_builder = SplitProductBuilder {
                mle: exponentiated_nodes
            };
            let _ = prod_builder.build_expression();
            exponentiated_nodes = prod_builder.next_layer(LayerId::Layer(20+i), None);
        }
        println!("final multiset 1. {:?}", exponentiated_nodes);

        // ------ LAST LAYER ID: 20+(TREE_HEIGHT-1)  ------

        assert_eq!(exponentiated_nodes.mle_ref().bookkeeping_table,
        vec![
            Fr::from(6+8342).pow(&[4_u64, 0, 0, 0]) *
            (Fr::from(3) + (Fr::from(1) + Fr::from(6) * Fr::from(4845552296702772050_u64))).pow(&[1_u64, 0, 0, 0]) *
            (Fr::from(3) + (Fr::from(2) + Fr::from(6) * Fr::from(3424474836643299239_u64))).pow(&[3_u64, 0, 0, 0])
        ]);


        // ********** MULTISET PART II **********
        // depending on the implementation / form factor of decision_node_paths_mle / leaf_node_paths_mle
        // (i) vectors of decision_node_path / leaf_node_path,
        // then
        // decision/leaf packing: 1 layer
        // concat: 1 layer
        // r-x: 1 layer
        // prod w pre_prod: 1 layer
        // 4 * NUM_DUMMY_INPUTS layers +  log2(TREE_HEIGHT) layers for the final product
        //
        // (ii) straight mle of all decision_node_paths / leaf_node_paths
        // then
        // decision/leaf packing: 1 layer
        // concat: 1 layer
        // r-x: 1 layer
        // literally 3 layers + log2(TREE_HEIGHT * NUM_DUMMY_INPUTS) layers for final product (binary product)

        let mut prev_prod_x_path_packed: DenseMle<Fr, Fr> = DenseMle::new_from_raw([Fr::from(1); TREE_HEIGHT].to_vec(), LayerId::Input(0), None);

        for i in 0..NUM_DUMMY_INPUTS {

            // PATH: decision nodes packing
            let decision_path_packing_builder = DecisionPackingBuilder{
                mle: dummy_decision_node_paths_mle[i].clone(),
                r,
                r_packings
            };
            let _ = decision_path_packing_builder.build_expression();
            let decision_path_packed = decision_path_packing_builder.next_layer(LayerId::Layer(0), None);
            assert_eq!(decision_path_packed.mle_ref().bookkeeping_table,
            vec![ Fr::from(8342).neg()]);

            // PATH: leaf nodes packing
            let leaf_path_packing_builder = LeafPackingBuilder{
                mle: dummy_leaf_node_paths_mle[i].clone(),
                r,
                r_packing: another_r
            };
            let _ = leaf_path_packing_builder.build_expression();
            let leaf_path_packed = leaf_path_packing_builder.next_layer(LayerId::Layer(0), None);

            if i == 1 {
                assert_eq!(leaf_path_packed.mle_ref().bookkeeping_table,
                    vec![
                        (Fr::from(3) - (Fr::from(1) + Fr::from(6) * Fr::from(4845552296702772050_u64)))
                    ]);
            } else {
                assert_eq!(leaf_path_packed.mle_ref().bookkeeping_table,
                    vec![
                        (Fr::from(3) - (Fr::from(2) + Fr::from(6) * Fr::from(3424474836643299239_u64)))
                    ]);
            }

            // r, r_packings, another_r: 3, (5, 4), 6
            // println!("path decision mle{:?}", dummy_decision_node_paths_mle_vec[i].clone());
            // println!("path decision {:?}", decision_path_packed);
            // println!("path leaf mle{:?}", dummy_leaf_node_paths_mle_vec[i].clone());
            // println!("path leaf {:?}", leaf_path_packed);

            // assert_eq!(decision_path_packed.mle_ref().bookkeeping_table,
            // DenseMle::new(vec![ Fr::from(3) - (Fr::from(0) + Fr::from(5) * Fr::from(0) + Fr::from(4) * Fr::from(1206))]).mle_ref().bookkeeping_table);
            // assert_eq!(leaf_path_packed.mle_ref().bookkeeping_table,
            // DenseMle::new(vec![ Fr::from(3) - (Fr::from(2) + Fr::from(6) * Fr::from(17299145535799709783 as u64))]).mle_ref().bookkeeping_table);
            //

            let decision_leaf_path_concat_builder = ConcatBuilder{
                mle_1: decision_path_packed,
                mle_2: leaf_path_packed
            };
            let _ = decision_leaf_path_concat_builder.build_expression();
            let x_path_packed = decision_leaf_path_concat_builder.next_layer(LayerId::Layer(0), None);

            if i == 1 {
                assert_eq!(x_path_packed.mle_ref().bookkeeping_table,
                    vec![Fr::from(8342).neg(),
                        (Fr::from(3) - (Fr::from(1) + Fr::from(6) * Fr::from(4845552296702772050_u64)))
                    ]);
            } else {
                assert_eq!(x_path_packed.mle_ref().bookkeeping_table,
                    vec![Fr::from(8342).neg(),
                        (Fr::from(3) - (Fr::from(2) + Fr::from(6) * Fr::from(3424474836643299239_u64)))
                    ]);
            }

            let r_minus_x_builder =  RMinusXBuilder {
                packed_x: x_path_packed,
                r: another_r,
            };
            let _ = r_minus_x_builder.build_expression();
            let curr_x_path_packed = r_minus_x_builder.next_layer(LayerId::Layer(0), None);

            if i == 1 {
                assert_eq!(curr_x_path_packed.mle_ref().bookkeeping_table,
                    vec![Fr::from(6+8342),
                        (Fr::from(3) + (Fr::from(1) + Fr::from(6) * Fr::from(4845552296702772050_u64)))
                    ]);
            } else {
                assert_eq!(curr_x_path_packed.mle_ref().bookkeeping_table,
                    vec![Fr::from(6+8342),
                        (Fr::from(3) + (Fr::from(2) + Fr::from(6) * Fr::from(3424474836643299239_u64)))
                    ]);
            }

            let prod_builder = ProductBuilder {
                multiplier: prev_prod_x_path_packed,
                prev_prod: curr_x_path_packed
            };

            let _ = prod_builder.build_expression();
            prev_prod_x_path_packed = prod_builder.next_layer(LayerId::Layer(0), None);

            // println!("prod path {:?}", prev_prod_x_path_packed);
        }

        let mut path_exponentiated = prev_prod_x_path_packed;

        assert_eq!(path_exponentiated.mle_ref().bookkeeping_table,
        vec![Fr::from(6+8342).pow(&[4_u64, 0, 0, 0]),
            (Fr::from(3) + (Fr::from(1) + Fr::from(6) * Fr::from(4845552296702772050_u64))).pow(&[1_u64, 0, 0, 0]) *
            (Fr::from(3) + (Fr::from(2) + Fr::from(6) * Fr::from(3424474836643299239_u64))).pow(&[3_u64, 0, 0, 0])
        ]);

        // dbg!(&path_exponentiated, "path exponentiated");

        for _ in 0..log2(TREE_HEIGHT) {
            let prod_builder = SplitProductBuilder {
                mle: path_exponentiated
            };
            let _ = prod_builder.build_expression();
            path_exponentiated = prod_builder.next_layer(LayerId::Layer(0), None);
        }
        assert_eq!(path_exponentiated.mle_ref().bookkeeping_table,
        vec![Fr::from(6+8342).pow(&[4_u64, 0, 0, 0]) *
            (Fr::from(3) + (Fr::from(1) + Fr::from(6) * Fr::from(4845552296702772050_u64))).pow(&[1_u64, 0, 0, 0]) *
            (Fr::from(3) + (Fr::from(2) + Fr::from(6) * Fr::from(3424474836643299239_u64))).pow(&[3_u64, 0, 0, 0])
        ]);

        println!("final multiset 2. {:?}", path_exponentiated);

        assert_eq!(exponentiated_nodes.mle_ref().bookkeeping_table,
            path_exponentiated.mle_ref().bookkeeping_table);

        println!("multi SET!!")
    }

    #[test]
    fn test_attribute_consistency_builder_dummy() {

        let mut zero_vec = vec![];
        for _ in 0..(TREE_HEIGHT-1) {
            zero_vec.push(Fr::from(0));
        }

        let BatchedDummyMles {
            dummy_permuted_input_data_mle,
            dummy_decision_node_paths_mle, ..
        } = generate_dummy_mles_batch::<Fr>();

        for i in 0..NUM_DUMMY_INPUTS {
            let attribute_consistency_build = AttributeConsistencyBuilder {
                mle_input: dummy_permuted_input_data_mle[i].clone(),
                mle_path: dummy_decision_node_paths_mle[i].clone(),
                tree_height: TREE_HEIGHT
            };
            let _ = attribute_consistency_build.build_expression();
            let difference_mle = attribute_consistency_build.next_layer(LayerId::Layer(0), None);

            assert_eq!(difference_mle.mle_ref().bookkeeping_table,
                    zero_vec)
        }
    }

    #[test]
    fn test_attribute_consistency_builder_catboost() {

        let (BatchedCatboostMles {
            permuted_input_data_mle_vec,
            decision_node_paths_mle_vec, ..
        }, (tree_height, _input_len)) = generate_mles_batch_catboost_single_tree::<Fr>(1, Path::new("upshot_data/"));

        let num_dummy_inputs = permuted_input_data_mle_vec.len();

        let mut zero_vec = vec![];
        for _ in 0..(tree_height-1) {
            zero_vec.push(Fr::from(0));
        }

        for i in 0..num_dummy_inputs {
            let attribute_consistency_build = AttributeConsistencyBuilder {
                mle_input: permuted_input_data_mle_vec[i].clone(),
                mle_path: decision_node_paths_mle_vec[i].clone(),
                tree_height
            };
            let _ = attribute_consistency_build.build_expression();
            let difference_mle = attribute_consistency_build.next_layer(LayerId::Layer(0), None);

            assert_eq!(difference_mle.mle_ref().bookkeeping_table,
                    zero_vec)
        }
    }

    #[test]
    fn test_permutation_builder_catboost() {

        let (BatchedCatboostMles {
            input_data_mle_vec,
            permuted_input_data_mle_vec, ..
        }, (_tree_height, input_len)) = generate_mles_batch_catboost_single_tree::<Fr>(1, Path::new("upshot_data/"));

        let num_dummy_inputs = permuted_input_data_mle_vec.len();

        let (r, _r_packings) = (Fr::from(3), (Fr::from(5), Fr::from(4)));
        let _another_r = Fr::from(6);
        let r_packing = Fr::from(3);

        for i in 0..num_dummy_inputs {

            let input_packing_builder = InputPackingBuilder{
                mle: input_data_mle_vec[i].clone(),
                r,
                r_packing
            };
            let _ = input_packing_builder.build_expression();
            let input_packed = input_packing_builder.next_layer(LayerId::Layer(0), None);

            let permuted_input_packing_builder = InputPackingBuilder{
                mle: permuted_input_data_mle_vec[i].clone(),
                r,
                r_packing
            };
            let _ = permuted_input_packing_builder.build_expression();
            let permuted_input_packed = permuted_input_packing_builder.next_layer(LayerId::Layer(0), None);

            let mut input_prod = input_packed;
            let mut permuted_input_prod = permuted_input_packed;

            // dbg!(&input_prod);
            // dbg!(&permuted_input_prod);

            for _ in 0..log2(input_len) {
                let prod_builder = SplitProductBuilder {
                    mle: input_prod
                };
                let _ = prod_builder.build_expression();
                input_prod = prod_builder.next_layer(LayerId::Layer(0), None);

                let permuted_prod_builder = SplitProductBuilder {
                    mle: permuted_input_prod
                };
                let _ = permuted_prod_builder.build_expression();
                permuted_input_prod = permuted_prod_builder.next_layer(LayerId::Layer(0), None);
            }

            // dbg!(input_prod.mle_ref().bookkeeping_table);
            assert_eq!(input_prod.mle_ref().bookkeeping_table,
                    permuted_input_prod.mle_ref().bookkeeping_table)
        }
    }

    #[test]
    fn test_multiset_builder_catboost() {
        // hand compute
        // for this to pass, change the parameters into the following:
        // const NUM_DUMMY_INPUTS: usize = 1 << 2;
        // const DUMMY_INPUT_LEN: usize = 1 << 1;
        // const TREE_HEIGHT: usize = 2;
        // RMinusXBuilder -> (SquaringBuilder -> BitExponentiationBuilder -> ProductBuilder ->)
        let (BatchedCatboostMles {decision_node_paths_mle_vec,
            leaf_node_paths_mle_vec,
            multiplicities_bin_decomp_mle_decision,
            multiplicities_bin_decomp_mle_leaf,
            decision_nodes_mle,
            leaf_nodes_mle, ..}, (tree_height, _input_len)) = generate_mles_batch_catboost_single_tree::<Fr>(1, Path::new("upshot_data/"));

        let num_dummy_inputs = decision_node_paths_mle_vec.len();

        // println!("node_paths nodes: {:?}", dummy_decision_node_paths_mle);
        // println!("leaf_paths nodes: {:?}", dummy_leaf_node_paths_mle);
        // println!("multiplicities {:?}", dummy_multiplicities_bin_decomp_mle);
        // println!("decision nodes: {:?}", dummy_decision_nodes_mle);
        // println!("leaf nodes: {:?}", dummy_leaf_nodes_mle);
        // return;

        let (r, r_packings) = (Fr::from(3), (Fr::from(5), Fr::from(4)));
        let another_r = Fr::from(5);


        // ********** MULTISET PART I **********
        // ------ LAYER ID: 0 ------

        // WHOLE TREE: decision nodes packing
        let decision_packing_builder = DecisionPackingBuilder{
            mle: decision_nodes_mle,
            r,
            r_packings
        };
        let _ = decision_packing_builder.build_expression();
        let decision_packed = decision_packing_builder.next_layer(LayerId::Layer(0), None);

        // WHOLE TREE: leaf nodes packing
        let leaf_packing_builder = LeafPackingBuilder{
            mle: leaf_nodes_mle,
            r,
            r_packing: another_r
        };
        let _ = leaf_packing_builder.build_expression();
        let leaf_packed = leaf_packing_builder.next_layer(LayerId::Layer(0), None);

        // ------ LAYER ID: 1 ------

        // r-x
        let r_minus_x_builder_decision =  RMinusXBuilder {
            packed_x: decision_packed,
            r: another_r,
        };
        let _ = r_minus_x_builder_decision.build_expression();
        let mut r_minus_x_decision = r_minus_x_builder_decision.next_layer(LayerId::Layer(1), None);

        let r_minus_x_builder_leaf =  RMinusXBuilder {
            packed_x: leaf_packed,
            r: another_r,
        };
        let _ = r_minus_x_builder_leaf.build_expression();
        let mut r_minus_x_leaf = r_minus_x_builder_leaf.next_layer(LayerId::Layer(1), None);

        // ------ LAYER ID: 2 ------

        // b_ij * (r-x) + (1 - b_ij), j = 0
        let prev_prod_builder_decision = BitExponentiationBuilderCatBoost {
            bin_decomp: multiplicities_bin_decomp_mle_decision.clone(),
            bit_index: 0,
            r_minus_x_power: r_minus_x_decision.clone()
        };
        let _ = prev_prod_builder_decision.build_expression();
        let mut prev_prod_decision = prev_prod_builder_decision.next_layer(LayerId::Layer(2), None);

        let prev_prod_builder_leaf = BitExponentiationBuilderCatBoost {
            bin_decomp: multiplicities_bin_decomp_mle_leaf.clone(),
            bit_index: 0,
            r_minus_x_power: r_minus_x_leaf.clone()
        };
        let _ = prev_prod_builder_leaf.build_expression();
        let mut prev_prod_leaf = prev_prod_builder_leaf.next_layer(LayerId::Layer(2), None);

        for i in 1..16 {

            // ------ LAYER ID: 2 ------
            // (r-x)^2 can be put in the same layer as: b_ij * (r-x) + (1 - b_ij), j = 0
            // in general, (r-x)^(2^(j+1)) can be put in the same lyaer as : b_ij * (r-x) + (1 - b_ij), j = j

            // (r-x)^2
            let r_minus_x_square_builder_decision = SquaringBuilder {
                mle: r_minus_x_decision
            };
            let _ = r_minus_x_square_builder_decision.build_expression();
            let r_minus_x_square_decision = r_minus_x_square_builder_decision.next_layer(LayerId::Layer(i+1), None);

            let r_minus_x_square_builder_leaf = SquaringBuilder {
                mle: r_minus_x_leaf
            };
            let _ = r_minus_x_square_builder_leaf.build_expression();
            let r_minus_x_square_leaf = r_minus_x_square_builder_leaf.next_layer(LayerId::Layer(i+1), None);

            // b_ij * (r-x)^2 + (1 - b_ij), j = 1..15
            let curr_prod_builder_decision = BitExponentiationBuilderCatBoost {
                bin_decomp: multiplicities_bin_decomp_mle_decision.clone(),
                bit_index: i,
                r_minus_x_power: r_minus_x_square_decision.clone()
            };
            let _ = curr_prod_builder_decision.build_expression();
            let curr_prod_decision = curr_prod_builder_decision.next_layer(LayerId::Layer(i+2), None);

            let curr_prod_builder_leaf = BitExponentiationBuilderCatBoost {
                bin_decomp: multiplicities_bin_decomp_mle_leaf.clone(),
                bit_index: i,
                r_minus_x_power: r_minus_x_square_leaf.clone()
            };
            let _ = curr_prod_builder_leaf.build_expression();
            let curr_prod_leaf = curr_prod_builder_leaf.next_layer(LayerId::Layer(i+2), None);

            // PROD(b_ij * (r-x) + (1 - b_ij))
            let prod_builder_decision = ProductBuilder {
                multiplier: curr_prod_decision,
                prev_prod: prev_prod_decision
            };
            let _ = prod_builder_decision.build_expression();
            prev_prod_decision = prod_builder_decision.next_layer(LayerId::Layer(i+4), None);

            let prod_builder_leaf = ProductBuilder {
                multiplier: curr_prod_leaf,
                prev_prod: prev_prod_leaf
            };
            let _ = prod_builder_leaf.build_expression();
            prev_prod_leaf = prod_builder_leaf.next_layer(LayerId::Layer(i+4), None);

            r_minus_x_decision = r_minus_x_square_decision;
            r_minus_x_leaf = r_minus_x_square_leaf;

        }

        // ------ NEXT LAYER ID: 19 ------
        // the last layer of prev_prod is LayerId::Layer(15+4) = LayerId::Layer(18)

        let mut exponentiated_decision = prev_prod_decision;
        for _i in 0..tree_height-1 {
            let prod_builder = SplitProductBuilder {
                mle: exponentiated_decision
            };
            let _ = prod_builder.build_expression();
            exponentiated_decision = prod_builder.next_layer(LayerId::Layer(0), None);
        }

        let mut exponentiated_leaf = prev_prod_leaf;
        for _i in 0..tree_height-1 {
            let prod_builder = SplitProductBuilder {
                mle: exponentiated_leaf
            };
            let _ = prod_builder.build_expression();
            exponentiated_leaf = prod_builder.next_layer(LayerId::Layer(0), None);
        }

        let prod_builder_nodes = ProductBuilder {
            multiplier: exponentiated_decision,
            prev_prod: exponentiated_leaf
        };
        let _ = prod_builder_nodes.build_expression();
        let exponentiated_nodes = prod_builder_nodes.next_layer(LayerId::Layer(0), None);

        println!("final multiset 1. {:?}", exponentiated_nodes);
        
        // ------ LAST LAYER ID: 20+(tree_height-1)  ------


        // ********** MULTISET PART II **********
        // depending on the implementation / form factor of decision_node_paths_mle / leaf_node_paths_mle
        // (i) vectors of decision_node_path / leaf_node_path,
        // then
        // decision/leaf packing: 1 layer
        // concat: 1 layer
        // r-x: 1 layer
        // prod w pre_prod: 1 layer
        // 4 * NUM_DUMMY_INPUTS layers +  log2(TREE_HEIGHT) layers for the final product
        //
        // (ii) straight mle of all decision_node_paths / leaf_node_paths
        // then
        // decision/leaf packing: 1 layer
        // concat: 1 layer
        // r-x: 1 layer
        // literally 3 layers + log2(TREE_HEIGHT * NUM_DUMMY_INPUTS) layers for final product (binary product)
        let mut one_vec = vec![];
        for _ in 0..tree_height {
            one_vec.push(Fr::from(1));
        }
        let mut prev_prod_x_path_packed: DenseMle<Fr, Fr> = DenseMle::new_from_raw(one_vec.to_vec(), LayerId::Input(0), None);

        for i in 0..num_dummy_inputs {

            // PATH: decision nodes packing
            let decision_path_packing_builder = DecisionPackingBuilder{
                mle: decision_node_paths_mle_vec[i].clone(),
                r,
                r_packings
            };
            let _ = decision_path_packing_builder.build_expression();
            let decision_path_packed = decision_path_packing_builder.next_layer(LayerId::Layer(0), None);

            // PATH: leaf nodes packing
            let leaf_path_packing_builder = LeafPackingBuilder{
                mle: leaf_node_paths_mle_vec[i].clone(),
                r,
                r_packing: another_r
            };
            let _ = leaf_path_packing_builder.build_expression();
            let leaf_path_packed = leaf_path_packing_builder.next_layer(LayerId::Layer(0), None);

            // r, r_packings, another_r: 3, (5, 4), 6
            // println!("path decision mle{:?}", dummy_decision_node_paths_mle_vec[i].clone());
            // println!("path decision {:?}", decision_path_packed);
            // println!("path leaf mle{:?}", dummy_leaf_node_paths_mle_vec[i].clone());
            // println!("path leaf {:?}", leaf_path_packed);

            // assert_eq!(decision_path_packed.mle_ref().bookkeeping_table,
            // DenseMle::new(vec![ Fr::from(3) - (Fr::from(0) + Fr::from(5) * Fr::from(0) + Fr::from(4) * Fr::from(1206))]).mle_ref().bookkeeping_table);
            // assert_eq!(leaf_path_packed.mle_ref().bookkeeping_table,
            // DenseMle::new(vec![ Fr::from(3) - (Fr::from(2) + Fr::from(6) * Fr::from(17299145535799709783 as u64))]).mle_ref().bookkeeping_table);
            //

            let decision_leaf_path_concat_builder = ConcatBuilder{
                mle_1: decision_path_packed,
                mle_2: leaf_path_packed
            };
            let _ = decision_leaf_path_concat_builder.build_expression();
            let x_path_packed = decision_leaf_path_concat_builder.next_layer(LayerId::Layer(0), None);

            let r_minus_x_builder =  RMinusXBuilder {
                packed_x: x_path_packed,
                r: another_r,
            };
            let _ = r_minus_x_builder.build_expression();
            let curr_x_path_packed = r_minus_x_builder.next_layer(LayerId::Layer(0), None);

            let prod_builder = ProductBuilder {
                multiplier: prev_prod_x_path_packed,
                prev_prod: curr_x_path_packed
            };

            let _ = prod_builder.build_expression();
            prev_prod_x_path_packed = prod_builder.next_layer(LayerId::Layer(0), None);

            // println!("prod path {:?}", prev_prod_x_path_packed);
        }

        let mut path_exponentiated = prev_prod_x_path_packed;

        // dbg!(&path_exponentiated, "path exponentiated");

        for _ in 0..log2(tree_height) {
            let prod_builder = SplitProductBuilder {
                mle: path_exponentiated
            };
            let _ = prod_builder.build_expression();
            path_exponentiated = prod_builder.next_layer(LayerId::Layer(0), None);
        }

        println!("final multiset 2. {:?}", path_exponentiated);

        assert_eq!(exponentiated_nodes.mle_ref().bookkeeping_table,
            path_exponentiated.mle_ref().bookkeeping_table);

        println!("multi SET!!")
    }
}