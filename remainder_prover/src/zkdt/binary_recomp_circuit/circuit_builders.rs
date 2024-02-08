//!The LayerBuilders that build the ZKDT Circuit

use crate::expression::generic_expr::Expression;
use crate::expression::prover_expr::ProverExpressionMle;
use crate::layer::{LayerBuilder, LayerId};
use crate::mle::MleRef;
use crate::mle::dense::{DenseMle};
use crate::mle::{zero::ZeroMleRef, Mle, MleIndex};
use remainder_shared_types::FieldExt;
use super::super::structs::{BinDecomp16Bit, InputAttribute, DecisionNode};

/// For asserting that the binary decomposition of the bits which purportedly
/// make up the difference between the node's threshold value and the actual
/// attribute's value on the path recomposes (in a bitwise fashion) to create
/// the actual difference.
/// 
/// This builder computes the positive binary recomposition
/// \sum_{b_1, ... b_{15}} 2^{15 - i} * b_i
pub struct BinaryRecompBuilder<F: FieldExt> {
    diff_signed_bin_decomp: DenseMle<F, BinDecomp16Bit<F>>,
}
impl<F: FieldExt> LayerBuilder<F> for BinaryRecompBuilder<F> {
    type Successor = DenseMle<F, F>;

    fn build_expression(&self) -> Expression<F, ProverExpressionMle> {
        let bit_mle_refs = self.diff_signed_bin_decomp.mle_bit_refs();

        // --- Let's just do a linear accumulator for now ---
        // TODO!(ryancao): Rewrite this expression but as a tree
        let b_s_initial_acc = Expression::constant(F::zero());

        bit_mle_refs.into_iter().rev().enumerate().skip(1).fold(
            b_s_initial_acc,
            |acc_expr, (bit_idx, bin_decomp_mle)| {

                // --- Coeff MLE ref (i.e. b_i) ---
                let b_i_mle_expression_ptr = Box::new(Expression::mle(bin_decomp_mle));

                // --- Compute (coeff) * 2^{15 - bit_idx} ---
                let base = F::from(2_u64.pow((16 - (bit_idx + 1)) as u32));
                let b_s_times_coeff_times_base =
                    Expression::scaled(b_i_mle_expression_ptr, base);

                acc_expr + b_s_times_coeff_times_base
            },
        )
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {

        let result_iter = self.diff_signed_bin_decomp.into_iter().map(
            |signed_bin_decomp| {
                signed_bin_decomp.bits.into_iter().rev().enumerate().skip(1).fold(F::zero(), |acc, (bit_idx, cur_bit)| {
                    let base = F::from(2_u64.pow((16 - (bit_idx + 1)) as u32));
                    acc + base * cur_bit
                })
            }
        );
        
        DenseMle::new_from_iter(result_iter, id, prefix_bits)
    }
}
impl<F: FieldExt> BinaryRecompBuilder<F> {
    /// Constructor
    pub fn new(diff_signed_bin_decomp: DenseMle<F, BinDecomp16Bit<F>>) -> Self {
        Self {
            diff_signed_bin_decomp
        }
    }
}

/// For asserting that the binary decomposition of the bits which purportedly
/// make up the difference between the node's threshold value and the actual
/// attribute's value on the path recomposes (in a bitwise fashion) to create
/// the actual difference.
/// 
/// This builder computes the difference between \bar{x}.val - path_x.thr
pub struct NodePathDiffBuilder<F: FieldExt> {
    decision_node_path_mle: DenseMle<F, DecisionNode<F>>,
    permuted_inputs_mle: DenseMle<F, InputAttribute<F>>,
}
impl<F: FieldExt> LayerBuilder<F> for NodePathDiffBuilder<F> {
    type Successor = DenseMle<F, F>;

    fn build_expression(&self) -> Expression<F, ProverExpressionMle> {
        let decision_node_thr_mle_ref = self.decision_node_path_mle.threshold();
        let permuted_inputs_val_mle_ref = self.permuted_inputs_mle.attr_val(Some(decision_node_thr_mle_ref.num_vars()));

        Expression::mle(permuted_inputs_val_mle_ref) - Expression::mle(decision_node_thr_mle_ref)
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {

        // --- TODO!(ryancao): Should we really be using MleRef here? ---
        let decision_node_thr_mle_ref = self.decision_node_path_mle.threshold();
        let permuted_inputs_val_mle_ref = self.permuted_inputs_mle.attr_val(Some(decision_node_thr_mle_ref.num_vars()));

        let diff_iter = self.decision_node_path_mle.into_iter().zip(permuted_inputs_val_mle_ref.bookkeeping_table()).map(|(decision_node, attr_val)| {
            *attr_val - decision_node.threshold
        });
        DenseMle::new_from_iter(diff_iter, id, prefix_bits)
    }
}
impl<F: FieldExt> NodePathDiffBuilder<F> {
    /// Constructor
    pub fn new(
        decision_node_path_mle: DenseMle<F, DecisionNode<F>>,
        permuted_inputs_mle: DenseMle<F, InputAttribute<F>>,
    ) -> Self {
        Self {
            decision_node_path_mle,
            permuted_inputs_mle
        }
    }
}

/// For asserting that the binary decomposition of the bits which purportedly
/// make up the difference between the node's threshold value and the actual
/// attribute's value on the path recomposes (in a bitwise fashion) to create
/// the actual difference.
/// 
/// This builder computes the value `pos_recomp` - `diff` + 2 * `sign_bit` * `diff`.
/// Note that this is equivalent to
/// (1 - b_s)(`pos_recomp` - `diff`) + `b_s`(`pos_recomp` + `diff`)
pub struct BinaryRecompCheckerBuilder<F: FieldExt> {
    input_path_diff_mle: DenseMle<F, F>,
    diff_signed_bit_decomp_mle: DenseMle<F, BinDecomp16Bit<F>>,
    positive_recomp_mle: DenseMle<F, F>,
}
impl<F: FieldExt> LayerBuilder<F> for BinaryRecompCheckerBuilder<F> {
    type Successor = ZeroMleRef<F>;

    fn build_expression(&self) -> Expression<F, ProverExpressionMle> {

        // --- Grab MLE refs ---
        let positive_recomp_mle_ref = self.positive_recomp_mle.mle_ref();
        let signed_bit_mle_ref = self.diff_signed_bit_decomp_mle.mle_bit_refs()[self.diff_signed_bit_decomp_mle.mle_bit_refs().len() - 1].clone();
        let diff_mle_ref = self.input_path_diff_mle.mle_ref();

        // --- LHS of addition ---
        let pos_recomp_minus_diff = Expression::mle(positive_recomp_mle_ref) - Expression::mle(diff_mle_ref.clone());

        // --- RHS of addition ---
        let sign_bit_times_diff_ptr = Box::new(Expression::products(vec![signed_bit_mle_ref, diff_mle_ref]));
        let two_times_sign_bit_times_diff = Expression::scaled(sign_bit_times_diff_ptr, F::from(2));

        pos_recomp_minus_diff + two_times_sign_bit_times_diff
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {

        // --- Collect b_s ---
        // let sign_bit_iter = self.diff_signed_bit_decomp_mle.into_iter().map(
        //     |bin_decomp| {
        //         bin_decomp.bits[bin_decomp.bits.len() - 1]
        //     }
        // );

        // let diff_signed_bit_decomp_mle_sign_ref = self.diff_signed_bit_decomp_mle.mle_bit_refs()[self.diff_signed_bit_decomp_mle.mle_bit_refs().len() - 1].clone();
        // let sign_bit_iter = diff_signed_bit_decomp_mle_sign_ref.bookkeeping_table().iter();

        // // --- Compute the formula from above ---
        // let ret_mle_iter = self.positive_recomp_mle.into_iter().zip(sign_bit_iter.zip(self.input_path_diff_mle.into_iter())).map(
        //     |(positive_recomp, (sign_bit, diff))| {
        //         positive_recomp - diff + F::from(2) * sign_bit * diff
        //     }
        // );

        // let actual_result = DenseMle::new_from_iter(ret_mle_iter, id, prefix_bits.clone());
        // dbg!(actual_result);

        ZeroMleRef::new(self.positive_recomp_mle.num_iterated_vars(), prefix_bits, id)
    }
}
impl<F: FieldExt> BinaryRecompCheckerBuilder<F> {
    /// Constructor
    pub fn new(
        input_path_diff_mle: DenseMle<F, F>,
        diff_signed_bit_decomp_mle: DenseMle<F, BinDecomp16Bit<F>>,
        positive_recomp_mle: DenseMle<F, F>,
    ) -> Self {
        Self {
            input_path_diff_mle,
            diff_signed_bit_decomp_mle,
            positive_recomp_mle
        }
    }
}

/// Simply for testing the part where we only grab some bits from the MLE
pub struct PartialBitsCheckerBuilder<F: FieldExt> {
    permuted_inputs_mle: DenseMle<F, InputAttribute<F>>,
    decision_node_paths_mle: DenseMle<F, DecisionNode<F>>,
    num_vars_to_grab: usize,
}
impl<F: FieldExt> LayerBuilder<F> for PartialBitsCheckerBuilder<F> {
    type Successor = ZeroMleRef<F>;

    fn build_expression(&self) -> Expression<F, ProverExpressionMle> {

        // --- Grab MLE refs ---
        let permuted_inputs_mle_ref = self.permuted_inputs_mle.attr_id(Some(self.num_vars_to_grab));
        let decision_node_paths_mle_ref = self.decision_node_paths_mle.threshold();

        // --- Actual expression is just subtracting from itself ---
        Expression::mle(permuted_inputs_mle_ref.clone()) - Expression::mle(permuted_inputs_mle_ref) + 
        Expression::mle(decision_node_paths_mle_ref.clone()) - Expression::mle(decision_node_paths_mle_ref)
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        ZeroMleRef::new(self.decision_node_paths_mle.num_iterated_vars(), prefix_bits, id)
    }
}
impl<F: FieldExt> PartialBitsCheckerBuilder<F> {
    /// Constructor
    pub fn new(
        permuted_inputs_mle: DenseMle<F, InputAttribute<F>>,
        decision_node_paths_mle: DenseMle<F, DecisionNode<F>>,
        num_vars_to_grab: usize,
    ) -> Self {
        Self {
            permuted_inputs_mle,
            decision_node_paths_mle,
            num_vars_to_grab
        }
    }
}