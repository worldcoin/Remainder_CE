use itertools::Itertools;

use crate::expression::{Expression, ExpressionStandard};
use crate::layer::{LayerBuilder, LayerId};
use crate::mle::dense::{DenseMle, Tuple2};
use crate::mle::{zero::ZeroMleRef, Mle, MleIndex};
use crate::FieldExt;

use super::structs::{BinDecomp16Bit, InputAttribute, DecisionNode, LeafNode};

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
        let mut flat_mle: DenseMle<F, F> = self
            .mle
            .into_iter()
            .map(|(first, second)| first * second)
            .collect();
        flat_mle.add_prefix_bits(prefix_bits);
        flat_mle.define_layer_id(id);
        flat_mle
    }
}

struct DecisionNodePackingBuilder<F: FieldExt> {
    mle: DenseMle<F, DecisionNode<F>>,
    r: F,
    r_packings: (F, F)
}

impl<F: FieldExt> LayerBuilder<F> for DecisionNodePackingBuilder<F> {
    type Successor = DenseMle<F, F>;

    // expressions = r - (x.node_id + r_packing[0] * x.attr_id + r_packing[1] * x.threshold)
    fn build_expression(&self) -> ExpressionStandard<F> {
        ExpressionStandard::Constant(self.r) - (ExpressionStandard::Mle(self.mle.node_id()) + 
        ExpressionStandard::Scaled(Box::new(ExpressionStandard::Mle(self.mle.attr_id())), self.r_packings.0) + 
        ExpressionStandard::Scaled(Box::new(ExpressionStandard::Mle(self.mle.threshold())), self.r_packings.1))
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        let mut flat_mle:DenseMle<F, F> = self.mle.into_iter().map(
            |((node_id, attr_id), threshold)| 
            self.r - (node_id + self.r_packings.0 * attr_id + self.r_packings.1 * threshold)
        ).collect();
        flat_mle.add_prefix_bits(prefix_bits);
        flat_mle.define_layer_id(id);
        flat_mle
    }
}

struct InputPackingBuilder<F: FieldExt> {
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
        let mut flat_mle:DenseMle<F, F> = self.mle.into_iter().map(|(id, val)| self.r - (id + self.r_packing * val)).collect();
        flat_mle.add_prefix_bits(prefix_bits);
        flat_mle.define_layer_id(id);
        flat_mle
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
        let mut expressions = decomp_bit_mle
            .into_iter()
            .map(|bit| {
                let b = ExpressionStandard::Mle(bit.clone());
                let b_squared = ExpressionStandard::Product(vec![bit.clone(), bit.clone()]);
                b - b_squared
            })
            .collect_vec();

        let chunk_and_concat = |expr: &[ExpressionStandard<F>]| {
            let chunks = expr.chunks(2);
            chunks
                .map(|chunk| chunk[0].clone().concat(chunk[1].clone()))
                .collect_vec()
        };

        let expr1 = chunk_and_concat(&expressions);
        let expr2 = chunk_and_concat(&expr1);
        let expr3 = chunk_and_concat(&expr2);

        return expr3[0].clone().concat(expr3[1].clone());
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        ZeroMleRef::new(self.mle.num_vars() + 4, prefix_bits, id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{mle::{dense::DenseMle, MleRef}, zkdt::zkdt_circuit::generate_dummy_mles};
    use ark_bn254::Fr;
    use FieldExt;

    #[test]
    fn test_product_tree_builder_next_layer() {
        let tuple_vec = vec![
            (Fr::from(0), Fr::from(1)),
            (Fr::from(2), Fr::from(3)),
            (Fr::from(4), Fr::from(5)),
            (Fr::from(6), Fr::from(7)),
        ];

        let mle = tuple_vec
            .clone()
            .into_iter()
            .map(Tuple2::from)
            .collect::<DenseMle<Fr, Tuple2<Fr>>>();

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
        let (_, _, _, _, dummy_binary_decomp_diffs_mle, _, _, _) = generate_dummy_mles::<Fr>();

        let first_bin_decomp_bit_mle = dummy_binary_decomp_diffs_mle.mle_bit_refs();
        let first_bin_decomp_bit_expr =
            ExpressionStandard::Mle(first_bin_decomp_bit_mle[0].clone());

        let binary_decomp_builder = BinaryDecompBuilder {
            mle: dummy_binary_decomp_diffs_mle,
        };

        let binary_decomp_expr: ExpressionStandard<_>= binary_decomp_builder.build_expression();
        assert_eq!(1, 1)
    }

    #[test]
    fn test_input_packing_builder() {

        let (dummy_input_data_mle,
            dummy_permuted_input_data_mle,
            _, _, _, _, _, _) = generate_dummy_mles::<Fr>();

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
                            .clone().iter()
                            .zip(dummy_input_data_mle.attr_val(None).bookkeeping_table.clone().iter())
                            .map(|(a, b)| {r - (a + &(r_packing * b))})
                            .collect_vec();
       
        assert_eq!(next_layer.mle_ref().bookkeeping_table, next_layer_should_be);
        println!("layer mle: {:?}", next_layer.mle_ref().bookkeeping_table);

        // hand compute
        // for this to pass, change the parameters into the following:
        // const DUMMY_INPUT_LEN: usize = 1 << 1;
        // const TREE_HEIGHT: usize = 2;
        // the attr_id: [0, 1], its values are: [3233, 2380],  r = 3, r_packing = 5
        // characteristic poly w packing: [3 - (0 + 5 * 3233), 3 - (1 + 5 * 2380)], which is [-16162, -11898]
        assert_eq!(next_layer_should_be, DenseMle::new(vec![Fr::from(-16162), Fr::from(-11898)]).mle_ref().bookkeeping_table);

    }

    #[test]
    fn test_decision_node_packing_builder() {

        let (_,_, dummy_decision_node_paths_mle, _, _, _, _, _) = generate_dummy_mles::<Fr>();

        let (r, r_packings) = (Fr::from(3), (Fr::from(5), Fr::from(4)));
        let input_packing_builder = DecisionNodePackingBuilder{
                                                                                            mle: dummy_decision_node_paths_mle.clone(),
                                                                                            r,
                                                                                            r_packings
                                                                                        };
        let input_packed_expression = input_packing_builder.build_expression();
        println!("layer expression: {:?}", input_packed_expression);

        let next_layer = input_packing_builder.next_layer(LayerId::Layer(0), None);
        let next_layer_should_be = dummy_decision_node_paths_mle.node_id().bookkeeping_table
                            .clone().iter()
                            .zip(dummy_decision_node_paths_mle.attr_id().bookkeeping_table.clone().iter())
                            .zip(dummy_decision_node_paths_mle.threshold().bookkeeping_table.clone().iter())
                            .map(|((a, b), c)| {r - (a + &(r_packings.0 * b) + &(r_packings.1 * c))})
                            .collect_vec();
       
        assert_eq!(next_layer.mle_ref().bookkeeping_table, next_layer_should_be);
        println!("layer mle: {:?}", next_layer.mle_ref().bookkeeping_table);

        // hand compute
        // for this to pass, change the parameters into the following:
        // const DUMMY_INPUT_LEN: usize = 1 << 1;
        // const TREE_HEIGHT: usize = 2;
        // the node_id: [0], its attr_id is: [0], its threshold is: [1206].  r = 3, r_packings = 5, 4
        // characteristic poly w packing: [3 - (0 + 5 * 0 + 4 * 1206 )], which is [-4821]
        println!("{:?}", dummy_decision_node_paths_mle);
        assert_eq!(next_layer_should_be, DenseMle::new(vec![Fr::from(-4821)]).mle_ref().bookkeeping_table);

    }

}
