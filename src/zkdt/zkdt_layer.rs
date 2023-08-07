use ark_ff::Field;
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

/// Takes x, outputs tuple [ r-x, b_ij*(r-x) + (1-b_ij) ]
/// which is [repeated squares, continued exponetiations]
/// first step in exponantiation
struct ExpoBuilderInit<F: FieldExt> {
    packed_x: DenseMle<F, F>,
    bin_decomp: BinDecomp16Bit<F>,
    bit_index: usize,
    r: F,
}

impl<F: FieldExt> LayerBuilder<F> for ExpoBuilderInit<F> {
    type Successor = (DenseMle<F, F>, DenseMle<F, F>);

    fn build_expression(&self) -> ExpressionStandard<F> {
        let expression_1 = ExpressionStandard::Constant(self.r) - 
                                                    (ExpressionStandard::Mle(self.packed_x.mle_ref()));

        let curr_bit = self.bin_decomp.bits[self.bit_index];

        let expression_2 = ExpressionStandard::Scaled(Box::new(expression_1.clone()), curr_bit) + 
                                                    ExpressionStandard::Constant(F::one()) - ExpressionStandard::Constant(curr_bit);

        expression_1.concat(expression_2)
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        let mut r_minus_x: DenseMle<F, F> = self.packed_x.clone().into_iter().map(
            |x| self.r - x
        ).collect();
        r_minus_x.add_prefix_bits(prefix_bits.clone()); 
        r_minus_x.define_layer_id(id.clone());

        let b_ij = self.bin_decomp.bits[self.bit_index];

        let mut prev_prod: DenseMle<F, F> = r_minus_x.clone().into_iter().map(
            |r_minus_x| b_ij * r_minus_x + (F::one() - b_ij)
        ).collect();
        prev_prod.add_prefix_bits(prefix_bits); 
        prev_prod.define_layer_id(id);

        (r_minus_x, prev_prod)
    }
}

/// Takes x, outputs x^2
/// used in repeated squaring in exponantiation
struct SquaringBuilder<F: FieldExt> {
    mle: DenseMle<F, F>,
}

impl<F: FieldExt> LayerBuilder<F> for SquaringBuilder<F> {
    type Successor = DenseMle<F, F>;
    fn build_expression(&self) -> ExpressionStandard<F> {
        ExpressionStandard::products(vec![self.mle.mle_ref(), self.mle.mle_ref()])
    }
    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        let mut squared: DenseMle<F, F> = self.mle
            .clone()
            .into_iter()
            .map(|x| x * x)
            .collect();
        squared.add_prefix_bits(prefix_bits);
        squared.define_layer_id(id);
        squared
    }
}

struct ExpoBuilderBitExpo<F: FieldExt> {
    bin_decomp: BinDecomp16Bit<F>,
    bit_index: usize,
    squared: DenseMle<F, F>,
}

/// Takes squared (r-x_i)^j, outputs b_ij * (r-x_i)^k + (1-b_ij)
impl<F: FieldExt> LayerBuilder<F> for ExpoBuilderBitExpo<F> {
    type Successor = DenseMle<F, F>;
    fn build_expression(&self) -> ExpressionStandard<F> {
        let b_ij = self.bin_decomp.bits[self.bit_index];
        ExpressionStandard::Sum(Box::new(ExpressionStandard::Scaled(Box::new(ExpressionStandard::Mle(self.squared.mle_ref())),
                                                                     b_ij)),
                                Box::new(ExpressionStandard::Constant(F::one()) - ExpressionStandard::Constant(b_ij)))
    }
    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        let b_ij = self.bin_decomp.bits[self.bit_index];
        let mut bit_expoed: DenseMle<F, F> = self.squared
            .clone()
            .into_iter()
            .map(
            |squared| (b_ij * squared + (F::one() - b_ij))
        ).collect();

        bit_expoed.add_prefix_bits(prefix_bits);
        bit_expoed.define_layer_id(id);
        bit_expoed
    }
}

/// Takes (1) b_ij * (r-x_i)^j + (1-b_ij), (2) prev_prods PROD(b_ij * (r-x_i)^j + (1-b_ij)) across j
/// Outputs (1) * (2). naming (1) as multiplier
struct ExpoBuilderProduct<F: FieldExt> {
    multiplier: DenseMle<F, F>,
    prev_prod: DenseMle<F, F>,
}

impl<F: FieldExt> LayerBuilder<F> for ExpoBuilderProduct<F> {
    type Successor = DenseMle<F, F>;

    fn build_expression(&self) -> ExpressionStandard<F> {
        ExpressionStandard::products(vec![self.multiplier.mle_ref(), self.prev_prod.mle_ref()])
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {

        let mut prod: DenseMle<F, F> = self.prev_prod
            .clone()
            .into_iter()
            .zip(self.multiplier.clone().into_iter())
            .map(
            |(prev_prod, multiplier)| prev_prod * multiplier
        ).collect();

        prod.add_prefix_bits(prefix_bits);
        prod.define_layer_id(id);
        prod
        
    }
}

struct LeafNodePackingBuilder<F: FieldExt> {
    mle: DenseMle<F, LeafNode<F>>,
    r: F,
    r_packing: F
}

impl<F: FieldExt> LayerBuilder<F> for LeafNodePackingBuilder<F> {
    type Successor = DenseMle<F, F>;

    // expressions = r - (x.node_id + r_packing * x.node_val)
    fn build_expression(&self) -> ExpressionStandard<F> {
        ExpressionStandard::Constant(self.r) - (ExpressionStandard::Mle(self.mle.node_id()) + 
        ExpressionStandard::Scaled(Box::new(ExpressionStandard::Mle(self.mle.node_val())), self.r_packing))
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        let mut flat_mle:DenseMle<F, F> = self.mle.into_iter().map(
            |(node_id, node_val)| 
            self.r - (node_id + self.r_packing * node_val)
        ).collect();
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

    #[test]
    fn test_leaf_node_packing_builder() {

        let (_,_, _, dummy_leaf_node_paths_mle, _, _, _, _) = generate_dummy_mles::<Fr>();

        let (r, r_packing) = (Fr::from(3), Fr::from(5));
        let input_packing_builder = LeafNodePackingBuilder{
                                                                                            mle: dummy_leaf_node_paths_mle.clone(),
                                                                                            r,
                                                                                            r_packing
                                                                                        };
        let input_packed_expression = input_packing_builder.build_expression();
        println!("layer expression: {:?}", input_packed_expression);

        let next_layer = input_packing_builder.next_layer(LayerId::Layer(0), None);
        let next_layer_should_be = dummy_leaf_node_paths_mle.node_id().bookkeeping_table
                            .clone().iter()
                            .zip(dummy_leaf_node_paths_mle.node_val().bookkeeping_table.clone().iter())
                            .map(|(a, b)| {r - (a + &(r_packing * b))})
                            .collect_vec();
       
        assert_eq!(next_layer.mle_ref().bookkeeping_table, next_layer_should_be);
        println!("layer mle: {:?}", next_layer.mle_ref().bookkeeping_table);

        // hand compute
        // for this to pass, change the parameters into the following:
        // const DUMMY_INPUT_LEN: usize = 1 << 1;
        // const TREE_HEIGHT: usize = 2;
        // the node_id: [2], its node_val is: 17299145535799709783. r = 3, r_packing = 5
        // characteristic poly w packing: [3 - (2 + 5 * 17299145535799709783)], which is [-4821]

        println!("{:?}", dummy_leaf_node_paths_mle);
        assert_eq!(next_layer_should_be, DenseMle::new(vec![ Fr::from(3) - (Fr::from(2) + Fr::from(5) * Fr::from(17299145535799709783 as u64))]).mle_ref().bookkeeping_table);

    }

}
