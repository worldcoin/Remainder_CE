use itertools::Itertools;

use crate::expression::{Expression, ExpressionStandard};
use crate::layer::{LayerBuilder, LayerId};
use crate::mle::dense::{DenseMle, Tuple2};
use crate::mle::{zero::ZeroMleRef, Mle, MleIndex};
use crate::FieldExt;

use super::structs::BinDecomp16Bit;

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
}
