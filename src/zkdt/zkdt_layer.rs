use std::slice::Chunks;

use ark_bn254::Fr;

use crate::expression::{ExpressionStandard, Expression};
use crate::layer::{Layer, LayerBuilder, LayerId};
use crate::mle::dense::{DenseMle, DenseMleRef, Tuple2};
use crate::mle::{zero::ZeroMleRef, Mle, MleIndex, MleRef};
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


   //
    // Returns an expression that checks if the bits are binary.
    fn build_expression(&self) -> ExpressionStandard<F> {
        let decomp_bit_mle = self.mle.mle_bit_refs();
        let mut expressions =
         decomp_bit_mle.into_iter().map(|bit|  {
            let b = ExpressionStandard::Mle(bit.clone());
            let b_squared = ExpressionStandard::Product(vec![bit.clone(), bit.clone()]);
            b - b_squared
         }
        ).collect::<Vec<ExpressionStandard<F>>>();
         
        let chunk_and_concat = |expr: &[ExpressionStandard<F>]| {
           let chunks = expr.chunks(2);
           chunks.map(
               |chunk| {
                   chunk[0].clone().concat(chunk[1].clone())
               }
           ).collect::<Vec<ExpressionStandard<F>>>()
        };
     
        let expr1 = chunk_and_concat(&expressions);

        let expr2 = chunk_and_concat(&expr1);

        let expr3 = chunk_and_concat(&expr2);

        let expr4 = chunk_and_concat(&expr3);

        return expr4[0].clone().concat(expr4[1].clone());
    }



    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        ZeroMleRef::new(self.mle.num_vars() + 4, prefix_bits, id)
    }
}


#[cfg(test)]
mod tests { 
    use super::*;
    use crate::expression::ExpressionStandard;
    use crate::layer::LayerBuilder;
    use crate::mle::dense::DenseMle;
    use crate::mle::Mle;
    use crate::FieldExt;
    use ark_bn254::Fr;
    use ark_ff::Zero;

    #[test]
    fn test_product_tree_builder() {
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


    }

    #[test]
    fn test_binary_decomp_builder() {
 
    }
}
        