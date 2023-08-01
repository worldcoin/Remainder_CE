use ark_bn254::Fr;

use crate::FieldExt;
use crate::layer::{LayerBuilder, LayerId, Layer};
use crate::expression::{ExpressionStandard }; 
use crate::mle::{MleIndex, zero::ZeroMleRef, Mle, MleRef};
use crate::mle::dense::{DenseMle, Tuple2, DenseMleRef};

use super::structs::BinDecomp16Bit;


struct ProductTreeBuilder<F:FieldExt> {
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
        let mut flat_mle: DenseMle<F, F> = self.mle.into_iter().map( |(first, second)| first * second).collect();
        flat_mle.add_prefix_bits(prefix_bits);
        flat_mle.define_layer_id(id);
        flat_mle
    }
}


struct BinaryDecompBuilder<F:FieldExt> {
    mle: DenseMle<F, BinDecomp16Bit<F>>,
}

impl<F: FieldExt> LayerBuilder<F> for BinaryDecompBuilder<F> {
    
        type Successor = ZeroMleRef<F>;
        
        // Returns an expression that checks if the bits are binary. 
        fn build_expression(&self) -> ExpressionStandard<F> {
            let decomp_bit_mle = self.mle.mle_bit_refs();
            let b = ExpressionStandard::Mle(decomp_bit_mle[0].clone());

            let b_squared = ExpressionStandard::Product(vec![
                decomp_bit_mle[0].clone(),
                decomp_bit_mle[0].clone(),
            ]);
             
            // b * (1 - b) = b - b^2 
            b - b_squared
        }
    
        fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
            ZeroMleRef::new(self.mle.num_vars(), prefix_bits, id)
        }
}




