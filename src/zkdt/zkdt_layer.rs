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

    // Returns an expression that checks if the bits are binary.
    fn build_expression(&self) -> ExpressionStandard<F> {
        let decomp_bit_mle = self.mle.mle_bit_refs();

        // Split the list of expressions into 8 chunks
        let chunks = decomp_bit_mle.chunks(2);

        // Concatenate the two elements of each chunk and reduce it 4 times

        let final_expression = chunks.map(
            |chunk| {
                let first_b = ExpressionStandard::Mle(chunk[0].clone());
                let second_b = ExpressionStandard::Mle(chunk[1].clone());
                let first_squared = ExpressionStandard::Product(vec![chunk[0].clone(), chunk[0].clone()]);
                let second_squared = ExpressionStandard::Product(vec![chunk[1].clone(), chunk[1].clone()]);
                (first_b - first_squared).concat(second_b - second_squared)
            }
        ).reduce(|a, b| a.concat(b)).unwrap();

        final_expression
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        ZeroMleRef::new(self.mle.num_vars() + 4, prefix_bits, id)
    }
}
