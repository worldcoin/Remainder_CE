use crate::FieldExt;
use crate::layer::{LayerBuilder, LayerId};
use crate::expression::{Expression, ExpressionStandard }; 
use crate::mle::{MleIndex, Mle};
use crate::mle::dense::{DenseMle, Tuple2};
struct ZKDTLayer<F:FieldExt> {
    id: LayerId,
    mle: DenseMle<F, Tuple2<F>>,
    expression: ExpressionStandard<F>,
} 

impl<F: FieldExt> LayerBuilder<F> for ZKDTLayer<F> { 
    type Successor = ZKDTLayer<F>;
    fn build_expression(&self) -> ExpressionStandard<F> {
        // Defines a build_expression function that multiplies the parts of the tuple pair-wise
        return ExpressionStandard::products(vec![self.mle.first(), self.mle.second()]);
    }
    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        // Create flatmle from tuple mle
        let flatMle: DenseMle<F, F> = self.mle.into_iter().map( |(first, second)| first * second).collect();
        flatMle.add_prefix_bits(&prefix_bits.unwrap());
        return ZKDTLayer {
            id: id,
            mle: flatMle,
            expression: self.build_expression(),
        };
        
    }
}





