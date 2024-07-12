use remainder_shared_types::{layer::LayerId, FieldExt};

use crate::{
    layouter::nodes::{
        circuit_inputs::{InputLayerNode, InputShred},
        Context,
    },
    mle::Mle,
};

use super::{
    dense::DenseMle,
    evals::{Evaluations, MultilinearExtension},
    MleIndex,
};

use itertools::Itertools;

/// helper function for converting Field Elements into FlatMles
/// essentially
pub fn to_flat_mles<F: FieldExt, const N: usize>(inputs: Vec<[F; N]>) -> [Vec<F>; N] {
    // converts a [Vec<F; N>] into a [Vec<F>; N]
    let out: [Vec<F>; N] = {
        let mut iters = inputs.into_iter().map(|arr| arr.into_iter());
        std::array::from_fn(|_| {
            iters
                .by_ref()
                .map(|mut iter| iter.next().unwrap())
                .collect()
        })
    };
    out
}

/// A trait for a MLE(s) that are the input(s) to a circuit,
/// but are bundled together for semantic reasons.
pub trait CircuitMle<F: FieldExt, const N: usize> {
    /// returns the references to all the underlying MLEs
    fn get_mle_refs(&self) -> &[DenseMle<F>; N];

    /// returns all the MLEs as InputShreds
    fn make_input_shreds(
        &self,
        ctx: &Context,
        source: Option<&InputLayerNode<F>>,
    ) -> [InputShred<F>; N];
}

/// A struct that bundles N MLEs together for semantic reasons.
#[derive(Debug, Clone)]
pub struct FlatMles<F: FieldExt, const N: usize> {
    mles: [DenseMle<F>; N],
}

impl<F: FieldExt, const N: usize> CircuitMle<F, N> for FlatMles<F, N> {
    fn get_mle_refs(&self) -> &[DenseMle<F>; N] {
        &self.mles
    }

    fn make_input_shreds(
        &self,
        ctx: &Context,
        source: Option<&InputLayerNode<F>>,
    ) -> [InputShred<F>; N] {
        self.mles
            .clone()
            .into_iter()
            .map(|mle| {
                // this part needs changes
                let mle: MultilinearExtension<F> = MultilinearExtension::new_from_evals(
                    Evaluations::<F>::new(mle.num_iterated_vars(), mle.get_padded_evaluations()),
                );
                InputShred::new(ctx, mle, source)
            })
            .collect_vec()
            .try_into()
            .unwrap()
    }
}

impl<F: FieldExt, const N: usize> FlatMles<F, N> {
    /// Creates a new [FlatMles] from raw data.
    pub fn new_from_raw(
        data: [Vec<F>; N],
        layer_id: LayerId,
        prefix_bits: Option<Vec<MleIndex<F>>>,
    ) -> Self {
        let mles = data
            .into_iter()
            .map(|data| {
                let mut out = DenseMle::new_from_raw(data, layer_id);
                if let Some(prefix_bits) = prefix_bits.clone() {
                    out.add_prefix_bits(prefix_bits);
                }
                out
            })
            .collect_vec()
            .try_into()
            .unwrap();

        Self { mles }
    }
}

#[cfg(test)]
mod tests {

    use remainder_shared_types::layer::LayerId;
    use remainder_shared_types::Fr;

    use crate::mle::circuit_mle::CircuitMle;
    use crate::mle::circuit_mle::FlatMles;
    use crate::mle::Mle;

    #[test]
    fn create_dense_tuple_mle_from_vec() {
        let tuple_vec = [
            vec![Fr::from(0), Fr::from(2), Fr::from(4), Fr::from(6)],
            vec![Fr::from(1), Fr::from(3), Fr::from(5), Fr::from(7)],
        ];

        let tuple2_mle =
            FlatMles::<Fr, 2>::new_from_raw(tuple_vec.clone(), LayerId::Input(0), None);

        let mles = tuple2_mle.get_mle_refs();

        assert!(
            mles[0].get_padded_evaluations() == tuple_vec[0]
                && mles[1].get_padded_evaluations() == tuple_vec[1]
        );
    }
}
