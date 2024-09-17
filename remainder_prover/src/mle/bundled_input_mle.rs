use remainder_shared_types::Field;

use crate::{
    layer::LayerId,
    layouter::nodes::{
        circuit_inputs::{InputLayerNode, InputShred, InputShredData},
        CircuitNode, Context,
    },
    mle::Mle,
};

use super::{
    dense::DenseMle,
    evals::{Evaluations, MultilinearExtension},
    MleIndex,
};

use itertools::Itertools;

/// Helper function that converts a [Vec<F; N>] into a [Vec<F>; N]
pub fn to_flat_mles<F: Field, const N: usize>(inputs: Vec<[F; N]>) -> [Vec<F>; N] {
    // converts a [Vec<F; N>] into a [Vec<F>; N]
    let mut result: [Vec<F>; N] = vec![Vec::new(); N].try_into().unwrap();
    for subarray in inputs {
        for (i, element) in subarray.iter().enumerate() {
            result[i].push(*element);
        }
    }
    result
}

// Test that demonstrates the incorrectness of the previous implementation of to_flat_mles.
#[test]
fn test_to_flat_mles() {
    use remainder_shared_types::Fr;
    let inputs = vec![
        [Fr::from(1), Fr::from(2), Fr::from(3)],
        [Fr::from(4), Fr::from(5), Fr::from(6)],
    ];
    let expected = [
        vec![Fr::from(1), Fr::from(4)],
        vec![Fr::from(2), Fr::from(5)],
        vec![Fr::from(3), Fr::from(6)],
    ];
    let actual = to_flat_mles(inputs);
    assert_eq!(actual, expected);
}

/// A trait for a MLE(s) that are the input(s) to a circuit,
/// but are bundled together for semantic reasons.
pub trait BundledInputMle<F: Field, const N: usize> {
    /// returns the references to all the underlying MLEs
    fn get_mle_refs(&self) -> &[DenseMle<F>; N];

    /// returns all the MLEs as InputShreds
    fn make_input_shreds(
        &self,
        ctx: &Context,
        source: &InputLayerNode,
    ) -> (Vec<InputShred>, Vec<InputShredData<F>>);
}

/// A struct that bundles N MLEs together for semantic reasons.
#[derive(Debug, Clone)]
pub struct FlatMles<F: Field, const N: usize> {
    mles: [DenseMle<F>; N],
}

impl<F: Field, const N: usize> BundledInputMle<F, N> for FlatMles<F, N> {
    fn get_mle_refs(&self) -> &[DenseMle<F>; N] {
        &self.mles
    }

    fn make_input_shreds(
        &self,
        ctx: &Context,
        source: &InputLayerNode,
    ) -> (Vec<InputShred>, Vec<InputShredData<F>>) {
        self.mles
            .clone()
            .into_iter()
            .map(|mle| {
                // this part needs changes
                let mle: MultilinearExtension<F> = MultilinearExtension::new_from_evals(
                    Evaluations::<F>::new(mle.num_iterated_vars(), mle.get_padded_evaluations()),
                );
                let input_shred = InputShred::new(ctx, mle.num_vars(), source);
                let input_shred_data = InputShredData::new(input_shred.id(), mle);
                (input_shred, input_shred_data)
            })
            .unzip()
            .try_into()
            .unwrap()
    }
}

impl<F: Field, const N: usize> FlatMles<F, N> {
    /// Creates a new [FlatMles] from raw data.
    pub fn new_from_raw(data: [Vec<F>; N], layer_id: LayerId) -> Self {
        let mles = data
            .into_iter()
            .map(|data| DenseMle::new_from_raw(data, layer_id))
            .collect_vec()
            .try_into()
            .unwrap();

        Self { mles }
    }
}

#[cfg(test)]
mod tests {

    use crate::layer::LayerId;
    use remainder_shared_types::Fr;

    use crate::mle::bundled_input_mle::BundledInputMle;
    use crate::mle::bundled_input_mle::FlatMles;
    use crate::mle::Mle;

    #[test]
    fn create_dense_tuple_mle_from_vec() {
        let tuple_vec = [
            vec![Fr::from(0), Fr::from(2), Fr::from(4), Fr::from(6)],
            vec![Fr::from(1), Fr::from(3), Fr::from(5), Fr::from(7)],
        ];

        let tuple2_mle = FlatMles::<Fr, 2>::new_from_raw(tuple_vec.clone(), LayerId::Input(0));

        let mles = tuple2_mle.get_mle_refs();

        assert!(
            mles[0].get_padded_evaluations() == tuple_vec[0]
                && mles[1].get_padded_evaluations() == tuple_vec[1]
        );
    }
}
