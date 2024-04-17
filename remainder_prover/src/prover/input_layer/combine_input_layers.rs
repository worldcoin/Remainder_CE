use itertools::Itertools;
use remainder_ligero::{
    ligero_structs::LigeroEncoding, poseidon_ligero::PoseidonSpongeHasher, LcCommit,
    LcProofAuxiliaryInfo, LcRoot,
};
use remainder_shared_types::FieldExt;

use crate::{
    layer::LayerId,
    mle::{dense::DenseMle, Mle},
    utils::{argsort, pad_to_nearest_power_of_two},
};

use super::{ligero_input_layer::LigeroInputLayer, MleInputLayer};

/// Takes an MLE bookkeeping table interpreted as (big/little)-endian,
/// and converts it into a bookkeeping table interpreted as (little/big)-endian.
///
/// ## Arguments
/// * `bookkeeping_table` - Original MLE bookkeeping table
///
/// ## Returns
/// * `opposite_endian_bookkeeping_table` - MLE bookkeeping table, which, when
///     indexed (b_n, ..., b_1) rather than (b_1, ..., b_n), yields the same
///     result.
fn invert_mle_bookkeeping_table<F: FieldExt>(bookkeeping_table: Vec<F>) -> Vec<F> {
    // --- This should only happen the first time!!! ---
    let padded_bookkeeping_table = pad_to_nearest_power_of_two(bookkeeping_table);

    // --- 2 or fewer elements: No-op ---
    if padded_bookkeeping_table.len() <= 2 {
        return padded_bookkeeping_table;
    }

    // --- Grab the table by pairs, and create iterators over each half ---
    let tuples: (Vec<F>, Vec<F>) = padded_bookkeeping_table
        .chunks(2)
        .map(|pair| (pair[0], pair[1]))
        .unzip();

    // --- Recursively flip each half ---
    let inverted_first_half = invert_mle_bookkeeping_table(tuples.0);
    let inverted_second_half = invert_mle_bookkeeping_table(tuples.1);

    // --- Return the concatenation of the two ---
    inverted_first_half
        .into_iter()
        .chain(inverted_second_half.into_iter())
        .collect()
}

/// An interface for defining the set of MLEs you want to combine into a single InputLayer.
pub struct InputLayerBuilder<F> {
    /// The mles you wish to combine.
    mles: Vec<Box<dyn Mle<F>>>,
    /// The ID associated with this layer.
    layer_id: LayerId,
}

impl<F: FieldExt> InputLayerBuilder<F> {
    /// Creates a new InputLayerBuilder that will yield an InputLayer from many MLEs.
    pub fn new(input_mles: Vec<Box<&mut (dyn Mle<F> + 'static)>>, layer_id: LayerId) -> Self {
        let input_mles = input_mles
            .into_iter()
            .map(|mle| {
                let mle_deref = *mle;
                assert_eq!(mle_deref.layer_id(), layer_id);
                dyn_clone::clone_box(mle_deref)
            })
            .collect_vec();
        Self {
            mles: input_mles,
            layer_id,
        }
    }

    /// Combines the list of input MLEs in the input layer into one giant MLE by interleaving them
    /// assuming that the indices of the bookkeeping table are stored in little endian.
    fn combine_input_mles(&self) -> DenseMle<F, F> {
        let input_mles = &self.mles;
        let mle_combine_indices = argsort(
            &input_mles
                .iter()
                .map(|mle| mle.num_iterated_vars())
                .collect_vec(),
            true,
        );

        let final_bookkeeping_table = mle_combine_indices.into_iter().fold(
            vec![],
            |current_bookkeeping_table, input_mle_idx| {
                // --- Grab from the list of input MLEs OR the input-output MLE if the index calls for it ---
                let input_mle = &input_mles[input_mle_idx];

                // --- Basically, everything is stored in big-endian (including bookkeeping tables ---
                // --- and indices), BUT the indexing functions all happen as if we're interpreting ---
                // --- the indices as little-endian. Therefore we need to merge the input MLEs via ---
                // --- interleaving, or alternatively by converting everything to "big-endian", ---
                // --- merging the usual big-endian way, and re-converting the merged version back to ---
                // --- "little-endian" ---
                let inverted_input_mle =
                    invert_mle_bookkeeping_table(input_mle.get_padded_evaluations());

                // --- Fold the new (padded) bookkeeping table with the old ones ---
                // let padded_bookkeeping_table = input_mle.get_padded_evaluations();
                current_bookkeeping_table
                    .into_iter()
                    .chain(inverted_input_mle.into_iter())
                    .collect()
            },
        );

        // --- Convert the final bookkeeping table back to "little-endian" ---
        let re_inverted_final_bookkeeping_table =
            invert_mle_bookkeeping_table(final_bookkeeping_table);
        DenseMle::new_from_raw(re_inverted_final_bookkeeping_table, self.layer_id, None)
    }

    /// Turn this builder into an input layer.
    pub fn to_input_layer<I: MleInputLayer<F>>(self) -> I {
        let final_mle: DenseMle<F, F> = self.combine_input_mles();
        I::new(final_mle, self.layer_id)
    }

    /// Turn the builder into an input layer WITH a pre-commitment.
    pub fn to_input_layer_with_precommit(
        self,
        ligero_comm: LcCommit<PoseidonSpongeHasher<F>, LigeroEncoding<F>, F>,
        ligero_aux: LcProofAuxiliaryInfo,
        ligero_root: LcRoot<LigeroEncoding<F>, F>,
        verifier_is_precommit: bool,
    ) -> LigeroInputLayer<F> {
        let final_mle: DenseMle<F, F> = self.combine_input_mles();
        LigeroInputLayer::<F>::new_with_ligero_commitment(
            final_mle,
            self.layer_id,
            ligero_comm,
            ligero_aux,
            ligero_root,
            verifier_is_precommit,
        )
    }

    /// Turn the builder into input layer with rho inv specified.
    pub fn to_input_layer_with_rho_inv(self, rho_inv: u8, ratio: f64) -> LigeroInputLayer<F> {
        let final_mle: DenseMle<F, F> = self.combine_input_mles();
        LigeroInputLayer::<F>::new_with_rho_inv_ratio(final_mle, self.layer_id, rho_inv, ratio)
    }
}
