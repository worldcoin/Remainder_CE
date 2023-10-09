use std::{
    cmp,
    iter::{Cloned, Map, Zip},
};

use crate::{
    mle::{
        dense::{get_padded_evaluations_for_list, DenseMle, DenseMleRef},
        Mle, MleAble, MleIndex, MleRef,
    }, layer::{batched::combine_mles, LayerId},
};
use remainder_shared_types::FieldExt;
use ark_std::log2;
use itertools::{repeat_n, Itertools};
use serde::{Serialize, Deserialize};

// ------------------------------------ ACTUAL DATA STRUCTS ------------------------------------

/// --- Path nodes within the tree and in the path hint ---
#[derive(Copy, Debug, Clone, Serialize, Deserialize)]
pub struct DecisionNode<F> {
    ///The id of this node in the tree
    pub(crate) node_id: F,
    ///The id of the attribute this node involves
    pub(crate) attr_id: F,
    ///The treshold of this node
    pub(crate) threshold: F,
}

#[derive(Copy, Debug, Clone, Serialize, Deserialize)]
///The Leafs of the tree
pub struct LeafNode<F> {
    ///The id of this leaf in the tree
    pub(crate) node_id: F,
    ///The value of this leaf
    pub(crate) node_val: F,
}

/// --- 16-bit binary decomposition ---
/// Used for the following components of the (circuit) input:
/// a) The binary decomposition of the path node hints (i.e. x.val - path_x.thr)
/// b) The binary decomposition of the multiplicity coefficients $c_j$
#[derive(Copy, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinDecomp16Bit<F> {
    ///The 16 bits that make up this decomposition
    ///
    /// Should all be 1 or 0
    pub bits: [F; 16],
}

#[derive(Copy, Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Used for the attribute multiplicities
pub struct BinDecomp4Bit<F> {
    ///The 4 bits that make up this decomposition
    ///
    /// Should all be 1 or 0
    pub bits: [F; 4],
}

/// --- Input element to the tree, i.e. a list of input attributes ---
/// Used for the following components of the (circuit) input:
/// a) The actual input attributes, i.e. x
/// b) The permuted input attributes, i.e. \bar{x}
#[derive(Copy, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InputAttribute<F> {
    // pub attr_idx: F,
    ///The attr id of this input
    pub attr_id: F,
    ///The threshold value of this input
    pub attr_val: F,
}

// ------------------------------------ MLEABLE IMPLEMENTATIONS ------------------------------------

impl<F: FieldExt> MleAble<F> for DecisionNode<F> {
    type Repr = [Vec<F>; 3];

    fn get_padded_evaluations(items: &Self::Repr) -> Vec<F> {
        get_padded_evaluations_for_list(items)
    }

    type IntoIter<'a> = Map<Zip<Zip<Cloned<std::slice::Iter<'a, F>>, Cloned<std::slice::Iter<'a, F>>>, Cloned<std::slice::Iter<'a, F>>>, fn(((F, F), F)) -> DecisionNode<F>> where Self: 'a;

    fn from_iter(iter: impl IntoIterator<Item = Self>) -> Self::Repr {
        let iter = iter.into_iter();
        let (node_ids, attr_ids, thresholds): (Vec<F>, Vec<F>, Vec<F>) = iter
            .map(|x| (x.node_id, x.attr_id, x.threshold))
            .multiunzip();

        [node_ids, attr_ids, thresholds]
    }

    fn to_iter(items: &Self::Repr) -> Self::IntoIter<'_> {
        items[0]
            .iter()
            .cloned()
            .zip(items[1].iter().cloned())
            .zip(items[2].iter().cloned())
            .map(|((node_id, attr_id), threshold)| DecisionNode {
                node_id,
                attr_id,
                threshold,
            })
    }

    fn num_vars(items: &Self::Repr) -> usize {
        log2(items[0].len() + items[1].len() + items[2].len()) as usize
    }
}


impl<F: FieldExt> From<Vec<bool>> for BinDecomp16Bit<F> {
    fn from(bits: Vec<bool>) -> Self {
        BinDecomp16Bit::<F> {
            bits: bits.iter().map(|x| F::from(*x)).collect::<Vec<F>>().try_into().unwrap()
        }
    }
}

impl<F: FieldExt> From<Vec<bool>> for BinDecomp4Bit<F> {
    fn from(bits: Vec<bool>) -> Self {
        BinDecomp4Bit::<F> {
            bits: bits.iter().map(|x| F::from(*x)).collect::<Vec<F>>().try_into().unwrap()
        }
    }
}

/// for input layer stuff, combining refs together
/// TODO!(ende): refactor
/// Takes the individual bookkeeping tables from the MleRefs within an MLE
/// and merges them with padding, using a little-endian representation
/// merge strategy. Assumes that ALL MleRefs are the same size.
pub fn combine_mle_refs<F: FieldExt>(items: Vec<DenseMleRef<F>>) -> DenseMle<F, F> {

    let num_fields = items.len();

    // --- All the items within should be the same size ---
    let max_size = items
        .iter()
        .map(|mle_ref| mle_ref.bookkeeping_table.len())
        .max().unwrap();

    let part_size = 1 << log2(max_size);
    let part_count = 2_u32.pow(log2(num_fields)) as usize;

    // --- Number of "part" slots which need to filled with padding ---
    let padding_count = part_count - num_fields;
    let total_size = part_size * part_count;
    let total_padding: usize = total_size - max_size * part_count;

    let result = (0..max_size)
        .flat_map(|index| {
            items
                .iter()
                .map(move |item| *item.bookkeeping_table().get(index).unwrap_or(&F::zero()))
                .chain(repeat_n(F::zero(), padding_count))
        })
        .chain(repeat_n(F::zero(), total_padding))
        .collect_vec();

    DenseMle::new_from_raw(result, LayerId::Input(0), None)
}


impl<F: FieldExt> DenseMle<F, DecisionNode<F>> {
    /// MleRef grabbing just the list of node IDs
    pub(crate) fn node_id(&'_ self) -> DenseMleRef<F> {
        let num_vars = self.num_iterated_vars();

        // --- There are four components to this MLE ---
        let _len = self.mle.len() / 4;

        DenseMleRef {
            bookkeeping_table: self.mle[0].to_vec(),
            // --- [0, 0, b_1, ..., b_n] ---
            // TODO!(ryancao): Does this give us the endian-ness we want???
            mle_indices: self
                .prefix_bits
                .clone()
                .into_iter()
                .flatten()
                .chain(
                    // --- NOTE that prefix bits HAVE to be in little-endian ---
                    std::iter::once(MleIndex::Fixed(false))
                        .chain(std::iter::once(MleIndex::Fixed(false)))
                        .chain(repeat_n(MleIndex::Iterated, num_vars - 2)),
                )
                .collect_vec(),
            num_vars: num_vars - 2,
            layer_id: self.layer_id,
            indexed: false,
        }
    }

    /// MleRef grabbing just the list of attribute IDs
    pub(crate) fn attr_id(&'_ self) -> DenseMleRef<F> {
        let num_vars = self.num_iterated_vars();

        // --- There are four components to this MLE ---
        let _len = self.mle.len() / 4;

        DenseMleRef {
            bookkeeping_table: self.mle[1].to_vec(),
            // --- [0, 1, b_1, ..., b_n] ---
            // TODO!(ryancao): Does this give us the endian-ness we want???
            mle_indices: self
                .prefix_bits
                .clone()
                .into_iter()
                .flatten()
                .chain(
                    // --- NOTE that prefix bits HAVE to be in little-endian ---
                    std::iter::once(MleIndex::Fixed(true))
                        .chain(std::iter::once(MleIndex::Fixed(false)))
                        .chain(repeat_n(MleIndex::Iterated, num_vars - 2)),
                )
                .collect_vec(),
            num_vars: num_vars - 2,
            layer_id: self.layer_id,
            indexed: false,
        }
    }

    /// MleRef grabbing just the list of thresholds
    pub(crate) fn threshold(&'_ self) -> DenseMleRef<F> {
        let num_vars = self.num_iterated_vars();

        // --- There are four components to this MLE ---
        // let len = self.mle.len() / 4;

        DenseMleRef {
            bookkeeping_table: self.mle[2].to_vec(),
            // --- [1, 0, b_1, ..., b_n] ---
            // TODO!(ryancao): Does this give us the endian-ness we want???
            // Answer: Lol not originally. The above comment is for ease of readability in big-endian
            mle_indices: self
                .prefix_bits
                .clone()
                .into_iter()
                .flatten()
                .chain(
                    // --- NOTE that prefix bits HAVE to be in little-endian ---
                    std::iter::once(MleIndex::Fixed(false))
                        .chain(std::iter::once(MleIndex::Fixed(true)))
                        .chain(repeat_n(MleIndex::Iterated, num_vars - 2)),
                )
                .collect_vec(),
            num_vars: num_vars - 2,
            layer_id: self.layer_id,
            indexed: false,
        }
    }

    /// Given a batch of `DenseMle<F, InputAttribute<F>>`, creates a single combined
    /// MLE bookkeeping table which represents first interleaving by node ID, attribute ID, and threshold,
    /// then interleaving by batched MLEs (should always be a power of two!)
    /// 
    /// Note that we interleave rather than stack since the indices are represented in little-endian.
    /// TODO!(ende): refactor
    pub fn combine_mle_batch(decision_mle_batch: Vec<DenseMle<F, DecisionNode<F>>>) -> DenseMle<F, F> {
        
        let batched_bits = log2(decision_mle_batch.len());

        let decision_mle_batch_ref_combined = decision_mle_batch
            
            .into_iter().map(
                |x| {
                    combine_mle_refs(
                        vec![x.node_id(), x.attr_id(), x.threshold()]
                    ).mle_ref()
                }
            ).collect_vec();

        let decision_mle_batch_ref_combined_ref =  combine_mles(decision_mle_batch_ref_combined, batched_bits as usize);

        DenseMle::new_from_raw(decision_mle_batch_ref_combined_ref.bookkeeping_table, LayerId::Input(0), None)
    }

}

// --- Leaf node ---
impl<F: FieldExt> MleAble<F> for LeafNode<F> {
    type Repr = [Vec<F>; 2];

    fn get_padded_evaluations(items: &Self::Repr) -> Vec<F> {
        get_padded_evaluations_for_list(items)
    }

    type IntoIter<'a> = Map<Zip<Cloned<std::slice::Iter<'a, F>>, Cloned<std::slice::Iter<'a, F>>>, fn((F, F)) -> Self> where Self: 'a;

    fn from_iter(iter: impl IntoIterator<Item = Self>) -> Self::Repr {
        let iter = iter.into_iter();
        let (node_ids, node_vals): (Vec<F>, Vec<F>) = iter.map(|x| (x.node_id, x.node_val)).unzip();
        [node_ids, node_vals]
    }

    fn to_iter(items: &Self::Repr) -> Self::IntoIter<'_> {
        items[0]
            .iter()
            .cloned()
            .zip(items[1].iter().cloned())
            .map(|(node_id, node_val)| LeafNode { node_id, node_val })
    }

    fn num_vars(items: &Self::Repr) -> usize {
        log2(items[0].len() + items[1].len()) as usize
    }
}

impl<F: FieldExt> DenseMle<F, LeafNode<F>> {
    /// MleRef grabbing just the list of node IDs
    pub(crate) fn node_id(&'_ self) -> DenseMleRef<F> {
        let num_vars = self.num_iterated_vars();

        // --- There are four components to this MLE ---
        let _len = self.mle.len() / 2;

        DenseMleRef {
            bookkeeping_table: self.mle[0].to_vec(),
            // --- [0, b_1, ..., b_n] ---
            // TODO!(ryancao): Does this give us the endian-ness we want???
            mle_indices: self
                .prefix_bits
                .clone()
                .into_iter()
                .flatten()
                .chain(
                    std::iter::once(MleIndex::Fixed(false))
                        .chain(repeat_n(MleIndex::Iterated, num_vars - 1)),
                )
                .collect_vec(),
            num_vars: num_vars - 1,
            layer_id: self.layer_id,
            indexed: false,
        }
    }

    /// MleRef grabbing just the list of attribute IDs
    pub(crate) fn node_val(&'_ self) -> DenseMleRef<F> {
        let num_vars = self.num_iterated_vars();

        DenseMleRef {
            bookkeeping_table: self.mle[1].to_vec(),
            // --- [1, b_1, ..., b_n] ---
            // TODO!(ryancao): Does this give us the endian-ness we want???
            mle_indices: self
                .prefix_bits
                .clone()
                .into_iter()
                .flatten()
                .chain(
                    std::iter::once(MleIndex::Fixed(true))
                        .chain(repeat_n(MleIndex::Iterated, num_vars - 1)),
                )
                .collect_vec(),
            num_vars: num_vars - 1,
            layer_id: self.layer_id,
            indexed: false,
        }
    }

    /// Given a batch of `DenseMle<F, InputAttribute<F>>`, creates a single combined
    /// MLE bookkeeping table which represents first interleaving by node ID and val,
    /// then interleaving by batched MLEs (should always be a power of two!)
    /// 
    /// Note that we interleave rather than stack since the indices are represented in little-endian.
    /// TODO!(ende): refactor
    pub fn combine_mle_batch(leaf_mle_batch: Vec<DenseMle<F, LeafNode<F>>>) -> DenseMle<F, F> {
        
        let batched_bits = log2(leaf_mle_batch.len());

        let leaf_mle_batch_ref_combined = leaf_mle_batch
            
            .into_iter().map(
                |x| {
                    combine_mle_refs(
                        vec![x.node_id(), x.node_val()]
                    ).mle_ref()
                }
            ).collect_vec();

        let leaf_mle_batch_ref_combined_ref =  combine_mles(leaf_mle_batch_ref_combined, batched_bits as usize);

        DenseMle::new_from_raw(leaf_mle_batch_ref_combined_ref.bookkeeping_table, LayerId::Input(0), None)

    }
}

// --- Input attribute ---
impl<F: FieldExt> MleAble<F> for InputAttribute<F> {
    type Repr = [Vec<F>; 2];

    fn get_padded_evaluations(items: &Self::Repr) -> Vec<F> {
        get_padded_evaluations_for_list(items)
    }

    type IntoIter<'a> = Map<Zip<Cloned<std::slice::Iter<'a, F>>, Cloned<std::slice::Iter<'a, F>>>, fn((F, F)) -> Self> where Self: 'a;

    fn from_iter(iter: impl IntoIterator<Item = Self>) -> Self::Repr {
        let iter = iter.into_iter();
        let (attr_ids, attr_vals): (Vec<F>, Vec<F>) = iter.map(|x| (x.attr_id, x.attr_val)).unzip();
        [attr_ids, attr_vals]
    }

    fn to_iter(items: &Self::Repr) -> Self::IntoIter<'_> {
        items[0]
            .iter()
            .cloned()
            .zip(items[1].iter().cloned())
            .map(|(attr_id, attr_val)| InputAttribute { attr_id, attr_val })
    }

    fn num_vars(items: &Self::Repr) -> usize {
        log2(items[0].len() + items[1].len()) as usize
    }
}

impl<F: FieldExt> DenseMle<F, InputAttribute<F>> {
    /// MleRef grabbing just the list of attribute IDs
    pub(crate) fn attr_id(&'_ self, num_vars: Option<usize>) -> DenseMleRef<F> {

        // --- Default to the entire (component of) the MLE ---
        let num_vars = num_vars.unwrap_or(self.num_iterated_vars() - 1);

        // TODO!(ryancao): Make this actually do error-handling
        assert!(num_vars < self.num_iterated_vars());

        // --- The length of the MLERef is just 2^{num_vars} ---
        let len = 2_u32.pow(num_vars as u32) as usize;
        let concrete_len = cmp::min(len, self.mle[0].to_vec().len());

        DenseMleRef {
            bookkeeping_table: self.mle[0][0..concrete_len].to_vec(),
            // --- [0; 0, ..., 0; b_1, ..., b_n] ---
            // TODO!(ryancao): Does this give us the endian-ness we want???
            mle_indices: self
                .prefix_bits
                .clone()
                .into_iter()
                .flatten()
                .chain(
                    std::iter::once(MleIndex::Fixed(false))
                        .chain(repeat_n(MleIndex::Iterated, num_vars))
                        .chain(repeat_n(
                            MleIndex::Fixed(false),
                            self.num_iterated_vars() - 1 - num_vars,
                        )),
                    // repeat_n(MleIndex::Iterated, num_vars)
                    // .chain(repeat_n(
                    //             MleIndex::Fixed(false),
                    //             self.num_iterated_vars() - 1 - num_vars))
                    // .chain(std::iter::once(MleIndex::Fixed(false)))
                )
                .collect_vec(),
            num_vars,
            layer_id: self.layer_id,
            indexed: false,
        }
    }

    /// MleRef grabbing just the list of attribute values
    pub(crate) fn attr_val(&'_ self, num_vars: Option<usize>) -> DenseMleRef<F> {
        // --- Default to the entire (component of) the MLE ---
        let num_vars = match num_vars {
            Some(num) => num,
            None => self.num_iterated_vars() - 1,
        };

        // TODO!(ryancao): Make this actually do error-handling
        assert!(num_vars < self.num_iterated_vars());

        // --- The length of the MLERef is just 2^{num_vars} ---
        let len = 2_u32.pow(num_vars as u32) as usize;
        let concrete_len = cmp::min(len, self.mle[1].to_vec().len());

        DenseMleRef {
            bookkeeping_table: self.mle[1][..concrete_len].to_vec(),
            // --- [1; 0, ..., 0; b_1, ..., b_n] ---
            // Note that the zeros are there to prefix all the things we chunked out
            // TODO!(ryancao): Does this give us the endian-ness we want???
            mle_indices: self
                .prefix_bits
                .clone()
                .into_iter()
                .flatten()
                .chain(
                    std::iter::once(MleIndex::Fixed(true))
                        .chain(repeat_n(MleIndex::Iterated, num_vars))
                        .chain(repeat_n(
                            MleIndex::Fixed(false),
                            self.num_iterated_vars() - 1 - num_vars,
                        )),
                    // repeat_n(MleIndex::Iterated, num_vars)
                    // .chain(repeat_n(
                    //             MleIndex::Fixed(false),
                    //             self.num_iterated_vars() - 1 - num_vars))
                    // .chain(std::iter::once(MleIndex::Fixed(true)))
                )
                .collect_vec(),
            num_vars,
            layer_id: self.layer_id,
            indexed: false,
        }
    }

    /// Given a batch of `DenseMle<F, InputAttribute<F>>`, creates a single combined
    /// MLE bookkeeping table which represents first interleaving by attribute ID and attribute val,
    /// then interleaving by batched MLEs (should always be a power of two!)
    /// 
    /// Note that we interleave rather than stack since the indices are represented in little-endian.
    /// TODO!(ende): refactor
    pub fn combine_mle_batch(input_mle_batch: Vec<DenseMle<F, InputAttribute<F>>>) -> DenseMle<F, F> {
        
        let batched_bits = log2(input_mle_batch.len());

        let input_mle_batch_ref_combined = input_mle_batch
            
            .into_iter().map(
                |x| {
                    combine_mle_refs(
                        vec![x.attr_id(None), x.attr_val(None)]
                    ).mle_ref()
                }
            ).collect_vec();

        let input_mle_batch_ref_combined_ref =  combine_mles(input_mle_batch_ref_combined, batched_bits as usize);

        DenseMle::new_from_raw(input_mle_batch_ref_combined_ref.bookkeeping_table, LayerId::Input(0), None)

    }
}

// --- Bin decomp ---
impl<F: FieldExt> MleAble<F> for BinDecomp16Bit<F> {
    type Repr = [Vec<F>; 16];

    fn get_padded_evaluations(items: &Self::Repr) -> Vec<F> {
        get_padded_evaluations_for_list(items)
    }

    type IntoIter<'a> = std::vec::IntoIter<BinDecomp16Bit<F>> where Self: 'a;

    fn from_iter(iter: impl IntoIterator<Item = Self>) -> Self::Repr {
        let iter = iter.into_iter();

        // --- TODO!(ryancao): This is genuinely horrible but we'll fix it later ---
        let mut ret: [Vec<F>; 16] = std::array::from_fn(|_| vec![]);
        iter.for_each(|tuple| {
            for (item, bit) in ret.iter_mut().zip(tuple.bits.iter()) {
                item.push(*bit);
            }
        });

        ret
    }

    fn to_iter(items: &Self::Repr) -> Self::IntoIter<'_> {
        let elems = (0..items[0].len()).map(
            |idx| {
                let bits = items.iter().map(
                    |item| {
                        item[idx]
                    }
                ).collect_vec();
                BinDecomp16Bit {
                    bits: bits.try_into().unwrap(),
                }
            }
        ).collect_vec();

        elems.into_iter()
    }

    fn num_vars(items: &Self::Repr) -> usize {
        log2(16 * items[0].len()) as usize
    }
}

// TODO!(ryancao): Make this stuff derivable
impl<F: FieldExt> DenseMle<F, BinDecomp16Bit<F>> {
    /// Returns a list of MLERefs, one for each bit
    /// TODO!(ryancao): Change this back to [DenseMleRef<F>; 16] and make it work!
    pub(crate) fn mle_bit_refs(&'_ self) -> Vec<DenseMleRef<F>> {
        let num_vars = self.num_iterated_vars();

        // --- There are sixteen components to this MLE ---
        let mut ret: Vec<DenseMleRef<F>> = vec![];

        for bit_idx in 0..16 {
            // --- Prefix bits need to be *literally* represented in little-endian ---
            let first_prefix = (bit_idx % 2) >= 1;
            let second_prefix = (bit_idx % 4) >= 2;
            let third_prefix = (bit_idx % 8) >= 4;
            let fourth_prefix = (bit_idx % 16) >= 8;

            let bit_mle_ref = DenseMleRef {
                bookkeeping_table: self.mle[bit_idx].to_vec(),
                // --- [0, 0, 0, 0, b_1, ..., b_n] ---
                mle_indices: self
                    .prefix_bits
                    .clone()
                    .into_iter()
                    .flatten()
                    .chain(
                        std::iter::once(MleIndex::Fixed(first_prefix))
                            .chain(std::iter::once(MleIndex::Fixed(second_prefix)))
                            .chain(std::iter::once(MleIndex::Fixed(third_prefix)))
                            .chain(std::iter::once(MleIndex::Fixed(fourth_prefix)))
                            .chain(repeat_n(MleIndex::Iterated, num_vars - 4)),
                    )
                    .collect_vec(),
                num_vars: num_vars - 4,
                layer_id: self.layer_id,
                indexed: false,
            };
            ret.push(bit_mle_ref);
        }

        ret
    }

    /// Returns the entire bin decomp MLE as a single MLE ref
    pub(crate) fn get_entire_mle_as_mle_ref(&'_ self) -> DenseMleRef<F> {
        // --- Just need to merge all of the bin decomps in an interleaved fashion ---
        // TODO!(ryancao): This is an awful hacky fix so that we can use `combine_mles`.
        // Note that we are manually inserting the extra iterated bits as prefix bits.
        // We should stop doing this once `combine_mles` works as it should!
        let self_mle_ref_vec = self.mle.clone().map(|mle_bookkeeping_table| {
            DenseMle::new_from_raw(
                mle_bookkeeping_table, 
                self.layer_id, 
                Some(
                    self.get_prefix_bits().iter().flatten().cloned().chain(
                        repeat_n(MleIndex::Iterated, 4)
                    ).collect_vec()
                )
            ).mle_ref()
        }).to_vec();
        combine_mles(self_mle_ref_vec, 4)
    }

    /// Combines the bookkeeping tables of each of the MleRefs within a
    /// `DenseMle<F, BinDecomp16Bit<F>>` into a single interleaved bookkeeping
    /// table such that referring to the merged table using little-endian indexing
    /// bits, followed by the appropriate MleRef indexing bits, gets us the same
    /// result as only using the same MleRef indexing bits on each MleRef from
    /// the `DenseMle<F, BinDecomp16Bit<F>>`.
    /// 
    /// TODO!(ende): refactor
    pub(crate) fn combine_mle_batch(input_mle_batch: Vec<DenseMle<F, BinDecomp16Bit<F>>>) -> DenseMle<F, F> {
        
        let batched_bits = log2(input_mle_batch.len());

        let input_mle_batch_ref_combined = input_mle_batch
            
            .into_iter().map(
                |x| {
                    combine_mle_refs(
                        x.mle_bit_refs()
                    ).mle_ref()
                }
            ).collect_vec();

        let input_mle_batch_ref_combined_ref = combine_mles(input_mle_batch_ref_combined, batched_bits as usize);

        DenseMle::new_from_raw(input_mle_batch_ref_combined_ref.bookkeeping_table, LayerId::Input(0), None)
    }
}

// --- Bin decomp but 4 bits ---
impl<F: FieldExt> MleAble<F> for BinDecomp4Bit<F> {
    type Repr = [Vec<F>; 4];

    fn get_padded_evaluations(items: &Self::Repr) -> Vec<F> {
        get_padded_evaluations_for_list(items)
    }

    type IntoIter<'a> = std::vec::IntoIter<BinDecomp4Bit<F>> where Self: 'a;

    fn from_iter(iter: impl IntoIterator<Item = Self>) -> Self::Repr {
        let iter = iter.into_iter();

        // --- TODO!(ryancao): This is genuinely horrible but we'll fix it later ---
        let mut ret: [Vec<F>; 4] = std::array::from_fn(|_| vec![]);
        iter.for_each(|tuple| {
            for (item, bit) in ret.iter_mut().zip(tuple.bits.iter()) {
                item.push(*bit);
            }
        });

        ret
    }

    fn to_iter(items: &Self::Repr) -> Self::IntoIter<'_> {
        let elems = (0..items[0].len()).map(
            |idx| {
                let bits = items.iter().map(
                    |item| {
                        item[idx]
                    }
                ).collect_vec();
                BinDecomp4Bit {
                    bits: bits.try_into().unwrap(),
                }
            }
        ).collect_vec();

        elems.into_iter()
    }

    fn num_vars(items: &Self::Repr) -> usize {
        log2(4 * items[0].len()) as usize
    }
}

// TODO!(ryancao): Make this stuff derivable
impl<F: FieldExt> DenseMle<F, BinDecomp4Bit<F>> {
    /// Returns a list of MLERefs, one for each bit
    /// TODO!(ryancao): Change this back to [DenseMleRef<F>; 4] and make it work!
    pub(crate) fn mle_bit_refs(&'_ self) -> Vec<DenseMleRef<F>> {
        let num_vars = self.num_iterated_vars();

        // --- There are sixteen components to this MLE ---
        let mut ret: Vec<DenseMleRef<F>> = vec![];

        for bit_idx in 0..4 {
            // --- Prefix bits need to be *literally* represented in little-endian ---
            let first_prefix = (bit_idx % 2) >= 1;
            let second_prefix = (bit_idx % 4) >= 2;

            let bit_mle_ref = DenseMleRef {
                bookkeeping_table: self.mle[bit_idx].to_vec(),
                // --- [0, 0, 0, 0, b_1, ..., b_n] ---
                mle_indices: self
                    .prefix_bits
                    .clone()
                    .into_iter()
                    .flatten()
                    .chain(
                        std::iter::once(MleIndex::Fixed(first_prefix))
                            .chain(std::iter::once(MleIndex::Fixed(second_prefix)))
                            .chain(repeat_n(MleIndex::Iterated, num_vars - 2)),
                    )
                    .collect_vec(),
                num_vars: num_vars - 2,
                layer_id: self.layer_id,
                indexed: false,
            };
            ret.push(bit_mle_ref);
        }

        ret
    }

    /// Combines the bookkeeping tables of each of the MleRefs within a
    /// `DenseMle<F, BinDecomp4Bit<F>>` into a single interleaved bookkeeping
    /// table such that referring to the merged table using little-endian indexing
    /// bits, followed by the appropriate MleRef indexing bits, gets us the same
    /// result as only using the same MleRef indexing bits on each MleRef from
    /// the `DenseMle<F, BinDecomp4Bit<F>>`.
    /// 
    /// TODO!(ende): refactor
    pub(crate) fn combine_mle_batch(input_mle_batch: Vec<DenseMle<F, BinDecomp4Bit<F>>>) -> DenseMle<F, F> {
        
        let batched_bits = log2(input_mle_batch.len());

        let input_mle_batch_ref_combined = input_mle_batch
            
            .into_iter().map(
                |x| {
                    combine_mle_refs(
                        x.mle_bit_refs()
                    ).mle_ref()
                }
            ).collect_vec();

        let input_mle_batch_ref_combined_ref =  combine_mles(input_mle_batch_ref_combined, batched_bits as usize);

        DenseMle::new_from_raw(input_mle_batch_ref_combined_ref.bookkeeping_table, LayerId::Input(0), None)
    }

    /// Returns the entire bin decomp MLE as a single MLE ref
    pub(crate) fn get_entire_mle_as_mle_ref(&'_ self) -> DenseMleRef<F> {
        // --- Just need to merge all of the bin decomps in an interleaved fashion ---
        // TODO!(ryancao): This is an awful hacky fix so that we can use `combine_mles`.
        // Note that we are manually inserting the extra iterated bits as prefix bits.
        // We should stop doing this once `combine_mles` works as it should!
        let self_mle_ref_vec = self.mle.clone().map(|mle_bookkeeping_table| {
            DenseMle::new_from_raw(
                mle_bookkeeping_table, 
                self.layer_id, 
                Some(
                    self.get_prefix_bits().iter().flatten().cloned().chain(
                        repeat_n(MleIndex::Iterated, 2)
                    ).collect_vec()
                )
            ).mle_ref()
        }).to_vec();
        combine_mles(self_mle_ref_vec, 4)
    }
}
