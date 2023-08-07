use std::{marker::PhantomData, cmp};

use crate::{FieldExt, mle::{MleAble, dense::{DenseMle, DenseMleRef}, MleIndex}, layer::LayerId};
use ark_std::log2;
use derive_more::{From, Into};
use itertools::{Itertools, repeat_n};

/// --- Path nodes within the tree and in the path hint ---
/// Used for the following components of the (circuit) input:
/// a)
#[derive(Copy, Debug, Clone, From, Into)]
pub struct DecisionNode<F: FieldExt> {
    pub node_id: F,
    pub attr_id: F,
    pub threshold: F,
}

#[derive(Copy, Debug, Clone, From, Into)]
pub struct LeafNode<F: FieldExt> {
    pub node_id: F,
    pub node_val: F,
}

/// --- 16-bit binary decomposition ---
/// Used for the following components of the (circuit) input:
/// a) The binary decomposition of the path node hints (i.e. path_x.thr - x.val)
/// b) The binary decomposition of the multiplicity coefficients $c_j$
#[derive(Copy, Debug, Clone, From, Into)]
pub struct BinDecomp16Bit<F: FieldExt> {
    pub bits: [F; 16],
}

/// --- Input element to the tree, i.e. a list of input attributes ---
/// Used for the following components of the (circuit) input:
/// a) The actual input attributes, i.e. x
/// b) The permuted input attributes, i.e. \bar{x}
#[derive(Copy, Debug, Clone, From, Into, PartialEq)]
pub struct InputAttribute<F: FieldExt> {
    // pub attr_idx: F,
    pub attr_id: F,
    pub attr_val: F,
}

// --- Just an enumeration of, uh, stuff...? ---
// To be honest this is basically just DenseMle<F>
// TODO!(ryancao)
// #[derive(Debug, Clone)]
// pub struct EnumerationRange<F: FieldExt> {
//     // TODO!(ryancao)
//     pub attr_id: F,
//     // pub attr_val: F,
// }

// Personally for the above, just give me a Vec<u32> and that should be great!
impl<F: FieldExt> MleAble<F> for DecisionNode<F> {
    type Repr = [Vec<F>; 3];
}

// TODO!(ryancao): Actually implement this correctly for PathNode<F>
// impl<'a, F: FieldExt> IntoIterator for &'a DenseMle<F, PathNode<F>> {
//     type Item = (F, F, F);

//     type IntoIter = Zip<Cloned<std::slice::Iter<'a, F>>, Cloned<std::slice::Iter<'a, F>>>;

//     fn into_iter(self) -> Self::IntoIter {
//         let len = self.mle.len() / 2;

//         self.mle[0]
//             .iter()
//             .cloned()
//             .zip(self.mle[1].iter().cloned())
//     }
// }

/// Conversion from a list of `PathNode<F>`s into a DenseMle
impl<F: FieldExt> FromIterator<DecisionNode<F>> for DenseMle<F, DecisionNode<F>> {
    fn from_iter<T: IntoIterator<Item = DecisionNode<F>>>(iter: T) -> Self {
        // --- The physical storage is [node_id_1, ...] + [attr_id_1, ...] + [threshold_1, ...] ---
        let iter = iter.into_iter();
        let (node_ids, attr_ids, thresholds): (Vec<F>, Vec<F>, Vec<F>) = iter
            .map(|x| (x.node_id, x.attr_id, x.threshold))
            .multiunzip();

        // --- TODO!(ryancao): Does this pad correctly? (I.e. is this necessary?) ---
        let num_vars = log2(4 * node_ids.len()) as usize;

        Self {
            mle: [node_ids, attr_ids, thresholds],
            num_vars,
            _marker: PhantomData,
            layer_id: None,
            prefix_bits: None,
        }
    }
}

impl<F: FieldExt> DenseMle<F, DecisionNode<F>> {
    /// MleRef grabbing just the list of node IDs
    pub fn node_id(&'_ self) -> DenseMleRef<F> {
        let num_vars = self.num_vars;

        // --- There are four components to this MLE ---
        let len = self.mle.len() / 4;

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
                    std::iter::once(MleIndex::Fixed(false))
                        .chain(std::iter::once(MleIndex::Fixed(false)))
                        .chain(repeat_n(MleIndex::Iterated, num_vars - 2)),
                )
                .collect_vec(),
            num_vars: num_vars - 2,
            layer_id: self.layer_id.clone().unwrap(),
            indexed: false,
        }
    }

    /// MleRef grabbing just the list of attribute IDs
    pub fn attr_id(&'_ self) -> DenseMleRef<F> {
        let num_vars = self.num_vars;

        // --- There are four components to this MLE ---
        let len = self.mle.len() / 4;

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
                    std::iter::once(MleIndex::Fixed(false))
                        .chain(std::iter::once(MleIndex::Fixed(true)))
                        .chain(repeat_n(MleIndex::Iterated, num_vars - 2)),
                )
                .collect_vec(),
            num_vars: num_vars - 2,
            layer_id: self.layer_id.clone().unwrap(),
            indexed: false,
        }
    }

    /// MleRef grabbing just the list of thresholds
    pub fn threshold(&'_ self) -> DenseMleRef<F> {
        let num_vars = self.num_vars;

        // --- There are four components to this MLE ---
        // let len = self.mle.len() / 4;

        DenseMleRef {
            bookkeeping_table: self.mle[2].to_vec(),
            // --- [1, 0, b_1, ..., b_n] ---
            // TODO!(ryancao): Does this give us the endian-ness we want???
            mle_indices: self
                .prefix_bits
                .clone()
                .into_iter()
                .flatten()
                .chain(
                    std::iter::once(MleIndex::Fixed(true))
                        .chain(std::iter::once(MleIndex::Fixed(false)))
                        .chain(repeat_n(MleIndex::Iterated, num_vars - 2)),
                )
                .collect_vec(),
            num_vars: num_vars - 2,
            layer_id: self.layer_id.clone().unwrap(),
            indexed: false,
        }
    }
}

// --- Leaf node ---
impl<F: FieldExt> MleAble<F> for LeafNode<F> {
    type Repr = [Vec<F>; 2];
}

// TODO!(ryancao): Actually implement this correctly for LeafNode<F>
// impl<'a, F: FieldExt> IntoIterator for &'a DenseMle<F, LeafNode<F>> {
//     type Item = (F, F, F);

//     type IntoIter = Zip<Cloned<std::slice::Iter<'a, F>>, Cloned<std::slice::Iter<'a, F>>>;

//     fn into_iter(self) -> Self::IntoIter {
//         let len = self.mle.len() / 2;

//         self.mle[0]
//             .iter()
//             .cloned()
//             .zip(self.mle[1].iter().cloned())
//     }
// }

/// Conversion from a list of `LeafNode<F>`s into a DenseMle
impl<F: FieldExt> FromIterator<LeafNode<F>> for DenseMle<F, LeafNode<F>> {
    fn from_iter<T: IntoIterator<Item = LeafNode<F>>>(iter: T) -> Self {
        // --- The physical storage is [node_id_1, ...] + [attr_id_1, ...] + [threshold_1, ...] ---
        let iter = iter.into_iter();
        let (node_ids, node_vals): (Vec<F>, Vec<F>) = iter.map(|x| (x.node_id, x.node_val)).unzip();

        // --- TODO!(ryancao): Does this pad correctly? (I.e. is this necessary?) ---
        let num_vars = log2(2 * node_ids.len()) as usize;

        Self {
            mle: [node_ids, node_vals],
            num_vars,
            _marker: PhantomData,
            layer_id: None,
            prefix_bits: None,
        }
    }
}

impl<F: FieldExt> DenseMle<F, LeafNode<F>> {
    /// MleRef grabbing just the list of node IDs
    pub fn node_id(&'_ self) -> DenseMleRef<F> {
        let num_vars = self.num_vars;

        // --- There are four components to this MLE ---
        let len = self.mle.len() / 2;

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
            layer_id: self.layer_id.clone().unwrap(),
            indexed: false,
        }
    }

    /// MleRef grabbing just the list of attribute IDs
    pub fn node_val(&'_ self) -> DenseMleRef<F> {
        let num_vars = self.num_vars;

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
            layer_id: self.layer_id.clone().unwrap(),
            indexed: false,
        }
    }
}

// --- Input attribute ---
impl<F: FieldExt> MleAble<F> for InputAttribute<F> {
    type Repr = [Vec<F>; 2];
}

// TODO!(ryancao): Actually implement this correctly for InputAttribute<F>
// impl<'a, F: FieldExt> IntoIterator for &'a DenseMle<F, InputAttribute<F>> {
//     type Item = (F, F, F);

//     type IntoIter = Zip<Cloned<std::slice::Iter<'a, F>>, Cloned<std::slice::Iter<'a, F>>>;

//     fn into_iter(self) -> Self::IntoIter {
//         let len = self.mle.len() / 2;

//         self.mle[0]
//             .iter()
//             .cloned()
//             .zip(self.mle[1].iter().cloned())
//     }
// }

/// Conversion from a list of `InputAttribute<F>`s into a DenseMle
impl<F: FieldExt> FromIterator<InputAttribute<F>> for DenseMle<F, InputAttribute<F>> {
    fn from_iter<T: IntoIterator<Item = InputAttribute<F>>>(iter: T) -> Self {
        // --- The physical storage is [node_id_1, ...] + [attr_id_1, ...] + [threshold_1, ...] ---
        let iter = iter.into_iter();
        let (attr_ids, attr_vals): (Vec<F>, Vec<F>) = iter.map(|x| (x.attr_id, x.attr_val)).unzip();

        // --- TODO!(ryancao): Does this pad correctly? (I.e. is this necessary?) ---
        let num_vars = log2(2 * attr_ids.len()) as usize;

        Self {
            mle: [attr_ids, attr_vals],
            num_vars,
            _marker: PhantomData,
            layer_id: None,
            prefix_bits: None,
        }
    }
}

impl<F: FieldExt> DenseMle<F, InputAttribute<F>> {
    /// MleRef grabbing just the list of attribute IDs
    pub fn attr_id(&'_ self, num_vars: Option<usize>) -> DenseMleRef<F> {
        // --- Default to the entire (component of) the MLE ---
        let num_vars = num_vars.unwrap_or(self.num_vars - 1);

        // TODO!(ryancao): Make this actually do error-handling
        assert!(num_vars <= self.num_vars - 1);

        // --- The length of the MLERef is just 2^{num_vars} ---
        let len = 2_u32.pow(num_vars as u32) as usize;
        let concrete_len = cmp::min(len, self.mle[0].to_vec().len());

        DenseMleRef {
            bookkeeping_table: self.mle[0].to_vec()[..concrete_len].to_vec(),
            // --- [0; 0, ..., 0; b_1, ..., b_n] ---
            // TODO!(ryancao): Does this give us the endian-ness we want???
            mle_indices: self
                .prefix_bits
                .clone()
                .into_iter()
                .flatten()
                .chain(
                    std::iter::once(MleIndex::Fixed(false))
                        .chain(repeat_n(
                            MleIndex::Fixed(false),
                            self.num_vars - 1 - num_vars,
                        ))
                        .chain(repeat_n(MleIndex::Iterated, num_vars)),
                )
                .collect_vec(),
            num_vars,
            layer_id: self.layer_id.clone().unwrap(),
            indexed: false,
        }
    }

    /// MleRef grabbing just the list of attribute values
    pub fn attr_val(&'_ self, num_vars: Option<usize>) -> DenseMleRef<F> {
        // --- Default to the entire (component of) the MLE ---
        let num_vars = match num_vars {
            Some(num) => num,
            None => self.num_vars - 1,
        };

        // TODO!(ryancao): Make this actually do error-handling
        assert!(num_vars <= self.num_vars - 1);

        // --- The length of the MLERef is just 2^{num_vars} ---
        let len = 2_u32.pow(num_vars as u32) as usize;
        let concrete_len = cmp::min(len, self.mle[1].to_vec().len());

        DenseMleRef {
            bookkeeping_table: self.mle[1].to_vec()[..concrete_len].to_vec(),
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
                        .chain(repeat_n(
                            MleIndex::Fixed(false),
                            self.num_vars - 1 - num_vars,
                        ))
                        .chain(repeat_n(MleIndex::Iterated, num_vars)),
                )
                .collect_vec(),
            num_vars,
            layer_id: self.layer_id.clone().unwrap(),
            indexed: false,
        }
    }
}

// --- Bin decomp ---
impl<F: FieldExt> MleAble<F> for BinDecomp16Bit<F> {
    type Repr = [Vec<F>; 16];
}

/// Conversion from a list of `BinDecomp16Bit<F>`s into a DenseMle
// TODO!(ryancao): Make this stuff derivable
impl<F: FieldExt> FromIterator<BinDecomp16Bit<F>> for DenseMle<F, BinDecomp16Bit<F>> {
    fn from_iter<T: IntoIterator<Item = BinDecomp16Bit<F>>>(iter: T) -> Self {
        // --- The physical storage is [bit_0, ...] + [bit_1, ...] + [bit_2, ...], ... ---
        let iter = iter.into_iter();

        // --- TODO!(ryancao): This is genuinely horrible but we'll fix it later ---
        let mut ret: [Vec<F>; 16] = std::array::from_fn(|_| vec![]);
        iter.for_each(|tuple| {
            for idx in 0..tuple.bits.len() {
                ret[idx].push(tuple.bits[idx]);
            }
        });

        // --- TODO!(ryancao): Does this pad correctly? (I.e. is this necessary?) ---
        let num_vars = log2(16 * ret[0].len()) as usize;

        Self {
            mle: ret,
            num_vars,
            _marker: PhantomData,
            layer_id: None,
            prefix_bits: None,
        }
    }
}

// TODO!(ryancao): Make this stuff derivable
impl<F: FieldExt> DenseMle<F, BinDecomp16Bit<F>> {
    /// Returns a list of MLERefs, one for each bit
    /// TODO!(ryancao): Change this back to [DenseMleRef<F>; 16] and make it work!
    pub fn mle_bit_refs(&'_ self) -> Vec<DenseMleRef<F>> {
        let num_vars = self.num_vars;

        // --- There are sixteen components to this MLE ---
        let mut ret: Vec<DenseMleRef<F>> = vec![];

        for bit_idx in 0..16 {
            let first_prefix = (bit_idx % 16) >= 8;
            let second_prefix = (bit_idx % 8) >= 4;
            let third_prefix = (bit_idx % 4) >= 2;
            let fourth_prefix = (bit_idx % 2) >= 1;
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
                layer_id: self.layer_id.clone().unwrap(),
                indexed: false,
            };
            ret.push(bit_mle_ref);
        }

        ret
    }
}
