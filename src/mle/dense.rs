use std::{
    cmp,
    fmt::Debug,
    iter::{Cloned, Zip},
    marker::PhantomData,
};

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{cfg_into_iter, cfg_iter, log2};
use derive_more::{From, Into};
use itertools::{repeat_n, Itertools};
use rayon::{prelude::ParallelIterator, slice::ParallelSlice};

use super::{Mle, MleAble, MleIndex, MleRef};
use crate::layer::Claim;
use crate::{
    zkdt::structs::LeafNode,
    {layer::LayerId, FieldExt},
};

use super::super::zkdt::structs::{BinDecomp16Bit, DecisionNode, InputAttribute};
use thiserror::Error;

#[derive(Error, Debug, Clone)]

/// Error for handling beta table updates
pub enum BetaError {
    #[error("claim index is 0, cannot take inverse")]
    NoInverse,
    #[error("not enough claims to compute beta table")]
    NotEnoughClaims,
    #[error("no initialized beta table")]
    NoBetaTable,
}

#[derive(Clone, Debug)]
///An [Mle] that is dense
pub struct DenseMle<F: FieldExt, T: Send + Sync + Clone + Debug + MleAble<F>> {
    mle: T::Repr,
    num_vars: usize,
    layer_id: Option<LayerId>,
    prefix_bits: Option<Vec<MleIndex<F>>>,
    _marker: PhantomData<F>,
}

impl<F: FieldExt, T> Mle<F, T> for DenseMle<F, T>
where
    T: Send + Sync + Clone + Debug + MleAble<F>,
{
    type MleRef = DenseMleRef<F>;

    type MultiLinearExtention = Vec<F>;

    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn define_layer_id(&mut self, id: LayerId) {
        self.layer_id = Some(id);
    }

    fn add_prefix_bits(&mut self, prefix: Option<Vec<MleIndex<F>>>) {
        self.prefix_bits = prefix;
    }
}

impl<F: FieldExt> IntoIterator for DenseMle<F, F> {
    type Item = F;

    type IntoIter = std::vec::IntoIter<F>;

    fn into_iter(self) -> Self::IntoIter {
        self.mle.into_iter()
    }
}

impl<F: FieldExt> FromIterator<F> for DenseMle<F, F> {
    fn from_iter<T: IntoIterator<Item = F>>(iter: T) -> Self {
        let evaluations = iter.into_iter().collect_vec();

        let num_vars = log2(evaluations.len()) as usize;

        Self {
            mle: evaluations,
            num_vars,
            layer_id: None,
            prefix_bits: None,
            _marker: PhantomData,
        }
    }
}

impl<F: FieldExt> MleAble<F> for F {
    type Repr = Vec<F>;
}

impl<F: FieldExt> DenseMle<F, F> {
    pub fn new(mle: Vec<F>) -> Self {
        let num_vars = log2(mle.len()) as usize;
        Self {
            mle,
            num_vars,
            layer_id: None,
            prefix_bits: None,
            _marker: PhantomData,
        }
    }

    pub fn mle_ref(&self) -> DenseMleRef<F> {
        DenseMleRef {
            bookkeeping_table: self.mle.clone(),
            mle_indices: self
                .prefix_bits
                .clone()
                .into_iter()
                .flatten()
                .chain((0..self.num_vars()).map(|_| MleIndex::Iterated))
                .collect(),
            num_vars: self.num_vars,
            layer_id: self.layer_id.clone(),
        }
    }
}

#[derive(Debug, Clone, From, Into)]
pub struct Tuple2<F: FieldExt>((F, F));

impl<F: FieldExt> MleAble<F> for Tuple2<F> {
    type Repr = [Vec<F>; 2];
}

//TODO!(Fix this so that it clones less)
impl<'a, F: FieldExt> IntoIterator for &'a DenseMle<F, Tuple2<F>> {
    type Item = (F, F);

    type IntoIter = Zip<Cloned<std::slice::Iter<'a, F>>, Cloned<std::slice::Iter<'a, F>>>;

    fn into_iter(self) -> Self::IntoIter {
        let len = self.mle.len() / 2;

        self.mle[0].iter().cloned().zip(self.mle[1].iter().cloned())
    }
}

impl<F: FieldExt> FromIterator<Tuple2<F>> for DenseMle<F, Tuple2<F>> {
    fn from_iter<T: IntoIterator<Item = Tuple2<F>>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let (first, second): (Vec<F>, Vec<F>) = iter.map(|x| (x.0 .0, x.0 .1)).unzip();

        let num_vars = log2(first.len() + second.len()) as usize;

        Self {
            mle: [first, second],
            num_vars,
            layer_id: None,
            prefix_bits: None,
            _marker: PhantomData,
        }
    }
}

impl<F: FieldExt> DenseMle<F, Tuple2<F>> {
    ///Gets an MleRef to the first element in the tuple
    pub fn first(&'_ self) -> DenseMleRef<F> {
        let num_vars = self.num_vars;

        let len = self.mle.len() / 2;

        DenseMleRef {
            bookkeeping_table: self.mle[0].to_vec(),
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
            num_vars,
            layer_id: self.layer_id.clone(),
        }
    }

    ///Gets an MleRef to the second element in the tuple
    pub fn second(&'_ self) -> DenseMleRef<F> {
        let num_vars = self.num_vars;

        let len = self.mle.len() / 2;

        DenseMleRef {
            bookkeeping_table: self.mle[1].to_vec(),
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
            num_vars,
            layer_id: self.layer_id.clone(),
        }
    }
}

// --------------------------- zkDT struct stuff ---------------------------

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
            layer_id: self.layer_id.clone(),
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
            layer_id: None,
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
            layer_id: None,
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
            layer_id: self.layer_id.clone(),
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
            layer_id: self.layer_id.clone(),
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
            layer_id: self.layer_id.clone(),
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
            layer_id: None,
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

        // let ret: [Vec<F>; 16] = (0..16).into_iter().map(|index| {
        //     iter.map(|tuple| {
        //         tuple.bits[index]
        //     }).collect()
        // }).collect_vec().try_into().unwrap();

        // For debugging
        // ret.clone().into_iter().for_each(|x| {
        //     dbg!(x.len());
        // });

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
        // TODO!(ryancao): Get rid of all the magic numbers
        let len = self.mle.len() / 16;

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
                layer_id: None,
            };
            ret.push(bit_mle_ref);
        }

        ret
    }
}

// --------------------------- MleRef stuff ---------------------------

/// An [MleRef] that is dense
#[derive(Clone, Debug)]
pub struct DenseMleRef<F: FieldExt> {
    bookkeeping_table: Vec<F>,
    mle_indices: Vec<MleIndex<F>>,
    /// Number of non-fixed variables within this MLE
    /// (warning: this gets modified destructively DURING sumcheck)
    num_vars: usize,
    layer_id: Option<LayerId>,
}

impl<'a, F: FieldExt> MleRef for DenseMleRef<F> {
    type F = F;

    fn bookkeeping_table(&self) -> &[F] {
        &self.bookkeeping_table
    }

    fn mle_indices(&self) -> &[MleIndex<Self::F>] {
        &self.mle_indices
    }

    // fn relabel_mle_indices(&mut self, new_indices: &[MleIndex<F>]) {
    //     self.mle_indices = new_indices
    //         .iter()
    //         .cloned()
    //         .chain(self.mle_indices.drain(..))
    //         .collect();
    // }

    fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Ryan's note -- I assume this function updates the bookkeeping tables as
    /// described by [Tha13].
    fn fix_variable(
        &mut self,
        round_index: usize,
        challenge: Self::F,
    ) -> Option<Claim<Self::F>> {
        // --- Bind the current indexed bit to the challenge value ---
        for mle_index in self.mle_indices.iter_mut() {
            if *mle_index == MleIndex::IndexedBit(round_index) {
                *mle_index = MleIndex::Bound(challenge);
            }
        }

        // --- One fewer iterated bit to sumcheck through ---
        self.num_vars -= 1;

        let transform = |chunk: &[F]| {
            let zero = F::zero();
            let first = chunk[0];
            let second = chunk.get(1).unwrap_or(&zero);

            // (1 - r) * V(i) + r * V(i + 1)
            first + (*second - first) * challenge
        };

        // --- So this goes through and applies the formula from [Tha13], bottom ---
        // --- of page 23 ---
        #[cfg(feature = "parallel")]
        let new = self.bookkeeping_table().par_chunks(2).map(transform);

        #[cfg(not(feature = "parallel"))]
        let new = self.mle().chunks(2).map(transform);

        // --- Note that MLE is destructively modified into the new bookkeeping table here ---
        self.bookkeeping_table = new.collect();

        // --- Just returns the final value if we've collapsed the table into a single value ---
        if self.bookkeeping_table.len() == 1 {
            Some((self.mle_indices.iter().map(|x| match x {
                MleIndex::Bound(chal) => *chal,
                MleIndex::Fixed(bit) => if *bit { F::one() } else { F::zero() },
                _ => panic!("All bits should be bound!")
            }).collect_vec(), self.bookkeeping_table[0]))
        } else {
            None
        }
    }

    fn index_mle_indices(&mut self, curr_index: usize) -> usize {
        let mut new_indices = 0;
        for mle_index in self.mle_indices.iter_mut() {
            if *mle_index == MleIndex::Iterated {
                *mle_index = MleIndex::IndexedBit(curr_index + new_indices);
                new_indices += 1;
            }
        }

        curr_index + new_indices
    }

    fn get_layer_id(&self) -> Option<LayerId> {
        self.layer_id.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sumcheck::SumOrEvals;
    use ark_bn254::Fr;
    use ark_ff::UniformRand;
    use ark_std::test_rng;
    use ark_std::One;
    use ark_std::Zero;

    #[test]
    ///test fixing variables in an mle with two variables
    fn fix_variable_twovars() {
        let layer_claims = (vec![Fr::from(3), Fr::from(4)], Fr::one());
        let mle_vec = vec![Fr::from(5), Fr::from(2), Fr::from(1), Fr::from(3)];
        let mle: DenseMle<Fr, Fr> = DenseMle::new(mle_vec);
        let mut mle_ref = mle.mle_ref();
        mle_ref.fix_variable(1, Fr::from(1));

        let mle_vec_exp = vec![Fr::from(2), Fr::from(3)];
        let mle_exp: DenseMle<Fr, Fr> = DenseMle::new(mle_vec_exp);
        assert_eq!(mle_ref.bookkeeping_table, mle_exp.mle);
    }
    #[test]
    ///test fixing variables in an mle with three variables
    fn fix_variable_threevars() {
        let layer_claims = (vec![Fr::from(3), Fr::from(4)], Fr::one());
        let mle_vec = vec![
            Fr::from(0),
            Fr::from(2),
            Fr::from(0),
            Fr::from(2),
            Fr::from(0),
            Fr::from(3),
            Fr::from(1),
            Fr::from(4),
        ];
        let mle: DenseMle<Fr, Fr> = DenseMle::new(mle_vec);
        let mut mle_ref = mle.mle_ref();
        mle_ref.fix_variable(1, Fr::from(3));

        let mle_vec_exp = vec![Fr::from(6), Fr::from(6), Fr::from(9), Fr::from(10)];
        let mle_exp: DenseMle<Fr, Fr> = DenseMle::new(mle_vec_exp);
        assert_eq!(mle_ref.bookkeeping_table, mle_exp.mle);
    }

    #[test]
    ///test nested fixing variables in an mle with three variables
    fn fix_variable_nested() {
        let layer_claims = (vec![Fr::from(3), Fr::from(4)], Fr::one());
        let mle_vec = vec![
            Fr::from(0),
            Fr::from(2),
            Fr::from(0),
            Fr::from(2),
            Fr::from(0),
            Fr::from(3),
            Fr::from(1),
            Fr::from(4),
        ];
        let mle: DenseMle<Fr, Fr> = DenseMle::new(mle_vec);
        let mut mle_ref = mle.mle_ref();
        mle_ref.fix_variable(1, Fr::from(3));
        mle_ref.fix_variable(2, Fr::from(2));

        let mle_vec_exp = vec![Fr::from(6), Fr::from(11)];
        let mle_exp: DenseMle<Fr, Fr> = DenseMle::new(mle_vec_exp);
        assert_eq!(mle_ref.bookkeeping_table, mle_exp.mle);
    }

    #[test]
    ///test nested fixing all the wayyyy
    fn fix_variable_full() {
        let layer_claims = (vec![Fr::from(3), Fr::from(4)], Fr::one());
        let mle_vec = vec![
            Fr::from(0),
            Fr::from(2),
            Fr::from(0),
            Fr::from(2),
            Fr::from(0),
            Fr::from(3),
            Fr::from(1),
            Fr::from(4),
        ];
        let mle: DenseMle<Fr, Fr> = DenseMle::new(mle_vec);
        let mut mle_ref = mle.mle_ref();
        let _ = mle_ref.index_mle_indices(0);
        mle_ref.fix_variable(0, Fr::from(3));
        mle_ref.fix_variable(1, Fr::from(2));
        mle_ref.fix_variable(2, Fr::from(4));

        let mle_vec_exp = vec![Fr::from(26)];
        let mle_exp: DenseMle<Fr, Fr> = DenseMle::new(mle_vec_exp);
        assert_eq!(mle_ref.bookkeeping_table, mle_exp.mle);
    }

    #[test]
    fn create_dense_mle_from_vec() {
        let mle_vec = vec![
            Fr::from(0),
            Fr::from(1),
            Fr::from(2),
            Fr::from(3),
            Fr::from(4),
            Fr::from(5),
            Fr::from(6),
            Fr::from(7),
        ];

        //DON'T do this normally, it clones the vec, if you have a flat MLE just use Mle::new
        let mle_iter = mle_vec.clone().into_iter().collect::<DenseMle<Fr, Fr>>();

        let mle_new: DenseMle<Fr, Fr> = DenseMle::new(mle_vec);

        assert!(mle_iter.mle == mle_new.mle);
        assert!(
            mle_iter.num_vars() == 3 && mle_new.num_vars() == 3,
            "Num vars must be the log_2 of the length of the vector"
        );
    }

    #[test]
    fn create_dense_tuple_mle_from_vec() {
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

        let (first, second): (Vec<Fr>, Vec<_>) = tuple_vec.into_iter().unzip();

        assert!(mle.mle == [first, second]);
        assert!(mle.num_vars() == 3);
    }

    #[test]
    fn create_dense_mle_ref_from_flat_mle() {
        let mle_vec = vec![
            Fr::from(0),
            Fr::from(1),
            Fr::from(2),
            Fr::from(3),
            Fr::from(4),
            Fr::from(5),
            Fr::from(6),
            Fr::from(7),
        ];

        let mle: DenseMle<Fr, Fr> = DenseMle::new(mle_vec.clone());

        let mle_ref: DenseMleRef<Fr> = mle.mle_ref();

        assert!(
            mle_ref.mle_indices == vec![MleIndex::Iterated, MleIndex::Iterated, MleIndex::Iterated]
        );
        assert!(mle_ref.bookkeeping_table == mle_vec);
    }

    #[test]
    fn create_dense_mle_ref_from_tuple_mle() {
        let tuple_vec = vec![
            (Fr::from(0), Fr::from(1)),
            (Fr::from(2), Fr::from(3)),
            (Fr::from(4), Fr::from(5)),
            (Fr::from(6), Fr::from(7)),
        ];

        let mle = tuple_vec
            .into_iter()
            .map(Tuple2::from)
            .collect::<DenseMle<Fr, Tuple2<Fr>>>();

        let first = mle.first();
        let second = mle.second();

        assert!(
            first.mle_indices
                == vec![
                    MleIndex::Fixed(false),
                    MleIndex::Iterated,
                    MleIndex::Iterated
                ]
        );
        assert!(
            second.mle_indices
                == vec![
                    MleIndex::Fixed(true),
                    MleIndex::Iterated,
                    MleIndex::Iterated
                ]
        );

        assert!(first.bookkeeping_table() == &[Fr::from(0), Fr::from(2), Fr::from(4), Fr::from(6)]);
        assert!(second.bookkeeping_table() == &[Fr::from(1), Fr::from(3), Fr::from(5), Fr::from(7)]);
    }

    // #[test]
    // fn relabel_claim_dense_mle() {
    //     let mle_vec = vec![
    //         Fr::from(0),
    //         Fr::from(1),
    //         Fr::from(2),
    //         Fr::from(3),
    //         Fr::from(4),
    //         Fr::from(5),
    //         Fr::from(6),
    //         Fr::from(7),
    //     ];

    //     let mle: DenseMle<Fr, Fr> = DenseMle::new(mle_vec);

    //     let mut mle_ref: DenseMleRef<Fr> = mle.mle_ref();

    //     mle_ref.relabel_mle_indices(&[MleIndex::Fixed(true), MleIndex::Fixed(false)]);

    //     assert!(
    //         mle_ref.mle_indices
    //             == vec![
    //                 MleIndex::Fixed(true),
    //                 MleIndex::Fixed(false),
    //                 MleIndex::Iterated,
    //                 MleIndex::Iterated,
    //                 MleIndex::Iterated
    //             ]
    //     );
    // }
}
