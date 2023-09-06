use std::{
    fmt::Debug,
    iter::{Cloned, Map, Zip},
    marker::PhantomData,
};

use ark_std::log2;
// use derive_more::{From, Into};
use itertools::{repeat_n, Itertools, MapInto};
use rayon::{prelude::ParallelIterator, slice::ParallelSlice};
use serde::{Deserialize, Serialize};

use super::{mle_enum::MleEnum, Mle, MleAble, MleIndex, MleRef};
use crate::layer::LayerId;
use crate::{expression::ExpressionStandard, layer::Claim};
use remainder_shared_types::FieldExt;

#[derive(Clone, Debug, Serialize, Deserialize)]
///An [Mle] that is dense
pub struct DenseMle<F, T: Send + Sync + Clone + Debug + MleAble<F>> {
    ///The underlying data
    pub mle: T::Repr,
    num_iterated_vars: usize,
    ///The layer_id this data belongs to
    pub layer_id: LayerId,
    ///Any prefix bits that must be added to any MleRefs yielded by this Mle
    pub prefix_bits: Option<Vec<MleIndex<F>>>,
    ///marker
    pub _marker: PhantomData<F>,
}

impl<F: FieldExt, T> Mle<F> for DenseMle<F, T>
where
    // Self: IntoIterator<Item = T> + FromIterator<T>
    T: Send + Sync + Clone + Debug + MleAble<F>,
{
    fn num_iterated_vars(&self) -> usize {
        self.num_iterated_vars
    }
    fn get_padded_evaluations(&self) -> Vec<F> {
        T::get_padded_evaluations(&self.mle)
    }

    fn add_prefix_bits(&mut self, new_bits: Option<Vec<MleIndex<F>>>) {
        self.prefix_bits = new_bits;
    }

    fn get_prefix_bits(&self) -> Option<Vec<MleIndex<F>>> {
        self.prefix_bits.clone()
    }
}

impl<F: FieldExt, T: Send + Sync + Clone + Debug + MleAble<F>> DenseMle<F, T> {
    pub fn new_from_iter(
        iter: impl Iterator<Item = T>,
        layer_id: LayerId,
        prefix_bits: Option<Vec<MleIndex<F>>>,
    ) -> Self {
        let items = T::from_iter(iter);
        let num_vars = T::num_vars(&items);
        Self {
            mle: items,
            num_iterated_vars: num_vars,
            layer_id,
            prefix_bits,
            _marker: PhantomData,
        }
    }

    pub fn new_from_raw(
        items: T::Repr,
        layer_id: LayerId,
        prefix_bits: Option<Vec<MleIndex<F>>>,
    ) -> Self {
        let num_vars = T::num_vars(&items);
        Self {
            mle: items,
            num_iterated_vars: num_vars,
            layer_id,
            prefix_bits,
            _marker: PhantomData,
        }
    }
}

impl<'a, F: FieldExt, T: Send + Sync + Clone + Debug + MleAble<F>> IntoIterator
    for &'a DenseMle<F, T>
{
    type Item = T;

    type IntoIter = T::IntoIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        T::to_iter(&self.mle)
    }
}

// impl<F: FieldExt> FromIterator<F> for DenseMle<F, F> {
//     fn from_iter<T: IntoIterator<Item = F>>(iter: T) -> Self {
//         let evaluations = iter.into_iter().collect_vec();

//         let num_vars = log2(evaluations.len()) as usize;

//         Self {
//             mle: evaluations,
//             num_vars,
//             layer_id: None,
//             prefix_bits: None,
//             _marker: PhantomData,
//         }
//     }
// }

/// Takes the individual bookkeeping tables from the MleRefs within an MLE
/// and merges them with padding, using a little-endian representation
/// merge strategy. Assumes that ALL MleRefs are the same size.
pub(crate) fn get_padded_evaluations_for_list<F: FieldExt, const L: usize>(items: &[Vec<F>; L]) -> Vec<F> {

    // --- All the items within should be the same size ---
    let max_size = items
        .iter()
        .map(|mle_ref| mle_ref.len())
        .max().unwrap();

    let part_size = 1 << log2(max_size);
    let part_count = 2_u32.pow(log2(L)) as usize;

    // --- Number of "part" slots which need to filled with padding ---
    let padding_count = part_count - L;
    let total_size = part_size * part_count;
    let total_padding: usize = total_size - max_size * part_count;

    // items.into_iter().cloned().map(|mut items| {
    //     let padding = part_size - items.len();
    //     items.extend(repeat_n(F::zero(), padding));
    //     items
    // }).flatten().chain(repeat_n(F::zero(), padding_count * part_size)).collect()

    (0..max_size)
        .flat_map(|index| {
            items
                .iter()
                .map(move |item| item.get(index).unwrap_or(&F::zero()).clone())
                .chain(repeat_n(F::zero(), padding_count))
        })
        .chain(repeat_n(F::zero(), total_padding))
        .collect()
}

impl<F: FieldExt> MleAble<F> for F {
    type Repr = Vec<F>;
    type IntoIter<'a> = Cloned<std::slice::Iter<'a, F>>;

    fn get_padded_evaluations(items: &Self::Repr) -> Vec<F> {
        let size: usize = 1 << log2(items.len());
        let padding = size - items.len();

        items
            .iter()
            .cloned()
            .chain(repeat_n(F::zero(), padding))
            .collect()
    }

    fn from_iter(iter: impl IntoIterator<Item = Self>) -> Self::Repr {
        iter.into_iter().collect_vec()
    }

    fn to_iter<'a>(items: &'a Self::Repr) -> Self::IntoIter<'a> {
        items.iter().cloned()
    }

    fn num_vars(items: &Self::Repr) -> usize {
        log2(items.len()) as usize
    }
}

impl<F: FieldExt> DenseMle<F, F> {
    ///Creates a flat DenseMle from a Vec<F>
    // pub fn new(mle: Vec<F>) -> Self {
    //     let num_vars = log2(mle.len()) as usize;
    //     Self {
    //         mle,
    //         num_vars,
    //         layer_id: None,
    //         prefix_bits: None,
    //         _marker: PhantomData,
    //     }
    // }

    ///Creates a DenseMleRef from this DenseMle
    pub fn mle_ref(&self) -> DenseMleRef<F> {
        DenseMleRef {
            bookkeeping_table: self.mle.clone(),
            mle_indices: self
                .prefix_bits
                .clone()
                .into_iter()
                .flatten()
                .chain((0..self.num_iterated_vars()).map(|_| MleIndex::Iterated))
                .collect(),
            num_vars: self.num_iterated_vars,
            layer_id: self.layer_id.clone(),
            indexed: false,
        }
    }

    ///Splits the mle into a new mle with a tuple of size 2 as it's element
    pub fn split(&self, padding: F) -> DenseMle<F, Tuple2<F>> {
        DenseMle::new_from_iter(
            self.mle
                .chunks(2)
                .map(|items| (items[0], items.get(1).cloned().unwrap_or(padding)).into()),
            self.layer_id.clone(),
            self.prefix_bits.clone(),
        )
    }

    pub fn one(mle_len: usize, layer_id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> DenseMle<F, F> {
        let mut one_vec = vec![];
        for _ in 0..mle_len {
            one_vec.push(F::one())
        }
        DenseMle::new_from_raw(
            one_vec,
            layer_id,
            prefix_bits
        )
    }
}

#[derive(Debug, Clone)]
///Newtype around a tuple of field elements
pub struct Tuple2<F: FieldExt>(pub (F, F));

impl<F: FieldExt> MleAble<F> for Tuple2<F> {
    type Repr = [Vec<F>; 2];

    fn get_padded_evaluations(items: &Self::Repr) -> Vec<F> {
        get_padded_evaluations_for_list(items)
    }

    type IntoIter<'a> = Map<Zip<std::slice::Iter<'a, F>, std::slice::Iter<'a, F>>, fn((&F, &F)) -> Self> where Self: 'a;

    fn from_iter(iter: impl IntoIterator<Item = Self>) -> Self::Repr {
        let iter = iter.into_iter();
        let (first, second): (Vec<F>, Vec<F>) = iter.map(|x| (x.0 .0, x.0 .1)).unzip();
        [first, second]
    }

    fn to_iter<'a>(items: &'a Self::Repr) -> Self::IntoIter<'a> {
        items[0]
            .iter()
            .zip(items[1].iter())
            .map(|(first, second)| Tuple2((first.clone(), second.clone())))
    }

    fn num_vars(items: &Self::Repr) -> usize {
        log2(items[0].len() + items[1].len()) as usize
    }
}

impl<F: FieldExt> From<(F, F)> for Tuple2<F> {
    fn from(value: (F, F)) -> Self {
        Self(value)
    }
}

//TODO!(Fix this so that it clones less)
// impl<'a, F: FieldExt> IntoIterator for &'a DenseMle<F, Tuple2<F>> {
//     type Item = (F, F);

//     type IntoIter = Zip<Cloned<std::slice::Iter<'a, F>>, Cloned<std::slice::Iter<'a, F>>>;

//     fn into_iter(self) -> Self::IntoIter {
//         self.mle[0].iter().cloned().zip(self.mle[1].iter().cloned())
//     }
// }

// impl<F: FieldExt> FromIterator<Tuple2<F>> for DenseMle<F, Tuple2<F>> {
//     fn from_iter<T: IntoIterator<Item = Tuple2<F>>>(iter: T) -> Self {
//         let iter = iter.into_iter();
//         let (first, second): (Vec<F>, Vec<F>) = iter.map(|x| (x.0 .0, x.0 .1)).unzip();

//         let num_vars: usize = log2(first.len() + second.len()) as usize;

//         Self {
//             mle: [first, second],
//             num_vars,
//             layer_id: None,
//             prefix_bits: None,
//             _marker: PhantomData,
//         }
//     }
// }

impl<F: FieldExt> DenseMle<F, Tuple2<F>> {
    ///Gets an MleRef to the first element in the tuple
    pub fn first(&'_ self) -> DenseMleRef<F> {
        // --- Number of *remaining* iterated variables ---
        let new_num_iterated_vars = self.num_iterated_vars - 1;

        DenseMleRef {
            bookkeeping_table: self.mle[0].to_vec(),
            mle_indices: self
                .prefix_bits
                .clone()
                .into_iter()
                .flatten()
                .chain(
                    std::iter::once(MleIndex::Fixed(false))
                        .chain(repeat_n(MleIndex::Iterated, new_num_iterated_vars)),
                )
                .collect_vec(),
            num_vars: new_num_iterated_vars,
            layer_id: self.layer_id.clone(),
            indexed: false,
        }
    }

    ///Gets an MleRef to the second element in the tuple
    pub fn second(&'_ self) -> DenseMleRef<F> {
        let new_num_iterated_vars = self.num_iterated_vars - 1;

        DenseMleRef {
            bookkeeping_table: self.mle[1].to_vec(),
            mle_indices: self
                .prefix_bits
                .clone()
                .into_iter()
                .flatten()
                .chain(
                    std::iter::once(MleIndex::Fixed(true))
                        .chain(repeat_n(MleIndex::Iterated, new_num_iterated_vars)),
                )
                .collect_vec(),
            num_vars: new_num_iterated_vars,
            layer_id: self.layer_id.clone(),
            indexed: false,
        }
    }
}

// --------------------------- MleRef stuff ---------------------------

/// An [MleRef] that is dense
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DenseMleRef<F> {
    ///The bookkeeping table of this MleRefs evaluations over the boolean hypercube
    pub bookkeeping_table: Vec<F>,
    ///The MleIndices of this MleRef e.g. V(0, 1, r_1, r_2)
    pub mle_indices: Vec<MleIndex<F>>,
    /// Number of non-fixed variables within this MLE
    /// (warning: this gets modified destructively DURING sumcheck)
    pub num_vars: usize,
    /// The layer this MleRef is a reference to
    pub layer_id: LayerId,
    /// A marker that keeps track of if this MleRef is indexed
    pub indexed: bool,
}

impl<F: FieldExt> DenseMleRef<F> {
    ///Convienence function for wrapping this in an Expression
    pub fn expression(self) -> ExpressionStandard<F> {
        ExpressionStandard::Mle(self)
    }
}

impl<F: FieldExt> MleRef for DenseMleRef<F> {
    type F = F;

    fn bookkeeping_table(&self) -> &[F] {
        &self.bookkeeping_table
    }

    fn mle_indices(&self) -> &[MleIndex<Self::F>] {
        &self.mle_indices
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn indexed(&self) -> bool {
        self.indexed
    }

    /// Ryan's note -- I assume this function updates the bookkeeping tables as
    /// described by [Tha13].
    fn fix_variable(&mut self, round_index: usize, challenge: Self::F) -> Option<Claim<Self::F>> {
        // --- Bind the current indexed bit to the challenge value ---
        for mle_index in self.mle_indices.iter_mut() {
            if *mle_index == MleIndex::IndexedBit(round_index) {
                mle_index.bind_index(challenge);
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
        let new = self.bookkeeping_table().chunks(2).map(transform);

        // --- Note that MLE is destructively modified into the new bookkeeping table here ---
        self.bookkeeping_table = new.collect();
        // --- Just returns the final value if we've collapsed the table into a single value ---
        if self.bookkeeping_table.len() == 1 {
            Some((
                self.mle_indices
                    .iter()
                    .map(|index| index.val().unwrap())
                    .collect_vec(),
                self.bookkeeping_table[0],
            ))
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

        self.indexed = true;
        curr_index + new_indices
    }

    fn get_layer_id(&self) -> LayerId {
        self.layer_id.clone()
    }

    fn push_mle_indices(&mut self, new_indices: &[MleIndex<Self::F>]) {
        self.mle_indices.append(&mut new_indices.to_vec());
    }

    fn get_enum(self) -> MleEnum<Self::F> {
        MleEnum::Dense(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_base::halo2_proofs::halo2curves::bn256::Fr;

    #[test]
    ///test fixing variables in an mle with two variables
    fn fix_variable_twovars() {
        let _layer_claims = (vec![Fr::from(3), Fr::from(4)], Fr::one());
        let mle_vec = vec![Fr::from(5), Fr::from(2), Fr::from(1), Fr::from(3)];
        let mle: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0), None);
        let mut mle_ref = mle.mle_ref();
        mle_ref.fix_variable(1, Fr::from(1));

        let mle_vec_exp = vec![Fr::from(2), Fr::from(3)];
        let mle_exp: DenseMle<Fr, Fr> =
            DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);
        assert_eq!(mle_ref.bookkeeping_table, mle_exp.mle);
    }
    #[test]
    ///test fixing variables in an mle with three variables
    fn fix_variable_threevars() {
        let _layer_claims = (vec![Fr::from(3), Fr::from(4)], Fr::one());
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
        let mle: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0), None);
        let mut mle_ref = mle.mle_ref();
        mle_ref.fix_variable(1, Fr::from(3));

        let mle_vec_exp = vec![Fr::from(6), Fr::from(6), Fr::from(9), Fr::from(10)];
        let mle_exp: DenseMle<Fr, Fr> =
            DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);
        assert_eq!(mle_ref.bookkeeping_table, mle_exp.mle);
    }

    #[test]
    ///test nested fixing variables in an mle with three variables
    fn fix_variable_nested() {
        let _layer_claims = (vec![Fr::from(3), Fr::from(4)], Fr::one());
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
        let mle: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0), None);
        let mut mle_ref = mle.mle_ref();
        mle_ref.fix_variable(1, Fr::from(3));
        mle_ref.fix_variable(2, Fr::from(2));

        let mle_vec_exp = vec![Fr::from(6), Fr::from(11)];
        let mle_exp: DenseMle<Fr, Fr> =
            DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);
        assert_eq!(mle_ref.bookkeeping_table, mle_exp.mle);
    }

    #[test]
    ///test nested fixing all the wayyyy
    fn fix_variable_full() {
        let _layer_claims = (vec![Fr::from(3), Fr::from(4)], Fr::one());
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
        let mle: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0), None);
        let mut mle_ref = mle.mle_ref();
        let _ = mle_ref.index_mle_indices(0);
        mle_ref.fix_variable(0, Fr::from(3));
        mle_ref.fix_variable(1, Fr::from(2));
        mle_ref.fix_variable(2, Fr::from(4));

        let mle_vec_exp = vec![Fr::from(26)];
        let mle_exp: DenseMle<Fr, Fr> =
            DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);
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
        let mle_iter =
            DenseMle::new_from_iter(mle_vec.clone().into_iter(), LayerId::Input(0), None);

        let mle_new: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0), None);

        assert!(mle_iter.mle == mle_new.mle);
        assert!(
            mle_iter.num_iterated_vars() == 3 && mle_new.num_iterated_vars() == 3,
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

        let mle = DenseMle::new_from_iter(
            tuple_vec.clone().into_iter().map(Tuple2::from),
            LayerId::Input(0),
            None,
        );

        let (first, second): (Vec<Fr>, Vec<_>) = tuple_vec.into_iter().unzip();

        assert!(mle.mle == [first, second]);
        assert!(mle.num_iterated_vars() == 3);
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

        let mle: DenseMle<Fr, Fr> =
            DenseMle::new_from_raw(mle_vec.clone(), LayerId::Input(0), None);

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

        let mle = DenseMle::new_from_iter(
            tuple_vec.into_iter().map(Tuple2::from),
            LayerId::Input(0),
            None,
        );

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
        assert!(
            second.bookkeeping_table() == &[Fr::from(1), Fr::from(3), Fr::from(5), Fr::from(7)]
        );
    }
}
