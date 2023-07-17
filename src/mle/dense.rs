use std::{
    fmt::Debug,
    iter::{Cloned, Zip},
    marker::PhantomData,
};


use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{log2, cfg_iter, cfg_into_iter};
use derive_more::{From, Into};
use itertools::{repeat_n, Itertools};
use rayon::{prelude::ParallelIterator, slice::ParallelSlice};

use crate::FieldExt;

use super::{Mle, MleIndex, MleRef, MleAble};

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
///An [Mle] that is dense
pub struct DenseMle<F: FieldExt, T: Send + Sync + Clone + Debug + MleAble<F>> {
    mle: T::Repr,
    num_vars: usize,
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
            _marker: PhantomData,
        }
    }

    pub fn mle_ref(&self) -> DenseMleRef<F> {
        DenseMleRef {
            mle: self.mle.clone(),
            mle_indices: (0..self.num_vars()).map(|_| MleIndex::Iterated).collect(),
            num_vars: self.num_vars,
            layer_id: None,
        }
    }
}

#[derive(Debug, Clone, From, Into)]
struct Tuple2<F: FieldExt>((F, F));

impl<F: FieldExt> MleAble<F> for Tuple2<F> {
    type Repr = [Vec<F>; 2];
}

//TODO!(Fix this so that it clones less)
impl<'a, F: FieldExt> IntoIterator for &'a DenseMle<F, Tuple2<F>> {
    type Item = (F, F);

    type IntoIter = Zip<Cloned<std::slice::Iter<'a, F>>, Cloned<std::slice::Iter<'a, F>>>;

    fn into_iter(self) -> Self::IntoIter {
        let len = self.mle.len() / 2;

        self.mle[0]
            .iter()
            .cloned()
            .zip(self.mle[1].iter().cloned())
    }
}

impl<F: FieldExt> FromIterator<Tuple2<F>> for DenseMle<F, Tuple2<F>> {
    fn from_iter<T: IntoIterator<Item = Tuple2<F>>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let (first, second): (Vec<F>, Vec<F>) = iter.map(|x| (x.0.0, x.0.1)).unzip();

        let num_vars = log2(first.len() + second.len()) as usize;

        Self {
            mle: [first, second],
            num_vars,
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
            mle: self.mle[0].to_vec(),
            mle_indices: std::iter::once(MleIndex::Fixed(false))
                .chain(repeat_n(MleIndex::Iterated, num_vars - 1))
                .collect_vec(),
            num_vars,
            layer_id: None,
        }
    }

    ///Gets an MleRef to the second element in the tuple
    pub fn second(&'_ self) -> DenseMleRef<F> {
        let num_vars = self.num_vars;

        let len = self.mle.len() / 2;

        DenseMleRef {
            mle: self.mle[1].to_vec(),
            mle_indices: std::iter::once(MleIndex::Fixed(true))
                .chain(repeat_n(MleIndex::Iterated, num_vars - 1))
                .collect_vec(),
            num_vars,
            layer_id: None,
        }
    }
}

/// An [MleRef] that is dense
#[derive(Clone, Debug)]
pub struct DenseMleRef<F: FieldExt> {
    mle: Vec<F>,
    mle_indices: Vec<MleIndex<F>>,
    num_vars: usize,
    layer_id: Option<usize>,
}

impl<'a, F: FieldExt> MleRef for DenseMleRef<F> {
    type Mle = Vec<F>;
    type F = F;

    fn mle_owned(&self) -> Self::Mle {
        self.mle.clone()
    }

    fn mle(&self) -> &[F] {
        &self.mle
    }

    fn mle_indices(&self) -> &[MleIndex<Self::F>] {
        &self.mle_indices
    }

    fn get_mle_indices(&self) -> &[MleIndex<Self::F>] {
        &self.mle_indices
    }

    fn relabel_mle_indices(&mut self, new_indices: &[MleIndex<F>]) {
        self.mle_indices = new_indices
            .iter()
            .cloned()
            .chain(self.mle_indices.drain(..))
            .collect();
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }


    /// Ryan's note -- I assume this function updates the bookkeeping tables as
    /// described by [Tha13]. 
    fn fix_variable(
        &mut self,
        round_index: usize,
        challenge: Self::F,
    ) -> Option<(F, Vec<MleIndex<F>>)> {

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
        let new = self.mle().par_chunks(2).map(transform);

        #[cfg(not(feature = "parallel"))]
        let new = self.mle().chunks(2).map(transform);

        // --- Note that MLE is destructively modified into the new bookkeeping table here ---
        self.mle = new.collect();

        // --- Just returns the final value if we've collapsed the table into a single value ---
        if self.mle.len() == 1 {
            Some((self.mle[0], self.mle_indices.clone()))
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

    fn get_layer_id(&self) -> Option<usize> {
        self.layer_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;

    #[test]
    ///test fixing variables in an mle with two variables
    fn fix_variable_twovars() {
        let mle_vec = vec![Fr::from(5), Fr::from(2), Fr::from(1), Fr::from(3)];
        let mle: DenseMle<Fr, Fr> = DenseMle::new(mle_vec);
        let mut mle_ref = mle.mle_ref();
        mle_ref.fix_variable(1, Fr::from(1));

        let mle_vec_exp = vec![Fr::from(2), Fr::from(3)];
        let mle_exp: DenseMle<Fr, Fr> = DenseMle::new(mle_vec_exp);
        assert_eq!(mle_ref.mle, mle_exp.mle);
    }
    #[test]
    ///test fixing variables in an mle with three variables
    fn fix_variable_threevars() {
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
        assert_eq!(mle_ref.mle, mle_exp.mle);
    }

    #[test]
    ///test nested fixing variables in an mle with three variables
    fn fix_variable_nested() {
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
        let next_mle: DenseMle<Fr, Fr> = DenseMle::new(mle_ref.mle);
        let mut next_mle_ref = next_mle.mle_ref();
        next_mle_ref.fix_variable(2, Fr::from(2));

        let mle_vec_exp = vec![Fr::from(6), Fr::from(11)];
        let mle_exp: DenseMle<Fr, Fr> = DenseMle::new(mle_vec_exp);
        assert_eq!(next_mle_ref.mle, mle_exp.mle);
    }

    #[test]
    ///test nested fixing all the wayyyy
    fn fix_variable_full() {
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
        let next_mle: DenseMle<Fr, Fr> = DenseMle::new(mle_ref.mle);
        let mut next_mle_ref = next_mle.mle_ref();
        next_mle_ref.fix_variable(2, Fr::from(2));
        let next2_mle: DenseMle<Fr, Fr> = DenseMle::new(next_mle_ref.mle);
        let mut next2_mle_ref = next2_mle.mle_ref();
        next2_mle_ref.fix_variable(3, Fr::from(4));

        let mle_vec_exp = vec![Fr::from(26)];
        let mle_exp: DenseMle<Fr, Fr> = DenseMle::new(mle_vec_exp);
        assert_eq!(next2_mle_ref.mle, mle_exp.mle);
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

        let mle = tuple_vec.clone().into_iter().map(Tuple2::from).collect::<DenseMle<Fr, Tuple2<Fr>>>();

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
        assert!(mle_ref.mle == mle_vec);
    }

    #[test]
    fn create_dense_mle_ref_from_tuple_mle() {
        let tuple_vec = vec![
            (Fr::from(0), Fr::from(1)),
            (Fr::from(2), Fr::from(3)),
            (Fr::from(4), Fr::from(5)),
            (Fr::from(6), Fr::from(7)),
        ];

        let mle = tuple_vec.into_iter().map(Tuple2::from).collect::<DenseMle<Fr, Tuple2<Fr>>>();

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

        assert!(first.mle_owned() == vec![Fr::from(0), Fr::from(2), Fr::from(4), Fr::from(6)]);
        assert!(second.mle_owned() == vec![Fr::from(1), Fr::from(3), Fr::from(5), Fr::from(7)]);
    }

    #[test]
    fn relabel_claim_dense_mle() {
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

        let mle: DenseMle<Fr, Fr> = DenseMle::new(mle_vec);

        let mut mle_ref: DenseMleRef<Fr> = mle.mle_ref();

        mle_ref.relabel_mle_indices(&[MleIndex::Fixed(true), MleIndex::Fixed(false)]);

        assert!(
            mle_ref.mle_indices
                == vec![
                    MleIndex::Fixed(true),
                    MleIndex::Fixed(false),
                    MleIndex::Iterated,
                    MleIndex::Iterated,
                    MleIndex::Iterated
                ]
        );
    }
}
