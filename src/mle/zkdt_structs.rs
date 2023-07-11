use std::{
    fmt::Debug,
    iter::{Cloned, Zip},
    marker::PhantomData,
};

use ark_std::log2;
use derive_more::{From, Into};
use itertools::{repeat_n, Itertools};

use crate::FieldExt;

use super::{MleIndex, MleAble};
use super::dense::DenseMle;

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
            bookkeeping_table: self.mle[0].to_vec(),
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
            bookkeeping_table: self.mle[1].to_vec(),
            mle_indices: std::iter::once(MleIndex::Fixed(true))
                .chain(repeat_n(MleIndex::Iterated, num_vars - 1))
                .collect_vec(),
            num_vars,
            layer_id: None,
        }
    }
}