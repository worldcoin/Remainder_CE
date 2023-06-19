use std::{
    fmt::Debug,
    iter::{Zip, Cloned},
    marker::PhantomData,
    ops::{Range},
};

use ark_poly::DenseMultilinearExtension;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::log2;
use itertools::{Itertools, repeat_n};

use crate::FieldExt;

use super::{Mle, MleRef};

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
///An [Mle] that is dense
pub struct DenseMle<F: FieldExt, T: Send + Sync + Clone + Debug> {
    mle: DenseMultilinearExtension<F>,
    _marker: PhantomData<T>,
}

impl<F: FieldExt, T> Mle<F, T> for DenseMle<F, T>
where
    T: Send + Sync + Clone + Debug,
    Self: IntoIterator<Item = T> + FromIterator<T>,
{
    type MleRef<'a> = DenseMleRef<'a, F> where Self: 'a;

    type MultiLinearExtention = DenseMultilinearExtension<F>;

    fn mle(&self) -> &Self::MultiLinearExtention {
        &self.mle
    }

    fn mle_ref(&'_ self) -> Self::MleRef<'_> {
        DenseMleRef {
            mle: &self.mle,
            claim: (0..self.num_vars()).map(|_| None).collect(),
            range: 0..self.mle.evaluations.len(),
        }
    }

    fn new(mle: Self::MultiLinearExtention) -> Self {
        Self {
            mle,
            _marker: PhantomData,
        }
    }

    fn num_vars(&self) -> usize {
        self.mle.num_vars
    }
}

impl<F: FieldExt> IntoIterator for DenseMle<F, F> {
    type Item = F;

    type IntoIter = std::vec::IntoIter<F>;

    fn into_iter(self) -> Self::IntoIter {
        self.mle.evaluations.into_iter()
    }
}

impl<F: FieldExt> FromIterator<F> for DenseMle<F, F> {
    fn from_iter<T: IntoIterator<Item = F>>(iter: T) -> Self {
        let evaluations = iter.into_iter().collect_vec();

        let num_vars = log2(evaluations.len());

        let mle = DenseMultilinearExtension::from_evaluations_vec(num_vars as usize, evaluations);

        Self {
            mle,
            _marker: PhantomData,
        }
    }
}

//TODO!(Fix this so that it clones less)
impl<'a, F: FieldExt> IntoIterator for &'a DenseMle<F, (F, F)> {
    type Item = (F, F);

    type IntoIter = Zip<Cloned<std::slice::Iter<'a, F>>, Cloned<std::slice::Iter<'a, F>>>;

    fn into_iter(self) -> Self::IntoIter {
        let len = self.mle.evaluations.len() / 2;

        self.mle.evaluations[..len].iter().cloned()
            .zip(self.mle.evaluations[len..].iter().cloned())
    }
}

impl<F: FieldExt> FromIterator<(F, F)> for DenseMle<F, (F, F)> {
    fn from_iter<T: IntoIterator<Item = (F, F)>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let (mut first, second): (Vec<F>, Vec<F>) = iter.unzip();

        first.extend(second.into_iter());

        let vec = first;
        let num_vars = log2(vec.len()) as usize;

        Self {
            mle: DenseMultilinearExtension::from_evaluations_vec(num_vars, vec),
            _marker: PhantomData,
        }
    }
}

impl<F: FieldExt> DenseMle<F, (F, F)> {
    ///Gets an MleRef to the first element in the tuple
    pub fn first(&'_ self) -> DenseMleRef<'_, F> {
        let num_vars = self.mle.num_vars;

        let len = self.mle.evaluations.len() / 2;

        DenseMleRef {
            mle: &self.mle,
            claim: std::iter::once(Some(false)).chain(repeat_n(None, num_vars - 1)).collect_vec(),
            range: 0..len,
        }
    }

    ///Gets an MleRef to the second element in the tuple
    pub fn second(&'_ self) -> DenseMleRef<'_, F> {
        let num_vars = self.mle.num_vars;

        let len = self.mle.evaluations.len() / 2;

        DenseMleRef {
            mle: &self.mle,
            claim: std::iter::once(Some(true)).chain(repeat_n(None, num_vars - 1)).collect_vec(),
            range: len..self.mle.evaluations.len(),
        }
    }
}

///A [MleRef] that is dense
pub struct DenseMleRef<'a, F: FieldExt> {
    mle: &'a DenseMultilinearExtension<F>,
    claim: Vec<Option<bool>>,
    range: Range<usize>,
}

impl<'a, F: FieldExt> MleRef for DenseMleRef<'a, F> {
    type Mle = DenseMultilinearExtension<F>;

    fn mle_owned(&self) -> Self::Mle {
        DenseMultilinearExtension::from_evaluations_slice(
            self.mle.num_vars,
            &self.mle.evaluations[self.range.clone()],
        )
    }

    fn mle(&self) -> &'a Self::Mle {
        self.mle
    }

    fn claim(&self) -> &[Option<bool>] {
        &self.claim
    }

    fn relabel_claim(&mut self, new_claim: &[Option<bool>]) {
        self.claim = new_claim
            .iter()
            .cloned()
            .chain(self.claim.drain(..))
            .collect();
    }
}
