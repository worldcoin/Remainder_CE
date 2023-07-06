use std::{
    fmt::Debug,
    iter::{Cloned, Zip},
    marker::PhantomData,
    ops::Range,
};

use ark_poly::DenseMultilinearExtension;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::log2;
use itertools::{repeat_n, Itertools};
use rayon::{slice::ParallelSlice, prelude::ParallelIterator};

use crate::FieldExt;

use super::{Mle, MleIndex, MleRef};

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
///An [Mle] that is dense
pub struct DenseMle<F: FieldExt, T: Send + Sync + Clone + Debug> {
    mle: Vec<F>,
    num_vars: usize,
    _marker: PhantomData<T>,
}

impl<F: FieldExt, T> Mle<F, T> for DenseMle<F, T>
where
    T: Send + Sync + Clone + Debug,
{
    type MleRef = DenseMleRef<F>;

    type MultiLinearExtention = Vec<F>;

    fn mle(&self) -> &Self::MultiLinearExtention {
        &self.mle
    }

    fn mle_ref(&'_ self) -> Self::MleRef {
        DenseMleRef {
            mle: self.mle.clone(),
            mle_indices: (0..self.num_vars()).map(|_| MleIndex::Iterated).collect(),
            range: 0..self.mle.len(),
            num_vars: self.num_vars,
        }
    }

    fn new(mle: Self::MultiLinearExtention) -> Self {
        let num_vars = log2(mle.len()) as usize;
        Self {
            mle,
            num_vars,
            _marker: PhantomData,
        }
    }

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

//TODO!(Fix this so that it clones less)
impl<'a, F: FieldExt> IntoIterator for &'a DenseMle<F, (F, F)> {
    type Item = (F, F);

    type IntoIter = Zip<Cloned<std::slice::Iter<'a, F>>, Cloned<std::slice::Iter<'a, F>>>;

    fn into_iter(self) -> Self::IntoIter {
        let len = self.mle.len() / 2;

        self.mle[..len]
            .iter()
            .cloned()
            .zip(self.mle[len..].iter().cloned())
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
            mle: vec,
            num_vars,
            _marker: PhantomData,
        }
    }
}

impl<F: FieldExt> DenseMle<F, (F, F)> {
    ///Gets an MleRef to the first element in the tuple
    pub fn first(&'_ self) -> DenseMleRef<F> {
        let num_vars = self.num_vars;

        let len = self.mle.len() / 2;

        DenseMleRef {
            mle: self.mle[0..len].to_vec(),
            mle_indices: std::iter::once(MleIndex::Fixed(false))
                .chain(repeat_n(MleIndex::Iterated, num_vars - 1))
                .collect_vec(),
            range: 0..len,
            num_vars,
        }
    }

    ///Gets an MleRef to the second element in the tuple
    pub fn second(&'_ self) -> DenseMleRef<F> {
        let num_vars = self.num_vars;

        let len = self.mle.len() / 2;

        DenseMleRef {
            mle: self.mle[len..self.mle.len()].to_vec(),
            mle_indices: std::iter::once(MleIndex::Fixed(true))
                .chain(repeat_n(MleIndex::Iterated, num_vars - 1))
                .collect_vec(),
            range: len..self.mle.len(),
            num_vars,
        }
    }
}

///A [MleRef] that is dense
#[derive(Clone, Debug)]
pub struct DenseMleRef<F: FieldExt> {
    mle: Vec<F>,
    mle_indices: Vec<MleIndex<F>>,
    range: Range<usize>,
    num_vars: usize,
}

impl<'a, F: FieldExt> MleRef for DenseMleRef<F> {
    type Mle = Vec<F>;
    type F = F;

    fn mle_owned(&self) -> Self::Mle {
        self.mle[self.range.clone()].to_vec()
    }

    fn mle(&self) -> &[F] {
        &self.mle[self.range.clone()]
    }

    fn mle_indices(&self) -> &[MleIndex<Self::F>] {
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

    fn fix_variable(&mut self, round_index: usize, challenge: Self::F) {
        for mle_index in self.mle_indices.iter_mut() {
            if *mle_index == MleIndex::IndexedBit(round_index) {
                *mle_index = MleIndex::Bound(challenge);
            }
        }
        
        let transform = |chunk: &[F]| {
            let zero = F::zero();
            let first = chunk[0];
            let second = chunk.get(1).unwrap_or(&zero);
    
            first + (*second - first) * challenge
        };

        #[cfg(feature = "parallel")]
        let new = self.mle().par_chunks(2).map(transform);

        #[cfg(not(feature = "parallel"))]
        let new = self.mle().par_chunks(2).map(transform);

        self.mle = new.collect();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;

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

        let mle = tuple_vec.into_iter().collect::<DenseMle<Fr, (Fr, Fr)>>();

        let mle_vec = vec![
            Fr::from(0),
            Fr::from(2),
            Fr::from(4),
            Fr::from(6),
            Fr::from(1),
            Fr::from(3),
            Fr::from(5),
            Fr::from(7),
        ];

        assert!(mle.mle == mle_vec);
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

        assert!(mle_ref.mle_indices == vec![MleIndex::Iterated, MleIndex::Iterated, MleIndex::Iterated]);
        assert!(mle_ref.mle == mle_vec);
        assert!(mle_ref.range.eq(0..mle_vec.len()));
    }

    #[test]
    fn create_dense_mle_ref_from_tuple_mle() {
        let tuple_vec = vec![
            (Fr::from(0), Fr::from(1)),
            (Fr::from(2), Fr::from(3)),
            (Fr::from(4), Fr::from(5)),
            (Fr::from(6), Fr::from(7)),
        ];

        let mle = tuple_vec.into_iter().collect::<DenseMle<Fr, (Fr, Fr)>>();

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

        assert!(first.range.start == 0 && first.range.end == 4);
        assert!(second.range.start == 4 && second.range.end == 8)
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
