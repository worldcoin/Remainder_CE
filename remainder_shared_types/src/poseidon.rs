use std::marker::PhantomData;

use itertools::Either;
use poseidon_circuit::{NativeSponge, Hashable, poseidon::primitives::{Absorbing, Spec, Domain, P128Pow5T3, Squeezing, ConstantLength}};

use crate::FieldExt;

#[derive(Clone)]
pub struct Poseidon<F: FieldExt, const WIDTH: usize, const RATE: usize> 
{
    sponge: Either<NativeSponge<F, P128Pow5T3<F>, Absorbing<F, 2>, 3, 2>, NativeSponge<F, P128Pow5T3<F>, Squeezing<F, 2>, 3, 2>>
}

impl<F: FieldExt, const WIDTH: usize, const RATE: usize> Poseidon<F, WIDTH, RATE>
{
    pub fn new(r_f: usize, r_p: usize) -> Self {
        dbg!("Calling a new sponge!!");
        Self {
            sponge: Either::Left(NativeSponge::new(<ConstantLength::<2> as Domain<F, 2>>::initial_capacity_element(), <ConstantLength::<2> as Domain<F, 2>>::layout(WIDTH)))
        }
    }

    pub fn update(&mut self, messages: &[F]) {
        if self.sponge.is_right() {
            self.sponge = Either::Left(NativeSponge::start_absorbing(self.sponge.clone().right().unwrap()));
        }

        let sponge = self.sponge.as_mut().left().unwrap();
        for message in messages {
            sponge.absorb(*message);
        }
    }

    pub fn squeeze(&mut self) -> F {
        match &mut self.sponge {
            Either::Left(sponge) => {
                self.sponge = Either::Right(sponge.clone().finish_absorbing());
            },
            Either::Right(_) => {
                
            },
        }

        match &mut self.sponge {
            Either::Left(_) => unreachable!(),
            Either::Right(sponge) => {
                let out = sponge.squeeze();
                out
            }
        }
    }
}

