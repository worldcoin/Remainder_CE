use ark_std::test_rng;
use rand::Rng;

use super::*;
use crate::halo2curves::bn256::G1 as Bn256;

#[test]
fn test_curve_ops() {
    let zero = Bn256::zero();
    let g = Bn256::generator();
    // check that doubling works
    assert_eq!(PrimeOrderCurve::double(&g), g + g);

    // generator better not be equal to twice itself!
    assert_ne!(g, PrimeOrderCurve::double(&g));

    // check that equality behaves as expected (i.e. it's not just comparing projective coordinates - this test isn't guaranteed to catch it out, of course)
    assert_eq!(PrimeOrderCurve::double(&g) - g, g);

    // check scalar multiplication
    let scalar = Fr::from(4u64);
    assert_eq!(g * scalar, g + g + g + g);
    // also by negative scalars
    assert_eq!(g * scalar.neg(), -(g + g + g + g));

    // check the affine coords of the identity
    // NB if these fail, you've likely upgraded halo2curves, see note in the implementation of PrimeOrderCurve.
    assert_eq!(None, zero.affine_coordinates());
    // .. of the generator
    let (x, y) = g.affine_coordinates().unwrap(); // should not panic (since generator is not the identity!)
    assert_eq!(x, Fq::from(1u64));
    assert_eq!(y, Fq::from(2u64));

    // check the projective coordinates
    // .. of the identity
    let (x, y, z) = zero.projective_coordinates();
    assert_eq!(x, Fq::ZERO);
    assert!(y != Fq::ZERO);
    assert_eq!(z, Fq::ZERO);
    // .. of the generator
    let (_x, _y, z) = g.projective_coordinates();
    assert!(z != Fq::ZERO); // only the identity has z=0

    // check that e.g. AddAssign works
    let mut acc = zero;
    acc += g;
    acc += g;
    assert_eq!(acc, PrimeOrderCurve::double(&g));

    // check that -= works
    let mut acc = zero;
    acc -= g;
    assert_eq!(acc, -g);

    // check that *= works
    let mut acc = g;
    acc *= scalar;
    assert_eq!(acc, g * scalar);

    // check that random works
    let r = <halo2curves::bn256::G1 as PrimeOrderCurve>::random(&mut rand::thread_rng());
    assert_ne!(r, g); // improbable that they are equal, for large groups!
}

#[test]
fn test_bn256_implementation() {
    test_curve_ops();
}

#[test]
/// Ensures that doing `from_bytes_le` and `to_bytes_le` gives the same value.
///
/// TODO(ryancao): Do another manual test to ensure that the integer-interpretable
/// values of the translation between an [Fr] and an [Fq] element are equal
/// (and in particular, equal to the original `u64` value)!
fn test_byte_repr_identity() {
    let mut rng = test_rng();

    (0..100).for_each(|_| {
        let orig_value_u64 = rng.gen::<u64>();
        let orig_value = Fr::from(orig_value_u64);
        let orig_value_bytes = orig_value.to_bytes_le();
        let new_value = Fr::from_bytes_le(&orig_value_bytes);
        assert_eq!(orig_value, new_value);
    })
}
