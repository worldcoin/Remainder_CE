// use remainder_shared_types::curves::PrimeOrderCurve;
// use remainder_shared_types::halo2curves::group::ff::Field;
// use remainder_shared_types::transcript::Transcript;
// use serde::Deserialize;
// use serde::Serialize;
// use std::ops::{Add, Mul};

// /// Append the affine coordinates of a point on the curve to the transcript.
// pub fn append_x_y_to_transcript_single<C: PrimeOrderCurve>(
//     point: &C,
//     transcript: &mut impl Transcript<C::Scalar, C::Base>,
//     x_label: &'static str,
//     y_label: &'static str,
// ) {
//     let (x_coord, y_coord) = point_to_coords(point);
//     transcript
//         .append_base_field_element(x_label, x_coord)
//         .unwrap();
//     transcript
//         .append_base_field_element(y_label, y_coord)
//         .unwrap();
// }

// /// Sample a scalar field element from the transcript.
// /// Uses truncation.
// /// FIXME this works for BN254 since the scalar field is smaller than the base field, but this is not always the case!
// pub fn sample_scalar_field_element<C: PrimeOrderCurve>(
//     transcript: &mut impl Transcript<C::Scalar, C::Base>,
//     label: &'static str,
// ) -> C::Scalar {
//     // assumption: `from_bytes_le` crashes out of range
//     // FIXME check assumption
//     let base_field_element = transcript.get_scalar_field_challenge(label).unwrap();
//     C::Scalar::from_bytes_le(&(base_field_element.to_bytes_le()))
// }

// /// Helper function to convert a point on the curve into its affine coordinates, and to return
// /// (by convention) (0,1) if the point is the identity (and so has no (x,y) affine coordinates).
// pub fn point_to_coords<C: PrimeOrderCurve>(point: &C) -> (C::Base, C::Base) {
//     let affine = point.affine_coordinates();
//     match affine {
//         Some((x, y)) => (x, y),
//         None => (C::Base::zero(), C::Base::one()),
//     }
// }
