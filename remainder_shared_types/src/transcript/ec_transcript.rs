use std::fmt::Display;

use itertools::Itertools;

use crate::{field::HasByteRepresentation, transcript::utils::sha256_hash_chain_on_field_elems};

use crate::curves::PrimeOrderCurve;

use super::{ProverTranscript, Transcript, TranscriptSponge};

pub trait ECTranscriptSponge<C: PrimeOrderCurve>: TranscriptSponge<C::Base> {
    /// Absorb a single field element `elem`.
    fn absorb_ec_point(&mut self, elem: C);

    /// Absorb a list of field elements sequentially.
    fn absorb_ec_points(&mut self, elements: &[C]);
}

impl<C, Tr> ECTranscriptSponge<C> for Tr
where
    C: PrimeOrderCurve,
    Tr: TranscriptSponge<C::Base>,
{
    fn absorb_ec_point(&mut self, elem: C) {
        let (x, y) = elem.affine_coordinates().unwrap();
        self.absorb(x);
        self.absorb(y);
    }

    fn absorb_ec_points(&mut self, elements: &[C]) {
        elements.iter().for_each(|elem| {
            let (x, y) = elem.affine_coordinates().unwrap();
            self.absorb(x);
            self.absorb(y);
        });
    }
}

/// The purposes of this trait is simply to hide (i.e. abstract away) the generic for the sponge
/// type from the prover and verifier code.
pub trait ECTranscriptTrait<C: PrimeOrderCurve>: Display {
    fn append_ec_point(&mut self, label: &str, elem: C);

    fn append_ec_points(&mut self, label: &str, elements: &[C]);

    fn append_scalar_field_elem(&mut self, label: &str, elem: C::Scalar);

    fn append_scalar_field_elems(&mut self, label: &str, elements: &[C::Scalar]);

    /// This function absorbs elliptic curve points as individual base field
    /// elements, and additionally absorbs the hash chain digest of the
    /// base field elements.
    fn append_input_ec_points(&mut self, label: &str, elements: Vec<C>);

    /// This function absorbs scalar field elements into the transcript sponge,
    /// and additionally absorbs the hash chain digest of these elements.
    fn append_input_scalar_field_elems(&mut self, label: &str, elements: &[C::Scalar]);

    /// This function absorbs base field elements into the transcript sponge.
    fn append_base_field_elems(&mut self, label: &str, elements: &[C::Base]);

    fn get_scalar_field_challenge(&mut self, label: &str) -> C::Scalar;

    fn get_scalar_field_challenges(&mut self, label: &str, num_elements: usize) -> Vec<C::Scalar>;

    fn get_ec_challenge(&mut self, label: &str) -> C;

    fn get_ec_challenges(&mut self, label: &str, num_elements: usize) -> Vec<C>;
}

/// A transcript that operates over the base field of a prime-order curve, while also allowing for
/// the absorption and sampling of scalar field elements (and of course, EC points).
pub struct ECTranscript<C: PrimeOrderCurve, T> {
    /// The sponge that this writer is using to append/squeeze elements.
    sponge: T,

    /// A mutable transcript which keeps a record of all the append/squeeze
    /// operations.
    transcript: Transcript<C::Base>,

    /// Whether to print debug information.
    debug: bool,
}

impl<C: PrimeOrderCurve, Tr: ECTranscriptSponge<C> + Default> ECTranscript<C, Tr> {
    /// Destructively extract the transcript produced by this writer.
    /// This should be the last operation performed on a `TranscriptWriter`.
    pub fn get_transcript(self) -> Transcript<C::Base> {
        self.transcript
    }

    /// Creates an empty sponge.
    /// `label` is an identifier used for debugging purposes.
    pub fn new(label: &str) -> Self {
        Self {
            sponge: Tr::default(),
            transcript: Transcript::new(label),
            debug: false,
        }
    }

    /// Creates an empty sponge in debug mode (i.e. with debug information printed).
    /// `label` is an identifier used for debugging purposes.
    pub fn new_with_debug(label: &str) -> Self {
        Self {
            sponge: Tr::default(),
            transcript: Transcript::new(label),
            debug: true,
        }
    }
}

impl<C: PrimeOrderCurve, Tr: ECTranscriptSponge<C> + Default> ECTranscriptTrait<C>
    for ECTranscript<C, Tr>
{
    fn append_ec_point(&mut self, label: &str, elem: C) {
        let (x_coord, y_coord) = elem.affine_coordinates().unwrap();
        self.append_elements(label, &[x_coord, y_coord]);
    }

    fn append_ec_points(&mut self, label: &str, elements: &[C]) {
        elements.iter().for_each(|elem| {
            let (x_coord, y_coord) = elem.affine_coordinates().unwrap();
            self.append_elements(label, &[x_coord, y_coord]);
        });
    }

    fn append_scalar_field_elem(&mut self, label: &str, elem: C::Scalar) {
        let base_elem = C::Base::from_bytes_le(&elem.to_bytes_le());
        self.append(label, base_elem);
    }

    fn append_scalar_field_elems(&mut self, label: &str, elements: &[C::Scalar]) {
        elements.iter().for_each(|elem| {
            let base_elem = C::Base::from_bytes_le(&elem.to_bytes_le());
            self.append(label, base_elem);
        });
    }

    /// Literally takes the byte representation of the base field element and
    /// dumps it (TODO: in an unsafe manner! Make this return an error rather
    /// than just panicking) into a scalar field element's representation.
    fn get_scalar_field_challenge(&mut self, label: &str) -> <C as PrimeOrderCurve>::Scalar {
        let base_field_challenge = self.get_challenge(label);
        C::Scalar::from_bytes_le(&base_field_challenge.to_bytes_le())
    }

    fn get_scalar_field_challenges(
        &mut self,
        label: &str,
        num_elements: usize,
    ) -> Vec<<C as PrimeOrderCurve>::Scalar> {
        let base_field_challenges = self.get_challenges(label, num_elements);
        base_field_challenges
            .iter()
            .map(|base_field_challenge| {
                C::Scalar::from_bytes_le(&base_field_challenge.to_bytes_le())
            })
            .collect()
    }

    /// Generates two base field elements, and uses only the parity of the second
    /// to determine the actual `y`-coordinate to be used.
    ///
    /// WARNING/TODO(ryancao): USING THIS FUNCTION `num_elements` TIMES WILL
    /// NOT PRODUCE THE SAME EC CHALLENGES AS CALLING [Self::get_ec_challenges]
    /// WITH `num_elements` AS A PARAMETER!!!
    ///
    /// IN PARTICULAR, THIS FUNCTION
    /// GENERATES (x, y) ELEMENTS IN INDIVIDUAL PAIRS, WHILE THE
    /// [Self::get_ec_challenges] FUNCTION GENERATES (x, y) ELEMENTS BY FIRST
    /// GENERATING ALL x-coordinates AND THEN GENERATING ALL ELEMENTS DETERMINING
    /// THE PARITY OF THE CORRESPONDING y-coordinates.
    fn get_ec_challenge(&mut self, label: &str) -> C {
        let x_coord_label = label.to_string() + ": x-coord";
        let x_coord = self.get_challenge(&x_coord_label);

        let y_coord_sign_elem_label = label.to_string() + ": y-coord sign elem";
        let y_coord_sign_elem = self.get_challenge(&y_coord_sign_elem_label);
        let y_coord_sign = y_coord_sign_elem.to_bytes_le()[0] & 1;

        C::from_x_and_sign_y(x_coord, y_coord_sign)
    }

    /// Generates two base field elements for each element requested, by FIRST
    /// generating ALL of the x-coords and AFTERWARDS generating ALL of the
    /// base field elements whose parity determines the sign of the corresponding
    /// y-coord.
    ///
    /// WARNING/TODO(ryancao): SEE WARNING FOR [Self::get_ec_challenge]!!!
    fn get_ec_challenges(&mut self, label: &str, num_elements: usize) -> Vec<C> {
        let x_coord_label = label.to_string() + ": x-coords";
        let y_coord_sign_elem_label = label.to_string() + ": y-coord sign elems";

        let x_coords = self.get_challenges(&x_coord_label, num_elements);
        let y_coord_sign_elems = self.get_challenges(&y_coord_sign_elem_label, num_elements);

        let y_coord_signs = y_coord_sign_elems
            .iter()
            .map(|y_coord_sign_elem| y_coord_sign_elem.to_bytes_le()[0] & 1);

        x_coords
            .into_iter()
            .zip(y_coord_signs)
            .map(|(x_coord, y_coord_sign)| C::from_x_and_sign_y(x_coord, y_coord_sign))
            .collect()
    }

    fn append_input_ec_points(&mut self, label: &str, elements: Vec<C>) {
        // We compute the list of all x-coordinates interwoven with all
        // y-coordinates, i.e. [x_1, y_1, x_2, y_2, ...]
        let elements_interwoven_x_y_coords = elements
            .into_iter()
            .map(|ec_element| ec_element.affine_coordinates().unwrap())
            .flat_map(|(x, y)| vec![x, y])
            .collect_vec();
        // We then compute the hash chain digest of the list.
        let hash_chain_digest = sha256_hash_chain_on_field_elems(&elements_interwoven_x_y_coords);
        // We first absorb the interwoven x/y coordinates, then the hash chain (both as native base field elements).
        self.append_base_field_elems(label, &elements_interwoven_x_y_coords);
        self.append_base_field_elems(label, &hash_chain_digest);
    }

    fn append_input_scalar_field_elems(
        &mut self,
        label: &str,
        elements: &[<C as PrimeOrderCurve>::Scalar],
    ) {
        // First, compute the hash chain digest of the elements.
        let hash_chain_digest = sha256_hash_chain_on_field_elems(elements);
        // Next, we simply absorb the elements and then the hash chain digest of the elements.
        self.append_scalar_field_elems(label, elements);
        self.append_scalar_field_elems(label, &hash_chain_digest);
    }

    fn append_base_field_elems(&mut self, label: &str, elements: &[<C as PrimeOrderCurve>::Base]) {
        // This is a simple wrapper around the `ProverTranscript<C::Base>` trait.
        self.append_elements(label, elements);
    }
}

impl<C: PrimeOrderCurve, Sp: ECTranscriptSponge<C>> std::fmt::Display for ECTranscript<C, Sp> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.transcript.fmt(f)
    }
}

impl<C: PrimeOrderCurve, Sp: TranscriptSponge<C::Base>> ProverTranscript<C::Base>
    for ECTranscript<C, Sp>
{
    fn append(&mut self, label: &str, elem: C::Base) {
        if self.debug {
            println!("Appending element (\"{label}\"): {elem:?}");
        }
        self.sponge.absorb(elem);
        self.transcript.append_elements(label, &[elem]);
    }

    fn append_elements(&mut self, label: &str, elements: &[C::Base]) {
        if !elements.is_empty() {
            if self.debug {
                println!(
                    "Appending {} elements (\"{}\"): [{:?}, .., ]",
                    elements.len(),
                    label,
                    elements[0]
                );
            }
            self.sponge.absorb_elements(elements);
            self.transcript.append_elements(label, elements);
        }
    }

    fn get_challenge(&mut self, label: &str) -> C::Base {
        let challenge = self.sponge.squeeze();
        self.transcript.squeeze_elements(label, 1);
        if self.debug {
            println!("Squeezing challenge (\"{label}\"): {challenge:?}");
        }
        challenge
    }

    fn get_challenges(&mut self, label: &str, num_elements: usize) -> Vec<C::Base> {
        if num_elements == 0 {
            vec![]
        } else {
            let challenges = self.sponge.squeeze_elements(num_elements);
            self.transcript.squeeze_elements(label, num_elements);
            if self.debug {
                println!(
                    "Squeezing {} challenges (\"{}\"): [{:?}, .., ]",
                    num_elements, label, challenges[0]
                );
            }
            challenges
        }
    }

    fn append_input_elements(&mut self, label: &str, elements: &[C::Base]) {
        let hash_chain_digest = sha256_hash_chain_on_field_elems(elements);
        self.transcript
            .append_input_elements(label, elements, &hash_chain_digest);
    }

    fn get_extension_field_challenge<E: crate::field::P3StyleExtensionField<C::Base>>(
        &mut self,
        label: &str,
    ) -> E {
        self.transcript.squeeze_elements(label, E::N_COEFF);
        let base_field_coeffs = self.sponge.squeeze_elements(E::N_COEFF);
        E::from_basis_elem_coeffs(&base_field_coeffs)
    }

    fn append_extension_field_elements<E: crate::field::P3StyleExtensionField<C::Base>>(
        &mut self,
        label: &str,
        elements: &[E],
    ) {
        // First, convert to base field coefficients
        let base_field_coeffs = elements
            .iter()
            .flat_map(|extension_field_elem| extension_field_elem.to_basis_elem_coeffs())
            .collect_vec();
        // Then, absorb as usual
        self.sponge.absorb_elements(&base_field_coeffs);
        self.transcript.append_elements(label, &base_field_coeffs);
    }

    /// TODO(ryancao): We should *also* compute scalar field challenges from this.
    fn get_extension_field_challenges<E: crate::field::P3StyleExtensionField<C::Base>>(
        &mut self,
        label: &str,
        num_elements: usize,
    ) -> Vec<E> {
        // TODO: Should we log this within the `self.transcript`?
        if num_elements == 0 {
            vec![]
        } else {
            (0..num_elements)
                .map(|ext_field_elem_number| {
                    // For each extension field element we wish to sample,
                    // simply squeeze the corresponding number of base field
                    // coefficients from the transcript.
                    let base_field_coeffs = self.sponge.squeeze_elements(E::N_COEFF);
                    self.transcript.squeeze_elements(
                        &format!(
                            "{}: extension field element number {}",
                            label, ext_field_elem_number
                        ),
                        E::N_COEFF,
                    );
                    E::from_basis_elem_coeffs(&base_field_coeffs)
                })
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ECTranscript, ECTranscriptTrait};
    use crate::{
        curves::PrimeOrderCurve,
        transcript::{poseidon_sponge::PoseidonSponge, ProverTranscript},
        Base, Bn256Point, Scalar,
    };
    use ark_std::test_rng;
    use itertools::Itertools;
    use rand::Rng;

    /// A basic test which ensures that the transcript challenge which would
    /// have been derived from simply calling `append_scalar_field_elems` is
    /// NOT the same as the one which would've been derived from calling
    /// `append_input_scalar_field_elems`.
    #[test]
    fn test_ec_scalar_input_hash_soundness() {
        // Create transcripts to compare
        let mut ec_transcript_1: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
            ECTranscript::new("test ec_transcript_1");
        let mut ec_transcript_2: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
            ECTranscript::new("test ec_transcript_2");

        // Generate random input elements
        let mut rng = test_rng();
        let random_input_elems = (0..1000)
            .map(|_| Scalar::from(rng.gen::<u64>()))
            .collect_vec();
        ec_transcript_1
            .append_scalar_field_elems("test append scalar field elems", &random_input_elems);
        ec_transcript_2.append_input_scalar_field_elems(
            "test append input scalar field elems",
            &random_input_elems,
        );

        // Generate challenges from each and assert they are not the same
        assert_ne!(
            ec_transcript_1.get_challenge("get challenge 1"),
            ec_transcript_2.get_challenge("get challenge 2")
        );
    }

    /// A basic test which ensures that the transcript challenge which would
    /// have been derived from simply calling `append_ec_points` is
    /// NOT the same as the one which would've been derived from calling
    /// `append_input_ec_points`.
    #[test]
    fn test_ec_point_input_hash_soundness() {
        // Create transcripts to compare
        let mut ec_transcript_1: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
            ECTranscript::new("test ec_transcript_1");
        let mut ec_transcript_2: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
            ECTranscript::new("test ec_transcript_2");

        // Generate random input elements
        let mut rng = test_rng();
        let random_input_ec_points = (0..1000)
            .map(|_| Bn256Point::random(&mut rng))
            .collect_vec();
        ec_transcript_1.append_ec_points("test append ec elems", &random_input_ec_points);
        ec_transcript_2
            .append_input_ec_points("test append input ec elems", random_input_ec_points);

        // Generate challenges from each and assert they are not the same
        assert_ne!(
            ec_transcript_1.get_challenge("get challenge 1"),
            ec_transcript_2.get_challenge("get challenge 2")
        );
    }
}
