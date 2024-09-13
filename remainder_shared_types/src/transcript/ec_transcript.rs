use crate::{curves, HasByteRepresentation};
use ff::PrimeField;
use itertools::Itertools;
use tracing::warn;

use crate::{curves::PrimeOrderCurve, Field};

use super::{
    Operation, ProverTranscript, Transcript, TranscriptReaderError, TranscriptSponge,
    VerifierTranscript,
};

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

pub trait ECProverTranscript<C: PrimeOrderCurve>
where
    Self: ProverTranscript<C::Base>,
{
    fn append_ec_point(&mut self, label: &str, elem: C);

    fn append_ec_points(&mut self, label: &str, elements: &[C]);

    fn append_scalar_point(&mut self, label: &str, elem: C::Scalar);

    fn append_scalar_points(&mut self, label: &str, elements: &[C::Scalar]);

    fn get_scalar_field_challenge(&mut self, label: &str) -> C::Scalar;

    fn get_scalar_field_challenges(&mut self, label: &str, num_elements: usize) -> Vec<C::Scalar>;

    fn get_ec_challenge(&mut self, label: &str) -> C;

    fn get_ec_challenges(&mut self, label: &str, num_elements: usize) -> Vec<C>;
}

impl<C: PrimeOrderCurve, Tr> ECProverTranscript<C> for Tr
where
    Tr: ProverTranscript<C::Base>,
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

    fn append_scalar_point(&mut self, label: &str, elem: C::Scalar) {
        let base_elem = C::Base::from_bytes_le(elem.to_bytes_le());
        self.append(label, base_elem);
    }

    fn append_scalar_points(&mut self, label: &str, elements: &[C::Scalar]) {
        elements.iter().for_each(|elem| {
            let base_elem = C::Base::from_bytes_le(elem.to_bytes_le());
            self.append(label, base_elem);
        });
    }

    /// Literally takes the byte representation of the base field element and
    /// dumps it (TODO: in an unsafe manner! Make this return an error rather
    /// than just panicking) into a scalar field element's representation.
    fn get_scalar_field_challenge(&mut self, label: &str) -> <C as PrimeOrderCurve>::Scalar {
        let base_field_challenge = self.get_challenge(label);
        C::Scalar::from_bytes_le(base_field_challenge.to_bytes_le())
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
                C::Scalar::from_bytes_le(base_field_challenge.to_bytes_le())
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
}

/// The prover-side interface for interacting with a transcript sponge. A
/// `ECTranscriptWriter` acts as a wrapper around a `ECTranscriptSponge` and
/// additionally keeps track of all the append/squeeze operations to be able to
/// generate a serializable `Transcript`.
pub struct ECTranscriptWriter<C: PrimeOrderCurve, T> {
    /// The sponge that this writer is using to append/squeeze elements.
    sponge: T,

    /// A mutable transcript which keeps a record of all the append/squeeze
    /// operations.
    transcript: Transcript<<C::Base as PrimeField>::Repr>,
}

impl<C: PrimeOrderCurve, Sp: TranscriptSponge<C::Base>> ProverTranscript<C::Base>
    for ECTranscriptWriter<C, Sp>
{
    fn append(&mut self, label: &str, elem: C::Base) {
        self.sponge.absorb(elem);
        self.transcript.append_elements(label, &[elem.to_repr()]);
    }

    fn append_elements(&mut self, label: &str, elements: &[C::Base]) {
        if !elements.is_empty() {
            let elements_repr = elements.iter().map(|elem| elem.to_repr()).collect_vec();
            self.sponge.absorb_elements(elements);
            self.transcript.append_elements(label, &elements_repr);
        }
    }

    fn get_challenge(&mut self, label: &str) -> C::Base {
        let challenge = self.sponge.squeeze();
        self.transcript.squeeze_elements(label, 1);
        challenge
    }

    fn get_challenges(&mut self, label: &str, num_elements: usize) -> Vec<C::Base> {
        if num_elements == 0 {
            vec![]
        } else {
            let challenges = self.sponge.squeeze_elements(num_elements);
            self.transcript.squeeze_elements(label, num_elements);
            challenges
        }
    }
}

impl<C: PrimeOrderCurve, Tr: ECTranscriptSponge<C> + Default> ECTranscriptWriter<C, Tr> {
    /// Destructively extract the transcript produced by this writer.
    /// This should be the last operation performed on a `TranscriptWriter`.
    pub fn get_transcript(self) -> Transcript<<C::Base as PrimeField>::Repr> {
        self.transcript
    }

    /// Creates an empty sponge.
    /// `label` is an identifier used for debugging purposes.
    pub fn new(label: &str) -> Self {
        Self {
            sponge: Tr::default(),
            transcript: Transcript::new(label),
        }
    }
}

pub trait ECVerifierTranscript<C: PrimeOrderCurve>
where
    Self: VerifierTranscript<C::Base>,
{
    fn consume_ec_point(&mut self, label: &'static str) -> Result<C, TranscriptReaderError>;

    fn consume_ec_points(
        &mut self,
        label: &'static str,
        num_elements: usize,
    ) -> Result<Vec<C>, TranscriptReaderError>;

    fn consume_scalar_point(
        &mut self,
        label: &'static str,
    ) -> Result<C::Scalar, TranscriptReaderError>;

    fn consume_scalar_points(
        &mut self,
        label: &'static str,
        num_elements: usize,
    ) -> Result<Vec<C::Scalar>, TranscriptReaderError>;

    fn get_scalar_field_challenge(
        &mut self,
        label: &'static str,
    ) -> Result<C::Scalar, TranscriptReaderError>;

    fn get_scalar_field_challenges(
        &mut self,
        label: &'static str,
        num_elements: usize,
    ) -> Result<Vec<C::Scalar>, TranscriptReaderError>;

    fn get_ec_challenge(&mut self, label: &'static str) -> Result<C, TranscriptReaderError>;

    fn get_ec_challenges(
        &mut self,
        label: &'static str,
        num_elements: usize,
    ) -> Result<Vec<C>, TranscriptReaderError>;
}

impl<C: PrimeOrderCurve, Tr> ECVerifierTranscript<C> for Tr
where
    Tr: VerifierTranscript<C::Base>,
{
    fn consume_ec_point(&mut self, label: &'static str) -> Result<C, TranscriptReaderError> {
        let points = self.consume_elements(label, 2)?;
        Ok(C::from_xy(points[0], points[1]))
    }

    fn consume_ec_points(
        &mut self,
        label: &'static str,
        num_elements: usize,
    ) -> Result<Vec<C>, TranscriptReaderError> {
        let points = self.consume_elements(label, 2 * num_elements)?;
        Ok(points
            .chunks(2)
            .map(|points| C::from_xy(points[0], points[1]))
            .collect())
    }

    fn consume_scalar_point(
        &mut self,
        label: &'static str,
    ) -> Result<C::Scalar, TranscriptReaderError> {
        let base_point = self.consume_element(label)?;
        let scalar_point = C::Scalar::from_bytes_le(base_point.to_bytes_le());
        Ok(scalar_point)
    }

    fn consume_scalar_points(
        &mut self,
        label: &'static str,
        num_elements: usize,
    ) -> Result<Vec<C::Scalar>, TranscriptReaderError> {
        let points = self.consume_elements(label, num_elements)?;
        Ok(points
            .iter()
            .map(|point| {
                let scalar_point = C::Scalar::from_bytes_le(point.to_bytes_le());
                scalar_point
            })
            .collect_vec())
    }

    /// Literally takes the byte representation of the base field element and
    /// dumps it (TODO: in an unsafe manner! Make this return an error rather
    /// than just panicking) into a scalar field element's representation.
    fn get_scalar_field_challenge(
        &mut self,
        label: &'static str,
    ) -> Result<<C as curves::PrimeOrderCurve>::Scalar, TranscriptReaderError> {
        let base_field_challenge = self.get_challenge(label)?;
        Ok(C::Scalar::from_bytes_le(base_field_challenge.to_bytes_le()))
    }

    fn get_scalar_field_challenges(
        &mut self,
        label: &'static str,
        num_elements: usize,
    ) -> Result<Vec<<C as curves::PrimeOrderCurve>::Scalar>, TranscriptReaderError> {
        let base_field_challenges = self.get_challenges(label, num_elements)?;
        Ok(base_field_challenges
            .iter()
            .map(|base_field_challenge| {
                C::Scalar::from_bytes_le(base_field_challenge.to_bytes_le())
            })
            .collect())
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
    fn get_ec_challenge(&mut self, label: &'static str) -> Result<C, TranscriptReaderError> {
        let x_coord_label = label.to_string() + ": x-coord";
        let x_coord = self.get_challenge(Box::leak(x_coord_label.into_boxed_str()))?;

        let y_coord_sign_elem_label = label.to_string() + ": y-coord sign elem";
        let y_coord_sign_elem =
            self.get_challenge(Box::leak(y_coord_sign_elem_label.into_boxed_str()))?;
        let y_coord_sign = y_coord_sign_elem.to_bytes_le()[0] & 1;

        Ok(C::from_x_and_sign_y(x_coord, y_coord_sign))
    }

    /// Generates two base field elements for each element requested, by FIRST
    /// generating ALL of the x-coords and AFTERWARDS generating ALL of the
    /// base field elements whose parity determines the sign of the corresponding
    /// y-coord.
    ///
    /// WARNING/TODO(ryancao): SEE WARNING FOR [Self::get_ec_challenge]!!!
    fn get_ec_challenges(
        &mut self,
        label: &str,
        num_elements: usize,
    ) -> Result<Vec<C>, TranscriptReaderError> {
        let x_coord_label = label.to_string() + ": x-coords";
        let y_coord_sign_elem_label = label.to_string() + ": y-coord sign elems";

        let x_coords =
            self.get_challenges(Box::leak(x_coord_label.into_boxed_str()), num_elements)?;
        let y_coord_sign_elems = self.get_challenges(
            Box::leak(y_coord_sign_elem_label.into_boxed_str()),
            num_elements,
        )?;

        let y_coord_signs = y_coord_sign_elems
            .iter()
            .map(|y_coord_sign_elem| y_coord_sign_elem.to_bytes_le()[0] & 1);

        Ok(x_coords
            .into_iter()
            .zip(y_coord_signs)
            .map(|(x_coord, y_coord_sign)| C::from_x_and_sign_y(x_coord, y_coord_sign))
            .collect())
    }
}

pub struct ECTranscriptReader<C: PrimeOrderCurve, T> {
    /// The sponge that this reader is wrapping around.
    sponge: T,

    /// The transcript that this reader is using to consume elements from and
    /// verify the order of operations is valid.
    transcript: Transcript<<C::Base as PrimeField>::Repr>,

    /// An internal state representing the position of the next operation on the
    /// transcript.
    next_element: (usize, usize),
}

impl<C: PrimeOrderCurve, T: Default> ECTranscriptReader<C, T> {
    /// Generate a new `TranscriptReader` to operate on a given `transcript`.
    pub fn new(transcript: Transcript<<C::Base as PrimeField>::Repr>) -> Self {
        Self {
            sponge: T::default(),
            transcript,
            next_element: (0, 0),
        }
    }

    /// Internal method used to advance the internal state to the next
    /// operation.
    fn advance_indices(&mut self) -> Result<(), TranscriptReaderError> {
        let (operation_idx, element_idx) = self.next_element;

        match self.transcript.transcript_operations.get(operation_idx) {
            None => Err(TranscriptReaderError::InternalIndicesError),
            Some(Operation::Append(_, v)) => {
                if element_idx + 1 >= v.len() {
                    self.next_element = (operation_idx + 1, 0);
                } else {
                    self.next_element = (operation_idx, element_idx + 1);
                }
                Ok(())
            }
            Some(Operation::Squeeze(_, num_elements)) => {
                if element_idx + 1 >= *num_elements {
                    self.next_element = (operation_idx + 1, 0);
                } else {
                    self.next_element = (operation_idx, element_idx + 1);
                }
                Ok(())
            }
        }
    }
}

impl<
        C: PrimeOrderCurve,
        F: Field<Repr = <C::Base as PrimeField>::Repr>,
        T: TranscriptSponge<F>,
    > VerifierTranscript<F> for ECTranscriptReader<C, T>
{
    /// Reads off a single element from the transcript and returns it if
    /// successful.
    /// The operation can fail with:
    /// * `TranscriptReaderError::ConsumeError`: if there are no more elements
    ///   to consume or if a squeeze was expected.
    /// * `TranscriptReaderError::InternalIndicesError`: if the internal state
    ///   is invalid. This is an internal error which should never appear under
    ///   normal circumstances.
    /// TODO(Makis): Consider turning the internal error into a panic.
    ///
    /// The `label` is used for sanity checking against the label that was used
    /// by the `TranscriptWriter` for the corresponding operation. If the labels
    /// don't match, a trace warning message is produced, but the caller is not
    /// otherwise notified of this discrepancy.
    fn consume_element(&mut self, label: &'static str) -> Result<F, TranscriptReaderError> {
        let (operation_idx, element_idx) = self.next_element;

        match self.transcript.transcript_operations.get(operation_idx) {
            None => Err(TranscriptReaderError::ConsumeError),
            Some(Operation::Squeeze(_, _)) => Err(TranscriptReaderError::ConsumeError),
            Some(Operation::Append(expected_label, v)) => {
                if label != expected_label {
                    // dbg!("Label mismatch on TranscriptReader consume_element. Expected \"{}\" but instead got \"{}\".", expected_label, label);

                    warn!("Label mismatch on TranscriptReader consume_element. Expected \"{}\" but instead got \"{}\".", expected_label, label);
                }
                match v.get(element_idx) {
                    None => Err(TranscriptReaderError::InternalIndicesError),
                    Some(&element) => {
                        let element = F::from_repr(element).unwrap();
                        self.advance_indices()?;
                        self.sponge.absorb(element);
                        Ok(element)
                    }
                }
            }
        }
    }

    /// A multi-element version of the `consume_element` method. Reads off a
    /// sequence of `num_elements` elements from the transcript and returns a
    /// vector of them if successful.
    /// The operation can fail with:
    /// * `TranscriptReaderError::ConsumeError`: if less than `num_elements`
    ///   elements remain in the transcript or if a squeeze operation was
    ///   expected to occur at any point before the consumption of
    ///   `num_elements` elements.
    /// * `TranscriptReaderError::InternalIndicesError`: if the internal state
    ///   is invalid. This is an internal error which should never appear under
    ///   normal circumstances.
    /// TODO(Makis): Consider turning the internal error into a panic.
    ///
    /// The `label` is used for sanity checking against the label that was used
    /// by the `TranscriptWriter` for the corresponding operations. In
    /// particular, the `TranscriptWriter` may have appended either a sequence
    /// of elements using `TranscriptWritter::append_elements` or may have
    /// called `TranscriptWritter::append` multiple times. Both scenarios are
    /// valid and in both cases, `label` should match with the corresponding
    /// labels used on the writer side. If there is a label mismatch for any of
    /// the `num_elements` elements, a trace warning message is produced, but
    /// the caller is not otherwise notified of this discrepancy.
    fn consume_elements(
        &mut self,
        label: &'static str,
        num_elements: usize,
    ) -> Result<Vec<F>, TranscriptReaderError> {
        (0..num_elements)
            .map(|_| self.consume_element(label))
            .collect()
    }

    /// Squeezes the sponge once and returns a single element if successful.
    /// The operation can fail with:
    /// * `TranscriptReaderError::SqueezeError`: if a squeeze is requested at a
    ///   time when either a consume operation was expected or no more
    ///   operations were expected.
    /// * `TranscriptReaderError::InternalIndicesError`: if the internal state
    ///   is invalid. This is an internal error which should never appear under
    ///   normal circumstances.
    /// TODO(Makis): Consider turning the internal error into a panic.
    ///
    /// The `label` is used for sanity checking against the label that was used
    /// by the `TranscriptWriter` for the corresponding operation. If the labels
    /// don't match, a trace warning message is produced, but the caller is not
    /// otherwise notified of this discrepancy.
    fn get_challenge(&mut self, label: &'static str) -> Result<F, TranscriptReaderError> {
        let (operation_idx, element_idx) = self.next_element;

        match self.transcript.transcript_operations.get(operation_idx) {
            None => Err(TranscriptReaderError::SqueezeError),
            Some(Operation::Append(_, _)) => Err(TranscriptReaderError::SqueezeError),
            Some(Operation::Squeeze(expected_label, num_elements)) => {
                if label != expected_label {
                    warn!("Label mismatch on TranscriptReader get_challenge. Expected \"{}\" but instead got \"{}\".", expected_label, label);
                }
                if element_idx >= *num_elements {
                    Err(TranscriptReaderError::SqueezeError)
                } else {
                    self.advance_indices()?;
                    Ok(self.sponge.squeeze())
                }
            }
        }
    }

    /// Squeezes the sponge `num_elements` times and returns a vector of the
    /// resulting elements if successful.
    /// The operation can fail with:
    /// * `TranscriptReaderError::SqueezeError`: if any of the squeeze
    ///   operations requested does not correspond to a squeeze operation
    ///   performed by the `TranscriptWriter` that produced the transcript.
    /// * `TranscriptReaderError::InternalIndicesError`: if the internal state
    ///   is invalid. This is an internal error which should never appear under
    ///   normal circumstances.
    /// TODO(Makis): Consider turning the internal error into a panic.
    ///
    /// The `label` is used for sanity checking against the label that was used
    /// by the `TranscriptWriter` for the corresponding operations. In
    /// particular, the `TranscriptWriter` may have squeezed either a sequence
    /// of elements using `TranscriptWriter::get_challenges` or may have called
    /// `TranscriptWriter::get_challenge` multiple times. Both scenarios are
    /// valid and in both cases, `label` should match with the corresponding
    /// labels used on the writer side. If there is a label mismatch for any of
    /// the `num_elements` elements, a trace warning message is produced, but
    /// the caller is not otherwise notified of this discrepancy.
    fn get_challenges(
        &mut self,
        label: &'static str,
        num_elements: usize,
    ) -> Result<Vec<F>, TranscriptReaderError> {
        (0..num_elements)
            .map(|_| self.get_challenge(label))
            .collect()
    }
}
