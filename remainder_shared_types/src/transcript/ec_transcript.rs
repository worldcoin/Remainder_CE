use ff::PrimeField;
use itertools::Itertools;
use tracing::warn;

use crate::FieldExt;
use crate::ec::CurveAffine;

use super::{
    Operation, ProverTranscript, Transcript, TranscriptReaderError, TranscriptSponge,
    VerifierTranscript,
};

pub trait ECTranscriptSponge<C: CurveAffine>:
    TranscriptSponge<C::Base> + TranscriptSponge<C::ScalarExt>
{
    /// Absorb a single field element `elem`.
    fn absorb_ec_point(&mut self, elem: C);

    /// Absorb a list of field elements sequentially.
    fn absorb_ec_points(&mut self, elements: &[C]);
}

impl<C, Tr> ECTranscriptSponge<C> for Tr
where
    C: CurveAffine,
    Tr: TranscriptSponge<C::Base> + TranscriptSponge<C::ScalarExt>,
    C::Base: FieldExt,
    C::ScalarExt: FieldExt,
{
    fn absorb_ec_point(&mut self, elem: C) {
        let coords = elem.coordinates().unwrap();
        self.absorb(*coords.x());
        self.absorb(*coords.y());
    }

    fn absorb_ec_points(&mut self, elements: &[C]) {
        elements.iter().for_each(|elem| {
            let coords = elem.coordinates().unwrap();
            self.absorb(*coords.x());
            self.absorb(*coords.y());
        });
    }
}

pub trait ECProverTranscript<C: CurveAffine>
where
    Self: ProverTranscript<C::Base> + ProverTranscript<C::ScalarExt>,
{
    fn append_ec_point(&mut self, label: &str, elem: C);

    fn append_ec_points(&mut self, label: &str, elements: &[C]);
}

impl<C: CurveAffine, Tr> ECProverTranscript<C> for Tr
where
    Tr: ProverTranscript<C::Base> + ProverTranscript<C::ScalarExt>,
{
    fn append_ec_point(&mut self, label: &str, elem: C) {
        let coords = elem.coordinates().unwrap();
        self.append_elements(label, &[*coords.x(), *coords.y()]);
    }

    fn append_ec_points(&mut self, label: &str, elements: &[C]) {
        elements.iter().for_each(|elem| {
            let coords = elem.coordinates().unwrap();
            self.append_elements(label, &[*coords.x(), *coords.y()]);
        });
    }
}

/// The prover-side interface for interacting with a transcript sponge. A
/// `ECTranscriptWriter` acts as a wrapper around a `ECTranscriptSponge` and
/// additionally keeps track of all the append/squeeze operations to be able to
/// generate a serializable `Transcript`.
pub struct ECTranscriptWriter<C: CurveAffine, T> {
    /// The sponge that this writer is using to append/squeeze elements.
    sponge: T,

    /// A mutable transcript which keeps a record of all the append/squeeze
    /// operations.
    transcript: Transcript<<C::Base as PrimeField>::Repr>,
}

impl<
        C: CurveAffine,
        F: FieldExt<Repr = <C::Base as PrimeField>::Repr>,
        Sp: TranscriptSponge<F>,
    > ProverTranscript<F> for ECTranscriptWriter<C, Sp>
{
    fn append(&mut self, label: &str, elem: F) {
        self.sponge.absorb(elem);
        self.transcript.append_elements(label, &[elem.to_repr()]);
    }

    fn append_elements(&mut self, label: &str, elements: &[F]) {
        if !elements.is_empty() {
            let elements_repr = elements.iter().map(|elem| elem.to_repr()).collect_vec();
            self.sponge.absorb_elements(elements);
            self.transcript.append_elements(label, &elements_repr);
        }
    }

    fn get_challenge(&mut self, label: &str) -> F {
        let challenge = self.sponge.squeeze();
        self.transcript.squeeze_elements(label, 1);
        challenge
    }

    fn get_challenges(&mut self, label: &str, num_elements: usize) -> Vec<F> {
        if num_elements == 0 {
            vec![]
        } else {
            let challenges = self.sponge.squeeze_elements(num_elements);
            self.transcript.squeeze_elements(label, num_elements);
            challenges
        }
    }
}

impl<C: CurveAffine, Tr: ECTranscriptSponge<C>> ECTranscriptWriter<C, Tr> {
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

pub trait ECVerifierTranscript<C: CurveAffine>
where
    Self: VerifierTranscript<C::Base> + VerifierTranscript<C::ScalarExt>,
{
    fn consume_ec_point(&mut self, label: &'static str) -> Result<C, TranscriptReaderError>;

    fn consume_ec_points(
        &mut self,
        label: &'static str,
        num_elements: usize,
    ) -> Result<Vec<C>, TranscriptReaderError>;
}

impl<C: CurveAffine, Tr> ECVerifierTranscript<C> for Tr
where
    Tr: VerifierTranscript<C::Base> + VerifierTranscript<C::ScalarExt>,
{
    fn consume_ec_point(&mut self, label: &'static str) -> Result<C, TranscriptReaderError> {
        let points = self.consume_elements(label, 2)?;
        Ok(C::from_xy(points[0], points[1]).unwrap())
    }

    fn consume_ec_points(
        &mut self,
        label: &'static str,
        num_elements: usize,
    ) -> Result<Vec<C>, TranscriptReaderError> {
        let points = self.consume_elements(label, 2 * num_elements)?;
        Ok(points
            .chunks(2)
            .map(|points| C::from_xy(points[0], points[1]).unwrap())
            .collect())
    }
}

pub struct ECTranscriptReader<C: CurveAffine, T> {
    /// The sponge that this reader is wrapping around.
    sponge: T,

    /// The transcript that this reader is using to consume elements from and
    /// verify the order of operations is valid.
    transcript: Transcript<<C::Base as PrimeField>::Repr>,

    /// An internal state representing the position of the next operation on the
    /// transcript.
    next_element: (usize, usize),
}

impl<C: CurveAffine, T: Default> ECTranscriptReader<C, T> {
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

impl<C: CurveAffine, F: FieldExt<Repr = <C::Base as PrimeField>::Repr>, T: TranscriptSponge<F>>
    VerifierTranscript<F> for ECTranscriptReader<C, T>
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
