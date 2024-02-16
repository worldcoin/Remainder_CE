//!A type that is responsible for FS over the interative version of the protocol

// use core::num;
// use std::collections::btree_map::OccupiedError;

use itertools::Itertools;
use poseidon::Poseidon;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{trace, warn};

use crate::FieldExt;

pub mod poseidon_transcript;

// ///An error representing the things that can go wrong when working with a Transcript
// #[derive(Error, Debug, Clone)]
// pub enum TranscriptError {
//     #[error("The challenges generated don't match challenges given!")]
//     TranscriptMatchError,
// }
//
// ///A type that is responsible for FS over the interative version of the protocol
// pub trait Transcript<F>: Clone {
//     ///Create an empty transcript
//     fn new(label: &'static str) -> Self;
//
//     ///Append a single field element to the transcript
//     fn append_field_element(
//         &mut self,
//         label: &'static str,
//         element: F,
//     ) -> Result<(), TranscriptError>;
//
//     ///Append a list of field elements to the transcript
//     fn append_field_elements(
//         &mut self,
//         label: &'static str,
//         elements: &[F],
//     ) -> Result<(), TranscriptError>;
//
//     ///Generate a random challenge and add it to the transcript
//     fn get_challenge(&mut self, label: &'static str) -> Result<F, TranscriptError>;
//
//     ///Generate a list of random challenges and add it to the transcript
//     fn get_challenges(
//         &mut self,
//         label: &'static str,
//         len: usize,
//     ) -> Result<Vec<F>, TranscriptError>;
// }

/// A `TranscriptSponge` provides the basic interface for a cryptographic sponge
/// operating on field elements. It is typically used for representing the
/// transcript of an interactive protocol.
pub trait TranscriptSponge<F: FieldExt>: Clone + Send + Sync {
    /// Create an empty transcript sponge.
    fn new() -> Self;

    /// Absorb a single field element `elem`.
    fn absorb(&mut self, elem: F);

    /// Absorb a list of field elements sequentially.
    fn absorb_elements(&mut self, elements: &[F]);

    /// Generate a field element by squeezing the sponge. Internal state is
    /// modified.
    fn squeeze(&mut self) -> F;

    /// Generate a sequence of field elements by squeezing the sponge
    /// `num_elements` times.
    fn squeeze_elements(&mut self, num_elements: usize) -> Vec<F>;
}

/// Describes an elementary operation on a transcript.
#[derive(PartialEq, Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "F: FieldExt")]
enum Operation<F: FieldExt> {
    /// An append operation consists of a label (used for debugging purposes)
    /// and a vector of field elements to be appended in order to the
    /// transcript.
    Append(String, Vec<F>),
    /// A squeeze operation consists of a label (used for debugging purposes)
    /// and a counter of how many elements are to be squeezed from the sponge.
    Squeeze(String, usize),
}

#[derive(PartialEq, Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "F: FieldExt")]
pub struct Transcript<F: FieldExt> {
    label: String,
    appended_elements: Vec<Operation<F>>,
}

impl<F: FieldExt> Transcript<F> {
    pub fn new(label: &str) -> Self {
        Self {
            label: String::from(label),
            appended_elements: vec![],
        }
    }

    pub fn append_elements(&mut self, label: &str, elements: &[F]) {
        self.appended_elements
            .push(Operation::Append(String::from(label), elements.to_vec()));
    }

    pub fn squeeze_elements(&mut self, label: &str, num_elements: usize) {
        self.appended_elements
            .push(Operation::Squeeze(String::from(label), num_elements));
    }

    // pub fn get_element(&self, operation_idx: usize, element_idx: usize) -> Option<F> {
    //     match self.appended_elements.get(operation_idx) {
    //         None => None,
    //         Some(v) => match v.get(element_idx) {
    //             None => None,
    //             Some(elem) => Some(elem),
    //         }
    //     }
    // }
}

pub struct TranscriptWriter<F: FieldExt, T: TranscriptSponge<F>> {
    sponge: T,
    transcript: Transcript<F>,
}

impl<F: FieldExt, T: TranscriptSponge<F>> TranscriptWriter<F, T> {
    pub fn new(label: &str) -> Self {
        Self {
            sponge: T::new(),
            transcript: Transcript::<F>::new(label),
        }
    }

    pub fn append(&mut self, label: &str, elem: F) {
        self.sponge.absorb(elem);
        self.transcript.append_elements(label, &[elem]);
    }

    pub fn append_elements(&mut self, label: &str, elements: &[F]) {
        self.sponge.absorb_elements(elements);
        self.transcript.append_elements(label, elements);
    }

    pub fn get_challenge(&mut self, label: &str) -> F {
        let challenge = self.sponge.squeeze();
        self.transcript.squeeze_elements(label, 1);
        challenge
    }

    pub fn get_challenges(&mut self, label: &str, num_elements: usize) -> Vec<F> {
        let challenges = self.sponge.squeeze_elements(num_elements);
        self.transcript.squeeze_elements(label, num_elements);
        challenges
    }

    pub fn get_transcript(&self) -> Transcript<F> {
        self.transcript.clone()
    }
}

#[derive(Error, Debug, Clone)]
pub enum TranscriptReaderError {
    #[error("Transcript indices out of bounds")]
    InternalIndicesError,
    #[error("An unexpected consume was requested")]
    ConsumeError,
    #[error("An unexpected squeeze was requested")]
    SqueezeError,
}

pub struct TranscriptReader<F: FieldExt, T: TranscriptSponge<F>> {
    sponge: T,
    transcript: Transcript<F>,
    next_element: (usize, usize),
}

impl<F: FieldExt, T: TranscriptSponge<F>> TranscriptReader<F, T> {
    pub fn new(transcript: Transcript<F>) -> Self {
        Self {
            sponge: T::new(),
            transcript,
            next_element: (0, 0),
        }
    }

    fn advance_indices(&mut self) -> Result<(), TranscriptReaderError> {
        let (operation_idx, element_idx) = self.next_element;

        match self.transcript.appended_elements.get(operation_idx) {
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

    pub fn consume_element(&mut self, label: &'static str) -> Result<F, TranscriptReaderError> {
        let (operation_idx, element_idx) = self.next_element;

        match self.transcript.appended_elements.get(operation_idx) {
            None => Err(TranscriptReaderError::ConsumeError),
            Some(Operation::Squeeze(_, _)) => Err(TranscriptReaderError::ConsumeError),
            Some(Operation::Append(expected_label, v)) => {
                if label != expected_label {
                    warn!("Label mismatch on TranscriptReader consume_element. Expected \"{}\" but instead got \"{}\".", expected_label, label);
                }
                match v.get(element_idx) {
                    None => Err(TranscriptReaderError::InternalIndicesError),
                    Some(element) => {
                        let element = *element;
                        self.advance_indices()?;
                        self.sponge.absorb(element);
                        Ok(element)
                    }
                }
            }
        }
    }

    pub fn consume_elements(
        &mut self,
        label: &'static str,
        num_elements: usize,
    ) -> Result<Vec<F>, TranscriptReaderError> {
        (0..num_elements)
            .map(|_| self.consume_element(label))
            .collect()
    }

    pub fn get_challenge(&mut self, label: &'static str) -> Result<F, TranscriptReaderError> {
        let (operation_idx, element_idx) = self.next_element;

        match self.transcript.appended_elements.get(operation_idx) {
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

    pub fn get_challenges(
        &mut self,
        label: &'static str,
        num_elements: usize,
    ) -> Result<Vec<F>, TranscriptReaderError> {
        (0..num_elements)
            .map(|_| self.get_challenge(label))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use self::poseidon_transcript::PoseidonSponge;

    use super::*;
    use crate::Fr;

    fn generate_test_transcript() -> Transcript<Fr> {
        let mut transcript_writer = TranscriptWriter::<Fr, PoseidonSponge<Fr>>::new("New tw");

        transcript_writer.append("A1", Fr::from(1));
        transcript_writer.append("A2", Fr::from(2));
        let _ = transcript_writer.get_challenge("C1");
        transcript_writer.append_elements("A3", &[Fr::from(3), Fr::from(4)]);
        transcript_writer.append("A4", Fr::from(5));
        let _ = transcript_writer.get_challenge("C2");
        let _ = transcript_writer.get_challenges("C3", 5);
        transcript_writer.append_elements("A5", &[Fr::from(6), Fr::from(7), Fr::from(8)]);
        let _ = transcript_writer.get_challenge("C4");

        transcript_writer.get_transcript()
    }

    #[test]
    fn test_transcript_writer_internal_state() {
        let transcript = generate_test_transcript();

        let appended_elements = vec![
            Operation::Append(String::from("A1"), vec![Fr::from(1)]),
            Operation::Append(String::from("A2"), vec![Fr::from(2)]),
            Operation::Squeeze(String::from("C1"), 1),
            Operation::Append(String::from("A3"), vec![Fr::from(3), Fr::from(4)]),
            Operation::Append(String::from("A4"), vec![Fr::from(5)]),
            Operation::Squeeze(String::from("C2"), 1),
            Operation::Squeeze(String::from("C3"), 5),
            Operation::Append(
                String::from("A5"),
                vec![Fr::from(6), Fr::from(7), Fr::from(8)],
            ),
            Operation::Squeeze(String::from("C4"), 1),
        ];

        let expected_transcript = Transcript::<Fr> {
            label: String::from("New tw"),
            appended_elements,
        };

        assert_eq!(transcript, expected_transcript);
    }

    #[test]
    fn test_transcript_reader() {
        let transcript = generate_test_transcript();
        let mut transcript_reader = TranscriptReader::<Fr, PoseidonSponge<Fr>>::new(transcript);

        assert_eq!(
            transcript_reader.consume_element("A1").unwrap(),
            Fr::from(1)
        );
        assert_eq!(
            transcript_reader.consume_element("A2").unwrap(),
            Fr::from(2)
        );
        assert!(transcript_reader.get_challenge("C1").is_ok());
        assert_eq!(
            transcript_reader.consume_element("A3").unwrap(),
            Fr::from(3)
        );
        assert_eq!(
            transcript_reader.consume_element("A3").unwrap(),
            Fr::from(4)
        );
        assert_eq!(
            transcript_reader.consume_element("A4").unwrap(),
            Fr::from(5)
        );
        assert!(transcript_reader.get_challenge("C2").is_ok());
        assert!(transcript_reader.get_challenges("C3", 5).is_ok());
        assert_eq!(
            transcript_reader.consume_element("A5").unwrap(),
            Fr::from(6)
        );
        assert_eq!(
            transcript_reader.consume_element("A5").unwrap(),
            Fr::from(7)
        );
        assert_eq!(
            transcript_reader.consume_element("A5").unwrap(),
            Fr::from(8)
        );
        assert!(transcript_reader.get_challenge("C4").is_ok());

        assert!(transcript_reader.get_challenge("C5").is_err());
        assert!(transcript_reader.consume_element("A6").is_err());
    }
}
