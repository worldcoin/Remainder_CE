//! A transcript that uses the Poseidon hash function; Useful for recursive proving

use ark_crypto_primitives::sponge::{
    poseidon::{find_poseidon_ark_and_mds, PoseidonConfig, PoseidonSponge},
    CryptographicSponge, FieldBasedCryptographicSponge,
};
use tracing::trace;

use crate::FieldExt;

use super::{Transcript, TranscriptError};

/// A transcript that uses the Poseidon hash function; Useful for recursive proving
pub struct PoseidonTranscript<F: FieldExt> {
    sponge: PoseidonSponge<F>,
}

impl<F: FieldExt> Transcript<F> for PoseidonTranscript<F> {
    fn new(label: &'static str) -> Self {
        trace!(module = "Transcript", label);
        //TODO!(This sucks, generating them anew every time is slow, stupid, and likely to lead to problems integrating with Marcin. Need to read these from somewhere. Touch base with Marcin on these constants)
        let (ark, mds) = find_poseidon_ark_and_mds::<F>(F::MODULUS_BIT_SIZE as u64, 2, 8, 60, 0);

        let params = PoseidonConfig::new(8, 60, 5, mds, ark, 2, 1);
        Self {
            sponge: PoseidonSponge::new(&params),
        }
    }

    fn append_field_element(
        &mut self,
        label: &'static str,
        element: F,
    ) -> Result<(), TranscriptError> {
        trace!(module = "Transcript", "Absorbing: {}, {:?}", label, element);
        self.sponge.absorb(&element);
        Ok(())
    }

    fn append_field_elements(
        &mut self,
        label: &'static str,
        elements: &[F],
    ) -> Result<(), TranscriptError> {
        trace!(
            module = "Transcript",
            "Absorbing: {}, {:?}",
            label,
            elements
        );
        self.sponge.absorb(&elements);
        Ok(())
    }

    fn get_challenge(&mut self, label: &'static str) -> Result<F, TranscriptError> {
        let output = self.sponge.squeeze_native_field_elements(1)[0];
        trace!(module = "Transcript", "Squeezing: {}, {:?}", label, output);
        Ok(output)
    }

    fn get_challenges(
        &mut self,
        label: &'static str,
        len: usize,
    ) -> Result<Vec<F>, TranscriptError> {
        let output = self.sponge.squeeze_native_field_elements(len);
        trace!(module = "Transcript", "Squeezing: {}, {:?}", label, output);
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use ark_ff::One;
    use itertools::Itertools;

    use crate::transcript::Transcript;

    use super::PoseidonTranscript;

    #[test]
    fn test_poseidon_transcript() {
        let mut transcript = PoseidonTranscript::<Fr>::new("New Transcript");
        transcript
            .append_field_element("Random crap", Fr::from(6))
            .unwrap();
        transcript
            .append_field_element("More crap", Fr::from(1))
            .unwrap();
        let out = transcript.get_challenge("Random Challenge").unwrap();
        assert!(
            out.to_string()
                == "20217826739391221062203418242768992766073309943336995961793742009429567399088"
        );
        transcript
            .append_field_elements("Random crap Vec", &[Fr::one(), Fr::from(2)])
            .unwrap();
        let outs = transcript
            .get_challenges("Random Challenges", 3)
            .unwrap()
            .into_iter()
            .map(|item| item.to_string())
            .collect_vec();
        assert!(
            outs == vec![
                "677756219746287980320990361612560322696521318112015859068333315837721133606",
                "7851947639380962889981762755637082147943839049218250815910426501550542257261",
                "2465518972688468036944219314823116581897801957391227587219603682466320590865"
            ]
        );
    }
}
