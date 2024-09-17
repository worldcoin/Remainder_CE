#[cfg(test)]
mod tests {
    use super::super::*;
    use remainder_shared_types::halo2curves::bn256::Fr;
    use remainder_shared_types::halo2curves::bn256::G1 as Bn256Point;
    /// Tests for the Pedersen commitment scheme using the BN254 (aka BN256) curve and its scalar field (Fr).
    use remainder_shared_types::halo2curves::CurveExt;
    use remainder_shared_types::transcript::ec_transcript::ECTranscriptReader;
    use remainder_shared_types::transcript::ec_transcript::ECTranscriptWriter;
    use remainder_shared_types::transcript::poseidon_transcript::PoseidonSponge;

    type Base = <Bn256Point as CurveExt>::Base;

    // a static string constant
    const INIT_STR: &str = "modulus modulus modulus modulus modulus modulus";

    #[test]
    fn test_completeness() {
        let committer = PedersenCommitter::<Bn256Point>::new(1, INIT_STR, None);

        let mut prover_transcript: ECTranscriptWriter<Bn256Point, PoseidonSponge<Base>> =
            ECTranscriptWriter::new("testing proof of opening (completeness) - prover");

        let x = committer.committed_scalar(&Fr::from(23 as u64), &Fr::from(2 as u64));
        let proof = ProofOfOpening::prove(
            &x,
            &committer,
            &mut rand::thread_rng(),
            &mut prover_transcript,
        );

        let transcript = prover_transcript.get_transcript();
        let mut verifier_transcript: ECTranscriptReader<Bn256Point, PoseidonSponge<Base>> =
            ECTranscriptReader::new(transcript);

        proof.verify(x.commitment, &committer, &mut verifier_transcript);
    }

    #[test]
    #[should_panic]
    fn test_soundness() {
        // test soundness - try to verify for a different commitment value than the one in the proof
        let committer = PedersenCommitter::<Bn256Point>::new(1, INIT_STR, None);

        let mut prover_transcript: ECTranscriptWriter<Bn256Point, PoseidonSponge<Base>> =
            ECTranscriptWriter::new("testing proof of opening (soundness) - prover");

        let x = committer.committed_scalar(&Fr::from(23 as u64), &Fr::from(2 as u64));
        let proof = ProofOfOpening::prove(
            &x,
            &committer,
            &mut rand::thread_rng(),
            &mut prover_transcript,
        );

        let transcript = prover_transcript.get_transcript();
        let mut verifier_transcript: ECTranscriptReader<Bn256Point, PoseidonSponge<Base>> =
            ECTranscriptReader::new(transcript);

        proof.verify(
            x.commitment + x.commitment,
            &committer,
            &mut verifier_transcript,
        );
    }
}
