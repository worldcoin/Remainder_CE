use std::{collections::HashMap, marker::PhantomData};

use crate::utils::vandermonde::VandermondeInverse;
use hyrax_input_layer::{HyraxInputLayerProof, InputProofEnum};
use hyrax_layer::HyraxClaim;
use hyrax_output_layer::HyraxOutputLayerProof;
use itertools::Itertools;
use rand::Rng;
use remainder::mle::Mle;
use remainder::{
    claims::wlx_eval::ClaimMle,
    input_layer::{
        enum_input_layer::InputLayerEnum, public_input_layer::PublicInputLayer,
        random_input_layer::RandomInputLayer,
    },
    layer::{LayerId, PostSumcheckEvaluation, SumcheckLayer},
};

use crate::pedersen::{CommittedScalar, PedersenCommitter};

use remainder_shared_types::{
    curves::PrimeOrderCurve,
    transcript::ec_transcript::{ECProverTranscript, ECVerifierTranscript},
};

use self::{hyrax_layer::HyraxLayerProof, hyrax_output_layer::HyraxOutputLayer};

/// The module that contains all functions necessary to do operations on an
/// output layer, [HyraxInputLayer]
pub mod hyrax_input_layer;
/// The module that has all functions and implementations necessary to produce
/// a [HyraxLayerProof]
pub mod hyrax_layer;
/// The module that contains all functions necessary to do operations on an
/// output layer, [HyraxOutputLayer]
pub mod hyrax_output_layer;
/// The testing module for worldcoin circuit
pub mod test_worldcoin;
/// The testing module for integration tests
pub mod tests;

/// The struct that holds all the respective proofs that the verifier needs in order
/// to verify a HyraxGKRProof
pub struct HyraxProof<
    C: PrimeOrderCurve,
    L: SumcheckLayer<C::Scalar> + PostSumcheckEvaluation<C::Scalar>,
> {
    /// The [HyraxLayerProof] for each of the intermediate layers in this circuit.
    layer_proofs: Vec<HyraxLayerProof<C>>,
    /// The [HyraxInputLayerProof] for each of the input polynomial commitments using the Hyrax PCS.
    input_layer_proofs: Vec<InputProofEnum<C>>,
    /// A commitment to the output of the circuit, i.e. what the final value of the output layer is.
    output_layer_proofs: Vec<HyraxOutputLayerProof<C>>,
    _marker: PhantomData<L>,
}

/// The struct that holds all the necessary information to describe a circuit.
pub struct HyraxCircuit<
    C: PrimeOrderCurve,
    L: SumcheckLayer<C::Scalar> + PostSumcheckEvaluation<C::Scalar>,
> {
    pub input_layers: Vec<InputLayerEnum<C::Scalar>>,
    pub layers: Vec<L>,
    pub output_layers: Vec<HyraxOutputLayer<C>>,
    pub input_commitments: Vec<CommitmentEnum<C, C::Scalar>>,
}

impl<C: PrimeOrderCurve, L: SumcheckLayer<C::Scalar> + PostSumcheckEvaluation<C::Scalar>>
    HyraxProof<C, L>
{
    /// TODO(vishady) riad audit comments: add in comments the ordering of the proofs every time they are in a vec

    /// The Hyrax GKR prover for a full circuit, including output layers, intermediate layers,
    /// and input layers.
    /// The calling context is responsible for appending to the transcript both the circuit
    /// description and the values and/or commitments of the input layer (which is appropriate
    /// unless already added further upstream).
    pub fn prove(
        circuit: &mut HyraxCircuit<C, L>,
        committer: &PedersenCommitter<C>,
        mut blinding_rng: &mut impl Rng,
        transcript: &mut impl ECProverTranscript<C>,
        converter: &mut VandermondeInverse<C::Scalar>,
    ) -> Self {
        let HyraxCircuit {
            input_layers,
            layers,
            output_layers,
            input_commitments,
        } = circuit;

        // HashMap to keep track of all claims made on each layer
        let mut claim_tracker: HashMap<LayerId, Vec<HyraxClaim<C::Scalar, CommittedScalar<C>>>> =
            HashMap::new();

        let output_layer_proofs = output_layers
            .into_iter()
            .map(|output_layer| {
                // Create the HyraxOutputLayerProof
                let (output_layer_proof, committed_output_claim) = HyraxOutputLayerProof::prove(
                    output_layer,
                    transcript,
                    &mut blinding_rng,
                    committer,
                );
                // Add the output claim to the claims table
                let output_layer_id = output_layer.underlying_mle.layer_id();
                claim_tracker.insert(output_layer_id, vec![committed_output_claim]);
                output_layer_proof
            })
            .collect_vec();

        let layer_proofs = layers
            .into_iter()
            .rev()
            .map(|layer| {
                let claims = claim_tracker.get(&layer.id()).unwrap().clone();

                let (layer_proof, claims_from_layer) = HyraxLayerProof::prove(
                    layer,
                    &claims,
                    &committer,
                    &mut blinding_rng,
                    transcript,
                    converter,
                );
                // add new claims to the claim tracking table, and add each new claim to the transcript
                for claim in claims_from_layer.into_iter() {
                    if let Some(curr_claims) = claim_tracker.get_mut(&claim.to_layer_id) {
                        curr_claims.push(claim);
                    } else {
                        claim_tracker.insert(claim.to_layer_id, vec![claim]);
                    }
                }

                layer_proof
            })
            .collect_vec();

        // Input layer proofs

        let input_layer_proofs = input_layers
            .into_iter()
            .zip(input_commitments)
            .map(|(input_layer, commitment)| {
                let layer_id = input_layer.layer_id();
                let committed_claims = claim_tracker.get(&layer_id).unwrap();
                match input_layer {
                    InputLayerEnum::HyraxInputLayer(hyrax_input_layer) => {
                        let hyrax_commitment = match commitment {
                            CommitmentEnum::HyraxCommitment(hyrax_commitment) => hyrax_commitment,
                            _ => {
                                panic!("Unsupported commitment type for Hyrax");
                            }
                        };
                        let input_proof = HyraxInputLayerProof::prove(
                            &hyrax_input_layer,
                            &hyrax_commitment,
                            &committed_claims,
                            &committer,
                            &mut blinding_rng,
                            transcript,
                            converter,
                        );
                        InputProofEnum::HyraxInputLayerProof(input_proof)
                    }

                    // @vishady this bit is new
                    // For the other input layers, the prover just hands over the (CommittedScalar-valued) HyraxClaim for each claim.
                    // The verifier will need to check that each of the claims is consistent with the input layer.
                    InputLayerEnum::PublicInputLayer(layer) => {
                        InputProofEnum::PublicInputLayerProof(
                            layer.clone(),
                            committed_claims.clone(),
                        )
                    }
                    InputLayerEnum::RandomInputLayer(layer) => {
                        InputProofEnum::RandomInputLayerProof(
                            layer.clone(),
                            committed_claims.clone(),
                        )
                    }
                    _ => {
                        panic!("Input layer type not supported by Hyrax");
                    }
                }
            })
            .collect_vec();

        Self {
            layer_proofs,
            input_layer_proofs,
            output_layer_proofs,
            _marker: PhantomData,
        }
    }

    /// This is the verification of a GKR proof. It essentially calls the verify functions of the underlying proofs
    /// The calling context is responsible for appending to the transcript both the circuit
    /// description and the values and/or commitments of the input layer (which is appropriate
    /// unless already added further upstream).
    pub fn verify(
        proof: &HyraxProof<C, L>,
        circuit_description: &CircuitDescription<C>,
        committer: &PedersenCommitter<C>,
        transcript: &mut impl ECVerifierTranscript<C>,
    ) {
        let HyraxProof {
            layer_proofs,
            input_layer_proofs,
            output_layer_proofs,
            _marker,
        } = proof;

        // Keep track of all claim commitments for the hyrax layer verifier
        let mut claim_tracker: HashMap<LayerId, Vec<HyraxClaim<C::Scalar, C>>> = HashMap::new();

        // Output layer verification
        output_layer_proofs
            .iter()
            .zip(circuit_description.output_layers.iter())
            .for_each(|(output_layer_proof, output_layer_desc)| {
                let output_layer_claim = HyraxOutputLayerProof::verify(
                    &output_layer_proof,
                    &output_layer_desc,
                    transcript,
                );

                // Add the output claim to the claims table
                claim_tracker.insert(output_layer_claim.to_layer_id, vec![output_layer_claim]);
            });

        // Intermediate layer verification
        (layer_proofs
            .into_iter()
            .zip(circuit_description.layers.clone().into_iter().rev()))
        .for_each(|(layer_proof, layer_desc)| {
            // Get the unaggregated claims for this layer
            // V checked that these claims had the expected form before adding them to the claim tracking table
            let layer_claims_vec = claim_tracker.remove(&layer_desc.id).unwrap().clone();
            let claim_commits_for_layer = HyraxLayerProof::verify(
                &layer_proof,
                &layer_desc,
                &layer_claims_vec,
                &committer,
                transcript,
            );

            for claim in claim_commits_for_layer {
                if let Some(curr_claims) = claim_tracker.get_mut(&claim.to_layer_id) {
                    curr_claims.push(claim);
                } else {
                    claim_tracker.insert(claim.to_layer_id, vec![claim]);
                }
            }
        });

        // Input layers verification
        input_layer_proofs
            .into_iter()
            .for_each(|input_layer_proof| match input_layer_proof {
                InputProofEnum::HyraxInputLayerProof(hyrax_input_proof) => {
                    let layer_id = hyrax_input_proof.layer_id;
                    let layer_claims_vec = claim_tracker.remove(&layer_id).unwrap().clone();
                    hyrax_input_proof.verify(&layer_claims_vec, &committer, transcript);
                }
                // @vishady this part is new
                InputProofEnum::PublicInputLayerProof(layer, committed_claims) => {
                    let claims_as_commitments =
                        claim_tracker.remove(&layer.layer_id()).unwrap().clone();
                    let plaintext_claims =
                        Self::match_claims(&claims_as_commitments, &committed_claims, committer);
                    plaintext_claims.into_iter().for_each(|claim| {
                        PublicInputLayer::<C>::verify(
                            &layer.clone().commit().unwrap(),
                            &(),
                            claim,
                            transcript,
                        )
                        .unwrap();
                    });
                }
                InputProofEnum::RandomInputLayerProof(layer, committed_claims) => {
                    let claims_as_commitments =
                        claim_tracker.remove(&layer.layer_id()).unwrap().clone();
                    let plaintext_claims =
                        Self::match_claims(&claims_as_commitments, &committed_claims, committer);
                    plaintext_claims.into_iter().for_each(|claim| {
                        RandomInputLayer::<C>::verify(
                            &layer.clone().commit().unwrap(),
                            &(),
                            claim,
                            transcript,
                        )
                        .unwrap();
                    });
                }
            });

        // @vishady this is new, so that we can be sure that the prover didn't e.g. leave out an input layer proof!  (I changed all the claims.get() to claims.remove() above)
        // Check that there aren't any claims left in our claim tracking table!
        assert_eq!(claim_tracker.len(), 0);
    }

    /// Match up the claims from the verifier with the claims from the prover. Used for input layer
    /// proofs, where the proof (in the case of public and random layers) consists of the prover
    /// simply opening the commitments in the claims, or equivalently just handing over the
    /// CommittedScalars. Panics if a verifier claim can not be matched to a prover claim (and
    /// doesn't worry about prover claims that don't have a verifier counterpart).
    fn match_claims(
        verifier_claims: &Vec<HyraxClaim<C::Scalar, C>>,
        prover_claims: &Vec<HyraxClaim<C::Scalar, CommittedScalar<C>>>,
        committer: &PedersenCommitter<C>,
    ) -> Vec<ClaimMle<C::Scalar>> {
        verifier_claims
            .iter()
            .map(|claim| {
                // find the corresponding committed claim
                if let Some(committed_claim) = prover_claims.iter().find(|committed_claim| {
                    (committed_claim.point == claim.point)
                        & (committed_claim.evaluation.commitment == claim.evaluation)
                }) {
                    // verify that the committed claim is consistent with the committer
                    // (necessary in order to conclude that the plain-text value is the correct one)
                    committed_claim.evaluation.verify(committer);
                    // ok, return the claim
                    committed_claim.to_claim()
                } else {
                    // TODO return an error instead of panicking
                    panic!("Claim has not counterpart in committed claims!");
                }
            })
            .collect()
    }
}
