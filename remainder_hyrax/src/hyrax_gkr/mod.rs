use std::{collections::HashMap, marker::PhantomData};

use crate::pedersen::{CommittedScalar, PedersenCommitter};
use crate::utils::vandermonde::VandermondeInverse;
use ark_std::{end_timer, start_timer};
use hyrax_circuit_inputs::HyraxInputLayerData;
use hyrax_input_layer::{commit_to_input_values, HyraxInputCommitment, HyraxInputLayerProof}
;
use hyrax_layer::HyraxClaim;
use hyrax_output_layer::HyraxOutputLayerProof;
use itertools::Itertools;
use rand::Rng;
use remainder::input_layer::fiat_shamir_challenge::FiatShamirChallenge;
use remainder::input_layer::hyrax_input_layer::HyraxInputLayerDescription;
use remainder::layer::{Layer, LayerDescription};
use remainder::layouter::component::{Component, ComponentSet};
use remainder::layouter::nodes::node_enum::NodeEnum;
use remainder::layouter::nodes::{Context, NodeId};
use remainder::mle::evals::MultilinearExtension;
use remainder::mle::Mle;
use remainder::prover::{generate_circuit_description, GKRCircuitDescription, InstantiatedCircuit};
use remainder::{claims::wlx_eval::ClaimMle, layer::LayerId};

use remainder_shared_types::{
    curves::PrimeOrderCurve,
    transcript::ec_transcript::{ECProverTranscript, ECVerifierTranscript},
};

use self::hyrax_layer::HyraxLayerProof;

pub mod hyrax_circuit_inputs;
/// The module that contains all functions necessary to do operations on an
/// output layer, [HyraxInputLayer]
pub mod hyrax_input_layer;
/// The module that has all functions and implementations necessary to produce
/// a [HyraxLayerProof]
pub mod hyrax_layer;
/// The module that contains all functions necessary to do operations on an
/// output layer, [HyraxOutputLayer]
pub mod hyrax_output_layer;
#[cfg(test)]
/// The testing module for integration tests
pub mod tests;

/// The struct that holds all the respective proofs that the verifier needs in order
/// to verify a Hyrax proof, i.e. the circuit proof along with the proofs for each input layer.
pub struct HyraxProof<C: PrimeOrderCurve> {
    /// The MLEs of the public inputs, along with their layer ids
    pub public_inputs: Vec<(LayerId, MultilinearExtension<C::Scalar>)>,
    /// The proof for the circuit proper, i.e. the intermediate layers and output layers.
    pub circuit_proof: HyraxCircuitProof<C>,
    /// The prover's claims on public input layers, in CommittedScalar form,
    /// i.e. including the blinding factors (since the verifier needs to check these itself).
    pub claims_on_public_values: Vec<HyraxClaim<C::Scalar, CommittedScalar<C>>>,
    /// The [HyraxInputLayerProof] for each of the input polynomial commitments using the Hyrax PCS.
    pub hyrax_input_proofs: Vec<HyraxInputLayerProof<C>>,
}

impl<C: PrimeOrderCurve> HyraxProof<C> {
    pub fn prove(
        inputs: &HashMap<LayerId, MultilinearExtension<C::Scalar>>,
        hyrax_input_layers: &HashMap<LayerId, (HyraxInputLayerDescription<C::Scalar>, Option<HyraxInputCommitment<C>>)>,
        circuit_description: &GKRCircuitDescription<C::Scalar>,
        committer: &PedersenCommitter<C>,
        blinding_rng: &mut impl Rng,
        converter: &mut VandermondeInverse<C::Scalar>,
        transcript: &mut impl ECProverTranscript<C>,
    ) -> HyraxProof<C> {
        // Add the input values of any public (i.e. non-hyrax) input layers to transcript.
        // Select the public input layers from the input layers, and sort them by layer id, and append
        // their input values to the transcript.
        inputs
            .keys()
            .filter(|layer_id| !hyrax_input_layers.contains_key(layer_id))
            .sorted_by_key(|layer_id| layer_id.get_input_layer_id())
            .for_each(|layer_id| {
                let mle = inputs.get(layer_id).unwrap();
                transcript.append_scalar_points("input layer", mle.get_evals_vector());
            });

        // For each hyrax input layer, calculate commitments if not already provided, and then append each
        // commitment to the transcript.
        let mut hyrax_input_commitments = HashMap::<LayerId, HyraxInputCommitment<C>>::new();
        hyrax_input_layers
            .keys()
            .sorted_by_key(|layer_id| layer_id.get_input_layer_id())
            .for_each(|layer_id| {
                // Commit to the Hyrax input layer, if it is not already committed to.
                let (desc, maybe_commitment) = hyrax_input_layers.get(layer_id).unwrap();
                let prover_commitment = if let Some(prover_commitment) = maybe_commitment {
                    // Use the commitment provided by the calling context
                    prover_commitment.clone()
                } else {
                    // Commit to the values of the input layer
                    let input_mle = inputs.get(layer_id).unwrap();
                    commit_to_input_values(desc, input_mle, blinding_rng)
                };

                // Add the verifier's view of the commitment to transcript
                transcript.append_ec_points("Hyrax input layer values", &prover_commitment.commitment);

                // Store the prover's view for later use in the evaluation proofs.
                hyrax_input_commitments.insert(layer_id.clone(), prover_commitment.clone());
            });

        let mut challenge_sampler = |size| {
            transcript.get_scalar_field_challenges("Verifier challenges", size)
        };
        let mut instantiated_circuit = circuit_description.instantiate(&inputs, &mut challenge_sampler);
        
        let (circuit_proof, claims_on_input_layers) = HyraxCircuitProof::prove(
            &mut instantiated_circuit,
            &committer,
            blinding_rng,
            converter,
            transcript,
        );

        // Collect the values of the public inputs
        let public_inputs = inputs
            .keys()
            .filter(|layer_id| !hyrax_input_layers.contains_key(layer_id))
            .map(|layer_id| {
                let mle = inputs.get(layer_id).unwrap();
                (layer_id.clone(), mle.clone())
            }).collect_vec();
        
        // Separate the claims on input layers into claims on public input layers vs claims on Hyrax
        // input layers
        let mut claims_on_public_values = vec![];
        let mut claims_on_hyrax_input_layers = HashMap::<LayerId, Vec<HyraxClaim<C::Scalar, CommittedScalar<C>>>>::new();
        claims_on_input_layers.iter().for_each(|claim| {
            if hyrax_input_layers.contains_key(&claim.to_layer_id) {
                if let Some(curr_claims) = claims_on_hyrax_input_layers.get_mut(&claim.to_layer_id) {
                    curr_claims.push(claim.clone());
                } else {
                    claims_on_hyrax_input_layers.insert(claim.to_layer_id, vec![claim.clone()]);
                }
            } else {
                claims_on_public_values.push(claim.clone());
            }
        });

        // If in debug mode, then check the claims on all input layers.
        if !cfg!(debug_assertions) {
            for claim in claims_on_input_layers.iter() {
                let input_mle = inputs.get(&claim.to_layer_id).unwrap();
                let evaluation = input_mle.evaluate_at_point(&claim.point);
                if evaluation != claim.evaluation.value {
                    panic!("Claim on input layer {} does not match evaluation", claim.to_layer_id);
                }
            }
        }
        
        // FIXME(Ben) we have the familiar challenge of ordering the claims!
        // Prove the claims on the Hyrax input layers
        let hyrax_input_proofs = hyrax_input_layers
            .keys()
            .sorted_by_key(|layer_id| layer_id.get_input_layer_id())
            .map(|layer_id| {
                let (desc, _) = hyrax_input_layers.get(layer_id).unwrap();
                let commitment = hyrax_input_commitments.get(layer_id).unwrap();
                let committed_claims = claims_on_hyrax_input_layers.get(layer_id).unwrap();
                let input_proof = HyraxInputLayerProof::prove(
                    desc,
                    commitment,
                    committed_claims,
                    committer,
                    &mut blinding_rng,
                    transcript,
                    converter,
                );
                input_proof
            })
            .collect_vec();

        // Check that now Hyrax input layer claims remain
        assert!(claims_on_hyrax_input_layers.is_empty());

        HyraxProof {
            public_inputs,
            circuit_proof,
            claims_on_public_values,
            hyrax_input_proofs,
        }
    }
}

/// The struct that holds all the information that the prover sends to the verifier about the circuit proof, i.e. the proof that transforms the claims on the output layers to claims on the input layers.
pub struct HyraxCircuitProof<C: PrimeOrderCurve> {
    /// The [HyraxLayerProof] for each of the intermediate layers in this circuit.
    pub layer_proofs: Vec<HyraxLayerProof<C>>,
    /// A commitment to the output of the circuit, i.e. what the final value of the output layer is.
    pub output_layer_proofs: Vec<HyraxOutputLayerProof<C>>,
    /// The prover's claims on verifier challenges, in CommittedScalar form,
    /// i.e. including the blinding factors (since the verifier needs to check these itself).
    pub fiat_shamir_claims: Vec<HyraxClaim<C::Scalar, CommittedScalar<C>>>,
}

impl<C: PrimeOrderCurve> HyraxCircuitProof<C> {

    /// The calling context is responsible for appending to the transcript both the circuit
    /// description and the values and/or commitments of the input layer (which is appropriate
    /// unless already added further upstream).
    /// Returns the [HyraxCircuitProof] instance, along with a vector of claims on input layers, to
    /// be proven by the calling context.
    pub fn prove(
        instantiated_circuit: &mut InstantiatedCircuit<C::Scalar>,
        committer: &PedersenCommitter<C>,
        blinding_rng: &mut impl Rng,
        converter: &mut VandermondeInverse<C::Scalar>,
        transcript: &mut impl ECProverTranscript<C>,
    ) -> (HyraxCircuitProof<C>, Vec<HyraxClaim<C::Scalar, CommittedScalar<C>>>) {
        let InstantiatedCircuit {
            input_layers,
            fiat_shamir_challenges,
            layers,
            output_layers,
            layer_map,
        } = instantiated_circuit;

        // HashMap to keep track of all claims made on each layer
        let mut claim_tracker: HashMap<LayerId, Vec<HyraxClaim<C::Scalar, CommittedScalar<C>>>> =
            HashMap::new();

        let output_layer_proofs = output_layers
            .iter_mut()
            .map(|output_layer| {
                // Create the HyraxOutputLayerProof
                let (output_layer_proof, committed_output_claim) = HyraxOutputLayerProof::prove(
                    output_layer,
                    transcript,
                    blinding_rng,
                    committer,
                );
                // Add the output claim to the claims table
                let output_layer_id = output_layer.get_mle().layer_id();
                claim_tracker.insert(output_layer_id, vec![committed_output_claim]);
                output_layer_proof
            })
            .collect_vec();

        let layer_proofs = layers.layers
            .iter_mut()
            .rev()
            .map(|layer| {
                let claims = claim_tracker.get(&layer.layer_id()).unwrap().clone();
                let output_mles_from_layer = layer_map.get(&layer.layer_id()).unwrap();
                let (layer_proof, claims_from_layer) = HyraxLayerProof::prove(
                    layer,
                    &claims,
                    output_mles_from_layer,
                    committer,
                    blinding_rng,
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

        let claims_on_input_layers = input_layers
            .iter()
            .filter_map(|input_layer| {
                claim_tracker.get(&input_layer.layer_id)
            })
            .flatten()
            .cloned()
            .collect_vec();

        // --------- Verifier Challenges ---------
        // The the claims on verifier challenges are checked directly
        // by the verifier, without aggregation, so there is almost nothing to do here.  However,
        // the verifier received the prover's claims in committed form (this is just how we
        // implemented layer proof) and so we provide here the CommittedScalar forms in order for V
        // to be able to actually verify the claims.
        let fiat_shamir_claims = fiat_shamir_challenges
            .iter()
            .flat_map(|fiat_shamir_challenge| {
                claim_tracker
                    .get(&fiat_shamir_challenge.layer_id())
                    .unwrap()
            })
            .cloned()
            .collect_vec();

        (HyraxCircuitProof {
            layer_proofs,
            output_layer_proofs,
            fiat_shamir_claims,
        }, claims_on_input_layers)
    }

    /// The calling context is responsible for appending to the transcript both the circuit
    /// description and the values and/or commitments of the input layer (which is appropriate
    /// unless already added further upstream).
    /// Returns a vector of claims on the input layers.
    pub fn verify(
        &self,
        circuit_description: &GKRCircuitDescription<C::Scalar>,
        committer: &PedersenCommitter<C>,
        fiat_shamir_challenges: Vec<FiatShamirChallenge<C::Scalar>>,
        transcript: &mut impl ECVerifierTranscript<C>,
    ) -> Vec<HyraxClaim<C::Scalar, C>> {
        // Unpack the Hyrax proof.
        let HyraxCircuitProof {
            layer_proofs,
            output_layer_proofs,
            fiat_shamir_claims,
        } = &self;

        // Keep track of all claim commitments for the hyrax layer verifier
        let mut claim_tracker: HashMap<LayerId, Vec<HyraxClaim<C::Scalar, C>>> = HashMap::new();

        // Output layer verification
        output_layer_proofs
            .iter()
            .zip(circuit_description.output_layers.iter())
            .for_each(|(output_layer_proof, output_layer_desc)| {
                let output_layer_claim = HyraxOutputLayerProof::verify(
                    output_layer_proof,
                    output_layer_desc,
                    transcript,
                );

                // Add the output claim to the claims table
                claim_tracker.insert(output_layer_claim.to_layer_id, vec![output_layer_claim]);
            });

        // Intermediate layer verification
        (layer_proofs
            .iter()
            .zip(circuit_description.intermediate_layers.iter().rev()))
        .for_each(|(layer_proof, layer_desc)| {
            // Get the unaggregated claims for this layer
            // V checked that these claims had the expected form before adding them to the claim tracking table
            let layer_claims_vec = claim_tracker
                .remove(&layer_desc.layer_id())
                .unwrap()
                .clone();
            let claim_commits_for_layer = HyraxLayerProof::verify(
                layer_proof,
                layer_desc,
                &layer_claims_vec,
                committer,
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

        // Check the claims on the verifier challenges
        fiat_shamir_challenges
            .iter()
            .for_each(|fiat_shamir_challenge| {
                let claims_as_commitments = claim_tracker
                    .remove(&fiat_shamir_challenge.layer_id())
                    .unwrap();
                match_claims(&claims_as_commitments, fiat_shamir_claims, committer)
                    .iter()
                    .for_each(|plaintext_claim| {
                        fiat_shamir_challenge
                            .verify(plaintext_claim.get_claim())
                            .unwrap();
                    });
            });

        // Collect the claims on the input layers
        let input_layer_claims = circuit_description
            .input_layers
            .iter()
            .filter_map(|input_layer| {
                claim_tracker.get(&input_layer.layer_id)
            })
            .flatten()
            .cloned()
            .collect_vec();

        // Check that there aren't any claims left in our claim tracking table!
        assert_eq!(claim_tracker.len(), 0);

        input_layer_claims
    }

}

pub struct FiatShamirChallengeProof<C: PrimeOrderCurve> {
    pub layer: FiatShamirChallenge<C::Scalar>,
    pub claims: Vec<HyraxClaim<C::Scalar, CommittedScalar<C>>>,
}

/// The struct that holds all the necessary information to describe a circuit.
pub struct HyraxProver<
    'a,
    C: PrimeOrderCurve,
    Fn: FnMut(
        &Context,
    ) -> (
        ComponentSet<NodeEnum<C::Scalar>>,
        Vec<HyraxInputLayerData<C>>,
    ),
    R: Rng,
> {
    pub committer: &'a PedersenCommitter<C>,
    pub blinding_rng: R,
    pub converter: &'a mut VandermondeInverse<C::Scalar>,
    pub _marker: PhantomData<Fn>,
}

impl<
        'a,
        C: PrimeOrderCurve,
        Fn: FnMut(
            &Context,
        ) -> (
            ComponentSet<NodeEnum<C::Scalar>>,
            Vec<HyraxInputLayerData<C>>,
        ),
        R: Rng,
    > HyraxProver<'a, C, Fn, R>
{
    pub fn new(
        committer: &'a PedersenCommitter<C>,
        blinding_rng: R,
        converter: &'a mut VandermondeInverse<C::Scalar>,
    ) -> Self {
        Self {
            committer,
            blinding_rng,
            converter,
            _marker: PhantomData,
        }
    }

    pub fn prove_gkr_circuit(
        &mut self,
        mut witness_function: Fn,
        transcript_writer: &mut impl ECProverTranscript<C>,
    ) -> (
        GKRCircuitDescription<C::Scalar>,
        HyraxProof<C>,
    ) {
        let ctx = Context::new();
        let (component, input_layer_data) = (witness_function)(&ctx);

        // Convert the input layer data into a map that maps the input shred ID
        // i.e. adapt witness builder output to the instantate() function.
        // This can be removed once witness builders are removed.
        let mut shred_id_to_data = HashMap::<NodeId, MultilinearExtension<C::Scalar>>::new();
        input_layer_data.into_iter().for_each(|input_layer_data| {
            input_layer_data.data.into_iter().for_each(|input_shred_data| {
                shred_id_to_data.insert(
                    input_shred_data.corresponding_input_shred_id,
                    input_shred_data.data,
                );
            });
        });

        let (circuit_description, input_builder, _) =
            generate_circuit_description(component.yield_nodes()).unwrap();

        let inputs = input_builder(shred_id_to_data).unwrap();

        // Add the inputs to transcript.
        // In the future flow, the inputs will be added to the transcript in the calling context.
        circuit_description.input_layers.iter().for_each(|input_layer| {
            // FIXME do this
        });

        let mut challenge_sampler = |size| {
            transcript_writer.get_scalar_field_challenges("Verifier challenges", size)
        };
        let mut instantiated_circuit = circuit_description.instantiate(&inputs, &mut challenge_sampler);

        let prove_timer = start_timer!(|| "prove hyrax circuit");
        let (circuit_proof, claims_on_input_layers) = self.prove(&mut instantiated_circuit, transcript_writer);
        end_timer!(prove_timer);

        // Collect the values of the public inputs
        let public_inputs = todo!();
        
        // Filter through the claims on input layers to get the claims on public input layers
        let claims_on_public_values = todo!();
        
        // Prove the claims on the Hyrax input layers
        let hyrax_input_proofs = todo!();
        //             let hyrax_commitment = hyrax_input_layer.comm.as_ref().unwrap();
        //             let input_proof = HyraxInputLayerProof::prove(
        //                 hyrax_input_layer,
        //                 hyrax_commitment,
        //                 committed_claims,
        //                 committer,
        //                 &mut blinding_rng,
        //                 transcript,
        //                 converter,
        //             );
        //             input_proof

        let proof = HyraxProof {
            public_inputs,
            circuit_proof,
            claims_on_public_values,
            hyrax_input_proofs,
        };
        (circuit_description, proof)
    }

    #[allow(clippy::type_complexity)]
    /// TODO(vishady) riad audit comments: add in comments the ordering of the proofs every time they are in a vec

    /// The Hyrax GKR prover for a circuit, including output layers, intermediate layers,
    /// but not input layers.
    /// The calling context is responsible for appending to the transcript both the circuit
    /// description and the values and/or commitments of the input layer (which is appropriate
    /// unless already added further upstream).
    /// Returns the [HyraxCircuitProof] instance, along with a vector of claims on input layers, to
    /// be proven by the calling context.
    pub fn prove(
        &mut self,
        instantiated_circuit: &mut InstantiatedCircuit<C::Scalar>,
        transcript: &mut impl ECProverTranscript<C>,
    ) -> (HyraxCircuitProof<C>, Vec<HyraxClaim<C::Scalar, CommittedScalar<C>>>) {
        let committer = self.committer;
        let mut blinding_rng = &mut self.blinding_rng;
        let converter = &mut self.converter;
        let InstantiatedCircuit {
            input_layers,
            fiat_shamir_challenges,
            layers,
            output_layers,
            layer_map,
        } = instantiated_circuit;

        // HashMap to keep track of all claims made on each layer
        let mut claim_tracker: HashMap<LayerId, Vec<HyraxClaim<C::Scalar, CommittedScalar<C>>>> =
            HashMap::new();

        let output_layer_proofs = output_layers
            .iter_mut()
            .map(|output_layer| {
                // Create the HyraxOutputLayerProof
                let (output_layer_proof, committed_output_claim) = HyraxOutputLayerProof::prove(
                    output_layer,
                    transcript,
                    &mut blinding_rng,
                    committer,
                );
                // Add the output claim to the claims table
                let output_layer_id = output_layer.get_mle().layer_id();
                claim_tracker.insert(output_layer_id, vec![committed_output_claim]);
                output_layer_proof
            })
            .collect_vec();

        let layer_proofs = layers.layers
            .iter_mut()
            .rev()
            .map(|layer| {
                let claims = claim_tracker.get(&layer.layer_id()).unwrap().clone();
                let output_mles_from_layer = layer_map.get(&layer.layer_id()).unwrap();
                let (layer_proof, claims_from_layer) = HyraxLayerProof::prove(
                    layer,
                    &claims,
                    output_mles_from_layer,
                    committer,
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

        // Hyrax input layer proofs
        let hyrax_input_layer_claims = input_layers
            .iter()
            .filter_map(|input_layer| {
                let layer_id = input_layer.layer_id;
                claim_tracker.get(&layer_id)
            })
            .collect_vec();

        let claims_on_input_layers = input_layers
            .iter()
            .filter_map(|input_layer| {
                claim_tracker.get(&input_layer.layer_id)
            })
            .flatten()
            .cloned()
            .collect_vec();

        // --------- Verifier Challenges ---------
        // The the claims on verifier challenges are checked directly
        // by the verifier, without aggregation, so there is almost nothing to do here.  However,
        // the verifier received the prover's claims as committed form (this is just how we
        // implemented layer proof) and so we provide here the CommittedScalar forms in order for V
        // to be able to actually verify the claims.
        let fiat_shamir_claims = fiat_shamir_challenges
            .iter()
            .flat_map(|fiat_shamir_challenge| {
                claim_tracker
                    .get(&fiat_shamir_challenge.layer_id())
                    .unwrap()
            })
            .cloned()
            .collect_vec();

        (HyraxCircuitProof {
            layer_proofs,
            output_layer_proofs,
            fiat_shamir_claims,
        }, claims_on_input_layers)
    }

    pub fn verify_gkr_circuit(
        &self,
        proof: &HyraxProof<C>,
        circuit_description: &mut GKRCircuitDescription<C::Scalar>,
        verifier_transcript: &mut impl ECVerifierTranscript<C>,
    ) {
        // TODO(vishady, ryancao): add circuit description to verifier transcript as well!!

        // Append the public inputs to the transcript
        proof
            .public_inputs
            .iter()
            .for_each(|(_layer_id, public_input)| {
                let public_input = public_input.get_evals_vector();
                let transcript_values = verifier_transcript
                    .consume_scalar_points("public values", public_input.len())
                    .unwrap();
                assert_eq!(&transcript_values, public_input);
            });

        // Append the commitments to the private inputs to transcript
        proof.hyrax_input_proofs.iter().for_each(|input_proof| {
            let hyrax_commit = &input_proof.input_commitment;
            let transcript_hyrax_commit = verifier_transcript
                .consume_ec_points("hyrax pcs commitment", hyrax_commit.len())
                .unwrap();
            assert_eq!(&transcript_hyrax_commit, hyrax_commit);
        });

        // Get the verifier challenges from the transcript.
        let fiat_shamir_challenges: Vec<FiatShamirChallenge<C::Scalar>> = circuit_description
            .fiat_shamir_challenges
            .iter()
            .map(|fs_desc| {
                let num_evals = 1 << fs_desc.num_bits;
                let values = verifier_transcript
                    .get_scalar_field_challenges("Verifier challenges", num_evals)
                    .unwrap();
                fs_desc.instantiate(values)
            })
            .collect();

        circuit_description.index_mle_indices(0);

        // Get the commitments for hyrax input layers
        // FIXME(Ben) in the next refactor the HyraxVerifierCommitmentEnum will be removed, so will
        // the following transformation.
        let hyrax_input_commitments = proof
            .hyrax_input_proofs
            .iter()
            .map(|input_proof| (&input_proof.layer_id, &input_proof.input_commitment))
            .collect_vec();

        let verify_timer = start_timer!(|| "verify hyrax circuit");
        let input_layer_claims = Self::verify(
            &proof.circuit_proof,
            circuit_description,
            self.committer,
            fiat_shamir_challenges,
            verifier_transcript,
        );

        // Check the claims on the public input layers
        // TODO Something like:
        // let public_input_layer_claims: Vec<HyraxClaim<C::Scalar, C>> = todo!();
        // public_inputs.iter().for_each(|(layer_id, values)| {
        //     let claims_as_commitments = claim_tracker.remove(layer_id).unwrap();
        //     let plaintext_claims =
        //         Self::match_claims(&claims_as_commitments, claims_on_public_values, committer);
        //     plaintext_claims.into_iter().for_each(|claim| {
        //         verify_claim::<C::Scalar>(values, claim.get_claim());
        //     });
        // });

        // Verify the hyrax input layer proofs
        // TODO remember to match these up with the claims we obtained from calling verify() above
        // TODO Something like:
        // hyrax_input_proofs
        //     .iter()
        //     .zip(hyrax_input_commitments)
        //     .for_each(|(hyrax_input_proof, (layer_id, hyrax_input_commit))| {
        //         // Check that the commitment given also matches with the commitment in the proof
        //         assert_eq!(layer_id, &hyrax_input_proof.layer_id);
        //         assert_eq!(&hyrax_input_proof.input_commitment, hyrax_input_commit);
        //         let layer_id = hyrax_input_proof.layer_id;
        //         let layer_claims_vec = claim_tracker.remove(&layer_id).unwrap().clone();
        //         hyrax_input_proof.verify(&layer_claims_vec, committer, transcript);
        //     });

        end_timer!(verify_timer);
    }
    /// This is the verification of a GKR proof. It essentially calls the verify functions of the underlying proofs
    /// The calling context is responsible for appending to the transcript both the circuit
    /// description and the values and/or commitments of the input layer (which is appropriate
    /// unless already added further upstream).
    /// Returns a vector of claims on the input layers.
    pub fn verify(
        proof: &HyraxCircuitProof<C>,
        circuit_description: &GKRCircuitDescription<C::Scalar>,
        committer: &PedersenCommitter<C>,
        fiat_shamir_challenges: Vec<FiatShamirChallenge<C::Scalar>>,
        transcript: &mut impl ECVerifierTranscript<C>,
    ) -> Vec<HyraxClaim<C::Scalar, C>> {
        // Unpack the Hyrax proof.
        let HyraxCircuitProof {
            layer_proofs,
            output_layer_proofs,
            fiat_shamir_claims,
        } = proof;

        // Keep track of all claim commitments for the hyrax layer verifier
        let mut claim_tracker: HashMap<LayerId, Vec<HyraxClaim<C::Scalar, C>>> = HashMap::new();

        // Output layer verification
        output_layer_proofs
            .iter()
            .zip(circuit_description.output_layers.iter())
            .for_each(|(output_layer_proof, output_layer_desc)| {
                let output_layer_claim = HyraxOutputLayerProof::verify(
                    output_layer_proof,
                    output_layer_desc,
                    transcript,
                );

                // Add the output claim to the claims table
                claim_tracker.insert(output_layer_claim.to_layer_id, vec![output_layer_claim]);
            });

        // Intermediate layer verification
        (layer_proofs
            .iter()
            .zip(circuit_description.intermediate_layers.iter().rev()))
        .for_each(|(layer_proof, layer_desc)| {
            // Get the unaggregated claims for this layer
            // V checked that these claims had the expected form before adding them to the claim tracking table
            let layer_claims_vec = claim_tracker
                .remove(&layer_desc.layer_id())
                .unwrap()
                .clone();
            let claim_commits_for_layer = HyraxLayerProof::verify(
                layer_proof,
                layer_desc,
                &layer_claims_vec,
                committer,
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

        // Check the claims on the verifier challenges
        fiat_shamir_challenges
            .iter()
            .for_each(|fiat_shamir_challenge| {
                let claims_as_commitments = claim_tracker
                    .remove(&fiat_shamir_challenge.layer_id())
                    .unwrap();
                match_claims(&claims_as_commitments, fiat_shamir_claims, committer)
                    .iter()
                    .for_each(|plaintext_claim| {
                        fiat_shamir_challenge
                            .verify(plaintext_claim.get_claim())
                            .unwrap();
                    });
            });

        // Collect the claims on the input layers
        let input_layer_claims = circuit_description
            .input_layers
            .iter()
            .filter_map(|input_layer| {
                claim_tracker.get(&input_layer.layer_id)
            })
            .flatten()
            .cloned()
            .collect_vec();

        // Check that there aren't any claims left in our claim tracking table!
        assert_eq!(claim_tracker.len(), 0);

        input_layer_claims
    }
}

/// Match up the claims from the verifier with the claims from the prover. Used for proofs of
/// evaluation on public values where the proof (in the case of [PublicInputLayer] and
/// [FiatShamirChallenge] ) consists of the prover simply opening the commitments in the claims,
/// or equivalently just handing over the CommittedScalars. Panics if a verifier claim can not
/// be matched to a prover claim (but doesn't worry about prover claims that don't have a
/// verifier counterpart).
/// Also checks that any matched claims are consistent with the committer (panics if not).
pub fn match_claims<C: PrimeOrderCurve>(
    verifier_claims: &[HyraxClaim<C::Scalar, C>],
    prover_claims: &[HyraxClaim<C::Scalar, CommittedScalar<C>>],
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
