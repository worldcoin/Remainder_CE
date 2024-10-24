#![allow(clippy::type_complexity)]
use std::collections::HashMap;

use crate::utils::vandermonde::VandermondeInverse;
use hyrax_input_layer::{
    commit_to_input_values, verify_claim, HyraxInputLayerDescription, HyraxInputLayerProof,
    HyraxProverInputCommitment,
};
use hyrax_layer::HyraxClaim;
use hyrax_output_layer::HyraxOutputLayerProof;
use itertools::Itertools;
use rand::Rng;
use remainder::claims::{Claim, RawClaim};
use remainder::expression::circuit_expr::filter_bookkeeping_table;
use remainder::input_layer::fiat_shamir_challenge::FiatShamirChallenge;
use remainder::input_layer::{InputLayer, InputLayerDescription};
use remainder::layer::layer_enum::LayerEnum;
use remainder::layer::LayerId;
use remainder::layer::{Layer, LayerDescription};
use remainder::mle::evals::MultilinearExtension;
use remainder::mle::Mle;
use remainder::prover::{generate_circuit_description, GKRCircuitDescription, InstantiatedCircuit};

use remainder_shared_types::curves::PrimeOrderCurve;
use remainder_shared_types::pedersen::{CommittedScalar, PedersenCommitter};
use remainder_shared_types::transcript::ec_transcript::ECTranscriptTrait;

use self::hyrax_layer::HyraxLayerProof;

pub mod helpers;
/// The module that contains all functions necessary to do operations on a Hyrax input layer using the Hyrax PCS.
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
    /// The MLEs of the public inputs, along with their layer ids.
    /// To be appended to transcript in order of LayerId ascending.
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
    /// Create a [HyraxProof].
    /// Values of public input layers are appended to transcript in order of `LayerId` value,
    /// ascending. Then Hyrax commitments are appended to transcript in order of `LayerId` value,
    /// ascending; this is also the ordering of `HyraxProof.hyrax_input_proofs`.
    ///
    /// # Arguments:
    /// * `inputs` - The MLEs of _all_ inputs (including Hyrax inputs), along with their layer ids.
    /// * `hyrax_input_layers` - The descriptions of the Hyrax input layers, along with (optionally) precommits.
    /// * `circuit_description` - The description of the circuit to be proven.
    /// * `committer` - The Pedersen committer to be used for commitments.
    /// * `rng` - The random number generator to be used for randomness.
    /// * `converter` - The Vandermonde inverse converter to be used for the proof.
    /// * `transcript` - The transcript to be used for Fiat-Shamir challenges.
    ///
    /// # Requires:
    ///   * `circuit_description.index_mle_indices(0)` has been called
    pub fn prove(
        inputs: &HashMap<LayerId, MultilinearExtension<C::Scalar>>,
        hyrax_input_layers: &HashMap<
            LayerId,
            (
                HyraxInputLayerDescription,
                Option<HyraxProverInputCommitment<C>>,
            ),
        >,
        circuit_description: &GKRCircuitDescription<C::Scalar>,
        committer: &PedersenCommitter<C>,
        mut rng: &mut impl Rng,
        converter: &mut VandermondeInverse<C::Scalar>,
        transcript: &mut impl ECTranscriptTrait<C>,
    ) -> HyraxProof<C> {
        // Add the input values of any public (i.e. non-hyrax) input layers to transcript.
        // Select the public input layers from the input layers, and sort them by layer id, and append
        // their input values to the transcript.
        inputs
            .keys()
            .filter(|layer_id| !hyrax_input_layers.contains_key(layer_id))
            .sorted_by_key(|layer_id| layer_id.get_raw_input_layer_id())
            .for_each(|layer_id| {
                let mle = inputs.get(layer_id).unwrap();
                transcript.append_scalar_points("input layer", &mle.f.iter().collect_vec());
            });

        // For each hyrax input layer, calculate commitments if not already provided, and then append each
        // commitment to the transcript.
        let mut hyrax_input_commitments = HashMap::<LayerId, HyraxProverInputCommitment<C>>::new();
        hyrax_input_layers
            .keys()
            .sorted_by_key(|layer_id| layer_id.get_raw_input_layer_id())
            .for_each(|layer_id| {
                // Commit to the Hyrax input layer, if it is not already committed to.
                let (desc, maybe_precommitment) = hyrax_input_layers.get(layer_id).unwrap();
                let prover_commitment: HyraxProverInputCommitment<C> =
                    if let Some(prover_commitment) = maybe_precommitment {
                        // Use the commitment provided by the calling context
                        (*prover_commitment).clone()
                    } else {
                        // Commit to the values of the input layer
                        let input_mle = inputs.get(layer_id).unwrap();
                        commit_to_input_values(desc, input_mle, committer, &mut rng)
                    };

                // Add the verifier's view of the commitment to transcript
                transcript
                    .append_ec_points("Hyrax input layer values", &prover_commitment.commitment);

                // Store the prover's view for later use in the evaluation proofs.
                hyrax_input_commitments.insert(*layer_id, prover_commitment);
            });

        // Get the verifier challenges from the transcript.
        let mut challenge_sampler =
            |size| transcript.get_scalar_field_challenges("Verifier challenges", size);
        let mut instantiated_circuit =
            circuit_description.instantiate(inputs, &mut challenge_sampler);

        let (circuit_proof, claims_on_input_layers) = HyraxCircuitProof::prove(
            &mut instantiated_circuit,
            committer,
            &mut rng,
            converter,
            transcript,
        );

        // Collect the values of the public inputs
        let public_inputs = inputs
            .keys()
            .filter(|layer_id| !hyrax_input_layers.contains_key(layer_id))
            .map(|layer_id| {
                let mle = inputs.get(layer_id).unwrap();
                (*layer_id, mle.clone())
            })
            .collect_vec();

        // Separate the claims on input layers into claims on public input layers vs claims on Hyrax
        // input layers
        let mut claims_on_public_values = vec![];
        let mut claims_on_hyrax_input_layers =
            HashMap::<LayerId, Vec<HyraxClaim<C::Scalar, CommittedScalar<C>>>>::new();
        claims_on_input_layers.iter().for_each(|claim| {
            if hyrax_input_layers.contains_key(&claim.to_layer_id) {
                if let Some(curr_claims) = claims_on_hyrax_input_layers.get_mut(&claim.to_layer_id)
                {
                    curr_claims.push(claim.clone());
                } else {
                    claims_on_hyrax_input_layers.insert(claim.to_layer_id, vec![claim.clone()]);
                }
            } else {
                claims_on_public_values.push(claim.clone());
            }
        });

        // If in debug mode, then check the claims on all input layers.
        if cfg!(debug_assertions) {
            for claim in claims_on_input_layers.iter() {
                let input_mle = inputs.get(&claim.to_layer_id).unwrap();
                let evaluation = input_mle.evaluate_at_point(&claim.point);
                if evaluation != claim.evaluation.value {
                    panic!(
                        "Claim on input layer {} does not match evaluation",
                        claim.to_layer_id
                    );
                }
            }
        }

        // Prove the claims on the Hyrax input layers
        let hyrax_input_proofs = hyrax_input_layers
            .keys()
            .sorted_by_key(|layer_id| layer_id.get_raw_input_layer_id())
            .map(|layer_id| {
                let (desc, _) = hyrax_input_layers.get(layer_id).unwrap();
                let commitment = hyrax_input_commitments.get(layer_id).unwrap();
                let committed_claims = claims_on_hyrax_input_layers.remove(layer_id).unwrap();

                HyraxInputLayerProof::prove(
                    desc,
                    commitment,
                    &committed_claims,
                    committer,
                    &mut rng,
                    transcript,
                    converter,
                )
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

    /// Verify this [HyraxProof] instance, matching it against the provided circuit description and
    /// descriptions of the hyrax input layers.
    /// Panics if verification fails.
    /// # Requires:
    ///   * `circuit_description.index_mle_indices(0)` has been called
    pub fn verify(
        &self,
        hyrax_input_layers: &HashMap<LayerId, HyraxInputLayerDescription>,
        circuit_description: &GKRCircuitDescription<C::Scalar>,
        committer: &PedersenCommitter<C>,
        transcript: &mut impl ECTranscriptTrait<C>,
    ) {
        // Append the public inputs to the transcript
        self.public_inputs
            .iter()
            .sorted_by_key(|(layer_id, _)| layer_id.get_raw_input_layer_id())
            .for_each(|(_layer_id, mle)| {
                transcript.append_scalar_points("input layer", &mle.f.iter().collect_vec());
            });

        // For each Hyrax input layer commitment (in order of LayerId), consume elements from the transcript and check they match the commitments contained in the HyraxInputLayerProof.
        self.hyrax_input_proofs
            .iter()
            .sorted_by_key(|input_proof| input_proof.layer_id.get_raw_input_layer_id())
            .for_each(|input_proof| {
                let hyrax_commit = &input_proof.input_commitment;
                transcript.append_ec_points("Hyrax input layer values", hyrax_commit);
            });

        // Get the verifier challenges from the transcript.
        let fiat_shamir_challenges: Vec<FiatShamirChallenge<C::Scalar>> = circuit_description
            .fiat_shamir_challenges
            .iter()
            .map(|fs_desc| {
                let num_evals = 1 << fs_desc.num_bits;
                let values =
                    transcript.get_scalar_field_challenges("Verifier challenges", num_evals);
                fs_desc.instantiate(values)
            })
            .collect();

        // Verify the circuit proof, and obtain the claims on the input layers
        let input_layer_claims_vec = self.circuit_proof.verify(
            circuit_description,
            committer,
            fiat_shamir_challenges,
            transcript,
        );
        let mut input_layer_claims: HashMap<LayerId, Vec<HyraxClaim<C::Scalar, C>>> =
            HashMap::new();
        input_layer_claims_vec.into_iter().for_each(|claim| {
            if let std::collections::hash_map::Entry::Vacant(e) =
                input_layer_claims.entry(claim.to_layer_id)
            {
                e.insert(vec![claim]);
            } else {
                input_layer_claims
                    .get_mut(&claim.to_layer_id)
                    .unwrap()
                    .push(claim);
            }
        });

        // For each public input layer, pop the claims, match them up with the corresponding claims
        // on public values provided by the prover, and verify them directly.
        self.public_inputs.iter().for_each(|(layer_id, values)| {
            let claims_as_commitments = input_layer_claims.remove(layer_id).unwrap();
            let plaintext_claims = match_claims(
                &claims_as_commitments,
                &self.claims_on_public_values,
                committer,
            );
            plaintext_claims.into_iter().for_each(|claim| {
                verify_claim::<C::Scalar>(&values.f.iter().collect_vec(), &claim);
            });
        });

        // Verify the hyrax input layer proofs.
        self.hyrax_input_proofs
            .iter()
            .for_each(|hyrax_input_proof| {
                let layer_id = &hyrax_input_proof.layer_id;
                let desc = hyrax_input_layers.get(layer_id).unwrap();
                let layer_claims_vec = input_layer_claims.remove(layer_id).unwrap();
                hyrax_input_proof.verify(desc, &layer_claims_vec, committer, transcript);
            });

        // Check that there are no claims left in the input layer claims table.
        assert!(input_layer_claims.is_empty());
    }
}

/// The struct that holds all the information that the prover sends to the verifier about the circuit proof, i.e. the proof that transforms the claims on the output layers to claims on the input layers.
pub struct HyraxCircuitProof<C: PrimeOrderCurve> {
    /// The [HyraxLayerProof] for each of the intermediate layers in this circuit.
    pub layer_proofs: Vec<(LayerId, HyraxLayerProof<C>)>,
    /// A commitment to the output of the circuit, i.e. what the final value of the output layer is.
    pub output_layer_proofs: Vec<(LayerId, HyraxOutputLayerProof<C>)>,
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
        transcript: &mut impl ECTranscriptTrait<C>,
    ) -> (
        HyraxCircuitProof<C>,
        Vec<HyraxClaim<C::Scalar, CommittedScalar<C>>>,
    ) {
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
            .sorted_by_key(|output_layer| output_layer.layer_id().get_raw_layer_id())
            .map(|output_layer| {
                // Create the HyraxOutputLayerProof
                let (output_layer_proof, committed_output_claim) =
                    HyraxOutputLayerProof::prove(output_layer, transcript, blinding_rng, committer);
                // Add the output claim to the claims table
                let output_layer_id = output_layer.get_mle().layer_id();
                claim_tracker.insert(output_layer_id, vec![committed_output_claim]);
                (output_layer_id, output_layer_proof)
            })
            .collect_vec();

        let layer_proofs = layers
            .layers
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

                (layer.layer_id(), layer_proof)
            })
            .collect_vec();

        let claims_on_input_layers = input_layers
            .iter()
            .filter_map(|input_layer| claim_tracker.get(&input_layer.layer_id))
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

        (
            HyraxCircuitProof {
                layer_proofs,
                output_layer_proofs,
                fiat_shamir_claims,
            },
            claims_on_input_layers,
        )
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
        transcript: &mut impl ECTranscriptTrait<C>,
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
            .sorted_by_key(|(output_layer_id, _)| output_layer_id.get_raw_layer_id())
            .for_each(|(output_layer_id, output_layer_proof)| {
                let output_layer_desc = circuit_description
                    .output_layers
                    .iter()
                    .find(|output_layer_desc| output_layer_desc.layer_id() == *output_layer_id)
                    .unwrap();
                let output_layer_claim = HyraxOutputLayerProof::verify(
                    output_layer_proof,
                    output_layer_desc,
                    transcript,
                );

                // Add the output claim to the claims table
                claim_tracker.insert(output_layer_claim.to_layer_id, vec![output_layer_claim]);
            });

        // Intermediate layer verification
        layer_proofs.iter().for_each(|(layer_id, layer_proof)| {
            // Get the layer description
            let layer_desc = circuit_description
                .intermediate_layers
                .iter()
                .find(|layer_desc| layer_desc.layer_id() == *layer_id)
                .unwrap();
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
                        fiat_shamir_challenge.verify(plaintext_claim).unwrap();
                    });
            });

        // Collect the claims on the input layers
        let input_layer_claims = circuit_description
            .input_layers
            .iter()
            .filter_map(|input_layer| claim_tracker.remove(&input_layer.layer_id))
            .flatten()
            .collect_vec();

        // Check that there aren't any claims left in our claim tracking table!
        assert_eq!(claim_tracker.len(), 0);

        input_layer_claims
    }
}

/// The struct that holds all the information that the prover sends to the verifier about the proof
/// of a [FiatShamirChallenge].  This mainly exists since in the main prover flow, P sends V
/// commitments to the claimed values, so V needs the prover versions of the commitments to check
/// the claims on the Fiat-Shamir challenges.
pub struct FiatShamirChallengeProof<C: PrimeOrderCurve> {
    pub layer: FiatShamirChallenge<C::Scalar>,
    pub claims: Vec<HyraxClaim<C::Scalar, CommittedScalar<C>>>,
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
) -> Vec<RawClaim<C::Scalar>> {
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
                committed_claim.to_raw_claim()
            } else {
                // TODO return an error instead of panicking
                panic!("Claim has not counterpart in committed claims!");
            }
        })
        .collect()
}
