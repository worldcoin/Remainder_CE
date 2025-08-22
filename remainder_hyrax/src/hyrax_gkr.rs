#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
use std::arch::aarch64::vreinterpret_f32_p16;
use std::collections::{HashMap, HashSet};

use crate::circuit_layout::{HyraxProvableCircuit, HyraxVerifiableCircuit};
use crate::utils::vandermonde::VandermondeInverse;
use anyhow::bail;
use ark_std::{end_timer, start_timer};
use hyrax_input_layer::{
    commit_to_input_values, HyraxInputLayerDescription, HyraxInputLayerProof,
    HyraxProverInputCommitment,
};
use hyrax_layer::HyraxClaim;
use hyrax_output_layer::HyraxOutputLayerProof;
use itertools::Itertools;
use rand::{CryptoRng, RngCore};
use remainder::claims::RawClaim;
use remainder::input_layer::fiat_shamir_challenge::FiatShamirChallenge;
use remainder::layer::LayerId;
use remainder::layer::{Layer, LayerDescription};
use remainder::mle::evals::MultilinearExtension;
use remainder::mle::Mle;
use remainder::prover::helpers::get_circuit_description_hash_as_field_elems;
use remainder::prover::{GKRCircuitDescription, InstantiatedCircuit};

use remainder::utils::mle::verify_claim;
use remainder_shared_types::config::global_config::{
    get_current_global_prover_config, get_current_global_verifier_config,
    global_prover_circuit_description_hash_type, global_verifier_circuit_description_hash_type,
};
use remainder_shared_types::config::ProofConfig;
use remainder_shared_types::curves::PrimeOrderCurve;
use remainder_shared_types::pedersen::{CommittedScalar, PedersenCommitter};
use remainder_shared_types::transcript::ec_transcript::ECTranscriptTrait;
use serde::{Deserialize, Serialize};

use self::hyrax_layer::HyraxLayerProof;

pub mod helpers;
/// The module that contains all functions necessary to do operations on a Hyrax
/// input layer using the Hyrax PCS.
pub mod hyrax_input_layer;
/// The module that has all functions and implementations necessary to produce a
/// [HyraxLayerProof]
pub mod hyrax_layer;
/// The module that contains all functions necessary to do operations on an
/// output layer.
pub mod hyrax_output_layer;

/// The struct that holds all the respective proofs that the verifier needs in
/// order to verify a Hyrax proof, i.e. the circuit proof along with the proofs
/// for each input layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "C: PrimeOrderCurve")]
pub struct HyraxProof<C: PrimeOrderCurve> {
    /// The MLEs of the public inputs, along with their layer ids. To be
    /// appended to transcript in order of LayerId ascending.
    pub public_inputs: Vec<(LayerId, Option<MultilinearExtension<C::Scalar>>)>,
    /// The proof for the circuit proper, i.e. the intermediate layers and
    /// output layers.
    pub circuit_proof: HyraxCircuitProof<C>,
    /// The prover's claims on public input layers, in CommittedScalar form,
    /// i.e. including the blinding factors (since the verifier needs to check
    /// these itself).
    pub claims_on_public_values: Vec<HyraxClaim<C::Scalar, CommittedScalar<C>>>,
    /// The [HyraxInputLayerProof] for each of the input polynomial commitments
    /// using the Hyrax PCS.
    pub hyrax_input_proofs: Vec<HyraxInputLayerProof<C>>,
}

impl<C: PrimeOrderCurve> HyraxProof<C> {
    pub fn remove_public_input_layer_by_id(&mut self, id_to_remove: LayerId) {
        let pos = self
            .public_inputs
            .iter()
            .position(|(id, _)| id == &id_to_remove)
            .unwrap();

        self.public_inputs[pos] = (id_to_remove, None);
    }

    pub fn insert_aux_public_data_by_id(
        &mut self,
        aux_mle: &MultilinearExtension<C::Scalar>,
        id_to_insert: LayerId,
    ) {
        let aux_mle_pos = self
            .public_inputs
            .iter()
            .position(|(id, _)| id == &id_to_insert)
            .unwrap();
        self.public_inputs[aux_mle_pos] = (id_to_insert, Some(aux_mle.clone()));
    }

    #[cfg(feature = "print-trace")]
    pub fn print_size(&self) {
        let public_inputs_size = bincode::serialize(&self.public_inputs).unwrap().len();
        let circuit_proof_size = bincode::serialize(&self.circuit_proof).unwrap().len();
        let claims_size = bincode::serialize(&self.claims_on_public_values)
            .unwrap()
            .len();
        let hyrax_input_proofs = bincode::serialize(&self.hyrax_input_proofs).unwrap().len();

        println!("LayerIDs of public input: ");
        for pi in &self.public_inputs {
            println!(
                "{:?} of size {} MB, ",
                pi.0,
                bincode::serialize(&pi.1).unwrap().len() as f64 / 1_000_000.0
            );
        }
        println!(
            "{} MB, {} MB, {} MB, {} MB",
            public_inputs_size as f64 / 1_000_000.0,
            circuit_proof_size as f64 / 1_000_000.0,
            claims_size as f64 / 1_000_000.0,
            hyrax_input_proofs as f64 / 1_000_000.0,
        );
    }
    /// Create a [HyraxProof]. Values of public input layers are appended to
    /// transcript in order of `LayerId` value, ascending. Then Hyrax
    /// commitments are appended to transcript in order of `LayerId` value,
    /// ascending; this is also the ordering of `HyraxProof.hyrax_input_proofs`.
    ///
    /// # Arguments:
    /// * `inputs` - The MLEs of _all_ inputs (including Hyrax inputs), along
    ///   with their layer ids.
    /// * `hyrax_input_layers` - The descriptions of the Hyrax input layers,
    ///   along with (optionally) precommits.
    /// * `circuit_description` - The description of the circuit to be proven.
    /// * `committer` - The Pedersen committer to be used for commitments.
    /// * `rng` - The random number generator to be used for randomness.
    /// * `converter` - The Vandermonde inverse converter to be used for the
    ///   proof.
    /// * `transcript` - The transcript to be used for Fiat-Shamir challenges.
    ///
    /// # Requires:
    ///   * `circuit_description.index_mle_indices(0)` has been called
    pub fn prove(
        provable_circuit: &mut HyraxProvableCircuit<C>,
        /*
        inputs: &HashMap<LayerId, MultilinearExtension<C::Scalar>>,
        hyrax_input_layers: &HashMap<
            LayerId,
            (
                HyraxInputLayerDescription,
                Option<HyraxProverInputCommitment<C>>,
            ),
        >,
        circuit_description: &GKRCircuitDescription<C::Scalar>,
        */
        committer: &PedersenCommitter<C>,
        mut rng: &mut (impl CryptoRng + RngCore),
        converter: &mut VandermondeInverse<C::Scalar>,
        transcript: &mut impl ECTranscriptTrait<C>,
    ) -> (HyraxProof<C>, ProofConfig) {
        // Get proof config from global config
        let proof_config = ProofConfig::new_from_prover_config(&get_current_global_prover_config());

        // Generate circuit description hash and append to transcript
        let hash_value_as_field_elems = get_circuit_description_hash_as_field_elems(
            provable_circuit.get_gkr_circuit_description_ref(),
            global_prover_circuit_description_hash_type(),
        );
        transcript
            .append_scalar_field_elems("Circuit description hash", &hash_value_as_field_elems);

        // Add the input values of any public (i.e. non-hyrax) input layers to
        // transcript. Select the public input layers from the input layers, and
        // sort them by layer id, and append their input values to the
        // transcript.
        provable_circuit
            .get_inputs_ref()
            .keys()
            .filter(|layer_id| {
                !provable_circuit
                    .get_private_input_layer_ids()
                    .contains(layer_id)
            })
            .sorted_by_key(|layer_id| layer_id.get_raw_input_layer_id())
            .for_each(|layer_id| {
                let mle = provable_circuit.get_input_mle(*layer_id).unwrap();
                let public_il_to_transcript_timer =
                    start_timer!(|| format!("adding il elements to transcript for {layer_id}"));
                transcript.append_input_scalar_field_elems(
                    "Public input layer values",
                    &mle.f.iter().collect_vec(),
                );
                end_timer!(public_il_to_transcript_timer);
            });

        provable_circuit.commit(committer, rng, transcript);

        // Get the verifier challenges from the transcript.
        let mut challenge_sampler =
            |size| transcript.get_scalar_field_challenges("Verifier challenges", size);
        let instantiation_timer = start_timer!(|| "instantiate circuit");

        // Instantiate the circuit description given the data from sampling
        // verifier challenges.
        let mut instantiated_circuit = provable_circuit
            .get_gkr_circuit_description_ref()
            .clone()
            .instantiate(provable_circuit.get_inputs_ref(), &mut challenge_sampler);
        end_timer!(instantiation_timer);

        // Generate the circuit proof, which is, starting from claims generated
        // on the output layers, is the proof of the intermediate layers,
        // resulting in claims on the input layers.
        //
        // NOTE: The `claims_on_input_layers` are in a deterministic order;
        // namely the claims are in reverse order of the layers making the
        // claim. The verifier has a claim tracker populated in the same order.
        // Additionally, claims generated are always made from "left to right"
        // when viewing a layer as an expression in terms of other layers.
        let layer_proving_timer = start_timer!(|| "proving intermediate layers");
        let (circuit_proof, claims_on_input_layers) = HyraxCircuitProof::prove(
            &mut instantiated_circuit,
            committer,
            &mut rng,
            converter,
            transcript,
        );
        end_timer!(layer_proving_timer);

        // Collect the values of the public inputs
        let public_inputs = provable_circuit
            .get_inputs_ref()
            .keys()
            .filter(|layer_id| !provable_circuit.is_private_input_layer(**layer_id))
            .map(|layer_id| {
                let mle = provable_circuit.get_input_mle(*layer_id).unwrap();
                (*layer_id, Some(mle.clone()))
            })
            .collect_vec();

        // Separate the claims on input layers into claims on public input
        // layers vs claims on Hyrax input layers
        let mut claims_on_public_values = vec![];
        let mut claims_on_hyrax_input_layers =
            HashMap::<LayerId, Vec<HyraxClaim<C::Scalar, CommittedScalar<C>>>>::new();
        claims_on_input_layers.iter().for_each(|claim| {
            if provable_circuit.is_private_input_layer(claim.to_layer_id) {
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
                let input_mle = provable_circuit.get_input_mle(claim.to_layer_id).unwrap();
                let public_il_verification_timer =
                    start_timer!(|| format!("public il eval for {0}", claim.to_layer_id));
                let evaluation = input_mle.evaluate_at_point(&claim.point);
                if evaluation != claim.evaluation.value {
                    panic!(
                        "Claim on input layer {} does not match evaluation",
                        claim.to_layer_id
                    );
                }
                end_timer!(public_il_verification_timer);
            }
        }

        // Prove the claims on the Hyrax input layers
        let hyrax_input_proofs = provable_circuit
            .get_private_input_layer_ids()
            .iter()
            .sorted_by_key(|layer_id| layer_id.get_raw_input_layer_id())
            .map(|layer_id| {
                if let (desc, Some(commitment)) = provable_circuit
                    .get_private_input_layer_mut_ref(*layer_id)
                    .unwrap()
                {
                    // let commitment = provable_circuit.get_commitment_mut_ref(layer_id).unwrap();
                    let committed_claims = claims_on_hyrax_input_layers.remove(layer_id).unwrap();

                    let hyrax_il_verification_timer = start_timer!(|| format!(
                        "HyraxInputLayer::prove for {0} with {1} claims",
                        layer_id,
                        committed_claims.len()
                    ));
                    let il_proof = HyraxInputLayerProof::prove(
                        &desc,
                        commitment,
                        &committed_claims,
                        committer,
                        &mut rng,
                        transcript,
                    );
                    end_timer!(hyrax_il_verification_timer);
                    il_proof
                } else {
                    panic!("Input layer with ID {layer_id} missing committment!")
                }
            })
            .collect_vec();

        // Check that now Hyrax input layer claims remain
        assert!(claims_on_hyrax_input_layers.is_empty());

        (
            HyraxProof {
                public_inputs,
                circuit_proof,
                claims_on_public_values,
                hyrax_input_proofs,
            },
            proof_config,
        )
    }
}

/// Verify this [HyraxProof] instance, matching it against the provided circuit
/// description and descriptions of the hyrax input layers. Panics if
/// verification fails.
/// # Requires:
///   * `circuit_description.index_mle_indices(0)` has been called
pub fn verify_hyrax_proof<C: PrimeOrderCurve>(
    hyrax_proof: &HyraxProof<C>,
    verifiable_circuit: &HyraxVerifiableCircuit<C>,
    /*
    hyrax_input_layers: &HashMap<LayerId, HyraxInputLayerDescription>,
    circuit_description: &GKRCircuitDescription<C::Scalar>,
    */
    committer: &PedersenCommitter<C>,
    transcript: &mut impl ECTranscriptTrait<C>,
    proof_config: &ProofConfig,
) {
    // Check that the verifier config we are about to run with matches the proof config.
    if !get_current_global_verifier_config().matches_proof_config(proof_config) {
        panic!("Error: Attempting to verify with a different config than the one proposed by the proof.");
    }

    // Check that the number of layers in the circuit description matches the number of layers
    // in the proof.
    assert_eq!(
        hyrax_proof.hyrax_input_proofs.len() + hyrax_proof.public_inputs.len(),
        verifiable_circuit
            .get_gkr_circuit_description_ref()
            .input_layers
            .len()
    );
    assert_eq!(
        verifiable_circuit.get_private_inputs_ref().len(),
        hyrax_proof.hyrax_input_proofs.len()
    );
    assert_eq!(
        hyrax_proof.circuit_proof.layer_proofs.len(),
        verifiable_circuit
            .get_gkr_circuit_description_ref()
            .intermediate_layers
            .len()
    );
    assert_eq!(
        hyrax_proof.circuit_proof.output_layer_proofs.len(),
        verifiable_circuit
            .get_gkr_circuit_description_ref()
            .output_layers
            .len()
    );

    // Generate circuit description hash and append to transcript Note that this
    // is different from the GKR case since there we are checking against the
    // prover-provided transcript elements but here we are simply building up
    // our own transcript.
    let hash_timer = start_timer!(|| "Hashing circuit description");
    let hash_value_as_field_elems = get_circuit_description_hash_as_field_elems(
        verifiable_circuit.get_gkr_circuit_description_ref(),
        global_verifier_circuit_description_hash_type(),
    );
    end_timer!(hash_timer);
    transcript.append_scalar_field_elems("Circuit description hash", &hash_value_as_field_elems);

    // Append the public inputs to the transcript
    hyrax_proof
        .public_inputs
        .iter()
        .sorted_by_key(|(layer_id, _)| layer_id.get_raw_input_layer_id())
        .for_each(|(_layer_id, mle)| {
            let il_timer = start_timer!(|| format!(
                "Appending public input layer for {_layer_id} to transcript"
            ));
            transcript.append_input_scalar_field_elems(
                "Public input layer values",
                &mle.as_ref().unwrap().f.iter().collect_vec(),
            );
            end_timer!(il_timer);
        });

    // For each Hyrax input layer commitment (in order of LayerId), consume
    // elements from the transcript and check they match the commitments
    // contained in the HyraxInputLayerProof.
    let hyrax_input_timer = start_timer!(|| "Checking Hyrax input layer commitments");
    hyrax_proof
        .hyrax_input_proofs
        .iter()
        .sorted_by_key(|input_proof| input_proof.layer_id.get_raw_input_layer_id())
        .for_each(|input_proof| {
            transcript.append_input_ec_points(
                "Hyrax input layer commitment",
                input_proof.input_commitment.clone(),
            );
        });
    end_timer!(hyrax_input_timer);

    // Get the verifier challenges from the transcript.
    let fiat_shamir_challenges: Vec<FiatShamirChallenge<C::Scalar>> = verifiable_circuit
        .get_gkr_circuit_description_ref()
        .fiat_shamir_challenges
        .iter()
        .map(|fs_desc| {
            let num_evals = 1 << fs_desc.num_bits;
            let values = transcript.get_scalar_field_challenges("Verifier challenges", num_evals);
            fs_desc.instantiate(values)
        })
        .collect();

    // Verify the circuit proof, and obtain the claims on the input layers
    //
    // NOTE: The `input_layer_claims_vec` are in a deterministic order; namely
    // the claims are in reverse order of the layers making the claim. The
    // prover has a claim tracker populated in the same order. Additionally,
    // claims generated are always made from "left to right" when viewing
    // a layer as an expression in terms of other layers.
    let circuit_timer = start_timer!(|| "Verifying circuit proof");
    let input_layer_claims_vec = hyrax_proof.circuit_proof.verify(
        verifiable_circuit.get_gkr_circuit_description_ref(),
        committer,
        fiat_shamir_challenges,
        transcript,
    );
    end_timer!(circuit_timer);
    let mut input_layer_claims: HashMap<LayerId, Vec<HyraxClaim<C::Scalar, C>>> = HashMap::new();
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

    // For each public input layer, pop the claims, match them up with the
    // corresponding claims on public values provided by the prover, and verify
    // them directly.
    hyrax_proof
        .public_inputs
        .iter()
        .for_each(|(layer_id, values)| {
            let values = values.as_ref().unwrap();
            let input_layer_description = verifiable_circuit
                .get_gkr_circuit_description_ref()
                .input_layers
                .iter()
                .find(|input_layer| input_layer.layer_id == *layer_id)
                .unwrap();
            // Check the shape of the input layer description against the input layer in the proof.
            assert_eq!(input_layer_description.num_vars, values.num_vars());
            let claims_as_commitments = input_layer_claims.remove(layer_id).unwrap();
            let plaintext_claims = match_claims(
                &claims_as_commitments,
                &hyrax_proof.claims_on_public_values,
                committer,
            );
            let timer = start_timer!(|| format!(
                "Verifying {0} claims for {1} (public)",
                plaintext_claims.len(),
                layer_id
            ));
            plaintext_claims.into_iter().for_each(|claim| {
                verify_claim::<C::Scalar>(&values.f.iter().collect_vec(), &claim);
            });
            end_timer!(timer);
        });

    // Verify the hyrax input layer proofs.
    hyrax_proof
        .hyrax_input_proofs
        .iter()
        .for_each(|hyrax_input_proof| {
            let layer_id = &hyrax_input_proof.layer_id;
            let desc = verifiable_circuit
                .get_private_inputs_ref()
                .get(layer_id)
                .unwrap();
            let layer_claims_vec = input_layer_claims.remove(layer_id).unwrap();
            let timer = start_timer!(|| format!(
                "Verifying {0} claims for Hyrax input layer {1}",
                layer_claims_vec.len(),
                layer_id
            ));
            hyrax_input_proof.verify(desc, &layer_claims_vec, committer, transcript);
            end_timer!(timer);
        });

    // Check that there are no claims left in the input layer claims table.
    assert!(input_layer_claims.is_empty());
}

/// The struct that holds all the information that the prover sends to the
/// verifier about the circuit proof, i.e. the proof that transforms the claims
/// on the output layers to claims on the input layers.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "C: PrimeOrderCurve")]
pub struct HyraxCircuitProof<C: PrimeOrderCurve> {
    /// The [HyraxLayerProof] for each of the intermediate layers in this
    /// circuit.
    pub layer_proofs: Vec<(LayerId, HyraxLayerProof<C>)>,
    /// A commitment to the output of the circuit, i.e. what the final value of
    /// the output layer is.
    pub output_layer_proofs: Vec<(LayerId, HyraxOutputLayerProof<C>)>,
    /// The prover's claims on verifier challenges, in CommittedScalar form,
    /// i.e. including the blinding factors (since the verifier needs to check
    /// these itself).
    pub fiat_shamir_claims: Vec<HyraxClaim<C::Scalar, CommittedScalar<C>>>,
}

impl<C: PrimeOrderCurve> HyraxCircuitProof<C> {
    /// The calling context is responsible for appending to the transcript both
    /// the circuit description and the values and/or commitments of the input
    /// layer (which is appropriate unless already added further upstream).
    /// Returns the [HyraxCircuitProof] instance, along with a vector of claims
    /// on input layers, to be proven by the calling context.
    pub fn prove(
        instantiated_circuit: &mut InstantiatedCircuit<C::Scalar>,
        committer: &PedersenCommitter<C>,
        blinding_rng: &mut (impl CryptoRng + RngCore),
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
                let output_mles_from_layer = layer_map.remove(&layer.layer_id()).unwrap();
                let layer_timer = start_timer!(|| format!("Proving layer {}", layer.layer_id()));
                let (layer_proof, claims_from_layer) = HyraxLayerProof::prove(
                    layer,
                    &claims,
                    output_mles_from_layer,
                    committer,
                    blinding_rng,
                    transcript,
                    converter,
                );
                // Add new claims to the claim tracking table.
                //
                // NOTE: Claims are always added in the order of claims made by
                // the layers in reverse order. These are stored in a Vec to
                // preserve order, which is important for the proof of claim
                // aggregation. Additionally, claims generated are always made
                // from "left to right" when viewing a layer as an expression in
                // terms of other layers.
                for claim in claims_from_layer.into_iter() {
                    if let Some(curr_claims) = claim_tracker.get_mut(&claim.to_layer_id) {
                        curr_claims.push(claim);
                    } else {
                        claim_tracker.insert(claim.to_layer_id, vec![claim]);
                    }
                }
                end_timer!(layer_timer);
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
        //
        // The the claims on verifier challenges are checked directly by the
        // verifier, without aggregation, so there is almost nothing to do here.
        // However, the verifier received the prover's claims in committed form
        // (this is just how we implemented layer proof) and so we provide here
        // the CommittedScalar forms in order for V to be able to actually
        // verify the claims.
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

    /// The calling context is responsible for appending to the transcript both
    /// the circuit description and the values and/or commitments of the input
    /// layer (which is appropriate unless already added further upstream).
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

        // Check that all output layers have *exactly* one proof associated
        // with them.
        let output_layer_ids: HashSet<LayerId> = circuit_description
            .output_layers
            .iter()
            .map(|output_layer| output_layer.layer_id())
            .collect();
        let output_layer_proof_ids: HashSet<LayerId> = output_layer_proofs
            .iter()
            .map(|(layer_id, _output_layer)| *layer_id)
            .collect();
        assert_eq!(output_layer_ids, output_layer_proof_ids);
        assert_eq!(
            circuit_description.output_layers.len(),
            output_layer_proofs.len()
        );

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
            // Get the unaggregated claims for this layer V checked that these
            // claims had the expected form before adding them to the claim
            // tracking table
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

            // Add new claims to the claim tracking table.
            //
            // NOTE: Claims are always added in the order of claims made by the
            // layers in reverse order. These are stored in a Vec to preserve
            // order, which is important for the proof of claim aggregation.
            // Additionally, claims generated are always made from "left to
            // right" when viewing a layer as an expression in terms of other
            // layers.
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
                        verify_claim(&fiat_shamir_challenge.mle.to_vec(), plaintext_claim);
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

/// The struct that holds all the information that the prover sends to the
/// verifier about the proof of a [FiatShamirChallenge].  This mainly exists
/// since in the main prover flow, P sends V commitments to the claimed values,
/// so V needs the prover versions of the commitments to check the claims on the
/// Fiat-Shamir challenges.
pub struct FiatShamirChallengeProof<C: PrimeOrderCurve> {
    pub layer: FiatShamirChallenge<C::Scalar>,
    pub claims: Vec<HyraxClaim<C::Scalar, CommittedScalar<C>>>,
}

/// Match up the claims from the verifier with the claims from the prover. Used
/// for proofs of evaluation on public values where the proof (in the case of
/// "public" input layers and [FiatShamirChallenge] ) consists of the prover simply
/// opening the commitments in the claims, or equivalently just handing over the
/// CommittedScalars. Panics if a verifier claim can not be matched to a prover
/// claim (but doesn't worry about prover claims that don't have a verifier
/// counterpart). Also checks that any matched claims are consistent with the
/// committer (panics if not).
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
                // verify that the committed claim is consistent with the
                // committer (necessary in order to conclude that the plain-text
                // value is the correct one)
                committed_claim.evaluation.verify(committer);
                // ok, return the claim
                committed_claim.to_raw_claim()
            } else {
                panic!("Claim has not counterpart in committed claims!");
            }
        })
        .collect()
}
