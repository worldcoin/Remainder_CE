use std::collections::hash_map::Entry;
use std::collections::HashSet;
use std::{collections::HashMap, marker::PhantomData};

use crate::pedersen::{CommittedScalar, PedersenCommitter};
use crate::utils::vandermonde::VandermondeInverse;
use ark_std::{end_timer, start_timer};
use hyrax_circuit_inputs::HyraxInputLayerData;
use hyrax_input_layer::{
    verify_public_and_random_input_layer, HyraxInputLayer, HyraxInputLayerEnum,
    HyraxInputLayerProof, HyraxProverCommitmentEnum, HyraxVerifierCommitmentEnum, InputProofEnum
};
use hyrax_layer::HyraxClaim;
use hyrax_output_layer::HyraxOutputLayerProof;
use itertools::Itertools;
use rand::Rng;
use remainder::expression::circuit_expr::{filter_bookkeeping_table, CircuitMle};
use remainder::input_layer::enum_input_layer::{
    CircuitInputLayerEnum, InputLayerEnumVerifierCommitment,
};
use remainder::input_layer::verifier_challenge_input_layer::VerifierChallenge;
use remainder::input_layer::{CircuitInputLayer, InputLayer};
use remainder::layer::layer_enum::{CircuitLayerEnum, LayerEnum};
use remainder::layer::{CircuitLayer, Layer};
use remainder::layouter::component::ComponentSet;
use remainder::layouter::layouting::{
    CircuitLocation, CircuitMap, InputLayerHintMap, InputNodeMap,
};
use remainder::layouter::nodes::circuit_inputs::compile_inputs::combine_input_mles;
use remainder::layouter::nodes::node_enum::NodeEnum;
use remainder::layouter::nodes::{Context, NodeId};
use remainder::mle::evals::MultilinearExtension;
use remainder::mle::Mle;
use remainder::prover::{generate_circuit_description, GKRCircuitDescription};
use remainder::{claims::wlx_eval::ClaimMle, layer::LayerId};

use remainder_shared_types::{
    curves::PrimeOrderCurve,
    transcript::ec_transcript::{ECProverTranscript, ECVerifierTranscript},
};

use self::{hyrax_layer::HyraxLayerProof, hyrax_output_layer::HyraxOutputLayer};

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
/// to verify a HyraxGKRProof
pub struct HyraxProof<C: PrimeOrderCurve> {
    /// The [HyraxLayerProof] for each of the intermediate layers in this circuit.
    layer_proofs: Vec<HyraxLayerProof<C>>,
    /// The [HyraxInputLayerProof] for each of the input polynomial commitments using the Hyrax PCS.
    input_layer_proofs: Vec<InputProofEnum<C>>,
    verifier_challenge_proofs: Vec<VerifierChallengeProof<C>>,
    /// A commitment to the output of the circuit, i.e. what the final value of the output layer is.
    output_layer_proofs: Vec<HyraxOutputLayerProof<C>>,
}

pub struct VerifierChallengeProof<C: PrimeOrderCurve> {
    pub layer: VerifierChallenge<C::Scalar>,
    pub claims: Vec<HyraxClaim<C::Scalar, CommittedScalar<C>>>,
}

pub struct HyraxInstantiatedCircuit<C: PrimeOrderCurve> {
    pub input_layers: Vec<HyraxInputLayerEnum<C>>,
    /// The verifier challenges
    pub verifier_challenges: Vec<VerifierChallenge<C::Scalar>>,
    pub layers: Vec<LayerEnum<C::Scalar>>,
    pub output_layers: Vec<HyraxOutputLayer<C>>,
}

type CircuitDescriptionAndAux<F> = (GKRCircuitDescription<F>, InputNodeMap, InputLayerHintMap<F>);

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

    pub fn generate_circuit_description(
        mut witness_function: Fn,
    ) -> (
        CircuitDescriptionAndAux<C::Scalar>,
        Vec<HyraxInputLayerData<C>>,
    ) {
        let ctx = Context::new();
        let (component, input_layer_data) = (witness_function)(&ctx);
        let circuit_description_timer = start_timer!(|| "generating circuit description");
        let result = (
            generate_circuit_description(component, ctx).unwrap(),
            input_layer_data,
        );
        end_timer!(circuit_description_timer);
        result
    }

    pub fn populate_hyrax_circuit(
        &self,
        gkr_circuit_description: &GKRCircuitDescription<C::Scalar>,
        input_layer_to_node_map: InputNodeMap,
        input_layer_hint_map: InputLayerHintMap<C::Scalar>,
        data_input_layers: Vec<HyraxInputLayerData<C>>,
        transcript_writer: &mut impl ECProverTranscript<C>,
    ) -> (
        HyraxInstantiatedCircuit<C>,
        Vec<HyraxVerifierCommitmentEnum<C>>,
    ) {
        let GKRCircuitDescription {
            input_layers: input_layer_descriptions,
            verifier_challenges: verifier_challenge_descriptions,
            intermediate_layers: intermediate_layer_descriptions,
            output_layers: output_layer_descriptions,
        } = gkr_circuit_description;

        let hyrax_populate_circuit_timer =
            start_timer!(|| "generating hyrax circuit from gkr circuit description");

        // Forward pass through input layer data to map input layer ID to the data that the circuit builder provides.
        let mut input_id_data_map = HashMap::<NodeId, &HyraxInputLayerData<C>>::new();
        data_input_layers.iter().for_each(|input_layer_data| {
            input_id_data_map.insert(
                input_layer_data.corresponding_input_node_id,
                input_layer_data,
            );
        });

        // Forward pass to get the map of circuit MLEs whose data is expected to be "compiled"
        // for future layers.
        let mut mle_claim_map = HashMap::<LayerId, HashSet<&CircuitMle<C::Scalar>>>::new();
        intermediate_layer_descriptions
            .iter()
            .for_each(|intermediate_layer| {
                let layer_source_circuit_mles = intermediate_layer.get_circuit_mles();
                layer_source_circuit_mles
                    .into_iter()
                    .for_each(|circuit_mle| {
                        let layer_id = circuit_mle.layer_id();
                        if let Entry::Vacant(e) = mle_claim_map.entry(layer_id) {
                            e.insert(HashSet::from([circuit_mle]));
                        } else {
                            mle_claim_map
                                .get_mut(&layer_id)
                                .unwrap()
                                .insert(circuit_mle);
                        }
                    })
            });

        output_layer_descriptions.iter().for_each(|output_layer| {
            let layer_source_mle = &output_layer.mle;
            let layer_id = layer_source_mle.layer_id();
            if let Entry::Vacant(e) = mle_claim_map.entry(layer_id) {
                e.insert(HashSet::from([&output_layer.mle]));
            } else {
                mle_claim_map
                    .get_mut(&layer_id)
                    .unwrap()
                    .insert(&output_layer.mle);
            }
        });

        let mut circuit_map = CircuitMap::new();

        // input layers
        // go through input data, map it to the inputlayernode it corresponds to
        // for each input layer node, take the input data it corresponds to and combine it to form one big bookkeeping table,
        // we convert the circuit input layer into a prover input layer using this big bookkeeping table
        // we add the data in the input data corresopnding with the circuit location for each input data struct into the circuit map
        let mut prover_input_layers: Vec<HyraxInputLayerEnum<C>> = Vec::new();
        let mut verifier_challenges = Vec::new();
        let mut input_commitments: Vec<HyraxVerifierCommitmentEnum<C>> = Vec::new();
        let mut hint_input_layers: Vec<&CircuitInputLayerEnum<C::Scalar>> = Vec::new();
        input_layer_descriptions
            .iter()
            .for_each(|input_layer_description| {
                let input_layer_id = input_layer_description.layer_id();
                let maybe_input_node_id = input_layer_to_node_map.get_node_id(&input_layer_id);
                if let Some(input_node_id) = maybe_input_node_id {
                    assert!(input_id_data_map.contains_key(input_node_id));
                    let corresponding_input_data = *(input_id_data_map.get(input_node_id).unwrap());
                    let input_mles = corresponding_input_data
                        .data
                        .iter()
                        .map(|input_shred_data| &input_shred_data.data);


                    let combined_mle = combine_input_mles(&input_mles.collect_vec());
                    let mle_outputs_necessary = mle_claim_map.get(&input_layer_id).unwrap();
                    mle_outputs_necessary.iter().for_each(|mle_output| {
                        let prefix_bits = mle_output.prefix_bits();
                        let output = filter_bookkeeping_table(&combined_mle, &prefix_bits);
                        circuit_map
                            .add_node(CircuitLocation::new(input_layer_id, prefix_bits), output);
                    });

                    match input_layer_description {
                        CircuitInputLayerEnum::HyraxInputLayer(circuit_hyrax_input_layer) => {
                            let (hyrax_commit, hyrax_prover_input_layer) = if let Some(HyraxProverCommitmentEnum::HyraxCommitment((hyrax_precommit, hyrax_blinding_factors))) = &corresponding_input_data.precommit {
                                let dtype = &corresponding_input_data.input_data_type;
                                let hyrax_input_layer = HyraxInputLayer::new_with_hyrax_commitment(
                                    combined_mle,
                                    dtype,
                                    input_layer_id,
                                    self.committer,
                                    hyrax_blinding_factors.clone(),
                                    circuit_hyrax_input_layer.log_num_cols,
                                    hyrax_precommit.clone());
                                (hyrax_precommit.clone(), hyrax_input_layer)
                            } else if corresponding_input_data.precommit.is_none() {
                                let dtype = &corresponding_input_data.input_data_type;
                                let mut hyrax_input_layer = HyraxInputLayer::new_with_committer(combined_mle, input_layer_id, self.committer, dtype);
                                let hyrax_commit_timer = start_timer!(|| format!("committing to hyrax input layer, {:?}", hyrax_input_layer.layer_id));
                                let hyrax_commit = hyrax_input_layer.commit();
                                end_timer!(hyrax_commit_timer);
                                (hyrax_commit, hyrax_input_layer)
                            } else {
                                panic!("We should only have no precommit or a hyrax precommit for hyrax input layers!")
                            };
                            transcript_writer.append_ec_points("Hyrax commitment", &hyrax_commit);
                            input_commitments.push(HyraxVerifierCommitmentEnum::HyraxCommitment(hyrax_commit.clone()));
                            prover_input_layers.push(HyraxInputLayerEnum::HyraxInputLayer(hyrax_prover_input_layer));

                        }
                        CircuitInputLayerEnum::PublicInputLayer(circuit_public_input_layer) => {
                            assert!(corresponding_input_data.precommit.is_none(), "public input layers should not have precommits");
                            let mut prover_public_input_layer = circuit_public_input_layer.convert_into_prover_input_layer(combined_mle, &None);
                            let public_commit_timer = start_timer!(|| format!("committing to public input layer, {:?}", prover_public_input_layer.layer_id()));
                            let commitment = prover_public_input_layer.commit().unwrap();
                            end_timer!(public_commit_timer);
                            if let InputLayerEnumVerifierCommitment::PublicInputLayer(public_input_coefficients) = commitment {
                                transcript_writer.append_scalar_points(
                                    "Input Coefficients for Public Input",
                                    &public_input_coefficients,
                                );
                                input_commitments.push(HyraxVerifierCommitmentEnum::PublicCommitment(public_input_coefficients));
                            }
                            prover_input_layers.push(HyraxInputLayerEnum::from_input_layer_enum(prover_public_input_layer));
                        },
                        CircuitInputLayerEnum::LigeroInputLayer(_circuit_ligero_input_layer) => {
                            panic!("Hyrax proof system does not support ligero input layers because the PCS implementation is not zero knowledge")
                        },
                    }
                } else {
                    hint_input_layers.push(input_layer_description);
                    assert!(input_layer_hint_map.0.contains_key(&input_layer_id));
                }
            });

        verifier_challenge_descriptions
            .iter()
            .for_each(|verifier_challenge_description| {
                let verifier_challenge_mle = MultilinearExtension::new(transcript_writer.get_scalar_field_challenges(
                    "Verifier challenges",
                    1 << verifier_challenge_description.num_bits,
                ));
                circuit_map.add_node(
                    CircuitLocation::new(verifier_challenge_description.layer_id(), vec![]),
                    verifier_challenge_mle.clone(),
                );
                verifier_challenges.push(VerifierChallenge::new(
                    verifier_challenge_mle,
                    verifier_challenge_description.layer_id(),
                ));
            });

        // forward pass of the layers
        // convert the circuit layer into a prover layer using circuit map -> populate a GKRCircuit as you do this
        // prover layer ( mle_claim_map ) -> populates circuit map
        let mut uninstantiated_intermediate_layers: Vec<&CircuitLayerEnum<C::Scalar>> = Vec::new();
        intermediate_layer_descriptions
            .iter()
            .for_each(|intermediate_layer_description| {
                let mle_outputs_necessary = mle_claim_map
                    .get(&intermediate_layer_description.layer_id())
                    .unwrap();
                let populatable = intermediate_layer_description
                    .compute_data_outputs(mle_outputs_necessary, &mut circuit_map);
                if !populatable {
                    uninstantiated_intermediate_layers.push(intermediate_layer_description);
                }
            });

        while !hint_input_layers.is_empty() || !uninstantiated_intermediate_layers.is_empty() {
            let mut data_updated = false;
            hint_input_layers = hint_input_layers
                .iter()
                .filter_map(|hint_input_layer_description| {
                    let (hint_circuit_location, hint_function) = input_layer_hint_map
                        .get_hint_function(&hint_input_layer_description.layer_id());
                    if let Some(data) = circuit_map.get_data_from_location(hint_circuit_location) {
                        let function_applied_to_data = hint_function(data);
                        circuit_map.add_node(
                            CircuitLocation::new(hint_input_layer_description.layer_id(), vec![]),
                            function_applied_to_data.clone(),
                        );
                        match hint_input_layer_description {
                            CircuitInputLayerEnum::HyraxInputLayer(_circuit_hyrax_input_layer) => {
                                let mut hyrax_input_layer = HyraxInputLayer::new_with_committer(function_applied_to_data, hint_input_layer_description.layer_id(), self.committer, &None);
                                let hyrax_commit = hyrax_input_layer.commit();
                                transcript_writer.append_ec_points("Hyrax commitment", &hyrax_commit);
                                input_commitments.push(HyraxVerifierCommitmentEnum::HyraxCommitment(hyrax_commit));
                                prover_input_layers.push(HyraxInputLayerEnum::HyraxInputLayer(hyrax_input_layer));
                            }
                            CircuitInputLayerEnum::PublicInputLayer(circuit_public_input_layer) => {
                                let mut prover_public_input_layer = circuit_public_input_layer.convert_into_prover_input_layer(function_applied_to_data, &None);
                                let commitment = prover_public_input_layer.commit().unwrap();
                                if let InputLayerEnumVerifierCommitment::PublicInputLayer(public_input_coefficients) = commitment {
                                    transcript_writer.append_scalar_points(
                                        "Input Coefficients for Public Input",
                                        &public_input_coefficients,
                                    );
                                    input_commitments.push(HyraxVerifierCommitmentEnum::PublicCommitment(public_input_coefficients));
                                }
                                prover_input_layers.push(HyraxInputLayerEnum::from_input_layer_enum(prover_public_input_layer));
                            },
                            CircuitInputLayerEnum::LigeroInputLayer(_circuit_ligero_input_layer) => {
                                panic!("Hyrax proof system does not support ligero input layers because the PCS implementation is not zero knowledge")
                            },
                        }
                        data_updated = true;
                        None
                    } else {
                        Some(*hint_input_layer_description)
                    }
                })
                .collect_vec();
            uninstantiated_intermediate_layers = uninstantiated_intermediate_layers
                .iter()
                .filter_map(|uninstantiated_intermediate_layer| {
                    let mle_outputs_necessary = mle_claim_map
                        .get(&uninstantiated_intermediate_layer.layer_id())
                        .unwrap();
                    let populatable = uninstantiated_intermediate_layer
                        .compute_data_outputs(mle_outputs_necessary, &mut circuit_map);
                    if populatable {
                        data_updated = true;
                        None
                    } else {
                        Some(*uninstantiated_intermediate_layer)
                    }
                })
                .collect();
            assert!(data_updated);
        }
        // assert_eq!(circuit_description_map.0.len(), circuit_map.0.len());

        let mut prover_intermediate_layers: Vec<LayerEnum<C::Scalar>> =
            Vec::with_capacity(intermediate_layer_descriptions.len());
        intermediate_layer_descriptions
            .iter()
            .for_each(|intermediate_layer_description| {
                let prover_intermediate_layer =
                    intermediate_layer_description.convert_into_prover_layer(&circuit_map);
                prover_intermediate_layers.push(prover_intermediate_layer)
            });

        let mut prover_output_layers: Vec<HyraxOutputLayer<C>> = Vec::new();
        output_layer_descriptions
            .iter()
            .for_each(|output_layer_description| {
                let prover_output_layer =
                    output_layer_description.into_prover_output_layer(&circuit_map);
                let hyrax_output_layer = HyraxOutputLayer {
                    underlying_mle: prover_output_layer.get_mle().clone(),
                    _marker: PhantomData,
                };
                prover_output_layers.push(hyrax_output_layer)
            });
        let hyrax_circuit = HyraxInstantiatedCircuit {
            input_layers: prover_input_layers,
            verifier_challenges: verifier_challenges,
            layers: prover_intermediate_layers,
            output_layers: prover_output_layers,
        };
        end_timer!(hyrax_populate_circuit_timer);

        (hyrax_circuit, input_commitments)
    }

    pub fn prove_gkr_circuit(
        &mut self,
        witness_function: Fn,
        transcript_writer: &mut impl ECProverTranscript<C>,
    ) -> (
        Vec<HyraxVerifierCommitmentEnum<C>>,
        GKRCircuitDescription<C::Scalar>,
        HyraxProof<C>,
    ) {
        let ((circuit_description, input_layer_to_node_map, input_hint_map), input_data) =
            Self::generate_circuit_description(witness_function);
        let (mut instantiated_circuit, commitments) = self.populate_hyrax_circuit(
            &circuit_description,
            input_layer_to_node_map,
            input_hint_map,
            input_data,
            transcript_writer,
        );
        let prove_timer = start_timer!(|| "prove hyrax circuit");
        let proof = self.prove(&mut instantiated_circuit, transcript_writer);
        end_timer!(prove_timer);
        (commitments, circuit_description, proof)
    }

    #[allow(clippy::type_complexity)]
    /// TODO(vishady) riad audit comments: add in comments the ordering of the proofs every time they are in a vec

    /// The Hyrax GKR prover for a full circuit, including output layers, intermediate layers,
    /// and input layers.
    /// The calling context is responsible for appending to the transcript both the circuit
    /// description and the values and/or commitments of the input layer (which is appropriate
    /// unless already added further upstream).
    pub fn prove(
        &mut self,
        instantiated_circuit: &mut HyraxInstantiatedCircuit<C>,
        transcript: &mut impl ECProverTranscript<C>,
    ) -> HyraxProof<C> {
        let committer = self.committer;
        let mut blinding_rng = &mut self.blinding_rng;
        let converter = &mut self.converter;
        let HyraxInstantiatedCircuit {
            input_layers,
            verifier_challenges: _verifier_challenges,
            layers,
            output_layers,
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
                let output_layer_id = output_layer.underlying_mle.layer_id();
                claim_tracker.insert(output_layer_id, vec![committed_output_claim]);
                output_layer_proof
            })
            .collect_vec();

        let layer_proofs = layers
            .iter_mut()
            .rev()
            .map(|layer| {
                let claims = claim_tracker.get(&layer.layer_id()).unwrap().clone();

                let (layer_proof, claims_from_layer) = HyraxLayerProof::prove(
                    layer,
                    &claims,
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

        // Input layer proofs
        let input_layer_proofs = input_layers
            .iter_mut()
            .map(|input_layer| {
                let layer_id = input_layer.layer_id();
                let committed_claims = claim_tracker.get(&layer_id).unwrap();
                match input_layer {
                    HyraxInputLayerEnum::HyraxInputLayer(hyrax_input_layer) => {
                        let hyrax_commitment = hyrax_input_layer.comm.as_ref().unwrap();
                        let input_proof = HyraxInputLayerProof::prove(
                            hyrax_input_layer,
                            hyrax_commitment,
                            committed_claims,
                            committer,
                            &mut blinding_rng,
                            transcript,
                            converter,
                        );
                        InputProofEnum::HyraxInputLayerProof(input_proof)
                    }
                    // For the other input layers, the prover just hands over the (CommittedScalar-valued) HyraxClaim for each claim.
                    // The verifier will need to check that each of the claims is consistent with the input layer.
                    HyraxInputLayerEnum::PublicInputLayer(layer) => {
                        InputProofEnum::PublicInputLayerProof(
                            layer.clone(),
                            committed_claims.clone(),
                        )
                    }
                }
            })
            .collect_vec();

        // --------- Verifier Challenges ---------
        // There is nothing to be done here, since the claims on verifier challenges are checked
        // directly by the verifier, without aggregation.

        HyraxProof {
            layer_proofs,
            input_layer_proofs,
            output_layer_proofs,
        }
    }

    pub fn verify_gkr_circuit(
        &self,
        proof: &HyraxProof<C>,
        circuit_description: &mut GKRCircuitDescription<C::Scalar>,
        commitments: &[HyraxVerifierCommitmentEnum<C>],
        verifier_transcript: &mut impl ECVerifierTranscript<C>,
    ) {
        // TODO(vishady, ryancao): add circuit description to verifier transcript as well!!

        // First consume all input layer commitments from the transcript
        commitments.iter().for_each(|commitment| match commitment {
            HyraxVerifierCommitmentEnum::HyraxCommitment(hyrax_commit) => {
                let transcript_hyrax_commit = verifier_transcript
                    .consume_ec_points("hyrax pcs commitment", hyrax_commit.len())
                    .unwrap();
                assert_eq!(&transcript_hyrax_commit, hyrax_commit);
            }
            HyraxVerifierCommitmentEnum::PublicCommitment(public_commit) => {
                let transcript_public_commit = verifier_transcript
                    .consume_scalar_points("public commitment", public_commit.len())
                    .unwrap();
                assert_eq!(&transcript_public_commit, public_commit);
            }
            HyraxVerifierCommitmentEnum::RandomCommitment(random_commit) => {
                let transcript_random_commit = verifier_transcript
                    .get_scalar_field_challenges("random commitment", random_commit.len())
                    .unwrap();
                assert_eq!(&transcript_random_commit, random_commit);
            }
        });

        circuit_description.index_mle_indices(0);

        let verify_timer = start_timer!(|| "verify hyrax circuit");
        Self::verify(
            proof,
            circuit_description,
            self.committer,
            commitments,
            verifier_transcript,
        );
        end_timer!(verify_timer);
    }
    /// This is the verification of a GKR proof. It essentially calls the verify functions of the underlying proofs
    /// The calling context is responsible for appending to the transcript both the circuit
    /// description and the values and/or commitments of the input layer (which is appropriate
    /// unless already added further upstream).
    pub fn verify(
        proof: &HyraxProof<C>,
        circuit_description: &GKRCircuitDescription<C::Scalar>,
        committer: &PedersenCommitter<C>,
        commitments: &[HyraxVerifierCommitmentEnum<C>],
        transcript: &mut impl ECVerifierTranscript<C>,
    ) {
        // Unpack the Hyrax proof.
        let HyraxProof {
            layer_proofs,
            input_layer_proofs,
            verifier_challenge_proofs,
            output_layer_proofs,
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

        // Input layers verification
        input_layer_proofs
            .iter()
            .zip(commitments)
            .for_each(
                |(input_layer_proof, input_commit)| match input_layer_proof {
                    InputProofEnum::HyraxInputLayerProof(hyrax_input_proof) => {
                        // Check that the commitment given also matches with the commitment in the proof
                        match input_commit {
                            HyraxVerifierCommitmentEnum::HyraxCommitment(hyrax_commit) => {
                                assert_eq!(&hyrax_input_proof.input_commitment, hyrax_commit);
                            }
                            _ => panic!("should have a hyrax commitment here!"),
                        }
                        let layer_id = hyrax_input_proof.layer_id;
                        let layer_claims_vec = claim_tracker.remove(&layer_id).unwrap().clone();
                        hyrax_input_proof.verify(&layer_claims_vec, committer, transcript);
                    }
                    InputProofEnum::PublicInputLayerProof(layer, committed_claims) => {
                        let public_commit_from_proof = layer.clone().commit().unwrap();
                        // Check that the commitment given also matches with the commitment in the proof
                        match input_commit {
                            HyraxVerifierCommitmentEnum::PublicCommitment(public_commit) => {
                                assert_eq!(&public_commit_from_proof, public_commit);
                            }
                            _ => panic!("should have a public commitment here!"),
                        }
                        let claims_as_commitments =
                            claim_tracker.remove(&layer.layer_id()).unwrap().clone();
                        let plaintext_claims =
                            Self::match_claims(&claims_as_commitments, committed_claims, committer);
                        plaintext_claims.into_iter().for_each(|claim| {
                            verify_public_and_random_input_layer::<C>(
                                &public_commit_from_proof,
                                claim.get_claim(),
                            );
                        });
                    }
                },
            );

        // TODO Check the claims on the verifier challenges
        // the following is incorrect, we need to look up OUR versions of the layer?
        // verifier_challenge_proofs
        //     .iter()
        //     .for_each(|vcp| {
        //         let claims_as_commitments = claim_tracker.remove(&vcp.layer.layer_id()).unwrap().clone();
        //         let plaintext_claims =
        //             Self::match_claims(&claims_as_commitments, vcp.claims, committer);
        //         plaintext_claims.into_iter().for_each(|claim| {
        //             verify_public_and_random_input_layer::<C>(
        //                 &random_commit_from_proof,
        //                 claim.get_claim(),
        //             );
        //         });
        //     }

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
}
