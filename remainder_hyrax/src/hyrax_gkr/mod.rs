use std::collections::hash_map::Entry;
use std::collections::HashSet;
use std::{collections::HashMap, marker::PhantomData};

use crate::pedersen::{CommittedScalar, PedersenCommitter};
use crate::utils::vandermonde::VandermondeInverse;
use ark_std::{end_timer, start_timer};
use hyrax_circuit_inputs::HyraxInputLayerData;
use hyrax_input_layer::{
    verify_claim, HyraxInputLayer, HyraxInputLayerEnum, HyraxInputLayerProof,
    HyraxProverCommitmentEnum
};
use hyrax_layer::HyraxClaim;
use hyrax_output_layer::HyraxOutputLayerProof;
use itertools::Itertools;
use rand::Rng;
use remainder::expression::circuit_expr::{filter_bookkeeping_table, MleDescription};
use remainder::input_layer::enum_input_layer::{
    InputLayerDescriptionEnum, InputLayerEnumVerifierCommitment,
};
use remainder::input_layer::fiat_shamir_challenge::FiatShamirChallenge;
use remainder::input_layer::{InputLayer, InputLayerDescription};
use remainder::layer::layer_enum::LayerEnum;
use remainder::layer::{Layer, LayerDescription};
use remainder::layouter::component::{Component, ComponentSet};
use remainder::layouter::layouting::{CircuitLocation, CircuitMap, InputNodeMap, LayerMap};
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
    /// The MLEs of the public inputs, along with their layer ids
    public_inputs: Vec<(LayerId, Vec<C::Scalar>)>,
    /// The [HyraxInputLayerProof] for each of the input polynomial commitments using the Hyrax PCS.
    hyrax_input_proofs: Vec<HyraxInputLayerProof<C>>,
    /// The prover's claims on public input layers and verifier challenges, in CommittedScalar form, i.e. including the blinding factors.
    claims_on_public_values: Vec<HyraxClaim<C::Scalar, CommittedScalar<C>>>,
    /// A commitment to the output of the circuit, i.e. what the final value of the output layer is.
    output_layer_proofs: Vec<HyraxOutputLayerProof<C>>,
}

pub struct FiatShamirChallengeProof<C: PrimeOrderCurve> {
    pub layer: FiatShamirChallenge<C::Scalar>,
    pub claims: Vec<HyraxClaim<C::Scalar, CommittedScalar<C>>>,
}

pub struct HyraxInstantiatedCircuit<C: PrimeOrderCurve> {
    pub input_layers: Vec<HyraxInputLayerEnum<C>>,
    /// The verifier challenges
    pub fiat_shamir_challenges: Vec<FiatShamirChallenge<C::Scalar>>,
    pub layers: Vec<LayerEnum<C::Scalar>>,
    pub output_layers: Vec<HyraxOutputLayer<C>>,
}

type CircuitDescriptionAndAux<F> = (GKRCircuitDescription<F>, InputNodeMap);

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
        let circ_desc = generate_circuit_description(component.yield_nodes()).unwrap();
        let result = ((circ_desc.0, circ_desc.1), input_layer_data);
        end_timer!(circuit_description_timer);
        result
    }

    pub fn populate_hyrax_circuit(
        &self,
        gkr_circuit_description: &GKRCircuitDescription<C::Scalar>,
        data_input_layers: Vec<HyraxInputLayerData<C>>,
        transcript_writer: &mut impl ECProverTranscript<C>,
        input_layer_to_node_map: InputNodeMap,
    ) -> (
        HyraxInstantiatedCircuit<C>,
        LayerMap<C::Scalar>,
    ) {
        let GKRCircuitDescription {
            input_layers: input_layer_descriptions,
            fiat_shamir_challenges: fiat_shamir_challenge_descriptions,
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
        let mut mle_claim_map = HashMap::<LayerId, HashSet<&MleDescription<C::Scalar>>>::new();
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
        let mut fiat_shamir_challenges = Vec::new();
        input_layer_descriptions
            .iter()
            .for_each(|input_layer_description| {
                let input_layer_id = input_layer_description.layer_id();
                let input_node_id = input_layer_to_node_map.get_node_id(input_layer_id).unwrap();
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
                    InputLayerDescriptionEnum::HyraxInputLayer(circuit_hyrax_input_layer) => {
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
                        prover_input_layers.push(HyraxInputLayerEnum::HyraxInputLayer(hyrax_prover_input_layer));

                    }
                    InputLayerDescriptionEnum::PublicInputLayer(circuit_public_input_layer) => {
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
                        }
                        prover_input_layers.push(HyraxInputLayerEnum::from_input_layer_enum(prover_public_input_layer));
                    },
                    InputLayerDescriptionEnum::LigeroInputLayer(_circuit_ligero_input_layer) => {
                        panic!("Hyrax proof system does not support ligero input layers because the PCS implementation is not zero knowledge")
                    },
                }
            });

        fiat_shamir_challenge_descriptions
            .iter()
            .for_each(|fiat_shamir_challenge_description| {
                let fiat_shamir_challenge_mle =
                    MultilinearExtension::new(transcript_writer.get_scalar_field_challenges(
                        "Verifier challenges",
                        1 << fiat_shamir_challenge_description.num_bits,
                    ));
                circuit_map.add_node(
                    CircuitLocation::new(fiat_shamir_challenge_description.layer_id(), vec![]),
                    fiat_shamir_challenge_mle.clone(),
                );
                fiat_shamir_challenges.push(FiatShamirChallenge::new(
                    fiat_shamir_challenge_mle,
                    fiat_shamir_challenge_description.layer_id(),
                ));
            });

        // forward pass of the layers
        // convert the circuit layer into a prover layer using circuit map -> populate a GKRCircuit as you do this
        // prover layer ( mle_claim_map ) -> populates circuit map
        intermediate_layer_descriptions
            .iter()
            .for_each(|intermediate_layer_description| {
                let mle_outputs_necessary = mle_claim_map
                    .get(&intermediate_layer_description.layer_id())
                    .unwrap();
                intermediate_layer_description
                    .compute_data_outputs(mle_outputs_necessary, &mut circuit_map);
            });

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
            fiat_shamir_challenges,
            layers: prover_intermediate_layers,
            output_layers: prover_output_layers,
        };
        let layer_map = circuit_map.convert_to_layer_map();
        end_timer!(hyrax_populate_circuit_timer);

        (hyrax_circuit, layer_map)
    }

    pub fn prove_gkr_circuit(
        &mut self,
        witness_function: Fn,
        transcript_writer: &mut impl ECProverTranscript<C>,
    ) -> (
        GKRCircuitDescription<C::Scalar>,
        HyraxProof<C>,
    ) {
        let ((circuit_description, input_layer_to_node_map), input_data) =
            Self::generate_circuit_description(witness_function);
        let (mut instantiated_circuit, mut layer_map) = self.populate_hyrax_circuit(
            &circuit_description,
            input_data,
            transcript_writer,
            input_layer_to_node_map,
        );
        let prove_timer = start_timer!(|| "prove hyrax circuit");
        let proof = self.prove(&mut instantiated_circuit, &mut layer_map, transcript_writer);
        end_timer!(prove_timer);
        (circuit_description, proof)
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
        layer_map: &mut LayerMap<C::Scalar>,
        transcript: &mut impl ECProverTranscript<C>,
    ) -> HyraxProof<C> {
        let committer = self.committer;
        let mut blinding_rng = &mut self.blinding_rng;
        let converter = &mut self.converter;
        let HyraxInstantiatedCircuit {
            input_layers,
            fiat_shamir_challenges,
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
        let hyrax_input_layer_proofs = input_layers
            .iter_mut()
            .filter_map(|input_layer| {
                let layer_id = input_layer.layer_id();
                let committed_claims = claim_tracker.get(&layer_id).unwrap();
                if let HyraxInputLayerEnum::HyraxInputLayer(hyrax_input_layer) = input_layer {
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
                    Some(input_proof)
                } else {
                    None
                }
            })
            .collect_vec();

        // --------- Public Input Layers & Verifier Challenges ---------
        // The the claims on both public input layers and verifier challenges are checked directly
        // by the verifier, without aggregation, so there is almost nothing to do here.  However,
        // the verifier received the prover's claims as committed form (this is just how we
        // implemented layer proof) and so we provide here the CommittedScalar forms in order for V
        // to be able to actually verify the claims. 
        let claims_on_public_values = input_layers
            .iter()
            .filter_map(|input_layer| {
                if let HyraxInputLayerEnum::PublicInputLayer(layer) = input_layer {
                    let layer_id = layer.layer_id();
                    Some(claim_tracker.get(&layer_id).unwrap())
                } else {
                    None
                }
            })
            .flatten()
            .chain(fiat_shamir_challenges.iter().flat_map(|fiat_shamir_challenge| {
                claim_tracker.get(&fiat_shamir_challenge.layer_id()).unwrap()
            })).cloned().collect_vec();

        // Collect the values of the public inputs
        let public_inputs = input_layers
            .iter()
            .filter_map(|input_layer| {
                if let HyraxInputLayerEnum::PublicInputLayer(layer) = input_layer {
                    let layer_id = layer.layer_id();
                    // FIXME(Ben) remove the clone here somehow
                    let public_values = layer.get_evaluations_as_vec().clone();
                    Some((layer_id, public_values))
                } else {
                    None
                }
            })
            .collect_vec();

        HyraxProof {
            layer_proofs,
            public_inputs,
            hyrax_input_proofs: hyrax_input_layer_proofs,
            claims_on_public_values,
            output_layer_proofs,
        }
    }

    pub fn verify_gkr_circuit(
        &self,
        proof: &HyraxProof<C>,
        circuit_description: &mut GKRCircuitDescription<C::Scalar>,
        verifier_transcript: &mut impl ECVerifierTranscript<C>,
    ) {
        // TODO(vishady, ryancao): add circuit description to verifier transcript as well!!

        // Append the commitments to the private inputs to transcript
        proof.hyrax_input_proofs.iter().for_each(|input_proof| {
            let hyrax_commit = &input_proof.input_commitment;
            let transcript_hyrax_commit = verifier_transcript
                .consume_ec_points("hyrax pcs commitment", hyrax_commit.len())
                .unwrap();
            assert_eq!(&transcript_hyrax_commit, hyrax_commit);
        });
        
        // Append the public inputs to the transcript
        proof.public_inputs.iter().for_each(|(_layer_id, public_input)| {
            let transcript_values = verifier_transcript
                .consume_scalar_points("public values", public_input.len())
                .unwrap();
            assert_eq!(&transcript_values, public_input);
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
        let hyrax_input_commitments = proof.hyrax_input_proofs
            .iter()
            .map(|input_proof| {
                (&input_proof.layer_id, &input_proof.input_commitment)
            })
            .collect_vec();

        let verify_timer = start_timer!(|| "verify hyrax circuit");
        Self::verify(
            proof,
            circuit_description,
            self.committer,
            &proof.public_inputs,
            hyrax_input_commitments,
            fiat_shamir_challenges,
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
        public_inputs: &Vec<(LayerId, Vec<C::Scalar>)>,
        hyrax_input_commitments: Vec<(&LayerId, &Vec<C>)>,
        fiat_shamir_challenges: Vec<FiatShamirChallenge<C::Scalar>>,
        transcript: &mut impl ECVerifierTranscript<C>,
    ) {
        // Unpack the Hyrax proof.
        let HyraxProof {
            layer_proofs,
            public_inputs,
            hyrax_input_proofs,
            claims_on_public_values,
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

        // Verify the hyrax input layer proofs
        hyrax_input_proofs
            .iter()
            .zip(hyrax_input_commitments.into_iter())
            .for_each(
                |(hyrax_input_proof, (layer_id, hyrax_input_commit))| {
                    // Check that the commitment given also matches with the commitment in the proof
                    assert_eq!(layer_id, &hyrax_input_proof.layer_id);
                    assert_eq!(&hyrax_input_proof.input_commitment, hyrax_input_commit);
                    let layer_id = hyrax_input_proof.layer_id;
                    let layer_claims_vec = claim_tracker.remove(&layer_id).unwrap().clone();
                    hyrax_input_proof.verify(&layer_claims_vec, committer, transcript);
                }
            );

        // Check the claims on the public input layers
        public_inputs
            .iter()
            .for_each(|(layer_id, values)| {
                let claims_as_commitments = claim_tracker.remove(&layer_id).unwrap();
                let plaintext_claims = Self::match_claims(
                    &claims_as_commitments,
                    claims_on_public_values,
                    committer,
                );
                plaintext_claims.into_iter().for_each(|claim| {
                    verify_claim::<C::Scalar>(
                        &values,
                        claim.get_claim(),
                    );
                });
            });

        // Check the claims on the verifier challenges
        fiat_shamir_challenges
            .iter()
            .for_each(|fiat_shamir_challenge| {
                let claims_as_commitments = claim_tracker
                    .remove(&fiat_shamir_challenge.layer_id())
                    .unwrap();
                Self::match_claims(&claims_as_commitments, claims_on_public_values, committer)
                    .iter()
                    .for_each(|plaintext_claim| {
                        fiat_shamir_challenge
                            .verify(plaintext_claim.get_claim())
                            .unwrap();
                    });
            });

        // Check that there aren't any claims left in our claim tracking table!
        assert_eq!(claim_tracker.len(), 0);
    }

    /// Match up the claims from the verifier with the claims from the prover. Used for proofs of
    /// evaluation on public values where the proof (in the case of [PublicInputLayer] and
    /// [FiatShamirChallenge] ) consists of the prover simply opening the commitments in the claims,
    /// or equivalently just handing over the CommittedScalars. Panics if a verifier claim can not
    /// be matched to a prover claim (but doesn't worry about prover claims that don't have a
    /// verifier counterpart).
    /// Also checks that any matched claims are consistent with the committer (panics if not).
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
