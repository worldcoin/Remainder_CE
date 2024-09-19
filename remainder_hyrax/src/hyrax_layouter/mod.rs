use std::{
    collections::{HashMap, HashSet},
    marker::PhantomData,
};

use itertools::Itertools;
use rand::Rng;
use remainder::{
    expression::circuit_expr::{filter_bookkeeping_table, CircuitMle},
    input_layer::{
        enum_input_layer::{CircuitInputLayerEnum, InputLayerEnumVerifierCommitment},
        CircuitInputLayer, InputLayer,
    },
    layer::{
        layer_enum::{CircuitLayerEnum, LayerEnum},
        CircuitLayer, LayerId,
    },
    layouter::{
        compiling::LayouterCircuit,
        component::ComponentSet,
        layouting::{CircuitLocation, CircuitMap, InputLayerHintMap, InputNodeMap},
        nodes::{
            circuit_inputs::{compile_inputs::combine_input_mles, InputLayerData},
            node_enum::NodeEnum,
            Context, NodeId,
        },
    },
    mle::evals::MultilinearExtension,
    prover::GKRCircuitDescription,
};
use remainder_shared_types::{
    curves::PrimeOrderCurve,
    transcript::ec_transcript::{ECProverTranscript, ECVerifierTranscript},
};

use crate::{
    hyrax_gkr::{
        hyrax_input_layer::{CommitmentEnum, HyraxCircuitInputLayerEnum},
        hyrax_output_layer::HyraxOutputLayer,
        HyraxCircuit, HyraxProof,
    },
    pedersen::PedersenCommitter,
    utils::vandermonde::VandermondeInverse,
};

impl<
        C: PrimeOrderCurve,
        Fn: FnMut(
            &Context,
        ) -> (
            ComponentSet<NodeEnum<C::Scalar>>,
            Vec<InputLayerData<C::Scalar>>,
        ),
    > HyraxCircuit<C, Fn>
{
    pub fn new_from_gkr_circuit(
        gkr_circuit: &mut LayouterCircuit<C::Scalar, ComponentSet<NodeEnum<C::Scalar>>, Fn>,
        committer: &PedersenCommitter<C>,
        blinding_factors_matrix: Option<Vec<<C as PrimeOrderCurve>::Scalar>>,
        log_num_cols: Option<usize>,
        commitment: Option<Vec<C>>,
        transcript_writer: &mut impl ECProverTranscript<C>,
    ) -> (
        Self,
        Vec<CommitmentEnum<C>>,
        GKRCircuitDescription<C::Scalar>,
    ) {
        let ctx = Context::new();
        let (component, input_layer_data) = (gkr_circuit.witness_builder)(&ctx);
        let (gkr_circuit_description, input_node_map, input_hint_map) = gkr_circuit
            .generate_circuit_description(component, ctx)
            .unwrap();

        let (hyrax_circuit, input_commits) = Self::populate_hyrax_circuit(
            &gkr_circuit_description,
            input_node_map,
            input_hint_map,
            input_layer_data,
            transcript_writer,
        );
        (hyrax_circuit, input_commits, gkr_circuit_description)
    }
    pub fn populate_hyrax_circuit(
        gkr_circuit_description: &GKRCircuitDescription<C::Scalar>,
        input_layer_to_node_map: InputNodeMap,
        input_layer_hint_map: InputLayerHintMap<C::Scalar>,
        data_input_layers: Vec<InputLayerData<C::Scalar>>,
        transcript_writer: &mut impl ECProverTranscript<C>,
    ) -> (Self, Vec<CommitmentEnum<C>>) {
        let GKRCircuitDescription {
            input_layers: input_layer_descriptions,
            intermediate_layers: intermediate_layer_descriptions,
            output_layers: output_layer_descriptions,
        } = gkr_circuit_description;

        // Forward pass through input layer data to map input layer ID to the data that the circuit builder provides.
        let mut input_id_data_map = HashMap::<NodeId, &InputLayerData<C::Scalar>>::new();
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
                        if mle_claim_map.get(&layer_id).is_none() {
                            mle_claim_map.insert(layer_id, HashSet::from([circuit_mle]));
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
            if mle_claim_map.get(&layer_id).is_none() {
                mle_claim_map.insert(layer_id, HashSet::from([&output_layer.mle]));
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
        let mut prover_input_layers: Vec<HyraxCircuitInputLayerEnum<C>> = Vec::new();
        let mut input_commitments: Vec<CommitmentEnum<C>> = Vec::new();
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

                    let mut prover_input_layer = input_layer_description
                        .into_prover_input_layer(combined_mle, &corresponding_input_data.precommit);
                    let commitment = prover_input_layer.commit().unwrap();
                    match commitment {
                        InputLayerEnumVerifierCommitment::LigeroInputLayer(_ligero_commit) => {
                            panic!("The verifier challenge layer should not be populated yet, and does not need to be committed to.");

                        }
                        InputLayerEnumVerifierCommitment::PublicInputLayer(
                            public_input_coefficients,
                        ) => {
                            transcript_writer.append_scalar_points(
                                "Input Coefficients for Public Input",
                                &public_input_coefficients,
                            );
                            input_commitments.push(CommitmentEnum::PublicCommitment(public_input_coefficients));
                        }
                        InputLayerEnumVerifierCommitment::RandomInputLayer(_verifier_challenge) => {
                            panic!("The verifier challenge layer should not be populated yet, and does not need to be committed to.");
                        }
                    }
                    prover_input_layers.push(HyraxCircuitInputLayerEnum::from_input_layer_enum(prover_input_layer));
                } else if let CircuitInputLayerEnum::RandomInputLayer(
                    verifier_challenge_input_layer_description,
                ) = input_layer_description
                {
                    let verifier_challenge_mle =
                        MultilinearExtension::new(transcript_writer.get_scalar_field_challenges(
                            "Verifier challenges for fiat shamir",
                            1 << verifier_challenge_input_layer_description.num_bits,
                        ));
                    input_commitments.push(CommitmentEnum::RandomCommitment(verifier_challenge_mle.get_evals_vector().clone()));
                    circuit_map.add_node(
                        CircuitLocation::new(
                            verifier_challenge_input_layer_description.layer_id(),
                            vec![],
                        ),
                        verifier_challenge_mle.clone(),
                    );
                    let verifier_challenge_layer = input_layer_description
                        .into_prover_input_layer(verifier_challenge_mle, &None);
                    prover_input_layers.push(HyraxCircuitInputLayerEnum::from_input_layer_enum(verifier_challenge_layer));
                } else {
                    hint_input_layers.push(input_layer_description);
                    assert!(input_layer_hint_map.0.contains_key(&input_layer_id));
                }
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
                        // also here @ryan do we actually need to add to circuit map? also there are several places (see: logup) where we never add to circuit
                        // map, even before this refactor, actually... this means we won't make claims on these so that's actually fine? idk
                        circuit_map.add_node(
                            CircuitLocation::new(hint_input_layer_description.layer_id(), vec![]),
                            function_applied_to_data.clone(),
                        );
                        let mut prover_input_layer = hint_input_layer_description
                            .into_prover_input_layer(function_applied_to_data, &None);
                        // LOL wait my brain can't think right now TODO (@ryan) do we really need to do this commitment part?
                        let prover_input_commit = prover_input_layer.commit().unwrap();
                        match prover_input_commit {
                            InputLayerEnumVerifierCommitment::LigeroInputLayer(_ligero_commit) => {
                                panic!("Hyrax does not handle ligero commitments")
                            }
                            InputLayerEnumVerifierCommitment::PublicInputLayer(
                                public_input_coefficients,
                            ) => {
                                transcript_writer.append_scalar_points(
                                    "Input Coefficients for Public Input",
                                    &public_input_coefficients,
                                );
                                input_commitments.push(CommitmentEnum::PublicCommitment(public_input_coefficients));
                            }
                            InputLayerEnumVerifierCommitment::RandomInputLayer(_verifier_challenge) => {
                                panic!("The verifier challenge layer should not be populated yet, and does not need to be committed to.");
                            }
                        }
                        prover_input_layers.push(HyraxCircuitInputLayerEnum::from_input_layer_enum(prover_input_layer));
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
                    intermediate_layer_description.into_prover_layer(&circuit_map);
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
        let hyrax_circuit = Self {
            input_layers: prover_input_layers,
            layers: prover_intermediate_layers,
            output_layers: prover_output_layers,
            _marker: PhantomData,
        };

        (hyrax_circuit, input_commitments)
    }

    /// Called after proving in order to set up the verification so that we can call [HyraxProof::verify].
    fn setup_verification(
        circuit_description: &GKRCircuitDescription<C::Scalar>,
        commitments: &[CommitmentEnum<C>],
        verifier_transcript: &mut impl ECVerifierTranscript<C>,
    ) {
        // TODO(vishady, ryancao): add circuit description to verifier transcript as well!!

        // First consume all input layer commitments from the transcript
        commitments.iter().for_each(|commitment| match commitment {
            CommitmentEnum::HyraxCommitment(hyrax_commit) => {
                let transcript_hyrax_commit = verifier_transcript
                    .consume_ec_points("hyrax pcs commitment", hyrax_commit.len())
                    .unwrap();
                assert_eq!(&transcript_hyrax_commit, hyrax_commit);
            }
            CommitmentEnum::PublicCommitment(public_commit) => {
                let transcript_public_commit = verifier_transcript
                    .consume_scalar_points("public commitment", public_commit.len())
                    .unwrap();
                assert_eq!(&transcript_public_commit, public_commit);
                dbg!("HELLOOOOO");
            }
            CommitmentEnum::RandomCommitment(random_commit) => {
                let transcript_random_commit = verifier_transcript
                    .get_scalar_field_challenges("random commitment", random_commit.len())
                    .unwrap();
                assert_eq!(&transcript_random_commit, random_commit);
            }
        });
    }

    /// Proves a circuit using the Hyrax IP given a GKR circuit!
    pub fn prove_gkr_circuit(
        gkr_circuit: &mut LayouterCircuit<C::Scalar, ComponentSet<NodeEnum<C::Scalar>>, Fn>,
        committer: &PedersenCommitter<C>,
        blinding_factors_matrix: Option<Vec<<C as PrimeOrderCurve>::Scalar>>,
        log_num_cols: Option<usize>,
        commitment: Option<Vec<C>>,
        blinding_rng: &mut impl Rng,
        converter: &mut VandermondeInverse<C::Scalar>,
        prover_transcript: &mut impl ECProverTranscript<C>,
    ) -> (
        HyraxProof<C, Fn>,
        Vec<CommitmentEnum<C>>,
        GKRCircuitDescription<C::Scalar>,
    ) {
        // Create the hyrax circuit from the GKR circuit
        let (mut hyrax_circuit, input_commits, circuit_description) =
            HyraxCircuit::new_from_gkr_circuit(
                gkr_circuit,
                committer,
                blinding_factors_matrix,
                log_num_cols,
                commitment,
                prover_transcript,
            );

        // Create the hyrax proof from the Hyrax circuit
        let hyrax_proof = HyraxProof::prove(
            &mut hyrax_circuit,
            committer,
            blinding_rng,
            prover_transcript,
            converter,
        );

        (hyrax_proof, input_commits, circuit_description)
    }

    /// Verifies a circuit using the Hyrax IP given a [HyraxProof]!!
    pub fn verify_gkr_circuit(
        hyrax_proof: HyraxProof<C, Fn>,
        input_commits: Vec<CommitmentEnum<C>>,
        circuit_description: &mut GKRCircuitDescription<C::Scalar>,
        committer: &PedersenCommitter<C>,
        verifier_transcript: &mut impl ECVerifierTranscript<C>,
    ) {
        // Setup verification by adding necessary commitments to transcript
        HyraxCircuit::<C, Fn>::setup_verification(
            &circuit_description,
            &input_commits,
            verifier_transcript,
        );

        circuit_description.index_mle_indices(0);

        // Verify the proof
        HyraxProof::verify(
            &hyrax_proof,
            &circuit_description,
            committer,
            input_commits,
            verifier_transcript,
        );
    }
}
