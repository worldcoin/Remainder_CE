use std::convert;
use std::{collections::HashMap, marker::PhantomData};

use crate::pedersen::{CommittedScalar, PedersenCommitter};
use crate::utils::vandermonde::VandermondeInverse;
use ark_std::{end_timer, start_timer};
use hyrax_input_layer::{
    verify_public_and_random_input_layer, CommitmentEnum, HyraxCircuitInputLayerEnum,
    HyraxInputLayer, HyraxInputLayerProof, InputProofEnum,
};
use hyrax_layer::HyraxClaim;
use hyrax_output_layer::HyraxOutputLayerProof;
use itertools::Itertools;
use rand::Rng;
use remainder::input_layer::InputLayer;
use remainder::layer::layer_enum::LayerEnum;
use remainder::layer::{CircuitLayer, Layer};
use remainder::layouter::compiling::LayouterCircuit;
use remainder::layouter::component::ComponentSet;
use remainder::layouter::nodes::node_enum::NodeEnum;
use remainder::layouter::nodes::Context;
use remainder::mle::Mle;
use remainder::prover::proof_system::DefaultProofSystem;
use remainder::prover::{GKRCircuit, GKRVerifierKey, Witness};
use remainder::{
    claims::wlx_eval::ClaimMle, input_layer::enum_input_layer::InputLayerEnum, layer::LayerId,
};

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

// /// The struct that holds all the respective proofs that the verifier needs in order
// /// to verify a HyraxGKRProof
// pub struct HyraxProof<C: PrimeOrderCurve, Fn: FnMut(&Context) -> ComponentSet<NodeEnum<C::Scalar>>>
// {
//     /// The [HyraxLayerProof] for each of the intermediate layers in this circuit.
//     layer_proofs: Vec<HyraxLayerProof<C>>,
//     /// The [HyraxInputLayerProof] for each of the input polynomial commitments using the Hyrax PCS.
//     input_layer_proofs: Vec<InputProofEnum<C>>,
//     /// A commitment to the output of the circuit, i.e. what the final value of the output layer is.
//     output_layer_proofs: Vec<HyraxOutputLayerProof<C>>,
//     _marker: PhantomData<Fn>,
// }

// /// The struct that holds all the necessary information to describe a circuit.
// pub struct HyraxCircuit<
//     C: PrimeOrderCurve,
//     Fn: FnMut(&Context) -> ComponentSet<NodeEnum<C::Scalar>>,
// > {
//     pub input_layers: Vec<HyraxCircuitInputLayerEnum<C>>,
//     pub layers: Vec<LayerEnum<C::Scalar>>,
//     pub output_layers: Vec<HyraxOutputLayer<C>>,
//     _phantom: PhantomData<Fn>,
// }

// impl<C: PrimeOrderCurve, Fn: FnMut(&Context) -> ComponentSet<NodeEnum<C::Scalar>>>
//     HyraxCircuit<C, Fn>
// {
//     /// WARNING: THIS CURRENTLY ASSUMES THAT THERE IS EXACTLY ONE [InputLayerEnum::HyraxPrecommitPlaceholderInputLayer]!!!
//     /// DUE TO THE LAYOUTER NOT ALLOWING LAYERS TO BE CONSTRUCTED WITH ANYTHING OTHER THAN THEIR DEFAULT `.new()`
//     /// CONSTRUCTORS! WE'RE JUST THROWING AN ERROR FOR NOW BUT SHOULD FIX THIS ASAP BECAUSE LIGERO IS ALSO
//     /// COMPLETELY BROKEN IN NEWMAINDER FOR THE SAME REASON
//     ///
//     /// TODO(ryancao, vishady): make the committer and other things optional
//     pub fn new_from_gkr_circuit(
//         gkr_circuit: &mut LayouterCircuit<C::Scalar, ComponentSet<NodeEnum<C::Scalar>>, Fn>,
//         committer: &PedersenCommitter<C>,
//         blinding_factors_matrix: Option<Vec<<C as PrimeOrderCurve>::Scalar>>,
//         log_num_cols: Option<usize>,
//         commitment: Option<Vec<C>>,
//         prover_transcript: &mut impl ECProverTranscript<C>,
//     ) -> (
//         Self,
//         Vec<CommitmentEnum<C>>,
//         GKRVerifierKey<C::Scalar, DefaultProofSystem>,
//     ) {
//         let gkr_circ_gen_timer = start_timer!(|| "generate gkr circuit");
//         let witness: Witness<<C as PrimeOrderCurve>::Scalar, DefaultProofSystem> =
//             gkr_circuit.synthesize();

//         let circuit_description = witness.generate_verifier_key().unwrap();

//         // TODO(ryancao, vishady): ADD CIRCUIT DESCRIPTION TO TRANSCRIPT!

//         let Witness {
//             input_layers,
//             layers,
//             output_layers,
//         } = witness;
//         end_timer!(gkr_circ_gen_timer);

//         let hyrax_circ_gen_timer = start_timer!(|| "generate hyrax circuit");
//         let hyrax_output_layers = output_layers
//             .iter()
//             .map(|output_layer| HyraxOutputLayer {
//                 underlying_mle: output_layer.get_mle().clone(),
//                 _marker: PhantomData,
//             })
//             .collect();

//         let mut hyrax_precommit_counter = 0;

//         let (hyrax_input_layers, commitments) = input_layers
//             .into_iter()
//             .map(|input_layer| match input_layer {
//                 InputLayerEnum::LigeroInputLayer(_) => None,
//                 InputLayerEnum::PublicInputLayer(mut public_input_layer) => {
//                     let public_commit = public_input_layer.commit().unwrap();
//                     prover_transcript
//                         .append_scalar_points("public input layer commit", &public_commit);
//                     Some((
//                         HyraxCircuitInputLayerEnum::PublicInputLayer(*public_input_layer),
//                         CommitmentEnum::PublicCommitment(public_commit),
//                     ))
//                 }
//                 InputLayerEnum::RandomInputLayer(mut random_input_layer) => {
//                     let random_commit = random_input_layer.commit().unwrap();
//                     prover_transcript
//                         .append_scalar_points("random input layer commit", &random_commit);
//                     Some((
//                         HyraxCircuitInputLayerEnum::RandomInputLayer(*random_input_layer),
//                         CommitmentEnum::RandomCommitment(random_commit),
//                     ))
//                 }
//                 InputLayerEnum::HyraxPlaceholderInputLayer(hyrax_placeholder_input_layer) => {
//                     let mut hyrax_input_layer =
//                         HyraxInputLayer::new_from_placeholder_with_committer(
//                             *hyrax_placeholder_input_layer,
//                             committer,
//                         );
//                     let hyrax_commit = hyrax_input_layer.commit();
//                     prover_transcript.append_ec_points("hyrax pcs commitment", &hyrax_commit);
//                     Some((
//                         HyraxCircuitInputLayerEnum::HyraxInputLayer(hyrax_input_layer),
//                         CommitmentEnum::HyraxCommitment(hyrax_commit),
//                     ))
//                 }
//                 InputLayerEnum::HyraxPrecommitPlaceholderInputLayer(
//                     hyrax_precommit_placeholder_input_layer,
//                 ) => {
//                     let precommit = commitment.clone().unwrap();
//                     hyrax_precommit_counter += 1;
//                     prover_transcript.append_ec_points("hyrax pcs commitment", &precommit);
//                     let hyrax_precommit_layer =
//                         HyraxInputLayer::new_from_placeholder_with_commitment(
//                             *hyrax_precommit_placeholder_input_layer,
//                             committer,
//                             blinding_factors_matrix.clone().unwrap(),
//                             log_num_cols.unwrap(),
//                             precommit.clone(),
//                         );

//                     Some((
//                         HyraxCircuitInputLayerEnum::HyraxInputLayer(hyrax_precommit_layer),
//                         CommitmentEnum::HyraxCommitment(precommit),
//                     ))
//                 }
//             })
//             .filter_map(|x| x)
//             .unzip();

//         if hyrax_precommit_counter > 1 {
//             panic!("ERROR: MORE THAN ONE HYRAX PRECOMMIT LAYER FOUND IN CIRCUIT! THIS IS NOT SUPPORTED AT THE MOMENT (READ THE ERROR)");
//         }

//         end_timer!(hyrax_circ_gen_timer);

//         (
//             Self {
//                 input_layers: hyrax_input_layers,
//                 layers: layers.layers,
//                 output_layers: hyrax_output_layers,
//                 _phantom: PhantomData,
//             },
//             commitments,
//             circuit_description,
//         )
//     }

//     /// Called after proving in order to set up the verification so that we can call [HyraxProof::verify].
//     fn setup_verification(
//         circuit_description: &GKRVerifierKey<C::Scalar, DefaultProofSystem>,
//         commitments: &[CommitmentEnum<C>],
//         verifier_transcript: &mut impl ECVerifierTranscript<C>,
//     ) {
//         // TODO(vishady, ryancao): add circuit description to verifier transcript as well!!

//         // First consume all input layer commitments from the transcript
//         commitments.iter().for_each(|commitment| match commitment {
//             CommitmentEnum::HyraxCommitment(hyrax_commit) => {
//                 let transcript_hyrax_commit = verifier_transcript
//                     .consume_ec_points("hyrax pcs commitment", hyrax_commit.len())
//                     .unwrap();
//                 assert_eq!(&transcript_hyrax_commit, hyrax_commit);
//             }
//             CommitmentEnum::PublicCommitment(public_commit) => {
//                 let transcript_public_commit = verifier_transcript
//                     .consume_scalar_points("public commitment", public_commit.len())
//                     .unwrap();
//                 assert_eq!(&transcript_public_commit, public_commit);
//             }
//             CommitmentEnum::RandomCommitment(random_commit) => {
//                 let transcript_random_commit = verifier_transcript
//                     .consume_scalar_points("random commitment", random_commit.len())
//                     .unwrap();
//                 assert_eq!(&transcript_random_commit, random_commit);
//             }
//         });
//     }

//     /// Proves a circuit using the Hyrax IP given a GKR circuit!
//     pub fn prove_gkr_circuit(
//         gkr_circuit: &mut LayouterCircuit<C::Scalar, ComponentSet<NodeEnum<C::Scalar>>, Fn>,
//         committer: &PedersenCommitter<C>,
//         blinding_factors_matrix: Option<Vec<<C as PrimeOrderCurve>::Scalar>>,
//         log_num_cols: Option<usize>,
//         commitment: Option<Vec<C>>,
//         blinding_rng: &mut impl Rng,
//         converter: &mut VandermondeInverse<C::Scalar>,
//         prover_transcript: &mut impl ECProverTranscript<C>,
//     ) -> (
//         HyraxProof<C, Fn>,
//         Vec<CommitmentEnum<C>>,
//         GKRVerifierKey<C::Scalar, DefaultProofSystem>,
//     ) {
//         // Create the hyrax circuit from the GKR circuit
//         let (mut hyrax_circuit, input_commits, circuit_description) =
//             HyraxCircuit::new_from_gkr_circuit(
//                 gkr_circuit,
//                 committer,
//                 blinding_factors_matrix,
//                 log_num_cols,
//                 commitment,
//                 prover_transcript,
//             );

//         let prove_timer = start_timer!(|| "prove timer");
//         // Create the hyrax proof from the Hyrax circuit
//         let hyrax_proof = HyraxProof::prove(
//             &mut hyrax_circuit,
//             committer,
//             blinding_rng,
//             prover_transcript,
//             converter,
//         );
//         end_timer!(prove_timer);

//         (hyrax_proof, input_commits, circuit_description)
//     }

//     /// Verifies a circuit using the Hyrax IP given a [HyraxProof]!!
//     pub fn verify_gkr_circuit(
//         hyrax_proof: HyraxProof<C, Fn>,
//         input_commits: Vec<CommitmentEnum<C>>,
//         circuit_description: &GKRVerifierKey<C::Scalar, DefaultProofSystem>,
//         committer: &PedersenCommitter<C>,
//         verifier_transcript: &mut impl ECVerifierTranscript<C>,
//     ) {
//         // Setup verification by adding necessary commitments to transcript
//         HyraxCircuit::<C, Fn>::setup_verification(
//             &circuit_description,
//             &input_commits,
//             verifier_transcript,
//         );

//         // Verify the proof
//         HyraxProof::verify(
//             &hyrax_proof,
//             &circuit_description,
//             committer,
//             input_commits,
//             verifier_transcript,
//         );
//     }
// }

// impl<C: PrimeOrderCurve, Fn: FnMut(&Context) -> ComponentSet<NodeEnum<C::Scalar>>>
//     HyraxProof<C, Fn>
// {
//     /// TODO(vishady) riad audit comments: add in comments the ordering of the proofs every time they are in a vec

//     /// The Hyrax GKR prover for a full circuit, including output layers, intermediate layers,
//     /// and input layers.
//     /// The calling context is responsible for appending to the transcript both the circuit
//     /// description and the values and/or commitments of the input layer (which is appropriate
//     /// unless already added further upstream).
//     pub fn prove(
//         circuit: &mut HyraxCircuit<C, Fn>,
//         committer: &PedersenCommitter<C>,
//         mut blinding_rng: &mut impl Rng,
//         transcript: &mut impl ECProverTranscript<C>,
//         converter: &mut VandermondeInverse<C::Scalar>,
//     ) -> Self {
//         let HyraxCircuit {
//             input_layers,
//             layers,
//             output_layers,
//             _phantom,
//         } = circuit;

//         // HashMap to keep track of all claims made on each layer
//         let mut claim_tracker: HashMap<LayerId, Vec<HyraxClaim<C::Scalar, CommittedScalar<C>>>> =
//             HashMap::new();

//         let output_layer_proofs = output_layers
//             .into_iter()
//             .map(|output_layer| {
//                 // Create the HyraxOutputLayerProof
//                 let (output_layer_proof, committed_output_claim) = HyraxOutputLayerProof::prove(
//                     output_layer,
//                     transcript,
//                     &mut blinding_rng,
//                     committer,
//                 );
//                 // Add the output claim to the claims table
//                 let output_layer_id = output_layer.underlying_mle.layer_id();
//                 claim_tracker.insert(output_layer_id, vec![committed_output_claim]);
//                 output_layer_proof
//             })
//             .collect_vec();

//         let layer_proofs = layers
//             .into_iter()
//             .rev()
//             .map(|layer| {
//                 let claims = claim_tracker.get(&layer.layer_id()).unwrap().clone();

//                 let (layer_proof, claims_from_layer) = HyraxLayerProof::prove(
//                     layer,
//                     &claims,
//                     &committer,
//                     &mut blinding_rng,
//                     transcript,
//                     converter,
//                 );
//                 // add new claims to the claim tracking table, and add each new claim to the transcript
//                 for claim in claims_from_layer.into_iter() {
//                     if let Some(curr_claims) = claim_tracker.get_mut(&claim.to_layer_id) {
//                         curr_claims.push(claim);
//                     } else {
//                         claim_tracker.insert(claim.to_layer_id, vec![claim]);
//                     }
//                 }

//                 layer_proof
//             })
//             .collect_vec();

//         // Input layer proofs
//         let input_layer_proofs = input_layers
//             .into_iter()
//             .map(|input_layer| {
//                 let layer_id = input_layer.layer_id();
//                 let committed_claims = claim_tracker.get(&layer_id).unwrap();
//                 match input_layer {
//                     HyraxCircuitInputLayerEnum::HyraxInputLayer(hyrax_input_layer) => {
//                         let hyrax_commitment = hyrax_input_layer.comm.as_ref().unwrap();
//                         let input_proof = HyraxInputLayerProof::prove(
//                             &hyrax_input_layer,
//                             hyrax_commitment,
//                             &committed_claims,
//                             &committer,
//                             &mut blinding_rng,
//                             transcript,
//                             converter,
//                         );
//                         InputProofEnum::HyraxInputLayerProof(input_proof)
//                     }
//                     // For the other input layers, the prover just hands over the (CommittedScalar-valued) HyraxClaim for each claim.
//                     // The verifier will need to check that each of the claims is consistent with the input layer.
//                     HyraxCircuitInputLayerEnum::PublicInputLayer(layer) => {
//                         InputProofEnum::PublicInputLayerProof(
//                             layer.clone(),
//                             committed_claims.clone(),
//                         )
//                     }
//                     HyraxCircuitInputLayerEnum::RandomInputLayer(layer) => {
//                         InputProofEnum::RandomInputLayerProof(
//                             layer.clone(),
//                             committed_claims.clone(),
//                         )
//                     }
//                 }
//             })
//             .collect_vec();

//         Self {
//             layer_proofs,
//             input_layer_proofs,
//             output_layer_proofs,
//             _marker: PhantomData,
//         }
//     }

//     /// This is the verification of a GKR proof. It essentially calls the verify functions of the underlying proofs
//     /// The calling context is responsible for appending to the transcript both the circuit
//     /// description and the values and/or commitments of the input layer (which is appropriate
//     /// unless already added further upstream).
//     pub fn verify(
//         proof: &HyraxProof<C, Fn>,
//         circuit_description: &GKRVerifierKey<C::Scalar, DefaultProofSystem>,
//         committer: &PedersenCommitter<C>,
//         commitments: Vec<CommitmentEnum<C>>,
//         transcript: &mut impl ECVerifierTranscript<C>,
//     ) {
//         // Unpack the Hyrax proof.
//         let HyraxProof {
//             layer_proofs,
//             input_layer_proofs,
//             output_layer_proofs,
//             _marker,
//         } = proof;

//         // Keep track of all claim commitments for the hyrax layer verifier
//         let mut claim_tracker: HashMap<LayerId, Vec<HyraxClaim<C::Scalar, C>>> = HashMap::new();

//         // Output layer verification
//         output_layer_proofs
//             .iter()
//             .zip(circuit_description.output_layers.iter())
//             .for_each(|(output_layer_proof, output_layer_desc)| {
//                 let output_layer_claim = HyraxOutputLayerProof::verify(
//                     &output_layer_proof,
//                     output_layer_desc,
//                     transcript,
//                 );

//                 // Add the output claim to the claims table
//                 claim_tracker.insert(output_layer_claim.to_layer_id, vec![output_layer_claim]);
//             });

//         // Intermediate layer verification
//         (layer_proofs
//             .into_iter()
//             .zip(circuit_description.intermediate_layers.iter().rev()))
//         .for_each(|(layer_proof, layer_desc)| {
//             // Get the unaggregated claims for this layer
//             // V checked that these claims had the expected form before adding them to the claim tracking table
//             let layer_claims_vec = claim_tracker
//                 .remove(&layer_desc.layer_id())
//                 .unwrap()
//                 .clone();
//             let claim_commits_for_layer = HyraxLayerProof::verify(
//                 &layer_proof,
//                 layer_desc,
//                 &layer_claims_vec,
//                 &committer,
//                 transcript,
//             );

//             for claim in claim_commits_for_layer {
//                 if let Some(curr_claims) = claim_tracker.get_mut(&claim.to_layer_id) {
//                     curr_claims.push(claim);
//                 } else {
//                     claim_tracker.insert(claim.to_layer_id, vec![claim]);
//                 }
//             }
//         });

//         // Input layers verification
//         input_layer_proofs
//             .into_iter()
//             .zip(commitments.into_iter())
//             .for_each(
//                 |(input_layer_proof, input_commit)| match input_layer_proof {
//                     InputProofEnum::HyraxInputLayerProof(hyrax_input_proof) => {
//                         // Check that the commitment given also matches with the commitment in the proof
//                         match input_commit {
//                             CommitmentEnum::HyraxCommitment(hyrax_commit) => {
//                                 assert_eq!(hyrax_input_proof.input_commitment, hyrax_commit);
//                             }
//                             _ => panic!("should have a hyrax commitment here!"),
//                         }
//                         let layer_id = hyrax_input_proof.layer_id;
//                         let layer_claims_vec = claim_tracker.remove(&layer_id).unwrap().clone();
//                         hyrax_input_proof.verify(&layer_claims_vec, &committer, transcript);
//                     }
//                     InputProofEnum::PublicInputLayerProof(layer, committed_claims) => {
//                         let public_commit_from_proof = layer.clone().commit().unwrap();
//                         // Check that the commitment given also matches with the commitment in the proof
//                         match input_commit {
//                             CommitmentEnum::PublicCommitment(public_commit) => {
//                                 assert_eq!(public_commit_from_proof, public_commit);
//                             }
//                             _ => panic!("should have a public commitment here!"),
//                         }
//                         let claims_as_commitments =
//                             claim_tracker.remove(&layer.layer_id()).unwrap().clone();
//                         let plaintext_claims = Self::match_claims(
//                             &claims_as_commitments,
//                             &committed_claims,
//                             committer,
//                         );
//                         plaintext_claims.into_iter().for_each(|claim| {
//                             verify_public_and_random_input_layer::<C>(
//                                 &public_commit_from_proof,
//                                 claim.get_claim(),
//                             );
//                         });
//                     }
//                     InputProofEnum::RandomInputLayerProof(layer, committed_claims) => {
//                         let random_commit_from_proof = layer.clone().commit().unwrap();
//                         // Check that the commitment given also matches with the commitment in the proof
//                         match input_commit {
//                             CommitmentEnum::RandomCommitment(random_commit) => {
//                                 assert_eq!(random_commit_from_proof, random_commit);
//                             }
//                             _ => panic!("should have a public commitment here!"),
//                         }
//                         let claims_as_commitments =
//                             claim_tracker.remove(&layer.layer_id()).unwrap().clone();
//                         let plaintext_claims = Self::match_claims(
//                             &claims_as_commitments,
//                             &committed_claims,
//                             committer,
//                         );
//                         plaintext_claims.into_iter().for_each(|claim| {
//                             verify_public_and_random_input_layer::<C>(
//                                 &random_commit_from_proof,
//                                 claim.get_claim(),
//                             );
//                         });
//                     }
//                 },
//             );

//         // @vishady this is new, so that we can be sure that the prover didn't e.g. leave out an input layer proof!  (I changed all the claims.get() to claims.remove() above)
//         // Check that there aren't any claims left in our claim tracking table!
//         assert_eq!(claim_tracker.len(), 0);
//     }

//     /// Match up the claims from the verifier with the claims from the prover. Used for input layer
//     /// proofs, where the proof (in the case of public and random layers) consists of the prover
//     /// simply opening the commitments in the claims, or equivalently just handing over the
//     /// CommittedScalars. Panics if a verifier claim can not be matched to a prover claim (and
//     /// doesn't worry about prover claims that don't have a verifier counterpart).
//     fn match_claims(
//         verifier_claims: &Vec<HyraxClaim<C::Scalar, C>>,
//         prover_claims: &Vec<HyraxClaim<C::Scalar, CommittedScalar<C>>>,
//         committer: &PedersenCommitter<C>,
//     ) -> Vec<ClaimMle<C::Scalar>> {
//         verifier_claims
//             .iter()
//             .map(|claim| {
//                 // find the corresponding committed claim
//                 if let Some(committed_claim) = prover_claims.iter().find(|committed_claim| {
//                     (committed_claim.point == claim.point)
//                         & (committed_claim.evaluation.commitment == claim.evaluation)
//                 }) {
//                     // verify that the committed claim is consistent with the committer
//                     // (necessary in order to conclude that the plain-text value is the correct one)
//                     committed_claim.evaluation.verify(committer);
//                     // ok, return the claim
//                     committed_claim.to_claim()
//                 } else {
//                     // TODO return an error instead of panicking
//                     panic!("Claim has not counterpart in committed claims!");
//                 }
//             })
//             .collect()
//     }
// }
