//! Identity gate id(z, x) determines whether the xth gate from the
//! i + 1th layer contributes to the zth gate in the ith layer.

use std::collections::HashSet;

use ark_std::cfg_into_iter;
use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::{
    claims::{
        wlx_eval::{get_num_wlx_evaluations, ClaimMle, YieldWLXEvals},
        Claim, ClaimError, YieldClaim,
    },
    expression::{circuit_expr::CircuitMle, verifier_expr::VerifierMle},
    layer::{gate::gate_helpers::bind_round_identity, LayerError, VerificationError},
    layouter::layouting::{CircuitLocation, CircuitMap},
    mle::{
        betavalues::BetaValues, dense::DenseMle, evals::MultilinearExtension, mle_enum::MleEnum,
        Mle, MleIndex,
    },
    sumcheck::*,
};
use remainder_shared_types::{
    transcript::{ProverTranscript, VerifierTranscript},
    Field,
};

use crate::layer::gate::gate_helpers::compute_sumcheck_message_identity;

use thiserror::Error;

use super::{
    gate::{
        gate_helpers::{compute_full_gate_identity, evaluate_mle_ref_product_no_beta_table},
        index_mle_indices_gate, GateError,
    },
    layer_enum::{LayerEnum, VerifierLayerEnum},
    product::{PostSumcheckLayer, Product},
    regular_layer::claims::CLAIM_AGGREGATION_CONSTANT_COLUMN_OPTIMIZATION,
    CircuitLayer, Layer, LayerId, VerifierLayer,
};

#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// The Circuit Description for an [IdentityGate].
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound = "F: Field")]
pub struct CircuitIdentityGateLayer<F: Field> {
    /// The layer id associated with this gate layer.
    id: LayerId,

    /// A vector of tuples representing the "nonzero" gates, especially useful
    /// in the sparse case the format is (z, x) where the gate at label z is
    /// the output of adding all values from labels x.
    wiring: Vec<(usize, usize)>,

    /// The source MLE of the expression, i.e. the mle that makes up the "x"
    /// variables.
    source_mle: CircuitMle<F>,
}

impl<F: Field> CircuitIdentityGateLayer<F> {
    /// Constructor for the [CircuitIdentityGateLayer] using the gate wiring, the source mle
    /// for the rerouting, and the layer_id.
    pub fn new(id: LayerId, wiring: Vec<(usize, usize)>, source_mle: CircuitMle<F>) -> Self {
        Self {
            id,
            wiring,
            source_mle,
        }
    }
}

/// Degree of independent variable is always quadratic!
///
/// V_i(g_1) = \sum_{x} f_1(g_1, x) (V_{i + 1}(x))
const NON_DATAPARALLEL_ROUND_ID_NUM_EVALS: usize = 3;

impl<F: Field> CircuitLayer<F> for CircuitIdentityGateLayer<F> {
    type VerifierLayer = VerifierIdentityGateLayer<F>;

    fn layer_id(&self) -> LayerId {
        self.id
    }

    /// Note that this ONLY verifies for non-dataparallel identity gate!!!
    ///
    /// TODO(vishady, ryancao): Implement dataparallel identity gate prover + verifier
    fn verify_rounds(
        &self,
        claim: Claim<F>,
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<VerifierLayerEnum<F>, VerificationError> {
        let num_sumcheck_rounds = self.sumcheck_round_indices().len();

        // --- Store challenges for later claim generation ---
        let mut challenges = vec![];

        // --- Grab the first round prover sumcheck message g_1(x) ---
        let mut sumcheck_messages: Vec<Vec<F>> = vec![];
        let first_round_sumcheck_messages = transcript_reader.consume_elements(
            "Initial sumcheck evaluations",
            NON_DATAPARALLEL_ROUND_ID_NUM_EVALS,
        )?;
        sumcheck_messages.push(first_round_sumcheck_messages.clone());

        // Check: V_i(g_1) =? g_1(0) + g_1(1)
        // TODO(ryancao): SUPER overloaded notation (in e.g. above comments); fix across the board
        if first_round_sumcheck_messages[0] + first_round_sumcheck_messages[1] != claim.get_result()
        {
            return Err(VerificationError::SumcheckStartFailed);
        }

        for _sumcheck_round_idx in 1..num_sumcheck_rounds {
            // --- Read challenge r_{i - 1} from transcript ---
            let challenge = transcript_reader
                .get_challenge("Sumcheck challenge")
                .unwrap();
            let g_i_minus_1_evals = sumcheck_messages[sumcheck_messages.len() - 1].clone();

            // --- Evaluate g_{i - 1}(r_{i - 1}) ---
            let prev_at_r = evaluate_at_a_point(&g_i_minus_1_evals, challenge).unwrap();

            // --- Read off g_i(0), g_i(1), ..., g_i(d) from transcript ---
            let curr_evals = transcript_reader
                .consume_elements("Sumcheck evaluations", NON_DATAPARALLEL_ROUND_ID_NUM_EVALS)
                .unwrap();

            // --- Check: g_i(0) + g_i(1) =? g_{i - 1}(r_{i - 1}) ---
            if prev_at_r != curr_evals[0] + curr_evals[1] {
                return Err(VerificationError::SumcheckFailed);
            };

            // --- Add the prover message to the sumcheck messages ---
            sumcheck_messages.push(curr_evals);

            // --- Store all challenges from transcript ---
            challenges.push(challenge);
        }

        // final round of sumcheck
        let final_chal = transcript_reader
            .get_challenge("Final Sumcheck challenge")
            .unwrap();
        challenges.push(final_chal);

        let verifier_id_gate_layer = self
            .convert_into_verifier_layer(&challenges, claim.get_point(), transcript_reader)
            .unwrap();
        let final_result = verifier_id_gate_layer.evaluate(&claim);

        // Finally, compute g_n(r_n).
        let g_n_evals = sumcheck_messages[sumcheck_messages.len() - 1].clone();
        let prev_at_r = evaluate_at_a_point(&g_n_evals, final_chal).unwrap();

        // error if this doesn't match the last round of sumcheck
        if final_result != prev_at_r {
            return Err(VerificationError::FinalSumcheckFailed);
        }

        Ok(VerifierLayerEnum::IdentityGate(verifier_id_gate_layer))
    }

    fn sumcheck_round_indices(&self) -> Vec<usize> {
        // --- SAME ISSUE HERE AS IN GATE.RS: JUST ASSUMING THAT TOTAL NUMBER
        // --- OF SUMCHECK ROUNDS IS GOING TO BE THE NUMBER OF NON-FIXED BITS
        let num_sumcheck_rounds = self
            .source_mle
            .mle_indices()
            .iter()
            .fold(0_usize, |acc, idx| {
                acc + match idx {
                    MleIndex::Fixed(_) => 0,
                    _ => 1,
                }
            });
        (0..num_sumcheck_rounds).collect_vec()
    }

    fn convert_into_verifier_layer(
        &self,
        sumcheck_challenges: &[F],
        _claim_point: &[F],
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<Self::VerifierLayer, VerificationError> {
        let src_verifier_mle = self
            .source_mle
            .into_verifier_mle(sumcheck_challenges, transcript_reader)
            .unwrap();

        // --- Create the resulting verifier layer for claim tracking ---
        // TODO(ryancao): This is not necessary; we only need to pass back the actual claims
        let verifier_id_gate_layer = VerifierIdentityGateLayer {
            layer_id: self.layer_id(),
            wiring: self.wiring.clone(),
            source_mle: src_verifier_mle,
            first_u_challenges: sumcheck_challenges.to_vec(),
        };

        Ok(verifier_id_gate_layer)
    }

    fn get_post_sumcheck_layer(
        &self,
        round_challenges: &[F],
        claim_challenges: &[F],
    ) -> PostSumcheckLayer<F, Option<F>> {
        let beta_u = BetaValues::new_beta_equality_mle(round_challenges.to_vec());
        let beta_g = BetaValues::new_beta_equality_mle(claim_challenges.to_vec());
        let f_1_uv = self
            .wiring
            .clone()
            .into_iter()
            .fold(F::ZERO, |acc, (z_ind, x_ind)| {
                let gz = *beta_g.bookkeeping_table().get(z_ind).unwrap_or(&F::ZERO);
                let ux = *beta_u.bookkeeping_table().get(x_ind).unwrap_or(&F::ZERO);
                acc + gz * ux
            });

        PostSumcheckLayer(vec![Product::<F, Option<F>>::new(
            &[self.source_mle.clone()],
            f_1_uv,
            round_challenges,
        )])
    }

    fn max_degree(&self) -> usize {
        2
    }

    fn get_circuit_mles(&self) -> Vec<&CircuitMle<F>> {
        vec![&self.source_mle]
    }

    fn convert_into_prover_layer(&self, circuit_map: &CircuitMap<F>) -> LayerEnum<F> {
        let source_mle = self.source_mle.into_dense_mle(circuit_map);
        let id_gate_layer = IdentityGate::new(self.layer_id(), self.wiring.clone(), source_mle);
        id_gate_layer.into()
    }

    fn index_mle_indices(&mut self, start_index: usize) {
        self.source_mle.index_mle_indices(start_index);
    }

    fn compute_data_outputs(
        &self,
        mle_outputs_necessary: &HashSet<&CircuitMle<F>>,
        circuit_map: &mut CircuitMap<F>,
    ) {
        assert_eq!(mle_outputs_necessary.len(), 1);
        let mle_output_necessary = mle_outputs_necessary.iter().next().unwrap();
        let source_mle_data = circuit_map
            .get_data_from_circuit_mle(&self.source_mle)
            .unwrap();

        let max_gate_val = self
            .wiring
            .iter()
            .fold(&0, |acc, (z, _)| std::cmp::max(acc, z));

        let mut remap_table = vec![F::ZERO; (max_gate_val + 1).next_power_of_two()];

        self.wiring.iter().for_each(|(z, x)| {
            let zero = F::ZERO;
            let id_val = source_mle_data.get_evals_vector().get(*x).unwrap_or(&zero);
            remap_table[*z] = *id_val;
        });

        let output_data = MultilinearExtension::new(remap_table);
        assert_eq!(
            output_data.num_vars(),
            mle_output_necessary.mle_indices().len()
        );

        circuit_map.add_node(CircuitLocation::new(self.layer_id(), vec![]), output_data);
    }
}

impl<F: Field> VerifierIdentityGateLayer<F> {
    /// Computes the oracle query's value for a given [IdentityGateVerifierLayer].
    pub fn evaluate(&self, claim: &Claim<F>) -> F {
        // compute the sum over all the variables of the gate function
        let beta_u = BetaValues::new_beta_equality_mle(self.first_u_challenges.clone());
        let beta_g = BetaValues::new_beta_equality_mle(claim.get_point().clone());

        let f_1_uv = self
            .wiring
            .clone()
            .into_iter()
            .fold(F::ZERO, |acc, (z_ind, x_ind)| {
                let gz = *beta_g.bookkeeping_table().get(z_ind).unwrap_or(&F::ZERO);
                let ux = *beta_u.bookkeeping_table().get(x_ind).unwrap_or(&F::ZERO);

                acc + gz * ux
            });

        // get the fully evaluated "expression"

        f_1_uv * self.source_mle.value()
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound = "F: Field")]
/// The layer representing a fully bound [IdentityGate].
pub struct VerifierIdentityGateLayer<F: Field> {
    /// The layer id associated with this gate layer.
    layer_id: LayerId,

    /// A vector of tuples representing the "nonzero" gates, especially useful
    /// in the sparse case the format is (z, x) where the gate at label z is
    /// the output of adding all values from labels x.
    wiring: Vec<(usize, usize)>,

    /// The source MLE of the expression, i.e. the mle that makes up the "x"
    /// variables.
    source_mle: VerifierMle<F>,

    /// The challenges for `x`, as derived from sumcheck.
    first_u_challenges: Vec<F>,
}

impl<F: Field> VerifierLayer<F> for VerifierIdentityGateLayer<F> {
    fn layer_id(&self) -> LayerId {
        self.layer_id
    }
}

/// implement the layer trait for identitygate struct
impl<F: Field> Layer<F> for IdentityGate<F> {
    fn prove_rounds(
        &mut self,
        claim: Claim<F>,
        transcript_writer: &mut impl ProverTranscript<F>,
    ) -> Result<(), LayerError> {
        // initialization, get the first sumcheck message
        let first_message = self
            .init_phase_1(claim)
            .expect("could not evaluate original lhs and rhs");

        let phase_1_mle_refs = self
            .phase_1_mles
            .as_mut()
            .ok_or(GateError::Phase1InitError)
            .unwrap();

        let mut challenges: Vec<F> = vec![];
        transcript_writer.append_elements("Initial Sumcheck evaluations", &first_message);
        let num_rounds = self.mle_ref.num_iterated_vars();

        // sumcheck rounds (binding x)
        let _sumcheck_rounds: Vec<Vec<F>> = std::iter::once(Ok(first_message))
            .chain((1..num_rounds).map(|round| {
                let challenge = transcript_writer.get_challenge("Sumcheck challenge");
                challenges.push(challenge);
                // if there are copy bits, we want to start at that index
                bind_round_identity(round, challenge, phase_1_mle_refs);
                let phase_1_mle_references: Vec<&DenseMle<F>> = phase_1_mle_refs.iter().collect();
                let eval =
                    compute_sumcheck_message_identity(round, &phase_1_mle_references).unwrap();
                transcript_writer.append_elements("Sumcheck evaluations", &eval);
                Ok::<_, LayerError>(eval)
            }))
            .try_collect()?;

        // final challenge after binding x (left side of the sum)
        let final_chal = transcript_writer.get_challenge("Final Sumcheck challenge for binding x");
        challenges.push(final_chal);

        phase_1_mle_refs.iter_mut().for_each(|mle| {
            mle.fix_variable(num_rounds - 1, final_chal);
        });

        // --- Finally, send the claimed values for each of the bound MLE to the verifier ---
        // First, send the claimed value of V_{i + 1}(u)
        let source_mle_reduced = self.phase_1_mles.clone().unwrap()[1].clone();
        debug_assert!(source_mle_reduced.bookkeeping_table().len() == 1);
        transcript_writer.append(
            "Evaluation of V_{i + 1}(g_2, u)",
            source_mle_reduced.bookkeeping_table()[0],
        );
        Ok(())
    }

    fn layer_id(&self) -> LayerId {
        self.layer_id
    }

    fn initialize_sumcheck(&mut self, claim_point: &[F]) -> Result<(), LayerError> {
        let beta_g = BetaValues::new_beta_equality_mle(claim_point.to_vec());
        self.set_beta_g(beta_g);

        self.mle_ref.index_mle_indices(0);
        let num_vars = self.mle_ref.num_iterated_vars();

        let mut a_hg_mle_ref = vec![F::ZERO; 1 << num_vars];

        self.nonzero_gates
            .clone()
            .into_iter()
            .for_each(|(z_ind, x_ind)| {
                let beta_g_at_z = *self
                    .beta_g
                    .as_ref()
                    .unwrap()
                    .bookkeeping_table()
                    .get(z_ind)
                    .unwrap_or(&F::ZERO);
                a_hg_mle_ref[x_ind] += beta_g_at_z;
            });

        let mut phase_1 = [
            DenseMle::new_from_raw(a_hg_mle_ref, LayerId::Input(0)),
            self.mle_ref.clone(),
        ];

        index_mle_indices_gate(&mut phase_1, 0);
        self.set_phase_1(phase_1.clone());

        Ok(())
    }

    fn compute_round_sumcheck_message(&self, round_index: usize) -> Result<Vec<F>, LayerError> {
        let mles: Vec<&DenseMle<F>> = self.phase_1_mles.as_ref().unwrap().iter().collect();
        let independent_variable = mles
            .iter()
            .map(|mle_ref| {
                mle_ref
                    .mle_indices()
                    .contains(&MleIndex::IndexedBit(round_index))
            })
            .reduce(|acc, item| acc | item)
            .unwrap();
        let evals = evaluate_mle_ref_product_no_beta_table(&mles, independent_variable, mles.len())
            .unwrap();
        let SumcheckEvals(evaluations) = evals;
        Ok(evaluations)
    }

    fn bind_round_variable(&mut self, round_index: usize, challenge: F) -> Result<(), LayerError> {
        let mles = self.phase_1_mles.as_mut().unwrap();
        mles.iter_mut().for_each(|mle_ref| {
            mle_ref.fix_variable(round_index, challenge);
        });
        Ok(())
    }

    fn sumcheck_round_indices(&self) -> Vec<usize> {
        (0..self.mle_ref.num_iterated_vars()).collect_vec()
    }

    fn max_degree(&self) -> usize {
        2
    }

    fn get_post_sumcheck_layer(
        &self,
        round_challenges: &[F],
        _claim_challenges: &[F],
    ) -> PostSumcheckLayer<F, F> {
        let [_, mle_ref] = self.phase_1_mles.as_ref().unwrap();
        let beta_u = BetaValues::new_beta_equality_mle(round_challenges.to_vec());

        let f_1_uv = self
            .nonzero_gates
            .clone()
            .into_iter()
            .fold(F::ZERO, |acc, (z_ind, x_ind)| {
                let gz = *self
                    .beta_g
                    .as_ref()
                    .unwrap()
                    .bookkeeping_table()
                    .get(z_ind)
                    .unwrap_or(&F::ZERO);
                let ux = *beta_u.bookkeeping_table().get(x_ind).unwrap_or(&F::ZERO);

                acc + gz * ux
            });

        PostSumcheckLayer(vec![Product::<F, F>::new(&[mle_ref.clone()], f_1_uv)])
    }
}

impl<F: Field> YieldClaim<ClaimMle<F>> for IdentityGate<F> {
    /// Get the claims that this layer makes on other layers
    fn get_claims(&self) -> Result<Vec<ClaimMle<F>>, LayerError> {
        let mut claims = vec![];
        let mut fixed_mle_indices_u: Vec<F> = vec![];

        // check the left side of the sum (f2(u)) against the challenges made to bind that variable
        if let Some([_, mle_ref]) = &self.phase_1_mles {
            for index in mle_ref.mle_indices() {
                fixed_mle_indices_u.push(
                    index
                        .val()
                        .ok_or(LayerError::ClaimError(ClaimError::ClaimMleIndexError))?,
                );
            }
            let val = mle_ref.bookkeeping_table()[0];
            let claim: ClaimMle<F> = ClaimMle::new(
                fixed_mle_indices_u,
                val,
                Some(self.layer_id()),
                Some(mle_ref.get_layer_id()),
                Some(MleEnum::Dense(mle_ref.clone())),
            );
            claims.push(claim);
        } else {
            return Err(LayerError::LayerNotReady);
        }

        Ok(claims)
    }
}

impl<F: Field> YieldWLXEvals<F> for IdentityGate<F> {
    fn get_wlx_evaluations(
        &self,
        claim_vecs: &[Vec<F>],
        claimed_vals: &[F],
        _claimed_mles: Vec<MleEnum<F>>,
        num_claims: usize,
        num_idx: usize,
    ) -> Result<Vec<F>, ClaimError> {
        // get the number of evaluations
        let num_evals = if CLAIM_AGGREGATION_CONSTANT_COLUMN_OPTIMIZATION {
            let (num_evals, _, _) = get_num_wlx_evaluations(claim_vecs);
            num_evals
        } else {
            ((num_claims - 1) * num_idx) + 1
        };

        // we already have the first #claims evaluations, get the next num_evals - #claims evaluations
        let next_evals: Vec<F> = (num_claims..num_evals)
            .map(|idx| {
                // get the challenge l(idx)
                let new_chal: Vec<F> = cfg_into_iter!(0..num_idx)
                    .map(|claim_idx| {
                        let evals: Vec<F> = cfg_into_iter!(&claim_vecs)
                            .map(|claim| claim[claim_idx])
                            .collect();

                        evaluate_at_a_point(&evals, F::from(idx as u64)).unwrap()
                    })
                    .collect();

                compute_full_gate_identity(new_chal, &mut self.mle_ref.clone(), &self.nonzero_gates)
            })
            .collect();

        // concat this with the first k evaluations from the claims to get num_evals evaluations
        let mut claimed_vals = claimed_vals.to_vec();

        claimed_vals.extend(&next_evals);
        let wlx_evals = claimed_vals;
        Ok(wlx_evals)
    }
}

impl<F: Field> YieldClaim<ClaimMle<F>> for VerifierIdentityGateLayer<F> {
    fn get_claims(&self) -> Result<Vec<ClaimMle<F>>, LayerError> {
        // Grab the claim on the left side.
        // TODO!(ryancao): Do error handling here!
        let source_vars = self.source_mle.mle_indices();
        let source_point = source_vars
            .iter()
            .map(|idx| match idx {
                MleIndex::Bound(chal, _bit_idx) => *chal,
                MleIndex::Fixed(val) => {
                    if *val {
                        F::ONE
                    } else {
                        F::ZERO
                    }
                }
                _ => panic!("Error: Not fully bound"),
            })
            .collect_vec();
        let source_val = self.source_mle.value();

        // WARNING: DO NOT TRUST THIS MLE! IT IS INCORRECT
        let dummy_source_mle = DenseMle::new_from_raw(vec![source_val], self.layer_id());

        let source_claim: ClaimMle<F> = ClaimMle::new(
            source_point,
            source_val,
            Some(self.layer_id()),
            Some(self.source_mle.layer_id()),
            Some(MleEnum::Dense(dummy_source_mle)),
        );

        Ok(vec![source_claim])
    }
}

/// identity gate struct
/// wiring is used to be able to select specific elements from an MLE
/// (equivalent to an add gate with a zero mle ref)
#[derive(Error, Debug, Serialize, Deserialize, Clone)]
#[serde(bound = "F: Field")]
pub struct IdentityGate<F: Field> {
    /// layer id that this gate is found
    pub layer_id: LayerId,
    /// we only need a single incoming gate and a single outgoing gate so this is a
    /// tuple of 2 integers representing which label maps to which
    pub nonzero_gates: Vec<(usize, usize)>,
    /// the mle ref in question from which we are selecting specific indices
    pub mle_ref: DenseMle<F>,
    /// the beta table which enumerates the incoming claim's challenge points
    beta_g: Option<DenseMle<F>>,
    /// the mles that are created from the initial phase, where we automatically
    /// filter through the nonzero gates using the libra trick
    pub phase_1_mles: Option<[DenseMle<F>; 2]>,
}

impl<F: Field> IdentityGate<F> {
    /// new addgate mle (wrapper constructor)
    pub fn new(
        layer_id: LayerId,
        nonzero_gates: Vec<(usize, usize)>,
        mle_ref: DenseMle<F>,
    ) -> IdentityGate<F> {
        IdentityGate {
            layer_id,
            nonzero_gates,
            mle_ref,
            beta_g: None,
            phase_1_mles: None,
        }
    }

    fn set_beta_g(&mut self, beta_g: DenseMle<F>) {
        self.beta_g = Some(beta_g);
    }

    /// bookkeeping tables necessary for binding x
    fn set_phase_1(&mut self, mle_refs: [DenseMle<F>; 2]) {
        self.phase_1_mles = Some(mle_refs);
    }

    /// initialize necessary bookkeeping tables by traversing the nonzero gates
    pub fn init_phase_1(&mut self, claim: Claim<F>) -> Result<Vec<F>, GateError> {
        let beta_g = BetaValues::new_beta_equality_mle(claim.get_point().clone());
        self.set_beta_g(beta_g);

        self.mle_ref.index_mle_indices(0);
        let num_vars = self.mle_ref.num_iterated_vars();

        let mut a_hg_mle_ref = vec![F::ZERO; 1 << num_vars];

        self.nonzero_gates
            .clone()
            .into_iter()
            .for_each(|(z_ind, x_ind)| {
                let beta_g_at_z = *self
                    .beta_g
                    .as_ref()
                    .unwrap()
                    .bookkeeping_table()
                    .get(z_ind)
                    .unwrap_or(&F::ZERO);
                a_hg_mle_ref[x_ind] += beta_g_at_z;
            });

        let mut phase_1 = [
            DenseMle::new_from_raw(a_hg_mle_ref, LayerId::Input(0)),
            self.mle_ref.clone(),
        ];

        index_mle_indices_gate(&mut phase_1, 0);
        self.set_phase_1(phase_1.clone());

        let independent_variable = phase_1
            .iter()
            .map(|mle_ref| mle_ref.mle_indices().contains(&MleIndex::IndexedBit(0)))
            .reduce(|acc, item| acc | item)
            .ok_or(GateError::EmptyMleList)?;
        let phase_1_mle_references: Vec<&DenseMle<F>> = phase_1.iter().collect();
        let evals = evaluate_mle_ref_product_no_beta_table(
            &phase_1_mle_references,
            independent_variable,
            phase_1.len(),
        )
        .unwrap();

        let SumcheckEvals(evaluations) = evals;
        Ok(evaluations)
    }
}

/// For circuit serialization to hash the circuit description into the transcript.
impl<F: std::fmt::Debug + Field> IdentityGate<F> {
    pub(crate) fn circuit_description_fmt(&self) -> impl std::fmt::Display + '_ {
        // --- Dummy struct which simply exists to implement `std::fmt::Display` ---
        // --- so that it can be returned as an `impl std::fmt::Display` ---
        struct IdentityGateCircuitDesc<'a, F: std::fmt::Debug + Field>(&'a IdentityGate<F>);

        impl<'a, F: std::fmt::Debug + Field> std::fmt::Display for IdentityGateCircuitDesc<'a, F> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct("IdentityGate")
                    .field("mle_ref_layer_id", &self.0.mle_ref.get_layer_id())
                    .field("mle_ref_mle_indices", &self.0.mle_ref.mle_indices())
                    .field("identity_nonzero_gates", &self.0.nonzero_gates)
                    .finish()
            }
        }
        IdentityGateCircuitDesc(self)
    }
}
