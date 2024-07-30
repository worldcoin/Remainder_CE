//! Identity gate id(z, x) determines whether the xth gate from the
//! i + 1th layer contributes to the zth gate in the ith layer.

use std::marker::PhantomData;

use ark_std::cfg_into_iter;
use itertools::Itertools;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::{
    claims::{
        wlx_eval::{get_num_wlx_evaluations, ClaimMle, YieldWLXEvals},
        Claim, ClaimError, YieldClaim,
    },
    expression::{circuit_expr::CircuitMle, verifier_expr::VerifierMle},
    layer::{LayerError, VerificationError},
    mle::{betavalues::BetaValues, dense::DenseMle, mle_enum::MleEnum, Mle, MleIndex},
    prover::SumcheckProof,
    sumcheck::*,
};
use remainder_shared_types::{
    transcript::{ProverTranscript, VerifierTranscript},
    FieldExt,
};

use crate::layer::gate::gate_helpers::prove_round_identity;

use thiserror::Error;

use super::{
    gate::{
        check_fully_bound,
        gate_helpers::{compute_full_gate_identity, evaluate_mle_ref_product_no_beta_table},
        index_mle_indices_gate, GateError,
    },
    product::{PostSumcheckLayer, Product},
    CircuitLayer, Layer, LayerId, PostSumcheckEvaluation, SumcheckLayer, VerifierLayer,
};

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound = "F: FieldExt")]
pub struct IdentityGateCircuitLayer<F: FieldExt> {
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

/// Degree of independent variable is always quadratic!
///
/// V_i(g_2, g_1) = \sum_{p_2, x} \beta(g_2, p_2) f_1(g_1, x) (V_{i + 1}(p_2, x))
const DATAPARALLEL_ROUND_ID_NUM_EVALS: usize = 3;
const NON_DATAPARALLEL_ROUND_ID_NUM_EVALS: usize = 3;

impl<F: FieldExt> CircuitLayer<F> for IdentityGateCircuitLayer<F> {
    type VerifierLayer = IdentityGateVerifierLayer<F>;

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
    ) -> Result<Self::VerifierLayer, VerificationError> {
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

        // --- Store challenges for later claim generation ---
        let mut challenges = vec![];

        // --- Grab the first round prover sumcheck message g_1(x) ---
        let sumcheck_messages: Vec<Vec<F>> = vec![];
        let first_round_sumcheck_messages = transcript_reader.consume_elements(
            "Initial sumcheck evaluations",
            NON_DATAPARALLEL_ROUND_ID_NUM_EVALS,
        )?;
        sumcheck_messages.push(first_round_sumcheck_messages.clone());

        // Check: V_i(g_1) =? g_1(0) + g_1(1)
        // TODO(ryancao): SUPER overloaded notation (in e.g. above comments); fix across the board
        if first_round_sumcheck_messages[0] + first_round_sumcheck_messages[1] != claim.get_result()
        {
            return Err(LayerError::VerificationError(
                VerificationError::SumcheckStartFailed,
            ));
        }

        for sumcheck_round_idx in 1..num_sumcheck_rounds {
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

        let src_verifier_mle = self
            .source_mle
            .into_verifier_mle(&challenges, transcript_reader)
            .unwrap();

        // --- Create the resulting verifier layer for claim tracking ---
        // TODO(ryancao): This is not necessary; we only need to pass back the actual claims
        let verifier_id_gate_layer = IdentityGateVerifierLayer {
            layer_id: self.layer_id(),
            wiring: self.wiring.clone(),
            source_mle: self.source_mle,
            claim_challenge_points: claim.get_point().clone(),
            first_u_challenges: challenges,
        };
        let final_result = verifier_id_gate_layer.evaluate(&claim);

        // Finally, compute g_n(r_n).
        let g_n_evals = sumcheck_messages[sumcheck_messages.len() - 1].clone();
        let prev_at_r = evaluate_at_a_point(&g_n_evals, final_chal).unwrap();

        // error if this doesn't match the last round of sumcheck
        if final_result != prev_at_r {
            return Err(LayerError::VerificationError(
                VerificationError::FinalSumcheckFailed,
            ));
        }

        Ok(())
    }
}

impl<F: FieldExt> IdentityGateVerifierLayer<F> {
    /// Computes the oracle query's value for a given [IdentityGateVerifierLayer].
    pub fn evaluate(&self, claim: &Claim<F>) -> F {
        // compute the sum over all the variables of the gate function
        let beta_u = BetaValues::new_beta_equality_mle(self.claim_challenge_points);
        let beta_g = BetaValues::new_beta_equality_mle(claim.get_point().clone());

        let f_1_uv = self
            .wiring
            .into_iter()
            .fold(F::ZERO, |acc, (z_ind, x_ind)| {
                let gz = *beta_g.bookkeeping_table().get(z_ind).unwrap_or(&F::ZERO);
                let ux = *beta_u.bookkeeping_table().get(x_ind).unwrap_or(&F::ZERO);

                acc + gz * ux
            });

        // get the fully evaluated "expression"
        let fully_evaluated = f_1_uv * self.source_mle.value();
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound = "F: FieldExt")]
pub struct IdentityGateVerifierLayer<F: FieldExt> {
    /// The layer id associated with this gate layer.
    layer_id: LayerId,

    /// A vector of tuples representing the "nonzero" gates, especially useful
    /// in the sparse case the format is (z, x) where the gate at label z is
    /// the output of adding all values from labels x.
    wiring: Vec<(usize, usize)>,

    /// The source MLE of the expression, i.e. the mle that makes up the "x"
    /// variables.
    source_mle: VerifierMle<F>,

    /// The challenge points for the claim on the [IdentityGate] layer.
    claim_challenge_points: Vec<F>,

    /// The challenges for `x`, as derived from sumcheck.
    first_u_challenges: Vec<F>,
}

impl<F: FieldExt> VerifierLayer<F> for IdentityGateVerifierLayer<F> {
    fn layer_id(&self) -> LayerId {
        todo!()
    }
}

/// implement the layer trait for identitygate struct
impl<F: FieldExt> Layer<F> for IdentityGate<F> {
    type CircuitLayer = IdentityGateCircuitLayer<F>;

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
        let sumcheck_rounds: Vec<Vec<F>> = std::iter::once(Ok(first_message))
            .chain((1..num_rounds).map(|round| {
                let challenge = transcript_writer.get_challenge("Sumcheck challenge");
                challenges.push(challenge);
                // if there are copy bits, we want to start at that index
                let eval = prove_round_identity(round, challenge, phase_1_mle_refs).unwrap();
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

        Ok(sumcheck_rounds.into())
    }

    fn into_circuit_layer(&self) -> Result<Self::CircuitLayer, LayerError> {
        todo!()
    }

    fn layer_id(&self) -> LayerId {
        self.layer_id
    }
}

impl<F: FieldExt> YieldClaim<ClaimMle<F>> for IdentityGate<F> {
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
                Some(self.id().clone()),
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

impl<F: FieldExt> YieldWLXEvals<F> for IdentityGate<F> {
    fn get_wlx_evaluations(
        &self,
        claim_vecs: &[Vec<F>],
        claimed_vals: &[F],
        _claimed_mles: Vec<MleEnum<F>>,
        num_claims: usize,
        num_idx: usize,
    ) -> Result<Vec<F>, ClaimError> {
        // get the number of evaluations
        let (num_evals, _) = get_num_wlx_evaluations(claim_vecs);

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
/// identity gate struct
/// wiring is used to be able to select specific elements from an MLE
/// (equivalent to an add gate with a zero mle ref)
#[derive(Error, Debug, Serialize, Deserialize, Clone)]
#[serde(bound = "F: FieldExt")]
pub struct IdentityGate<F: FieldExt> {
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

impl<F: FieldExt> IdentityGate<F> {
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
        self.set_beta_g(beta_g.clone());

        self.mle_ref.index_mle_indices(0);
        let num_vars = self.mle_ref.num_iterated_vars();

        let mut a_hg_mle_ref = vec![F::ZERO; 1 << num_vars];

        self.nonzero_gates
            .clone()
            .into_iter()
            .for_each(|(z_ind, x_ind)| {
                let beta_g_at_z = *beta_g.bookkeeping_table().get(z_ind).unwrap_or(&F::ZERO);
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
        let evals =
            evaluate_mle_ref_product_no_beta_table(&phase_1, independent_variable, phase_1.len())
                .unwrap();

        let Evals(evaluations) = evals;
        Ok(evaluations)
    }
}

impl<F: FieldExt> PostSumcheckEvaluation<F> for IdentityGate<F> {
    fn get_post_sumcheck_layer(
        &self,
        round_challenges: &[F],
        claim_challenges: &[F],
    ) -> PostSumcheckLayer<F, F> {
        let [_, mle_ref] = self.phase_1_mles.as_ref().unwrap();
        let beta_u = BetaValues::new_beta_equality_mle(round_challenges.to_vec());
        let beta_g = BetaValues::new_beta_equality_mle(claim_challenges.to_vec());

        let f_1_uv = self
            .nonzero_gates
            .clone()
            .into_iter()
            .fold(F::ZERO, |acc, (z_ind, x_ind)| {
                let gz = *beta_g.bookkeeping_table().get(z_ind).unwrap_or(&F::ZERO);
                let ux = *beta_u.bookkeeping_table().get(x_ind).unwrap_or(&F::ZERO);

                acc + gz * ux
            });

        PostSumcheckLayer(vec![Product::<F, F>::new(&vec![mle_ref.clone()], f_1_uv)])
    }
}

/// For circuit serialization to hash the circuit description into the transcript.
impl<F: std::fmt::Debug + FieldExt> IdentityGate<F> {
    pub(crate) fn circuit_description_fmt<'a>(&'a self) -> impl std::fmt::Display + 'a {
        // --- Dummy struct which simply exists to implement `std::fmt::Display` ---
        // --- so that it can be returned as an `impl std::fmt::Display` ---
        struct IdentityGateCircuitDesc<'a, F: std::fmt::Debug + FieldExt>(&'a IdentityGate<F>);

        impl<'a, F: std::fmt::Debug + FieldExt> std::fmt::Display for IdentityGateCircuitDesc<'a, F> {
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

impl<F: FieldExt> SumcheckLayer<F> for IdentityGate<F> {
    fn initialize_sumcheck(&mut self, claim_point: &[F]) -> Result<(), LayerError> {
        let beta_g = BetaValues::new_beta_equality_mle(claim_point.to_vec());
        self.set_beta_g(beta_g.clone());

        self.mle_ref.index_mle_indices(0);
        let num_vars = self.mle_ref.num_iterated_vars();

        let mut a_hg_mle_ref = vec![F::ZERO; 1 << num_vars];

        self.nonzero_gates
            .clone()
            .into_iter()
            .for_each(|(z_ind, x_ind)| {
                let beta_g_at_z = *beta_g.bookkeeping_table().get(z_ind).unwrap_or(&F::ZERO);
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

    fn compute_round_sumcheck_message(&mut self, round_index: usize) -> Result<Vec<F>, LayerError> {
        let mles = self.phase_1_mles.as_mut().unwrap();
        let independent_variable = mles
            .iter()
            .map(|mle_ref| {
                mle_ref
                    .mle_indices()
                    .contains(&MleIndex::IndexedBit(round_index))
            })
            .reduce(|acc, item| acc | item)
            .unwrap();
        let evals =
            evaluate_mle_ref_product_no_beta_table(mles, independent_variable, mles.len()).unwrap();
        let Evals(evaluations) = evals;
        Ok(evaluations)
    }

    fn bind_round_variable(&mut self, round_index: usize, challenge: F) -> Result<(), LayerError> {
        let mles = self.phase_1_mles.as_mut().unwrap();
        mles.iter_mut().for_each(|mle_ref| {
            mle_ref.fix_variable(round_index, challenge);
        });
        Ok(())
    }

    fn num_sumcheck_rounds(&self) -> usize {
        self.mle_ref.num_iterated_vars()
    }

    fn max_degree(&self) -> usize {
        2
    }
}
