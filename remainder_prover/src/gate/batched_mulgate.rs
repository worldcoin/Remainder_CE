use std::marker::PhantomData;

use ark_std::cfg_into_iter;
use itertools::Itertools;
use remainder_shared_types::{transcript::Transcript, FieldExt};
use serde::{Deserialize, Serialize};

use crate::{
    layer::{
        claims::Claim,
        claims::{get_num_wlx_evaluations, ClaimError},
        layer_enum::LayerEnum,
        Layer, LayerBuilder, LayerError, LayerId, VerificationError,
    },
    mle::{
        beta::{compute_beta_over_two_challenges, BetaTable},
        dense::DenseMle, mle_enum::MleEnum,
    },
    prover::{SumcheckProof, ENABLE_OPTIMIZATION},
    sumcheck::evaluate_at_a_point,
};

use super::{
    gate_helpers::{
        check_fully_bound, compute_full_gate, libra_giraffe, prove_round_dataparallel_phase, GateError,
    },
    mulgate::MulGate,
};
use crate::mle::{dense::DenseMleRef, MleRef};
use rayon::{iter::IntoParallelIterator, prelude::ParallelIterator};

impl<F: FieldExt, Tr: Transcript<F>> Layer<F> for MulGateBatched<F, Tr> {
    type Transcript = Tr;

    fn prove_rounds(
        &mut self,
        claim: Claim<F>,
        transcript: &mut Self::Transcript,
    ) -> Result<SumcheckProof<F>, LayerError> {
        // initialization, first message comes from here
        let first_message = self
            .init_copy_phase(claim)
            .expect("could not evaluate original lhs and rhs");
        let beta_g1 = if !self.g1_challenges.clone().unwrap().is_empty() {
            BetaTable::new(self.g1_challenges.clone().unwrap()).unwrap()
        } else {
            BetaTable {
                layer_claim_vars: vec![],
                table: DenseMle::new_from_raw(vec![F::one()], LayerId::Input(0), None).mle_ref(),
                relevant_indices: vec![],
            }
        };
        let beta_g2 = self.beta_g2.as_mut().unwrap();
        let (lhs, rhs) = (&mut self.lhs, &mut self.rhs);


        // new bits is the number of bits representing which copy of the gate we are looking at
        transcript
            .append_field_elements("Initial Sumcheck evaluations", &first_message)
            .unwrap();
        let num_rounds_copy_phase = self.num_dataparallel_bits;

        // do the first copy bits number sumcheck rounds
        let mut sumcheck_rounds: Vec<Vec<F>> = std::iter::once(Ok(first_message))
            .chain((1..num_rounds_copy_phase).map(|round| {
                let challenge = transcript.get_challenge("Sumcheck challenge").unwrap();
                let eval = prove_round_dataparallel_phase(
                    lhs,
                    rhs,
                    &beta_g1,
                    beta_g2,
                    round,
                    challenge,
                    &self.nonzero_gates,
                    self.num_dataparallel_bits - round,
                )
                .unwrap();
                transcript
                    .append_field_elements("Sumcheck evaluations", &eval)
                    .unwrap();
                Ok::<_, LayerError>(eval)
            }))
            .try_collect()?;

        // bind the final challenge, update the final beta table
        let final_chal_copy = transcript
            .get_challenge("Final Sumcheck challenge")
            .unwrap();
        // fix the variable and everything as you would in the last round of sumcheck
        // the evaluations from this is what you return from the first round of sumcheck in the next phase!
        beta_g2
            .beta_update(num_rounds_copy_phase - 1, final_chal_copy)
            .unwrap();
        self.lhs
            .fix_variable(num_rounds_copy_phase - 1, final_chal_copy);
        self.rhs
            .fix_variable(num_rounds_copy_phase - 1, final_chal_copy);

        if beta_g2.table.bookkeeping_table.len() == 1 {
            let beta_g2 = beta_g2.table.bookkeeping_table()[0];
            let next_claims = Claim::new_raw(self.g1_challenges.clone().unwrap(), F::zero());

            // reduced gate is how we represent the rest of the protocol as a non-batched gate mle
            // this essentially takes in the two mles bound only at the copy bits
            let reduced_gate: MulGate<F, Tr> = MulGate::new(
                self.layer_id,
                self.nonzero_gates.clone(),
                self.lhs.clone(),
                self.rhs.clone(),
                self.num_dataparallel_bits,
                Some(beta_g2),
            );
            self.reduced_gate = Some(reduced_gate);
            let next_messages = self
                .reduced_gate
                .as_mut()
                .unwrap()
                .prove_rounds(next_claims, transcript)
                .unwrap();

            sumcheck_rounds.extend(next_messages.0);

            Ok(sumcheck_rounds.into())
        } else {
            Err(LayerError::LayerNotReady)
        }
    }

    /// Verifies the sumcheck protocol
    fn verify_rounds(
        &mut self,
        claim: Claim<F>,
        sumcheck_rounds: Vec<Vec<F>>,
        transcript: &mut Self::Transcript,
    ) -> Result<(), LayerError> {
        let mut prev_evals = &sumcheck_rounds[0];
        let mut challenges = vec![];
        let mut first_u_challenges = vec![];
        let mut last_v_challenges = vec![];
        let mut first_copy_challenges = vec![];
        let num_u = self.num_vars_l.unwrap();

        // first check!!!!
        let claimed_val = sumcheck_rounds[0][0] + sumcheck_rounds[0][1];
        if claimed_val != claim.get_result() {
            return Err(LayerError::VerificationError(
                VerificationError::SumcheckStartFailed,
            ));
        }

        transcript
            .append_field_elements("Initial Sumcheck evaluations", &sumcheck_rounds[0])
            .unwrap();

        // check each of the messages
        for (i, curr_evals) in sumcheck_rounds.iter().enumerate().skip(1) {
            let challenge = transcript.get_challenge("Sumcheck challenge").unwrap();
            let prev_at_r =
                evaluate_at_a_point(prev_evals, challenge).map_err(LayerError::InterpError)?;

            if prev_at_r != curr_evals[0] + curr_evals[1] {
                return Err(LayerError::VerificationError(
                    VerificationError::SumcheckFailed,
                ));
            };
            transcript
                .append_field_elements("Sumcheck evaluations", curr_evals)
                .unwrap();
            prev_evals = curr_evals;
            challenges.push(challenge);

            // we want to separate the challenges into which ones are from the copy bits, which ones
            // are for binding x, and which are for binding y (non-batched)
            if (..(self.num_dataparallel_bits + 1)).contains(&i) {
                first_copy_challenges.push(challenge);
            } else if (..(num_u + 1)).contains(&i) {
                first_u_challenges.push(challenge);
            } else {
                last_v_challenges.push(challenge);
            }
        }

        // final round of sumcheck
        let final_chal = transcript
            .get_challenge("Final Sumcheck challenge")
            .unwrap();
        challenges.push(final_chal);

        // this belongs in the last challenge bound to y
        if self.rhs.num_vars() == 0 {
            first_u_challenges.push(final_chal);
        } else {
            last_v_challenges.push(final_chal);
        }

        // we want to grab the mutated bookkeeping tables from the "reduced_gate", this is the non-batched version
        let [_, lhs_reduced] = self
            .reduced_gate
            .as_ref()
            .unwrap()
            .phase_1_mles
            .as_ref()
            .unwrap()
            .clone();
        let [_, rhs_reduced] = self
            .reduced_gate
            .as_ref()
            .unwrap()
            .phase_2_mles
            .as_ref()
            .unwrap()
            .clone();

        // since the original mles are batched, the challenges are the concat of the copy bits and the variable bound bits
        let lhs_challenges = [
            first_copy_challenges.clone().as_slice(),
            first_u_challenges.clone().as_slice(),
        ]
        .concat();
        let rhs_challenges = [
            first_copy_challenges.clone().as_slice(),
            last_v_challenges.clone().as_slice(),
        ]
        .concat();

        let g2_challenges = claim.get_point()[..self.num_dataparallel_bits].to_vec();
        let g1_challenges = claim.get_point()[self.num_dataparallel_bits..].to_vec();

        // compute the gate function bound at those variables
        let beta_u = BetaTable::new(first_u_challenges.clone()).unwrap();
        let beta_v = if !last_v_challenges.is_empty() {
            BetaTable::new(last_v_challenges.clone()).unwrap()
        } else {
            BetaTable {
                layer_claim_vars: vec![],
                table: DenseMle::new_from_raw(vec![F::one()], LayerId::Input(0), None).mle_ref(),
                relevant_indices: vec![],
            }
        };

        let beta_g = if !g1_challenges.is_empty() {
            BetaTable::new(g1_challenges).unwrap()
        } else {
            BetaTable {
                layer_claim_vars: vec![],
                table: DenseMle::new_from_raw(vec![F::one()], LayerId::Input(0), None).mle_ref(),
                relevant_indices: vec![],
            }
        };
        let f_1_uv =
            self.nonzero_gates
                .clone()
                .into_iter()
                .fold(F::zero(), |acc, (z_ind, x_ind, y_ind)| {
                    let gz = *beta_g
                        .table
                        .bookkeeping_table()
                        .get(z_ind)
                        .unwrap_or(&F::zero());
                    let ux = *beta_u
                        .table
                        .bookkeeping_table()
                        .get(x_ind)
                        .unwrap_or(&F::zero());
                    let vy = *beta_v
                        .table
                        .bookkeeping_table()
                        .get(y_ind)
                        .unwrap_or(&F::zero());
                    acc + gz * ux * vy
                });

        // check that the original mles have been bound correctly -- this is what we get from the reduced gate
        check_fully_bound(&mut [lhs_reduced.clone()], lhs_challenges).unwrap();
        check_fully_bound(&mut [rhs_reduced.clone()], rhs_challenges).unwrap();
        let f2_bound = lhs_reduced.bookkeeping_table()[0];
        let f3_bound = rhs_reduced.bookkeeping_table()[0];
        let beta_bound = compute_beta_over_two_challenges(&g2_challenges, &first_copy_challenges);

        // compute the final result of the bound expression
        let final_result = beta_bound * (f_1_uv * (f2_bound * f3_bound));

        let prev_at_r = evaluate_at_a_point(prev_evals, final_chal).unwrap();

        // final check in sumcheck
        if final_result != prev_at_r {
            return Err(LayerError::VerificationError(
                VerificationError::FinalSumcheckFailed,
            ));
        }

        Ok(())
    }

    ///Get the claims that this layer makes on other layers
    fn get_claims(&self) -> Result<Vec<Claim<F>>, LayerError> {
        // we are going to grab the claims from the reduced gate -- this is where the mles are finally mutated
        let [_, lhs_reduced] = self
            .reduced_gate
            .as_ref()
            .unwrap()
            .phase_1_mles
            .as_ref()
            .unwrap()
            .clone();

        let [_, rhs_reduced] = self
            .reduced_gate
            .as_ref()
            .unwrap()
            .phase_2_mles
            .as_ref()
            .unwrap()
            .clone();

        let mut claims = vec![];

        // grab the claim on the left sum
        let mut fixed_mle_indices_u: Vec<F> = vec![];
        for index in lhs_reduced.mle_indices() {
            fixed_mle_indices_u.push(
                index
                    .val()
                    .ok_or(LayerError::ClaimError(ClaimError::ClaimMleIndexError))?,
            );
        }
        let val = lhs_reduced.bookkeeping_table()[0];
        let claim: Claim<F> = Claim::new(
            fixed_mle_indices_u,
            val,
            Some(self.id().clone()),
            Some(self.lhs.get_layer_id()),
            Some(MleEnum::Dense(lhs_reduced.clone())),
        );
        claims.push(claim);

        // grab the claim on the right sum
        let mut fixed_mle_indices_v: Vec<F> = vec![];
        for index in rhs_reduced.mle_indices() {
            fixed_mle_indices_v.push(
                index
                    .val()
                    .ok_or(LayerError::ClaimError(ClaimError::ClaimMleIndexError))?,
            );
        }
        let val = rhs_reduced.bookkeeping_table()[0];
        let claim: Claim<F> = Claim::new(
            fixed_mle_indices_v,
            val,
            Some(self.id().clone()),
            Some(self.rhs.get_layer_id()),
            Some(MleEnum::Dense(rhs_reduced.clone())),
        );
        claims.push(claim);

        Ok(claims)
    }

    ///Gets this layers id
    fn id(&self) -> &LayerId {
        &self.layer_id
    }

    ///Create new ConcreteLayer from a LayerBuilder
    fn new<L: LayerBuilder<F>>(_builder: L, _id: LayerId) -> Self {
        unimplemented!()
    }

    fn get_wlx_evaluations(
        &self,
        claim_vecs: &Vec<Vec<F>>,
        claimed_vals: &Vec<F>,
        claimed_mles: Vec<MleEnum<F>>,
        num_claims: usize,
        num_idx: usize,
    ) -> Result<Vec<F>, ClaimError> {
        // get the number of evaluations
        let num_vars = std::cmp::max(self.lhs.num_vars(), self.rhs.num_vars());
        let (num_evals, _,) = get_num_wlx_evaluations(claim_vecs);

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

                compute_full_gate(
                    new_chal,
                    &mut self.lhs.clone(),
                    &mut self.rhs.clone(),
                    &self.nonzero_gates,
                    self.num_dataparallel_bits,
                )
            })
            .collect();

        // concat this with the first k evaluations from the claims to get num_evals evaluations
        let mut claimed_vals = claimed_vals.clone();

        claimed_vals.extend(&next_evals);
        let wlx_evals = claimed_vals;
        Ok(wlx_evals)
    }

    fn get_enum(self) -> LayerEnum<F, Self::Transcript> {
        LayerEnum::MulGateBatched(self)
    }
}

/// batched impl for gate
///
/// ## Fields
/// * `new_bits` - number of bits in p2
/// * `nonzero_gates` - Same as `AddGate` (non-batched)
/// * `lhs` - MLEs on the left side to be added.
/// * `rhs` - MLEs on the right side to be added.
/// * `num_vars_l` - Length of `x` (as in f_2(x))
/// * `num_vars_r` - Length of `y` (as in f_3(y))
/// * `copy_phase_mles` - List of MLEs for when we're binding the vars representing the batched bits
/// * `g1_challenges` - Literally g_1
/// * `g2_challenges` - Literally g_2
/// * `layer_id` - GKR layer number
/// * `reduced_gate` - the non-batched gate that this reduces to after the copy phase
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound = "F: FieldExt")]
pub struct MulGateBatched<F: FieldExt, Tr: Transcript<F>> {
    pub num_dataparallel_bits: usize,
    pub nonzero_gates: Vec<(usize, usize, usize)>,
    pub lhs: DenseMleRef<F>,
    pub rhs: DenseMleRef<F>,
    pub num_vars_l: Option<usize>,
    pub num_vars_r: Option<usize>,
    pub g1_challenges: Option<Vec<F>>,
    pub g2_challenges: Option<Vec<F>>,
    pub layer_id: LayerId,
    pub reduced_gate: Option<MulGate<F, Tr>>,
    pub beta_g2: Option<BetaTable<F>>,
    _marker: PhantomData<Tr>,
}

/// For circuit serialization to hash the circuit description into the transcript.
impl<F: std::fmt::Debug + FieldExt, Tr: Transcript<F>> MulGateBatched<F, Tr> {
    pub(crate) fn circuit_description_fmt<'a>(&'a self) -> impl std::fmt::Display + 'a {

        // --- Dummy struct which simply exists to implement `std::fmt::Display` ---
        // --- so that it can be returned as an `impl std::fmt::Display` ---
        struct MulGateBatchedCircuitDesc<'a, F: std::fmt::Debug + FieldExt, Tr: Transcript<F>>(&'a MulGateBatched<F, Tr>);

        impl<'a, F: std::fmt::Debug + FieldExt, Tr: Transcript<F>> std::fmt::Display for MulGateBatchedCircuitDesc<'a, F, Tr> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct("MulGateBatched")
                    .field("lhs_mle_ref_layer_id", &self.0.lhs.get_layer_id())
                    .field("lhs_mle_ref_mle_indices", &self.0.lhs.mle_indices())
                    .field("rhs_mle_ref_layer_id", &self.0.rhs.get_layer_id())
                    .field("rhs_mle_ref_mle_indices", &self.0.rhs.mle_indices())
                    .field("add_nonzero_gates", &self.0.nonzero_gates)
                    .field("num_dataparallel_bits", &self.0.num_dataparallel_bits)
                    .finish()
            }
        }
        MulGateBatchedCircuitDesc(self)
    }
}

impl<F: FieldExt, Tr: Transcript<F>> MulGateBatched<F, Tr> {
    /// new batched addgate thingy
    pub fn new(
        new_bits: usize,
        nonzero_gates: Vec<(usize, usize, usize)>,
        lhs: DenseMleRef<F>,
        rhs: DenseMleRef<F>,
        layer_id: LayerId,
    ) -> Self {
        MulGateBatched {
            num_dataparallel_bits: new_bits,
            nonzero_gates,
            lhs,
            rhs,
            num_vars_l: None,
            num_vars_r: None,
            g1_challenges: None,
            g2_challenges: None,
            layer_id,
            reduced_gate: None,
            beta_g2: None,
            _marker: PhantomData,
        }
    }

    /// sets all the attributes after the "copy phase" is initialized
    fn set_copy_phase(
        &mut self,
        g1_challenges: Vec<F>,
        g2_challenges: Vec<F>,
        lhs_num_vars: usize,
        rhs_num_vars: usize,
    ) {
        self.g1_challenges = Some(g1_challenges);
        self.g2_challenges = Some(g2_challenges);
        self.num_vars_l = Some(lhs_num_vars);
        self.num_vars_r = Some(rhs_num_vars);
    }

    /// initializes the copy phase
    ///
    /// ---
    ///
    /// The expression for this phase is as follows (note that we are binding `g_2`, i.e. the batch bits):
    /// * V_i(g_2, g_1) = \sum_{p_2, x, y} \beta(g_2, p_2) f_1(g_1, x, y) (f_2(p_2, x) + f_3(p_2, y))
    ///
    /// We thus need the following bookkeeping tables:
    /// * \beta(g_2, p_2)
    /// * a_f2(p_2) = \sum_{x, y} f_1(g_2, x, y) * f_2(p_2, x) = \sum_{p_2, z, x, y \in N_x} \beta(g_2, z) f_2(p_2, x)
    /// * a_f3(p_2) = \sum_{x, y} f_1(g_2, x, y) * f_3(p_2, y) = \sum_{p_2, z, x, y \in N_x} \beta(g_2, z) f_3(p_2, y)
    ///
    /// Note that --
    /// * The first one is computed via initializing a beta table.
    /// * The second/third ones are computed via iterating over all (sparse) (p_2, z, x, y) points and summing the terms above.
    fn init_copy_phase(&mut self, claim: Claim<F>) -> Result<Vec<F>, GateError> {
        let challenges = claim.get_point();
        let mut g2_challenges: Vec<F> = vec![];
        let mut g1_challenges: Vec<F> = vec![];

        // we split the claim challenges into two -- the first copy_bits number of challenges are referred
        // to as g2, and the rest are referred to as g1. this distinguishes batching from non-batching internally
        challenges
            .iter()
            .enumerate()
            .for_each(|(bit_idx, challenge)| {
                if bit_idx < self.num_dataparallel_bits {
                    g2_challenges.push(*challenge);
                } else {
                    g1_challenges.push(*challenge);
                }
            });

        // create two separate beta tables for each, as they are handled differently
        let mut beta_g2 = BetaTable::new(g2_challenges.clone()).unwrap();
        beta_g2.table.index_mle_indices(0);
        let beta_g1 = if !g1_challenges.is_empty() {
            BetaTable::new(g1_challenges.clone()).unwrap()
        } else {
            BetaTable {
                layer_claim_vars: vec![],
                table: DenseMle::new_from_raw(vec![F::one()], LayerId::Input(0), None).mle_ref(),
                relevant_indices: vec![],
            }
        };

        // index original bookkeeping tables to send over to the non-batched mul gate after the copy phase
        self.lhs.index_mle_indices(0);
        self.rhs.index_mle_indices(0);

        // --- Sets self internal state ---
        self.set_copy_phase(
            g1_challenges,
            g2_challenges,
            self.lhs.num_vars(),
            self.rhs.num_vars(),
        );

        // result of initializing is the first sumcheck message!
        let first_sumcheck_message = libra_giraffe(
            &self.lhs,
            &self.rhs,
            &beta_g2.table,
            &beta_g1.table,
            &self.nonzero_gates,
            self.num_dataparallel_bits,
        );

        // --- Need to set this to be used later ---
        self.beta_g2 = Some(beta_g2);

        first_sumcheck_message
    }
}
