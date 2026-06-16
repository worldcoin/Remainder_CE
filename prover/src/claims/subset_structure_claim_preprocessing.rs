use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
};

use itertools::Itertools;
use shared_types::{
    transcript::{ProverTranscript, VerifierTranscript},
    Field,
};

use crate::{claims::Claim, layer::LayerId};

/// A claim group representing a group of "subset structure" claims.
///
/// Note that `subset_vars` refers to the indices of variables which span {0, 1}
/// between the different claims. For example,
/// `[r_1, r_2, 0, r_4]; [r_1, r_2, 1, r_4]` would have `subset_vars = \{3\}`
#[derive(Clone, Debug)]
pub struct SubsetStructureClaimGroup<F: Field> {
    _claim_num_vars: usize,
    from_layer_id: LayerId,
    to_layer_id: LayerId,
    claims: Vec<Claim<F>>,
    subset_vars: HashSet<usize>,
    full_subset: bool, // If we don't have a "full subset" then we simply don't aggregate for now.
}

impl<F: Field> SubsetStructureClaimGroup<F> {
    /// Constructor from a (complete) subset of claims.
    pub fn new(claims: Vec<Claim<F>>) -> Self {
        // Sanitycheck: Non-empty claim group
        assert!(!claims.is_empty());
        // Sanitycheck: All claims have the same length and come from the same
        // source layer
        for (i, claim) in claims.iter().enumerate() {
            assert_eq!(
                claim.get_point().len(),
                claims[0].get_point().len(),
                "Claim {i} point length {} differs from claim 0 point length {}",
                claim.get_point().len(),
                claims[0].get_point().len(),
            );
            assert_eq!(
                claim.get_num_vars(),
                claims[0].get_num_vars(),
                "Claim {i} num_vars {} differs from claim 0 num_vars {}",
                claim.get_num_vars(),
                claims[0].get_num_vars(),
            );
            assert_eq!(
                claim.get_to_layer_id(),
                claims[0].get_to_layer_id(),
                "Claim {i} to_layer_id {:?} differs from claim 0 to_layer_id {:?}",
                claim.get_to_layer_id(),
                claims[0].get_to_layer_id(),
            );
        }
        // Compute the set of variables which differ between all the claims.
        let subset_vars = Self::compute_subset_vars_for_claim_group(&claims);
        Self {
            _claim_num_vars: claims[0].get_num_vars(),
            from_layer_id: claims[0].get_from_layer_id(),
            to_layer_id: claims[0].get_to_layer_id(),
            full_subset: claims.len() == 1 << subset_vars.len(),
            claims,
            subset_vars,
        }
    }

    /// Returns the set of variable indices which differ in \{0, 1\} between at
    /// least a pair of claims.
    fn compute_subset_vars_for_claim_group(claims: &[Claim<F>]) -> HashSet<usize> {
        let mut subset_vars = HashSet::new();
        let first_claim = claims.first().unwrap();
        claims.iter().for_each(|claim| {
            claim
                .get_point()
                .iter()
                .enumerate()
                .zip(first_claim.get_point().iter())
                .for_each(|((chal_idx, claim_chal), first_claim_chal)| {
                    // If the challeges are binary _and_ they do not match, add
                    // to the list of challenges.
                    if (*claim_chal == F::ZERO || *claim_chal == F::ONE)
                        && claim_chal != first_claim_chal
                    {
                        subset_vars.insert(chal_idx);
                    }
                })
        });
        subset_vars
    }

    /// Has just one claim, i.e. no aggregation needed.
    pub fn is_trivial(&self) -> bool {
        self.claims.len() == 1
    }

    /// Performs subset structure aggregation and returns the resulting claim,
    /// writing all prover messages (including the resulting claim) into the
    /// provided `prover_transcript`.
    pub fn aggregate(self, challenge_sampler: &mut impl FnMut(usize, usize) -> F) -> Vec<Claim<F>> {
        if self.is_trivial() {
            // If there is only one claim, no need for additional aggregation.
            vec![self.claims[0].clone()]
        } else if !self.full_subset {
            // If we don't have a full subset, just return all the claims without doing any aggregation.
            self.claims
        } else {
            // Otherwise, perform exact-subset aggregation. In theory _all_
            // of the subsets should be present, but this may not be true.

            // First, sample challenges for each of the subset variables.
            let challenges_for_subset_vars: HashMap<usize, F> = self
                .subset_vars
                .iter()
                .sorted()
                .map(|subset_var_idx| {
                    let transcript_challenge =
                        challenge_sampler(self.from_layer_id.get_raw_layer_id(), *subset_var_idx);
                    (*subset_var_idx, transcript_challenge)
                })
                .collect();

            // Get the resulting claimed point by inserting challenges into the
            // appropriate positions
            let resulting_claim_point = self.claims[0]
                .get_point()
                .to_vec()
                .into_iter()
                .enumerate()
                .map(|(idx, chal)| *(challenges_for_subset_vars.get(&idx).unwrap_or(&chal)))
                .collect_vec();

            // Next, aggregate by summing over all the claims and multiplying
            // by the correct challenge values.
            let resulting_claimed_value = self.claims.into_iter().fold(F::ZERO, |acc, claim| {
                let coeff_product = challenges_for_subset_vars.iter().fold(
                    F::ONE,
                    |inner_acc, (chal_idx, chal)| {
                        inner_acc
                            * if claim.get_point()[*chal_idx] == F::ZERO {
                                F::ONE - *chal
                            } else {
                                *chal
                            }
                    },
                );
                acc + coeff_product * claim.get_eval()
            });

            vec![Claim::new(
                resulting_claim_point,
                resulting_claimed_value,
                self.from_layer_id,
                self.to_layer_id,
            )]
        }
    }
}

/// Idea here is to aggregate claims with an _exact_ subset-like structure.
/// We will only "trivially" aggregate those where the claim points are either
/// all identical challenges or exactly form 0/1 pairs.
///
/// For now, we assert that _all_ of the subset claims are present.
///
/// See [create_subset_structure_claim_groups()] for more details.
pub fn prover_aggregate_subset_structure_claims<F: Field>(
    all_claims_on_layer: &[Claim<F>],
    prover_transcript: &mut impl ProverTranscript<F>,
) -> Vec<Claim<F>> {
    // First, we form the exact-subset claim groups.
    let exact_subset_claim_groups = create_subset_structure_claim_groups(all_claims_on_layer);

    // For each of the exact-subset claim groups, we aggregate into a single layer.
    let ret: Vec<Claim<F>> = exact_subset_claim_groups
        .into_iter()
        .flat_map(|exact_subset_claim_group| {
            let mut challenge_sampler = |from_layer_id, var_idx| {
                prover_transcript.get_challenge(&format!(
                    "Challenge for subset aggregation: from_layer {from_layer_id}, var_idx {var_idx}"
                ))
            };
            // Note that no values need to be added to transcript here, since
            // the verifier can compute the claimed evaluation on its own.
            exact_subset_claim_group.aggregate(&mut challenge_sampler)
        })
        .collect();

    ret
}

/// The verifier's analogous function to
/// [prover_aggregate_subset_structure_claims()]. See the prover's function for
/// more details.
///
/// TODO(ryancao): Refactor this to also use the `challenge_sampler` like the
/// prover's version.
pub fn verifier_aggregate_subset_structure_claims<F: Field>(
    all_claims_on_layer: &[Claim<F>],
    verifier_transcript: &mut impl VerifierTranscript<F>,
) -> Vec<Claim<F>> {
    // First, we form the exact-subset claim groups.
    let exact_subset_claim_groups = create_subset_structure_claim_groups(all_claims_on_layer);

    // For each of the exact-subset claim groups, we aggregate into a single layer.
    exact_subset_claim_groups
        .into_iter()
        .flat_map(|exact_subset_claim_group| {
            let mut challenge_sampler = |from_layer_id, var_idx| {
                let label = Box::leak(
                    format!(
                        "Challenge for subset aggregation: from_layer {from_layer_id}, var_idx {var_idx}",
                    )
                    .into_boxed_str(),
                );
                verifier_transcript.get_challenge(label).unwrap()
            };
            // Note that no values need to be added to transcript here, since
            // the verifier can compute the claimed evaluation on its own.
            exact_subset_claim_group.aggregate(&mut challenge_sampler)
        })
        .collect()
}

/// Compares two claims lexicographically, ignoring all non-binary claim point
/// values. Note that this function is not currently used but will be used if
/// the functionality of not requiring complete subsets is implemented.
#[allow(dead_code)]
fn compare_claims_in_exact_subset<F: Field>(claim_1: &Claim<F>, claim_2: &Claim<F>) -> Ordering {
    assert_eq!(claim_1.get_point().len(), claim_2.get_point().len());
    let mut order = None;
    // Iterate through challenges lexicographically.
    for (claim_1_chal, claim_2_chal) in claim_1.get_point().iter().zip(claim_2.get_point().iter()) {
        // Both must be binary...
        if *claim_1_chal == F::ZERO || *claim_1_chal == F::ONE {
            // If they are not equal, bingo!
            if claim_1_chal != claim_2_chal {
                if *claim_1_chal == F::ZERO {
                    order = Some(Ordering::Less);
                } else {
                    order = Some(Ordering::Greater);
                }
                break;
            }
        }
    }

    // This causes us to crash rather than fail silently if there were no binary
    // claim point values which differed.
    order.unwrap()
}

/// For now we are just going to do grouping and nothing else.
///
/// The `Vec<Vec<Claim<F>>>` structure is as follows: the outer `Vec` is all of
/// the groups, while the inner `Vec` is the set of claims held within a single
/// group. Within a single group the above properties are guaranteed.
///
/// There is a problem of duplicates. We do not allow duplicates, and more
/// generally do not allow for a (0, r) match or a (1, r) match.
///
/// In order to be part of a group, you must "match" against every other claim
/// within that group.
pub fn create_subset_structure_claim_groups<F: Field>(
    all_claims_on_layer: &[Claim<F>],
) -> Vec<SubsetStructureClaimGroup<F>> {
    // Every layer should have at least one claim on it...
    assert!(!all_claims_on_layer.is_empty());

    // We do this dynamically.
    let mut all_groupings: Vec<Vec<Claim<F>>> = vec![];
    all_groupings.push(vec![all_claims_on_layer[0].clone()]);
    all_claims_on_layer[1..]
        .iter()
        .for_each(|claim_to_consider| {
            // We greedily assign to the first grouping where all claims in that
            // group "match" with the current claim. This is quite slow but we assume
            // that it's still much faster than non-trivial claim aggregation.
            let mut maybe_belongs_idx = None;
            all_groupings
                .iter()
                .enumerate()
                .for_each(|(grouping_idx, grouping)| {
                    let belongs = grouping
                        .iter()
                        .all(|grouping_claim| claims_match(grouping_claim, claim_to_consider));
                    if belongs {
                        maybe_belongs_idx = Some(grouping_idx);
                    }
                });
            match maybe_belongs_idx {
                Some(belongs_idx) => {
                    // If this claim belonged to a group, put it in that
                    // grouping.
                    all_groupings[belongs_idx].push(claim_to_consider.clone());
                }
                None => {
                    // If this claim didn't belong to any grouping, put it in
                    // its own grouping.
                    all_groupings.push(vec![claim_to_consider.clone()]);
                }
            }
        });

    all_groupings
        .into_iter()
        .map(|raw_claim_grouping| SubsetStructureClaimGroup::new(raw_claim_grouping))
        .collect()
}

/// Two claims "match" if...
/// a) All of their non-binary challenges match, AND
/// b) Of their binary challenges, at least one is non-similar
///
/// Note that implicitly, two identical claims with no binary challenges will
/// _not_ be seen as a match.
fn claims_match<F: Field>(claim_1: &Claim<F>, claim_2: &Claim<F>) -> bool {
    assert!(claim_1.get_point().len() == claim_2.get_point().len());
    let mut nonbinary_challenges_match = true;
    let mut binary_challenges_at_least_one_non_match = false;
    claim_1
        .get_point()
        .iter()
        .zip(claim_2.get_point())
        .for_each(|(claim_1_chal, claim_2_chal)| {
            if *claim_1_chal != F::ZERO && *claim_1_chal != F::ONE
                || (*claim_2_chal != F::ZERO && *claim_2_chal != F::ONE)
            {
                // If the challenge is non-binary, they must match
                nonbinary_challenges_match &= *claim_1_chal == *claim_2_chal;
            } else {
                // If the challenge is non-binary, at least one must NOT match
                binary_challenges_at_least_one_non_match |= *claim_1_chal != *claim_2_chal;
            }
        });
    nonbinary_challenges_match && binary_challenges_at_least_one_non_match
}

// -------------------- INPUT LAYER CLAIM WRAPPERS --------------------

/// Wrapper around [prover_aggregate_subset_structure_claims()] which first
/// splits claims by their respective input layer.
pub(crate) fn prover_input_layer_subset_structure_claim_agg<F: Field>(
    input_layer_claims: &[Claim<F>],
    prover_transcript: &mut impl ProverTranscript<F>,
) -> Vec<Claim<F>> {
    // First group by input layer ID...
    let mut claims_grouped_by_input_layer: HashMap<LayerId, Vec<Claim<F>>> = HashMap::new();
    input_layer_claims.iter().sorted().for_each(|claim| {
        claims_grouped_by_input_layer
            .entry(claim.get_to_layer_id())
            .or_default()
            .push(claim.clone());
    });

    // Then (potentially) aggregate using the subset structure aggregation strategy.
    claims_grouped_by_input_layer
        .into_iter()
        .sorted_by(|x, y| x.0.cmp(&y.0))
        .flat_map(|(_input_layer_id, claims_on_one_input_layer)| {
            prover_aggregate_subset_structure_claims(&claims_on_one_input_layer, prover_transcript)
        })
        .collect()
}

/// Wrapper around [verifier_aggregate_subset_structure_claims()] which first
/// splits claims by their respective input layer.
pub(crate) fn verifier_input_layer_subset_structure_claim_agg<F: Field>(
    input_layer_claims: &[Claim<F>],
    verifier_transcript: &mut impl VerifierTranscript<F>,
) -> Vec<Claim<F>> {
    // First group by input layer ID...
    let mut claims_grouped_by_input_layer: HashMap<LayerId, Vec<Claim<F>>> = HashMap::new();
    input_layer_claims.iter().sorted().for_each(|claim| {
        claims_grouped_by_input_layer
            .entry(claim.get_to_layer_id())
            .or_default()
            .push(claim.clone());
    });

    // Then (potentially) aggregate using the subset structure aggregation strategy.
    claims_grouped_by_input_layer
        .into_iter()
        .sorted_by(|x, y| x.0.cmp(&y.0))
        .flat_map(|(_input_layer_id, claims_on_one_input_layer)| {
            verifier_aggregate_subset_structure_claims(
                &claims_on_one_input_layer,
                verifier_transcript,
            )
        })
        .collect()
}

/// Tests for subset structure claim aggregation.
#[cfg(test)]
pub mod tests {
    use crate::{
        claims::{
            subset_structure_claim_preprocessing::{
                prover_aggregate_subset_structure_claims,
                verifier_aggregate_subset_structure_claims,
            },
            Claim,
        },
        mle::evals::MultilinearExtension,
    };
    use shared_types::{
        transcript::{poseidon_sponge::PoseidonSponge, TranscriptReader, TranscriptWriter},
        Field, Fr,
    };

    fn setup_two_claim_groups_simple<F: Field>() -> (Vec<Claim<F>>, MultilinearExtension<F>) {
        use ark_std::test_rng;
        use ligero::utils::get_random_coeffs_for_multilinear_poly;

        use crate::{claims::Claim, mle::evals::MultilinearExtension};

        // First claim group: [r_1, _, _,]
        // Second claim group: [_, _, _,]

        let mut rng = test_rng();
        let mle = MultilinearExtension::new(get_random_coeffs_for_multilinear_poly(3, &mut rng));

        let all_claim_points = vec![
            // [r_1, _, _]
            vec![F::from(10), F::ZERO, F::ZERO],
            vec![F::from(10), F::ZERO, F::ONE],
            vec![F::from(10), F::ONE, F::ZERO],
            vec![F::from(10), F::ONE, F::ONE],
            // [_, _, _]
            vec![F::ZERO, F::ZERO, F::ZERO],
            vec![F::ZERO, F::ZERO, F::ONE],
            vec![F::ZERO, F::ONE, F::ZERO],
            vec![F::ZERO, F::ONE, F::ONE],
            vec![F::ONE, F::ZERO, F::ZERO],
            vec![F::ONE, F::ZERO, F::ONE],
            vec![F::ONE, F::ONE, F::ZERO],
            vec![F::ONE, F::ONE, F::ONE],
        ];

        // Create claims from the claim points and the source MLE.
        (
            all_claim_points
                .iter()
                .map(|claim_point| {
                    let evaluation = mle.evaluate_at_point(claim_point);
                    Claim::new(
                        claim_point.clone(),
                        evaluation,
                        crate::layer::LayerId::Layer(1),
                        crate::layer::LayerId::Layer(0),
                    )
                })
                .collect(),
            mle,
        )
    }

    #[test]
    fn test_num_groups_two_claim_groups_simple() {
        let (claims, _source_mle) = setup_two_claim_groups_simple::<Fr>();

        let mut prover_transcript: TranscriptWriter<Fr, PoseidonSponge<Fr>> =
            TranscriptWriter::new("Test two claim groups simple");

        // Perform aggregation process.
        let aggregated = prover_aggregate_subset_structure_claims(&claims, &mut prover_transcript);

        // There should be two final claims.
        assert_eq!(aggregated.len(), 2);
    }

    #[test]
    fn test_prover_verifier_flow_two_claim_groups_simple() {
        // ------- PROVER SIDE -------
        let (claims, source_mle) = setup_two_claim_groups_simple::<Fr>();

        let mut prover_transcript: TranscriptWriter<Fr, PoseidonSponge<Fr>> =
            TranscriptWriter::new("Test two claim groups simple");

        // Perform aggregation process.
        let prover_aggregated =
            prover_aggregate_subset_structure_claims(&claims, &mut prover_transcript);

        // ------- VERIFIER SIDE -------
        let mut verifier_transcript: TranscriptReader<Fr, PoseidonSponge<Fr>> =
            TranscriptReader::new(prover_transcript.get_transcript());
        let verifier_aggregated =
            verifier_aggregate_subset_structure_claims(&claims, &mut verifier_transcript);

        // Check that the prover + verifier receive the same aggregated claims.
        assert_eq!(verifier_aggregated, prover_aggregated);

        // Check that the prover's claims are correct.
        assert!(verifier_aggregated
            .iter()
            .all(|claim| { claim.get_eval() == source_mle.evaluate_at_point(claim.get_point()) }));
    }
}
