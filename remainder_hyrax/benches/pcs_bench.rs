use ark_std::test_rng;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use halo2curves::bn256::G1 as Bn256Point;
use rand::Rng;
use remainder_hyrax::hyrax_pcs::{HyraxPCSEvaluationProof, MleCoefficientsVector};
use remainder_shared_types::{
    curves::PrimeOrderCurve, halo2curves, pedersen::PedersenCommitter, Field,
};
type Bn256Scalar = <Bn256Point as PrimeOrderCurve>::Scalar;

fn create_random_field_vec<F: Field>(num_items: usize, mut rng: impl Rng) -> Vec<F> {
    (0..num_items).map(|_| F::random(&mut rng)).collect()
}
/// BenchmarksIDs:
///  * `PCS benchmark for 2^{1, 2, 3, 4, 5, 6, 7, 8, 9} by 2^{1, 2, 3, 4, 5, 6, 7, 8, 9} matrix`
///
/// Bench the PCS commitment step for Hyrax.
fn bench_pcs_commit(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench group");

    // Set the sample size to a smaller number
    group.sample_size(10);

    for log_num_elems in [1, 2, 3, 4, 5, 6, 7, 8, 9] {
        group.bench_function(
            &format!("PCS benchmark for 2^{log_num_elems} by 2^{log_num_elems} matrix"),
            |b| {
                b.iter_batched(
                    || {
                        let mut rng = test_rng();
                        let matrix_to_commit = create_random_field_vec::<Bn256Scalar>(
                            (1 << log_num_elems) * (1 << log_num_elems),
                            &mut rng,
                        );
                        let blinding_factors =
                            create_random_field_vec::<Bn256Scalar>(1 << log_num_elems, &mut rng);
                        let committer = PedersenCommitter::<Bn256Point>::new(
                            1 << log_num_elems,
                            "Testing Pedersen Committer Trying to get to 32 Characters",
                            None,
                        );
                        let mle_coeffs_data =
                            MleCoefficientsVector::ScalarFieldVector(matrix_to_commit);
                        (mle_coeffs_data, committer, blinding_factors)
                    },
                    |(data, committer, blinding_factors)| {
                        HyraxPCSEvaluationProof::compute_matrix_commitments(
                            log_num_elems,
                            &data,
                            &committer,
                            &blinding_factors,
                        )
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }
}

criterion_group!(benches, bench_pcs_commit);
criterion_main!(benches);
