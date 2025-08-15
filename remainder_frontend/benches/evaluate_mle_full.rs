use ark_std::test_rng;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use halo2curves::bn256::G1 as Bn256Point;
use rand::Rng;
use remainder::{
    mle::evals::MultilinearExtension,
    utils::mle::{
        evaluate_mle_at_a_point_gray_codes, evaluate_mle_at_a_point_gray_codes_parallel,
        evaluate_mle_at_a_point_lexicographic_order, evaluate_mle_destructive,
    },
};
use remainder_shared_types::{curves::PrimeOrderCurve, halo2curves, Field};
type Bn256Scalar = <Bn256Point as PrimeOrderCurve>::Scalar;

fn create_random_field_vec<F: Field>(num_items: usize, mut rng: impl Rng) -> Vec<F> {
    (0..num_items).map(|_| F::random(&mut rng)).collect()
}

/// BenchmarksIDs:
///  * "Fully evaluate an MLE with {2, 4, 6, 8, 10, 12, 14, 16, 18, 20} at a random point using the gray codes algorithm."
///
/// Benchmarks evaluating an MLE non-destructively in O(2^n) time where n is the number of variables, using
/// the gray codes algorithm as described in [GrayCodeIterator].
fn bench_gray_codes_mle_evaluation(c: &mut Criterion) {
    for log_num_variables in [22, 23, 24, 25, 26] {
        let mut group = c.benchmark_group("bench_gray_codes_mle_evaluation");
        group.sample_size(25);
        group.bench_function(
            &format!("MLE Eval, gray codes, {log_num_variables} variables."),
            |b| {
                b.iter_batched(
                    || {
                        let mut rng = test_rng();
                        let random_mle_vec = create_random_field_vec::<Bn256Scalar>(
                            1 << log_num_variables,
                            &mut rng,
                        );
                        let mle = MultilinearExtension::new(random_mle_vec);
                        let random_point =
                            create_random_field_vec::<Bn256Scalar>(log_num_variables, &mut rng);
                        (mle, random_point)
                    },
                    |(mle, random_point)| evaluate_mle_at_a_point_gray_codes(&mle, &random_point),
                    BatchSize::SmallInput,
                )
            },
        );
    }
}

/// BenchmarksIDs:
///  * "Fully evaluate an MLE with {2, 4, 6, 8, 10, 12, 14, 16, 18, 20} at a random point by destructively fixing variable."
///
/// Benchmarks evaluating an MLE destructively in O(2^n) time where n is the number of variables, using
/// the fix-variable algorithm iteratively until all the variables are bound to compare with the above.
fn bench_destructive_mle_evaluation(c: &mut Criterion) {
    for log_num_variables in [22, 23, 24, 25, 26] {
        let mut group = c.benchmark_group("bench_destructive_mle_evaluation");
        group.sample_size(25);
        group.bench_function(
            &format!("MLE Eval, destructively fixing variable, {log_num_variables} variables."),
            |b| {
                b.iter_batched(
                    || {
                        let mut rng = test_rng();
                        let random_mle_vec = create_random_field_vec::<Bn256Scalar>(
                            1 << log_num_variables,
                            &mut rng,
                        );
                        let mle = MultilinearExtension::new(random_mle_vec);
                        let random_point =
                            create_random_field_vec::<Bn256Scalar>(log_num_variables, &mut rng);
                        (mle, random_point)
                    },
                    |(mut mle, random_point)| evaluate_mle_destructive(&mut mle, &random_point),
                    BatchSize::SmallInput,
                )
            },
        );
    }
}

/// BenchmarksIDs:
///  * "Fully evaluate an MLE with {2, 4, 6, 8, 10, 12, 14, 16, 18, 20} at a random point using the gray codes algorithm."
///
/// Benchmarks evaluating an MLE non-destructively in O(2^n) time where n is the number of variables, using
/// the gray codes algorithm as described in [GrayCode].
///
/// Parallelized with K threads.
fn bench_gray_codes_mle_evaluation_parallel<const K: usize>(c: &mut Criterion) {
    for log_num_variables in [22, 23, 24, 25, 26] {
        let mut group = c.benchmark_group("bench_gray_codes_mle_evaluation_parallel");
        group.sample_size(25);
        group.bench_function(
            &format!("MLE Eval, parallel {K} thread gray codes, {log_num_variables} variables."),
            |b| {
                b.iter_batched(
                    || {
                        let mut rng = test_rng();
                        let random_mle_vec = create_random_field_vec::<Bn256Scalar>(
                            1 << log_num_variables,
                            &mut rng,
                        );
                        let mle = MultilinearExtension::new(random_mle_vec);
                        let random_point =
                            create_random_field_vec::<Bn256Scalar>(log_num_variables, &mut rng);
                        (mle, random_point)
                    },
                    |(mle, random_point)| {
                        evaluate_mle_at_a_point_gray_codes_parallel::<remainder_shared_types::Fr, K>(&mle, &random_point)
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }
}

/// BenchmarksIDs:
///  * "Fully evaluate an MLE with {2, 4, 6, 8, 10, 12, 14, 16, 18, 20} at a random point using the lexicographic order."
///
/// Benchmarks evaluating an MLE non-destructively in O(2^n) time where n is the number of variables, using
/// the lexicographic order as described in [LexicographicLE].
fn bench_lexicographic_mle_evaluation(c: &mut Criterion) {
    for log_num_variables in [22, 23, 24, 25, 26] {
        let mut group = c.benchmark_group("bench_lexicographic_mle_evaluation");
        group.sample_size(25);
        group.bench_function(
            &format!("MLE Eval, fixing with lexicographic order, {log_num_variables} variables."),
            |b| {
                b.iter_batched(
                    || {
                        let mut rng = test_rng();
                        let random_mle_vec = create_random_field_vec::<Bn256Scalar>(
                            1 << log_num_variables,
                            &mut rng,
                        );
                        let mle = MultilinearExtension::new(random_mle_vec);
                        let random_point =
                            create_random_field_vec::<Bn256Scalar>(log_num_variables, &mut rng);
                        (mle, random_point)
                    },
                    |(mle, random_point)| {
                        evaluate_mle_at_a_point_lexicographic_order(&mle, &random_point)
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }
}

criterion_group!(
    benches,
    bench_gray_codes_mle_evaluation,
    bench_gray_codes_mle_evaluation_parallel::<4>,
    bench_gray_codes_mle_evaluation_parallel::<8>,
    bench_gray_codes_mle_evaluation_parallel::<16>,
    bench_destructive_mle_evaluation,
    bench_lexicographic_mle_evaluation
);
criterion_main!(benches);
