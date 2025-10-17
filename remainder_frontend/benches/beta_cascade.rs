use ark_std::test_rng;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use itertools::Itertools;
use rand::Rng;
use remainder::{
    layer::LayerId,
    mle::{dense::DenseMle, Mle},
    sumcheck::beta_cascade,
};
use remainder_shared_types::{curves::PrimeOrderCurve, Bn256Point, Field};
type Bn256Scalar = <Bn256Point as PrimeOrderCurve>::Scalar;

fn create_random_field_vec<F: Field>(num_items: usize, mut rng: impl Rng) -> Vec<F> {
    (0..num_items).map(|_| F::random(&mut rng)).collect()
}

/// BenchmarksIDs:
///  * "bench_beta_cascade_{10, 15, 20, 25}_log_vars_{2, 4, 8}_mles"
///
/// Benchmarks for beta cascade, the algorithm which computes sumcheck message evaluations
/// by factoring a beta table (as detailed in [BetaValues]).
fn bench_beta_cascade(c: &mut Criterion) {
    for log_num_variables in [10, 15, 20, 25] {
        for num_mles in [2, 4, 8] {
            let mut group = c.benchmark_group("beta cascade benchmark");
            group.bench_function(
                &format!("bench_beta_cascade_{log_num_variables}_log_vars_{num_mles}_mles"),
                |b| {
                    b.iter_batched(
                        || {
                            let mut rng = test_rng();
                            let mle_vec = (0..num_mles)
                                .map(|_idx| {
                                    let random_mle_vec = create_random_field_vec::<Bn256Scalar>(
                                        1 << log_num_variables,
                                        &mut rng,
                                    );
                                    let mut mle =
                                        DenseMle::new_from_raw(random_mle_vec, LayerId::Input(0));
                                    mle.index_mle_indices(0);
                                    mle
                                })
                                .collect_vec();
                            let beta_vals =
                                create_random_field_vec::<Bn256Scalar>(log_num_variables, &mut rng);
                            let beta_updated_vals = vec![vec![]];
                            let round_index = 0;
                            let degree = num_mles + 1;
                            (mle_vec, beta_vals, beta_updated_vals, round_index, degree)
                        },
                        |(mle_vec, beta_vals, beta_updated_vals, round_index, degree)| {
                            beta_cascade(
                                &mle_vec.iter().collect_vec(),
                                degree,
                                round_index,
                                &[beta_vals],
                                &beta_updated_vals,
                                &[Bn256Scalar::one()],
                            )
                        },
                        BatchSize::SmallInput,
                    )
                },
            );
        }
    }
}

criterion_group!(benches, bench_beta_cascade);
criterion_main!(benches);
