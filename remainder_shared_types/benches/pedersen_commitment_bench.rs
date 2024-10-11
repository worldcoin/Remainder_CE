use ark_std::test_rng;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use halo2curves::bn256::G1 as Bn256Point;
use rand::Rng;
use remainder_shared_types::curves::PrimeOrderCurve;
type Bn256Scalar = <Bn256Point as PrimeOrderCurve>::Scalar;
use remainder_shared_types::pedersen::PedersenCommitter;
use remainder_shared_types::{ff_field, Field};

fn create_random_field_vec<F: Field>(num_items: usize, mut rng: impl Rng) -> Vec<F> {
    (0..num_items).map(|_| F::random(&mut rng)).collect()
}

/// BenchmarksIDs:
///  * `pedersen_vector_commit_bench_log_{1, 2, 3, 4, 5, 6, 7, 8, 9}`
///
/// Bench how long it takes to do a Pedersen commitment for 2^`log_num_elems` provided.
fn bench_vector_commitment(c: &mut Criterion) {
    for log_num_elems in [1, 2, 3, 4, 5, 6, 7, 8, 9] {
        c.bench_function(
            &format!("pedersen_vector_commit_bench_log_{log_num_elems}"),
            |b| {
                b.iter_batched(
                    || {
                        let mut rng = test_rng();
                        let vec_to_commit =
                            create_random_field_vec::<Bn256Scalar>(1 << log_num_elems, &mut rng);
                        let blinding = Bn256Scalar::random(&mut rng);
                        let committer = PedersenCommitter::<Bn256Point>::new(
                            1 << log_num_elems,
                            "Testing Pedersen Committer Trying to get to 32 Characters",
                            None,
                        );
                        (committer, vec_to_commit, blinding)
                    },
                    |(committer, vec_to_commit, blinding_factor)| {
                        committer.vector_commit(&vec_to_commit, &blinding_factor);
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }
}

criterion_group!(benches, bench_vector_commitment);
criterion_main!(benches);
