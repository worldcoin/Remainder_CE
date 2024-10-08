use ark_std::test_rng;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use halo2curves::bn256::G1 as Bn256Point;
use rand::Rng;
use remainder_hyrax::{
    hyrax_pcs::{HyraxPCSProof, MleCoefficientsVector},
    pedersen::PedersenCommitter,
};
use remainder_shared_types::{curves::PrimeOrderCurve, halo2curves, Field};
type Bn256Scalar = <Bn256Point as PrimeOrderCurve>::Scalar;

fn create_random_field_vec<F: Field>(num_items: usize, mut rng: impl Rng) -> Vec<F> {
    (0..num_items).map(|_| F::random(&mut rng)).collect()
}

fn bench_pcs(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench group");

    // Set the sample size to a smaller number, e.g., 10
    group.sample_size(10);

    group.bench_function(&format!("PCS benchmark for 2^9 by 2^9 matrix"), |b| {
        b.iter_batched(
            || {
                let mut rng = test_rng();
                let matrix_to_commit =
                    create_random_field_vec::<Bn256Scalar>((1 << 9) * (1 << 9), &mut rng);
                let blinding_factors = create_random_field_vec::<Bn256Scalar>((1 << 9), &mut rng);
                let committer = PedersenCommitter::<Bn256Point>::new(
                    1 << 9,
                    "Testing Pedersen Committer Trying to get to 32 Characters",
                    None,
                );
                let mle_coeffs_data = MleCoefficientsVector::ScalarFieldVector(matrix_to_commit);
                (mle_coeffs_data, committer, blinding_factors)
            },
            |(data, committer, blinding_factors)| {
                HyraxPCSProof::compute_matrix_commitments(9, &data, &committer, &blinding_factors)
            },
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, bench_pcs);
criterion_main!(benches);
