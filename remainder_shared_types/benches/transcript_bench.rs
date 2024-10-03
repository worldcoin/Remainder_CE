//! ============== INSTRUCTIONS ==============
//! Each benchmark is identified by a BenchmarkID string which is documented in
//! the docstring comment of each benchmark function in this file.
//!
//! To run all the benchmarks in this file which match a regular expression
//! REGEX, use the following command:
//!
//!  > cargo bench --bench transcript_bench -- 'REGEX'

use ark_std::test_rng;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
type Bn256Scalar = <Bn256Point as PrimeOrderCurve>::Scalar;
type Bn256Base = <Bn256Point as PrimeOrderCurve>::Base;
use halo2curves::bn256::G1 as Bn256Point;
use rand::Rng;
use remainder_shared_types::curves::PrimeOrderCurve;
use remainder_shared_types::transcript::ec_transcript::ECProverTranscript;
use remainder_shared_types::{
    transcript::{
        ec_transcript::ECTranscriptWriter, poseidon_transcript::PoseidonSponge, ProverTranscript,
        TranscriptWriter,
    },
    Field,
};

// =========== Helper functions ===========

/// Literally creates a vector of uniformly random field elements.
fn create_random_field_vec<F: Field>(num_items: usize, mut rng: impl Rng) -> Vec<F> {
    (0..num_items).map(|_| F::random(&mut rng)).collect()
}

// ========================================

/// BenchmarksIDs:
///  * `bn254_scalar_field_transcript_absorb_{1, 10, 100, 1000, 10000}`
///
/// Benchmarks the [TranscriptWriter::append_elements()] functionality for
/// a Poseidon sponge over the scalar field of BN-254.
fn bench_bn254_scalar_field_transcript_absorb(c: &mut Criterion) {
    for num_elems in [1, 10, 100, 1000, 10000] {
        c.bench_function(
            &format!("bn254_scalar_field_transcript_absorb_{num_elems}"),
            |b| {
                b.iter_batched(
                    || {
                        let rng = test_rng();
                        (
                            create_random_field_vec::<Bn256Scalar>(num_elems, rng),
                            TranscriptWriter::<Bn256Scalar, PoseidonSponge<Bn256Scalar>>::new(
                                "Test transcript",
                            ),
                        )
                    },
                    |(to_be_absorbed, mut transcript_writer)| {
                        transcript_writer.append_elements("benchmark", &to_be_absorbed);
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }
}

/// BenchmarksIDs:
///  * `bn254_scalar_field_transcript_squeeze_{1, 10, 100, 1000, 10000}`
///
/// Benchmarks the [TranscriptWriter::get_challenges()] functionality for
/// a Poseidon sponge over the scalar field of BN-254.
fn bench_bn254_scalar_field_transcript_squeeze(c: &mut Criterion) {
    for num_elems in [1, 10, 100, 1000, 10000] {
        c.bench_function(
            &format!("bn254_scalar_field_transcript_squeeze_{num_elems}"),
            |b| {
                b.iter_batched(
                    || {
                        TranscriptWriter::<Bn256Scalar, PoseidonSponge<Bn256Scalar>>::new(
                            "Test transcript",
                        )
                    },
                    |mut transcript_writer| {
                        transcript_writer.get_challenges("benchmark", num_elems)
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }
}

/// BenchmarksIDs:
///  * `bn254_base_field_transcript_absorb_scalar_{1, 10, 100, 1000, 10000}`
///
/// Benchmarks the [ECTranscriptWriter::append_scalar_points()] functionality for
/// a Poseidon sponge over the base field of BN-254 absorbing scalar field
/// elements of BN-254.
fn bench_bn254_base_field_transcript_absorb_scalar(c: &mut Criterion) {
    for num_elems in [1, 10, 100, 1000, 10000] {
        c.bench_function(
            &format!("bn254_base_field_transcript_absorb_scalar_{num_elems}"),
            |b| {
                b.iter_batched(
                    || {
                        let rng = test_rng();
                        (
                            create_random_field_vec::<Bn256Scalar>(num_elems, rng),
                            ECTranscriptWriter::<Bn256Point, PoseidonSponge<Bn256Base>>::new(
                                "Test transcript",
                            ),
                        )
                    },
                    |(to_be_absorbed, mut transcript_writer)| {
                        // TODO(ryancao): I don't know why the straightforward function call doesn't work
                        <ECTranscriptWriter<
                            Bn256Point,
                            PoseidonSponge<Bn256Base>,
                        > as ECProverTranscript<Bn256Point>>::append_scalar_points(
                            &mut transcript_writer,
                            "benchmark",
                            &to_be_absorbed,
                        );
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }
}

/// BenchmarksIDs:
///  * `bn254_base_field_transcript_squeeze_scalar_{1, 10, 100, 1000, 10000}`
///
/// Benchmarks the [TranscriptWriter::get_scalar_field_challenges()] functionality for
/// a Poseidon sponge over the base field of BN-254 squeezing scalar field
/// verifier challenges.
fn bench_bn254_base_field_transcript_squeeze_scalar(c: &mut Criterion) {
    for num_elems in [1, 10, 100, 1000, 10000] {
        c.bench_function(
            &format!("bn254_base_field_transcript_squeeze_scalar_{num_elems}"),
            |b| {
                b.iter_batched(
                    || {
                        ECTranscriptWriter::<Bn256Point, PoseidonSponge<Bn256Base>>::new(
                            "Test transcript",
                        )
                    },
                    |mut transcript_writer| {
                        <ECTranscriptWriter<
                            Bn256Point,
                            PoseidonSponge<Bn256Base>,
                        > as ECProverTranscript<Bn256Point>>::get_scalar_field_challenges(
                            &mut transcript_writer,
                            "benchmark",
                            num_elems,
                        )
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }
}

criterion_group!(
    benches,
    bench_bn254_scalar_field_transcript_absorb,
    bench_bn254_scalar_field_transcript_squeeze,
    bench_bn254_base_field_transcript_absorb_scalar,
    bench_bn254_base_field_transcript_squeeze_scalar
);
criterion_main!(benches);
