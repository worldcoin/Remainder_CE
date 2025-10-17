//! ============== INSTRUCTIONS ==============
//! Each benchmark is identified by a BenchmarkID string which is documented in
//! the docstring comment of each benchmark function in this file.
//!
//! To run all the benchmarks in this file which match a regular expression
//! REGEX, use the following command:
//!
//!  > cargo bench --bench fix_variable -- 'REGEX'
//!
//! Examples:
//! * To benchmark different implementations of `fix_variable_at_index` for MLEs
//!   of size 20, edit the code in `src/mle/evals.rs` by uncommenting the
//!   desired implementation and run the command bellow. Each time the command
//!   is run, criterion will report the percentage improvement/regression
//!   compared to the previously run benchmark:
//!  > cargo bench --bench fix_variable -- '^fix_var_at_index_20/first$'
//!
//! * To compare `fix_variable` with `fix_variable_at_index` for MLEs of size
//!   2^20, we need to choose the benchmarks with IDs "fix_var_20" and
//!   "fix_var_at_index_20/first". We can pick those using the regular
//!   expression in the following command:
//!  > cargo bench --bench fix_variable -- '^fix_var(_at_index)?_20(/first)?$'
//!
//! * To benchmark `fix_variable_at_index` for MLE of size 2^20 and compare the
//!   time to fix the first, middle and last element run:
//!
//!  > cargo bench --bench fix_variable -- '^fix_var_at_index_20'
//!
//! * ONLY ON PIKNIKS: To benchmark the effects of Rayon parallelism on
//!   `fix_variable` on MLEs of size 2^15, we may want to run benchmarks with
//!   varying number of threads available. The following command runs benchmarks
//!   for 1, 2, 4, 8, 16, 32, 64 and 128 threads:
//!
//!  > cargo bench --bench fix_variable -- `^rayon_fix_var_vars_15`

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use rayon::ThreadPoolBuilder;
use remainder::mle::evals::{Evaluations, MultilinearExtension};
use remainder_shared_types::Fr;

// =========== Helper functions ===========

/// Return an MLE on `num_vars` variables on evaluations:
/// `[0, 1, ..., 2^num_vars - 1]`.
fn create_dummy_mle(num_vars: usize) -> MultilinearExtension<Fr> {
    // Note: It is conceivable that using the same field element repeatedly in a
    // bookkeeping table, and especially a small field element whose higher
    // order bits are all zero, might introduce some opportunities for (1)
    // compiler-level optimizations (2) CPU-level optimizations if the Field
    // crate is handling zero bits differently. To address the first point, we
    // wrap the elements around `black_box()` to disable the compiler optimizer
    // on them. As for the second point, we performed benchmarks using random
    // field elements and we didn't notice any statistically significant
    // regression in performance for `halo2curves::bn256::Fr`.
    let evals: Vec<Fr> = (0..(1 << num_vars)).map(Fr::from).collect();

    let f = Evaluations::new(num_vars, evals);

    MultilinearExtension::new_from_evals(f)
}

/// Run `fix_variable` on `mle` instructing Rayon to use up to `num_threads`
/// threads.
fn fix_variable_parallel(num_threads: usize, mut mle: MultilinearExtension<Fr>) {
    let pool = ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .unwrap();

    pool.install(|| {
        mle.fix_variable(Fr::one());
    });
}

/// Run `fix_variable_at_index` on index 0 of `mle` instructing Rayon to use up
/// to `num_threads` threads.
fn fix_variable_at_index_parallel(num_threads: usize, mut mle: MultilinearExtension<Fr>) {
    let pool = ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .unwrap();

    pool.install(|| {
        mle.fix_variable_at_index(0, Fr::one());
    });
}

// ========================================

/// BenchmarksIDs:
///  * `fix_var_{10, 15, 20, 23, 25}`
///
/// Benchmark `MultilinearExtension::fix_variable` on MLEs of various sizes
/// using the default Rayon configuration.
fn bench_fix_variable(c: &mut Criterion) {
    for num_vars in [10, 15, 20, 23, 25] {
        let f_tilde = create_dummy_mle(num_vars);

        c.bench_function(&format!("fix_var_{num_vars}"), |b| {
            b.iter_batched(
                || f_tilde.clone(),
                |mut mle| mle.fix_variable(black_box(Fr::one())),
                BatchSize::SmallInput,
            )
        });
    }
}

/// BenchmarksIDs:
///  * `fix_var_at_index_{10, 15, 20, 23, 25}/{first, middle, last}`
///
/// Benchmark `MultilinearExtension::fix_variable_at_index` on MLEs of various
/// sizes using the default Rayon configuration.
/// The index to be fixed is the first, middle and last respectively.
fn bench_fix_variable_at_index(c: &mut Criterion) {
    for num_vars in [10, 15, 20, 23, 25] {
        let f_tilde = create_dummy_mle(num_vars);

        let mut group = c.benchmark_group(format!("fix_var_at_index_{num_vars}"));

        // Fix first variable.
        group.bench_function("first".to_string(), |b| {
            b.iter_batched(
                || f_tilde.clone(),
                |mut mle| mle.fix_variable_at_index(0, black_box(Fr::one())),
                BatchSize::SmallInput,
            )
        });

        // Fix middle variable.
        group.bench_function("middle".to_string(), |b| {
            b.iter_batched(
                || f_tilde.clone(),
                |mut mle| mle.fix_variable_at_index(num_vars / 2, black_box(Fr::one())),
                BatchSize::SmallInput,
            )
        });

        // Fix last variable.
        group.bench_function("last".to_string(), |b| {
            b.iter_batched(
                || f_tilde.clone(),
                |mut mle| mle.fix_variable_at_index(num_vars - 1, black_box(Fr::one())),
                BatchSize::SmallInput,
            )
        });
    }
}

/// BenchmarksIDs:
///  * `rayon_fix_var_vars_{10, 15, 20, 23, 25}_threads/{1, 2, 4, 8, 16, 32, 64, 128}`
///  * `rayon_fix_var_default/{1, 2, 4, 8, 16, 32, 64, 128}`
///
/// Benchmarks for evaluating the effects of Rayon parallelism.
/// Runs `fix_variable` on various MLE sizes and Rayon configurations
/// (i.e. number of threads available).
fn bench_fix_variable_parallelism(c: &mut Criterion) {
    // ---- VARY both `num_threads` and `num_vars` ----
    for num_vars in [10, 15, 20, 23, 25] {
        let f_tilde = create_dummy_mle(num_vars);

        let mut group = c.benchmark_group(format!("rayon_fix_var_vars_{num_vars}"));

        // For machines with less than 128 cores, modify the following line
        // accordingly:
        let num_threads: Vec<usize> = (0..=7).map(|i: usize| 1 << i).collect();

        for n in num_threads {
            group.bench_with_input(
                BenchmarkId::new(format!("rayon_fix_var_vars_{num_vars}_threads"), n),
                &n,
                |b, &n| {
                    b.iter_batched(
                        || f_tilde.clone(),
                        |mle| fix_variable_parallel(black_box(n), black_box(mle)),
                        BatchSize::SmallInput,
                    )
                },
            );
        }
    }

    // ---- DEFAULT `num_threads`, VARY num_vars ----
    // Here we set `num_threads` to zero which instructs Rayon to use a default
    // value.
    let num_threads = 0;
    let mut group = c.benchmark_group("rayon_fix_var_default");
    for num_vars in [10, 15, 20, 23] {
        group.bench_with_input(
            BenchmarkId::new("rayon_fix_var_default".to_string(), num_vars),
            &num_threads,
            |b, &num_threads| {
                b.iter_batched(
                    || create_dummy_mle(num_vars),
                    |mle| fix_variable_parallel(black_box(num_threads), black_box(mle)),
                    BatchSize::SmallInput,
                )
            },
        );
    }
}

/// BenchmarksIDs:
///  * `rayon_fix_var_at_index_vars_{10, 15, 20, 23, 25}_threads/{1, 2, 4, 8, 16, 32, 64, 128}`
///  * `rayon_fix_var_at_index_default/{1, 2, 4, 8, 16, 32, 64, 128}`
///
/// Benchmarks for evaluating the effects of Rayon parallelism.
/// Runs `fix_variable_at_index` on various MLE sizes and Rayon configurations
/// (i.e. number of threads available).
fn bench_fix_variable_at_index_parallelism(c: &mut Criterion) {
    // ---- VARY both `num_threads` and `num_vars` ----
    for num_vars in [10, 15, 20, 23, 25] {
        let f_tilde = create_dummy_mle(num_vars);

        let mut group = c.benchmark_group(format!("rayon_fix_var_at_index_vars_{num_vars}"));

        // For machines with less than 128 cores, modify the following line
        // accordingly:
        let num_threads: Vec<usize> = (0..=7).map(|i: usize| 1 << i).collect();

        for num_threads in num_threads {
            group.bench_with_input(
                BenchmarkId::new(
                    format!("rayon_fix_var_at_index_vars_{num_vars}_threads"),
                    num_threads,
                ),
                &num_threads,
                |b, &num_threads| {
                    b.iter_batched(
                        || f_tilde.clone(),
                        |mle| fix_variable_at_index_parallel(num_threads, mle),
                        BatchSize::SmallInput,
                    )
                },
            );
        }
    }

    // ---- DEFAULT `num_threads`, VARY num_vars ----
    // Here we set `num_threads` to zero which instructs Rayon to use a default
    // value.
    let num_threads = 0;
    let mut group = c.benchmark_group("rayon_fix_var_at_index_default");
    for num_vars in [10, 15, 20, 23] {
        group.bench_with_input(
            BenchmarkId::new("rayon_fix_var_at_index_default".to_string(), num_vars),
            &num_threads,
            |b, &num_threads| {
                b.iter_batched(
                    || create_dummy_mle(num_vars),
                    |mle| fix_variable_at_index_parallel(num_threads, mle),
                    BatchSize::SmallInput,
                )
            },
        );
    }
}

criterion_group!(
    benches,
    bench_fix_variable,
    bench_fix_variable_at_index,
    bench_fix_variable_parallelism,
    bench_fix_variable_at_index_parallelism
);
criterion_main!(benches);
