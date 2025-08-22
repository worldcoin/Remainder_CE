use ark_std::test_rng;
use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use itertools::Itertools;
use rand::Rng;
use remainder::{
    claims::RawClaim,
    expression::generic_expr::Expression,
    layer::{regular_layer::RegularLayer, Layer, LayerId},
    mle::{betavalues::BetaValues, dense::DenseMle},
    sumcheck::SumcheckEvals,
};
use remainder_shared_types::{
    transcript::{poseidon_sponge::PoseidonSponge, TranscriptWriter},
    Fr,
};

// =========== Helper functions ===========

/// Return a `DenseMle` on `num_vars` variables on evaluations:
/// `[1, 1, ..., 1]`.
fn create_dummy_mle(num_vars: usize) -> DenseMle<Fr> {
    // Note: It is conceivable that using the same field element repeatedly in a
    // bookkeeping table, and especially a small field element whose higher
    // order bits are all zero, might introduce some opportunities for (1)
    // compiler-level optimizations (2) CPU-level optimizations if the Field
    // crate is handling zero bits differently. To address the first point, we
    // wrap the elements around `black_box()` to disable the compiler optimizer
    // on them. As for the second point, we performed benchmarks using random
    // field elements and we didn't notice any statistically significant
    // regression in performance for `halo2curves::bn256::Fr`.
    let evals: Vec<Fr> = (0..(1 << num_vars)).map(|_| black_box(Fr::one())).collect();

    // Uncomment the following instead for random evaluations:
    // let evals: Vec<Fr> = (0..(1 << num_vars)).map(|_|
    // Fr::random(OsRng)).collect();

    DenseMle::<Fr>::new_from_raw(evals, LayerId::Layer(0))
}

/// Evaluates (at a random field point) the Multilinear Extension of the boolean
/// function defined by `expression`.
/// It's worth noting that this function is *not* equivalent to
/// `expression.evaluate_expr()` which evaluates the (possibly non-linear)
/// function defined by `expression`. For example, if `expression` is a product,
/// this function computes the MLE of the product, as opposed to the product of
/// MLEs that is returned by `expression.evaluate_expr()`.
fn get_dummy_expression_eval(
    expression: &Expression<Fr, DenseMle<Fr>>,
    rng: &mut impl Rng,
) -> RawClaim<Fr> {
    let mut expression = expression.clone();
    let mut bind_list = Vec::new();
    let num_vars = expression.index_mle_indices(0, &mut bind_list);

    let expression_nonlinear_indices = expression.get_all_nonlinear_rounds();
    let expression_linear_indices = expression.get_all_linear_rounds();

    let challenges = (0..num_vars)
        .map(|_| (Fr::from(rng.gen::<u64>())))
        .collect_vec();
    let challenges_enumerate = expression_nonlinear_indices
        .iter()
        .map(|index| (*index, challenges[*index]))
        .collect_vec();
    expression_linear_indices.iter().for_each(|round| {
        expression.fix_variable_at_index(*round, challenges[*round], &mut bind_list)
    });

    let beta = BetaValues::new(challenges_enumerate);
    let eval = expression.evaluate_sumcheck_beta_cascade(
        &[&beta],
        &[Fr::one()],
        expression_linear_indices.len(),
        2,
        &bind_list,
    );
    let SumcheckEvals(evals) = eval;

    let result = if !expression_nonlinear_indices.is_empty() {
        debug_assert!(evals.len() > 1);
        evals[0] + evals[1]
    } else {
        debug_assert!(evals.len() == 1 || evals[0] == evals[1]);
        evals[0]
    };

    RawClaim::new(challenges, result)
}

/// A collection of custom [Expression] structures to benchmark proving
/// [RegularLayer] over.
#[allow(dead_code)]
#[derive(Debug)]
enum ExpressionStructure {
    /// Simple, linear expression summing two MLEs of the same size:
    /// ```ignore
    ///      +
    ///  /      \
    /// MLE    MLE
    /// ```
    Sum,
    /// Product of a configurable number of MLEs of the same size:
    /// ```ignore
    ///      *
    ///   /  |   \
    /// MLE ... MLE
    /// ```
    Product,
    /// A balanced tree consisting of two levels of selectors and a level of
    /// products of MLEs of uniform size and degree:
    /// ```ignore
    ///                         Sel
    ///                    /            \
    ///                 /                  \
    ///              /                       \
    ///            Sel                       Sel
    ///         /       \                  /      \
    ///       /           \             /           \
    ///      *            *            *            *
    ///   /  |   \    /   |   \     /  |   \    /   |   \
    /// MLE ... MLE  MLE ... MLE  MLE ... MLE  MLE ... MLE
    /// ```
    BalancedTree,
    /// A left-leaning tree consisting of a chain of selectors and a level of
    /// products of MLEs. The field [BenchLayerConfig::degree] controls the
    /// degree of the product node and the field [BenchLayerConfig::mle_size]
    /// controls the size of the MLE *closer* to the root. The diagram depits
    /// the sizes of MLEs where `n == mle_size`:
    /// ```ignore
    ///                              Sel
    ///                         /            \
    ///                      /                  \
    ///                   /                       \
    ///                 Sel                      MLE(n)
    ///              /       \        
    ///            /           \      
    ///           *           MLE(n-1)
    ///   /       |      \  
    /// MLE(n-2)   ...  MLE(n-2)
    /// ```
    UnbalancedTree,
}

/// Configures the [Expression] used to define the [RegularLayer] to be
/// benchmarked.
struct BenchLayerConfig {
    /// Can take a value among a set of custom [Expression] structures.
    expression_structure: ExpressionStructure,
    /// The degree of every product node in the expression tree.
    degree: usize,
    /// The number of variables of each MLE on the leaves of the expression
    /// tree.
    mle_size: usize,
}

/// Generates a [RegularLayer] according to `config` along with a [Claim] at a
/// random point.
fn create_dummy_regular_layer(config: BenchLayerConfig) -> (RegularLayer<Fr>, RawClaim<Fr>) {
    let mut rng = test_rng();

    let leaf_mle = create_dummy_mle(config.mle_size);

    match config.expression_structure {
        ExpressionStructure::Sum => {
            let leaf_expression = Expression::<Fr, DenseMle<Fr>>::mle(leaf_mle);
            let layer_expression = leaf_expression.clone() + leaf_expression.clone();

            let claim = get_dummy_expression_eval(&layer_expression, &mut rng);
            let layer = RegularLayer::new_raw(LayerId::Layer(0), layer_expression, Vec::new());

            (layer, claim)
        }
        ExpressionStructure::Product => {
            let layer_expression =
                Expression::<Fr, DenseMle<Fr>>::products(vec![leaf_mle; config.degree]);

            let claim = get_dummy_expression_eval(&layer_expression, &mut rng);
            let layer = RegularLayer::new_raw(LayerId::Layer(0), layer_expression, Vec::new());

            (layer, claim)
        }
        ExpressionStructure::BalancedTree => {
            let prod_expression =
                Expression::<Fr, DenseMle<Fr>>::products(vec![leaf_mle.clone(); config.degree]);
            let prod_expression_copy = prod_expression.clone();
            let level1_expression =
                Expression::<Fr, DenseMle<Fr>>::select(prod_expression, prod_expression_copy);
            let level1_expression_copy = level1_expression.clone();
            let level0_expression =
                Expression::<Fr, DenseMle<Fr>>::select(level1_expression, level1_expression_copy);

            let claim = get_dummy_expression_eval(&level0_expression, &mut rng);
            let layer = RegularLayer::new_raw(LayerId::Layer(0), level0_expression, Vec::new());

            (layer, claim)
        }
        ExpressionStructure::UnbalancedTree => {
            // We need a different leaf MLE in this case.
            assert!(config.mle_size >= 2);
            let mle_n_2 = create_dummy_mle(config.mle_size - 2);

            let mle_n_1 = create_dummy_mle(config.mle_size - 1);
            let mle_n_1_expression = Expression::<Fr, DenseMle<Fr>>::mle(mle_n_1);

            let mle_n = create_dummy_mle(config.mle_size);
            let mle_n_expression = Expression::<Fr, DenseMle<Fr>>::mle(mle_n);

            let prod_expression =
                Expression::<Fr, DenseMle<Fr>>::products(vec![mle_n_2; config.degree]);

            let level1_expression =
                Expression::<Fr, DenseMle<Fr>>::select(prod_expression, mle_n_1_expression);
            let level0_expression =
                Expression::<Fr, DenseMle<Fr>>::select(level1_expression, mle_n_expression);

            let claim = get_dummy_expression_eval(&level0_expression, &mut rng);
            let layer = RegularLayer::new_raw(LayerId::Layer(0), level0_expression, Vec::new());

            (layer, claim)
        }
    }
}

// ========================================

/// BenchmarksIDs:
///  * `regular_layer/product_four/{10, 15, 20}`
///  * `regular_layer/product_eight/{10, 15, 20}`
///  * `regular_layer/balanced_two/{10, 15, 20}`
///  * `regular_layer/unbalanced_two/{10, 15, 20}`
///
/// Benchmark [RegularLayer::prove_rounds] on layers of different structure and
/// sizes.
fn bench_regular_layer(c: &mut Criterion) {
    let mut group = c.benchmark_group("regular_layer".to_string());

    for mle_size in [10, 15, 20] {
        let config = BenchLayerConfig {
            expression_structure: ExpressionStructure::Product,
            degree: 4,
            mle_size,
        };
        let (layer, claim) = create_dummy_regular_layer(config);

        group.bench_with_input(
            BenchmarkId::new("product_four".to_string(), mle_size),
            &mle_size,
            |b, _mle_size| {
                b.iter_batched(
                    || {
                        (
                            layer.clone(),
                            claim.clone(),
                            TranscriptWriter::<Fr, PoseidonSponge<Fr>>::new(
                                "Benchmark transcript writer",
                            ),
                        )
                    },
                    |(mut layer, claim, mut transcript)| layer.prove(&[&claim], &mut transcript),
                    BatchSize::SmallInput,
                )
            },
        );

        let config = BenchLayerConfig {
            expression_structure: ExpressionStructure::Product,
            degree: 8,
            mle_size,
        };
        let (layer, claim) = create_dummy_regular_layer(config);

        group.bench_with_input(
            BenchmarkId::new("product_eight".to_string(), mle_size),
            &mle_size,
            |b, _mle_size| {
                b.iter_batched(
                    || {
                        (
                            layer.clone(),
                            claim.clone(),
                            TranscriptWriter::<Fr, PoseidonSponge<Fr>>::new(
                                "Benchmark transcript writer",
                            ),
                        )
                    },
                    |(mut layer, claim, mut transcript)| layer.prove(&[&claim], &mut transcript),
                    BatchSize::SmallInput,
                )
            },
        );

        let config = BenchLayerConfig {
            expression_structure: ExpressionStructure::BalancedTree,
            degree: 2,
            mle_size,
        };
        let (layer, claim) = create_dummy_regular_layer(config);

        group.bench_with_input(
            BenchmarkId::new("balanced_two".to_string(), mle_size),
            &mle_size,
            |b, _mle_size| {
                b.iter_batched(
                    || {
                        (
                            layer.clone(),
                            claim.clone(),
                            TranscriptWriter::<Fr, PoseidonSponge<Fr>>::new(
                                "Benchmark transcript writer",
                            ),
                        )
                    },
                    |(mut layer, claim, mut transcript)| layer.prove(&[&claim], &mut transcript),
                    BatchSize::SmallInput,
                )
            },
        );

        let config = BenchLayerConfig {
            expression_structure: ExpressionStructure::UnbalancedTree,
            degree: 2,
            mle_size,
        };
        let (layer, claim) = create_dummy_regular_layer(config);

        group.bench_with_input(
            BenchmarkId::new("unbalanced_two".to_string(), mle_size),
            &mle_size,
            |b, _mle_size| {
                b.iter_batched(
                    || {
                        (
                            layer.clone(),
                            claim.clone(),
                            TranscriptWriter::<Fr, PoseidonSponge<Fr>>::new(
                                "Benchmark transcript writer",
                            ),
                        )
                    },
                    |(mut layer, claim, mut transcript)| layer.prove(&[&claim], &mut transcript),
                    BatchSize::SmallInput,
                )
            },
        );
    }
}

criterion_group!(benches, bench_regular_layer);
criterion_main!(benches);
