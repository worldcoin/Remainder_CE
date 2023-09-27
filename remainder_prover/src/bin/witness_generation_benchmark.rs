use remainder::zkdt::data_pipeline::dt2zkdt::*;
use halo2_base::halo2_proofs::halo2curves::bn256::Fr;
use std::time::Instant;

const TREES_FN: &str = "src/zkdt/data_pipeline/upshot_data/quantized-upshot-model.json";
const SAMPLES_FN: &str = "src/zkdt/data_pipeline/upshot_data/upshot-quantized-samples-10k.npy";

fn main() {
    println!("Starting tree loading ...");
    let start_time = Instant::now();
    let raw_trees_model: RawTreesModel = load_raw_trees_model(TREES_FN);
    println!("Loading trees from JSON took: {:?}", start_time.elapsed());
    
    println!("Starting sample loading ...");
    let start_time = Instant::now();
    let mut raw_samples: RawSamples = load_raw_samples(SAMPLES_FN);
    println!("Loading samples took: {:?}", start_time.elapsed());

    // use just a small batch of samples
    raw_samples = raw_samples.slice(0, 1000);
    
    println!("Starting tree witness generation ...");
    let start_time = Instant::now();
    let trees_model: TreesModel = (&raw_trees_model).into();
    let _ctrees: CircuitizedTrees<Fr> = (&trees_model).into();
    println!("Trees witness generation ({}) took: {:?}", raw_trees_model, start_time.elapsed());

    println!("Starting samples witness generation ...");
    let start_time = Instant::now();
    let samples = to_samples(&raw_samples, &trees_model);
    let _csamples = circuitize_samples::<Fr>(&samples, &trees_model);
    println!("Samples witness generation ({}) took: {:?}", raw_samples, start_time.elapsed());
}
