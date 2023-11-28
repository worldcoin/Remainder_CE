for i in {0..7}
do
    scp -i ~/modulus/rocky_backend_key.pem ~/modulus/remainder/zkdt_proof_tree_${i}_benchmark.json ubuntu@ec2-35-86-196-95.us-west-2.compute.amazonaws.com:/home/ubuntu/halo2-gkr/pipeline_data
done