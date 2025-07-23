.PHONY: all pr check test-dev test-ignored test mem-lim-cgroups mem-lim-docker prod prod-seq bin bin-seq mem-profile-prover mobile clean help

# Memory limit in MB.
MEM_LIM=110

all: help

pr:  ## Prepare for a PR; run all GitHub CI Actions.
	$(MAKE) check
	$(MAKE) test
	$(MAKE) mem-lim-docker

check:  ## GitHub Action #1 - compile, run formatter and linter.
	@if find remainder_*/src -name mod.rs | grep -q .; then \
		echo; \
	 	echo "Found the following 'mod.rs' files in a src directory:"; \
		find remainder_*/src -name mod.rs; \
		echo; \
		echo "Please follow the '[module_name].rs' instead of '[module_name]/mod.rs' file naming convention."; \
		echo;  \
		exit 1; \
	fi
	cargo check
	cargo check --features parallel
	cargo fmt --all -- --check
	cargo clippy --no-deps -- -D warnings

test-dev: test-dev-seq test-dev-par  ## GitHub Action #2a - Basic unit testing. With and without parallel features.

test-dev-seq:
	cargo test --profile=dev-opt

test-dev-par:
	cargo test --profile=dev-opt --features parallel

test-ignored:  ## GitHub Action #2b - Run some slow tests that are normally ignored.
	cargo test --profile=dev-opt --features parallel --package remainder-hyrax --lib -- --ignored hyrax_worldcoin::test_worldcoin
	cargo test --profile=dev-opt --features parallel --package remainder --lib -- --ignored worldcoin::tests

test: test-dev test-ignored  ## Comprehensive testing. Equivalent to `test-dev` followed by `test-ignored`.

mem-lim-cgroups:  ## GitHub Action #3 - Run sequential world prover with a memory limit. Only available on Linux!
	$(MAKE) prod-seq
	echo ${MEM_LIM}M | sudo tee /sys/fs/cgroup/makefile_memory_limited_group/memory.max
	echo 0 | sudo tee /sys/fs/cgroup/makefile_memory_limited_group/memory.swap.max
	sudo cgexec -g memory:makefile_memory_limited_group ./target/release/world_prove --circuit world.circuit --input iriscode_pcp_example --output-dir world_zkp

mem-lim-docker: Dockerfile  ## Run sequential World prover inside a Docker container with a memory limit.
	docker build -t remainder-mem .
	docker run --rm --memory=${MEM_LIM}m --memory-swap=${MEM_LIM}m remainder-mem:latest
	echo "World V3 + MPC prover runs in under ${MEM_LIM} MB!"

circuit: prod  ## Generate iris code upgrade circuit (V3 + MPC).
	./target/release/world_gen_iriscode_secret_share_circuit_descriptions --circuit world.circuit

prod:  ## Build World binaries for production; optimizations + rayon parallelism, NO print-trace.
	cargo build --release --features parallel --bin world_gen_iriscode_secret_share_circuit_descriptions
	cargo build --release --features parallel --bin world_prove
	cargo build --release --features parallel --bin world_upgrade_verify
	cargo build --release --features parallel --bin world_verify_ampc_party

prod-seq:  ## Similar to 'prod', but NO rayon parallelism.
	cargo build --release --bin world_gen_iriscode_secret_share_circuit_descriptions
	cargo build --release --bin world_prove
	cargo build --release --bin world_upgrade_verify
	cargo build --release --bin world_verify_ampc_party

bin:  ## Build the binaries for efficient debugging; optimizations + rayon parallelism + print-trace.
	cargo build --release --features "parallel, print-trace" --bin world_gen_iriscode_secret_share_circuit_descriptions
	cargo build --release --features "parallel, print-trace" --bin world_prove
	cargo build --release --features "parallel, print-trace" --bin world_upgrade_verify
	cargo build --release --features "parallel, print-trace" --bin world_verify_ampc_party

bin-seq:  ## Similar to `make bin`, but NO rayon parallelism.
	cargo build --release --features print-trace --bin world_gen_iriscode_secret_share_circuit_descriptions
	cargo build --release --features print-trace --bin world_prove
	cargo build --release --features print-trace --bin world_upgrade_verify
	cargo build --release --features print-trace --bin world_verify_ampc_party

mem-profile-prover:  ## Use Valgrind to profile memory usage of the World prover. E.g. `make mem-profile-prover name=v2.0`. Only available on Linux!
	cargo build --profile=mem-bench --bin world_prove
	mkdir -p massif
	valgrind --tool=massif --massif-out-file=massif/massif.$(name).out --pages-as-heap=yes ./target/mem-bench/world_prove --circuit world.circuit --input iriscode_pcp_example --output-dir world_zkp
	ms_print massif/massif.$(name).out | less

mobile:  ## DEPRECATED - Compile World prover optimized for binary size.
	cargo build --profile mobile --bin world_prove

clean:  ## Equivalent to "cargo clean"
	cargo clean

# Got the idea from https://stackoverflow.com/a/47107132.
help:  ## Show this help message.
	@sed -ne '/@sed/!s/:.*## /:/p' $(MAKEFILE_LIST) | column -t -s':'

