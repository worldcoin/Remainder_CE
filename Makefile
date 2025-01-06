.PHONY: all pr check test mem-lim test-dev prod prod-seq bin bin-seq bench bench-mpc mobile clean help

all: help

pr:  ## Prepare for a PR; run all GitHub CI Actions.
	$(MAKE) check
	$(MAKE) test
	$(MAKE) mem-lim

check:  ## GitHub Action #1 - compile, run formatter and linter.
	cargo check
	cargo check --features parallel
	cargo fmt --all -- --check
	cargo clippy --no-deps -- -D warnings

test:  ## GitHub Action #2 - Slow but Comprehensive testing.
	cargo test
	cargo test --features parallel
	cargo test --release --features parallel --package remainder-hyrax --lib -- --ignored hyrax_worldcoin::test_worldcoin
	cargo test --release --features parallel --package remainder --lib -- --ignored worldcoin::tests

mem-lim:  ## GitHub Action #3 - run sequential worldcoin binary with a memory limit.
	$(MAKE) prod-seq
	echo 350M | sudo tee /sys/fs/cgroup/makefile_memory_limited_group/memory.max
	echo 0 | sudo tee /sys/fs/cgroup/makefile_memory_limited_group/memory.swap.max
	sudo cgexec -g memory:makefile_memory_limited_group ./target/release/worldcoin_mpc prove worldcoin_mpc.circuit iriscode_pcp_example worldcoin_mpc.zkp

test-dev:  ## Faster alternative to "make test"; uses `--release` flag and ignores slow tests.
	cargo test --release
	cargo test --release --features parallel

prod:  ## Build worldcoin binary for production; optimizations + rayon parallelism, NO print-trace.
	cargo build --release --features parallel --bin worldcoin_mpc

prod-seq:  ## Similar to 'prod', but NO rayon parallelism.
	cargo build --release --bin worldcoin_mpc

prod-no-mpc:  ## Build deprecated worldcoin binary (no MPC circuit) for production; optimizations + rayon parallelism, NO print-trace.
	cargo build --release --features parallel --bin worldcoin

bin:  ## Build the binaries for efficient debugging; optimizations + rayon parallelism + print-trace.
	cargo build --bins --release --features "parallel, print-trace"

bin-seq:  ## Similar to "make bin", but NO rayon parallelism.
	cargo build --bins --release --features "parallel, print-trace"

bench:  ## Use Valgrind to profile memory usage, for worldcoin ic circuit. Example - make bench name=v2.0
	cargo build --profile=opt-with-debug --bin worldcoin_mpc
	mkdir -p massif
	valgrind --tool=massif --massif-out-file=massif/massif.$(name).out --pages-as-heap=yes ./target/opt-with-debug/worldcoin_mpc prove worldcoin_mpc.circuit iriscode_pcp_example worldcoin_mpc.zkp
	ms_print massif/massif.$(name).out | less

bench-no-mpc:  ## Use Valgrind to profile memory usage, for worldcoin mpc circuit. Example - make bench-mpc
	cargo build --profile=opt-with-debug --bin worldcoin
	mkdir -p massif
	valgrind --tool=massif --massif-out-file=massif/massif.out --pages-as-heap=yes ./target/opt-with-debug/worldcoin prove worldcoin.circuit iriscode_pcp_example worldcoin.zkp
	ms_print massif/massif.out | less

bench-single:  ## Use Valgrind to profile memory usage of the proving and verifying of a single iriscode circuit (i.e. just one eye, just iris).
	cargo build --profile=opt-with-debug --bin run_iriscode_circuit
	mkdir -p massif
	valgrind --tool=massif --massif-out-file=massif/massif.$(name).out --pages-as-heap=yes ./target/opt-with-debug/run_iriscode_circuit --image-filepath remainder_prover/src/worldcoin/constants/v3-split-images/iris/test_image.bin
	ms_print massif/massif.$(name).out | less

mobile:  ## Compile worldcoin binary optimized for binary size.
	cargo build --profile mobile --bin worldcoin

clean:  ## Equivalent to "cargo clean"
	cargo clean

# Got the idea from https://stackoverflow.com/a/47107132.
help:  ## Show this help message.
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST) | column -t -s':'

