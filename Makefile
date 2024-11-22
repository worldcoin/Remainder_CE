.PHONY: all bench prod prod-seq bin bin-seq test test-dev mobile clean

all: bin

# Example: make bench name=hyrax.opt
bench:
	cargo build --profile=opt-with-debug --bin worldcoin
	valgrind --tool=massif --massif-out-file=massif/massif.$(name).out --pages-as-heap=yes ./target/opt-with-debug/worldcoin prove worldcoin.circuit iriscode_pcp_example worldcoin.zkp
	ms_print massif/massif.$(name).out | less

prod:
	cargo build --release --features parallel --bin worldcoin

prod-seq:
	cargo build --release --bin worldcoin

bin:
	cargo build --bin worldcoin --release --features "parallel, print-trace"

bin-seq:
	cargo build --bin worldcoin --release --features "parallel, print-trace"

test: test-dev
	cargo test --release --features parallel --package remainder-hyrax --lib -- --ignored hyrax_worldcoin::test_worldcoin
	cargo test --release --features parallel --package remainder --lib -- --ignored worldcoin::tests

test-dev:
	cargo test --release
	cargo test --release --features parallel

mobile:
	cargo build --profile mobile --bin worldcoin

clean:
	cargo clean
