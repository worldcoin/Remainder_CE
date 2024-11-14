.PHONY: all bench prod prod-seq mobile test test-dev clean

all: prod

# Example: make bench name=hyrax.opt
bench:
	cargo build --profile=opt-with-debug --bin worldcoin
	valgrind --tool=massif --massif-out-file=massif/massif.$(name).out --pages-as-heap=yes ./target/opt-with-debug/worldcoin
	ms_print massif/massif.$(name).out | less

prod:
	cargo build --release --features "parallel" --bin worldcoin

prod-seq:
	cargo build --release --bin worldcoin

test: test-dev
	cargo test --release --features parallel  --package remainder-hyrax --lib -- --ignored hyrax_worldcoin::test_worldcoin --test-threads=1
	cargo test --release --features parallel  --package remainder --lib -- --ignored worldcoin::tests --test-threads=1

test-dev:
	cargo test --release --features parallel -- --test-threads=1

mobile:
	cargo build --profile mobile --bin worldcoin

clean:
	cargo clean
