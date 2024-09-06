.PHONY: all bench prod mobile test clean

all: bench prod mobile

# Example: make bench name=hyrax.opt
bench:
	RUSTFLAGS=-Awarnings cargo build --profile=opt-with-debug --bin worldcoin &&\
		valgrind --tool=massif --massif-out-file=massif.$(name).out ./target/opt-with-debug/worldcoin &&\
		ms_print massif.$(name).out | less

prod:
	RUSTFLAGS=-Awarnings cargo build --release --features "parallel" --bin worldcoin

test:
	RUSTFLAGS=-Awarnings cargo test --release --features parallel -- --test-threads=1

mobile:
	RUSTFLAGS=-Awarnings cargo build --profile mobile --bin worldcoin

clean:
	cargo clean
