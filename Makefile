.PHONY: all bench prod test clean

all: bench prod

# Example: make bench name=hyrax.opt
bench:
	RUSTFLAGS=-Awarnings cargo build --profile=opt-with-debug --bin worldcoin &&\
		valgrind --tool=massif --massif-out-file=massif/massif.$(name).out ./target/opt-with-debug/worldcoin &&\
		ms_print massif/massif.$(name).out | less

prod:
	cargo build --release --features "parallel" --bin worldcoin
	cp target/release/worldcoin ./worldcoin_prod

test:
	cargo test --release --features parallel -- --test-threads=1

clean:
	cargo clean
	$(RM) ./worldcoin ./worldcoin_bench
