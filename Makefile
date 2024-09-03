.PHONY: all bench prod test clean

all: bench prod

# Example: make bench name=hyrax.opt
bench:
	RUSTFLAGS=-Awarnings cargo build --profile=opt-with-debug --bin worldcoin &&\
		cp target/opt-with-debug/worldcoin ./worldcoin_bench &&\
		valgrind --tool=massif --massif-out-file=massif.$(name).out ./worldcoin_bench &&\
		ms_print massif.$(name).out | less

prod:
	cargo build --release --features "parallel" --bin worldcoin
	cp target/release/worldcoin ./worldcoin_prod

test:
	cargo test --release --features parallel -- --test-threads=1

clean:
	cargo clean
	$(RM) ./worldcoin ./worldcoin_bench
