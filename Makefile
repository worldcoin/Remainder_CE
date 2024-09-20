.PHONY: all bench prod prod-seq mobile test clean

all: prod

# Example: make bench name=hyrax.opt
bench:
	cargo build --profile=opt-with-debug --bin worldcoin &&\
		valgrind --tool=massif --massif-out-file=massif/massif.$(name).out --pages-as-heap=yes ./target/opt-with-debug/worldcoin &&\
		ms_print massif/massif.$(name).out | less

prod:
	cargo build --release --features "parallel" --bin worldcoin

prod-seq:
	cargo build --release --bin worldcoin

# Currently using release for faster running.
test:
	cargo test --release --features parallel -- --test-threads=1

mobile:
	cargo build --profile mobile --bin worldcoin

clean:
	cargo clean
