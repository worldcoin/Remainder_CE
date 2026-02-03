# _Remainder_: <ins>Re</ins>asonable <ins>Ma</ins>chine learn<ins>in</ins>g <ins>D</ins>oubly-<ins>E</ins>fficient prove<ins>r</ins>

## Overview
_Remainder_ is an open-source "GKR-ish" implementation, with a custom circuit creation frontend, prover, and verifier. _Remainder_ is fully written in Rust and only has CPU bindings, and seamlessly bundles together a basket of techniques, including
* Structured layers and linear-time sumcheck prover from [Tha13](https://ia.cr/2013/351)
* Time-optimal sumcheck for matrix multiplication from [Tha13](https://ia.cr/2013/351)
* Linear-time dataparallel $\widetilde{\text{add}}$ and $\widetilde{\text{mul}}$ prover algorithms a la [Mod24](https://github.com/Modulus-Labs/Papers/blob/master/remainder-paper.pdf), which combines ideas from [WJB+17](https://eprint.iacr.org/2017/242) and [XZZ+19](https://eprint.iacr.org/2019/317)
* Interpolative claim aggregation from [Tha13](https://ia.cr/2013/351) and random linear combination from [XZZ+19](https://eprint.iacr.org/2019/317)
* Polynomial commitment schemes implicit in [AHIV22](https://eprint.iacr.org/2022/1608) ("Ligero PCS") and [WTS+17](https://eprint.iacr.org/2017/1132) ("Hyrax PCS")
* Pedersen commitment-based zero-knowledge wrapper for all of the above, drawing heavily from [WTS+17](https://eprint.iacr.org/2017/1132)

See our companion [_Remainder_ book](https://worldcoin.github.io/remainder-documentation/) for an in-depth overview of all the above components and more!

## Status
Parts of _Remainder_ have been audited in the past, although we provide **no official guarantees** on the security of the current implementation. Past audits can be found in our [`audit_reports/`](./audit_reports/) directory. 

## Installation
_Remainder_ is fully implemented in Rust, and requires the (stable) version specified in the `rust-toolchain.toml` file. Once you have [installed Rust](https://doc.rust-lang.org/book/ch01-01-installation.html) -- 
```bash
git clone https://github.com/worldcoin/Remainder.git
cd Remainder
make check # Compiles + lints
make test-dev # Runs all fast unit tests; skips slower integration tests
```
This will compile, lint, and run all tests but the particularly expensive ones. You may run `make test` to run all tests (this will compile in release mode and may take >5 minutes). 

## Repository Structure
* [`frontend`](./frontend/): Crate containing circuit creation components. 
* [`prover`](./prover/): Crate containing modules for GKR proving and verifying proofs over a pre-defined circuit.
* [`ligero`](./ligero/): Crate containing an implementation of the Ligero polynomial commitment scheme.
* [`hyrax`](./hyrax/): Crate containing an implementation of the Hyrax zero-knowledge Pedersen commitment GKR wrapper.
* [`shared_types`](./shared_types/): Crate defining base traits/types (finite field, elliptic curve, multilinear extension, transcript, etc). 

## Examples
See [`frontend/examples`](./frontend/examples) directory for circuit examples! Additionally, see the _Remainder_ book's [quickstart](https://worldcoin.github.io/remainder-documentation/quickstart.html) and [frontend](https://worldcoin.github.io/remainder-documentation/frontend/frontend_components.html) tutorial sections for more details. To run the basic example highlighted in the quickstart:
```bash
cargo run --package frontend --example tutorial
```

## Contributing
Note: We plan to accept contributions at a later date, and have minimal bandwidth to review PRs currently.

Likewise, we are providing this source code for the benefit of the community, but cannot commit to any SemVer or API stability guarantees. Be warned: we may change things in a backwards-incompatible way at any time!

For soundness or other security-related issues, see [SECURITY.md](./SECURITY.md). 

## License
Unless otherwise specified, all code in this repository is dual-licensed under either:

* MIT License ([LICENSE-MIT](./LICENSE-MIT))
* Apache License, Version 2.0, with LLVM Exceptions ([LICENSE-APACHE](./LICENSE-APACHE)) at your option. This means you may select the license you prefer to use.

Any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.