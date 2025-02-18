# Builder stage
# FROM rust:latest as builder
FROM rust:1.84.1-bookworm as builder
WORKDIR /app
COPY . .
RUN make prod-seq

# Runtime stage
# FROM ubuntu:latest
FROM ubuntu:24.04
WORKDIR /app
# Copy necessary files for running `world_prove`.
COPY ./world.circuit .
COPY ./iriscode_pcp_example ./iriscode_pcp_example
# Copy the built binary from the builder stage
COPY --from=builder /app/target/release/world_prove ./world_prove
CMD ["./world_prove", "--circuit", "world.circuit", "--input", "iriscode_pcp_example/", "--output-dir", "world_zkp"]
