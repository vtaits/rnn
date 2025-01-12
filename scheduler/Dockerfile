FROM rust:1.82 as builder

WORKDIR /app

COPY Cargo.toml Cargo.lock ./

RUN mkdir src && echo "fn main() {}" > src/main.rs

RUN cargo build --release && rm -rf src

COPY . .

RUN cargo build --release

FROM debian:buster-slim

WORKDIR /app

COPY --from=builder /app/target/release/scheduler .

CMD ["./scheduler"]
