# The scoring image's base: the ors-scorer binary and nothing else. A deploy adds one
# plan bundle (plan.json + models/) at /app/model, so no Rust build or pip install
# happens per deploy:
#
#   FROM ors-scorer:<version>
#   COPY ./bundle /app/model
#
# ONNX Runtime is linked into the binary statically.

FROM rust:1.90-bookworm AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked \
    && cp target/release/ors-scorer /usr/local/bin/ors-scorer

FROM debian:bookworm-slim
COPY --from=build /usr/local/bin/ors-scorer /usr/local/bin/ors-scorer
RUN useradd --system --uid 10001 scorer
USER scorer
ENV MODEL_DIR=/app/model
EXPOSE 5001
ENTRYPOINT ["ors-scorer"]
CMD ["serve"]
