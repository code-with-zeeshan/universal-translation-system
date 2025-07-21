# Stage 1: Build the encoder core
FROM ubuntu:20.04 AS builder

RUN apt-get update && apt-get install -y \
    cmake \
    g++ \
    git \
    wget \
    make \
    liblz4-dev

WORKDIR /encoder_core
COPY encoder_core/ .
RUN mkdir -p build && cd build && cmake .. && make -j$(nproc)

# Stage 2: Minimal runtime image for CI/CD artifact or edge packaging
FROM ubuntu:20.04
COPY --from=builder /encoder_core/build/libuniversal_encoder_core.so /usr/lib/
COPY --from=builder /encoder_core/include /usr/include/universal_encoder_core
# Optionally copy test binaries or examples if needed for CI
# COPY --from=builder /encoder_core/build/tests /tests
# COPY --from=builder /encoder_core/build/examples /examples

# No entrypoint: this image is for CI/CD artifact packaging, not serving