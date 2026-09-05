# goofi as one public container. DRAFT — never built yet.
#
# The build path IS the run path: `goofi_init::repo_root()` is baked in at compile time from
# CARGO_MANIFEST_DIR, so the binary looks for its two venvs where the build left them. Both stages
# are /app, and the runtime stage carries the venvs and the interpreter they point at.

FROM rust:1.97.1-bookworm AS build

# uv and npm are the two tools goofi-init demands; libasound2-dev is cpal's, which is compiled in
# whether or not a demo ever opens a device.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl libasound2-dev nodejs npm pkg-config \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

# A named directory rather than uv's default under HOME, so the runtime stage copies one known path.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python

WORKDIR /app
COPY . .

# The one setup step, then the build. goofi-init makes both venvs, installs both wheels and the
# frontend's dependencies; the frontend and every shipped node are compiled INTO the binary.
RUN cargo run -p goofi-init
RUN cargo build --release -p goofi-cli

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libasound2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=build /opt/uv-python /opt/uv-python
COPY --from=build /app/.gfivenv /app/.gfivenv
COPY --from=build /app/.gfivenv-ft /app/.gfivenv-ft
COPY --from=build /app/target/release/goofi /usr/local/bin/goofi

# The embedded interpreter is linked against the free-threaded build, whose shared library lives
# with the interpreter rather than on the loader's default path.
RUN printf '/opt/uv-python/*/lib\n' > /etc/ld.so.conf.d/uv-python.conf && ldconfig

# Sessions and any saved patch belong on a mounted volume, not in the layer.
ENV GOOFI_HOME=/data
ENV GOOFI_DEMO=1
RUN mkdir -p /data

# `--port` rather than a PORT variable goofi would have to know the name of.
CMD ["sh", "-c", "exec goofi serve --bind 0.0.0.0 --port ${PORT:-8000}"]
