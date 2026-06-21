FROM alpine:3.24 AS build

RUN apk add --no-cache \
    linux-headers \
    git \
    cmake \
    make \
    g++ \
    pkgconf \
    && apk add --no-cache --repository=http://dl-cdn.alpinelinux.org/alpine/edge/testing/ \
    gperftools-dev \
    tcmalloc

WORKDIR /workspace
COPY . /workspace

RUN cd /workspace && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    make install

# ---

FROM alpine:3.24

ENV PATH="/usr/local/bin:${PATH}"
COPY --from=build /usr/local /usr/local

# Add runtime dependencies.
RUN apk add --no-cache --repository=http://dl-cdn.alpinelinux.org/alpine/edge/testing/ \
    gperftools-dev tcmalloc
