ARG BASE_CONTAINER=python:3.10-alpine3.17

FROM $BASE_CONTAINER as builder

ARG python=3.10
ARG dask=2022.12.0
ARG gcsfs=2022.11.0
ARG zarr=2.13.3

ENV DASK_VERSION=${dask}
ENV GCSFS_VERSION=${gcsfs}
ENV ZARR_VERSION=${zarr}

# Since Alpine is very lightweight and Python wheels don't
# exist for all packages we need, some needs to be compiled
# from source. The following does:
# 1. Install tini
# 2. Install tagged temporary build dependencies
# 3. Install dask[array] and dask[distributed]
# 4. Install gcsfs and zarr
# 5. Clean up bloat (pip, caches, python lib files)
RUN apk update \
    && apk add --no-cache tini \
    && apk add --no-cache \
        --virtual build-deps \
          gcc musl-dev linux-headers libffi-dev jpeg-dev \
          libjpeg zlib-dev g++ build-base libzmq zeromq-dev

RUN pip install --upgrade pip \
    pip install \
        dask[array,distributed]==${DASK_VERSION} \
        gcsfs==${GCSFS_VERSION} \
        zarr==${ZARR_VERSION}

RUN pip cache purge \
    && pip uninstall -y pillow pip setuptools \
    && apk del build-deps \
    && find /usr/local/lib/python3.10/site-packages -follow -type f -name '*.a' -delete \
    && find /usr/local/lib/python3.10/site-packages -follow -type f -name '*.pyc' -delete \
    && find /usr/local/lib/python3.10/site-packages -follow -type f -name '*.js.map' -delete
#    && find /usr/local/lib/python3.10/site-packages/bokeh/server/static -follow -type f -name '*.js' ! -name '*.min.js' -delete

FROM $BASE_CONTAINER
LABEL org.opencontainers.image.source=https://github.com/tgsai/mdio-python
LABEL org.opencontainers.image.description="Minimal Dask image with zarr and gcsfs."
LABEL org.opencontainers.image.documentation="https://mdio-python.readthedocs.io"
LABEL org.opencontainers.image.licenses="Apache-2.0"

COPY --from=builder /usr/local /usr/local

COPY dask-prepare.sh /usr/bin/dask-prepare.sh
RUN chmod +x /usr/bin/dask-prepare.sh

ENTRYPOINT ["tini", "-g", "--", "/usr/bin/dask-prepare.sh"]
