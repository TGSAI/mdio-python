FROM python:3.10-slim-bullseye

ARG MDIO_VERSION=0.2.5

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    SHELL=/bin/bash \
    NUMBA_CACHE_DIR=/tmp

RUN pip install "multidimio[lossy,distributed,cloud]==$MDIO_VERSION"

ENTRYPOINT ["mdio"]
