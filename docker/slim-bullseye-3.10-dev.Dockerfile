FROM python:3.10-slim-bullseye

# Build time args for dev tools etc
ARG MDIO_SRC_DIR=/mdio-python
ARG POETRY_VERSION=1.2.2
ARG NOX_VERSION=2022.8.7
ARG NOX_POETRY_VERSION=1.0.1

ENV PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    # Fake user home dir under source directory
    HOME=$MDIO_SRC_DIR/.cache


COPY . $MDIO_SRC_DIR/
WORKDIR $MDIO_SRC_DIR

RUN apt-get update \
    && apt-get install -y git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install \
      "poetry==$POETRY_VERSION"  \
      "nox==$NOX_VERSION" \
      "nox-poetry==$NOX_POETRY_VERSION" \
    && poetry config virtualenvs.create false \
    && poetry install \
      --with dev \
      --with interactive \
      --all-extras \
      --no-ansi
