# Make venv and install MDIO dependencies
FROM python:3.10-slim-bullseye as venv_base

# Build time args for dev tools etc
ARG POETRY_VERSION=1.2.2
ARG NOX_VERSION=2022.11.21
ARG NOX_POETRY_VERSION=1.0.2

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml poetry.lock /

RUN pip install \
      "poetry==$POETRY_VERSION"  \
      "nox==$NOX_VERSION" \
      "nox-poetry==$NOX_POETRY_VERSION" \
    && poetry config virtualenvs.create false \
    && poetry install \
      --no-root \
      --with dev \
      --with interactive \
      --all-extras \
      --no-ansi

# Install Git
FROM python:3.10-slim-bullseye as system_tools

RUN apt-get update \
    && apt-get install -y --no-install-recommends  \
      git \
      graphviz \
    && rm -rf /var/lib/apt/lists/*

# Final Stage (git + venv)
FROM system_tools

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:$PATH" \
    SHELL=/bin/bash \
    PYTHONPATH=/mdio-python/src

COPY --from=venv_base --chmod=777 /opt/venv /opt/venv

WORKDIR /mdio-python
