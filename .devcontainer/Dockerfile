# This is a multi-stage build to minimize image size
# You can specify python version and linux distro as build args
# Stage 1: Build python venv + dependencies
# Stage 2: Install some system level developer tools
# Stage 3: Combine Stage 2 + Venv

ARG PYTHON_VERSION=3.11
ARG LINUX_DISTRO=slim-bookworm

FROM python:${PYTHON_VERSION}-${LINUX_DISTRO} as venv_base
# Stage 1: Create venv and install MDIO dependencies only

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR mdio-python

# Install all of MDIO dependencies
# 1. Get latest release from MDIO
# 2. Unpack it to WORKDIR
# 3. Install poetry and nox (dev tools)
# 4. Install all of MDIO dependencies with poetry
# && curl -s https://api.github.com/repos/tgsai/mdio-python/releases/latest \
#     | python -c 'import sys, json; print(json.load(sys.stdin)["tarball_url"])' \
#     | xargs curl -LJ \
#     | tar zx --strip=1 \
COPY . /mdio-python

#    --with dev \
#    --all-extras \
RUN apt-get update \
    && apt-get install -y --no-install-recommends  \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install \
    msgpack \
    nox \
    nox-poetry \
    "poetry>=1.6.1" \
    && poetry config virtualenvs.create false \
    && poetry install --no-root --no-ansi --with dev


FROM python:${PYTHON_VERSION}-${LINUX_DISTRO} as system_tools
# STAGE 2
# Install `git` and graphviz
# - git for pre-commit stuff
# - graphviz for debugging dask graphs

RUN apt-get update \
    && apt-get install -y --no-install-recommends  \
    git \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

FROM system_tools
# STAGE 3
# Based on STAGE 2
# 1. Set Python interpreter
# 2. Add expected source dir to PYTHONPATH
# 3. Copy the venv from STAGE 1
# 4. Set user

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:$PATH" \
    SHELL=/bin/bash \
    PYTHONPATH=/mdio-python/src

COPY --from=venv_base --chmod=777 /opt/venv /opt/venv

RUN python3 -m pip install \
    msgpack

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && chmod -r 777 /opt/venv

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME

WORKDIR /mdio-python
