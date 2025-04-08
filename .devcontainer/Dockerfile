ARG PYTHON_VERSION=3.13
ARG LINUX_DISTRO=bookworm

FROM mcr.microsoft.com/devcontainers/python:1-${PYTHON_VERSION}-${LINUX_DISTRO}

# Install git for nox pre-commit
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    git \
  && rm -rf /var/lib/apt/lists/*

# Install UV dependency manager
ARG UV_VERSION="0.6.11"
RUN if [ "${UV_VERSION}" != "none" ]; then bash -c "umask 0002 && pip3 install uv==${UV_VERSION}"; fi

# Nox installation using UV (system-level tool)
ARG NOX_VERSION="2025.2.9"
RUN if [ "${NOX_VERSION}" != "none" ]; then bash -c "umask 0002 && uv tool install nox==${NOX_VERSION}"; fi

# Extra python packages for dev (global/system-level)
RUN bash -c "umask 0002 && uv tool install msgpack ipykernel"

# Create virtual environment
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Allow users to update venv
RUN chmod -R 777 /opt/venv
