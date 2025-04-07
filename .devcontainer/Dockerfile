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

# Nox installation using UV (system-level, so we pass --system)
ARG NOX_VERSION="2025.2.9"
RUN if [ "${NOX_VERSION}" != "none" ]; then bash -c "umask 0002 && uv pip install --system nox==${NOX_VERSION}"; fi

# Extra python packages for dev (global/system-level)
RUN bash -c "umask 0002 && uv pip install --system msgpack ipykernel"

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Extra python packages for dev (inside the venv)
RUN bash -c "umask 0002 && uv pip install msgpack ipykernel"

# Allow users to update venv
RUN chmod -R 777 /opt/venv
