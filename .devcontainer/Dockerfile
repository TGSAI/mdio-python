ARG PYTHON_VERSION=3.12
ARG LINUX_DISTRO=bookworm

FROM mcr.microsoft.com/devcontainers/python:1-${PYTHON_VERSION}-${LINUX_DISTRO}

# Install git for nox pre-commit
RUN apt-get update \
  && apt-get install -y --no-install-recommends  \
  git \
  && rm -rf /var/lib/apt/lists/*

# Poetry
ARG POETRY_VERSION="2.1.2"
RUN if [ "${POETRY_VERSION}" != "none" ]; then bash -c "umask 0002 && pip3 install poetry==${POETRY_VERSION}"; fi

# Nox
ARG NOX_VERSION="2025.2.9"
RUN if [ "${NOX_VERSION}" != "none" ]; then bash -c "umask 0002 && pip3 install nox-poetry nox==${NOX_VERSION}"; fi

# Nox poetry
ARG NOX_POETRY_VERSION="1.2.0"
RUN if [ "${NOX_POETRY_VERSION}" != "none" ]; then bash -c "umask 0002 && pip3 install nox-poetry nox-poetry==${NOX_POETRY_VERSION}"; fi

# Extra python packages for dev
RUN bash -c "umask 0002 &&  pip3 install msgpack ipykernel"

# Create venv
RUN python -m venv /opt/venv
RUN poetry config virtualenvs.create false
ENV PATH="/opt/venv/bin:$PATH"

# Extra python packages for dev
RUN bash -c "umask 0002 &&  pip3 install msgpack ipykernel"

# Allow users to update venv
RUN chmod -R 777 /opt/venv
