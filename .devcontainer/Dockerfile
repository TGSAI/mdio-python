


ARG PYTHON_VERSION=3.11
ARG LINUX_DISTRO=bookworm

# FROM python:${PYTHON_VERSION}-${LINUX_DISTRO}
FROM mcr.microsoft.com/devcontainers/python:1-${PYTHON_VERSION}-${LINUX_DISTRO}

# Install git for nox pre-commit
RUN apt-get update \
    && apt-get install -y --no-install-recommends  \
    git \
    && rm -rf /var/lib/apt/lists/*

# Poetry
ARG POETRY_VERSION="1.6.1"
# RUN if [ "${POETRY_VERSION}" != "none" ]; then su vscode -c "umask 0002 && pip3 install poetry==${POETRY_VERSION}"; fi
RUN if [ "${POETRY_VERSION}" != "none" ]; then bash -c "umask 0002 && pip3 install poetry==${POETRY_VERSION}"; fi

# Nox
ARG NOX_VERSION="2023.4.22"
# RUN if [ "${NOX_VERSION}" != "none" ]; then su vscode -c "umask 0002 && pip3 install nox-poetry nox==${NOX_VERSION}"; fi
RUN if [ "${NOX_VERSION}" != "none" ]; then bash -c "umask 0002 && pip3 install nox-poetry nox==${NOX_VERSION}"; fi

# Nox poetry
ARG NOX_POETRY_VERSION="1.0.3"
# RUN if [ "${NOX_POETRY_VERSION}" != "none" ]; then su vscode -c "umask 0002 && pip3 install nox-poetry nox-poetry==${NOX_POETRY_VERSION}"; fi
RUN if [ "${NOX_POETRY_VERSION}" != "none" ]; then bash -c "umask 0002 && pip3 install nox-poetry nox-poetry==${NOX_POETRY_VERSION}"; fi
