#!/bin/sh

set -x

# We start by adding extra apt packages, since pip modules may required library
if [ "$EXTRA_APT_PACKAGES" ]; then
    echo "EXTRA_APT_PACKAGES environment variable found."
    echo "Note this is an Alpine build so it will be from apk."
    apk update -y
    apk add --no-cache $EXTRA_APT_PACKAGES
fi

if [ "$USE_MAMBA" == "true" ]; then
    echo "USE_MAMBA detected. Ignoring since this is a pip based image."
fi

if [ -e "/opt/app/environment.yml" ]; then
    echo "environment.yml found. Ignoring since this is a pip based image."
fi

if [ "$EXTRA_CONDA_PACKAGES" ]; then
    echo "EXTRA_CONDA_PACKAGES environment variable found."
    echo "Ignoring conda packages since this is a pip based image."
fi

if [ "$EXTRA_PIP_PACKAGES" ]; then
    echo "EXTRA_PIP_PACKAGES environment variable found. Installing".
    python -m ensurepip
    /usr/local/bin/pip3 install $EXTRA_PIP_PACKAGES
fi

# Run extra commands
exec "$@"
