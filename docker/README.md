# Docker Images

## User

## Developer

### Build Docker Image from Scratch

To run the developer container build first by (from source directory):

```shell
DOCKER_BUILDKIT=1 docker build -f docker/slim-bullseye-3.10-dev.Dockerfile -t mdio-dev .
```

The command above will build the dev environment and tag is as `mdio-dev`

### Start Container Instance

Then you can run the container in the background like below. This will ensure the following:

1. Host UNIX user is propagated, including home directory.
2. Host networking will be used.

```shell
docker run \
  --name mdio_dev_noroot \
  --tty \
  --detach \
  --network host \
  --user $(id -u):$(id -g) \
  -e USER=$USER \
  -e HOME=$HOME \
  -v $HOME:$HOME \
  -v $SRC_DIR:/mdio-python \
  mdio-dev
```

Where `SRC_DIR` environment variable is the path to MDIO source code on host.

If you want to run with default Docker permissions (root), i.e. in a cloud
environment, you can omit the `--user` option, environment variables `{USER, HOME}`
and the volume mount `HOME`.

Now you can:

- Use the container as a remote interpreter.
- Run MDIO developer tools like tests, build docs, etc.

### Running Developer Tools

Since the container now has developer tools, they can be executed with this pattern:

```shell
docker exec -it $CONTAINER_NAME $COMMAND
```

The `it` flags make the command interactive.

#### Examples

##### Linting

Run linting tools like `black` and `isort`.

```shell
docker exec -it mdio_dev_noroot black src
docker exec -it mdio_dev_noroot isort src
```

##### CI/CD Tools

Run some CI/CD like pre-commit, testing or building documentation.  
The `-rs` flag in `nox` tells it to run a session by reusing an existing virtual env.

```shell
docker exec -it mdio_dev_noroot nox -rs pre-commit  # Run pre-commit CI/CD
docker exec -it mdio_dev_noroot nox -rs tests  # Run tests
docker exec -it mdio_dev_noroot nox -rs docs-build  # Build docs locally
```

##### Jupyter Lab Server

This should launch a Jupyter Lab server that points to source directory at port 8888.

```shell
docker exec -it mdio_dev_noroot jupyter lab --ip $HOST_IP --no-browser
```

You can kill the server with `Control-C`.

The `$HOST_IP` can be omitted if docker is running locally. If you are
logged into another server via SSH, `$HOST_IP` must be provided or the
port has to be forwarded explicitly.
