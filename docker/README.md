# Docker Images

## User

## Developer

### Build Docker Image from Scratch

To run the developer container build first by (from source directory):

```shell
docker build -t mdio-dev -f docker/slim-bullseye-3.10-dev.Dockerfile .
```

The command above will build the dev environment and tag is as `mdio-dev`

### Start Container Instance

Then you can run the container in the background like below. This will ensure the following:

1. UNIX permissions are propagated.
2. Host networking will be used.
3. Source files from your host are mounted to the container.

```shell
docker run \
  --tty \
  --detach \
  --user $UID:$GID \
  --name mdio_dev_noroot \
  --network host \
  --volume $HOST_SOURCE:/mdio-python \
  mdio-dev
```

Where `$UID` is your linux user ID, `$GID` is your linux group ID.  
The `$HOST_SOURCE` is where your source code is on your host machine.

If you want to run with default Docker permissions (root), i.e. in a cloud
environment, you can omit the `--user $UID:$GID` option.

Now you can:

- Use the container as a remote interpreter.
- Run MDIO developer tools like tests, build docs, etc.

### Running Developer Tools

Since the container now has developer tools, they can be executed with this pattern:

```shell
docker exec $CONTAINER_NAME $COMMAND
```

#### Examples

##### Linting

Run linting tools like `black` and `isort`.

```shell
docker exec mdio_dev_noroot black src
docker exec mdio_dev_noroot isort src
```

##### CI/CD Tools

Run some CI/CD like pre-commit, testing or building documentation.  
The `-rs` flag in `nox` tells it to run a session by reusing an existing virtual env.

```shell
docker exec mdio_dev_noroot nox -rs pre-commit  # Run pre-commit CI/CD
docker exec mdio_dev_noroot nox -rs tests  # Run tests
docker exec mdio_dev_noroot nox -rs docs-build  # Build docs locally
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
