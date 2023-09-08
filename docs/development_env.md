# Development Environment

To facilitate development on different systems a [dev container](https://containers.dev/) has been added. This should seamlessly enable development for users of [VSCode](https://code.visualstudio.com/docs/devcontainers/containers) on systems with docker installed.

For contributing guidelines please look here [link](../CONTRIBUTING.md)

### known issues:

- Some effort was take to run without using root inside the container. However nox always seemed to have permissions issues which I've been unable to fix.
- `git config --global --add safe.directory \`pwd\` ` Might be needed inside the container.
