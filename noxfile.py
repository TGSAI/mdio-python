"""Nox sessions."""

import os
import shutil
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import nox
from nox import Session
from nox import session


package = "mdio"
python_versions = ["3.12", "3.11", "3.10"]
nox.needs_version = ">= 2022.1.7"
nox.options.sessions = (
    "pre-commit",
    "safety",
    "mypy",
    "tests",
    "typeguard",
    "xdoctest",
    "docs-build",
)


@session(name="pre-commit", python=python_versions[0])
def precommit(session: Session) -> None:
    """Lint using pre-commit."""
    args = session.posargs or [
        "run",
        "--all-files",
        "--hook-stage=manual",
        "--show-diff-on-failure",
    ]
    session.run(
        "uv",
        "pip",
        "install",
        "black",
        "darglint",
        "flake8",
        "flake8-bandit",
        "flake8-bugbear",
        "flake8-docstrings",
        "flake8-rst-docstrings",
        "isort",
        "pep8-naming",
        "pre-commit",
        "pre-commit-hooks",
        "pyupgrade",
        external=True,
    )
    session.run("pre-commit", *args)


@session(python=python_versions[0])
def safety(session: Session) -> None:
    """Scan dependencies for insecure packages."""
    with NamedTemporaryFile(delete=False) as requirements:
        session.run(
            "uv",
            "pip",
            "compile",
            "pyproject.toml",
            "--output-file",
            requirements.name,
            external=True,
        )
        session.run("uv", "pip", "install", "safety", external=True)
        # TODO(Altay): Remove the CVE ignore once its resolved.
        # It's not critical, so ignoring now.
        ignore = ["70612"]
        try:
            session.run(
                "safety",
                "check",
                "--full-report",
                f"--file={requirements.name}",
                f"--ignore={','.join(ignore)}",
            )
        finally:
            os.remove(requirements.name)


@session(python=python_versions)
def mypy(session: Session) -> None:
    """Type-check using mypy."""
    args = session.posargs or ["src", "tests", "docs/conf.py"]
    session.run_always("uv", "pip", "install", "-e", ".", external=True)
    session.run("uv", "pip", "install", "mypy", "pytest", external=True)
    session.run("mypy", *args)
    if not session.posargs:
        session.run("mypy", f"--python-executable={sys.executable}", "noxfile.py")


@session(python=python_versions)
def tests(session: Session) -> None:
    """Run the test suite."""
    session.run_always("uv", "pip", "install", "-e", ".[cloud]", external=True)
    session.run(
        "uv",
        "pip",
        "install",
        "coverage[toml]",
        "pytest",
        "pygments",
        "pytest-dependency",
        "s3fs",
        external=True,
    )
    try:
        session.run("coverage", "run", "--parallel", "-m", "pytest", *session.posargs)
    finally:
        if session.interactive:
            session.notify("coverage", posargs=[])


@session(python=python_versions[0])
def coverage(session: Session) -> None:
    """Produce the coverage report."""
    args = session.posargs or ["report"]

    session.run("uv", "pip", "install", "coverage[toml]", external=True)

    if not session.posargs and any(Path().glob(".coverage.*")):
        session.run("coverage", "combine", external=True)

    session.run("coverage", *args, external=True)


@session(python=python_versions[0])
def typeguard(session: Session) -> None:
    """Runtime type checking using Typeguard."""
    session.run_always("uv", "pip", "install", "-e", ".", external=True)
    session.run(
        "uv", "pip", "install", "pytest", "typeguard", "pygments", external=True
    )
    session.run("pytest", f"--typeguard-packages={package}", *session.posargs)


@session(python=python_versions)
def xdoctest(session: Session) -> None:
    """Run examples with xdoctest."""
    if session.posargs:
        args = [package, *session.posargs]
    else:
        args = [f"--modname={package}", "--command=all"]
        if "FORCE_COLOR" in os.environ:
            args.append("--colored=1")

    session.run_always("uv", "pip", "install", "-e", ".", external=True)
    session.run("uv", "pip", "install", "xdoctest[colors]", external=True)
    session.run("python", "-m", "xdoctest", *args)


@session(name="docs-build", python=python_versions[0])
def docs_build(session: Session) -> None:
    """Build the documentation."""
    args = session.posargs or ["docs", "docs/_build"]
    if not session.posargs and "FORCE_COLOR" in os.environ:
        args.insert(0, "--color")

    session.run_always("uv", "pip", "install", "-e", ".", external=True)
    session.run(
        "uv",
        "pip",
        "install",
        "sphinx",
        "sphinx-click",
        "sphinx-copybutton",
        "furo",
        "myst-nb",
        "linkify-it-py",
        external=True,
    )

    build_dir = Path("docs", "_build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    session.run("sphinx-build", *args)


@session(python=python_versions[0])
def docs(session: Session) -> None:
    """Build and serve the documentation with live reloading on file changes."""
    args = session.posargs or ["--open-browser", "docs", "docs/_build"]
    session.run_always("uv", "pip", "install", "-e", ".", external=True)
    session.run(
        "uv",
        "pip",
        "install",
        "sphinx",
        "sphinx-autobuild",
        "sphinx-click",
        "sphinx-copybutton",
        "furo",
        "myst-nb",
        "linkify-it-py",
        external=True,
    )

    build_dir = Path("docs", "_build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    session.run("sphinx-autobuild", *args)
