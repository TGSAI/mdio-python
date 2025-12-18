"""Nox sessions."""

import os
import shlex
import shutil
import sys
from pathlib import Path
from textwrap import dedent

try:
    import nox
    from nox import Session
    from nox import session
except ImportError:
    message = f"""\
    Nox failed to import.

    Please install it using the following command:

    {sys.executable} -m pip install nox[uv]"""
    raise SystemExit(dedent(message)) from None

package = "mdio"
python_versions = ["3.13", "3.12", "3.11"]
nox.needs_version = ">=2025.2.9"
nox.options.default_venv_backend = "uv"
nox.options.sessions = ("pre-commit", "safety", "mypy", "tests", "typeguard", "xdoctest", "docs-build")


def session_install_uv(
    session: Session,
    install_project: bool = True,
    install_dev: bool = False,
    install_docs: bool = False,
) -> None:
    """Install root project into the session's virtual environment using uv."""
    env = {"UV_PROJECT_ENVIRONMENT": session.virtualenv.location}

    args = ["uv", "sync", "--frozen"]
    if not install_project:
        args.append("--no-install-project")
    if not install_dev:
        args.append("--no-dev")
    if install_docs:
        args.extend(["--group", "docs"])

    session.run_install(*args, silent=True, env=env)


def session_install_uv_package(session: Session, packages: list[str]) -> None:
    """Install packages into the session's virtual environment using uv lockfile."""
    env = {"UV_PROJECT_ENVIRONMENT": session.virtualenv.location}

    # Export requirements.txt to session temp dir using uv with locked dependencies
    requirements_tmp = str(Path(session.create_tmp()) / "requirements.txt")
    export_args = ["uv", "export", "--only-dev", "--no-hashes", "-o", requirements_tmp]
    session.run_install(*export_args, silent=True, env=env)

    # Install requested packages with requirements.txt constraints
    session.install(*packages, "--constraint", requirements_tmp)


def activate_virtualenv_in_precommit_hooks(session: Session) -> None:
    """Activate virtualenv in hooks installed by pre-commit.

    This function patches git hooks installed by pre-commit to activate the
    session's virtual environment. This allows pre-commit to locate hooks in
    that environment when invoked from git.

    Args:
        session: The Session object.
    """
    assert session.bin is not None  # noqa: S101

    # Only patch hooks containing a reference to this session's bindir. Support
    # quoting rules for Python and bash, but strip the outermost quotes so we
    # can detect paths within the bindir, like <bindir>/python.
    bindirs = [
        bindir[1:-1] if bindir[0] in "'\"" else bindir for bindir in (repr(session.bin), shlex.quote(session.bin))
    ]

    virtualenv = session.env.get("VIRTUAL_ENV")
    if virtualenv is None:
        return

    headers = {
        # pre-commit < 2.16.0
        "python": f"""\
            import os
            os.environ["VIRTUAL_ENV"] = {virtualenv!r}
            os.environ["PATH"] = os.pathsep.join((
                {session.bin!r},
                os.environ.get("PATH", ""),
            ))
            """,
        # pre-commit >= 2.16.0
        "bash": f"""\
            VIRTUAL_ENV={shlex.quote(virtualenv)}
            PATH={shlex.quote(session.bin)}"{os.pathsep}$PATH"
            """,
        # pre-commit >= 2.17.0 on Windows forces sh shebang
        "/bin/sh": f"""\
            VIRTUAL_ENV={shlex.quote(virtualenv)}
            PATH={shlex.quote(session.bin)}"{os.pathsep}$PATH"
            """,
    }

    hookdir = Path(".git") / "hooks"
    if not hookdir.is_dir():
        return

    for hook in hookdir.iterdir():
        if hook.name.endswith(".sample") or not hook.is_file():
            continue

        if not hook.read_bytes().startswith(b"#!"):
            continue

        text = hook.read_text()

        if not any(Path("A") == Path("a") and bindir.lower() in text.lower() or bindir in text for bindir in bindirs):
            continue

        lines = text.splitlines()

        for executable, header in headers.items():
            if executable in lines[0].lower():
                lines.insert(1, dedent(header))
                hook.write_text("\n".join(lines))
                break


@session(name="pre-commit", python=python_versions[0])
def precommit(session: Session) -> None:
    """Lint using pre-commit."""
    args = session.posargs or ["run", "--all-files", "--hook-stage=manual", "--show-diff-on-failure"]
    session_install_uv_package(session, ["ruff", "pre-commit", "pre-commit-hooks"])
    session.run("pre-commit", *args)
    if args and args[0] == "install":
        activate_virtualenv_in_precommit_hooks(session)


@session(python=python_versions[0])
def safety(session: Session) -> None:
    """Scan dependencies for insecure packages."""
    env = {"UV_PROJECT_ENVIRONMENT": session.virtualenv.location}
    requirements_tmp = str(Path(session.create_tmp()) / "requirements.txt")
    export_args = ["uv", "export", "--all-groups", "-o", requirements_tmp]
    session.run_install(*export_args, silent=True, env=env)
    session_install_uv_package(session, ["safety"])

    # CVE-2019-8341, jinja2: not a problem for us
    ignore = ["70612"]
    session.run("safety", "check", "--full-report", f"--file={requirements_tmp}", f"--ignore={','.join(ignore)}")


@session(python=python_versions)
def mypy(session: Session) -> None:
    """Type-check using mypy."""
    args = session.posargs or ["src", "tests", "docs/conf.py"]
    session_install_uv(session)
    session_install_uv_package(session, ["mypy", "pytest"])
    session.run("mypy", *args)
    if not session.posargs:
        session.run("mypy", f"--python-executable={sys.executable}", "noxfile.py")


@session(python=python_versions)
def tests(session: Session) -> None:
    """Run the test suite."""
    session_install_uv(session)
    session_install_uv_package(
        session, ["coverage[toml]", "pytest", "pygments", "pytest-dependency", "s3fs", "distributed", "zfpy"]
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

    session_install_uv_package(session, ["coverage[toml]"])

    if not session.posargs and any(Path().glob(".coverage.*")):
        session.run("coverage", "combine")

    session.run("coverage", *args)


@session(python=python_versions[0])
def typeguard(session: Session) -> None:
    """Runtime type checking using Typeguard."""
    session_install_uv(session)
    session_install_uv_package(session, ["pytest", "typeguard", "pygments"])
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

    session_install_uv(session)
    session_install_uv_package(session, ["xdoctest[colors]"])
    session.run("python", "-m", "xdoctest", *args)


@session(name="docs-build", python=python_versions[0])
def docs_build(session: Session) -> None:
    """Build the documentation."""
    args = session.posargs or ["docs", "docs/_build"]
    if not session.posargs and "FORCE_COLOR" in os.environ:
        args.insert(0, "--color")

    session_install_uv(session)
    session_install_uv_package(
        session,
        [
            "aiohttp",
            "autodoc-pydantic",
            "furo",
            "linkify-it-py",
            "matplotlib",
            "myst-nb",
            "sphinx",
            "sphinx-click",
            "sphinx-copybutton",
            "sphinx-design",
            "ipywidgets",
        ],
    )

    build_dir = Path("docs", "_build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    session.run("sphinx-build", *args)


@session(python=python_versions[0])
def docs(session: Session) -> None:
    """Build and serve the documentation with live reloading on file changes."""
    args = session.posargs or ["--open-browser", "docs", "docs/_build"]
    session_install_uv(session)
    session_install_uv_package(
        session,
        [
            "aiohttp",
            "autodoc-pydantic",
            "furo",
            "linkify-it-py",
            "matplotlib",
            "myst-nb",
            "sphinx",
            "sphinx-autobuild",
            "sphinx-click",
            "sphinx-copybutton",
            "sphinx-design",
            "ipywidgets",
        ],
    )

    build_dir = Path("docs", "_build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    session.run("sphinx-autobuild", *args)
