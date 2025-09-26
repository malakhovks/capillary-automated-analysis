"""Utility helpers to make the project work smoothly on Google Colab.

The functions defined here are intentionally lightweight so they can be
executed directly inside a Colab notebook.  They cover the two most
common issues we have faced when running the pipeline on Colab:

1.  Installing python dependencies using a requirements file that does
    not rely on conda-specific wheels.
2.  Making sure the repository sub-packages are added to ``sys.path`` so
    that direct module imports (for example ``from Image_Analysis ...``)
    behave the same way they do in a local development environment.

Typical usage inside a Colab notebook::

    import colab_setup
    colab_setup.install_requirements()
    colab_setup.configure_python_path()

Both functions are idempotent which means they can safely be executed
multiple times without producing duplicate work.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent

# Subdirectories that expose python packages used across the project.  In a
# traditional development environment these folders are added to ``PYTHONPATH``
# through editable installs.  Colab does not preserve such configuration, so we
# expose a helper to add them programmatically.
PACKAGE_ROOTS: tuple[str, ...] = (
    "Data_Preprocess",
    "Flow_Velocity_Measurement",
    "Image_Analysis",
    "Image_Segmentation",
    "Keypoint_Detection",
    "Object_Detection",
    "Video_Process",
)


def _call_subprocess(command: Iterable[str]) -> None:
    """Execute *command* and stream the output to the notebook.

    ``subprocess.run`` is used instead of ``os.system`` to keep the code
    testable and to surface non-zero return codes to the caller.
    """

    try:
        completed = subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        joined = " ".join(str(part) for part in exc.cmd)
        raise RuntimeError(
            "Command execution failed. Scroll up for the captured output and rerun the cell "
            f"after addressing the issue. (exit code {exc.returncode}: {joined})"
        ) from exc
    else:
        if completed.returncode != 0:
            joined = " ".join(str(part) for part in command)
            raise RuntimeError(
                f"Command {joined} finished with return code {completed.returncode}."
            )


def install_requirements(requirements_path: str | Path = "requirements.txt") -> None:
    """Install python dependencies using pip.

    Parameters
    ----------
    requirements_path:
        Path to a requirements file.  The default points at the repository level
        ``requirements.txt`` that is compatible with the Colab python runtime.
    """

    requirements_path = Path(requirements_path)
    if not requirements_path.exists():
        raise FileNotFoundError(f"Could not find requirements file: {requirements_path}")
    # Colab images occasionally miss one or more of these tools, and older
    # versions bundled with the runtime can fail to build wheels for packages
    # that ship only source distributions.  Upgrading them first keeps the
    # installation process predictable.
    _call_subprocess(
        (
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
            "setuptools",
            "wheel",
        )
    )
    _call_subprocess((sys.executable, "-m", "pip", "install", "-r", str(requirements_path)))


def configure_python_path(additional_roots: Iterable[str | Path] | None = None) -> None:
    """Ensure the repository sub-packages are importable in Colab.

    Parameters
    ----------
    additional_roots:
        Optional iterable of extra directories to add to ``sys.path`` in addition
        to the default project folders defined in :data:`PACKAGE_ROOTS`.
    """

    roots = list(PACKAGE_ROOTS)
    if additional_roots:
        roots.extend(str(Path(root)) for root in additional_roots)

    for relative_path in roots:
        path = (PROJECT_ROOT / relative_path).resolve()
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
        else:
            # We silently skip missing optional paths to keep the helper usable
            # even when certain datasets or components are not checked out.
            continue


__all__ = [
    "configure_python_path",
    "install_requirements",
]
