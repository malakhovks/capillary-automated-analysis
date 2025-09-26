"""Utility helpers for configuring this project inside Google Colab notebooks.

The default project layout relies on relative imports which often break on
Colab because the repository root is not available on ``sys.path``.  The helper
below centralises the few steps that repeatedly come up when we demo the
project in notebooks:

* ensure the repository root is importable;
* optionally install the Python dependencies; and
* create frequently used output folders so downstream code can save files
  without additional boilerplate.

Example
-------
```python
from google.colab import drive
from pathlib import Path

# 1) Mount Google Drive (optional)
drive.mount('/content/drive')

# 2) Clone the repository and change into it
!git clone https://github.com/<user>/capillary-automated-analysis.git
%cd capillary-automated-analysis

# 3) Prepare the runtime
from colab_setup import prepare_colab_environment
prepare_colab_environment(install_requirements=True)
```
``prepare_colab_environment`` returns the resolved project root so that a
notebook can build paths relative to it when loading data.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable


DEFAULT_OUTPUT_DIRS: tuple[str, ...] = (
    "outputs",
    "outputs_evaluation",
    "Flow_Velocity_Measurement/output",
    "Flow_Velocity_Measurement/output_table",
    "Image_Segmentation/image_segmentation/result",
)


def _as_path(value: str | Path) -> Path:
    """Return ``value`` as a resolved :class:`~pathlib.Path` instance."""

    return Path(value).expanduser().resolve()


def prepare_colab_environment(
    project_root: str | Path = ".",
    *,
    requirements_path: str | Path = "requirements.txt",
    install_requirements: bool = False,
    extra_output_dirs: Iterable[str | Path] | None = None,
) -> Path:
    """Bootstrap a Colab runtime for this repository.

    Parameters
    ----------
    project_root:
        Location of the repository.  When running inside a cloned checkout in
        Colab the default value (``'.'``) is sufficient.
    requirements_path:
        Path to the requirements file that should be installed when
        ``install_requirements`` is set to :data:`True`.
    install_requirements:
        Whether ``pip install`` should be executed.  The flag defaults to
        :data:`False` so that users can manage package installation manually if
        they prefer a custom environment.
    extra_output_dirs:
        Additional directories that should exist when the function returns.

    Returns
    -------
    :class:`~pathlib.Path`
        The resolved repository root.
    """

    root = _as_path(project_root)

    # Make sure project imports like ``from Image_Segmentation...`` work.
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Create directories that the pipeline expects during execution.  This
    # mirrors the structure created by the original bash scripts and avoids
    # littering notebooks with ``mkdir`` commands.
    output_dirs = list(DEFAULT_OUTPUT_DIRS)
    if extra_output_dirs:
        output_dirs.extend(str(_as_path(root / Path(p))) for p in extra_output_dirs)

    for directory in output_dirs:
        path = Path(directory)
        if not path.is_absolute():
            path = root / directory
        path.mkdir(parents=True, exist_ok=True)

    if install_requirements:
        req_path = _as_path(root / requirements_path)
        command = [sys.executable, "-m", "pip", "install", "-r", str(req_path)]
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            combined_output = "\n\n".join(part for part in (stdout, stderr) if part)

            message_text = "\n\n" + combined_output if combined_output else ""
            compatibility_hint = ""
            if "No matching distribution found" in combined_output:
                compatibility_hint = (
                    "\n\nThe pinned dependencies in the requirements file do not "
                    "appear to provide wheels for Python "
                    f"{sys.version_info.major}.{sys.version_info.minor}. Consider "
                    "running `prepare_colab_environment(install_requirements=False)` "
                    "and managing packages manually or updating the requirements "
                    "for your Python version."
                )

            raise RuntimeError(
                "Failed to install dependencies via pip.\n\n"
                f"Command: {shlex.join(command)}\n"
                f"Exit code: {result.returncode}" + message_text + compatibility_hint
            )

    return root


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a Colab runtime.")
    parser.add_argument(
        "--project-root",
        default=Path.cwd(),
        help="Path to the repository (defaults to the current working directory).",
    )
    parser.add_argument(
        "--requirements-path",
        default="requirements.txt",
        help="Requirements file to install when --install-requirements is set.",
    )
    parser.add_argument(
        "--install-requirements",
        action="store_true",
        help="Install the dependencies via pip before returning.",
    )
    parser.add_argument(
        "--extra-output-dir",
        action="append",
        default=[],
        help="Additional directories to create (can be provided multiple times).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    prepare_colab_environment(
        project_root=args.project_root,
        requirements_path=args.requirements_path,
        install_requirements=args.install_requirements,
        extra_output_dirs=args.extra_output_dir,
    )


if __name__ == "__main__":
    main()
