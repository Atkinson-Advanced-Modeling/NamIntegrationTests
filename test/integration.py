"""
Integration helpers (e.g. running external tools like NeuralAmpModelerCore loadmodel).

Expects the trainer (neural-amp-modeler) and core (NeuralAmpModelerCore) to be
cloned as siblings of this repo.
"""

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_NEURAL_AMP_MODELER_CORE_DIR = _REPO_ROOT.parent / "NeuralAmpModelerCore"


def loadmodel_exe_path() -> Path | None:
    """Path to the loadmodel executable if it exists, else None."""
    if not _NEURAL_AMP_MODELER_CORE_DIR.exists():
        return None
    build = _NEURAL_AMP_MODELER_CORE_DIR / "build"
    if not build.exists():
        return None
    if sys.platform == "win32":
        exe = build / "tools" / "loadmodel.exe"
    else:
        exe = build / "tools" / "loadmodel"
    return exe if exe.exists() else None


def run_loadmodel(
    model_path: Path, *, timeout: float = 10.0
) -> subprocess.CompletedProcess:
    """
    Run NeuralAmpModelerCore's loadmodel tool on a .nam model path.

    :param model_path: Path to a .nam file (or directory containing one).
    :param timeout: Seconds before the subprocess is killed.
    :return: CompletedProcess from subprocess.run.
    :raises: FileNotFoundError if loadmodel executable is not found.
    """
    exe = loadmodel_exe_path()
    if exe is None:
        raise FileNotFoundError(
            "NeuralAmpModelerCore loadmodel not found: either "
            f"{_NEURAL_AMP_MODELER_CORE_DIR!s} is missing or build/tools/loadmodel is not built."
        )
    return subprocess.run(
        [str(exe), str(model_path)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


_has_loadmodel = loadmodel_exe_path() is not None

requires_loadmodel = pytest.mark.skipif(
    not _has_loadmodel,
    reason="NeuralAmpModelerCore not present or loadmodel tool not built",
)
