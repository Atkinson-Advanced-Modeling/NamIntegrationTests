"""
Integration helpers (e.g. running external tools like NeuralAmpModelerCore loadmodel).

Expects the trainer (neural-amp-modeler) and core (NeuralAmpModelerCore) to be
cloned as siblings of this repo. In CI, set NAM_CORE_DIR to the core checkout path.
"""

import os as _os
import subprocess as _subprocess
import sys as _sys
from pathlib import Path as _Path

import pytest as _pytest

_REPO_ROOT = _Path(__file__).resolve().parents[1]
_NAM_CORE_DIR = _os.environ.get("NAM_CORE_DIR")
_NEURAL_AMP_MODELER_CORE_DIR = (
    _Path(_NAM_CORE_DIR)
    if _NAM_CORE_DIR
    else _REPO_ROOT.parent / "NeuralAmpModelerCore"
)


def loadmodel_exe_path() -> _Path | None:
    """Path to the loadmodel executable if it exists, else None."""
    if not _NEURAL_AMP_MODELER_CORE_DIR.exists():
        return None
    build = _NEURAL_AMP_MODELER_CORE_DIR / "build"
    if not build.exists():
        return None
    if _sys.platform == "win32":
        exe = build / "tools" / "loadmodel.exe"
    else:
        exe = build / "tools" / "loadmodel"
    return exe if exe.exists() else None


def render_exe_path() -> _Path | None:
    """Path to the render executable if it exists, else None."""
    if not _NEURAL_AMP_MODELER_CORE_DIR.exists():
        return None
    build = _NEURAL_AMP_MODELER_CORE_DIR / "build"
    if not build.exists():
        return None
    if _sys.platform == "win32":
        exe = build / "tools" / "render.exe"
    else:
        exe = build / "tools" / "render"
    return exe if exe.exists() else None


def run_loadmodel(
    model_path: _Path, *, timeout: float = 10.0
) -> _subprocess.CompletedProcess:
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
    return _subprocess.run(
        [str(exe), str(model_path)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def run_render(
    model_path: _Path,
    input_wav_path: _Path,
    output_path: _Path,
    *,
    timeout: float = 30.0,
) -> _subprocess.CompletedProcess:
    """
    Run NeuralAmpModelerCore's render tool on a .nam model and input WAV.

    :param model_path: Path to a .nam file.
    :param input_wav_path: Path to input WAV file.
    :param output_path: Path for output WAV file.
    :param timeout: Seconds before the subprocess is killed.
    :return: CompletedProcess from subprocess.run.
    :raises: FileNotFoundError if render executable is not found.
    """
    exe = render_exe_path()
    if exe is None:
        raise FileNotFoundError(
            "NeuralAmpModelerCore render not found: either "
            f"{_NEURAL_AMP_MODELER_CORE_DIR!s} is missing or build/tools/render is not built."
        )
    return _subprocess.run(
        [str(exe), str(model_path), str(input_wav_path), str(output_path)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


_has_loadmodel = loadmodel_exe_path() is not None

requires_loadmodel = _pytest.mark.skipif(
    not _has_loadmodel,
    reason="NeuralAmpModelerCore not present or loadmodel tool not built",
)

_has_render = render_exe_path() is not None

requires_render = _pytest.mark.skipif(
    not _has_render,
    reason="NeuralAmpModelerCore not present or render tool not built",
)
