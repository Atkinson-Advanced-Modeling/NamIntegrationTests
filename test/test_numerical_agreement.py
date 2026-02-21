"""
Numerical agreement tests: trainer (PyTorch) vs NeuralAmpModelerCore.

Assert that predictions from the trainer match predictions from the core's
render tool when given the same input.
"""

import json as _json
from pathlib import Path as _Path
from tempfile import TemporaryDirectory as _TemporaryDirectory

import numpy as _np
import pytest as _pytest

from _integration import run_render, requires_render
from nam.data import np_to_wav, wav_to_np
from nam.train.lightning_module import LightningModule as _LightningModule


def _load_demonet_config() -> dict:
    path = _Path(__file__).resolve().parents[1] / "configs" / "demonet.json"
    data = _json.loads(path.read_text())
    data.pop("_notes", None)
    data.pop("_comments", None)
    return data


_RTOL = 1e-5
_ATOL = 1e-6


@requires_render
def test_trainer_core_numerical_agreement(demonet_config):
    """
    Export with include_snapshot -> render through core -> compare outputs.
    """
    config = _load_demonet_config()
    module = _LightningModule.init_from_config(config)
    module.net.sample_rate = 48000
    sample_rate = 48000

    with _TemporaryDirectory() as tmpdir:
        outdir = _Path(tmpdir)
        module.net.export(outdir, basename="model", include_snapshot=True)

        nam_path = outdir / "model.nam"
        assert nam_path.exists()

        test_inputs_path = outdir / "test_inputs.npy"
        test_outputs_path = outdir / "test_outputs.npy"
        assert test_inputs_path.exists()
        assert test_outputs_path.exists()

        input_npy = _np.load(test_inputs_path)
        expected_npy = _np.load(test_outputs_path)

        input_wav_path = outdir / "input.wav"
        np_to_wav(input_npy, input_wav_path, rate=sample_rate)

        output_wav_path = outdir / "output.wav"
        result = run_render(nam_path, input_wav_path, output_wav_path)

        assert (
            result.returncode == 0
        ), f"render failed: stderr={result.stderr!r} stdout={result.stdout!r}"

        actual = wav_to_np(output_wav_path)

        expected_flat = _np.squeeze(expected_npy)
        actual_flat = _np.squeeze(actual)

        assert (
            expected_flat.shape == actual_flat.shape
        ), f"Shape mismatch: expected {expected_flat.shape}, got {actual_flat.shape}"

        assert _np.allclose(
            actual_flat, expected_flat, rtol=_RTOL, atol=_ATOL
        ), f"Numerical mismatch: max |diff| = {_np.max(_np.abs(actual_flat - expected_flat))}"
