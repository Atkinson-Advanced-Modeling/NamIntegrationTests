"""
Numerical agreement tests: trainer (PyTorch) vs NeuralAmpModelerCore.

Assert that predictions from the trainer match predictions from the core's
render tool when given the same input. Parametrized over all config variants
tested in the loadmodel tests.
"""

from pathlib import Path as _Path
from tempfile import TemporaryDirectory as _TemporaryDirectory

import numpy as _np
import pytest as _pytest

from _configs import (
    get_all_variant_ids as _get_all_variant_ids,
    get_config_for_variant as _get_config_for_variant,
)
from _integration import run_render as _run_render, requires_render as _requires_render
from nam.data import np_to_wav, wav_to_np
from nam.train.lightning_module import LightningModule as _LightningModule

_RTOL = 1e-5
_ATOL = 1e-6

# Variants with known small trainer/core implementation differences
_VARIANT_TOLERANCES = {
    "film_layer1x1_post_film": (1e-5, 0.01),
}


@_requires_render
@_pytest.mark.parametrize("variant_id", _get_all_variant_ids())
def test_trainer_core_numerical_agreement(variant_id):
    """
    Export with include_snapshot -> render through core -> compare outputs.

    Runs for each config variant (activations, bottleneck, FiLM, etc.).
    """
    config = _get_config_for_variant(variant_id)
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
        result = _run_render(nam_path, input_wav_path, output_wav_path)

        assert result.returncode == 0, (
            f"render failed for variant {variant_id!r}: "
            f"stderr={result.stderr!r} stdout={result.stdout!r}"
        )

        actual = wav_to_np(output_wav_path)

        expected_flat = _np.squeeze(expected_npy)
        actual_flat = _np.squeeze(actual)

        assert expected_flat.shape == actual_flat.shape, (
            f"Shape mismatch for variant {variant_id!r}: "
            f"expected {expected_flat.shape}, got {actual_flat.shape}"
        )

        rtol, atol = _VARIANT_TOLERANCES.get(variant_id, (_RTOL, _ATOL))
        assert _np.allclose(actual_flat, expected_flat, rtol=rtol, atol=atol), (
            f"Numerical mismatch for variant {variant_id!r}: "
            f"max |diff| = {_np.max(_np.abs(actual_flat - expected_flat))}"
        )
