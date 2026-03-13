"""
Numerical agreement tests for slimmable WaveNet at slimmed sizes.

Asserts Python (trainer) output matches NeuralAmpModelerCore render at each slim
ratio. Uses a context manager to set the Python net's slim value and the
--slim argument for the C++ render tool. The core detects slimmable from the
config (allowed_channels) and instantiates the slimmable variant when loading
a WaveNet .nam. Full-size (1.0) is covered by
test_numerical_agreement.test_trainer_core_numerical_agreement.
"""

from contextlib import contextmanager as _contextmanager
from pathlib import Path as _Path
from tempfile import TemporaryDirectory as _TemporaryDirectory

import numpy as _np
import pytest as _pytest
import torch as _torch
from _configs import get_config_for_variant as _get_config_for_variant
from _integration import requires_render as _requires_render
from _integration import run_render as _run_render
from nam.data import np_to_wav, wav_to_np
from nam.train.lightning_module import LightningModule as _LightningModule

_RTOL = 1e-5
_ATOL = 1e-6

# Slim ratios to test (exclude 1.0; full size is in test_numerical_agreement).
_SLIM_RATIOS = [0.0, 0.25, 0.5, 0.75]


def _get_slimmable_net(module):
    """Extract the underlying _WaveNet (Slimmable) from LightningModule's net."""
    return module.net._net


@_contextmanager
def _slim_context(net, value: float):
    """Context manager: set slimming to value, restore to 1.0 on exit."""
    net.set_slimming(value)
    try:
        yield
    finally:
        net.set_slimming(1.0)


@_requires_render
@_pytest.mark.parametrize("slim_ratio", _SLIM_RATIOS)
def test_slimmable_trainer_core_numerical_agreement_at_slimmed_size(slim_ratio):
    """
    At each slim ratio, Python forward matches core render --slim <ratio>.

    Exports as SlimmableWavenet-compatible .nam, then compares Python output
    (with set_slimming) to core render (with --slim).
    """
    config = _get_config_for_variant("slimmable")
    module = _LightningModule.init_from_config(config)
    module.net.sample_rate = 48000
    sample_rate = 48000
    module.eval()

    with _TemporaryDirectory() as tmpdir:
        outdir = _Path(tmpdir)

        # Export as WaveNet (full-size weights); core detects slimmable from config
        _get_slimmable_net(module).set_slimming(1.0)
        module.net.export(outdir, basename="model", include_snapshot=True)

        nam_path = outdir / "model.nam"
        input_npy = _np.load(outdir / "test_inputs.npy")
        input_wav_path = outdir / "input.wav"
        np_to_wav(input_npy, input_wav_path, rate=sample_rate)

        # Python: run forward at slim_ratio using context manager
        x = _torch.from_numpy(input_npy).float()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        with _slim_context(_get_slimmable_net(module), slim_ratio):
            with _torch.no_grad():
                expected = module.net(x, pad_start=True).cpu().numpy()

        # C++: render with --slim (core instantiates slimmable from config)
        output_wav_path = outdir / "output.wav"
        result = _run_render(nam_path, input_wav_path, output_wav_path, slim=slim_ratio)

        assert result.returncode == 0, (
            f"render failed for slim_ratio={slim_ratio}: "
            f"stderr={result.stderr!r} stdout={result.stdout!r}"
        )

        actual = wav_to_np(output_wav_path)

        expected_flat = _np.squeeze(expected)
        actual_flat = _np.squeeze(actual)

        assert expected_flat.shape == actual_flat.shape, (
            f"Shape mismatch for slim_ratio={slim_ratio}: "
            f"expected {expected_flat.shape}, got {actual_flat.shape}"
        )

        assert _np.allclose(expected_flat, actual_flat, rtol=_RTOL, atol=_ATOL), (
            f"Numerical mismatch for slim_ratio={slim_ratio}: "
            f"max |diff| = {_np.max(_np.abs(expected_flat - actual_flat))}"
        )
