"""
Numerical agreement tests for slimmable WaveNet.

Unlike the standard numerical agreement tests, these include tests at different
slimmed sizes (slim ratios 0.0–1.0). At full size (1.0), we compare Python
output to NeuralAmpModelerCore render. At slimmed sizes, we verify Python
forward pass determinism (core comparison requires SlimmableWavenet export).
"""

import math as _math
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

# Slim ratios to test. With allowed_channels [2, 4]: 0.0->2ch, 0.5->2ch, 1.0->4ch.
# Add 0.25, 0.75 to cover intermediate behavior.
_SLIM_RATIOS = [0.0, 0.25, 0.5, 0.75, 1.0]


def _get_slimmable_net(module):
    """Extract the underlying _WaveNet (Slimmable) from LightningModule's net."""
    return module.net._net


def _make_test_input(module, sample_rate=48000):
    """Create the same test input used by export (zeros, sin, zeros)."""
    x = _torch.cat(
        [
            _torch.zeros((sample_rate,)),
            0.5
            * _torch.sin(
                2.0 * _math.pi * 220.0 * _torch.linspace(0.0, 1.0, sample_rate + 1)[:-1]
            ),
            _torch.zeros((sample_rate,)),
        ]
    )
    return x


@_requires_render
def test_slimmable_trainer_core_numerical_agreement_full_size():
    """
    Slimmable WaveNet at full size (slim=1.0): Python export matches core render.

    The slimmable model exports as regular WaveNet; at full size the outputs
    should agree.
    """
    config = _get_config_for_variant("slimmable")
    module = _LightningModule.init_from_config(config)
    module.net.sample_rate = 48000
    sample_rate = 48000

    # Ensure full size
    _get_slimmable_net(module).set_slimming(1.0)

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
            "render failed for slimmable full size: "
            f"stderr={result.stderr!r} stdout={result.stdout!r}"
        )

        actual = wav_to_np(output_wav_path)

        expected_flat = _np.squeeze(expected_npy)
        actual_flat = _np.squeeze(actual)

        assert (
            expected_flat.shape == actual_flat.shape
        ), f"Shape mismatch: expected {expected_flat.shape}, got {actual_flat.shape}"

        assert _np.allclose(
            actual_flat, expected_flat, rtol=_RTOL, atol=_ATOL
        ), f"Numerical mismatch: max |diff| = {_np.max(_np.abs(actual_flat - expected_flat))}"


@_pytest.mark.parametrize("slim_ratio", _SLIM_RATIOS)
def test_slimmable_python_determinism_at_slimmed_sizes(slim_ratio):
    """
    At each slim ratio, running forward twice with the same input yields the same output.

    Verifies the slimmable forward pass is deterministic across slimmed sizes.
    """
    config = _get_config_for_variant("slimmable")
    module = _LightningModule.init_from_config(config)
    module.net.sample_rate = 48000
    module.eval()

    net = _get_slimmable_net(module)
    net.set_slimming(slim_ratio)

    x = _make_test_input(module)

    with _torch.no_grad():
        y1 = module.net(x, pad_start=True)
        y2 = module.net(x, pad_start=True)

    y1_np = y1.cpu().numpy()
    y2_np = y2.cpu().numpy()

    assert _np.allclose(y1_np, y2_np, rtol=_RTOL, atol=_ATOL), (
        f"Slimmable forward at ratio {slim_ratio} not deterministic: "
        f"max |diff| = {_np.max(_np.abs(y1_np - y2_np))}"
    )


@_pytest.mark.parametrize("slim_ratio", _SLIM_RATIOS)
def test_slimmable_different_ratios_produce_different_outputs(slim_ratio):
    """
    Different slim ratios produce different outputs (sanity check for slimmable logic).

    With allowed_channels [2, 4], ratio 0.0 and 0.5 both map to 2ch, so they match.
    Ratio 1.0 uses 4ch and should differ. We assert that at least one pair differs.
    """
    config = _get_config_for_variant("slimmable")
    module = _LightningModule.init_from_config(config)
    module.net.sample_rate = 48000
    module.eval()

    x = _make_test_input(module)
    outputs = {}

    with _torch.no_grad():
        for r in _SLIM_RATIOS:
            _get_slimmable_net(module).set_slimming(r)
            outputs[r] = module.net(x, pad_start=True).cpu().numpy()

    # Full size (1.0) should differ from min size (0.0) when allowed_channels has >1 option
    full_out = outputs[1.0]
    min_out = outputs[0.0]
    assert not _np.allclose(
        full_out, min_out, rtol=_RTOL, atol=_ATOL
    ), "Full and min slim sizes produced identical output (expected different channel counts)"
