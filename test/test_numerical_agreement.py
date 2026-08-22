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
from _configs import get_all_variant_ids as _get_all_variant_ids
from _configs import get_config_for_variant as _get_config_for_variant
from _integration import requires_render as _requires_render
from _integration import run_render as _run_render
from nam.data import np_to_wav, wav_to_np
from nam.train.lightning_module import LightningModule as _LightningModule

_RTOL = 1e-5
_ATOL = 1e-6


def _variant_ids_parametrize():
    # layer1x1_post_film is covered across its affected gating modes below.
    return [vid for vid in _get_all_variant_ids() if vid != "film_layer1x1_post_film"]


def _assert_trainer_core_numerical_agreement(config, case_id):
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
            f"render failed for {case_id!r}: "
            f"stderr={result.stderr!r} stdout={result.stdout!r}"
        )

        actual = wav_to_np(output_wav_path)

        expected_flat = _np.squeeze(expected_npy)
        actual_flat = _np.squeeze(actual)

        assert expected_flat.shape == actual_flat.shape, (
            f"Shape mismatch for {case_id!r}: "
            f"expected {expected_flat.shape}, got {actual_flat.shape}"
        )

        assert _np.allclose(actual_flat, expected_flat, rtol=_RTOL, atol=_ATOL), (
            f"Numerical mismatch for {case_id!r}: "
            f"max |diff| = {_np.max(_np.abs(actual_flat - expected_flat))}"
        )


@_requires_render
@_pytest.mark.parametrize("variant_id", _variant_ids_parametrize())
def test_trainer_core_numerical_agreement(variant_id):
    """
    Export with include_snapshot -> render through core -> compare outputs.

    Runs for each config variant (activations, bottleneck, FiLM, etc.).
    """
    config = _get_config_for_variant(variant_id)
    _assert_trainer_core_numerical_agreement(config, variant_id)


@_requires_render
@_pytest.mark.parametrize(
    ("activation", "gating_mode"),
    (
        _pytest.param("Tanh", "none", id="none"),
        _pytest.param(
            {
                "name": "PairMultiply",
                "primary": "Tanh",
                "secondary": "Sigmoid",
            },
            "gated",
            id="gated",
        ),
    ),
)
def test_trainer_core_numerical_agreement_layer1x1_post_film(activation, gating_mode):
    """Regression coverage for layer1x1 post-FiLM with non-blended gating."""
    config = _get_config_for_variant("film_layer1x1_post_film")
    for layer in config["net"]["config"]["layers_configs"]:
        layer["activation"] = activation

    _assert_trainer_core_numerical_agreement(
        config, f"layer1x1_post_film_{gating_mode}"
    )


@_requires_render
@_pytest.mark.parametrize("kernel_sizes", ([1], [1, 1], [3, 3], [3, 2]))
def test_trainer_core_numerical_agreement_wavenet_head_kernel_sizes(kernel_sizes):
    """
    Regression test for post-stack WaveNet head activation/conv chaining in core.

    Verifies numerical agreement for multiple head stack depths and kernel layouts.
    """
    config = _get_config_for_variant("wavenet_head")
    config["net"]["config"]["head"]["kernel_sizes"] = list(kernel_sizes)
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
            f"render failed for kernel_sizes {kernel_sizes!r}: "
            f"stderr={result.stderr!r} stdout={result.stdout!r}"
        )

        actual = wav_to_np(output_wav_path)

        expected_flat = _np.squeeze(expected_npy)
        actual_flat = _np.squeeze(actual)

        assert expected_flat.shape == actual_flat.shape, (
            f"Shape mismatch for kernel_sizes {kernel_sizes!r}: "
            f"expected {expected_flat.shape}, got {actual_flat.shape}"
        )

        assert _np.allclose(actual_flat, expected_flat, rtol=_RTOL, atol=_ATOL), (
            f"Numerical mismatch for kernel_sizes {kernel_sizes!r}: "
            f"max |diff| = {_np.max(_np.abs(actual_flat - expected_flat))}"
        )


@_requires_render
def test_trainer_core_numerical_agreement_layer_head_rechannel_kernel_gt_1():
    """
    Layer-array head rechannel (``layers[i].head``) with ``kernel_size`` > 1:
    trainer snapshot vs Core ``render`` must match.

    This is distinct from ``test_trainer_core_numerical_agreement_wavenet_head_kernel_sizes``,
    which varies the post-stack WaveNet ``head`` conv stack, not per-layer-array rechannel.
    """
    config = _get_config_for_variant("condition_dsp")
    for layer in config["net"]["config"]["layers_configs"]:
        layer["head"]["kernel_size"] = 3
    cond = config["net"]["config"].get("condition_dsp")
    if cond is not None:
        for layer in cond["config"]["layers_configs"]:
            layer["head"]["kernel_size"] = 3
    module = _LightningModule.init_from_config(config)
    module.net.sample_rate = 48000
    sample_rate = 48000

    with _TemporaryDirectory() as tmpdir:
        outdir = _Path(tmpdir)
        module.net.export(outdir, basename="model", include_snapshot=True)

        nam_path = outdir / "model.nam"
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
            "render failed for layer head kernel_size=3: "
            f"stderr={result.stderr!r} stdout={result.stdout!r}"
        )

        actual = wav_to_np(output_wav_path)
        expected_flat = _np.squeeze(expected_npy)
        actual_flat = _np.squeeze(actual)

        assert expected_flat.shape == actual_flat.shape, (
            f"Shape mismatch for layer head rechannel k=3: "
            f"expected {expected_flat.shape}, got {actual_flat.shape}"
        )

        assert _np.allclose(actual_flat, expected_flat, rtol=_RTOL, atol=_ATOL), (
            "Numerical mismatch for layer head rechannel kernel_size=3: "
            f"max |diff| = {_np.max(_np.abs(actual_flat - expected_flat))}"
        )
