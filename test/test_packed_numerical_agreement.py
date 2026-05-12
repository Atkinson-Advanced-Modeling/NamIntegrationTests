"""
Numerical agreement for packed WaveNet exports.

The packed Stew config exports as a SlimmableContainer with one ordinary
WaveNet per packed submodel. Verify that each Core-selected submodel produces
the same signal as the matching PyTorch packed output channel.
"""

from pathlib import Path as _Path
from tempfile import TemporaryDirectory as _TemporaryDirectory

import numpy as _np
import torch as _torch
from _configs import load_stew_packed_config as _load_stew_packed_config
from _integration import requires_render as _requires_render
from _integration import run_render as _run_render
from nam.data import np_to_wav, wav_to_np
from nam.train.lightning_module import PackedLightningModule as _PackedLightningModule

_RTOL = 1e-5
_ATOL = 1e-6
_SAMPLE_RATE = 48_000
_INPUT_NUM_SAMPLES = 4096


def _make_test_input() -> _np.ndarray:
    t = _np.arange(_INPUT_NUM_SAMPLES, dtype=_np.float32) / _SAMPLE_RATE
    x = (
        0.12 * _np.sin(2.0 * _np.pi * 220.0 * t)
        + 0.03 * _np.sin(2.0 * _np.pi * 997.0 * t)
    ).astype(_np.float32)
    x[0] = 0.25
    x[_INPUT_NUM_SAMPLES // 4] = -0.2
    return x


def _slim_values_for_submodels(container: dict) -> list[float]:
    max_values = [float(item["max_value"]) for item in container["config"]["submodels"]]
    assert max_values == sorted(max_values)
    assert max_values[-1] == 1.0

    values = []
    low = 0.0
    for max_value in max_values:
        high = min(max_value, 1.0)
        values.append(low + 0.5 * (high - low))
        low = max_value
    return values


@_requires_render
def test_stew_packed_export_matches_core_for_both_submodels():
    config = _load_stew_packed_config()
    assert config["net"]["name"] == "PackedWaveNet"

    _torch.manual_seed(0)
    module = _PackedLightningModule.init_from_config(config)
    module.net.sample_rate = _SAMPLE_RATE
    module.eval()

    input_npy = _make_test_input()
    input_tensor = _torch.from_numpy(input_npy).float().unsqueeze(0)
    with _torch.no_grad():
        expected = (
            module.net(input_tensor, pad_start=True).squeeze(0).detach().cpu().numpy()
        )

    assert expected.shape == (2, _INPUT_NUM_SAMPLES)

    with _TemporaryDirectory() as tmpdir:
        outdir = _Path(tmpdir)
        container = module.net.export_container(outdir, basename="model")
        nam_path = outdir / "model.nam"
        assert nam_path.exists()
        assert container["architecture"] == "SlimmableContainer"
        assert len(container["config"]["submodels"]) == 2

        input_wav_path = outdir / "input.wav"
        np_to_wav(input_npy, input_wav_path, rate=_SAMPLE_RATE)

        for submodel_index, slim_value in enumerate(
            _slim_values_for_submodels(container)
        ):
            output_wav_path = outdir / f"output_submodel_{submodel_index}.wav"
            result = _run_render(
                nam_path,
                input_wav_path,
                output_wav_path,
                slim=slim_value,
            )

            assert result.returncode == 0, (
                f"render failed for packed submodel {submodel_index} "
                f"(slim={slim_value}): stderr={result.stderr!r} "
                f"stdout={result.stdout!r}"
            )

            actual = _np.squeeze(wav_to_np(output_wav_path))
            expected_submodel = _np.squeeze(expected[submodel_index])

            assert expected_submodel.shape == actual.shape, (
                f"Shape mismatch for packed submodel {submodel_index}: "
                f"expected {expected_submodel.shape}, got {actual.shape}"
            )

            assert _np.allclose(expected_submodel, actual, rtol=_RTOL, atol=_ATOL), (
                f"Numerical mismatch for packed submodel {submodel_index}: "
                f"max |diff| = {_np.max(_np.abs(expected_submodel - actual))}"
            )
