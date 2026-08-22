"""Sequential model contract tests across the trainer and Core repositories."""

import json as _json
from pathlib import Path as _Path
from tempfile import TemporaryDirectory as _TemporaryDirectory

import numpy as _np
import torch as _torch
from _integration import requires_loadmodel as _requires_loadmodel
from _integration import requires_render as _requires_render
from _integration import run_loadmodel as _run_loadmodel
from _integration import run_render as _run_render
from nam.data import np_to_wav, wav_to_np
from nam.models.linear import Linear as _Linear
from nam.models.sequential import Sequential as _Sequential

_SAMPLE_RATE = 48_000
_RTOL = 1.0e-5
_ATOL = 1.0e-6


def _linear(weights: list[float]) -> _Linear:
    model = _Linear(receptive_field=len(weights), sample_rate=_SAMPLE_RATE)
    model.import_weights(_torch.tensor(weights))
    return model


def _nested_sequential_model() -> _Sequential:
    inner = _Sequential(models=[_linear([0.5, -0.25, 0.125]), _linear([1.2, -0.1])])
    return _Sequential(models=[inner, _linear([-0.75, 0.2, 0.05])])


def _export(model: _Sequential, outdir: _Path) -> tuple[_Path, dict]:
    model.export(outdir, basename="sequential")
    path = outdir / "sequential.nam"
    assert path.exists()
    return path, _json.loads(path.read_text(encoding="utf-8"))


def _assert_complete_model(model: dict) -> None:
    assert {"version", "architecture", "config", "weights"}.issubset(model)


@_requires_loadmodel
def test_trainer_sequential_export_matches_core_file_contract():
    model = _nested_sequential_model()

    with _TemporaryDirectory() as tmpdir:
        path, exported = _export(model, _Path(tmpdir))

        assert exported["version"] == "0.7.0"
        assert exported["architecture"] == "Sequential"
        assert exported["weights"] == []
        assert exported["sample_rate"] == _SAMPLE_RATE
        assert "weights_version" not in exported["config"]

        children = exported["config"]["models"]
        assert [child["architecture"] for child in children] == [
            "Sequential",
            "Linear",
        ]
        for child in children:
            _assert_complete_model(child)
        for grandchild in children[0]["config"]["models"]:
            _assert_complete_model(grandchild)

        result = _run_loadmodel(path)
        assert result.returncode == 0, (
            "Core loadmodel rejected the trainer's Sequential export: "
            f"stderr={result.stderr!r} stdout={result.stdout!r}"
        )


@_requires_render
def test_trainer_and_core_sequential_numerical_agreement():
    model = _nested_sequential_model()
    model.eval()
    t = _np.arange(4096, dtype=_np.float32) / _SAMPLE_RATE
    input_signal = (
        0.12 * _np.sin(2.0 * _np.pi * 220.0 * t)
        + 0.03 * _np.sin(2.0 * _np.pi * 997.0 * t)
    ).astype(_np.float32)
    input_signal[0] = 0.25
    input_signal[1024] = -0.2

    with _torch.no_grad():
        expected = (
            model(_torch.from_numpy(input_signal), pad_start=True)
            .detach()
            .cpu()
            .numpy()
        )

    with _TemporaryDirectory() as tmpdir:
        outdir = _Path(tmpdir)
        model_path, _ = _export(model, outdir)
        input_path = outdir / "input.wav"
        output_path = outdir / "output.wav"
        np_to_wav(input_signal, input_path, rate=_SAMPLE_RATE)

        result = _run_render(model_path, input_path, output_path)
        assert result.returncode == 0, (
            "Core render rejected the trainer's Sequential export: "
            f"stderr={result.stderr!r} stdout={result.stdout!r}"
        )

        actual = _np.squeeze(wav_to_np(output_path))
        assert actual.shape == expected.shape
        assert _np.allclose(actual, expected, rtol=_RTOL, atol=_ATOL), (
            "Sequential trainer/Core numerical mismatch: "
            f"max |diff| = {_np.max(_np.abs(actual - expected))}"
        )
