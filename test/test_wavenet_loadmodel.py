"""
WaveNet export -> loadmodel integration tests.

Assert that models exported by the trainer (neural-amp-modeler) can be loaded
by the core's loadmodel tool.
"""

import json as _json
from pathlib import Path as _Path
from tempfile import TemporaryDirectory as _TemporaryDirectory

import pytest as _pytest

from _integration import run_loadmodel, requires_loadmodel
from nam.train.lightning_module import LightningModule as _LightningModule

# Activations supported by both Python _activations and NeuralAmpModelerCore loadmodel.
# (Fasttanh is C++-only; omit it since we build the model in Python.)
_LOADMODEL_ACTIVATIONS = [
    "Tanh",
    "Hardtanh",
    "ReLU",
    "LeakyReLU",
    "PReLU",
    "Sigmoid",
    "SiLU",
    "Hardswish",
    "LeakyHardtanh",
    "Softsign",
    {"name": "PairBlend", "primary": "Tanh", "secondary": "Sigmoid"},
    {"name": "PairMultiply", "primary": "Tanh", "secondary": "Sigmoid"},
]

_FILM_SLOTS = (
    "conv_pre_film",
    "conv_post_film",
    "input_mixin_pre_film",
    "input_mixin_post_film",
    "activation_pre_film",
    "activation_post_film",
    "layer1x1_post_film",
    "head1x1_post_film",
)


def _load_demonet_config() -> dict:
    path = _Path(__file__).resolve().parents[1] / "configs" / "demonet.json"
    data = _json.loads(path.read_text())
    data.pop("_notes", None)
    data.pop("_comments", None)
    return data


@requires_loadmodel
@_pytest.mark.parametrize("activation", _LOADMODEL_ACTIVATIONS)
def test_export_nam_loadmodel_can_load(demonet_config, activation):
    """
    LightningModule.init_from_config(demonet with activation replaced) -> .export()
    -> loadmodel can load the resulting .nam.
    """
    config = _load_demonet_config()
    for layer in config["net"]["config"]["layers_configs"]:
        layer["activation"] = activation
    module = _LightningModule.init_from_config(config)
    module.net.sample_rate = 48000
    with _TemporaryDirectory() as tmpdir:
        outdir = _Path(tmpdir)
        module.net.export(outdir, basename="model")
        nam_path = outdir / "model.nam"
        assert nam_path.exists()
        result = run_loadmodel(nam_path)
        assert result.returncode == 0, (
            f"loadmodel failed for activation={activation!r}: "
            f"stderr={result.stderr!r} stdout={result.stdout!r}"
        )


@requires_loadmodel
def test_export_nam_loadmodel_can_load_with_bottleneck(demonet_config):
    """
    LightningModule with bottleneck -> .export() -> loadmodel can load the .nam.
    """
    config = _load_demonet_config()
    for layer in config["net"]["config"]["layers_configs"]:
        layer["bottleneck"] = 2
    module = _LightningModule.init_from_config(config)
    module.net.sample_rate = 48000
    with _TemporaryDirectory() as tmpdir:
        outdir = _Path(tmpdir)
        module.net.export(outdir, basename="model")
        nam_path = outdir / "model.nam"
        assert nam_path.exists()
        result = run_loadmodel(nam_path)
        assert result.returncode == 0, (
            "loadmodel failed for bottleneck: "
            f"stderr={result.stderr!r} stdout={result.stdout!r}"
        )


@requires_loadmodel
def test_export_nam_loadmodel_can_load_with_groups_input(demonet_config):
    """
    LightningModule with groups_input=2 -> .export() -> loadmodel can load the .nam.
    """
    config = _load_demonet_config()
    layers_configs = config["net"]["config"]["layers_configs"]
    layers_configs[1]["groups_input"] = 2
    module = _LightningModule.init_from_config(config)
    module.net.sample_rate = 48000
    with _TemporaryDirectory() as tmpdir:
        outdir = _Path(tmpdir)
        module.net.export(outdir, basename="model")
        nam_path = outdir / "model.nam"
        assert nam_path.exists()
        result = run_loadmodel(nam_path)
        assert result.returncode == 0, (
            "loadmodel failed for groups_input=2: "
            f"stderr={result.stderr!r} stdout={result.stdout!r}"
        )


@requires_loadmodel
def test_export_nam_loadmodel_can_load_with_head1x1(demonet_config):
    """
    LightningModule with head1x1 active -> .export() -> loadmodel can load the .nam.
    """
    config = _load_demonet_config()
    layers_configs = config["net"]["config"]["layers_configs"]
    head1x1_out_channels = layers_configs[0]["head_size"]
    for layer in layers_configs:
        layer["head_1x1_config"] = {
            "active": True,
            "out_channels": head1x1_out_channels,
            "groups": 1,
        }
    module = _LightningModule.init_from_config(config)
    module.net.sample_rate = 48000
    with _TemporaryDirectory() as tmpdir:
        outdir = _Path(tmpdir)
        module.net.export(outdir, basename="model")
        nam_path = outdir / "model.nam"
        assert nam_path.exists()
        result = run_loadmodel(nam_path)
        assert result.returncode == 0, (
            "loadmodel failed for head1x1: "
            f"stderr={result.stderr!r} stdout={result.stdout!r}"
        )


@requires_loadmodel
def test_export_nam_loadmodel_can_load_different_activation_per_layer(demonet_config):
    """
    Same as test_export_nam_loadmodel_can_load but with a different activation
    for each layer in the layer array (loadmodel still loads the .nam).
    """
    config = _load_demonet_config()
    layers_configs = config["net"]["config"]["layers_configs"]
    per_layer_activations = ["Tanh", "ReLU"]
    for i, layer in enumerate(layers_configs):
        layer["activation"] = per_layer_activations[i]
    module = _LightningModule.init_from_config(config)
    module.net.sample_rate = 48000
    with _TemporaryDirectory() as tmpdir:
        outdir = _Path(tmpdir)
        module.net.export(outdir, basename="model")
        nam_path = outdir / "model.nam"
        assert nam_path.exists()
        result = run_loadmodel(nam_path)
        assert result.returncode == 0, (
            "loadmodel failed for per-layer activations: "
            f"stderr={result.stderr!r} stdout={result.stdout!r}"
        )


@requires_loadmodel
@_pytest.mark.parametrize("film_slot", _FILM_SLOTS)
def test_export_nam_loadmodel_can_load_with_film(demonet_config, film_slot):
    """
    LightningModule with one FiLM slot active -> .export() -> loadmodel can load the .nam.
    """
    config = _load_demonet_config()
    layers_configs = config["net"]["config"]["layers_configs"]
    head_size = layers_configs[0]["head_size"]
    for layer in layers_configs:
        layer["film_params"] = {
            film_slot: {"active": True, "shift": True, "groups": 1},
        }
        if film_slot == "head1x1_post_film":
            layer["head_1x1_config"] = {
                "active": True,
                "out_channels": head_size,
                "groups": 1,
            }
    module = _LightningModule.init_from_config(config)
    module.net.sample_rate = 48000
    with _TemporaryDirectory() as tmpdir:
        outdir = _Path(tmpdir)
        module.net.export(outdir, basename="model")
        nam_path = outdir / "model.nam"
        assert nam_path.exists()
        result = run_loadmodel(nam_path)
        assert result.returncode == 0, (
            f"loadmodel failed for FiLM slot {film_slot!r}: "
            f"stderr={result.stderr!r} stdout={result.stdout!r}"
        )


@requires_loadmodel
def test_export_nam_loadmodel_can_load_with_condition_dsp():
    """
    WaveNet with condition_dsp: export to .nam -> loadmodel can load the file.
    """
    config = {
        "net": {
            "name": "WaveNet",
            "config": {
                "condition_dsp": {
                    "name": "WaveNet",
                    "config": {
                        "layers_configs": [
                            {
                                "input_size": 1,
                                "condition_size": 1,
                                "head_size": 2,
                                "channels": 2,
                                "kernel_size": 2,
                                "dilations": [1],
                                "activation": "Tanh",
                            }
                        ],
                        "head_scale": 1.0,
                    },
                },
                "layers_configs": [
                    {
                        "input_size": 1,
                        "condition_size": 2,
                        "head_size": 1,
                        "channels": 2,
                        "kernel_size": 2,
                        "dilations": [1],
                        "activation": "Tanh",
                    }
                ],
                "head_scale": 1.0,
            },
        },
        "optimizer": {"lr": 0.004},
        "lr_scheduler": {"class": "ExponentialLR", "kwargs": {"gamma": 0.993}},
    }
    module = _LightningModule.init_from_config(config)
    module.net.sample_rate = 48000
    with _TemporaryDirectory() as tmpdir:
        outdir = _Path(tmpdir)
        module.net.export(outdir, basename="model")
        nam_path = outdir / "model.nam"
        assert nam_path.exists()
        result = run_loadmodel(nam_path)
        assert result.returncode == 0, (
            "loadmodel failed for condition_dsp: "
            f"stderr={result.stderr!r} stdout={result.stdout!r}"
        )
