"""
WaveNet export -> loadmodel integration tests.

Assert that models exported by the trainer (neural-amp-modeler) can be loaded
by the core's loadmodel tool.
"""

import json as _json
from pathlib import Path as _Path
from tempfile import TemporaryDirectory as _TemporaryDirectory

import pytest as _pytest
from _configs import FILM_SLOTS as _FILM_SLOTS
from _configs import LOADMODEL_ACTIVATIONS as _LOADMODEL_ACTIVATIONS
from _configs import get_config_for_variant as _get_config_for_variant
from _integration import requires_loadmodel as _requires_loadmodel
from _integration import run_loadmodel as _run_loadmodel
from nam.train.lightning_module import LightningModule as _LightningModule


@_requires_loadmodel
@_pytest.mark.parametrize("activation", _LOADMODEL_ACTIVATIONS)
def test_export_nam_loadmodel_can_load(demonet_config, activation):
    """
    LightningModule.init_from_config(demonet with activation replaced) -> .export()
    -> loadmodel can load the resulting .nam.
    """
    act_id = activation if isinstance(activation, str) else activation.get("name")
    config = _get_config_for_variant(f"activation_{act_id}")
    module = _LightningModule.init_from_config(config)
    module.net.sample_rate = 48000
    with _TemporaryDirectory() as tmpdir:
        outdir = _Path(tmpdir)
        module.net.export(outdir, basename="model")
        nam_path = outdir / "model.nam"
        assert nam_path.exists()
        result = _run_loadmodel(nam_path)
        assert result.returncode == 0, (
            f"loadmodel failed for activation={activation!r}: "
            f"stderr={result.stderr!r} stdout={result.stdout!r}"
        )


@_requires_loadmodel
def test_export_nam_loadmodel_can_load_with_bottleneck(demonet_config):
    """
    LightningModule with bottleneck -> .export() -> loadmodel can load the .nam.
    """
    config = _get_config_for_variant("bottleneck")
    module = _LightningModule.init_from_config(config)
    module.net.sample_rate = 48000
    with _TemporaryDirectory() as tmpdir:
        outdir = _Path(tmpdir)
        module.net.export(outdir, basename="model")
        nam_path = outdir / "model.nam"
        assert nam_path.exists()
        result = _run_loadmodel(nam_path)
        assert result.returncode == 0, (
            "loadmodel failed for bottleneck: "
            f"stderr={result.stderr!r} stdout={result.stdout!r}"
        )


@_requires_loadmodel
def test_export_nam_loadmodel_can_load_with_groups_input(demonet_config):
    """
    LightningModule with groups_input=2 -> .export() -> loadmodel can load the .nam.
    """
    config = _get_config_for_variant("groups_input")
    module = _LightningModule.init_from_config(config)
    module.net.sample_rate = 48000
    with _TemporaryDirectory() as tmpdir:
        outdir = _Path(tmpdir)
        module.net.export(outdir, basename="model")
        nam_path = outdir / "model.nam"
        assert nam_path.exists()
        result = _run_loadmodel(nam_path)
        assert result.returncode == 0, (
            "loadmodel failed for groups_input=2: "
            f"stderr={result.stderr!r} stdout={result.stdout!r}"
        )


@_requires_loadmodel
def test_export_nam_loadmodel_can_load_with_head1x1(demonet_config):
    """
    LightningModule with head1x1 active -> .export() -> loadmodel can load the .nam.
    """
    config = _get_config_for_variant("head1x1")
    module = _LightningModule.init_from_config(config)
    module.net.sample_rate = 48000
    with _TemporaryDirectory() as tmpdir:
        outdir = _Path(tmpdir)
        module.net.export(outdir, basename="model")
        nam_path = outdir / "model.nam"
        assert nam_path.exists()
        result = _run_loadmodel(nam_path)
        assert result.returncode == 0, (
            "loadmodel failed for head1x1: "
            f"stderr={result.stderr!r} stdout={result.stdout!r}"
        )


@_requires_loadmodel
def test_export_nam_loadmodel_can_load_with_wavenet_head(demonet_config):
    """
    WaveNet with post-stack ``head`` (Conv1d stack) -> export -> loadmodel can load the .nam.
    """
    config = _get_config_for_variant("wavenet_head")
    module = _LightningModule.init_from_config(config)
    module.net.sample_rate = 48000
    with _TemporaryDirectory() as tmpdir:
        outdir = _Path(tmpdir)
        module.net.export(outdir, basename="model")
        nam_path = outdir / "model.nam"
        assert nam_path.exists()
        result = _run_loadmodel(nam_path)
        assert result.returncode == 0, (
            "loadmodel failed for wavenet_head: "
            f"stderr={result.stderr!r} stdout={result.stdout!r}"
        )


@_requires_loadmodel
def test_export_nam_loadmodel_can_load_different_activation_per_layer(demonet_config):
    """
    Same as test_export_nam_loadmodel_can_load but with a different activation
    for each layer in the layer array (loadmodel still loads the .nam).
    """
    config = _get_config_for_variant("per_layer_activations")
    module = _LightningModule.init_from_config(config)
    module.net.sample_rate = 48000
    with _TemporaryDirectory() as tmpdir:
        outdir = _Path(tmpdir)
        module.net.export(outdir, basename="model")
        nam_path = outdir / "model.nam"
        assert nam_path.exists()
        result = _run_loadmodel(nam_path)
        assert result.returncode == 0, (
            "loadmodel failed for per-layer activations: "
            f"stderr={result.stderr!r} stdout={result.stdout!r}"
        )


@_requires_loadmodel
@_pytest.mark.parametrize("film_slot", _FILM_SLOTS)
def test_export_nam_loadmodel_can_load_with_film(demonet_config, film_slot):
    """
    LightningModule with one FiLM slot active -> .export() -> loadmodel can load the .nam.
    """
    config = _get_config_for_variant(f"film_{film_slot}")
    module = _LightningModule.init_from_config(config)
    module.net.sample_rate = 48000
    with _TemporaryDirectory() as tmpdir:
        outdir = _Path(tmpdir)
        module.net.export(outdir, basename="model")
        nam_path = outdir / "model.nam"
        assert nam_path.exists()
        result = _run_loadmodel(nam_path)
        assert result.returncode == 0, (
            f"loadmodel failed for FiLM slot {film_slot!r}: "
            f"stderr={result.stderr!r} stdout={result.stdout!r}"
        )


@_requires_loadmodel
def test_export_nam_loadmodel_can_load_with_condition_dsp():
    """
    WaveNet with condition_dsp: export to .nam -> loadmodel can load the file.
    """
    config = _get_config_for_variant("condition_dsp")
    module = _LightningModule.init_from_config(config)
    module.net.sample_rate = 48000
    with _TemporaryDirectory() as tmpdir:
        outdir = _Path(tmpdir)
        module.net.export(outdir, basename="model")
        nam_path = outdir / "model.nam"
        assert nam_path.exists()
        result = _run_loadmodel(nam_path)
        assert result.returncode == 0, (
            "loadmodel failed for condition_dsp: "
            f"stderr={result.stderr!r} stdout={result.stdout!r}"
        )


@_requires_loadmodel
def test_export_nam_loadmodel_can_load_layer_head_rechannel_kernel_gt_1():
    """
    Layer-array head rechannel with kernel_size > 1: export -> Core loadmodel.
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
    with _TemporaryDirectory() as tmpdir:
        outdir = _Path(tmpdir)
        module.net.export(outdir, basename="model")
        nam_path = outdir / "model.nam"
        assert nam_path.exists()
        result = _run_loadmodel(nam_path)
        assert result.returncode == 0, (
            "loadmodel failed for layer head kernel_size=3: "
            f"stderr={result.stderr!r} stdout={result.stdout!r}"
        )


@_requires_loadmodel
def test_export_nam_loadmodel_accepts_legacy_head_size_head_bias():
    """
    NeuralAmpModelerCore must load a .nam whose layer arrays use legacy
    head_size / head_bias (no nested head object). We export with the new
    schema, strip to legacy keys, and assert loadmodel succeeds.
    """
    config = _get_config_for_variant("condition_dsp")
    module = _LightningModule.init_from_config(config)
    module.net.sample_rate = 48000
    with _TemporaryDirectory() as tmpdir:
        outdir = _Path(tmpdir)
        module.net.export(outdir, basename="model")
        nam_path = outdir / "model.nam"
        nam_obj = _json.loads(nam_path.read_text(encoding="utf-8"))
        for layer in nam_obj["config"]["layers"]:
            head = layer.pop("head")
            layer["head_size"] = head["out_channels"]
            layer["head_bias"] = head["bias"]
        legacy_path = outdir / "model_legacy_head.nam"
        legacy_path.write_text(_json.dumps(nam_obj), encoding="utf-8")
        result = _run_loadmodel(legacy_path)
        assert result.returncode == 0, (
            "loadmodel failed for legacy head_size/head_bias: "
            f"stderr={result.stderr!r} stdout={result.stdout!r}"
        )
