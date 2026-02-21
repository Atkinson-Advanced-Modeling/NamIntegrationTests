"""
WaveNet export -> loadmodel integration tests.

Assert that models exported by the trainer (neural-amp-modeler) can be loaded
by the core's loadmodel tool.
"""

from pathlib import Path as _Path
from tempfile import TemporaryDirectory as _TemporaryDirectory

import pytest as _pytest

from _configs import (
    FILM_SLOTS as _FILM_SLOTS,
    LOADMODEL_ACTIVATIONS as _LOADMODEL_ACTIVATIONS,
    get_config_for_variant as _get_config_for_variant,
)
from _integration import (
    run_loadmodel as _run_loadmodel,
    requires_loadmodel as _requires_loadmodel,
)
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
