"""
Shared config definitions for WaveNet integration tests.

Used by both loadmodel and numerical agreement tests.
"""

import copy as _copy
import json as _json
from pathlib import Path as _Path
from typing import Any as _Any

# Activations supported by both Python _activations and NeuralAmpModelerCore loadmodel.
# (Fasttanh is C++-only; omit it since we build the model in Python.)
LOADMODEL_ACTIVATIONS = [
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

FILM_SLOTS = (
    "conv_pre_film",
    "conv_post_film",
    "input_mixin_pre_film",
    "input_mixin_post_film",
    "activation_pre_film",
    "activation_post_film",
    "layer1x1_post_film",
    "head1x1_post_film",
)


def load_demonet_config() -> dict:
    """Load demonet config (mirrors nam_full_configs/models/demonet.json)."""
    path = _Path(__file__).resolve().parents[1] / "configs" / "demonet.json"
    data = _json.loads(path.read_text())
    data.pop("_notes", None)
    data.pop("_comments", None)
    return data


def _condition_dsp_config() -> dict:
    """WaveNet config with condition_dsp (different structure than demonet)."""
    return {
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


def _apply_activation(config: dict, activation: _Any) -> None:
    for layer in config["net"]["config"]["layers_configs"]:
        layer["activation"] = activation


def _apply_bottleneck(config: dict) -> None:
    for layer in config["net"]["config"]["layers_configs"]:
        layer["bottleneck"] = 2


def _apply_groups_input(config: dict) -> None:
    config["net"]["config"]["layers_configs"][1]["groups_input"] = 2


def _apply_head1x1(config: dict) -> None:
    layers_configs = config["net"]["config"]["layers_configs"]
    head1x1_out_channels = layers_configs[0]["head_size"]
    for layer in layers_configs:
        layer["head_1x1_config"] = {
            "active": True,
            "out_channels": head1x1_out_channels,
            "groups": 1,
        }


def _apply_per_layer_activations(config: dict) -> None:
    layers_configs = config["net"]["config"]["layers_configs"]
    per_layer_activations = ["Tanh", "ReLU"]
    for i, layer in enumerate(layers_configs):
        layer["activation"] = per_layer_activations[i]


def _apply_film(config: dict, film_slot: str) -> None:
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


def get_config_for_variant(variant_id: str) -> dict:
    """
    Build a config dict for the given variant_id.

    :param variant_id: One of "base", "activation_<name>", "bottleneck",
        "groups_input", "head1x1", "per_layer_activations", "film_<slot>",
        "condition_dsp".
    :return: Deep copy of config, ready for module init.
    """
    if variant_id == "condition_dsp":
        return _copy.deepcopy(_condition_dsp_config())

    config = _copy.deepcopy(load_demonet_config())

    if variant_id == "base":
        return config

    if variant_id.startswith("activation_"):
        act_suffix = variant_id[len("activation_") :]
        activation = None
        for a in LOADMODEL_ACTIVATIONS:
            if isinstance(a, dict):
                if a.get("name") == act_suffix:
                    activation = a
                    break
            elif a == act_suffix:
                activation = a
                break
        if activation is None:
            raise ValueError(f"Unknown activation variant: {variant_id}")
        _apply_activation(config, activation)
        return config

    if variant_id == "bottleneck":
        _apply_bottleneck(config)
        return config

    if variant_id == "groups_input":
        _apply_groups_input(config)
        return config

    if variant_id == "head1x1":
        _apply_head1x1(config)
        return config

    if variant_id == "per_layer_activations":
        _apply_per_layer_activations(config)
        return config

    if variant_id.startswith("film_"):
        film_slot = variant_id[len("film_") :]
        if film_slot not in FILM_SLOTS:
            raise ValueError(f"Unknown film slot: {film_slot}")
        _apply_film(config, film_slot)
        return config

    raise ValueError(f"Unknown variant_id: {variant_id}")


def get_all_variant_ids() -> list[str]:
    """All variant IDs for parametrized tests."""
    ids_ = ["base"]
    for i, a in enumerate(LOADMODEL_ACTIVATIONS):
        if isinstance(a, dict):
            ids_.append(f"activation_{a.get('name', i)}")
        else:
            ids_.append(f"activation_{a}")
    ids_.extend(
        [
            "bottleneck",
            "groups_input",
            "head1x1",
            "per_layer_activations",
            "condition_dsp",
        ]
    )
    ids_.extend(f"film_{slot}" for slot in FILM_SLOTS)
    return ids_
