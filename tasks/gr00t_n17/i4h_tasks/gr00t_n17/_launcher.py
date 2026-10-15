# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pre-launch shim for Isaac-GR00T's ``launch_finetune.py``.

A ``runpy`` wrapper that can set config fields ``launch_finetune.py`` has no
CLI flag for. Options for the shim come before the launch script; everything
after it is the launcher's own argv.

Run with ``-m`` rather than invoked as a path: that puts the working directory
the fine-tune subprocess is given, the Isaac-GR00T checkout root, at the front
of ``sys.path``, which is where the launcher resolves its own ``gr00t`` imports
from. Running the launch script directly would put the script's own directory
there instead.
"""

from __future__ import annotations

import runpy
import sys

from i4h_tasks.gr00t_n17._patch_embed import install_linear_patch_embed

VIDEO_BACKEND_FLAG = "--video-backend="
VIDEO_BACKENDS = ("torchcodec", "decord")


def split_shim_args(argv: list[str]) -> tuple[dict[str, dict[str, object]], list[str]]:
    """Separate the shim's own options from the launch script and its argv.

    The overrides are keyed by the config section they apply to.
    """
    overrides: dict[str, dict[str, object]] = {}
    rest = list(argv)
    while rest and rest[0].startswith("--"):
        option = rest.pop(0)
        if option.startswith(VIDEO_BACKEND_FLAG):
            backend = option[len(VIDEO_BACKEND_FLAG) :]
            if backend not in VIDEO_BACKENDS:
                sys.exit(f"{VIDEO_BACKEND_FLAG} expects one of {', '.join(VIDEO_BACKENDS)}, got {backend!r}")
            overrides.setdefault("data", {})["video_backend"] = backend
        else:
            sys.exit(f"unknown launcher option {option!r}")
    return overrides, rest


def install_config_overrides(overrides: dict[str, dict[str, object]]) -> None:
    """Make every config the launcher builds carry the overrides.

    Patched on the module rather than on the config classes: the launcher
    imports ``get_default_config`` by name when it runs, so it picks up this
    wrapper, while a changed class attribute would not reach the dataclass
    ``__init__`` defaults that ``Config()`` actually uses.

    The overrides are applied again after every ``load_dict``. It rebuilds each
    section it is given from the dataclass defaults, and the launcher passes it
    a ``data`` section, so a data override set only once would be discarded
    before training reads it. ``load_dict`` is wrapped on the class: the config
    is saved with ``yaml.dump(self)``, which would write an instance attribute
    into the checkpoint's ``config.yaml``.
    """
    if not overrides:
        return
    from gr00t.configs import base_config

    original = base_config.get_default_config
    original_load_dict = base_config.Config.load_dict

    def apply(config):
        for section_name, fields in overrides.items():
            section = getattr(config, section_name)
            for name, value in fields.items():
                if not hasattr(section, name):
                    raise AttributeError(f"GR00T {section_name} config has no field {name!r}")
                setattr(section, name, value)
        return config

    def get_default_config():
        return apply(original())

    def load_dict(self, data: dict):
        return apply(original_load_dict(self, data))

    base_config.get_default_config = get_default_config
    base_config.Config.load_dict = load_dict


def main() -> None:
    overrides, rest = split_shim_args(sys.argv[1:])
    if not rest:
        sys.exit("usage: python -m i4h_tasks.gr00t_n17._launcher [--video-backend=NAME] <launch_finetune.py> [args...]")
    install_config_overrides(overrides)
    install_linear_patch_embed()
    launch_script = rest[0]
    sys.argv = [launch_script, *rest[1:]]
    runpy.run_path(launch_script, run_name="__main__")


if __name__ == "__main__":
    main()
