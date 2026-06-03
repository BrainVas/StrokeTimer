# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
"""Minimal experiment logger used by the StrokeTimer runner."""

from __future__ import annotations

import os
from pathlib import Path


class Logger:
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "log.txt"

    def log_cfg(self, config: dict):
        cfg_path = self.log_dir / "cfg_logged.yaml"
        with open(cfg_path, "w", encoding="utf-8") as f:
            try:
                import yaml
                yaml.safe_dump(config, f, sort_keys=False, allow_unicode=False)
            except ModuleNotFoundError:
                f.write(repr(config))
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write("Configuration saved to " + os.fspath(cfg_path) + "\n")

    def log(self, message):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(str(message) + "\n")
