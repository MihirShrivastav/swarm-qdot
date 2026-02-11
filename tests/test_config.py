from copy import deepcopy

import pytest

from utils.config import DEFAULT_CONFIG, validate_config


def test_validate_config_accepts_default():
    cfg = deepcopy(DEFAULT_CONFIG)
    validate_config(cfg)


def test_validate_config_rejects_bad_lr_schedule_factor():
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg["training"]["lr_schedule"]["enabled"] = True
    cfg["training"]["lr_schedule"]["factor"] = 1.2
    with pytest.raises(ValueError, match="factor"):
        validate_config(cfg)


def test_validate_config_rejects_bad_lr_schedule_alpha():
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg["training"]["lr_schedule"]["enabled"] = True
    cfg["training"]["lr_schedule"]["monitor_ema_alpha"] = 0.0
    with pytest.raises(ValueError, match="monitor_ema_alpha"):
        validate_config(cfg)
