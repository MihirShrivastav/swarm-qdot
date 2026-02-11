from experiments.run_single_config import _extract_lr_drop_events


def test_extract_lr_drop_events():
    history = [
        {"step": 1, "lr": 1e-4, "lr_reduced": False, "lr_monitor": 10.0},
        {"step": 2, "lr": 1e-4, "lr_reduced": False, "lr_monitor": 9.0},
        {"step": 3, "lr": 5e-5, "lr_reduced": True, "lr_monitor": 8.0},
        {"step": 4, "lr": 5e-5, "lr_reduced": False, "lr_monitor": 7.0},
        {"step": 5, "lr": 2.5e-5, "lr_reduced": True, "lr_monitor": 6.5},
    ]
    out = _extract_lr_drop_events(history)
    assert out["num_drops"] == 2
    assert out["initial_lr"] == 1e-4
    assert out["final_lr"] == 2.5e-5
    assert out["min_lr_seen"] == 2.5e-5
    assert out["drops"][0]["step"] == 3
    assert out["drops"][0]["lr_prev"] == 1e-4
    assert out["drops"][0]["lr_new"] == 5e-5
