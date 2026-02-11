from utils.run_manager import create_run


def test_create_run_unique(tmp_path):
    cfg = {"solver": {"K": 2, "M": 4}}
    ctx1 = create_run(tmp_path, "exp", cfg, seed=1)
    ctx2 = create_run(tmp_path, "exp", cfg, seed=1)
    assert ctx1.run_dir != ctx2.run_dir
    assert ctx1.logs_dir.exists()
    assert ctx2.reports_dir.exists()
