import random
import re

from experiments.run_default_random import random_experiment_name


def test_random_experiment_name_format():
    rng = random.Random(123)
    name = random_experiment_name(rng)
    assert re.fullmatch(r"[a-z]+-[a-z]+", name) is not None
