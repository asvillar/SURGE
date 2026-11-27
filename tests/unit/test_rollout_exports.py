from surge.inference import rollout


def test_all_names_exist():
    missing = [n for n in rollout.__all__ if not hasattr(rollout, n)]
    assert not missing
