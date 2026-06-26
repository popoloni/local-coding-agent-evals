from metrics import collect_metrics


def test_first_metric():
    assert collect_metrics("loss", 1.0) == {"loss": 1.0}


def test_second_metric_starts_empty():
    assert collect_metrics("accuracy", 0.9) == {"accuracy": 0.9}
