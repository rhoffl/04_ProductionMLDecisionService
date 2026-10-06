import numpy as np

from ml.economics import choose_threshold, threshold_report


def test_threshold_respects_capacity():
    y = np.array([1, 1, 1, 0, 0, 0])
    p = np.array([0.95, 0.80, 0.55, 0.70, 0.30, 0.10])
    report = threshold_report(y, p, capacity=3, retained_margin=300, save_probability=0.25, intervention_cost=12)
    selected = choose_threshold(report)
    assert selected["contacted"] <= 3
    assert selected["feasible"]

