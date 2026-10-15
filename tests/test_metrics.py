import numpy as np
import pytest

from metrics import (accuracy, bayes_decide, bootstrap_ci, choose_tau, cus, macro_f1,
                     risk_coverage, summarize)

U3 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.9], [0.0, 0.9, 1.0]])


def test_cus_hand_computed():
    assert cus(np.array([0, 1, 2]), np.array([0, 2, 2]), U3) == pytest.approx((1 + 0.9 + 1) / 3)


def test_bayes_rule_prefers_expected_utility_over_argmax():
    proba = np.array([[0.40, 0.35, 0.25]])
    # EU(k) = sum_j p_j U[j,k] = [0.40, 0.575, 0.565]
    assert proba.argmax(1)[0] == 0
    assert bayes_decide(proba, U3)[0] == 1


def test_macro_f1_ignores_classes_without_support():
    per, m = macro_f1(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), 3)
    assert per[0] == pytest.approx(2 / 3) and per[1] == pytest.approx(0.8)
    assert m == pytest.approx((2 / 3 + 0.8) / 2)


def test_bootstrap_ci_on_perfect_predictions():
    y = np.arange(20) % 3
    ci = bootstrap_ci(accuracy, y, np.stack([y, y]), n_boot=50)
    assert ci == {"mean": 1.0, "lo": 1.0, "hi": 1.0}


def test_risk_coverage_and_tau():
    y = np.array([0, 1, 0, 1])
    proba = np.array([[0.95, 0.05], [0.9, 0.1], [0.6, 0.4], [0.2, 0.8]])
    rows = {r["tau"]: r for r in risk_coverage(y, proba)}
    assert rows[0.0]["coverage"] == 1.0 and rows[0.0]["accuracy"] == 0.75
    assert rows[0.95]["coverage"] == 0.25 and rows[0.95]["accuracy"] == 1.0
    assert choose_tau(list(rows.values()), target_acc=0.99, min_coverage=0.9) == 0.0


def test_summarize_keys():
    s = summarize(np.array([0, 1, 2]), np.array([0, 1, 1]), U3)
    assert set(s) == {"accuracy", "macro_f1", "cus", "per_class_f1", "confusion"}
    assert np.array(s["confusion"]).sum() == 3
