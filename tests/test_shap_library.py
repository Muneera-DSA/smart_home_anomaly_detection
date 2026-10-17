"""Cross-check: our TreeSHAP against the `shap` package (runs in CI, where shap is installed)."""

import numpy as np
import pytest

shap = pytest.importorskip("shap")

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

from smart_energy import treeshap  # noqa: E402


def test_matches_shap_tree_explainer_classifier():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(800, 6))
    y = (X[:, 0] * 2 + X[:, 1] * X[:, 2] + rng.normal(scale=0.5, size=800) > 0.5).astype(int)
    model = HistGradientBoostingClassifier(max_iter=50, max_leaf_nodes=15, random_state=0).fit(X, y)
    ours, base = treeshap.shap_values(model, X[:25])
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    theirs = np.asarray(explainer.shap_values(X[:25]))
    np.testing.assert_allclose(ours, theirs.reshape(ours.shape), atol=1e-4, rtol=1e-4)
