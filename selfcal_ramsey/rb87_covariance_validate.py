"""Regression audit for physically admissible likelihood interpolation."""
import json
import numpy as np
from scipy.interpolate import CubicSpline
from rb87_adaptive_experiment import Estimator, OUT


def main():
    rows = []
    grid = np.linspace(-.8, .8, 10001)
    for filename in ["rb87_refined_lookup.json", "rb87_refined_effective_lookup.json"]:
        source = json.loads((OUT / filename).read_text())
        for entry in source["designs"]:
            values = np.array([r["covariance"] for r in source["tables"][entry["name"]]])
            original = CubicSpline(source["grid"], values)
            estimator = Estimator(entry, source["grid"], source["tables"][entry["name"]])
            minimum = float(np.linalg.eigvalsh(estimator.cov(grid)).min())
            error = float(np.max(abs(estimator.cov(source["grid"])-values)))
            assert minimum > 0 and error < 1e-12
            row = dict(lookup=filename, name=entry["name"], method=estimator.cov.interpolation_method,
                       original_minimum_eigenvalue=float(np.linalg.eigvalsh(original(grid)).min()),
                       minimum_eigenvalue=minimum, node_reconstruction_error=error)
            if row["method"] == "certified_positive_cubic":
                assert np.array_equal(estimator.cov(grid), original(grid))
            rows.append(row)
    bad = next(r for r in rows if "effective" in r["lookup"] and r["name"] == "fixed_shape")
    assert bad["original_minimum_eigenvalue"] < 0 and bad["method"] == "log_cholesky_cubic"
    assert all(r["method"] == "certified_positive_cubic" for r in rows if "effective" not in r["lookup"])
    result = dict(passed=True, rows=rows, scope="Bernstein certificate or positive factor interpolation over the closed lookup interval; node fidelity and 10001-point regression. Gaussian likelihood approximation remains separate.")
    (OUT / "rb87_covariance_validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
